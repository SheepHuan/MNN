//
//  PagedAttentionBufExecution.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"
#include "core/MNNFileUtils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace MNN {
namespace OpenCL {
namespace {

static uint32_t _floatBits(float value) {
    uint32_t out = 0;
    ::memcpy(&out, &value, sizeof(out));
    return out;
}

static float _bitsFloat(uint32_t value) {
    float out = 0.0f;
    ::memcpy(&out, &value, sizeof(out));
    return out;
}

static float _halfToFloat(uint16_t h) {
    const uint16_t hExp = h & 0x7c00u;
    const uint16_t hSig = h & 0x03ffu;
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t out = 0;
    if (hExp == 0) {
        if (hSig == 0) {
            out = sign;
        } else {
            uint16_t sig = hSig;
            int exp = -1;
            do {
                ++exp;
                sig <<= 1;
            } while ((sig & 0x0400u) == 0);
            sig &= 0x03ffu;
            out = sign | (static_cast<uint32_t>(127 - 15 - exp) << 23) | (static_cast<uint32_t>(sig) << 13);
        }
    } else if (hExp == 0x7c00u) {
        out = sign | 0x7f800000u | (static_cast<uint32_t>(hSig) << 13);
    } else {
        out = sign | (static_cast<uint32_t>((hExp >> 10) + (127 - 15)) << 23) |
              (static_cast<uint32_t>(hSig) << 13);
    }
    return _bitsFloat(out);
}

static uint16_t _floatToHalf(float value) {
    uint32_t bits = _floatBits(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

static inline float _readScalar(const int8_t* ptr, int index, int bytes) {
    if (bytes == 2) {
        return _halfToFloat(reinterpret_cast<const uint16_t*>(ptr)[index]);
    }
    return reinterpret_cast<const float*>(ptr)[index];
}

static inline void _writeScalar(int8_t* ptr, int index, float value, int bytes) {
    if (bytes == 2) {
        reinterpret_cast<uint16_t*>(ptr)[index] = _floatToHalf(value);
        return;
    }
    reinterpret_cast<float*>(ptr)[index] = value;
}

static int _reverseCount(const PagedKVMeta* meta) {
    if (meta == nullptr || meta->n_reserve <= 0 || meta->reserve == nullptr) {
        return 0;
    }
    return std::max(0, meta->computeReverseSize());
}

static bool _profilePagedAttention() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static uint64_t _nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool _writeBinaryFile(const std::string& path, const std::vector<int8_t>& data) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    if (!data.empty()) {
        os.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    return os.good();
}

static std::string _externalLayerKey(const std::string& keyPath, const std::string& valuePath) {
    return keyPath + "\n" + valuePath;
}

static bool _adviseFileWillNeed(const std::string& path) {
#if defined(__linux__) && defined(POSIX_FADV_WILLNEED)
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return false;
    }
    int ret = ::posix_fadvise(fd, 0, st.st_size, POSIX_FADV_WILLNEED);
    ::close(fd);
    return ret == 0;
#else
    (void)path;
    return false;
#endif
}

static std::mutex gExternalLayerPrefetchMutex;
static std::unordered_set<std::string> gExternalLayerPrefetched;
static std::vector<std::shared_future<void>> gExternalLayerPrefetchTasks;

static int _lastExternalLayerIndex(const PagedKVMeta* meta) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return -1;
    }
    int last = meta->layer_nums > 0 ? meta->layer_nums - 1 : -1;
    for (const auto& segment : meta->external_segments) {
        for (const auto& layer : segment.layers) {
            last = std::max(last, layer.layerIndex);
        }
    }
    return last;
}

static void _prefetchExternalLayersFrom(const PagedKVMeta* meta, int startLayer) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return;
    }
    const int lastLayer = _lastExternalLayerIndex(meta);
    if (startLayer < 0 || lastLayer < startLayer) {
        return;
    }
    std::vector<std::pair<std::string, std::string>> jobs;
    {
        std::lock_guard<std::mutex> lock(gExternalLayerPrefetchMutex);
        for (int layerIndex = startLayer; layerIndex <= lastLayer; ++layerIndex) {
            if (meta->externalLayerLoaded(layerIndex)) {
                continue;
            }
            for (const auto& segment : meta->external_segments) {
                auto layer = segment.layer(layerIndex);
                if (layer == nullptr) {
                    continue;
                }
                const auto key = _externalLayerKey(layer->keyPath, layer->valuePath);
                if (gExternalLayerPrefetched.insert(key).second) {
                    jobs.emplace_back(layer->keyPath, layer->valuePath);
                }
            }
        }
    }
    if (jobs.empty()) {
        return;
    }
    auto future = std::async(std::launch::async, [jobs]() {
        for (const auto& job : jobs) {
            (void)_adviseFileWillNeed(job.first);
            (void)_adviseFileWillNeed(job.second);
        }
    }).share();
    std::lock_guard<std::mutex> lock(gExternalLayerPrefetchMutex);
    gExternalLayerPrefetchTasks.emplace_back(std::move(future));
}

static bool _readBinaryFileRange(const std::string& path, size_t offsetBytes, void* dst, size_t expectedBytes) {
    if (dst == nullptr || expectedBytes == 0) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto fileSize = is.tellg();
    if (fileSize < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(fileSize));
    if (offsetBytes > fileBytes || fileBytes - offsetBytes < expectedBytes) {
        return false;
    }
    is.seekg(static_cast<std::streamoff>(offsetBytes), std::ios::beg);
    char* cursor = reinterpret_cast<char*>(dst);
    size_t remaining = expectedBytes;
    while (remaining > 0) {
        const auto chunk = static_cast<std::streamsize>(
            std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<std::streamsize>::max())));
        is.read(cursor, chunk);
        if (is.gcount() != chunk) {
            return false;
        }
        cursor += chunk;
        remaining -= static_cast<size_t>(chunk);
    }
    return true;
}

static bool _readBinaryFilePrefix(const std::string& path, void* dst, size_t expectedBytes) {
    return _readBinaryFileRange(path, 0, dst, expectedBytes);
}

enum class SourceCLReadStatus {
    Success,
    MapFailed,
    ReadFailed,
};

static SourceCLReadStatus _readBinaryFileRangeToMappedCLBuffer(const std::string& path, size_t offsetBytes,
                                                               cl::Buffer& buffer, size_t expectedBytes,
                                                               cl::CommandQueue& queue);
static bool _readBinaryFileRangeToCLBufferFallback(const std::string& path, size_t offsetBytes, cl::Buffer& buffer,
                                                   size_t expectedBytes, cl::CommandQueue& queue);
static bool _readBinaryFileRangeToSourceCLBuffer(const std::string& path, size_t offsetBytes, cl::Buffer& buffer,
                                                 size_t expectedBytes, cl::CommandQueue& queue);

static SourceCLReadStatus _readBinaryFileToMappedCLBuffer(const std::string& path, cl::Buffer& buffer,
                                                          size_t expectedBytes, cl::CommandQueue& queue) {
    return _readBinaryFileRangeToMappedCLBuffer(path, 0, buffer, expectedBytes, queue);
}

static SourceCLReadStatus _readBinaryFileRangeToMappedCLBuffer(const std::string& path, size_t offsetBytes,
                                                               cl::Buffer& buffer, size_t expectedBytes,
                                                               cl::CommandQueue& queue) {
    if (expectedBytes == 0) {
        return SourceCLReadStatus::ReadFailed;
    }
    cl_int error = CL_SUCCESS;
    void* ptr = queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, expectedBytes, nullptr, nullptr, &error);
    if (ptr == nullptr || error != CL_SUCCESS) {
        return SourceCLReadStatus::MapFailed;
    }
    bool ok = _readBinaryFileRange(path, offsetBytes, ptr, expectedBytes);
    auto unmap = queue.enqueueUnmapMemObject(buffer, ptr);
    if (!ok) {
        return SourceCLReadStatus::ReadFailed;
    }
    return unmap == CL_SUCCESS ? SourceCLReadStatus::Success : SourceCLReadStatus::MapFailed;
}

static bool _readBinaryFileToCLBufferFallback(const std::string& path, cl::Buffer& buffer, size_t expectedBytes,
                                              cl::CommandQueue& queue) {
    return _readBinaryFileRangeToCLBufferFallback(path, 0, buffer, expectedBytes, queue);
}

static bool _readBinaryFileRangeToCLBufferFallback(const std::string& path, size_t offsetBytes, cl::Buffer& buffer,
                                                   size_t expectedBytes, cl::CommandQueue& queue) {
    std::vector<int8_t> data(expectedBytes);
    if (!_readBinaryFileRange(path, offsetBytes, data.data(), expectedBytes)) {
        return false;
    }
    return queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, expectedBytes, data.data()) == CL_SUCCESS;
}

static bool _readBinaryFileToSourceCLBuffer(const std::string& path, cl::Buffer& buffer, size_t expectedBytes,
                                            cl::CommandQueue& queue) {
    return _readBinaryFileRangeToSourceCLBuffer(path, 0, buffer, expectedBytes, queue);
}

static bool _readBinaryFileRangeToSourceCLBuffer(const std::string& path, size_t offsetBytes, cl::Buffer& buffer,
                                                 size_t expectedBytes, cl::CommandQueue& queue) {
    auto status = _readBinaryFileRangeToMappedCLBuffer(path, offsetBytes, buffer, expectedBytes, queue);
    if (status == SourceCLReadStatus::Success) {
        return true;
    }
    if (status == SourceCLReadStatus::ReadFailed) {
        return false;
    }
    static std::once_flag fallbackOnce;
    std::call_once(fallbackOnce, []() {
        MNN_PRINT("OpenCLPagedAttention: mapped source CL buffer unavailable, fallback to enqueueWriteBuffer\n");
    });
    return _readBinaryFileRangeToCLBufferFallback(path, offsetBytes, buffer, expectedBytes, queue);
}

static bool _readExternalValueSegment(const std::string& path, void* dst, size_t expectedBytes, int batch,
                                      int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset,
                                      size_t tokenCount, int headDim, int bytes) {
    if (dst == nullptr || expectedBytes == 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0 || bytes <= 0) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto fileSize = is.tellg();
    if (fileSize < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(fileSize));
    const size_t tokenBytes = static_cast<size_t>(headDim) * bytes;
    const size_t segmentBytes = tokenCount * tokenBytes;
    const size_t requiredBytes = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * tokenBytes;
    if (sourceTokenOffset + tokenCount > sourceTokenCount || fileBytes < requiredBytes ||
        expectedBytes < static_cast<size_t>(batch) * kvHeads * segmentBytes) {
        return false;
    }
    char* out = reinterpret_cast<char*>(dst);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dstOffset = (static_cast<size_t>(b) * kvHeads + h) * segmentBytes;
            is.seekg(static_cast<std::streamoff>(src), std::ios::beg);
            is.read(out + dstOffset, static_cast<std::streamsize>(segmentBytes));
            if (is.gcount() != static_cast<std::streamsize>(segmentBytes)) {
                return false;
            }
        }
    }
    return true;
}

static SourceCLReadStatus _readExternalValueSegmentToMappedCLBuffer(
    const std::string& path, cl::Buffer& buffer, size_t expectedBytes, cl::CommandQueue& queue, int batch,
    int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount, int headDim, int bytes) {
    if (expectedBytes == 0) {
        return SourceCLReadStatus::ReadFailed;
    }
    cl_int error = CL_SUCCESS;
    void* ptr = queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, expectedBytes, nullptr, nullptr, &error);
    if (ptr == nullptr || error != CL_SUCCESS) {
        return SourceCLReadStatus::MapFailed;
    }
    bool ok = _readExternalValueSegment(path, ptr, expectedBytes, batch, kvHeads, sourceTokenCount,
                                        sourceTokenOffset, tokenCount, headDim, bytes);
    auto unmap = queue.enqueueUnmapMemObject(buffer, ptr);
    if (!ok) {
        return SourceCLReadStatus::ReadFailed;
    }
    return unmap == CL_SUCCESS ? SourceCLReadStatus::Success : SourceCLReadStatus::MapFailed;
}

static bool _readExternalValueSegmentToCLBufferFallback(
    const std::string& path, cl::Buffer& buffer, size_t expectedBytes, cl::CommandQueue& queue, int batch,
    int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount, int headDim, int bytes) {
    std::vector<int8_t> data(expectedBytes);
    if (!_readExternalValueSegment(path, data.data(), expectedBytes, batch, kvHeads, sourceTokenCount,
                                   sourceTokenOffset, tokenCount, headDim, bytes)) {
        return false;
    }
    return queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, expectedBytes, data.data()) == CL_SUCCESS;
}

static bool _readExternalValueSegmentToSourceCLBuffer(
    const std::string& path, cl::Buffer& buffer, size_t expectedBytes, cl::CommandQueue& queue, int batch,
    int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount, int headDim, int bytes) {
    auto status = _readExternalValueSegmentToMappedCLBuffer(path, buffer, expectedBytes, queue, batch, kvHeads,
                                                            sourceTokenCount, sourceTokenOffset, tokenCount, headDim,
                                                            bytes);
    if (status == SourceCLReadStatus::Success) {
        return true;
    }
    if (status == SourceCLReadStatus::ReadFailed) {
        return false;
    }
    static std::once_flag fallbackOnce;
    std::call_once(fallbackOnce, []() {
        MNN_PRINT("OpenCLPagedAttention: mapped value segment CL buffer unavailable, fallback to enqueueWriteBuffer\n");
    });
    return _readExternalValueSegmentToCLBufferFallback(path, buffer, expectedBytes, queue, batch, kvHeads,
                                                       sourceTokenCount, sourceTokenOffset, tokenCount, headDim,
                                                       bytes);
}

static int _ropeDimForExport(const PagedKVMeta* meta, int headDim) {
    if (meta == nullptr || meta->rope_dim <= 0) {
        return headDim;
    }
    return std::min(headDim, meta->rope_dim);
}

static float _ropeInvFreq(float theta, const std::string& ropeType, float factor, float lowFreqFactor,
                          float highFreqFactor, int oldContext, int pairIndex, int ropeDim) {
    float invFreq = std::pow(theta, -static_cast<float>(2 * pairIndex) / static_cast<float>(ropeDim));
    if (ropeType != "llama3") {
        return invFreq;
    }
    factor = std::max(factor, 1.0f);
    lowFreqFactor = std::max(lowFreqFactor, 1.0e-6f);
    highFreqFactor = std::max(highFreqFactor, 1.0e-6f);
    if (oldContext <= 0 || factor == 1.0f || lowFreqFactor == highFreqFactor) {
        return invFreq;
    }
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float wavelen = kTwoPi / invFreq;
    const float lowFreqWavelen = static_cast<float>(oldContext) / lowFreqFactor;
    const float highFreqWavelen = static_cast<float>(oldContext) / highFreqFactor;
    float scaled = wavelen > lowFreqWavelen ? invFreq / factor : invFreq;
    if (wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen) {
        const float smooth = (static_cast<float>(oldContext) / wavelen - lowFreqFactor) /
                             (highFreqFactor - lowFreqFactor);
        scaled = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
    }
    return scaled;
}

static float _ropeInvFreqForExport(const PagedKVMeta* meta, int pairIndex, int ropeDim) {
    const float theta = (meta != nullptr && meta->rope_theta > 0.0f) ? meta->rope_theta : 10000.0f;
    return _ropeInvFreq(theta, meta != nullptr ? meta->rope_type : "default",
                        meta != nullptr ? meta->rope_scaling_factor : 1.0f,
                        meta != nullptr ? meta->rope_scaling_low_freq_factor : 1.0f,
                        meta != nullptr ? meta->rope_scaling_high_freq_factor : 4.0f,
                        meta != nullptr && meta->rope_scaling_original_max_position_embeddings > 0
                            ? meta->rope_scaling_original_max_position_embeddings
                            : (meta != nullptr ? meta->max_position_embeddings : 0),
                        pairIndex, ropeDim);
}

static void _inverseRopeKeyData(std::vector<int8_t>& keyData, int tokenCount, int batch, int kvHeads, int headDim,
                                int bytes, const PagedKVMeta* meta) {
    int ropeDim = (_ropeDimForExport(meta, headDim) / 2) * 2;
    if (ropeDim <= 0) {
        return;
    }
    const int half = ropeDim / 2;
    const float invAttentionScale = (meta != nullptr && meta->rope_attention_scaling > 0.0f)
        ? (1.0f / meta->rope_attention_scaling)
        : 1.0f;
    for (int l = 0; l < tokenCount; ++l) {
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < kvHeads; ++h) {
                int base = ((l * batch + b) * kvHeads + h) * headDim;
                for (int p = 0; p < half; ++p) {
                    const float angle = static_cast<float>(l) * _ropeInvFreqForExport(meta, p, ropeDim);
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const int first = base + p;
                    const int second = base + p + half;
                    const float y0 = _readScalar(keyData.data(), first, bytes);
                    const float y1 = _readScalar(keyData.data(), second, bytes);
                    _writeScalar(keyData.data(), first, (y0 * c + y1 * s) * invAttentionScale, bytes);
                    _writeScalar(keyData.data(), second, (y1 * c - y0 * s) * invAttentionScale, bytes);
                }
            }
        }
    }
}

static bool _writeShapeFile(const std::string& path, int batch, int kvHeads, int headDim, int tokenCount, int bytes,
                            const PagedKVMeta* meta) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    int ropeDim = _ropeDimForExport(meta, headDim);
    os << "{\n"
       << "  \"format\": \"mnn-paged-attention-kv-shape-v1\",\n"
       << "  \"batch\": " << batch << ",\n"
       << "  \"kv_heads\": " << kvHeads << ",\n"
       << "  \"head_dim\": " << headDim << ",\n"
       << "  \"token_count\": " << tokenCount << ",\n"
       << "  \"dtype_bytes\": " << bytes << ",\n"
       << "  \"key_layout\": \"[token,batch,kv_head,head_dim]\",\n"
       << "  \"value_layout\": \"[batch,kv_head,token,head_dim]\",\n"
       << "  \"key_rope_state\": \"canonical_no_rope\",\n"
       << "  \"rope_pairing\": \"half\",\n"
       << "  \"rope_theta\": " << ((meta != nullptr && meta->rope_theta > 0.0f) ? meta->rope_theta : 10000.0f) << ",\n"
       << "  \"rope_dim\": " << ropeDim << ",\n"
       << "  \"rope_type\": \"" << ((meta != nullptr && !meta->rope_type.empty()) ? meta->rope_type : "default") << "\",\n"
       << "  \"rope_scaling_factor\": " << (meta != nullptr ? meta->rope_scaling_factor : 1.0f) << ",\n"
       << "  \"rope_scaling_low_freq_factor\": " << (meta != nullptr ? meta->rope_scaling_low_freq_factor : 1.0f) << ",\n"
       << "  \"rope_scaling_high_freq_factor\": " << (meta != nullptr ? meta->rope_scaling_high_freq_factor : 4.0f) << ",\n"
       << "  \"rope_scaling_original_max_position_embeddings\": "
       << (meta != nullptr ? meta->rope_scaling_original_max_position_embeddings : 0) << ",\n"
       << "  \"max_position_embeddings\": " << (meta != nullptr ? meta->max_position_embeddings : 0) << "\n"
       << "}\n";
    return os.good();
}

} // namespace

PagedAttentionBufExecution::PagedAttentionBufExecution(const MNN::Op* op, Backend* backend)
    : CommonExecution(backend, op), mOpenCLBackend(static_cast<OpenCLBackend*>(backend)) {
    auto param = op->main_as_AttentionParam();
    if (param != nullptr) {
        mLayerIndex = param->layer_index();
        mKVSharedLayerIndex = param->kv_shared_layer_index();
        mIsKVShared = mKVSharedLayerIndex >= 0;
    }
    mMeta = static_cast<PagedKVMeta*>(backend->getMetaPtr());
    mCache.reset(new SharedPagedCache);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mCopyKernel = runtime->buildKernel("paged_attention_buf", "copy_paged_kv", {}, mOpenCLBackend->getPrecision());
    mAttentionKernel = runtime->buildKernel("paged_attention_buf", "paged_attention", {}, mOpenCLBackend->getPrecision());
    mAttentionRowKernel = runtime->buildKernel("paged_attention_buf", "paged_attention_row", {},
                                               mOpenCLBackend->getPrecision());
    mPackPagedKVKernel = runtime->buildKernel("paged_attention_buf", "pack_paged_kv_prefill", {},
                                              mOpenCLBackend->getPrecision());
    mHydrateExternalKernel = runtime->buildKernel("paged_attention_buf", "pic_page_attention_hydrate_kv", {},
                                                  mOpenCLBackend->getPrecision());
    mCacheBlendScoreKernel = runtime->buildKernel("paged_attention_buf", "pic_cacheblend_value_score", {},
                                                  mOpenCLBackend->getPrecision());
    mCacheBlendTopKKernel = runtime->buildKernel("paged_attention_buf", "pic_cacheblend_topk", {},
                                                 mOpenCLBackend->getPrecision());
    mRearrangeQKernel = runtime->buildKernel("attention_buf", "rearrange_q", {}, mOpenCLBackend->getPrecision());
    mRearrangeMaskKernel = runtime->buildKernel("attention_buf", "rearrange_mask_shortprefill", {"-DADD_MASK"},
                                                mOpenCLBackend->getPrecision());
    mSoftmaxKernel = runtime->buildKernel("softmax_buf", "softmax_v4_buf", {"-DSOFTMAX_LOCAL_SIZE=64"},
                                          mOpenCLBackend->getPrecision());
    mZeroKernel = runtime->buildKernel("paged_attention_buf", "zero_output", {}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mCopyKernel);
    OPENCL_CHECK_KERNEL_CTOR(mAttentionKernel);
    OPENCL_CHECK_KERNEL_CTOR(mAttentionRowKernel);
    OPENCL_CHECK_KERNEL_CTOR(mPackPagedKVKernel);
    OPENCL_CHECK_KERNEL_CTOR(mHydrateExternalKernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendScoreKernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendTopKKernel);
    OPENCL_CHECK_KERNEL_CTOR(mRearrangeQKernel);
    OPENCL_CHECK_KERNEL_CTOR(mRearrangeMaskKernel);
    OPENCL_CHECK_KERNEL_CTOR(mSoftmaxKernel);
    OPENCL_CHECK_KERNEL_CTOR(mZeroKernel);
}

ErrorCode PagedAttentionBufExecution::ensureCache(int maxSlots, int batch, int kvHeads, int headDim) {
    if (maxSlots <= 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0) {
        return INVALID_VALUE;
    }
    if (mCache && mCache->key && mCache->value && mCache->slotTable && mCache->sparseQuery &&
        mCache->maxSlots == maxSlots && mCache->batch == batch && mCache->kvHeads == kvHeads &&
        mCache->headDim == headDim && mCache->bytes == mBytes) {
        return NO_ERROR;
    }
    if (!mCache) {
        mCache.reset(new SharedPagedCache);
    }
    if (mBytes == 4) {
        mCache->key.reset(Tensor::createDevice<float>({maxSlots, batch, kvHeads, headDim}));
        mCache->value.reset(Tensor::createDevice<float>({batch, kvHeads, maxSlots, headDim}));
    } else {
        mCache->key.reset(Tensor::createDevice<uint16_t>({maxSlots, batch, kvHeads, headDim}));
        mCache->value.reset(Tensor::createDevice<uint16_t>({batch, kvHeads, maxSlots, headDim}));
    }
    mCache->slotTable.reset(Tensor::createDevice<int>({maxSlots}));
    mCache->sparseQuery.reset(Tensor::createDevice<int>({maxSlots}));
    if (!mCache->key || !mCache->value || !mCache->slotTable || !mCache->sparseQuery) {
        return OUT_OF_MEMORY;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->key.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->value.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->slotTable.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->sparseQuery.get(), Backend::STATIC));
    const size_t keyElements = static_cast<size_t>(maxSlots) * batch * kvHeads * headDim;
    const size_t valueElements = static_cast<size_t>(batch) * kvHeads * maxSlots * headDim;
    if (keyElements > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        valueElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return OUT_OF_MEMORY;
    }
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mZeroKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mZeroKernel->get().setArg(idx++, static_cast<int>(keyElements));
    MNN_CHECK_CL_SUCCESS(ret, "setArg zero_paged_key_cache");
    queue.enqueueNDRangeKernel(mZeroKernel->get(), cl::NullRange, cl::NDRange(keyElements), cl::NullRange);
    idx = 0;
    ret = CL_SUCCESS;
    ret |= mZeroKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mZeroKernel->get().setArg(idx++, static_cast<int>(valueElements));
    MNN_CHECK_CL_SUCCESS(ret, "setArg zero_paged_value_cache");
    queue.enqueueNDRangeKernel(mZeroKernel->get(), cl::NullRange, cl::NDRange(valueElements), cl::NullRange);
    mCache->maxSlots = maxSlots;
    mCache->batch = batch;
    mCache->kvHeads = kvHeads;
    mCache->headDim = headDim;
    mCache->bytes = mBytes;
    mCache->slotTableVersion = -1;
    mCache->slotTableLength = 0;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::syncSlotTable(int requiredSlots) {
    if (!mCache || !mCache->slotTable || requiredSlots > mCache->maxSlots) {
        return OUT_OF_MEMORY;
    }
    std::vector<int> identity;
    const int* hostPtr = nullptr;
    int version = 0;
    if (mMeta != nullptr) {
        if (!mMeta->ensureLogicalCapacity(requiredSlots)) {
            return OUT_OF_MEMORY;
        }
        hostPtr = mMeta->slot_table_host.data();
        version = mMeta->slot_table_version;
    } else {
        identity.resize(requiredSlots);
        for (int i = 0; i < requiredSlots; ++i) {
            identity[i] = i;
        }
        hostPtr = identity.data();
    }
    if (mCache->slotTableVersion == version && mCache->slotTableLength >= requiredSlots && mMeta != nullptr) {
        return NO_ERROR;
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
        openCLBuffer(mCache->slotTable.get()), CL_TRUE, 0, requiredSlots * sizeof(int), hostPtr);
    mCache->slotTableVersion = version;
    mCache->slotTableLength = requiredSlots;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::syncSparseQuery(int insertLen) {
    if (insertLen <= 0 || mMeta == nullptr || !mMeta->sparse_query_active) {
        return NO_ERROR;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < insertLen) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
        openCLBuffer(mCache->sparseQuery.get()), CL_TRUE, 0, insertLen * sizeof(int),
        mMeta->sparse_query_logical_indices.data());
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureFastPrefillTemps(int seqLen, int kvLen) {
    if (seqLen <= 0 || kvLen <= 0) {
        return INVALID_VALUE;
    }
    if (mTempQ && mTempK && mTempV && mTempMask && mTempQK && mTempSoftmax &&
        mFastSeqLen == seqLen && mFastKvLen == kvLen) {
        return NO_ERROR;
    }
    const int seqPack = ROUND_UP(seqLen, 4);
    const int kvPack = ROUND_UP(kvLen, 4);
    const int headPack4 = ROUND_UP(mHeadDim, 4);
    const int headPack8 = ROUND_UP(mHeadDim, 8);
    mTempQ.reset(Tensor::createDevice<float>({seqPack * headPack4 * mNumHead * mBatch}));
    mTempK.reset(Tensor::createDevice<float>({kvPack * headPack4 * mKvNumHead * mBatch}));
    mTempV.reset(Tensor::createDevice<float>({kvPack * headPack8 * mKvNumHead * mBatch}));
    mTempMask.reset(Tensor::createDevice<float>({seqPack * kvPack * mBatch}));
    mTempQK.reset(Tensor::createDevice<float>({seqPack * kvLen * mNumHead * mBatch}));
    mTempSoftmax.reset(Tensor::createDevice<float>({seqPack * kvLen * mNumHead * mBatch}));
    if (!mTempQ || !mTempK || !mTempV || !mTempMask || !mTempQK || !mTempSoftmax) {
        return OUT_OF_MEMORY;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempSoftmax.get(), Backend::STATIC));
    mFastSeqLen = seqLen;
    mFastKvLen = kvLen;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureExternalTemps(size_t keyElements, size_t valueElements) {
    if (keyElements == 0 || valueElements == 0) {
        return INVALID_VALUE;
    }
    if (keyElements > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        valueElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return OUT_OF_MEMORY;
    }
    if (mExternalKey && mExternalValue && mExternalKeyElements >= keyElements &&
        mExternalValueElements >= valueElements) {
        return NO_ERROR;
    }
    if (mBytes == 2) {
        mExternalKey.reset(Tensor::createDevice<uint16_t>({static_cast<int>(keyElements)}));
        mExternalValue.reset(Tensor::createDevice<uint16_t>({static_cast<int>(valueElements)}));
    } else {
        mExternalKey.reset(Tensor::createDevice<float>({static_cast<int>(keyElements)}));
        mExternalValue.reset(Tensor::createDevice<float>({static_cast<int>(valueElements)}));
    }
    if (!mExternalKey || !mExternalValue) {
        return OUT_OF_MEMORY;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mExternalKey.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mExternalValue.get(), Backend::STATIC));
    mExternalKeyElements = keyElements;
    mExternalValueElements = valueElements;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureCacheBlendScoreTemps(int scoreCount, int indexCount) {
    if (scoreCount < 0 || indexCount < 0) {
        return INVALID_VALUE;
    }
    if (scoreCount == 0 && indexCount == 0) {
        return NO_ERROR;
    }
    if (mCacheBlendScores && mCacheBlendIndices && mCacheBlendScoreCount >= scoreCount &&
        mCacheBlendIndexCount >= indexCount) {
        return NO_ERROR;
    }
    if (scoreCount > 0) {
        mCacheBlendScores.reset(Tensor::createDevice<float>({scoreCount}));
        if (!mCacheBlendScores) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCacheBlendScores.get(), Backend::STATIC));
    }
    if (indexCount > 0) {
        mCacheBlendIndices.reset(Tensor::createDevice<int>({indexCount}));
        if (!mCacheBlendIndices) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCacheBlendIndices.get(), Backend::STATIC));
    }
    mCacheBlendScoreCount = scoreCount;
    mCacheBlendIndexCount = indexCount;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::hydrateExternalSegments(int layerIndex, int kvLen) {
    if (mMeta == nullptr || mMeta->external_segments.empty() || mMeta->externalLayerLoaded(layerIndex)) {
        return NO_ERROR;
    }
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    size_t totalTokens = 0;
    _prefetchExternalLayersFrom(mMeta, layerIndex + 1);
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    for (const auto& segment : mMeta->external_segments) {
        totalTokens += segment.tokenCount;
        if (segment.tokenCount == 0) {
            continue;
        }
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            MNN_ERROR("OpenCLPagedAttention layer %d missing external PIC KV for cache %s\n", layerIndex,
                      segment.cacheName.c_str());
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : mBatch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : mKvNumHead;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : mHeadDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : mBytes;
        if (segBatch != mBatch || segKvHeads != mKvNumHead || segHeadDim != mHeadDim || segBytes != mBytes) {
            return INVALID_VALUE;
        }
        if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
            return INVALID_VALUE;
        }
        if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
            return INVALID_VALUE;
        }
        const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset
                                                                  : segment.sourceTokenOffset;
        const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
            ? layer->sourceTokenCount
            : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
        const size_t sourceEnd = sourceTokenOffset + segment.tokenCount;
        if (sourceEnd > sourceTokenCount) {
            return INVALID_VALUE;
        }
        const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
        if (segment.logicalStart > maxInt || segment.tokenCount > maxInt || sourceTokenOffset > maxInt ||
            sourceTokenCount > maxInt) {
            MNN_ERROR("OpenCLPagedAttention external PIC KV indices exceed int range at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        const size_t keyTokenBytes = static_cast<size_t>(mBatch) * mKvNumHead * mHeadDim * mBytes;
        const size_t keySourceOffsetBytes = sourceTokenOffset * keyTokenBytes;
        const size_t keySegmentBytes = segment.tokenCount * keyTokenBytes;
        const size_t valueTokenBytes = static_cast<size_t>(mHeadDim) * mBytes;
        const size_t valueSegmentBytes = static_cast<size_t>(mBatch) * mKvNumHead * segment.tokenCount *
                                         valueTokenBytes;
        auto err = ensureExternalTemps(keySegmentBytes / mBytes, valueSegmentBytes / mBytes);
        if (err != NO_ERROR) {
            return err;
        }
        auto& externalKeyBuffer = openCLBuffer(mExternalKey.get());
        auto& externalValueBuffer = openCLBuffer(mExternalValue.get());
        if (!_readBinaryFileRangeToSourceCLBuffer(layer->keyPath, keySourceOffsetBytes, externalKeyBuffer,
                                                  keySegmentBytes, queue) ||
            !_readExternalValueSegmentToSourceCLBuffer(layer->valuePath, externalValueBuffer, valueSegmentBytes,
                                                       queue, mBatch, mKvNumHead, sourceTokenCount,
                                                       sourceTokenOffset, segment.tokenCount, mHeadDim, mBytes)) {
            return INVALID_VALUE;
        }

        int ropeDim = segment.ropeDim > 0 ? segment.ropeDim : mHeadDim;
        ropeDim = std::min(ropeDim, mHeadDim);
        ropeDim = (ropeDim / 2) * 2;
        const int ropeTypeLlama3 = segment.ropeType == "llama3" ? 1 : 0;
        const int oldContext = segment.ropeScalingOriginalMaxPositionEmbeddings > 0
            ? segment.ropeScalingOriginalMaxPositionEmbeddings
            : segment.maxPositionEmbeddings;
        const size_t totalElements = segment.tokenCount * static_cast<size_t>(mBatch) * mKvNumHead * mHeadDim;
        if (totalElements > maxInt) {
            MNN_ERROR("OpenCLPagedAttention external PIC KV hydrate element count exceeds int range at layer %d\n",
                      layerIndex);
            return INVALID_VALUE;
        }
        const int total = static_cast<int>(totalElements);
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mExternalKey.get()));
        ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mExternalValue.get()));
        ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mHydrateExternalKernel->get().setArg(idx++, mBatch);
        ret |= mHydrateExternalKernel->get().setArg(idx++, mKvNumHead);
        ret |= mHydrateExternalKernel->get().setArg(idx++, mHeadDim);
        ret |= mHydrateExternalKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mHydrateExternalKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
        ret |= mHydrateExternalKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
        ret |= mHydrateExternalKernel->get().setArg(idx++, ropeDim);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeTheta);
        ret |= mHydrateExternalKernel->get().setArg(idx++, ropeTypeLlama3);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeScalingFactor);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeScalingLowFreqFactor);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeScalingHighFreqFactor);
        ret |= mHydrateExternalKernel->get().setArg(idx++, oldContext);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.maxPositionEmbeddings);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeAttentionScaling);
        ret |= mHydrateExternalKernel->get().setArg(idx++, total);
        MNN_CHECK_CL_SUCCESS(ret, "setArg pic_page_attention_hydrate_kv");
        queue.enqueueNDRangeKernel(mHydrateExternalKernel->get(), cl::NullRange, cl::NDRange(total), cl::NullRange);
    }
    mMeta->markExternalLayerLoaded(layerIndex);
    _prefetchExternalLayersFrom(mMeta, layerIndex + 1);
    if (profile) {
        queue.finish();
        MNN_PRINT("OpenCLPagedAttention profile op=hydrate layer=%d tokens=%d kv_len=%d us=%llu\n",
                  layerIndex, static_cast<int>(totalTokens), kvLen,
                  static_cast<unsigned long long>(_nowUs() - startUs));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runCacheBlendScoring(int layerIndex, int kvLen) {
    if (mMeta == nullptr || !mMeta->needsCacheBlendScoring(layerIndex)) {
        return NO_ERROR;
    }
    const int picTokenCount = mMeta->cacheblend_score_pic_token_count;
    const int topK = mMeta->cacheblend_score_top_k;
    if (picTokenCount < 0 || topK < 0 || topK > picTokenCount ||
        mMeta->cacheblend_score_pic_start + picTokenCount > kvLen) {
        return INVALID_VALUE;
    }
    if (topK == 0 || picTokenCount == 0) {
        mMeta->setCacheBlendScoringResult({});
        return NO_ERROR;
    }
    auto err = ensureCacheBlendScoreTemps(picTokenCount, topK);
    if (err != NO_ERROR) {
        return err;
    }
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    size_t scoreOffset = 0;
    for (const auto& segment : mMeta->cacheblend_score_segments) {
        if (segment.tokenCount == 0) {
            continue;
        }
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : mBatch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : mKvNumHead;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : mHeadDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : mBytes;
        if (segBatch != mBatch || segKvHeads != mKvNumHead || segHeadDim != mHeadDim || segBytes != mBytes) {
            return INVALID_VALUE;
        }
        const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset
                                                                  : segment.sourceTokenOffset;
        const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
            ? layer->sourceTokenCount
            : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
        const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
        if (sourceTokenOffset + segment.tokenCount > sourceTokenCount ||
            segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen) ||
            scoreOffset + segment.tokenCount > static_cast<size_t>(picTokenCount) ||
            segment.logicalStart > maxInt || segment.tokenCount > maxInt || scoreOffset > maxInt) {
            return INVALID_VALUE;
        }
        const size_t valueTokenBytes = static_cast<size_t>(mHeadDim) * mBytes;
        const size_t valueSegmentBytes = static_cast<size_t>(mBatch) * mKvNumHead * segment.tokenCount *
                                         valueTokenBytes;
        err = ensureExternalTemps(1, valueSegmentBytes / mBytes);
        if (err != NO_ERROR) {
            return err;
        }
        auto& externalValueBuffer = openCLBuffer(mExternalValue.get());
        if (!_readExternalValueSegmentToSourceCLBuffer(layer->valuePath, externalValueBuffer, valueSegmentBytes,
                                                       queue, mBatch, mKvNumHead, sourceTokenCount,
                                                       sourceTokenOffset, segment.tokenCount, mHeadDim, mBytes)) {
            return INVALID_VALUE;
        }
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mExternalValue.get()));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mBatch);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mKvNumHead);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mHeadDim);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(scoreOffset));
        MNN_CHECK_CL_SUCCESS(ret, "setArg pic_cacheblend_value_score");
        ret = queue.enqueueNDRangeKernel(mCacheBlendScoreKernel->get(), cl::NullRange,
                                         cl::NDRange(static_cast<int>(segment.tokenCount)), cl::NullRange);
        MNN_CHECK_CL_SUCCESS(ret, "enqueue pic_cacheblend_value_score");
        scoreOffset += segment.tokenCount;
    }
    if (scoreOffset != static_cast<size_t>(picTokenCount)) {
        return INVALID_VALUE;
    }
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendIndices.get()));
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, picTokenCount);
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, topK);
    MNN_CHECK_CL_SUCCESS(ret, "setArg pic_cacheblend_topk");
    constexpr int topKLocalSize = 64;
    ret = queue.enqueueNDRangeKernel(mCacheBlendTopKKernel->get(), cl::NullRange, cl::NDRange(topKLocalSize),
                                     cl::NDRange(topKLocalSize));
    MNN_CHECK_CL_SUCCESS(ret, "enqueue pic_cacheblend_topk");
    std::vector<int> selected(topK);
    if (queue.enqueueReadBuffer(openCLBuffer(mCacheBlendIndices.get()), CL_TRUE, 0,
                                static_cast<size_t>(topK) * sizeof(int), selected.data()) != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(picTokenCount), 0);
    for (int index : selected) {
        if (index < 0 || index >= picTokenCount || seen[static_cast<size_t>(index)] != 0) {
            return INVALID_VALUE;
        }
        seen[static_cast<size_t>(index)] = 1;
    }
    mMeta->setCacheBlendScoringResult(selected);
    if (_profilePagedAttention()) {
        queue.finish();
        MNN_PRINT("OpenCLPagedAttention profile op=cacheblend_score layer=%d pic_tokens=%d top_k=%d\n",
                  layerIndex, picTokenCount, topK);
    }
    return NO_ERROR;
}

bool PagedAttentionBufExecution::canUseFastPrefill(const Tensor* mask, int baseLogical, int insertLen, int kvLen,
                                                   bool sparseQuery, bool externalHydrated, int* maskKeyLen) const {
    if (maskKeyLen != nullptr) {
        *maskKeyLen = 0;
    }
    if (sparseQuery || mIsKVShared || mMeta == nullptr) {
        return false;
    }
    const bool hasExternal = !mMeta->external_segments.empty();
    if (hasExternal && !externalHydrated) {
        return false;
    }
    if (!hasExternal && (baseLogical != 0 || kvLen != insertLen)) {
        return false;
    }
    if (baseLogical < 0 || insertLen != mQuerySeqLen || insertLen <= 1 || kvLen < insertLen) {
        return false;
    }
    if (mHeadDim <= 0 || mHeadDim % 8 != 0 || mNumHead % mKvNumHead != 0) {
        return false;
    }
    if (mask == nullptr || mask->elementSize() <= 1 || mask->getType().code != halide_type_float) {
        return false;
    }
    const int maskElements = static_cast<int>(mask->elementSize());
    const int64_t fullMaskElements = static_cast<int64_t>(insertLen) * kvLen;
    const int64_t shortMaskElements = static_cast<int64_t>(insertLen) * insertLen;
    if (maskElements >= fullMaskElements) {
        if (maskKeyLen != nullptr) {
            *maskKeyLen = kvLen;
        }
        return true;
    }
    if (hasExternal && maskElements >= shortMaskElements) {
        if (maskKeyLen != nullptr) {
            *maskKeyLen = insertLen;
        }
        return true;
    }
    return false;
}

ErrorCode PagedAttentionBufExecution::runFastPrefill(const std::vector<Tensor*>& inputs,
                                                     const std::vector<Tensor*>& outputs, int kvLen, int maskKeyLen) {
    if (maskKeyLen <= 0 || maskKeyLen > kvLen) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto mask = inputs[3];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    auto err = ensureFastPrefillTemps(mQuerySeqLen, kvLen);
    if (err != NO_ERROR) {
        return err;
    }
    if (!mQKKernel || !mQKVKernel) {
        const int groupSize = mNumHead / mKvNumHead;
        mQKKernel = runtime->buildKernel("attention_buf", "matmul_qk_div_mask_prefill",
                                         {"-DADD_MASK", "-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                         mOpenCLBackend->getPrecision());
        mQKVKernel = runtime->buildKernel("attention_buf", "matmul_qkv_prefill",
                                          {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                          mOpenCLBackend->getPrecision());
        OPENCL_CHECK_KERNEL(mQKKernel);
        OPENCL_CHECK_KERNEL(mQKVKernel);
    }
    auto run3D = [&](const std::shared_ptr<KernelWrap>& kernel, std::vector<uint32_t> gws,
                    const std::string& kernelName, const std::string& programName) {
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(kernel));
        auto lws = localWS3DDefault(gws, maxWorkGroupSize, runtime, kernelName, kernel,
                                    mOpenCLBackend->getCLTuneLevel(), programName).first;
        gws[0] = ROUND_UP(gws[0], std::max((uint32_t)1, lws[0]));
        gws[1] = ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]));
        gws[2] = ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]));
        run3DKernelDefault(kernel, gws, lws, runtime);
    };

    const int seqPack = ROUND_UP(mQuerySeqLen, 4);
    const int kvPack = ROUND_UP(kvLen, 4);
    const int headPack4 = ROUND_UP(mHeadDim, 4);
    const int headPack8 = ROUND_UP(mHeadDim, 8);
    const float scale = (mMeta && mMeta->attn_scale > 0)
        ? mMeta->attn_scale
        : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    std::vector<uint32_t> gws;

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mNumHead * mBatch)};
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[0]);
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[1]);
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[2]);
    ret |= mRearrangeQKernel->get().setArg(idx++, openCLBuffer(query));
    ret |= mRearrangeQKernel->get().setArg(idx++, openCLBuffer(mTempQ.get()));
    ret |= mRearrangeQKernel->get().setArg(idx++, mQuerySeqLen);
    ret |= mRearrangeQKernel->get().setArg(idx++, mHeadDim);
    ret |= mRearrangeQKernel->get().setArg(idx++, mNumHead);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast rearrange_q");
    run3D(mRearrangeQKernel, gws, "rearrange_q", "attention_buf");

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(kvLen, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mKvNumHead * mBatch)};
    ret = CL_SUCCESS;
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[0]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[1]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[2]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mTempK.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mTempV.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, mBatch);
    ret |= mPackPagedKVKernel->get().setArg(idx++, kvLen);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mKvNumHead);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mHeadDim);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mCache->maxSlots);
    MNN_CHECK_CL_SUCCESS(ret, "setArg pack_paged_kv_prefill");
    run3D(mPackPagedKVKernel, gws, "pack_paged_kv_prefill", "paged_attention_buf");

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)), static_cast<uint32_t>(UP_DIV(maskKeyLen, 4)),
           static_cast<uint32_t>(mBatch)};
    ret = CL_SUCCESS;
    ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[0]);
    ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[1]);
    ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[2]);
    ret |= mRearrangeMaskKernel->get().setArg(idx++, openCLBuffer(mask));
    ret |= mRearrangeMaskKernel->get().setArg(idx++, openCLBuffer(mTempMask.get()));
    ret |= mRearrangeMaskKernel->get().setArg(idx++, mQuerySeqLen);
    ret |= mRearrangeMaskKernel->get().setArg(idx++, maskKeyLen);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast rearrange_mask_shortprefill");
    run3D(mRearrangeMaskKernel, gws, "rearrange_mask_shortprefill", "attention_buf");

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)), static_cast<uint32_t>(UP_DIV(kvLen, 4)),
           static_cast<uint32_t>(mNumHead * mBatch)};
    ret = CL_SUCCESS;
    ret |= mQKKernel->get().setArg(idx++, gws[0]);
    ret |= mQKKernel->get().setArg(idx++, gws[1]);
    ret |= mQKKernel->get().setArg(idx++, gws[2]);
    ret |= mQKKernel->get().setArg(idx++, openCLBuffer(mTempQ.get()));
    ret |= mQKKernel->get().setArg(idx++, openCLBuffer(mTempK.get()));
    ret |= mQKKernel->get().setArg(idx++, openCLBuffer(mTempMask.get()));
    ret |= mQKKernel->get().setArg(idx++, openCLBuffer(mTempQK.get()));
    ret |= mQKKernel->get().setArg(idx++, scale);
    ret |= mQKKernel->get().setArg(idx++, mQuerySeqLen);
    ret |= mQKKernel->get().setArg(idx++, maskKeyLen);
    ret |= mQKKernel->get().setArg(idx++, kvLen);
    ret |= mQKKernel->get().setArg(idx++, kvPack);
    ret |= mQKKernel->get().setArg(idx++, mNumHead);
    ret |= mQKKernel->get().setArg(idx++, headPack4);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast matmul_qk_div_mask_prefill");
    run3D(mQKKernel, gws, "matmul_qk_div_mask_prefill", "attention_buf");

    idx = 0;
    gws = {64u, static_cast<uint32_t>(UP_DIV(seqPack, 4)), static_cast<uint32_t>(mNumHead * mBatch)};
    ret = CL_SUCCESS;
    ret |= mSoftmaxKernel->get().setArg(idx++, gws[0]);
    ret |= mSoftmaxKernel->get().setArg(idx++, gws[1]);
    ret |= mSoftmaxKernel->get().setArg(idx++, gws[2]);
    ret |= mSoftmaxKernel->get().setArg(idx++, openCLBuffer(mTempQK.get()));
    ret |= mSoftmaxKernel->get().setArg(idx++, openCLBuffer(mTempSoftmax.get()));
    ret |= mSoftmaxKernel->get().setArg(idx++, seqPack);
    ret |= mSoftmaxKernel->get().setArg(idx++, mNumHead * mBatch);
    ret |= mSoftmaxKernel->get().setArg(idx++, kvLen);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast softmax");
    run3DKernelDefault(mSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(mHeadDim, 8)), static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)),
           static_cast<uint32_t>(mNumHead * mBatch)};
    ret = CL_SUCCESS;
    ret |= mQKVKernel->get().setArg(idx++, gws[0]);
    ret |= mQKVKernel->get().setArg(idx++, gws[1]);
    ret |= mQKVKernel->get().setArg(idx++, gws[2]);
    ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(mTempSoftmax.get()));
    ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(mTempV.get()));
    ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(output));
    ret |= mQKVKernel->get().setArg(idx++, mQuerySeqLen);
    ret |= mQKVKernel->get().setArg(idx++, kvLen);
    ret |= mQKVKernel->get().setArg(idx++, kvPack);
    ret |= mQKVKernel->get().setArg(idx++, mNumHead);
    ret |= mQKVKernel->get().setArg(idx++, mKvNumHead);
    ret |= mQKVKernel->get().setArg(idx++, headPack8);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast matmul_qkv_prefill");
    run3D(mQKVKernel, gws, "matmul_qkv_prefill", "attention_buf");
    if (profile) {
        runtime->commandQueue().finish();
        int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
        MNN_PRINT("OpenCLPagedAttention profile op=fast_prefill layer=%d query=%d kv_len=%d mask_key_len=%d us=%llu\n",
                  layerIndex, mQuerySeqLen, kvLen, maskKeyLen,
                  static_cast<unsigned long long>(_nowUs() - startUs));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 3 || outputs.empty()) {
        return INVALID_VALUE;
    }
    mBytes = mOpenCLBackend->getPrecision() == BackendConfig::Precision_High ? 4 : 2;
    auto query = inputs[0];
    auto key = inputs[1];
    mBatch = query->length(0);
    mQuerySeqLen = query->length(1);
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    mNewKvSeqLen = key->length(1);
    if (mHeadDim <= 0 || mKvNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    mQKKernel.reset();
    mQKVKernel.reset();
    int maxSlots = mNewKvSeqLen;
    if (mMeta != nullptr) {
        maxSlots = std::max(maxSlots, mMeta->request_capacity > 0 ? mMeta->request_capacity : mMeta->max_tokens);
        if (maxSlots <= 0) {
            maxSlots = static_cast<int>(mMeta->previous) + mNewKvSeqLen;
        }
    }
    return ensureCache(maxSlots, mBatch, mKvNumHead, mHeadDim);
}

ErrorCode PagedAttentionBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const Tensor* mask = inputs.size() > 3 ? inputs[3] : nullptr;

    if (mMeta != nullptr && mMeta->request_capacity <= 0 && !mMeta->request_active) {
        mMeta->beginRequest(std::max(mNewKvSeqLen, mQuerySeqLen));
    }
    int reverse = _reverseCount(mMeta);
    int baseLogical = 0;
    int insertLen = mNewKvSeqLen;
    bool sparseQuery = mMeta != nullptr && mMeta->sparse_query_active;
    if (mMeta != nullptr) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        baseLogical = static_cast<int>(kept) + reverse;
        insertLen = mMeta->add > 0 ? static_cast<int>(std::min<size_t>(mMeta->add, mNewKvSeqLen)) : mNewKvSeqLen;
    }
    insertLen = std::min(insertLen, mQuerySeqLen);
    if (sparseQuery) {
        if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < insertLen) {
            return INVALID_VALUE;
        }
        baseLogical = 0;
    }
    int kvLen = sparseQuery ? std::max(0, mMeta->logical_length) : (baseLogical + insertLen);
    if (kvLen > mCache->maxSlots) {
        return OUT_OF_MEMORY;
    }
    auto err = syncSlotTable(kvLen);
    if (err != NO_ERROR) {
        return err;
    }
    err = syncSparseQuery(insertLen);
    if (err != NO_ERROR) {
        return err;
    }

    std::vector<int> physicalSlots(kvLen);
    for (int l = 0; l < kvLen; ++l) {
        physicalSlots[l] = mMeta ? mMeta->physicalSlot(l) : l;
    }
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : 0);
    if (mMeta != nullptr && !mMeta->external_segments.empty() && !mMeta->externalLayerLoaded(layerIndex)) {
        auto hydrate = hydrateExternalSegments(layerIndex, kvLen);
        if (hydrate != NO_ERROR) {
            return hydrate;
        }
    }

    if (!mIsKVShared && insertLen > 0) {
        const int total = mBatch * insertLen * mKvNumHead * mHeadDim;
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(key));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(value));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mCopyKernel->get().setArg(idx++, mBatch);
        ret |= mCopyKernel->get().setArg(idx++, mNewKvSeqLen);
        ret |= mCopyKernel->get().setArg(idx++, insertLen);
        ret |= mCopyKernel->get().setArg(idx++, mKvNumHead);
        ret |= mCopyKernel->get().setArg(idx++, mHeadDim);
        ret |= mCopyKernel->get().setArg(idx++, baseLogical);
        ret |= mCopyKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mCopyKernel->get().setArg(idx++, sparseQuery ? 1 : 0);
        ret |= mCopyKernel->get().setArg(idx++, total);
        MNN_CHECK_CL_SUCCESS(ret, "setArg copy_paged_kv");
        queue.enqueueNDRangeKernel(mCopyKernel->get(), cl::NullRange, cl::NDRange(total), cl::NullRange);
    }

    err = runCacheBlendScoring(layerIndex, kvLen);
    if (err != NO_ERROR) {
        return err;
    }

    if (mMeta != nullptr && !mMeta->file_name.empty() && mMeta->file_flag == KVMeta::PendingWrite && kvLen > 0) {
        auto prefixDir = mOpenCLBackend->getRuntime()->hint().prefixcacheDirPath;
        MNNCreateDir(prefixDir.c_str());
        std::string basePath = MNNFilePathConcat(prefixDir, mMeta->file_name) + "_" + std::to_string(layerIndex);
        const size_t keyBytes = static_cast<size_t>(mCache->maxSlots) * mBatch * mKvNumHead * mHeadDim * mBytes;
        const size_t valueBytes = static_cast<size_t>(mBatch) * mKvNumHead * mCache->maxSlots * mHeadDim * mBytes;
        std::vector<int8_t> keyStorage(keyBytes);
        std::vector<int8_t> valueStorage(valueBytes);
        queue.enqueueReadBuffer(openCLBuffer(mCache->key.get()), CL_TRUE, 0, keyBytes, keyStorage.data());
        queue.enqueueReadBuffer(openCLBuffer(mCache->value.get()), CL_TRUE, 0, valueBytes, valueStorage.data());
        std::vector<int8_t> keyData(static_cast<size_t>(kvLen) * mBatch * mKvNumHead * mHeadDim * mBytes);
        std::vector<int8_t> valueData(static_cast<size_t>(mBatch) * mKvNumHead * kvLen * mHeadDim * mBytes);
        for (int l = 0; l < kvLen; ++l) {
            int slot = physicalSlots[l];
            if (slot < 0 || slot >= mCache->maxSlots) {
                continue;
            }
            for (int b = 0; b < mBatch; ++b) {
                for (int h = 0; h < mKvNumHead; ++h) {
                    const int8_t* srcK = keyStorage.data() + ((slot * mBatch + b) * mKvNumHead + h) * mHeadDim * mBytes;
                    int8_t* dstK = keyData.data() + ((l * mBatch + b) * mKvNumHead + h) * mHeadDim * mBytes;
                    ::memcpy(dstK, srcK, mHeadDim * mBytes);
                    const int8_t* srcV = valueStorage.data() + ((b * mKvNumHead + h) * mCache->maxSlots + slot) * mHeadDim * mBytes;
                    int8_t* dstV = valueData.data() + ((b * mKvNumHead + h) * kvLen + l) * mHeadDim * mBytes;
                    ::memcpy(dstV, srcV, mHeadDim * mBytes);
                }
            }
        }
        _inverseRopeKeyData(keyData, kvLen, mBatch, mKvNumHead, mHeadDim, mBytes, mMeta);
        if (!_writeBinaryFile(basePath + ".k", keyData)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export key cache: %s\n", (basePath + ".k").c_str());
        }
        if (!_writeBinaryFile(basePath + ".v", valueData)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export value cache: %s\n", (basePath + ".v").c_str());
        }
        if (!_writeShapeFile(basePath + ".json", mBatch, mKvNumHead, mHeadDim, kvLen, mBytes, mMeta)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export shape metadata: %s\n", (basePath + ".json").c_str());
        }
        if (mLayerIndex < 0) {
            mMeta->layer_index = (mMeta->layer_index + 1) % std::max(1, mMeta->layer_nums);
        }
    }

    if (insertLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }
    mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    bool useMask = mask != nullptr && mask->elementSize() > 1 && mask->getType().code == halide_type_float;
    int maskElements = useMask ? static_cast<int>(mask->elementSize()) : 0;
    bool externalHydrated = mMeta == nullptr || mMeta->external_segments.empty() || mMeta->externalLayerLoaded(layerIndex);
    int fastMaskKeyLen = 0;
    if (canUseFastPrefill(mask, baseLogical, insertLen, kvLen, sparseQuery, externalHydrated, &fastMaskKeyLen)) {
        return runFastPrefill(inputs, outputs, kvLen, fastMaskKeyLen);
    }
    const bool profileFallback = _profilePagedAttention();
    const uint64_t fallbackStartUs = profileFallback ? _nowUs() : 0;
    const int outputElements = mBatch * mQuerySeqLen * mNumHead * mHeadDim;
    if (outputElements > 0) {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mZeroKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mZeroKernel->get().setArg(idx++, outputElements);
        MNN_CHECK_CL_SUCCESS(ret, "setArg zero_output");
        queue.enqueueNDRangeKernel(mZeroKernel->get(), cl::NullRange, cl::NDRange(outputElements), cl::NullRange);
    }
    if (insertLen >= 1 && mHeadDim <= 256) {
        const int totalRows = mBatch * insertLen * mNumHead;
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mAttentionRowKernel->get().setArg(idx++, useMask ? openCLBuffer(mask) : openCLBuffer(output));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mAttentionRowKernel->get().setArg(idx++, maskElements);
        ret |= mAttentionRowKernel->get().setArg(idx++, mBatch);
        ret |= mAttentionRowKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= mAttentionRowKernel->get().setArg(idx++, insertLen);
        ret |= mAttentionRowKernel->get().setArg(idx++, mNumHead);
        ret |= mAttentionRowKernel->get().setArg(idx++, mKvNumHead);
        ret |= mAttentionRowKernel->get().setArg(idx++, mHeadDim);
        ret |= mAttentionRowKernel->get().setArg(idx++, baseLogical);
        ret |= mAttentionRowKernel->get().setArg(idx++, kvLen);
        ret |= mAttentionRowKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mAttentionRowKernel->get().setArg(idx++, mScale);
        ret |= mAttentionRowKernel->get().setArg(idx++, sparseQuery ? 1 : 0);
        ret |= mAttentionRowKernel->get().setArg(idx++, totalRows);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged_attention_row");
        queue.enqueueNDRangeKernel(mAttentionRowKernel->get(), cl::NullRange, cl::NDRange(totalRows), cl::NullRange);
        if (profileFallback) {
            queue.finish();
            MNN_PRINT("OpenCLPagedAttention profile op=row layer=%d query=%d insert=%d kv_len=%d sparse=%d us=%llu\n",
                      layerIndex, mQuerySeqLen, insertLen, kvLen, sparseQuery ? 1 : 0,
                      static_cast<unsigned long long>(_nowUs() - fallbackStartUs));
        }
        return NO_ERROR;
    }
    const int total = mBatch * insertLen * mNumHead * mHeadDim;
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(query));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(output));
    ret |= mAttentionKernel->get().setArg(idx++, useMask ? openCLBuffer(mask) : openCLBuffer(output));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
    ret |= mAttentionKernel->get().setArg(idx++, maskElements);
    ret |= mAttentionKernel->get().setArg(idx++, mBatch);
    ret |= mAttentionKernel->get().setArg(idx++, mQuerySeqLen);
    ret |= mAttentionKernel->get().setArg(idx++, insertLen);
    ret |= mAttentionKernel->get().setArg(idx++, mNumHead);
    ret |= mAttentionKernel->get().setArg(idx++, mKvNumHead);
    ret |= mAttentionKernel->get().setArg(idx++, mHeadDim);
    ret |= mAttentionKernel->get().setArg(idx++, baseLogical);
    ret |= mAttentionKernel->get().setArg(idx++, kvLen);
    ret |= mAttentionKernel->get().setArg(idx++, mCache->maxSlots);
    ret |= mAttentionKernel->get().setArg(idx++, mScale);
    ret |= mAttentionKernel->get().setArg(idx++, sparseQuery ? 1 : 0);
    ret |= mAttentionKernel->get().setArg(idx++, total);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged_attention");
    queue.enqueueNDRangeKernel(mAttentionKernel->get(), cl::NullRange, cl::NDRange(total), cl::NullRange);
    if (profileFallback) {
        queue.finish();
        MNN_PRINT("OpenCLPagedAttention profile op=generic layer=%d query=%d insert=%d kv_len=%d sparse=%d us=%llu\n",
                  layerIndex, mQuerySeqLen, insertLen, kvLen, sparseQuery ? 1 : 0,
                  static_cast<unsigned long long>(_nowUs() - fallbackStartUs));
    }
    return NO_ERROR;
}

bool PagedAttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    auto tmp = new PagedAttentionBufExecution(op, bn);
    tmp->mCache = mCache;
    tmp->mMeta = mMeta;
    tmp->mLayerIndex = mLayerIndex;
    tmp->mKVSharedLayerIndex = mKVSharedLayerIndex;
    tmp->mIsKVShared = mIsKVShared;
    tmp->mBytes = mBytes;
    tmp->mBatch = mBatch;
    tmp->mQuerySeqLen = mQuerySeqLen;
    tmp->mNumHead = mNumHead;
    tmp->mHeadDim = mHeadDim;
    tmp->mKvNumHead = mKvNumHead;
    tmp->mNewKvSeqLen = mNewKvSeqLen;
    tmp->mScale = mScale;
    *dst = tmp;
    return true;
}

class PagedAttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (auto* input : inputs) {
            TensorUtils::setTensorSupportPack(input, false);
        }
        for (auto* output : outputs) {
            TensorUtils::setTensorSupportPack(output, false);
        }
        OPENCL_CREATOR_CHECK(new PagedAttentionBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(PagedAttentionBufCreator, OpType_PagedAttention, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
