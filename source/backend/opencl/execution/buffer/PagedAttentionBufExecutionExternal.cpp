//
//  PagedAttentionBufExecutionExternal.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecutionExternal.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MNN {
namespace OpenCL {

static bool _envFlagEnabled(const char* name, bool defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return value[0] != '0';
}

static bool _profilePagedAttention() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE", false);
}

static bool _legacyB863976OpenCL() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_LEGACY_B863976", false);
}

static uint64_t _nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool _allowDirectPagedCacheOpenCL() {
    if (_legacyB863976OpenCL()) {
        return false;
    }
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_ZERO_COPY_CACHE", true)) {
        return false;
    }
    return true;
}

static bool _requireDirectPagedCacheOpenCL() {
    if (_legacyB863976OpenCL()) {
        return false;
    }
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_ZERO_COPY_CACHE", true)) {
        return false;
    }
    return !_envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK", false);
}

static bool _shouldUseDirectPagedCacheOpenCL() {
    return _allowDirectPagedCacheOpenCL();
}

static bool _useAdrenoSourceSlotValueHydrate(OpenCLRuntime* runtime) {
    return PagedAttentionAdreno::useAdrenoSourceSlotValueHydrate(runtime, _legacyB863976OpenCL());
}

struct ExternalLayerReadSegment {
    bool ok = false;
    bool directWritten = false;
    bool valueAlreadyHydrated = false;
    int sourceSlotStart = 0;
    int valueSourceStart = 0;
    int valueSourceStride = 0;
    std::string error;
    std::vector<int8_t> keyData;
    std::vector<int8_t> valueData;
};

struct ExternalLayerReadResult {
    bool ok = true;
    int layerIndex = -1;
    std::string requestKey;
    std::string error;
    std::vector<ExternalLayerReadSegment> segments;
};

struct ExternalLayerReadTask {
    std::string requestKey;
    int layerIndex = -1;
    std::shared_future<std::shared_ptr<ExternalLayerReadResult>> future;
};

static std::mutex gExternalLayerReadMutex;
static std::unordered_map<std::string, ExternalLayerReadTask> gExternalLayerReadTasks;

int _alignPicCacheSourceSlots(int slots) {
    constexpr int kSourceSlotAlignment = 32;
    if (slots <= 0) {
        return slots;
    }
    if (slots > std::numeric_limits<int>::max() - kSourceSlotAlignment) {
        return std::numeric_limits<int>::max();
    }
    return ROUND_UP(slots, kSourceSlotAlignment);
}

static size_t _segmentSourceSlotSpan(const std::vector<PagedKVExternalSegment>& segments) {
    size_t cursor = 0;
    for (const auto& segment : segments) {
        if (cursor > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return cursor;
        }
        cursor = static_cast<size_t>(_alignPicCacheSourceSlots(static_cast<int>(cursor)));
        cursor += segment.tokenCount;
    }
    if (cursor > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return cursor;
    }
    return static_cast<size_t>(_alignPicCacheSourceSlots(static_cast<int>(cursor)));
}

int _picCacheSourceSlotBase(const PagedKVMeta* meta, int kvLen) {
    if (meta == nullptr) {
        return kvLen;
    }
    return _alignPicCacheSourceSlots(std::max(kvLen, meta->request_capacity));
}

size_t _picCacheSourceSlotCount(const PagedKVMeta* meta) {
    if (meta == nullptr) {
        return 0;
    }
    return std::max({static_cast<size_t>(_alignPicCacheSourceSlots(
                         static_cast<int>(std::min<size_t>(
                             meta->external_source_slot_reserve,
                             static_cast<size_t>(std::numeric_limits<int>::max()))))),
                     _segmentSourceSlotSpan(meta->external_segments),
                     _segmentSourceSlotSpan(meta->cacheblend_score_segments)});
}

struct ExternalLayerMappedTarget {
    std::shared_ptr<Tensor> key;
    std::shared_ptr<Tensor> value;
    std::shared_ptr<cl::CommandQueue> queue;
    bool valueSourceHydrate = false;
    int batch = 0;
    int kvHeads = 0;
    int headDim = 0;
    int bytes = 0;
    int maxSlots = 0;
};

struct ExternalLayerMappedTargetRef {
    std::weak_ptr<Tensor> key;
    std::weak_ptr<Tensor> value;
    std::shared_ptr<cl::CommandQueue> queue;
    bool valueSourceHydrate = false;
    int batch = 0;
    int kvHeads = 0;
    int headDim = 0;
    int bytes = 0;
    int maxSlots = 0;
};

static std::mutex gExternalLayerMappedTargetMutex;
static std::unordered_map<std::string, ExternalLayerMappedTargetRef> gExternalLayerMappedTargets;

static std::string _externalLayerMappedTargetKey(const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads,
                                                 int headDim, int bytes) {
    std::ostringstream os;
    os << reinterpret_cast<uintptr_t>(meta) << ":" << (meta != nullptr ? meta->request_generation : 0) << ":"
       << layerIndex << ":" << batch << ":" << kvHeads << ":" << headDim << ":" << bytes;
    return os.str();
}

void _registerExternalLayerMappedTarget(const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads,
                                               int headDim, int bytes, int maxSlots,
                                               const std::shared_ptr<Tensor>& key,
                                               const std::shared_ptr<Tensor>& value,
                                               cl::CommandQueue& queue, bool valueSourceHydrate) {
    if (!_shouldUseDirectPagedCacheOpenCL() || meta == nullptr || layerIndex < 0 || key == nullptr ||
        value == nullptr || key->deviceId() == 0 || value->deviceId() == 0) {
        return;
    }
    ExternalLayerMappedTargetRef ref;
    ref.key = key;
    ref.value = value;
    ref.queue.reset(new cl::CommandQueue(queue));
    ref.valueSourceHydrate = valueSourceHydrate;
    ref.batch = batch;
    ref.kvHeads = kvHeads;
    ref.headDim = headDim;
    ref.bytes = bytes;
    ref.maxSlots = maxSlots;
    std::lock_guard<std::mutex> lock(gExternalLayerMappedTargetMutex);
    gExternalLayerMappedTargets[_externalLayerMappedTargetKey(meta, layerIndex, batch, kvHeads, headDim, bytes)] =
        std::move(ref);
}

static ExternalLayerMappedTarget _lookupExternalLayerMappedTarget(const PagedKVMeta* meta, int layerIndex, int batch,
                                                                  int kvHeads, int headDim, int bytes) {
    ExternalLayerMappedTarget target;
    if (!_shouldUseDirectPagedCacheOpenCL()) {
        return target;
    }
    std::lock_guard<std::mutex> lock(gExternalLayerMappedTargetMutex);
    auto iter = gExternalLayerMappedTargets.find(
        _externalLayerMappedTargetKey(meta, layerIndex, batch, kvHeads, headDim, bytes));
    if (iter == gExternalLayerMappedTargets.end()) {
        return target;
    }
    target.key = iter->second.key.lock();
    target.value = iter->second.value.lock();
    target.queue = iter->second.queue;
    target.valueSourceHydrate = iter->second.valueSourceHydrate;
    target.batch = iter->second.batch;
    target.kvHeads = iter->second.kvHeads;
    target.headDim = iter->second.headDim;
    target.bytes = iter->second.bytes;
    target.maxSlots = iter->second.maxSlots;
    if (target.key == nullptr || target.value == nullptr || target.queue == nullptr) {
        gExternalLayerMappedTargets.erase(iter);
        target = ExternalLayerMappedTarget();
    }
    return target;
}

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

static int _externalLayerRequiredSlots(const PagedKVMeta* meta, int kvLen) {
    const size_t sourceSlots = _picCacheSourceSlotCount(meta);
    if (sourceSlots == 0) {
        return kvLen;
    }
    const int sourceBase = _picCacheSourceSlotBase(meta, kvLen);
    if (sourceSlots > static_cast<size_t>(std::numeric_limits<int>::max() - sourceBase)) {
        return std::numeric_limits<int>::max();
    }
    return std::max(kvLen, sourceBase + static_cast<int>(sourceSlots));
}

static bool _externalLayerTargetReady(const ExternalLayerMappedTarget& target, int batch, int kvHeads, int headDim,
                                      int bytes, int requiredSlots) {
    return target.key != nullptr && target.value != nullptr && target.queue != nullptr &&
        target.batch == batch && target.kvHeads == kvHeads && target.headDim == headDim &&
        target.bytes == bytes && target.maxSlots >= requiredSlots;
}

static int _externalLayerReadWindow() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PREFETCH_WINDOW");
    if (value == nullptr || value[0] == '\0') {
        return -1;
    }
    return std::atoi(value);
}

static std::string _externalLayerRequestKey(const PagedKVMeta* meta, int batch, int kvHeads, int headDim, int bytes,
                                            int kvLen) {
    std::ostringstream os;
    os << reinterpret_cast<uintptr_t>(meta) << ":" << (meta != nullptr ? meta->request_generation : 0) << ":"
       << batch << ":" << kvHeads << ":" << headDim << ":" << bytes << ":" << kvLen;
    if (meta != nullptr) {
        for (const auto& segment : meta->external_segments) {
            os << "|seg:" << segment.cacheName << ":" << segment.logicalStart << ":" << segment.tokenCount << ":"
               << segment.sourceTokenOffset << ":" << segment.sourceTokenCount;
            for (const auto& layer : segment.layers) {
                os << "|layer:" << layer.layerIndex << ":" << layer.keyPath << ":" << layer.valuePath << ":"
                   << layer.hasSourceOverride << ":" << layer.sourceTokenOffset << ":" << layer.sourceTokenCount;
            }
        }
    }
    return os.str();
}

static std::string _externalLayerTaskKey(const std::string& requestKey, int layerIndex) {
    return requestKey + "\n" + std::to_string(layerIndex);
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

static bool _readBinaryFileRangeToMappedCLBufferOffset(const std::string& path, size_t fileOffsetBytes,
                                                       cl::Buffer& buffer, size_t bufferOffsetBytes,
                                                       size_t expectedBytes, cl::CommandQueue& queue) {
    if (expectedBytes == 0) {
        return false;
    }
    cl_int error = CL_SUCCESS;
    void* ptr = queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, bufferOffsetBytes, expectedBytes,
                                       nullptr, nullptr, &error);
    if (ptr == nullptr || error != CL_SUCCESS) {
        return false;
    }
    bool ok = _readBinaryFileRange(path, fileOffsetBytes, ptr, expectedBytes);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    cl_int unmap = queue.enqueueUnmapMemObject(buffer, ptr);
    cl_int finish = queue.finish();
    return ok && unmap == CL_SUCCESS && finish == CL_SUCCESS;
}

static bool _readKeySegmentToPagedCacheSourceSlotsOpenCL(const std::string& path, cl::Buffer& keyCache,
                                                         cl::CommandQueue& queue, int batch, int kvHeads,
                                                         int headDim, int bytes, size_t sourceTokenOffset,
                                                         size_t tokenCount, size_t sourceSlotStart,
                                                         int maxSlots) {
    if (batch <= 0 || kvHeads <= 0 || headDim <= 0 || bytes <= 0 || tokenCount == 0 ||
        sourceSlotStart + tokenCount > static_cast<size_t>(maxSlots)) {
        return false;
    }
    const size_t tokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
    return _readBinaryFileRangeToMappedCLBufferOffset(path, sourceTokenOffset * tokenBytes, keyCache,
                                                      sourceSlotStart * tokenBytes, tokenCount * tokenBytes,
                                                      queue);
}

enum class SourceCLReadStatus {
    Success,
    MapFailed,
    ReadFailed,
};

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
    std::atomic_thread_fence(std::memory_order_seq_cst);
    cl_int unmap = queue.enqueueUnmapMemObject(buffer, ptr);
    cl_int finish = queue.finish();
    if (!ok) {
        return SourceCLReadStatus::ReadFailed;
    }
    return unmap == CL_SUCCESS && finish == CL_SUCCESS ? SourceCLReadStatus::Success : SourceCLReadStatus::MapFailed;
}

static bool _readBinaryFileRangeToCLBufferFallback(const std::string& path, size_t offsetBytes, cl::Buffer& buffer,
                                                   size_t expectedBytes, cl::CommandQueue& queue) {
    std::vector<int8_t> data(expectedBytes);
    if (!_readBinaryFileRange(path, offsetBytes, data.data(), expectedBytes)) {
        return false;
    }
    return queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, expectedBytes, data.data()) == CL_SUCCESS;
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
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK", false)) {
        static std::once_flag mapRequiredOnce;
        std::call_once(mapRequiredOnce, []() {
            MNN_PRINT("OpenCLPagedAttention: mapped source CL buffer is required for PIC cache key hydrate; "
                      "set MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK=1 only for debug fallback\n");
        });
        return false;
    }
    static std::once_flag fallbackOnce;
    std::call_once(fallbackOnce, []() {
        MNN_PRINT("OpenCLPagedAttention: debug fallback copies PIC cache key source with enqueueWriteBuffer\n");
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
    auto finish = queue.finish();
    if (!ok) {
        return SourceCLReadStatus::ReadFailed;
    }
    return unmap == CL_SUCCESS && finish == CL_SUCCESS ? SourceCLReadStatus::Success : SourceCLReadStatus::MapFailed;
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

bool _readExternalValueSegmentToSourceCLBuffer(
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
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK", false)) {
        static std::once_flag mapRequiredOnce;
        std::call_once(mapRequiredOnce, []() {
            MNN_PRINT("OpenCLPagedAttention: mapped PagedCache source slots are required for PIC cache hydrate; "
                      "set MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK=1 only for debug fallback\n");
        });
        return false;
    }
    static std::once_flag fallbackOnce;
    std::call_once(fallbackOnce, []() {
        MNN_PRINT("OpenCLPagedAttention: debug fallback copies PIC cache source data with enqueueWriteBuffer\n");
    });
    return _readExternalValueSegmentToCLBufferFallback(path, buffer, expectedBytes, queue, batch, kvHeads,
                                                       sourceTokenCount, sourceTokenOffset, tokenCount, headDim,
                                                       bytes);
}

bool _readExternalValueSegmentToPagedCacheOpenCL(
    const std::string& path, cl::Buffer& valueCache, cl::CommandQueue& queue, int batch, int kvHeads, int maxSlots,
    size_t logicalStart, size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount, int headDim,
    int bytes) {
    if (batch <= 0 || kvHeads <= 0 || maxSlots <= 0 || headDim <= 0 || bytes <= 0 || tokenCount == 0 ||
        logicalStart + tokenCount > static_cast<size_t>(maxSlots)) {
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
    if (sourceTokenOffset + tokenCount > sourceTokenCount || fileBytes < requiredBytes) {
        return false;
    }
    bool ok = true;
    for (int b = 0; ok && b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dst = ((static_cast<size_t>(b) * kvHeads + h) * static_cast<size_t>(maxSlots) +
                                logicalStart) * tokenBytes;
            cl_int error = CL_SUCCESS;
            void* ptr = queue.enqueueMapBuffer(valueCache, CL_TRUE, CL_MAP_WRITE, dst, segmentBytes, nullptr,
                                               nullptr, &error);
            if (ptr == nullptr || error != CL_SUCCESS) {
                ok = false;
                break;
            }
            is.seekg(static_cast<std::streamoff>(src), std::ios::beg);
            is.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(segmentBytes));
            if (is.gcount() != static_cast<std::streamsize>(segmentBytes)) {
                ok = false;
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            cl_int unmap = queue.enqueueUnmapMemObject(valueCache, ptr);
            if (unmap != CL_SUCCESS) {
                ok = false;
                break;
            }
        }
    }
    cl_int finish = queue.finish();
    return ok && finish == CL_SUCCESS;
}

static bool _readExternalValueSegmentToPhysicalPagedCacheOpenCL(
    const std::string& path, cl::Buffer& valueCache, cl::CommandQueue& queue, const PagedKVMeta* meta, int batch,
    int kvHeads, int maxSlots, size_t logicalStart, size_t sourceTokenCount, size_t sourceTokenOffset,
    size_t tokenCount, int headDim, int bytes) {
    if (batch <= 0 || kvHeads <= 0 || maxSlots <= 0 || headDim <= 0 || bytes <= 0 || tokenCount == 0) {
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
    const size_t requiredBytes = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * tokenBytes;
    if (sourceTokenOffset + tokenCount > sourceTokenCount || fileBytes < requiredBytes) {
        return false;
    }
    bool ok = true;
    for (int b = 0; ok && b < batch; ++b) {
        for (int h = 0; ok && h < kvHeads; ++h) {
            size_t local = 0;
            while (local < tokenCount) {
                const size_t logical = logicalStart + local;
                const int slot = static_cast<int>(logical);
                if (slot < 0 || slot >= maxSlots) {
                    ok = false;
                    break;
                }
                size_t run = 1;
                while (local + run < tokenCount) {
                    const size_t nextLogical = logicalStart + local + run;
                    const int nextSlot = static_cast<int>(nextLogical);
                    if (nextSlot != slot + static_cast<int>(run) || nextSlot < 0 || nextSlot >= maxSlots) {
                        break;
                    }
                    ++run;
                }
                const size_t bytesToRead = run * tokenBytes;
                const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                    sourceTokenOffset + local) * tokenBytes;
                const size_t dst = ((static_cast<size_t>(b) * kvHeads + h) * static_cast<size_t>(maxSlots) +
                                    static_cast<size_t>(slot)) * tokenBytes;
                cl_int error = CL_SUCCESS;
                void* ptr = queue.enqueueMapBuffer(valueCache, CL_TRUE, CL_MAP_WRITE, dst, bytesToRead, nullptr,
                                                   nullptr, &error);
                if (ptr == nullptr || error != CL_SUCCESS) {
                    ok = false;
                    break;
                }
                is.seekg(static_cast<std::streamoff>(src), std::ios::beg);
                is.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(bytesToRead));
                if (is.gcount() != static_cast<std::streamsize>(bytesToRead)) {
                    ok = false;
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                cl_int unmap = queue.enqueueUnmapMemObject(valueCache, ptr);
                if (unmap != CL_SUCCESS) {
                    ok = false;
                    break;
                }
                local += run;
            }
        }
    }
    cl_int finish = queue.finish();
    return ok && finish == CL_SUCCESS;
}

static bool _readExternalLayerSegmentOpenCL(const PagedKVMeta* meta, const PagedKVExternalSegment& segment,
                                            int layerIndex, int batch, int kvHeads, int headDim, int bytes, int kvLen,
                                            int sourceSlotStart,
                                            const ExternalLayerMappedTarget* target,
                                            ExternalLayerReadSegment& out) {
    out = ExternalLayerReadSegment();
    if (segment.tokenCount == 0) {
        out.ok = true;
        return true;
    }
    auto layer = segment.layer(layerIndex);
    if (layer == nullptr) {
        out.error = "missing persistent PIC cache layer " + std::to_string(layerIndex) + " for cache " +
            segment.cacheName;
        return false;
    }
    const int segBatch = segment.batch > 0 ? segment.batch : batch;
    const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
    const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
    const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
    if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
        out.error = "persistent PIC cache shape mismatch at layer " + std::to_string(layerIndex);
        return false;
    }
    if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
        out.error = "persistent PIC cache key must be canonical_no_rope/half";
        return false;
    }
    if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
        out.error = "persistent PIC cache range exceeds visible KV length at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset : segment.sourceTokenOffset;
    const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
        ? layer->sourceTokenCount
        : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
    if (sourceTokenOffset + segment.tokenCount > sourceTokenCount) {
        out.error = "persistent PIC cache source range is invalid at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
    if (segment.logicalStart > maxInt || segment.tokenCount > maxInt || sourceTokenOffset > maxInt ||
        sourceTokenCount > maxInt) {
        out.error = "persistent PIC cache indices exceed int range at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t keyTokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
    const size_t keySegmentBytes = segment.tokenCount * keyTokenBytes;
    const bool requireDirect = _requireDirectPagedCacheOpenCL();
    const bool targetOk = target != nullptr && target->key != nullptr && target->value != nullptr &&
        target->queue != nullptr && target->batch == batch && target->kvHeads == kvHeads &&
        target->headDim == headDim && target->bytes == bytes && target->maxSlots >= kvLen &&
        sourceSlotStart >= 0 && static_cast<size_t>(sourceSlotStart) + segment.tokenCount <=
            static_cast<size_t>(target->maxSlots);
    if (requireDirect && !targetOk) {
        out.error = "OpenCL direct PagedCache target unavailable at layer " + std::to_string(layerIndex);
        return false;
    }
    if (targetOk) {
        auto& keyBuffer = openCLBuffer(target->key.get());
        auto& valueBuffer = openCLBuffer(target->value.get());
        const bool directKeyOk = _readKeySegmentToPagedCacheSourceSlotsOpenCL(
            layer->keyPath, keyBuffer, *target->queue, batch, kvHeads, headDim, bytes, sourceTokenOffset,
            segment.tokenCount, static_cast<size_t>(sourceSlotStart), target->maxSlots);
        bool directValueOk = false;
        bool valueAlreadyHydrated = false;
        if (directKeyOk && target->valueSourceHydrate) {
            directValueOk = _readExternalValueSegmentToPagedCacheOpenCL(
                layer->valuePath, valueBuffer, *target->queue, batch, kvHeads, target->maxSlots,
                static_cast<size_t>(sourceSlotStart), sourceTokenCount, sourceTokenOffset,
                segment.tokenCount, headDim, bytes);
        } else if (directKeyOk) {
            directValueOk = _readExternalValueSegmentToPhysicalPagedCacheOpenCL(
                layer->valuePath, valueBuffer, *target->queue, meta, batch, kvHeads, target->maxSlots,
                segment.logicalStart, sourceTokenCount, sourceTokenOffset, segment.tokenCount, headDim, bytes);
            valueAlreadyHydrated = directValueOk;
        }
        if (directKeyOk && directValueOk) {
            out.directWritten = true;
            out.valueAlreadyHydrated = valueAlreadyHydrated;
            out.sourceSlotStart = sourceSlotStart;
            out.valueSourceStart = target->valueSourceHydrate ? sourceSlotStart : 0;
            out.valueSourceStride = target->valueSourceHydrate ? target->maxSlots
                                                               : static_cast<int>(segment.tokenCount);
            out.ok = true;
            return true;
        }
        if (requireDirect) {
            out.error = "OpenCL direct map read into PagedCache failed at layer " + std::to_string(layerIndex);
            return false;
        }
    }
    out.keyData.resize(keySegmentBytes);
    if (!_readBinaryFileRange(layer->keyPath, sourceTokenOffset * keyTokenBytes, out.keyData.data(),
                              keySegmentBytes)) {
        out.error = "failed to read persistent PIC cache key range at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t valueTokenBytes = static_cast<size_t>(headDim) * bytes;
    const size_t valueSegmentBytes = static_cast<size_t>(batch) * kvHeads * segment.tokenCount * valueTokenBytes;
    out.valueData.resize(valueSegmentBytes);
    if (!_readExternalValueSegment(layer->valuePath, out.valueData.data(), valueSegmentBytes, batch, kvHeads,
                                   sourceTokenCount, sourceTokenOffset, segment.tokenCount, headDim, bytes)) {
        out.error = "failed to read persistent PIC cache value range at layer " + std::to_string(layerIndex);
        return false;
    }
    out.ok = true;
    return true;
}

static std::shared_ptr<ExternalLayerReadResult> _readExternalLayerOpenCL(
    const PagedKVMeta* meta, std::string requestKey, std::vector<PagedKVExternalSegment> segments, int layerIndex,
    int batch, int kvHeads, int headDim, int bytes, int kvLen, ExternalLayerMappedTarget target) {
    auto result = std::make_shared<ExternalLayerReadResult>();
    result->requestKey = std::move(requestKey);
    result->layerIndex = layerIndex;
    result->segments.resize(segments.size());
    const bool hasTarget = _externalLayerTargetReady(
        target, batch, kvHeads, headDim, bytes, _externalLayerRequiredSlots(meta, kvLen));
    int sourceSlotCursor = _picCacheSourceSlotBase(meta, kvLen);
    for (size_t i = 0; i < segments.size(); ++i) {
        sourceSlotCursor = _alignPicCacheSourceSlots(sourceSlotCursor);
        const int sourceSlotStart = sourceSlotCursor;
        sourceSlotCursor += static_cast<int>(segments[i].tokenCount);
        if (!_readExternalLayerSegmentOpenCL(meta, segments[i], layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                            sourceSlotStart, hasTarget ? &target : nullptr,
                                            result->segments[i])) {
            result->ok = false;
            result->error = result->segments[i].error;
            break;
        }
    }
    return result;
}

void _scheduleExternalLayerReadsFrom(const PagedKVMeta* meta, int startLayer, int batch, int kvHeads,
                                            int headDim, int bytes, int kvLen) {
    if (_legacyB863976OpenCL()) {
        return;
    }
    if (meta == nullptr || meta->external_segments.empty()) {
        return;
    }
    const int lastLayer = _lastExternalLayerIndex(meta);
    if (startLayer < 0 || lastLayer < startLayer) {
        return;
    }
    const int requiredSlots = _externalLayerRequiredSlots(meta, kvLen);
    const int window = _externalLayerReadWindow();
    if (window == 0) {
        return;
    }
    const int endLayer = window > 0 ? std::min(lastLayer, startLayer + window - 1) : lastLayer;
    const std::string requestKey = _externalLayerRequestKey(meta, batch, kvHeads, headDim, bytes, kvLen);
    const std::string requestPrefix = requestKey + "\n";
    std::vector<std::pair<int, ExternalLayerMappedTarget>> layersToSchedule;
    {
        std::lock_guard<std::mutex> lock(gExternalLayerReadMutex);
        for (auto it = gExternalLayerReadTasks.begin(); it != gExternalLayerReadTasks.end();) {
            if (it->first.rfind(requestPrefix, 0) != 0) {
                it = gExternalLayerReadTasks.erase(it);
            } else {
                ++it;
            }
        }
        for (int layerIndex = startLayer; layerIndex <= endLayer; ++layerIndex) {
            if (meta->externalLayerLoaded(layerIndex)) {
                continue;
            }
            const auto key = _externalLayerTaskKey(requestKey, layerIndex);
            if (gExternalLayerReadTasks.find(key) == gExternalLayerReadTasks.end()) {
                auto target = _lookupExternalLayerMappedTarget(meta, layerIndex, batch, kvHeads, headDim, bytes);
                if (!_externalLayerTargetReady(target, batch, kvHeads, headDim, bytes, requiredSlots)) {
                    break;
                }
                layersToSchedule.emplace_back(layerIndex, std::move(target));
            }
        }
        auto segments = meta->external_segments;
        for (const auto& item : layersToSchedule) {
            const int layerIndex = item.first;
            auto target = item.second;
            const auto key = _externalLayerTaskKey(requestKey, layerIndex);
            auto future = std::async(std::launch::async,
                                     [meta, requestKey, segments, layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                      target]() {
                                         return _readExternalLayerOpenCL(meta, requestKey, segments, layerIndex,
                                                                         batch, kvHeads, headDim, bytes, kvLen,
                                                                         target);
                                     }).share();
            ExternalLayerReadTask task;
            task.requestKey = requestKey;
            task.layerIndex = layerIndex;
            task.future = std::move(future);
            gExternalLayerReadTasks[key] = std::move(task);
        }
    }
}

static std::shared_ptr<ExternalLayerReadResult> _takeExternalLayerRead(const PagedKVMeta* meta, int layerIndex,
                                                                       int batch, int kvHeads, int headDim, int bytes,
                                                                       int kvLen) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return nullptr;
    }
    const auto requestKey = _externalLayerRequestKey(meta, batch, kvHeads, headDim, bytes, kvLen);
    const auto taskKey = _externalLayerTaskKey(requestKey, layerIndex);
    std::shared_future<std::shared_ptr<ExternalLayerReadResult>> future;
    {
        std::lock_guard<std::mutex> lock(gExternalLayerReadMutex);
        auto iter = gExternalLayerReadTasks.find(taskKey);
        if (iter == gExternalLayerReadTasks.end()) {
            return nullptr;
        }
        future = iter->second.future;
        gExternalLayerReadTasks.erase(iter);
    }
    if (!future.valid()) {
        return nullptr;
    }
    return future.get();
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

ErrorCode PagedAttentionBufExecution::hydrateExternalSegments(int layerIndex, int kvLen) {
    if (mMeta == nullptr || mMeta->external_segments.empty() || mMeta->externalLayerLoaded(layerIndex)) {
        return NO_ERROR;
    }
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    size_t totalTokens = 0;
    size_t directTokens = 0;
    size_t fallbackTokens = 0;
    int directSegments = 0;
    const bool legacy = _legacyB863976OpenCL();
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    if (!legacy && mCache != nullptr && mCache->key != nullptr && mCache->value != nullptr) {
        _registerExternalLayerMappedTarget(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes,
                                           mCache->maxSlots, mCache->key, mCache->value, queue,
                                           _useAdrenoSourceSlotValueHydrate(mOpenCLBackend->getOpenCLRuntime()));
    }
    auto prefetched = legacy ? nullptr
                             : _takeExternalLayerRead(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes, kvLen);
    if (prefetched != nullptr && !prefetched->ok) {
        MNN_ERROR("OpenCLPagedAttention async persistent PIC cache read failed at layer %d: %s\n", layerIndex,
                  prefetched->error.c_str());
        return INVALID_VALUE;
    }
    if (prefetched != nullptr && prefetched->segments.size() != mMeta->external_segments.size()) {
        MNN_ERROR("OpenCLPagedAttention async persistent PIC cache read segment count mismatch at layer %d\n",
                  layerIndex);
        return INVALID_VALUE;
    }
    auto directTarget = _lookupExternalLayerMappedTarget(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes);
    const bool hasDirectTarget = directTarget.key != nullptr && directTarget.value != nullptr &&
                                 directTarget.queue != nullptr;
    int sourceSlotCursor = _picCacheSourceSlotBase(mMeta, kvLen);
    for (size_t segmentIndex = 0; segmentIndex < mMeta->external_segments.size(); ++segmentIndex) {
        const auto& segment = mMeta->external_segments[segmentIndex];
        sourceSlotCursor = _alignPicCacheSourceSlots(sourceSlotCursor);
        const int sourceSlotStart = sourceSlotCursor;
        sourceSlotCursor += static_cast<int>(segment.tokenCount);
        totalTokens += segment.tokenCount;
        if (segment.tokenCount == 0) {
            continue;
        }
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            MNN_ERROR("OpenCLPagedAttention layer %d missing persistent PIC cache for cache %s\n", layerIndex,
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
            MNN_ERROR("OpenCLPagedAttention persistent PIC cache indices exceed int range at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        const size_t keyTokenBytes = static_cast<size_t>(mBatch) * mKvNumHead * mHeadDim * mBytes;
        const size_t keySourceOffsetBytes = sourceTokenOffset * keyTokenBytes;
        const size_t keySegmentBytes = segment.tokenCount * keyTokenBytes;
        const size_t valueTokenBytes = static_cast<size_t>(mHeadDim) * mBytes;
        const size_t valueSegmentBytes = static_cast<size_t>(mBatch) * mKvNumHead * segment.tokenCount *
                                         valueTokenBytes;
        ExternalLayerReadSegment syncRead;
        const ExternalLayerReadSegment* loadedSegment = nullptr;
        if (prefetched != nullptr) {
            loadedSegment = &prefetched->segments[segmentIndex];
            if (!loadedSegment->ok) {
                MNN_ERROR("OpenCLPagedAttention async persistent PIC cache segment read failed at layer %d: %s\n",
                          layerIndex, loadedSegment->error.c_str());
                return INVALID_VALUE;
            }
        } else {
            if (!_readExternalLayerSegmentOpenCL(mMeta, segment, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes,
                                                kvLen, sourceSlotStart,
                                                hasDirectTarget ? &directTarget : nullptr, syncRead)) {
                MNN_ERROR("OpenCLPagedAttention failed to read persistent PIC cache files for layer %d: %s\n",
                          layerIndex, syncRead.error.c_str());
                return INVALID_VALUE;
            }
            loadedSegment = &syncRead;
        }

        cl::Buffer* sourceKeyBuffer = nullptr;
        cl::Buffer* sourceValueBuffer = nullptr;
        int keySourceLogicalStart = 0;
        int valueSourceStart = 0;
        int valueSourceStride = static_cast<int>(segment.tokenCount);
        int hydrateValue = 1;
        if (loadedSegment != nullptr && loadedSegment->directWritten) {
            sourceKeyBuffer = &openCLBuffer(mCache->key.get());
            sourceValueBuffer = &openCLBuffer(mCache->value.get());
            keySourceLogicalStart = loadedSegment->sourceSlotStart;
            valueSourceStart = loadedSegment->valueSourceStart;
            valueSourceStride = loadedSegment->valueSourceStride > 0
                ? loadedSegment->valueSourceStride
                : static_cast<int>(segment.tokenCount);
            hydrateValue = loadedSegment->valueAlreadyHydrated ? 0 : 1;
            directTokens += segment.tokenCount;
            ++directSegments;
        } else {
            fallbackTokens += segment.tokenCount;
            auto err = ensureExternalTemps(keySegmentBytes / mBytes, valueSegmentBytes / mBytes);
            if (err != NO_ERROR) {
                return err;
            }
            auto& externalKeyBuffer = openCLBuffer(mExternalKey.get());
            auto& externalValueBuffer = openCLBuffer(mExternalValue.get());
            if (loadedSegment != nullptr) {
                if (loadedSegment->keyData.size() < keySegmentBytes ||
                    loadedSegment->valueData.size() < valueSegmentBytes) {
                    return INVALID_VALUE;
                }
                if (queue.enqueueWriteBuffer(externalKeyBuffer, CL_TRUE, 0, keySegmentBytes,
                                             loadedSegment->keyData.data()) != CL_SUCCESS ||
                    queue.enqueueWriteBuffer(externalValueBuffer, CL_TRUE, 0, valueSegmentBytes,
                                             loadedSegment->valueData.data()) != CL_SUCCESS) {
                    return INVALID_VALUE;
                }
            } else {
                if (!_readBinaryFileRangeToSourceCLBuffer(layer->keyPath, keySourceOffsetBytes, externalKeyBuffer,
                                                          keySegmentBytes, queue) ||
                    !_readExternalValueSegmentToSourceCLBuffer(
                        layer->valuePath, externalValueBuffer, valueSegmentBytes, queue, mBatch, mKvNumHead,
                        sourceTokenCount, sourceTokenOffset, segment.tokenCount, mHeadDim, mBytes)) {
                    return INVALID_VALUE;
                }
            }
            sourceKeyBuffer = &externalKeyBuffer;
            sourceValueBuffer = &externalValueBuffer;
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
            MNN_ERROR("OpenCLPagedAttention persistent PIC cache hydrate element count exceeds int range at layer %d\n",
                      layerIndex);
            return INVALID_VALUE;
        }
        const int total = static_cast<int>(totalElements);
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        if (loadedSegment != nullptr && loadedSegment->directWritten) {
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, mBatch);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, mKvNumHead);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, mHeadDim);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, mCache->maxSlots);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, keySourceLogicalStart);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, valueSourceStart);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, valueSourceStride);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, hydrateValue);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, ropeDim);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeTheta);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, ropeTypeLlama3);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeScalingFactor);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeScalingLowFreqFactor);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeScalingHighFreqFactor);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, oldContext);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.maxPositionEmbeddings);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeAttentionScaling);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, total);
            MNN_CHECK_CL_SUCCESS(ret, "setArg pic_page_attention_hydrate_kv_inplace");
            queue.enqueueNDRangeKernel(mHydrateExternalInplaceKernel->get(), cl::NullRange, cl::NDRange(total),
                                       cl::NullRange);
        } else {
            ret |= mHydrateExternalKernel->get().setArg(idx++, *sourceKeyBuffer);
            ret |= mHydrateExternalKernel->get().setArg(idx++, *sourceValueBuffer);
            ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
            ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
            ret |= mHydrateExternalKernel->get().setArg(idx++, mBatch);
            ret |= mHydrateExternalKernel->get().setArg(idx++, mKvNumHead);
            ret |= mHydrateExternalKernel->get().setArg(idx++, mHeadDim);
            ret |= mHydrateExternalKernel->get().setArg(idx++, mCache->maxSlots);
            ret |= mHydrateExternalKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
            ret |= mHydrateExternalKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
            ret |= mHydrateExternalKernel->get().setArg(idx++, keySourceLogicalStart);
            ret |= mHydrateExternalKernel->get().setArg(idx++, valueSourceStart);
            ret |= mHydrateExternalKernel->get().setArg(idx++, valueSourceStride);
            ret |= mHydrateExternalKernel->get().setArg(idx++, hydrateValue);
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
            queue.enqueueNDRangeKernel(mHydrateExternalKernel->get(), cl::NullRange, cl::NDRange(total),
                                       cl::NullRange);
        }
    }
    mMeta->markExternalLayerLoaded(layerIndex);
    if (profile) {
        queue.finish();
        MNN_PRINT("OpenCLPagedAttention profile op=hydrate layer=%d tokens=%d kv_len=%d async_read=%d "
                  "direct_segments=%d direct_tokens=%d fallback_tokens=%d us=%llu\n",
                  layerIndex, static_cast<int>(totalTokens), kvLen, prefetched != nullptr ? 1 : 0,
                  directSegments, static_cast<int>(directTokens), static_cast<int>(fallbackTokens),
                  static_cast<unsigned long long>(_nowUs() - startUs));
    }
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN

#endif // MNN_OPENCL_BUFFER_CLOSED
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
