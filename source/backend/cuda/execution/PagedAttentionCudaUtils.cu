#include "PagedAttentionCudaUtils.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

CUDAPagedAttention::SharedPagedCache::MappedBuffer::~MappedBuffer() {
    if (host == nullptr) {
        return;
    }
    if (deviceId >= 0) {
        cudaSetDevice(deviceId);
    }
    cudaFreeHost(host);
    host = nullptr;
    device = nullptr;
    bytes = 0;
}

static size_t segmentTokenCountCUDA(const std::vector<PagedKVExternalSegment>& segments) {
    size_t total = 0;
    for (const auto& segment : segments) {
        total += segment.tokenCount;
    }
    return total;
}

int picCacheSourceSlotBaseCUDA(const PagedKVMeta* meta, int kvLen) {
    if (meta == nullptr) {
        return kvLen;
    }
    return std::max(kvLen, meta->request_capacity);
}

size_t picCacheSourceSlotCountCUDA(const PagedKVMeta* meta) {
    if (meta == nullptr) {
        return 0;
    }
    return std::max(meta->external_source_slot_reserve,
                    std::max(segmentTokenCountCUDA(meta->external_segments),
                             segmentTokenCountCUDA(meta->cacheblend_score_segments)));
}

int maxSlotsWithPicSourceSlotsCUDA(const PagedKVMeta* meta, int kvLen) {
    const size_t sourceSlots = picCacheSourceSlotCountCUDA(meta);
    if (sourceSlots == 0) {
        return kvLen;
    }
    const int sourceBase = picCacheSourceSlotBaseCUDA(meta, kvLen);
    if (sourceSlots > static_cast<size_t>(std::numeric_limits<int>::max() - sourceBase)) {
        return std::numeric_limits<int>::max();
    }
    return std::max(kvLen, sourceBase + static_cast<int>(sourceSlots));
}

bool writeBinaryFile(const std::string& path, const void* data, size_t bytes) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    if (data != nullptr && bytes > 0) {
        const char* cursor = reinterpret_cast<const char*>(data);
        size_t remaining = bytes;
        while (remaining > 0) {
            const auto chunk = static_cast<std::streamsize>(
                std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<std::streamsize>::max())));
            os.write(cursor, chunk);
            if (!os.good()) {
                return false;
            }
            cursor += chunk;
            remaining -= static_cast<size_t>(chunk);
        }
    }
    return os.good();
}

static bool readStreamRange(std::ifstream& is, size_t offsetBytes, void* dst, size_t expectedBytes) {
    if (dst == nullptr || expectedBytes == 0) {
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

bool readBinaryFileRange(const std::string& path, size_t offsetBytes, void* dst, size_t expectedBytes) {
    if (dst == nullptr || expectedBytes == 0) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto size = is.tellg();
    if (size < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(size));
    if (offsetBytes > fileBytes || fileBytes - offsetBytes < expectedBytes) {
        return false;
    }
    return readStreamRange(is, offsetBytes, dst, expectedBytes);
}

bool readExternalValueSegmentToBuffer(const std::string& path, void* dstBuffer, size_t dstBytes, int batch,
                                      int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset,
                                      size_t tokenCount, int headDim, int bytes) {
    if (batch <= 0 || kvHeads <= 0 || headDim <= 0 || bytes <= 0 || tokenCount == 0) {
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
    const size_t outputBytes = static_cast<size_t>(batch) * kvHeads * segmentBytes;
    if (dstBuffer == nullptr || dstBytes < outputBytes) {
        return false;
    }
    auto* dstBase = reinterpret_cast<int8_t*>(dstBuffer);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dst = (static_cast<size_t>(b) * kvHeads + h) * segmentBytes;
            if (!readStreamRange(is, src, dstBase + dst, segmentBytes)) {
                return false;
            }
        }
    }
    return true;
}

bool readExternalValueSegmentToPagedCache(const std::string& path, void* valueCacheHost, int batch,
                                          int kvHeads, int maxSlots, size_t logicalStart,
                                          size_t sourceTokenCount, size_t sourceTokenOffset,
                                          size_t tokenCount, int headDim, int bytes) {
    if (valueCacheHost == nullptr || batch <= 0 || kvHeads <= 0 || maxSlots <= 0 || headDim <= 0 || bytes <= 0 ||
        tokenCount == 0 || logicalStart + tokenCount > static_cast<size_t>(maxSlots)) {
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
    auto* dstBase = reinterpret_cast<int8_t*>(valueCacheHost);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dst = ((static_cast<size_t>(b) * kvHeads + h) * static_cast<size_t>(maxSlots) +
                                logicalStart) * tokenBytes;
            if (!readStreamRange(is, src, dstBase + dst, segmentBytes)) {
                return false;
            }
        }
    }
    return true;
}

bool profilePagedAttention() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool nvtxPagedAttention() {
#ifdef MNN_CUDA_PROFILE
    const char* value = ::getenv("MNN_PAGED_ATTENTION_NVTX");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
#else
    return false;
#endif
}

uint64_t nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string nvtxLayerRangeName(const char* op, int layerIndex, int queryLen, int insertLen, int kvLen) {
    std::ostringstream os;
    os << "CUDAPagedAttention " << op << " layer=" << layerIndex;
    if (queryLen >= 0) {
        os << " query=" << queryLen;
    }
    if (insertLen >= 0) {
        os << " insert=" << insertLen;
    }
    if (kvLen >= 0) {
        os << " kv_len=" << kvLen;
    }
    return os.str();
}

bool envFlagEnabled(const char* name, bool defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return value[0] != '0';
}

std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> makeMappedPagedBuffer(size_t bytes,
                                                                                          int deviceId) {
    if (bytes == 0) {
        return nullptr;
    }
    if (deviceId >= 0) {
        cudaSetDevice(deviceId);
    }
    void* host = nullptr;
    auto err = cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess || host == nullptr) {
        return nullptr;
    }
    void* device = nullptr;
    err = cudaHostGetDevicePointer(&device, host, 0);
    if (err != cudaSuccess || device == nullptr) {
        cudaFreeHost(host);
        return nullptr;
    }
    auto out = std::make_shared<CUDAPagedAttention::SharedPagedCache::MappedBuffer>();
    out->host = host;
    out->device = device;
    out->bytes = bytes;
    out->deviceId = deviceId;
    return out;
}

static size_t roundMappedWorkspaceBytes(size_t bytes) {
    constexpr size_t kAlign = 64 * 1024 * 1024;
    if (bytes == 0) {
        return 0;
    }
    return ((bytes + kAlign - 1) / kAlign) * kAlign;
}

bool ensureMappedHostWorkspace(std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>* buffer,
                               size_t bytes, int deviceId) {
    if (buffer == nullptr || bytes == 0) {
        return false;
    }
    if (*buffer != nullptr && (*buffer)->host != nullptr && (*buffer)->device != nullptr &&
        (*buffer)->bytes >= bytes) {
        return true;
    }
    *buffer = makeMappedPagedBuffer(roundMappedWorkspaceBytes(bytes), deviceId);
    return *buffer != nullptr && (*buffer)->host != nullptr && (*buffer)->device != nullptr;
}

MappedPagedExportWorkspace& mappedPagedExportWorkspace(int deviceId) {
    static std::mutex mutex;
    static std::unordered_map<int, std::shared_ptr<MappedPagedExportWorkspace>> workspaces;
    std::lock_guard<std::mutex> lock(mutex);
    auto& workspace = workspaces[deviceId];
    if (workspace == nullptr) {
        workspace.reset(new MappedPagedExportWorkspace);
    }
    return *workspace;
}

int cudaBackendDeviceId(CUDABackend* backend) {
    if (backend == nullptr || backend->getCUDARuntime() == nullptr) {
        return -1;
    }
    return backend->getCUDARuntime()->device_id();
}

bool shouldUseMappedPagedCache(CUDABackend* backend) {
    if (backend == nullptr || !envFlagEnabled("MNN_PAGED_ATTENTION_ZERO_COPY_CACHE", true)) {
        return false;
    }
    auto runtime = backend->getCUDARuntime();
    if (runtime == nullptr) {
        return false;
    }
    return runtime->prop().integrated != 0 || envFlagEnabled("MNN_PAGED_ATTENTION_FORCE_ZERO_COPY_CACHE", false);
}

int ropeDimForExport(const KVMeta* meta, int headDim) {
    if (meta == nullptr || meta->rope_dim <= 0) {
        return headDim;
    }
    return std::min(headDim, meta->rope_dim);
}

int ropeTypeCode(const KVMeta* meta) {
    if (meta != nullptr && meta->rope_type == "llama3") {
        return 1;
    }
    return 0;
}

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
