#ifndef MNN_CUDA_PAGED_ATTENTION_CUDA_UTILS_HPP
#define MNN_CUDA_PAGED_ATTENTION_CUDA_UTILS_HPP

#include "PagedAttentionExecution.hpp"
#include "backend/cuda/core/CUDATools.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

struct CUDAPagedAttention::SharedPagedCache::MappedBuffer {
    void* host = nullptr;
    void* device = nullptr;
    size_t bytes = 0;
    int deviceId = -1;

    ~MappedBuffer();
};

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(std::string name, bool enabled = true) {
#ifdef MNN_CUDA_PROFILE
        mEnabled = enabled;
        if (mEnabled) {
            mName = std::move(name);
            NVTX_PUSH(mName.c_str());
        }
#else
        (void)name;
        (void)enabled;
#endif
    }
    ~ScopedNvtxRange() {
#ifdef MNN_CUDA_PROFILE
        if (mEnabled) {
            NVTX_POP();
        }
#endif
    }

private:
#ifdef MNN_CUDA_PROFILE
    bool mEnabled = false;
    std::string mName;
#endif
};

struct MappedPagedExportWorkspace {
    std::mutex mutex;
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> scratch;
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> valueStorage;
};

bool profilePagedAttention();
bool nvtxPagedAttention();
uint64_t nowUs();
std::string nvtxLayerRangeName(const char* op, int layerIndex, int queryLen, int insertLen, int kvLen);
bool envFlagEnabled(const char* name, bool defaultValue);

std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> makeMappedPagedBuffer(size_t bytes,
                                                                                          int deviceId);
bool ensureMappedHostWorkspace(std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>* buffer,
                               size_t bytes, int deviceId);
MappedPagedExportWorkspace& mappedPagedExportWorkspace(int deviceId);
int cudaBackendDeviceId(CUDABackend* backend);
bool shouldUseMappedPagedCache(CUDABackend* backend);

int picCacheSourceSlotBaseCUDA(const PagedKVMeta* meta, int kvLen);
size_t picCacheSourceSlotCountCUDA(const PagedKVMeta* meta);
int maxSlotsWithPicSourceSlotsCUDA(const PagedKVMeta* meta, int kvLen);

bool writeBinaryFile(const std::string& path, const void* data, size_t bytes);
bool readBinaryFileRange(const std::string& path, size_t offsetBytes, void* dst, size_t expectedBytes);
bool readExternalValueSegmentToBuffer(const std::string& path, void* dstBuffer, size_t dstBytes, int batch,
                                      int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset,
                                      size_t tokenCount, int headDim, int bytes);
bool readExternalValueSegmentToPagedCache(const std::string& path, void* valueCacheHost, int batch,
                                          int kvHeads, int maxSlots, size_t logicalStart,
                                          size_t sourceTokenCount, size_t sourceTokenOffset,
                                          size_t tokenCount, int headDim, int bytes);

int ropeDimForExport(const KVMeta* meta, int headDim);
int ropeTypeCode(const KVMeta* meta);

void registerExternalLayerMappedTarget(
    const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim, int bytes, int maxSlots,
    const std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>& key,
    const std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>& value);
ErrorCode restoreExternalSegmentsCUDA(PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim,
                                      int bytes, int maxSlots, int kvLen, void* keyCacheDevice);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN

#endif // MNN_CUDA_PAGED_ATTENTION_CUDA_UTILS_HPP
