#include "AttentionExecution.hpp"
#include "core/MNNFileUtils.h"
#include "core/PagedKVCachePlan.hpp"
#include "core/PrefixCachePath.hpp"
#include "core/PrefixKVHostCache.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace {

constexpr int kCudaPagedPrefixPageSize = 64;
constexpr const char* kCudaPagedPrefixLayout = "cuda-paged-canonical-no-rope-v1";

template <typename T = void>
static inline T* getTensorDevicePtr(const Tensor* tensor) {
    if (!tensor || tensor->deviceId() == 0) {
        return nullptr;
    }
    return reinterpret_cast<T*>(tensor->deviceId());
}

template <typename T>
__device__ static T castPrefixScalar(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __half castPrefixScalar<__half>(float value) {
    return __float2half(value);
}

template <typename T>
__global__ void applyCudaPagedPrefixRopeKernel(T* keyPages, int physicalStart, int logicalStart,
                                               int pageSize, int kvHeads, int headDim,
                                               int srcLen, int ropeDim, float ropeTheta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int half = ropeDim / 2;
    int total = kvHeads * srcLen * half;
    if (index >= total) {
        return;
    }

    int freqDim = index % half;
    int token = (index / half) % srcLen;
    int head = index / (half * srcLen);
    int logicalSeq = logicalStart + token;
    int physicalSlot = physicalStart + token;
    int physicalBlock = physicalSlot / pageSize;
    int pageOffset = physicalSlot - physicalBlock * pageSize;
    size_t base = (((static_cast<size_t>(physicalBlock) * pageSize + pageOffset) *
                    kvHeads + head) * headDim);
    int leftDim = freqDim;
    int rightDim = freqDim + half;
    float left = static_cast<float>(keyPages[base + leftDim]);
    float right = static_cast<float>(keyPages[base + rightDim]);
    float invFreq = 1.0f / powf(ropeTheta, static_cast<float>(2 * freqDim) / static_cast<float>(ropeDim));
    float angle = static_cast<float>(logicalSeq) * invFreq;
    float cosValue = cosf(angle);
    float sinValue = sinf(angle);
    keyPages[base + leftDim] = castPrefixScalar<T>(left * cosValue - right * sinValue);
    keyPages[base + rightDim] = castPrefixScalar<T>(right * cosValue + left * sinValue);
}

template <typename T>
__global__ void loadCudaPagedPrefixKVKernel(const T* rawKeyPages, const T* rawValuePages,
                                            T* keyPages, T* valuePages, const int* blockTable,
                                            int pageSize, int kvHeads, int headDim,
                                            int srcLen, int dstStart,
                                            int ropeDim, float ropeTheta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = kvHeads * srcLen * headDim;
    if (index >= total) {
        return;
    }

    int dim = index % headDim;
    int token = (index / headDim) % srcLen;
    int head = index / (headDim * srcLen);
    int dstSeq = dstStart + token;

    int srcBlock = token / pageSize;
    int srcPageOffset = token - srcBlock * pageSize;
    size_t srcKeyIndex = ((static_cast<size_t>(srcBlock) * pageSize + srcPageOffset) * kvHeads + head) *
                         headDim + dim;
    size_t srcValueIndex = ((static_cast<size_t>(srcBlock) * kvHeads + head) * pageSize + srcPageOffset) *
                           headDim + dim;
    float keyValue = static_cast<float>(rawKeyPages[srcKeyIndex]);
    float value = static_cast<float>(rawValuePages[srcValueIndex]);

    if (dim < ropeDim) {
        int half = ropeDim / 2;
        int pairDim = dim < half ? dim + half : dim - half;
        int freqDim = dim < half ? dim : dim - half;
        size_t pairIndex = ((static_cast<size_t>(srcBlock) * pageSize + srcPageOffset) * kvHeads + head) *
                           headDim + pairDim;
        float pairValue = static_cast<float>(rawKeyPages[pairIndex]);
        float left = dim < half ? keyValue : pairValue;
        float right = dim < half ? pairValue : keyValue;
        float invFreq = 1.0f / powf(ropeTheta, static_cast<float>(2 * freqDim) / static_cast<float>(ropeDim));
        float angle = static_cast<float>(dstSeq) * invFreq;
        float cosValue = cosf(angle);
        float sinValue = sinf(angle);
        keyValue = dim < half ? left * cosValue - right * sinValue : right * cosValue + left * sinValue;
    }

    int logicalBlock = dstSeq / pageSize;
    int pageOffset = dstSeq - logicalBlock * pageSize;
    int physicalBlock = blockTable[logicalBlock];
    keyPages[((static_cast<size_t>(physicalBlock) * pageSize + pageOffset) * kvHeads + head) *
             headDim + dim] = castPrefixScalar<T>(keyValue);
    valuePages[((static_cast<size_t>(physicalBlock) * kvHeads + head) * pageSize + pageOffset) *
               headDim + dim] = castPrefixScalar<T>(value);
}

static size_t cudaPagedFileBytes(int tokenCount, int pageSize, int kvHeads, int headDim, int bytes) {
    int blocks = UP_DIV(tokenCount, pageSize);
    return static_cast<size_t>(blocks) * pageSize * kvHeads * headDim * bytes;
}

static int alignToPage(int value, int pageSize) {
    return UP_DIV(value, pageSize) * pageSize;
}

static double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

struct CudaPrefixSegmentPlacement {
    int logicalStart = 0;
    int physicalStart = 0;
    int tokenCount = 0;
    int physicalLength = 0;
    bool deviceCacheHit = false;
    double deviceCacheLookupMs = 0.0;
    std::string deviceCacheKey;
    std::string keyPath;
    std::string valuePath;
    std::string cacheName;
    cudaEvent_t readyEvent = nullptr;
};

struct CudaPinnedPrefixKVEntry {
    ~CudaPinnedPrefixKVEntry() {
        if (pinned != nullptr) {
            cudaFreeHost(pinned);
        }
    }

    const void* data() const {
        if (pinned != nullptr) {
            return pinned;
        }
        return fallback != nullptr ? fallback->data() : nullptr;
    }

    void* pinned = nullptr;
    std::shared_ptr<std::vector<uint8_t>> fallback;
    bool ok = false;
    bool isPinned = false;
    bool sourceHostCacheHit = false;
    double prefetchWaitMs = 0.0;
    double diskReadMs = 0.0;
    double pinCopyMs = 0.0;
    size_t fileSize = 0;
    std::string error;
};

struct CudaPinnedPrefixKVRead {
    std::shared_ptr<CudaPinnedPrefixKVEntry> entry;
    bool ok = false;
    bool pinnedCacheHit = false;
    bool hostCacheHit = false;
    bool isPinned = false;
    double prefetchWaitMs = 0.0;
    double diskReadMs = 0.0;
    double pinCopyMs = 0.0;
    size_t fileSize = 0;
    std::string error;

    const void* data() const {
        return entry != nullptr ? entry->data() : nullptr;
    }
};

class CudaPinnedPrefixKVHostCache {
public:
    static CudaPinnedPrefixKVHostCache& get() {
        static CudaPinnedPrefixKVHostCache cache;
        return cache;
    }

    CudaPinnedPrefixKVRead readFile(const std::string& path) {
        CudaPinnedPrefixKVRead result;
        std::string identity;
        size_t fileSize = 0;
        if (!MNN::detail::prefixKVFileIdentity(path, identity, &fileSize)) {
            result.error = "prefix KV file does not exist: " + path;
            return result;
        }
        result.fileSize = fileSize;

        std::shared_future<std::shared_ptr<CudaPinnedPrefixKVEntry>> future;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto iter = mEntries.find(identity);
            if (iter != mEntries.end()) {
                future = iter->second;
                result.pinnedCacheHit = future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            } else {
                future = std::async(std::launch::async, [path]() {
                    return readPinnedNow(path);
                }).share();
                mEntries.emplace(identity, future);
            }
        }

        double waitStart = nowMs();
        auto entry = future.get();
        result.prefetchWaitMs = nowMs() - waitStart;
        if (entry != nullptr) {
            result.entry = entry;
            result.ok = entry->ok;
            result.hostCacheHit = result.pinnedCacheHit || entry->sourceHostCacheHit;
            result.isPinned = entry->isPinned;
            if (!result.pinnedCacheHit) {
                result.prefetchWaitMs += entry->prefetchWaitMs;
                result.diskReadMs = entry->diskReadMs;
                result.pinCopyMs = entry->pinCopyMs;
            }
            result.fileSize = entry->fileSize;
            result.error = entry->error;
        }
        if (!result.ok) {
            std::lock_guard<std::mutex> lock(mMutex);
            mEntries.erase(identity);
        }
        return result;
    }

private:
    static std::shared_ptr<CudaPinnedPrefixKVEntry> readPinnedNow(const std::string& path) {
        auto entry = std::make_shared<CudaPinnedPrefixKVEntry>();
        auto source = readPrefixKVHostCacheFile(path);
        entry->sourceHostCacheHit = source.host_cache_hit;
        entry->prefetchWaitMs = source.prefetch_wait_ms;
        entry->diskReadMs = source.disk_read_ms;
        entry->fileSize = source.file_size;
        if (!source.ok || source.bytes == nullptr) {
            entry->error = source.error;
            return entry;
        }

        double pinStart = nowMs();
        void* pinned = nullptr;
        auto allocCode = cudaMallocHost(&pinned, source.bytes->size());
        if (allocCode == cudaSuccess && pinned != nullptr) {
            ::memcpy(pinned, source.bytes->data(), source.bytes->size());
            entry->pinned = pinned;
            entry->isPinned = true;
        } else {
            entry->fallback = source.bytes;
            entry->isPinned = false;
            entry->error = std::string("cudaMallocHost failed, fallback to pageable host memory: ") +
                           cudaGetErrorString(allocCode);
        }
        entry->pinCopyMs = nowMs() - pinStart;
        entry->ok = true;
        entry->fileSize = source.bytes->size();
        return entry;
    }

    std::mutex mMutex;
    std::unordered_map<std::string, std::shared_future<std::shared_ptr<CudaPinnedPrefixKVEntry>>> mEntries;
};

struct CudaDevicePrefixDocumentRecord {
    int physicalStart = 0;
    int physicalLength = 0;
    int tokenCount = 0;
    int ropeDim = 0;
    float ropeTheta = 10000.0f;
    cudaEvent_t readyEvent = nullptr;
};

struct CudaDevicePrefixDocumentPool {
    std::mutex mutex;
    std::unordered_map<std::string, CudaDevicePrefixDocumentRecord> records;
    int nextPhysicalSlot = 0;
};

static std::mutex& cudaDocumentPoolRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<const AttentionExecution::SharedCache*, std::shared_ptr<CudaDevicePrefixDocumentPool>>&
cudaDocumentPoolRegistry() {
    static std::unordered_map<const AttentionExecution::SharedCache*, std::shared_ptr<CudaDevicePrefixDocumentPool>> value;
    return value;
}

static std::shared_ptr<CudaDevicePrefixDocumentPool> cudaDocumentPoolForCache(
        const std::shared_ptr<AttentionExecution::SharedCache>& cache) {
    if (cache == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(cudaDocumentPoolRegistryMutex());
    auto& registry = cudaDocumentPoolRegistry();
    auto iter = registry.find(cache.get());
    if (iter != registry.end()) {
        return iter->second;
    }
    auto pool = std::make_shared<CudaDevicePrefixDocumentPool>();
    registry.emplace(cache.get(), pool);
    return pool;
}

static std::string cudaDocumentPageCacheKey(const std::string& keyPath, const std::string& valuePath,
                                            const KVMeta::PrefixSegment& segment,
                                            int layerIndex, int precision, int kvHeads, int headDim,
                                            int expectedRopeDim, float expectedRopeTheta) {
    std::string keyIdentity;
    std::string valueIdentity;
    size_t keySize = 0;
    size_t valueSize = 0;
    if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keySize) ||
        !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueSize)) {
        return "";
    }
    return "backend=cuda|layout=" + segment.layout +
           "|cache=" + segment.cache_name +
           "|layer=" + std::to_string(layerIndex) +
           "|dtype=" + segment.dtype +
           "|precision=" + std::to_string(precision) +
           "|kv_heads=" + std::to_string(kvHeads) +
           "|head_dim=" + std::to_string(headDim) +
           "|tokens=" + std::to_string(segment.token_count) +
           "|page_size=" + std::to_string(segment.page_size) +
           "|rope_dim=" + std::to_string(expectedRopeDim) +
           "|rope_theta=" + std::to_string(expectedRopeTheta) +
           "|key=" + keyIdentity +
           "|value=" + valueIdentity +
           "|key_size=" + std::to_string(keySize) +
           "|value_size=" + std::to_string(valueSize);
}

} // namespace

class CUDAPrefixAttentionExecution : public AttentionExecution {
public:
    CUDAPrefixAttentionExecution(Backend* backend, bool kvCache, int layerIndex)
        : AttentionExecution(backend, kvCache), mLayerIndex(layerIndex) {
    }

    ~CUDAPrefixAttentionExecution() override {
        unregisterForDevicePrefetch();
        clearDevicePrefetchEvent();
    }

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override {
        auto code = AttentionExecution::onResize(inputs, outputs);
        if (code == NO_ERROR) {
            registerForDevicePrefetch();
        }
        return code;
    }

protected:
    virtual ErrorCode onPrepareKVCacheBeforeAppend(const std::vector<Tensor*>& inputs, cudaStream_t stream,
                                                   bool& prepared, int& appendKvSeqLen) override {
        prepared = false;
        appendKvSeqLen = mNewKvSeqLen;
        if (!mIsKVCacheEnabled || mMeta == nullptr ||
            mMeta->file_flag != KVMeta::PendingReadSegments || mMeta->prefix_segments.empty()) {
            return NO_ERROR;
        }
        if (mBatch != 1) {
            MNN_ERROR("[Error]: CUDA PrefixAttention direct_segments only supports batch=1 now\n");
            return NOT_SUPPORT;
        }
        if (mMeta->prefix_cache_dir.empty()) {
            MNN_ERROR("[Error]: CUDA PrefixAttention missing prefix_cache_dir in KVMeta\n");
            return INVALID_VALUE;
        }
        if ((mMeta->remove > 0 || mMeta->n_reserve > 0) && mMeta->previous != mMeta->remove) {
            MNN_ERROR("[Error]: CUDA PrefixAttention direct_segments append does not support remove/reserve compaction yet\n");
            return NOT_SUPPORT;
        }

        if (mMeta->prefix_device_prefetch && mMeta->prefix_request_id != 0) {
            auto submitErr = submitDevicePrefetchForRequestIfNeeded();
            if (submitErr != NO_ERROR) {
                return submitErr;
            }
            bool prefetchHit = false;
            auto waitErr = consumeDevicePrefetch(stream, prefetchHit, appendKvSeqLen);
            if (waitErr != NO_ERROR) {
                return waitErr;
            }
            if (prefetchHit) {
                prepared = true;
                return NO_ERROR;
            }
            MNN_PRINT("[CUDAPrefixAttention] layer=%d request_id=%llu device_prefetch=1 device_prefetch_hit=0 fallback_sync=1\n",
                      mLayerIndex, static_cast<unsigned long long>(mMeta->prefix_request_id));
        }

        PrefixPrepareSummary summary;
        auto prepareErr = preparePagedPrefixKV(stream, true, false, appendKvSeqLen, summary);
        if (prepareErr != NO_ERROR) {
            return prepareErr;
        }
        prepared = true;
        return NO_ERROR;
    }

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (dst == nullptr) {
            return true;
        }
        auto exe = new CUDAPrefixAttentionExecution(bn, mIsKVCacheEnabled, mLayerIndex);
        exe->mCache = mCache;
        exe->mMeta = mMeta;
        *dst = exe;
        return true;
    }

private:
    struct PrefixPrepareSummary {
        int layerIndex = -1;
        int basePast = 0;
        int segmentTotalTokens = 0;
        int promptTokens = 0;
        int requiredTotal = 0;
        int requiredPhysicalTotal = 0;
        int promptPhysicalStart = 0;
    };

    ErrorCode preparePagedPrefixKV(cudaStream_t stream, bool allowMetaLayerCursor, bool devicePrefetch,
                                   int& appendKvSeqLen, PrefixPrepareSummary& summary) {
        int segmentTotalTokens = mMeta->segment_total_tokens;
        if (segmentTotalTokens <= 0) {
            for (const auto& segment : mMeta->prefix_segments) {
                segmentTotalTokens += segment.token_count;
            }
            mMeta->segment_total_tokens = segmentTotalTokens;
        }
        if (segmentTotalTokens <= 0) {
            return INVALID_VALUE;
        }

        appendKvSeqLen = static_cast<int>(mMeta->add);
        if (appendKvSeqLen <= 0) {
            appendKvSeqLen = mNewKvSeqLen;
        }
        if (!buildPrefixRuntimeKVBlockTablePlan(mPagedKVPlan, mMeta, appendKvSeqLen)) {
            return INVALID_VALUE;
        }
        int basePast = 0;
        if (mMeta->previous != mMeta->remove) {
            basePast = mCache != nullptr ? mCache->mPastLength : 0;
            if (basePast <= 0) {
                basePast = static_cast<int>(mMeta->previous - mMeta->remove);
            }
            if (!mCache->mPagedActive || !mCache->mPagedKey || !mCache->mPagedValue ||
                !mCache->mPagedBlockTable || !mCache->mPagedRopeTable) {
                MNN_ERROR("[Error]: CUDA PrefixAttention direct_segments append requires existing paged runtime KV cache; reset before first direct_segments request\n");
                return NOT_SUPPORT;
            }
        }
        const int requiredTotal = basePast + segmentTotalTokens + appendKvSeqLen;

        int layerIndex = mLayerIndex;
        if (layerIndex < 0) {
            if (!allowMetaLayerCursor) {
                MNN_ERROR("[Error]: CUDA PrefixAttention device_prefetch requires explicit AttentionParam.layer_index\n");
                return NOT_SUPPORT;
            }
            layerIndex = mMeta->layer_index;
            if (mMeta->layer_nums > 0) {
                mMeta->layer_index = (mMeta->layer_index + 1) % mMeta->layer_nums;
            } else {
                mMeta->layer_index++;
            }
        }

        auto docPool = cudaDocumentPoolForCache(mCache);
        if (docPool == nullptr) {
            return OUT_OF_MEMORY;
        }

        std::vector<CudaPrefixSegmentPlacement> placements;
        placements.reserve(mMeta->prefix_segments.size());
        int logicalCursor = basePast;
        int physicalCursor = std::max(mCache != nullptr ? mCache->mPagedPhysicalLength : 0, basePast);
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            physicalCursor = std::max(physicalCursor, docPool->nextPhysicalSlot);
        }
        int ropeDimForRequest = 0;
        float ropeThetaForRequest = 10000.0f;

        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            if (segment.key_rope_state != KVMeta::KeyRopeCanonicalNoRope ||
                segment.rope_pairing != KVMeta::RopePairingHalf) {
                MNN_ERROR("[Error]: CUDA PrefixAttention requires canonical_no_rope half-split prefix cache: %s\n",
                          segment.cache_name.c_str());
                return NOT_SUPPORT;
            }
            if (segment.backend != "cuda" || segment.layout != kCudaPagedPrefixLayout ||
                segment.page_size != kCudaPagedPrefixPageSize) {
                MNN_ERROR("[Error]: CUDA PrefixAttention requires backend-native paged cache for %s, got backend=%s layout=%s page_size=%d\n",
                          segment.cache_name.c_str(), segment.backend.c_str(), segment.layout.c_str(), segment.page_size);
                return NOT_SUPPORT;
            }
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : mHeadDim;
            ropeDim = std::min(ropeDim, mHeadDim);
            if (ropeDim <= 0 || (ropeDim % 2) != 0 || segment.rope_theta <= 0.0f) {
                MNN_ERROR("[Error]: CUDA PrefixAttention invalid RoPE metadata for cache %s\n",
                          segment.cache_name.c_str());
                return INVALID_VALUE;
            }
            if (ropeDimForRequest == 0) {
                ropeDimForRequest = ropeDim;
                ropeThetaForRequest = segment.rope_theta;
            } else if (ropeDimForRequest != ropeDim || std::fabs(ropeThetaForRequest - segment.rope_theta) > 1e-5f) {
                MNN_ERROR("[Error]: CUDA PrefixAttention requires consistent RoPE metadata across direct_segments\n");
                return NOT_SUPPORT;
            }

            auto basePath = prefixCacheLayerBase(mMeta->prefix_cache_dir, "cuda", segment.cache_name, layerIndex);
            auto keyPath = basePath + ".k";
            auto valuePath = basePath + ".v";
            auto expectedBytes = cudaPagedFileBytes(segment.token_count, kCudaPagedPrefixPageSize,
                                                   mKvNumHead, mHeadDim, mPrecision);
            std::string keyIdentity;
            std::string valueIdentity;
            size_t keyFileSize = 0;
            size_t valueFileSize = 0;
            if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keyFileSize) ||
                !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueFileSize)) {
                MNN_ERROR("[Error]: CUDA PrefixAttention missing native paged KV file for %s layer %d\n",
                          segment.cache_name.c_str(), layerIndex);
                return FILE_OPEN_FAILED;
            }
            if (keyFileSize != expectedBytes || valueFileSize != expectedBytes) {
                MNN_ERROR("[Error]: CUDA PrefixAttention invalid native paged KV size for %s layer %d: key=%zu value=%zu expected=%zu\n",
                          segment.cache_name.c_str(), layerIndex, keyFileSize, valueFileSize, expectedBytes);
                return INVALID_VALUE;
            }

            auto cacheKey = cudaDocumentPageCacheKey(keyPath, valuePath, segment, layerIndex,
                                                     mPrecision, mKvNumHead, mHeadDim,
                                                     ropeDim, segment.rope_theta);
            if (cacheKey.empty()) {
                return FILE_OPEN_FAILED;
            }
            CudaPrefixSegmentPlacement placement;
            placement.logicalStart = logicalCursor;
            placement.tokenCount = segment.token_count;
            placement.physicalLength = alignToPage(segment.token_count, kCudaPagedPrefixPageSize);
            placement.deviceCacheKey = cacheKey;
            placement.keyPath = keyPath;
            placement.valuePath = valuePath;
            placement.cacheName = segment.cache_name;

            double lookupStart = nowMs();
            {
                std::lock_guard<std::mutex> lock(docPool->mutex);
                auto iter = docPool->records.find(cacheKey);
                if (iter != docPool->records.end() &&
                    iter->second.tokenCount == segment.token_count &&
                    iter->second.ropeDim == ropeDim &&
                    std::fabs(iter->second.ropeTheta - segment.rope_theta) <= 1e-5f) {
                    placement.deviceCacheHit = true;
                    placement.physicalStart = iter->second.physicalStart;
                    placement.physicalLength = iter->second.physicalLength;
                    placement.readyEvent = iter->second.readyEvent;
                } else {
                    int physicalStart = alignToPage(std::max(physicalCursor, docPool->nextPhysicalSlot),
                                                    kCudaPagedPrefixPageSize);
                    placement.physicalStart = physicalStart;
                    docPool->nextPhysicalSlot = physicalStart + placement.physicalLength;
                    physicalCursor = docPool->nextPhysicalSlot;
                }
            }
            placement.deviceCacheLookupMs = nowMs() - lookupStart;
            physicalCursor = std::max(physicalCursor, placement.physicalStart + placement.physicalLength);
            placements.emplace_back(placement);
            logicalCursor += segment.token_count;
        }

        const int promptLogicalStart = basePast + segmentTotalTokens;
        const int promptPhysicalStart = alignToPage(physicalCursor, kCudaPagedPrefixPageSize);
        physicalCursor = promptPhysicalStart + appendKvSeqLen;
        const int requiredPhysicalTotal = alignToPage(physicalCursor, kCudaPagedPrefixPageSize);
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            docPool->nextPhysicalSlot = std::max(docPool->nextPhysicalSlot, requiredPhysicalTotal);
        }

        auto err = reallocKVCache_gpu(requiredTotal, mBatch, mKvNumHead, mHeadDim, stream);
        if (err != NO_ERROR) {
            return err;
        }
        auto pagedErr = ensurePagedKVCache_gpu(requiredPhysicalTotal, mBatch, mKvNumHead, mHeadDim, stream);
        if (pagedErr != NO_ERROR) {
            return pagedErr;
        }
        resetPagedKVCacheState(basePast);

        std::vector<int> slotTable;
        std::vector<int> ropeTable;
        slotTable.resize(std::max(0, requiredTotal - basePast));
        ropeTable.resize(slotTable.size(), 0);
        for (const auto& placement : placements) {
            for (int i = 0; i < placement.tokenCount; ++i) {
                int logicalOffset = placement.logicalStart + i - basePast;
                slotTable[logicalOffset] = placement.physicalStart + i;
                ropeTable[logicalOffset] = 1;
            }
        }
        for (int i = 0; i < appendKvSeqLen; ++i) {
            int logicalOffset = promptLogicalStart + i - basePast;
            slotTable[logicalOffset] = promptPhysicalStart + i;
            ropeTable[logicalOffset] = 0;
        }
        if (!slotTable.empty()) {
            auto tableCopy = cudaMemcpyAsync(getTensorDevicePtr<int>(mCache->mPagedBlockTable.get()) + basePast,
                                             slotTable.data(), slotTable.size() * sizeof(int),
                                             cudaMemcpyHostToDevice, stream);
            auto ropeCopy = cudaMemcpyAsync(getTensorDevicePtr<int>(mCache->mPagedRopeTable.get()) + basePast,
                                            ropeTable.data(), ropeTable.size() * sizeof(int),
                                            cudaMemcpyHostToDevice, stream);
            if (tableCopy != cudaSuccess || ropeCopy != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to upload paged token tables: %s / %s\n",
                          cudaGetErrorString(tableCopy), cudaGetErrorString(ropeCopy));
                return INVALID_VALUE;
            }
        }
        mCache->mPagedTokenTableCustom = true;
        mCache->mPagedPhysicalLength = requiredPhysicalTotal;
        mCache->mPagedRopeDim = ropeDimForRequest;
        mCache->mPagedRopeTheta = ropeThetaForRequest;

        size_t placementIndex = 0;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            const auto& placement = placements[placementIndex++];
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : mHeadDim;
            ropeDim = std::min(ropeDim, mHeadDim);
            const size_t expectedBytes = cudaPagedFileBytes(segment.token_count, kCudaPagedPrefixPageSize,
                                                           mKvNumHead, mHeadDim, mPrecision);

            bool hostCacheHit = false;
            bool pinnedHost = false;
            bool pinnedCacheHit = false;
            double prefetchWaitMs = 0.0;
            double diskReadMs = 0.0;
            double pinnedCopyMs = 0.0;
            double h2dEnqueueMs = 0.0;
            double materializeEnqueueMs = 0.0;
            double attentionWaitMs = 0.0;
            size_t uploadedBytes = 0;

            if (placement.deviceCacheHit) {
                if (placement.readyEvent != nullptr) {
                    double waitStart = nowMs();
                    auto waitErr = cudaStreamWaitEvent(stream, placement.readyEvent, 0);
                    attentionWaitMs = nowMs() - waitStart;
                    if (waitErr != cudaSuccess) {
                        MNN_ERROR("[Error]: CUDA PrefixAttention failed to wait document page event: %s\n",
                                  cudaGetErrorString(waitErr));
                        return INVALID_VALUE;
                    }
                }
            } else {
                auto keyRead = CudaPinnedPrefixKVHostCache::get().readFile(placement.keyPath);
                auto valueRead = CudaPinnedPrefixKVHostCache::get().readFile(placement.valuePath);
                if (!keyRead.ok || !valueRead.ok || keyRead.data() == nullptr || valueRead.data() == nullptr) {
                    MNN_ERROR("[Error]: CUDA PrefixAttention failed to read prefix KV cache: %s / %s, key_error=%s value_error=%s\n",
                              placement.keyPath.c_str(), placement.valuePath.c_str(), keyRead.error.c_str(), valueRead.error.c_str());
                    return FILE_OPEN_FAILED;
                }
                if (keyRead.fileSize != expectedBytes || valueRead.fileSize != expectedBytes) {
                    MNN_ERROR("[Error]: CUDA PrefixAttention invalid native paged KV size for %s layer %d: key=%zu value=%zu expected=%zu\n",
                              segment.cache_name.c_str(), layerIndex, keyRead.fileSize, valueRead.fileSize, expectedBytes);
                    return INVALID_VALUE;
                }

                double h2dStart = nowMs();
                auto* keyPages = getTensorDevicePtr<uint8_t>(mCache->mPagedKey.get());
                auto* valuePages = getTensorDevicePtr<uint8_t>(mCache->mPagedValue.get());
                size_t dstByteOffset = static_cast<size_t>(placement.physicalStart) * mKvNumHead * mHeadDim * mPrecision;
                auto keyCopy = cudaMemcpyAsync(keyPages + dstByteOffset, keyRead.data(), keyRead.fileSize, cudaMemcpyHostToDevice, stream);
                auto valueCopy = cudaMemcpyAsync(valuePages + dstByteOffset, valueRead.data(), valueRead.fileSize, cudaMemcpyHostToDevice, stream);
                if (keyCopy != cudaSuccess || valueCopy != cudaSuccess) {
                    MNN_ERROR("[Error]: CUDA PrefixAttention failed to copy prefix KV into runtime pages: %s / %s\n",
                              cudaGetErrorString(keyCopy), cudaGetErrorString(valueCopy));
                    return INVALID_VALUE;
                }
                h2dEnqueueMs = nowMs() - h2dStart;
                uploadedBytes = keyRead.fileSize + valueRead.fileSize;
                hostCacheHit = keyRead.hostCacheHit && valueRead.hostCacheHit;
                pinnedHost = keyRead.isPinned && valueRead.isPinned;
                pinnedCacheHit = keyRead.pinnedCacheHit && valueRead.pinnedCacheHit;
                prefetchWaitMs = keyRead.prefetchWaitMs + valueRead.prefetchWaitMs;
                diskReadMs = keyRead.diskReadMs + valueRead.diskReadMs;
                pinnedCopyMs = keyRead.pinCopyMs + valueRead.pinCopyMs;

                cudaEvent_t readyEvent = nullptr;
                auto eventErr = cudaEventCreateWithFlags(&readyEvent, cudaEventDisableTiming);
                if (eventErr == cudaSuccess) {
                    eventErr = cudaEventRecord(readyEvent, stream);
                }
                if (eventErr != cudaSuccess) {
                    if (readyEvent != nullptr) {
                        cudaEventDestroy(readyEvent);
                    }
                    MNN_ERROR("[Error]: CUDA PrefixAttention failed to record document page event: %s\n",
                              cudaGetErrorString(eventErr));
                    return INVALID_VALUE;
                }
                {
                    std::lock_guard<std::mutex> lock(docPool->mutex);
                    CudaDevicePrefixDocumentRecord record;
                    record.physicalStart = placement.physicalStart;
                    record.physicalLength = placement.physicalLength;
                    record.tokenCount = segment.token_count;
                    record.ropeDim = ropeDim;
                    record.ropeTheta = segment.rope_theta;
                    record.readyEvent = readyEvent;
                    docPool->records[placement.deviceCacheKey] = record;
                }
            }

            MNN_PRINT("[CUDAPrefixAttention] layer=%d cache=%s tokens=%d logical_start=%d physical_start=%d host_cache_hit=%d pinned_host=%d pinned_cache_hit=%d device_cache_hit=%d device_cache_lookup_ms=%.3f device_prefetch=%d device_prefetch_submit=%d prefetch_wait_ms=%.3f disk_read_ms=%.3f pinned_copy_ms=%.3f host_to_device_ms=%.3f materialize_ms=%.3f attention_wait_ms=%.3f h2d_enqueue_bytes=%zu staging_path=0 on_read_rope=1\n",
                      layerIndex, placement.cacheName.c_str(), segment.token_count,
                      placement.logicalStart, placement.physicalStart,
                      hostCacheHit ? 1 : 0,
                      pinnedHost ? 1 : 0,
                      pinnedCacheHit ? 1 : 0,
                      placement.deviceCacheHit ? 1 : 0,
                      placement.deviceCacheLookupMs,
                      devicePrefetch ? 1 : 0,
                      devicePrefetch ? 1 : 0,
                      prefetchWaitMs,
                      diskReadMs,
                      pinnedCopyMs,
                      h2dEnqueueMs,
                      materializeEnqueueMs,
                      attentionWaitMs,
                      uploadedBytes);
        }

        if (logicalCursor != basePast + segmentTotalTokens) {
            MNN_ERROR("[Error]: CUDA PrefixAttention copied prefix length %d but expected %d\n",
                      logicalCursor - basePast, segmentTotalTokens);
            return INVALID_VALUE;
        }

        mCache->mPastLength = basePast + segmentTotalTokens;
        mCache->mPagedLength = basePast + segmentTotalTokens;
        if (layerIndex == 0) {
            MNN_PRINT("[CUDAPrefixAttention] block_table base_past=%d prefix_tokens=%d prompt_tokens=%d required_total=%d physical_total=%d page_size=%d logical_blocks=%d physical_blocks=%d prefix_blocks=%d prompt_blocks=%d prompt_physical_start=%d device_prefetch=%d\n",
                      basePast, segmentTotalTokens, appendKvSeqLen, requiredTotal, requiredPhysicalTotal,
                      kCudaPagedPrefixPageSize,
                      UP_DIV(requiredTotal, kCudaPagedPrefixPageSize),
                      UP_DIV(requiredPhysicalTotal, kCudaPagedPrefixPageSize),
                      UP_DIV(segmentTotalTokens, kCudaPagedPrefixPageSize),
                      UP_DIV(appendKvSeqLen, kCudaPagedPrefixPageSize),
                      promptPhysicalStart,
                      devicePrefetch ? 1 : 0);
        }
        summary.layerIndex = layerIndex;
        summary.basePast = basePast;
        summary.segmentTotalTokens = segmentTotalTokens;
        summary.promptTokens = appendKvSeqLen;
        summary.requiredTotal = requiredTotal;
        summary.requiredPhysicalTotal = requiredPhysicalTotal;
        summary.promptPhysicalStart = promptPhysicalStart;
        return NO_ERROR;
    }

    ErrorCode enqueueDevicePrefetch(cudaStream_t copyStream, uint64_t requestId) {
        if (mDevicePrefetchRequestId == requestId) {
            return mDevicePrefetchStatus;
        }
        clearDevicePrefetchEvent();
        mDevicePrefetchRequestId = requestId;
        mDevicePrefetchStatus = NO_ERROR;
        mDevicePrefetchAppendKvSeqLen = 0;
        mDevicePrefetchSummary = PrefixPrepareSummary();

        auto start = nowMs();
        auto err = preparePagedPrefixKV(copyStream, false, true,
                                        mDevicePrefetchAppendKvSeqLen,
                                        mDevicePrefetchSummary);
        if (err != NO_ERROR) {
            mDevicePrefetchStatus = err;
            return err;
        }
        cudaEvent_t event = nullptr;
        auto eventErr = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (eventErr != cudaSuccess) {
            MNN_ERROR("[Error]: CUDA PrefixAttention failed to create device_prefetch event: %s\n",
                      cudaGetErrorString(eventErr));
            mDevicePrefetchStatus = INVALID_VALUE;
            return mDevicePrefetchStatus;
        }
        eventErr = cudaEventRecord(event, copyStream);
        if (eventErr != cudaSuccess) {
            MNN_ERROR("[Error]: CUDA PrefixAttention failed to record device_prefetch event: %s\n",
                      cudaGetErrorString(eventErr));
            cudaEventDestroy(event);
            mDevicePrefetchStatus = INVALID_VALUE;
            return mDevicePrefetchStatus;
        }
        mDevicePrefetchReadyEvent = event;
        MNN_PRINT("[CUDAPrefixAttentionPrefetch] layer=%d request_id=%llu submit_ms=%.3f prefix_tokens=%d prompt_tokens=%d physical_total=%d\n",
                  mDevicePrefetchSummary.layerIndex,
                  static_cast<unsigned long long>(requestId),
                  nowMs() - start,
                  mDevicePrefetchSummary.segmentTotalTokens,
                  mDevicePrefetchSummary.promptTokens,
                  mDevicePrefetchSummary.requiredPhysicalTotal);
        return NO_ERROR;
    }

    ErrorCode consumeDevicePrefetch(cudaStream_t stream, bool& hit, int& appendKvSeqLen) {
        hit = false;
        if (mMeta == nullptr || mDevicePrefetchRequestId != mMeta->prefix_request_id ||
            mDevicePrefetchRequestId == 0) {
            return NO_ERROR;
        }
        if (mDevicePrefetchStatus != NO_ERROR) {
            return mDevicePrefetchStatus;
        }
        if (mDevicePrefetchReadyEvent == nullptr) {
            return NO_ERROR;
        }
        double waitStart = nowMs();
        auto waitErr = cudaStreamWaitEvent(stream, mDevicePrefetchReadyEvent, 0);
        double waitMs = nowMs() - waitStart;
        if (waitErr != cudaSuccess) {
            MNN_ERROR("[Error]: CUDA PrefixAttention failed to wait device_prefetch event: %s\n",
                      cudaGetErrorString(waitErr));
            return INVALID_VALUE;
        }
        appendKvSeqLen = mDevicePrefetchAppendKvSeqLen;
        hit = true;
        MNN_PRINT("[CUDAPrefixAttention] layer=%d request_id=%llu device_prefetch=1 device_prefetch_hit=1 attention_wait_ms=%.3f prefix_tokens=%d prompt_tokens=%d required_total=%d physical_total=%d prompt_physical_start=%d\n",
                  mDevicePrefetchSummary.layerIndex,
                  static_cast<unsigned long long>(mDevicePrefetchRequestId),
                  waitMs,
                  mDevicePrefetchSummary.segmentTotalTokens,
                  mDevicePrefetchSummary.promptTokens,
                  mDevicePrefetchSummary.requiredTotal,
                  mDevicePrefetchSummary.requiredPhysicalTotal,
                  mDevicePrefetchSummary.promptPhysicalStart);
        return NO_ERROR;
    }

    ErrorCode submitDevicePrefetchForRequestIfNeeded() {
        if (mMeta == nullptr || !mMeta->prefix_device_prefetch || mMeta->prefix_request_id == 0) {
            return NO_ERROR;
        }
        if (mLayerIndex != 0) {
            return NO_ERROR;
        }
        auto copyStream = devicePrefetchStream();
        if (copyStream == nullptr) {
            MNN_PRINT("[CUDAPrefixAttentionPrefetch] request_id=%llu unavailable_copy_stream=1 fallback_sync=1\n",
                      static_cast<unsigned long long>(mMeta->prefix_request_id));
            return NO_ERROR;
        }

        std::vector<CUDAPrefixAttentionExecution*> targets;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            auto& submitted = submittedRequests();
            auto submittedIter = submitted.find(mMeta);
            if (submittedIter != submitted.end() && submittedIter->second == mMeta->prefix_request_id) {
                return NO_ERROR;
            }

            auto registryIter = registry().find(mMeta);
            if (registryIter != registry().end()) {
                std::unordered_map<int, CUDAPrefixAttentionExecution*> byLayer;
                for (auto* exe : registryIter->second) {
                    if (exe == nullptr || exe->mMeta != mMeta || exe->mLayerIndex < 0 ||
                        exe->mBatch <= 0 || exe->mKvNumHead <= 0 || exe->mHeadDim <= 0 ||
                        exe->mCache == nullptr) {
                        continue;
                    }
                    byLayer[exe->mLayerIndex] = exe;
                }
                targets.reserve(byLayer.size());
                for (auto& iter : byLayer) {
                    targets.emplace_back(iter.second);
                }
            }
            if (targets.empty()) {
                return NO_ERROR;
            }
            submitted[mMeta] = mMeta->prefix_request_id;
        }

        std::sort(targets.begin(), targets.end(), [](const CUDAPrefixAttentionExecution* left,
                                                     const CUDAPrefixAttentionExecution* right) {
            return left->mLayerIndex < right->mLayerIndex;
        });

        MNN_PRINT("[CUDAPrefixAttentionPrefetch] request_id=%llu submit_layers=%zu stream=copy\n",
                  static_cast<unsigned long long>(mMeta->prefix_request_id),
                  targets.size());
        for (auto* target : targets) {
            auto err = target->enqueueDevicePrefetch(copyStream, mMeta->prefix_request_id);
            if (err != NO_ERROR) {
                return err;
            }
        }
        return NO_ERROR;
    }

    void clearDevicePrefetchEvent() {
        if (mDevicePrefetchReadyEvent != nullptr) {
            cudaEventDestroy(mDevicePrefetchReadyEvent);
            mDevicePrefetchReadyEvent = nullptr;
        }
    }

    void registerForDevicePrefetch() {
        if (mRegisteredForDevicePrefetch || mMeta == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(registryMutex());
        auto& list = registry()[mMeta];
        if (std::find(list.begin(), list.end(), this) == list.end()) {
            list.emplace_back(this);
        }
        mRegisteredForDevicePrefetch = true;
    }

    void unregisterForDevicePrefetch() {
        if (!mRegisteredForDevicePrefetch || mMeta == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(registryMutex());
        auto iter = registry().find(mMeta);
        if (iter != registry().end()) {
            auto& list = iter->second;
            list.erase(std::remove(list.begin(), list.end(), this), list.end());
            if (list.empty()) {
                registry().erase(iter);
                submittedRequests().erase(mMeta);
            }
        }
        mRegisteredForDevicePrefetch = false;
    }

    static cudaStream_t devicePrefetchStream() {
        static cudaStream_t stream = []() {
            cudaStream_t created = nullptr;
            auto err = cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to create device prefetch stream: %s\n",
                          cudaGetErrorString(err));
                return static_cast<cudaStream_t>(nullptr);
            }
            return created;
        }();
        return stream;
    }

    static std::mutex& registryMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_map<KVMeta*, std::vector<CUDAPrefixAttentionExecution*>>& registry() {
        static std::unordered_map<KVMeta*, std::vector<CUDAPrefixAttentionExecution*>> value;
        return value;
    }

    static std::unordered_map<KVMeta*, uint64_t>& submittedRequests() {
        static std::unordered_map<KVMeta*, uint64_t> value;
        return value;
    }

    int mLayerIndex = -1;
    RuntimeKVBlockTablePlan mPagedKVPlan;
    bool mRegisteredForDevicePrefetch = false;
    uint64_t mDevicePrefetchRequestId = 0;
    cudaEvent_t mDevicePrefetchReadyEvent = nullptr;
    ErrorCode mDevicePrefetchStatus = NO_ERROR;
    int mDevicePrefetchAppendKvSeqLen = 0;
    PrefixPrepareSummary mDevicePrefetchSummary;
};

class CUDAPrefixAttentionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        bool kvCache = true;
        int layerIndex = -1;
        if (op->main_type() == OpParameter_AttentionParam) {
            auto param = op->main_as_AttentionParam();
            if (param != nullptr) {
                kvCache = param->kv_cache();
                layerIndex = param->layer_index();
            }
        }
        return new CUDAPrefixAttentionExecution(backend, kvCache, layerIndex);
    }
};

static CUDACreatorRegister<CUDAPrefixAttentionCreator> __init_prefix_attention(OpType_PrefixAttention);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
