#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/AttentionBufExecution.hpp"
#include "core/MNNFileUtils.h"
#include "core/PagedKVCachePlan.hpp"
#include "core/PrefixCachePath.hpp"
#include "core/PrefixKVHostCache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace MNN {
namespace OpenCL {

namespace {

constexpr const char* kOpenCLPrefixLayout = "opencl-buffer-canonical-no-rope-v1";

static size_t openCLNativeFileBytes(int tokenCount, int kvHeads, int headDim, int bytes) {
    int alignedTokens = ROUND_UP(tokenCount, 4);
    return static_cast<size_t>(alignedTokens) * kvHeads * headDim * bytes;
}

static double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

static cl_int enqueueWaitForEvent(cl::CommandQueue& queue, const cl::Event& event) {
    if (event() == nullptr) {
        return CL_SUCCESS;
    }
    cl_event rawEvent = event();
    return clEnqueueWaitForEvents(queue(), 1, &rawEvent);
}

static int alignToPage(int value, int pageSize) {
    return UP_DIV(value, pageSize) * pageSize;
}

struct OpenCLPrefixSegmentPlacement {
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
    cl::Event readyEvent;
};

struct OpenCLDevicePrefixDocumentRecord {
    int physicalStart = 0;
    int physicalLength = 0;
    int tokenCount = 0;
    int ropeDim = 0;
    float ropeTheta = 10000.0f;
    cl::Event readyEvent;
};

struct OpenCLDevicePrefixDocumentPool {
    std::mutex mutex;
    std::unordered_map<std::string, OpenCLDevicePrefixDocumentRecord> records;
    int nextPhysicalSlot = 0;
};

static std::mutex& openCLDocumentPoolRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<const KVCacheCLManager*, std::shared_ptr<OpenCLDevicePrefixDocumentPool>>&
openCLDocumentPoolRegistry() {
    static std::unordered_map<const KVCacheCLManager*, std::shared_ptr<OpenCLDevicePrefixDocumentPool>> value;
    return value;
}

static std::shared_ptr<OpenCLDevicePrefixDocumentPool> openCLDocumentPoolForCache(
        const std::shared_ptr<KVCacheCLManager>& cache) {
    if (cache == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(openCLDocumentPoolRegistryMutex());
    auto& registry = openCLDocumentPoolRegistry();
    auto iter = registry.find(cache.get());
    if (iter != registry.end()) {
        return iter->second;
    }
    auto pool = std::make_shared<OpenCLDevicePrefixDocumentPool>();
    registry.emplace(cache.get(), pool);
    return pool;
}

static std::mutex& openCLPrefixLayerCacheRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<int, std::shared_ptr<KVCacheCLManager>>&
openCLPrefixLayerCacheRegistry() {
    static std::unordered_map<int, std::shared_ptr<KVCacheCLManager>> value;
    return value;
}

static std::string openCLDocumentPageCacheKey(const std::string& keyPath, const std::string& valuePath,
                                              const KVMeta::PrefixSegment& segment,
                                              int layerIndex, int runtimeBytes, int kvHeads, int headDim,
                                              int expectedRopeDim, float expectedRopeTheta) {
    std::string keyIdentity;
    std::string valueIdentity;
    size_t keySize = 0;
    size_t valueSize = 0;
    if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keySize) ||
        !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueSize)) {
        return "";
    }
    return "backend=opencl|layout=" + segment.layout +
           "|cache=" + segment.cache_name +
           "|layer=" + std::to_string(layerIndex) +
           "|dtype=" + segment.dtype +
           "|runtime_bytes=" + std::to_string(runtimeBytes) +
           "|kv_heads=" + std::to_string(kvHeads) +
           "|head_dim=" + std::to_string(headDim) +
           "|tokens=" + std::to_string(segment.token_count) +
           "|rope_dim=" + std::to_string(expectedRopeDim) +
           "|rope_theta=" + std::to_string(expectedRopeTheta) +
           "|key=" + keyIdentity +
           "|value=" + valueIdentity +
           "|key_size=" + std::to_string(keySize) +
           "|value_size=" + std::to_string(valueSize);
}

} // namespace

class OpenCLPrefixAttentionBufExecution : public AttentionBufExecution {
public:
    OpenCLPrefixAttentionBufExecution(const MNN::Op* op, Backend* backend, bool kvCache, int layerIndex)
        : AttentionBufExecution(op, backend, kvCache), mLayerIndex(layerIndex) {
        adoptSharedKVCacheManager();
    }

    OpenCLPrefixAttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op* op,
                                      Backend* backend, int layerIndex)
        : AttentionBufExecution(manager, op, backend), mLayerIndex(layerIndex) {
        adoptSharedKVCacheManager();
    }

    ~OpenCLPrefixAttentionBufExecution() override {
        unregisterForDevicePrefetch();
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        ErrorCode code = NO_ERROR;
        if (isDirectSegmentsPrefix()) {
            setRecordClose closeRecord(mOpenCLBackend);
            code = AttentionBufExecution::onResize(inputs, outputs);
        } else {
            code = AttentionBufExecution::onResize(inputs, outputs);
        }
        if (code == NO_ERROR) {
            registerForDevicePrefetch();
        }
        return code;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (isDirectSegmentsPrefix()) {
            setRecordClose closeRecord(mOpenCLBackend);
            return AttentionBufExecution::onExecute(inputs, outputs);
        }
        return AttentionBufExecution::onExecute(inputs, outputs);
    }

protected:
    virtual int onResizePrefixKVLength(const std::vector<Tensor*>& inputs, int seqlen) override {
        (void)seqlen;
        if (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingReadSegments ||
            mMeta->prefix_segments.empty()) {
            return 0;
        }
        int segmentTotalTokens = mMeta->segment_total_tokens;
        if (segmentTotalTokens <= 0) {
            for (const auto& segment : mMeta->prefix_segments) {
                segmentTotalTokens += segment.token_count;
            }
        }
        if (segmentTotalTokens > 0 && mKVCacheCLManager != nullptr &&
            buildPrefixRuntimeKVBlockTablePlan(mPagedKVPlan, mMeta, seqlen)) {
            int basePast = 0;
            if (mMeta->previous != mMeta->remove) {
                basePast = mKVCacheCLManager->pastKvLength();
                if (basePast <= 0) {
                    basePast = static_cast<int>(mMeta->previous - mMeta->remove);
                }
                if (!mKVCacheCLManager->pagedActive()) {
                    int headDim = inputs.size() > 1 && inputs[1] != nullptr &&
                                  inputs[1]->dimensions() > 3 ? inputs[1]->shape()[3] : 0;
                    int restoreRopeDim = !mMeta->prefix_segments.empty() &&
                                         mMeta->prefix_segments[0].rope_dim > 0 ?
                                         mMeta->prefix_segments[0].rope_dim : headDim;
                    float restoreRopeTheta = !mMeta->prefix_segments.empty() &&
                                             mMeta->prefix_segments[0].rope_theta > 0.0f ?
                                             mMeta->prefix_segments[0].rope_theta : 10000.0f;
                    if (!mKVCacheCLManager->ensurePagedStateForExistingHistory(
                            basePast, alignToPage(basePast, RuntimeKVBlockTablePlan::kDefaultPageSize),
                            RuntimeKVBlockTablePlan::kDefaultPageSize,
                            restoreRopeDim, restoreRopeTheta)) {
                        return 0;
                    }
                }
            }
            int ropeDim = 0;
            float ropeTheta = 10000.0f;
            if (!mMeta->prefix_segments.empty()) {
                ropeDim = mMeta->prefix_segments[0].rope_dim;
                ropeTheta = mMeta->prefix_segments[0].rope_theta > 0.0f ?
                    mMeta->prefix_segments[0].rope_theta : ropeTheta;
            }
            constexpr int pageSize = RuntimeKVBlockTablePlan::kDefaultPageSize;
            int physicalBase = basePast > 0 ? ROUND_UP(mKVCacheCLManager->pagedPhysicalLength(), pageSize) : 0;
            int logicalTotal = basePast + segmentTotalTokens + seqlen;
            int physicalTotal = physicalBase + mPagedKVPlan.physicalLength();
            if (mKVCacheCLManager->ensurePagedCapacity(physicalTotal, logicalTotal)) {
                mKVCacheCLManager->setPagedState(basePast + segmentTotalTokens, physicalTotal,
                                                 mPagedKVPlan.pageSize(), ropeDim, ropeTheta);
            }
        }
        int basePast = mMeta->previous != mMeta->remove ?
            static_cast<int>(mMeta->previous - mMeta->remove) : 0;
        return basePast + segmentTotalTokens;
    }

    virtual ErrorCode onPrepareKVCacheBeforeAppend(const std::vector<Tensor*>& inputs, bool& prepared,
                                                   int& appendKvSeqLen) override {
        prepared = false;
        appendKvSeqLen = inputs[1]->shape()[1];
        if (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingReadSegments ||
            mMeta->prefix_segments.empty()) {
            return NO_ERROR;
        }

        if (mMeta->prefix_device_prefetch && mMeta->prefix_request_id != 0) {
            auto submitErr = submitDevicePrefetchForRequestIfNeeded(inputs);
            if (submitErr != NO_ERROR) {
                return submitErr;
            }
            bool prefetchHit = false;
            auto waitErr = consumeDevicePrefetch(prefetchHit, appendKvSeqLen);
            if (waitErr != NO_ERROR) {
                return waitErr;
            }
            if (prefetchHit) {
                prepared = true;
                return NO_ERROR;
            }
            MNN_PRINT("[OpenCLPrefixAttention] layer=%d request_id=%llu device_prefetch=1 device_prefetch_hit=0 fallback_sync=1\n",
                      mLayerIndex, static_cast<unsigned long long>(mMeta->prefix_request_id));
        }

        PrefixPrepareSummary summary;
        auto err = preparePrefixKVOnQueue(inputs, mOpenCLBackend->getOpenCLRuntime()->commandQueue(),
                                          true, false, appendKvSeqLen, summary, nullptr);
        if (err != NO_ERROR) {
            return err;
        }
        prepared = summary.layerIndex >= 0;
        return NO_ERROR;
    }

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (dst == nullptr) {
            return true;
        }
        auto* exe = new OpenCLPrefixAttentionBufExecution(mKVCacheCLManager, op, bn, mLayerIndex);
        exe->mMeta = mMeta;
        exe->adoptSharedKVCacheManager();
        *dst = exe;
        return true;
    }

private:
    struct PrefixPrepareSummary {
        int layerIndex = -1;
        int basePast = 0;
        int segmentTotalTokens = 0;
        int promptTokens = 0;
        int pastAfterPrefix = 0;
    };

    struct SharedPrefetchRecord {
        cl::Event event;
        bool hasEvent = false;
        ErrorCode status = NO_ERROR;
        int appendKvSeqLen = 0;
        PrefixPrepareSummary summary;
    };

    bool isDirectSegmentsPrefix() const {
        return mMeta != nullptr && mMeta->file_flag == KVMeta::PendingReadSegments &&
               !mMeta->prefix_segments.empty();
    }

    void adoptSharedKVCacheManager() {
        if (mLayerIndex < 0 || mKVCacheCLManager == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(openCLPrefixLayerCacheRegistryMutex());
        auto& slot = openCLPrefixLayerCacheRegistry()[mLayerIndex];
        if (slot != nullptr) {
            mKVCacheCLManager = slot;
            return;
        }
        slot = mKVCacheCLManager;
    }

    ErrorCode preparePrefixKVOnQueue(const std::vector<Tensor*>& inputs, cl::CommandQueue& queue,
                                     bool allowMetaLayerCursor, bool devicePrefetch,
                                     int& appendKvSeqLen, PrefixPrepareSummary& summary,
                                     cl::Event* readyEvent) {

        auto query = inputs[0];
        auto key = inputs[1];
        int batch = query->shape()[0];
        int promptLen = key->shape()[1];
        int kvHeads = key->shape()[2];
        int headDim = key->shape()[3];
        if (batch != 1) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention direct_segments only supports batch=1 now\n");
            return NOT_SUPPORT;
        }
        if ((mMeta->remove > 0 || mMeta->n_reserve > 0) && mMeta->previous != mMeta->remove) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention direct_segments append does not support remove/reserve compaction yet\n");
            return NOT_SUPPORT;
        }
        if (mMeta->prefix_cache_dir.empty()) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention missing prefix_cache_dir in KVMeta\n");
            return INVALID_VALUE;
        }

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
        if (!buildPrefixRuntimeKVBlockTablePlan(mPagedKVPlan, mMeta, promptLen)) {
            return INVALID_VALUE;
        }

        int basePast = 0;
        if (mMeta->previous != mMeta->remove) {
            basePast = mKVCacheCLManager->pastKvLength();
            if (basePast <= 0) {
                basePast = static_cast<int>(mMeta->previous - mMeta->remove);
            }
            if (!mKVCacheCLManager->pagedActive()) {
                if (!mKVCacheCLManager->ensurePagedStateForExistingHistory(
                        basePast, alignToPage(basePast, RuntimeKVBlockTablePlan::kDefaultPageSize),
                        RuntimeKVBlockTablePlan::kDefaultPageSize, headDim, 10000.0f)) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention direct_segments append requires existing paged runtime KV cache; "
                              "reset before first direct_segments request. layer=%d manager=%p previous=%zu remove=%zu "
                              "base_past=%d manager_past=%d\n",
                              mLayerIndex, mKVCacheCLManager.get(), mMeta->previous, mMeta->remove,
                              basePast, mKVCacheCLManager->pastKvLength());
                    return NOT_SUPPORT;
                }
            }
        }

        int runtimeBytes = mKVCacheCLManager->byte();
        if (runtimeBytes != 2 && runtimeBytes != 4) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention only supports fp16/fp32 runtime KV cache\n");
            return NOT_SUPPORT;
        }

        int layerIndex = mLayerIndex;
        if (layerIndex < 0) {
            if (!allowMetaLayerCursor) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention device_prefetch requires explicit AttentionParam.layer_index\n");
                return NOT_SUPPORT;
            }
            layerIndex = mMeta->layer_index;
            if (mMeta->layer_nums > 0) {
                mMeta->layer_index = (mMeta->layer_index + 1) % mMeta->layer_nums;
            } else {
                mMeta->layer_index++;
            }
        }

        auto docPool = openCLDocumentPoolForCache(mKVCacheCLManager);
        if (docPool == nullptr) {
            return OUT_OF_MEMORY;
        }

        constexpr int pageSize = RuntimeKVBlockTablePlan::kDefaultPageSize;
        int physicalCursor = basePast > 0 ? std::max(mKVCacheCLManager->pagedPhysicalLength(), basePast) : 0;
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            physicalCursor = std::max(physicalCursor, docPool->nextPhysicalSlot);
        }
        std::vector<OpenCLPrefixSegmentPlacement> placements;
        placements.reserve(mMeta->prefix_segments.size());
        int logicalCursor = basePast;
        int segmentIndex = 0;
        int ropeDimForRequest = 0;
        float ropeThetaForRequest = 10000.0f;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                segmentIndex++;
                continue;
            }
            if (segment.key_rope_state != KVMeta::KeyRopeCanonicalNoRope ||
                segment.rope_pairing != KVMeta::RopePairingHalf) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention requires canonical_no_rope half-split prefix cache: %s\n",
                          segment.cache_name.c_str());
                return NOT_SUPPORT;
            }
            if (segment.backend != "opencl" || segment.layout != kOpenCLPrefixLayout) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention requires backend-native OpenCL cache for %s, got backend=%s layout=%s\n",
                          segment.cache_name.c_str(), segment.backend.c_str(), segment.layout.c_str());
                return NOT_SUPPORT;
            }

            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : headDim;
            ropeDim = std::min(ropeDim, headDim);
            if (ropeDim <= 0 || (ropeDim % 2) != 0 || segment.rope_theta <= 0.0f) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention invalid RoPE metadata for cache %s\n",
                          segment.cache_name.c_str());
                return INVALID_VALUE;
            }
            if (ropeDimForRequest <= 0) {
                ropeDimForRequest = ropeDim;
                ropeThetaForRequest = segment.rope_theta;
            } else if (ropeDimForRequest != ropeDim ||
                       std::fabs(ropeThetaForRequest - segment.rope_theta) > 1e-5f) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention requires consistent RoPE metadata across direct_segments\n");
                return NOT_SUPPORT;
            }

            auto basePath = prefixCacheLayerBase(mMeta->prefix_cache_dir, "opencl", segment.cache_name, layerIndex);
            auto keyPath = basePath + ".k";
            auto valuePath = basePath + ".v";
            size_t keyFileSize = 0;
            size_t valueFileSize = 0;
            std::string keyIdentity;
            std::string valueIdentity;
            if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keyFileSize) ||
                !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueFileSize)) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention missing native KV file for %s layer %d\n",
                          segment.cache_name.c_str(), layerIndex);
                return FILE_OPEN_FAILED;
            }
            size_t expectedBytes = openCLNativeFileBytes(segment.token_count, kvHeads, headDim, runtimeBytes);
            if (keyFileSize != expectedBytes || valueFileSize != expectedBytes) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention invalid native KV size for %s layer %d: key=%zu value=%zu expected=%zu\n",
                          segment.cache_name.c_str(), layerIndex, keyFileSize, valueFileSize, expectedBytes);
                return INVALID_VALUE;
            }
            auto cacheKey = openCLDocumentPageCacheKey(keyPath, valuePath, segment, layerIndex,
                                                       runtimeBytes, kvHeads, headDim,
                                                       ropeDim, segment.rope_theta);
            if (cacheKey.empty()) {
                return FILE_OPEN_FAILED;
            }

            OpenCLPrefixSegmentPlacement placement;
            placement.logicalStart = logicalCursor;
            placement.tokenCount = segment.token_count;
            placement.physicalLength = alignToPage(segment.token_count, pageSize);
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
                    int physicalStart = alignToPage(std::max(physicalCursor, docPool->nextPhysicalSlot), pageSize);
                    placement.physicalStart = physicalStart;
                    docPool->nextPhysicalSlot = physicalStart + placement.physicalLength;
                    physicalCursor = docPool->nextPhysicalSlot;
                }
            }
            placement.deviceCacheLookupMs = nowMs() - lookupStart;
            physicalCursor = std::max(physicalCursor, placement.physicalStart + placement.physicalLength);
            placements.emplace_back(placement);
            logicalCursor += segment.token_count;
            segmentIndex++;
        }

        if (logicalCursor != basePast + segmentTotalTokens) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention planned prefix length %d but expected %d\n",
                      logicalCursor - basePast, segmentTotalTokens);
            return INVALID_VALUE;
        }

        const int promptLogicalStart = basePast + segmentTotalTokens;
        const int promptPhysicalStart = alignToPage(physicalCursor, pageSize);
        physicalCursor = promptPhysicalStart + promptLen;
        const int requiredTotal = basePast + segmentTotalTokens + promptLen;
        const int requiredPhysicalTotal = alignToPage(physicalCursor, pageSize);
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            docPool->nextPhysicalSlot = std::max(docPool->nextPhysicalSlot, requiredPhysicalTotal);
        }
        if (!mKVCacheCLManager->ensurePagedCapacity(requiredPhysicalTotal, requiredTotal)) {
            return OUT_OF_MEMORY;
        }

        int maxLen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
        mPendingHostUploads.clear();

        std::vector<int> tokenTable(requiredTotal - basePast, 0);
        std::vector<int> ropeTable(segmentTotalTokens + promptLen, 0);
        for (const auto& placement : placements) {
            for (int i = 0; i < placement.tokenCount; ++i) {
                int offset = placement.logicalStart + i - basePast;
                tokenTable[offset] = placement.physicalStart + i;
                ropeTable[offset] = 1;
            }
        }
        for (int i = 0; i < promptLen; ++i) {
            int offset = promptLogicalStart + i - basePast;
            tokenTable[offset] = promptPhysicalStart + i;
            ropeTable[offset] = 0;
        }
        cl_int tableStatus = CL_SUCCESS;
        cl::Event tableEvent;
        tableStatus |= queue.enqueueWriteBuffer(*mKVCacheCLManager->mutableTokenTable(), CL_TRUE,
                                                static_cast<size_t>(basePast) * sizeof(int),
                                                tokenTable.size() * sizeof(int), tokenTable.data());
        tableStatus |= queue.enqueueWriteBuffer(*mKVCacheCLManager->mutableRopeTable(), CL_TRUE,
                                                static_cast<size_t>(basePast) * sizeof(int),
                                                ropeTable.size() * sizeof(int), ropeTable.data(),
                                                nullptr, &tableEvent);
        if (tableStatus != CL_SUCCESS) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention failed to upload paged token table: %d\n", tableStatus);
            return INVALID_VALUE;
        }

        cl::Event lastEvent = tableEvent;
        bool hasLastEvent = tableEvent() != nullptr;
        logicalCursor = basePast;
        segmentIndex = 0;
        size_t placementIndex = 0;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                segmentIndex++;
                continue;
            }
            const auto& placement = placements[placementIndex++];
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : headDim;
            ropeDim = std::min(ropeDim, headDim);

            bool hostCacheHit = false;
            double prefetchWaitMs = 0.0;
            double diskReadMs = 0.0;
            double uploadEnqueueMs = 0.0;
            double attentionWaitMs = 0.0;
            size_t uploadBytes = 0;
            const size_t elemBytes = static_cast<size_t>(runtimeBytes);

            if (placement.deviceCacheHit) {
                double waitStart = nowMs();
                cl_int waitStatus = enqueueWaitForEvent(queue, placement.readyEvent);
                attentionWaitMs = nowMs() - waitStart;
                if (waitStatus != CL_SUCCESS) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention failed to wait document page event: %d\n", waitStatus);
                    return INVALID_VALUE;
                }
            } else {
                auto keyRead = readPrefixKVHostCacheFile(placement.keyPath);
                auto valueRead = readPrefixKVHostCacheFile(placement.valuePath);
                if (!keyRead.ok || !valueRead.ok || keyRead.bytes == nullptr || valueRead.bytes == nullptr) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention failed to read prefix KV cache: %s / %s, key_error=%s value_error=%s\n",
                              placement.keyPath.c_str(), placement.valuePath.c_str(),
                              keyRead.error.c_str(), valueRead.error.c_str());
                    return FILE_OPEN_FAILED;
                }
                auto& keyBytes = *keyRead.bytes;
                auto& valueBytes = *valueRead.bytes;

                size_t expectedBytes = openCLNativeFileBytes(segment.token_count, kvHeads, headDim, runtimeBytes);
                if (keyBytes.size() != expectedBytes || valueBytes.size() != expectedBytes) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention invalid native KV size for %s layer %d: key=%zu value=%zu expected=%zu\n",
                              segment.cache_name.c_str(), layerIndex, keyBytes.size(), valueBytes.size(), expectedBytes);
                    return INVALID_VALUE;
                }

                double uploadStart = nowMs();
                cl_int uploadStatus = CL_SUCCESS;
                int alignedTokens = ROUND_UP(segment.token_count, 4);
                for (const auto& ref : mPagedKVPlan.refs()) {
                    if (ref.source != RuntimeKVBlockRef::Document || ref.segmentIndex != segmentIndex ||
                        ref.tokenCount <= 0) {
                        continue;
                    }
                    for (int head = 0; head < kvHeads; ++head) {
                        for (int dim = 0; dim < headDim; ++dim) {
                            size_t srcOffset = ((static_cast<size_t>(head) * headDim + dim) *
                                                static_cast<size_t>(alignedTokens) +
                                                static_cast<size_t>(ref.sourceTokenStart)) * elemBytes;
                            size_t dstOffset = ((static_cast<size_t>(head) * headDim + dim) *
                                                static_cast<size_t>(maxLen) +
                                                static_cast<size_t>(placement.physicalStart + ref.sourceTokenStart)) *
                                               elemBytes;
                            cl::Event writeEvent;
                            uploadStatus |= queue.enqueueWriteBuffer(*mKVCacheCLManager->mutableKey(), CL_FALSE,
                                                                     dstOffset,
                                                                     static_cast<size_t>(ref.tokenCount) * elemBytes,
                                                                     keyBytes.data() + srcOffset, nullptr, &writeEvent);
                            lastEvent = writeEvent;
                            hasLastEvent = writeEvent() != nullptr;
                        }
                        size_t srcValueOffset = (static_cast<size_t>(head) * static_cast<size_t>(alignedTokens) +
                                                 static_cast<size_t>(ref.sourceTokenStart)) *
                                                static_cast<size_t>(headDim) * elemBytes;
                        size_t dstValueOffset = (static_cast<size_t>(head) * static_cast<size_t>(maxLen) +
                                                 static_cast<size_t>(placement.physicalStart + ref.sourceTokenStart)) *
                                                static_cast<size_t>(headDim) * elemBytes;
                        cl::Event writeEvent;
                        uploadStatus |= queue.enqueueWriteBuffer(*mKVCacheCLManager->mutableValue(), CL_FALSE,
                                                                 dstValueOffset,
                                                                 static_cast<size_t>(ref.tokenCount) *
                                                                 static_cast<size_t>(headDim) * elemBytes,
                                                                 valueBytes.data() + srcValueOffset, nullptr, &writeEvent);
                        lastEvent = writeEvent;
                        hasLastEvent = writeEvent() != nullptr;
                    }
                }
                if (uploadStatus != CL_SUCCESS) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention failed to upload prefix KV to paged runtime cache: "
                              "layer=%d cache=%s err=%d key_bytes=%zu value_bytes=%zu\n",
                              layerIndex, segment.cache_name.c_str(), uploadStatus,
                              keyBytes.size(), valueBytes.size());
                    return INVALID_VALUE;
                }
                uploadEnqueueMs = nowMs() - uploadStart;
                uploadBytes = keyRead.file_size + valueRead.file_size;
                hostCacheHit = keyRead.host_cache_hit && valueRead.host_cache_hit;
                prefetchWaitMs = keyRead.prefetch_wait_ms + valueRead.prefetch_wait_ms;
                diskReadMs = keyRead.disk_read_ms + valueRead.disk_read_ms;
                mPendingHostUploads.emplace_back(keyRead.bytes);
                mPendingHostUploads.emplace_back(valueRead.bytes);

                if (hasLastEvent) {
                    OpenCLDevicePrefixDocumentRecord record;
                    record.physicalStart = placement.physicalStart;
                    record.physicalLength = placement.physicalLength;
                    record.tokenCount = segment.token_count;
                    record.ropeDim = ropeDim;
                    record.ropeTheta = segment.rope_theta;
                    record.readyEvent = lastEvent;
                    std::lock_guard<std::mutex> lock(docPool->mutex);
                    docPool->records[placement.deviceCacheKey] = record;
                }
            }

            const double waitMs = (!devicePrefetch && mMeta->prefix_device_prefetch) ?
                prefetchWaitMs : attentionWaitMs;
            MNN_PRINT("[OpenCLPrefixAttention] layer=%d cache=%s tokens=%d logical_start=%d physical_start=%d host_cache_hit=%d device_cache_hit=%d device_cache_lookup_ms=%.3f device_prefetch=%d device_prefetch_submit=%d device_prefetch_hit=0 prefetch_wait_ms=%.3f disk_read_ms=%.3f host_to_device_ms=%.3f materialize_ms=0.000 attention_wait_ms=%.3f upload_enqueue_bytes=%zu staging_path=0 direct_runtime_upload=0 page_pool=1 on_read_rope=1 page_size=%d\n",
                      layerIndex, placement.cacheName.c_str(), segment.token_count,
                      placement.logicalStart, placement.physicalStart,
                      hostCacheHit ? 1 : 0,
                      placement.deviceCacheHit ? 1 : 0,
                      placement.deviceCacheLookupMs,
                      mMeta->prefix_device_prefetch ? 1 : 0,
                      devicePrefetch ? 1 : 0,
                      prefetchWaitMs,
                      diskReadMs,
                      uploadEnqueueMs,
                      waitMs,
                      uploadBytes,
                      pageSize);
            logicalCursor += segment.token_count;
            segmentIndex++;
        }

        queue.flush();

        if (logicalCursor != basePast + segmentTotalTokens) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention copied prefix length %d but expected %d\n",
                      logicalCursor - basePast, segmentTotalTokens);
            return INVALID_VALUE;
        }

        mKVCacheCLManager->setPagedState(basePast + segmentTotalTokens, requiredPhysicalTotal, pageSize,
                                         ropeDimForRequest > 0 ? ropeDimForRequest : headDim,
                                         ropeThetaForRequest);
        mKVCacheCLManager->setPastKvLength(basePast + segmentTotalTokens);
        if (layerIndex == 0) {
            MNN_PRINT("[OpenCLPrefixAttention] block_table base_past=%d prefix_tokens=%d prompt_tokens=%d required_total=%d physical_total=%d page_size=%d logical_blocks=%d physical_blocks=%d prefix_blocks=%d prompt_blocks=%d prompt_physical_start=%d device_prefetch=%d\n",
                      basePast, segmentTotalTokens, promptLen, requiredTotal, requiredPhysicalTotal,
                      pageSize,
                      UP_DIV(requiredTotal, pageSize),
                      UP_DIV(requiredPhysicalTotal, pageSize),
                      UP_DIV(segmentTotalTokens, pageSize),
                      UP_DIV(promptLen, pageSize),
                      promptPhysicalStart,
                      devicePrefetch ? 1 : 0);
        }
        if (readyEvent != nullptr && hasLastEvent) {
            *readyEvent = lastEvent;
        }
        appendKvSeqLen = promptLen;
        summary.layerIndex = layerIndex;
        summary.basePast = basePast;
        summary.segmentTotalTokens = segmentTotalTokens;
        summary.promptTokens = promptLen;
        summary.pastAfterPrefix = basePast + segmentTotalTokens;
        return NO_ERROR;
    }

    ErrorCode enqueueDevicePrefetch(const std::vector<Tensor*>& inputs, cl::CommandQueue& queue,
                                    uint64_t requestId) {
        if (mDevicePrefetchRequestId == requestId) {
            return mDevicePrefetchStatus;
        }
        mDevicePrefetchRequestId = requestId;
        mDevicePrefetchStatus = NO_ERROR;
        mDevicePrefetchReadyEvent = cl::Event();
        mDevicePrefetchHasEvent = false;
        mDevicePrefetchAppendKvSeqLen = 0;
        mDevicePrefetchSummary = PrefixPrepareSummary();

        double start = nowMs();
        cl::Event event;
        auto err = preparePrefixKVOnQueue(inputs, queue, false, true,
                                          mDevicePrefetchAppendKvSeqLen,
                                          mDevicePrefetchSummary, &event);
        if (err != NO_ERROR) {
            mDevicePrefetchStatus = err;
            return err;
        }
        if (mDevicePrefetchSummary.layerIndex < 0) {
            return NO_ERROR;
        }
        mDevicePrefetchReadyEvent = event;
        mDevicePrefetchHasEvent = event() != nullptr;
        auto record = std::make_shared<SharedPrefetchRecord>();
        record->event = mDevicePrefetchReadyEvent;
        record->hasEvent = mDevicePrefetchHasEvent;
        record->status = mDevicePrefetchStatus;
        record->appendKvSeqLen = mDevicePrefetchAppendKvSeqLen;
        record->summary = mDevicePrefetchSummary;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            sharedPrefetchRecords()[mMeta][requestId][mDevicePrefetchSummary.layerIndex] = record;
        }
        MNN_PRINT("[OpenCLPrefixAttentionPrefetch] layer=%d request_id=%llu submit_ms=%.3f prefix_tokens=%d prompt_tokens=%d manager=%p paged_active=%d manager_past=%d\n",
                  mDevicePrefetchSummary.layerIndex,
                  static_cast<unsigned long long>(requestId),
                  nowMs() - start,
                  mDevicePrefetchSummary.segmentTotalTokens,
                  mDevicePrefetchSummary.promptTokens,
                  mKVCacheCLManager.get(),
                  (mKVCacheCLManager != nullptr && mKVCacheCLManager->pagedActive()) ? 1 : 0,
                  mKVCacheCLManager != nullptr ? mKVCacheCLManager->pastKvLength() : 0);
        return NO_ERROR;
    }

    ErrorCode consumeDevicePrefetch(bool& hit, int& appendKvSeqLen) {
        hit = false;
        if (mMeta == nullptr || mDevicePrefetchRequestId != mMeta->prefix_request_id ||
            mDevicePrefetchRequestId == 0) {
            if (!loadSharedDevicePrefetchRecord()) {
                return NO_ERROR;
            }
        }
        if (mDevicePrefetchStatus != NO_ERROR) {
            return mDevicePrefetchStatus;
        }
        if (!mDevicePrefetchHasEvent || mDevicePrefetchReadyEvent() == nullptr) {
            return NO_ERROR;
        }
        double waitStart = nowMs();
        cl_int res = enqueueWaitForEvent(mOpenCLBackend->getOpenCLRuntime()->commandQueue(),
                                         mDevicePrefetchReadyEvent);
        double waitMs = nowMs() - waitStart;
        if (res != CL_SUCCESS) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention failed to enqueue device_prefetch event wait: %d\n", res);
            return INVALID_VALUE;
        }
        mKVCacheCLManager->setPastKvLength(mDevicePrefetchSummary.pastAfterPrefix);
        appendKvSeqLen = mDevicePrefetchAppendKvSeqLen;
        hit = true;
        MNN_PRINT("[OpenCLPrefixAttention] layer=%d request_id=%llu device_prefetch=1 device_prefetch_hit=1 attention_wait_ms=%.3f queue_event_wait=1 base_past=%d prefix_tokens=%d prompt_tokens=%d\n",
                  mDevicePrefetchSummary.layerIndex,
                  static_cast<unsigned long long>(mDevicePrefetchRequestId),
                  waitMs,
                  mDevicePrefetchSummary.basePast,
                  mDevicePrefetchSummary.segmentTotalTokens,
                  mDevicePrefetchSummary.promptTokens);
        return NO_ERROR;
    }

    bool loadSharedDevicePrefetchRecord() {
        if (mMeta == nullptr || mMeta->prefix_request_id == 0 || mLayerIndex < 0) {
            return false;
        }
        std::shared_ptr<SharedPrefetchRecord> record;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            auto metaIter = sharedPrefetchRecords().find(mMeta);
            if (metaIter == sharedPrefetchRecords().end()) {
                return false;
            }
            auto requestIter = metaIter->second.find(mMeta->prefix_request_id);
            if (requestIter == metaIter->second.end()) {
                return false;
            }
            auto layerIter = requestIter->second.find(mLayerIndex);
            if (layerIter == requestIter->second.end()) {
                return false;
            }
            record = layerIter->second;
        }
        if (record == nullptr) {
            return false;
        }
        mDevicePrefetchRequestId = mMeta->prefix_request_id;
        mDevicePrefetchStatus = record->status;
        mDevicePrefetchReadyEvent = record->event;
        mDevicePrefetchHasEvent = record->hasEvent;
        mDevicePrefetchAppendKvSeqLen = record->appendKvSeqLen;
        mDevicePrefetchSummary = record->summary;
        return true;
    }

    ErrorCode submitDevicePrefetchForRequestIfNeeded(const std::vector<Tensor*>& inputs) {
        if (mMeta == nullptr || !mMeta->prefix_device_prefetch || mMeta->prefix_request_id == 0) {
            return NO_ERROR;
        }
        if (mLayerIndex != 0) {
            return NO_ERROR;
        }
        auto queue = devicePrefetchQueue(mOpenCLBackend->getOpenCLRuntime());
        if (queue == nullptr) {
            MNN_PRINT("[OpenCLPrefixAttentionPrefetch] request_id=%llu unavailable_copy_queue=1 fallback_sync=1\n",
                      static_cast<unsigned long long>(mMeta->prefix_request_id));
            return NO_ERROR;
        }

        std::vector<OpenCLPrefixAttentionBufExecution*> targets;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            auto& submitted = submittedRequests();
            auto submittedIter = submitted.find(mMeta);
            if (submittedIter != submitted.end() && submittedIter->second == mMeta->prefix_request_id) {
                return NO_ERROR;
            }

            auto registryIter = registry().find(mMeta);
            if (registryIter != registry().end()) {
                std::unordered_map<int, OpenCLPrefixAttentionBufExecution*> byLayer;
                const bool needExistingPagedState = (mMeta->previous != mMeta->remove);
                auto preferCandidate = [needExistingPagedState](OpenCLPrefixAttentionBufExecution* current,
                                                                 OpenCLPrefixAttentionBufExecution* candidate) {
                    if (current == nullptr) {
                        return candidate;
                    }
                    if (candidate == nullptr) {
                        return current;
                    }
                    if (needExistingPagedState) {
                        const bool currentPaged = current->mKVCacheCLManager != nullptr &&
                                                  current->mKVCacheCLManager->pagedActive();
                        const bool candidatePaged = candidate->mKVCacheCLManager != nullptr &&
                                                    candidate->mKVCacheCLManager->pagedActive();
                        if (currentPaged != candidatePaged) {
                            return candidatePaged ? candidate : current;
                        }
                        const int currentPast = current->mKVCacheCLManager != nullptr ?
                            current->mKVCacheCLManager->pastKvLength() : 0;
                        const int candidatePast = candidate->mKVCacheCLManager != nullptr ?
                            candidate->mKVCacheCLManager->pastKvLength() : 0;
                        if (currentPast != candidatePast) {
                            return candidatePast > currentPast ? candidate : current;
                        }
                    }
                    return candidate;
                };
                for (auto* exe : registryIter->second) {
                    if (exe == nullptr || exe->mMeta != mMeta || exe->mLayerIndex < 0 ||
                        exe->mKVCacheCLManager == nullptr) {
                        continue;
                    }
                    auto iter = byLayer.find(exe->mLayerIndex);
                    byLayer[exe->mLayerIndex] = preferCandidate(
                        iter == byLayer.end() ? nullptr : iter->second, exe);
                }
                targets.reserve(byLayer.size());
                for (auto& iter : byLayer) {
                    targets.emplace_back(iter.second);
                }
            }
            if (targets.empty()) {
                return NO_ERROR;
            }
            sharedPrefetchRecords()[mMeta].clear();
            submitted[mMeta] = mMeta->prefix_request_id;
        }

        std::sort(targets.begin(), targets.end(), [](const OpenCLPrefixAttentionBufExecution* left,
                                                     const OpenCLPrefixAttentionBufExecution* right) {
            return left->mLayerIndex < right->mLayerIndex;
        });

        MNN_PRINT("[OpenCLPrefixAttentionPrefetch] request_id=%llu submit_layers=%zu queue=copy\n",
                  static_cast<unsigned long long>(mMeta->prefix_request_id),
                  targets.size());
        for (auto* target : targets) {
            auto err = target->enqueueDevicePrefetch(inputs, *queue, mMeta->prefix_request_id);
            if (err != NO_ERROR) {
                return err;
            }
        }
        return NO_ERROR;
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
                sharedPrefetchRecords().erase(mMeta);
            }
        }
        mRegisteredForDevicePrefetch = false;
    }

    static std::shared_ptr<cl::CommandQueue> devicePrefetchQueue(OpenCLRuntime* runtime) {
        if (runtime == nullptr) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(queueMutex());
        auto& queues = copyQueues();
        auto iter = queues.find(runtime);
        if (iter != queues.end()) {
            return iter->second;
        }
        cl_int res = CL_SUCCESS;
        std::vector<cl::Device> devices = runtime->context().getInfo<CL_CONTEXT_DEVICES>(&res);
        if (res != CL_SUCCESS || devices.empty()) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention failed to get context device for prefetch queue: %d\n", res);
            return nullptr;
        }
        auto queue = std::make_shared<cl::CommandQueue>(runtime->context(), devices[0], 0, &res);
        if (res != CL_SUCCESS) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention failed to create device prefetch queue: %d\n", res);
            return nullptr;
        }
        queues[runtime] = queue;
        return queue;
    }

    static std::mutex& registryMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_map<KVMeta*, std::vector<OpenCLPrefixAttentionBufExecution*>>& registry() {
        static std::unordered_map<KVMeta*, std::vector<OpenCLPrefixAttentionBufExecution*>> value;
        return value;
    }

    static std::unordered_map<KVMeta*, uint64_t>& submittedRequests() {
        static std::unordered_map<KVMeta*, uint64_t> value;
        return value;
    }

    static std::unordered_map<KVMeta*, std::unordered_map<uint64_t, std::unordered_map<int, std::shared_ptr<SharedPrefetchRecord>>>>& sharedPrefetchRecords() {
        static std::unordered_map<KVMeta*, std::unordered_map<uint64_t, std::unordered_map<int, std::shared_ptr<SharedPrefetchRecord>>>> value;
        return value;
    }

    static std::mutex& queueMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_map<OpenCLRuntime*, std::shared_ptr<cl::CommandQueue>>& copyQueues() {
        static std::unordered_map<OpenCLRuntime*, std::shared_ptr<cl::CommandQueue>> value;
        return value;
    }

    int mLayerIndex = -1;
    RuntimeKVBlockTablePlan mPagedKVPlan;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> mPendingHostUploads;
    bool mRegisteredForDevicePrefetch = false;
    uint64_t mDevicePrefetchRequestId = 0;
    cl::Event mDevicePrefetchReadyEvent;
    bool mDevicePrefetchHasEvent = false;
    ErrorCode mDevicePrefetchStatus = NO_ERROR;
    int mDevicePrefetchAppendKvSeqLen = 0;
    PrefixPrepareSummary mDevicePrefetchSummary;
};

class OpenCLPrefixAttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        bool kvCache = true;
        int layerIndex = -1;
        auto param = op->main_as_AttentionParam();
        if (param != nullptr) {
            kvCache = param->kv_cache();
            layerIndex = param->layer_index();
        }
        OPENCL_CREATOR_CHECK(new OpenCLPrefixAttentionBufExecution(op, backend, kvCache, layerIndex));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(OpenCLPrefixAttentionBufCreator, OpType_PrefixAttention, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
