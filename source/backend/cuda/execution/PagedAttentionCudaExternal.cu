#include "PagedAttentionCudaUtils.hpp"
#include "core/Macro.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
namespace {

template <typename T>
__device__ inline float externalToFloat(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float externalToFloat<half>(half v) {
    return __half2float(v);
}

template <typename T>
__device__ inline T externalFromFloat(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline half externalFromFloat<half>(float v) {
    return __float2half(v);
}

__device__ inline float externalRopeInvFreq(int pairIndex, int ropeDim, float theta, int ropeType, float factor,
                                            float lowFreqFactor, float highFreqFactor, int oldContext) {
    float invFreq = powf(theta, -static_cast<float>(2 * pairIndex) / static_cast<float>(ropeDim));
    if (ropeType != 1 || oldContext <= 0 || factor == 1.0f || lowFreqFactor == highFreqFactor) {
        return invFreq;
    }
    constexpr float kTwoPi = 6.28318530717958647692f;
    float wavelen = kTwoPi / invFreq;
    float lowFreqWavelen = static_cast<float>(oldContext) / lowFreqFactor;
    float highFreqWavelen = static_cast<float>(oldContext) / highFreqFactor;
    float scaled = wavelen > lowFreqWavelen ? invFreq / factor : invFreq;
    if (wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen) {
        float smooth = (static_cast<float>(oldContext) / wavelen - lowFreqFactor) /
                       (highFreqFactor - lowFreqFactor);
        scaled = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
    }
    return scaled;
}

template <typename T>
__global__ void hydrateExternalPagedKeyKernel(const T* keyIn, T* keyCache, int batch, int tokenCount, int kvHeads,
                                              int headDim, int maxSlots, int logicalStart, int ropeDim,
                                              float ropeTheta, int ropeType, float factor, float lowFreqFactor,
                                              float highFreqFactor, int oldContext, float attentionScale, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int tmp = idx / headDim;
    int h = tmp % kvHeads;
    tmp /= kvHeads;
    int b = tmp % batch;
    int local = tmp / batch;
    if (local >= tokenCount) {
        return;
    }
    int logical = logicalStart + local;
    if (logical < 0 || logical >= maxSlots) {
        return;
    }
    int slot = logical;
    int srcBase = ((local * batch + b) * kvHeads + h) * headDim;
    int dst = ((slot * batch + b) * kvHeads + h) * headDim + d;
    if (ropeDim > 0 && d < ropeDim) {
        int halfDim = ropeDim / 2;
        if (d >= halfDim) {
            return;
        }
        int pair = d;
        float angle = static_cast<float>(logical) *
            externalRopeInvFreq(pair, ropeDim, ropeTheta, ropeType, factor, lowFreqFactor,
                                highFreqFactor, oldContext);
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = externalToFloat<T>(keyIn[srcBase + pair]);
        float x1 = externalToFloat<T>(keyIn[srcBase + pair + halfDim]);
        float scale = attentionScale > 0.0f ? attentionScale : 1.0f;
        keyCache[dst] = externalFromFloat<T>((x0 * c - x1 * s) * scale);
        keyCache[dst + halfDim] = externalFromFloat<T>((x1 * c + x0 * s) * scale);
        return;
    }
    keyCache[dst] = keyIn[srcBase + d];
}

int externalLayerRequiredSlotsCUDA(const PagedKVMeta* meta, int kvLen) {
    return maxSlotsWithPicSourceSlotsCUDA(meta, kvLen);
}

struct ExternalLayerReadSegment {
    bool ok = false;
    bool directWritten = false;
    std::string error;
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

std::mutex gExternalLayerReadMutex;
std::unordered_map<std::string, ExternalLayerReadTask> gExternalLayerReadTasks;

struct ExternalLayerMappedTarget {
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> key;
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> value;
    int batch = 0;
    int kvHeads = 0;
    int headDim = 0;
    int bytes = 0;
    int maxSlots = 0;
};

struct ExternalLayerMappedTargetRef {
    std::weak_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> key;
    std::weak_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> value;
    int batch = 0;
    int kvHeads = 0;
    int headDim = 0;
    int bytes = 0;
    int maxSlots = 0;
};

std::mutex gExternalLayerMappedTargetMutex;
std::unordered_map<std::string, ExternalLayerMappedTargetRef> gExternalLayerMappedTargets;

std::string externalLayerMappedTargetKey(const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads,
                                         int headDim, int bytes) {
    std::ostringstream os;
    os << reinterpret_cast<uintptr_t>(meta) << ":" << layerIndex << ":" << batch << ":" << kvHeads << ":"
       << headDim << ":" << bytes;
    return os.str();
}

ExternalLayerMappedTarget lookupExternalLayerMappedTarget(const PagedKVMeta* meta, int layerIndex, int batch,
                                                          int kvHeads, int headDim, int bytes) {
    ExternalLayerMappedTarget target;
    std::lock_guard<std::mutex> lock(gExternalLayerMappedTargetMutex);
    auto iter = gExternalLayerMappedTargets.find(
        externalLayerMappedTargetKey(meta, layerIndex, batch, kvHeads, headDim, bytes));
    if (iter == gExternalLayerMappedTargets.end()) {
        return target;
    }
    target.key = iter->second.key.lock();
    target.value = iter->second.value.lock();
    target.batch = iter->second.batch;
    target.kvHeads = iter->second.kvHeads;
    target.headDim = iter->second.headDim;
    target.bytes = iter->second.bytes;
    target.maxSlots = iter->second.maxSlots;
    if (target.key == nullptr || target.value == nullptr) {
        gExternalLayerMappedTargets.erase(iter);
        target = ExternalLayerMappedTarget();
    }
    return target;
}

bool externalLayerTargetReadyCUDA(const ExternalLayerMappedTarget& target, int batch, int kvHeads,
                                  int headDim, int bytes, int requiredSlots) {
    return target.key != nullptr && target.value != nullptr && target.key->host != nullptr &&
        target.value->host != nullptr && target.batch == batch && target.kvHeads == kvHeads &&
        target.headDim == headDim && target.bytes == bytes && target.maxSlots >= requiredSlots;
}

int lastExternalLayerIndex(const PagedKVMeta* meta) {
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

int externalLayerReadWindow() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PREFETCH_WINDOW");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    return std::max(0, std::atoi(value));
}

std::string externalLayerRequestKey(const PagedKVMeta* meta, int batch, int kvHeads, int headDim, int bytes,
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

std::string externalLayerTaskKey(const std::string& requestKey, int layerIndex) {
    return requestKey + "\n" + std::to_string(layerIndex);
}

bool readExternalLayerSegmentCUDA(const PagedKVExternalSegment& segment, int layerIndex, int batch,
                                  int kvHeads, int headDim, int bytes, int kvLen,
                                  const ExternalLayerMappedTarget* target,
                                  ExternalLayerReadSegment& out) {
    out = ExternalLayerReadSegment();
    auto layer = segment.layer(layerIndex);
    if (layer == nullptr) {
        out.error = "missing external PIC KV layer " + std::to_string(layerIndex);
        return false;
    }
    const int segBatch = segment.batch > 0 ? segment.batch : batch;
    const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
    const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
    const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
    if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
        out.error = "external PIC KV shape mismatch at layer " + std::to_string(layerIndex);
        return false;
    }
    if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
        out.error = "external PIC KV must be canonical_no_rope/half";
        return false;
    }
    if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
        out.error = "external PIC KV range exceeds visible KV length at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset : segment.sourceTokenOffset;
    const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
        ? layer->sourceTokenCount
        : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
    if (sourceTokenOffset + segment.tokenCount > sourceTokenCount) {
        out.error = "external PIC KV source range is invalid at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
    if (segment.logicalStart > maxInt || segment.tokenCount > maxInt || sourceTokenOffset > maxInt ||
        sourceTokenCount > maxInt) {
        out.error = "external PIC KV indices exceed int range at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t keyTokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
    const size_t keySegmentBytes = segment.tokenCount * keyTokenBytes;
    const bool targetOk = target != nullptr && target->key != nullptr && target->value != nullptr &&
        target->key->host != nullptr && target->value->host != nullptr && target->batch == batch &&
        target->kvHeads == kvHeads && target->headDim == headDim && target->bytes == bytes &&
        target->maxSlots >= kvLen && segment.logicalStart + segment.tokenCount <=
            static_cast<size_t>(target->maxSlots);
    if (!targetOk) {
        out.error = "CUDA external PIC KV requires direct mapped PagedCache at layer " +
                    std::to_string(layerIndex);
        return false;
    }
    const size_t keyDstOffset = segment.logicalStart * keyTokenBytes;
    const size_t keyEnd = keyDstOffset + keySegmentBytes;
    if (keyEnd > target->key->bytes) {
        out.error = "mapped external PIC key target is too small at layer " + std::to_string(layerIndex);
        return false;
    }
    if (!readBinaryFileRange(layer->keyPath, sourceTokenOffset * keyTokenBytes,
                             reinterpret_cast<int8_t*>(target->key->host) + keyDstOffset, keySegmentBytes)) {
        out.error = "failed to read external PIC key directly to PagedCache at layer " +
                    std::to_string(layerIndex);
        return false;
    }
    if (!readExternalValueSegmentToPagedCache(layer->valuePath, target->value->host, batch, kvHeads,
                                              target->maxSlots, segment.logicalStart, sourceTokenCount,
                                              sourceTokenOffset, segment.tokenCount, headDim, bytes)) {
        out.error = "failed to read external PIC value directly to PagedCache at layer " +
                    std::to_string(layerIndex);
        return false;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    out.directWritten = true;
    out.ok = true;
    return true;
}

std::shared_ptr<ExternalLayerReadResult> readExternalLayerCUDA(
    const PagedKVMeta* meta, std::string requestKey, std::vector<PagedKVExternalSegment> segments, int layerIndex,
    int batch, int kvHeads, int headDim, int bytes, int kvLen, ExternalLayerMappedTarget target) {
    ScopedNvtxRange nvtx(nvtxLayerRangeName("pic_async_read_from_disk", layerIndex, -1, -1, kvLen),
                         nvtxPagedAttention());
    auto result = std::make_shared<ExternalLayerReadResult>();
    result->requestKey = std::move(requestKey);
    result->layerIndex = layerIndex;
    result->segments.resize(segments.size());
    const bool hasTarget = externalLayerTargetReadyCUDA(
        target, batch, kvHeads, headDim, bytes, externalLayerRequiredSlotsCUDA(meta, kvLen));
    for (size_t i = 0; i < segments.size(); ++i) {
        if (!readExternalLayerSegmentCUDA(segments[i], layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                          hasTarget ? &target : nullptr, result->segments[i])) {
            result->ok = false;
            result->error = result->segments[i].error;
            break;
        }
    }
    return result;
}

void scheduleExternalLayerReadsFrom(const PagedKVMeta* meta, int startLayer, int batch, int kvHeads,
                                    int headDim, int bytes, int kvLen) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return;
    }
    const int lastLayer = lastExternalLayerIndex(meta);
    if (startLayer < 0 || lastLayer < startLayer) {
        return;
    }
    const int requiredSlots = externalLayerRequiredSlotsCUDA(meta, kvLen);
    const int window = externalLayerReadWindow();
    const int endLayer = window > 0 ? std::min(lastLayer, startLayer + window - 1) : lastLayer;
    const std::string requestKey = externalLayerRequestKey(meta, batch, kvHeads, headDim, bytes, kvLen);
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
            const auto key = externalLayerTaskKey(requestKey, layerIndex);
            if (gExternalLayerReadTasks.find(key) == gExternalLayerReadTasks.end()) {
                auto target = lookupExternalLayerMappedTarget(meta, layerIndex, batch, kvHeads, headDim, bytes);
                if (!externalLayerTargetReadyCUDA(target, batch, kvHeads, headDim, bytes, requiredSlots)) {
                    break;
                }
                layersToSchedule.emplace_back(layerIndex, std::move(target));
            }
        }
        auto segments = meta->external_segments;
        for (const auto& item : layersToSchedule) {
            const int layerIndex = item.first;
            auto target = item.second;
            const auto key = externalLayerTaskKey(requestKey, layerIndex);
            auto future = std::async(std::launch::async,
                                     [meta, requestKey, segments, layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                      target]() {
                                         return readExternalLayerCUDA(meta, requestKey, segments, layerIndex, batch,
                                                                      kvHeads, headDim, bytes, kvLen, target);
                                     }).share();
            ExternalLayerReadTask task;
            task.requestKey = requestKey;
            task.layerIndex = layerIndex;
            task.future = std::move(future);
            gExternalLayerReadTasks[key] = std::move(task);
        }
    }
}

std::shared_ptr<ExternalLayerReadResult> takeExternalLayerRead(const PagedKVMeta* meta, int layerIndex,
                                                               int batch, int kvHeads, int headDim, int bytes,
                                                               int kvLen) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return nullptr;
    }
    const auto requestKey = externalLayerRequestKey(meta, batch, kvHeads, headDim, bytes, kvLen);
    const auto taskKey = externalLayerTaskKey(requestKey, layerIndex);
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

int ropeTypeCode(const PagedKVExternalSegment& segment) {
    return segment.ropeType == "llama3" ? 1 : 0;
}

} // namespace

void registerExternalLayerMappedTarget(
    const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim, int bytes, int maxSlots,
    const std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>& key,
    const std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>& value) {
    if (meta == nullptr || layerIndex < 0 || key == nullptr || value == nullptr || key->host == nullptr ||
        value->host == nullptr || key->device == nullptr || value->device == nullptr) {
        return;
    }
    ExternalLayerMappedTargetRef ref;
    ref.key = key;
    ref.value = value;
    ref.batch = batch;
    ref.kvHeads = kvHeads;
    ref.headDim = headDim;
    ref.bytes = bytes;
    ref.maxSlots = maxSlots;
    std::lock_guard<std::mutex> lock(gExternalLayerMappedTargetMutex);
    gExternalLayerMappedTargets[externalLayerMappedTargetKey(meta, layerIndex, batch, kvHeads, headDim, bytes)] =
        std::move(ref);
}

ErrorCode restoreExternalSegmentsCUDA(PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim,
                                      int bytes, int maxSlots, int kvLen, void* keyCacheDevice) {
    if (meta == nullptr || meta->external_segments.empty() || meta->externalLayerLoaded(layerIndex)) {
        return NO_ERROR;
    }
    const bool profile = profilePagedAttention();
    const bool nvtx = nvtxPagedAttention();
    auto directTarget = lookupExternalLayerMappedTarget(meta, layerIndex, batch, kvHeads, headDim, bytes);
    const bool hasDirectTarget = directTarget.key != nullptr && directTarget.value != nullptr;
    const char* restoreOp = "pic_restore_mapped_kv_for_attention_total";
    ScopedNvtxRange restoreNvtx(nvtxLayerRangeName(restoreOp, layerIndex, -1, -1, kvLen), nvtx);
    if (!hasDirectTarget) {
        MNN_ERROR("CUDAPagedAttention external PIC KV requires direct mapped PagedCache at layer %d\n", layerIndex);
        return INVALID_VALUE;
    }
    const uint64_t startUs = profile ? nowUs() : 0;
    size_t totalTokens = 0;
    size_t directTokens = 0;
    int directSegments = 0;
    scheduleExternalLayerReadsFrom(meta, layerIndex + 1, batch, kvHeads, headDim, bytes, kvLen);
    std::shared_ptr<ExternalLayerReadResult> prefetched;
    {
        ScopedNvtxRange waitNvtx(nvtxLayerRangeName("pic_wait_async_read_from_disk", layerIndex, -1, -1, kvLen),
                                 nvtx);
        prefetched = takeExternalLayerRead(meta, layerIndex, batch, kvHeads, headDim, bytes, kvLen);
    }
    if (prefetched != nullptr && !prefetched->ok) {
        MNN_ERROR("CUDAPagedAttention async external PIC KV read failed at layer %d: %s\n", layerIndex,
                  prefetched->error.c_str());
        return INVALID_VALUE;
    }
    if (prefetched != nullptr && prefetched->segments.size() != meta->external_segments.size()) {
        MNN_ERROR("CUDAPagedAttention async external PIC KV read segment count mismatch at layer %d\n", layerIndex);
        return INVALID_VALUE;
    }
    for (size_t segmentIndex = 0; segmentIndex < meta->external_segments.size(); ++segmentIndex) {
        const auto& segment = meta->external_segments[segmentIndex];
        totalTokens += segment.tokenCount;
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            MNN_ERROR("CUDAPagedAttention layer %d missing external PIC KV for cache %s\n", layerIndex,
                      segment.cacheName.c_str());
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : batch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
        if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
            MNN_ERROR("CUDAPagedAttention external KV shape mismatch at layer %d, cache %s\n", layerIndex,
                      segment.cacheName.c_str());
            return INVALID_VALUE;
        }
        if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
            MNN_ERROR("CUDAPagedAttention external KV must be canonical_no_rope/half, got %s/%s\n",
                      segment.keyRopeState.c_str(), segment.ropePairing.c_str());
            return INVALID_VALUE;
        }
        if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
            MNN_ERROR("CUDAPagedAttention external KV range exceeds visible KV length at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        if (segment.tokenCount == 0) {
            continue;
        }
        ExternalLayerReadSegment syncRead;
        const ExternalLayerReadSegment* loadedSegment = nullptr;
        if (prefetched != nullptr) {
            loadedSegment = &prefetched->segments[segmentIndex];
            if (!loadedSegment->ok) {
                MNN_ERROR("CUDAPagedAttention async external PIC KV segment read failed at layer %d: %s\n",
                          layerIndex, loadedSegment->error.c_str());
                return INVALID_VALUE;
            }
        } else {
            if (!readExternalLayerSegmentCUDA(segment, layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                              hasDirectTarget ? &directTarget : nullptr, syncRead)) {
                MNN_ERROR("CUDAPagedAttention failed to read external PIC KV files for layer %d: %s\n",
                          layerIndex, syncRead.error.c_str());
                return INVALID_VALUE;
            }
            loadedSegment = &syncRead;
        }
        if (loadedSegment == nullptr || !loadedSegment->directWritten) {
            MNN_ERROR("CUDAPagedAttention external PIC KV requires direct mapped PagedCache at layer %d\n",
                      layerIndex);
            return INVALID_VALUE;
        }
        directTokens += segment.tokenCount;
        ++directSegments;
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
            MNN_ERROR("CUDAPagedAttention external PIC KV indices exceed int range at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        for (size_t local = 0; local < segment.tokenCount; ++local) {
            int logical = static_cast<int>(segment.logicalStart + local);
            int slot = logical;
            if (slot < 0 || slot >= maxSlots) {
                return OUT_OF_MEMORY;
            }
        }
        const size_t keyTokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
        const int8_t* keySourceDevice = reinterpret_cast<const int8_t*>(keyCacheDevice) +
                                        segment.logicalStart * keyTokenBytes;
        int ropeDim = segment.ropeDim > 0 ? segment.ropeDim : headDim;
        ropeDim = std::min(ropeDim, headDim);
        ropeDim = (ropeDim / 2) * 2;
        const int oldContext = segment.ropeScalingOriginalMaxPositionEmbeddings > 0
            ? segment.ropeScalingOriginalMaxPositionEmbeddings
            : segment.maxPositionEmbeddings;
        const size_t totalElements = segment.tokenCount * static_cast<size_t>(batch) * kvHeads * headDim;
        if (totalElements > maxInt) {
            MNN_ERROR("CUDAPagedAttention external PIC KV hydrate element count exceeds int range at layer %d\n",
                      layerIndex);
            return INVALID_VALUE;
        }
        const int totalKey = static_cast<int>(totalElements);
        const int threads = 256;
        const int blocks = UP_DIV(totalKey, threads);
        const char* keyRopeOp = "pic_apply_rope_to_mapped_key_cache";
        {
            ScopedNvtxRange keyRopeNvtx(nvtxLayerRangeName(keyRopeOp, layerIndex, -1, -1, kvLen), nvtx);
            if (bytes == 4) {
                hydrateExternalPagedKeyKernel<float><<<blocks, threads>>>(
                    reinterpret_cast<const float*>(keySourceDevice), reinterpret_cast<float*>(keyCacheDevice),
                    batch, static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots,
                    static_cast<int>(segment.logicalStart), ropeDim,
                    segment.ropeTheta > 0.0f ? segment.ropeTheta : 10000.0f, ropeTypeCode(segment),
                    std::max(segment.ropeScalingFactor, 1.0f), std::max(segment.ropeScalingLowFreqFactor, 1.0e-6f),
                    std::max(segment.ropeScalingHighFreqFactor, 1.0e-6f), oldContext,
                    segment.ropeAttentionScaling > 0.0f ? segment.ropeAttentionScaling : 1.0f, totalKey);
            } else {
                hydrateExternalPagedKeyKernel<half><<<blocks, threads>>>(
                    reinterpret_cast<const half*>(keySourceDevice), reinterpret_cast<half*>(keyCacheDevice),
                    batch, static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots,
                    static_cast<int>(segment.logicalStart), ropeDim,
                    segment.ropeTheta > 0.0f ? segment.ropeTheta : 10000.0f, ropeTypeCode(segment),
                    std::max(segment.ropeScalingFactor, 1.0f), std::max(segment.ropeScalingLowFreqFactor, 1.0e-6f),
                    std::max(segment.ropeScalingHighFreqFactor, 1.0e-6f), oldContext,
                    segment.ropeAttentionScaling > 0.0f ? segment.ropeAttentionScaling : 1.0f, totalKey);
            }
        }
        auto keyKernel = cudaGetLastError();
        if (keyKernel != cudaSuccess) {
            MNN_ERROR("CUDAPagedAttention failed to hydrate external PIC key on GPU at layer %d: %s\n",
                      layerIndex, cudaGetErrorString(keyKernel));
            return INVALID_VALUE;
        }
    }
    meta->markExternalLayerLoaded(layerIndex);
    if (profile) {
        cudaDeviceSynchronize();
        MNN_PRINT("CUDAPagedAttention profile op=%s layer=%d tokens=%d kv_len=%d async_read=%d "
                  "direct_segments=%d direct_tokens=%d us=%llu\n",
                  restoreOp, layerIndex, static_cast<int>(totalTokens), kvLen, prefetched != nullptr ? 1 : 0,
                  directSegments, static_cast<int>(directTokens), static_cast<unsigned long long>(nowUs() - startUs));
    }
    return NO_ERROR;
}

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
