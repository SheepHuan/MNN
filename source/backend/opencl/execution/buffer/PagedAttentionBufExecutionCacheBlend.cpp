//
//  PagedAttentionBufExecutionCacheBlend.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionBufExecutionExternal.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace MNN {
namespace OpenCL {
namespace {

static thread_local int gSteadyStateTuneMeasureDepth = 0;

enum CacheBlendTopKFamily : uint32_t {
    kCacheBlendTopKFamilyLegacy = 0,
    kCacheBlendTopKFamilyStage1024 = 1,
    kCacheBlendTopKFamilyStage2048 = 2,
};

enum TuneSelectionSource : uint32_t {
    kTuneSelectionSourceDefault = 0,
    kTuneSelectionSourceCache = 1,
    kTuneSelectionSourceOnlineTuned = 2,
    kTuneSelectionSourceBenchOverride = 3,
};

struct CacheBlendTopKDispatch {
    uint32_t family = kCacheBlendTopKFamilyLegacy;
    bool useStage = false;
    int stage1LocalSize = 256;
    int stage2LocalSize = 256;
    int blockSize = 0;
    int blockCount = 0;
    int stageCandidateCount = 0;
    int stageSortSize = 0;
};

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

static bool _profilePagedAttentionTune() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_TUNE", false);
}

static bool _legacyB863976OpenCL() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_LEGACY_B863976", false);
}

static uint64_t _nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

template <typename Runner>
static bool _measureSteadyStateCandidate(OpenCLRuntime* runtime, Runner&& runner, uint64_t* elapsedUs) {
    if (runtime == nullptr || elapsedUs == nullptr) {
        return false;
    }
    if (gSteadyStateTuneMeasureDepth > 0) {
        const uint64_t t0 = _nowUs();
        auto timedErr = runner();
        if (timedErr != NO_ERROR) {
            return false;
        }
        runtime->commandQueue().finish();
        *elapsedUs = _nowUs() - t0;
        return true;
    }
    ++gSteadyStateTuneMeasureDepth;
    auto warmErr = runner();
    if (warmErr != NO_ERROR) {
        --gSteadyStateTuneMeasureDepth;
        return false;
    }
    runtime->commandQueue().finish();
    const uint64_t t0 = _nowUs();
    auto timedErr = runner();
    if (timedErr != NO_ERROR) {
        --gSteadyStateTuneMeasureDepth;
        return false;
    }
    runtime->commandQueue().finish();
    *elapsedUs = _nowUs() - t0;
    --gSteadyStateTuneMeasureDepth;
    return true;
}

static const char* _tuneSelectionSourceName(uint32_t source) {
    switch (source) {
        case kTuneSelectionSourceCache:
            return "cache";
        case kTuneSelectionSourceOnlineTuned:
            return "online_tuned";
        case kTuneSelectionSourceBenchOverride:
            return "bench_override";
        case kTuneSelectionSourceDefault:
        default:
            return "default";
    }
}

static bool _compareCacheBlendTopK() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_COMPARE_CACHEBLEND_TOPK", false);
}

static std::string _openCLTuneDeviceKey(OpenCLRuntime* runtime) {
    if (runtime == nullptr) {
        return "unknown";
    }
    switch (runtime->getGpuType()) {
        case GpuType::MALI:
            return "mali";
        case GpuType::ADRENO:
            return _legacyB863976OpenCL() ? "adreno_legacy" : "adreno";
        case GpuType::RADEON:
            return "radeon";
        case GpuType::INTEL:
            return "intel";
        case GpuType::OTHER:
        default:
            return "other";
    }
}

static const char* _cacheBlendTopKFamilyName(uint32_t family) {
    switch (family) {
        case kCacheBlendTopKFamilyLegacy:
            return "legacy";
        case kCacheBlendTopKFamilyStage1024:
            return "stage1024";
        case kCacheBlendTopKFamilyStage2048:
            return "stage2048";
        default:
            return "unknown";
    }
}

static int _cacheBlendTopKNextPow2(int value) {
    if (value <= 1) {
        return 1;
    }
    int rounded = 1;
    while (rounded < value && rounded < (1 << 30)) {
        rounded <<= 1;
    }
    return rounded;
}

static bool _cacheBlendTopKDispatchForFamily(OpenCLRuntime* runtime,
                                             const std::shared_ptr<KernelWrap>& stage1Kernel,
                                             const std::shared_ptr<KernelWrap>& stage2Kernel,
                                             int picTokenCount,
                                             int topK,
                                             uint32_t family,
                                             CacheBlendTopKDispatch* dispatch) {
    if (dispatch == nullptr || runtime == nullptr) {
        return false;
    }
    dispatch->family = family;
    dispatch->useStage = false;
    dispatch->stage1LocalSize = 256;
    dispatch->stage2LocalSize = 256;
    dispatch->blockSize = 0;
    dispatch->blockCount = 0;
    dispatch->stageCandidateCount = 0;
    dispatch->stageSortSize = 0;
    if (family == kCacheBlendTopKFamilyLegacy) {
        return true;
    }
    if (picTokenCount <= 1024 || topK <= 0) {
        return false;
    }
    if (runtime->getGpuType() == GpuType::MALI) {
        // Mali-G610 returns different or even invalid indices from the staged
        // path at high context lengths. Keep Mali on the deterministic legacy
        // top-k until the staged kernel is fixed.
        return false;
    }
    int blockSize = 0;
    switch (family) {
        case kCacheBlendTopKFamilyStage1024:
            blockSize = 1024;
            break;
        case kCacheBlendTopKFamilyStage2048:
            blockSize = 2048;
            break;
        default:
            return false;
    }
    if (topK > blockSize) {
        return false;
    }
    const int blockCount = UP_DIV(picTokenCount, blockSize);
    const int candidateCount = blockCount * topK;
    const int sortSize = _cacheBlendTopKNextPow2(candidateCount);
    constexpr int kTopKMaxSortSize = 4096;
    if (blockCount <= 0 || candidateCount <= 0 || sortSize > kTopKMaxSortSize) {
        return false;
    }
    const size_t stage1Bytes = static_cast<size_t>(blockSize) * (sizeof(float) + sizeof(int));
    const size_t stage2Bytes = static_cast<size_t>(sortSize) * (sizeof(float) + sizeof(int));
    size_t localMem = runtime->getMaxLocalMem();
    if (std::max(stage1Bytes, stage2Bytes) > localMem) {
        return false;
    }
    constexpr int kMaxPreferredLocalSize = 256;
    const int stage2LocalSize = std::max(1, std::min(kMaxPreferredLocalSize, sortSize));
    if (runtime->getMaxWorkGroupSize(stage1Kernel) < kMaxPreferredLocalSize ||
        runtime->getMaxWorkGroupSize(stage2Kernel) < stage2LocalSize) {
        return false;
    }
    dispatch->useStage = true;
    dispatch->stage1LocalSize = kMaxPreferredLocalSize;
    dispatch->stage2LocalSize = stage2LocalSize;
    dispatch->blockSize = blockSize;
    dispatch->blockCount = blockCount;
    dispatch->stageCandidateCount = candidateCount;
    dispatch->stageSortSize = sortSize;
    return true;
}

static std::vector<uint32_t> _cacheBlendTopKFamilyCandidates(OpenCLRuntime* runtime,
                                                             const std::shared_ptr<KernelWrap>& stage1Kernel,
                                                             const std::shared_ptr<KernelWrap>& stage2Kernel,
                                                             int picTokenCount,
                                                             int topK) {
    std::vector<uint32_t> candidates;
    if (picTokenCount <= 1024 || topK <= 0 || runtime == nullptr) {
        return candidates;
    }
    CacheBlendTopKDispatch dispatch;
    for (uint32_t family : {kCacheBlendTopKFamilyStage1024, kCacheBlendTopKFamilyStage2048}) {
        if (_cacheBlendTopKDispatchForFamily(runtime, stage1Kernel, stage2Kernel, picTokenCount, topK, family,
                                             &dispatch)) {
            candidates.emplace_back(family);
        }
    }
    return candidates;
}

static bool _shouldTuneCacheBlendTopKFamily(int picTokenCount, int topK, bool profile, int tuneLevel) {
    if ((profile && !_profilePagedAttentionTune()) || picTokenCount <= 1024 || topK <= 0) {
        return false;
    }
    return tuneLevel == Heavy || tuneLevel == Wide;
}

static bool _useAdrenoCacheBlendValueImage(OpenCLRuntime* runtime, int batch, int kvHeads, int tokenCount,
                                           int headDim) {
    return PagedAttentionAdreno::useAdrenoCacheBlendValueImage(runtime, _legacyB863976OpenCL(), batch, kvHeads,
                                                               tokenCount, headDim);
}

static bool _computeAdrenoCacheBlendValueImageShape(OpenCLRuntime* runtime, int batch, int kvHeads,
                                                    int tokenCount, int headDim,
                                                    int* imageWidth, int* imageHeight) {
    return PagedAttentionAdreno::computeAdrenoCacheBlendValueImageShape(runtime, _legacyB863976OpenCL(), batch,
                                                                        kvHeads, tokenCount, headDim,
                                                                        imageWidth, imageHeight);
}

} // namespace

ErrorCode PagedAttentionBufExecution::ensureCacheBlendScoreTemps(int scoreCount, int indexCount,
                                                                 int stageCandidateCount) {
    if (scoreCount < 0 || indexCount < 0) {
        return INVALID_VALUE;
    }
    if (stageCandidateCount < 0) {
        return INVALID_VALUE;
    }
    if (scoreCount == 0 && indexCount == 0 && stageCandidateCount == 0) {
        return NO_ERROR;
    }
    if (mCacheBlendScores && mCacheBlendIndices && mCacheBlendScoreCount >= scoreCount &&
        mCacheBlendIndexCount >= indexCount &&
        (stageCandidateCount == 0 ||
         (mCacheBlendStageValues && mCacheBlendStageIndices &&
          mCacheBlendStageCandidateCount >= stageCandidateCount))) {
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
    if (stageCandidateCount > 0) {
        mCacheBlendStageValues.reset(Tensor::createDevice<float>({stageCandidateCount}));
        if (!mCacheBlendStageValues) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCacheBlendStageValues.get(), Backend::STATIC));
        mCacheBlendStageIndices.reset(Tensor::createDevice<int>({stageCandidateCount}));
        if (!mCacheBlendStageIndices) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCacheBlendStageIndices.get(), Backend::STATIC));
    }
    mCacheBlendScoreCount = scoreCount;
    mCacheBlendIndexCount = indexCount;
    mCacheBlendStageCandidateCount = stageCandidateCount;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureAdrenoCacheBlendValueImage(int tokenCapacity) {
    if (tokenCapacity <= 0 || mBatch <= 0 || mKvNumHead <= 0 || mHeadDim <= 0) {
        return INVALID_VALUE;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr) {
        return INVALID_VALUE;
    }
    int imageWidth = 0;
    int imageHeight = 0;
    if (!_computeAdrenoCacheBlendValueImageShape(runtime, mBatch, mKvNumHead, tokenCapacity, mHeadDim,
                                                 &imageWidth, &imageHeight)) {
        return NOT_SUPPORT;
    }
    const bool useFp32 = mBytes == static_cast<int>(sizeof(float));
    if (!mCopyBufferToImageLinearKernel || mCopyBufferToImageLinearUseFp32 != useFp32) {
        std::set<std::string> buildOptions;
        if (useFp32) {
            buildOptions.emplace("-DBUFFER_INP_FP32");
        }
        mCopyBufferToImageLinearKernel = runtime->buildKernel("copy_buffer_to_image2d", "copy_buffer_to_image2d",
                                                              buildOptions, mOpenCLBackend->getPrecision());
        if (mCopyBufferToImageLinearKernel == nullptr) {
            return NOT_SUPPORT;
        }
        mCopyBufferToImageLinearUseFp32 = useFp32;
    }
    if (mCacheBlendSourceValueImage != nullptr && mCacheBlendSourceValueTokenCapacity >= tokenCapacity &&
        mCacheBlendSourceValueImageHeight == imageHeight && mCacheBlendSourceValueBytes == mBytes) {
        return NO_ERROR;
    }
    if (!mAdrenoImagePool) {
        mAdrenoImagePool.reset(new ImagePool(runtime->context()));
    }
    auto image = mAdrenoImagePool->alloc(imageWidth, imageHeight, mOpenCLBackend->fpType());
    if (image == nullptr) {
        return OUT_OF_MEMORY;
    }
    mCacheBlendSourceValueImage = image;
    mCacheBlendSourceValueTokenCapacity = tokenCapacity;
    mCacheBlendSourceValueImageWidth = imageWidth;
    mCacheBlendSourceValueImageHeight = imageHeight;
    mCacheBlendSourceValueBytes = mBytes;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runCacheBlendScoring(int layerIndex, int kvLen) {
    if (mMeta == nullptr || !mMeta->needsCacheBlendScoring(layerIndex)) {
        return NO_ERROR;
    }
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t readUs = 0;
    uint64_t imagePrepUs = 0;
    uint64_t scoreKernelUs = 0;
    uint64_t topKUs = 0;
    uint64_t readbackUs = 0;
    int imageSegments = 0;
    size_t imageTokens = 0;
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
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto topKFamilies = _cacheBlendTopKFamilyCandidates(runtime, mCacheBlendTopKStage1Kernel,
                                                        mCacheBlendTopKStage2Kernel, picTokenCount, topK);
    CacheBlendTopKDispatch topKDispatch;
    uint32_t topKFamilySource = kTuneSelectionSourceDefault;
    const uint32_t defaultTopKFamily =
        _cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel, mCacheBlendTopKStage2Kernel,
                                         picTokenCount, topK, kCacheBlendTopKFamilyStage2048, &topKDispatch)
        ? kCacheBlendTopKFamilyStage2048
        : (topKFamilies.empty() ? kCacheBlendTopKFamilyLegacy : topKFamilies.front());
    if (defaultTopKFamily != topKDispatch.family) {
        _cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel, mCacheBlendTopKStage2Kernel,
                                         picTokenCount, topK, defaultTopKFamily, &topKDispatch);
    }
    int stageCandidateCapacity = 0;
    for (uint32_t family : topKFamilies) {
        CacheBlendTopKDispatch candidateDispatch;
        if (_cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel, mCacheBlendTopKStage2Kernel,
                                             picTokenCount, topK, family, &candidateDispatch)) {
            stageCandidateCapacity = std::max(stageCandidateCapacity, candidateDispatch.stageCandidateCount);
        }
    }
    auto err = ensureCacheBlendScoreTemps(picTokenCount, topK, stageCandidateCapacity);
    if (err != NO_ERROR) {
        return err;
    }
    auto& queue = runtime->commandQueue();
    size_t scoreOffset = 0;
    int sourceSlotCursor = _picCacheSourceSlotBase(mMeta, kvLen);
    for (const auto& segment : mMeta->cacheblend_score_segments) {
        sourceSlotCursor = _alignPicCacheSourceSlots(sourceSlotCursor);
        const int sourceSlotStart = sourceSlotCursor;
        sourceSlotCursor += static_cast<int>(segment.tokenCount);
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
        if (sourceSlotStart < 0 || static_cast<size_t>(sourceSlotStart) + segment.tokenCount >
            static_cast<size_t>(mCache->maxSlots)) {
            return OUT_OF_MEMORY;
        }
        const size_t valueTokenBytes = static_cast<size_t>(mHeadDim) * mBytes;
        const size_t valueSegmentBytes = static_cast<size_t>(mBatch) * mKvNumHead * segment.tokenCount *
                                         valueTokenBytes;
        auto& cacheValueBuffer = openCLBuffer(mCache->value.get());
        const bool preferImageScore = _useAdrenoCacheBlendValueImage(runtime, mBatch, mKvNumHead,
                                                                     static_cast<int>(segment.tokenCount), mHeadDim);
        bool usedImageScore = false;
        uint64_t opStartUs = profileDetail ? _nowUs() : 0;
        if (preferImageScore) {
            const size_t valueSegmentElements = valueSegmentBytes / static_cast<size_t>(mBytes);
            if (ensureExternalTemps(0, valueSegmentElements) == NO_ERROR &&
                ensureAdrenoCacheBlendValueImage(static_cast<int>(segment.tokenCount)) == NO_ERROR &&
                _readExternalValueSegmentToSourceCLBuffer(
                    layer->valuePath, openCLBuffer(mExternalValue.get()), valueSegmentBytes, queue, mBatch,
                    mKvNumHead, sourceTokenCount, sourceTokenOffset, segment.tokenCount, mHeadDim, mBytes)) {
                if (profileDetail) {
                    queue.finish();
                    readUs += _nowUs() - opStartUs;
                }
                uint32_t copyIdx = 0;
                cl_int copyRet = CL_SUCCESS;
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, openCLBuffer(mExternalValue.get()));
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, *mCacheBlendSourceValueImage);
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, mCacheBlendSourceValueImageWidth);
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, mCacheBlendSourceValueImageHeight);
                MNN_CHECK_CL_SUCCESS(copyRet, "setArg copy_buffer_to_image2d");
                if (copyRet == CL_SUCCESS) {
                    opStartUs = profileDetail ? _nowUs() : 0;
                    copyRet = queue.enqueueNDRangeKernel(mCopyBufferToImageLinearKernel->get(), cl::NullRange,
                                                         cl::NDRange(mCacheBlendSourceValueImageWidth,
                                                                     mCacheBlendSourceValueImageHeight),
                                                         cl::NullRange);
                    MNN_CHECK_CL_SUCCESS(copyRet, "enqueue copy_buffer_to_image2d");
                    if (copyRet == CL_SUCCESS) {
                        if (profileDetail) {
                            queue.finish();
                            imagePrepUs += _nowUs() - opStartUs;
                        }
                        uint32_t idx = 0;
                        cl_int ret = CL_SUCCESS;
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, *mCacheBlendSourceValueImage);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mBatch);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mKvNumHead);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mHeadDim);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mCache->maxSlots);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mCacheBlendSourceValueTokenCapacity);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mCacheBlendSourceValueImageWidth);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, static_cast<int>(scoreOffset));
                        MNN_CHECK_CL_SUCCESS(ret, "setArg pic_cacheblend_value_score_cached_image");
                        if (ret == CL_SUCCESS) {
                            opStartUs = profileDetail ? _nowUs() : 0;
                            ret = queue.enqueueNDRangeKernel(mCacheBlendScoreImageKernel->get(), cl::NullRange,
                                                             cl::NDRange(static_cast<int>(segment.tokenCount)),
                                                             cl::NullRange);
                            MNN_CHECK_CL_SUCCESS(ret, "enqueue pic_cacheblend_value_score_cached_image");
                            if (ret == CL_SUCCESS) {
                                if (profileDetail) {
                                    queue.finish();
                                    scoreKernelUs += _nowUs() - opStartUs;
                                }
                                usedImageScore = true;
                                ++imageSegments;
                                imageTokens += segment.tokenCount;
                            }
                        }
                    }
                }
            }
        }
        if (!usedImageScore) {
            opStartUs = profileDetail ? _nowUs() : 0;
            if (!_readExternalValueSegmentToPagedCacheOpenCL(
                    layer->valuePath, cacheValueBuffer, queue, mBatch, mKvNumHead, mCache->maxSlots,
                    static_cast<size_t>(sourceSlotStart), sourceTokenCount, sourceTokenOffset, segment.tokenCount,
                    mHeadDim, mBytes)) {
                return INVALID_VALUE;
            }
            if (profileDetail) {
                queue.finish();
                readUs += _nowUs() - opStartUs;
            }
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, mBatch);
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, mKvNumHead);
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, mHeadDim);
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, mCache->maxSlots);
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, mCache->maxSlots);
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, sourceSlotStart);
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
            ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(scoreOffset));
            MNN_CHECK_CL_SUCCESS(ret, "setArg pic_cacheblend_value_score");
            opStartUs = profileDetail ? _nowUs() : 0;
            ret = queue.enqueueNDRangeKernel(mCacheBlendScoreKernel->get(), cl::NullRange,
                                             cl::NDRange(static_cast<int>(segment.tokenCount)), cl::NullRange);
            MNN_CHECK_CL_SUCCESS(ret, "enqueue pic_cacheblend_value_score");
            if (profileDetail) {
                queue.finish();
                scoreKernelUs += _nowUs() - opStartUs;
            }
        }
        scoreOffset += segment.tokenCount;
    }
    if (scoreOffset != static_cast<size_t>(picTokenCount)) {
        return INVALID_VALUE;
    }
    auto runTopKFamily = [&](const CacheBlendTopKDispatch& dispatch) -> ErrorCode {
        cl_int localRet = CL_SUCCESS;
        if (!dispatch.useStage) {
            uint32_t idx = 0;
            localRet |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
            localRet |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendIndices.get()));
            localRet |= mCacheBlendTopKKernel->get().setArg(idx++, picTokenCount);
            localRet |= mCacheBlendTopKKernel->get().setArg(idx++, topK);
            MNN_CHECK_CL_SUCCESS(localRet, "setArg pic_cacheblend_topk");
            if (localRet != CL_SUCCESS) {
                return INVALID_VALUE;
            }
            localRet = queue.enqueueNDRangeKernel(mCacheBlendTopKKernel->get(), cl::NullRange,
                                                  cl::NDRange(256), cl::NDRange(256));
            MNN_CHECK_CL_SUCCESS(localRet, "enqueue pic_cacheblend_topk");
            if (localRet != CL_SUCCESS) {
                return INVALID_VALUE;
            }
            return NO_ERROR;
        }
        uint32_t idx = 0;
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendStageValues.get()));
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendStageIndices.get()));
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, picTokenCount);
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, topK);
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, dispatch.blockSize);
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(
            idx++, cl::Local(static_cast<size_t>(dispatch.blockSize) * sizeof(float)));
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(
            idx++, cl::Local(static_cast<size_t>(dispatch.blockSize) * sizeof(int)));
        MNN_CHECK_CL_SUCCESS(localRet, "setArg pic_cacheblend_topk_stage1");
        if (localRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        localRet = queue.enqueueNDRangeKernel(mCacheBlendTopKStage1Kernel->get(), cl::NullRange,
                                              cl::NDRange(dispatch.blockCount * dispatch.stage1LocalSize),
                                              cl::NDRange(dispatch.stage1LocalSize));
        MNN_CHECK_CL_SUCCESS(localRet, "enqueue pic_cacheblend_topk_stage1");
        if (localRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        idx = 0;
        localRet = CL_SUCCESS;
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendStageValues.get()));
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendStageIndices.get()));
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendIndices.get()));
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, dispatch.stageCandidateCount);
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, topK);
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, dispatch.stageSortSize);
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(
            idx++, cl::Local(static_cast<size_t>(dispatch.stageSortSize) * sizeof(float)));
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(
            idx++, cl::Local(static_cast<size_t>(dispatch.stageSortSize) * sizeof(int)));
        MNN_CHECK_CL_SUCCESS(localRet, "setArg pic_cacheblend_topk_stage2");
        if (localRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        localRet = queue.enqueueNDRangeKernel(mCacheBlendTopKStage2Kernel->get(), cl::NullRange,
                                              cl::NDRange(dispatch.stage2LocalSize),
                                              cl::NDRange(dispatch.stage2LocalSize));
        MNN_CHECK_CL_SUCCESS(localRet, "enqueue pic_cacheblend_topk_stage2");
        if (localRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        return NO_ERROR;
    };
    if (_shouldTuneCacheBlendTopKFamily(picTokenCount, topK, profile, mOpenCLBackend->getCLTuneLevel()) &&
        !topKFamilies.empty()) {
        const std::string tuneKey =
            std::string("paged_cacheblend_topk_family_") + _openCLTuneDeviceKey(runtime) + "_" +
            std::to_string(mBatch) + "_" + std::to_string(mNumHead) + "_" +
            std::to_string(mKvNumHead) + "_" + std::to_string(mHeadDim);
        const std::vector<uint32_t> tuneShape = {
            static_cast<uint32_t>(picTokenCount),
            static_cast<uint32_t>(topK),
            static_cast<uint32_t>((static_cast<uint64_t>(topK) * 1000u) /
                                  static_cast<uint64_t>(std::max(1, picTokenCount))),
        };
        std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
        if (getTunedInfo(tuneKey, tuneShape, tuneInfo, runtime) && !tuneInfo.first.empty()) {
            CacheBlendTopKDispatch cachedDispatch;
            if (_cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel,
                                                 mCacheBlendTopKStage2Kernel, picTokenCount, topK,
                                                 tuneInfo.first[0], &cachedDispatch)) {
                topKDispatch = cachedDispatch;
                topKFamilySource = kTuneSelectionSourceCache;
            }
        } else if (topKFamilies.size() > 1) {
            bool tuned = false;
            uint64_t bestUs = std::numeric_limits<uint64_t>::max();
            CacheBlendTopKDispatch bestDispatch = topKDispatch;
            for (uint32_t family : topKFamilies) {
                CacheBlendTopKDispatch candidateDispatch;
                if (!_cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel,
                                                      mCacheBlendTopKStage2Kernel, picTokenCount, topK, family,
                                                      &candidateDispatch)) {
                    continue;
                }
                uint64_t candidateUs = 0;
                if (!_measureSteadyStateCandidate(runtime, [&]() {
                        return runTopKFamily(candidateDispatch);
                    }, &candidateUs)) {
                    continue;
                }
                if (!tuned || candidateUs < bestUs) {
                    tuned = true;
                    bestUs = candidateUs;
                    bestDispatch = candidateDispatch;
                }
            }
            if (tuned) {
                std::pair<std::vector<uint32_t>, uint32_t> bestInfo = std::make_pair(
                    std::vector<uint32_t>{bestDispatch.family},
                    static_cast<uint32_t>(std::min<uint64_t>(bestUs, std::numeric_limits<uint32_t>::max())));
                setTunedInfo(tuneKey, tuneShape, bestInfo, runtime, "attention_buf");
                topKDispatch = bestDispatch;
                topKFamilySource = kTuneSelectionSourceOnlineTuned;
            }
        }
        if (profile) {
            MNN_PRINT("OpenCLPagedAttention profile op=cacheblend_topk_family layer=%d pic_tokens=%d top_k=%d family=%s source=%s\n",
                      layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                      _tuneSelectionSourceName(topKFamilySource));
        }
    }
    uint64_t opStartUs = profileDetail ? _nowUs() : 0;
    err = runTopKFamily(topKDispatch);
    if (err != NO_ERROR && !topKDispatch.useStage) {
        return err;
    }
    if (err != NO_ERROR && topKDispatch.useStage) {
        MNN_ERROR("OpenCLPagedAttention cacheblend staged top-k failed, retrying legacy path layer=%d "
                  "pic_tokens=%d top_k=%d family=%s\n",
                  layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family));
        CacheBlendTopKDispatch legacyDispatch;
        auto legacyErr = runTopKFamily(legacyDispatch);
        if (legacyErr != NO_ERROR) {
            return legacyErr;
        }
        topKDispatch = legacyDispatch;
        topKFamilySource = kTuneSelectionSourceDefault;
    }
    if (profileDetail) {
        queue.finish();
        topKUs += _nowUs() - opStartUs;
    }
    std::vector<int> selected(topK);
    auto readAndValidateSelected = [&](std::vector<int>* out, std::string* reason) -> ErrorCode {
        if (out == nullptr) {
            return INVALID_VALUE;
        }
        out->assign(static_cast<size_t>(topK), -1);
        cl_int readRet = queue.enqueueReadBuffer(openCLBuffer(mCacheBlendIndices.get()), CL_TRUE, 0,
                                                 static_cast<size_t>(topK) * sizeof(int), out->data());
        MNN_CHECK_CL_SUCCESS(readRet, "read pic_cacheblend_topk indices");
        if (readRet != CL_SUCCESS) {
            if (reason != nullptr) {
                *reason = std::string("read_failed ret=") + std::to_string(static_cast<int>(readRet));
            }
            return INVALID_VALUE;
        }
        std::vector<uint8_t> seen(static_cast<size_t>(picTokenCount), 0);
        for (int i = 0; i < topK; ++i) {
            const int index = (*out)[static_cast<size_t>(i)];
            if (index < 0 || index >= picTokenCount) {
                if (reason != nullptr) {
                    *reason = std::string("out_of_range at=") + std::to_string(i) +
                              " index=" + std::to_string(index);
                }
                return INVALID_VALUE;
            }
            if (seen[static_cast<size_t>(index)] != 0) {
                if (reason != nullptr) {
                    *reason = std::string("duplicate at=") + std::to_string(i) +
                              " index=" + std::to_string(index);
                }
                return INVALID_VALUE;
            }
            seen[static_cast<size_t>(index)] = 1;
        }
        return NO_ERROR;
    };
    opStartUs = profileDetail ? _nowUs() : 0;
    std::string invalidReason;
    err = readAndValidateSelected(&selected, &invalidReason);
    if (profileDetail) {
        readbackUs += _nowUs() - opStartUs;
    }
    if (_compareCacheBlendTopK() && topKDispatch.useStage) {
        const auto stagedErr = err;
        const auto stagedReason = invalidReason;
        const auto stagedSelected = selected;
        CacheBlendTopKDispatch legacyDispatch;
        std::vector<int> legacySelected;
        std::string legacyReason;
        auto legacyErr = runTopKFamily(legacyDispatch);
        if (legacyErr == NO_ERROR) {
            legacyErr = readAndValidateSelected(&legacySelected, &legacyReason);
        }
        auto summarize = [](const std::vector<int>& values) -> std::string {
            std::ostringstream os;
            os << "[";
            const int count = std::min<int>(static_cast<int>(values.size()), 16);
            for (int i = 0; i < count; ++i) {
                if (i > 0) {
                    os << ",";
                }
                os << values[static_cast<size_t>(i)];
            }
            if (static_cast<int>(values.size()) > count) {
                os << ",...";
            }
            os << "]";
            return os.str();
        };
        auto setEqual = [](std::vector<int> lhs, std::vector<int> rhs) -> bool {
            std::sort(lhs.begin(), lhs.end());
            std::sort(rhs.begin(), rhs.end());
            return lhs == rhs;
        };
        int firstMismatch = -1;
        const int compareCount = std::min<int>(static_cast<int>(stagedSelected.size()),
                                               static_cast<int>(legacySelected.size()));
        for (int i = 0; i < compareCount; ++i) {
            if (stagedSelected[static_cast<size_t>(i)] != legacySelected[static_cast<size_t>(i)]) {
                firstMismatch = i;
                break;
            }
        }
        if (firstMismatch < 0 && stagedSelected.size() != legacySelected.size()) {
            firstMismatch = compareCount;
        }
        const bool bothValid = stagedErr == NO_ERROR && legacyErr == NO_ERROR;
        const bool orderedEqual = bothValid && stagedSelected == legacySelected;
        const bool sameSet = bothValid && setEqual(stagedSelected, legacySelected);
        if (orderedEqual) {
            MNN_PRINT("OpenCLPagedAttention cacheblend top-k compare layer=%d pic_tokens=%d top_k=%d "
                      "staged=%s legacy=legacy ordered_equal=1 set_equal=1 sample=%s\n",
                      layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                      summarize(stagedSelected).c_str());
        } else {
            MNN_ERROR("OpenCLPagedAttention cacheblend top-k compare mismatch layer=%d pic_tokens=%d top_k=%d "
                      "staged=%s staged_err=%d staged_reason=%s legacy_err=%d legacy_reason=%s "
                      "ordered_equal=%d set_equal=%d first_mismatch=%d staged_sample=%s legacy_sample=%s\n",
                      layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                      static_cast<int>(stagedErr), stagedReason.c_str(), static_cast<int>(legacyErr),
                      legacyReason.c_str(), orderedEqual ? 1 : 0, sameSet ? 1 : 0, firstMismatch,
                      summarize(stagedSelected).c_str(), summarize(legacySelected).c_str());
        }
    }
    if (err != NO_ERROR && topKDispatch.useStage) {
        MNN_ERROR("OpenCLPagedAttention cacheblend staged top-k produced invalid selected rows, "
                  "retrying legacy path layer=%d pic_tokens=%d top_k=%d family=%s reason=%s\n",
                  layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                  invalidReason.c_str());
        CacheBlendTopKDispatch legacyDispatch;
        err = runTopKFamily(legacyDispatch);
        if (err != NO_ERROR) {
            return err;
        }
        topKDispatch = legacyDispatch;
        topKFamilySource = kTuneSelectionSourceDefault;
        opStartUs = profileDetail ? _nowUs() : 0;
        invalidReason.clear();
        err = readAndValidateSelected(&selected, &invalidReason);
        if (profileDetail) {
            readbackUs += _nowUs() - opStartUs;
        }
    }
    if (err != NO_ERROR) {
        MNN_ERROR("OpenCLPagedAttention cacheblend top-k selected rows invalid layer=%d pic_tokens=%d top_k=%d "
                  "path=%s reason=%s\n",
                  layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                  invalidReason.c_str());
        return err;
    }
    mMeta->setCacheBlendScoringResult(selected);
    if (profile) {
        queue.finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=cacheblend_score layer=%d pic_tokens=%d top_k=%d us=%llu "
                  "read_us=%llu image_prep_us=%llu score_kernel_us=%llu topk_us=%llu readback_us=%llu "
                  "detail=%d topk_path=%s image_segments=%d image_tokens=%d stage_candidates=%d "
                  "stage_block=%d stage_sort=%d\n",
                  layerIndex, picTokenCount, topK,
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(readUs),
                  static_cast<unsigned long long>(imagePrepUs),
                  static_cast<unsigned long long>(scoreKernelUs),
                  static_cast<unsigned long long>(topKUs),
                  static_cast<unsigned long long>(readbackUs),
                  profileDetail ? 1 : 0,
                  _cacheBlendTopKFamilyName(topKDispatch.family),
                  imageSegments,
                  static_cast<int>(imageTokens),
                  topKDispatch.useStage ? topKDispatch.stageCandidateCount : 0,
                  topKDispatch.useStage ? topKDispatch.blockSize : 0,
                  topKDispatch.useStage ? topKDispatch.stageSortSize : 0);
    }
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN

#endif // MNN_OPENCL_BUFFER_CLOSED
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
