//
//  CPUKVCacheManager.cpp
//  MNN
//
//  Created by MNN on 2024/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CPUKVCacheManager.hpp"
#include "core/Concurrency.h"
#include "core/PagedKVCachePlan.hpp"
#include "core/PrefixCachePath.hpp"
#include "core/PrefixKVHostCache.hpp"

#include <chrono>
#include <cmath>

namespace MNN {

namespace {
constexpr const char* kCpuPrefixLayout = "cpu-flash-packed-canonical-no-rope-v1";

static double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}
}

/*
**  @brief  Expand the size of kvcache and copy it from the old tensor in memory to the new tensor in memory
**          Finally reset the pointer to the new tensor
*/
void CPUKVCacheManager::expandKVCacheInMem(int oldMaxLength) {
    /*===================================  Key  ===================================*/
    auto new_key = Tensor::createDevice<int8_t>({mKvNumHead, (int)mCurrentKeySizePerHead});
    mBackend->onAcquireBuffer(new_key, Backend::STATIC);
    if (mKeyQuantMode != KVQuantMode::None) {
        memset(new_key->host<int8_t>(), 0, mKvNumHead * mCurrentKeySizePerHead);
    }
    for (int h = 0; h < mKvNumHead; h++) {
        memcpy(
               new_key->host<int8_t>() + h * mCurrentKeySizePerHead,
               mPastKey->host<int8_t>() + h * mPastKey->stride(0),
               mPastKey->stride(0)
               );
        if (mKeyQuantMode != KVQuantMode::Int8 && (new_key->stride(0) - mPastKey->stride(0)) > 0) {
            memset(new_key->host<int8_t>() + h * new_key->stride(0) + mPastKey->stride(0), 0, (new_key->stride(0) - mPastKey->stride(0)));
        }
    }
    mPastKey.reset(new_key);
    /*===================================  Value  ===================================*/
    auto newValue = Tensor::createDevice<int8_t>({mKvNumHead, (int)mCurrentValueSizePerHead});
    mBackend->onAcquireBuffer(newValue, Backend::STATIC);

    if (mUseFlashAttention) { // [mKvNumHead, UP_DIV(mMaxLength, mFlashAttentionUpperKv), UP_DIV(mHeadDim, hP), UP_DIV(mFlashAttentionUpperKv, lP), hP, lP]
        for (int h = 0; h < mKvNumHead; h++) {
            memset(newValue->host<int8_t>() + h * newValue->stride(0), 0, newValue->stride(0));
            memcpy(
                newValue->host<int8_t>() + h * newValue->stride(0),
                mPastValue->host<int8_t>() + h * mPastValue->stride(0),
                mPastValue->stride(0)
            );
        }
    } else {
        if (mValueQuantMode == KVQuantMode::Int8) { // [mKvNumHead, UP_DIV(mHeadDim, hP8), (UP_DIV(mMaxLength,
                                                    // lP8)*hP8*lP8+2*hP8*sizeof(float)) ]
            auto currentWeightInside = ROUND_UP(mMaxLength, lP8) * hP8;
            auto currentStride1 = currentWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
            auto currentStride0 = currentStride1 * UP_DIV(mHeadDim, hP8);

            auto prevWeightInside = ROUND_UP(oldMaxLength, lP8) * hP8;
            auto prevStride1 = prevWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
            auto prevStride0 = prevStride1 * UP_DIV(mHeadDim, hP8);
            for (int h = 0; h < mKvNumHead; ++h) {
                for (int d = 0; d < UP_DIV(mHeadDim, hP8); ++d) {
                    auto dstPtr = newValue->host<int8_t>() + h * currentStride0 + d * currentStride1;
                    auto srcPtr = mPastValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                    // initialize 0 for weightInt8
                    memset(dstPtr, 0, currentWeightInside);
                    // copy inner side weightInt8
                    memcpy(dstPtr, srcPtr, prevWeightInside);
                    // copy hP8 scale&bias
                    memcpy(dstPtr + currentWeightInside, srcPtr + prevWeightInside, 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES);
                }
            }
        } else { // [mKvNumHead, UP_DIV(mHeadDim, hP), UP_DIV(mMaxLength, lP), hP, lP]
            auto currentStride1 = ROUND_UP(mMaxLength, lP) * hP * mBytes;
            auto currentStride0 = ROUND_UP(mMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;

            auto prevStride1 = ROUND_UP(oldMaxLength, lP) * hP * mBytes;
            auto prevStride0 = ROUND_UP(oldMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;
            for (int h = 0; h < mKvNumHead; ++h) {
                for (int d = 0; d < UP_DIV(mHeadDim, hP); ++d) {
                    auto dstPtr = newValue->host<int8_t>() + h * currentStride0 + d * currentStride1;
                    auto srcPtr = mPastValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                    // initialize 0 for weight
                    if (lP > 1) {
                        memset(dstPtr, 0, currentStride1);
                    }
                    // copy inner side weight
                    memcpy(dstPtr, srcPtr, prevStride1);
                }
            }
        }
    }
    mPastValue.reset(newValue);
}

/*
**  @brief  Move the kvcache from memory to the memory-mapped kvcache files in disk
**          Then release the memory buffer of old kvcache
*/
void CPUKVCacheManager::moveKVCacheFromMemToDisk(int oldMaxLength) {
    /*===================================  Key  ===================================*/
    size_t prevKeySizePerHead = 0;
    if (mKeyQuantMode == KVQuantMode::Int8) {
        prevKeySizePerHead = ROUND_UP(oldMaxLength, hP8) * ROUND_UP(mHeadDim, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(oldMaxLength, hP8);
    } else {
        prevKeySizePerHead = UP_DIV(oldMaxLength, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
    }
    if (mHeadDim % lP || (mKeyQuantMode == KVQuantMode::Int8)) {
        memset(mMapKeyAddr, 0, mKvNumHead * mCurrentKeySizePerHead);
    }
    for (int h = 0; h < mKvNumHead; h++) {
        memcpy(
            mMapKeyAddr + h * mCurrentKeySizePerHead,
            mPastKey->host<int8_t>() + h * prevKeySizePerHead,
            prevKeySizePerHead
        );
    }
    mBackend->onReleaseBuffer(mPastKey.get(), Backend::STATIC);
    mPastKey.reset();
    /*===================================  Value  ===================================*/
    {
        size_t prevValueSizePerHead = 0;
        if (mValueQuantMode == KVQuantMode::Int8) {
            prevValueSizePerHead = UP_DIV(oldMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mFlashAttentionUpperKv, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mHeadDim, hP8));
        } else {
            prevValueSizePerHead = UP_DIV(oldMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
        }
        if (lP > 1 || (mValueQuantMode == KVQuantMode::Int8)) {
            memset(mMapValueAddr, 0, mKvNumHead * mCurrentValueSizePerHead);
        }

        if (mUseFlashAttention) {
            for (int h = 0; h < mKvNumHead; h++) {
                memcpy(
                       mMapValueAddr + h * mCurrentValueSizePerHead,
                       mPastValue->host<int8_t>() + h * prevValueSizePerHead,
                       prevValueSizePerHead
                       );
            }
        } else {
            if (mValueQuantMode == KVQuantMode::Int8) { // [mKvNumHead, UP_DIV(mHeadDim, hP8), (UP_DIV(mMaxLength,
                                                        // lP8)*hP8*lP8+2*hP8*sizeof(float)) ]
                auto currentWeightInside = ROUND_UP(mMaxLength, lP8) * hP8;
                auto currentStride1 = currentWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
                auto currentStride0 = currentStride1 * UP_DIV(mHeadDim, hP8);

                auto prevWeightInside = ROUND_UP(oldMaxLength, lP8) * hP8;
                auto prevStride1 = prevWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
                auto prevStride0 = prevStride1 * UP_DIV(mHeadDim, hP8);
                for (int h = 0; h < mKvNumHead; ++h) {
                    for (int d = 0; d < UP_DIV(mHeadDim, hP8); ++d) {
                        auto dstPtr = mMapValueAddr + h * currentStride0 + d * currentStride1;
                        auto srcPtr = mPastValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                        // initialize 0 for weightInt8
                        memset(dstPtr, 0, currentWeightInside);
                        // copy inner side weightInt8
                        memcpy(dstPtr, srcPtr, prevWeightInside);
                        // copy hP8 scale&bias
                        memcpy(dstPtr + currentWeightInside, srcPtr + prevWeightInside, 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES);
                    }
                }
            } else { // [mKvNumHead, UP_DIV(mHeadDim, hP), UP_DIV(mMaxLength, lP), hP, lP]
                auto currentStride1 = ROUND_UP(mMaxLength, lP) * hP * mBytes;
                auto currentStride0 = ROUND_UP(mMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;

                auto prevStride1 = ROUND_UP(oldMaxLength, lP) * hP * mBytes;
                auto prevStride0 = ROUND_UP(oldMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;
                for (int h = 0; h < mKvNumHead; ++h) {
                    for (int d = 0; d < UP_DIV(mHeadDim, hP); ++d) {
                        auto dstPtr = mMapValueAddr + h * currentStride0 + d * currentStride1;
                        auto srcPtr = mPastValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                        // initialize 0 for weight
                        if (lP > 1) {
                            memset(dstPtr, 0, currentStride1);
                        }
                        // copy inner side weight
                        memcpy(dstPtr, srcPtr, prevStride1);
                    }
                }
            }
        }
        mBackend->onReleaseBuffer(mPastValue.get(), Backend::STATIC);
        mPastValue.reset();
    }
}

/*
**  @brief  Expand the size of kvcache files in disk
*/
void CPUKVCacheManager::expandKVCacheInDisk(int oldMaxLength, int oldKeySize, int oldValueSize, int keySize, int valueSize, file_t specKeyFile, file_t specValueFile) {
    // Step 1: Copy the old kvcache from files to temporary buffers in memory
    auto prevKeySizePerHead = oldKeySize / mKvNumHead;
    auto prevValueSizePerHead = oldValueSize / mKvNumHead;
    std::shared_ptr<Tensor> prevKey, prevValue;
    prevKey.reset(Tensor::createDevice<int8_t>({mKvNumHead, prevKeySizePerHead}));
    prevValue.reset(Tensor::createDevice<int8_t>({mKvNumHead, prevValueSizePerHead}));

    mBackend->onAcquireBuffer(prevKey.get(), Backend::STATIC);
    mBackend->onAcquireBuffer(prevValue.get(), Backend::STATIC);
    if (mHeadDim % lP) {
        memset(prevKey->host<uint8_t>(), 0, prevKey->length(0) * prevKey->stride(0));
    }
    if (lP > 1) {
        // can't be mMaxLenth % lP, since mMaxLength may be larger than seq_len for prefilling, we should ensure the (mMaxLength - seq_len)'s buffer is 0.
        // computing L is seq_len
        memset(prevValue->host<uint8_t>(), 0, prevValue->length(0) * prevValue->stride(0));
    }
    mmapKVCache(oldKeySize, oldValueSize, specKeyFile, specValueFile);
    memcpy(prevKey->host<int8_t>(),   mMapKeyAddr,   oldKeySize);
    memcpy(prevValue->host<int8_t>(), mMapValueAddr, oldValueSize);
    // Step 2: Resize the kvcache files and remap them
    unmapKVCache(oldKeySize, oldValueSize);
    resetKVCacheFileSize(keySize, valueSize);
    mmapKVCache(keySize, valueSize);
    // Step 3: Move the kvcache from temporary buffers in memory to disk
    memset(mMapKeyAddr, 0, keySize);
    memset(mMapValueAddr, 0, valueSize);

    for (int h = 0; h < mKvNumHead; h++) {
        memcpy(mMapKeyAddr + h * mCurrentKeySizePerHead, prevKey->host<int8_t>() + h * prevKeySizePerHead, prevKeySizePerHead);
    }

    if (mUseFlashAttention) {
        for (int h = 0; h < mKvNumHead; h++) {
            memcpy(mMapValueAddr + h * mCurrentValueSizePerHead, prevValue->host<int8_t>() + h * prevValueSizePerHead, prevValueSizePerHead);
        }
    } else {
        if (mValueQuantMode == KVQuantMode::Int8) {
            auto currentWeightInside = ROUND_UP(mMaxLength, lP8) * hP8;
            auto currentStride1 = currentWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
            auto currentStride0 = currentStride1 * UP_DIV(mHeadDim, hP8);

            auto prevWeightInside = ROUND_UP(oldMaxLength, lP8) * hP8;
            auto prevStride1 = prevWeightInside + 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES;
            auto prevStride0 = prevStride1 * UP_DIV(mHeadDim, hP8);

            for (int h = 0; h < mKvNumHead; ++h) {
                for (int d = 0; d < UP_DIV(mHeadDim, hP8); ++d) {
                    auto dstPtr = mMapValueAddr + h * currentStride0 + d * currentStride1;
                    auto srcPtr = prevValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                    // initialize 0 for weightInt8
                    memset(dstPtr, 0, currentWeightInside);
                    // copy inner side weightInt8
                    memcpy(dstPtr, srcPtr, prevWeightInside);
                    // copy hP8 scale&bias
                    memcpy(dstPtr + currentWeightInside, srcPtr + prevWeightInside, 2 * mConfig.mBlockNum * hP8 * QUANT_INFO_BYTES);
                }
            }
        } else {
            auto currentStride1 = ROUND_UP(mMaxLength, lP) * hP * mBytes;
            auto currentStride0 = ROUND_UP(mMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;

            auto prevStride1 = ROUND_UP(oldMaxLength, lP) * hP * mBytes;
            auto prevStride0 = ROUND_UP(oldMaxLength, lP) * hP * UP_DIV(mHeadDim, hP) * mBytes;
            for (int h = 0; h < mKvNumHead; ++h) {
                for (int d = 0; d < UP_DIV(mHeadDim, hP); ++d) {
                    auto dstPtr = mMapValueAddr + h * currentStride0 + d * currentStride1;
                    auto srcPtr = prevValue->host<int8_t>() + h * prevStride0 + d * prevStride1;

                    // initialize 0 for weight
                    if (lP > 1) {
                        memset(dstPtr, 0, currentStride1);
                    }
                    // copy inner side weight
                    memcpy(dstPtr, srcPtr, prevStride1);
                }
            }
        }
    }

    // Step 4: Release the temporary buffers
    mBackend->onReleaseBuffer(prevKey.get(), Backend::STATIC);
    mBackend->onReleaseBuffer(prevValue.get(), Backend::STATIC);
}

void CPUKVCacheManager::onResize(int kv_num_head, int head_dim) {
    mKvNumHead = kv_num_head;
    mHeadDim = head_dim;
    auto core  = static_cast<CPUBackend *>(mBackend)->functions();
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mBytes = core->bytes;
    mThreadNum = static_cast<CPUBackend *>(mBackend)->threadNumber();
    if (mThreadNum > mKvNumHead) {
        mThreadNum = mKvNumHead;
    }

    static_cast<CPUBackend *>(mBackend)->int8Functions()->MNNGetGemmUnit(&hP8, &lP8, &eP8);
    mQuantKeyFunc = core->MNNQuantAttentionKey;
    mQuantValueFunc = core->MNNQuantAttentionValue;

}

ErrorCode CPUKVCacheManager::onAllocPrefixSegments(KVMeta* meta, int seq_len, int layer_index) {
    mMeta = meta;
    if (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingReadSegments || mMeta->prefix_segments.empty()) {
        return INVALID_VALUE;
    }
    if (mKeyQuantMode != KVQuantMode::None || mValueQuantMode != KVQuantMode::None || !mUseFlashAttention) {
        MNN_ERROR("[Error]: PrefixAttention direct segment prefix cache only supports CPU flash attention without KV quantization\n");
        return NOT_SUPPORT;
    }

    int segmentTotalTokens = mMeta->segment_total_tokens;
    if (segmentTotalTokens <= 0) {
        for (const auto& segment : mMeta->prefix_segments) {
            segmentTotalTokens += segment.token_count;
        }
        mMeta->segment_total_tokens = segmentTotalTokens;
    }

    int resolvedLayerIndex = layer_index;
    if (resolvedLayerIndex < 0) {
        resolvedLayerIndex = mMeta->layer_index;
        if (mMeta->layer_nums > 0) {
            mMeta->layer_index = (mMeta->layer_index + 1) % mMeta->layer_nums;
        } else {
            mMeta->layer_index++;
        }
    }

    if (mMeta->remove > 0 || mMeta->n_reserve > 0) {
        MNN_ERROR("[Error]: PrefixAttention direct segment prefix cache does not support remove/reserve compaction yet\n");
        return NOT_SUPPORT;
    }
    int metaBasePast = static_cast<int>(mMeta->previous);
    int basePast = (metaBasePast > 0 && mPastLength > 0) ? mPastLength : metaBasePast;
    if (resolvedLayerIndex == 0 && metaBasePast != basePast) {
        MNN_PRINT("[CPUPrefixAttention] base_past adjusted to runtime cache length: meta=%d cache=%d\n",
                  metaBasePast, mPastLength);
    }
    if (basePast < 0 || mPastLength != basePast) {
        MNN_ERROR("[Error]: PrefixAttention direct segment base history mismatch: meta=%d cache=%d\n",
                  metaBasePast, mPastLength);
        return INVALID_VALUE;
    }

    int promptAppendLen = (int)mMeta->add;
    if (promptAppendLen <= 0) {
        promptAppendLen = seq_len;
    }
    RuntimeKVBlockTablePlan pagePlan;
    if (!buildPrefixRuntimeKVBlockTablePlan(pagePlan, mMeta, promptAppendLen)) {
        return INVALID_VALUE;
    }
    const int basePhysicalReserved = basePast > 0 ? ROUND_UP(basePast, pagePlan.pageSize()) : 0;
    const int logicalLength = basePast + pagePlan.logicalLength();
    const int physicalLength = basePhysicalReserved + pagePlan.physicalLength();
    if (logicalLength <= basePast + segmentTotalTokens || physicalLength < logicalLength) {
        MNN_ERROR("[Error]: invalid CPU PrefixAttention page plan, logical=%d physical=%d base=%d prefix=%d prompt=%d\n",
                  logicalLength, physicalLength, basePast, segmentTotalTokens, promptAppendLen);
        return INVALID_VALUE;
    }

    auto computeFloatFlashKVSize = [&]() {
        mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * mBytes;
        mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) *
            (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
    };

    if (basePast == 0) {
        mMaxLength = physicalLength + mConfig.mExpandChunk;
        setFlashAttentionUpperKv(MNN_FLASH_ATTENTION_BLOCK_SIZE);
        computeFloatFlashKVSize();
        size_t keySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
        size_t valueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;

        createKVCacheFile();
        resetKVCacheFileSize(keySize, valueSize);
        mmapKVCache(keySize, valueSize);
        if (mMapKeyAddr != nullptr) {
            ::memset(mMapKeyAddr, 0, keySize);
        }
        if (mMapValueAddr != nullptr) {
            ::memset(mMapValueAddr, 0, valueSize);
        }
        mKVCacheInDisk = true;
    } else if (physicalLength > mMaxLength) {
        int oldMaxLength = mMaxLength;
        size_t oldKeySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
        size_t oldValueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;
        mMaxLength = physicalLength + mConfig.mExpandChunk;
        setFlashAttentionUpperKv(MNN_FLASH_ATTENTION_BLOCK_SIZE);
        computeFloatFlashKVSize();
        size_t keySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
        size_t valueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;
        if (mKVCacheInDisk) {
            expandKVCacheInDisk(oldMaxLength, oldKeySize, oldValueSize, keySize, valueSize);
        } else {
            expandKVCacheInMem(oldMaxLength);
        }
    }

    mPagedPrefixActive = true;
    mPagedPrefixPageSize = pagePlan.pageSize();
    mPagedPrefixLogicalLength = logicalLength;
    mPagedPrefixPhysicalLength = physicalLength;
    mPagedPrefixLogicalToPhysical.assign(logicalLength, -1);
    mPagedPrefixKeyNeedsRoPE.assign(logicalLength, 0);
    mPagedPrefixRopeDim.assign(logicalLength, 0);
    mPagedPrefixRopeTheta.assign(logicalLength, 0.0f);
    mPagedPrefixRopePairing.assign(logicalLength, KVMeta::RopePairingHalf);
    for (int i = 0; i < basePast; ++i) {
        mPagedPrefixLogicalToPhysical[i] = i;
    }
    for (const auto& ref : pagePlan.refs()) {
        const KVMeta::PrefixSegment* segment = nullptr;
        if (ref.source == RuntimeKVBlockRef::Document) {
            if (ref.segmentIndex < 0 || ref.segmentIndex >= (int)mMeta->prefix_segments.size()) {
                return INVALID_VALUE;
            }
            segment = &mMeta->prefix_segments[ref.segmentIndex];
        }
        for (int i = 0; i < ref.tokenCount; ++i) {
            int logical = basePast + ref.logicalTokenStart + i;
            int physical = basePhysicalReserved + ref.physicalTokenStart + i;
            if (logical < 0 || logical >= logicalLength || physical < 0 || physical >= physicalLength) {
                return INVALID_VALUE;
            }
            mPagedPrefixLogicalToPhysical[logical] = physical;
            if (segment != nullptr) {
                mPagedPrefixKeyNeedsRoPE[logical] = 1;
                mPagedPrefixRopeDim[logical] = segment->rope_dim;
                mPagedPrefixRopeTheta[logical] = segment->rope_theta;
                mPagedPrefixRopePairing[logical] = segment->rope_pairing;
            }
        }
    }
    for (int i = 0; i < logicalLength; ++i) {
        if (mPagedPrefixLogicalToPhysical[i] < 0) {
            MNN_ERROR("[Error]: CPU PrefixAttention page table has unmapped logical token %d/%d\n", i, logicalLength);
            return INVALID_VALUE;
        }
    }

    int dstStart = basePast;
    for (const auto& segment : mMeta->prefix_segments) {
        if (segment.token_count <= 0) {
            continue;
        }
        if (segment.key_rope_state != KVMeta::KeyRopeCanonicalNoRope) {
            MNN_ERROR("[Error]: PrefixAttention direct segment prefix cache requires canonical_no_rope key: %s\n",
                      segment.cache_name.c_str());
            return NOT_SUPPORT;
        }
        if (segment.backend != "cpu" || segment.layout != kCpuPrefixLayout) {
            MNN_ERROR("[Error]: CPU PrefixAttention requires backend-native CPU cache for %s, got backend=%s layout=%s\n",
                      segment.cache_name.c_str(), segment.backend.c_str(), segment.layout.c_str());
            return NOT_SUPPORT;
        }

        std::string base = prefixCacheLayerBase(mConfig.mPrefixCacheDir, "cpu", segment.cache_name, resolvedLayerIndex);
        std::string pathk = base + ".k";
        std::string pathv = base + ".v";
        auto keyRead = readPrefixKVHostCacheFile(pathk);
        auto valueRead = readPrefixKVHostCacheFile(pathv);
        if (!keyRead.ok || !valueRead.ok || keyRead.bytes == nullptr || valueRead.bytes == nullptr) {
            MNN_ERROR("[Error]: Failed to read segment prefix cache files: %s / %s, key_error=%s value_error=%s\n",
                      pathk.c_str(), pathv.c_str(), keyRead.error.c_str(), valueRead.error.c_str());
            return FILE_OPEN_FAILED;
        }

        auto srcKeySize = keyRead.bytes->size();
        auto srcValueSize = valueRead.bytes->size();
        size_t srcKeyCapacity = srcKeySize / ((size_t)mKvNumHead * ROUND_UP(mHeadDim, lP) * mBytes);
        size_t srcValueCapacity = srcValueSize / ((size_t)mKvNumHead * ROUND_UP(mHeadDim, hP) * mBytes);
        size_t srcCapacity = ALIMIN(srcKeyCapacity, srcValueCapacity);
        if ((size_t)segment.token_count > srcCapacity) {
            MNN_ERROR("[Error]: Segment prefix token_count %d exceeds KV file capacity %zu for cache %s layer %d\n",
                      segment.token_count, srcCapacity, segment.cache_name.c_str(), resolvedLayerIndex);
            return INVALID_VALUE;
        }

        auto srcKeyAddr = reinterpret_cast<const int8_t*>(keyRead.bytes->data());
        auto srcValueAddr = reinterpret_cast<const int8_t*>(valueRead.bytes->data());

        double materializeStart = nowMs();
        copyPrefixSegmentKV(srcKeyAddr, srcKeySize / mKvNumHead, srcValueAddr, srcValueSize / mKvNumHead,
                            segment.token_count, dstStart);
        double materializeMs = nowMs() - materializeStart;
        dstStart += segment.token_count;
        const double waitMs = mMeta->prefix_device_prefetch ?
            (keyRead.prefetch_wait_ms + valueRead.prefetch_wait_ms) : 0.0;

        MNN_PRINT("[CPUPrefixAttention] layer=%d cache=%s tokens=%d host_cache_hit=%d device_cache_hit=0 device_prefetch=%d device_prefetch_hit=%d prefetch_wait_ms=%.3f disk_read_ms=%.3f host_to_device_ms=0.000 materialize_ms=%.3f attention_wait_ms=%.3f bytes=%zu host_cache_threads=%d\n",
                  resolvedLayerIndex, segment.cache_name.c_str(), segment.token_count,
                  (keyRead.host_cache_hit && valueRead.host_cache_hit) ? 1 : 0,
                  mMeta->prefix_device_prefetch ? 1 : 0,
                  (mMeta->prefix_device_prefetch && keyRead.host_cache_hit && valueRead.host_cache_hit) ? 1 : 0,
                  keyRead.prefetch_wait_ms + valueRead.prefetch_wait_ms,
                  keyRead.disk_read_ms + valueRead.disk_read_ms,
                  materializeMs,
                  waitMs,
                  keyRead.file_size + valueRead.file_size,
                  prefixKVHostCacheWorkerCount());
    }

    mPastLength = dstStart;
    if (mPastLength != basePast + segmentTotalTokens) {
        MNN_ERROR("[Error]: PrefixAttention direct segment prefix copied length %d does not match expected length %d\n",
                  mPastLength - basePast, segmentTotalTokens);
        return INVALID_VALUE;
    }
    return NO_ERROR;
}

void CPUKVCacheManager::onAlloc(KVMeta* meta, int seq_len) {
    mMeta = meta;

    // load disk prefix kvcache
    if(mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingRead) {
        // create new files
        auto base = prefixCacheLayerBase(mConfig.mPrefixCacheDir, "cpu", mMeta->file_name, mMeta->layer_index);
        std::string pathk    = base + ".k";
        std::string pathv    = base + ".v";
        mMeta->layer_index++;
        mMeta->layer_index = mMeta->layer_index % mMeta->layer_nums;
        auto old_key_fd   = MNNOpenFile(pathk.c_str(), MNN_FILE_WRITE);
        auto old_value_fd = MNNOpenFile(pathv.c_str(), MNN_FILE_WRITE);
        if (old_key_fd == INVALID_FILE) {
            MNN_PRINT("Failed to open the file: %s\n", pathk.c_str());
        }
        if (old_value_fd == INVALID_FILE) {
            MNN_PRINT("Failed to open the file: %s\n", pathv.c_str());
        }

        // get kv cache file info
        auto oldKeySize = MNNGetFileSize(old_key_fd);
        auto oldValueSize = MNNGetFileSize(old_value_fd);

        size_t oldMaxLength = 0;
        if (mKeyQuantMode != KVQuantMode::None || mValueQuantMode != KVQuantMode::None) {
            MNN_ERROR("[Error]: Currently, kvcache save in disk not support quantized key/value\n");
        } else {
            size_t oldKeyMaxLength = oldKeySize / (mKvNumHead * ROUND_UP(mHeadDim, lP) * mBytes);
            size_t oldValueMaxLength = oldValueSize / (mKvNumHead * ROUND_UP(mHeadDim, hP) * mBytes);
            oldMaxLength = ALIMIN(oldKeyMaxLength, oldValueMaxLength);
        }
        if(oldMaxLength < meta->seqlen_in_disk) {
            MNN_ERROR("[Error]: Kvcache in disk size smaller than saved lengthInDiskToload:%d\n", (int)meta->seqlen_in_disk);
        }

        // Update mMaxLength first, then setFlashAttentionUpperKv to avoid division by zero
        int kv_seq_len = meta->add + meta->seqlen_in_disk;
        mMaxLength = kv_seq_len > oldMaxLength ? kv_seq_len + mConfig.mExpandChunk : oldMaxLength;
        if (mUseFlashAttention) {
            setFlashAttentionUpperKv(MNN_FLASH_ATTENTION_BLOCK_SIZE);
        } else {
            setFlashAttentionUpperKv(mMaxLength);
        }
        size_t keySize = (size_t)mKvNumHead * ROUND_UP(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * mBytes;
        size_t valueSize = (size_t)mKvNumHead * UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);

        keySize = ALIMAX(keySize, oldKeySize);
        valueSize = ALIMAX(valueSize, oldValueSize);

        if (mKeyQuantMode == KVQuantMode::TQ3) {
            int tq3BytesPerSeq = (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
            mCurrentKeySizePerHead = (size_t)mMaxLength * tq3BytesPerSeq;
        } else if (mKeyQuantMode == KVQuantMode::TQ4) {
            int tq4BytesPerSeq = (mHeadDim / TQ4_BLOCK_SIZE) * TQ4_BYTES_PER_BLOCK;
            mCurrentKeySizePerHead = (size_t)mMaxLength * tq4BytesPerSeq;
        } else if (mKeyQuantMode == KVQuantMode::Int8) {
            mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP8) * ROUND_UP(mHeadDim, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mMaxLength, hP8);
        } else {
            mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * mBytes;
        }
        if (mValueQuantMode == KVQuantMode::TQ3) {
            int tq3BytesPerSeq = (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
            mCurrentValueSizePerHead = (size_t)mMaxLength * tq3BytesPerSeq;
        } else if (mValueQuantMode == KVQuantMode::TQ4) {
            int tq4BytesPerSeq = (mHeadDim / TQ4_BLOCK_SIZE) * TQ4_BYTES_PER_BLOCK;
            mCurrentValueSizePerHead = (size_t)mMaxLength * tq4BytesPerSeq;
        } else if (mValueQuantMode == KVQuantMode::Int8) {
            mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mFlashAttentionUpperKv, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mHeadDim, hP8));
        } else {
            mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
        }

        createKVCacheFile();
        resetKVCacheFileSize(keySize, valueSize);
        expandKVCacheInDisk(oldMaxLength, oldKeySize, oldValueSize, keySize, valueSize, old_key_fd, old_value_fd);
        mPastLength = meta->seqlen_in_disk;
        mKVCacheInDisk = true;
        if (mMeta->key_rope_state == KVMeta::KeyRopeCanonicalNoRope) {
            // 单 prefix cache 若保存为 canonical_no_rope，加载后也按原始 prefix 位置重新编码。
            if (mBytes == 2) {
                applyPackedKeyRoPE<FLOAT16_T>(0, mPastLength, 0, mMeta->rope_dim, mMeta->rope_theta, mMeta->rope_pairing, false);
            } else {
                applyPackedKeyRoPE<float>(0, mPastLength, 0, mMeta->rope_dim, mMeta->rope_theta, mMeta->rope_pairing, false);
            }
        }

        return;
    }

    // Do not use mMeta->add, because in VL models or Qnn case, mMeta->add is 0 or mMeta is nullptr.
    int kv_seq_len = seq_len;
    mMaxLength = kv_seq_len + mConfig.mExpandChunk;
    if (mUseFlashAttention) {
        setFlashAttentionUpperKv(MNN_FLASH_ATTENTION_BLOCK_SIZE);
    } else {
        setFlashAttentionUpperKv(mMaxLength);
    }

    // 1. compute size
    if (mKeyQuantMode == KVQuantMode::TQ3) {
        mCurrentKeySizePerHead = (size_t)mMaxLength * (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
    } else if (mKeyQuantMode == KVQuantMode::TQ4) {
        mCurrentKeySizePerHead = (size_t)mMaxLength * (mHeadDim / TQ4_BLOCK_SIZE) * TQ4_BYTES_PER_BLOCK;
    } else if (mKeyQuantMode == KVQuantMode::Int8) {
        mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP8) * ROUND_UP(mHeadDim, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mMaxLength, hP8);
    } else {
        mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * mBytes;
    }
    if (mValueQuantMode == KVQuantMode::TQ3) {
        mCurrentValueSizePerHead = (size_t)mMaxLength * (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
    } else if (mValueQuantMode == KVQuantMode::TQ4) {
        mCurrentValueSizePerHead = (size_t)mMaxLength * (mHeadDim / TQ4_BLOCK_SIZE) * TQ4_BYTES_PER_BLOCK;
    } else if (mValueQuantMode == KVQuantMode::Int8) {
        mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mFlashAttentionUpperKv, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mHeadDim, hP8));
    } else {
        mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
    }
    size_t keySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
    size_t valueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;

    // 2. allocate buffer

    // case1: key&value size exceeds the limited size
    // case2: multi prompts share a common prefix kv cache info
    bool storeKvInDisk  = !mConfig.mKVCacheDir.empty();
    bool sharePrefixKv = mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingWrite;

    if (sharePrefixKv) {
        mSaveShareKvPrefix = true;
        if(!ensurePrefixCacheObjectDirs(mConfig.mPrefixCacheDir, "cpu", mMeta->file_name)) {
            MNN_PRINT("Failed to create prefix cache file dir: %s\n", mConfig.mPrefixCacheDir.c_str());
        }
    }
    if (storeKvInDisk || sharePrefixKv) { // store kv in disk
        std::string keyStoredDst = "";
        std::string valueStoredDst = "";
        if(mMeta != nullptr) {
            mBasePrefixFileName = prefixCacheLayerBase(mConfig.mPrefixCacheDir, "cpu", mMeta->file_name, mMeta->layer_index);
            keyStoredDst = sharePrefixKv ? mBasePrefixFileName + ".k" : "";
            valueStoredDst = sharePrefixKv ? mBasePrefixFileName + ".v" : "";
            mMeta->layer_index++;
            mMeta->layer_index = mMeta->layer_index % mMeta->layer_nums;
        }
        createKVCacheFile(keyStoredDst, valueStoredDst);
        resetKVCacheFileSize(keySize, valueSize);
        mmapKVCache(keySize, valueSize);
        mKVCacheInDisk = true;
    } else { // store kv in memory
        mPastKey.reset(Tensor::createDevice<int8_t>({mKvNumHead, (int)mCurrentKeySizePerHead}));
        mPastValue.reset(Tensor::createDevice<int8_t>({mKvNumHead, (int)mCurrentValueSizePerHead}));

        mBackend->onAcquireBuffer(mPastKey.get(), Backend::STATIC);
        mBackend->onAcquireBuffer(mPastValue.get(), Backend::STATIC);

        // initilize 0
        if (mHeadDim % lP || mKeyQuantMode != KVQuantMode::None) {
            memset(mPastKey->host<int8_t>(), 0, mPastKey->length(0) * mPastKey->stride(0));
        }
        if (lP > 1 || mValueQuantMode != KVQuantMode::None) {
            memset(mPastValue->host<int8_t>(), 0, mPastValue->length(0) * mPastValue->stride(0));
        }
    }
    // scale, zero point and sum of key for quantization
    if (mKeyQuantMode == KVQuantMode::Int8) { // quant K
        mKeySum.reset(Tensor::createDevice<int8_t>({mKvNumHead, ROUND_UP(mMaxLength, hP8) * QUANT_INFO_BYTES}));
        mKeyMax.reset(Tensor::createDevice<int8_t>({mKvNumHead, mHeadDim * QUANT_INFO_BYTES}));
        mBackend->onAcquireBuffer(mKeySum.get(), Backend::STATIC);
        mBackend->onAcquireBuffer(mKeyMax.get(), Backend::STATIC);

        for (int ks = 0; ks < mKvNumHead * mHeadDim; ++ks) {
            mKeyMax->host<float>()[ks] = std::numeric_limits<float>::lowest();
        }
        if (mBytes == 2) {
            auto core = static_cast<CPUBackend*>(mBackend)->functions();
            core->MNNFp32ToLowp(mKeyMax->host<float>(), (int16_t*)(mKeyMax->host<float>()), mKvNumHead * mHeadDim);
        }
    }
    if (mValueQuantMode == KVQuantMode::Int8) {
        mValueSum.reset(Tensor::createDevice<int8_t>({mKvNumHead, (int)UP_DIV(mMaxLength, mFlashAttentionUpperKv), ROUND_UP(mHeadDim, hP8) * QUANT_INFO_BYTES}));
        mBackend->onAcquireBuffer(mValueSum.get(), Backend::STATIC);
        memset(mValueSum->host<int8_t>(), 0, mValueSum->stride(0) * mValueSum->length(0));
    }
}

void CPUKVCacheManager::onRealloc(KVMeta* meta) {
    auto kv_seq_len = meta->previous + meta->add - meta->remove + meta->computeReverseSize();
    if (kv_seq_len > mMaxLength) {
        // Realloc
        int oldMaxLength = mMaxLength;
        mMaxLength = (int)kv_seq_len + mConfig.mExpandChunk;
        if (mUseFlashAttention) {
            setFlashAttentionUpperKv(MNN_FLASH_ATTENTION_BLOCK_SIZE);
        } else {
            setFlashAttentionUpperKv(mMaxLength);
        }
        size_t oldKeySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
        size_t oldValueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;

        // update current key size per head
        if (mKeyQuantMode == KVQuantMode::TQ3) {
            mCurrentKeySizePerHead = (size_t)mMaxLength * (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
        } else if (mKeyQuantMode == KVQuantMode::TQ4) {
            mCurrentKeySizePerHead = (size_t)mMaxLength * (mHeadDim / TQ4_BLOCK_SIZE) * TQ4_BYTES_PER_BLOCK;
        } else if (mKeyQuantMode == KVQuantMode::Int8) {
            mCurrentKeySizePerHead = ROUND_UP(mMaxLength, hP8) * ROUND_UP(mHeadDim, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mMaxLength, hP8);
        } else {
            mCurrentKeySizePerHead = UP_DIV(mMaxLength, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
        }
        // update current value size per head
        if (mValueQuantMode == KVQuantMode::TQ3) {
            mCurrentValueSizePerHead = (size_t)mMaxLength * (mHeadDim / TQ3_BLOCK_SIZE) * TQ3_BYTES_PER_BLOCK;
        } else if (mValueQuantMode == KVQuantMode::TQ4) {
            mCurrentValueSizePerHead = (size_t)mMaxLength * (mHeadDim / TQ4_BLOCK_SIZE) * TQ4_BYTES_PER_BLOCK;
        } else if (mValueQuantMode == KVQuantMode::Int8) {
            mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mFlashAttentionUpperKv, lP8) + 2 * QUANT_INFO_BYTES * mConfig.mBlockNum * ROUND_UP(mHeadDim, hP8));
        } else {
            mCurrentValueSizePerHead = UP_DIV(mMaxLength, mFlashAttentionUpperKv) * (ROUND_UP(mHeadDim, hP) * ROUND_UP(mFlashAttentionUpperKv, lP) * mBytes);
        }
        size_t keySize = (size_t)mKvNumHead * mCurrentKeySizePerHead;
        size_t valueSize = (size_t)mKvNumHead * mCurrentValueSizePerHead;

        /*==== No limit for kvcache ====*/
        if (mKVCacheInDisk == false) {
            expandKVCacheInMem(oldMaxLength);
        } else {
            expandKVCacheInDisk(oldMaxLength, oldKeySize, oldValueSize, keySize, valueSize);
        }
        /* No matter where is the kvcache, the scales and zero points are always in memory, since their size is very small */
        if (mKeyQuantMode == KVQuantMode::Int8) {
            auto newKeySumTensor = Tensor::createDevice<int32_t>({mKvNumHead, UP_DIV(mMaxLength, hP8), hP8});
            mBackend->onAcquireBuffer(newKeySumTensor, Backend::STATIC);
            for (int h = 0; h < mKvNumHead; h++) {
                memcpy(newKeySumTensor->host<int8_t>() + h * UP_DIV(mMaxLength, hP8) * hP8 * 4, mKeySum->host<int8_t>() + h * UP_DIV(oldMaxLength, hP8) * hP8 * 4, UP_DIV(oldMaxLength, hP8) * hP8 * 4);
            }
            mKeySum.reset(newKeySumTensor);
        }
        if (mValueQuantMode == KVQuantMode::Int8) {
            auto newValueSumTensor = Tensor::createDevice<int8_t>({mKvNumHead, (int)UP_DIV(mMaxLength, mFlashAttentionUpperKv), ROUND_UP(mHeadDim, hP8) * QUANT_INFO_BYTES});
            mBackend->onAcquireBuffer(newValueSumTensor, Backend::STATIC);
            auto remainSizePerHead = mValueSum->stride(0);
            auto increSizePerHead = newValueSumTensor->stride(0) - mValueSum->stride(0);
            for (int h = 0; h < mKvNumHead; ++h) {
                memcpy(newValueSumTensor->host<int8_t>() + h * newValueSumTensor->stride(0) , mValueSum->host<int8_t>() + h * mValueSum->stride(0), remainSizePerHead);
                // memset 0
                if (increSizePerHead > 0) {
                    memset(newValueSumTensor->host<int8_t>() + h * newValueSumTensor->stride(0) + remainSizePerHead, 0, increSizePerHead);
                }
            }
            mValueSum.reset(newValueSumTensor);
        }
    }
    // Remove
    auto start = mPastLength - meta->remove;
    if (0 == meta->n_reserve || mKeyQuantMode != KVQuantMode::None ||
        mValueQuantMode != KVQuantMode::None) { // n_reserve > 0 is not currently supported when K or V is quantized.
        mPastLength = start;
        return;
    }
#if 1
    auto dstIndex = start;
    for (int n = 0; n < meta->n_reserve; ++n) {
        auto begin = meta->reserve[2 * n];
        auto size  = meta->reserve[2 * n + 1];
        auto srcIndex = start + begin;
        if (mBytes == 2) {
            moveKV<FLOAT16_T>(srcIndex, dstIndex, size);
        } else {
            moveKV<float>(srcIndex, dstIndex, size);
        }
        dstIndex += size;
    }
    mPastLength = dstIndex;
#else
    // Don't support not align reserve
    auto align = hP;
    auto dstStart = start;
    auto lastValidSrcEnd = start;
    for (int n=0; n<meta->n_reserve; ++n) {
        auto lastEndAlign = UP_DIV(lastValidSrcEnd, align) * align;
        auto begin = meta->reserve[2 * n];
        auto size = meta->reserve[2 * n + 1];
        auto startAlign = ((begin + start) / align) * align;
        if (startAlign <= lastEndAlign) {
            // Fullly reserve
            dstStart = dstStart + size;
            lastValidSrcEnd = begin + start + size;
            continue;
        }
        auto end = begin + start + size;
        auto endAlign = UP_DIV(end, align) * align;

        auto sizeUnit = (endAlign - startAlign) / align;
        auto dstStartAlign = UP_DIV(dstStart, align) * align;

        //TODO: Support Quant
//        mPastKey.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mMaxLength, hP), mHeadDim, hP}));

        // Move K
        auto keyStride = UP_DIV(mMaxLength, align) * align * ROUND_UP(mHeadDim, lP);
        auto dstKAddr = keyAddr() + dstStartAlign * ROUND_UP(mHeadDim, lP) * mBytes;
        auto srcKAddr = keyAddr() + startAlign * ROUND_UP(mHeadDim, lP) * mBytes;
        for (int i=0; i<mKvNumHead; ++i) {
            auto dst = dstKAddr + i * keyStride * mBytes;
            auto src = srcKAddr + i * keyStride * mBytes;
            for (int j=0; j<sizeUnit; ++j) {
                ::memcpy(dst + j * align * ROUND_UP(mHeadDim, lP) * mBytes, src + j * align * ROUND_UP(mHeadDim, lP) * mBytes, align * ROUND_UP(mHeadDim, lP) * mBytes);
            }
        }


        // Move V
        auto dstVAddr = valudAddr() + dstStartAlign * align * mBytes;
        auto srcVAddr = valudAddr() + startAlign * align * mBytes;
        auto number = mKvNumHead * UP_DIV(mHeadDim, align);
        for (int i=0; i<number; ++i) {
            auto dst = dstVAddr + i * ROUND_UP(mMaxLength, lP) * align * mBytes;
            auto src = srcVAddr + i * ROUND_UP(mMaxLength, lP) * align * mBytes;
            for (int j=0; j<sizeUnit; ++j) {
                ::memcpy(dst + j * align * align * mBytes, src + j * align * align * mBytes, align * align * mBytes);
            }
        }
        dstStart = dstStart + size;
        lastValidSrcEnd = begin + start + size;
    }
    mPastLength = dstStart;
#endif
}

void CPUKVCacheManager::saveKVCacheInDisk() {
    // get original kv cache info
    auto keySize = MNNGetFileSize(mKeyCacheFD);
    auto valueSize = MNNGetFileSize(mValueCacheFD);
    mmapKVCache(keySize, valueSize);
    if(!ensurePrefixCacheObjectDirs(mConfig.mPrefixCacheDir, "cpu", mMeta->file_name)) {
        MNN_PRINT("Failed to create prefix cache file dir: %s\n", mConfig.mPrefixCacheDir.c_str());
    }

    // create new files
    auto base = prefixCacheLayerBase(mConfig.mPrefixCacheDir, "cpu", mMeta->file_name, mMeta->layer_index);
    std::string pathk    = base + ".k";
    std::string pathv    = base + ".v";
    mMeta->layer_index++;
    mMeta->layer_index = mMeta->layer_index % mMeta->layer_nums;

    auto new_key_fd   = MNNCreateFile(pathk.c_str());
    auto new_value_fd = MNNCreateFile(pathv.c_str());
    if (new_key_fd == INVALID_FILE) {
        MNN_PRINT("Failed to create the file: %s\n", pathk.c_str());
    }
    if (new_value_fd == INVALID_FILE) {
        MNN_PRINT("Failed to create the file: %s\n", pathv.c_str());
    }
    // set new file size
    if (MNNSetFileSize(new_key_fd, keySize) != MNN::NO_ERROR || MNNSetFileSize(new_value_fd, valueSize) != MNN::NO_ERROR) {
        MNN_PRINT("Failed to resize the kvcache files!\n");
    }
    // mmap files
    int8_t* mMapNewKeyAddr = (int8_t *)MNNMmapFile(new_key_fd, keySize);
    if (mMapNewKeyAddr == nullptr) {
        MNN_PRINT("Failed to memory-map the new kvcache!\n");
    }
    int8_t* mMapNewValueAddr =(int8_t *)MNNMmapFile(new_value_fd, valueSize);
    if (mMapNewValueAddr == nullptr) {
        MNN_PRINT("Failed to memory-map the kvcache!\n");
    }

    // copy
    memcpy(mMapNewKeyAddr,   mMapKeyAddr,   keySize);
    memcpy(mMapNewValueAddr, mMapValueAddr, valueSize);

    // unmap new files
    if (mMapNewKeyAddr != nullptr) {
        MNNUnmapFile(mMapNewKeyAddr, keySize);
        mMapNewKeyAddr = nullptr;
    }
    if (mMapNewValueAddr != nullptr) {
        MNNUnmapFile(mMapNewValueAddr, valueSize);
        mMapNewValueAddr = nullptr;
    }
    // close file
    if (new_key_fd != INVALID_FILE) {
        MNNCloseFile(new_key_fd);
        new_key_fd = INVALID_FILE;
    }
    if (new_value_fd != INVALID_FILE) {
        MNNCloseFile(new_value_fd);
        new_value_fd = INVALID_FILE;
    }
}

void CPUKVCacheManager::onClear() {
    if (mKVCacheInDisk) {
        // mSaveShareKvPrefix 表示这份 mmap 文件就是可复用 prefix cache。
        // 清理运行时状态时只能解除 mmap，不能删除 .cache/prefixcache 下的正式文件。
        unmapKVCache(mCurrentKeySizePerHead * (size_t)mKvNumHead, mCurrentValueSizePerHead * (size_t)mKvNumHead);
        if(!mSaveShareKvPrefix) {
            // 普通推理/读取 prefix 时创建的是临时运行时 KV 文件，可以随 cache 一起删除。
            removeKVCacheFile();
        }
        mKVCacheInDisk = false;
    }
    mSaveShareKvPrefix = false;
    mPastKey.reset();
    mPastValue.reset();
    mKeySum.reset();
    mKeyMax.reset();
    mValueSum.reset();
    mMaxLength = mPastLength = 0;
    mPagedPrefixActive = false;
    mPagedPrefixLogicalLength = 0;
    mPagedPrefixPhysicalLength = 0;
    mPagedPrefixLogicalToPhysical.clear();
    mPagedPrefixKeyNeedsRoPE.clear();
    mPagedPrefixRopeDim.clear();
    mPagedPrefixRopeTheta.clear();
    mPagedPrefixRopePairing.clear();
}

template <typename T>
void CPUKVCacheManager::ProcessKey(const Tensor* key, int seqLen, int kvHead) {
    if ((mKeyQuantMode == KVQuantMode::TQ3) || (mKeyQuantMode == KVQuantMode::TQ4)) {
        int bytesPerBlock = (mKeyQuantMode == KVQuantMode::TQ3) ? TQ3_BYTES_PER_BLOCK : TQ4_BYTES_PER_BLOCK;
        int blockSize = (mKeyQuantMode == KVQuantMode::TQ3) ? TQ3_BLOCK_SIZE : TQ4_BLOCK_SIZE;
        int bytesPerSeq = (mHeadDim / blockSize) * bytesPerBlock;
        uint8_t* keyDst = (uint8_t*)addrOfKey(kvHead);
        for (int i = 0; i < seqLen; i++) {
            T* src = key->host<T>() + i * mKvNumHead * mHeadDim + kvHead * mHeadDim;
            uint8_t* dst = keyDst + (mPastLength + i) * bytesPerSeq;
            float block[TQ4_BLOCK_SIZE]; // TQ4_BLOCK_SIZE >= TQ3_BLOCK_SIZE
            for (int b = 0; b < mHeadDim / blockSize; b++) {
                for (int k = 0; k < blockSize; k++) {
                    block[k] = (float)src[b * blockSize + k];
                }
                if (mKeyQuantMode == KVQuantMode::TQ3) {
                    tq3_quantize_block(dst + b * bytesPerBlock, block);
                } else {
                    tq4_quantize_block(dst + b * bytesPerBlock, block);
                }
            }
        }
    } else if (mKeyQuantMode ==
               KVQuantMode::Int8) { // [seqLen, headDim] -> [maxlen/hP8, blockNum, (headDim/blockNum)/lP8, hP8, lP8]
        int8_t * keyDst = reinterpret_cast<int8_t*>(addrOfKey(kvHead));
        float * sumDst = reinterpret_cast<float*>(addrOfKeySum(kvHead));

        auto blockL = UP_DIV(mHeadDim, mConfig.mBlockNum);
        auto weightStride1 = ROUND_UP(blockL, lP8) * hP8;
        auto weightStride2 = lP8 * hP8;
        auto packedWeightStride1 = weightStride1 + 2 * QUANT_INFO_BYTES * hP8;

        T* keyMax = reinterpret_cast<T*>(addrOfKeyMax(kvHead));
        int32_t params[] = {mKvNumHead, seqLen, mHeadDim, mConfig.mBlockNum, eP8, lP8, hP8, mPastLength, kvHead};
        mQuantKeyFunc(keyDst, key->host<float>(), sumDst, (float*)keyMax, params);
    } else { // target: [maxlen/hP, headdim/lP, hP, lP]
        T * key_dst = reinterpret_cast<T*>(addrOfKey(kvHead));
        for (int i = 0; i < seqLen; i++) {
            T * key_src = key->host<T>() + i * mKvNumHead * mHeadDim + kvHead * mHeadDim;
            for (int j = 0; j < mHeadDim; j++) {
                key_dst[keyIndex(mPastLength + i, j)] = key_src[j];
            }
        }
    }
}

template <typename T>
void CPUKVCacheManager::ProcessValue(const Tensor* value, int seqLen, int kvHead) { // [headdim/hP, maxlen, hP]
    if ((mValueQuantMode == KVQuantMode::TQ3) || (mValueQuantMode == KVQuantMode::TQ4)) {
        int bytesPerBlock = (mValueQuantMode == KVQuantMode::TQ3) ? TQ3_BYTES_PER_BLOCK : TQ4_BYTES_PER_BLOCK;
        int blockSize = (mValueQuantMode == KVQuantMode::TQ3) ? TQ3_BLOCK_SIZE : TQ4_BLOCK_SIZE;
        int bytesPerSeq = (mHeadDim / blockSize) * bytesPerBlock;
        uint8_t* valueDst = (uint8_t*)addrOfValue(kvHead);
        for (int i = 0; i < seqLen; i++) {
            T* src = value->host<T>() + i * mKvNumHead * mHeadDim + kvHead * mHeadDim;
            uint8_t* dst = valueDst + (mPastLength + i) * bytesPerSeq;
            float block[TQ4_BLOCK_SIZE];
            for (int b = 0; b < mHeadDim / blockSize; b++) {
                for (int k = 0; k < blockSize; k++) {
                    block[k] = (float)src[b * blockSize + k];
                }
                if (mValueQuantMode == KVQuantMode::TQ3) {
                    tq3_quantize_block(dst + b * bytesPerBlock, block);
                } else {
                    tq4_quantize_block(dst + b * bytesPerBlock, block);
                }
            }
        }
    } else if (mValueQuantMode == KVQuantMode::Int8) {
        int8_t* valueDst = reinterpret_cast<int8_t*>(addrOfValue(kvHead));
        float* valueSum = reinterpret_cast<float*>(addrOfValueSum(kvHead));

        int32_t params[] = {mKvNumHead, seqLen, mHeadDim, mConfig.mBlockNum, mMaxLength, lP8, hP8, mPastLength, kvHead, (int32_t)mFlashAttentionUpperKv};
        mQuantValueFunc(valueDst, value->host<float>(), valueSum, params);
    } else {
        T * value_dst = reinterpret_cast<T*>(addrOfValue(kvHead));
        for (int i = 0; i < seqLen; i++) {
            T * value_src = value->host<T>() + i * mKvNumHead * mHeadDim + kvHead * mHeadDim;
            for (int j = 0; j < mHeadDim; j++) {
                value_dst[flashValueIndex(mPastLength + i, j)] = value_src[j];
            }
        }
    }
}

int CPUKVCacheManager::physicalSlotForLogical(int seq) const {
    if (mPagedPrefixActive && seq >= 0 && seq < (int)mPagedPrefixLogicalToPhysical.size()) {
        int physical = mPagedPrefixLogicalToPhysical[seq];
        if (physical >= 0) {
            return physical;
        }
    }
    return seq;
}

size_t CPUKVCacheManager::keyIndexPhysical(int seq, int dim) const {
    return (seq / hP) * ROUND_UP(mHeadDim, lP) * hP +
           (dim / lP) * hP * lP +
           (seq % hP) * lP +
           (dim % lP);
}

size_t CPUKVCacheManager::valueIndexPhysical(int seq, int dim) const {
    return (dim / hP) * ROUND_UP(mMaxLength, lP) * hP +
           (seq / lP) * hP * lP +
           (dim % hP) * lP +
           (seq % lP);
}

size_t CPUKVCacheManager::flashValueIndexPhysical(int seq, int dim) const {
    auto weightStride2 = lP * hP;
    auto weightStride1 = UP_DIV((int32_t)mFlashAttentionUpperKv, lP) * weightStride2;
    auto weightStride0 = weightStride1 * UP_DIV(mHeadDim, hP);
    auto idxInner = (seq / (int32_t)mFlashAttentionUpperKv) * weightStride0 +
                    (seq % (int32_t)mFlashAttentionUpperKv) / lP * weightStride2 +
                    (seq % (int32_t)mFlashAttentionUpperKv) % lP;
    auto idxBase = (dim / hP) * weightStride1 + (dim % hP) * lP;
    return idxBase + idxInner;
}

size_t CPUKVCacheManager::keyIndex(int seq, int dim) const {
    return keyIndexPhysical(physicalSlotForLogical(seq), dim);
}

size_t CPUKVCacheManager::valueIndex(int seq, int dim) const {
    return valueIndexPhysical(physicalSlotForLogical(seq), dim);
}

size_t CPUKVCacheManager::flashValueIndex(int seq, int dim) const {
    return flashValueIndexPhysical(physicalSlotForLogical(seq), dim);
}

size_t CPUKVCacheManager::packedKeyIndex(int seq, int dim) const {
    return keyIndex(seq, dim);
}

size_t CPUKVCacheManager::packedFlashValueIndex(int seq, int dim) const {
    return flashValueIndex(seq, dim);
}

template <typename T>
void CPUKVCacheManager::applyPackedKeyRoPE(int start, int len, int positionBase, int ropeDim, float ropeTheta,
                                           int ropePairing, bool inverse) {
    if (len <= 0) {
        return;
    }
    if (mKeyQuantMode != KVQuantMode::None) {
        MNN_ERROR("[Error]: prefix key RoPE canonicalization does not support KV quantization\n");
        return;
    }
    if (ropePairing != KVMeta::RopePairingHalf) {
        MNN_ERROR("[Error]: prefix key RoPE canonicalization only supports half-split pairing now\n");
        return;
    }
    if (ropeDim <= 0) {
        ropeDim = mHeadDim;
    }
    ropeDim = ALIMIN(ropeDim, mHeadDim);
    if (ropeDim <= 0 || (ropeDim % 2) != 0 || ropeTheta <= 0.0f) {
        MNN_ERROR("[Error]: invalid RoPE params, rope_dim=%d, head_dim=%d, theta=%f\n", ropeDim, mHeadDim, ropeTheta);
        return;
    }

    // MNN 导出的 Llama 类 RoPE 使用 rotate_half：前半维和后半维成对旋转。
    // inverse=true 时把 sin 取反，即可从 RoPE(position) * K_raw 恢复 canonical K_raw。
    int half = ropeDim / 2;
    for (int h = 0; h < mKvNumHead; ++h) {
        auto keyPtr = reinterpret_cast<T*>(addrOfKey(h));
        for (int i = 0; i < len; ++i) {
            int seq = start + i;
            int position = positionBase + i;
            for (int d = 0; d < half; ++d) {
                double invFreq = 1.0 / std::pow(static_cast<double>(ropeTheta), static_cast<double>(2 * d) / ropeDim);
                double angle = static_cast<double>(position) * invFreq;
                float cosValue = static_cast<float>(std::cos(angle));
                float sinValue = static_cast<float>(std::sin(angle));
                if (inverse) {
                    sinValue = -sinValue;
                }
                size_t leftIndex = keyIndex(seq, d);
                size_t rightIndex = keyIndex(seq, d + half);
                float left = static_cast<float>(keyPtr[leftIndex]);
                float right = static_cast<float>(keyPtr[rightIndex]);
                keyPtr[leftIndex] = static_cast<T>(left * cosValue - right * sinValue);
                keyPtr[rightIndex] = static_cast<T>(right * cosValue + left * sinValue);
            }
        }
    }
}

void CPUKVCacheManager::copyPrefixSegmentKV(const int8_t* srcKey, size_t srcKeySizePerHead, const int8_t* srcValue,
                                            size_t srcValueSizePerHead, int srcLen, int dstStart) {
    // direct_segments 需要把多个独立 prefix 文件按真实 token 数紧密拼接。
    // 文件本身按 hP/lP/blockKV 做了容量 padding，因此这里逐 token / dim 复制，避免把 padding 放进运行时 KV。
    for (int h = 0; h < mKvNumHead; ++h) {
        auto srcKeyHead = srcKey + h * srcKeySizePerHead;
        auto srcValueHead = srcValue + h * srcValueSizePerHead;
        auto dstKeyHead = addrOfKey(h);
        auto dstValueHead = addrOfValue(h);
        for (int s = 0; s < srcLen; ++s) {
            int dstSeq = dstStart + s;
            for (int d = 0; d < mHeadDim; ++d) {
                ::memcpy(dstKeyHead + packedKeyIndex(dstSeq, d) * mBytes,
                         srcKeyHead + keyIndexPhysical(s, d) * mBytes,
                         mBytes);
                ::memcpy(dstValueHead + packedFlashValueIndex(dstSeq, d) * mBytes,
                         srcValueHead + flashValueIndexPhysical(s, d) * mBytes,
                         mBytes);
            }
        }
    }
}

void CPUKVCacheManager::copyPagedPrefixBlock(int logicalStart, int tokenCount, int kvHead,
                                             int8_t* keyDst, int8_t* valueDst) const {
    if (tokenCount <= 0 || keyDst == nullptr || valueDst == nullptr) {
        return;
    }
    size_t keyBytes = (size_t)UP_DIV(tokenCount, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
    size_t valueBytes = (size_t)UP_DIV((int32_t)mFlashAttentionUpperKv, lP) * lP *
        ROUND_UP(mHeadDim, hP) * mBytes;
    ::memset(keyDst, 0, keyBytes);
    ::memset(valueDst, 0, valueBytes);

    auto self = const_cast<CPUKVCacheManager*>(this);
    auto keySrc = self->addrOfKey(kvHead);
    auto valueSrc = self->addrOfValue(kvHead);
    for (int s = 0; s < tokenCount; ++s) {
        int logicalSeq = logicalStart + s;
        int physicalSeq = physicalSlotForLogical(logicalSeq);
        for (int d = 0; d < mHeadDim; ++d) {
            ::memcpy(keyDst + keyIndexPhysical(s, d) * mBytes,
                     keySrc + keyIndexPhysical(physicalSeq, d) * mBytes,
                     mBytes);
            ::memcpy(valueDst + flashValueIndexPhysical(s, d) * mBytes,
                     valueSrc + flashValueIndexPhysical(physicalSeq, d) * mBytes,
                     mBytes);
        }

        if (!mPagedPrefixActive || logicalSeq < 0 ||
            logicalSeq >= (int)mPagedPrefixKeyNeedsRoPE.size() ||
            mPagedPrefixKeyNeedsRoPE[logicalSeq] == 0) {
            continue;
        }
        int ropePairing = mPagedPrefixRopePairing[logicalSeq];
        int ropeDim = mPagedPrefixRopeDim[logicalSeq] > 0 ? mPagedPrefixRopeDim[logicalSeq] : mHeadDim;
        float ropeTheta = mPagedPrefixRopeTheta[logicalSeq] > 0.0f ? mPagedPrefixRopeTheta[logicalSeq] : 10000.0f;
        ropeDim = ALIMIN(ropeDim, mHeadDim);
        if (ropePairing != KVMeta::RopePairingHalf || ropeDim <= 0 || (ropeDim % 2) != 0 || ropeTheta <= 0.0f) {
            continue;
        }
        int half = ropeDim / 2;
        if (mBytes == 2) {
            auto typedKeyDst = reinterpret_cast<FLOAT16_T*>(keyDst);
            for (int d = 0; d < half; ++d) {
                double invFreq = 1.0 / std::pow(static_cast<double>(ropeTheta), static_cast<double>(2 * d) / ropeDim);
                double angle = static_cast<double>(logicalSeq) * invFreq;
                float cosValue = static_cast<float>(std::cos(angle));
                float sinValue = static_cast<float>(std::sin(angle));
                size_t leftIndex = keyIndexPhysical(s, d);
                size_t rightIndex = keyIndexPhysical(s, d + half);
                float left = static_cast<float>(typedKeyDst[leftIndex]);
                float right = static_cast<float>(typedKeyDst[rightIndex]);
                typedKeyDst[leftIndex] = static_cast<FLOAT16_T>(left * cosValue - right * sinValue);
                typedKeyDst[rightIndex] = static_cast<FLOAT16_T>(right * cosValue + left * sinValue);
            }
        } else {
            auto typedKeyDst = reinterpret_cast<float*>(keyDst);
            for (int d = 0; d < half; ++d) {
                double invFreq = 1.0 / std::pow(static_cast<double>(ropeTheta), static_cast<double>(2 * d) / ropeDim);
                double angle = static_cast<double>(logicalSeq) * invFreq;
                float cosValue = static_cast<float>(std::cos(angle));
                float sinValue = static_cast<float>(std::sin(angle));
                size_t leftIndex = keyIndexPhysical(s, d);
                size_t rightIndex = keyIndexPhysical(s, d + half);
                float left = typedKeyDst[leftIndex];
                float right = typedKeyDst[rightIndex];
                typedKeyDst[leftIndex] = left * cosValue - right * sinValue;
                typedKeyDst[rightIndex] = right * cosValue + left * sinValue;
            }
        }
    }
}

template <typename T>
void CPUKVCacheManager::moveKV(int src, int dst, int size) {
    for (int h = 0; h < mKvNumHead; ++h) {
        auto kPtr = reinterpret_cast<T*>(addrOfKey(h));
        auto vPtr = reinterpret_cast<T*>(addrOfValue(h));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < mHeadDim; j++) {
                kPtr[keyIndex(dst + i, j)]   = kPtr[keyIndex(src + i, j)];
                vPtr[valueIndex(dst + i, j)] = vPtr[valueIndex(src + i, j)];
            }
        }
    }
}

void CPUKVCacheManager::onUpdateKV(const Tensor * key, const Tensor * value, int add) {
    auto core = static_cast<CPUBackend*>(mBackend)->functions();
    int seq_len = add;
    auto divPart = UP_DIV(mKvNumHead, 1);
    MNN_CONCURRENCY_BEGIN(tId, 1) {
        auto remainPart = mKvNumHead - tId * divPart;
        if (remainPart > 0) {
            remainPart = ALIMIN(divPart, remainPart);
            int startIdx = tId * divPart;
            int endIdx = startIdx + remainPart;
            for (int h = startIdx; h < endIdx; ++h) {
                if (mBytes == 2) {
                    ProcessKey<FLOAT16_T>(key, seq_len, h);
                    ProcessValue<FLOAT16_T>(value, seq_len, h);
                } else {
                    ProcessKey<float>(key, seq_len, h);
                    ProcessValue<float>(value, seq_len, h);
                }
            }
        }
    } MNN_CONCURRENCY_END();
    if (mMeta != nullptr && mMeta->file_flag == KVMeta::PendingWrite &&
        mMeta->key_rope_state == KVMeta::KeyRopeCanonicalNoRope) {
        // 保存可复用文档 prefix 时，图里的 key 已经带 local position RoPE。
        // 在写入磁盘前对本次新增区间逆 RoPE，磁盘上只保留无位置信息的 canonical key。
        int positionBase = mMeta->source_position_base + mPastLength;
        if (mBytes == 2) {
            applyPackedKeyRoPE<FLOAT16_T>(mPastLength, seq_len, positionBase, mMeta->rope_dim,
                                          mMeta->rope_theta, mMeta->rope_pairing, true);
        } else {
            applyPackedKeyRoPE<float>(mPastLength, seq_len, positionBase, mMeta->rope_dim,
                                      mMeta->rope_theta, mMeta->rope_pairing, true);
        }
    }
    mPastLength += seq_len;
}

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
