//
//  PagedAttentionBufExecution.hpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef PagedAttentionBufExecution_hpp
#define PagedAttentionBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "backend/opencl/core/ImagePool.hpp"
#include "core/PagedKVMeta.hpp"
#include <cstdint>
#include <vector>

namespace MNN {
namespace OpenCL {

class PagedAttentionBufExecution : public CommonExecution {
public:
    struct SharedPagedCache {
        std::shared_ptr<Tensor> key;         // [max_slots, B, H_kv, D]
        std::shared_ptr<Tensor> decodeKey;   // [B, H_kv, D, max_slots], decode-only transposed K view
        std::shared_ptr<Tensor> value;       // [B, H_kv, max_slots, D]
        std::shared_ptr<Tensor> sparseQuery; // [max_slots], int32
        int maxSlots = 0;
        int batch = 0;
        int kvHeads = 0;
        int headDim = 0;
        int bytes = 4;
        int decodeKeyReadyLength = 0;
        std::vector<int> sparseQueryHost;
    };

    PagedAttentionBufExecution(const MNN::Op* op, Backend* backend);
    virtual ~PagedAttentionBufExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    ErrorCode prepareDecodePrefix(int kvLen);

private:
    ErrorCode ensureCache(int maxSlots, int batch, int kvHeads, int headDim);
    ErrorCode syncSparseQuery(int attnLen);
    ErrorCode ensureFastPrefillTemps(int seqLen, int kvLen, int qChunkLen, bool staticWorkspace);
    ErrorCode ensureSparseFlashTemps(int seqLen, int kvLen, bool staticWorkspace);
    ErrorCode ensureAdrenoGemmPrefillTemps(int seqLen, int kvLen, int qSplitNum);
    ErrorCode ensureSparseFlashKernel();
    ErrorCode ensureDecodeCausalKernel();
    ErrorCode ensureDecodeCausalKernelHD128Identity();
    ErrorCode ensureDecodeTransposedKKernel();
    ErrorCode ensureDecodeRepairSlotIdentitySplitKernel();
    ErrorCode ensureDecodeAttentionRankKernel();
    ErrorCode ensureDecodeTransposedTemps(int kvLen, int rows = 1);
    ErrorCode ensureDecodeKeyReady(int requiredLen, bool insideDecode = false);
    ErrorCode appendDecodeKeyValueHD128(const std::vector<Tensor*>& inputs, int baseLogical,
                                        bool profileDetail, uint64_t* appendUs);
    ErrorCode ensureExternalTemps(size_t keyElements, size_t valueElements);
    ErrorCode ensureCacheBlendScoreTemps(int scoreCount, int indexCount, int stageCandidateCount = 0);
    ErrorCode ensureAdrenoCacheBlendValueImage(int tokenCapacity);
    ErrorCode ensureAdrenoSparseFlashPackedKeyImage(int kvPack);
    ErrorCode ensureAdrenoSparseFlashPackedKVImages(int kvPack);
    ErrorCode syncDecodeAttentionHeadIds();
    ErrorCode hydrateExternalSegments(int layerIndex, int kvLen);
    ErrorCode runCacheBlendScoring(int layerIndex, int kvLen);
    ErrorCode runFusionRAGOnlineScoring(int layerIndex, int kvLen, const Tensor* query);
    bool canUseFastPrefill(const Tensor* mask, int baseLogical, int attnLen, int kvLen, bool sparseQuery,
                           bool externalHydrated, int* maskKeyLen) const;
    bool canUseSparseFastPrefill(const Tensor* mask, int attnLen, int kvLen, bool externalHydrated,
                                 bool queryRowsAreFull) const;
    bool canUseSparseQSplitPrefill(const Tensor* mask, int attnLen, int kvLen, bool externalHydrated,
                                   bool queryRowsAreFull, int* maskKeyLen) const;
    ErrorCode runFastPrefill(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen,
                             int maskKeyLen);
    ErrorCode runAdrenoGemmPrefill(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   int kvLen, int qSplitNum);
    ErrorCode runSparseFastPrefill(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen,
                                   int attnLen, bool queryRowsAreFull);
    ErrorCode runSparseQSplitPrefill(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                     int kvLen, int attnLen, bool queryRowsAreFull, int maskKeyLen);
    ErrorCode runDecodeCausalAttention(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                       int kvLen, int attnLen, int baseLogical, bool sparseQuery,
                                       bool queryRowsAreFull);
    ErrorCode runDecodeCausalAttentionHD128Identity(const std::vector<Tensor*>& inputs,
                                                    const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
                                                    int baseLogical, int layerIndex);
    ErrorCode runDecodeCausalAttentionHD128IdentityFusedKV(const std::vector<Tensor*>& inputs,
                                                           const std::vector<Tensor*>& outputs, int kvLen,
                                                           int attnLen, int baseLogical, int layerIndex);
    ErrorCode runDecodeCausalAttentionHD128IdentityFusedKVGQA(const std::vector<Tensor*>& inputs,
                                                              const std::vector<Tensor*>& outputs, int kvLen,
                                                              int attnLen, int baseLogical, int layerIndex);
    ErrorCode runDecodeCausalAttentionHD128TransposedKFusedKV(const std::vector<Tensor*>& inputs,
                                                              const std::vector<Tensor*>& outputs, int kvLen,
                                                              int attnLen, int baseLogical, int layerIndex);
    ErrorCode runDecodeCausalAttentionHD128TransposedKFusedKVGQA(const std::vector<Tensor*>& inputs,
                                                                 const std::vector<Tensor*>& outputs, int kvLen,
                                                                 int attnLen, int baseLogical, int layerIndex);
    ErrorCode runDecodeCausalAttentionHD128TransposedKFusedKVRecord(const std::vector<Tensor*>& inputs,
                                                                    const std::vector<Tensor*>& outputs, int kvLen,
                                                                    int attnLen, int baseLogical, int layerIndex,
                                                                    uint32_t lanes,
                                                                    std::shared_ptr<KernelWrap> kernel);
    ErrorCode runDecodeCausalAttentionHD128TransposedKAppendReadonly(const std::vector<Tensor*>& inputs,
                                                                     const std::vector<Tensor*>& outputs,
                                                                     int kvLen, int attnLen, int baseLogical,
                                                                     int layerIndex, bool appendCurrent);
    ErrorCode runDecodeCausalAttentionHD128SplitProfile(const std::vector<Tensor*>& inputs,
                                                        const std::vector<Tensor*>& outputs, int kvLen,
                                                        int attnLen, int baseLogical, int layerIndex,
                                                        bool transposedK);
    ErrorCode runDecodeCausalAttentionHD128SlotIdentitySplit(const std::vector<Tensor*>& inputs,
                                                             const std::vector<Tensor*>& outputs, int kvLen,
                                                             int attnLen, int layerIndex);
    ErrorCode runDecodeCausalAttentionHD128TransposedKAppendReadonlyRecord(
        const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
        int baseLogical, int layerIndex, uint32_t lanes, std::shared_ptr<KernelWrap> attentionKernel);
    ErrorCode runDecodeCausalAttentionHD128TransposedKSparse(const std::vector<Tensor*>& inputs,
                                                             const std::vector<Tensor*>& outputs, int kvLen,
                                                             int attnLen, int layerIndex);
    ErrorCode runDecodeCausalAttentionHD128TransposedKSparseRecord(
        const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen,
        int attnLen, int layerIndex, uint32_t lanes, int qTile, std::shared_ptr<KernelWrap> kernel);
    // Negative A/B routes are intentionally undeclared: q1 identity fused-KV,
    // keycache qtile v2, identity qtile, sparse GQA, and fused-append record all
    // regressed decode TPOT versus the transposed-K qtile family.
    ErrorCode runDecodeCausalAttentionHD128IdentityRecord(const std::vector<Tensor*>& inputs,
                                                          const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
                                                          int baseLogical, int layerIndex, uint32_t lanes,
                                                          std::shared_ptr<KernelWrap> kernel);
    ErrorCode runDecodeAttentionRankCaptureOpenCL(const Tensor* query, int layerIndex, int kvLen, int attnLen,
                                                  bool queryRowsAreFull);

    OpenCLBackend* mOpenCLBackend = nullptr;
    PagedKVMeta* mMeta = nullptr;
    std::shared_ptr<SharedPagedCache> mCache;
    std::shared_ptr<KernelWrap> mCopyKernel;
    std::shared_ptr<KernelWrap> mAttentionKernel;
    std::shared_ptr<KernelWrap> mAttentionRowKernel;
    std::shared_ptr<KernelWrap> mPackPagedKVKernel;
    std::shared_ptr<KernelWrap> mPackPagedKeyKernel;
    std::shared_ptr<KernelWrap> mPackPagedKeyToImageKernel;
    std::shared_ptr<KernelWrap> mHydrateExternalKernel;
    std::shared_ptr<KernelWrap> mHydrateExternalInplaceKernel;
    std::shared_ptr<KernelWrap> mExportCanonicalKeyKernel;
    std::shared_ptr<KernelWrap> mCacheBlendScoreKernel;
    std::shared_ptr<KernelWrap> mCacheBlendTopKKernel;
    std::shared_ptr<KernelWrap> mCacheBlendTopKStage1Kernel;
    std::shared_ptr<KernelWrap> mCacheBlendTopKStage2Kernel;
    std::shared_ptr<KernelWrap> mCacheBlendScoreImageKernel;
    std::shared_ptr<KernelWrap> mCopyBufferToImageLinearKernel;
    std::shared_ptr<KernelWrap> mRearrangeQKernel;
    std::shared_ptr<KernelWrap> mRearrangeSparseQKernel;
    std::shared_ptr<KernelWrap> mRearrangeMaskKernel;
    std::shared_ptr<KernelWrap> mQKKernel;
    std::shared_ptr<KernelWrap> mSoftmaxKernel;
    std::shared_ptr<KernelWrap> mQKVKernel;
    std::shared_ptr<KernelWrap> mAdrenoGemmRearrangeQKernel;
    std::shared_ptr<KernelWrap> mAdrenoGemmPackPagedKVKernel;
    std::shared_ptr<KernelWrap> mAdrenoGemmMaskKernel;
    std::shared_ptr<KernelWrap> mAdrenoGemmClipKernel;
    std::shared_ptr<KernelWrap> mSparseFlashKernel32;
    std::shared_ptr<KernelWrap> mSparseFlashKernel64;
    std::shared_ptr<KernelWrap> mSparseFlashKernelMQTileHD64Q4K16;
    std::shared_ptr<KernelWrap> mSparseFlashKernelMQTileHD128Q4K16;
    std::shared_ptr<KernelWrap> mSparseFlashKernelMQTileHD128Q4K8;
    std::shared_ptr<KernelWrap> mSparseFlashKernelMQTileHD128Q4K8KImage;
    std::shared_ptr<KernelWrap> mSparseFlashKernelMQTileHD128Q4K8KVImage;
    std::shared_ptr<KernelWrap> mSparseFlashKernelMQTileHD128Q8K16;
    std::shared_ptr<KernelWrap> mSparseFlashKernelMQTileHD128Q8K16KImage;
    std::shared_ptr<KernelWrap> mSparseFlashKernelMQTileHD128Q8K16KVImage;
    std::shared_ptr<KernelWrap> mDecodeCausalKernel32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernel64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityRow32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityRow64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityRow128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityFusedKVRow32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityFusedKVRow64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityFusedKVRow128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityFusedKVGQARow32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityFusedKVGQARow64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128IdentityFusedKVGQARow128;
    std::shared_ptr<KernelWrap> mDecodeQKIdentityKernel;
    std::shared_ptr<KernelWrap> mDecodeKeyTransposeKernel;
    std::shared_ptr<KernelWrap> mDecodeKeyAppendKernel;
    std::shared_ptr<KernelWrap> mDecodeKeyAppendSparseKernel;
    std::shared_ptr<KernelWrap> mDecodeKeyAppendSparseSlotIdentityKernel;
    std::shared_ptr<KernelWrap> mDecodeQKTransposedKernel;
    std::shared_ptr<KernelWrap> mDecodeQKVTransposedKernel;
    std::shared_ptr<KernelWrap> mDecodeQKRepairSlotIdentityKernel;
    std::shared_ptr<KernelWrap> mDecodeQKVRepairSlotIdentityKernel;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKFusedKVRow32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKFusedKVRow64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKFusedKVRow128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKFusedKVGQARow32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKFusedKVGQARow64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKFusedKVGQARow128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKReadonlyRow32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKReadonlyRow64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKReadonlyRow128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKSparseRow32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKSparseRow64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKSparseRow128;
    // PoC (x0 only, env MNN_PIC_DECODE_SPARSE_DCONTIG_K): d-continuous K read
    // from key_cache via vload4 instead of transposed decode_key gathers.
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKSparseDcontigRow128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile1Row32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile1Row64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile1Row128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile2Row32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile2Row64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile2Row128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile4Row32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile4Row64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile4Row128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile8Row32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile8Row64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTile8Row128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused1Row32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused1Row64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused1Row128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused2Row32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused2Row64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused2Row128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused4Row32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused4Row64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused4Row128;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused8Row32;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused8Row64;
    std::shared_ptr<KernelWrap> mDecodeCausalKernelHD128TransposedKQTileFused8Row128;
    std::shared_ptr<KernelWrap> mDecodeSoftmaxKernel;
    std::shared_ptr<KernelWrap> mDecodeAttentionRankScoreKernelHD128;
    std::shared_ptr<KernelWrap> mZeroKernel;
    std::shared_ptr<Tensor> mTempQ;
    std::shared_ptr<Tensor> mTempK;
    std::shared_ptr<Tensor> mTempV;
    std::shared_ptr<Tensor> mTempMask;
    std::shared_ptr<Tensor> mTempQK;
    std::shared_ptr<Tensor> mTempSoftmax;
    std::shared_ptr<Tensor> mTempQKV;
    std::shared_ptr<Tensor> mExternalKey;
    std::shared_ptr<Tensor> mExternalValue;
    std::shared_ptr<Tensor> mCacheBlendScores;
    std::shared_ptr<Tensor> mCacheBlendIndices;
    std::shared_ptr<Tensor> mCacheBlendStageValues;
    std::shared_ptr<Tensor> mCacheBlendStageIndices;
    std::shared_ptr<Tensor> mDecodeAttentionHeadIds;
    std::shared_ptr<ImagePool> mAdrenoImagePool;
    cl::Image* mCacheBlendSourceValueImage = nullptr;
    cl::Image* mSparseFlashPackedKeyImage = nullptr;
    cl::Image* mSparseFlashPackedValueImage = nullptr;
    size_t mExternalKeyElements = 0;
    size_t mExternalValueElements = 0;
    int mCacheBlendScoreCount = 0;
    int mCacheBlendIndexCount = 0;
    int mCacheBlendStageCandidateCount = 0;
    int mDecodeAttentionHeadIdCapacity = 0;
    int mCacheBlendSourceValueTokenCapacity = 0;
    int mCacheBlendSourceValueImageWidth = 0;
    int mCacheBlendSourceValueImageHeight = 0;
    int mCacheBlendSourceValueBytes = 0;
    int mSparseFlashPackedKVLen = 0;
    int mSparseFlashPackedKeyImageWidth = 0;
    int mSparseFlashPackedKeyImageHeight = 0;
    int mSparseFlashPackedValueImageWidth = 0;
    int mSparseFlashPackedValueImageHeight = 0;
    int mSparseFlashPackedImageBytes = 0;
    int mLayerIndex = -1;
    int mKVSharedLayerIndex = -1;
    int mPicAttentionMode = 0; // 0: full, 1: score layer, 2: sparse layer
    bool mIsKVShared = false;
    int mBytes = 4;
    int mBatch = 0;
    int mQuerySeqLen = 0;
    int mNumHead = 0;
    int mHeadDim = 0;
    int mKvNumHead = 0;
    int mNewKvSeqLen = 0;
    int mFastSeqLen = 0;
    int mFastKvLen = 0;
    int mFastQChunkLen = 0;
    int mAdrenoGemmSeqLen = 0;
    int mAdrenoGemmKvLen = 0;
    int mAdrenoGemmQSplitNum = 0;
    int mAdrenoGemmQKVElements = 0;
    int mSparseFlashKernelGroupSize = 0;
    int mDecodeCausalKernelGroupSize = 0;
    int mDecodeCausalHD128IdentityKernelGroupSize = 0;
    int mDecodeTransposedKKernelGroupSize = 0;
    int mDecodeRepairSlotIdentitySplitKernelGroupSize = 0;
    int mDecodeTransposedTempKvLen = 0;
    int mDecodeTransposedTempRows = 1;
    int mLastDecodeKeyRequiredLen = 0;
    uint64_t mLastDecodeKeyPrepareUs = 0;
    bool mLastDecodeKeyReadyHit = false;
    bool mLastDecodeKeyPrepared = false;
    bool mLastDecodeKeyPrepareInsideDecode = false;
    bool mLastDecodeKeySlotIdentity = false;
    bool mLastDecodeKeyPrefixStable = false;
    RecordUpdateInfo mDecodeIdentityRecordUpdateInfo;
    std::vector<RecordUpdateInfo*> mDecodeIdentityRecordUpdateInfos;
    bool mDecodeIdentityRecordValid = false;
    uint32_t mDecodeIdentityRecordLanes = 0;
    uint32_t mDecodeIdentityRecordHeads = 0;
    int mDecodeIdentityRecordAttnLen = 0;
    uint32_t mDecodeIdentityRecordGws0 = 0;
    uint32_t mDecodeIdentityRecordGws1 = 0;
    uint32_t mDecodeIdentityRecordGws2 = 0;
    int mDecodeIdentityRecordQuerySeqLen = 0;
    int mDecodeIdentityRecordBaseLogical = 0;
    int mDecodeIdentityRecordKvLen = 0;
    int mDecodeIdentityRecordMaxSlots = 0;
    RecordUpdateInfo mDecodeTransposedFusedRecordUpdateInfo;
    std::vector<RecordUpdateInfo*> mDecodeTransposedFusedRecordUpdateInfos;
    bool mDecodeTransposedFusedRecordValid = false;
    uint32_t mDecodeTransposedFusedRecordLanes = 0;
    uint32_t mDecodeTransposedFusedRecordHeads = 0;
    int mDecodeTransposedFusedRecordAttnLen = 0;
    int mDecodeTransposedFusedRecordGroupSize = 0;
    uint32_t mDecodeTransposedFusedRecordGws0 = 0;
    uint32_t mDecodeTransposedFusedRecordGws1 = 0;
    uint32_t mDecodeTransposedFusedRecordGws2 = 0;
    int mDecodeTransposedFusedRecordQuerySeqLen = 0;
    int mDecodeTransposedFusedRecordBaseLogical = 0;
    int mDecodeTransposedFusedRecordKvLen = 0;
    int mDecodeTransposedFusedRecordMaxSlots = 0;
    float mDecodeTransposedFusedRecordScale = 1.0f;
    RecordUpdateInfo mDecodeTransposedAppendRecordUpdateInfo;
    RecordUpdateInfo mDecodeTransposedReadonlyRecordUpdateInfo;
    std::vector<RecordUpdateInfo*> mDecodeTransposedAppendReadonlyRecordUpdateInfos;
    bool mDecodeTransposedAppendReadonlyRecordValid = false;
    uint32_t mDecodeTransposedAppendReadonlyRecordLanes = 0;
    uint32_t mDecodeTransposedAppendReadonlyRecordHeads = 0;
    int mDecodeTransposedAppendReadonlyRecordAttnLen = 0;
    int mDecodeTransposedAppendReadonlyRecordGroupSize = 0;
    uint32_t mDecodeTransposedAppendReadonlyRecordAppendGws0 = 0;
    uint32_t mDecodeTransposedAppendReadonlyRecordAppendGws1 = 0;
    uint32_t mDecodeTransposedAppendReadonlyRecordGws0 = 0;
    uint32_t mDecodeTransposedAppendReadonlyRecordGws1 = 0;
    uint32_t mDecodeTransposedAppendReadonlyRecordGws2 = 0;
    int mDecodeTransposedAppendReadonlyRecordQuerySeqLen = 0;
    int mDecodeTransposedAppendReadonlyRecordBaseLogical = 0;
    int mDecodeTransposedAppendReadonlyRecordKvLen = 0;
    int mDecodeTransposedAppendReadonlyRecordMaxSlots = 0;
    float mDecodeTransposedAppendReadonlyRecordScale = 1.0f;
    RecordUpdateInfo mDecodeTransposedSparseAppendRecordUpdateInfo;
    RecordUpdateInfo mDecodeTransposedSparseAttentionRecordUpdateInfo;
    std::vector<RecordUpdateInfo*> mDecodeTransposedSparseRecordUpdateInfos;
    bool mDecodeTransposedSparseRecordValid = false;
    uint32_t mDecodeTransposedSparseRecordLanes = 0;
    uint32_t mDecodeTransposedSparseRecordHeads = 0;
    int mDecodeTransposedSparseRecordAttnLen = 0;
    int mDecodeTransposedSparseRecordQTile = 0;
    uint32_t mDecodeTransposedSparseRecordAppendGws0 = 0;
    uint32_t mDecodeTransposedSparseRecordAppendGws1 = 0;
    uint32_t mDecodeTransposedSparseRecordAppendGws2 = 0;
    uint32_t mDecodeTransposedSparseRecordGws0 = 0;
    uint32_t mDecodeTransposedSparseRecordGws1 = 0;
    uint32_t mDecodeTransposedSparseRecordGws2 = 0;
    int mDecodeTransposedSparseRecordQuerySeqLen = 0;
    int mDecodeTransposedSparseRecordNewKvSeqLen = 0;
    int mDecodeTransposedSparseRecordKvLen = 0;
    int mDecodeTransposedSparseRecordMaxSlots = 0;
    float mDecodeTransposedSparseRecordScale = 1.0f;
    // Dead record state for the negative A/B variants above is not kept here;
    // the active decode-repair record path only records append + transposed-K qtile.
    int mDecodeAttentionRankKernelGroupSize = 0;
    bool mFastStaticWorkspace = false;
    bool mFastKernelStatic = false;
    bool mFastKernelSparse = false;
    bool mFastKernelAddMask = false;
    bool mCopyBufferToImageLinearUseFp32 = false;
    std::vector<std::shared_ptr<KernelWrap>> mAdrenoGemmQKKernels;
    std::vector<std::shared_ptr<KernelWrap>> mAdrenoGemmSoftmaxKernels;
    std::vector<std::shared_ptr<KernelWrap>> mAdrenoGemmTransKernels;
    std::vector<std::shared_ptr<KernelWrap>> mAdrenoGemmQKVKernels;
    std::vector<int> mDecodeAttentionHeadIdsHost;
    float mScale = 1.0f;
};

} // namespace OpenCL
} // namespace MNN

#endif /* PagedAttentionBufExecution_hpp */

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
