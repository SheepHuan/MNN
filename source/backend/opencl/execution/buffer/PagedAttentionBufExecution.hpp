//
//  PagedAttentionBufExecution.hpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef PagedAttentionBufExecution_hpp
#define PagedAttentionBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "core/PagedKVMeta.hpp"

namespace MNN {
namespace OpenCL {

class PagedAttentionBufExecution : public CommonExecution {
public:
    struct SharedPagedCache {
        std::shared_ptr<Tensor> key;         // [max_slots, B, H_kv, D]
        std::shared_ptr<Tensor> value;       // [B, H_kv, max_slots, D]
        std::shared_ptr<Tensor> slotTable;   // [max_slots], int32
        std::shared_ptr<Tensor> sparseQuery; // [max_slots], int32
        int maxSlots = 0;
        int batch = 0;
        int kvHeads = 0;
        int headDim = 0;
        int bytes = 4;
        int slotTableVersion = -1;
        int slotTableLength = 0;
    };

    PagedAttentionBufExecution(const MNN::Op* op, Backend* backend);
    virtual ~PagedAttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    ErrorCode ensureCache(int maxSlots, int batch, int kvHeads, int headDim);
    ErrorCode syncSlotTable(int requiredSlots);
    ErrorCode syncSparseQuery(int attnLen);
    ErrorCode ensureFastPrefillTemps(int seqLen, int kvLen, int qChunkLen, bool staticWorkspace);
    ErrorCode ensureSparseFlashKernel();
    ErrorCode ensureExternalTemps(size_t keyElements, size_t valueElements);
    ErrorCode ensureCacheBlendScoreTemps(int scoreCount, int indexCount);
    ErrorCode hydrateExternalSegments(int layerIndex, int kvLen);
    ErrorCode runCacheBlendScoring(int layerIndex, int kvLen);
    bool canUseFastPrefill(const Tensor* mask, int baseLogical, int attnLen, int kvLen, bool sparseQuery,
                           bool externalHydrated, int* maskKeyLen) const;
    bool canUseSparseFastPrefill(const Tensor* mask, int attnLen, int kvLen, bool externalHydrated,
                                 bool queryRowsAreFull) const;
    ErrorCode runFastPrefill(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen,
                             int maskKeyLen);
    ErrorCode runSparseFastPrefill(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen,
                                   int attnLen, bool queryRowsAreFull);

    OpenCLBackend* mOpenCLBackend = nullptr;
    PagedKVMeta* mMeta = nullptr;
    std::shared_ptr<SharedPagedCache> mCache;
    std::shared_ptr<KernelWrap> mCopyKernel;
    std::shared_ptr<KernelWrap> mAttentionKernel;
    std::shared_ptr<KernelWrap> mAttentionRowKernel;
    std::shared_ptr<KernelWrap> mPackPagedKVKernel;
    std::shared_ptr<KernelWrap> mPackPagedKeyKernel;
    std::shared_ptr<KernelWrap> mHydrateExternalKernel;
    std::shared_ptr<KernelWrap> mExportCanonicalKeyKernel;
    std::shared_ptr<KernelWrap> mCacheBlendScoreKernel;
    std::shared_ptr<KernelWrap> mCacheBlendTopKKernel;
    std::shared_ptr<KernelWrap> mRearrangeQKernel;
    std::shared_ptr<KernelWrap> mRearrangeMaskKernel;
    std::shared_ptr<KernelWrap> mQKKernel;
    std::shared_ptr<KernelWrap> mSoftmaxKernel;
    std::shared_ptr<KernelWrap> mQKVKernel;
    std::shared_ptr<KernelWrap> mSparseFlashKernel32;
    std::shared_ptr<KernelWrap> mSparseFlashKernel64;
    std::shared_ptr<KernelWrap> mZeroKernel;
    std::shared_ptr<Tensor> mTempQ;
    std::shared_ptr<Tensor> mTempK;
    std::shared_ptr<Tensor> mTempV;
    std::shared_ptr<Tensor> mTempMask;
    std::shared_ptr<Tensor> mTempQK;
    std::shared_ptr<Tensor> mTempSoftmax;
    std::shared_ptr<Tensor> mExternalKey;
    std::shared_ptr<Tensor> mExternalValue;
    std::shared_ptr<Tensor> mCacheBlendScores;
    std::shared_ptr<Tensor> mCacheBlendIndices;
    size_t mExternalKeyElements = 0;
    size_t mExternalValueElements = 0;
    int mCacheBlendScoreCount = 0;
    int mCacheBlendIndexCount = 0;
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
    int mSparseFlashKernelGroupSize = 0;
    bool mFastStaticWorkspace = false;
    bool mFastKernelStatic = false;
    bool mFastKernelSparse = false;
    float mScale = 1.0f;
};

} // namespace OpenCL
} // namespace MNN

#endif /* PagedAttentionBufExecution_hpp */

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
