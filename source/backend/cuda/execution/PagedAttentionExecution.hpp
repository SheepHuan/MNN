#ifndef MNN_CUDA_PAGED_ATTENTION_EXECUTION_HPP
#define MNN_CUDA_PAGED_ATTENTION_EXECUTION_HPP

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "core/PagedKVMeta.hpp"

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

class CUDAPagedAttention : public Execution {
public:
    struct SharedPagedCache {
        std::shared_ptr<Tensor> key;       // [max_slots, B, H_kv, D]
        std::shared_ptr<Tensor> value;     // [B, H_kv, max_slots, D]
        std::shared_ptr<Tensor> slotTable; // [max_slots], int32
        std::shared_ptr<Tensor> sparseQuery; // [max_slots], int32 logical indices for sparse recompute
        int maxSlots = 0;
        int batch = 0;
        int kvHeads = 0;
        int headDim = 0;
        int precision = 4;
        int slotTableVersion = -1;
        int slotTableLength = 0;
    };

    CUDAPagedAttention(Backend* backend, const MNN::Op* op);
    virtual ~CUDAPagedAttention();
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    ErrorCode ensureCache(int maxSlots, int batch, int kvHeads, int headDim);
    ErrorCode syncSlotTable(int requiredSlots);
    bool ensurePrefillTemp(size_t elements);

    CUDABackend* mCudaBackend = nullptr;
    PagedKVMeta* mMeta = nullptr;
    std::shared_ptr<SharedPagedCache> mCache;
    int mLayerIndex = -1;
    int mKVSharedLayerIndex = -1;
    bool mIsKVShared = false;
    int mPrecision = 4;
    int mBatch = 0;
    int mQuerySeqLen = 0;
    int mNumHead = 0;
    int mHeadDim = 0;
    int mKvNumHead = 0;
    int mNewKvSeqLen = 0;
    float mScale = 1.0f;
    float* mPrefillQK = nullptr;
    float* mPrefillSoftmax = nullptr;
    size_t mPrefillElements = 0;
};

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN

#endif // MNN_CUDA_PAGED_ATTENTION_EXECUTION_HPP
