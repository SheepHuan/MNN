//
//  CPUPagedAttention.hpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPU_PAGED_ATTENTION_HPP
#define CPU_PAGED_ATTENTION_HPP

#include "core/Execution.hpp"
#include "core/PagedKVMeta.hpp"

namespace MNN {

class CPUPagedAttention : public Execution {
public:
    CPUPagedAttention(Backend* backend, const Op* op);
    virtual ~CPUPagedAttention() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    struct LayerCache {
        std::shared_ptr<Tensor> key;
        std::shared_ptr<Tensor> value;
        int maxSlots = 0;
        int batch = 0;
        int kvHeads = 0;
        int headDim = 0;
        int bytes = 4;
    };

    ErrorCode ensureCache(int maxSlots, int batch, int kvHeads, int headDim);

    int mLayerIndex = -1;
    int mKVSharedLayerIndex = -1;
    int mPicAttentionMode = 0; // 0: full, 1: score layer, 2: sparse layer
    bool mIsKVShared = false;
    int mBytes = 4;
    float mScale = 1.0f;
    PagedKVMeta* mMeta = nullptr;
    std::shared_ptr<LayerCache> mCache;
};

} // namespace MNN

#endif // CPU_PAGED_ATTENTION_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
