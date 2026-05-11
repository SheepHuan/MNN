//
//  CPUPrefixAttention.cpp
//  MNN
//
//  PrefixAttention owns prefix-segment KV materialization. The dense attention
//  math is inherited from CPUAttention so the CPU kernel behavior stays aligned.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CPUAttention.hpp"
#include "CPUBackend.hpp"
#include "core/PagedKVCachePlan.hpp"

namespace MNN {

class CPUPrefixAttention : public CPUAttention {
public:
    CPUPrefixAttention(Backend* backend, bool kv_cache, int layerIndex)
        : CPUAttention(backend, kv_cache), mLayerIndex(layerIndex) {
    }

protected:
    virtual ErrorCode onPrepareKVCache(const std::vector<Tensor*>& inputs, int seqLen, int& insertLen) override {
        if (mIsKVShared) {
            insertLen = mMeta ? (int)mMeta->add : seqLen;
            return NO_ERROR;
        }

        if (mKVCache && mMeta != nullptr && mMeta->previous == mMeta->remove &&
            mMeta->file_flag == KVMeta::PendingReadSegments && !mMeta->prefix_segments.empty()) {
            int promptAppendLen = (int)mMeta->add;
            if (promptAppendLen <= 0) {
                promptAppendLen = seqLen;
            }
            if (!buildPrefixRuntimeKVBlockTablePlan(mPagedKVPlan, mMeta, promptAppendLen)) {
                return INVALID_VALUE;
            }
            mKVCacheManager->onClear();
            auto code = mKVCacheManager->onAllocPrefixSegments(mMeta, seqLen, mLayerIndex);
            if (code != NO_ERROR) {
                mKVCacheManager->onClear();
                return code;
            }
            insertLen = promptAppendLen;
            mKVCacheManager->onUpdateKV(inputs[1], inputs[2], insertLen);
            return NO_ERROR;
        }

        return CPUAttention::onPrepareKVCache(inputs, seqLen, insertLen);
    }

    virtual CPUAttention* onCreateClone(Backend* bn) const override {
        return new CPUPrefixAttention(bn, mKVCache, mLayerIndex);
    }

private:
    int mLayerIndex = -1;
    RuntimeKVBlockTablePlan mPagedKVPlan;
};

class CPUPrefixAttentionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_AttentionParam();
        bool kvCache = param ? param->kv_cache() : true;
        int layerIndex = param ? param->layer_index() : -1;
        return new CPUPrefixAttention(backend, kvCache, layerIndex);
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUPrefixAttentionCreator, OpType_PrefixAttention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
