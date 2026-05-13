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

        if (mKVCache && mMeta != nullptr &&
            mMeta->file_flag == KVMeta::PendingReadSegments && !mMeta->prefix_segments.empty()) {
            if (mMeta->remove > 0 || mMeta->n_reserve > 0) {
                MNN_ERROR("[Error]: CPU PrefixAttention direct_segments does not support remove/reserve compaction yet\n");
                return NOT_SUPPORT;
            }
            int promptAppendLen = (int)mMeta->add;
            if (promptAppendLen <= 0) {
                promptAppendLen = seqLen;
            }
            if (!buildPrefixRuntimeKVBlockTablePlan(mPagedKVPlan, mMeta, promptAppendLen)) {
                return INVALID_VALUE;
            }
            const int metaBasePast = static_cast<int>(mMeta->previous);
            const int cacheBasePast = mKVCacheManager->kvLength();
            const int basePast = (metaBasePast > 0 && cacheBasePast > 0) ? cacheBasePast : metaBasePast;
            if (mLayerIndex == 0 && !mPagedKVPlan.empty()) {
                if (metaBasePast != basePast) {
                    MNN_PRINT("[CPUPrefixAttention] base_past adjusted to runtime cache length: meta=%d cache=%d\n",
                              metaBasePast, cacheBasePast);
                }
                int prefixBlocks = 0;
                int promptBlocks = 0;
                int promptPhysicalStart = -1;
                for (const auto& ref : mPagedKVPlan.refs()) {
                    if (ref.source == RuntimeKVBlockRef::Document) {
                        prefixBlocks++;
                    } else if (ref.source == RuntimeKVBlockRef::Prompt) {
                        promptBlocks++;
                        if (promptPhysicalStart < 0) {
                            promptPhysicalStart = ref.physicalTokenStart;
                        }
                    }
                }
                MNN_PRINT("[CPUPrefixAttention] block_table base_past=%d prefix_tokens=%d prompt_tokens=%d "
                          "required_total=%d physical_total=%d page_size=%d logical_blocks=%d physical_blocks=%d "
                          "prefix_blocks=%d prompt_blocks=%d prompt_physical_start=%d page_pool=1 scratch_gather=1 transitional_contiguous=0\n",
                          basePast, mMeta->segment_total_tokens, promptAppendLen,
                          basePast + mPagedKVPlan.logicalLength(),
                          basePast + mPagedKVPlan.physicalLength(),
                          mPagedKVPlan.pageSize(),
                          UP_DIV(basePast + mPagedKVPlan.logicalLength(), mPagedKVPlan.pageSize()),
                          UP_DIV(basePast + mPagedKVPlan.physicalLength(), mPagedKVPlan.pageSize()),
                          prefixBlocks, promptBlocks, promptPhysicalStart);
            }
            if (basePast == 0) {
                mKVCacheManager->onClear();
            } else if (mKVCacheManager->kvLength() != basePast) {
                MNN_ERROR("[Error]: CPU PrefixAttention direct_segments base history mismatch: meta=%d cache=%d\n",
                          basePast, mKVCacheManager->kvLength());
                return INVALID_VALUE;
            }
            auto code = mKVCacheManager->onAllocPrefixSegments(mMeta, seqLen, mLayerIndex);
            if (code != NO_ERROR) {
                if (basePast == 0) {
                    mKVCacheManager->onClear();
                }
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
