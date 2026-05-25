//
//  CPUPagedAttention.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CPUPagedAttention.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace MNN {

static inline float _pagedRead(const int8_t* ptr, int index, int bytes) {
#ifdef __aarch64__
    if (bytes == 2) {
        return static_cast<float>(((const __fp16*)ptr)[index]);
    }
#endif
    return ((const float*)ptr)[index];
}

static inline void _pagedWrite(int8_t* ptr, int index, float val, int bytes) {
#ifdef __aarch64__
    if (bytes == 2) {
        ((__fp16*)ptr)[index] = static_cast<__fp16>(val);
        return;
    }
#endif
    ((float*)ptr)[index] = val;
}

static inline int _reverseCount(const KVMeta* meta) {
    if (meta == nullptr || meta->n_reserve <= 0 || meta->reserve == nullptr) {
        return 0;
    }
    int revert = meta->computeReverseSize();
    return std::max(0, revert);
}

static inline float _readFloatMask(const Tensor* mask, int q, int logicalK, int queryLen, int kvLen, int bytes) {
    if (mask == nullptr || mask->elementSize() <= 1) {
        return 0.0f;
    }
    if (mask->getType().code != halide_type_float) {
        return 0.0f;
    }
    int elements = static_cast<int>(mask->elementSize());
    int maskCols = elements >= queryLen * kvLen ? kvLen : queryLen;
    if (maskCols <= 0) {
        return 0.0f;
    }
    int gap = kvLen - maskCols;
    if (logicalK < gap) {
        return 0.0f;
    }
    int col = logicalK - gap;
    int idx = q * maskCols + col;
    if (col < 0 || col >= maskCols || idx < 0 || idx >= elements) {
        return 0.0f;
    }
    return _pagedRead(mask->host<int8_t>(), idx, bytes);
}

CPUPagedAttention::CPUPagedAttention(Backend* backend, const Op* op) : Execution(backend) {
    auto param = op->main_as_AttentionParam();
    if (param != nullptr) {
        mLayerIndex = param->layer_index();
        mKVSharedLayerIndex = param->kv_shared_layer_index();
        mIsKVShared = mKVSharedLayerIndex >= 0;
    }
    mMeta = static_cast<PagedKVMeta*>(backend->getMetaPtr());
    mCache.reset(new LayerCache);
}

ErrorCode CPUPagedAttention::ensureCache(int maxSlots, int batch, int kvHeads, int headDim) {
    if (maxSlots <= 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0) {
        return INVALID_VALUE;
    }
    if (mCache && mCache->key && mCache->value && mCache->maxSlots == maxSlots && mCache->batch == batch &&
        mCache->kvHeads == kvHeads && mCache->headDim == headDim && mCache->bytes == mBytes) {
        return NO_ERROR;
    }

    if (!mCache) {
        mCache.reset(new LayerCache);
    }
    mCache->key.reset(Tensor::createDevice<int8_t>({maxSlots * batch * kvHeads * headDim * mBytes}));
    mCache->value.reset(Tensor::createDevice<int8_t>({batch * kvHeads * maxSlots * headDim * mBytes}));
    if (!mCache->key || !mCache->value) {
        return OUT_OF_MEMORY;
    }
    if (!backend()->onAcquireBuffer(mCache->key.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    if (!backend()->onAcquireBuffer(mCache->value.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    ::memset(mCache->key->host<int8_t>(), 0, mCache->key->elementSize());
    ::memset(mCache->value->host<int8_t>(), 0, mCache->value->elementSize());
    mCache->maxSlots = maxSlots;
    mCache->batch = batch;
    mCache->kvHeads = kvHeads;
    mCache->headDim = headDim;
    mCache->bytes = mBytes;
    return NO_ERROR;
}

ErrorCode CPUPagedAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 3 || outputs.empty()) {
        return INVALID_VALUE;
    }
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mBytes = core->bytes;
    auto query = inputs[0];
    auto key = inputs[1];
    int batch = query->length(0);
    int kvHeads = key->length(2);
    int headDim = query->length(3);
    int required = key->length(1);
    if (mMeta != nullptr) {
        required = std::max(required, mMeta->request_capacity > 0 ? mMeta->request_capacity : mMeta->max_tokens);
        if (required <= 0) {
            required = static_cast<int>(mMeta->previous) + key->length(1);
        }
    }
    return ensureCache(required, batch, kvHeads, headDim);
}

ErrorCode CPUPagedAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    const Tensor* mask = inputs.size() > 3 ? inputs[3] : nullptr;
    auto output = outputs[0];

    int batch = query->length(0);
    int queryLen = query->length(1);
    int numHeads = query->length(2);
    int headDim = query->length(3);
    int newKvLen = key->length(1);
    int kvHeads = key->length(2);
    if (batch != key->length(0) || batch != value->length(0) || kvHeads <= 0 || numHeads % kvHeads != 0) {
        return INVALID_VALUE;
    }
    if (mMeta != nullptr && mMeta->request_capacity <= 0 && !mMeta->request_active) {
        mMeta->beginRequest(std::max(newKvLen, queryLen));
    }

    int reverse = _reverseCount(mMeta);
    int baseLogical = 0;
    int insertLen = newKvLen;
    if (mMeta != nullptr) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        baseLogical = static_cast<int>(kept) + reverse;
        insertLen = mMeta->add > 0 ? static_cast<int>(std::min<size_t>(mMeta->add, newKvLen)) : newKvLen;
    }
    insertLen = std::min(insertLen, queryLen);
    int kvLen = baseLogical + insertLen;
    if (kvLen > mCache->maxSlots) {
        MNN_ERROR("CPUPagedAttention layer %d needs %d slots, cache capacity is %d\n", mLayerIndex, kvLen,
                  mCache->maxSlots);
        return OUT_OF_MEMORY;
    }
    if (mMeta != nullptr && !mMeta->ensureLogicalCapacity(kvLen)) {
        MNN_ERROR("CPUPagedAttention layer %d logical length %d exceeds request capacity %d\n", mLayerIndex, kvLen,
                  mMeta->request_capacity);
        return OUT_OF_MEMORY;
    }

    auto kCache = mCache->key->host<int8_t>();
    auto vCache = mCache->value->host<int8_t>();
    const auto kInput = key->host<int8_t>();
    const auto vInput = value->host<int8_t>();
    std::vector<int> physicalSlots(kvLen);
    for (int k = 0; k < kvLen; ++k) {
        int slot = mMeta ? mMeta->physicalSlot(k) : k;
        if (slot < 0 || slot >= mCache->maxSlots) {
            return OUT_OF_MEMORY;
        }
        physicalSlots[k] = slot;
    }
    if (!mIsKVShared) {
        for (int b = 0; b < batch; ++b) {
            for (int l = 0; l < insertLen; ++l) {
                int slot = physicalSlots[baseLogical + l];
                for (int h = 0; h < kvHeads; ++h) {
                    int inOffset = ((b * newKvLen + l) * kvHeads + h) * headDim;
                    int kOffset = ((slot * batch + b) * kvHeads + h) * headDim;
                    int vOffset = ((b * kvHeads + h) * mCache->maxSlots + slot) * headDim;
                    ::memcpy(kCache + kOffset * mBytes, kInput + inOffset * mBytes, headDim * mBytes);
                    ::memcpy(vCache + vOffset * mBytes, vInput + inOffset * mBytes, headDim * mBytes);
                }
            }
        }
    }
    ::memset(output->host<int8_t>(), 0, output->elementSize() * mBytes);
    if (insertLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }

    mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0f / std::sqrt(static_cast<float>(headDim)));
    const int group = numHeads / kvHeads;
    const auto qInput = query->host<int8_t>();
    auto outPtr = output->host<int8_t>();
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    int totalWork = batch * insertLen * numHeads;
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        std::vector<float> scores(kvLen);
        for (int index = (int)tId; index < totalWork; index += threadNum) {
            int h = index % numHeads;
            int tmp = index / numHeads;
            int q = tmp % insertLen;
            int b = tmp / insertLen;
            int kvHead = h / group;
            int qLogical = baseLogical + q;
            int validLen = std::min(kvLen, qLogical + 1);
            float maxScore = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < validLen; ++k) {
                int slot = physicalSlots[k];
                float score = 0.0f;
                if (mBytes == 4) {
                    const float* qPtr = reinterpret_cast<const float*>(qInput) +
                        ((b * queryLen + q) * numHeads + h) * headDim;
                    const float* kPtr = reinterpret_cast<const float*>(kCache) +
                        ((slot * batch + b) * kvHeads + kvHead) * headDim;
                    for (int d = 0; d < headDim; ++d) {
                        score += qPtr[d] * kPtr[d];
                    }
                } else {
                    for (int d = 0; d < headDim; ++d) {
                        int qOffset = ((b * queryLen + q) * numHeads + h) * headDim + d;
                        int kOffset = ((slot * batch + b) * kvHeads + kvHead) * headDim + d;
                        score += _pagedRead(qInput, qOffset, mBytes) * _pagedRead(kCache, kOffset, mBytes);
                    }
                }
                score = score * mScale + _readFloatMask(mask, q, k, insertLen, kvLen, mBytes);
                scores[k] = score;
                maxScore = std::max(maxScore, score);
            }
            float sum = 0.0f;
            for (int k = 0; k < validLen; ++k) {
                scores[k] = std::exp(scores[k] - maxScore);
                sum += scores[k];
            }
            float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < headDim; ++d) {
                float acc = 0.0f;
                for (int k = 0; k < validLen; ++k) {
                    if (scores[k] <= 0.0f) {
                        continue;
                    }
                    int slot = physicalSlots[k];
                    int vOffset = ((b * kvHeads + kvHead) * mCache->maxSlots + slot) * headDim + d;
                    float v = mBytes == 4 ? reinterpret_cast<const float*>(vCache)[vOffset] : _pagedRead(vCache, vOffset, mBytes);
                    acc += scores[k] * invSum * v;
                }
                int outOffset = (b * queryLen * numHeads * headDim) + q * numHeads * headDim + h * headDim + d;
                if (mBytes == 4) {
                    reinterpret_cast<float*>(outPtr)[outOffset] = acc;
                } else {
                    _pagedWrite(outPtr, outOffset, acc, mBytes);
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

bool CPUPagedAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    auto tmp = new CPUPagedAttention(bn, op);
    tmp->mCache = mCache;
    auto param = op->main_as_AttentionParam();
    tmp->mIsKVShared = param != nullptr && param->kv_shared_layer_index() >= 0;
    *dst = tmp;
    return true;
}

class CPUPagedAttentionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUPagedAttention(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUPagedAttentionCreator, OpType_PagedAttention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
