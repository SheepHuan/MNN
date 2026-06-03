#include "PagedAttentionExecution.hpp"
#include "core/Macro.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <float.h>
#include <vector>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

template <typename T = void>
static inline T* pagedDevPtr(const Tensor* t) {
    if (t == nullptr || t->deviceId() == 0) {
        return nullptr;
    }
    return reinterpret_cast<T*>(t->deviceId());
}

template <typename T>
__device__ inline float pagedToFloat(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float pagedToFloat<half>(half v) {
    return __half2float(v);
}

template <typename T>
__device__ inline T pagedFromFloat(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline half pagedFromFloat<half>(float v) {
    return __float2half(v);
}

template <typename T>
__global__ void copyPagedKVKernel(const T* keyInput, const T* valueInput, T* keyCache, T* valueCache,
                                  const int* slotTable, int batch, int inputLen, int insertLen, int kvHeads,
                                  int headDim, int baseLogical, int maxSlots) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z * blockDim.z + threadIdx.z;
    if (d >= headDim || l >= insertLen || bh >= batch * kvHeads) {
        return;
    }
    int b = bh / kvHeads;
    int h = bh % kvHeads;
    int logical = baseLogical + l;
    int slot = slotTable ? slotTable[logical] : logical;
    if (slot < 0 || slot >= maxSlots) {
        return;
    }
    int inputOffset = ((b * inputLen + l) * kvHeads + h) * headDim + d;
    T k = keyInput[inputOffset];
    T v = valueInput[inputOffset];
    int keyOffset = ((slot * batch + b) * kvHeads + h) * headDim + d;
    int valueOffset = ((b * kvHeads + h) * maxSlots + slot) * headDim + d;
    keyCache[keyOffset] = k;
    valueCache[valueOffset] = v;
}

template <typename T>
__global__ void pagedAttentionKernel(const T* query, const T* keyCache, const T* valueCache, T* output,
                                     const T* mask, const int* slotTable, int maskElements, int batch,
                                     int queryLen, int insertLen, int numHeads, int kvHeads, int headDim,
                                     int baseLogical, int kvLen, int maxSlots, float scale) {
    extern __shared__ float smem[];
    float* scores = smem;
    float* reduce = smem + kvLen;
    int q = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    if (b >= batch || h >= numHeads || q >= insertLen) {
        return;
    }

    int group = numHeads / kvHeads;
    int kvHead = h / group;
    int qLogical = baseLogical + q;
    float localMax = -FLT_MAX;
    int maskCols = 0;
    int maskGap = 0;
    if (mask != nullptr && maskElements > 1) {
        maskCols = maskElements >= insertLen * kvLen ? kvLen : insertLen;
        maskGap = kvLen - maskCols;
    }

    for (int k = tid; k < kvLen; k += blockDim.x) {
        if (k > qLogical) {
            scores[k] = -FLT_MAX;
            continue;
        }
        int slot = slotTable ? slotTable[k] : k;
        if (slot < 0 || slot >= maxSlots) {
            scores[k] = -FLT_MAX;
            continue;
        }
        float score = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            int qOffset = ((b * queryLen + q) * numHeads + h) * headDim + d;
            int kOffset = ((slot * batch + b) * kvHeads + kvHead) * headDim + d;
            score += pagedToFloat<T>(query[qOffset]) * pagedToFloat<T>(keyCache[kOffset]);
        }
        score *= scale;
        if (mask != nullptr && maskCols > 0 && k >= maskGap) {
            int col = k - maskGap;
            int maskIdx = q * maskCols + col;
            if (maskIdx >= 0 && maskIdx < maskElements) {
                score += pagedToFloat<T>(mask[maskIdx]);
            }
        }
        scores[k] = score;
        localMax = fmaxf(localMax, score);
    }

    reduce[tid] = localMax;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        }
        __syncthreads();
    }
    float maxScore = reduce[0];

    float localSum = 0.0f;
    for (int k = tid; k < kvLen; k += blockDim.x) {
        if (scores[k] == -FLT_MAX) {
            continue;
        }
        scores[k] = expf(scores[k] - maxScore);
        localSum += scores[k];
    }
    reduce[tid] = localSum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }
    float sum = reduce[0];
    float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int d = tid; d < headDim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < kvLen; ++k) {
            if (scores[k] <= 0.0f) {
                continue;
            }
            int slot = slotTable ? slotTable[k] : k;
            int vOffset = ((b * kvHeads + kvHead) * maxSlots + slot) * headDim + d;
            acc += scores[k] * invSum * pagedToFloat<T>(valueCache[vOffset]);
        }
        int outOffset = (b * queryLen * numHeads * headDim) + q * numHeads * headDim + h * headDim + d;
        output[outOffset] = pagedFromFloat<T>(acc);
    }
}

static inline int reverseCount(const KVMeta* meta) {
    if (meta == nullptr || meta->n_reserve <= 0 || meta->reserve == nullptr) {
        return 0;
    }
    int reverse = meta->computeReverseSize();
    return std::max(0, reverse);
}

CUDAPagedAttention::CUDAPagedAttention(Backend* backend, const MNN::Op* op)
    : Execution(backend), mCudaBackend(static_cast<CUDABackend*>(backend)) {
    auto param = op->main_as_AttentionParam();
    if (param != nullptr) {
        mLayerIndex = param->layer_index();
        mKVSharedLayerIndex = param->kv_shared_layer_index();
        mIsKVShared = mKVSharedLayerIndex >= 0;
    }
    mMeta = static_cast<PagedKVMeta*>(mCudaBackend->getMetaPtr());
    mCache.reset(new SharedPagedCache);
}

ErrorCode CUDAPagedAttention::ensureCache(int maxSlots, int batch, int kvHeads, int headDim) {
    if (maxSlots <= 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0) {
        return INVALID_VALUE;
    }
    if (mCache && mCache->key && mCache->value && mCache->slotTable && mCache->maxSlots == maxSlots &&
        mCache->batch == batch && mCache->kvHeads == kvHeads && mCache->headDim == headDim &&
        mCache->precision == mPrecision) {
        return NO_ERROR;
    }
    if (!mCache) {
        mCache.reset(new SharedPagedCache);
    }
    if (mPrecision == 4) {
        mCache->key.reset(Tensor::createDevice<float>({maxSlots, batch, kvHeads, headDim}));
        mCache->value.reset(Tensor::createDevice<float>({batch, kvHeads, maxSlots, headDim}));
    } else {
        mCache->key.reset(Tensor::createDevice<uint16_t>({maxSlots, batch, kvHeads, headDim}));
        mCache->value.reset(Tensor::createDevice<uint16_t>({batch, kvHeads, maxSlots, headDim}));
    }
    mCache->slotTable.reset(Tensor::createDevice<int>({maxSlots}));
    if (!mCache->key || !mCache->value || !mCache->slotTable) {
        return OUT_OF_MEMORY;
    }
    if (!mCudaBackend->onAcquireBuffer(mCache->key.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    if (!mCudaBackend->onAcquireBuffer(mCache->value.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    if (!mCudaBackend->onAcquireBuffer(mCache->slotTable.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    cudaMemset(pagedDevPtr<void>(mCache->key.get()), 0, static_cast<size_t>(maxSlots) * batch * kvHeads * headDim * mPrecision);
    cudaMemset(pagedDevPtr<void>(mCache->value.get()), 0, static_cast<size_t>(batch) * kvHeads * maxSlots * headDim * mPrecision);
    mCache->maxSlots = maxSlots;
    mCache->batch = batch;
    mCache->kvHeads = kvHeads;
    mCache->headDim = headDim;
    mCache->precision = mPrecision;
    mCache->slotTableVersion = -1;
    mCache->slotTableLength = 0;
    return NO_ERROR;
}

ErrorCode CUDAPagedAttention::syncSlotTable(int requiredSlots) {
    if (!mCache || !mCache->slotTable || requiredSlots > mCache->maxSlots) {
        return OUT_OF_MEMORY;
    }
    std::vector<int> identity;
    const int* hostPtr = nullptr;
    int version = 0;
    if (mMeta != nullptr) {
        if (!mMeta->ensureLogicalCapacity(requiredSlots)) {
            return OUT_OF_MEMORY;
        }
        hostPtr = mMeta->slot_table_host.data();
        version = mMeta->slot_table_version;
    } else {
        identity.resize(requiredSlots);
        for (int i = 0; i < requiredSlots; ++i) {
            identity[i] = i;
        }
        hostPtr = identity.data();
    }
    if (mCache->slotTableVersion == version && mCache->slotTableLength >= requiredSlots && mMeta != nullptr) {
        return NO_ERROR;
    }
    cudaMemcpy(pagedDevPtr<int>(mCache->slotTable.get()), hostPtr, requiredSlots * sizeof(int), cudaMemcpyHostToDevice);
    mCache->slotTableVersion = version;
    mCache->slotTableLength = requiredSlots;
    return NO_ERROR;
}

ErrorCode CUDAPagedAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 3 || outputs.empty()) {
        return INVALID_VALUE;
    }
    mPrecision = mCudaBackend->useFp16() ? 2 : 4;
    auto query = inputs[0];
    auto key = inputs[1];
    mBatch = query->length(0);
    mQuerySeqLen = query->length(1);
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    mNewKvSeqLen = key->length(1);
    if (mHeadDim <= 0 || mKvNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    int maxSlots = mNewKvSeqLen;
    if (mMeta != nullptr) {
        maxSlots = std::max(maxSlots, mMeta->request_capacity > 0 ? mMeta->request_capacity : mMeta->max_tokens);
        if (maxSlots <= 0) {
            maxSlots = static_cast<int>(mMeta->previous) + mNewKvSeqLen;
        }
    }
    return ensureCache(maxSlots, mBatch, mKvNumHead, mHeadDim);
}

ErrorCode CUDAPagedAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const Tensor* mask = inputs.size() > 3 ? inputs[3] : nullptr;

    if (mMeta != nullptr && mMeta->request_capacity <= 0 && !mMeta->request_active) {
        mMeta->beginRequest(std::max(mNewKvSeqLen, mQuerySeqLen));
    }

    int reverse = reverseCount(mMeta);
    int baseLogical = 0;
    int insertLen = mNewKvSeqLen;
    if (mMeta != nullptr) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        baseLogical = static_cast<int>(kept) + reverse;
        insertLen = mMeta->add > 0 ? static_cast<int>(std::min<size_t>(mMeta->add, mNewKvSeqLen)) : mNewKvSeqLen;
    }
    insertLen = std::min(insertLen, mQuerySeqLen);
    int kvLen = baseLogical + insertLen;
    if (kvLen > mCache->maxSlots) {
        MNN_ERROR("CUDAPagedAttention layer %d needs %d slots, cache capacity is %d\n", mLayerIndex, kvLen,
                  mCache->maxSlots);
        return OUT_OF_MEMORY;
    }
    auto err = syncSlotTable(kvLen);
    if (err != NO_ERROR) {
        return err;
    }

    cudaStream_t stream = 0;
    if (!mIsKVShared && insertLen > 0) {
        dim3 block(32, 8, 1);
        dim3 grid(UP_DIV(mHeadDim, block.x), UP_DIV(insertLen, block.y), UP_DIV(mBatch * mKvNumHead, block.z));
        if (mPrecision == 4) {
            copyPagedKVKernel<float><<<grid, block, 0, stream>>>(
                pagedDevPtr<float>(key), pagedDevPtr<float>(value), pagedDevPtr<float>(mCache->key.get()),
                pagedDevPtr<float>(mCache->value.get()), pagedDevPtr<int>(mCache->slotTable.get()), mBatch,
                mNewKvSeqLen, insertLen, mKvNumHead, mHeadDim, baseLogical, mCache->maxSlots);
        } else {
            copyPagedKVKernel<half><<<grid, block, 0, stream>>>(
                pagedDevPtr<half>(key), pagedDevPtr<half>(value), pagedDevPtr<half>(mCache->key.get()),
                pagedDevPtr<half>(mCache->value.get()), pagedDevPtr<int>(mCache->slotTable.get()), mBatch,
                mNewKvSeqLen, insertLen, mKvNumHead, mHeadDim, baseLogical, mCache->maxSlots);
        }
        checkKernelErrors;
    }

    size_t outputBytes = static_cast<size_t>(mBatch) * mQuerySeqLen * mNumHead * mHeadDim * mPrecision;
    cudaMemset(pagedDevPtr<void>(output), 0, outputBytes);
    if (insertLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }

    mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    bool useMask = mask != nullptr && mask->elementSize() > 1 && mask->getType().code == halide_type_float;
    int maskElements = useMask ? static_cast<int>(mask->elementSize()) : 0;
    dim3 grid(insertLen, mNumHead, mBatch);
    int blockSize = 128;
    int sharedBytes = (kvLen + blockSize) * sizeof(float);
    if (mPrecision == 4) {
        pagedAttentionKernel<float><<<grid, blockSize, sharedBytes, stream>>>(
            pagedDevPtr<float>(query), pagedDevPtr<float>(mCache->key.get()), pagedDevPtr<float>(mCache->value.get()),
            pagedDevPtr<float>(output), useMask ? pagedDevPtr<float>(mask) : nullptr,
            pagedDevPtr<int>(mCache->slotTable.get()), maskElements, mBatch, mQuerySeqLen, insertLen, mNumHead,
            mKvNumHead, mHeadDim, baseLogical, kvLen, mCache->maxSlots, mScale);
    } else {
        pagedAttentionKernel<half><<<grid, blockSize, sharedBytes, stream>>>(
            pagedDevPtr<half>(query), pagedDevPtr<half>(mCache->key.get()), pagedDevPtr<half>(mCache->value.get()),
            pagedDevPtr<half>(output), useMask ? pagedDevPtr<half>(mask) : nullptr,
            pagedDevPtr<int>(mCache->slotTable.get()), maskElements, mBatch, mQuerySeqLen, insertLen, mNumHead,
            mKvNumHead, mHeadDim, baseLogical, kvLen, mCache->maxSlots, mScale);
    }
    checkKernelErrors;
    return NO_ERROR;
}

bool CUDAPagedAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    auto tmp = new CUDAPagedAttention(bn, op);
    tmp->mCache = mCache;
    auto param = op->main_as_AttentionParam();
    tmp->mIsKVShared = param != nullptr && param->kv_shared_layer_index() >= 0;
    *dst = tmp;
    return true;
}

class PagedAttentionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CUDAPagedAttention(backend, op);
    }
};

static CUDACreatorRegister<PagedAttentionCreator> __init_paged_attention(OpType_PagedAttention);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
