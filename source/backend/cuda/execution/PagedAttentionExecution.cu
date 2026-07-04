#include "PagedAttentionExecution.hpp"
#include "SoftmaxExecution.hpp"
#include "backend/cuda/core/CUDATools.hpp"
#include "core/Macro.h"
#include "core/MNNFileUtils.h"
#include "core/TensorUtils.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
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

static std::string tensorShapeString(const Tensor* t) {
    if (t == nullptr) {
        return "<null>";
    }
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < t->dimensions(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << t->length(i);
    }
    os << "]";
    return os.str();
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

static bool profilePagedAttention();
static uint64_t nowUs();

template <typename T>
__global__ void copyPagedKVKernel(const T* keyInput, const T* valueInput, T* keyCache, T* valueCache,
                                  int batch, int inputLen, int insertLen, int kvHeads, int headDim,
                                  int baseLogical, int maxSlots, const int* queryLogicalIndices) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z * blockDim.z + threadIdx.z;
    if (d >= headDim || l >= insertLen || bh >= batch * kvHeads) {
        return;
    }
    if (l >= inputLen) {
        return;
    }
    int b = bh / kvHeads;
    int h = bh % kvHeads;
    int logical = queryLogicalIndices ? queryLogicalIndices[l] : (baseLogical + l);
    if (logical < 0 || logical >= maxSlots) {
        return;
    }
    int slot = logical;
    int inputOffset = ((b * inputLen + l) * kvHeads + h) * headDim + d;
    T k = keyInput[inputOffset];
    T v = valueInput[inputOffset];
    int keyOffset = ((slot * batch + b) * kvHeads + h) * headDim + d;
    int valueOffset = ((b * kvHeads + h) * maxSlots + slot) * headDim + d;
    keyCache[keyOffset] = k;
    valueCache[valueOffset] = v;
}

__device__ inline float pagedRopeInvFreq(int pairIndex, int ropeDim, float theta, int ropeType, float factor,
                                         float lowFreqFactor, float highFreqFactor, int oldContext) {
    float invFreq = powf(theta, -static_cast<float>(2 * pairIndex) / static_cast<float>(ropeDim));
    if (ropeType != 1 || oldContext <= 0 || factor == 1.0f || lowFreqFactor == highFreqFactor) {
        return invFreq;
    }
    constexpr float kTwoPi = 6.28318530717958647692f;
    float wavelen = kTwoPi / invFreq;
    float lowFreqWavelen = static_cast<float>(oldContext) / lowFreqFactor;
    float highFreqWavelen = static_cast<float>(oldContext) / highFreqFactor;
    float scaled = wavelen > lowFreqWavelen ? invFreq / factor : invFreq;
    if (wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen) {
        float smooth = (static_cast<float>(oldContext) / wavelen - lowFreqFactor) /
                       (highFreqFactor - lowFreqFactor);
        scaled = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
    }
    return scaled;
}

template <typename T>
__global__ void exportCanonicalPagedKeyKernel(const T* keyCache, T* keyOut, int batch, int kvLen, int kvHeads,
                                              int headDim, int maxSlots, int ropeDim,
                                              float ropeTheta, int ropeType, float factor, float lowFreqFactor,
                                              float highFreqFactor, int oldContext, float attentionScale) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z * blockDim.z + threadIdx.z;
    if (d >= headDim || l >= kvLen || bh >= batch * kvHeads) {
        return;
    }
    int b = bh / kvHeads;
    int h = bh % kvHeads;
    int slot = l;
    int srcBase = ((slot * batch + b) * kvHeads + h) * headDim;
    int dst = ((l * batch + b) * kvHeads + h) * headDim + d;
    float out = pagedToFloat<T>(keyCache[srcBase + d]);
    if (ropeDim > 0 && d < ropeDim) {
        int half = ropeDim / 2;
        int pair = d < half ? d : d - half;
        float angle = static_cast<float>(l) *
                      pagedRopeInvFreq(pair, ropeDim, ropeTheta, ropeType, factor, lowFreqFactor,
                                       highFreqFactor, oldContext);
        float c = cosf(angle);
        float s = sinf(angle);
        float y0 = pagedToFloat<T>(keyCache[srcBase + pair]);
        float y1 = pagedToFloat<T>(keyCache[srcBase + pair + half]);
        float invScale = attentionScale > 0.0f ? 1.0f / attentionScale : 1.0f;
        out = d < half ? (y0 * c + y1 * s) * invScale : (y1 * c - y0 * s) * invScale;
    }
    keyOut[dst] = pagedFromFloat<T>(out);
}

template <typename T>
__global__ void hydratePagedKeyKernel(const T* keyIn, T* keyCache, int batch, int tokenCount, int kvHeads,
                                      int headDim, int maxSlots, int logicalStart, int ropeDim,
                                      float ropeTheta, int ropeType, float factor, float lowFreqFactor,
                                      float highFreqFactor, int oldContext, float attentionScale, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int tmp = idx / headDim;
    int h = tmp % kvHeads;
    tmp /= kvHeads;
    int b = tmp % batch;
    int local = tmp / batch;
    if (local >= tokenCount) {
        return;
    }
    int logical = logicalStart + local;
    if (logical < 0 || logical >= maxSlots) {
        return;
    }
    int slot = logical;
    int srcBase = ((local * batch + b) * kvHeads + h) * headDim;
    int dst = ((slot * batch + b) * kvHeads + h) * headDim + d;
    if (ropeDim > 0 && d < ropeDim) {
        int half = ropeDim / 2;
        if (d >= half) {
            return;
        }
        int pair = d;
        float angle = static_cast<float>(logical) *
                      pagedRopeInvFreq(pair, ropeDim, ropeTheta, ropeType, factor, lowFreqFactor,
                                       highFreqFactor, oldContext);
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = pagedToFloat<T>(keyIn[srcBase + pair]);
        float x1 = pagedToFloat<T>(keyIn[srcBase + pair + half]);
        float scale = attentionScale > 0.0f ? attentionScale : 1.0f;
        keyCache[dst] = pagedFromFloat<T>((x0 * c - x1 * s) * scale);
        keyCache[dst + half] = pagedFromFloat<T>((x1 * c + x0 * s) * scale);
        return;
    }
    keyCache[dst] = keyIn[srcBase + d];
}

template <typename T>
__global__ void hydratePagedValueKernel(const T* valueIn, T* valueCache, int batch, int tokenCount, int kvHeads,
                                        int headDim, int maxSlots, int logicalStart, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int tmp = idx / headDim;
    int local = tmp % tokenCount;
    tmp /= tokenCount;
    int h = tmp % kvHeads;
    int b = tmp / kvHeads;
    int logical = logicalStart + local;
    if (b >= batch || logical < 0 || logical >= maxSlots) {
        return;
    }
    int slot = logical;
    int src = ((b * kvHeads + h) * tokenCount + local) * headDim + d;
    int dst = ((b * kvHeads + h) * maxSlots + slot) * headDim + d;
    valueCache[dst] = valueIn[src];
}

template <typename T>
__global__ void cacheBlendValueScoreKernel(const T* referenceValueCache, const T* cachedValue, float* scores,
                                           int batch, int tokenCount, int kvHeads, int headDim,
                                           int maxSlots, int logicalStart, int scoreOffset) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= tokenCount) {
        return;
    }
    int logical = logicalStart + local;
    if (logical < 0 || logical >= maxSlots) {
        scores[scoreOffset + local] = -FLT_MAX;
        return;
    }
    int slot = logical;
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            int refBase = ((b * kvHeads + h) * maxSlots + slot) * headDim;
            int cachedBase = ((b * kvHeads + h) * tokenCount + local) * headDim;
            for (int d = 0; d < headDim; ++d) {
                acc += fabsf(pagedToFloat<T>(referenceValueCache[refBase + d]) -
                             pagedToFloat<T>(cachedValue[cachedBase + d]));
            }
        }
    }
    const int denomInt = batch * kvHeads * headDim > 0 ? batch * kvHeads * headDim : 1;
    const float denom = static_cast<float>(denomInt);
    scores[scoreOffset + local] = acc / denom;
}

template <typename T>
__global__ void cacheBlendValueScoreFromPagedSourceKernel(const T* valueCache, float* scores,
                                                          int batch, int tokenCount, int kvHeads, int headDim,
                                                          int maxSlots, int sourceSlotStart, int logicalStart,
                                                          int scoreOffset) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= tokenCount) {
        return;
    }
    int logical = logicalStart + local;
    int sourceSlot = sourceSlotStart + local;
    if (logical < 0 || logical >= maxSlots || sourceSlot < 0 || sourceSlot >= maxSlots) {
        scores[scoreOffset + local] = -FLT_MAX;
        return;
    }
    int slot = logical;
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            int refBase = ((b * kvHeads + h) * maxSlots + slot) * headDim;
            int sourceBase = ((b * kvHeads + h) * maxSlots + sourceSlot) * headDim;
            for (int d = 0; d < headDim; ++d) {
                acc += fabsf(pagedToFloat<T>(valueCache[refBase + d]) -
                             pagedToFloat<T>(valueCache[sourceBase + d]));
            }
        }
    }
    const int denomInt = batch * kvHeads * headDim > 0 ? batch * kvHeads * headDim : 1;
    const float denom = static_cast<float>(denomInt);
    scores[scoreOffset + local] = acc / denom;
}

__global__ void cacheBlendTopKKernel(const float* scores, int* selected, unsigned char* used, int tokenCount,
                                     int topK) {
    __shared__ float bestValues[256];
    __shared__ int bestIndices[256];
    int tid = threadIdx.x;
    if (blockIdx.x != 0 || tid >= 256) {
        return;
    }
    for (int k = 0; k < topK; ++k) {
        float best = -FLT_MAX;
        int bestIndex = -1;
        for (int i = tid; i < tokenCount; i += blockDim.x) {
            if (used != nullptr && used[i] != 0) {
                continue;
            }
            float value = scores[i];
            if (value != value) {
                value = -FLT_MAX;
            }
            if (bestIndex < 0 || value > best || (value == best && i < bestIndex)) {
                best = value;
                bestIndex = i;
            }
        }
        bestValues[tid] = best;
        bestIndices[tid] = bestIndex;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float otherValue = bestValues[tid + stride];
                int otherIndex = bestIndices[tid + stride];
                if (otherValue > bestValues[tid] ||
                    (otherValue == bestValues[tid] && otherIndex >= 0 &&
                     (bestIndices[tid] < 0 || otherIndex < bestIndices[tid]))) {
                    bestValues[tid] = otherValue;
                    bestIndices[tid] = otherIndex;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int chosen = bestIndices[0];
            selected[k] = chosen;
            if (used != nullptr && chosen >= 0 && chosen < tokenCount) {
                used[chosen] = 1;
                __threadfence_block();
            }
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void decodeAttentionPicScoreKernel(const T* query, const T* keyCache, const int* headIds,
                                              float* scores, int batch, int queryLen,
                                              int numHeads, int kvHeads, int headDim, int maxSlots,
                                              int picStart, int picTokenCount, int qRow, int qLogical,
                                              int selectedHeadCount, float scale) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= picTokenCount) {
        return;
    }
    const int logical = picStart + local;
    if (logical < 0 || logical > qLogical || logical >= maxSlots || qRow < 0 || qRow >= queryLen ||
        numHeads <= 0 || kvHeads <= 0 || headDim <= 0) {
        scores[local] = -FLT_MAX;
        return;
    }
    const int slot = logical;
    const int group = numHeads / kvHeads;
    if (group <= 0) {
        scores[local] = -FLT_MAX;
        return;
    }
    float acc = 0.0f;
    int used = 0;
    const int loopHeads = selectedHeadCount > 0 ? selectedHeadCount : numHeads;
    for (int hi = 0; hi < loopHeads; ++hi) {
        const int h = selectedHeadCount > 0 ? headIds[hi] : hi;
        if (h < 0 || h >= numHeads) {
            continue;
        }
        const int kvHead = h / group;
        if (kvHead < 0 || kvHead >= kvHeads) {
            continue;
        }
        for (int b = 0; b < batch; ++b) {
            float score = 0.0f;
            const int qBase = ((b * queryLen + qRow) * numHeads + h) * headDim;
            const int kBase = ((slot * batch + b) * kvHeads + kvHead) * headDim;
            for (int d = 0; d < headDim; ++d) {
                score += pagedToFloat<T>(query[qBase + d]) * pagedToFloat<T>(keyCache[kBase + d]);
            }
            acc += score * scale;
            ++used;
        }
    }
    scores[local] = used > 0 ? acc / static_cast<float>(used) : -FLT_MAX;
}

struct DecodeAttentionRankCaptureCUDA {
    bool requested = false;
    bool fused = false;
    int picStart = 0;
    int picTokenCount = 0;
    int topM = 0;
    int qRow = 0;
    int qLogical = 0;
    const int* headIdsDevice = nullptr;
    int headIdCount = 0;
    int validHeadCount = 0;
    float scoreScale = 1.0f;
};

static ErrorCode ensureDecodeAttentionRankWorkspacesCUDA(int picTokenCount, int topM,
                                                         float** scoreWorkspace, size_t* scoreWorkspaceCount,
                                                         int** indexWorkspace, size_t* indexWorkspaceCount,
                                                         unsigned char** usedWorkspace,
                                                         size_t* usedWorkspaceCount) {
    if (picTokenCount <= 0 || topM <= 0 || scoreWorkspace == nullptr || scoreWorkspaceCount == nullptr ||
        indexWorkspace == nullptr || indexWorkspaceCount == nullptr || usedWorkspace == nullptr ||
        usedWorkspaceCount == nullptr) {
        return INVALID_VALUE;
    }
    if (*scoreWorkspace == nullptr || *scoreWorkspaceCount < static_cast<size_t>(picTokenCount)) {
        if (*scoreWorkspace != nullptr) {
            cudaFree(*scoreWorkspace);
            *scoreWorkspace = nullptr;
            *scoreWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(scoreWorkspace), static_cast<size_t>(picTokenCount) * sizeof(float)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *scoreWorkspaceCount = static_cast<size_t>(picTokenCount);
    }
    if (*indexWorkspace == nullptr || *indexWorkspaceCount < static_cast<size_t>(topM)) {
        if (*indexWorkspace != nullptr) {
            cudaFree(*indexWorkspace);
            *indexWorkspace = nullptr;
            *indexWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(indexWorkspace), static_cast<size_t>(topM) * sizeof(int)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *indexWorkspaceCount = static_cast<size_t>(topM);
    }
    if (*usedWorkspace == nullptr || *usedWorkspaceCount < static_cast<size_t>(picTokenCount)) {
        if (*usedWorkspace != nullptr) {
            cudaFree(*usedWorkspace);
            *usedWorkspace = nullptr;
            *usedWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(usedWorkspace), static_cast<size_t>(picTokenCount)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *usedWorkspaceCount = static_cast<size_t>(picTokenCount);
    }
    return NO_ERROR;
}

static ErrorCode finishDecodeAttentionRankCaptureCUDA(PagedKVMeta* meta, int layerIndex, int picTokenCount,
                                                       int topM, int qLogical, int headCount,
                                                       float* scoreWorkspace, int* indexWorkspace,
                                                       unsigned char* usedWorkspace, bool fusedScoreInAttention,
                                                       uint64_t prepUs) {
    if (meta == nullptr || scoreWorkspace == nullptr || indexWorkspace == nullptr || usedWorkspace == nullptr ||
        picTokenCount <= 0 || topM <= 0) {
        return INVALID_VALUE;
    }
    const bool profile = profilePagedAttention();
    uint64_t topKInitUs = 0;
    uint64_t topKUs = 0;
    uint64_t d2hUs = 0;
    const uint64_t totalStartUs = profile ? nowUs() : 0;
    uint64_t topKInitStartUs = profile ? nowUs() : 0;
    if (cudaMemset(usedWorkspace, 0, static_cast<size_t>(picTokenCount)) != cudaSuccess) {
        return INVALID_VALUE;
    }
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        topKInitUs = nowUs() - topKInitStartUs;
    }
    uint64_t topKStartUs = profile ? nowUs() : 0;
    cacheBlendTopKKernel<<<1, 256>>>(scoreWorkspace, indexWorkspace, usedWorkspace, picTokenCount, topM);
    if (cudaGetLastError() != cudaSuccess) {
        return INVALID_VALUE;
    }
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        topKUs = nowUs() - topKStartUs;
    }
    std::vector<int> selected(static_cast<size_t>(topM));
    uint64_t d2hStartUs = profile ? nowUs() : 0;
    if (cudaMemcpy(selected.data(), indexWorkspace, static_cast<size_t>(topM) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return INVALID_VALUE;
    }
    if (profile) {
        d2hUs = nowUs() - d2hStartUs;
    }
    meta->setPicDecodeAttentionRankResult(selected, meta->pic_decode_attention_step_idx);
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        MNN_PRINT("CUDAPagedAttention profile op=decode_attention_rank layer=%d pic_tokens=%d top_m=%d "
                  "heads=%d q_logical=%d fused_score_in_attention=%d us=%llu prep_us=%llu "
                  "topk_init_us=%llu topk_us=%llu d2h_us=%llu\n",
                  layerIndex, picTokenCount, topM, headCount, qLogical, fusedScoreInAttention ? 1 : 0,
                  static_cast<unsigned long long>(nowUs() - totalStartUs + prepUs),
                  static_cast<unsigned long long>(prepUs),
                  static_cast<unsigned long long>(topKInitUs),
                  static_cast<unsigned long long>(topKUs),
                  static_cast<unsigned long long>(d2hUs));
    }
    return NO_ERROR;
}

template <typename T>
__global__ void pagedAttentionKernel(const T* query, const T* keyCache, const T* valueCache, T* output,
                                     const float* mask, int maskElements, int batch,
                                     int queryLen, int outputLen, int insertLen, int numHeads, int kvHeads, int headDim,
                                     int baseLogical, int kvLen, int maxSlots, float scale,
                                     const int* queryLogicalIndices, int queryRowsAreFull) {
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
    int qLogical = queryLogicalIndices ? queryLogicalIndices[q] : (baseLogical + q);
    int qRow = queryRowsAreFull ? qLogical : q;
    if (qRow < 0 || qRow >= queryLen) {
        return;
    }
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
        int slot = k;
        float score = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            int qOffset = ((b * queryLen + qRow) * numHeads + h) * headDim + d;
            int kOffset = ((slot * batch + b) * kvHeads + kvHead) * headDim + d;
            score += pagedToFloat<T>(query[qOffset]) * pagedToFloat<T>(keyCache[kOffset]);
        }
        score *= scale;
        if (mask != nullptr && maskCols > 0 && k >= maskGap) {
            int col = k - maskGap;
            int maskIdx = (queryRowsAreFull ? qLogical : q) * maskCols + col;
            if (maskIdx >= 0 && maskIdx < maskElements) {
                score += mask[maskIdx];
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
            int slot = k;
            int vOffset = ((b * kvHeads + kvHead) * maxSlots + slot) * headDim + d;
            acc += scores[k] * invSum * pagedToFloat<T>(valueCache[vOffset]);
        }
        int outOffset = (b * outputLen * numHeads * headDim) + q * numHeads * headDim + h * headDim + d;
        output[outOffset] = pagedFromFloat<T>(acc);
    }
}

template <typename T>
__global__ void pagedAttentionRowCompressedMaskKernel(
    const T* query, const T* keyCache, const T* valueCache, T* output, const float* mask,
    int maskElements, int batch, int queryLen, int outputLen, int insertLen, int numHeads, int kvHeads, int headDim,
    int baseLogical, int kvLen, int maxSlots, float scale, const int* queryLogicalIndices,
    int queryRowsAreFull, float* decodeAttentionScores, int decodeAttentionPicStart,
    int decodeAttentionPicTokenCount, int decodeAttentionQueryIndex, const int* decodeAttentionHeadIds,
    int decodeAttentionHeadIdCount, float decodeAttentionScoreScale) {
    extern __shared__ float smem[];
    float* qShared = smem;
    float* outShared = qShared + headDim;
    float* scoreTile = outShared + headDim;
    float* reduce = scoreTile + blockDim.x;
    int q = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    if (b >= batch || h >= numHeads || q >= insertLen) {
        return;
    }

    int group = numHeads / kvHeads;
    int kvHead = h / group;
    const bool captureDecodeRank = decodeAttentionScores != nullptr && decodeAttentionPicTokenCount > 0 &&
        q == decodeAttentionQueryIndex;
    bool captureHead = captureDecodeRank;
    if (captureDecodeRank && decodeAttentionHeadIdCount > 0) {
        captureHead = false;
        if (decodeAttentionHeadIds != nullptr) {
            for (int i = 0; i < decodeAttentionHeadIdCount; ++i) {
                if (decodeAttentionHeadIds[i] == h) {
                    captureHead = true;
                    break;
                }
            }
        }
    }
    int qLogical = queryLogicalIndices ? queryLogicalIndices[q] : (baseLogical + q);
    int qRow = queryRowsAreFull ? qLogical : q;
    if (qRow < 0 || qRow >= queryLen) {
        return;
    }
    int rowEnd = qLogical + 1;
    rowEnd = rowEnd < 0 ? 0 : rowEnd;
    rowEnd = rowEnd > kvLen ? kvLen : rowEnd;
    int maskCols = 0;
    int maskGap = 0;
    if (mask != nullptr && maskElements > 1) {
        maskCols = maskElements >= insertLen * kvLen ? kvLen : insertLen;
        maskGap = kvLen - maskCols;
    }

    int qBase = ((b * queryLen + qRow) * numHeads + h) * headDim;
    for (int d = tid; d < headDim; d += blockDim.x) {
        qShared[d] = pagedToFloat<T>(query[qBase + d]);
        outShared[d] = 0.0f;
    }
    __syncthreads();

    float runningMax = -FLT_MAX;
    float runningSum = 0.0f;
    for (int kStart = 0; kStart < rowEnd; kStart += blockDim.x) {
        int tileCount = rowEnd - kStart;
        tileCount = tileCount > static_cast<int>(blockDim.x) ? static_cast<int>(blockDim.x) : tileCount;

        float localScore = -FLT_MAX;
        if (tid < tileCount) {
            int k = kStart + tid;
            int slot = k;
            localScore = 0.0f;
            for (int d = 0; d < headDim; ++d) {
                int kOffset = ((slot * batch + b) * kvHeads + kvHead) * headDim + d;
                localScore += qShared[d] * pagedToFloat<T>(keyCache[kOffset]);
            }
            localScore *= scale;
            if (mask != nullptr && maskCols > 0 && k >= maskGap) {
                int col = k - maskGap;
                int maskIdx = (queryRowsAreFull ? qLogical : q) * maskCols + col;
                if (maskIdx >= 0 && maskIdx < maskElements) {
                    localScore += mask[maskIdx];
                }
            }
            if (captureHead && k >= decodeAttentionPicStart &&
                k < decodeAttentionPicStart + decodeAttentionPicTokenCount &&
                localScore != -FLT_MAX) {
                atomicAdd(decodeAttentionScores + (k - decodeAttentionPicStart),
                          localScore * decodeAttentionScoreScale);
            }
            scoreTile[tid] = localScore;
        }

        reduce[tid] = localScore;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            }
            __syncthreads();
        }
        float tileMax = reduce[0];

        float localSum = 0.0f;
        if (tid < tileCount && tileMax != -FLT_MAX) {
            float score = scoreTile[tid];
            if (score != -FLT_MAX) {
                localSum = expf(score - tileMax);
            }
        }
        reduce[tid] = localSum;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        float tileSum = reduce[0];

        if (tileSum > 0.0f) {
            float newMax = runningSum > 0.0f ? fmaxf(runningMax, tileMax) : tileMax;
            float oldScale = runningSum > 0.0f ? expf(runningMax - newMax) : 0.0f;
            float tileScale = expf(tileMax - newMax);
            for (int d = tid; d < headDim; d += blockDim.x) {
                float tileAcc = 0.0f;
                for (int kk = 0; kk < tileCount; ++kk) {
                    float score = scoreTile[kk];
                    if (score == -FLT_MAX) {
                        continue;
                    }
                    int k = kStart + kk;
                    int slot = k;
                    int vOffset = ((b * kvHeads + kvHead) * maxSlots + slot) * headDim + d;
                    tileAcc += expf(score - tileMax) * pagedToFloat<T>(valueCache[vOffset]);
                }
                outShared[d] = outShared[d] * oldScale + tileAcc * tileScale;
            }
            runningSum = runningSum * oldScale + tileSum * tileScale;
            runningMax = newMax;
        }
        __syncthreads();
    }

    float invSum = runningSum > 0.0f ? 1.0f / runningSum : 0.0f;
    for (int d = tid; d < headDim; d += blockDim.x) {
        int outOffset = (b * outputLen * numHeads * headDim) + q * numHeads * headDim + h * headDim + d;
        output[outOffset] = pagedFromFloat<T>(outShared[d] * invSum);
    }
}

static __device__ __forceinline__ float pagedWarpMax(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return __shfl_sync(0xffffffff, v, 0);
}

static __device__ __forceinline__ float pagedWarpSum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return __shfl_sync(0xffffffff, v, 0);
}

enum CudaSparseQTileVariant : int {
    kCudaSparseQTileNone = 0,
    kCudaSparseQTileHD64Q4K16 = 2,
    kCudaSparseQTileHD128Q4K16 = 3,
    kCudaSparseQTileHD128Q4K8 = 4,
    kCudaSparseQTileHD128Q8K16 = 5,
    kCudaSparseQTileHD128Q8K8 = 6,
    kCudaSparseQTileHD128Q2K16 = 7,
    kCudaSparseQTileHD128Q2K8 = 8,
    kCudaSparseQTileHD128Q16K16 = 9,
    kCudaSparseQTileHD128Q16K8 = 10,
};

static const char* cudaSparseQTileVariantName(CudaSparseQTileVariant variant) {
    switch (variant) {
        case kCudaSparseQTileHD64Q4K16:
            return "hd64_q4k16";
        case kCudaSparseQTileHD128Q4K16:
            return "hd128_q4k16";
        case kCudaSparseQTileHD128Q4K8:
            return "hd128_q4k8";
        case kCudaSparseQTileHD128Q8K16:
            return "hd128_q8k16";
        case kCudaSparseQTileHD128Q8K8:
            return "hd128_q8k8";
        case kCudaSparseQTileHD128Q2K16:
            return "hd128_q2k16";
        case kCudaSparseQTileHD128Q2K8:
            return "hd128_q2k8";
        case kCudaSparseQTileHD128Q16K16:
            return "hd128_q16k16";
        case kCudaSparseQTileHD128Q16K8:
            return "hd128_q16k8";
        case kCudaSparseQTileNone:
        default:
            return "none";
    }
}

static CudaSparseQTileVariant selectCudaSparseQTileVariant(int headDim, int attnLen, bool cacheBlendSparse,
                                                          bool fixedPlanSparse) {
    // Very small active-row batches are launch/occupancy bound on Xavier; keep
    // the existing non-qtile sparse attention implementation.
    const bool tinyHD128SparseRows = headDim == 128 && attnLen < 64;
    if (tinyHD128SparseRows) {
        return kCudaSparseQTileNone;
    }
    if (headDim != 128) {
        return kCudaSparseQTileNone;
    }
    // CacheBlend selected rows are often scattered across the full PIC span, so
    // keep its accepted q4/k16 default. Fixed-plan sparse rows are more regular
    // and can use wider-Q variants without adding a separate implementation.
    if (fixedPlanSparse && !cacheBlendSparse) {
        if (attnLen >= 384) {
            return kCudaSparseQTileHD128Q16K16;
        }
        if (attnLen >= 256) {
            return kCudaSparseQTileHD128Q8K16;
        }
        if (attnLen >= 128) {
            return kCudaSparseQTileHD128Q8K8;
        }
        return kCudaSparseQTileHD128Q4K8;
    }
    return kCudaSparseQTileHD128Q4K16;
}

template <typename T, int HEAD_DIM, int Q_TILE, int K_TILE, int DLANES = 16>
__global__ void pagedSparseFlashMQTileKernel(
    const T* query, const T* keyCache, const T* valueCache, T* output, const float* mask,
    int maskElements, int batch, int queryLen, int outputLen, int insertLen, int numHeads, int kvHeads,
    int baseLogical, int kvLen, int maxSlots, float scale, const int* queryLogicalIndices, int queryRowsAreFull,
    float* decodeAttentionScores, int decodeAttentionPicStart, int decodeAttentionPicTokenCount,
    int decodeAttentionQueryIndex, const int* decodeAttentionHeadIds, int decodeAttentionHeadIdCount,
    float decodeAttentionScoreScale) {
    static_assert(HEAD_DIM % DLANES == 0, "HEAD_DIM must be divisible by DLANES");
    constexpr int VEC = HEAD_DIM / DLANES;
    __shared__ T kShared[K_TILE][HEAD_DIM];
    __shared__ T vShared[K_TILE][HEAD_DIM];
    __shared__ float partialShared[Q_TILE][K_TILE][DLANES];
    __shared__ float betaShared[Q_TILE][K_TILE];
    __shared__ float outShared[Q_TILE][HEAD_DIM];
    __shared__ int qLogicalShared[Q_TILE];
    __shared__ int qRowShared[Q_TILE];
    __shared__ int qValidShared[Q_TILE];
    __shared__ int kSlotShared[K_TILE];
    __shared__ int qMinLogicalShared;
    __shared__ int qMaxLogicalShared;
    __shared__ float rowMaxShared[Q_TILE];
    __shared__ float rowSumShared[Q_TILE];
    __shared__ float rowAlphaShared[Q_TILE];
    __shared__ float rowNewMaxShared[Q_TILE];
    __shared__ int tileValidShared[Q_TILE];

    const int lane = threadIdx.x;
    const int qLane = threadIdx.y;
    const int linearTid = qLane * blockDim.x + lane;
    const int linearThreads = blockDim.x * blockDim.y;
    const int qBaseTile = blockIdx.x * Q_TILE;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch || h >= numHeads || lane >= DLANES || qLane >= Q_TILE) {
        return;
    }

    const int group = numHeads / kvHeads;
    const int kvHead = h / group;
    const int q = qBaseTile + qLane;
    const bool captureDecodeRank = decodeAttentionScores != nullptr && decodeAttentionPicTokenCount > 0 &&
        q == decodeAttentionQueryIndex;
    bool captureHead = captureDecodeRank;
    if (captureDecodeRank && decodeAttentionHeadIdCount > 0) {
        captureHead = false;
        if (decodeAttentionHeadIds != nullptr) {
            for (int i = 0; i < decodeAttentionHeadIdCount; ++i) {
                if (decodeAttentionHeadIds[i] == h) {
                    captureHead = true;
                    break;
                }
            }
        }
    }
    const int qLogical = q < insertLen ? (queryLogicalIndices ? queryLogicalIndices[q] : (baseLogical + q)) : -1;
    const int qRow = queryRowsAreFull ? qLogical : q;
    const int qValid = (q < insertLen && qRow >= 0 && qRow < queryLen) ? 1 : 0;
    if (lane == 0) {
        qLogicalShared[qLane] = qLogical;
        qRowShared[qLane] = qRow;
        qValidShared[qLane] = qValid;
        rowMaxShared[qLane] = -FLT_MAX;
        rowSumShared[qLane] = 0.0f;
        rowAlphaShared[qLane] = 0.0f;
        rowNewMaxShared[qLane] = -FLT_MAX;
        tileValidShared[qLane] = 0;
        if (qLane == 0) {
            qMinLogicalShared = kvLen;
            qMaxLogicalShared = -1;
        }
    }
    __syncthreads();

    if (linearTid == 0) {
        int qMinLogical = kvLen;
        int qMaxLogical = -1;
#pragma unroll
        for (int i = 0; i < Q_TILE; ++i) {
            const int logical = qLogicalShared[i];
            if (!qValidShared[i] || logical < 0) {
                continue;
            }
            qMinLogical = min(qMinLogical, logical);
            qMaxLogical = max(qMaxLogical, logical);
        }
        qMinLogicalShared = qMaxLogical >= 0 ? qMinLogical : -1;
        qMaxLogicalShared = qMaxLogical;
    }
    __syncthreads();

    float qFrag[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        qFrag[i] = 0.0f;
    }
    if (qValid) {
        const int qBase = ((b * queryLen + qRow) * numHeads + h) * HEAD_DIM + lane * VEC;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            qFrag[i] = pagedToFloat<T>(query[qBase + i]);
        }
    }

    for (int idx = linearTid; idx < Q_TILE * HEAD_DIM; idx += linearThreads) {
        const int qLocal = idx / HEAD_DIM;
        const int d = idx - qLocal * HEAD_DIM;
        outShared[qLocal][d] = 0.0f;
    }
    __syncthreads();

    int maskCols = 0;
    int maskGap = 0;
    if (mask != nullptr && maskElements > 1) {
        maskCols = maskElements >= insertLen * kvLen ? kvLen : insertLen;
        maskGap = kvLen - maskCols;
    }

    int causalKLimit = qMaxLogicalShared + 1;
    if (causalKLimit < 0) {
        causalKLimit = 0;
    }
    if (causalKLimit > kvLen) {
        causalKLimit = kvLen;
    }
    const T zero = pagedFromFloat<T>(0.0f);
    for (int kBase = 0; kBase < causalKLimit; kBase += K_TILE) {
        const int kTile = min(K_TILE, causalKLimit - kBase);
        const int kTileEnd = kBase + kTile - 1;
        const bool kTileFull = qMinLogicalShared >= 0 && kTileEnd <= qMinLogicalShared;

        if (linearTid < K_TILE) {
            const int logical = kBase + linearTid;
            int slot = -1;
            if (logical < causalKLimit) {
                slot = logical;
            }
            kSlotShared[linearTid] = slot;
        }
        __syncthreads();

        for (int idx = linearTid; idx < K_TILE * HEAD_DIM; idx += linearThreads) {
            const int kLocal = idx / HEAD_DIM;
            const int d = idx - kLocal * HEAD_DIM;
            const int slot = kSlotShared[kLocal];
            T kValue = zero;
            T vValue = zero;
            if (slot >= 0 && kLocal < kTile) {
                const int kOffset = ((slot * batch + b) * kvHeads + kvHead) * HEAD_DIM + d;
                const int vOffset = ((b * kvHeads + kvHead) * maxSlots + slot) * HEAD_DIM + d;
                kValue = keyCache[kOffset];
                vValue = valueCache[vOffset];
            }
            kShared[kLocal][d] = kValue;
            vShared[kLocal][d] = vValue;
        }
        __syncthreads();

#pragma unroll
        for (int kLocal = 0; kLocal < K_TILE; ++kLocal) {
            float partial = 0.0f;
            if (qValid && kLocal < kTile) {
                const int dBase = lane * VEC;
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    partial += qFrag[i] * pagedToFloat<T>(kShared[kLocal][dBase + i]);
                }
            }
            partialShared[qLane][kLocal][lane] = partial;
        }
        __syncthreads();

        if (lane < K_TILE) {
            float score = -FLT_MAX;
            if (qValid && lane < kTile) {
                const int logical = kBase + lane;
                if (kTileFull || logical <= qLogical) {
                    score = 0.0f;
#pragma unroll
                    for (int dLane = 0; dLane < DLANES; ++dLane) {
                        score += partialShared[qLane][lane][dLane];
                    }
                    score *= scale;
                    if (mask != nullptr && maskCols > 0 && logical >= maskGap) {
                        const int col = logical - maskGap;
                        const int maskRow = queryRowsAreFull ? qLogical : q;
                        const int maskIdx = maskRow * maskCols + col;
                        if (maskIdx >= 0 && maskIdx < maskElements) {
                            score += mask[maskIdx];
                        }
                    }
                    if (captureHead && logical >= decodeAttentionPicStart &&
                        logical < decodeAttentionPicStart + decodeAttentionPicTokenCount) {
                        atomicAdd(decodeAttentionScores + (logical - decodeAttentionPicStart),
                                  score * decodeAttentionScoreScale);
                    }
                }
            }
            betaShared[qLane][lane] = score;
        }
        __syncthreads();

        if (lane == 0) {
            float tileMax = -FLT_MAX;
            if (qValid) {
#pragma unroll
                for (int kLocal = 0; kLocal < K_TILE; ++kLocal) {
                    if (kLocal < kTile) {
                        tileMax = fmaxf(tileMax, betaShared[qLane][kLocal]);
                    }
                }
                const float mergedMax = rowSumShared[qLane] > 0.0f ? fmaxf(rowMaxShared[qLane], tileMax) : tileMax;
                rowAlphaShared[qLane] = rowSumShared[qLane] > 0.0f ? expf(rowMaxShared[qLane] - mergedMax) : 0.0f;
                rowNewMaxShared[qLane] = mergedMax;
            } else {
                rowAlphaShared[qLane] = 0.0f;
                rowNewMaxShared[qLane] = -FLT_MAX;
            }
        }
        __syncthreads();

        if (lane < K_TILE) {
            float beta = 0.0f;
            if (qValid && lane < kTile) {
                const float score = betaShared[qLane][lane];
                if (score != -FLT_MAX) {
                    beta = expf(score - rowNewMaxShared[qLane]);
                }
            }
            betaShared[qLane][lane] = beta;
        }
        __syncthreads();

        if (lane == 0) {
            float tileSum = 0.0f;
            if (qValid) {
#pragma unroll
                for (int kLocal = 0; kLocal < K_TILE; ++kLocal) {
                    if (kLocal < kTile) {
                        tileSum += betaShared[qLane][kLocal];
                    }
                }
                rowSumShared[qLane] = rowSumShared[qLane] * rowAlphaShared[qLane] + tileSum;
                rowMaxShared[qLane] = rowNewMaxShared[qLane];
                tileValidShared[qLane] = tileSum > 0.0f ? 1 : 0;
            } else {
                rowSumShared[qLane] = 0.0f;
                rowMaxShared[qLane] = -FLT_MAX;
                tileValidShared[qLane] = 0;
            }
        }
        __syncthreads();

        for (int idx = linearTid; idx < Q_TILE * HEAD_DIM; idx += linearThreads) {
            const int qLocal = idx / HEAD_DIM;
            const int d = idx - qLocal * HEAD_DIM;
            if (qValidShared[qLocal] && tileValidShared[qLocal]) {
                float tileAcc = 0.0f;
#pragma unroll
                for (int kLocal = 0; kLocal < K_TILE; ++kLocal) {
                    if (kLocal < kTile) {
                        tileAcc += betaShared[qLocal][kLocal] * pagedToFloat<T>(vShared[kLocal][d]);
                    }
                }
                outShared[qLocal][d] = outShared[qLocal][d] * rowAlphaShared[qLocal] + tileAcc;
            }
        }
        __syncthreads();
    }

    for (int idx = linearTid; idx < Q_TILE * HEAD_DIM; idx += linearThreads) {
        const int qLocal = idx / HEAD_DIM;
        const int d = idx - qLocal * HEAD_DIM;
        const int outQ = qBaseTile + qLocal;
        if (qValidShared[qLocal] && outQ < outputLen) {
            const float denom = rowSumShared[qLocal] > 0.0f ? rowSumShared[qLocal] : 1.0f;
            const int outOffset = (b * outputLen * numHeads * HEAD_DIM) + outQ * numHeads * HEAD_DIM + h * HEAD_DIM + d;
            output[outOffset] = pagedFromFloat<T>(outShared[qLocal][d] / denom);
        }
    }
}

static bool launchCudaSparseQTileVariant(CudaSparseQTileVariant variant, int precision, const void* query,
                                         const void* keyCache, const void* valueCache, void* output,
                                         const float* mask, int maskElements, int batch,
                                         int queryLen, int outputLen, int attnLen, int numHeads, int kvHeads,
                                         int baseLogical, int kvLen, int maxSlots, float scale,
                                         const int* queryLogicalIndices, bool queryRowsAreFull,
                                         cudaStream_t stream, float* decodeAttentionScores = nullptr,
                                         int decodeAttentionPicStart = 0, int decodeAttentionPicTokenCount = 0,
                                         int decodeAttentionQueryIndex = -1,
                                         const int* decodeAttentionHeadIds = nullptr,
                                         int decodeAttentionHeadIdCount = 0,
                                         float decodeAttentionScoreScale = 1.0f) {
    switch (variant) {
        case kCudaSparseQTileHD64Q4K16: {
            dim3 block(16, 4, 1);
            dim3 grid(UP_DIV(attnLen, 4), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 64, 4, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 64, 4, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        case kCudaSparseQTileHD128Q4K16: {
            dim3 block(16, 4, 1);
            dim3 grid(UP_DIV(attnLen, 4), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 128, 4, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 128, 4, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        case kCudaSparseQTileHD128Q4K8: {
            dim3 block(16, 4, 1);
            dim3 grid(UP_DIV(attnLen, 4), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 128, 4, 8><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 128, 4, 8><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        case kCudaSparseQTileHD128Q8K16: {
            dim3 block(16, 8, 1);
            dim3 grid(UP_DIV(attnLen, 8), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 128, 8, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 128, 8, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        case kCudaSparseQTileHD128Q8K8: {
            dim3 block(16, 8, 1);
            dim3 grid(UP_DIV(attnLen, 8), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 128, 8, 8><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 128, 8, 8><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        case kCudaSparseQTileHD128Q2K16: {
            dim3 block(16, 2, 1);
            dim3 grid(UP_DIV(attnLen, 2), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 128, 2, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 128, 2, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        case kCudaSparseQTileHD128Q2K8: {
            dim3 block(16, 2, 1);
            dim3 grid(UP_DIV(attnLen, 2), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 128, 2, 8><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 128, 2, 8><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        case kCudaSparseQTileHD128Q16K16: {
            dim3 block(16, 16, 1);
            dim3 grid(UP_DIV(attnLen, 16), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 128, 16, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 128, 16, 16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        case kCudaSparseQTileHD128Q16K8: {
            dim3 block(16, 16, 1);
            dim3 grid(UP_DIV(attnLen, 16), numHeads, batch);
            if (precision == 4) {
                pagedSparseFlashMQTileKernel<float, 128, 16, 8><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(query), reinterpret_cast<const float*>(keyCache),
                    reinterpret_cast<const float*>(valueCache), reinterpret_cast<float*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            } else {
                pagedSparseFlashMQTileKernel<half, 128, 16, 8><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const half*>(query), reinterpret_cast<const half*>(keyCache),
                    reinterpret_cast<const half*>(valueCache), reinterpret_cast<half*>(output), mask,
                    maskElements, batch, queryLen, outputLen, attnLen, numHeads, kvHeads, baseLogical, kvLen, maxSlots,
                    scale, queryLogicalIndices, queryRowsAreFull ? 1 : 0, decodeAttentionScores,
                    decodeAttentionPicStart, decodeAttentionPicTokenCount, decodeAttentionQueryIndex,
                    decodeAttentionHeadIds, decodeAttentionHeadIdCount, decodeAttentionScoreScale);
            }
            return true;
        }
        default:
            return false;
    }
}

template <typename T, int Q_TILE, int K_TILE>
__global__ void pagedSparseFlashTileKernel(
    const T* query, const T* keyCache, const T* valueCache, T* output, const float* mask,
    int maskElements, int batch, int queryLen, int outputLen, int insertLen, int numHeads, int kvHeads,
    int baseLogical, int kvLen, int maxSlots, float scale, const int* queryLogicalIndices, int queryRowsAreFull) {
    constexpr int HEAD_DIM = 64;
    __shared__ float qShared[Q_TILE][HEAD_DIM + 1];
    __shared__ float kShared[K_TILE][HEAD_DIM + 1];
    __shared__ float vShared[K_TILE][HEAD_DIM + 1];
    __shared__ float scoreShared[Q_TILE][K_TILE + 1];
    __shared__ float outShared[Q_TILE][HEAD_DIM + 1];
    __shared__ int qLogicalShared[Q_TILE];
    __shared__ int qRowShared[Q_TILE];
    __shared__ int qValidShared[Q_TILE];
    __shared__ int kSlotShared[K_TILE];
    __shared__ int qMaxLogicalShared;
    __shared__ float runningMax[Q_TILE];
    __shared__ float runningSum[Q_TILE];
    __shared__ float oldScaleShared[Q_TILE];
    __shared__ float tileScaleShared[Q_TILE];
    __shared__ int tileValidShared[Q_TILE];

    const int lane = threadIdx.x;
    const int qLane = threadIdx.y;
    const int linearTid = qLane * blockDim.x + lane;
    const int linearThreads = blockDim.x * blockDim.y;
    const int qBaseTile = blockIdx.x * Q_TILE;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch || h >= numHeads) {
        return;
    }
    const int group = numHeads / kvHeads;
    const int kvHead = h / group;

    if (lane == 0) {
        const int q = qBaseTile + qLane;
        const int qLogical = q < insertLen ? (queryLogicalIndices ? queryLogicalIndices[q] : (baseLogical + q)) : -1;
        const int qRow = queryRowsAreFull ? qLogical : q;
        qLogicalShared[qLane] = qLogical;
        qRowShared[qLane] = qRow;
        qValidShared[qLane] = (q < insertLen && qRow >= 0 && qRow < queryLen) ? 1 : 0;
        runningMax[qLane] = -FLT_MAX;
        runningSum[qLane] = 0.0f;
        oldScaleShared[qLane] = 1.0f;
        tileScaleShared[qLane] = 0.0f;
        tileValidShared[qLane] = 0;
        if (qLane == 0) {
            qMaxLogicalShared = -1;
        }
    }
    __syncthreads();

    if (linearTid == 0) {
        int maxLogical = -1;
#pragma unroll
        for (int i = 0; i < Q_TILE; ++i) {
            if (qValidShared[i] && qLogicalShared[i] > maxLogical) {
                maxLogical = qLogicalShared[i];
            }
        }
        qMaxLogicalShared = maxLogical;
    }
    __syncthreads();

    for (int idx = linearTid; idx < Q_TILE * HEAD_DIM; idx += linearThreads) {
        const int qLocal = idx / HEAD_DIM;
        const int d = idx - qLocal * HEAD_DIM;
        float qValue = 0.0f;
        if (qValidShared[qLocal]) {
            const int qOffset = ((b * queryLen + qRowShared[qLocal]) * numHeads + h) * HEAD_DIM + d;
            qValue = pagedToFloat<T>(query[qOffset]);
        }
        qShared[qLocal][d] = qValue;
        outShared[qLocal][d] = 0.0f;
    }
    __syncthreads();

    int maskCols = 0;
    int maskGap = 0;
    if (mask != nullptr && maskElements > 1) {
        maskCols = maskElements >= insertLen * kvLen ? kvLen : insertLen;
        maskGap = kvLen - maskCols;
    }

    int causalKLimit = qMaxLogicalShared + 1;
    if (causalKLimit < 0) {
        causalKLimit = 0;
    }
    if (causalKLimit > kvLen) {
        causalKLimit = kvLen;
    }
    for (int kStart = 0; kStart < causalKLimit; kStart += K_TILE) {
        if (linearTid < K_TILE) {
            const int logical = kStart + linearTid;
            int slot = -1;
            if (logical < causalKLimit) {
                slot = logical;
            }
            kSlotShared[linearTid] = slot;
        }
        __syncthreads();

        for (int idx = linearTid; idx < K_TILE * HEAD_DIM; idx += linearThreads) {
            const int kk = idx / HEAD_DIM;
            const int d = idx - kk * HEAD_DIM;
            const int slot = kSlotShared[kk];
            float kValue = 0.0f;
            if (slot >= 0) {
                const int kOffset = ((slot * batch + b) * kvHeads + kvHead) * HEAD_DIM + d;
                kValue = pagedToFloat<T>(keyCache[kOffset]);
            }
            kShared[kk][d] = kValue;
        }
        __syncthreads();

        float score = -FLT_MAX;
        const int qLocal = qLane;
        const int kk = lane;
        const int logical = kStart + kk;
        if (qValidShared[qLocal] && logical < causalKLimit && logical <= qLogicalShared[qLocal]) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += qShared[qLocal][d] * kShared[kk][d];
            }
            score = dot * scale;
            if (mask != nullptr && maskCols > 0 && logical >= maskGap) {
                const int col = logical - maskGap;
                const int maskRow = queryRowsAreFull ? qLogicalShared[qLocal] : (qBaseTile + qLocal);
                const int maskIdx = maskRow * maskCols + col;
                if (maskIdx >= 0 && maskIdx < maskElements) {
                    score += mask[maskIdx];
                }
            }
        }
        const float tileMax = pagedWarpMax(score);
        float prob = 0.0f;
        if (score != -FLT_MAX && tileMax != -FLT_MAX) {
            prob = expf(score - tileMax);
        }
        const float tileSum = pagedWarpSum(prob);
        scoreShared[qLocal][kk] = prob;
        if (lane == 0) {
            if (tileSum > 0.0f) {
                const float newMax = runningSum[qLocal] > 0.0f ? fmaxf(runningMax[qLocal], tileMax) : tileMax;
                oldScaleShared[qLocal] = runningSum[qLocal] > 0.0f ? expf(runningMax[qLocal] - newMax) : 0.0f;
                tileScaleShared[qLocal] = expf(tileMax - newMax);
                runningSum[qLocal] = runningSum[qLocal] * oldScaleShared[qLocal] + tileSum * tileScaleShared[qLocal];
                runningMax[qLocal] = newMax;
                tileValidShared[qLocal] = 1;
            } else {
                oldScaleShared[qLocal] = 1.0f;
                tileScaleShared[qLocal] = 0.0f;
                tileValidShared[qLocal] = 0;
            }
        }
        __syncthreads();

        for (int idx = linearTid; idx < K_TILE * HEAD_DIM; idx += linearThreads) {
            const int kLocal = idx / HEAD_DIM;
            const int d = idx - kLocal * HEAD_DIM;
            const int slot = kSlotShared[kLocal];
            float vValue = 0.0f;
            if (slot >= 0) {
                const int vOffset = ((b * kvHeads + kvHead) * maxSlots + slot) * HEAD_DIM + d;
                vValue = pagedToFloat<T>(valueCache[vOffset]);
            }
            vShared[kLocal][d] = vValue;
        }
        __syncthreads();

        for (int idx = linearTid; idx < Q_TILE * HEAD_DIM; idx += linearThreads) {
            const int qAcc = idx / HEAD_DIM;
            const int d = idx - qAcc * HEAD_DIM;
            if (qValidShared[qAcc] && tileValidShared[qAcc]) {
                float tileAcc = 0.0f;
#pragma unroll
                for (int kLocal = 0; kLocal < K_TILE; ++kLocal) {
                    tileAcc += scoreShared[qAcc][kLocal] * vShared[kLocal][d];
                }
                outShared[qAcc][d] = outShared[qAcc][d] * oldScaleShared[qAcc] +
                                     tileAcc * tileScaleShared[qAcc];
            }
        }
        __syncthreads();
    }

    for (int idx = linearTid; idx < Q_TILE * HEAD_DIM; idx += linearThreads) {
        const int qLocal = idx / HEAD_DIM;
        const int d = idx - qLocal * HEAD_DIM;
        const int q = qBaseTile + qLocal;
        if (qValidShared[qLocal] && q < outputLen) {
            const float invSum = runningSum[qLocal] > 0.0f ? 1.0f / runningSum[qLocal] : 0.0f;
            const int outOffset = (b * outputLen * numHeads * HEAD_DIM) + q * numHeads * HEAD_DIM + h * HEAD_DIM + d;
            output[outOffset] = pagedFromFloat<T>(outShared[qLocal][d] * invSum);
        }
    }
}

template <typename T>
__global__ void pagedPrefillQKKernel(const T* query, const T* keyCache, float* scores, const float* mask,
                                     int maskElements,
                                     int batch, int queryLen, int insertLen, int numHeads, int kvHeads,
                                     int headDim, int baseLogical, int qStart, int qPieceLen, int kvLen, int maxSlots,
                                     float scale,
                                     const int* queryLogicalIndices, int queryRowsAreFull) {
    constexpr int TILE_Q = 16;
    constexpr int TILE_K = 16;
    constexpr int TILE_D = 32;
    __shared__ float qTile[TILE_Q][TILE_D + 1];
    __shared__ float kTile[TILE_K][TILE_D + 1];

    int k = blockIdx.x * TILE_K + threadIdx.x;
    int qLocal = blockIdx.y * TILE_Q + threadIdx.y;
    int q = qStart + qLocal;
    int bh = blockIdx.z;
    if (bh >= batch * numHeads) {
        return;
    }
    int b = bh / numHeads;
    int h = bh % numHeads;
    int group = numHeads / kvHeads;
    int kvHead = h / group;
    bool validQ = qLocal < qPieceLen && q < insertLen;
    int qLogical = validQ ? (queryLogicalIndices ? queryLogicalIndices[q] : (baseLogical + q)) : -1;
    int qRow = queryRowsAreFull ? qLogical : q;
    validQ = validQ && qRow >= 0 && qRow < queryLen;
    int scoreOffset = ((b * numHeads + h) * qPieceLen + qLocal) * kvLen + k;

    int slot = k < kvLen ? k : -1;
    bool validK = k < kvLen && slot >= 0 && slot < maxSlots;
    bool validScore = validQ && validK && k <= qLogical;
    int maskCols = 0;
    int maskGap = 0;
    if (mask != nullptr && maskElements > 1) {
        maskCols = maskElements >= insertLen * kvLen ? kvLen : insertLen;
        maskGap = kvLen - maskCols;
    }

    float score = 0.0f;
    for (int dStart = 0; dStart < headDim; dStart += TILE_D) {
        for (int dd = threadIdx.x; dd < TILE_D; dd += TILE_K) {
            int d = dStart + dd;
            float value = 0.0f;
            if (validQ && d < headDim) {
                int qOffset = ((b * queryLen + qRow) * numHeads + h) * headDim + d;
                value = pagedToFloat<T>(query[qOffset]);
            }
            qTile[threadIdx.y][dd] = value;
        }
        for (int dd = threadIdx.y; dd < TILE_D; dd += TILE_Q) {
            int d = dStart + dd;
            float value = 0.0f;
            if (validK && d < headDim) {
                int kOffset = ((slot * batch + b) * kvHeads + kvHead) * headDim + d;
                value = pagedToFloat<T>(keyCache[kOffset]);
            }
            kTile[threadIdx.x][dd] = value;
        }
        __syncthreads();

        int dEnd = min(TILE_D, headDim - dStart);
#pragma unroll
        for (int dd = 0; dd < TILE_D; ++dd) {
            if (dd < dEnd) {
                score += qTile[threadIdx.y][dd] * kTile[threadIdx.x][dd];
            }
        }
        __syncthreads();
    }
    if (validQ && k < kvLen) {
        float finalScore = validScore ? score * scale : -FLT_MAX;
        if (validScore && mask != nullptr && maskCols > 0 && k >= maskGap) {
            int col = k - maskGap;
            int maskIdx = (queryRowsAreFull ? qLogical : q) * maskCols + col;
            if (maskIdx >= 0 && maskIdx < maskElements) {
                finalScore += mask[maskIdx];
            }
        }
        scores[scoreOffset] = finalScore;
    }
}

template <typename T>
__global__ void pagedPrefillQKVKernel(const float* probs, const T* valueCache, T* output, int batch,
                                      int outputLen, int qStart, int qPieceLen, int numHeads, int kvHeads,
                                      int headDim, int kvLen, int maxSlots) {
    constexpr int TILE_D = 32;
    constexpr int TILE_Q = 8;
    constexpr int TILE_K = 32;
    __shared__ float probTile[TILE_Q][TILE_K + 1];
    __shared__ float valueTile[TILE_K][TILE_D + 1];

    int d = blockIdx.x * TILE_D + threadIdx.x;
    int qLocal = blockIdx.y * TILE_Q + threadIdx.y;
    int q = qStart + qLocal;
    int bh = blockIdx.z;
    if (bh >= batch * numHeads) {
        return;
    }
    int b = bh / numHeads;
    int h = bh % numHeads;
    int group = numHeads / kvHeads;
    int kvHead = h / group;
    const float* probBase = probs + ((b * numHeads + h) * qPieceLen) * kvLen;
    const T* valueBase = valueCache + ((b * kvHeads + kvHead) * maxSlots) * headDim;

    float acc = 0.0f;
    int linearTid = threadIdx.y * blockDim.x + threadIdx.x;
    int linearThreads = blockDim.x * blockDim.y;
    for (int kStart = 0; kStart < kvLen; kStart += TILE_K) {
        int kForProb = kStart + threadIdx.x;
        if (qLocal < qPieceLen && kForProb < kvLen) {
            probTile[threadIdx.y][threadIdx.x] = probBase[qLocal * kvLen + kForProb];
        } else {
            probTile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        for (int idx = linearTid; idx < TILE_K * TILE_D; idx += linearThreads) {
            int kk = idx / TILE_D;
            int dd = idx - kk * TILE_D;
            int logical = kStart + kk;
            int valueD = blockIdx.x * TILE_D + dd;
            float value = 0.0f;
            if (logical < kvLen && valueD < headDim) {
                int slot = logical;
                value = pagedToFloat<T>(valueBase[slot * headDim + valueD]);
            }
            valueTile[kk][dd] = value;
        }
        __syncthreads();

        if (qLocal < qPieceLen && q < outputLen && d < headDim) {
#pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                if (kStart + kk < kvLen) {
                    acc += probTile[threadIdx.y][kk] * valueTile[kk][threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    if (qLocal < qPieceLen && q < outputLen && d < headDim) {
        int outOffset = (b * outputLen * numHeads * headDim) + q * numHeads * headDim + h * headDim + d;
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

static size_t segmentTokenCountCUDA(const std::vector<PagedKVExternalSegment>& segments) {
    size_t total = 0;
    for (const auto& segment : segments) {
        total += segment.tokenCount;
    }
    return total;
}

static int picCacheSourceSlotBaseCUDA(const PagedKVMeta* meta, int kvLen) {
    if (meta == nullptr) {
        return kvLen;
    }
    return std::max(kvLen, meta->request_capacity);
}

static size_t picCacheSourceSlotCountCUDA(const PagedKVMeta* meta) {
    if (meta == nullptr) {
        return 0;
    }
    return std::max(meta->external_source_slot_reserve,
                    std::max(segmentTokenCountCUDA(meta->external_segments),
                             segmentTokenCountCUDA(meta->cacheblend_score_segments)));
}

static int maxSlotsWithPicSourceSlotsCUDA(const PagedKVMeta* meta, int kvLen) {
    const size_t sourceSlots = picCacheSourceSlotCountCUDA(meta);
    if (sourceSlots == 0) {
        return kvLen;
    }
    const int sourceBase = picCacheSourceSlotBaseCUDA(meta, kvLen);
    if (sourceSlots > static_cast<size_t>(std::numeric_limits<int>::max() - sourceBase)) {
        return std::numeric_limits<int>::max();
    }
    return std::max(kvLen, sourceBase + static_cast<int>(sourceSlots));
}

static int externalLayerRequiredSlotsCUDA(const PagedKVMeta* meta, int kvLen) {
    return maxSlotsWithPicSourceSlotsCUDA(meta, kvLen);
}

static bool writeBinaryFile(const std::string& path, const void* data, size_t bytes) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    if (data != nullptr && bytes > 0) {
        const char* cursor = reinterpret_cast<const char*>(data);
        size_t remaining = bytes;
        while (remaining > 0) {
            const auto chunk = static_cast<std::streamsize>(
                std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<std::streamsize>::max())));
            os.write(cursor, chunk);
            if (!os.good()) {
                return false;
            }
            cursor += chunk;
            remaining -= static_cast<size_t>(chunk);
        }
    }
    return os.good();
}

static bool readStreamRange(std::ifstream& is, size_t offsetBytes, void* dst, size_t expectedBytes) {
    if (dst == nullptr || expectedBytes == 0) {
        return false;
    }
    is.seekg(static_cast<std::streamoff>(offsetBytes), std::ios::beg);
    char* cursor = reinterpret_cast<char*>(dst);
    size_t remaining = expectedBytes;
    while (remaining > 0) {
        const auto chunk = static_cast<std::streamsize>(
            std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<std::streamsize>::max())));
        is.read(cursor, chunk);
        if (is.gcount() != chunk) {
            return false;
        }
        cursor += chunk;
        remaining -= static_cast<size_t>(chunk);
    }
    return true;
}

static bool readBinaryFileRange(const std::string& path, size_t offsetBytes, void* dst, size_t expectedBytes) {
    if (dst == nullptr || expectedBytes == 0) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto size = is.tellg();
    if (size < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(size));
    if (offsetBytes > fileBytes || fileBytes - offsetBytes < expectedBytes) {
        return false;
    }
    return readStreamRange(is, offsetBytes, dst, expectedBytes);
}

static bool readExternalValueSegmentToBuffer(const std::string& path, void* dstBuffer, size_t dstBytes, int batch,
                                             int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset,
                                             size_t tokenCount, int headDim, int bytes) {
    if (batch <= 0 || kvHeads <= 0 || headDim <= 0 || bytes <= 0 || tokenCount == 0) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto fileSize = is.tellg();
    if (fileSize < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(fileSize));
    const size_t tokenBytes = static_cast<size_t>(headDim) * bytes;
    const size_t segmentBytes = tokenCount * tokenBytes;
    const size_t requiredBytes = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * tokenBytes;
    if (sourceTokenOffset + tokenCount > sourceTokenCount || fileBytes < requiredBytes) {
        return false;
    }
    const size_t outputBytes = static_cast<size_t>(batch) * kvHeads * segmentBytes;
    if (dstBuffer == nullptr || dstBytes < outputBytes) {
        return false;
    }
    auto* dstBase = reinterpret_cast<int8_t*>(dstBuffer);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dst = (static_cast<size_t>(b) * kvHeads + h) * segmentBytes;
            if (!readStreamRange(is, src, dstBase + dst, segmentBytes)) {
                return false;
            }
        }
    }
    return true;
}

static bool readExternalValueSegmentToPagedCache(const std::string& path, void* valueCacheHost, int batch,
                                                 int kvHeads, int maxSlots, size_t logicalStart,
                                                 size_t sourceTokenCount, size_t sourceTokenOffset,
                                                 size_t tokenCount, int headDim, int bytes) {
    if (valueCacheHost == nullptr || batch <= 0 || kvHeads <= 0 || maxSlots <= 0 || headDim <= 0 || bytes <= 0 ||
        tokenCount == 0 || logicalStart + tokenCount > static_cast<size_t>(maxSlots)) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto fileSize = is.tellg();
    if (fileSize < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(fileSize));
    const size_t tokenBytes = static_cast<size_t>(headDim) * bytes;
    const size_t segmentBytes = tokenCount * tokenBytes;
    const size_t requiredBytes = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * tokenBytes;
    if (sourceTokenOffset + tokenCount > sourceTokenCount || fileBytes < requiredBytes) {
        return false;
    }
    auto* dstBase = reinterpret_cast<int8_t*>(valueCacheHost);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dst = ((static_cast<size_t>(b) * kvHeads + h) * static_cast<size_t>(maxSlots) +
                                logicalStart) * tokenBytes;
            if (!readStreamRange(is, src, dstBase + dst, segmentBytes)) {
                return false;
            }
        }
    }
    return true;
}

static bool profilePagedAttention() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool nvtxPagedAttention() {
#ifdef MNN_CUDA_PROFILE
    const char* value = ::getenv("MNN_PAGED_ATTENTION_NVTX");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
#else
    return false;
#endif
}

static uint64_t nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static std::string nvtxLayerRangeName(const char* op, int layerIndex, int queryLen, int insertLen, int kvLen) {
    std::ostringstream os;
    os << "CUDAPagedAttention " << op << " layer=" << layerIndex;
    if (queryLen >= 0) {
        os << " query=" << queryLen;
    }
    if (insertLen >= 0) {
        os << " insert=" << insertLen;
    }
    if (kvLen >= 0) {
        os << " kv_len=" << kvLen;
    }
    return os.str();
}

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(std::string name, bool enabled = true) {
#ifdef MNN_CUDA_PROFILE
        mEnabled = enabled;
        if (mEnabled) {
            mName = std::move(name);
            NVTX_PUSH(mName.c_str());
        }
#else
        (void)name;
        (void)enabled;
#endif
    }
    ~ScopedNvtxRange() {
#ifdef MNN_CUDA_PROFILE
        if (mEnabled) {
            NVTX_POP();
        }
#endif
    }

private:
#ifdef MNN_CUDA_PROFILE
    bool mEnabled = false;
    std::string mName;
#endif
};

struct ExternalLayerReadSegment {
    bool ok = false;
    bool directWritten = false;
    std::string error;
};

struct ExternalLayerReadResult {
    bool ok = true;
    int layerIndex = -1;
    std::string requestKey;
    std::string error;
    std::vector<ExternalLayerReadSegment> segments;
};

struct ExternalLayerReadTask {
    std::string requestKey;
    int layerIndex = -1;
    std::shared_future<std::shared_ptr<ExternalLayerReadResult>> future;
};

static std::mutex gExternalLayerReadMutex;
static std::unordered_map<std::string, ExternalLayerReadTask> gExternalLayerReadTasks;

struct CUDAPagedAttention::SharedPagedCache::MappedBuffer {
    void* host = nullptr;
    void* device = nullptr;
    size_t bytes = 0;
    int deviceId = -1;

    ~MappedBuffer() {
        if (host == nullptr) {
            return;
        }
        if (deviceId >= 0) {
            cudaSetDevice(deviceId);
        }
        cudaFreeHost(host);
        host = nullptr;
        device = nullptr;
        bytes = 0;
    }
};

static bool envFlagEnabled(const char* name, bool defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return value[0] != '0';
}

static CudaSparseQTileVariant selectCudaDecodeRepairQTileVariant(int headDim, int attnLen) {
    // Decode repair uses one implementation family for x=0/1/3/5/7: x=0 is the
    // active-row=1 degenerate case, and x>0 only increases the compact row count.
    if (headDim == 128 && attnLen >= 1) {
        return kCudaSparseQTileHD128Q8K16;
    }
    if (headDim == 64 && attnLen >= 16) {
        return kCudaSparseQTileHD64Q4K16;
    }
    return kCudaSparseQTileNone;
}

static bool cudaDecodeRepairQTileEnabled(int headDim, int attnLen) {
    if (selectCudaDecodeRepairQTileVariant(headDim, attnLen) == kCudaSparseQTileNone) {
        return false;
    }
    return true;
}

static ErrorCode debugCheckPagedAttentionKernel(bool enabled, const char* op, int layerIndex, int piece, int qStart,
                                                int qPieceLen, int kvLen, int numHeads, int kvHeads, int headDim) {
    if (!enabled) {
        return NO_ERROR;
    }
    auto launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        MNN_PRINT("CUDAPagedAttention: debug launch failed op=%s layer=%d piece=%d q_start=%d q_piece=%d kv_len=%d "
                  "heads=%d kv_heads=%d head_dim=%d err=%s\n",
                  op, layerIndex, piece, qStart, qPieceLen, kvLen, numHeads, kvHeads, headDim,
                  cudaGetErrorString(launchErr));
        return INVALID_VALUE;
    }
    auto syncErr = cudaDeviceSynchronize();
    if (syncErr == cudaSuccess) {
        return NO_ERROR;
    }
    MNN_PRINT("CUDAPagedAttention: debug sync failed op=%s layer=%d piece=%d q_start=%d q_piece=%d kv_len=%d "
              "heads=%d kv_heads=%d head_dim=%d err=%s\n",
              op, layerIndex, piece, qStart, qPieceLen, kvLen, numHeads, kvHeads, headDim,
              cudaGetErrorString(syncErr));
    return INVALID_VALUE;
}

static std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> makeMappedPagedBuffer(size_t bytes,
                                                                                                  int deviceId) {
    if (bytes == 0) {
        return nullptr;
    }
    if (deviceId >= 0) {
        cudaSetDevice(deviceId);
    }
    void* host = nullptr;
    auto err = cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess || host == nullptr) {
        return nullptr;
    }
    void* device = nullptr;
    err = cudaHostGetDevicePointer(&device, host, 0);
    if (err != cudaSuccess || device == nullptr) {
        cudaFreeHost(host);
        return nullptr;
    }
    auto out = std::make_shared<CUDAPagedAttention::SharedPagedCache::MappedBuffer>();
    out->host = host;
    out->device = device;
    out->bytes = bytes;
    out->deviceId = deviceId;
    return out;
}

static size_t roundMappedWorkspaceBytes(size_t bytes) {
    constexpr size_t kAlign = 64 * 1024 * 1024;
    if (bytes == 0) {
        return 0;
    }
    return ((bytes + kAlign - 1) / kAlign) * kAlign;
}

static bool ensureMappedHostWorkspace(
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>* buffer, size_t bytes, int deviceId) {
    if (buffer == nullptr || bytes == 0) {
        return false;
    }
    if (*buffer != nullptr && (*buffer)->host != nullptr && (*buffer)->device != nullptr &&
        (*buffer)->bytes >= bytes) {
        return true;
    }
    *buffer = makeMappedPagedBuffer(roundMappedWorkspaceBytes(bytes), deviceId);
    return *buffer != nullptr && (*buffer)->host != nullptr && (*buffer)->device != nullptr;
}

struct MappedPagedExportWorkspace {
    std::mutex mutex;
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> scratch;
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> valueStorage;
};

static MappedPagedExportWorkspace& mappedPagedExportWorkspace(int deviceId) {
    static std::mutex mutex;
    static std::unordered_map<int, std::shared_ptr<MappedPagedExportWorkspace>> workspaces;
    std::lock_guard<std::mutex> lock(mutex);
    auto& workspace = workspaces[deviceId];
    if (workspace == nullptr) {
        workspace.reset(new MappedPagedExportWorkspace);
    }
    return *workspace;
}

static int cudaBackendDeviceId(CUDABackend* backend) {
    if (backend == nullptr || backend->getCUDARuntime() == nullptr) {
        return -1;
    }
    return backend->getCUDARuntime()->device_id();
}

static bool shouldUseMappedPagedCache(CUDABackend* backend) {
    if (backend == nullptr || !envFlagEnabled("MNN_PAGED_ATTENTION_ZERO_COPY_CACHE", true)) {
        return false;
    }
    auto runtime = backend->getCUDARuntime();
    if (runtime == nullptr) {
        return false;
    }
    return runtime->prop().integrated != 0 || envFlagEnabled("MNN_PAGED_ATTENTION_FORCE_ZERO_COPY_CACHE", false);
}

struct ExternalLayerMappedTarget {
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> key;
    std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> value;
    int batch = 0;
    int kvHeads = 0;
    int headDim = 0;
    int bytes = 0;
    int maxSlots = 0;
};

struct ExternalLayerMappedTargetRef {
    std::weak_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> key;
    std::weak_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer> value;
    int batch = 0;
    int kvHeads = 0;
    int headDim = 0;
    int bytes = 0;
    int maxSlots = 0;
};

static std::mutex gExternalLayerMappedTargetMutex;
static std::unordered_map<std::string, ExternalLayerMappedTargetRef> gExternalLayerMappedTargets;

static std::string externalLayerMappedTargetKey(const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads,
                                                int headDim, int bytes) {
    std::ostringstream os;
    os << reinterpret_cast<uintptr_t>(meta) << ":" << layerIndex << ":" << batch << ":" << kvHeads << ":"
       << headDim << ":" << bytes;
    return os.str();
}

static void registerExternalLayerMappedTarget(
    const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim, int bytes, int maxSlots,
    const std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>& key,
    const std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>& value) {
    if (meta == nullptr || layerIndex < 0 || key == nullptr || value == nullptr || key->host == nullptr ||
        value->host == nullptr || key->device == nullptr || value->device == nullptr) {
        return;
    }
    ExternalLayerMappedTargetRef ref;
    ref.key = key;
    ref.value = value;
    ref.batch = batch;
    ref.kvHeads = kvHeads;
    ref.headDim = headDim;
    ref.bytes = bytes;
    ref.maxSlots = maxSlots;
    std::lock_guard<std::mutex> lock(gExternalLayerMappedTargetMutex);
    gExternalLayerMappedTargets[externalLayerMappedTargetKey(meta, layerIndex, batch, kvHeads, headDim, bytes)] =
        std::move(ref);
}

static ExternalLayerMappedTarget lookupExternalLayerMappedTarget(const PagedKVMeta* meta, int layerIndex, int batch,
                                                                 int kvHeads, int headDim, int bytes) {
    ExternalLayerMappedTarget target;
    std::lock_guard<std::mutex> lock(gExternalLayerMappedTargetMutex);
    auto iter = gExternalLayerMappedTargets.find(
        externalLayerMappedTargetKey(meta, layerIndex, batch, kvHeads, headDim, bytes));
    if (iter == gExternalLayerMappedTargets.end()) {
        return target;
    }
    target.key = iter->second.key.lock();
    target.value = iter->second.value.lock();
    target.batch = iter->second.batch;
    target.kvHeads = iter->second.kvHeads;
    target.headDim = iter->second.headDim;
    target.bytes = iter->second.bytes;
    target.maxSlots = iter->second.maxSlots;
    if (target.key == nullptr || target.value == nullptr) {
        gExternalLayerMappedTargets.erase(iter);
        target = ExternalLayerMappedTarget();
    }
    return target;
}

static bool externalLayerTargetReadyCUDA(const ExternalLayerMappedTarget& target, int batch, int kvHeads,
                                         int headDim, int bytes, int requiredSlots) {
    return target.key != nullptr && target.value != nullptr && target.key->host != nullptr &&
        target.value->host != nullptr && target.batch == batch && target.kvHeads == kvHeads &&
        target.headDim == headDim && target.bytes == bytes && target.maxSlots >= requiredSlots;
}

static int lastExternalLayerIndex(const PagedKVMeta* meta) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return -1;
    }
    int last = meta->layer_nums > 0 ? meta->layer_nums - 1 : -1;
    for (const auto& segment : meta->external_segments) {
        for (const auto& layer : segment.layers) {
            last = std::max(last, layer.layerIndex);
        }
    }
    return last;
}

static int externalLayerReadWindow() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PREFETCH_WINDOW");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    return std::max(0, std::atoi(value));
}

static std::string externalLayerRequestKey(const PagedKVMeta* meta, int batch, int kvHeads, int headDim, int bytes,
                                           int kvLen) {
    std::ostringstream os;
    os << reinterpret_cast<uintptr_t>(meta) << ":" << (meta != nullptr ? meta->request_generation : 0) << ":"
       << batch << ":" << kvHeads << ":" << headDim << ":" << bytes << ":" << kvLen;
    if (meta != nullptr) {
        for (const auto& segment : meta->external_segments) {
            os << "|seg:" << segment.cacheName << ":" << segment.logicalStart << ":" << segment.tokenCount << ":"
               << segment.sourceTokenOffset << ":" << segment.sourceTokenCount;
            for (const auto& layer : segment.layers) {
                os << "|layer:" << layer.layerIndex << ":" << layer.keyPath << ":" << layer.valuePath << ":"
                   << layer.hasSourceOverride << ":" << layer.sourceTokenOffset << ":" << layer.sourceTokenCount;
            }
        }
    }
    return os.str();
}

static std::string externalLayerTaskKey(const std::string& requestKey, int layerIndex) {
    return requestKey + "\n" + std::to_string(layerIndex);
}

static bool readExternalLayerSegmentCUDA(const PagedKVExternalSegment& segment, int layerIndex, int batch,
                                         int kvHeads, int headDim, int bytes, int kvLen,
                                         const ExternalLayerMappedTarget* target,
                                         ExternalLayerReadSegment& out) {
    out = ExternalLayerReadSegment();
    if (segment.tokenCount == 0) {
        out.ok = true;
        return true;
    }
    auto layer = segment.layer(layerIndex);
    if (layer == nullptr) {
        out.error = "missing external PIC KV layer " + std::to_string(layerIndex) + " for cache " + segment.cacheName;
        return false;
    }
    const int segBatch = segment.batch > 0 ? segment.batch : batch;
    const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
    const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
    const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
    if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
        out.error = "external PIC KV shape mismatch at layer " + std::to_string(layerIndex);
        return false;
    }
    if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
        out.error = "external PIC KV must be canonical_no_rope/half";
        return false;
    }
    if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
        out.error = "external PIC KV range exceeds visible KV length at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset : segment.sourceTokenOffset;
    const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
        ? layer->sourceTokenCount
        : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
    if (sourceTokenOffset + segment.tokenCount > sourceTokenCount) {
        out.error = "external PIC KV source range is invalid at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
    if (segment.logicalStart > maxInt || segment.tokenCount > maxInt || sourceTokenOffset > maxInt ||
        sourceTokenCount > maxInt) {
        out.error = "external PIC KV indices exceed int range at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t keyTokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
    const size_t keySegmentBytes = segment.tokenCount * keyTokenBytes;
    const bool targetOk = target != nullptr && target->key != nullptr && target->value != nullptr &&
        target->key->host != nullptr && target->value->host != nullptr && target->batch == batch &&
        target->kvHeads == kvHeads && target->headDim == headDim && target->bytes == bytes &&
        target->maxSlots >= kvLen && segment.logicalStart + segment.tokenCount <=
            static_cast<size_t>(target->maxSlots);
    if (!targetOk) {
        out.error = "CUDA external PIC KV requires direct mapped PagedCache at layer " +
                    std::to_string(layerIndex);
        return false;
    }
    const size_t keyDstOffset = segment.logicalStart * keyTokenBytes;
    const size_t keyEnd = keyDstOffset + keySegmentBytes;
    if (keyEnd > target->key->bytes) {
        out.error = "mapped external PIC key target is too small at layer " + std::to_string(layerIndex);
        return false;
    }
    if (!readBinaryFileRange(layer->keyPath, sourceTokenOffset * keyTokenBytes,
                             reinterpret_cast<int8_t*>(target->key->host) + keyDstOffset, keySegmentBytes)) {
        out.error = "failed to read external PIC key directly to PagedCache at layer " +
                    std::to_string(layerIndex);
        return false;
    }
    if (!readExternalValueSegmentToPagedCache(layer->valuePath, target->value->host, batch, kvHeads,
                                              target->maxSlots, segment.logicalStart, sourceTokenCount,
                                              sourceTokenOffset, segment.tokenCount, headDim, bytes)) {
        out.error = "failed to read external PIC value directly to PagedCache at layer " +
                    std::to_string(layerIndex);
        return false;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    out.directWritten = true;
    out.ok = true;
    return true;
}

static std::shared_ptr<ExternalLayerReadResult> readExternalLayerCUDA(
    const PagedKVMeta* meta, std::string requestKey, std::vector<PagedKVExternalSegment> segments, int layerIndex,
    int batch, int kvHeads, int headDim, int bytes, int kvLen, ExternalLayerMappedTarget target) {
    ScopedNvtxRange nvtx(nvtxLayerRangeName("pic_async_read_from_disk", layerIndex, -1, -1, kvLen),
                         nvtxPagedAttention());
    auto result = std::make_shared<ExternalLayerReadResult>();
    result->requestKey = std::move(requestKey);
    result->layerIndex = layerIndex;
    result->segments.resize(segments.size());
    const bool hasTarget = externalLayerTargetReadyCUDA(
        target, batch, kvHeads, headDim, bytes, externalLayerRequiredSlotsCUDA(meta, kvLen));
    for (size_t i = 0; i < segments.size(); ++i) {
        if (!readExternalLayerSegmentCUDA(segments[i], layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                          hasTarget ? &target : nullptr, result->segments[i])) {
            result->ok = false;
            result->error = result->segments[i].error;
            break;
        }
    }
    return result;
}

static void scheduleExternalLayerReadsFrom(const PagedKVMeta* meta, int startLayer, int batch, int kvHeads,
                                           int headDim, int bytes, int kvLen) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return;
    }
    const int lastLayer = lastExternalLayerIndex(meta);
    if (startLayer < 0 || lastLayer < startLayer) {
        return;
    }
    const int requiredSlots = externalLayerRequiredSlotsCUDA(meta, kvLen);
    const int window = externalLayerReadWindow();
    const int endLayer = window > 0 ? std::min(lastLayer, startLayer + window - 1) : lastLayer;
    const std::string requestKey = externalLayerRequestKey(meta, batch, kvHeads, headDim, bytes, kvLen);
    const std::string requestPrefix = requestKey + "\n";
    std::vector<std::pair<int, ExternalLayerMappedTarget>> layersToSchedule;
    {
        std::lock_guard<std::mutex> lock(gExternalLayerReadMutex);
        for (auto it = gExternalLayerReadTasks.begin(); it != gExternalLayerReadTasks.end();) {
            if (it->first.rfind(requestPrefix, 0) != 0) {
                it = gExternalLayerReadTasks.erase(it);
            } else {
                ++it;
            }
        }
        for (int layerIndex = startLayer; layerIndex <= endLayer; ++layerIndex) {
            if (meta->externalLayerLoaded(layerIndex)) {
                continue;
            }
            const auto key = externalLayerTaskKey(requestKey, layerIndex);
            if (gExternalLayerReadTasks.find(key) == gExternalLayerReadTasks.end()) {
                auto target = lookupExternalLayerMappedTarget(meta, layerIndex, batch, kvHeads, headDim, bytes);
                if (!externalLayerTargetReadyCUDA(target, batch, kvHeads, headDim, bytes, requiredSlots)) {
                    break;
                }
                layersToSchedule.emplace_back(layerIndex, std::move(target));
            }
        }
        auto segments = meta->external_segments;
        for (const auto& item : layersToSchedule) {
            const int layerIndex = item.first;
            auto target = item.second;
            const auto key = externalLayerTaskKey(requestKey, layerIndex);
            auto future = std::async(std::launch::async,
                                     [meta, requestKey, segments, layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                      target]() {
                                         return readExternalLayerCUDA(meta, requestKey, segments, layerIndex, batch,
                                                                      kvHeads, headDim, bytes, kvLen, target);
                                     }).share();
            ExternalLayerReadTask task;
            task.requestKey = requestKey;
            task.layerIndex = layerIndex;
            task.future = std::move(future);
            gExternalLayerReadTasks[key] = std::move(task);
        }
    }
}

static std::shared_ptr<ExternalLayerReadResult> takeExternalLayerRead(const PagedKVMeta* meta, int layerIndex,
                                                                      int batch, int kvHeads, int headDim, int bytes,
                                                                      int kvLen) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return nullptr;
    }
    const auto requestKey = externalLayerRequestKey(meta, batch, kvHeads, headDim, bytes, kvLen);
    const auto taskKey = externalLayerTaskKey(requestKey, layerIndex);
    std::shared_future<std::shared_ptr<ExternalLayerReadResult>> future;
    {
        std::lock_guard<std::mutex> lock(gExternalLayerReadMutex);
        auto iter = gExternalLayerReadTasks.find(taskKey);
        if (iter == gExternalLayerReadTasks.end()) {
            return nullptr;
        }
        future = iter->second.future;
        gExternalLayerReadTasks.erase(iter);
    }
    if (!future.valid()) {
        return nullptr;
    }
    return future.get();
}

static int ropeDimForExport(const KVMeta* meta, int headDim) {
    if (meta == nullptr || meta->rope_dim <= 0) {
        return headDim;
    }
    return std::min(headDim, meta->rope_dim);
}

static int ropeTypeCode(const KVMeta* meta) {
    if (meta != nullptr && meta->rope_type == "llama3") {
        return 1;
    }
    return 0;
}

static int ropeTypeCode(const PagedKVExternalSegment& segment) {
    return segment.ropeType == "llama3" ? 1 : 0;
}

static ErrorCode restoreExternalSegmentsCUDA(PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim,
                                             int bytes, int maxSlots, int kvLen, void* keyCacheDevice) {
    if (meta == nullptr || meta->external_segments.empty() || meta->externalLayerLoaded(layerIndex)) {
        return NO_ERROR;
    }
    const bool profile = profilePagedAttention();
    const bool nvtx = nvtxPagedAttention();
    auto directTarget = lookupExternalLayerMappedTarget(meta, layerIndex, batch, kvHeads, headDim, bytes);
    const bool hasDirectTarget = directTarget.key != nullptr && directTarget.value != nullptr;
    const char* restoreOp = "pic_restore_mapped_kv_for_attention_total";
    ScopedNvtxRange restoreNvtx(nvtxLayerRangeName(restoreOp, layerIndex, -1, -1, kvLen), nvtx);
    if (!hasDirectTarget) {
        MNN_ERROR("CUDAPagedAttention external PIC KV requires direct mapped PagedCache at layer %d\n", layerIndex);
        return INVALID_VALUE;
    }
    const uint64_t startUs = profile ? nowUs() : 0;
    size_t totalTokens = 0;
    size_t directTokens = 0;
    int directSegments = 0;
    scheduleExternalLayerReadsFrom(meta, layerIndex + 1, batch, kvHeads, headDim, bytes, kvLen);
    std::shared_ptr<ExternalLayerReadResult> prefetched;
    {
        ScopedNvtxRange waitNvtx(nvtxLayerRangeName("pic_wait_async_read_from_disk", layerIndex, -1, -1, kvLen),
                                 nvtx);
        prefetched = takeExternalLayerRead(meta, layerIndex, batch, kvHeads, headDim, bytes, kvLen);
    }
    if (prefetched != nullptr && !prefetched->ok) {
        MNN_ERROR("CUDAPagedAttention async external PIC KV read failed at layer %d: %s\n", layerIndex,
                  prefetched->error.c_str());
        return INVALID_VALUE;
    }
    if (prefetched != nullptr && prefetched->segments.size() != meta->external_segments.size()) {
        MNN_ERROR("CUDAPagedAttention async external PIC KV read segment count mismatch at layer %d\n", layerIndex);
        return INVALID_VALUE;
    }
    for (size_t segmentIndex = 0; segmentIndex < meta->external_segments.size(); ++segmentIndex) {
        const auto& segment = meta->external_segments[segmentIndex];
        totalTokens += segment.tokenCount;
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            MNN_ERROR("CUDAPagedAttention layer %d missing external PIC KV for cache %s\n", layerIndex,
                      segment.cacheName.c_str());
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : batch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
        if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
            MNN_ERROR("CUDAPagedAttention external KV shape mismatch at layer %d, cache %s\n", layerIndex,
                      segment.cacheName.c_str());
            return INVALID_VALUE;
        }
        if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
            MNN_ERROR("CUDAPagedAttention external KV must be canonical_no_rope/half, got %s/%s\n",
                      segment.keyRopeState.c_str(), segment.ropePairing.c_str());
            return INVALID_VALUE;
        }
        if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
            MNN_ERROR("CUDAPagedAttention external KV range exceeds visible KV length at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        if (segment.tokenCount == 0) {
            continue;
        }
        ExternalLayerReadSegment syncRead;
        const ExternalLayerReadSegment* loadedSegment = nullptr;
        if (prefetched != nullptr) {
            loadedSegment = &prefetched->segments[segmentIndex];
            if (!loadedSegment->ok) {
                MNN_ERROR("CUDAPagedAttention async external PIC KV segment read failed at layer %d: %s\n",
                          layerIndex, loadedSegment->error.c_str());
                return INVALID_VALUE;
            }
        } else {
            if (!readExternalLayerSegmentCUDA(segment, layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                              hasDirectTarget ? &directTarget : nullptr, syncRead)) {
                MNN_ERROR("CUDAPagedAttention failed to read external PIC KV files for layer %d: %s\n",
                          layerIndex, syncRead.error.c_str());
                return INVALID_VALUE;
            }
            loadedSegment = &syncRead;
        }
        if (loadedSegment == nullptr || !loadedSegment->directWritten) {
            MNN_ERROR("CUDAPagedAttention external PIC KV requires direct mapped PagedCache at layer %d\n",
                      layerIndex);
            return INVALID_VALUE;
        }
        directTokens += segment.tokenCount;
        ++directSegments;
        const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset
                                                                  : segment.sourceTokenOffset;
        const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
            ? layer->sourceTokenCount
            : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
        const size_t sourceEnd = sourceTokenOffset + segment.tokenCount;
        if (sourceEnd > sourceTokenCount) {
            return INVALID_VALUE;
        }
        const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
        if (segment.logicalStart > maxInt || segment.tokenCount > maxInt || sourceTokenOffset > maxInt ||
            sourceTokenCount > maxInt) {
            MNN_ERROR("CUDAPagedAttention external PIC KV indices exceed int range at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        for (size_t local = 0; local < segment.tokenCount; ++local) {
            int logical = static_cast<int>(segment.logicalStart + local);
            int slot = logical;
            if (slot < 0 || slot >= maxSlots) {
                return OUT_OF_MEMORY;
            }
        }
        const size_t keyTokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
        const int8_t* keySourceDevice = reinterpret_cast<const int8_t*>(keyCacheDevice) +
                                        segment.logicalStart * keyTokenBytes;
        int ropeDim = segment.ropeDim > 0 ? segment.ropeDim : headDim;
        ropeDim = std::min(ropeDim, headDim);
        ropeDim = (ropeDim / 2) * 2;
        const int oldContext = segment.ropeScalingOriginalMaxPositionEmbeddings > 0
            ? segment.ropeScalingOriginalMaxPositionEmbeddings
            : segment.maxPositionEmbeddings;
        const size_t totalElements = segment.tokenCount * static_cast<size_t>(batch) * kvHeads * headDim;
        if (totalElements > maxInt) {
            MNN_ERROR("CUDAPagedAttention external PIC KV hydrate element count exceeds int range at layer %d\n",
                      layerIndex);
            return INVALID_VALUE;
        }
        const int totalKey = static_cast<int>(totalElements);
        const int threads = 256;
        const int blocks = UP_DIV(totalKey, threads);
        const char* keyRopeOp = "pic_apply_rope_to_mapped_key_cache";
        {
            ScopedNvtxRange keyRopeNvtx(nvtxLayerRangeName(keyRopeOp, layerIndex, -1, -1, kvLen), nvtx);
            if (bytes == 4) {
                hydratePagedKeyKernel<float><<<blocks, threads>>>(
                    reinterpret_cast<const float*>(keySourceDevice), reinterpret_cast<float*>(keyCacheDevice),
                    batch, static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots,
                    static_cast<int>(segment.logicalStart), ropeDim,
                    segment.ropeTheta > 0.0f ? segment.ropeTheta : 10000.0f, ropeTypeCode(segment),
                    std::max(segment.ropeScalingFactor, 1.0f), std::max(segment.ropeScalingLowFreqFactor, 1.0e-6f),
                    std::max(segment.ropeScalingHighFreqFactor, 1.0e-6f), oldContext,
                    segment.ropeAttentionScaling > 0.0f ? segment.ropeAttentionScaling : 1.0f, totalKey);
            } else {
                hydratePagedKeyKernel<half><<<blocks, threads>>>(
                    reinterpret_cast<const half*>(keySourceDevice), reinterpret_cast<half*>(keyCacheDevice),
                    batch, static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots,
                    static_cast<int>(segment.logicalStart), ropeDim,
                    segment.ropeTheta > 0.0f ? segment.ropeTheta : 10000.0f, ropeTypeCode(segment),
                    std::max(segment.ropeScalingFactor, 1.0f), std::max(segment.ropeScalingLowFreqFactor, 1.0e-6f),
                    std::max(segment.ropeScalingHighFreqFactor, 1.0e-6f), oldContext,
                    segment.ropeAttentionScaling > 0.0f ? segment.ropeAttentionScaling : 1.0f, totalKey);
            }
        }
        auto keyKernel = cudaGetLastError();
        if (keyKernel != cudaSuccess) {
            MNN_ERROR("CUDAPagedAttention failed to hydrate external PIC key on GPU at layer %d: %s\n",
                      layerIndex, cudaGetErrorString(keyKernel));
            return INVALID_VALUE;
        }
    }
    meta->markExternalLayerLoaded(layerIndex);
    if (profile) {
        cudaDeviceSynchronize();
        MNN_PRINT("CUDAPagedAttention profile op=%s layer=%d tokens=%d kv_len=%d async_read=%d "
                  "direct_segments=%d direct_tokens=%d us=%llu\n",
                  restoreOp, layerIndex, static_cast<int>(totalTokens), kvLen, prefetched != nullptr ? 1 : 0,
                  directSegments, static_cast<int>(directTokens), static_cast<unsigned long long>(nowUs() - startUs));
    }
    return NO_ERROR;
}

static ErrorCode runCacheBlendScoringCUDA(PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim,
                                          int bytes, int maxSlots, int kvLen, const void* valueCacheDevice,
                                          void* valueCacheHost, int sourceSlotBase, int deviceId,
                                          std::shared_ptr<CUDAPagedAttention::SharedPagedCache::MappedBuffer>*
                                              valueWorkspace,
                                          float** scoreWorkspace, size_t* scoreWorkspaceCount, int** indexWorkspace,
                                          size_t* indexWorkspaceCount, unsigned char** usedWorkspace,
                                          size_t* usedWorkspaceCount) {
    if (meta == nullptr || !meta->needsCacheBlendScoring(layerIndex)) {
        return NO_ERROR;
    }
    const int picTokenCount = meta->cacheblend_score_pic_token_count;
    const int topK = meta->cacheblend_score_top_k;
    if (picTokenCount < 0 || topK < 0 || topK > picTokenCount ||
        meta->cacheblend_score_pic_start + picTokenCount > kvLen) {
        return INVALID_VALUE;
    }
    if (topK == 0 || picTokenCount == 0) {
        meta->setCacheBlendScoringResult({});
        return NO_ERROR;
    }
    const bool profile = profilePagedAttention();
    uint64_t totalStartUs = 0;
    uint64_t allocUs = 0;
    uint64_t readUs = 0;
    uint64_t h2dUs = 0;
    uint64_t scoreKernelUs = 0;
    uint64_t topKInitUs = 0;
    uint64_t topKUs = 0;
    uint64_t d2hUs = 0;
    uint64_t metadataUs = 0;
    size_t sourceSlotTokens = 0;
    int sourceSlotSegments = 0;
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        totalStartUs = nowUs();
    }
    if (valueWorkspace == nullptr || scoreWorkspace == nullptr || scoreWorkspaceCount == nullptr ||
        indexWorkspace == nullptr || indexWorkspaceCount == nullptr || usedWorkspace == nullptr ||
        usedWorkspaceCount == nullptr) {
        return INVALID_VALUE;
    }
    uint64_t allocStartUs = profile ? nowUs() : 0;
    if (*scoreWorkspace == nullptr || *scoreWorkspaceCount < static_cast<size_t>(picTokenCount)) {
        if (*scoreWorkspace != nullptr) {
            cudaFree(*scoreWorkspace);
            *scoreWorkspace = nullptr;
            *scoreWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(scoreWorkspace), static_cast<size_t>(picTokenCount) * sizeof(float)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *scoreWorkspaceCount = static_cast<size_t>(picTokenCount);
    }
    if (*indexWorkspace == nullptr || *indexWorkspaceCount < static_cast<size_t>(topK)) {
        if (*indexWorkspace != nullptr) {
            cudaFree(*indexWorkspace);
            *indexWorkspace = nullptr;
            *indexWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(indexWorkspace), static_cast<size_t>(topK) * sizeof(int)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *indexWorkspaceCount = static_cast<size_t>(topK);
    }
    if (*usedWorkspace == nullptr || *usedWorkspaceCount < static_cast<size_t>(picTokenCount)) {
        if (*usedWorkspace != nullptr) {
            cudaFree(*usedWorkspace);
            *usedWorkspace = nullptr;
            *usedWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(usedWorkspace), static_cast<size_t>(picTokenCount)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *usedWorkspaceCount = static_cast<size_t>(picTokenCount);
    }
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        allocUs += nowUs() - allocStartUs;
    }
    size_t scoreOffset = 0;
    for (const auto& segment : meta->cacheblend_score_segments) {
        if (segment.tokenCount == 0) {
            continue;
        }
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : batch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
        if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
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
        const int threads = 128;
        const int blocks = UP_DIV(static_cast<int>(segment.tokenCount), threads);
        const int sourceSlotStart = sourceSlotBase + static_cast<int>(scoreOffset);
        const bool useSourceSlots = valueCacheHost != nullptr && sourceSlotBase >= 0 &&
            sourceSlotStart >= 0 && sourceSlotStart + static_cast<int>(segment.tokenCount) <= maxSlots;
        if (useSourceSlots) {
            uint64_t readStartUs = profile ? nowUs() : 0;
            if (!readExternalValueSegmentToPagedCache(layer->valuePath, valueCacheHost, batch, kvHeads, maxSlots,
                                                      static_cast<size_t>(sourceSlotStart), sourceTokenCount,
                                                      sourceTokenOffset, segment.tokenCount, headDim, bytes)) {
                return INVALID_VALUE;
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (profile) {
                readUs += nowUs() - readStartUs;
            }
            sourceSlotTokens += segment.tokenCount;
            ++sourceSlotSegments;
        } else {
            const size_t tokenBytes = static_cast<size_t>(headDim) * bytes;
            const size_t segmentBytes = segment.tokenCount * tokenBytes;
            const size_t valueBytes = static_cast<size_t>(batch) * kvHeads * segmentBytes;
            uint64_t valueAllocStartUs = profile ? nowUs() : 0;
            if (!ensureMappedHostWorkspace(valueWorkspace, valueBytes, deviceId)) {
                return OUT_OF_MEMORY;
            }
            if (profile) {
                allocUs += nowUs() - valueAllocStartUs;
            }
            uint64_t readStartUs = profile ? nowUs() : 0;
            if (!readExternalValueSegmentToBuffer(layer->valuePath, (*valueWorkspace)->host,
                                                  (*valueWorkspace)->bytes, batch, kvHeads, sourceTokenCount,
                                                  sourceTokenOffset, segment.tokenCount, headDim, bytes)) {
                return INVALID_VALUE;
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (profile) {
                readUs += nowUs() - readStartUs;
            }
        }
        uint64_t scoreStartUs = profile ? nowUs() : 0;
        if (useSourceSlots && bytes == 4) {
            cacheBlendValueScoreFromPagedSourceKernel<float><<<blocks, threads>>>(
                reinterpret_cast<const float*>(valueCacheDevice), *scoreWorkspace, batch,
                static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots, sourceSlotStart,
                static_cast<int>(segment.logicalStart), static_cast<int>(scoreOffset));
        } else if (useSourceSlots) {
            cacheBlendValueScoreFromPagedSourceKernel<half><<<blocks, threads>>>(
                reinterpret_cast<const half*>(valueCacheDevice), *scoreWorkspace, batch,
                static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots, sourceSlotStart,
                static_cast<int>(segment.logicalStart), static_cast<int>(scoreOffset));
        } else if (bytes == 4) {
            cacheBlendValueScoreKernel<float><<<blocks, threads>>>(
                reinterpret_cast<const float*>(valueCacheDevice),
                reinterpret_cast<const float*>((*valueWorkspace)->device),
                *scoreWorkspace, batch, static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots,
                static_cast<int>(segment.logicalStart), static_cast<int>(scoreOffset));
        } else {
            cacheBlendValueScoreKernel<half><<<blocks, threads>>>(
                reinterpret_cast<const half*>(valueCacheDevice),
                reinterpret_cast<const half*>((*valueWorkspace)->device),
                *scoreWorkspace, batch, static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots,
                static_cast<int>(segment.logicalStart), static_cast<int>(scoreOffset));
        }
        if (cudaGetLastError() != cudaSuccess) {
            return INVALID_VALUE;
        }
        if (profile) {
            if (cudaDeviceSynchronize() != cudaSuccess) {
                return INVALID_VALUE;
            }
            scoreKernelUs += nowUs() - scoreStartUs;
        }
        scoreOffset += segment.tokenCount;
    }
    if (scoreOffset != static_cast<size_t>(picTokenCount)) {
        return INVALID_VALUE;
    }
    uint64_t topKInitStartUs = profile ? nowUs() : 0;
    if (cudaMemset(*usedWorkspace, 0, static_cast<size_t>(picTokenCount)) != cudaSuccess) {
        return INVALID_VALUE;
    }
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        topKInitUs = nowUs() - topKInitStartUs;
    }
    uint64_t topKStartUs = profile ? nowUs() : 0;
    cacheBlendTopKKernel<<<1, 256>>>(*scoreWorkspace, *indexWorkspace, *usedWorkspace, picTokenCount, topK);
    if (cudaGetLastError() != cudaSuccess) {
        return INVALID_VALUE;
    }
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        topKUs = nowUs() - topKStartUs;
    }
    std::vector<int> selected(topK);
    uint64_t d2hStartUs = profile ? nowUs() : 0;
    if (cudaMemcpy(selected.data(), *indexWorkspace, static_cast<size_t>(topK) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return INVALID_VALUE;
    }
    if (profile) {
        d2hUs = nowUs() - d2hStartUs;
    }
    uint64_t metadataStartUs = profile ? nowUs() : 0;
    std::vector<uint8_t> seen(static_cast<size_t>(picTokenCount), 0);
    for (int index : selected) {
        if (index < 0 || index >= picTokenCount || seen[static_cast<size_t>(index)] != 0) {
            return INVALID_VALUE;
        }
        seen[static_cast<size_t>(index)] = 1;
    }
    meta->setCacheBlendScoringResult(selected);
    if (profile) {
        metadataUs = nowUs() - metadataStartUs;
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        MNN_PRINT("CUDAPagedAttention profile op=cacheblend_score layer=%d pic_tokens=%d top_k=%d "
                  "segments=%d source_slot_segments=%d source_slot_tokens=%d us=%llu alloc_us=%llu "
                  "read_us=%llu h2d_us=%llu score_kernel_us=%llu "
                  "topk_init_us=%llu topk_us=%llu d2h_us=%llu metadata_us=%llu\n",
                  layerIndex, picTokenCount, topK, static_cast<int>(meta->cacheblend_score_segments.size()),
                  sourceSlotSegments, static_cast<int>(sourceSlotTokens),
                  static_cast<unsigned long long>(nowUs() - totalStartUs),
                  static_cast<unsigned long long>(allocUs),
                  static_cast<unsigned long long>(readUs),
                  static_cast<unsigned long long>(h2dUs),
                  static_cast<unsigned long long>(scoreKernelUs),
                  static_cast<unsigned long long>(topKInitUs),
                  static_cast<unsigned long long>(topKUs),
                  static_cast<unsigned long long>(d2hUs),
                  static_cast<unsigned long long>(metadataUs));
    }
    return NO_ERROR;
}

static ErrorCode runDecodeAttentionRankCUDA(PagedKVMeta* meta, int layerIndex, int batch, int queryLen,
                                            int numHeads, int kvHeads, int headDim, int bytes, int maxSlots,
                                            int kvLen, const void* queryDevice, const void* keyCacheDevice,
                                            const int* headIdsDevice, int headIdCount, float scale,
                                            float** scoreWorkspace,
                                            size_t* scoreWorkspaceCount, int** indexWorkspace,
                                            size_t* indexWorkspaceCount, unsigned char** usedWorkspace,
                                            size_t* usedWorkspaceCount) {
    if (meta == nullptr || !meta->needsPicDecodeAttentionRankCapture(layerIndex)) {
        return NO_ERROR;
    }
    if (queryDevice == nullptr || keyCacheDevice == nullptr || scoreWorkspace == nullptr ||
        scoreWorkspaceCount == nullptr || indexWorkspace == nullptr ||
        indexWorkspaceCount == nullptr || usedWorkspace == nullptr || usedWorkspaceCount == nullptr) {
        return INVALID_VALUE;
    }
    const int picStart = meta->pic_decode_attention_pic_start;
    const int picTokenCount = meta->pic_decode_attention_pic_token_count;
    const int topM = std::min(meta->pic_decode_attention_top_m, picTokenCount);
    if (picStart < 0 || picTokenCount <= 0 || topM <= 0 || picStart + picTokenCount > kvLen ||
        queryLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || numHeads % kvHeads != 0) {
        return INVALID_VALUE;
    }
    if (static_cast<int>(meta->sparse_query_logical_indices.size()) < queryLen) {
        return INVALID_VALUE;
    }
    const int qRow = queryLen - 1;
    const int qLogical = meta->sparse_query_logical_indices[static_cast<size_t>(qRow)];
    if (qLogical < 0 || qLogical >= kvLen) {
        return INVALID_VALUE;
    }
    if (*scoreWorkspace == nullptr || *scoreWorkspaceCount < static_cast<size_t>(picTokenCount)) {
        if (*scoreWorkspace != nullptr) {
            cudaFree(*scoreWorkspace);
            *scoreWorkspace = nullptr;
            *scoreWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(scoreWorkspace), static_cast<size_t>(picTokenCount) * sizeof(float)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *scoreWorkspaceCount = static_cast<size_t>(picTokenCount);
    }
    if (*indexWorkspace == nullptr || *indexWorkspaceCount < static_cast<size_t>(topM)) {
        if (*indexWorkspace != nullptr) {
            cudaFree(*indexWorkspace);
            *indexWorkspace = nullptr;
            *indexWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(indexWorkspace), static_cast<size_t>(topM) * sizeof(int)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *indexWorkspaceCount = static_cast<size_t>(topM);
    }
    if (*usedWorkspace == nullptr || *usedWorkspaceCount < static_cast<size_t>(picTokenCount)) {
        if (*usedWorkspace != nullptr) {
            cudaFree(*usedWorkspace);
            *usedWorkspace = nullptr;
            *usedWorkspaceCount = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(usedWorkspace), static_cast<size_t>(picTokenCount)) !=
            cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        *usedWorkspaceCount = static_cast<size_t>(picTokenCount);
    }

    const bool profile = profilePagedAttention();
    uint64_t startUs = 0;
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        startUs = nowUs();
    }
    const int threads = 128;
    const int blocks = UP_DIV(picTokenCount, threads);
    if (bytes == 4) {
        decodeAttentionPicScoreKernel<float><<<blocks, threads>>>(
            reinterpret_cast<const float*>(queryDevice), reinterpret_cast<const float*>(keyCacheDevice),
            headIdsDevice, *scoreWorkspace, batch, queryLen, numHeads, kvHeads, headDim, maxSlots, picStart,
            picTokenCount, qRow, qLogical, headIdCount, scale);
    } else {
        decodeAttentionPicScoreKernel<half><<<blocks, threads>>>(
            reinterpret_cast<const half*>(queryDevice), reinterpret_cast<const half*>(keyCacheDevice),
            headIdsDevice, *scoreWorkspace, batch, queryLen, numHeads, kvHeads, headDim, maxSlots, picStart,
            picTokenCount, qRow, qLogical, headIdCount, scale);
    }
    if (cudaGetLastError() != cudaSuccess) {
        return INVALID_VALUE;
    }
    if (cudaMemset(*usedWorkspace, 0, static_cast<size_t>(picTokenCount)) != cudaSuccess) {
        return INVALID_VALUE;
    }
    cacheBlendTopKKernel<<<1, 256>>>(*scoreWorkspace, *indexWorkspace, *usedWorkspace, picTokenCount, topM);
    if (cudaGetLastError() != cudaSuccess) {
        return INVALID_VALUE;
    }
    std::vector<int> selected(static_cast<size_t>(topM));
    if (cudaMemcpy(selected.data(), *indexWorkspace, static_cast<size_t>(topM) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return INVALID_VALUE;
    }
    meta->setPicDecodeAttentionRankResult(selected, meta->pic_decode_attention_step_idx);
    if (profile) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return INVALID_VALUE;
        }
        MNN_PRINT("CUDAPagedAttention profile op=decode_attention_rank layer=%d pic_tokens=%d top_m=%d "
                  "heads=%d q_logical=%d us=%llu\n",
                  layerIndex, picTokenCount, topM, headIdCount > 0 ? headIdCount : numHeads, qLogical,
                  static_cast<unsigned long long>(nowUs() - startUs));
    }
    return NO_ERROR;
}

static ErrorCode emitActiveIndicesCUDA(PagedKVMeta* meta, int layerIndex, int kvLen, Tensor* output) {
    if (output == nullptr) {
        return NO_ERROR;
    }
    const int budget = static_cast<int>(output->elementSize());
    std::vector<int> active;
    if (meta != nullptr) {
        active = meta->buildBudgetActiveLogicalIndices(kvLen, budget);
    }
    if (active.empty() && budget > 0) {
        const int count = std::min(kvLen, budget);
        active.reserve(count);
        for (int i = 0; i < count; ++i) {
            active.emplace_back(i);
        }
    }
    if (static_cast<int>(active.size()) != budget) {
        MNN_ERROR("CUDAPagedAttention layer %d active index count mismatch, budget=%d active=%d\n",
                  layerIndex, budget, static_cast<int>(active.size()));
        return INVALID_VALUE;
    }
    if (meta != nullptr && !active.empty() && !meta->activatePicRows(active, layerIndex, kvLen)) {
        return INVALID_VALUE;
    }
    if (!active.empty()) {
        if (cudaMemcpy(pagedDevPtr<int>(output), active.data(), active.size() * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return INVALID_VALUE;
        }
    }
    return NO_ERROR;
}

static ErrorCode emitIdentityIndicesCUDA(int count, Tensor* output) {
    if (output == nullptr) {
        return NO_ERROR;
    }
    count = std::min(count, static_cast<int>(output->elementSize()));
    if (count <= 0) {
        return NO_ERROR;
    }
    std::vector<int> indices(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        indices[static_cast<size_t>(i)] = i;
    }
    if (cudaMemcpy(pagedDevPtr<int>(output), indices.data(), indices.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return INVALID_VALUE;
    }
    return NO_ERROR;
}

static ErrorCode emitDecodeRecomputeIndicesCUDA(PagedKVMeta* meta, int count, Tensor* output) {
    if (output == nullptr) {
        return NO_ERROR;
    }
    count = std::min(count, static_cast<int>(output->elementSize()));
    if (count <= 0) {
        return NO_ERROR;
    }
    if (meta == nullptr || static_cast<int>(meta->sparse_query_logical_indices.size()) < count) {
        return INVALID_VALUE;
    }
    return emitIdentityIndicesCUDA(count, output);
}

static uint64_t sparseLogicalWorkCUDA(const PagedKVMeta* meta, int activeLen, int kvLen) {
    if (meta == nullptr || activeLen <= 0 || kvLen <= 0 ||
        static_cast<int>(meta->sparse_query_logical_indices.size()) < activeLen) {
        return 0;
    }
    uint64_t work = 0;
    for (int i = 0; i < activeLen; ++i) {
        const int logical = meta->sparse_query_logical_indices[static_cast<size_t>(i)];
        work += static_cast<uint64_t>(std::max(0, std::min(kvLen, logical + 1)));
    }
    return work;
}

static bool writeShapeFile(const std::string& path, int batch, int kvHeads, int headDim, int tokenCount, int bytes,
                           const KVMeta* meta) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    int ropeDim = ropeDimForExport(meta, headDim);
    os << "{\n"
       << "  \"format\": \"mnn-paged-attention-kv-shape-v1\",\n"
       << "  \"batch\": " << batch << ",\n"
       << "  \"kv_heads\": " << kvHeads << ",\n"
       << "  \"head_dim\": " << headDim << ",\n"
       << "  \"token_count\": " << tokenCount << ",\n"
       << "  \"dtype_bytes\": " << bytes << ",\n"
       << "  \"key_layout\": \"[token,batch,kv_head,head_dim]\",\n"
       << "  \"value_layout\": \"[batch,kv_head,token,head_dim]\",\n"
       << "  \"key_rope_state\": \"canonical_no_rope\",\n"
       << "  \"rope_pairing\": \"half\",\n"
       << "  \"rope_theta\": " << ((meta != nullptr && meta->rope_theta > 0.0f) ? meta->rope_theta : 10000.0f) << ",\n"
       << "  \"rope_dim\": " << ropeDim << ",\n"
       << "  \"rope_type\": \"" << ((meta != nullptr && !meta->rope_type.empty()) ? meta->rope_type : "default") << "\",\n"
       << "  \"rope_scaling_factor\": " << (meta != nullptr ? meta->rope_scaling_factor : 1.0f) << ",\n"
       << "  \"rope_scaling_low_freq_factor\": " << (meta != nullptr ? meta->rope_scaling_low_freq_factor : 1.0f) << ",\n"
       << "  \"rope_scaling_high_freq_factor\": " << (meta != nullptr ? meta->rope_scaling_high_freq_factor : 4.0f) << ",\n"
       << "  \"rope_scaling_original_max_position_embeddings\": "
       << (meta != nullptr ? meta->rope_scaling_original_max_position_embeddings : 0) << ",\n"
       << "  \"max_position_embeddings\": " << (meta != nullptr ? meta->max_position_embeddings : 0) << ",\n"
       << "  \"rope_attention_scaling\": " << (meta != nullptr ? meta->rope_attention_scaling : 1.0f) << "\n"
       << "}\n";
    return os.good();
}

CUDAPagedAttention::CUDAPagedAttention(Backend* backend, const MNN::Op* op)
    : Execution(backend), mCudaBackend(static_cast<CUDABackend*>(backend)) {
    if (op != nullptr && op->type() == OpType_PicScoreAttention) {
        mPicAttentionMode = 1;
    } else if (op != nullptr && op->type() == OpType_PicSparseAttention) {
        mPicAttentionMode = 2;
    }
    auto param = op->main_as_AttentionParam();
    if (param != nullptr) {
        mLayerIndex = param->layer_index();
        mKVSharedLayerIndex = param->kv_shared_layer_index();
        mIsKVShared = mKVSharedLayerIndex >= 0;
    }
    mMeta = static_cast<PagedKVMeta*>(mCudaBackend->getMetaPtr());
    mCache.reset(new SharedPagedCache);
}

CUDAPagedAttention::~CUDAPagedAttention() {
    if (mPrefillQK != nullptr) {
        cudaFree(mPrefillQK);
        mPrefillQK = nullptr;
    }
    if (mPrefillSoftmax != nullptr) {
        cudaFree(mPrefillSoftmax);
        mPrefillSoftmax = nullptr;
    }
    if (mExternalKey != nullptr) {
        cudaFree(mExternalKey);
        mExternalKey = nullptr;
        mExternalKeyBytes = 0;
    }
    if (mExternalValue != nullptr) {
        cudaFree(mExternalValue);
        mExternalValue = nullptr;
        mExternalValueBytes = 0;
    }
    if (mCacheBlendScores != nullptr) {
        cudaFree(mCacheBlendScores);
        mCacheBlendScores = nullptr;
        mCacheBlendScoreCount = 0;
    }
    if (mCacheBlendIndices != nullptr) {
        cudaFree(mCacheBlendIndices);
        mCacheBlendIndices = nullptr;
        mCacheBlendIndexCount = 0;
    }
    if (mCacheBlendUsed != nullptr) {
        cudaFree(mCacheBlendUsed);
        mCacheBlendUsed = nullptr;
        mCacheBlendUsedCount = 0;
    }
    if (mDecodeAttentionHeadIds != nullptr) {
        cudaFree(mDecodeAttentionHeadIds);
        mDecodeAttentionHeadIds = nullptr;
        mDecodeAttentionHeadIdCount = 0;
        mDecodeAttentionHeadIdsHost.clear();
    }
}

ErrorCode CUDAPagedAttention::ensureCache(int maxSlots, int batch, int kvHeads, int headDim) {
    if (maxSlots <= 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0) {
        return INVALID_VALUE;
    }
    const bool prefixCacheWrite =
        mMeta != nullptr && !mMeta->file_name.empty() && mMeta->file_flag == KVMeta::PendingWrite;
    const bool requiresMappedPagedCache = !prefixCacheWrite && shouldUseMappedPagedCache(mCudaBackend);
    if (mCache && mCache->key && mCache->value && mCache->maxSlots == maxSlots &&
        mCache->batch == batch && mCache->kvHeads == kvHeads && mCache->headDim == headDim &&
        mCache->precision == mPrecision && !(prefixCacheWrite && mCache->zeroCopyKV) &&
        !(requiresMappedPagedCache && !mCache->zeroCopyKV)) {
        if (mCache->zeroCopyKV && mLayerIndex >= 0) {
            registerExternalLayerMappedTarget(mMeta, mLayerIndex, batch, kvHeads, headDim, mPrecision, maxSlots,
                                              mCache->mappedKey, mCache->mappedValue);
        }
        return NO_ERROR;
    }
    if (!mCache) {
        mCache.reset(new SharedPagedCache);
    }
    mCache->mappedKey.reset();
    mCache->mappedValue.reset();
    mCache->zeroCopyKV = false;
    if (mPrecision == 4) {
        mCache->key.reset(Tensor::createDevice<float>({maxSlots, batch, kvHeads, headDim}));
        mCache->value.reset(Tensor::createDevice<float>({batch, kvHeads, maxSlots, headDim}));
    } else {
        mCache->key.reset(Tensor::createDevice<uint16_t>({maxSlots, batch, kvHeads, headDim}));
        mCache->value.reset(Tensor::createDevice<uint16_t>({batch, kvHeads, maxSlots, headDim}));
    }
    mCache->sparseQuery.reset(Tensor::createDevice<int>({maxSlots}));
    if (!mCache->key || !mCache->value || !mCache->sparseQuery) {
        return OUT_OF_MEMORY;
    }
    const size_t keyBytes = static_cast<size_t>(maxSlots) * batch * kvHeads * headDim * mPrecision;
    const size_t valueBytes = static_cast<size_t>(batch) * kvHeads * maxSlots * headDim * mPrecision;
    if (requiresMappedPagedCache) {
        const int deviceId = mCudaBackend->getCUDARuntime()->device_id();
        auto mappedKey = makeMappedPagedBuffer(keyBytes, deviceId);
        auto mappedValue = makeMappedPagedBuffer(valueBytes, deviceId);
        if (mappedKey != nullptr && mappedValue != nullptr) {
            mCache->mappedKey = std::move(mappedKey);
            mCache->mappedValue = std::move(mappedValue);
            mCache->key->buffer().device = reinterpret_cast<uint64_t>(mCache->mappedKey->device);
            mCache->value->buffer().device = reinterpret_cast<uint64_t>(mCache->mappedValue->device);
            TensorUtils::getDescribeOrigin(mCache->key.get())->setBackend(mCudaBackend);
            TensorUtils::getDescribeOrigin(mCache->value.get())->setBackend(mCudaBackend);
            ::memset(mCache->mappedKey->host, 0, keyBytes);
            ::memset(mCache->mappedValue->host, 0, valueBytes);
            mCache->zeroCopyKV = true;
            if (mLayerIndex >= 0) {
                registerExternalLayerMappedTarget(mMeta, mLayerIndex, batch, kvHeads, headDim, mPrecision, maxSlots,
                                                  mCache->mappedKey, mCache->mappedValue);
            }
        } else {
            mCache->mappedKey.reset();
            mCache->mappedValue.reset();
        }
    }
    if (!mCache->zeroCopyKV) {
        if (!mCudaBackend->onAcquireBuffer(mCache->key.get(), Backend::STATIC)) {
            return OUT_OF_MEMORY;
        }
        if (!mCudaBackend->onAcquireBuffer(mCache->value.get(), Backend::STATIC)) {
            return OUT_OF_MEMORY;
        }
    }
    if (!mCudaBackend->onAcquireBuffer(mCache->sparseQuery.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    if (!mCache->zeroCopyKV) {
        cudaMemset(pagedDevPtr<void>(mCache->key.get()), 0, keyBytes);
        cudaMemset(pagedDevPtr<void>(mCache->value.get()), 0, valueBytes);
    }
    mCache->maxSlots = maxSlots;
    mCache->batch = batch;
    mCache->kvHeads = kvHeads;
    mCache->headDim = headDim;
    mCache->precision = mPrecision;
    mCache->sparseQueryHost.clear();
    return NO_ERROR;
}

bool CUDAPagedAttention::ensurePrefillTemp(size_t elements) {
    if (elements == 0) {
        return true;
    }
    if (mPrefillQK != nullptr && mPrefillSoftmax != nullptr && mPrefillElements >= elements) {
        return true;
    }
    if (mPrefillQK != nullptr) {
        cudaFree(mPrefillQK);
        mPrefillQK = nullptr;
    }
    if (mPrefillSoftmax != nullptr) {
        cudaFree(mPrefillSoftmax);
        mPrefillSoftmax = nullptr;
    }
    mPrefillElements = 0;
    size_t bytes = elements * sizeof(float);
    if (cudaMalloc(&mPrefillQK, bytes) != cudaSuccess) {
        mPrefillQK = nullptr;
        return false;
    }
    if (cudaMalloc(&mPrefillSoftmax, bytes) != cudaSuccess) {
        cudaFree(mPrefillQK);
        mPrefillQK = nullptr;
        mPrefillSoftmax = nullptr;
        return false;
    }
    mPrefillElements = elements;
    return true;
}

ErrorCode CUDAPagedAttention::syncSparseQuery(int attnLen, const int** devicePtr) {
    if (devicePtr != nullptr) {
        *devicePtr = nullptr;
    }
    if (attnLen <= 0 || mMeta == nullptr || !mMeta->sparse_query_active) {
        return NO_ERROR;
    }
    if (!mCache || !mCache->sparseQuery || attnLen > mCache->maxSlots) {
        return OUT_OF_MEMORY;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
        return INVALID_VALUE;
    }
    const int* hostPtr = mMeta->sparse_query_logical_indices.data();
    if (static_cast<int>(mCache->sparseQueryHost.size()) != attnLen ||
        !std::equal(mCache->sparseQueryHost.begin(), mCache->sparseQueryHost.end(), hostPtr)) {
        if (cudaMemcpy(pagedDevPtr<int>(mCache->sparseQuery.get()), hostPtr,
                       static_cast<size_t>(attnLen) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
            mCache->sparseQueryHost.clear();
            return INVALID_VALUE;
        }
        mCache->sparseQueryHost.assign(hostPtr, hostPtr + attnLen);
    }
    if (devicePtr != nullptr) {
        *devicePtr = pagedDevPtr<int>(mCache->sparseQuery.get());
    }
    return NO_ERROR;
}

ErrorCode CUDAPagedAttention::syncDecodeAttentionHeadIds(const int** devicePtr, int* count) {
    if (devicePtr != nullptr) {
        *devicePtr = nullptr;
    }
    if (count != nullptr) {
        *count = 0;
    }
    if (mMeta == nullptr || mMeta->pic_decode_attention_head_ids.empty()) {
        return NO_ERROR;
    }
    const auto& heads = mMeta->pic_decode_attention_head_ids;
    if (mDecodeAttentionHeadIds == nullptr || mDecodeAttentionHeadIdCount < heads.size()) {
        if (mDecodeAttentionHeadIds != nullptr) {
            cudaFree(mDecodeAttentionHeadIds);
            mDecodeAttentionHeadIds = nullptr;
            mDecodeAttentionHeadIdCount = 0;
            mDecodeAttentionHeadIdsHost.clear();
        }
        if (cudaMalloc(reinterpret_cast<void**>(&mDecodeAttentionHeadIds),
                       heads.size() * sizeof(int)) != cudaSuccess) {
            return OUT_OF_MEMORY;
        }
        mDecodeAttentionHeadIdCount = heads.size();
    }
    if (mDecodeAttentionHeadIdsHost != heads) {
        if (cudaMemcpy(mDecodeAttentionHeadIds, heads.data(), heads.size() * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            mDecodeAttentionHeadIdsHost.clear();
            return INVALID_VALUE;
        }
        mDecodeAttentionHeadIdsHost = heads;
    }
    if (devicePtr != nullptr) {
        *devicePtr = mDecodeAttentionHeadIds;
    }
    if (count != nullptr) {
        *count = static_cast<int>(heads.size());
    }
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
    int maxSlots = std::max(mNewKvSeqLen, mQuerySeqLen);
    if (mMeta != nullptr) {
        maxSlots = std::max(maxSlots, mMeta->request_capacity > 0 ? mMeta->request_capacity : mMeta->max_tokens);
        if (maxSlots <= 0) {
            maxSlots = static_cast<int>(mMeta->previous) + mNewKvSeqLen;
        }
        maxSlots = maxSlotsWithPicSourceSlotsCUDA(mMeta, maxSlots);
    }
    auto err = ensureCache(maxSlots, mBatch, mKvNumHead, mHeadDim);
    if (err != NO_ERROR) {
        return err;
    }
    return NO_ERROR;
}

ErrorCode CUDAPagedAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const Tensor* mask = inputs.size() > 3 ? inputs[3] : nullptr;

    if (mMeta != nullptr && mMeta->request_capacity <= 0 && !mMeta->request_active) {
        mMeta->beginRequest(std::max(std::max(mNewKvSeqLen, mQuerySeqLen), mMeta->max_tokens));
    }

    int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : 0);
    const bool decodeStep = mMeta != nullptr && mQuerySeqLen == 1 && mNewKvSeqLen == 1 &&
        mMeta->previous > 0 && !mMeta->cacheblend_score_active && !mMeta->pic_graph_active_plan_ready;
    const bool forcePlainSparseQuery = mMeta != nullptr && mMeta->sparse_query_active &&
        mMeta->sparse_query_force_plain_attention;
    const bool picRuntimeActive = mMeta != nullptr &&
        (mMeta->sparse_query_active || mMeta->cacheblend_score_active || mMeta->pic_graph_active_plan_ready);
    const bool sparseBlockedBeforeStart =
        !decodeStep && mMeta != nullptr && mMeta->sparseQueryBlockedBeforeLayer(layerIndex);
    const int effectivePicAttentionMode =
        (decodeStep || !picRuntimeActive || sparseBlockedBeforeStart || forcePlainSparseQuery) ? 0 : mPicAttentionMode;

    int reverse = reverseCount(mMeta);
    int baseLogical = 0;
    int kvWriteLen = mNewKvSeqLen;
    int attnLen = output->length(1);
    bool sparseQuery = mMeta != nullptr && !sparseBlockedBeforeStart && mMeta->sparseQueryActiveForLayer(layerIndex);
    const bool picDecodeRecompute = mMeta != nullptr && mMeta->pic_decode_recompute_active;
    const bool scoreAttention = effectivePicAttentionMode == 1 && !picDecodeRecompute;
    if (scoreAttention) {
        sparseQuery = false;
    }
    if (mMeta != nullptr) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        baseLogical = static_cast<int>(kept) + reverse;
        kvWriteLen = mMeta->add > 0 ? static_cast<int>(std::min<size_t>(mMeta->add, mNewKvSeqLen)) : mNewKvSeqLen;
    }
    if (sparseQuery) {
        if (!mMeta->validateSparseQueryRows(attnLen, mQuerySeqLen, false)) {
            MNN_ERROR("CUDAPagedAttention layer %d invalid sparse rows before K/V write, query=%d attn=%d "
                      "active=%d logical_length=%d\n",
                      layerIndex, mQuerySeqLen, attnLen,
                      static_cast<int>(mMeta->sparse_query_logical_indices.size()), mMeta->logical_length);
            return INVALID_VALUE;
        }
        if (mNewKvSeqLen < attnLen) {
            MNN_ERROR("CUDAPagedAttention layer %d sparse K/V rows %d smaller than active attention rows %d\n",
                      layerIndex, mNewKvSeqLen, attnLen);
            return INVALID_VALUE;
        }
        baseLogical = 0;
        kvWriteLen = attnLen;
    } else if (effectivePicAttentionMode == 2) {
        MNN_ERROR("CUDA PicSparseAttention layer %d requires active sparse rows from PicScoreAttention\n",
                  layerIndex);
        return INVALID_VALUE;
    }
    int kvLen = sparseQuery ? std::max(0, mMeta->logical_length) : (baseLogical + kvWriteLen);
    const bool prefixCacheWrite =
        mMeta != nullptr && !mMeta->file_name.empty() && mMeta->file_flag == KVMeta::PendingWrite;
    if (kvWriteLen > mNewKvSeqLen) {
        MNN_ERROR("CUDAPagedAttention layer %d invalid KV copy length, inputLen=%d kvWriteLen=%d "
                  "query=%d attn=%d base=%d kvLen=%d\n",
                  layerIndex, mNewKvSeqLen, kvWriteLen, mQuerySeqLen, attnLen, baseLogical, kvLen);
        return INVALID_VALUE;
    }
    if (prefixCacheWrite) {
        const bool keyShapeOk = key != nullptr && key->dimensions() >= 4 &&
            key->length(0) == mBatch && key->length(1) == mNewKvSeqLen &&
            key->length(2) == mKvNumHead && key->length(3) == mHeadDim;
        const bool valueShapeOk = value != nullptr && value->dimensions() >= 4 &&
            value->length(0) == mBatch && value->length(1) == mNewKvSeqLen &&
            value->length(2) == mKvNumHead && value->length(3) == mHeadDim;
        if (!keyShapeOk || !valueShapeOk) {
            MNN_ERROR("CUDAPagedAttention: PendingWrite K/V shape mismatch at layer %d, query=%s key=%s value=%s "
                      "batch=%d new_kv=%d kv_heads=%d head_dim=%d\n",
                      layerIndex, tensorShapeString(query).c_str(), tensorShapeString(key).c_str(),
                      tensorShapeString(value).c_str(), mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim);
            return INVALID_VALUE;
        }
    }
    if (prefixCacheWrite && mCache != nullptr && mCache->zeroCopyKV) {
        int stableSlots = std::max({mCache->maxSlots, kvLen, mQuerySeqLen, mNewKvSeqLen,
                                    mMeta->request_capacity > 0 ? mMeta->request_capacity : mMeta->max_tokens});
        auto resizeForPrefixWrite = ensureCache(std::max(1, stableSlots), mBatch, mKvNumHead, mHeadDim);
        if (resizeForPrefixWrite != NO_ERROR) {
            return resizeForPrefixWrite;
        }
    }
    if (kvLen > mCache->maxSlots) {
        MNN_ERROR("CUDAPagedAttention layer %d needs %d slots, cache capacity is %d\n", mLayerIndex, kvLen,
                  mCache->maxSlots);
        return OUT_OF_MEMORY;
    }
    ErrorCode err = NO_ERROR;
    const int* sparseQueryDevice = nullptr;
    if (sparseQuery) {
        err = syncSparseQuery(attnLen, &sparseQueryDevice);
        if (err != NO_ERROR) {
            return err;
        }
    }

    cudaStream_t stream = 0;
    const bool profile = profilePagedAttention();
    const bool nvtx = nvtxPagedAttention();
    const bool debugSync = envFlagEnabled("MNN_PAGED_ATTENTION_DEBUG_SYNC", false);
    ScopedNvtxRange layerNvtx(nvtxLayerRangeName("paged_attention_layer_total", layerIndex, mQuerySeqLen, attnLen,
                                                 kvLen), nvtx);
    if (mCache->zeroCopyKV) {
        registerExternalLayerMappedTarget(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mPrecision,
                                          mCache->maxSlots, mCache->mappedKey, mCache->mappedValue);
    }
    auto restore = restoreExternalSegmentsCUDA(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mPrecision,
                                               mCache->maxSlots, kvLen, pagedDevPtr<void>(mCache->key.get()));
    if (restore != NO_ERROR) {
        return restore;
    }

    if (!mIsKVShared && kvWriteLen > 0) {
        ScopedNvtxRange copyNvtx(nvtxLayerRangeName("write_current_kv_to_paged_cache", layerIndex, mQuerySeqLen,
                                                    kvWriteLen, kvLen), nvtx);
        if (prefixCacheWrite) {
            auto previousLaunchError = cudaGetLastError();
            auto previousSyncError = previousLaunchError == cudaSuccess ? cudaDeviceSynchronize() : previousLaunchError;
            if (previousSyncError != cudaSuccess) {
                MNN_PRINT("CUDAPagedAttention: CUDA error before pending-write KV copy at layer %d, "
                          "query=%s key=%s value=%s maxSlots=%d kvLen=%d kvWriteLen=%d "
                          "request_capacity=%d logical_length=%d previous=%zu remove=%zu add=%zu "
                          "launch=%s sync=%s\n",
                          layerIndex, tensorShapeString(query).c_str(), tensorShapeString(key).c_str(),
                          tensorShapeString(value).c_str(), mCache->maxSlots, kvLen, kvWriteLen,
                          mMeta ? mMeta->request_capacity : 0, mMeta ? mMeta->logical_length : 0,
                          mMeta ? mMeta->previous : 0, mMeta ? mMeta->remove : 0, mMeta ? mMeta->add : 0,
                          cudaGetErrorString(previousLaunchError), cudaGetErrorString(previousSyncError));
                return INVALID_VALUE;
            }
        }
        dim3 block(32, 8, 1);
        dim3 grid(UP_DIV(mHeadDim, block.x), UP_DIV(kvWriteLen, block.y), UP_DIV(mBatch * mKvNumHead, block.z));
        if (mPrecision == 4) {
            copyPagedKVKernel<float><<<grid, block, 0, stream>>>(
                pagedDevPtr<float>(key), pagedDevPtr<float>(value), pagedDevPtr<float>(mCache->key.get()),
                pagedDevPtr<float>(mCache->value.get()), mBatch, mNewKvSeqLen, kvWriteLen, mKvNumHead,
                mHeadDim, baseLogical, mCache->maxSlots, sparseQueryDevice);
        } else {
            copyPagedKVKernel<half><<<grid, block, 0, stream>>>(
                pagedDevPtr<half>(key), pagedDevPtr<half>(value), pagedDevPtr<half>(mCache->key.get()),
                pagedDevPtr<half>(mCache->value.get()), mBatch, mNewKvSeqLen, kvWriteLen, mKvNumHead,
                mHeadDim, baseLogical, mCache->maxSlots, sparseQueryDevice);
        }
        if (debugSync) {
            err = debugCheckPagedAttentionKernel(true, "copy_paged_kv", layerIndex, -1, baseLogical, kvWriteLen,
                                                 mNewKvSeqLen, mKvNumHead, mKvNumHead, mHeadDim);
            if (err != NO_ERROR) {
                return err;
            }
        } else if (prefixCacheWrite) {
            auto copyLaunch = cudaGetLastError();
            auto copySync = copyLaunch == cudaSuccess ? cudaDeviceSynchronize() : copyLaunch;
            if (copySync != cudaSuccess) {
                MNN_PRINT("CUDAPagedAttention: failed to sync pending-write KV copy at layer %d, "
                          "inputLen=%d query=%d attn=%d base=%d kvLen=%d maxSlots=%d "
                          "kvWriteLen=%d request_capacity=%d logical_length=%d previous=%zu "
                          "remove=%zu add=%zu keyShape=%s valueShape=%s launch=%s sync=%s\n",
                          layerIndex, mNewKvSeqLen, mQuerySeqLen, attnLen, baseLogical, kvLen,
                          mCache->maxSlots, kvWriteLen, mMeta ? mMeta->request_capacity : 0,
                          mMeta ? mMeta->logical_length : 0, mMeta ? mMeta->previous : 0,
                          mMeta ? mMeta->remove : 0, mMeta ? mMeta->add : 0, tensorShapeString(key).c_str(),
                          tensorShapeString(value).c_str(), cudaGetErrorString(copyLaunch),
                          cudaGetErrorString(copySync));
                return INVALID_VALUE;
            }
        } else {
            checkKernelErrors;
        }
    }

    void* mappedValueHost = (mCache->zeroCopyKV && mCache->mappedValue != nullptr) ? mCache->mappedValue->host : nullptr;
    const int sourceSlotBase = picCacheSourceSlotBaseCUDA(mMeta, kvLen);
    auto cacheBlendScore = runCacheBlendScoringCUDA(
        mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mPrecision, mCache->maxSlots, kvLen,
        pagedDevPtr<void>(mCache->value.get()), mappedValueHost, sourceSlotBase,
        cudaBackendDeviceId(mCudaBackend), &mExternalValueMapped,
        &mCacheBlendScores, &mCacheBlendScoreCount, &mCacheBlendIndices, &mCacheBlendIndexCount,
        &mCacheBlendUsed, &mCacheBlendUsedCount);
    if (cacheBlendScore != NO_ERROR) {
        return cacheBlendScore;
    }

    if ((scoreAttention || picDecodeRecompute) && outputs.size() > 1) {
        err = picDecodeRecompute ? emitDecodeRecomputeIndicesCUDA(mMeta, attnLen, outputs[1])
                                 : emitActiveIndicesCUDA(mMeta, layerIndex, kvLen, outputs[1]);
        if (err != NO_ERROR) {
            return err;
        }
        sparseQuery = mMeta != nullptr && mMeta->sparseQueryActiveForLayer(layerIndex);
        if (sparseQuery) {
            const bool allowFullQueryRows = scoreAttention && mQuerySeqLen > attnLen;
            if (!mMeta->validateSparseQueryRows(attnLen, mQuerySeqLen, allowFullQueryRows)) {
                MNN_ERROR("CUDAPagedAttention layer %d invalid sparse rows after active-index emit, query=%d "
                          "attn=%d active=%d logical_length=%d allow_full_q=%d\n",
                          layerIndex, mQuerySeqLen, attnLen,
                          static_cast<int>(mMeta->sparse_query_logical_indices.size()), mMeta->logical_length,
                          allowFullQueryRows ? 1 : 0);
                return INVALID_VALUE;
            }
            baseLogical = 0;
            kvLen = std::max(0, mMeta->logical_length);
            err = syncSparseQuery(attnLen, &sparseQueryDevice);
            if (err != NO_ERROR) {
                return err;
            }
        }
    } else if (outputs.size() > 1) {
        err = emitIdentityIndicesCUDA(attnLen, outputs[1]);
        if (err != NO_ERROR) {
            return err;
        }
    }

    if (mMeta != nullptr && !mMeta->file_name.empty() && mMeta->file_flag == KVMeta::PendingWrite && kvLen > 0) {
        auto prefixDir = mCudaBackend->getRuntime()->hint().prefixcacheDirPath;
        MNNCreateDir(prefixDir.c_str());
        int layerIndex = mLayerIndex >= 0 ? mLayerIndex : mMeta->layer_index;
        std::string basePath = MNNFilePathConcat(prefixDir, mMeta->file_name) + "_" + std::to_string(layerIndex);

        const int deviceId = cudaBackendDeviceId(mCudaBackend);
        const size_t keyBytes = static_cast<size_t>(kvLen) * mBatch * mKvNumHead * mHeadDim * mPrecision;
        size_t valueStorageBytes = static_cast<size_t>(mBatch) * mKvNumHead * mCache->maxSlots * mHeadDim * mPrecision;
        const size_t valueBytes = static_cast<size_t>(mBatch) * mKvNumHead * kvLen * mHeadDim * mPrecision;
        auto& exportWorkspace = mappedPagedExportWorkspace(deviceId);
        std::lock_guard<std::mutex> exportWorkspaceLock(exportWorkspace.mutex);
        bool keyReady = false;
        if (ensureMappedHostWorkspace(&exportWorkspace.scratch, keyBytes, deviceId)) {
            int ropeDim = ropeDimForExport(mMeta, mHeadDim);
            ropeDim = std::min(ropeDim, mHeadDim);
            ropeDim = (ropeDim / 2) * 2;
            int oldContext = mMeta != nullptr && mMeta->rope_scaling_original_max_position_embeddings > 0
                ? mMeta->rope_scaling_original_max_position_embeddings
                : (mMeta != nullptr ? mMeta->max_position_embeddings : 0);
            dim3 keyBlock(32, 8, 1);
            dim3 keyGrid(UP_DIV(mHeadDim, keyBlock.x), UP_DIV(kvLen, keyBlock.y),
                         UP_DIV(mBatch * mKvNumHead, keyBlock.z));
            if (mPrecision == 4) {
                exportCanonicalPagedKeyKernel<float><<<keyGrid, keyBlock, 0, stream>>>(
                    pagedDevPtr<float>(mCache->key.get()), reinterpret_cast<float*>(exportWorkspace.scratch->device),
                    mBatch, kvLen, mKvNumHead, mHeadDim, mCache->maxSlots, ropeDim,
                    mMeta && mMeta->rope_theta > 0.0f ? mMeta->rope_theta : 10000.0f,
                    ropeTypeCode(mMeta), mMeta ? std::max(mMeta->rope_scaling_factor, 1.0f) : 1.0f,
                    mMeta ? std::max(mMeta->rope_scaling_low_freq_factor, 1.0e-6f) : 1.0f,
                    mMeta ? std::max(mMeta->rope_scaling_high_freq_factor, 1.0e-6f) : 4.0f,
                    oldContext, mMeta ? mMeta->rope_attention_scaling : 1.0f);
            } else {
                exportCanonicalPagedKeyKernel<half><<<keyGrid, keyBlock, 0, stream>>>(
                    pagedDevPtr<half>(mCache->key.get()), reinterpret_cast<half*>(exportWorkspace.scratch->device),
                    mBatch, kvLen, mKvNumHead, mHeadDim, mCache->maxSlots, ropeDim,
                    mMeta && mMeta->rope_theta > 0.0f ? mMeta->rope_theta : 10000.0f,
                    ropeTypeCode(mMeta), mMeta ? std::max(mMeta->rope_scaling_factor, 1.0f) : 1.0f,
                    mMeta ? std::max(mMeta->rope_scaling_low_freq_factor, 1.0e-6f) : 1.0f,
                    mMeta ? std::max(mMeta->rope_scaling_high_freq_factor, 1.0e-6f) : 4.0f,
                    oldContext, mMeta ? mMeta->rope_attention_scaling : 1.0f);
            }
            auto keyKernel = cudaGetLastError();
            auto keySync = keyKernel == cudaSuccess ? cudaDeviceSynchronize() : keyKernel;
            if (keyKernel == cudaSuccess && keySync == cudaSuccess) {
                keyReady = true;
            } else {
                MNN_PRINT("CUDAPagedAttention: canonical mapped key export failed at layer %d, launch=%s sync=%s key=%zu maxSlots=%d kvLen=%d\n",
                          layerIndex, cudaGetErrorString(keyKernel), cudaGetErrorString(keySync),
                          keyBytes, mCache->maxSlots, kvLen);
            }
        }
        if (!keyReady) {
            MNN_PRINT("CUDAPagedAttention: failed to export canonical mapped key, key=%zu maxSlots=%d kvLen=%d\n",
                      keyBytes, mCache->maxSlots, kvLen);
            return OUT_OF_MEMORY;
        }
        if (!writeBinaryFile(basePath + ".k", exportWorkspace.scratch->host, keyBytes)) {
            MNN_PRINT("CUDAPagedAttention: failed to export key cache: %s\n", (basePath + ".k").c_str());
            return FILE_CREATE_FAILED;
        }
        bool valueStorageReady = false;
        const int8_t* valueStorageHost = nullptr;
        if (mCache->zeroCopyKV && mCache->mappedValue != nullptr && mCache->mappedValue->host != nullptr &&
            mCache->mappedValue->bytes >= valueStorageBytes) {
            valueStorageHost = reinterpret_cast<const int8_t*>(mCache->mappedValue->host);
            valueStorageReady = true;
        } else {
            if (ensureMappedHostWorkspace(&exportWorkspace.valueStorage, valueStorageBytes, deviceId) &&
                cudaMemcpy(exportWorkspace.valueStorage->host, pagedDevPtr<void>(mCache->value.get()),
                           valueStorageBytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
                valueStorageHost = reinterpret_cast<const int8_t*>(exportWorkspace.valueStorage->host);
                valueStorageReady = true;
            }
        }
        const bool valueDataReady = valueStorageReady &&
            ensureMappedHostWorkspace(&exportWorkspace.scratch, valueBytes, deviceId);
        if (valueDataReady) {
            auto* valueData = reinterpret_cast<int8_t*>(exportWorkspace.scratch->host);
            ::memset(valueData, 0, valueBytes);
            for (int l = 0; l < kvLen; ++l) {
                int slot = l;
                for (int b = 0; b < mBatch; ++b) {
                    for (int h = 0; h < mKvNumHead; ++h) {
                        const int8_t* srcV = valueStorageHost +
                            ((b * mKvNumHead + h) * mCache->maxSlots + slot) * mHeadDim * mPrecision;
                        int8_t* dstV = valueData +
                            ((b * mKvNumHead + h) * kvLen + l) * mHeadDim * mPrecision;
                        ::memcpy(dstV, srcV, mHeadDim * mPrecision);
                    }
                }
            }
            if (!writeBinaryFile(basePath + ".v", valueData, valueBytes)) {
                MNN_PRINT("CUDAPagedAttention: failed to export value cache: %s\n", (basePath + ".v").c_str());
                MNNRemoveFile((basePath + ".k").c_str());
                return FILE_CREATE_FAILED;
            }
            if (!writeShapeFile(basePath + ".json", mBatch, mKvNumHead, mHeadDim, kvLen, mPrecision, mMeta)) {
                MNN_PRINT("CUDAPagedAttention: failed to export shape metadata: %s\n", (basePath + ".json").c_str());
                MNNRemoveFile((basePath + ".k").c_str());
                MNNRemoveFile((basePath + ".v").c_str());
                return FILE_CREATE_FAILED;
            }
        } else {
            MNN_PRINT("CUDAPagedAttention: failed to stage mapped value for export, key=%zu valueStorage=%zu value=%zu maxSlots=%d kvLen=%d zeroCopy=%d\n",
                      keyBytes, valueStorageBytes, valueBytes, mCache->maxSlots, kvLen,
                      mCache->zeroCopyKV ? 1 : 0);
            return OUT_OF_MEMORY;
        }
        if (mLayerIndex < 0) {
            mMeta->layer_index = (mMeta->layer_index + 1) % std::max(1, mMeta->layer_nums);
        }
    }

    size_t outputBytes = output->elementSize() * static_cast<size_t>(mPrecision);
    cudaMemset(pagedDevPtr<void>(output), 0, outputBytes);
    if (attnLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }

    mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    bool useMask = mask != nullptr && mask->elementSize() > 1 && mask->getType().code == halide_type_float;
    int maskElements = useMask ? static_cast<int>(mask->elementSize()) : 0;
    const uint64_t attentionStartUs = profile ? nowUs() : 0;
    const bool queryRowsAreFull = scoreAttention && sparseQuery && mQuerySeqLen > attnLen;
    const bool fullCausalMask = mMeta != nullptr && mMeta->full_causal_attention_mask;
    const bool repairDecodeCausal = picDecodeRecompute && sparseQuery && fullCausalMask;
    const bool ordinaryDecodeCausal = decodeStep && !sparseQuery && !picDecodeRecompute && fullCausalMask &&
        attnLen == 1 && mQuerySeqLen == 1 && mNewKvSeqLen == 1;
    CudaSparseQTileVariant decodeRepairQTileVariant = kCudaSparseQTileNone;
    if (picDecodeRecompute && sparseQuery && attnLen >= 1 &&
        cudaDecodeRepairQTileEnabled(mHeadDim, attnLen)) {
        decodeRepairQTileVariant = selectCudaDecodeRepairQTileVariant(mHeadDim, attnLen);
    }
    const bool useDecodeRepairQTile = decodeRepairQTileVariant != kCudaSparseQTileNone;
    DecodeAttentionRankCaptureCUDA qtileDecodeRank;
    uint64_t qtileDecodeRankPrepUs = 0;
    auto prepareDecodeAttentionRankForQTile = [&]() -> ErrorCode {
        if (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex)) {
            return NO_ERROR;
        }
        const int qIndex = attnLen - 1;
        if (qIndex < 0 || static_cast<int>(mMeta->sparse_query_logical_indices.size()) <= qIndex) {
            return INVALID_VALUE;
        }
        const int qLogical = mMeta->sparse_query_logical_indices[static_cast<size_t>(qIndex)];
        const int qRow = queryRowsAreFull ? qLogical : qIndex;
        const int picStart = mMeta->pic_decode_attention_pic_start;
        const int picTokenCount = mMeta->pic_decode_attention_pic_token_count;
        const int topM = std::min(mMeta->pic_decode_attention_top_m, picTokenCount);
        qtileDecodeRank.requested = true;
        qtileDecodeRank.picStart = picStart;
        qtileDecodeRank.picTokenCount = picTokenCount;
        qtileDecodeRank.topM = topM;
        qtileDecodeRank.qRow = qRow;
        qtileDecodeRank.qLogical = qLogical;
        const auto& attentionHeadIds = mMeta->pic_decode_attention_head_ids;
        if (attentionHeadIds.empty()) {
            qtileDecodeRank.validHeadCount = mNumHead;
        } else {
            qtileDecodeRank.validHeadCount = 0;
            for (int head : attentionHeadIds) {
                if (head >= 0 && head < mNumHead) {
                    ++qtileDecodeRank.validHeadCount;
                }
            }
        }
        const bool canFuseRank = picStart >= 0 && picTokenCount > 0 && topM > 0 &&
            picStart + picTokenCount <= kvLen && picStart + picTokenCount <= qLogical + 1 &&
            qRow >= 0 && qRow < mQuerySeqLen && qtileDecodeRank.validHeadCount > 0 && mBatch > 0;
        if (!canFuseRank) {
            return NO_ERROR;
        }
        uint64_t prepStartUs = profile ? nowUs() : 0;
        const int* decodeAttentionHeadIdsDevice = nullptr;
        int decodeAttentionHeadIdCount = 0;
        auto prepErr = syncDecodeAttentionHeadIds(&decodeAttentionHeadIdsDevice, &decodeAttentionHeadIdCount);
        if (prepErr != NO_ERROR) {
            return prepErr;
        }
        prepErr = ensureDecodeAttentionRankWorkspacesCUDA(
            picTokenCount, topM, &mCacheBlendScores, &mCacheBlendScoreCount,
            &mCacheBlendIndices, &mCacheBlendIndexCount, &mCacheBlendUsed, &mCacheBlendUsedCount);
        if (prepErr != NO_ERROR) {
            return prepErr;
        }
        if (cudaMemset(mCacheBlendScores, 0, static_cast<size_t>(picTokenCount) * sizeof(float)) != cudaSuccess) {
            return INVALID_VALUE;
        }
        qtileDecodeRank.headIdsDevice = decodeAttentionHeadIdsDevice;
        qtileDecodeRank.headIdCount = decodeAttentionHeadIdCount;
        qtileDecodeRank.scoreScale =
            1.0f / static_cast<float>(qtileDecodeRank.validHeadCount * mBatch);
        if (profile) {
            if (cudaDeviceSynchronize() != cudaSuccess) {
                return INVALID_VALUE;
            }
            qtileDecodeRankPrepUs = nowUs() - prepStartUs;
        }
        qtileDecodeRank.fused = true;
        return NO_ERROR;
    };
    auto finishDecodeAttentionRankAfterQTile = [&]() -> ErrorCode {
        if (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex)) {
            return NO_ERROR;
        }
        if (qtileDecodeRank.fused) {
            return finishDecodeAttentionRankCaptureCUDA(
                mMeta, layerIndex, qtileDecodeRank.picTokenCount, qtileDecodeRank.topM,
                qtileDecodeRank.qLogical, qtileDecodeRank.validHeadCount,
                mCacheBlendScores, mCacheBlendIndices, mCacheBlendUsed, true, qtileDecodeRankPrepUs);
        }
        const int* decodeAttentionHeadIdsDevice = nullptr;
        int decodeAttentionHeadIdCount = 0;
        auto rankErr = syncDecodeAttentionHeadIds(&decodeAttentionHeadIdsDevice, &decodeAttentionHeadIdCount);
        if (rankErr != NO_ERROR) {
            return rankErr;
        }
        return runDecodeAttentionRankCUDA(
            mMeta, layerIndex, mBatch, mQuerySeqLen, mNumHead, mKvNumHead, mHeadDim, mPrecision,
            mCache->maxSlots, kvLen, pagedDevPtr<void>(query), pagedDevPtr<void>(mCache->key.get()),
            decodeAttentionHeadIdsDevice, decodeAttentionHeadIdCount, mScale, &mCacheBlendScores, &mCacheBlendScoreCount,
            &mCacheBlendIndices, &mCacheBlendIndexCount, &mCacheBlendUsed, &mCacheBlendUsedCount);
    };
    // Row-compressed attention is kept for decode and decode-recompute paths.
    // Prefill keeps the fast q-split path below as the default accelerator.
    if ((picDecodeRecompute && !useDecodeRepairQTile) || ordinaryDecodeCausal) {
        const int v2BlockSize = 128;
        const int v2SharedBytes = (mHeadDim * 2 + v2BlockSize * 2) * static_cast<int>(sizeof(float));
        const bool skipCausalMask = fullCausalMask;
        const bool decodeCausalAttention = repairDecodeCausal || ordinaryDecodeCausal;
        const char* v2OpName = decodeCausalAttention ? "decode_causal_attention" : "v2_row_compressed_mask";
        const float* v2Mask = (!skipCausalMask && useMask) ? pagedDevPtr<float>(mask) : nullptr;
        const int v2MaskElements = v2Mask != nullptr ? maskElements : 0;
        DecodeAttentionRankCaptureCUDA decodeRank;
        uint64_t decodeRankPrepUs = 0;
        const bool wantsDecodeAttentionRank = picDecodeRecompute && mQuerySeqLen == attnLen &&
            mMeta != nullptr && mMeta->needsPicDecodeAttentionRankCapture(layerIndex);
        if (wantsDecodeAttentionRank) {
            const int qIndex = attnLen - 1;
            if (qIndex < 0 || static_cast<int>(mMeta->sparse_query_logical_indices.size()) <= qIndex) {
                return INVALID_VALUE;
            }
            const int qLogical = mMeta->sparse_query_logical_indices[static_cast<size_t>(qIndex)];
            const int qRow = queryRowsAreFull ? qLogical : qIndex;
            const int picStart = mMeta->pic_decode_attention_pic_start;
            const int picTokenCount = mMeta->pic_decode_attention_pic_token_count;
            const int topM = std::min(mMeta->pic_decode_attention_top_m, picTokenCount);
            decodeRank.requested = true;
            decodeRank.picStart = picStart;
            decodeRank.picTokenCount = picTokenCount;
            decodeRank.topM = topM;
            decodeRank.qRow = qRow;
            decodeRank.qLogical = qLogical;
            const auto& attentionHeadIds = mMeta->pic_decode_attention_head_ids;
            if (attentionHeadIds.empty()) {
                decodeRank.validHeadCount = mNumHead;
            } else {
                decodeRank.validHeadCount = 0;
                for (int head : attentionHeadIds) {
                    if (head >= 0 && head < mNumHead) {
                        ++decodeRank.validHeadCount;
                    }
                }
            }
            const bool canFuseRank = picStart >= 0 && picTokenCount > 0 && topM > 0 &&
                picStart + picTokenCount <= kvLen && picStart + picTokenCount <= qLogical + 1 &&
                qRow >= 0 && qRow < mQuerySeqLen && decodeRank.validHeadCount > 0 && mBatch > 0;
            if (canFuseRank) {
                uint64_t prepStartUs = profile ? nowUs() : 0;
                err = syncDecodeAttentionHeadIds(&decodeRank.headIdsDevice, &decodeRank.headIdCount);
                if (err != NO_ERROR) {
                    return err;
                }
                err = ensureDecodeAttentionRankWorkspacesCUDA(
                    picTokenCount, topM, &mCacheBlendScores, &mCacheBlendScoreCount,
                    &mCacheBlendIndices, &mCacheBlendIndexCount, &mCacheBlendUsed, &mCacheBlendUsedCount);
                if (err != NO_ERROR) {
                    return err;
                }
                if (cudaMemset(mCacheBlendScores, 0, static_cast<size_t>(picTokenCount) * sizeof(float)) !=
                    cudaSuccess) {
                    return INVALID_VALUE;
                }
                decodeRank.scoreScale =
                    1.0f / static_cast<float>(decodeRank.validHeadCount * mBatch);
                if (profile) {
                    if (cudaDeviceSynchronize() != cudaSuccess) {
                        return INVALID_VALUE;
                    }
                    decodeRankPrepUs = nowUs() - prepStartUs;
                }
                decodeRank.fused = true;
            }
        }
        ScopedNvtxRange v2Nvtx(nvtxLayerRangeName(v2OpName, layerIndex,
                                                  mQuerySeqLen, attnLen, kvLen), nvtx);
        dim3 v2Grid(attnLen, mNumHead, mBatch);
        const uint64_t decodeAttentionKernelStartUs = profile ? nowUs() : 0;
        if (mPrecision == 4) {
            pagedAttentionRowCompressedMaskKernel<float><<<v2Grid, v2BlockSize, v2SharedBytes, stream>>>(
                pagedDevPtr<float>(query), pagedDevPtr<float>(mCache->key.get()),
                pagedDevPtr<float>(mCache->value.get()), pagedDevPtr<float>(output), v2Mask,
                v2MaskElements, mBatch, mQuerySeqLen, attnLen, attnLen, mNumHead, mKvNumHead, mHeadDim,
                baseLogical, kvLen, mCache->maxSlots, mScale, sparseQueryDevice, queryRowsAreFull ? 1 : 0,
                decodeRank.fused ? mCacheBlendScores : nullptr, decodeRank.picStart,
                decodeRank.picTokenCount, decodeRank.fused ? (attnLen - 1) : -1,
                decodeRank.headIdsDevice, decodeRank.headIdCount, decodeRank.scoreScale);
        } else {
            pagedAttentionRowCompressedMaskKernel<half><<<v2Grid, v2BlockSize, v2SharedBytes, stream>>>(
                pagedDevPtr<half>(query), pagedDevPtr<half>(mCache->key.get()),
                pagedDevPtr<half>(mCache->value.get()), pagedDevPtr<half>(output), v2Mask,
                v2MaskElements, mBatch, mQuerySeqLen, attnLen, attnLen, mNumHead, mKvNumHead, mHeadDim,
                baseLogical, kvLen, mCache->maxSlots, mScale, sparseQueryDevice, queryRowsAreFull ? 1 : 0,
                decodeRank.fused ? mCacheBlendScores : nullptr, decodeRank.picStart,
                decodeRank.picTokenCount, decodeRank.fused ? (attnLen - 1) : -1,
                decodeRank.headIdsDevice, decodeRank.headIdCount, decodeRank.scoreScale);
        }
        checkKernelErrors;
        uint64_t decodeAttentionKernelUs = 0;
        if (profile) {
            cudaDeviceSynchronize();
            decodeAttentionKernelUs = nowUs() - decodeAttentionKernelStartUs;
        }
        if (decodeRank.fused) {
            err = finishDecodeAttentionRankCaptureCUDA(
                mMeta, layerIndex, decodeRank.picTokenCount, decodeRank.topM, decodeRank.qLogical,
                decodeRank.validHeadCount,
                mCacheBlendScores, mCacheBlendIndices, mCacheBlendUsed, true, decodeRankPrepUs);
            if (err != NO_ERROR) {
                return err;
            }
        } else if (decodeRank.requested) {
            const int* decodeAttentionHeadIdsDevice = nullptr;
            int decodeAttentionHeadIdCount = 0;
            err = syncDecodeAttentionHeadIds(&decodeAttentionHeadIdsDevice, &decodeAttentionHeadIdCount);
            if (err != NO_ERROR) {
                return err;
            }
            err = runDecodeAttentionRankCUDA(
                mMeta, layerIndex, mBatch, mQuerySeqLen, mNumHead, mKvNumHead, mHeadDim, mPrecision,
                mCache->maxSlots, kvLen, pagedDevPtr<void>(query), pagedDevPtr<void>(mCache->key.get()),
                decodeAttentionHeadIdsDevice, decodeAttentionHeadIdCount, mScale, &mCacheBlendScores, &mCacheBlendScoreCount,
                &mCacheBlendIndices, &mCacheBlendIndexCount, &mCacheBlendUsed, &mCacheBlendUsedCount);
            if (err != NO_ERROR) {
                return err;
            }
        }
        if (profile) {
            const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
            uint64_t causalWork = denseWork;
            if (repairDecodeCausal) {
                causalWork = sparseLogicalWorkCUDA(mMeta, attnLen, kvLen);
            } else if (ordinaryDecodeCausal) {
                causalWork = static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + 1)));
            }
            MNN_PRINT("CUDAPagedAttention profile op=%s layer=%d query=%d attn=%d "
                      "kv_write=%d kv_len=%d sparse=%d full_q=%d mask_elements=%d input_mask_elements=%d "
                      "causal_mask_skipped=%d dense_kv_work=%llu causal_kv_work=%llu us=%llu\n",
                      v2OpName, layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                      queryRowsAreFull ? 1 : 0, v2MaskElements, maskElements, skipCausalMask ? 1 : 0,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      static_cast<unsigned long long>(decodeAttentionKernelUs));
        }
        return NO_ERROR;
    }
    const bool cacheBlendLargeSparseTile = sparseQuery && mMeta != nullptr &&
        mMeta->cacheblend_score_active && attnLen > 256;
    const bool fixedPlanSparseTile = sparseQuery && mMeta != nullptr && mMeta->pic_graph_active_plan_ready;
    const bool sparseTileSkipCausalMask = mMeta != nullptr && mMeta->full_causal_attention_mask;
    const float* qtileSparseMask = (!sparseTileSkipCausalMask && useMask) ? pagedDevPtr<float>(mask) : nullptr;
    const int qtileSparseMaskElements = qtileSparseMask != nullptr ? maskElements : 0;
    const bool qtileAllowedSparse = (!picDecodeRecompute || useDecodeRepairQTile) && sparseQuery && attnLen >= 1;
    CudaSparseQTileVariant sparseQTileVariant = qtileAllowedSparse
        ? (useDecodeRepairQTile ? decodeRepairQTileVariant :
           selectCudaSparseQTileVariant(mHeadDim, attnLen, cacheBlendLargeSparseTile, fixedPlanSparseTile))
        : kCudaSparseQTileNone;
    if (sparseQTileVariant != kCudaSparseQTileNone) {
        err = prepareDecodeAttentionRankForQTile();
        if (err != NO_ERROR) {
            return err;
        }
    }
    if (sparseQTileVariant != kCudaSparseQTileNone) {
        const char* variantName = cudaSparseQTileVariantName(sparseQTileVariant);
        ScopedNvtxRange flashNvtx(nvtxLayerRangeName("sparse_flash_qtile_attention", layerIndex,
                                                     mQuerySeqLen, attnLen, kvLen), nvtx);
        if (!launchCudaSparseQTileVariant(sparseQTileVariant, mPrecision, pagedDevPtr<void>(query),
                                          pagedDevPtr<void>(mCache->key.get()),
                                          pagedDevPtr<void>(mCache->value.get()), pagedDevPtr<void>(output),
                                          qtileSparseMask, qtileSparseMaskElements, mBatch, mQuerySeqLen, attnLen, attnLen,
                                          mNumHead, mKvNumHead, baseLogical, kvLen, mCache->maxSlots, mScale,
                                          sparseQueryDevice, queryRowsAreFull, stream,
                                          qtileDecodeRank.fused ? mCacheBlendScores : nullptr,
                                          qtileDecodeRank.picStart, qtileDecodeRank.picTokenCount,
                                          qtileDecodeRank.fused ? (attnLen - 1) : -1,
                                          qtileDecodeRank.headIdsDevice, qtileDecodeRank.headIdCount,
                                          qtileDecodeRank.scoreScale)) {
            return INVALID_VALUE;
        }
        checkKernelErrors;
        if (profile) {
            cudaDeviceSynchronize();
            MNN_PRINT("CUDAPagedAttention profile op=sparse_flash_qtile_attention layer=%d query=%d attn=%d "
                      "kv_write=%d kv_len=%d sparse=%d full_q=%d mask_elements=%d input_mask_elements=%d "
                      "causal_mask_skipped=%d variant=%s us=%llu\n",
                      layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                      queryRowsAreFull ? 1 : 0, qtileSparseMaskElements, maskElements,
                      sparseTileSkipCausalMask ? 1 : 0,
                      variantName, static_cast<unsigned long long>(nowUs() - attentionStartUs));
        }
        err = finishDecodeAttentionRankAfterQTile();
        if (err != NO_ERROR) {
            return err;
        }
        return NO_ERROR;
    }
    if ((fixedPlanSparseTile || cacheBlendLargeSparseTile) && mHeadDim == 64) {
        constexpr int qTile = 8;
        constexpr int qTileWide = 32;
        constexpr int kTile = 32;
        const bool useWideQTile = attnLen >= 384;
        dim3 flashBlock(kTile, useWideQTile ? qTileWide : qTile, 1);
        dim3 flashGrid(UP_DIV(attnLen, useWideQTile ? qTileWide : qTile), mNumHead, mBatch);
        const bool skipCausalMask = mMeta != nullptr && mMeta->full_causal_attention_mask;
        const float* sparseTileMask = (!skipCausalMask && useMask) ? pagedDevPtr<float>(mask) : nullptr;
        const int sparseTileMaskElements = sparseTileMask != nullptr ? maskElements : 0;
        ScopedNvtxRange flashNvtx(nvtxLayerRangeName("sparse_flash_tile_attention", layerIndex,
                                                     mQuerySeqLen, attnLen, kvLen), nvtx);
        if (useWideQTile && mPrecision == 4) {
            pagedSparseFlashTileKernel<float, qTileWide, kTile><<<flashGrid, flashBlock, 0, stream>>>(
                pagedDevPtr<float>(query), pagedDevPtr<float>(mCache->key.get()),
                pagedDevPtr<float>(mCache->value.get()), pagedDevPtr<float>(output),
                sparseTileMask, sparseTileMaskElements, mBatch, mQuerySeqLen, attnLen, attnLen, mNumHead, mKvNumHead, baseLogical,
                kvLen, mCache->maxSlots, mScale, sparseQueryDevice, queryRowsAreFull ? 1 : 0);
        } else if (useWideQTile) {
            pagedSparseFlashTileKernel<half, qTileWide, kTile><<<flashGrid, flashBlock, 0, stream>>>(
                pagedDevPtr<half>(query), pagedDevPtr<half>(mCache->key.get()),
                pagedDevPtr<half>(mCache->value.get()), pagedDevPtr<half>(output),
                sparseTileMask, sparseTileMaskElements, mBatch, mQuerySeqLen, attnLen, attnLen, mNumHead, mKvNumHead, baseLogical,
                kvLen, mCache->maxSlots, mScale, sparseQueryDevice, queryRowsAreFull ? 1 : 0);
        } else if (mPrecision == 4) {
            pagedSparseFlashTileKernel<float, qTile, kTile><<<flashGrid, flashBlock, 0, stream>>>(
                pagedDevPtr<float>(query), pagedDevPtr<float>(mCache->key.get()),
                pagedDevPtr<float>(mCache->value.get()), pagedDevPtr<float>(output),
                sparseTileMask, sparseTileMaskElements, mBatch, mQuerySeqLen, attnLen, attnLen, mNumHead, mKvNumHead, baseLogical,
                kvLen, mCache->maxSlots, mScale, sparseQueryDevice, queryRowsAreFull ? 1 : 0);
        } else {
            pagedSparseFlashTileKernel<half, qTile, kTile><<<flashGrid, flashBlock, 0, stream>>>(
                pagedDevPtr<half>(query), pagedDevPtr<half>(mCache->key.get()),
                pagedDevPtr<half>(mCache->value.get()), pagedDevPtr<half>(output),
                sparseTileMask, sparseTileMaskElements, mBatch, mQuerySeqLen, attnLen, attnLen, mNumHead, mKvNumHead, baseLogical,
                kvLen, mCache->maxSlots, mScale, sparseQueryDevice, queryRowsAreFull ? 1 : 0);
        }
        checkKernelErrors;
        if (profile) {
            cudaDeviceSynchronize();
            MNN_PRINT("CUDAPagedAttention profile op=sparse_flash_tile_attention layer=%d query=%d attn=%d "
                      "kv_write=%d kv_len=%d sparse=%d full_q=%d mask_elements=%d input_mask_elements=%d "
                      "causal_mask_skipped=%d q_tile=%d k_tile=%d us=%llu\n",
                      layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                      queryRowsAreFull ? 1 : 0, sparseTileMaskElements, maskElements, skipCausalMask ? 1 : 0,
                      useWideQTile ? qTileWide : qTile, kTile,
                      static_cast<unsigned long long>(nowUs() - attentionStartUs));
        }
        return NO_ERROR;
    }
    if (attnLen > 1 && !picDecodeRecompute) {
        int qSplitNum = 1;
        if (attnLen > 1024) {
            qSplitNum = UP_DIV(attnLen, 1024);
        } else if (attnLen > 256) {
            qSplitNum = UP_DIV(attnLen, 256);
        }
        const int maxPieceLen = UP_DIV(attnLen, qSplitNum);
        size_t prefillElements = static_cast<size_t>(mBatch) * mNumHead * maxPieceLen * kvLen;
        // Full-causal prefill is enforced in-kernel by logical position, so the additive causal mask
        // is redundant here and can be skipped safely.
        const float* prefillMask = (!fullCausalMask && useMask) ? pagedDevPtr<float>(mask) : nullptr;
        const int prefillMaskElements = prefillMask != nullptr ? maskElements : 0;
        if (ensurePrefillTemp(prefillElements)) {
            ScopedNvtxRange prefillNvtx(nvtxLayerRangeName("prefill_attention_fast_qk_softmax_qkv", layerIndex,
                                                           mQuerySeqLen, attnLen, kvLen), nvtx);
            for (int piece = 0; piece < qSplitNum; ++piece) {
                const int qStart = piece * maxPieceLen;
                const int qPieceLen = std::min(maxPieceLen, attnLen - qStart);
                if (qPieceLen <= 0) {
                    continue;
                }
                dim3 qkBlock(16, 16, 1);
                dim3 qkGrid(UP_DIV(kvLen, qkBlock.x), UP_DIV(qPieceLen, qkBlock.y), mBatch * mNumHead);
                if (mPrecision == 4) {
                    pagedPrefillQKKernel<float><<<qkGrid, qkBlock, 0, stream>>>(
                        pagedDevPtr<float>(query), pagedDevPtr<float>(mCache->key.get()), mPrefillQK,
                        prefillMask, prefillMaskElements, mBatch, mQuerySeqLen, attnLen, mNumHead, mKvNumHead, mHeadDim,
                        baseLogical, qStart, qPieceLen, kvLen, mCache->maxSlots, mScale, sparseQueryDevice,
                        queryRowsAreFull ? 1 : 0);
                } else {
                    pagedPrefillQKKernel<half><<<qkGrid, qkBlock, 0, stream>>>(
                        pagedDevPtr<half>(query), pagedDevPtr<half>(mCache->key.get()), mPrefillQK,
                        prefillMask, prefillMaskElements, mBatch, mQuerySeqLen, attnLen, mNumHead, mKvNumHead, mHeadDim,
                        baseLogical, qStart, qPieceLen, kvLen, mCache->maxSlots, mScale, sparseQueryDevice,
                        queryRowsAreFull ? 1 : 0);
                }
                if (debugSync) {
                    err = debugCheckPagedAttentionKernel(true, "prefill_qk", layerIndex, piece, qStart, qPieceLen,
                                                         kvLen, mNumHead, mKvNumHead, mHeadDim);
                    if (err != NO_ERROR) {
                        return err;
                    }
                } else {
                    checkKernelErrors;
                }

                const int axis = kvLen;
                const int outside = mBatch * mNumHead * qPieceLen;
                const int count = outside;
                if (axis <= 32) {
                    SOFTMAX_WARP_32<float><<<count, 32, 0, stream>>>(mPrefillQK, mPrefillSoftmax, 1, axis, outside, count);
                } else {
                    constexpr int threads = 256;
                    int calcMultiNum = UP_DIV(axis, threads);
                    SOFTMAX_AXIS_REDUCE<float><<<count, threads, 0, stream>>>(
                        mPrefillQK, mPrefillSoftmax, 1, axis, threads, calcMultiNum, outside, count);
                }
                if (debugSync) {
                    err = debugCheckPagedAttentionKernel(true, "prefill_softmax", layerIndex, piece, qStart,
                                                         qPieceLen, kvLen, mNumHead, mKvNumHead, mHeadDim);
                    if (err != NO_ERROR) {
                        return err;
                    }
                } else {
                    checkKernelErrors;
                }

                dim3 qkvBlock(32, 8, 1);
                dim3 qkvGrid(UP_DIV(mHeadDim, qkvBlock.x), UP_DIV(qPieceLen, qkvBlock.y), mBatch * mNumHead);
                if (mPrecision == 4) {
                    pagedPrefillQKVKernel<float><<<qkvGrid, qkvBlock, 0, stream>>>(
                        mPrefillSoftmax, pagedDevPtr<float>(mCache->value.get()), pagedDevPtr<float>(output),
                        mBatch, attnLen, qStart, qPieceLen, mNumHead, mKvNumHead, mHeadDim, kvLen, mCache->maxSlots);
                } else {
                    pagedPrefillQKVKernel<half><<<qkvGrid, qkvBlock, 0, stream>>>(
                        mPrefillSoftmax, pagedDevPtr<half>(mCache->value.get()), pagedDevPtr<half>(output),
                        mBatch, attnLen, qStart, qPieceLen, mNumHead, mKvNumHead, mHeadDim, kvLen, mCache->maxSlots);
                }
                if (debugSync) {
                    err = debugCheckPagedAttentionKernel(true, "prefill_qkv", layerIndex, piece, qStart, qPieceLen,
                                                         kvLen, mNumHead, mKvNumHead, mHeadDim);
                    if (err != NO_ERROR) {
                        return err;
                    }
                } else {
                    checkKernelErrors;
                }
            }
            if (profile) {
                cudaDeviceSynchronize();
                MNN_PRINT("CUDAPagedAttention profile op=prefill_attention_fast_qk_softmax_qkv layer=%d query=%d "
                          "attn=%d kv_write=%d kv_len=%d sparse=%d full_q=%d mask_elements=%d "
                          "input_mask_elements=%d causal_mask_skipped=%d q_split=%d us=%llu\n",
                          layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                          queryRowsAreFull ? 1 : 0, prefillMaskElements, maskElements,
                          fullCausalMask ? 1 : 0, qSplitNum,
                          static_cast<unsigned long long>(nowUs() - attentionStartUs));
            }
            return NO_ERROR;
        }
    }

    dim3 grid(attnLen, mNumHead, mBatch);
    int blockSize = 128;
    int sharedBytes = (kvLen + blockSize) * sizeof(float);
    const bool genericSkipCausalMask = fullCausalMask;
    // Generic full-causal attention also applies its own positional bound, so skip the additive mask.
    const float* genericMask = (!genericSkipCausalMask && useMask) ? pagedDevPtr<float>(mask) : nullptr;
    const int genericMaskElements = genericMask != nullptr ? maskElements : 0;
    ScopedNvtxRange genericNvtx(nvtxLayerRangeName("prefill_attention_generic_kernel", layerIndex, mQuerySeqLen,
                                                   attnLen, kvLen), nvtx);
    if (mPrecision == 4) {
        pagedAttentionKernel<float><<<grid, blockSize, sharedBytes, stream>>>(
            pagedDevPtr<float>(query), pagedDevPtr<float>(mCache->key.get()), pagedDevPtr<float>(mCache->value.get()),
            pagedDevPtr<float>(output), genericMask, genericMaskElements, mBatch, mQuerySeqLen, attnLen, attnLen,
            mNumHead, mKvNumHead, mHeadDim, baseLogical, kvLen, mCache->maxSlots, mScale, sparseQueryDevice,
            queryRowsAreFull ? 1 : 0);
    } else {
        pagedAttentionKernel<half><<<grid, blockSize, sharedBytes, stream>>>(
            pagedDevPtr<half>(query), pagedDevPtr<half>(mCache->key.get()), pagedDevPtr<half>(mCache->value.get()),
            pagedDevPtr<half>(output), genericMask, genericMaskElements, mBatch, mQuerySeqLen, attnLen, attnLen,
            mNumHead, mKvNumHead, mHeadDim, baseLogical, kvLen, mCache->maxSlots, mScale, sparseQueryDevice,
            queryRowsAreFull ? 1 : 0);
    }
    checkKernelErrors;
    if (profile) {
        cudaDeviceSynchronize();
        MNN_PRINT("CUDAPagedAttention profile op=generic layer=%d query=%d attn=%d kv_write=%d kv_len=%d "
                  "sparse=%d full_q=%d mask_elements=%d input_mask_elements=%d causal_mask_skipped=%d us=%llu\n",
                  layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                  queryRowsAreFull ? 1 : 0, genericMaskElements, maskElements,
                  genericSkipCausalMask ? 1 : 0,
                  static_cast<unsigned long long>(nowUs() - attentionStartUs));
    }
    return NO_ERROR;
}

bool CUDAPagedAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    auto tmp = new CUDAPagedAttention(bn, op);
    tmp->mCache = mCache;
    tmp->mPicAttentionMode = mPicAttentionMode;
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
static CUDACreatorRegister<PagedAttentionCreator> __init_pic_score_attention(OpType_PicScoreAttention);
static CUDACreatorRegister<PagedAttentionCreator> __init_pic_sparse_attention(OpType_PicSparseAttention);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
