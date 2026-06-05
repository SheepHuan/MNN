#include "PagedAttentionExecution.hpp"
#include "SoftmaxExecution.hpp"
#include "core/Macro.h"
#include "core/MNNFileUtils.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
                                  int headDim, int baseLogical, int maxSlots, const int* queryLogicalIndices) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z * blockDim.z + threadIdx.z;
    if (d >= headDim || l >= insertLen || bh >= batch * kvHeads) {
        return;
    }
    int b = bh / kvHeads;
    int h = bh % kvHeads;
    int logical = queryLogicalIndices ? queryLogicalIndices[l] : (baseLogical + l);
    if (logical < 0 || logical >= maxSlots) {
        return;
    }
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
__global__ void exportCanonicalPagedKeyKernel(const T* keyCache, T* keyOut, const int* slotTable, int batch,
                                              int kvLen, int kvHeads, int headDim, int maxSlots, int ropeDim,
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
    int slot = slotTable ? slotTable[l] : l;
    if (slot < 0 || slot >= maxSlots) {
        return;
    }
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
__global__ void hydratePagedKeyKernel(const T* keyIn, T* keyCache, const int* slotTable, int batch, int tokenCount,
                                      int kvHeads, int headDim, int maxSlots, int logicalStart, int ropeDim,
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
    int slot = slotTable ? slotTable[logical] : logical;
    if (slot < 0 || slot >= maxSlots) {
        return;
    }
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
__global__ void hydratePagedValueKernel(const T* valueIn, T* valueCache, const int* slotTable, int batch,
                                        int tokenCount, int kvHeads, int headDim, int maxSlots, int logicalStart,
                                        int total) {
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
    int slot = slotTable ? slotTable[logical] : logical;
    if (slot < 0 || slot >= maxSlots) {
        return;
    }
    int src = ((b * kvHeads + h) * tokenCount + local) * headDim + d;
    int dst = ((b * kvHeads + h) * maxSlots + slot) * headDim + d;
    valueCache[dst] = valueIn[src];
}

template <typename T>
__global__ void cacheBlendValueScoreKernel(const T* referenceValueCache, const T* cachedValue, const int* slotTable,
                                           float* scores, int batch, int tokenCount, int kvHeads, int headDim,
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
    int slot = slotTable ? slotTable[logical] : logical;
    if (slot < 0 || slot >= maxSlots) {
        scores[scoreOffset + local] = -FLT_MAX;
        return;
    }
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

__global__ void cacheBlendTopKKernel(const float* scores, int* selected, int tokenCount, int topK) {
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
            bool used = false;
            for (int prev = 0; prev < k; ++prev) {
                if (selected[prev] == i) {
                    used = true;
                    break;
                }
            }
            if (used) {
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
            selected[k] = bestIndices[0];
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void pagedAttentionKernel(const T* query, const T* keyCache, const T* valueCache, T* output,
                                     const float* mask, const int* slotTable, int maskElements, int batch,
                                     int queryLen, int insertLen, int numHeads, int kvHeads, int headDim,
                                     int baseLogical, int kvLen, int maxSlots, float scale,
                                     const int* queryLogicalIndices) {
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
            int slot = slotTable ? slotTable[k] : k;
            int vOffset = ((b * kvHeads + kvHead) * maxSlots + slot) * headDim + d;
            acc += scores[k] * invSum * pagedToFloat<T>(valueCache[vOffset]);
        }
        int outOffset = (b * queryLen * numHeads * headDim) + q * numHeads * headDim + h * headDim + d;
        output[outOffset] = pagedFromFloat<T>(acc);
    }
}

template <typename T>
__global__ void pagedPrefillQKKernel(const T* query, const T* keyCache, float* scores, const int* slotTable,
                                     const float* mask, int maskElements,
                                     int batch, int queryLen, int insertLen, int numHeads, int kvHeads,
                                     int headDim, int baseLogical, int qStart, int qPieceLen, int kvLen, int maxSlots,
                                     float scale,
                                     const int* queryLogicalIndices) {
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
    int scoreOffset = ((b * numHeads + h) * qPieceLen + qLocal) * kvLen + k;

    int slot = k < kvLen ? (slotTable ? slotTable[k] : k) : -1;
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
                int qOffset = ((b * queryLen + q) * numHeads + h) * headDim + d;
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
            int maskIdx = q * maskCols + col;
            if (maskIdx >= 0 && maskIdx < maskElements) {
                finalScore += mask[maskIdx];
            }
        }
        scores[scoreOffset] = finalScore;
    }
}

template <typename T>
__global__ void pagedPrefillQKVKernel(const float* probs, const T* valueCache, T* output, const int* slotTable,
                                      int batch, int queryLen, int qStart, int qPieceLen, int numHeads, int kvHeads,
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
                int slot = slotTable ? slotTable[logical] : logical;
                if (slot >= 0 && slot < maxSlots) {
                    value = pagedToFloat<T>(valueBase[slot * headDim + valueD]);
                }
            }
            valueTile[kk][dd] = value;
        }
        __syncthreads();

        if (qLocal < qPieceLen && q < queryLen && d < headDim) {
#pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                if (kStart + kk < kvLen) {
                    acc += probTile[threadIdx.y][kk] * valueTile[kk][threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    if (qLocal < qPieceLen && q < queryLen && d < headDim) {
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

static bool writeBinaryFile(const std::string& path, const std::vector<int8_t>& data) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    if (!data.empty()) {
        os.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    return os.good();
}

static bool readBinaryFile(const std::string& path, std::vector<int8_t>& data) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto size = is.tellg();
    if (size < 0) {
        return false;
    }
    data.resize(static_cast<size_t>(size));
    is.seekg(0, std::ios::beg);
    if (!data.empty()) {
        is.read(reinterpret_cast<char*>(data.data()), size);
    }
    return is.good() || is.eof();
}

static bool readExternalValueSegment(const std::string& path, std::vector<int8_t>& data, int batch, int kvHeads,
                                     size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount,
                                     int headDim, int bytes) {
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
    data.resize(static_cast<size_t>(batch) * kvHeads * segmentBytes);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dst = (static_cast<size_t>(b) * kvHeads + h) * segmentBytes;
            is.seekg(static_cast<std::streamoff>(src), std::ios::beg);
            is.read(reinterpret_cast<char*>(data.data() + dst), static_cast<std::streamsize>(segmentBytes));
            if (is.gcount() != static_cast<std::streamsize>(segmentBytes)) {
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

static uint64_t nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static std::string externalLayerKey(const std::string& keyPath, const std::string& valuePath) {
    return keyPath + "\n" + valuePath;
}

static bool adviseFileWillNeed(const std::string& path) {
#if defined(__linux__) && defined(POSIX_FADV_WILLNEED)
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return false;
    }
    int ret = ::posix_fadvise(fd, 0, st.st_size, POSIX_FADV_WILLNEED);
    ::close(fd);
    return ret == 0;
#else
    (void)path;
    return false;
#endif
}

static std::mutex gExternalLayerPrefetchMutex;
static std::unordered_set<std::string> gExternalLayerPrefetched;
static std::vector<std::shared_future<void>> gExternalLayerPrefetchTasks;

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

static void prefetchExternalLayersFrom(const PagedKVMeta* meta, int startLayer) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return;
    }
    const int lastLayer = lastExternalLayerIndex(meta);
    if (startLayer < 0 || lastLayer < startLayer) {
        return;
    }
    std::vector<std::pair<std::string, std::string>> jobs;
    {
        std::lock_guard<std::mutex> lock(gExternalLayerPrefetchMutex);
        for (int layerIndex = startLayer; layerIndex <= lastLayer; ++layerIndex) {
            if (meta->externalLayerLoaded(layerIndex)) {
                continue;
            }
            for (const auto& segment : meta->external_segments) {
                auto layer = segment.layer(layerIndex);
                if (layer == nullptr) {
                    continue;
                }
                const auto key = externalLayerKey(layer->keyPath, layer->valuePath);
                if (gExternalLayerPrefetched.insert(key).second) {
                    jobs.emplace_back(layer->keyPath, layer->valuePath);
                }
            }
        }
    }
    if (jobs.empty()) {
        return;
    }
    auto future = std::async(std::launch::async, [jobs]() {
        for (const auto& job : jobs) {
            (void)adviseFileWillNeed(job.first);
            (void)adviseFileWillNeed(job.second);
        }
    }).share();
    std::lock_guard<std::mutex> lock(gExternalLayerPrefetchMutex);
    gExternalLayerPrefetchTasks.emplace_back(std::move(future));
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
                                             int bytes, int maxSlots, const std::vector<int>& physicalSlots,
                                             int kvLen, void* keyCacheDevice, void* valueCacheDevice,
                                             const int* slotTableDevice, void** keyWorkspace,
                                             size_t* keyWorkspaceBytes, void** valueWorkspace,
                                             size_t* valueWorkspaceBytes) {
    if (meta == nullptr || meta->external_segments.empty() || meta->externalLayerLoaded(layerIndex)) {
        return NO_ERROR;
    }
    const bool profile = profilePagedAttention();
    const uint64_t startUs = profile ? nowUs() : 0;
    size_t totalTokens = 0;
    prefetchExternalLayersFrom(meta, layerIndex + 1);
    for (const auto& segment : meta->external_segments) {
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
        std::vector<int8_t> keyData;
        std::vector<int8_t> valueData;
        if (!readBinaryFile(layer->keyPath, keyData) || !readBinaryFile(layer->valuePath, valueData)) {
            MNN_ERROR("CUDAPagedAttention failed to read external PIC KV files for layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
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
        const size_t expectedKey = sourceTokenCount * static_cast<size_t>(batch) * kvHeads * headDim * bytes;
        const size_t expectedValue = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * headDim * bytes;
        if (keyData.size() < expectedKey || valueData.size() < expectedValue) {
            MNN_ERROR("CUDAPagedAttention external PIC KV file is too small at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        for (size_t local = 0; local < segment.tokenCount; ++local) {
            int logical = static_cast<int>(segment.logicalStart + local);
            int slot = logical >= 0 && logical < static_cast<int>(physicalSlots.size()) ? physicalSlots[logical] : -1;
            if (slot < 0 || slot >= maxSlots) {
                return OUT_OF_MEMORY;
            }
        }
        const size_t keyTokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
        const size_t keySourceBase = sourceTokenOffset * keyTokenBytes;
        const size_t keySegmentBytes = segment.tokenCount * keyTokenBytes;
        if (keyWorkspace == nullptr || keyWorkspaceBytes == nullptr) {
            return INVALID_VALUE;
        }
        if (*keyWorkspace == nullptr || *keyWorkspaceBytes < keySegmentBytes) {
            if (*keyWorkspace != nullptr) {
                cudaFree(*keyWorkspace);
                *keyWorkspace = nullptr;
                *keyWorkspaceBytes = 0;
            }
            auto keyAlloc = cudaMalloc(keyWorkspace, keySegmentBytes);
            if (keyAlloc != cudaSuccess) {
                return OUT_OF_MEMORY;
            }
            *keyWorkspaceBytes = keySegmentBytes;
        }
        auto keyCopy = cudaMemcpy(*keyWorkspace, keyData.data() + keySourceBase, keySegmentBytes,
                                  cudaMemcpyHostToDevice);
        if (keyCopy != cudaSuccess) {
            return INVALID_VALUE;
        }
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
        if (bytes == 4) {
            hydratePagedKeyKernel<float><<<blocks, threads>>>(
                reinterpret_cast<const float*>(*keyWorkspace), reinterpret_cast<float*>(keyCacheDevice),
                slotTableDevice, batch, static_cast<int>(segment.tokenCount), kvHeads,
                headDim, maxSlots, static_cast<int>(segment.logicalStart), ropeDim,
                segment.ropeTheta > 0.0f ? segment.ropeTheta : 10000.0f, ropeTypeCode(segment),
                std::max(segment.ropeScalingFactor, 1.0f), std::max(segment.ropeScalingLowFreqFactor, 1.0e-6f),
                std::max(segment.ropeScalingHighFreqFactor, 1.0e-6f), oldContext,
                segment.ropeAttentionScaling > 0.0f ? segment.ropeAttentionScaling : 1.0f, totalKey);
        } else {
            hydratePagedKeyKernel<half><<<blocks, threads>>>(
                reinterpret_cast<const half*>(*keyWorkspace), reinterpret_cast<half*>(keyCacheDevice),
                slotTableDevice, batch, static_cast<int>(segment.tokenCount), kvHeads,
                headDim, maxSlots, static_cast<int>(segment.logicalStart), ropeDim,
                segment.ropeTheta > 0.0f ? segment.ropeTheta : 10000.0f, ropeTypeCode(segment),
                std::max(segment.ropeScalingFactor, 1.0f), std::max(segment.ropeScalingLowFreqFactor, 1.0e-6f),
                std::max(segment.ropeScalingHighFreqFactor, 1.0e-6f), oldContext,
                segment.ropeAttentionScaling > 0.0f ? segment.ropeAttentionScaling : 1.0f, totalKey);
        }
        auto keyKernel = cudaGetLastError();
        if (keyKernel != cudaSuccess) {
            MNN_ERROR("CUDAPagedAttention failed to hydrate external PIC key on GPU at layer %d: %s\n",
                      layerIndex, cudaGetErrorString(keyKernel));
            return INVALID_VALUE;
        }

        if (valueWorkspace == nullptr || valueWorkspaceBytes == nullptr) {
            return INVALID_VALUE;
        }
        const size_t valueTokenBytes = static_cast<size_t>(headDim) * bytes;
        const size_t valueHeadSegmentBytes = segment.tokenCount * valueTokenBytes;
        const size_t valueSegmentBytes = static_cast<size_t>(batch) * kvHeads * valueHeadSegmentBytes;
        const int8_t* valueUploadPtr = valueData.data();
        std::vector<int8_t> compactValueData;
        if (sourceTokenOffset != 0 || segment.tokenCount != sourceTokenCount) {
            compactValueData.resize(valueSegmentBytes);
            for (int b = 0; b < batch; ++b) {
                for (int h = 0; h < kvHeads; ++h) {
                    const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                        sourceTokenOffset) * valueTokenBytes;
                    const size_t dst = (static_cast<size_t>(b) * kvHeads + h) * valueHeadSegmentBytes;
                    ::memcpy(compactValueData.data() + dst, valueData.data() + src, valueHeadSegmentBytes);
                }
            }
            valueUploadPtr = compactValueData.data();
        }
        if (*valueWorkspace == nullptr || *valueWorkspaceBytes < valueSegmentBytes) {
            if (*valueWorkspace != nullptr) {
                cudaFree(*valueWorkspace);
                *valueWorkspace = nullptr;
                *valueWorkspaceBytes = 0;
            }
            auto valueAlloc = cudaMalloc(valueWorkspace, valueSegmentBytes);
            if (valueAlloc != cudaSuccess) {
                return OUT_OF_MEMORY;
            }
            *valueWorkspaceBytes = valueSegmentBytes;
        }
        auto valueCopy = cudaMemcpy(*valueWorkspace, valueUploadPtr, valueSegmentBytes, cudaMemcpyHostToDevice);
        if (valueCopy != cudaSuccess) {
            return INVALID_VALUE;
        }
        const int totalValue = static_cast<int>(totalElements);
        if (bytes == 4) {
            hydratePagedValueKernel<float><<<blocks, threads>>>(
                reinterpret_cast<const float*>(*valueWorkspace), reinterpret_cast<float*>(valueCacheDevice),
                slotTableDevice, batch, static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots,
                static_cast<int>(segment.logicalStart), totalValue);
        } else {
            hydratePagedValueKernel<half><<<blocks, threads>>>(
                reinterpret_cast<const half*>(*valueWorkspace), reinterpret_cast<half*>(valueCacheDevice),
                slotTableDevice, batch, static_cast<int>(segment.tokenCount), kvHeads, headDim, maxSlots,
                static_cast<int>(segment.logicalStart), totalValue);
        }
        auto valueKernel = cudaGetLastError();
        if (valueKernel != cudaSuccess) {
            MNN_ERROR("CUDAPagedAttention failed to hydrate external PIC value on GPU at layer %d: %s\n",
                      layerIndex, cudaGetErrorString(valueKernel));
            return INVALID_VALUE;
        }
    }
    meta->markExternalLayerLoaded(layerIndex);
    if (profile) {
        cudaDeviceSynchronize();
        MNN_PRINT("CUDAPagedAttention profile op=hydrate layer=%d tokens=%d kv_len=%d us=%llu\n",
                  layerIndex, static_cast<int>(totalTokens), kvLen,
                  static_cast<unsigned long long>(nowUs() - startUs));
    }
    return NO_ERROR;
}

static ErrorCode runCacheBlendScoringCUDA(PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim,
                                          int bytes, int maxSlots, int kvLen, const void* valueCacheDevice,
                                          const int* slotTableDevice, void** valueWorkspace,
                                          size_t* valueWorkspaceBytes, float** scoreWorkspace,
                                          size_t* scoreWorkspaceCount, int** indexWorkspace,
                                          size_t* indexWorkspaceCount) {
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
    if (valueWorkspace == nullptr || valueWorkspaceBytes == nullptr || scoreWorkspace == nullptr ||
        scoreWorkspaceCount == nullptr || indexWorkspace == nullptr || indexWorkspaceCount == nullptr) {
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
        std::vector<int8_t> cachedValue;
        if (!readExternalValueSegment(layer->valuePath, cachedValue, batch, kvHeads, sourceTokenCount,
                                      sourceTokenOffset, segment.tokenCount, headDim, bytes)) {
            return INVALID_VALUE;
        }
        const size_t valueBytes = cachedValue.size();
        if (*valueWorkspace == nullptr || *valueWorkspaceBytes < valueBytes) {
            if (*valueWorkspace != nullptr) {
                cudaFree(*valueWorkspace);
                *valueWorkspace = nullptr;
                *valueWorkspaceBytes = 0;
            }
            if (cudaMalloc(valueWorkspace, valueBytes) != cudaSuccess) {
                return OUT_OF_MEMORY;
            }
            *valueWorkspaceBytes = valueBytes;
        }
        if (cudaMemcpy(*valueWorkspace, cachedValue.data(), valueBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            return INVALID_VALUE;
        }
        const int threads = 128;
        const int blocks = UP_DIV(static_cast<int>(segment.tokenCount), threads);
        if (bytes == 4) {
            cacheBlendValueScoreKernel<float><<<blocks, threads>>>(
                reinterpret_cast<const float*>(valueCacheDevice), reinterpret_cast<const float*>(*valueWorkspace),
                slotTableDevice, *scoreWorkspace, batch, static_cast<int>(segment.tokenCount), kvHeads, headDim,
                maxSlots, static_cast<int>(segment.logicalStart), static_cast<int>(scoreOffset));
        } else {
            cacheBlendValueScoreKernel<half><<<blocks, threads>>>(
                reinterpret_cast<const half*>(valueCacheDevice), reinterpret_cast<const half*>(*valueWorkspace),
                slotTableDevice, *scoreWorkspace, batch, static_cast<int>(segment.tokenCount), kvHeads, headDim,
                maxSlots, static_cast<int>(segment.logicalStart), static_cast<int>(scoreOffset));
        }
        if (cudaGetLastError() != cudaSuccess) {
            return INVALID_VALUE;
        }
        scoreOffset += segment.tokenCount;
    }
    if (scoreOffset != static_cast<size_t>(picTokenCount)) {
        return INVALID_VALUE;
    }
    cacheBlendTopKKernel<<<1, 256>>>(*scoreWorkspace, *indexWorkspace, picTokenCount, topK);
    if (cudaGetLastError() != cudaSuccess) {
        return INVALID_VALUE;
    }
    std::vector<int> selected(topK);
    if (cudaMemcpy(selected.data(), *indexWorkspace, static_cast<size_t>(topK) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return INVALID_VALUE;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(picTokenCount), 0);
    for (int index : selected) {
        if (index < 0 || index >= picTokenCount || seen[static_cast<size_t>(index)] != 0) {
            return INVALID_VALUE;
        }
        seen[static_cast<size_t>(index)] = 1;
    }
    meta->setCacheBlendScoringResult(selected);
    if (profilePagedAttention()) {
        cudaDeviceSynchronize();
        MNN_PRINT("CUDAPagedAttention profile op=cacheblend_score layer=%d pic_tokens=%d top_k=%d\n",
                  layerIndex, picTokenCount, topK);
    }
    return NO_ERROR;
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
       << "  \"max_position_embeddings\": " << (meta != nullptr ? meta->max_position_embeddings : 0) << "\n"
       << "}\n";
    return os.good();
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
    mCache->sparseQuery.reset(Tensor::createDevice<int>({maxSlots}));
    if (!mCache->key || !mCache->value || !mCache->slotTable || !mCache->sparseQuery) {
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
    if (!mCudaBackend->onAcquireBuffer(mCache->sparseQuery.get(), Backend::STATIC)) {
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
    bool sparseQuery = mMeta != nullptr && mMeta->sparse_query_active;
    if (mMeta != nullptr) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        baseLogical = static_cast<int>(kept) + reverse;
        insertLen = mMeta->add > 0 ? static_cast<int>(std::min<size_t>(mMeta->add, mNewKvSeqLen)) : mNewKvSeqLen;
    }
    insertLen = std::min(insertLen, mQuerySeqLen);
    if (sparseQuery) {
        if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < insertLen) {
            return INVALID_VALUE;
        }
        baseLogical = 0;
    }
    int kvLen = sparseQuery ? std::max(0, mMeta->logical_length) : (baseLogical + insertLen);
    if (kvLen > mCache->maxSlots) {
        MNN_ERROR("CUDAPagedAttention layer %d needs %d slots, cache capacity is %d\n", mLayerIndex, kvLen,
                  mCache->maxSlots);
        return OUT_OF_MEMORY;
    }
    auto err = syncSlotTable(kvLen);
    if (err != NO_ERROR) {
        return err;
    }
    std::vector<int> physicalSlots(kvLen);
    for (int l = 0; l < kvLen; ++l) {
        physicalSlots[l] = mMeta ? mMeta->physicalSlot(l) : l;
    }
    const int* sparseQueryDevice = nullptr;
    if (sparseQuery) {
        cudaMemcpy(pagedDevPtr<int>(mCache->sparseQuery.get()), mMeta->sparse_query_logical_indices.data(),
                   insertLen * sizeof(int), cudaMemcpyHostToDevice);
        sparseQueryDevice = pagedDevPtr<int>(mCache->sparseQuery.get());
    }

    cudaStream_t stream = 0;
    int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : 0);
    auto restore = restoreExternalSegmentsCUDA(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mPrecision,
                                               mCache->maxSlots, physicalSlots, kvLen,
                                               pagedDevPtr<void>(mCache->key.get()),
                                               pagedDevPtr<void>(mCache->value.get()),
                                               pagedDevPtr<int>(mCache->slotTable.get()), &mExternalKey,
                                               &mExternalKeyBytes, &mExternalValue, &mExternalValueBytes);
    if (restore != NO_ERROR) {
        return restore;
    }

    if (!mIsKVShared && insertLen > 0) {
        dim3 block(32, 8, 1);
        dim3 grid(UP_DIV(mHeadDim, block.x), UP_DIV(insertLen, block.y), UP_DIV(mBatch * mKvNumHead, block.z));
        if (mPrecision == 4) {
            copyPagedKVKernel<float><<<grid, block, 0, stream>>>(
                pagedDevPtr<float>(key), pagedDevPtr<float>(value), pagedDevPtr<float>(mCache->key.get()),
                pagedDevPtr<float>(mCache->value.get()), pagedDevPtr<int>(mCache->slotTable.get()), mBatch,
                mNewKvSeqLen, insertLen, mKvNumHead, mHeadDim, baseLogical, mCache->maxSlots, sparseQueryDevice);
        } else {
            copyPagedKVKernel<half><<<grid, block, 0, stream>>>(
                pagedDevPtr<half>(key), pagedDevPtr<half>(value), pagedDevPtr<half>(mCache->key.get()),
                pagedDevPtr<half>(mCache->value.get()), pagedDevPtr<int>(mCache->slotTable.get()), mBatch,
                mNewKvSeqLen, insertLen, mKvNumHead, mHeadDim, baseLogical, mCache->maxSlots, sparseQueryDevice);
        }
        checkKernelErrors;
    }

    auto cacheBlendScore = runCacheBlendScoringCUDA(
        mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mPrecision, mCache->maxSlots, kvLen,
        pagedDevPtr<void>(mCache->value.get()), pagedDevPtr<int>(mCache->slotTable.get()), &mExternalValue,
        &mExternalValueBytes, &mCacheBlendScores, &mCacheBlendScoreCount, &mCacheBlendIndices,
        &mCacheBlendIndexCount);
    if (cacheBlendScore != NO_ERROR) {
        return cacheBlendScore;
    }

    if (mMeta != nullptr && !mMeta->file_name.empty() && mMeta->file_flag == KVMeta::PendingWrite && kvLen > 0) {
        auto prefixDir = mCudaBackend->getRuntime()->hint().prefixcacheDirPath;
        MNNCreateDir(prefixDir.c_str());
        int layerIndex = mLayerIndex >= 0 ? mLayerIndex : mMeta->layer_index;
        std::string basePath = MNNFilePathConcat(prefixDir, mMeta->file_name) + "_" + std::to_string(layerIndex);

        size_t valueStorageBytes = static_cast<size_t>(mBatch) * mKvNumHead * mCache->maxSlots * mHeadDim * mPrecision;
        std::vector<int8_t> valueStorage(valueStorageBytes);
        std::vector<int8_t> keyData(static_cast<size_t>(kvLen) * mBatch * mKvNumHead * mHeadDim * mPrecision);
        void* keyExportDevice = nullptr;
        cudaError_t keyCopy = cudaMalloc(&keyExportDevice, keyData.size());
        if (keyCopy == cudaSuccess) {
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
                    pagedDevPtr<float>(mCache->key.get()), reinterpret_cast<float*>(keyExportDevice),
                    pagedDevPtr<int>(mCache->slotTable.get()), mBatch, kvLen, mKvNumHead, mHeadDim,
                    mCache->maxSlots, ropeDim, mMeta && mMeta->rope_theta > 0.0f ? mMeta->rope_theta : 10000.0f,
                    ropeTypeCode(mMeta), mMeta ? std::max(mMeta->rope_scaling_factor, 1.0f) : 1.0f,
                    mMeta ? std::max(mMeta->rope_scaling_low_freq_factor, 1.0e-6f) : 1.0f,
                    mMeta ? std::max(mMeta->rope_scaling_high_freq_factor, 1.0e-6f) : 4.0f,
                    oldContext, mMeta ? mMeta->rope_attention_scaling : 1.0f);
            } else {
                exportCanonicalPagedKeyKernel<half><<<keyGrid, keyBlock, 0, stream>>>(
                    pagedDevPtr<half>(mCache->key.get()), reinterpret_cast<half*>(keyExportDevice),
                    pagedDevPtr<int>(mCache->slotTable.get()), mBatch, kvLen, mKvNumHead, mHeadDim,
                    mCache->maxSlots, ropeDim, mMeta && mMeta->rope_theta > 0.0f ? mMeta->rope_theta : 10000.0f,
                    ropeTypeCode(mMeta), mMeta ? std::max(mMeta->rope_scaling_factor, 1.0f) : 1.0f,
                    mMeta ? std::max(mMeta->rope_scaling_low_freq_factor, 1.0e-6f) : 1.0f,
                    mMeta ? std::max(mMeta->rope_scaling_high_freq_factor, 1.0e-6f) : 4.0f,
                    oldContext, mMeta ? mMeta->rope_attention_scaling : 1.0f);
            }
            keyCopy = cudaGetLastError();
            if (keyCopy == cudaSuccess) {
                keyCopy = cudaMemcpy(keyData.data(), keyExportDevice, keyData.size(), cudaMemcpyDeviceToHost);
            }
            cudaFree(keyExportDevice);
        }
        cudaError_t valueCopy = cudaMemcpy(valueStorage.data(), pagedDevPtr<void>(mCache->value.get()), valueStorageBytes,
                                           cudaMemcpyDeviceToHost);
        if (keyCopy == cudaSuccess && valueCopy == cudaSuccess) {
            std::vector<int> physicalSlots(kvLen);
            for (int l = 0; l < kvLen; ++l) {
                physicalSlots[l] = mMeta ? mMeta->physicalSlot(l) : l;
            }
            std::vector<int8_t> valueData(static_cast<size_t>(mBatch) * mKvNumHead * kvLen * mHeadDim * mPrecision);
            for (int l = 0; l < kvLen; ++l) {
                int slot = physicalSlots[l];
                if (slot < 0 || slot >= mCache->maxSlots) {
                    continue;
                }
                for (int b = 0; b < mBatch; ++b) {
                    for (int h = 0; h < mKvNumHead; ++h) {
                        const int8_t* srcV = valueStorage.data() +
                            ((b * mKvNumHead + h) * mCache->maxSlots + slot) * mHeadDim * mPrecision;
                        int8_t* dstV = valueData.data() +
                            ((b * mKvNumHead + h) * kvLen + l) * mHeadDim * mPrecision;
                        ::memcpy(dstV, srcV, mHeadDim * mPrecision);
                    }
                }
            }
            if (!writeBinaryFile(basePath + ".k", keyData)) {
                MNN_PRINT("CUDAPagedAttention: failed to export key cache: %s\n", (basePath + ".k").c_str());
            }
            if (!writeBinaryFile(basePath + ".v", valueData)) {
                MNN_PRINT("CUDAPagedAttention: failed to export value cache: %s\n", (basePath + ".v").c_str());
            }
            if (!writeShapeFile(basePath + ".json", mBatch, mKvNumHead, mHeadDim, kvLen, mPrecision, mMeta)) {
                MNN_PRINT("CUDAPagedAttention: failed to export shape metadata: %s\n", (basePath + ".json").c_str());
            }
        } else {
            MNN_PRINT("CUDAPagedAttention: failed to copy paged KV to host for export\n");
        }
        if (mLayerIndex < 0) {
            mMeta->layer_index = (mMeta->layer_index + 1) % std::max(1, mMeta->layer_nums);
        }
    }

    size_t outputBytes = static_cast<size_t>(mBatch) * mQuerySeqLen * mNumHead * mHeadDim * mPrecision;
    cudaMemset(pagedDevPtr<void>(output), 0, outputBytes);
    if (insertLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }

    mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    bool useMask = mask != nullptr && mask->elementSize() > 1 && mask->getType().code == halide_type_float;
    int maskElements = useMask ? static_cast<int>(mask->elementSize()) : 0;
    const bool profile = profilePagedAttention();
    const uint64_t attentionStartUs = profile ? nowUs() : 0;
    if (insertLen > 1) {
        int qSplitNum = 1;
        if (insertLen > 1024) {
            qSplitNum = UP_DIV(insertLen, 1024);
        } else if (insertLen > 256) {
            qSplitNum = UP_DIV(insertLen, 256);
        }
        const int maxPieceLen = UP_DIV(insertLen, qSplitNum);
        size_t prefillElements = static_cast<size_t>(mBatch) * mNumHead * maxPieceLen * kvLen;
        if (ensurePrefillTemp(prefillElements)) {
            for (int piece = 0; piece < qSplitNum; ++piece) {
                const int qStart = piece * maxPieceLen;
                const int qPieceLen = std::min(maxPieceLen, insertLen - qStart);
                if (qPieceLen <= 0) {
                    continue;
                }
                dim3 qkBlock(16, 16, 1);
                dim3 qkGrid(UP_DIV(kvLen, qkBlock.x), UP_DIV(qPieceLen, qkBlock.y), mBatch * mNumHead);
                if (mPrecision == 4) {
                    pagedPrefillQKKernel<float><<<qkGrid, qkBlock, 0, stream>>>(
                        pagedDevPtr<float>(query), pagedDevPtr<float>(mCache->key.get()), mPrefillQK,
                        pagedDevPtr<int>(mCache->slotTable.get()), useMask ? pagedDevPtr<float>(mask) : nullptr,
                        maskElements, mBatch, mQuerySeqLen, insertLen, mNumHead, mKvNumHead, mHeadDim,
                        baseLogical, qStart, qPieceLen, kvLen, mCache->maxSlots, mScale, sparseQueryDevice);
                } else {
                    pagedPrefillQKKernel<half><<<qkGrid, qkBlock, 0, stream>>>(
                        pagedDevPtr<half>(query), pagedDevPtr<half>(mCache->key.get()), mPrefillQK,
                        pagedDevPtr<int>(mCache->slotTable.get()), useMask ? pagedDevPtr<float>(mask) : nullptr,
                        maskElements, mBatch, mQuerySeqLen, insertLen, mNumHead, mKvNumHead, mHeadDim,
                        baseLogical, qStart, qPieceLen, kvLen, mCache->maxSlots, mScale, sparseQueryDevice);
                }
                checkKernelErrors;

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
                checkKernelErrors;

                dim3 qkvBlock(32, 8, 1);
                dim3 qkvGrid(UP_DIV(mHeadDim, qkvBlock.x), UP_DIV(qPieceLen, qkvBlock.y), mBatch * mNumHead);
                if (mPrecision == 4) {
                    pagedPrefillQKVKernel<float><<<qkvGrid, qkvBlock, 0, stream>>>(
                        mPrefillSoftmax, pagedDevPtr<float>(mCache->value.get()), pagedDevPtr<float>(output),
                        pagedDevPtr<int>(mCache->slotTable.get()), mBatch, mQuerySeqLen, qStart, qPieceLen,
                        mNumHead, mKvNumHead, mHeadDim, kvLen, mCache->maxSlots);
                } else {
                    pagedPrefillQKVKernel<half><<<qkvGrid, qkvBlock, 0, stream>>>(
                        mPrefillSoftmax, pagedDevPtr<half>(mCache->value.get()), pagedDevPtr<half>(output),
                        pagedDevPtr<int>(mCache->slotTable.get()), mBatch, mQuerySeqLen, qStart, qPieceLen,
                        mNumHead, mKvNumHead, mHeadDim, kvLen, mCache->maxSlots);
                }
                checkKernelErrors;
            }
            if (profile) {
                cudaDeviceSynchronize();
                MNN_PRINT("CUDAPagedAttention profile op=fast_prefill layer=%d query=%d insert=%d kv_len=%d "
                          "mask_elements=%d q_split=%d us=%llu\n",
                          layerIndex, mQuerySeqLen, insertLen, kvLen, maskElements, qSplitNum,
                          static_cast<unsigned long long>(nowUs() - attentionStartUs));
            }
            return NO_ERROR;
        }
    }

    dim3 grid(insertLen, mNumHead, mBatch);
    int blockSize = 128;
    int sharedBytes = (kvLen + blockSize) * sizeof(float);
    if (mPrecision == 4) {
        pagedAttentionKernel<float><<<grid, blockSize, sharedBytes, stream>>>(
            pagedDevPtr<float>(query), pagedDevPtr<float>(mCache->key.get()), pagedDevPtr<float>(mCache->value.get()),
            pagedDevPtr<float>(output), useMask ? pagedDevPtr<float>(mask) : nullptr,
            pagedDevPtr<int>(mCache->slotTable.get()), maskElements, mBatch, mQuerySeqLen, insertLen, mNumHead,
            mKvNumHead, mHeadDim, baseLogical, kvLen, mCache->maxSlots, mScale, sparseQueryDevice);
    } else {
        pagedAttentionKernel<half><<<grid, blockSize, sharedBytes, stream>>>(
            pagedDevPtr<half>(query), pagedDevPtr<half>(mCache->key.get()), pagedDevPtr<half>(mCache->value.get()),
            pagedDevPtr<half>(output), useMask ? pagedDevPtr<float>(mask) : nullptr,
            pagedDevPtr<int>(mCache->slotTable.get()), maskElements, mBatch, mQuerySeqLen, insertLen, mNumHead,
            mKvNumHead, mHeadDim, baseLogical, kvLen, mCache->maxSlots, mScale, sparseQueryDevice);
    }
    checkKernelErrors;
    if (profile) {
        cudaDeviceSynchronize();
        MNN_PRINT("CUDAPagedAttention profile op=generic layer=%d query=%d insert=%d kv_len=%d sparse=%d "
                  "mask_elements=%d us=%llu\n",
                  layerIndex, mQuerySeqLen, insertLen, kvLen, sparseQuery ? 1 : 0, maskElements,
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
