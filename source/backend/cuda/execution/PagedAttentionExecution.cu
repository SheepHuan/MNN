#include "PagedAttentionExecution.hpp"
#include "SoftmaxExecution.hpp"
#include "core/Macro.h"
#include "core/MNNFileUtils.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <float.h>
#include <fstream>
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

static inline float hostPagedRead(const int8_t* ptr, int index, int bytes) {
    if (bytes == 2) {
        return __half2float(reinterpret_cast<const half*>(ptr)[index]);
    }
    return reinterpret_cast<const float*>(ptr)[index];
}

static inline void hostPagedWrite(int8_t* ptr, int index, float value, int bytes) {
    if (bytes == 2) {
        reinterpret_cast<half*>(ptr)[index] = __float2half(value);
        return;
    }
    reinterpret_cast<float*>(ptr)[index] = value;
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

static float ropeInvFreqForSegment(const PagedKVExternalSegment& segment, int pairIndex, int ropeDim) {
    const float theta = segment.ropeTheta > 0.0f ? segment.ropeTheta : 10000.0f;
    float invFreq = std::pow(theta, -static_cast<float>(2 * pairIndex) / static_cast<float>(ropeDim));
    if (segment.ropeType != "llama3") {
        return invFreq;
    }
    const float factor = std::max(segment.ropeScalingFactor, 1.0f);
    const float lowFreqFactor = std::max(segment.ropeScalingLowFreqFactor, 1.0e-6f);
    const float highFreqFactor = std::max(segment.ropeScalingHighFreqFactor, 1.0e-6f);
    const int oldContext = segment.ropeScalingOriginalMaxPositionEmbeddings > 0
        ? segment.ropeScalingOriginalMaxPositionEmbeddings
        : segment.maxPositionEmbeddings;
    if (oldContext <= 0 || factor == 1.0f || lowFreqFactor == highFreqFactor) {
        return invFreq;
    }
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float wavelen = kTwoPi / invFreq;
    const float lowFreqWavelen = static_cast<float>(oldContext) / lowFreqFactor;
    const float highFreqWavelen = static_cast<float>(oldContext) / highFreqFactor;
    float scaled = wavelen > lowFreqWavelen ? invFreq / factor : invFreq;
    if (wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen) {
        const float smooth = (static_cast<float>(oldContext) / wavelen - lowFreqFactor) /
                             (highFreqFactor - lowFreqFactor);
        scaled = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
    }
    return scaled;
}

static void applyForwardRopeToSegment(std::vector<int8_t>& keyData, const PagedKVExternalSegment& segment,
                                      size_t sourceTokenOffset, int batch, int kvHeads, int headDim, int bytes) {
    int ropeDim = segment.ropeDim > 0 ? segment.ropeDim : headDim;
    ropeDim = std::min(ropeDim, headDim);
    ropeDim = (ropeDim / 2) * 2;
    if (ropeDim <= 0) {
        return;
    }
    const int half = ropeDim / 2;
    const float attentionScale = segment.ropeAttentionScaling > 0.0f ? segment.ropeAttentionScaling : 1.0f;
    for (size_t local = 0; local < segment.tokenCount; ++local) {
        const int logical = static_cast<int>(segment.logicalStart + local);
        const int sourceToken = static_cast<int>(sourceTokenOffset + local);
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < kvHeads; ++h) {
                const int base = ((sourceToken * batch + b) * kvHeads + h) * headDim;
                for (int p = 0; p < half; ++p) {
                    const float angle = static_cast<float>(logical) * ropeInvFreqForSegment(segment, p, ropeDim);
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const float x0 = hostPagedRead(keyData.data(), base + p, bytes);
                    const float x1 = hostPagedRead(keyData.data(), base + p + half, bytes);
                    hostPagedWrite(keyData.data(), base + p, (x0 * c - x1 * s) * attentionScale, bytes);
                    hostPagedWrite(keyData.data(), base + p + half, (x1 * c + x0 * s) * attentionScale, bytes);
                }
            }
        }
    }
}

static ErrorCode restoreExternalSegmentsCUDA(PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim,
                                             int bytes, int maxSlots, const std::vector<int>& physicalSlots,
                                             int kvLen, void* keyCacheDevice, void* valueCacheDevice) {
    if (meta == nullptr || meta->external_segments.empty() || meta->externalLayerLoaded(layerIndex)) {
        return NO_ERROR;
    }
    auto keyBase = reinterpret_cast<int8_t*>(keyCacheDevice);
    auto valueBase = reinterpret_cast<int8_t*>(valueCacheDevice);
    for (const auto& segment : meta->external_segments) {
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
        const size_t expectedKey = sourceTokenCount * static_cast<size_t>(batch) * kvHeads * headDim * bytes;
        const size_t expectedValue = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * headDim * bytes;
        if (keyData.size() < expectedKey || valueData.size() < expectedValue) {
            MNN_ERROR("CUDAPagedAttention external PIC KV file is too small at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        applyForwardRopeToSegment(keyData, segment, sourceTokenOffset, batch, kvHeads, headDim, bytes);
        bool contiguousSlots = true;
        int firstSlot = -1;
        for (size_t local = 0; local < segment.tokenCount; ++local) {
            int logical = static_cast<int>(segment.logicalStart + local);
            int slot = logical >= 0 && logical < static_cast<int>(physicalSlots.size()) ? physicalSlots[logical] : -1;
            if (slot < 0 || slot >= maxSlots) {
                return OUT_OF_MEMORY;
            }
            if (local == 0) {
                firstSlot = slot;
            } else if (slot != firstSlot + static_cast<int>(local)) {
                contiguousSlots = false;
            }
        }
        const size_t keyTokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
        const size_t keySourceBase = sourceTokenOffset * keyTokenBytes;
        if (contiguousSlots) {
            cudaMemcpy(keyBase + static_cast<size_t>(firstSlot) * keyTokenBytes, keyData.data() + keySourceBase,
                       segment.tokenCount * keyTokenBytes, cudaMemcpyHostToDevice);
        } else {
            for (size_t local = 0; local < segment.tokenCount; ++local) {
                int logical = static_cast<int>(segment.logicalStart + local);
                int slot = physicalSlots[logical];
                cudaMemcpy(keyBase + static_cast<size_t>(slot) * keyTokenBytes,
                           keyData.data() + (sourceTokenOffset + local) * keyTokenBytes,
                           keyTokenBytes, cudaMemcpyHostToDevice);
            }
        }
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < kvHeads; ++h) {
                const size_t srcBase = (static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount * headDim * bytes;
                if (contiguousSlots) {
                    const size_t dstBase = ((static_cast<size_t>(b) * kvHeads + h) * maxSlots + firstSlot) *
                                           headDim * bytes;
                    cudaMemcpy(valueBase + dstBase,
                               valueData.data() + srcBase + sourceTokenOffset * static_cast<size_t>(headDim) * bytes,
                               segment.tokenCount * static_cast<size_t>(headDim) * bytes, cudaMemcpyHostToDevice);
                } else {
                    for (size_t local = 0; local < segment.tokenCount; ++local) {
                        int logical = static_cast<int>(segment.logicalStart + local);
                        int slot = physicalSlots[logical];
                        const size_t dstBase = ((static_cast<size_t>(b) * kvHeads + h) * maxSlots + slot) *
                                               headDim * bytes;
                        cudaMemcpy(valueBase + dstBase,
                                   valueData.data() + srcBase +
                                       (sourceTokenOffset + local) * static_cast<size_t>(headDim) * bytes,
                                   static_cast<size_t>(headDim) * bytes, cudaMemcpyHostToDevice);
                    }
                }
            }
        }
    }
    meta->markExternalLayerLoaded(layerIndex);
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
                                               pagedDevPtr<void>(mCache->value.get()));
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
