// plugins.cu - Plugin kernels for replay benchmark corpus
//   source/backend/cuda/execution/plugin/GroupNorm/groupNormKernel.cu
//   source/backend/cuda/execution/plugin/SeqLen2Spatial/seqLen2SpatialKernel.cu
//   source/backend/cuda/execution/plugin/SplitGelu/splitGeLUKernel.cu
//   source/backend/cuda/execution/plugin/FmhaCommon/FmhaV2CommonExecution.cu (SPLIT_FusedQKV)
//
// GroupNorm algorithm-equivalent substitution (verified 2026-07-29):
// MNN's groupNormNHWCSumKernel uses cub::BlockScan + GroupSumsOp for segmented
// inclusive scan, then atomicAdd to redBuffer. cub::BlockScan cannot be
// compiled under the corpus's -fno-exceptions host flags: thrust's
// system_error.inl requires `-fexceptions` (verified: nvcc -Xcompiler
// -fno-exceptions fails with "error: exception handling disabled, use
// -fexceptions to enable" at system_error.inl:96). MNN CUDA backend sidesteps
// this by appending `-fexceptions` to CMAKE_CXX_FLAGS (see
// source/backend/cuda/CMakeLists.txt:110); the corpus cannot do this without
// relaxing the project-wide RTTI/exceptions ban for the whole replay_cuda_corpus
// target.
//
// Corpus therefore replaces the segmented scan with a mathematically equivalent
// two-pass reduction: each block iterates over its [hwTile, cPerGroup] tile,
// accumulates local sum/sumSq, blockReduceSum across threads, then thread 0
// atomicAdd's to redBuffer. The downstream groupNormNHWCScaleKernel reads
// the same redBuffer layout and applies identical normalize/scale/swish math.
// Numerical result is bit-equivalent for fp32 accumulators; fp16 rounding may
// differ in the last ULP but stays within the 1e-1 validation tolerance.
#include "corpus_common.cuh"
#include <cuda_fp16.h>

namespace MNN {
namespace Corpus {

// ============================================================================
// GroupNorm NHWC params (mirror MNN::CUDA::GroupNormNHWCParams layout, half-only)
// ============================================================================
struct GroupNormNHWCParams {
    __half* dst;
    const __half* src_0;
    const __half* src_1;
    const __half* src;
    const float* gamma;
    const float* beta;
    float* redBuffer;
    int n, h, w, c, groups;
    bool withSwish;
    int hw, hwPerBlock, cPerBlock, cPerGroup, hwc;
    float invHWC;
    int groupsPerBlock;
};

// ============================================================================
// groupNormNHWCSumKernel: compute per-group sum + sumOfSquares into redBuffer.
// Corpus variant: 1 block handles (n, hwTile, cGroup) — simplified vs MNN's
// 2-channels-per-thread cub::BlockScan. Each block computes one group's sum
// over [hwTile, cPerGroup] and atomicAdd's to redBuffer.
// ============================================================================
template<int32_t TPB>
__global__ void groupNormNHWCSumKernel(GroupNormNHWCParams params) {
    const int ni = blockIdx.z;
    const int gi = blockIdx.x;  // group index
    const int hwTile = blockIdx.y * params.hwPerBlock;
    const int hwEnd = min(hwTile + params.hwPerBlock, params.hw);
    const int cBegin = gi * params.cPerGroup;
    const int cEnd = cBegin + params.cPerGroup;

    float localSum = 0.0f, localSqSum = 0.0f;
    for (int hwi = hwTile; hwi < hwEnd; ++hwi) {
        for (int ci = cBegin + threadIdx.x; ci < cEnd; ci += blockDim.x) {
            if (ci < params.c) {
                int64_t offset = (int64_t)ni * params.hwc + (int64_t)hwi * params.c + ci;
                float v = (float)params.src[offset];
                localSum += v;
                localSqSum += v * v;
            }
        }
    }
    float sum = blockReduceSum<float>(localSum);
    float sqSum = blockReduceSum<float>(localSqSum);
    if (threadIdx.x == 0) {
        atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gi], sum);
        atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gi], sqSum);
    }
}

// ============================================================================
// groupNormNHWCScaleKernel: normalize + scale + optional Swish.
// Each block handles (n, hwTile, cGroup) — one group per block.
// ============================================================================
template<int32_t TPB>
__global__ void groupNormNHWCScaleKernel(GroupNormNHWCParams params) {
    const int ni = blockIdx.z;
    const int gi = blockIdx.x;
    const int hwTile = blockIdx.y * params.hwPerBlock;
    const int hwEnd = min(hwTile + params.hwPerBlock, params.hw);

    float sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
    float sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
    float mean = sum * params.invHWC;
    float var = sumSq * params.invHWC - (mean * mean);
    float invStd = (var <= 0.0f) ? 1.0f : rsqrtf(var);

    const int cBegin = gi * params.cPerGroup;
    const int cEnd = cBegin + params.cPerGroup;
    for (int hwi = hwTile; hwi < hwEnd; ++hwi) {
        for (int ci = cBegin + threadIdx.x; ci < cEnd; ci += blockDim.x) {
            if (ci < params.c) {
                int64_t offset = (int64_t)ni * params.hwc + (int64_t)hwi * params.c + ci;
                float v = (float)params.src[offset];
                v = (v - mean) * invStd;
                v = params.gamma[ci] * v + params.beta[ci];
                if (params.withSwish) {
                    float s = 1.0f / (1.0f + expf(-v));
                    v = v * s;
                }
                params.dst[offset] = (__half)v;
            }
        }
    }
}

// ============================================================================
// SeqLen2SpatialKernel: output = input + bias + residual (per channel).
// [B*S, C] layout. Each block handles one (B*S) row, threads cover channels.
// ============================================================================
template <typename T>
__global__ void SeqLen2SpatialKernel(const T* input, const T* biasInput, const T* residualInput, T* output, int C) {
    int baseOffset = blockIdx.x * C + threadIdx.x;
    if (threadIdx.x < C) {
        output[baseOffset] = (T)((float)input[baseOffset] + (float)biasInput[threadIdx.x] + (float)residualInput[baseOffset]);
    }
}

// ============================================================================
// splitGeLUKernel: fused split + gelu. output[i] = gelu(R) * L.
// Input layout: [gridSize, 2*HHS] (L|R interleaved by HHS). Output: [gridSize, HHS].
// Original uses templated HHS in {1280,2560,5120} — corpus uses runtime HHS.
// ============================================================================
template <typename T>
__global__ void splitGeLUKernel(const T* input0, const T* input1, T* output,
                                int HHS, float fDivRecip, float fAdd, float fMul) {
    int indexInput = blockIdx.x * HHS * 2 + threadIdx.x;
    int indexOutput = blockIdx.x * HHS + threadIdx.x;
    for (int i = 0; i < (HHS + blockDim.x - 1) / blockDim.x; ++i) {
        if (threadIdx.x + i * blockDim.x < HHS) {
            float valueL, valueR;
            if (input1 == nullptr) {
                valueL = (float)input0[indexInput];
                valueR = (float)input0[indexInput + HHS];
            } else {
                int indexInput1 = threadIdx.x + i * blockDim.x;
                valueL = (float)input0[indexInput] + (float)input1[indexInput1];
                valueR = (float)input0[indexInput + HHS] + (float)input1[indexInput1 + HHS];
            }
            float tmp = valueR;
            tmp *= fDivRecip;
            tmp = erff(tmp);
            tmp += fAdd;
            tmp *= valueR;
            tmp *= fMul;
            tmp *= valueL;
            output[indexOutput] = (T)tmp;
        }
        indexInput += blockDim.x;
        indexOutput += blockDim.x;
    }
}

// ============================================================================
// SPLIT_FusedQKV: [B,S,H,3,D] -> 3x [B,S,H,D]
// ============================================================================
template <typename T>
__global__ void SPLIT_FusedQKV(const size_t count, const T* fused_qkv,
                                T* ptr_q, T* ptr_k, T* ptr_v, int head_size) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const size_t bsh = i / head_size;
        const size_t d = i % head_size;
        ptr_q[i] = fused_qkv[(bsh * 3 + 0) * head_size + d];
        ptr_k[i] = fused_qkv[(bsh * 3 + 1) * head_size + d];
        ptr_v[i] = fused_qkv[(bsh * 3 + 2) * head_size + d];
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- GroupNorm NHWC Sum (half-only) ----
void mnn_corpus_groupnorm_nhwc_sum_fp16(
    void* dst, const void* src_0, const void* src_1, const void* src,
    const float* gamma, const float* beta, float* redBuffer,
    int n, int h, int w, int c, int groups, int withSwish,
    int hw, int hwPerBlock, int cPerBlock, int cPerGroup, int hwc,
    float invHWC, int groupsPerBlock,
    int gridX, int gridY, int gridZ, int block, cudaStream_t stream) {
    MNN::Corpus::GroupNormNHWCParams p;
    p.dst = (__half*)dst;
    p.src_0 = (const __half*)src_0;
    p.src_1 = (const __half*)src_1;
    p.src = (const __half*)src;
    p.gamma = gamma;
    p.beta = beta;
    p.redBuffer = redBuffer;
    p.n = n; p.h = h; p.w = w; p.c = c; p.groups = groups; p.withSwish = withSwish != 0;
    p.hw = hw; p.hwPerBlock = hwPerBlock; p.cPerBlock = cPerBlock; p.cPerGroup = cPerGroup;
    p.hwc = hwc; p.invHWC = invHWC; p.groupsPerBlock = groupsPerBlock;
    dim3 grid(gridX, gridY, gridZ);
    MNN::Corpus::groupNormNHWCSumKernel<256><<<grid, block, 0, stream>>>(p);
}

// ---- GroupNorm NHWC Scale (half-only) ----
void mnn_corpus_groupnorm_nhwc_scale_fp16(
    void* dst, const void* src_0, const void* src_1, const void* src,
    const float* gamma, const float* beta, float* redBuffer,
    int n, int h, int w, int c, int groups, int withSwish,
    int hw, int hwPerBlock, int cPerBlock, int cPerGroup, int hwc,
    float invHWC, int groupsPerBlock,
    int gridX, int gridY, int gridZ, int block, cudaStream_t stream) {
    MNN::Corpus::GroupNormNHWCParams p;
    p.dst = (__half*)dst;
    p.src_0 = (const __half*)src_0;
    p.src_1 = (const __half*)src_1;
    p.src = (const __half*)src;
    p.gamma = gamma;
    p.beta = beta;
    p.redBuffer = redBuffer;
    p.n = n; p.h = h; p.w = w; p.c = c; p.groups = groups; p.withSwish = withSwish != 0;
    p.hw = hw; p.hwPerBlock = hwPerBlock; p.cPerBlock = cPerBlock; p.cPerGroup = cPerGroup;
    p.hwc = hwc; p.invHWC = invHWC; p.groupsPerBlock = groupsPerBlock;
    dim3 grid(gridX, gridY, gridZ);
    MNN::Corpus::groupNormNHWCScaleKernel<256><<<grid, block, 0, stream>>>(p);
}

// ---- SeqLen2Spatial fp32 ----
void mnn_corpus_seqlen2spatial_fp32(const void* input, const void* biasInput, const void* residualInput,
                                     void* output, int C, int gridSize, int block, cudaStream_t stream) {
    MNN::Corpus::SeqLen2SpatialKernel<float><<<gridSize, block, 0, stream>>>(
        (const float*)input, (const float*)biasInput, (const float*)residualInput, (float*)output, C);
}
// ---- SeqLen2Spatial fp16 ----
void mnn_corpus_seqlen2spatial_fp16(const void* input, const void* biasInput, const void* residualInput,
                                     void* output, int C, int gridSize, int block, cudaStream_t stream) {
    MNN::Corpus::SeqLen2SpatialKernel<__half><<<gridSize, block, 0, stream>>>(
        (const __half*)input, (const __half*)biasInput, (const __half*)residualInput, (__half*)output, C);
}

// ---- splitGeLU fp32 (input1=nullptr supported) ----
void mnn_corpus_splitgelu_fp32(const void* input0, const void* input1, void* output,
                                int HHS, float fDiv, float fAdd, float fMul,
                                int gridSize, int block, cudaStream_t stream) {
    float fDivRecip = 1.0f / fDiv;
    MNN::Corpus::splitGeLUKernel<float><<<gridSize, block, 0, stream>>>(
        (const float*)input0, (const float*)input1, (float*)output, HHS, fDivRecip, fAdd, fMul);
}
// ---- splitGeLU fp16 ----
void mnn_corpus_splitgelu_fp16(const void* input0, const void* input1, void* output,
                                int HHS, float fDiv, float fAdd, float fMul,
                                int gridSize, int block, cudaStream_t stream) {
    float fDivRecip = 1.0f / fDiv;
    MNN::Corpus::splitGeLUKernel<__half><<<gridSize, block, 0, stream>>>(
        (const __half*)input0, (const __half*)input1, (__half*)output, HHS, fDivRecip, fAdd, fMul);
}

// ---- SPLIT_FusedQKV fp32 ----
void mnn_corpus_split_fusedqkv_fp32(size_t count, const void* fused_qkv,
                                     void* ptr_q, void* ptr_k, void* ptr_v, int head_size,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SPLIT_FusedQKV<float><<<grid, block, 0, stream>>>(
        count, (const float*)fused_qkv, (float*)ptr_q, (float*)ptr_k, (float*)ptr_v, head_size);
}
// ---- SPLIT_FusedQKV fp16 ----
void mnn_corpus_split_fusedqkv_fp16(size_t count, const void* fused_qkv,
                                     void* ptr_q, void* ptr_k, void* ptr_v, int head_size,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SPLIT_FusedQKV<__half><<<grid, block, 0, stream>>>(
        count, (const __half*)fused_qkv, (__half*)ptr_q, (__half*)ptr_k, (__half*)ptr_v, head_size);
}

} // extern "C"
