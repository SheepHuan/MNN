// plugins.cu - Plugin kernels for replay benchmark corpus
//   source/backend/cuda/execution/plugin/GroupNorm/groupNormKernel.cu
//   source/backend/cuda/execution/plugin/SeqLen2Spatial/seqLen2SpatialKernel.cu
//   source/backend/cuda/execution/plugin/SplitGelu/splitGeLUKernel.cu
//   source/backend/cuda/execution/plugin/FmhaCommon/FmhaV2CommonExecution.cu (SPLIT_FusedQKV)
//
// GroupNorm is a faithful copy of MNN's groupNormKernel.cu, including the
// cub::BlockScan segmented scan. This requires -fexceptions on the host
// compiler (MNN CUDA backend adds it at source/backend/cuda/CMakeLists.txt:110;
// the corpus mirrors the same override in replay_benchmark/CMakeLists.txt via
// -Xcompiler -fexceptions). Without -fexceptions, thrust's system_error.inl
// fails with "exception handling disabled".
#include "corpus_common.cuh"
#include <cuda_fp16.h>
#include <cub/cub.cuh>

namespace MNN {
namespace Corpus {

static inline __device__ __host__ float groupNormSigmoid(float x)
{
    return 1.F / (1.F + expf(-x));
}

// ============================================================================
// GroupNorm NHWC params (faithful mirror of MNN::CUDA::GroupNormNHWCParams)
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

// Segmented scan structures (faithful copy of MNN GroupSums/GroupSumsOp)
struct GroupSums {
    int32_t flag;
    float sum;
    float sumSq;
};
struct GroupSumsOp {
    inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b) {
        GroupSums dst;
        dst.sum = b.flag ? b.sum : (a.sum + b.sum);
        dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
        dst.flag = a.flag + b.flag;
        return dst;
    }
};

// ============================================================================
// groupNormNHWCSumKernel: faithful copy from
// source/backend/cuda/execution/plugin/GroupNorm/groupNormKernel.cu
// Uses cub::BlockScan for segmented inclusive scan, then atomicAdd to redBuffer.
// ============================================================================
template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCSumKernel(GroupNormNHWCParams params)
{
    typedef cub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;
    __shared__ float2 smem[tTHREADS_PER_BLOCK];

    int32_t ni = blockIdx.z;
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    float sum = 0.F;
    float sumSq = 0.F;

    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
        int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hwi) * params.c + ci;
        __half2 h2(0, 0);
        float2 f2;
        f2.x = 0.0f; f2.y = 0.0f;
        if (ci < params.c) {
            if (params.src != nullptr) {
                h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
                f2 = __half22float2(h2);
            } else {
                int64_t offset_1 = static_cast<int64_t>(ni) * params.c + ci;
                __half2 h2_0 = *reinterpret_cast<__half2 const*>(&params.src_0[offset]);
                __half2 h2_1 = *reinterpret_cast<__half2 const*>(&params.src_1[offset_1]);
                float2 f2_0 = __half22float2(h2_0);
                float2 f2_1 = __half22float2(h2_1);
                f2.x = f2_0.x + f2_1.x;
                f2.y = f2_0.y + f2_1.y;
            }
        }
        sum += f2.x + f2.y;
        sumSq += f2.x * f2.x + f2.y * f2.y;
    }

    int32_t gi = threadIdx.x * 2 / params.cPerGroup;
    int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi;
    GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};
    GroupSums out;
    BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

    if (cj == params.cPerGroup - 2) {
        smem[gi] = make_float2(out.sum, out.sumSq);
    }
    __syncthreads();

    int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;
    if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups) {
        return;
    }
    float2 sums = smem[threadIdx.x];
    atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
    atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

// ============================================================================
// groupNormNHWCScaleKernel: faithful copy from groupNormKernel.cu
// ============================================================================
template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCScaleKernel(GroupNormNHWCParams params)
{
    int32_t ni = blockIdx.z;
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
    int32_t gi = ci / params.cPerGroup;

    float sum = 0.F, sumSq = 0.F;
    if (gi < params.groups) {
        sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
        sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
    }

    float2 gammaF2, betaF2;
    if (ci < params.c) {
        gammaF2 = *reinterpret_cast<float2 const*>(&params.gamma[ci]);
        betaF2 = *reinterpret_cast<float2 const*>(&params.beta[ci]);
    }

    float mean = sum * params.invHWC;
    float var = sumSq * params.invHWC - (mean * mean);
    float invStdDev = var <= 0.F ? 1.F : rsqrtf(var);

    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
        int64_t offset = (int64_t)ni * params.hwc + hwi * params.c + ci;
        __half2 h2(0, 0);
        float2 f2;
        f2.x = 0.0f; f2.y = 0.0f;
        if (ci < params.c) {
            if (params.src != nullptr) {
                h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
                f2 = __half22float2(h2);
            } else {
                int64_t offset_1 = static_cast<int64_t>(ni) * params.c + ci;
                __half2 h2_0 = *reinterpret_cast<__half2 const*>(&params.src_0[offset]);
                __half2 h2_1 = *reinterpret_cast<__half2 const*>(&params.src_1[offset_1]);
                float2 f2_0 = __half22float2(h2_0);
                float2 f2_1 = __half22float2(h2_1);
                f2.x = f2_0.x + f2_1.x;
                f2.y = f2_0.y + f2_1.y;
            }
        }
        f2.x = (f2.x - mean) * invStdDev;
        f2.y = (f2.y - mean) * invStdDev;
        f2.x = gammaF2.x * f2.x + betaF2.x;
        f2.y = gammaF2.y * f2.y + betaF2.y;
        if (params.withSwish) {
            f2.x = f2.x * groupNormSigmoid(f2.x);
            f2.y = f2.y * groupNormSigmoid(f2.y);
        }
        if (ci < params.c) {
            *reinterpret_cast<__half2*>(&params.dst[offset]) = __float22half2_rn(f2);
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

// ============================================================================
// SPLIT_FusedKV: [B,S,H,2,D] -> 2x [B,S,H,D]
// 忠实复制 MNN plugin/FmhaCommon/FmhaV2CommonExecution.cu:30
// ============================================================================
template <typename T>
__global__ void SPLIT_FusedKV(const size_t count, const T* fused_kv,
                               T* ptr_k, T* ptr_v, int head_size) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int bsh = i / head_size;
        const int d = i % head_size;
        ptr_k[i] = fused_kv[(bsh * 2 + 0) * head_size + d];
        ptr_v[i] = fused_kv[(bsh * 2 + 1) * head_size + d];
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
    // Faithful dispatch: MNN selects thread count by cPerBlock
    // (320→160, 480→256, 256→128, 128→64). The `block` arg from adapter is
    // ignored for the kernel launch to match MNN exactly.
    switch (cPerBlock) {
        case 320: MNN::Corpus::groupNormNHWCSumKernel<160><<<grid, 160, 0, stream>>>(p); break;
        case 480: MNN::Corpus::groupNormNHWCSumKernel<256><<<grid, 256, 0, stream>>>(p); break;
        case 256: MNN::Corpus::groupNormNHWCSumKernel<128><<<grid, 128, 0, stream>>>(p); break;
        case 128: MNN::Corpus::groupNormNHWCSumKernel<64><<<grid, 64, 0, stream>>>(p); break;
        default: break;
    }
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
    switch (cPerBlock) {
        case 320: MNN::Corpus::groupNormNHWCScaleKernel<160><<<grid, 160, 0, stream>>>(p); break;
        case 480: MNN::Corpus::groupNormNHWCScaleKernel<256><<<grid, 256, 0, stream>>>(p); break;
        case 256: MNN::Corpus::groupNormNHWCScaleKernel<128><<<grid, 128, 0, stream>>>(p); break;
        case 128: MNN::Corpus::groupNormNHWCScaleKernel<64><<<grid, 64, 0, stream>>>(p); break;
        default: break;
    }
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

// ---- SPLIT_FusedKV fp32 ----
void mnn_corpus_split_fusedkv_fp32(size_t count, const void* fused_kv,
                                    void* ptr_k, void* ptr_v, int head_size,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SPLIT_FusedKV<float><<<grid, block, 0, stream>>>(
        count, (const float*)fused_kv, (float*)ptr_k, (float*)ptr_v, head_size);
}
// ---- SPLIT_FusedKV fp16 ----
void mnn_corpus_split_fusedkv_fp16(size_t count, const void* fused_kv,
                                    void* ptr_k, void* ptr_v, int head_size,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SPLIT_FusedKV<__half><<<grid, block, 0, stream>>>(
        count, (const __half*)fused_kv, (__half*)ptr_k, (__half*)ptr_v, head_size);
}

} // extern "C"
