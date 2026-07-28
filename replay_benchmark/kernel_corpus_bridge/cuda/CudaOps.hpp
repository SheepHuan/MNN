#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CUDA_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CUDA_OPS_HPP

#include "CudaOpAdapter.hpp"
#include <vector>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// Shared launch geometry helpers used by all adapter .cpp/.cu files.
constexpr int kBlock = 128;
inline int gridFor(size_t count) { return static_cast<int>((count + kBlock - 1) / kBlock); }

// Shared input-data fillers (inline so each .cpp TU sees the same definition).
// Defined here instead of an anonymous namespace so that CudaOps.cpp and
// CudaOpsMisc.cpp can both use them without ODR violations.
inline void fillInputAlternating(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) {
        float mag = 0.5f * (i % 7);
        v[i] = (i % 2 == 0) ? mag : -mag;
    }
}
inline void fillInputRamp(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % 13);
}

// Reproduce MNN 1.2.0's adaptive block-size selection (CUDARuntime::blocks_num,
// source/backend/cuda/core/runtime/CUDARuntime.cpp at tag 1.2.0). 1.2.1+ and
// 3.6.0 fixed mThreadPerBlock = 128, so only 1.2.0 adapters should call this.
// maxThreadsPerBlock defaults to 1024 (typical for Ampere/Turing consumer GPUs);
// pass the device's actual value if known. The rule: pick the largest power-of-2
// bucket so that total_threads / bucket > maxNum, else fall back to 128.
inline int mnnBlock120(size_t total_threads, int maxThreadsPerBlock = 1024) {
    if (total_threads / 32 > (size_t)maxThreadsPerBlock)      return maxThreadsPerBlock;
    else if (total_threads / 16 > (size_t)maxThreadsPerBlock) return maxThreadsPerBlock / 2;
    else if (total_threads / 8 > (size_t)maxThreadsPerBlock)  return maxThreadsPerBlock / 4;
    else if (total_threads / 4 > (size_t)maxThreadsPerBlock)  return maxThreadsPerBlock / 8;
    return 128;
}
// Grid (block_num) for a given workload size and block size — matches MNN's
// (total + block - 1) / block in both 1.2.0 and 3.6.0.
inline int mnnGridFor(size_t total, int block) { return static_cast<int>((total + block - 1) / block); }

// ============================================================================
// CUDA corpus adapters. fp32 variants are in CudaOps.cpp (compiled by g++).
// fp16/int8/bf16 variants are in CudaOpsFp16.cu (compiled by nvcc, because
// __half is unavailable in plain g++ host code).
// ============================================================================

// ---- Unary (fp32/int8 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaReluFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "relu"; }
    const char* variant() const override { return "cuda_relu_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReluFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "relu"; }
    const char* variant() const override { return "cuda_relu_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReluInt8Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "relu"; }
    const char* variant() const override { return "cuda_relu_int8"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaClampFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "clamp"; }
    const char* variant() const override { return "cuda_clamp_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaClampFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "clamp"; }
    const char* variant() const override { return "cuda_clamp_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Cast ----
class CudaCastBoolKernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "castbool"; }
    const char* variant() const override { return "cuda_castbool_i32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCastF322I32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cuda_cast_f32_i32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCastI322F32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cuda_cast_i32_f32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCastI82I32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cuda_cast_i8_i32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCastI322U8Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cuda_cast_i32_u8"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCastU82I32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cuda_cast_u8_i32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Binary (fp32 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaBinaryAtan2Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "cuda_atan2_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryAtan2Fp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "cuda_atan2_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryModKernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "cuda_mod_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryModFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "cuda_mod_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryLogicalOrKernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "cuda_logicalor_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryLogicalOrFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "cuda_logicalor_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Range (fp32/i32 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaRangeFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "range"; }
    const char* variant() const override { return "cuda_range_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaRangeI32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "range"; }
    const char* variant() const override { return "cuda_range_i32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaRangeFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "range"; }
    const char* variant() const override { return "cuda_range_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Select (fp32 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaSelectFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "select"; }
    const char* variant() const override { return "cuda_select_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSelectFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "select"; }
    const char* variant() const override { return "cuda_select_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Softmax (fp32 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaSoftmaxFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "cuda_softmax_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSoftmaxFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "cuda_softmax_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- LayerNorm (fp32 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaLayerNormFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_layernorm_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaLayerNormFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_layernorm_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- PReLU (fp32 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaPreluFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "prelu"; }
    const char* variant() const override { return "cuda_prelu_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaPreluFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "prelu"; }
    const char* variant() const override { return "cuda_prelu_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Scale (fp32 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaScaleFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "scale"; }
    const char* variant() const override { return "cuda_scale_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaScaleFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "scale"; }
    const char* variant() const override { return "cuda_scale_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Pool (fp32 in CudaOps.cpp, fp16 in CudaOpsFp16.cu) ----
class CudaMaxPoolFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "maxpool"; }
    const char* variant() const override { return "cuda_maxpool_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaMaxPoolFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "maxpool"; }
    const char* variant() const override { return "cuda_maxpool_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaAvgPoolFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "avgpool"; }
    const char* variant() const override { return "cuda_avgpool_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaAvgPoolFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "avgpool"; }
    const char* variant() const override { return "cuda_avgpool_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGlobalAvgPoolFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "global_avgpool"; }
    const char* variant() const override { return "cuda_global_avgpool_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGlobalAvgPoolFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "global_avgpool"; }
    const char* variant() const override { return "cuda_global_avgpool_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGlobalMaxPoolFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "global_maxpool"; }
    const char* variant() const override { return "cuda_global_maxpool_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGlobalMaxPoolFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "global_maxpool"; }
    const char* variant() const override { return "cuda_global_maxpool_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// Registration functions. fp32/int8 adapters register in CudaOps.cpp; the
// fp16 variants and extra cast types register in CudaOpsFp16.cu.
void registerCudaOps();
void registerCudaOpsFp16();
// ---- B-class: GatherV2 / ArgMax / ArgMin / Interp / Transpose ----
class CudaGatherV2Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "gatherv2"; }
    const char* variant() const override { return "cuda_gatherv2_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaArgMaxFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "argmax"; }
    const char* variant() const override { return "cuda_argmax_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaArgMinFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "argmin"; }
    const char* variant() const override { return "cuda_argmin_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInterpNearestFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "interp_nearest"; }
    const char* variant() const override { return "cuda_interp_nearest_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaNhwc2NchwFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_nhwc2nchw_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaNchw2NhwcFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_nchw2nhwc_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

class CudaGridSampleNearestFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "grid_sample"; }
    const char* variant() const override { return "cuda_grid_sample_nearest_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- B-class: Reduction (SUM/MAX/MIN/MEAN/PROD) ----
class CudaReductionSumFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_sum_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionMeanFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_mean_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionMaxFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_max_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionMinFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_min_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionProdFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_prod_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- B-class: Interp bilinear / round ----
class CudaInterpBilinearFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "interp_bilinear"; }
    const char* variant() const override { return "cuda_interp_bilinear_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInterpNearestRoundFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "interp_nearest_round"; }
    const char* variant() const override { return "cuda_interp_nearest_round_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- B-class: GridSample bilinear ----
class CudaGridSampleBilinearFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "grid_sample_bilinear"; }
    const char* variant() const override { return "cuda_grid_sample_bilinear_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- TopKV2 ----
class CudaTopKV2Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "topkv2"; }
    const char* variant() const override { return "cuda_topkv2_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- Raster blitRegion ----
class CudaBlitRegionFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_blitregion_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- fp16 B-class ----
class CudaGatherV2Fp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "gatherv2"; }
    const char* variant() const override { return "cuda_gatherv2_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaArgMaxFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "argmax"; }
    const char* variant() const override { return "cuda_argmax_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaArgMinFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "argmin"; }
    const char* variant() const override { return "cuda_argmin_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- GridSample 3D ----
class CudaGridSampleNearest3dFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "grid_sample_3d"; }
    const char* variant() const override { return "cuda_grid_sample_nearest_3d_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGridSampleBilinear3dFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "grid_sample_3d"; }
    const char* variant() const override { return "cuda_grid_sample_bilinear_3d_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Conv DepthWise ----
class CudaConvDwFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_conv_dw_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Transpose fp16 ----
class CudaNhwc2NchwFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_nhwc2nchw_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaNchw2NhwcFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_nchw2nhwc_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- 1.2.0 tag variants (old kernel signatures) ----
class CudaPackC4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_pack_c4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaUnpackC4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_unpack_c4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSetZeroFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "scatternd"; }
    const char* variant() const override { return "cuda_setzero_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaAddBiasFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "matmul"; }
    const char* variant() const override { return "cuda_add_bias_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CUDA_OPS_HPP
