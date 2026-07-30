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
// Softmax variants are split into independent adapter classes per the
// "variant vs version" layering rule (see skills/kernel-adapt/SKILL.md).
//   naive         -> CudaSoftmaxFp32Kernel         (cuda_softmax_fp32)
//   warp32         -> CudaSoftmaxWarp32Fp32Kernel   (cuda_softmax_warp32_fp32)
//   axis_reduce    -> CudaSoftmaxAxisReduceFp32Kernel (cuda_softmax_axis_reduce_fp32)
// Each variant's tag branches live inside its own adapter.
class CudaSoftmaxFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "cuda_softmax_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSoftmaxWarp32Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "cuda_softmax_warp32_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSoftmaxAxisReduceFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "cuda_softmax_axis_reduce_fp32"; }
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
// Two-stage argmax variant: dim > 256, tag >= 2.5.0. Independent adapter per
// the variant-vs-version layering rule. Entry = mnn_corpus_argmax_twostage_fp32
// (3.6.0/2.8.4 fixed SECOND_STEP) or mnn_corpus_argmax_twostage_250_fp32
// (2.5.0-2.8.4 buggy SECOND_STEP, smoke-only).
class CudaArgMaxTwostageFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "argmax"; }
    const char* variant() const override { return "cuda_argmax_twostage_fp32"; }
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
// ---- Reduction axis-reduce variant (2.5.1+, axis >= 32) ----
// SUM_REDUCE_AXIS / MEAN_REDUCE_AXIS: blockReduceSum collapses the axis dim.
// Different kernel signature (per_block_size, calc_multi_num) and dispatch
// geometry (block=256|64, grid=count) vs naive → independent variant.
class CudaReductionSumAxisFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_sum_axis_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionMeanAxisFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_mean_axis_fp32"; }
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
class CudaInterpBilinearOptFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "interp_bilinear_opt"; }
    const char* variant() const override { return "cuda_interp_bilinear_opt_fp32"; }
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

// ---- RoPE (3.6.0) ----
// ropeC4Kernel: rotary position embedding, C4 packed. Simplified replay variant
// (no QNorm/KNorm path — gamma=nullptr, useNorm=false).
class CudaRopeC4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "rope"; }
    const char* variant() const override { return "cuda_rope_c4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- MatMul variants (3.6.0) ----
// GENERAL_BATCH_MATMUL: naive batched matmul for large-batch small-problem.
// matmul_gemv_kernel: GEMV for M=1 (decode stage optimization).
// PackPadFill is CUTLASS preprocessing — not included (cannot validate
// independently without the CUTLASS GEMM that follows).
class CudaGeneralBatchMatmulFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "matmul"; }
    const char* variant() const override { return "cuda_general_batch_matmul_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaMatmulGemvFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "matmul"; }
    const char* variant() const override { return "cuda_matmul_gemv_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- LayerNorm C4 variants (3.6.0) ----
// layernorm_c4: C4-packed, block-per-row with blockReduceSum.
// binary_layernorm_c4: fuses binary add + layernorm C4.
// Both differ from the naive LAYERNORM kernel (grid-stride loop) in dispatch
// geometry (one block per row) and reduction strategy (blockReduceSum vs serial).
class CudaLayerNormC4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_layernorm_c4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryLayerNormC4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_binary_layernorm_c4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// input_layernorm_<size>: size-specialized LayerNorm variants (3.6.0).
// Math equivalent to LAYERNORM but with unrolled per-size kernels.
// n is fixed per variant (320/512/1024/2048); adaptive handles arbitrary n.
class CudaInputLayerNorm320Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_input_layernorm_320_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInputLayerNorm512Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_input_layernorm_512_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInputLayerNorm1024Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_input_layernorm_1024_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInputLayerNorm2048Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_input_layernorm_2048_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInputLayerNormAdaptiveFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_input_layernorm_adaptive_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- ConvBase variants (3.6.0) ----
// Float22Half2: float→half2 packing (4 elements per thread).
// Float22BFloat16: float→bfloat16 packing (sm80+).
// Im2Col_FilterC: im2col for convolution preprocessing.
// WeightPackFill: weight reordering [Co,Ci,KhKw]→[Co,KhKw,Ci].
// These are independent kernels with distinct signatures → independent variants.
class CudaFloat22Half2Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cuda_float22half2_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaIm2ColFilterCFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "im2col"; }
    const char* variant() const override { return "cuda_im2col_filterc_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaWeightPackFillFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "weight_pack_fill"; }
    const char* variant() const override { return "cuda_weight_pack_fill_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ---- Transpose variants (3.6.0) ----
// TRANSPOSE: generic transpose using TransposeParam struct.
// PACKCOMMON/UNPACKCOMMON: pack/unpack C4 channel layout.
// blit_2_float: vec2 blit for raster.
// All are independent kernels with distinct signatures → independent variants.
class CudaTransposeFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_transpose_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaPackCommonFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_packcommon_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaUnpackCommonFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_unpackcommon_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBlit2FloatFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_blit_2_float_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFuseBlitFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_fuseblit_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFuseBlitLimitFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_fuseblit_limit_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- Cast PACK variants (2.5.3+) ----
// FLOAT_2_INT8_CAST_PACK: packed quantization (float→int8 with scale/zeroPoint/clamp).
// INT8_2_FLOAT_CAST_PACK: packed dequantization (int8→float with scale/zeroPoint).
class CudaFloat2Int8CastPackFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cuda_float2int8_cast_pack_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInt82FloatCastPackFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cuda_int82float_cast_pack_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- MultiInputDW preprocessing kernels (3.6.0) ----
// WeightPrepare: pack weight [C,Kh,Kw] → [C_p,Kh,Kw] with zero padding.
// BiasPrepare: pack bias [C] → [C_p] with zero padding.
// BiasZeroPrepare: zero-fill bias buffer [C_p].
class CudaWeightPrepareFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_weight_prepare_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBiasPrepareFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_bias_prepare_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBiasZeroPrepareFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_bias_zero_prepare_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- Deconv variants (3.6.0) ----
class CudaDeconvKernelReorderFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "deconv"; }
    const char* variant() const override { return "cuda_deconv_kernel_reorder_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCol2ImFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "deconv"; }
    const char* variant() const override { return "cuda_col2im_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCol2ImVec4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "deconv"; }
    const char* variant() const override { return "cuda_col2im_vec4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- PackPadFill / WeightPackFill_Implicit / transpose_BDL_to_BLD (3.6.0) ----
class CudaPackPadFillFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "matmul"; }
    const char* variant() const override { return "cuda_pack_pad_fill_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaWeightPackFillImplicitFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "weight_pack_fill"; }
    const char* variant() const override { return "cuda_weight_pack_fill_implicit_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaTransposeBdlToBldFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_transpose_bdl_to_bld_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// ---- PACKCOMMON_4 / UNPACKCOMMON_4 / blit_2_half (3.6.0) ----
class CudaPackCommon4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_packcommon_4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaUnpackCommon4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_unpackcommon_4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBlit2HalfFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_blit_2_half_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// A-class: CONV_DW fp16/half2 variants (5 kernels, CudaOpsFp16.cu)
// ============================================================================
class CudaConvDwOptFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_conv_dw_opt_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaConvDwHalf2OptFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_conv_dw_half2_opt_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaConvDw3x3Half2OptFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_conv_dw3x3_half2_opt_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaConvDwMultiWidth4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_conv_dw_multi_width4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaConvDwMultiWidthChannelFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw"; }
    const char* variant() const override { return "cuda_conv_dw_multi_width_channel_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// A-class: half-precision transpose/pack/fuseblit variants (5 kernels)
// ============================================================================
class CudaPackCommonHalf4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_packcommon_half_4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaPackCommonRearrangeHalf4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_packcommon_rearrange_half_4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaUnpackCommonRearrangeHalf4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "transpose"; }
    const char* variant() const override { return "cuda_unpackcommon_rearrange_half_4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFuseBlitHalf4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_fuseblit_half_4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaUnaryHalf2SigmoidFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "cuda_unary_half2_sigmoid_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// B-class: Winograd convolution variants (6 kernels)
// ============================================================================
class CudaWinoWeightReorderFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "convolution"; }
    const char* variant() const override { return "cuda_wino_weight_reorder_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaWinoInputTransFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "convolution"; }
    const char* variant() const override { return "cuda_wino_input_trans_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaWinoInputTransHalf2Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "convolution"; }
    const char* variant() const override { return "cuda_wino_input_trans_half2_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaWinoTrans2OutputFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "convolution"; }
    const char* variant() const override { return "cuda_wino_trans2output_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaWinoTrans2OutputHalf2Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "convolution"; }
    const char* variant() const override { return "cuda_wino_trans2output_half2_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaIm2ColFilterCVec4Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "convolution"; }
    const char* variant() const override { return "cuda_im2col_filterc_vec4_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// C-class: Attention/LinearAttention (12 kernels)
// ============================================================================
class CudaFlashDecodeFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_flash_decode_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFlashDecodeWithMaskFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_flash_decode_with_mask_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFlashDecodeSplitkFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_flash_decode_splitk_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFlashAttnCombineResultsFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_flash_attn_combine_results_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCompactKvCacheFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_compact_kv_cache_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCopyKvToCacheFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_copy_kv_to_cache_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaQkKernelTiledFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_qk_kernel_tiled_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaQkvKernelTiledFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_qkv_kernel_tiled_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaConv1dSiluFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "linear_attention"; }
    const char* variant() const override { return "cuda_conv1d_silu_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaShortConvFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "linear_attention"; }
    const char* variant() const override { return "cuda_short_conv_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaShortConvOutputFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "linear_attention"; }
    const char* variant() const override { return "cuda_short_conv_output_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGatedDeltaRuleDecodeFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "linear_attention"; }
    const char* variant() const override { return "cuda_gated_delta_rule_decode_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// P0 fp16 variants — Attention / LinearAttention / RoPE / TopKV2
// Mirror fp32 logic; buffers use __half packing for T-typed I/O.
// ============================================================================
class CudaFlashDecodeFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_flash_decode_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFlashDecodeWithMaskFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_flash_decode_with_mask_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFlashDecodeSplitkFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_flash_decode_splitk_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFlashAttnCombineResultsFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_flash_attn_combine_results_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaCopyKvToCacheFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_copy_kv_to_cache_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaQkKernelTiledFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_qk_kernel_tiled_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaQkvKernelTiledFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "cuda_qkv_kernel_tiled_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaConv1dSiluFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "linear_attention"; }
    const char* variant() const override { return "cuda_conv1d_silu_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaShortConvFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "linear_attention"; }
    const char* variant() const override { return "cuda_short_conv_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaShortConvOutputFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "linear_attention"; }
    const char* variant() const override { return "cuda_short_conv_output_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGatedDeltaRuleDecodeFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "linear_attention"; }
    const char* variant() const override { return "cuda_gated_delta_rule_decode_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaRopeC4Fp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "rope"; }
    const char* variant() const override { return "cuda_rope_c4_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaTopKV2Fp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "topkv2"; }
    const char* variant() const override { return "cuda_topkv2_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// P1 fp16 variants — Reduction(7) / Interp(4) / GridSample(4) / LayerNormC4(2)
// Only tag 3.6.0 (fp16 support introduced in 3.6.0).
// ============================================================================
class CudaReductionSumFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_sum_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionMeanFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_mean_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionMaxFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_max_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionMinFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_min_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionProdFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_prod_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionSumAxisFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_sum_axis_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaReductionMeanAxisFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "cuda_reduction_mean_axis_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInterpNearestFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "interp_nearest"; }
    const char* variant() const override { return "cuda_interp_nearest_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInterpBilinearFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "interp_bilinear"; }
    const char* variant() const override { return "cuda_interp_bilinear_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInterpNearestRoundFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "interp_nearest_round"; }
    const char* variant() const override { return "cuda_interp_nearest_round_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInterpBilinearOptFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "interp_bilinear_opt"; }
    const char* variant() const override { return "cuda_interp_bilinear_opt_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGridSampleNearestFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "grid_sample"; }
    const char* variant() const override { return "cuda_grid_sample_nearest_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGridSampleBilinearFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "grid_sample_bilinear"; }
    const char* variant() const override { return "cuda_grid_sample_bilinear_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGridSampleNearest3dFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "grid_sample_3d"; }
    const char* variant() const override { return "cuda_grid_sample_nearest_3d_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGridSampleBilinear3dFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "grid_sample_3d"; }
    const char* variant() const override { return "cuda_grid_sample_bilinear_3d_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaLayerNormC4Fp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_layernorm_c4_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryLayerNormC4Fp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "cuda_binary_layernorm_c4_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// P2 plugin kernels — GroupNorm(half) / SeqLen2Spatial / splitGeLU / SPLIT_FusedQKV
// ============================================================================
class CudaGroupNormNHWCSumFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "groupnorm"; }
    const char* variant() const override { return "cuda_groupnorm_nhwc_sum_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaGroupNormNHWCScaleFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "groupnorm"; }
    const char* variant() const override { return "cuda_groupnorm_nhwc_scale_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSeqLen2SpatialFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "seqlen2spatial"; }
    const char* variant() const override { return "cuda_seqlen2spatial_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSeqLen2SpatialFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "seqlen2spatial"; }
    const char* variant() const override { return "cuda_seqlen2spatial_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSplitGeluFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "splitgelu"; }
    const char* variant() const override { return "cuda_splitgelu_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSplitGeluFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "splitgelu"; }
    const char* variant() const override { return "cuda_splitgelu_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSplitFusedQKVFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "split_fusedqkv"; }
    const char* variant() const override { return "cuda_split_fusedqkv_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSplitFusedQKVFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "split_fusedqkv"; }
    const char* variant() const override { return "cuda_split_fusedqkv_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
// P4 扩展：SPLIT_FusedKV (2-way split)
class CudaSplitFusedKVFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "split_fusedkv"; }
    const char* variant() const override { return "cuda_split_fusedkv_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaSplitFusedKVFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "split_fusedkv"; }
    const char* variant() const override { return "cuda_split_fusedkv_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// P4 int8 kernels — FloatToInt8/Int8ToFloat/DequantWeight/ConvDW/Im2Col/BinaryInt8
// ============================================================================
class CudaFloat2Int8Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "float2int8"; }
    const char* variant() const override { return "cuda_float2int8_packed_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaFloat2Int8SingleFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "float2int8_single"; }
    const char* variant() const override { return "cuda_float2int8_single_packed_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInt82FloatFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "int82float"; }
    const char* variant() const override { return "cuda_int82float_packed_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaInt82FloatSingleFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "int82float_single"; }
    const char* variant() const override { return "cuda_int82float_single_packed_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaDequantizeInt8WeightFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "dequantize_int8_weight"; }
    const char* variant() const override { return "cuda_dequantize_int8_weight_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaDequantizeInt4WeightFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "dequantize_int4_weight"; }
    const char* variant() const override { return "cuda_dequantize_int4_weight_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaConvDwInt8Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw_int8"; }
    const char* variant() const override { return "cuda_conv_dw_int8_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaConvDw3x3S1Int8Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "conv_dw3x3s1_int8"; }
    const char* variant() const override { return "cuda_conv_dw3x3s1_int8_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaIm2ColPackC16Int8Fp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "im2col_packc16_int8"; }
    const char* variant() const override { return "cuda_im2col_packc16_int8_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaWeightInt8PackFillFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "weight_int8_pack_fill"; }
    const char* variant() const override { return "cuda_weight_int8_pack_fill_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryInt8AddFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "binary_int8_add"; }
    const char* variant() const override { return "cuda_binary_int8_add_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryInt8MulFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "binary_int8_mul"; }
    const char* variant() const override { return "cuda_binary_int8_mul_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// P3 扩展：BINARY_INT8 其余操作
#define DECL_BINARY_INT8_ADAPTER(ClassName, OpTypeName, VariantName) \
class ClassName : public CudaOpAdapter { \
public: \
    const char* opType() const override { return OpTypeName; } \
    const char* variant() const override { return VariantName; } \
    bool adapt(const CaseSpec&, AdaptedCase&) const override; \
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override; \
    bool validate(const AdaptedCase&, const std::vector<float>&) const override; \
};

DECL_BINARY_INT8_ADAPTER(CudaBinaryInt8SubFp32Kernel, "binary_int8_sub", "cuda_binary_int8_sub_fp32")
DECL_BINARY_INT8_ADAPTER(CudaBinaryInt8DivFp32Kernel, "binary_int8_div", "cuda_binary_int8_div_fp32")
DECL_BINARY_INT8_ADAPTER(CudaBinaryInt8MinimumFp32Kernel, "binary_int8_minimum", "cuda_binary_int8_minimum_fp32")
DECL_BINARY_INT8_ADAPTER(CudaBinaryInt8MaximumFp32Kernel, "binary_int8_maximum", "cuda_binary_int8_maximum_fp32")

// P3 扩展：BINARY_INT8_CHANNELWISE (per-channel scale)
DECL_BINARY_INT8_ADAPTER(CudaBinaryInt8ChannelwiseAddFp32Kernel, "binary_int8_channelwise_add", "cuda_binary_int8_channelwise_add_fp32")
DECL_BINARY_INT8_ADAPTER(CudaBinaryInt8ChannelwiseMulFp32Kernel, "binary_int8_channelwise_mul", "cuda_binary_int8_channelwise_mul_fp32")

// ============================================================================
// P5 Raster fused binary — BinaryADD/MUL + FuseAddADD/MUL + MidADD/MUL + MidLinear4ADD/MUL
// ============================================================================
class CudaBinaryAddRasterFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binary"; }
    const char* variant() const override { return "cuda_raster_binary_add_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryMulRasterFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binary"; }
    const char* variant() const override { return "cuda_raster_binary_mul_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryFuseAddAddFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binary_fuseadd"; }
    const char* variant() const override { return "cuda_raster_fuseadd_add_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryFuseAddMulFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binary_fuseadd"; }
    const char* variant() const override { return "cuda_raster_fuseadd_mul_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryMidAddFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymid"; }
    const char* variant() const override { return "cuda_raster_binarymid_add_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryMidMulFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymid"; }
    const char* variant() const override { return "cuda_raster_binarymid_mul_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryMidLinear4AddFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymidlinear4"; }
    const char* variant() const override { return "cuda_raster_binarymidlinear4_add_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryMidLinear4MulFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymidlinear4"; }
    const char* variant() const override { return "cuda_raster_binarymidlinear4_mul_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// ============================================================================
// P5 扩展：Raster 融合宏实例批量声明（SUB/DIV/MIN/MAX/FLOORDIV/FLOORMOD/...）
// 用宏一次性声明 adapter 类（每个 OP 一个类）
// ============================================================================

#define DECL_RASTER_BINARY_ADAPTER(ClassName, VariantName) \
class ClassName : public CudaOpAdapter { \
public: \
    const char* opType() const override { return "raster_binary"; } \
    const char* variant() const override { return VariantName; } \
    bool adapt(const CaseSpec&, AdaptedCase&) const override; \
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override; \
    bool validate(const AdaptedCase&, const std::vector<float>&) const override; \
};

#define DECL_RASTER_FUSEADD_ADAPTER(ClassName, VariantName) \
class ClassName : public CudaOpAdapter { \
public: \
    const char* opType() const override { return "raster_binary_fuseadd"; } \
    const char* variant() const override { return VariantName; } \
    bool adapt(const CaseSpec&, AdaptedCase&) const override; \
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override; \
    bool validate(const AdaptedCase&, const std::vector<float>&) const override; \
};

#define DECL_RASTER_BINARYMID_ADAPTER(ClassName, VariantName) \
class ClassName : public CudaOpAdapter { \
public: \
    const char* opType() const override { return "raster_binarymid"; } \
    const char* variant() const override { return VariantName; } \
    bool adapt(const CaseSpec&, AdaptedCase&) const override; \
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override; \
    bool validate(const AdaptedCase&, const std::vector<float>&) const override; \
};

#define DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(ClassName, VariantName) \
class ClassName : public CudaOpAdapter { \
public: \
    const char* opType() const override { return "raster_binarymidlinear4"; } \
    const char* variant() const override { return VariantName; } \
    bool adapt(const CaseSpec&, AdaptedCase&) const override; \
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override; \
    bool validate(const AdaptedCase&, const std::vector<float>&) const override; \
};

// raster_binary 扩展 (SUB/DIV/MIN/MAX/FLOORDIV/FLOORMOD/SQUAREDDIFF/POW)
DECL_RASTER_BINARY_ADAPTER(CudaBinarySubRasterFp32Kernel, "cuda_raster_binary_sub_fp32")
DECL_RASTER_BINARY_ADAPTER(CudaBinaryDivRasterFp32Kernel, "cuda_raster_binary_div_fp32")
DECL_RASTER_BINARY_ADAPTER(CudaBinaryMinimumRasterFp32Kernel, "cuda_raster_binary_minimum_fp32")
DECL_RASTER_BINARY_ADAPTER(CudaBinaryMaximumRasterFp32Kernel, "cuda_raster_binary_maximum_fp32")
DECL_RASTER_BINARY_ADAPTER(CudaBinaryFloordivRasterFp32Kernel, "cuda_raster_binary_floordiv_fp32")
DECL_RASTER_BINARY_ADAPTER(CudaBinaryFloormodRasterFp32Kernel, "cuda_raster_binary_floormod_fp32")
DECL_RASTER_BINARY_ADAPTER(CudaBinarySquaredDifferenceRasterFp32Kernel, "cuda_raster_binary_squared_difference_fp32")
DECL_RASTER_BINARY_ADAPTER(CudaBinaryPowRasterFp32Kernel, "cuda_raster_binary_pow_fp32")

// raster_binary_fuseadd 扩展
DECL_RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddSubFp32Kernel, "cuda_raster_fuseadd_sub_fp32")
DECL_RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddDivFp32Kernel, "cuda_raster_fuseadd_div_fp32")
DECL_RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddMinimumFp32Kernel, "cuda_raster_fuseadd_minimum_fp32")
DECL_RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddMaximumFp32Kernel, "cuda_raster_fuseadd_maximum_fp32")
DECL_RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddFloordivFp32Kernel, "cuda_raster_fuseadd_floordiv_fp32")
DECL_RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddFloormodFp32Kernel, "cuda_raster_fuseadd_floormod_fp32")
DECL_RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddSquaredDifferenceFp32Kernel, "cuda_raster_fuseadd_squared_difference_fp32")
DECL_RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddPowFp32Kernel, "cuda_raster_fuseadd_pow_fp32")

// raster_binarymid 扩展
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidSubFp32Kernel, "cuda_raster_binarymid_sub_fp32")
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidMulSiluFp32Kernel, "cuda_raster_binarymid_mul_silu_fp32")
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidDivFp32Kernel, "cuda_raster_binarymid_div_fp32")
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidMinimumFp32Kernel, "cuda_raster_binarymid_minimum_fp32")
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidMaximumFp32Kernel, "cuda_raster_binarymid_maximum_fp32")
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidFloordivFp32Kernel, "cuda_raster_binarymid_floordiv_fp32")
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidFloormodFp32Kernel, "cuda_raster_binarymid_floormod_fp32")
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidSquaredDifferenceFp32Kernel, "cuda_raster_binarymid_squared_difference_fp32")
DECL_RASTER_BINARYMID_ADAPTER(CudaBinaryMidPowFp32Kernel, "cuda_raster_binarymid_pow_fp32")

// raster_binarymidlinear4 扩展
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4SubFp32Kernel, "cuda_raster_binarymidlinear4_sub_fp32")
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4MulSiluFp32Kernel, "cuda_raster_binarymidlinear4_mul_silu_fp32")
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4DivFp32Kernel, "cuda_raster_binarymidlinear4_div_fp32")
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4MinimumFp32Kernel, "cuda_raster_binarymidlinear4_minimum_fp32")
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4MaximumFp32Kernel, "cuda_raster_binarymidlinear4_maximum_fp32")
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4FloordivFp32Kernel, "cuda_raster_binarymidlinear4_floordiv_fp32")
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4FloormodFp32Kernel, "cuda_raster_binarymidlinear4_floormod_fp32")
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4SquaredDifferenceFp32Kernel, "cuda_raster_binarymidlinear4_squared_difference_fp32")
DECL_RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4PowFp32Kernel, "cuda_raster_binarymidlinear4_pow_fp32")

// ============================================================================
// P2 扩展：BinaryMid4 / BinaryMidHalf2 / BinaryMidLinearHalf4 (PACK_NUMBER 向量化变体)
// ============================================================================

// BinaryMid4: float4 vectorized, stride 不含 X 维度（PACK_NUMBER=4）
class CudaBinaryMid4AddFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymid4"; }
    const char* variant() const override { return "cuda_raster_binarymid4_add_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryMid4MulFp32Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymid4"; }
    const char* variant() const override { return "cuda_raster_binarymid4_mul_fp32"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// BinaryMidHalf2: fp16 half2 vectorized (PACK_NUMBER=2)
class CudaBinaryMidHalf2AddFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymidhalf2"; }
    const char* variant() const override { return "cuda_raster_binarymidhalf2_add_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryMidHalf2MulFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymidhalf2"; }
    const char* variant() const override { return "cuda_raster_binarymidhalf2_mul_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

// BinaryMidLinearHalf4: fp16 half2 x2, 1D linear
class CudaBinaryMidLinearHalf4AddFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymidlinearhalf4"; }
    const char* variant() const override { return "cuda_raster_binarymidlinearhalf4_add_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};
class CudaBinaryMidLinearHalf4MulFp16Kernel : public CudaOpAdapter {
public:
    const char* opType() const override { return "raster_binarymidlinearhalf4"; }
    const char* variant() const override { return "cuda_raster_binarymidlinearhalf4_mul_fp16"; }
    bool adapt(const CaseSpec&, AdaptedCase&) const override;
    cudaError_t launch(const AdaptedCase&, const CudaLaunchCtx&) const override;
    bool validate(const AdaptedCase&, const std::vector<float>&) const override;
};

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CUDA_OPS_HPP
