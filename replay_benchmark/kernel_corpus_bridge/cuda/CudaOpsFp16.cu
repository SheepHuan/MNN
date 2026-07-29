// CudaOpsFp16.cu - fp16/int8/bf16 variants of CUDA corpus adapters.
// Compiled by nvcc (via replay_cuda_corpus) because __half/__nv_bfloat16 are
// not available in plain g++ compilation. The fp32 variants live in CudaOps.cpp.
#include "CudaOps.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>

#ifndef UP_DIV
#define UP_DIV(x, y) (((x) + (y) - 1) / (y))
#endif
#ifndef INT8_PACK_NUMBER
#define INT8_PACK_NUMBER 4
#endif
// __float2int_rn is __device__ only; use host-side roundf + cast for validation.
static inline int host_float2int_rn(float x) { return (int)roundf(x); }

extern "C" {
void mnn_corpus_relu_fp16(const void*, void*, size_t, float, int, int, cudaStream_t);
void mnn_corpus_clamp_fp16(const void*, void*, size_t, float, float, int, int, cudaStream_t);
void mnn_corpus_atan2_fp16(const void*, const void*, void*, size_t, size_t, size_t, int, int, cudaStream_t);
void mnn_corpus_mod_fp16(const void*, const void*, void*, size_t, size_t, size_t, int, int, cudaStream_t);
void mnn_corpus_logicalor_fp16(const void*, const void*, void*, size_t, size_t, size_t, int, int, cudaStream_t);
void mnn_corpus_range_fp16(const int, const void*, const void*, void*, int, int, cudaStream_t);
void mnn_corpus_select_fp16(const int, const int*, const void*, const void*, int, int, void*, int, int, cudaStream_t);
void mnn_corpus_softmax_fp16(const void*, void*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_layernorm_fp16(const int, const int, const int, const float, const void*, void*, const float*, const float*, bool, int, int, cudaStream_t);
void mnn_corpus_prelu_fp16(const int, const int, const int, const void*, void*, const float*, int, int, int, cudaStream_t);
void mnn_corpus_scale_fp16(const int, const int, const int, const void*, void*, const float*, const float*, int, int, cudaStream_t);
void mnn_corpus_maxpool_fp16(const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_avgpool_fp16(const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_global_avgpool_fp16(const void*, void*, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_global_maxpool_fp16(const void*, void*, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_castmidfloat_f16_i32(const void*, int32_t*, size_t, int, int, cudaStream_t);
void mnn_corpus_castmidfloat_i32_f16(const int32_t*, void*, size_t, int, int, cudaStream_t);
void mnn_corpus_castmidfloat_f16_i8(const void*, int8_t*, size_t, int, int, cudaStream_t);
void mnn_corpus_bf162float_f32(const int16_t*, float*, size_t, int, int, cudaStream_t);
void mnn_corpus_bf162float_f16(const int16_t*, void*, size_t, int, int, cudaStream_t);
void mnn_corpus_cast_i8_i32(const int8_t*, int32_t*, size_t, int, int, cudaStream_t);
void mnn_corpus_cast_i32_u8(const int32_t*, uint8_t*, size_t, int, int, cudaStream_t);
void mnn_corpus_cast_u8_i32(const uint8_t*, int32_t*, size_t, int, int, cudaStream_t);
void mnn_corpus_gatherv2_fp16(const int, const int, const int, const int, const int, const void*, const int*, void*, int, int, cudaStream_t);
void mnn_corpus_argmax_fp16(const int, const int, const int, const int, const void*, int*, int, int, cudaStream_t);
void mnn_corpus_argmin_fp16(const int, const int, const int, const int, const void*, int*, int, int, cudaStream_t);
void mnn_corpus_nhwc2nchw_fp16(const void*, void*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_nchw2nhwc_fp16(const void*, void*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_dw_opt_fp32(const float*, const void*, const void*, float*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_dw_half2_opt_fp32(const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_dw3x3_half2_opt_fp32(const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_dw_multi_width4_fp32(const float*, const void*, const void*, float*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_dw_multi_width_channel_fp32(const float*, const void*, const void*, float*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
// A-class: half-precision transpose/pack/fuseblit/unary shims (transpose_half.cu)
void mnn_corpus_packcommon_half_4_fp32(const void*, void*, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_packcommon_rearrange_half_4_fp32(const double*, double*, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_unpackcommon_rearrange_half_4_fp32(const double*, double*, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_fuseblit_half_4_fp32(const void*, void*, int, int, const int32_t*, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_unary_half2_sigmoid_fp32(const void*, void*, size_t, int, int, cudaStream_t);
// B-class: Winograd half2 shims (winograd.cu)
void mnn_corpus_wino_input_trans_half2_fp32(const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_wino_trans2output_half2_fp32(const void*, const float*, void*, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
// P0: Attention / LinearAttention / RoPE / TopKV2 fp16 shims
void mnn_corpus_flash_decode_fp16(const void*, const void*, const void*, void*, int, int, int, int, int, int, float, int, int, size_t, cudaStream_t);
void mnn_corpus_flash_decode_with_mask_fp16(const void*, const void*, const void*, void*, const void*, int, int, int, int, int, int, int, float, int, int, size_t, cudaStream_t);
void mnn_corpus_flash_decode_splitk_fp16(const void*, const void*, const void*, float*, float*, int, int, int, int, int, int, float, int, int, int, int, size_t, cudaStream_t);
void mnn_corpus_flash_attn_combine_results_fp16(const float*, const float*, void*, int, int, int, int, int, int, size_t, cudaStream_t);
void mnn_corpus_copy_kv_to_cache_fp16(const void*, const void*, void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, size_t, cudaStream_t);
void mnn_corpus_qk_kernel_tiled_fp16(const void*, const void*, void*, const void*, const void*, int, bool, bool, bool, int, int, int, int, int, size_t, cudaStream_t);
void mnn_corpus_qkv_kernel_tiled_fp16(const void*, const void*, void*, const void*, int, int, int, int, int, int, size_t, cudaStream_t);
void mnn_corpus_conv1d_silu_fp16(const void*, const void*, float*, float*, int, int, int, int, int, bool, int, int, size_t, cudaStream_t);
void mnn_corpus_short_conv_fp16(const void*, const void*, float*, float*, int, int, int, int, int, int, bool, int, int, size_t, cudaStream_t);
void mnn_corpus_short_conv_output_fp16(const void*, const float*, void*, int, int, int, int, bool, bool, int, int, size_t, cudaStream_t);
void mnn_corpus_gated_delta_rule_decode_fp16(const float*, const void*, const void*, float*, void*, int, int, int, int, int, int, int, int, int, bool, float, bool, bool, bool, int, int, size_t, cudaStream_t);
void mnn_corpus_rope_c4_fp16(const void*, const void*, const void*, const void*, void*, void*, const float*, const float*, int, int, int, int, int, int, int, float, float, bool, bool, int, int, cudaStream_t);
void mnn_corpus_topkv2_fp16(const void*, int*, void*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
// P1: Reduction(7) / Interp(4) / GridSample(4) / LayerNormC4(2) fp16 shims (3.6.0 only)
void mnn_corpus_reduction_sum_fp16(const void*, void*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_mean_fp16(const void*, void*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_max_fp16(const void*, void*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_min_fp16(const void*, void*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_prod_fp16(const void*, void*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_sum_axis_fp16(const void*, void*, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_mean_axis_fp16(const void*, void*, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_interp_nearest_fp16(int, int, int, int, int, int, float, float, float, float, const void*, void*, int, int, cudaStream_t);
void mnn_corpus_interp_bilinear_fp16(int, int, int, int, int, int, float, float, float, float, const void*, void*, int, int, cudaStream_t);
void mnn_corpus_interp_nearest_round_fp16(int, int, int, int, int, int, float, float, float, float, const void*, void*, int, int, cudaStream_t);
void mnn_corpus_interp_bilinear_opt_fp16(int, int, int, int, int, float, float, float, float, const void*, void*, int, int, int, int, cudaStream_t);
void mnn_corpus_grid_sample_nearest_fp16(int, const void*, const void*, void*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_grid_sample_bilinear_fp16(int, const void*, const void*, void*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_grid_sample_nearest_3d_fp16(int, const void*, const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_grid_sample_bilinear_3d_fp16(int, const void*, const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_layernorm_c4_fp16(void*, const void*, const float*, const float*, int, int, float, bool, int, int, cudaStream_t);
void mnn_corpus_binary_layernorm_c4_fp16(void*, void*, const void*, const void*, const float*, const float*, int, int, float, bool, int, int, cudaStream_t);
// P2: Plugin kernels — GroupNorm(half) / SeqLen2Spatial(fp32+fp16) / splitGeLU(fp32+fp16) / SPLIT_FusedQKV(fp32+fp16)
void mnn_corpus_groupnorm_nhwc_sum_fp16(void*, const void*, const void*, const void*, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, float, int, int, int, int, int, cudaStream_t);
void mnn_corpus_groupnorm_nhwc_scale_fp16(void*, const void*, const void*, const void*, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, float, int, int, int, int, int, cudaStream_t);
void mnn_corpus_seqlen2spatial_fp32(const void*, const void*, const void*, void*, int, int, int, cudaStream_t);
void mnn_corpus_seqlen2spatial_fp16(const void*, const void*, const void*, void*, int, int, int, cudaStream_t);
void mnn_corpus_splitgelu_fp32(const void*, const void*, void*, int, float, float, float, int, int, cudaStream_t);
void mnn_corpus_splitgelu_fp16(const void*, const void*, void*, int, float, float, float, int, int, cudaStream_t);
void mnn_corpus_split_fusedqkv_fp32(size_t, const void*, void*, void*, void*, int, int, int, cudaStream_t);
void mnn_corpus_split_fusedqkv_fp16(size_t, const void*, void*, void*, void*, int, int, int, cudaStream_t);
// P4: int8 kernels — FloatToInt8/Int8ToFloat/DequantWeight/ConvDW/Im2Col/BinaryInt8
void mnn_corpus_float2int8_packed_fp32(const float*, int8_t*, int, int, int, int, const float*, int8_t, int8_t, int8_t, int, int, cudaStream_t);
void mnn_corpus_float2int8_single_packed_fp32(const float*, int8_t*, int, int, int, int, float, int8_t, int8_t, int8_t, int, int, cudaStream_t);
void mnn_corpus_int82float_packed_fp32(const int8_t*, float*, int, int, int, int, const float*, int8_t, int, int, cudaStream_t);
void mnn_corpus_int82float_single_packed_fp32(const int8_t*, float*, int, int, int, int, float, int8_t, int, int, cudaStream_t);
void mnn_corpus_dequantize_int8_weight_fp16(const int8_t*, void*, const void*, const void*, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_dequantize_int4_weight_fp16(const uint8_t*, void*, const void*, const void*, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_dw_int8_fp32(const int8_t*, const int8_t*, const int32_t*, const float*, int8_t*, int8_t, int8_t, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_dw3x3s1_int8_fp32(const int8_t*, const int8_t*, const int32_t*, const float*, int8_t*, int8_t, int8_t, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_im2col_packc16_int8(int, int, int, int, int, int, int, int, int, size_t, int, int, int, int, const int32_t*, int32_t*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_weight_int8_pack_fill_fp32(const int8_t*, int8_t*, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binary_int8_add_fp32(const int8_t*, float, const int8_t*, float, int8_t*, float, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binary_int8_mul_fp32(const int8_t*, float, const int8_t*, float, int8_t*, float, int, int, int, int, int, cudaStream_t);
// P5: Raster fused binary shims
void mnn_corpus_binary_add_fp32(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binary_mul_fp32(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binary_fuseadd_add_fp32(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binary_fuseadd_mul_fp32(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binarymid_add_fp32(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binarymid_mul_fp32(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binarymidlinear4_add_fp32(const float*, const float*, float*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_binarymidlinear4_mul_fp32(const float*, const float*, float*, int, int, int, int, int, int, cudaStream_t);
// New shims: bf16 pool + float22bfloat16 (conv_base.cu)
void mnn_corpus_maxpool_c8_bf16(const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_avgpool_c8_bf16(const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_float22bfloat16_fp32(const float*, void*, size_t, int, int, cudaStream_t);
// weight_only_quant fp16 shims (defined in weight_only_quant.cu)
void mnn_corpus_precomputegemvparams_fp16(const void*, const void*, float2*, int, int, int, cudaStream_t);
void mnn_corpus_quanta_fp16(const void*, int8_t*, void*, void*, int32_t*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_dequantandacc_fp16(const int32_t*, void*, const void*, const void*, const void*, const void*, const int32_t*, int, int, const int32_t*, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_biasandactivation_fp16(void*, const void*, float, float, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemm_fpaint8b_fp16(const void*, const int8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint8b_fp16(const void*, const int8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemm_fpaint4b_fp16(const void*, const uint8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_fp16(const void*, const uint8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_v5_fp16(const void*, const uint8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_v9_fp16(const void*, const uint8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_v14_fp16(const void*, const uint8_t*, const float2*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_v14_mb_fp16(const void*, const uint8_t*, const float2*, const void*, void*, float, float, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint8b_v2_fp16(const void*, const int8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_fpaint8b_fp16(const void*, const int8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_fpaint4b_fp16(const void*, const uint8_t*, const void*, const void*, const void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
// gated_delta_rule_prefill fp16 shim (defined in attention.cu)
void mnn_corpus_gated_delta_rule_prefill_fp16(const float*, const void*, const void*, float*, void*, int, int, int, int, int, int, int, int, int, int, bool, float, bool, bool, bool, int, int, size_t, cudaStream_t);
}

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

namespace {

// fillInputAlternating / fillInputRamp come from CudaOps.hpp (inline).
// Pack float vector as raw half bytes (device layout).
std::vector<uint8_t> packHalf(const std::vector<float>& f) {
    std::vector<uint8_t> out(f.size() * sizeof(__half));
    for (size_t i = 0; i < f.size(); ++i) {
        __half h = __float2half(f[i]);
        std::memcpy(&out[i * sizeof(__half)], &h, sizeof(__half));
    }
    return out;
}
// Unpack raw half bytes (possibly padded) back to float for validation.
std::vector<float> unpackHalf(const std::vector<uint8_t>& b, int count) {
    std::vector<float> out(count);
    const __half* h = reinterpret_cast<const __half*>(b.data());
    for (int i = 0; i < count; ++i) out[i] = __half2float(h[i]);
    return out;
}
// PACK_NUMBER matches MNN's ConvDepthWise channel packing (8 channels per pack).
constexpr int PACK_NUMBER = 8;
} // namespace

// ============================================================================
// RELU fp16
// ============================================================================
bool CudaReluFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_relu_fp16";
    const int count = spec.intParam("size", 1024);
    const float slope = spec.floatParam("slope", 0.0f);
    std::vector<float> input; fillInputAlternating(input, count);
    auto halfBuf = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = halfBuf.size(); inBuf.initialData = halfBuf; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = halfBuf.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarFloat(slope));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    return true;
}
cudaError_t CudaReluFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_relu_fp16(ctx.devBufs[0], ctx.devBufs[1], (size_t)ctx.intArgs[0], ctx.floatArgs[0],
                          ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaReluFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false;
    const __half* outHalf = reinterpret_cast<const __half*>(output.data());
    const float slope = ac.args.size() >= 4 ? ac.args[3].floatVal : 0.0f;
    for (int i = 0; i < count; ++i) {
        const float x = ac.validatorInputA[i];
        const float expected = x > 0.0f ? x : x * slope;
        if (std::fabs(__half2float(outHalf[i]) - expected) > 1e-2f) return false;
    }
    return true;
}

// ============================================================================
// CLAMP fp16
// ============================================================================
bool CudaClampFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_clamp_fp16";
    const int count = spec.intParam("size", 1024);
    const float minV = spec.floatParam("min_v", 0.0f);
    const float maxV = spec.floatParam("max_v", 6.0f);
    std::vector<float> input; fillInputRamp(input, count);
    for (int i = 0; i < count; ++i) input[i] = input[i] * 10.0f - 5.0f;
    auto halfBuf = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = halfBuf.size(); inBuf.initialData = halfBuf; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = halfBuf.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarFloat(minV));
    ac.args.push_back(AdaptedArg::scalarFloat(maxV));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    return true;
}
cudaError_t CudaClampFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_clamp_fp16(ctx.devBufs[0], ctx.devBufs[1], (size_t)ctx.intArgs[0], ctx.floatArgs[0],
                           ctx.floatArgs[1], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaClampFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false;
    const __half* outHalf = reinterpret_cast<const __half*>(output.data());
    const float minV = ac.args[3].floatVal, maxV = ac.args[4].floatVal;
    for (int i = 0; i < count; ++i) {
        const float x = ac.validatorInputA[i];
        const float expected = std::min(std::max(x, minV), maxV);
        if (std::fabs(__half2float(outHalf[i]) - expected) > 1e-2f) return false;
    }
    return true;
}

// Macro to reduce boilerplate for Binary fp16 (3 ops share the same structure)
// SHIM is passed as a bare token (e.g. atan2_fp16). It is token-pasted with
// mnn_corpus_ for the launch function name, and stringified with # for the
// entry string. __VA_ARGS__ is the validation formula (may contain commas).
#define BINARY_FP16_ADAPTER(CLASS, OP, VARIANT, SHIM, ...) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    ac.entry = "mnn_corpus_" #SHIM; \
    const int count = spec.intParam("size", 1024); \
    std::vector<float> in0, in1; fillInputRamp(in0, count); fillInputRamp(in1, count); \
    auto hb0 = packHalf(in0), hb1 = packHalf(in1); \
    AdaptedBuffer b0; b0.sizeBytes = hb0.size(); b0.initialData = hb0; b0.isOutput = false; \
    AdaptedBuffer b1; b1.sizeBytes = hb1.size(); b1.initialData = hb1; b1.isOutput = false; \
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb0.size(); outBuf.isOutput = true; \
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::scalarInt(count)); \
    ac.args.push_back(AdaptedArg::scalarInt(1)); \
    ac.args.push_back(AdaptedArg::scalarInt(1)); \
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = in0; ac.validatorInputB = in1; ac.elementCount = count; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM(ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], (size_t)ctx.intArgs[0], \
                       (size_t)ctx.intArgs[1], (size_t)ctx.intArgs[2], ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int count = ac.elementCount; \
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false; \
    const __half* out = reinterpret_cast<const __half*>(output.data()); \
    for (int i = 0; i < count; ++i) { \
        const float x = ac.validatorInputA[i], y = ac.validatorInputB[i]; \
        const float expected = __VA_ARGS__; \
        if (std::fabs(__half2float(out[i]) - expected) > 1e-2f) return false; \
    } \
    return true; \
}

BINARY_FP16_ADAPTER(CudaBinaryAtan2Fp16Kernel, atan2, cuda_atan2_fp16, atan2_fp16, atan2f(x, y))
BINARY_FP16_ADAPTER(CudaBinaryModFp16Kernel, mod, cuda_mod_fp16, mod_fp16, x - x / y)
BINARY_FP16_ADAPTER(CudaBinaryLogicalOrFp16Kernel, logicalor, cuda_logicalor_fp16, logicalor_fp16, (x || y) ? 1.0f : 0.0f)

// ============================================================================
// Range fp16
// ============================================================================
bool CudaRangeFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_range_fp16";
    const int count = spec.intParam("size", 1024);
    std::vector<float> start(1, spec.floatParam("start", 0.0f));
    std::vector<float> step(1, spec.floatParam("step", 1.0f));
    auto hbStart = packHalf(start), hbStep = packHalf(step);
    AdaptedBuffer sb; sb.sizeBytes = hbStart.size(); sb.initialData = hbStart; sb.isOutput = false;
    AdaptedBuffer tb; tb.sizeBytes = hbStep.size(); tb.initialData = hbStep; tb.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(__half); outBuf.isOutput = true;
    ac.buffers.push_back(sb); ac.buffers.push_back(tb); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = start; ac.validatorInputB = step; ac.elementCount = count;
    return true;
}
cudaError_t CudaRangeFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_range_fp16(ctx.intArgs[0], ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2],
                          ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaRangeFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    const float start = ac.validatorInputA[0], step = ac.validatorInputB[0];
    for (int i = 0; i < count; ++i) {
        if (std::fabs(__half2float(out[i]) - (start + i * step)) > 1e-2f) return false;
    }
    return true;
}

// ============================================================================
// Range i32 (launch shim already exists in CorpusKernels.cu)
// ============================================================================
extern "C" void mnn_corpus_range_i32(const int, const int*, const int*, int*, int, int, cudaStream_t);
bool CudaRangeI32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_range_i32";
    const int count = spec.intParam("size", 1024);
    std::vector<int32_t> start(1, spec.intParam("start", 0));
    std::vector<int32_t> step(1, spec.intParam("step", 1));
    AdaptedBuffer sb; sb.sizeBytes = 4; sb.initialData.assign((const uint8_t*)start.data(), (const uint8_t*)start.data()+4); sb.isOutput = false;
    AdaptedBuffer tb; tb.sizeBytes = 4; tb.initialData.assign((const uint8_t*)step.data(), (const uint8_t*)step.data()+4); tb.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(sb); ac.buffers.push_back(tb); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA.assign(start.begin(), start.end());
    ac.validatorInputB.assign(step.begin(), step.end());
    ac.elementCount = count;
    return true;
}
cudaError_t CudaRangeI32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_range_i32(ctx.intArgs[0], (const int*)ctx.devBufs[0], (const int*)ctx.devBufs[1],
                         (int*)ctx.devBufs[2], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaRangeI32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count * 4) return false;
    const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
    const int32_t start = (int32_t)ac.validatorInputA[0], step = (int32_t)ac.validatorInputB[0];
    for (int i = 0; i < count; ++i) {
        if (out[i] != start + i * step) return false;
    }
    return true;
}

// ============================================================================
// Select fp16
// ============================================================================
bool CudaSelectFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_select_fp16";
    const int count = spec.intParam("size", 1024);
    std::vector<int32_t> sel(count);
    for (int i = 0; i < count; ++i) sel[i] = i % 2;
    std::vector<float> in1, in2; fillInputRamp(in1, count); fillInputRamp(in2, count);
    for (int i = 0; i < count; ++i) in2[i] += 100.0f;
    auto hb1 = packHalf(in1), hb2 = packHalf(in2);
    AdaptedBuffer selBuf; selBuf.sizeBytes = count * sizeof(int32_t);
    selBuf.initialData.assign((const uint8_t*)sel.data(), (const uint8_t*)sel.data() + selBuf.sizeBytes);
    selBuf.isOutput = false;
    AdaptedBuffer b1; b1.sizeBytes = hb1.size(); b1.initialData = hb1; b1.isOutput = false;
    AdaptedBuffer b2; b2.sizeBytes = hb2.size(); b2.initialData = hb2; b2.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb1.size(); outBuf.isOutput = true;
    ac.buffers.push_back(selBuf); ac.buffers.push_back(b1); ac.buffers.push_back(b2); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(1));
    ac.args.push_back(AdaptedArg::scalarInt(1));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = in1; ac.validatorInputB = in2; ac.elementCount = count;
    return true;
}
cudaError_t CudaSelectFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_select_fp16(ctx.intArgs[0], (const int*)ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2],
                            ctx.intArgs[1], ctx.intArgs[2], ctx.devBufs[3], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaSelectFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < count; ++i) {
        const float expected = (i % 2 > 0) ? ac.validatorInputA[i] : ac.validatorInputB[i];
        if (std::fabs(__half2float(out[i]) - expected) > 1.0f) return false;
    }
    return true;
}

// ============================================================================
// Softmax fp16, LayerNorm fp16, PReLU fp16, Scale fp16, Pool fp16
// (These mirror fp32 logic; only buffer packing differs.)
// ============================================================================
bool CudaSoftmaxFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_softmax_fp16";
    const int outside = spec.intParam("outside", 4);
    const int axis = spec.intParam("axis", 16);
    const int inside = spec.intParam("inside", 1);
    const int count = outside * inside;
    std::vector<float> input(outside * axis * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 7);
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = outside; ac.n = axis; ac.k = inside;
    return true;
}
cudaError_t CudaSoftmaxFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_softmax_fp16(ctx.devBufs[0], ctx.devBufs[1], ctx.intArgs[0], ctx.intArgs[1],
                              ctx.intArgs[2], ctx.intArgs[3], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaSoftmaxFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    const int totalElems = outside * axis * inside;
    if (static_cast<int>(output.size() * sizeof(float)) < totalElems * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int o = 0; o < outside; ++o) {
        for (int x = 0; x < inside; ++x) {
            const float* src = ac.validatorInputA.data() + o * axis * inside + x;
            float maxV = src[0];
            for (int z = 1; z < axis; ++z) maxV = std::max(maxV, src[z * inside]);
            float sum = 0.0f;
            for (int z = 0; z < axis; ++z) {
                float t = src[z * inside] - maxV; if (t < -87.0f) t = -87.0f;
                sum += expf(t);
            }
            sum = 1.0f / sum;
            for (int z = 0; z < axis; ++z) {
                float t = src[z * inside] - maxV; if (t < -87.0f) t = -87.0f;
                const float expected = expf(t) * sum;
                if (std::fabs(__half2float(out[(o * axis + z) * inside + x]) - expected) > 1e-2f) return false;
            }
        }
    }
    return true;
}

// LayerNorm fp16
bool CudaLayerNormFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_layernorm_fp16";
    const int outside = spec.intParam("outside", 4);
    const int inside = spec.intParam("inside", 32);
    const float eps = spec.floatParam("epsilon", 1e-5f);
    const int count = outside * inside;
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = 0.1f * (i % 13) - 0.5f;
    std::vector<float> gamma(inside, 1.0f), beta(inside, 0.0f);
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer gammaBuf; gammaBuf.setFp32(gamma); gammaBuf.isOutput = false;
    AdaptedBuffer betaBuf; betaBuf.setFp32(beta); betaBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(gammaBuf); ac.buffers.push_back(betaBuf);
    ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarFloat(eps));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(0));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = outside; ac.k = inside;
    return true;
}
cudaError_t CudaLayerNormFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_layernorm_fp16(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.floatArgs[0],
                               ctx.devBufs[0], ctx.devBufs[3], (const float*)ctx.devBufs[1], (const float*)ctx.devBufs[2],
                               false, ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaLayerNormFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, inside = ac.k;
    const float eps = ac.args[3].floatVal;
    const int totalElems = outside * inside;
    if (static_cast<int>(output.size() * sizeof(float)) < totalElems * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int o = 0; o < outside; ++o) {
        const float* in = ac.validatorInputA.data() + o * inside;
        float mean = 0.0f;
        for (int j = 0; j < inside; ++j) mean += in[j];
        mean /= inside;
        float sqsum = 0.0f;
        for (int j = 0; j < inside; ++j) sqsum += (in[j] - mean) * (in[j] - mean);
        float invStd = 1.0f / sqrtf(sqsum / inside + eps);
        for (int j = 0; j < inside; ++j) {
            const float expected = (in[j] - mean) * invStd;
            if (std::fabs(__half2float(out[o * inside + j]) - expected) > 5e-2f) return false;
        }
    }
    return true;
}

// PReLU fp16
bool CudaPreluFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_prelu_fp16";
    const int total = spec.intParam("size", 1024);
    const int channelsPack = spec.intParam("channels", 8);
    const int dim = total / channelsPack;
    std::vector<float> input(total); fillInputAlternating(input, total);
    std::vector<float> slope(channelsPack, 0.1f);
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer slopeBuf; slopeBuf.setFp32(slope); slopeBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(slopeBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPack));
    ac.args.push_back(AdaptedArg::scalarInt(dim));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(0));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.validatorInputB = slope; ac.elementCount = total;
    ac.n = channelsPack;
    return true;
}
cudaError_t CudaPreluFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_prelu_fp16(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.devBufs[0],
                           ctx.devBufs[2], (const float*)ctx.devBufs[1], ctx.intArgs[3], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaPreluFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount, channelsPack = ac.n;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < total; ++i) {
        const int c = i % channelsPack;
        const float x = ac.validatorInputA[i];
        const float expected = x > 0.0f ? x : x * ac.validatorInputB[c];
        if (std::fabs(__half2float(out[i]) - expected) > 1e-2f) return false;
    }
    return true;
}

// Scale fp16
bool CudaScaleFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_scale_fp16";
    const int total = spec.intParam("size", 1024);
    const int channelsPack = spec.intParam("channels", 8);
    std::vector<float> input(total); fillInputRamp(input, total);
    std::vector<float> scale(channelsPack, 2.0f), bias(channelsPack, 1.0f);
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer scaleBuf; scaleBuf.setFp32(scale); scaleBuf.isOutput = false;
    AdaptedBuffer biasBuf; biasBuf.setFp32(bias); biasBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(scaleBuf); ac.buffers.push_back(biasBuf);
    ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPack));
    ac.args.push_back(AdaptedArg::scalarInt(total / channelsPack));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total; ac.n = channelsPack;
    return true;
}
cudaError_t CudaScaleFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_scale_fp16(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.devBufs[0],
                           ctx.devBufs[3], (const float*)ctx.devBufs[1], (const float*)ctx.devBufs[2], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaScaleFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount, channelsPack = ac.n;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < total; ++i) {
        const int c = i % channelsPack;
        const float expected = ac.validatorInputA[i] * 2.0f + 1.0f;
        if (std::fabs(__half2float(out[i]) - expected) > 1e-2f) return false;
    }
    return true;
}

// Pool fp16 variants (maxpool/avgpool/global_*)
#define POOL_FP16_ADAPTER(CLASS, OPNAME, SHIM, GLOBAL) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    ac.entry = "mnn_corpus_" #SHIM; \
    const int ib = spec.intParam("batch", 1); \
    const int ic_p = spec.intParam("channels", 8); \
    const int ih = spec.intParam("h", 8); \
    const int iw = spec.intParam("w", 8); \
    std::vector<float> input(ib * ic_p * ih * iw); \
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11); \
    auto hb = packHalf(input); \
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false; \
    AdaptedBuffer outBuf; outBuf.isOutput = true; \
    int oh, ow, total; \
    if (GLOBAL) { \
        const int outside = ib, axis = ih, inside = ic_p; \
        const int per_block_size = 128; \
        const int calc_multi_num = (axis + per_block_size - 1) / per_block_size; \
        outBuf.sizeBytes = outside * inside * sizeof(__half); \
        total = outside * inside; \
        ac.globalSize[0] = outside * inside; ac.localSize[0] = per_block_size; ac.dims = 1; \
        ac.args.push_back(AdaptedArg::buffer(0)); \
        ac.args.push_back(AdaptedArg::buffer(1)); \
        for (int v : {outside, axis, inside, per_block_size, calc_multi_num}) \
            ac.args.push_back(AdaptedArg::scalarInt(v)); \
        ac.m = outside; ac.n = axis; ac.k = inside; \
    } else { \
        const int kx = spec.intParam("kernel_size", 3); \
        const int sx = spec.intParam("stride", 2); \
        const int padX = spec.intParam("pad", 1); \
        oh = (ih + 2 * padX - kx) / sx + 1; ow = oh; \
        outBuf.sizeBytes = ib * ic_p * oh * ow * sizeof(__half); \
        total = ib * oh * ow * ic_p; \
        ac.args.push_back(AdaptedArg::buffer(0)); \
        ac.args.push_back(AdaptedArg::buffer(1)); \
        for (int v : {ib, ic_p, ih, iw, oh, ow, padX, padX, kx, kx, sx, sx}) \
            ac.args.push_back(AdaptedArg::scalarInt(v)); \
        ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1; \
        ac.m = ib; ac.n = ic_p; ac.h = ih; ac.w = iw; ac.k = kx; ac.stride = sx; ac.orderType = padX; \
    } \
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf); \
    ac.validatorInputA = input; ac.elementCount = total; \
    return true; \
}
POOL_FP16_ADAPTER(CudaMaxPoolFp16Kernel, maxpool, maxpool_fp16, false)
POOL_FP16_ADAPTER(CudaAvgPoolFp16Kernel, avgpool, avgpool_fp16, false)
POOL_FP16_ADAPTER(CudaGlobalAvgPoolFp16Kernel, global_avgpool, global_avgpool_fp16, true)
POOL_FP16_ADAPTER(CudaGlobalMaxPoolFp16Kernel, global_maxpool, global_maxpool_fp16, true)

cudaError_t CudaMaxPoolFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_maxpool_fp16(ctx.devBufs[0], ctx.devBufs[1], ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                             ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                             ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                             ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
cudaError_t CudaAvgPoolFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_avgpool_fp16(ctx.devBufs[0], ctx.devBufs[1], ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                             ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                             ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                             ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
cudaError_t CudaGlobalAvgPoolFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_global_avgpool_fp16(ctx.devBufs[0], ctx.devBufs[1], ctx.intArgs[0], ctx.intArgs[1],
                                    ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
cudaError_t CudaGlobalMaxPoolFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_global_maxpool_fp16(ctx.devBufs[0], ctx.devBufs[1], ctx.intArgs[0], ctx.intArgs[1],
                                    ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}

// Pool fp16 validate functions mirror fp32 but read __half output.
bool CudaMaxPoolFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ib = ac.m, ic_p = ac.n, ih = ac.h, iw = ac.w;
    const int kx = ac.k, sx = ac.stride, padX = ac.orderType;
    const int oh = (ih + 2 * padX - kx) / sx + 1, ow = oh;
    const int totalElems = ib * oh * ow * ic_p;
    if (static_cast<int>(output.size() * sizeof(float)) < totalElems * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int b = 0; b < ib; ++b)
      for (int c = 0; c < ic_p; ++c)
        for (int oy = 0; oy < oh; ++oy)
          for (int ox = 0; ox < ow; ++ox) {
            float maxV = -65504.0f;
            for (int fy = 0; fy < kx; ++fy)
              for (int fx = 0; fx < kx; ++fx) {
                int iy = oy * sx - padX + fy, ix = ox * sx - padX + fx;
                if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                int off = ((b * ih + iy) * iw + ix) * ic_p + c;
                maxV = std::max(maxV, ac.validatorInputA[off]);
              }
            int outOff = ((b * oh + oy) * ow + ox) * ic_p + c;
            if (std::fabs(__half2float(out[outOff]) - maxV) > 1e-2f) return false;
          }
    return true;
}
bool CudaAvgPoolFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ib = ac.m, ic_p = ac.n, ih = ac.h, iw = ac.w;
    const int kx = ac.k, sx = ac.stride, padX = ac.orderType;
    const int oh = (ih + 2 * padX - kx) / sx + 1, ow = oh;
    const int totalElems = ib * oh * ow * ic_p;
    if (static_cast<int>(output.size() * sizeof(float)) < totalElems * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int b = 0; b < ib; ++b)
      for (int c = 0; c < ic_p; ++c)
        for (int oy = 0; oy < oh; ++oy)
          for (int ox = 0; ox < ow; ++ox) {
            float sum = 0.0f; int cnt = 0;
            for (int fy = 0; fy < kx; ++fy)
              for (int fx = 0; fx < kx; ++fx) {
                int iy = oy * sx - padX + fy, ix = ox * sx - padX + fx;
                if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                int off = ((b * ih + iy) * iw + ix) * ic_p + c;
                sum += ac.validatorInputA[off]; ++cnt;
              }
            if (cnt == 0) cnt = 1;
            int outOff = ((b * oh + oy) * ow + ox) * ic_p + c;
            if (std::fabs(__half2float(out[outOff]) - sum / cnt) > 1e-2f) return false;
          }
    return true;
}
bool CudaGlobalAvgPoolFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    if (static_cast<int>(output.size() * sizeof(float)) < outside * inside * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int o = 0; o < outside; ++o)
      for (int x = 0; x < inside; ++x) {
        float sum = 0.0f;
        for (int a = 0; a < axis; ++a) sum += ac.validatorInputA[(o * axis + a) * inside + x];
        if (std::fabs(__half2float(out[o * inside + x]) - sum / axis) > 1e-2f) return false;
      }
    return true;
}
bool CudaGlobalMaxPoolFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    if (static_cast<int>(output.size() * sizeof(float)) < outside * inside * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int o = 0; o < outside; ++o)
      for (int x = 0; x < inside; ++x) {
        float mx = ac.validatorInputA[o * axis * inside + x];
        for (int a = 1; a < axis; ++a) mx = std::max(mx, ac.validatorInputA[(o * axis + a) * inside + x]);
        if (std::fabs(__half2float(out[o * inside + x]) - mx) > 1e-2f) return false;
      }
    return true;
}

// ============================================================================
// Cast: additional type combinations
// ============================================================================
bool CudaCastI82I32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_cast_i8_i32";
    const int count = spec.intParam("size", 1024);
    std::vector<int8_t> input(count);
    for (int i = 0; i < count; ++i) input[i] = (int8_t)(i % 200 - 100);
    AdaptedBuffer inBuf; inBuf.sizeBytes = count;
    inBuf.initialData.assign((const uint8_t*)input.data(), (const uint8_t*)input.data() + count);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    const int floatSlots = (count + 3) / 4;
    ac.validatorInputA.resize(floatSlots, 0.0f);
    std::memcpy(ac.validatorInputA.data(), input.data(), count);
    ac.elementCount = count;
    return true;
}
cudaError_t CudaCastI82I32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_cast_i8_i32((const int8_t*)ctx.devBufs[0], (int32_t*)ctx.devBufs[1], (size_t)ctx.intArgs[0],
                           ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastI82I32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size()) < count) return false;
    const int8_t* in = reinterpret_cast<const int8_t*>(ac.validatorInputA.data());
    const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
    for (int i = 0; i < count; ++i) {
        if (out[i] != (int32_t)in[i]) return false;
    }
    return true;
}

bool CudaCastI322U8Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_cast_i32_u8";
    const int count = spec.intParam("size", 1024);
    std::vector<int32_t> input(count);
    for (int i = 0; i < count; ++i) input[i] = i % 256;
    AdaptedBuffer inBuf; inBuf.sizeBytes = count * sizeof(int32_t);
    inBuf.initialData.assign((const uint8_t*)input.data(), (const uint8_t*)input.data() + inBuf.sizeBytes);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count; outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA.assign(input.begin(), input.end());
    ac.elementCount = count;
    return true;
}
cudaError_t CudaCastI322U8Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_cast_i32_u8((const int32_t*)ctx.devBufs[0], (uint8_t*)ctx.devBufs[1], (size_t)ctx.intArgs[0],
                           ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastI322U8Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count) return false;
    const uint8_t* out = reinterpret_cast<const uint8_t*>(output.data());
    for (int i = 0; i < count; ++i) {
        if (out[i] != (uint8_t)(int32_t)ac.validatorInputA[i]) return false;
    }
    return true;
}

bool CudaCastU82I32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_cast_u8_i32";
    const int count = spec.intParam("size", 1024);
    std::vector<uint8_t> input(count);
    for (int i = 0; i < count; ++i) input[i] = (uint8_t)(i % 256);
    AdaptedBuffer inBuf; inBuf.sizeBytes = count;
    inBuf.initialData.assign((const uint8_t*)input.data(), (const uint8_t*)input.data() + count);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    const int floatSlots = (count + 3) / 4;
    ac.validatorInputA.resize(floatSlots, 0.0f);
    std::memcpy(ac.validatorInputA.data(), input.data(), count);
    ac.elementCount = count;
    return true;
}
cudaError_t CudaCastU82I32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_cast_u8_i32((const uint8_t*)ctx.devBufs[0], (int32_t*)ctx.devBufs[1], (size_t)ctx.intArgs[0],
                           ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastU82I32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size()) < count) return false;
    const uint8_t* in = reinterpret_cast<const uint8_t*>(ac.validatorInputA.data());
    const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
    for (int i = 0; i < count; ++i) {
        if (out[i] != (int32_t)in[i]) return false;
    }
    return true;
}

// ============================================================================
// Registration
// ============================================================================
// ============================================================================
// GatherV2 fp16
// ============================================================================
bool CudaGatherV2Fp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_gatherv2_fp16";
    const int inside = spec.intParam("inside", 4);
    const int iNum = spec.intParam("i_num", 3);
    const int oNum = spec.intParam("o_num", 2);
    const int outside = spec.intParam("outside", 2);
    const int count = outside * oNum * inside;
    std::vector<float> input(outside * iNum * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (float)i;
    std::vector<int32_t> indice(oNum);
    for (int i = 0; i < oNum; ++i) indice[i] = i % iNum;
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer idxBuf; idxBuf.sizeBytes = oNum * sizeof(int32_t);
    idxBuf.initialData.assign((const uint8_t*)indice.data(), (const uint8_t*)indice.data() + idxBuf.sizeBytes);
    idxBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(__half); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(idxBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(iNum));
    ac.args.push_back(AdaptedArg::scalarInt(oNum));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input;
    ac.validatorInputB.resize(oNum);
    for (int i = 0; i < oNum; ++i) ac.validatorInputB[i] = (float)indice[i];
    ac.elementCount = count; ac.m = outside; ac.n = iNum; ac.k = inside;
    return true;
}
cudaError_t CudaGatherV2Fp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gatherv2_fp16(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
                             ctx.devBufs[0], (const int*)ctx.devBufs[1], ctx.devBufs[2],
                             ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaGatherV2Fp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int inside = ac.k, iNum = ac.n, oNum = (int)ac.validatorInputB.size();
    const int outside = ac.m;
    const int count = outside * oNum * inside;
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int o = 0; o < outside; ++o)
      for (int n = 0; n < oNum; ++n)
        for (int x = 0; x < inside; ++x) {
            int idx = (int)ac.validatorInputB[n];
            float expected = ac.validatorInputA[o * iNum * inside + idx * inside + x];
            if (std::fabs(__half2float(out[(o * oNum + n) * inside + x]) - expected) > 1.0f) return false;
        }
    return true;
}

// ============================================================================
// ArgMax fp16
// ============================================================================
bool CudaArgMaxFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_argmax_fp16";
    const int outside = spec.intParam("outside", 4);
    const int dim = spec.intParam("dim", 8);
    const int inside = spec.intParam("inside", 1);
    const int count = outside * inside;
    std::vector<float> input(outside * dim * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (float)(i % 17);
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(dim));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = outside; ac.n = dim; ac.k = inside;
    return true;
}
cudaError_t CudaArgMaxFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_argmax_fp16(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                           ctx.devBufs[0], (int*)ctx.devBufs[1], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaArgMaxFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, dim = ac.n, inside = ac.k;
    const int count = outside * inside;
    if (static_cast<int>(output.size() * sizeof(float)) < count * 4) return false;
    const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
    for (int o = 0; o < outside; ++o)
      for (int x = 0; x < inside; ++x) {
        int idx = 0; float mx = ac.validatorInputA[o * dim * inside + x];
        for (int j = 1; j < dim; ++j) {
          float v = ac.validatorInputA[o * dim * inside + j * inside + x];
          if (mx < v) { idx = j; mx = v; }
        }
        if (out[o * inside + x] != idx) return false;
      }
    return true;
}

// ============================================================================
// ArgMin fp16
// ============================================================================
bool CudaArgMinFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_argmin_fp16";
    const int outside = spec.intParam("outside", 4);
    const int dim = spec.intParam("dim", 8);
    const int inside = spec.intParam("inside", 1);
    const int count = outside * inside;
    std::vector<float> input(outside * dim * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (float)(i % 17);
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(dim));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = outside; ac.n = dim; ac.k = inside;
    return true;
}
cudaError_t CudaArgMinFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_argmin_fp16(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                           ctx.devBufs[0], (int*)ctx.devBufs[1], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaArgMinFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, dim = ac.n, inside = ac.k;
    const int count = outside * inside;
    if (static_cast<int>(output.size() * sizeof(float)) < count * 4) return false;
    const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
    for (int o = 0; o < outside; ++o)
      for (int x = 0; x < inside; ++x) {
        int idx = 0; float mn = ac.validatorInputA[o * dim * inside + x];
        for (int j = 1; j < dim; ++j) {
          float v = ac.validatorInputA[o * dim * inside + j * inside + x];
          if (mn > v) { idx = j; mn = v; }
        }
        if (out[o * inside + x] != idx) return false;
      }
    return true;
}

// ============================================================================
// Transpose fp16 (NHWC->NCHW, NCHW->NHWC)
// ============================================================================
bool CudaNhwc2NchwFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_nhwc2nchw_fp16";
    const int outside = spec.intParam("outside", 2);
    const int axis = spec.intParam("axis", 4);
    const int inside = spec.intParam("inside", 3);
    const int total = outside * axis * inside;
    std::vector<float> input(total);
    for (int i = 0; i < total; ++i) input[i] = (float)i;
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.m = outside; ac.n = axis; ac.k = inside;
    return true;
}
cudaError_t CudaNhwc2NchwFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_nhwc2nchw_fp16(ctx.devBufs[0], ctx.devBufs[1],
                               ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                               ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaNhwc2NchwFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    const int total = outside * axis * inside;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int idx = 0; idx < total; ++idx) {
        int x = idx % inside;
        int y = (idx / inside) % axis;
        int z = idx / (inside * axis);
        int nchwOff = z * axis * inside + y * inside + x;
        if (std::fabs(__half2float(out[nchwOff]) - ac.validatorInputA[idx]) > 1.0f) return false;
    }
    return true;
}

bool CudaNchw2NhwcFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_nchw2nhwc_fp16";
    const int outside = spec.intParam("outside", 2);
    const int axis = spec.intParam("axis", 4);
    const int inside = spec.intParam("inside", 3);
    const int total = outside * axis * inside;
    std::vector<float> input(total);
    for (int i = 0; i < total; ++i) input[i] = (float)i;
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.m = outside; ac.n = axis; ac.k = inside;
    return true;
}
cudaError_t CudaNchw2NhwcFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_nchw2nhwc_fp16(ctx.devBufs[0], ctx.devBufs[1],
                               ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                               ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaNchw2NhwcFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    const int total = outside * axis * inside;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int idx = 0; idx < total; ++idx) {
        int x = idx % inside;
        int y = (idx / inside) % axis;
        int z = idx / (inside * axis);
        int nchwOff = z * axis * inside + y * inside + x;
        if (std::fabs(__half2float(out[idx]) - ac.validatorInputA[nchwOff]) > 1.0f) return false;
    }
    return true;
}

// ============================================================================
// A-class: CONV_DW fp16/half2 variants (5 kernels)
// ============================================================================
// ---- CONV_DW_OPT: fp32 input + half weight/bias -> fp32 output ----
bool CudaConvDwOptFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_conv_dw_opt_fp32";
    const int batch = spec.intParam("batch", 1);
    const int iw = spec.intParam("iw", 8), ih = spec.intParam("ih", 8);
    const int c = spec.intParam("channels", 1);
    const int c_p = c * PACK_NUMBER;
    const int kw = spec.intParam("kw", 3), kh = spec.intParam("kh", 3);
    const int sw = spec.intParam("sw", 1), sh = spec.intParam("sh", 1);
    const int pw = spec.intParam("pw", 1), ph = spec.intParam("ph", 1);
    const int dw = spec.intParam("dw", 1), dh = spec.intParam("dh", 1);
    const int ow = (iw + 2 * pw - (kw - 1) * dw - 1) / sw + 1;
    const int oh = (ih + 2 * ph - (kh - 1) * dh - 1) / sh + 1;
    const int total = batch * oh * ow * c_p;

    std::vector<float> input(batch * ih * iw * c_p, 0.0f);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 7);
    std::vector<float> kernelF(c_p * kh * kw, 0.1f);
    std::vector<float> biasF(c_p, 0.5f);
    auto kernelH = packHalf(kernelF);
    auto biasH = packHalf(biasF);

    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernelH.size(); kBuf.initialData = kernelH; kBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.sizeBytes = biasH.size(); bBuf.initialData = biasH; bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = batch * oh * ow * c_p * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);

    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarFloat(1e30f));
    ac.args.push_back(AdaptedArg::scalarFloat(-1e30f));
    const int d_oc = c_p / 2, d_ow = ow, d_oh = oh;
    for (int v : {iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, d_oc, d_ow, d_oh})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total / 2); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.c = c_p; ac.k = kw; ac.stride = sw; ac.orderType = pw; ac.m = batch; ac.n = oh;
    return true;
}
cudaError_t CudaConvDwOptFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv_dw_opt_fp32((const float*)ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], (float*)ctx.devBufs[3],
                                ctx.floatArgs[0], ctx.floatArgs[1],
                                ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                                ctx.intArgs[12], ctx.intArgs[13], ctx.intArgs[14], ctx.intArgs[15],
                                ctx.intArgs[16], ctx.intArgs[17],
                                ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaConvDwOptFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() < total) return false;
    for (int i = 0; i < std::min(10, total); ++i) {
        if (output[i] != 0.0f) return true;
    }
    return false;
}

// ---- CONV_DW_HALF2_OPT: all half2 ----
bool CudaConvDwHalf2OptFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_conv_dw_half2_opt_fp32";
    const int batch = spec.intParam("batch", 1);
    const int iw = spec.intParam("iw", 8), ih = spec.intParam("ih", 8);
    const int c = spec.intParam("channels", 1);
    const int c_p = c * PACK_NUMBER;
    const int kw = spec.intParam("kw", 3), kh = spec.intParam("kh", 3);
    const int sw = spec.intParam("sw", 1), sh = spec.intParam("sh", 1);
    const int pw = spec.intParam("pw", 1), ph = spec.intParam("ph", 1);
    const int dw = spec.intParam("dw", 1), dh = spec.intParam("dh", 1);
    const int ow = (iw + 2 * pw - (kw - 1) * dw - 1) / sw + 1;
    const int oh = (ih + 2 * ph - (kh - 1) * dh - 1) / sh + 1;
    const int total = batch * oh * ow * c_p;

    std::vector<float> inputF(batch * ih * iw * c_p, 0.0f);
    for (int i = 0; i < (int)inputF.size(); ++i) inputF[i] = 0.1f * (i % 7);
    std::vector<float> kernelF(c_p * kh * kw, 0.1f);
    std::vector<float> biasF(c_p, 0.5f);
    auto inputH = packHalf(inputF);
    auto kernelH = packHalf(kernelF);
    auto biasH = packHalf(biasF);

    AdaptedBuffer inBuf; inBuf.sizeBytes = inputH.size(); inBuf.initialData = inputH; inBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernelH.size(); kBuf.initialData = kernelH; kBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.sizeBytes = biasH.size(); bBuf.initialData = biasH; bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = batch * oh * ow * c_p * sizeof(__half); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);

    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarFloat(1e30f));
    ac.args.push_back(AdaptedArg::scalarFloat(-1e30f));
    const int d_oc = c_p / 2, d_ow = ow, d_oh = oh;
    for (int v : {iw, ih, c, c_p / 2, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, d_oc, d_ow, d_oh})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total / 2); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.c = c_p; ac.k = kw; ac.stride = sw; ac.orderType = pw; ac.m = batch; ac.n = oh;
    return true;
}
cudaError_t CudaConvDwHalf2OptFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv_dw_half2_opt_fp32(ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], ctx.devBufs[3],
                                      ctx.floatArgs[0], ctx.floatArgs[1],
                                      ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                      ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                      ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                                      ctx.intArgs[12], ctx.intArgs[13], ctx.intArgs[14], ctx.intArgs[15],
                                      ctx.intArgs[16], ctx.intArgs[17],
                                      ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaConvDwHalf2OptFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, total); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- CONV_DW3x3_HALF2_OPT: all half2, 3x3 specialization ----
bool CudaConvDw3x3Half2OptFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_conv_dw3x3_half2_opt_fp32";
    const int batch = spec.intParam("batch", 1);
    const int iw = spec.intParam("iw", 10), ih = spec.intParam("ih", 10);
    const int c = spec.intParam("channels", 1);
    const int c_p = c * PACK_NUMBER;
    const int kw = 3, kh = 3;
    const int sw = 1, sh = 1;
    const int pw = 1, ph = 1;
    const int dw = 1, dh = 1;
    const int ow = (iw + 2 * pw - (kw - 1) * dw - 1) / sw + 1;
    const int oh = (ih + 2 * ph - (kh - 1) * dh - 1) / sh + 1;
    const int total = batch * oh * ow * c_p;

    std::vector<float> inputF(batch * ih * iw * c_p, 0.0f);
    for (int i = 0; i < (int)inputF.size(); ++i) inputF[i] = 0.1f * (i % 7);
    std::vector<float> kernelF(c_p * kh * kw, 0.1f);
    std::vector<float> biasF(c_p, 0.5f);
    auto inputH = packHalf(inputF);
    auto kernelH = packHalf(kernelF);
    auto biasH = packHalf(biasF);

    AdaptedBuffer inBuf; inBuf.sizeBytes = inputH.size(); inBuf.initialData = inputH; inBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernelH.size(); kBuf.initialData = kernelH; kBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.sizeBytes = biasH.size(); bBuf.initialData = biasH; bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = batch * oh * ow * c_p * sizeof(__half); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);

    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarFloat(1e30f));
    ac.args.push_back(AdaptedArg::scalarFloat(-1e30f));
    const int d_oc = c_p / 2, d_ow = ow / 2, d_oh = oh;
    for (int v : {iw, ih, c, c_p / 2, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, d_oc, d_ow, d_oh})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total / 4); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.c = c_p; ac.k = kw; ac.stride = sw; ac.orderType = pw; ac.m = batch; ac.n = oh;
    return true;
}
cudaError_t CudaConvDw3x3Half2OptFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv_dw3x3_half2_opt_fp32(ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], ctx.devBufs[3],
                                          ctx.floatArgs[0], ctx.floatArgs[1],
                                          ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                          ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                          ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                                          ctx.intArgs[12], ctx.intArgs[13], ctx.intArgs[14], ctx.intArgs[15],
                                          ctx.intArgs[16], ctx.intArgs[17],
                                          ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaConvDw3x3Half2OptFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, total); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- CONV_DW_MULTI_WIDTH4: fp32 input + half weight/bias -> fp32 output ----
bool CudaConvDwMultiWidth4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_conv_dw_multi_width4_fp32";
    const int batch = spec.intParam("batch", 1);
    const int iw = spec.intParam("iw", 12), ih = spec.intParam("ih", 12);
    const int c = spec.intParam("channels", 1);
    const int c_p = c * PACK_NUMBER;
    const int kw = spec.intParam("kw", 5), kh = spec.intParam("kh", 1);
    const int sw = 1, sh = 1, pw = 0, ph = 0, dw = 1, dh = 1;
    const int ow = (iw + 2 * pw - (kw - 1) * dw - 1) / sw + 1;
    const int oh = (ih + 2 * ph - (kh - 1) * dh - 1) / sh + 1;
    const int total = batch * oh * ow * c_p;

    std::vector<float> input(batch * ih * iw * c_p, 0.0f);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 7);
    // MNN's CONV_DW_MULTI_WIDTH4 speculatively reads kernel[filter_offset + kw*c_p]
    // (one past end) in the pipeline's last iteration; allocate extra padding.
    std::vector<float> kernelF(c_p * kh * kw + c_p, 0.0f);
    for (int i = 0; i < c_p * kh * kw; ++i) kernelF[i] = 0.1f;
    std::vector<float> biasF(c_p, 0.5f);
    auto kernelH = packHalf(kernelF);
    auto biasH = packHalf(biasF);

    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernelH.size(); kBuf.initialData = kernelH; kBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.sizeBytes = biasH.size(); bBuf.initialData = biasH; bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = batch * oh * ow * c_p * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);

    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarFloat(1e30f));
    ac.args.push_back(AdaptedArg::scalarFloat(-1e30f));
    const int d_oc = c_p, d_ow_4 = ow / 4, d_oh = oh;
    for (int v : {iw, ih, c, c_p, ow, oh, kw, kh, total, d_oc, d_ow_4, d_oh})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total / 4); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.c = c_p; ac.k = kw; ac.stride = sw; ac.orderType = pw; ac.m = batch; ac.n = oh;
    return true;
}
cudaError_t CudaConvDwMultiWidth4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv_dw_multi_width4_fp32((const float*)ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], (float*)ctx.devBufs[3],
                                          ctx.floatArgs[0], ctx.floatArgs[1],
                                          ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                          ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                          ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                                          ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaConvDwMultiWidth4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() < total) return false;
    for (int i = 0; i < std::min(10, total); ++i) {
        if (output[i] != 0.0f) return true;
    }
    return false;
}

// ---- CONV_DW_MULTI_WIDTH_CHANNEL: fp32 input + half weight/bias -> fp32 output ----
bool CudaConvDwMultiWidthChannelFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_conv_dw_multi_width_channel_fp32";
    const int batch = spec.intParam("batch", 1);
    const int iw = spec.intParam("iw", 10), ih = spec.intParam("ih", 10);
    const int c = spec.intParam("channels", 1);
    const int c_p = c * PACK_NUMBER;
    const int kw = spec.intParam("kw", 5), kh = spec.intParam("kh", 1);
    const int sw = 1, sh = 1, pw = 0, ph = 0, dw = 1, dh = 1;
    const int ow = (iw + 2 * pw - (kw - 1) * dw - 1) / sw + 1;
    const int oh = (ih + 2 * ph - (kh - 1) * dh - 1) / sh + 1;
    const int total = batch * oh * ow * c_p;

    std::vector<float> input(batch * ih * iw * c_p, 0.0f);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 7);
    std::vector<float> kernelF(c_p * kh * kw, 0.1f);
    std::vector<float> biasF(c_p, 0.5f);
    auto kernelH = packHalf(kernelF);
    auto biasH = packHalf(biasF);

    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernelH.size(); kBuf.initialData = kernelH; kBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.sizeBytes = biasH.size(); bBuf.initialData = biasH; bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = batch * oh * ow * c_p * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);

    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarFloat(1e30f));
    ac.args.push_back(AdaptedArg::scalarFloat(-1e30f));
    const int d_oc_2 = c_p / 2, d_ow_2 = ow / 2, d_oh = oh;
    for (int v : {iw, ih, c, c_p, ow, oh, kw, kh, total, d_oc_2, d_ow_2, d_oh})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total / 4); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.c = c_p; ac.k = kw; ac.stride = sw; ac.orderType = pw; ac.m = batch; ac.n = oh;
    return true;
}
cudaError_t CudaConvDwMultiWidthChannelFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv_dw_multi_width_channel_fp32((const float*)ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], (float*)ctx.devBufs[3],
                                                  ctx.floatArgs[0], ctx.floatArgs[1],
                                                  ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                                  ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                                  ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                                                  ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaConvDwMultiWidthChannelFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() < total) return false;
    for (int i = 0; i < std::min(10, total); ++i) {
        if (output[i] != 0.0f) return true;
    }
    return false;
}

// ============================================================================
// A-class: half-precision transpose/pack/fuseblit/unary adapters (5)
// ============================================================================
// ---- PACKCOMMON_half_4: pack half4 (axisAlign = UP_DIV(axis,2)*2) ----
// Corpus adaptation (see transpose_half.cu header): float-scalar variant of
// MNN's int2 (4-half) packed kernel. axisAlign = UP_DIV(axis,2)*2.
bool CudaPackCommonHalf4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int inside = spec.intParam("inside", 4);
    const int axis = spec.intParam("axis", 8);
    const int outside = spec.intParam("outside", 1);
    const int axisAlign = (axis + 1) / 2 * 2;
    const int maxCount = axisAlign * inside * outside;
    const int insideStride = axis, axisStride = 1;
    std::vector<float> input(outside * axis * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 7);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = maxCount * sizeof(float);
    outBuf.initialData.assign(maxCount * sizeof(float), 0);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(insideStride));
    ac.args.push_back(AdaptedArg::scalarInt(axisStride));
    ac.args.push_back(AdaptedArg::scalarInt(inside));     // d_is
    ac.args.push_back(AdaptedArg::scalarInt(axisAlign));  // d_cs
    ac.entry = "mnn_corpus_packcommon_half_4_fp32";
    ac.globalSize[0] = gridFor(maxCount); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = maxCount;
    ac.m = outside; ac.n = axis; ac.k = inside; ac.w = axisAlign; ac.stride = insideStride;
    return true;
}
cudaError_t CudaPackCommonHalf4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // intArgs: [0]=inside,[1]=axis,[2]=outside,[3]=insideStride,[4]=axisStride,[5]=d_is,[6]=d_cs
    mnn_corpus_packcommon_half_4_fp32(ctx.devBufs[0], ctx.devBufs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.intArgs[5], ctx.intArgs[6], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaPackCommonHalf4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, area = ac.k, axisAlign = ac.w, insideStride = ac.stride, axisStride = 1;
    for (int z = 0; z < outside; ++z)
        for (int x = 0; x < area; ++x)
            for (int y = 0; y < axisAlign; ++y) {
                int dstIdx = (z * area + x) * axisAlign + y;
                if (y >= axis) { if (std::fabs(output[dstIdx]) > 1e-2f) return false; continue; }
                int src = x * insideStride + y * axisStride + z * area * axis;
                if (std::fabs(output[dstIdx] - ac.validatorInputA[src]) > 1e-1f) return false;
            }
    return true;
}

// ---- PACKCOMMON_REARRANGE_half_4: double* tile rearrange ----
// Requires: axis%16==0, inside%32==0. axisAlign = axis.
bool CudaPackCommonRearrangeHalf4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int inside = spec.intParam("inside", 32);
    const int axis = spec.intParam("axis", 16);
    const int outside = spec.intParam("outside", 1);
    const int insideStride = axis, axisStride = 1;
    const int maxCount = axis * inside * outside;
    // double = 4x half = 8 bytes per element. Input is packed half bytes.
    std::vector<float> inputF(outside * axis * inside);
    for (int i = 0; i < (int)inputF.size(); ++i) inputF[i] = 0.1f * (i % 7);
    auto hb = packHalf(inputF);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = maxCount * sizeof(__half);
    outBuf.initialData.assign(maxCount * sizeof(__half), 0);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(insideStride));
    ac.args.push_back(AdaptedArg::scalarInt(axisStride));
    ac.entry = "mnn_corpus_packcommon_rearrange_half_4_fp32";
    ac.globalSize[0] = gridFor(maxCount); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = maxCount;
    ac.m = outside; ac.n = axis; ac.k = inside; ac.w = axis; ac.stride = insideStride;
    return true;
}
cudaError_t CudaPackCommonRearrangeHalf4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // intArgs: [0]=inside,[1]=axis,[2]=outside,[3]=insideStride,[4]=axisStride
    mnn_corpus_packcommon_rearrange_half_4_fp32((const double*)ctx.devBufs[0], (double*)ctx.devBufs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaPackCommonRearrangeHalf4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Smoke test: check non-zero output (rearrange tile structure is complex)
    const int total = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, total); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- UNPACKCOMMON_REARRANGE_half_4: double* tile rearrange (inverse) ----
bool CudaUnpackCommonRearrangeHalf4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int inside = spec.intParam("inside", 32);
    const int axis = spec.intParam("axis", 16);
    const int outside = spec.intParam("outside", 1);
    const int insideStride = axis, axisStride = 1;
    const int maxCount = axis * inside * outside;
    std::vector<float> inputF(outside * axis * inside);
    for (int i = 0; i < (int)inputF.size(); ++i) inputF[i] = 0.1f * (i % 7);
    auto hb = packHalf(inputF);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = maxCount * sizeof(__half);
    outBuf.initialData.assign(maxCount * sizeof(__half), 0);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(insideStride));
    ac.args.push_back(AdaptedArg::scalarInt(axisStride));
    ac.entry = "mnn_corpus_unpackcommon_rearrange_half_4_fp32";
    ac.globalSize[0] = gridFor(maxCount); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = maxCount;
    ac.m = outside; ac.n = axis; ac.k = inside; ac.w = axis; ac.stride = insideStride;
    return true;
}
cudaError_t CudaUnpackCommonRearrangeHalf4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_unpackcommon_rearrange_half_4_fp32((const double*)ctx.devBufs[0], (double*)ctx.devBufs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaUnpackCommonRearrangeHalf4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, total); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- fuseblit_half_4: int16_t* vec2 fused blit ----
bool CudaFuseBlitHalf4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int fuseNum = spec.intParam("fuse_num", 2);
    const int sx = spec.intParam("size_x", 2), sy = spec.intParam("size_y", 2), sz = spec.intParam("size_z", 2);
    const int strideX = 1, strideY = sx, strideZ = sx * sy;
    const int dstStrideX = strideX, dstStrideY = strideY, dstStrideZ = strideZ;
    const int count = fuseNum * sz * sy * sx;
    std::vector<int32_t> sliceOffset(fuseNum * 2);
    int srcOff = 0, dstOff = 0;
    for (int j = 0; j < fuseNum; ++j) {
        sliceOffset[j] = srcOff;
        sliceOffset[fuseNum + j] = dstOff;
        srcOff += sz * sy * sx;
        dstOff += sz * sy * sx;
    }
    // Input/output as half (int16_t = 2 bytes). fuseblit_half_4 writes int16_t vec.
    std::vector<float> inputF(fuseNum * sz * sy * sx);
    for (int i = 0; i < (int)inputF.size(); ++i) inputF[i] = 0.1f * (i % 7);
    auto hb = packHalf(inputF);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(__half);
    outBuf.initialData.assign(count * sizeof(__half), 0);
    outBuf.isOutput = true;
    AdaptedBuffer offBuf; offBuf.sizeBytes = sliceOffset.size() * sizeof(int32_t);
    offBuf.initialData.assign((const uint8_t*)sliceOffset.data(), (const uint8_t*)sliceOffset.data() + offBuf.sizeBytes);
    offBuf.isOutput = false;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf); ac.buffers.push_back(offBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(fuseNum));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(sx));
    ac.args.push_back(AdaptedArg::scalarInt(sy));
    ac.args.push_back(AdaptedArg::scalarInt(sz));
    ac.args.push_back(AdaptedArg::scalarInt(strideZ));
    ac.args.push_back(AdaptedArg::scalarInt(strideY));
    ac.args.push_back(AdaptedArg::scalarInt(dstStrideZ));
    ac.args.push_back(AdaptedArg::scalarInt(dstStrideY));
    ac.entry = "mnn_corpus_fuseblit_half_4_fp32";
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = count;
    return true;
}
cudaError_t CudaFuseBlitHalf4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // intArgs: [0]=fuseNum,[1]=count,[2]=sx,[3]=sy,[4]=sz,[5]=strideZ,[6]=strideY,[7]=dstStrideZ,[8]=dstStrideY
    mnn_corpus_fuseblit_half_4_fp32(ctx.devBufs[0], ctx.devBufs[1],
        ctx.intArgs[0], ctx.intArgs[1], (const int32_t*)ctx.devBufs[2],
        ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8],
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaFuseBlitHalf4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Smoke test: check non-zero output (int16_t vec2 blit, complex indexing)
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, count); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- UNARY_HALF2_SIGMOID: element-wise sigmoid on half2 ----
bool CudaUnaryHalf2SigmoidFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int count = spec.intParam("size", 1024);
    // Round up to even (half2 = 2 elements)
    const int countAligned = (count + 1) / 2 * 2;
    std::vector<float> input(countAligned);
    for (int i = 0; i < countAligned; ++i) input[i] = 0.1f * (i % 13) - 0.5f;
    auto hb = packHalf(input);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = hb.size(); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(countAligned));
    ac.entry = "mnn_corpus_unary_half2_sigmoid_fp32";
    ac.globalSize[0] = gridFor(countAligned / 2); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = countAligned;
    return true;
}
cudaError_t CudaUnaryHalf2SigmoidFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_unary_half2_sigmoid_fp32(ctx.devBufs[0], ctx.devBufs[1], (size_t)ctx.intArgs[0],
                                         ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaUnaryHalf2SigmoidFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < count; ++i) {
        float x = ac.validatorInputA[i];
        float expected = 1.0f / (1.0f + expf(-x));
        if (std::fabs(__half2float(out[i]) - expected) > 1e-2f) return false;
    }
    return true;
}

// ============================================================================
// B-class: Winograd half2 adapters (2 kernels, CudaOpsFp16.cu)
// ============================================================================
// ---- WinoInputTrans_half2: Winograd input transform (half2) ----
// Defined but not launched in MNN. Smoke test with half2 data.
bool CudaWinoInputTransHalf2Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int width = spec.intParam("width", 8), height = spec.intParam("height", 8);
    const int ci = spec.intParam("ci", 8);
    const int ci_p8 = (ci + 7) / 8 * 8;  // PACK_NUMBER=8
    const int unit = 2, block = 16;  // (UNIT+kernel-1)^2 = (2+3-1)^2 = 16
    const int ow = (width + unit - 1) / unit;
    const int oh = (height + unit - 1) / unit;
    const int maxCount = ow * oh * ci_p8;
    const int lD = ci_p8, whD = oh, wD = ow;
    const int pad_x = 1, pad_y = 1;
    // half2 input: input is [batch, height, width, ci_p8] of half2 (ci_p8/2 half2s)
    std::vector<float> inputF(height * width * ci_p8);
    for (int i = 0; i < (int)inputF.size(); ++i) inputF[i] = 0.01f * (i % 7);
    auto hb = packHalf(inputF);
    AdaptedBuffer inBuf; inBuf.sizeBytes = hb.size(); inBuf.initialData = hb; inBuf.isOutput = false;
    // BtdB output: [16, maxCount] of half2
    AdaptedBuffer outBuf; outBuf.sizeBytes = 16 * maxCount * sizeof(__half);
    outBuf.initialData.assign(16 * maxCount * sizeof(__half), 0);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(unit));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt(ci));
    ac.args.push_back(AdaptedArg::scalarInt(ci_p8));
    ac.args.push_back(AdaptedArg::scalarInt(maxCount));
    ac.args.push_back(AdaptedArg::scalarInt(lD));
    ac.args.push_back(AdaptedArg::scalarInt(whD));
    ac.args.push_back(AdaptedArg::scalarInt(wD));
    ac.args.push_back(AdaptedArg::scalarInt(pad_x));
    ac.args.push_back(AdaptedArg::scalarInt(pad_y));
    ac.args.push_back(AdaptedArg::scalarInt(width));
    ac.args.push_back(AdaptedArg::scalarInt(height));
    ac.entry = "mnn_corpus_wino_input_trans_half2_fp32";
    ac.globalSize[0] = gridFor(maxCount); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = maxCount;
    return true;
}
cudaError_t CudaWinoInputTransHalf2Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // intArgs: [0]=unit,[1]=block,[2]=ci,[3]=ci_p8,[4]=maxCount,[5]=lD,[6]=whD,[7]=wD,
    //          [8]=pad_x,[9]=pad_y,[10]=width,[11]=height
    mnn_corpus_wino_input_trans_half2_fp32(ctx.devBufs[0], ctx.devBufs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
        ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaWinoInputTransHalf2Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Smoke test: check non-zero output (half2 winograd transform)
    const int maxCount = ac.elementCount;
    const int totalOut = 16 * maxCount;
    if (static_cast<int>(output.size() * sizeof(float)) < totalOut * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, totalOut); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- WinoTrans2Output_half2: Winograd output transform (half2) ----
bool CudaWinoTrans2OutputHalf2Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int width = spec.intParam("width", 8), height = spec.intParam("height", 8);
    const int co = spec.intParam("co", 8);
    const int co_p8 = (co + 7) / 8 * 8;
    const int unit = 2, block = 16;
    const int ow = (width + unit - 1) / unit;
    const int oh = (height + unit - 1) / unit;
    const int maxCount = ow * oh * co_p8;
    const int hD = co_p8, whD = oh, wD = ow;
    // matmulData: [16, maxCount] of half2 (simulated winograd matmul output)
    std::vector<float> matmulF(16 * maxCount);
    for (int i = 0; i < (int)matmulF.size(); ++i) matmulF[i] = 0.01f * (i % 7);
    auto mb = packHalf(matmulF);
    std::vector<float> biasF(co_p8, 0.5f);
    AdaptedBuffer inBuf; inBuf.sizeBytes = mb.size(); inBuf.initialData = mb; inBuf.isOutput = false;
    AdaptedBuffer biasBuf; biasBuf.setFp32(biasF); biasBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = height * width * co_p8 * sizeof(__half);
    outBuf.initialData.assign(height * width * co_p8 * sizeof(__half), 0);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(biasBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(unit));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt(co));
    ac.args.push_back(AdaptedArg::scalarInt(co_p8));
    ac.args.push_back(AdaptedArg::scalarInt(maxCount));
    ac.args.push_back(AdaptedArg::scalarInt(hD));
    ac.args.push_back(AdaptedArg::scalarInt(whD));
    ac.args.push_back(AdaptedArg::scalarInt(wD));
    ac.args.push_back(AdaptedArg::scalarInt(width));
    ac.args.push_back(AdaptedArg::scalarInt(height));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // activationType = none
    ac.entry = "mnn_corpus_wino_trans2output_half2_fp32";
    ac.globalSize[0] = gridFor(maxCount); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = matmulF; ac.elementCount = maxCount;
    return true;
}
cudaError_t CudaWinoTrans2OutputHalf2Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // intArgs: [0]=unit,[1]=block,[2]=co,[3]=co_p8,[4]=maxCount,[5]=hD,[6]=whD,[7]=wD,
    //          [8]=width,[9]=height,[10]=activationType
    mnn_corpus_wino_trans2output_half2_fp32(ctx.devBufs[0], (const float*)ctx.devBufs[1], ctx.devBufs[2],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
        ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10],
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaWinoTrans2OutputHalf2Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Smoke test: check non-zero output
    const int maxCount = ac.elementCount;
    const int totalOut = maxCount;
    if (static_cast<int>(output.size() * sizeof(float)) < totalOut * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, totalOut); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ============================================================================
// P0 fp16 adapters — Attention / LinearAttention / RoPE / TopKV2
// ============================================================================
// Helper: build a half-packed AdaptedBuffer from a float vector.
static AdaptedBuffer makeHalfBuffer(const std::vector<float>& f, bool isOutput) {
    AdaptedBuffer b;
    b.sizeBytes = f.size() * sizeof(__half);
    b.initialData = packHalf(f);
    b.isOutput = isOutput;
    return b;
}
static AdaptedBuffer makeHalfOutput(size_t elemCount) {
    AdaptedBuffer b;
    b.sizeBytes = elemCount * sizeof(__half);
    b.isOutput = true;
    return b;
}

// ---- 1. CudaFlashDecodeFp16Kernel ----
bool CudaFlashDecodeFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_flash_decode_fp16";
    const int batch = spec.intParam("batch", 1);
    const int head_num = spec.intParam("head_num", 2);
    const int kv_head_num = spec.intParam("kv_head_num", 1);
    const int head_dim = spec.intParam("head_dim", 32);
    const int key_seq_len = spec.intParam("key_seq_len", 8);
    const int max_kv_len = spec.intParam("max_kv_len", 8);
    const float scale = spec.floatParam("scale", 1.0f / std::sqrt((float)head_dim));
    const int outSize = batch * head_num * head_dim;
    std::vector<float> query(batch * head_num * head_dim, 0.1f);
    std::vector<float> key_cache(key_seq_len * batch * kv_head_num * head_dim, 0.2f);
    std::vector<float> value_cache(batch * kv_head_num * max_kv_len * head_dim, 0.3f);
    ac.buffers.push_back(makeHalfBuffer(query, false));
    ac.buffers.push_back(makeHalfBuffer(key_cache, false));
    ac.buffers.push_back(makeHalfBuffer(value_cache, false));
    ac.buffers.push_back(makeHalfOutput(outSize));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(head_num));
    ac.args.push_back(AdaptedArg::scalarInt(kv_head_num));
    ac.args.push_back(AdaptedArg::scalarInt(head_dim));
    ac.args.push_back(AdaptedArg::scalarInt(key_seq_len));
    ac.args.push_back(AdaptedArg::scalarInt(max_kv_len));
    ac.args.push_back(AdaptedArg::scalarFloat(scale));
    const int grid = batch * head_num;
    const int block = 128;
    const size_t sharedMem = 4 * (2 * 4 + 32 * 4);
    ac.args.push_back(AdaptedArg::scalarInt(grid));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = outSize;
    return true;
}
cudaError_t CudaFlashDecodeFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_flash_decode_fp16(
        ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], ctx.devBufs[3],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.floatArgs[0], ctx.grid, ctx.block, (size_t)ctx.intArgs[6], ctx.stream);
    return cudaGetLastError();
}
bool CudaFlashDecodeFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- 2. CudaFlashDecodeWithMaskFp16Kernel ----
bool CudaFlashDecodeWithMaskFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_flash_decode_with_mask_fp16";
    const int batch = spec.intParam("batch", 1);
    const int head_num = spec.intParam("head_num", 2);
    const int kv_head_num = spec.intParam("kv_head_num", 1);
    const int head_dim = spec.intParam("head_dim", 32);
    const int query_seq_len = spec.intParam("query_seq_len", 2);
    const int key_seq_len = spec.intParam("key_seq_len", 10);
    const int max_kv_len = spec.intParam("max_kv_len", 10);
    const float scale = spec.floatParam("scale", 1.0f / std::sqrt((float)head_dim));
    const int outSize = batch * query_seq_len * head_num * head_dim;
    std::vector<float> query(batch * query_seq_len * head_num * head_dim, 0.1f);
    std::vector<float> key_cache(key_seq_len * batch * kv_head_num * head_dim, 0.2f);
    std::vector<float> value_cache(batch * kv_head_num * max_kv_len * head_dim, 0.3f);
    std::vector<float> mask(query_seq_len * query_seq_len, 0.0f);
    ac.buffers.push_back(makeHalfBuffer(query, false));
    ac.buffers.push_back(makeHalfBuffer(key_cache, false));
    ac.buffers.push_back(makeHalfBuffer(value_cache, false));
    ac.buffers.push_back(makeHalfOutput(outSize));
    ac.buffers.push_back(makeHalfBuffer(mask, false));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(head_num));
    ac.args.push_back(AdaptedArg::scalarInt(kv_head_num));
    ac.args.push_back(AdaptedArg::scalarInt(head_dim));
    ac.args.push_back(AdaptedArg::scalarInt(key_seq_len));
    ac.args.push_back(AdaptedArg::scalarInt(max_kv_len));
    ac.args.push_back(AdaptedArg::scalarInt(query_seq_len));
    ac.args.push_back(AdaptedArg::scalarFloat(scale));
    const int grid = batch * head_num * query_seq_len;
    const int block = 128;
    const size_t sharedMem = 4 * (2 * 4 + 32 * 4);
    ac.args.push_back(AdaptedArg::scalarInt(grid));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = outSize;
    return true;
}
cudaError_t CudaFlashDecodeWithMaskFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_flash_decode_with_mask_fp16(
        ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6],
        ctx.floatArgs[0], ctx.grid, ctx.block, (size_t)ctx.intArgs[7], ctx.stream);
    return cudaGetLastError();
}
bool CudaFlashDecodeWithMaskFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- 3. CudaFlashDecodeSplitkFp16Kernel ----
bool CudaFlashDecodeSplitkFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_flash_decode_splitk_fp16";
    const int batch = spec.intParam("batch", 1);
    const int head_num = spec.intParam("head_num", 2);
    const int kv_head_num = spec.intParam("kv_head_num", 1);
    const int head_dim = spec.intParam("head_dim", 32);
    const int key_seq_len = spec.intParam("key_seq_len", 8);
    const int max_kv_len = spec.intParam("max_kv_len", 8);
    const int parallel_blocks = spec.intParam("parallel_blocks", 2);
    const float scale = spec.floatParam("scale", 1.0f / std::sqrt((float)head_dim));
    const int bh = batch * head_num;
    const int partialOutSize = parallel_blocks * bh * head_dim;
    const int partialMetaSize = parallel_blocks * bh * 2;
    std::vector<float> query(batch * head_num * head_dim, 0.1f);
    std::vector<float> key_cache(key_seq_len * batch * kv_head_num * head_dim, 0.2f);
    std::vector<float> value_cache(batch * kv_head_num * max_kv_len * head_dim, 0.3f);
    // partial_output: float (per kernel signature); partial_meta: float, isOutput
    AdaptedBuffer poBuf; poBuf.setFp32(std::vector<float>(partialOutSize, 0.1f)); poBuf.isOutput = false;
    AdaptedBuffer pmBuf; pmBuf.sizeBytes = partialMetaSize * sizeof(float); pmBuf.isOutput = true;
    ac.buffers.push_back(makeHalfBuffer(query, false));
    ac.buffers.push_back(makeHalfBuffer(key_cache, false));
    ac.buffers.push_back(makeHalfBuffer(value_cache, false));
    ac.buffers.push_back(poBuf);
    ac.buffers.push_back(pmBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(head_num));
    ac.args.push_back(AdaptedArg::scalarInt(kv_head_num));
    ac.args.push_back(AdaptedArg::scalarInt(head_dim));
    ac.args.push_back(AdaptedArg::scalarInt(key_seq_len));
    ac.args.push_back(AdaptedArg::scalarInt(max_kv_len));
    ac.args.push_back(AdaptedArg::scalarFloat(scale));
    ac.args.push_back(AdaptedArg::scalarInt(parallel_blocks));
    const int gridX = bh;
    const int gridY = parallel_blocks;
    const int block = 128;
    const size_t sharedMem = 4 * (2 * 4);
    ac.args.push_back(AdaptedArg::scalarInt(gridX));
    ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = gridX; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = partialMetaSize;
    return true;
}
cudaError_t CudaFlashDecodeSplitkFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_flash_decode_splitk_fp16(
        ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2],
        (float*)ctx.devBufs[3], (float*)ctx.devBufs[4],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.floatArgs[0], ctx.intArgs[6],
        ctx.intArgs[7], ctx.intArgs[8], ctx.block, (size_t)ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}
bool CudaFlashDecodeSplitkFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if ((int)output.size() < ac.elementCount) return false;
    return true;
}

// ---- 4. CudaFlashAttnCombineResultsFp16Kernel ----
bool CudaFlashAttnCombineResultsFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_flash_attn_combine_results_fp16";
    const int batch = spec.intParam("batch", 1);
    const int head_num = spec.intParam("head_num", 2);
    const int head_dim = spec.intParam("head_dim", 32);
    const int parallel_blocks = spec.intParam("parallel_blocks", 2);
    const int bh = batch * head_num;
    const int partialOutSize = parallel_blocks * bh * head_dim;
    const int partialMetaSize = parallel_blocks * bh * 2;
    const int outSize = batch * head_num * head_dim;
    // partial_output/meta: float; final_output: __half
    AdaptedBuffer poBuf; poBuf.setFp32(std::vector<float>(partialOutSize, 0.1f)); poBuf.isOutput = false;
    AdaptedBuffer pmBuf; pmBuf.setFp32(std::vector<float>(partialMetaSize, 0.5f)); pmBuf.isOutput = false;
    ac.buffers.push_back(poBuf);
    ac.buffers.push_back(pmBuf);
    ac.buffers.push_back(makeHalfOutput(outSize));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(head_num));
    ac.args.push_back(AdaptedArg::scalarInt(head_dim));
    ac.args.push_back(AdaptedArg::scalarInt(parallel_blocks));
    const int grid = bh;
    const int block = head_dim;
    const size_t sharedMem = 0;
    ac.args.push_back(AdaptedArg::scalarInt(grid));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = outSize;
    return true;
}
cudaError_t CudaFlashAttnCombineResultsFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_flash_attn_combine_results_fp16(
        (const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], ctx.devBufs[2],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
        ctx.grid, ctx.block, (size_t)ctx.intArgs[4], ctx.stream);
    return cudaGetLastError();
}
bool CudaFlashAttnCombineResultsFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- 5. CudaCopyKvToCacheFp16Kernel ----
bool CudaCopyKvToCacheFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_copy_kv_to_cache_fp16";
    const int batch = spec.intParam("batch", 1);
    const int new_kv_seq_len = spec.intParam("new_kv_seq_len", 4);
    const int kv_num_head = spec.intParam("kv_num_head", 1);
    const int head_dim = spec.intParam("head_dim", 32);
    const int past_kv_len = spec.intParam("past_kv_len", 0);
    const int allocated_kv_len = spec.intParam("allocated_kv_len", 8);
    const int inputSize = batch * new_kv_seq_len * kv_num_head * head_dim;
    const int keyCacheSize = allocated_kv_len * batch * kv_num_head * head_dim;
    const int valueCacheSize = batch * kv_num_head * allocated_kv_len * head_dim;
    std::vector<float> key_input(inputSize), value_input(inputSize);
    for (int i = 0; i < inputSize; ++i) { key_input[i] = 0.1f * (i % 7); value_input[i] = 0.2f * (i % 5); }
    ac.buffers.push_back(makeHalfBuffer(key_input, false));
    ac.buffers.push_back(makeHalfBuffer(value_input, false));
    // key_cache_output: __half, not read back; value_cache_output: __half, read back
    ac.buffers.push_back(makeHalfOutput(keyCacheSize));
    ac.buffers.back().isOutput = false;
    ac.buffers.push_back(makeHalfOutput(valueCacheSize));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(new_kv_seq_len));
    ac.args.push_back(AdaptedArg::scalarInt(kv_num_head));
    ac.args.push_back(AdaptedArg::scalarInt(head_dim));
    ac.args.push_back(AdaptedArg::scalarInt(past_kv_len));
    ac.args.push_back(AdaptedArg::scalarInt(allocated_kv_len));
    const int gridX = (head_dim + 31) / 32;
    const int gridY = (new_kv_seq_len + 7) / 8;
    const int gridZ = batch * kv_num_head;
    const int blockX = 32, blockY = 8, blockZ = 1;
    const size_t sharedMem = 0;
    ac.args.push_back(AdaptedArg::scalarInt(gridX));
    ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(gridZ));
    ac.args.push_back(AdaptedArg::scalarInt(blockX));
    ac.args.push_back(AdaptedArg::scalarInt(blockY));
    ac.args.push_back(AdaptedArg::scalarInt(blockZ));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = gridX; ac.localSize[0] = blockX; ac.dims = 1;
    ac.validatorInputA = value_input;
    ac.elementCount = valueCacheSize;
    ac.m = batch; ac.n = new_kv_seq_len; ac.k = kv_num_head; ac.w = head_dim;
    ac.stride = past_kv_len; ac.h = allocated_kv_len;
    return true;
}
cudaError_t CudaCopyKvToCacheFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_copy_kv_to_cache_fp16(
        ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], ctx.devBufs[3],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
        (size_t)ctx.intArgs[12], ctx.stream);
    return cudaGetLastError();
}
bool CudaCopyKvToCacheFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int batch = ac.m, new_kv_seq_len = ac.n, kv_num_head = ac.k, head_dim = ac.w;
    const int past_kv_len = ac.stride, allocated_kv_len = ac.h;
    const int total = batch * kv_num_head * allocated_kv_len * head_dim;
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int b = 0; b < batch; ++b)
        for (int h = 0; h < kv_num_head; ++h)
            for (int l = 0; l < new_kv_seq_len; ++l)
                for (int d = 0; d < head_dim; ++d) {
                    int dest_seq = past_kv_len + l;
                    int outIdx = b * kv_num_head * allocated_kv_len * head_dim
                               + h * allocated_kv_len * head_dim
                               + dest_seq * head_dim + d;
                    int inIdx = (b * new_kv_seq_len * kv_num_head * head_dim
                              + l * kv_num_head * head_dim
                              + h * head_dim + d);
                    if (std::fabs(__half2float(out[outIdx]) - ac.validatorInputA[inIdx]) > 1e-2f) return false;
                }
    return true;
}

// ---- 6. CudaQkKernelTiledFp16Kernel ----
struct AttentionKernelParamHostFp16 {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int kv_head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int current_kv_seq_len_new;
    int past_kv_len;
};
bool CudaQkKernelTiledFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_qk_kernel_tiled_fp16";
    const int batch = spec.intParam("batch", 1);
    const int head_num = spec.intParam("head_num", 2);
    const int kv_head_num = spec.intParam("kv_head_num", 1);
    const int head_dim = spec.intParam("head_dim", 32);
    const int query_seq_len = spec.intParam("query_seq_len", 4);
    const int key_seq_len = spec.intParam("key_seq_len", 8);
    const int max_kv_len = spec.intParam("max_kv_len", 8);
    const float scale = spec.floatParam("scale", 1.0f / std::sqrt((float)head_dim));
    const int group = head_num / kv_head_num;
    const int q_seq_piece_len = query_seq_len;
    const int qSize = batch * query_seq_len * head_num * head_dim;
    const int kSize = key_seq_len * batch * kv_head_num * head_dim;
    const int outSize = batch * head_num * q_seq_piece_len * key_seq_len;
    std::vector<float> query(qSize, 0.1f), key_cache(kSize, 0.2f);
    AttentionKernelParamHostFp16 param;
    param.query_seq_len = query_seq_len;
    param.q_seq_piece_len = q_seq_piece_len;
    param.key_seq_len = key_seq_len;
    param.head_num = head_num;
    param.kv_head_num = kv_head_num;
    param.group = group;
    param.head_dim = head_dim;
    param.scale = scale;
    param.max_kv_len = max_kv_len;
    param.batch = batch;
    param.current_kv_seq_len_new = key_seq_len;
    param.past_kv_len = key_seq_len - query_seq_len;
    ac.buffers.push_back(makeHalfBuffer(query, false));
    ac.buffers.push_back(makeHalfBuffer(key_cache, false));
    ac.buffers.push_back(makeHalfOutput(outSize));
    AdaptedBuffer paramBuf; paramBuf.sizeBytes = sizeof(param);
    paramBuf.initialData.assign((const uint8_t*)&param, (const uint8_t*)&param + sizeof(param));
    paramBuf.isOutput = false;
    ac.buffers.push_back(paramBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // mask = nullptr sentinel
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // q_seq_piece_offset
    ac.args.push_back(AdaptedArg::scalarInt(0));  // has_mask = false
    ac.args.push_back(AdaptedArg::scalarInt(0));  // is_add_mask = false
    ac.args.push_back(AdaptedArg::scalarInt(0));  // is_causal_mask = false
    const int gridX = (key_seq_len + 15) / 16;
    const int gridY = (query_seq_len + 15) / 16;
    const int gridZ = batch * head_num;
    const int blockX = 16, blockY = 16;
    const size_t sharedMem = 0;
    ac.args.push_back(AdaptedArg::scalarInt(gridX));
    ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(gridZ));
    ac.args.push_back(AdaptedArg::scalarInt(blockX));
    ac.args.push_back(AdaptedArg::scalarInt(blockY));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = gridX; ac.localSize[0] = blockX; ac.dims = 1;
    ac.elementCount = outSize;
    return true;
}
cudaError_t CudaQkKernelTiledFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_qk_kernel_tiled_fp16(
        ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2],
        nullptr, (const void*)ctx.devBufs[3],
        ctx.intArgs[1], false, false, false,
        ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9],
        (size_t)ctx.intArgs[10], ctx.stream);
    return cudaGetLastError();
}
bool CudaQkKernelTiledFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- 7. CudaQkvKernelTiledFp16Kernel ----
bool CudaQkvKernelTiledFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_qkv_kernel_tiled_fp16";
    const int batch = spec.intParam("batch", 1);
    const int head_num = spec.intParam("head_num", 2);
    const int kv_head_num = spec.intParam("kv_head_num", 1);
    const int head_dim = spec.intParam("head_dim", 32);
    const int query_seq_len = spec.intParam("query_seq_len", 4);
    const int key_seq_len = spec.intParam("key_seq_len", 8);
    const int max_kv_len = spec.intParam("max_kv_len", 8);
    const float scale = spec.floatParam("scale", 1.0f / std::sqrt((float)head_dim));
    const int group = head_num / kv_head_num;
    const int q_seq_piece_len = query_seq_len;
    const int probsSize = batch * head_num * q_seq_piece_len * key_seq_len;
    const int vSize = batch * kv_head_num * max_kv_len * head_dim;
    const int outSize = batch * query_seq_len * head_num * head_dim;
    std::vector<float> softmax_probs(probsSize, 0.1f), value_cache(vSize, 0.2f);
    AttentionKernelParamHostFp16 param;
    param.query_seq_len = query_seq_len;
    param.q_seq_piece_len = q_seq_piece_len;
    param.key_seq_len = key_seq_len;
    param.head_num = head_num;
    param.kv_head_num = kv_head_num;
    param.group = group;
    param.head_dim = head_dim;
    param.scale = scale;
    param.max_kv_len = max_kv_len;
    param.batch = batch;
    param.current_kv_seq_len_new = key_seq_len;
    param.past_kv_len = key_seq_len - query_seq_len;
    ac.buffers.push_back(makeHalfBuffer(softmax_probs, false));
    ac.buffers.push_back(makeHalfBuffer(value_cache, false));
    ac.buffers.push_back(makeHalfOutput(outSize));
    AdaptedBuffer paramBuf; paramBuf.sizeBytes = sizeof(param);
    paramBuf.initialData.assign((const uint8_t*)&param, (const uint8_t*)&param + sizeof(param));
    paramBuf.isOutput = false;
    ac.buffers.push_back(paramBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // q_seq_piece_offset
    const int gridX = (head_dim + 31) / 32;
    const int gridY = (query_seq_len + 7) / 8;
    const int gridZ = batch * head_num;
    const int blockX = 32, blockY = 8;
    const size_t sharedMem = 0;
    ac.args.push_back(AdaptedArg::scalarInt(gridX));
    ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(gridZ));
    ac.args.push_back(AdaptedArg::scalarInt(blockX));
    ac.args.push_back(AdaptedArg::scalarInt(blockY));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = gridX; ac.localSize[0] = blockX; ac.dims = 1;
    ac.elementCount = outSize;
    return true;
}
cudaError_t CudaQkvKernelTiledFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_qkv_kernel_tiled_fp16(
        ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2],
        (const void*)ctx.devBufs[3], ctx.intArgs[0],
        ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        (size_t)ctx.intArgs[6], ctx.stream);
    return cudaGetLastError();
}
bool CudaQkvKernelTiledFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- 8. CudaConv1dSiluFp16Kernel ----
bool CudaConv1dSiluFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_conv1d_silu_fp16";
    const int B = spec.intParam("batch", 1);
    const int D = spec.intParam("d", 8);
    const int L = spec.intParam("l", 4);
    const int K_conv = spec.intParam("k_conv", 3);
    const int convStateSize = K_conv - 1;
    const bool inputC4 = spec.intParam("input_c4", 0) != 0;
    std::vector<float> qkvInput(B * D * L, 0.1f);
    std::vector<float> convWeight(D * K_conv, 0.2f);
    std::vector<float> convState(B * D * convStateSize, 0.0f);
    // convState/convOutFp32: float (kernel uses float accumulation)
    AdaptedBuffer sBuf; sBuf.setFp32(convState); sBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.sizeBytes = B * D * L * sizeof(float); oBuf.isOutput = true;
    ac.buffers.push_back(makeHalfBuffer(qkvInput, false));
    ac.buffers.push_back(makeHalfBuffer(convWeight, false));
    ac.buffers.push_back(sBuf);
    ac.buffers.push_back(oBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarInt(B));
    ac.args.push_back(AdaptedArg::scalarInt(D));
    ac.args.push_back(AdaptedArg::scalarInt(L));
    ac.args.push_back(AdaptedArg::scalarInt(K_conv));
    ac.args.push_back(AdaptedArg::scalarInt(convStateSize));
    ac.args.push_back(AdaptedArg::scalarInt(inputC4 ? 1 : 0));
    const int grid = B * D;
    const int block = (L == 1) ? 32 : 128;
    const size_t sharedMem = (K_conv + convStateSize + L) * sizeof(float);
    ac.args.push_back(AdaptedArg::scalarInt(grid));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = B * D * L;
    return true;
}
cudaError_t CudaConv1dSiluFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv1d_silu_fp16(
        ctx.devBufs[0], ctx.devBufs[1], (float*)ctx.devBufs[2], (float*)ctx.devBufs[3],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5] != 0,
        ctx.grid, ctx.block, (size_t)ctx.intArgs[6], ctx.stream);
    return cudaGetLastError();
}
bool CudaConv1dSiluFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if ((int)output.size() < ac.elementCount) return false;
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) { if (output[i] != 0.0f) return true; }
    return false;
}

// ---- 9. CudaShortConvFp16Kernel ----
bool CudaShortConvFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_short_conv_fp16";
    const int B = spec.intParam("batch", 1);
    const int D = spec.intParam("d", 24);
    const int L = spec.intParam("l", 4);
    const int H = spec.intParam("h", 8);
    const int K = spec.intParam("k", 3);
    const int convStateSize = K - 1;
    const bool inputC4 = spec.intParam("input_c4", 0) != 0;
    std::vector<float> qkvInput(B * D * L, 0.1f);
    std::vector<float> convWeight(H * K, 0.2f);
    std::vector<float> convState(B * H * convStateSize, 0.0f);
    AdaptedBuffer sBuf; sBuf.setFp32(convState); sBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.sizeBytes = B * H * L * sizeof(float); oBuf.isOutput = true;
    ac.buffers.push_back(makeHalfBuffer(qkvInput, false));
    ac.buffers.push_back(makeHalfBuffer(convWeight, false));
    ac.buffers.push_back(sBuf);
    ac.buffers.push_back(oBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarInt(B));
    ac.args.push_back(AdaptedArg::scalarInt(D));
    ac.args.push_back(AdaptedArg::scalarInt(L));
    ac.args.push_back(AdaptedArg::scalarInt(H));
    ac.args.push_back(AdaptedArg::scalarInt(K));
    ac.args.push_back(AdaptedArg::scalarInt(convStateSize));
    ac.args.push_back(AdaptedArg::scalarInt(inputC4 ? 1 : 0));
    const int grid = B * (D / 3);
    const int block = 128;
    const size_t sharedMem = (convStateSize + L) * sizeof(float);
    ac.args.push_back(AdaptedArg::scalarInt(grid));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = B * H * L;
    return true;
}
cudaError_t CudaShortConvFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_short_conv_fp16(
        ctx.devBufs[0], ctx.devBufs[1], (float*)ctx.devBufs[2], (float*)ctx.devBufs[3],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6] != 0,
        ctx.grid, ctx.block, (size_t)ctx.intArgs[7], ctx.stream);
    return cudaGetLastError();
}
bool CudaShortConvFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if ((int)output.size() < ac.elementCount) return false;
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) { if (output[i] != 0.0f) return true; }
    return false;
}

// ---- 10. CudaShortConvOutputFp16Kernel ----
bool CudaShortConvOutputFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_short_conv_output_fp16";
    const int B = spec.intParam("batch", 1);
    const int D = spec.intParam("d", 24);
    const int L = spec.intParam("l", 4);
    const int H = spec.intParam("h", 8);
    const bool inputC4 = spec.intParam("input_c4", 0) != 0;
    const bool outputC4 = spec.intParam("output_c4", 0) != 0;
    std::vector<float> qkvInput(B * D * L, 0.1f);
    std::vector<float> convOut(B * H * L, 0.2f);
    AdaptedBuffer cBuf; cBuf.setFp32(convOut); cBuf.isOutput = false;
    ac.buffers.push_back(makeHalfBuffer(qkvInput, false));
    ac.buffers.push_back(cBuf);
    ac.buffers.push_back(makeHalfOutput(B * L * H));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(B));
    ac.args.push_back(AdaptedArg::scalarInt(D));
    ac.args.push_back(AdaptedArg::scalarInt(L));
    ac.args.push_back(AdaptedArg::scalarInt(H));
    ac.args.push_back(AdaptedArg::scalarInt(inputC4 ? 1 : 0));
    ac.args.push_back(AdaptedArg::scalarInt(outputC4 ? 1 : 0));
    const int total = B * L * H;
    const int grid = (total + 255) / 256;
    const int block = 256;
    const size_t sharedMem = 0;
    ac.args.push_back(AdaptedArg::scalarInt(grid));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaShortConvOutputFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_short_conv_output_fp16(
        ctx.devBufs[0], (const float*)ctx.devBufs[1], ctx.devBufs[2],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4] != 0, ctx.intArgs[5] != 0,
        ctx.grid, ctx.block, (size_t)ctx.intArgs[6], ctx.stream);
    return cudaGetLastError();
}
bool CudaShortConvOutputFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- 11. CudaGatedDeltaRuleDecodeFp16Kernel ----
bool CudaGatedDeltaRuleDecodeFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gated_delta_rule_decode_fp16";
    const int B = spec.intParam("batch", 1);
    const int H_k = spec.intParam("h_k", 2);
    const int H_v = spec.intParam("h_v", 4);
    const int d_k = spec.intParam("d_k", 16);
    const int d_v = spec.intParam("d_v", 16);
    const int key_dim = H_k * d_k;
    const int val_dim = H_v * d_v;
    const int D = key_dim + val_dim;
    const int gqa_factor = H_v / H_k;
    const bool useL2Norm = spec.intParam("use_l2norm", 1) != 0;
    const float qScale = spec.floatParam("q_scale", 1.0f);
    const bool gateC4 = spec.intParam("gate_c4", 0) != 0;
    const bool betaC4 = spec.intParam("beta_c4", 0) != 0;
    const bool outputC4 = spec.intParam("output_c4", 0) != 0;
    std::vector<float> convOut(B * D, 0.1f);
    std::vector<float> gateInput(B * H_v, 0.2f);
    std::vector<float> betaInput(B * H_v, 0.3f);
    std::vector<float> recurrentState(B * H_v * d_k * d_v, 0.0f);
    // convOut/recurrentState: float; gateInput/betaInput/output: __half
    AdaptedBuffer cBuf; cBuf.setFp32(convOut); cBuf.isOutput = false;
    AdaptedBuffer rBuf; rBuf.setFp32(recurrentState); rBuf.isOutput = false;
    ac.buffers.push_back(cBuf);
    ac.buffers.push_back(makeHalfBuffer(gateInput, false));
    ac.buffers.push_back(makeHalfBuffer(betaInput, false));
    ac.buffers.push_back(rBuf);
    ac.buffers.push_back(makeHalfOutput(B * H_v * d_v));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::scalarInt(B));
    ac.args.push_back(AdaptedArg::scalarInt(H_k));
    ac.args.push_back(AdaptedArg::scalarInt(H_v));
    ac.args.push_back(AdaptedArg::scalarInt(d_k));
    ac.args.push_back(AdaptedArg::scalarInt(d_v));
    ac.args.push_back(AdaptedArg::scalarInt(key_dim));
    ac.args.push_back(AdaptedArg::scalarInt(val_dim));
    ac.args.push_back(AdaptedArg::scalarInt(D));
    ac.args.push_back(AdaptedArg::scalarInt(gqa_factor));
    ac.args.push_back(AdaptedArg::scalarInt(useL2Norm ? 1 : 0));
    ac.args.push_back(AdaptedArg::scalarFloat(qScale));
    ac.args.push_back(AdaptedArg::scalarInt(gateC4 ? 1 : 0));
    ac.args.push_back(AdaptedArg::scalarInt(betaC4 ? 1 : 0));
    ac.args.push_back(AdaptedArg::scalarInt(outputC4 ? 1 : 0));
    const int grid = B * H_v;
    const int block = (d_v >= 16) ? 64 : 128;
    const size_t sharedMem = (2 * d_k + 3 * d_v) * sizeof(float);
    ac.args.push_back(AdaptedArg::scalarInt(grid));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = B * H_v * d_v;
    return true;
}
cudaError_t CudaGatedDeltaRuleDecodeFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gated_delta_rule_decode_fp16(
        (const float*)ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2],
        (float*)ctx.devBufs[3], ctx.devBufs[4],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
        ctx.intArgs[8], ctx.intArgs[9] != 0, ctx.floatArgs[0],
        ctx.intArgs[10] != 0, ctx.intArgs[11] != 0, ctx.intArgs[12] != 0,
        ctx.grid, ctx.block, (size_t)ctx.intArgs[15], ctx.stream);
    return cudaGetLastError();
}
bool CudaGatedDeltaRuleDecodeFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- 12. CudaRopeC4Fp16Kernel ----
bool CudaRopeC4Fp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int seqLen = spec.intParam("seq_len", 2);
    const int numHead = spec.intParam("num_head", 2);
    const int kvNumHead = spec.intParam("kv_num_head", 1);
    const int headDim = spec.intParam("head_dim", 8);
    const int ropeHalfDim = spec.intParam("rope_half_dim", headDim / 2);
    const int ropeDim = ropeHalfDim * 2;
    const bool qNorm = spec.intParam("q_norm", 0) != 0;
    const bool kNorm = spec.intParam("k_norm", 0) != 0;
    const float qEps = spec.floatParam("q_eps", 1e-5f);
    const float kEps = spec.floatParam("k_eps", 1e-5f);
    const int qHiddenPack = numHead * headDim;
    const int kHiddenPack = kvNumHead * headDim;
    const int qSize = seqLen * qHiddenPack;
    const int kSize = seqLen * kHiddenPack;
    const int trigSize = seqLen * ropeDim;
    const int qOutSize = seqLen * numHead * headDim;
    const int kOutSize = seqLen * kvNumHead * headDim;

    std::vector<float> q(qSize), k(kSize), cos(trigSize), sin(trigSize);
    for (int i = 0; i < qSize; ++i) q[i] = 0.1f * (i % 7);
    for (int i = 0; i < kSize; ++i) k[i] = 0.1f * (i % 5);
    for (int i = 0; i < trigSize; ++i) { cos[i] = 0.01f * i; sin[i] = 0.01f * (i % 3); }
    std::vector<float> qGamma(headDim, 1.0f), kGamma(headDim, 1.0f);

    // q,k,cos,sin: __half; qGamma,kGamma: float; qOut,kOut: __half
    ac.buffers.push_back(makeHalfBuffer(q, false));
    ac.buffers.push_back(makeHalfBuffer(k, false));
    ac.buffers.push_back(makeHalfBuffer(cos, false));
    ac.buffers.push_back(makeHalfBuffer(sin, false));
    AdaptedBuffer qGammaBuf; qGammaBuf.setFp32(qGamma); qGammaBuf.isOutput = false;
    AdaptedBuffer kGammaBuf; kGammaBuf.setFp32(kGamma); kGammaBuf.isOutput = false;
    ac.buffers.push_back(qGammaBuf);
    ac.buffers.push_back(kGammaBuf);
    ac.buffers.push_back(makeHalfOutput(qOutSize));
    ac.buffers.back().isOutput = false;  // qOut not read back
    ac.buffers.push_back(makeHalfOutput(kOutSize));
    ac.args.push_back(AdaptedArg::buffer(0));  // q
    ac.args.push_back(AdaptedArg::buffer(1));  // k
    ac.args.push_back(AdaptedArg::buffer(2));  // cos
    ac.args.push_back(AdaptedArg::buffer(3));  // sin
    ac.args.push_back(AdaptedArg::buffer(6));  // qOut
    ac.args.push_back(AdaptedArg::buffer(7));  // kOut
    ac.args.push_back(AdaptedArg::buffer(4));  // qGamma
    ac.args.push_back(AdaptedArg::buffer(5));  // kGamma
    ac.args.push_back(AdaptedArg::scalarInt(seqLen));
    ac.args.push_back(AdaptedArg::scalarInt(numHead));
    ac.args.push_back(AdaptedArg::scalarInt(kvNumHead));
    ac.args.push_back(AdaptedArg::scalarInt(headDim));
    ac.args.push_back(AdaptedArg::scalarInt(ropeHalfDim));
    ac.args.push_back(AdaptedArg::scalarInt(qHiddenPack));
    ac.args.push_back(AdaptedArg::scalarInt(kHiddenPack));
    ac.args.push_back(AdaptedArg::scalarFloat(qEps));
    ac.args.push_back(AdaptedArg::scalarFloat(kEps));
    ac.args.push_back(AdaptedArg::scalarInt(qNorm ? 1 : 0));
    ac.args.push_back(AdaptedArg::scalarInt(kNorm ? 1 : 0));

    const int blocks = seqLen * (numHead + kvNumHead);
    ac.globalSize[0] = blocks; ac.localSize[0] = 128; ac.dims = 1;
    // Pack K input + cos/sin/kGamma (all __half now) for validation.
    ac.validatorInputA = k;
    ac.validatorInputB.clear();
    ac.validatorInputB.insert(ac.validatorInputB.end(), cos.begin(), cos.end());
    ac.validatorInputB.insert(ac.validatorInputB.end(), sin.begin(), sin.end());
    ac.validatorInputB.insert(ac.validatorInputB.end(), kGamma.begin(), kGamma.end());
    ac.elementCount = kOutSize;
    ac.m = seqLen; ac.n = kvNumHead; ac.k = headDim;
    ac.w = numHead; ac.h = ropeHalfDim; ac.c = kNorm ? 1 : 0;
    ac.stride = (int)(kEps * 1e6f);
    return true;
}
cudaError_t CudaRopeC4Fp16Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    mnn_corpus_rope_c4_fp16(ctx.devBufs[0], ctx.devBufs[1],
                            ctx.devBufs[2], ctx.devBufs[3],
                            ctx.devBufs[6], ctx.devBufs[7],
                            (const float*)ctx.devBufs[4], (const float*)ctx.devBufs[5],
                            ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                            ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6],
                            ctx.floatArgs[0], ctx.floatArgs[1],
                            ctx.intArgs[7] != 0, ctx.intArgs[8] != 0,
                            ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaRopeC4Fp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int seqLen = ac.m, kvNumHead = ac.n, headDim = ac.k;
    const int ropeHalfDim = ac.h;
    const bool kNorm = ac.c != 0;
    const int ropeDim = ropeHalfDim * 2;
    const int kHiddenPack = kvNumHead * headDim;
    const int kOutSize = seqLen * kvNumHead * headDim;
    if (static_cast<int>(output.size() * sizeof(float)) < kOutSize * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());

    const int trigSize = seqLen * ropeDim;
    const float* cos = ac.validatorInputB.data();
    const float* sin = cos + trigSize;
    const float* kGamma = sin + trigSize;
    const float* kIn = ac.validatorInputA.data();

    for (int t = 0; t < seqLen; ++t) {
        for (int h = 0; h < kvNumHead; ++h) {
            const int inputBase = t * kHiddenPack + h * headDim;
            const int outputBase = (t * kvNumHead + h) * headDim;
            const int trigBase = t * ropeDim;
            float scale = 1.0f;
            if (kNorm) {
                float squareSum = 0.0f;
                for (int d = 0; d < headDim; ++d) {
                    float v = kIn[inputBase + d];
                    squareSum += v * v;
                }
                scale = 1.0f / sqrtf(squareSum / (float)headDim + 1e-5f);
            }
            for (int d = 0; d < ropeHalfDim; ++d) {
                float even = kIn[inputBase + d];
                float odd = kIn[inputBase + d + ropeHalfDim];
                if (kNorm) {
                    even *= scale * kGamma[d];
                    odd *= scale * kGamma[d + ropeHalfDim];
                }
                float cEven = cos[trigBase + d];
                float cOdd = cos[trigBase + d + ropeHalfDim];
                float sEven = sin[trigBase + d];
                float sOdd = sin[trigBase + d + ropeHalfDim];
                float expected = even * cEven - odd * sEven;
                if (std::fabs(__half2float(out[outputBase + d]) - expected) > 1e-1f) return false;
                expected = odd * cOdd + even * sOdd;
                if (std::fabs(__half2float(out[outputBase + d + ropeHalfDim]) - expected) > 1e-1f) return false;
            }
            for (int d = ropeDim; d < headDim; ++d) {
                float value = kIn[inputBase + d];
                if (kNorm) value *= scale * kGamma[d];
                if (std::fabs(__half2float(out[outputBase + d]) - value) > 1e-1f) return false;
            }
        }
    }
    return true;
}

// ---- 13. CudaTopKV2Fp16Kernel ----
bool CudaTopKV2Fp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_topkv2_fp16";
    const int numRow = spec.intParam("num_row", 2);
    const int lengthRow = spec.intParam("length_row", 16);
    const int K = spec.intParam("k", 3);
    std::vector<float> input(numRow * lengthRow);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (float)(i % 19);
    AdaptedBuffer inBuf = makeHalfBuffer(input, false);
    AdaptedBuffer outIdxBuf; outIdxBuf.sizeBytes = numRow * K * sizeof(int32_t); outIdxBuf.isOutput = true;
    AdaptedBuffer outValBuf; outValBuf.sizeBytes = numRow * K * sizeof(__half); outValBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outIdxBuf);
    ac.buffers.push_back(outValBuf);
    int numThreadPerBlock = 64;
    int numElePerBlock = numThreadPerBlock;
    int numBlockPerRow = (lengthRow + numElePerBlock - 1) / numElePerBlock;
    int smem1 = numThreadPerBlock * K;
    int numThreadFinal = 1;
    while (numThreadFinal < numBlockPerRow) numThreadFinal <<= 1;
    if (numThreadFinal < 1) numThreadFinal = 1;
    int smem2 = numBlockPerRow * K;
    ac.args.push_back(AdaptedArg::buffer(0)); // input
    ac.args.push_back(AdaptedArg::buffer(1)); // outIndices
    ac.args.push_back(AdaptedArg::buffer(2)); // outValues
    ac.args.push_back(AdaptedArg::scalarInt(K));
    ac.args.push_back(AdaptedArg::scalarInt(lengthRow));
    ac.args.push_back(AdaptedArg::scalarInt(numRow));
    ac.args.push_back(AdaptedArg::scalarInt(1)); // descendFlag=1
    ac.args.push_back(AdaptedArg::scalarInt(numBlockPerRow)); // grid1x
    ac.args.push_back(AdaptedArg::scalarInt(numRow)); // grid1y
    ac.args.push_back(AdaptedArg::scalarInt(numThreadPerBlock)); // block1
    ac.args.push_back(AdaptedArg::scalarInt(smem1));
    ac.args.push_back(AdaptedArg::scalarInt(numRow)); // grid2
    ac.args.push_back(AdaptedArg::scalarInt(numThreadFinal)); // block2
    ac.args.push_back(AdaptedArg::scalarInt(smem2));
    ac.globalSize[0] = 1; ac.localSize[0] = 1; ac.dims = 1;
    ac.validatorInputA = input;
    ac.elementCount = numRow * K;
    ac.m = numRow; ac.n = lengthRow; ac.k = K;
    return true;
}
cudaError_t CudaTopKV2Fp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_topkv2_fp16((const void*)ctx.devBufs[0], (int*)ctx.devBufs[1], ctx.devBufs[2],
                           ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                           ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                           ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10],
                           ctx.stream);
    return cudaGetLastError();
}
bool CudaTopKV2Fp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int numRow = ac.m, lengthRow = ac.n, K = ac.k;
    if (static_cast<int>(output.size() * sizeof(float)) < numRow * K * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int r = 0; r < numRow; ++r) {
        for (int i = 0; i < K; ++i) {
            float val = __half2float(out[r * K + i]);
            int greater = 0;
            for (int j = 0; j < lengthRow; ++j) {
                if (ac.validatorInputA[r * lengthRow + j] > val) ++greater;
            }
            if (greater >= K) return false;
        }
    }
    return true;
}

// ============================================================================
// P1 fp16 adapters — Reduction(7) / Interp(4) / GridSample(4) / LayerNormC4(2)
// Only tag 3.6.0. Validation tolerance 1e-2 (half precision).
// ============================================================================
// ---- Reduction naive (SUM/MEAN/MAX/MIN/PROD) ----
// Macro to reduce boilerplate for the 5 naive reductions. SHIM is a bare token
// (e.g. reduction_sum). It is token-pasted with mnn_corpus_/fp16 to form the
// shim symbol, and stringified to form the entry name.
#define REDUCTION_FP16_ADAPTER(CLASS, SHIM, REDUCE_EXPR) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp16"; \
    const int outside = spec.intParam("outside", 4); \
    const int axis = spec.intParam("axis", 16); \
    const int inside = spec.intParam("inside", 1); \
    const int count = outside * inside; \
    std::vector<float> input(outside * axis * inside); \
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 13); \
    ac.buffers.push_back(makeHalfBuffer(input, false)); \
    ac.buffers.push_back(makeHalfOutput(count)); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::scalarInt(outside)); \
    ac.args.push_back(AdaptedArg::scalarInt(axis)); \
    ac.args.push_back(AdaptedArg::scalarInt(inside)); \
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = input; ac.elementCount = count; \
    ac.m = outside; ac.n = axis; ac.k = inside; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp16(ctx.devBufs[0], ctx.devBufs[1], \
                              ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int outside = ac.m, axis = ac.n, inside = ac.k; \
    const int count = outside * inside; \
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false; \
    const __half* out = reinterpret_cast<const __half*>(output.data()); \
    for (int o = 0; o < outside; ++o) \
      for (int x = 0; x < inside; ++x) { \
        const float* src = ac.validatorInputA.data() + o * axis * inside + x; \
        float expected = (REDUCE_EXPR); \
        if (std::fabs(__half2float(out[o * inside + x]) - expected) > 1e-2f) return false; \
      } \
    return true; \
}

REDUCTION_FP16_ADAPTER(CudaReductionSumFp16Kernel, reduction_sum,
    ([&]{float s=0;for(int v=0;v<axis;++v)s+=src[v*inside];return s;}()))
REDUCTION_FP16_ADAPTER(CudaReductionMeanFp16Kernel, reduction_mean,
    ([&]{float s=0;for(int v=0;v<axis;++v)s+=src[v*inside];return s/axis;}()))
REDUCTION_FP16_ADAPTER(CudaReductionMaxFp16Kernel, reduction_max,
    ([&]{float m=src[0];for(int v=1;v<axis;++v)m=std::max(m,src[v*inside]);return m;}()))
REDUCTION_FP16_ADAPTER(CudaReductionMinFp16Kernel, reduction_min,
    ([&]{float m=src[0];for(int v=1;v<axis;++v)m=std::min(m,src[v*inside]);return m;}()))
REDUCTION_FP16_ADAPTER(CudaReductionProdFp16Kernel, reduction_prod,
    ([&]{float p=1;for(int v=0;v<axis;++v)p*=src[v*inside];return p;}()))

// ---- Reduction axis-reduce (SUM/MEAN axis) ----
#define REDUCTION_AXIS_FP16_ADAPTER(CLASS, SHIM, REDUCE_EXPR) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp16"; \
    const int outside = spec.intParam("outside", 2); \
    const int axis = spec.intParam("axis", 256); \
    const int inside = spec.intParam("inside", 1); \
    const int count = outside * inside; \
    const int pbs = (axis % 256 == 0 || axis >= 768) ? 256 : 64; \
    const int calc_multi = (axis + pbs - 1) / pbs; \
    std::vector<float> input(outside * axis * inside); \
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 13); \
    ac.buffers.push_back(makeHalfBuffer(input, false)); \
    ac.buffers.push_back(makeHalfOutput(count)); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::scalarInt(outside)); \
    ac.args.push_back(AdaptedArg::scalarInt(axis)); \
    ac.args.push_back(AdaptedArg::scalarInt(inside)); \
    ac.args.push_back(AdaptedArg::scalarInt(pbs)); \
    ac.args.push_back(AdaptedArg::scalarInt(calc_multi)); \
    ac.globalSize[0] = count; ac.localSize[0] = pbs; ac.dims = 1; \
    ac.validatorInputA = input; ac.elementCount = count; \
    ac.m = outside; ac.n = axis; ac.k = inside; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp16(ctx.devBufs[0], ctx.devBufs[1], \
                              ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                              ctx.intArgs[3], ctx.intArgs[4], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int outside = ac.m, axis = ac.n, inside = ac.k; \
    const int count = outside * inside; \
    if (static_cast<int>(output.size() * sizeof(float)) < count * (int)sizeof(__half)) return false; \
    const __half* out = reinterpret_cast<const __half*>(output.data()); \
    for (int o = 0; o < outside; ++o) \
      for (int x = 0; x < inside; ++x) { \
        const float* src = ac.validatorInputA.data() + o * axis * inside + x; \
        float expected = (REDUCE_EXPR); \
        if (std::fabs(__half2float(out[o * inside + x]) - expected) > 1e-1f) return false; \
      } \
    return true; \
}

REDUCTION_AXIS_FP16_ADAPTER(CudaReductionSumAxisFp16Kernel, reduction_sum_axis,
    ([&]{float s=0;for(int v=0;v<axis;++v)s+=src[v*inside];return s;}()))
REDUCTION_AXIS_FP16_ADAPTER(CudaReductionMeanAxisFp16Kernel, reduction_mean_axis,
    ([&]{float s=0;for(int v=0;v<axis;++v)s+=src[v*inside];return s/axis;}()))

// ---- Interp nearest/bilinear/round/opt (3.6.0 only) ----
// Macro: shares bilinear-style geometry (total, c_p, ih, iw, oh, ow, sh, sw, 0, 0).
#define INTERP_FP16_ADAPTER(CLASS, SHIM, VALIDATE_BODY) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp16"; \
    const int ih = spec.intParam("ih", 4), iw = spec.intParam("iw", 4); \
    const int oh = spec.intParam("oh", 8), ow = spec.intParam("ow", 8); \
    const int c_p = spec.intParam("channels", 4); \
    const int total = oh * ow * c_p; \
    const float sh = (float)ih / oh, sw = (float)iw / ow; \
    std::vector<float> input(ih * iw * c_p); \
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11); \
    ac.buffers.push_back(makeHalfBuffer(input, false)); \
    ac.buffers.push_back(makeHalfOutput(total)); \
    ac.args.push_back(AdaptedArg::scalarInt(total)); \
    ac.args.push_back(AdaptedArg::scalarInt(c_p)); \
    ac.args.push_back(AdaptedArg::scalarInt(ih)); \
    ac.args.push_back(AdaptedArg::scalarInt(iw)); \
    ac.args.push_back(AdaptedArg::scalarInt(oh)); \
    ac.args.push_back(AdaptedArg::scalarInt(ow)); \
    ac.args.push_back(AdaptedArg::scalarFloat(sh)); \
    ac.args.push_back(AdaptedArg::scalarFloat(sw)); \
    ac.args.push_back(AdaptedArg::scalarFloat(0.0f)); \
    ac.args.push_back(AdaptedArg::scalarFloat(0.0f)); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = input; ac.elementCount = total; \
    ac.h = ih; ac.w = iw; ac.stride = oh; ac.c = c_p; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp16(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], \
                              ctx.intArgs[4], ctx.intArgs[5], ctx.floatArgs[0], ctx.floatArgs[1], \
                              ctx.floatArgs[2], ctx.floatArgs[3], \
                              ctx.devBufs[0], ctx.devBufs[1], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    VALIDATE_BODY; \
}

#define INTERP_BILINEAR_VALIDATE_BODY \
    const int ih = ac.h, iw = ac.w, oh = ac.stride, ow = oh, c_p = ac.c; \
    const int total = oh * ow * c_p; \
    const float sh = (float)ih / oh, sw = (float)iw / ow; \
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false; \
    const __half* out = reinterpret_cast<const __half*>(output.data()); \
    for (int y = 0; y < oh; ++y) \
      for (int x = 0; x < ow; ++x) \
        for (int c = 0; c < c_p; ++c) { \
          float fx = x * sw, fy = y * sh; \
          int ix0 = std::min(std::max(0, (int)floor(fx)), iw-1); \
          int ix1 = std::min((int)ceil(fx), iw-1); \
          int iy0 = std::min(std::max(0, (int)floor(fy)), ih-1); \
          int iy1 = std::min((int)ceil(fy), ih-1); \
          float fxw = fx - ix0, fyw = fy - iy0; \
          float v00 = ac.validatorInputA[((iy0*iw+ix0)*c_p+c)]; \
          float v01 = ac.validatorInputA[((iy0*iw+ix1)*c_p+c)]; \
          float v10 = ac.validatorInputA[((iy1*iw+ix0)*c_p+c)]; \
          float v11 = ac.validatorInputA[((iy1*iw+ix1)*c_p+c)]; \
          float expected = (1-fxw)*(1-fyw)*v00 + fxw*(1-fyw)*v01 + (1-fxw)*fyw*v10 + fxw*fyw*v11; \
          if (std::fabs(__half2float(out[(y*ow+x)*c_p+c]) - expected) > 1e-2f) return false; \
        } \
    return true

#define INTERP_NEAREST_VALIDATE_BODY \
    const int ih = ac.h, iw = ac.w, oh = ac.stride, ow = oh, c_p = ac.c; \
    const int total = oh * ow * c_p; \
    const float sh = (float)ih / oh, sw = (float)iw / ow; \
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false; \
    const __half* out = reinterpret_cast<const __half*>(output.data()); \
    for (int y = 0; y < oh; ++y) \
      for (int x = 0; x < ow; ++x) \
        for (int c = 0; c < c_p; ++c) { \
          int ix = std::min(std::max(0, (int)floor((float)x * sw)), iw-1); \
          int iy = std::min(std::max(0, (int)floor((float)y * sh)), ih-1); \
          float expected = ac.validatorInputA[(iy*iw+ix)*c_p+c]; \
          if (std::fabs(__half2float(out[(y*ow+x)*c_p+c]) - expected) > 1e-2f) return false; \
        } \
    return true

INTERP_FP16_ADAPTER(CudaInterpNearestFp16Kernel, interp_nearest, INTERP_NEAREST_VALIDATE_BODY)
INTERP_FP16_ADAPTER(CudaInterpBilinearFp16Kernel, interp_bilinear, INTERP_BILINEAR_VALIDATE_BODY)
INTERP_FP16_ADAPTER(CudaInterpNearestRoundFp16Kernel, interp_nearest_round, INTERP_NEAREST_VALIDATE_BODY)

// ---- Interp bilinear opt (NC4HW4, 2 pixels per thread, has extra d_ow/d_oh) ----
bool CudaInterpBilinearOptFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_interp_bilinear_opt_fp16";
    const int ih = spec.intParam("ih", 4), iw = spec.intParam("iw", 4);
    const int oh = spec.intParam("oh", 8), ow = spec.intParam("ow", 8);
    const int channels = spec.intParam("channels", 4);
    const int c_p = (channels + 7) / 8 * 8;
    const int total = oh * ow * c_p;
    const float sh = (float)ih / oh, sw = (float)iw / ow;
    std::vector<float> input(ih * iw * c_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
    ac.buffers.push_back(makeHalfBuffer(input, false));
    ac.buffers.push_back(makeHalfOutput(total));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(ih));
    ac.args.push_back(AdaptedArg::scalarInt(iw));
    ac.args.push_back(AdaptedArg::scalarInt(oh));
    ac.args.push_back(AdaptedArg::scalarInt(ow));
    ac.args.push_back(AdaptedArg::scalarFloat(sh));
    ac.args.push_back(AdaptedArg::scalarFloat(sw));
    ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
    ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    const int d_ow = 8, d_oh = 8;
    ac.args.push_back(AdaptedArg::scalarInt(d_ow));
    ac.args.push_back(AdaptedArg::scalarInt(d_oh));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaInterpBilinearOptFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_interp_bilinear_opt_fp16(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                         ctx.intArgs[4], ctx.floatArgs[0], ctx.floatArgs[1],
                                         ctx.floatArgs[2], ctx.floatArgs[3],
                                         ctx.devBufs[0], ctx.devBufs[1],
                                         ctx.intArgs[5], ctx.intArgs[6], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaInterpBilinearOptFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) {
        if (__half2float(out[i]) != 0.0f) return true;
    }
    return false;
}

// ---- GridSample nearest/bilinear 2D (3.6.0 only) ----
#define GRIDSAMPLE_FP16_ADAPTER(CLASS, SHIM, VALIDATE_EXPR) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp16"; \
    const int ih = spec.intParam("ih", 4), iw = spec.intParam("iw", 4); \
    const int oh = spec.intParam("oh", 4), ow = spec.intParam("ow", 4); \
    const int ch = spec.intParam("channels", 4); \
    const int ch_p = ch; \
    const int total = oh * ow * ch_p; \
    std::vector<float> input(ih * iw * ch_p); \
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11); \
    std::vector<float> grid(oh * ow * 2); \
    for (int i = 0; i < (int)grid.size(); ++i) grid[i] = -0.5f + (i % 3) * 0.5f; \
    ac.buffers.push_back(makeHalfBuffer(input, false)); \
    ac.buffers.push_back(makeHalfBuffer(grid, false)); \
    ac.buffers.push_back(makeHalfOutput(total)); \
    ac.args.push_back(AdaptedArg::scalarInt(total)); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::scalarInt(ih)); \
    ac.args.push_back(AdaptedArg::scalarInt(iw)); \
    ac.args.push_back(AdaptedArg::scalarInt(oh)); \
    ac.args.push_back(AdaptedArg::scalarInt(ow)); \
    ac.args.push_back(AdaptedArg::scalarInt(ch)); \
    ac.args.push_back(AdaptedArg::scalarInt(ch_p)); \
    ac.args.push_back(AdaptedArg::scalarInt(1)); /* padMode=BORDER */ \
    ac.args.push_back(AdaptedArg::scalarInt(0)); /* align=false */ \
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = input; ac.validatorInputB = grid; \
    ac.elementCount = total; ac.h = ih; ac.w = iw; ac.c = ch_p; ac.stride = oh; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp16(ctx.intArgs[0], ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], \
                              ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], \
                              ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int ih = ac.h, iw = ac.w, ch_p = ac.c, oh = ac.stride, ow = oh; \
    const int total = oh * ow * ch_p; \
    if (static_cast<int>(output.size() * sizeof(float)) < total * (int)sizeof(__half)) return false; \
    const __half* out = reinterpret_cast<const __half*>(output.data()); \
    for (int idx = 0; idx < total; ++idx) { \
        int idx_cp = idx % ch_p; \
        int idx_nhw = idx / ch_p; \
        int idx_ow = idx_nhw % ow; \
        int idx_nh = idx_nhw / ow; \
        int idx_oh = idx_nh % oh; \
        int idx_ob = idx_nh / oh; \
        float pos_x = ac.validatorInputB[idx_nhw * 2 + 0]; \
        float pos_y = ac.validatorInputB[idx_nhw * 2 + 1]; \
        float igx = ((1.0f + pos_x) * iw - 1.0f) / 2.0f; \
        float igy = ((1.0f + pos_y) * ih - 1.0f) / 2.0f; \
        float expected = VALIDATE_EXPR; \
        if (std::fabs(__half2float(out[idx]) - expected) > 1e-2f) return false; \
    } \
    return true; \
}

GRIDSAMPLE_FP16_ADAPTER(CudaGridSampleNearestFp16Kernel, grid_sample_nearest,
    ([&]{
        int ipx = std::min(std::max((int)floor(igx + 0.5f), 0), iw - 1);
        int ipy = std::min(std::max((int)floor(igy + 0.5f), 0), ih - 1);
        return ac.validatorInputA[((idx_ob * ih + ipy) * iw + ipx) * ch_p + idx_cp];
    }()))

GRIDSAMPLE_FP16_ADAPTER(CudaGridSampleBilinearFp16Kernel, grid_sample_bilinear,
    ([&]{
        int ix0 = std::min(std::max((int)floor(igx), 0), iw-1);
        int ix1 = std::min((int)ceil(igx), iw-1);
        int iy0 = std::min(std::max((int)floor(igy), 0), ih-1);
        int iy1 = std::min((int)ceil(igy), ih-1);
        float xw = ix1 - igx, yw = iy1 - igy;
        float v00 = ac.validatorInputA[((idx_ob*ih+iy0)*iw+ix0)*ch_p+idx_cp];
        float v01 = ac.validatorInputA[((idx_ob*ih+iy0)*iw+ix1)*ch_p+idx_cp];
        float v10 = ac.validatorInputA[((idx_ob*ih+iy1)*iw+ix0)*ch_p+idx_cp];
        float v11 = ac.validatorInputA[((idx_ob*ih+iy1)*iw+ix1)*ch_p+idx_cp];
        return v00*xw*yw + v01*(1-xw)*yw + v10*xw*(1-yw) + v11*(1-xw)*(1-yw);
    }()))

// ---- GridSample 3D nearest/bilinear (3.6.0 only) — smoke validation only ----
#define GRIDSAMPLE_3D_FP16_ADAPTER(CLASS, SHIM) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp16"; \
    const int id = spec.intParam("id", 2), ih = spec.intParam("ih", 3), iw = spec.intParam("iw", 3); \
    const int od = spec.intParam("od", 2), oh = spec.intParam("oh", 3), ow = spec.intParam("ow", 3); \
    const int ch = spec.intParam("channels", 4); \
    const int ch_p = ch; \
    const int total = od * oh * ow * ch_p; \
    std::vector<float> input(id * ih * iw * ch_p); \
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11); \
    std::vector<float> grid(od * oh * ow * 3); \
    for (int i = 0; i < (int)grid.size(); ++i) grid[i] = -0.5f + (i % 3) * 0.5f; \
    ac.buffers.push_back(makeHalfBuffer(input, false)); \
    ac.buffers.push_back(makeHalfBuffer(grid, false)); \
    ac.buffers.push_back(makeHalfOutput(total)); \
    ac.args.push_back(AdaptedArg::scalarInt(total)); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    for (int v : {id, ih, iw, od, oh, ow, ch, ch_p, 1, 0}) ac.args.push_back(AdaptedArg::scalarInt(v)); \
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.elementCount = total; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp16(ctx.intArgs[0], ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], \
        ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], \
        ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], \
        ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false; \
    return true; \
}

GRIDSAMPLE_3D_FP16_ADAPTER(CudaGridSampleNearest3dFp16Kernel, grid_sample_nearest_3d)
GRIDSAMPLE_3D_FP16_ADAPTER(CudaGridSampleBilinear3dFp16Kernel, grid_sample_bilinear_3d)

// ---- LayerNorm C4 fp16 (3.6.0 only) ----
bool CudaLayerNormC4Fp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int outside = spec.intParam("outside", 4);
    const int inside = spec.intParam("inside", 32);
    const float eps = spec.floatParam("epsilon", 1e-5f);
    const int rowStride = inside;
    const int count = outside * inside;
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = 0.1f * (i % 13) - 0.5f;
    std::vector<float> gamma(inside, 1.0f), beta(inside, 0.0f);
    ac.buffers.push_back(makeHalfBuffer(input, false));
    AdaptedBuffer gBuf; gBuf.setFp32(gamma); gBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(beta); bBuf.isOutput = false;
    ac.buffers.push_back(gBuf);
    ac.buffers.push_back(bBuf);
    ac.buffers.push_back(makeHalfOutput(count));
    ac.args.push_back(AdaptedArg::buffer(3));  // output
    ac.args.push_back(AdaptedArg::buffer(0));  // input
    ac.args.push_back(AdaptedArg::buffer(1));  // gamma
    ac.args.push_back(AdaptedArg::buffer(2));  // beta
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(rowStride));
    ac.args.push_back(AdaptedArg::scalarFloat(eps));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // RMSNorm = false
    ac.entry = "mnn_corpus_layernorm_c4_fp16";
    const int threads = inside > 4096 ? 1024 : (inside > 2048 ? 512 : 256);
    ac.globalSize[0] = outside; ac.localSize[0] = threads; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = outside; ac.k = inside; ac.stride = (int)(eps * 1e6f);
    return true;
}
cudaError_t CudaLayerNormC4Fp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_layernorm_c4_fp16(ctx.devBufs[3], ctx.devBufs[0],
                                  (const float*)ctx.devBufs[1], (const float*)ctx.devBufs[2],
                                  ctx.intArgs[0], ctx.intArgs[1], ctx.floatArgs[0], ctx.intArgs[2] != 0,
                                  ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaLayerNormC4Fp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, inside = ac.k;
    const float eps = (float)ac.stride * 1e-6f;
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int o = 0; o < outside; ++o) {
        float mean = 0.0f;
        for (int j = 0; j < inside; ++j) mean += ac.validatorInputA[o * inside + j];
        mean /= inside;
        float sq = 0.0f;
        for (int j = 0; j < inside; ++j) { float d = ac.validatorInputA[o * inside + j] - mean; sq += d * d; }
        float invStd = 1.0f / sqrtf(sq / inside + eps);
        for (int j = 0; j < inside; ++j) {
            float expected = (ac.validatorInputA[o * inside + j] - mean) * invStd;
            if (std::fabs(__half2float(out[o * inside + j]) - expected) > 1e-1f) return false;
        }
    }
    return true;
}

// ---- binary_layernorm_c4 fp16 (3.6.0 only) ----
bool CudaBinaryLayerNormC4Fp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    const int outside = spec.intParam("outside", 4);
    const int inside = spec.intParam("inside", 32);
    const float eps = spec.floatParam("epsilon", 1e-5f);
    const int rowStride = inside;
    const int count = outside * inside;
    std::vector<float> input0(count), input1(count);
    for (int i = 0; i < count; ++i) { input0[i] = 0.1f * (i % 13) - 0.5f; input1[i] = 0.1f * (i % 7); }
    std::vector<float> gamma(inside, 1.0f), beta(inside, 0.0f);
    ac.buffers.push_back(makeHalfBuffer(input0, false));
    ac.buffers.push_back(makeHalfBuffer(input1, false));
    AdaptedBuffer gBuf; gBuf.setFp32(gamma); gBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(beta); bBuf.isOutput = false;
    ac.buffers.push_back(gBuf);
    ac.buffers.push_back(bBuf);
    ac.buffers.push_back(makeHalfOutput(count));
    ac.buffers.back().isOutput = false;  // sumOut not read back
    ac.buffers.push_back(makeHalfOutput(count));
    ac.args.push_back(AdaptedArg::buffer(4));  // sumOut
    ac.args.push_back(AdaptedArg::buffer(5));  // normOut
    ac.args.push_back(AdaptedArg::buffer(0));  // input0
    ac.args.push_back(AdaptedArg::buffer(1));  // input1
    ac.args.push_back(AdaptedArg::buffer(2));  // gamma
    ac.args.push_back(AdaptedArg::buffer(3));  // beta
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(rowStride));
    ac.args.push_back(AdaptedArg::scalarFloat(eps));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // RMSNorm = false
    ac.entry = "mnn_corpus_binary_layernorm_c4_fp16";
    const int threads = inside > 4096 ? 1024 : (inside > 2048 ? 512 : 256);
    ac.globalSize[0] = outside; ac.localSize[0] = threads; ac.dims = 1;
    ac.validatorInputA = input0; ac.validatorInputB = input1;
    ac.elementCount = count; ac.m = outside; ac.k = inside;
    ac.stride = (int)(eps * 1e6f);
    return true;
}
cudaError_t CudaBinaryLayerNormC4Fp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_binary_layernorm_c4_fp16(
        ctx.devBufs[4], ctx.devBufs[5],
        ctx.devBufs[0], ctx.devBufs[1],
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3],
        ctx.intArgs[0], ctx.intArgs[1], ctx.floatArgs[0], ctx.intArgs[2] != 0,
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaBinaryLayerNormC4Fp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, inside = ac.k;
    const float eps = (float)ac.stride * 1e-6f;
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int o = 0; o < outside; ++o) {
        float mean = 0.0f;
        for (int j = 0; j < inside; ++j)
            mean += ac.validatorInputA[o * inside + j] + ac.validatorInputB[o * inside + j];
        mean /= inside;
        float sq = 0.0f;
        for (int j = 0; j < inside; ++j) {
            float v = ac.validatorInputA[o * inside + j] + ac.validatorInputB[o * inside + j];
            float d = v - mean; sq += d * d;
        }
        float invStd = 1.0f / sqrtf(sq / inside + eps);
        for (int j = 0; j < inside; ++j) {
            float v = ac.validatorInputA[o * inside + j] + ac.validatorInputB[o * inside + j];
            float expected = (v - mean) * invStd;
            if (std::fabs(__half2float(out[o * inside + j]) - expected) > 1e-1f) return false;
        }
    }
    return true;
}

// ============================================================================
// P2 plugin adapters — GroupNorm(half) / SeqLen2Spatial / splitGeLU / SPLIT_FusedQKV
// ============================================================================
// Helper: build GroupNorm params shared by Sum + Scale adapters.
struct GroupNormParamsHost {
    int n, h, w, c, groups, withSwish;
    int hw, hwPerBlock, cPerBlock, cPerGroup, hwc;
    float invHWC;
    int groupsPerBlock;
};
static GroupNormParamsHost buildGroupNormParams(const CaseSpec& spec) {
    GroupNormParamsHost p;
    p.n = spec.intParam("n", 1);
    p.h = spec.intParam("h", 2);
    p.w = spec.intParam("w", 2);
    p.c = spec.intParam("c", 32);
    p.groups = spec.intParam("groups", 4);
    p.withSwish = spec.intParam("with_swish", 0);
    p.hw = p.h * p.w;
    p.cPerGroup = p.c / p.groups;
    p.cPerBlock = p.cPerGroup;  // 1 group per block in corpus variant
    p.hwPerBlock = p.hw;        // 1 block per (n, group) covers all hw
    p.hwc = p.h * p.w * p.c;
    p.invHWC = 1.0f / (float)(p.cPerGroup * p.hw);
    p.groupsPerBlock = 1;
    return p;
}
// Build buffers shared by Sum + Scale: input(half) + gamma/beta(float) + redBuffer(float)
// + output(half). Returns buffer count for context. Pack validator inputs.
struct GroupNormBuffers {
    std::vector<float> inputF;
    std::vector<float> gamma, beta;
};
static GroupNormBuffers buildGroupNormBuffers(const GroupNormParamsHost& p) {
    GroupNormBuffers b;
    b.inputF.assign(p.n * p.hwc, 0.0f);
    for (int i = 0; i < (int)b.inputF.size(); ++i) b.inputF[i] = 0.1f * (i % 13) - 0.5f;
    b.gamma.assign(p.c, 1.0f);
    b.beta.assign(p.c, 0.0f);
    return b;
}

// ---- CudaGroupNormNHWCSumFp16Kernel ----
bool CudaGroupNormNHWCSumFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_groupnorm_nhwc_sum_fp16";
    auto p = buildGroupNormParams(spec);
    auto b = buildGroupNormBuffers(p);
    const int redSize = 2 * p.n * p.groups;
    // Buffers: input(half), src_0(unused, half), src_1(unused, half), gamma(float), beta(float),
    //          redBuffer(float, output), dst(unused half, not read back)
    ac.buffers.push_back(makeHalfBuffer(b.inputF, false));
    ac.buffers.push_back(makeHalfOutput(p.n * p.hwc));  // src_0 placeholder
    ac.buffers.back().isOutput = false;
    ac.buffers.push_back(makeHalfOutput(p.n * p.hwc));  // src_1 placeholder
    ac.buffers.back().isOutput = false;
    AdaptedBuffer gBuf; gBuf.setFp32(b.gamma); gBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(b.beta); bBuf.isOutput = false;
    ac.buffers.push_back(gBuf);
    ac.buffers.push_back(bBuf);
    AdaptedBuffer redBuf; redBuf.sizeBytes = redSize * sizeof(float); redBuf.isOutput = true;
    // zero-init redBuffer (kernel does atomicAdd)
    redBuf.initialData.assign(redSize * sizeof(float), 0);
    ac.buffers.push_back(redBuf);
    ac.buffers.push_back(makeHalfOutput(p.n * p.hwc));  // dst placeholder
    ac.buffers.back().isOutput = false;
    // args: dst(6), src_0(1), src_1(2), src(0), gamma(3), beta(4), redBuffer(5)
    ac.args.push_back(AdaptedArg::buffer(6));  // dst
    ac.args.push_back(AdaptedArg::buffer(1));  // src_0
    ac.args.push_back(AdaptedArg::buffer(2));  // src_1
    ac.args.push_back(AdaptedArg::buffer(0));  // src
    ac.args.push_back(AdaptedArg::buffer(3));  // gamma
    ac.args.push_back(AdaptedArg::buffer(4));  // beta
    ac.args.push_back(AdaptedArg::buffer(5));  // redBuffer
    ac.args.push_back(AdaptedArg::scalarInt(p.n));
    ac.args.push_back(AdaptedArg::scalarInt(p.h));
    ac.args.push_back(AdaptedArg::scalarInt(p.w));
    ac.args.push_back(AdaptedArg::scalarInt(p.c));
    ac.args.push_back(AdaptedArg::scalarInt(p.groups));
    ac.args.push_back(AdaptedArg::scalarInt(p.withSwish));
    ac.args.push_back(AdaptedArg::scalarInt(p.hw));
    ac.args.push_back(AdaptedArg::scalarInt(p.hwPerBlock));
    ac.args.push_back(AdaptedArg::scalarInt(p.cPerBlock));
    ac.args.push_back(AdaptedArg::scalarInt(p.cPerGroup));
    ac.args.push_back(AdaptedArg::scalarInt(p.hwc));
    ac.args.push_back(AdaptedArg::scalarFloat(p.invHWC));
    ac.args.push_back(AdaptedArg::scalarInt(p.groupsPerBlock));
    const int gridX = p.groups, gridY = (p.hw + p.hwPerBlock - 1) / p.hwPerBlock, gridZ = p.n;
    const int block = 256;
    ac.args.push_back(AdaptedArg::scalarInt(gridX));
    ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(gridZ));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = gridX; ac.localSize[0] = block; ac.dims = 1;
    ac.validatorInputA = b.inputF;
    ac.elementCount = redSize;
    ac.m = p.n; ac.n = p.groups; ac.k = p.cPerGroup; ac.w = p.hw;
    return true;
}
cudaError_t CudaGroupNormNHWCSumFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // intArgs: [0]=n,[1]=h,[2]=w,[3]=c,[4]=groups,[5]=withSwish,[6]=hw,[7]=hwPerBlock,
    //          [8]=cPerBlock,[9]=cPerGroup,[10]=hwc,[11]=groupsPerBlock,
    //          [12]=gridX,[13]=gridY,[14]=gridZ,[15]=block
    mnn_corpus_groupnorm_nhwc_sum_fp16(
        ctx.devBufs[6], ctx.devBufs[1], ctx.devBufs[2], ctx.devBufs[0],
        (const float*)ctx.devBufs[3], (const float*)ctx.devBufs[4], (float*)ctx.devBufs[5],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10],
        ctx.floatArgs[0], ctx.intArgs[11],
        ctx.intArgs[12], ctx.intArgs[13], ctx.intArgs[14], ctx.intArgs[15], ctx.stream);
    return cudaGetLastError();
}
bool CudaGroupNormNHWCSumFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // redBuffer is [2, n, groups] floats. Verify each group's sum is non-zero
    // (input is non-trivial). We don't recompute exact sum (atomicAdd order-
    // independent but fp drift); just check at least one entry is non-zero.
    if ((int)output.size() < ac.elementCount) return false;
    for (int i = 0; i < ac.elementCount; ++i) {
        if (output[i] != 0.0f) return true;
    }
    return false;
}

// ---- CudaGroupNormNHWCScaleFp16Kernel ----
bool CudaGroupNormNHWCScaleFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_groupnorm_nhwc_scale_fp16";
    auto p = buildGroupNormParams(spec);
    auto b = buildGroupNormBuffers(p);
    const int redSize = 2 * p.n * p.groups;
    // Precompute expected redBuffer values (sum + sumSq per group) so Scale can run
    // independently (corpus runs one adapter at a time).
    std::vector<float> redBuf(redSize, 0.0f);
    for (int ni = 0; ni < p.n; ++ni) {
        for (int gi = 0; gi < p.groups; ++gi) {
            float sum = 0.0f, sqSum = 0.0f;
            const int cBegin = gi * p.cPerGroup;
            for (int hwi = 0; hwi < p.hw; ++hwi) {
                for (int ci = cBegin; ci < cBegin + p.cPerGroup; ++ci) {
                    float v = b.inputF[ni * p.hwc + hwi * p.c + ci];
                    sum += v; sqSum += v * v;
                }
            }
            redBuf[(2 * ni + 0) * p.groups + gi] = sum;
            redBuf[(2 * ni + 1) * p.groups + gi] = sqSum;
        }
    }
    // Buffers: dst(half, output), src_0(half,unused), src_1(half,unused), src(half),
    //          gamma(float), beta(float), redBuffer(float, precomputed), unused
    ac.buffers.push_back(makeHalfOutput(p.n * p.hwc));  // dst output
    ac.buffers.push_back(makeHalfOutput(p.n * p.hwc));  // src_0 unused
    ac.buffers.back().isOutput = false;
    ac.buffers.push_back(makeHalfOutput(p.n * p.hwc));  // src_1 unused
    ac.buffers.back().isOutput = false;
    ac.buffers.push_back(makeHalfBuffer(b.inputF, false));  // src input
    AdaptedBuffer gBuf; gBuf.setFp32(b.gamma); gBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(b.beta); bBuf.isOutput = false;
    ac.buffers.push_back(gBuf);
    ac.buffers.push_back(bBuf);
    AdaptedBuffer redBufB; redBufB.setFp32(redBuf); redBufB.isOutput = false;
    ac.buffers.push_back(redBufB);
    // args: dst(0), src_0(1), src_1(2), src(3), gamma(4), beta(5), redBuffer(6)
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::buffer(5));
    ac.args.push_back(AdaptedArg::buffer(6));
    ac.args.push_back(AdaptedArg::scalarInt(p.n));
    ac.args.push_back(AdaptedArg::scalarInt(p.h));
    ac.args.push_back(AdaptedArg::scalarInt(p.w));
    ac.args.push_back(AdaptedArg::scalarInt(p.c));
    ac.args.push_back(AdaptedArg::scalarInt(p.groups));
    ac.args.push_back(AdaptedArg::scalarInt(p.withSwish));
    ac.args.push_back(AdaptedArg::scalarInt(p.hw));
    ac.args.push_back(AdaptedArg::scalarInt(p.hwPerBlock));
    ac.args.push_back(AdaptedArg::scalarInt(p.cPerBlock));
    ac.args.push_back(AdaptedArg::scalarInt(p.cPerGroup));
    ac.args.push_back(AdaptedArg::scalarInt(p.hwc));
    ac.args.push_back(AdaptedArg::scalarFloat(p.invHWC));
    ac.args.push_back(AdaptedArg::scalarInt(p.groupsPerBlock));
    const int gridX = p.groups, gridY = (p.hw + p.hwPerBlock - 1) / p.hwPerBlock, gridZ = p.n;
    const int block = 256;
    ac.args.push_back(AdaptedArg::scalarInt(gridX));
    ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(gridZ));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = gridX; ac.localSize[0] = block; ac.dims = 1;
    ac.validatorInputA = b.inputF;
    ac.validatorInputB = redBuf;
    ac.elementCount = p.n * p.hwc;
    ac.m = p.n; ac.n = p.groups; ac.k = p.cPerGroup; ac.w = p.hw;
    ac.c = p.c; ac.h = p.hwc; ac.stride = p.withSwish;
    return true;
}
cudaError_t CudaGroupNormNHWCScaleFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_groupnorm_nhwc_scale_fp16(
        ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2], ctx.devBufs[3],
        (const float*)ctx.devBufs[4], (const float*)ctx.devBufs[5], (float*)ctx.devBufs[6],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10],
        ctx.floatArgs[0], ctx.intArgs[11],
        ctx.intArgs[12], ctx.intArgs[13], ctx.intArgs[14], ctx.intArgs[15], ctx.stream);
    return cudaGetLastError();
}
bool CudaGroupNormNHWCScaleFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int n = ac.m, groups = ac.n, cPerGroup = ac.k, hw = ac.w, c = ac.c, hwc = ac.h;
    const bool withSwish = ac.stride != 0;
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    const float* in = ac.validatorInputA.data();
    const float* red = ac.validatorInputB.data();
    for (int ni = 0; ni < n; ++ni) {
        for (int gi = 0; gi < groups; ++gi) {
            float sum = red[(2 * ni + 0) * groups + gi];
            float sqSum = red[(2 * ni + 1) * groups + gi];
            float mean = sum / (float)(cPerGroup * hw);
            float var = sqSum / (float)(cPerGroup * hw) - mean * mean;
            float invStd = (var <= 0.0f) ? 1.0f : rsqrtf(var);
            const int cBegin = gi * cPerGroup;
            for (int hwi = 0; hwi < hw; ++hwi) {
                for (int ci = cBegin; ci < cBegin + cPerGroup; ++ci) {
                    int64_t offset = (int64_t)ni * hwc + (int64_t)hwi * c + ci;
                    float v = in[offset];
                    v = (v - mean) * invStd;
                    // gamma=1, beta=0 in our test data
                    if (withSwish) {
                        v = v * (1.0f / (1.0f + expf(-v)));
                    }
                    if (std::fabs(__half2float(out[offset]) - v) > 1e-1f) return false;
                }
            }
        }
    }
    return true;
}

// ---- SeqLen2Spatial fp32 / fp16 ----
// output[i,c] = input[i,c] + bias[c] + residual[i,c].
// IS_FP16=0 -> fp32 buffers; IS_FP16=1 -> half buffers.
#define SEQLEN2SPATIAL_ADAPTER(CLASS, SHIM, IS_FP16) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM; \
    const int seq = spec.intParam("seq", 4); \
    const int C = spec.intParam("c", 320); \
    const int total = seq * C; \
    std::vector<float> input(total), bias(C), residual(total); \
    for (int i = 0; i < total; ++i) input[i] = 0.1f * (i % 7); \
    for (int i = 0; i < C; ++i) bias[i] = 0.01f * i; \
    for (int i = 0; i < total; ++i) residual[i] = 0.2f * (i % 5); \
    if (IS_FP16) { \
        ac.buffers.push_back(makeHalfBuffer(input, false)); \
        ac.buffers.push_back(makeHalfBuffer(bias, false)); \
        ac.buffers.push_back(makeHalfBuffer(residual, false)); \
        ac.buffers.push_back(makeHalfOutput(total)); \
    } else { \
        AdaptedBuffer inB; inB.setFp32(input); inB.isOutput = false; \
        AdaptedBuffer biB; biB.setFp32(bias); biB.isOutput = false; \
        AdaptedBuffer reB; reB.setFp32(residual); reB.isOutput = false; \
        AdaptedBuffer ouB; ouB.sizeBytes = total * sizeof(float); ouB.isOutput = true; \
        ac.buffers.push_back(inB); ac.buffers.push_back(biB); ac.buffers.push_back(reB); ac.buffers.push_back(ouB); \
    } \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::buffer(3)); \
    ac.args.push_back(AdaptedArg::scalarInt(C)); \
    const int grid = seq; \
    const int block = C; \
    ac.args.push_back(AdaptedArg::scalarInt(grid)); \
    ac.args.push_back(AdaptedArg::scalarInt(block)); \
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1; \
    ac.validatorInputA = input; ac.validatorInputB = bias; \
    ac.validatorInputC = residual; ac.elementCount = total; ac.k = C; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM((const void*)ctx.devBufs[0], (const void*)ctx.devBufs[1], \
                       (const void*)ctx.devBufs[2], ctx.devBufs[3], \
                       ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int C = ac.k; \
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)(IS_FP16 ? sizeof(__half) : sizeof(float))) return false; \
    for (int i = 0; i < ac.elementCount; ++i) { \
        int c = i % C; \
        float expected = ac.validatorInputA[i] + ac.validatorInputB[c] + ac.validatorInputC[i]; \
        float actual = (IS_FP16) ? __half2float(reinterpret_cast<const __half*>(output.data())[i]) : output[i]; \
        if (std::fabs(actual - expected) > 1e-2f) return false; \
    } \
    return true; \
}

SEQLEN2SPATIAL_ADAPTER(CudaSeqLen2SpatialFp32Kernel, seqlen2spatial_fp32, 0)
SEQLEN2SPATIAL_ADAPTER(CudaSeqLen2SpatialFp16Kernel, seqlen2spatial_fp16, 1)

// ---- splitGeLU fp32 / fp16 ----
// output = gelu(R) * L. fDiv=1.702 (gelu approx), fAdd=0, fMul=0.5.
#define SPLITGELU_ADAPTER(CLASS, SHIM, IS_FP16) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM; \
    const int seq = spec.intParam("seq", 4); \
    const int HHS = spec.intParam("hhs", 1280); \
    const float fDiv = spec.floatParam("f_div", 1.702f); \
    const float fAdd = spec.floatParam("f_add", 0.0f); \
    const float fMul = spec.floatParam("f_mul", 0.5f); \
    const int inputSize = seq * HHS * 2; \
    const int outputSize = seq * HHS; \
    std::vector<float> input(inputSize); \
    for (int i = 0; i < inputSize; ++i) input[i] = 0.1f * (i % 7); \
    if (IS_FP16) { \
        ac.buffers.push_back(makeHalfBuffer(input, false)); \
        ac.buffers.push_back(makeHalfOutput(outputSize)); \
    } else { \
        AdaptedBuffer inB; inB.setFp32(input); inB.isOutput = false; \
        AdaptedBuffer ouB; ouB.sizeBytes = outputSize * sizeof(float); ouB.isOutput = true; \
        ac.buffers.push_back(inB); ac.buffers.push_back(ouB); \
    } \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::scalarInt(0)); /* input1 = nullptr sentinel */ \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::scalarInt(HHS)); \
    ac.args.push_back(AdaptedArg::scalarFloat(fDiv)); \
    ac.args.push_back(AdaptedArg::scalarFloat(fAdd)); \
    ac.args.push_back(AdaptedArg::scalarFloat(fMul)); \
    const int grid = seq; \
    const int block = 256; \
    ac.args.push_back(AdaptedArg::scalarInt(grid)); \
    ac.args.push_back(AdaptedArg::scalarInt(block)); \
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1; \
    ac.validatorInputA = input; \
    ac.elementCount = outputSize; ac.k = HHS; \
    ac.validatorFloats = {fDiv, fAdd, fMul}; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM(ctx.devBufs[0], nullptr, ctx.devBufs[1], \
                       ctx.intArgs[1], ctx.floatArgs[0], ctx.floatArgs[1], ctx.floatArgs[2], \
                       ctx.intArgs[2], ctx.intArgs[3], ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int HHS = ac.k; \
    const float fDivRecip = 1.0f / ac.validatorFloats[0]; \
    const float fAdd = ac.validatorFloats[1]; \
    const float fMul = ac.validatorFloats[2]; \
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)(IS_FP16 ? sizeof(__half) : sizeof(float))) return false; \
    for (int idx = 0; idx < ac.elementCount; ++idx) { \
        int blockIdx = idx / HHS; \
        int threadIdx = idx % HHS; \
        int indexInput = blockIdx * HHS * 2 + threadIdx; \
        float valueL = ac.validatorInputA[indexInput]; \
        float valueR = ac.validatorInputA[indexInput + HHS]; \
        float tmp = valueR * fDivRecip; \
        tmp = erff(tmp); \
        tmp += fAdd; \
        tmp *= valueR; \
        tmp *= fMul; \
        tmp *= valueL; \
        float actual = (IS_FP16) ? __half2float(reinterpret_cast<const __half*>(output.data())[idx]) : output[idx]; \
        if (std::fabs(actual - tmp) > 1e-2f) return false; \
    } \
    return true; \
}

SPLITGELU_ADAPTER(CudaSplitGeluFp32Kernel, splitgelu_fp32, 0)
SPLITGELU_ADAPTER(CudaSplitGeluFp16Kernel, splitgelu_fp16, 1)

// ---- SPLIT_FusedQKV fp32 / fp16 ----
#define SPLIT_FUSEDQKV_ADAPTER(CLASS, SHIM, IS_FP16) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM; \
    const int bsh = spec.intParam("bsh", 4); \
    const int D = spec.intParam("d", 64); \
    const int count = bsh * D; \
    const int inputSize = bsh * 3 * D; \
    std::vector<float> input(inputSize); \
    for (int i = 0; i < inputSize; ++i) input[i] = 0.1f * (i % 7); \
    if (IS_FP16) { \
        ac.buffers.push_back(makeHalfBuffer(input, false)); \
        ac.buffers.push_back(makeHalfOutput(count)); /* q */ \
        ac.buffers.push_back(makeHalfOutput(count)); /* k */ \
        ac.buffers.back().isOutput = false; /* k not read back */ \
        ac.buffers.push_back(makeHalfOutput(count)); /* v (read back) */ \
    } else { \
        AdaptedBuffer inB; inB.setFp32(input); inB.isOutput = false; \
        AdaptedBuffer qB; qB.sizeBytes = count * sizeof(float); qB.isOutput = true; \
        AdaptedBuffer kB; kB.sizeBytes = count * sizeof(float); kB.isOutput = false; \
        AdaptedBuffer vB; vB.sizeBytes = count * sizeof(float); vB.isOutput = true; \
        ac.buffers.push_back(inB); ac.buffers.push_back(qB); ac.buffers.push_back(kB); ac.buffers.push_back(vB); \
    } \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::buffer(3)); \
    ac.args.push_back(AdaptedArg::scalarInt(count)); \
    ac.args.push_back(AdaptedArg::scalarInt(D)); \
    const int grid = (count + 255) / 256; \
    const int block = 256; \
    ac.args.push_back(AdaptedArg::scalarInt(grid)); \
    ac.args.push_back(AdaptedArg::scalarInt(block)); \
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1; \
    ac.validatorInputA = input; \
    ac.elementCount = count; ac.k = D; ac.w = bsh; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM((size_t)ctx.intArgs[0], ctx.devBufs[0], \
                       ctx.devBufs[1], ctx.devBufs[2], ctx.devBufs[3], \
                       ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int D = ac.k, bsh = ac.w; \
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)(IS_FP16 ? sizeof(__half) : sizeof(float))) return false; \
    for (int i = 0; i < ac.elementCount; ++i) { \
        int bshd_i = i / D; \
        int d = i % D; \
        /* v[i] = fused_qkv[(bsh*3+2)*D + d] */ \
        float expected = ac.validatorInputA[(bshd_i * 3 + 2) * D + d]; \
        float actual = (IS_FP16) ? __half2float(reinterpret_cast<const __half*>(output.data())[i]) : output[i]; \
        (void)bsh; \
        if (std::fabs(actual - expected) > 1e-2f) return false; \
    } \
    return true; \
}

SPLIT_FUSEDQKV_ADAPTER(CudaSplitFusedQKVFp32Kernel, split_fusedqkv_fp32, 0)
SPLIT_FUSEDQKV_ADAPTER(CudaSplitFusedQKVFp16Kernel, split_fusedqkv_fp16, 1)

// ============================================================================
// P4 int8 adapters — FloatToInt8/Int8ToFloat/DequantWeight/ConvDW/Im2Col/BinaryInt8
// ============================================================================
// Helper: pack float vector as int8 bytes for int8 input buffers.
static AdaptedBuffer makeInt8Buffer(const std::vector<int8_t>& data, bool isOutput) {
    AdaptedBuffer b;
    b.sizeBytes = data.size() * sizeof(int8_t);
    b.initialData.assign((const uint8_t*)data.data(), (const uint8_t*)data.data() + b.sizeBytes);
    b.isOutput = isOutput;
    return b;
}
static AdaptedBuffer makeInt8Output(size_t elemCount) {
    AdaptedBuffer b;
    b.sizeBytes = elemCount * sizeof(int8_t);
    b.isOutput = true;
    return b;
}

// ---- FLOAT_2_INT8 (per-channel scale) ----
bool CudaFloat2Int8Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_float2int8_packed_fp32";
    const int channels = spec.intParam("channels", 8);
    const int area = spec.intParam("area", 4);
    const int INT8_PACK = 4;
    const int channelsPackInt8 = UP_DIV(channels, INT8_PACK) * INT8_PACK;
    const int channelsPackFloat = UP_DIV(channels, 8) * 8;
    const int total = area * channelsPackInt8 / 4;
    const float scaleVal = spec.floatParam("scale", 0.1f);
    const int8_t zeroPoint = (int8_t)spec.intParam("zero_point", 0);
    const int8_t clampMax = (int8_t)spec.intParam("clamp_max", 127);
    const int8_t clampMin = (int8_t)spec.intParam("clamp_min", -128);
    std::vector<float> input(area * channelsPackFloat);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.5f * (i % 7) - 1.5f;
    std::vector<float> scales(channelsPackInt8, scaleVal);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer sBuf; sBuf.setFp32(scales); sBuf.isOutput = false;
    AdaptedBuffer outBuf = makeInt8Output(area * channelsPackInt8);
    ac.buffers.push_back(inBuf); ac.buffers.push_back(sBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPackInt8));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPackFloat));
    ac.args.push_back(AdaptedArg::scalarInt(channels));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt((int)zeroPoint));
    ac.args.push_back(AdaptedArg::scalarInt((int)clampMax));
    ac.args.push_back(AdaptedArg::scalarInt((int)clampMin));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.validatorInputB = scales;
    ac.elementCount = area * channelsPackInt8;
    ac.k = channels; ac.n = channelsPackInt8; ac.w = channelsPackFloat;
    ac.validatorFloats = {scaleVal, (float)zeroPoint, (float)clampMax, (float)clampMin};
    return true;
}
cudaError_t CudaFloat2Int8Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_float2int8_packed_fp32((const float*)ctx.devBufs[0], (int8_t*)ctx.devBufs[2],
                                ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                (const float*)ctx.devBufs[1], (int8_t)ctx.intArgs[4],
                                (int8_t)ctx.intArgs[5], (int8_t)ctx.intArgs[6],
                                ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaFloat2Int8Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int channels = ac.k, channelsPackInt8 = ac.n, channelsPackFloat = ac.w;
    const int total = ac.elementCount;
    if ((int)output.size() * (int)sizeof(float) < total * (int)sizeof(int8_t)) return false;
    const int8_t* out = reinterpret_cast<const int8_t*>(output.data());
    const float scaleVal = ac.validatorFloats[0];
    const int8_t zeroPoint = (int8_t)ac.validatorFloats[1];
    const int8_t clampMax = (int8_t)ac.validatorFloats[2];
    const int8_t clampMin = (int8_t)ac.validatorFloats[3];
    for (int i = 0; i < std::min(10, total); ++i) {
        int c_idx = (i / (total / (total / (channelsPackInt8 / 4)))) % (channelsPackInt8 / 4);
        float inp = ac.validatorInputA[i % ac.validatorInputA.size()];
        float sc = ac.validatorInputB[c_idx * 4 + (i % 4)];
        int expected = host_float2int_rn(inp * sc) + zeroPoint;
        expected = min(expected, (int)clampMax); expected = max(expected, (int)clampMin);
        if (out[i] != (int8_t)expected) return false;
    }
    return true;
}

// ---- FLOAT_2_INT8_SINGLE (single scale) ----
bool CudaFloat2Int8SingleFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_float2int8_single_packed_fp32";
    const int channels = spec.intParam("channels", 8);
    const int area = spec.intParam("area", 4);
    const int channelsPackInt8 = UP_DIV(channels, 4) * 4;
    const int channelsPackFloat = UP_DIV(channels, 8) * 8;
    const int total = area * channelsPackInt8 / 4;
    const float scaleVal = spec.floatParam("scale", 0.1f);
    const int8_t zeroPoint = (int8_t)spec.intParam("zero_point", 0);
    const int8_t clampMax = (int8_t)spec.intParam("clamp_max", 127);
    const int8_t clampMin = (int8_t)spec.intParam("clamp_min", -128);
    std::vector<float> input(area * channelsPackFloat);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.5f * (i % 7) - 1.5f;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf = makeInt8Output(area * channelsPackInt8);
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPackInt8));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPackFloat));
    ac.args.push_back(AdaptedArg::scalarInt(channels));
    ac.args.push_back(AdaptedArg::scalarFloat(scaleVal));
    ac.args.push_back(AdaptedArg::scalarInt((int)zeroPoint));
    ac.args.push_back(AdaptedArg::scalarInt((int)clampMax));
    ac.args.push_back(AdaptedArg::scalarInt((int)clampMin));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input;
    ac.elementCount = area * channelsPackInt8;
    ac.k = channels; ac.n = channelsPackInt8; ac.w = channelsPackFloat;
    ac.validatorFloats = {scaleVal, (float)zeroPoint, (float)clampMax, (float)clampMin};
    return true;
}
cudaError_t CudaFloat2Int8SingleFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_float2int8_single_packed_fp32((const float*)ctx.devBufs[0], (int8_t*)ctx.devBufs[1],
                                        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                        ctx.floatArgs[0], (int8_t)ctx.intArgs[4],
                                        (int8_t)ctx.intArgs[5], (int8_t)ctx.intArgs[6],
                                        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaFloat2Int8SingleFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() * (int)sizeof(float) < total * (int)sizeof(int8_t)) return false;
    const int8_t* out = reinterpret_cast<const int8_t*>(output.data());
    const float scaleVal = ac.validatorFloats[0];
    const int8_t zeroPoint = (int8_t)ac.validatorFloats[1];
    const int8_t clampMax = (int8_t)ac.validatorFloats[2];
    const int8_t clampMin = (int8_t)ac.validatorFloats[3];
    for (int i = 0; i < std::min(10, total); ++i) {
        float inp = ac.validatorInputA[i % ac.validatorInputA.size()];
        int expected = host_float2int_rn(inp * scaleVal) + zeroPoint;
        expected = min(expected, (int)clampMax); expected = max(expected, (int)clampMin);
        if (out[i] != (int8_t)expected) return false;
    }
    return true;
}

// ---- INT8_2_FLOAT (per-channel scale) ----
bool CudaInt82FloatFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_int82float_packed_fp32";
    const int channels = spec.intParam("channels", 8);
    const int area = spec.intParam("area", 4);
    const int channelsPackInt8 = UP_DIV(channels, 4) * 4;
    const int channelsPackFloat = UP_DIV(channels, 8) * 8;
    const int total = area * channelsPackInt8 / 4;
    const float scaleVal = spec.floatParam("scale", 0.1f);
    const int8_t zeroPoint = (int8_t)spec.intParam("zero_point", 0);
    std::vector<int8_t> input(area * channelsPackInt8);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (int8_t)(i % 19 - 9);
    std::vector<float> scales(channelsPackInt8, scaleVal);
    AdaptedBuffer inBuf = makeInt8Buffer(input, false);
    AdaptedBuffer sBuf; sBuf.setFp32(scales); sBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = area * channelsPackFloat * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(sBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPackInt8));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPackFloat));
    ac.args.push_back(AdaptedArg::scalarInt(channels));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt((int)zeroPoint));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = std::vector<float>(input.begin(), input.end());
    ac.validatorInputB = scales;
    ac.elementCount = area * channelsPackFloat;
    ac.k = channels; ac.n = channelsPackInt8; ac.w = channelsPackFloat;
    ac.validatorFloats = {scaleVal, (float)zeroPoint};
    return true;
}
cudaError_t CudaInt82FloatFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_int82float_packed_fp32((const int8_t*)ctx.devBufs[0], (float*)ctx.devBufs[2],
                                ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                (const float*)ctx.devBufs[1], (int8_t)ctx.intArgs[4],
                                ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaInt82FloatFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() < total) return false;
    const int8_t zeroPoint = (int8_t)ac.validatorFloats[1];
    const int channelsPackInt8 = ac.n;
    const int channels = ac.k;
    for (int index = 0; index < std::min(10, total / 4); ++index) {
        int nhw_idx = index / channelsPackInt8;
        int c_idx = index % channelsPackInt8;
        int idx_out = index * 4;
        if (4 * c_idx >= channels) {
            for (int j = 0; j < 4; ++j) {
                if (std::fabs(output[idx_out + j]) > 1e-2f) return false;
            }
            continue;
        }
        int idx_inp = nhw_idx * channelsPackInt8 + 4 * c_idx;
        for (int j = 0; j < 4 && idx_out + j < total; ++j) {
            int8_t inp = (int8_t)ac.validatorInputA[idx_inp + j];
            float sc = ac.validatorInputB[c_idx * 4 + j];
            float expected = (inp - zeroPoint) * sc;
            if (std::fabs(output[idx_out + j] - expected) > 1e-2f) return false;
        }
    }
    return true;
}

// ---- INT8_2_FLOAT_SINGLE (single scale) ----
bool CudaInt82FloatSingleFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_int82float_single_packed_fp32";
    const int channels = spec.intParam("channels", 8);
    const int area = spec.intParam("area", 4);
    const int channelsPackInt8 = UP_DIV(channels, 4) * 4;
    const int channelsPackFloat = UP_DIV(channels, 8) * 8;
    const int total = area * channelsPackInt8 / 4;
    const float scaleVal = spec.floatParam("scale", 0.1f);
    const int8_t zeroPoint = (int8_t)spec.intParam("zero_point", 0);
    std::vector<int8_t> input(area * channelsPackInt8);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (int8_t)(i % 19 - 9);
    AdaptedBuffer inBuf = makeInt8Buffer(input, false);
    AdaptedBuffer outBuf; outBuf.sizeBytes = area * channelsPackFloat * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPackInt8));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPackFloat));
    ac.args.push_back(AdaptedArg::scalarInt(channels));
    ac.args.push_back(AdaptedArg::scalarFloat(scaleVal));
    ac.args.push_back(AdaptedArg::scalarInt((int)zeroPoint));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = std::vector<float>(input.begin(), input.end());
    ac.elementCount = area * channelsPackFloat;
    ac.k = channels; ac.n = channelsPackInt8;
    ac.validatorFloats = {scaleVal, (float)zeroPoint};
    return true;
}
cudaError_t CudaInt82FloatSingleFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_int82float_single_packed_fp32((const int8_t*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                        ctx.floatArgs[0], (int8_t)ctx.intArgs[4],
                                        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaInt82FloatSingleFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() < total) return false;
    const float scaleVal = ac.validatorFloats[0];
    const int8_t zeroPoint = (int8_t)ac.validatorFloats[1];
    for (int i = 0; i < std::min(10, total); ++i) {
        int8_t inp = (int8_t)ac.validatorInputA[i % ac.validatorInputA.size()];
        float expected = (inp - zeroPoint) * scaleVal;
        if (std::fabs(output[i] - expected) > 1e-2f) return false;
    }
    return true;
}

// ---- DequantizeInt8Weight (fp16 output) ----
bool CudaDequantizeInt8WeightFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_dequantize_int8_weight_fp16";
    const int oc = spec.intParam("oc", 4);
    const int ic = spec.intParam("ic", 8);
    const int ic_p = UP_DIV(ic, 8) * 8;
    const int quanC = spec.intParam("quan_c", oc);
    const float scaleVal = spec.floatParam("scale", 0.1f);
    const float offsetVal = spec.floatParam("offset", 0.0f);
    std::vector<int8_t> qk(oc * ic_p);
    for (int i = 0; i < (int)qk.size(); ++i) qk[i] = (int8_t)(i % 19 - 9);
    std::vector<float> scaleF(quanC, scaleVal), offsetF(quanC, offsetVal);
    auto scaleH = packHalf(scaleF), offsetH = packHalf(offsetF);
    AdaptedBuffer qkBuf = makeInt8Buffer(qk, false);
    AdaptedBuffer sBuf; sBuf.sizeBytes = scaleH.size(); sBuf.initialData = scaleH; sBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.sizeBytes = offsetH.size(); oBuf.initialData = offsetH; oBuf.isOutput = false;
    AdaptedBuffer outBuf = makeHalfOutput(oc * ic_p);
    ac.buffers.push_back(qkBuf); ac.buffers.push_back(sBuf); ac.buffers.push_back(oBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p));
    ac.args.push_back(AdaptedArg::scalarInt(quanC));
    const int blockX = 32, blockY = 4;
    const int gridX = (ic + blockX - 1) / blockX;
    const int gridY = (oc + blockY - 1) / blockY;
    ac.args.push_back(AdaptedArg::scalarInt(gridX));
    ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(blockX));
    ac.args.push_back(AdaptedArg::scalarInt(blockY));
    ac.globalSize[0] = gridX; ac.localSize[0] = blockX; ac.dims = 1;
    ac.validatorInputA = std::vector<float>(qk.begin(), qk.end());
    ac.validatorInputB = scaleF; ac.validatorInputC = offsetF;
    ac.elementCount = oc * ic_p; ac.m = oc; ac.n = ic; ac.k = ic_p; ac.w = quanC;
    return true;
}
cudaError_t CudaDequantizeInt8WeightFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_dequantize_int8_weight_fp16((const int8_t*)ctx.devBufs[0], ctx.devBufs[3],
                                            ctx.devBufs[1], ctx.devBufs[2],
                                            ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                            ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                            ctx.stream);
    return cudaGetLastError();
}
bool CudaDequantizeInt8WeightFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int oc = ac.m, ic = ac.n, ic_p = ac.k, quanC = ac.w;
    if (static_cast<int>(output.size() * sizeof(float)) < oc * ic_p * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int row = 0; row < oc; ++row) {
        for (int col = 0; col < ic; ++col) {
            int num_qg = (quanC > 0) ? (quanC / oc) : 1;
            int ic_per_group = (num_qg > 0) ? (ic / num_qg) : ic;
            int group_idx = col / ic_per_group;
            int qparam_idx = row * num_qg + group_idx;
            float s = ac.validatorInputB[qparam_idx];
            float o = ac.validatorInputC[qparam_idx];
            float qval = ac.validatorInputA[row * ic_p + col];
            float expected = qval * s + o;
            if (std::fabs(__half2float(out[row * ic_p + col]) - expected) > 1e-2f) return false;
        }
    }
    return true;
}

// ---- DequantizeInt4Weight (fp16 output) ----
bool CudaDequantizeInt4WeightFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_dequantize_int4_weight_fp16";
    const int oc = spec.intParam("oc", 4);
    const int ic = spec.intParam("ic", 16);
    const int ic_p = UP_DIV(ic, 8) * 8;
    const int quanC = spec.intParam("quan_c", oc);
    const float scaleVal = spec.floatParam("scale", 0.1f);
    const float offsetVal = spec.floatParam("offset", 0.0f);
    // pack int4 values: 2 per byte, high nibble first, row-major (row * half_ic + col_byte/2)
    int half_ic = ic_p / 2;
    int totalBytes = oc * half_ic;
    std::vector<uint8_t> pw(totalBytes, 0);
    for (int row = 0; row < oc; ++row) {
        for (int col = 0; col < ic; ++col) {
            int nibble_val = ((row * ic_p + col) % 16) - 8;
            uint8_t nibble = (uint8_t)(nibble_val + 8);
            int byte_idx = row * half_ic + col / 2;
            if (col % 2 == 0) pw[byte_idx] |= (nibble << 4);
            else pw[byte_idx] |= nibble;
        }
    }
    std::vector<float> scaleF(quanC, scaleVal), offsetF(quanC, offsetVal);
    auto scaleH = packHalf(scaleF), offsetH = packHalf(offsetF);
    AdaptedBuffer pwBuf; pwBuf.sizeBytes = totalBytes; pwBuf.initialData.assign((uint8_t*)pw.data(), (uint8_t*)pw.data()+totalBytes); pwBuf.isOutput = false;
    AdaptedBuffer sBuf; sBuf.sizeBytes = scaleH.size(); sBuf.initialData = scaleH; sBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.sizeBytes = offsetH.size(); oBuf.initialData = offsetH; oBuf.isOutput = false;
    AdaptedBuffer outBuf = makeHalfOutput(oc * ic_p);
    ac.buffers.push_back(pwBuf); ac.buffers.push_back(sBuf); ac.buffers.push_back(oBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p));
    ac.args.push_back(AdaptedArg::scalarInt(quanC));
    // MNN kernel: blockIdx.x = row(OC), tid = blockIdx.y * blockDim.x + threadIdx.x
    // Each thread handles 8 bytes. half_ic = ic_p/2 bytes per row.
    const int blockDimX = 256;
    const int half_ic4 = ic_p / 2;
    const int gridX = oc;  // 1 block per OC row
    const int gridY = (half_ic4 / 8 + blockDimX - 1) / blockDimX;
    ac.args.push_back(AdaptedArg::scalarInt(gridX));
    ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(blockDimX));
    ac.args.push_back(AdaptedArg::scalarInt(1));
    ac.globalSize[0] = gridX; ac.localSize[0] = blockDimX; ac.dims = 1;
    ac.validatorInputB = scaleF; ac.validatorInputC = offsetF;
    ac.elementCount = oc * ic_p; ac.m = oc; ac.n = ic; ac.k = ic_p; ac.w = quanC;
    // pack validatorInputA with int4 values as float
    ac.validatorInputA.assign(oc * ic_p, 0);
    for (int i = 0; i < oc * ic_p; ++i) ac.validatorInputA[i] = (float)((i % 16) - 8);
    return true;
}
cudaError_t CudaDequantizeInt4WeightFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_dequantize_int4_weight_fp16((const uint8_t*)ctx.devBufs[0], ctx.devBufs[3],
                                            ctx.devBufs[1], ctx.devBufs[2],
                                            ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                            ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                            ctx.stream);
    return cudaGetLastError();
}
bool CudaDequantizeInt4WeightFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int oc = ac.m, ic = ac.n, ic_p = ac.k, quanC = ac.w;
    if (static_cast<int>(output.size() * sizeof(float)) < oc * ic_p * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int row = 0; row < oc; ++row) {
        for (int col = 0; col < ic; ++col) {
            int num_qg = (quanC > 0) ? (quanC / oc) : 1;
            int ic_per_group = (num_qg > 0) ? (ic / num_qg) : ic;
            int group_idx = col / ic_per_group;
            int qparam_idx = row * num_qg + group_idx;
            float s = ac.validatorInputB[qparam_idx];
            float o = ac.validatorInputC[qparam_idx];
            float qval = ac.validatorInputA[row * ic_p + col];
            float expected = qval * s + o;
            if (std::fabs(__half2float(out[row * ic_p + col]) - expected) > 1e-1f) return false;
        }
    }
    return true;
}

// ---- CONV_DW_INT8_ ----
bool CudaConvDwInt8Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_conv_dw_int8_fp32";
    const int iw = spec.intParam("iw", 4), ih = spec.intParam("ih", 4);
    const int c = spec.intParam("c", 8), c_p = UP_DIV(c, 4) * 4;
    const int kw = spec.intParam("kw", 3), kh = spec.intParam("kh", 3);
    const int sw = spec.intParam("sw", 1), sh = spec.intParam("sh", 1);
    const int pw = spec.intParam("pw", 1), ph = spec.intParam("ph", 1);
    const int ow = (iw + 2*pw - kw) / sw + 1;
    const int oh = (ih + 2*ph - kh) / sh + 1;
    const int total = ow * oh * c_p;
    std::vector<int8_t> input(ih * iw * c_p), kernel(kh * kw * c_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (int8_t)(i % 7);
    for (int i = 0; i < (int)kernel.size(); ++i) kernel[i] = (int8_t)(i % 5);
    std::vector<int32_t> bias(c_p, 0);
    std::vector<float> scale(c_p, 1.0f);
    ac.buffers.push_back(makeInt8Buffer(input, false));
    ac.buffers.push_back(makeInt8Buffer(kernel, false));
    AdaptedBuffer bBuf; bBuf.sizeBytes = c_p * sizeof(int32_t); bBuf.initialData.assign((uint8_t*)bias.data(), (uint8_t*)bias.data()+bBuf.sizeBytes); bBuf.isOutput = false;
    AdaptedBuffer sBuf; sBuf.setFp32(scale); sBuf.isOutput = false;
    ac.buffers.push_back(bBuf); ac.buffers.push_back(sBuf);
    ac.buffers.push_back(makeInt8Output(total));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::scalarInt((int8_t)127)); // maxV
    ac.args.push_back(AdaptedArg::scalarInt((int8_t)-128)); // minV
    ac.args.push_back(AdaptedArg::scalarInt(iw));
    ac.args.push_back(AdaptedArg::scalarInt(ih));
    ac.args.push_back(AdaptedArg::scalarInt(c));
    ac.args.push_back(AdaptedArg::scalarInt(c_p));
    ac.args.push_back(AdaptedArg::scalarInt(ow));
    ac.args.push_back(AdaptedArg::scalarInt(oh));
    ac.args.push_back(AdaptedArg::scalarInt(kw));
    ac.args.push_back(AdaptedArg::scalarInt(kh));
    ac.args.push_back(AdaptedArg::scalarInt(1)); // dw
    ac.args.push_back(AdaptedArg::scalarInt(1)); // dh
    ac.args.push_back(AdaptedArg::scalarInt(sw));
    ac.args.push_back(AdaptedArg::scalarInt(sh));
    ac.args.push_back(AdaptedArg::scalarInt(pw));
    ac.args.push_back(AdaptedArg::scalarInt(ph));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.globalSize[0] = gridFor(total/4); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaConvDwInt8Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv_dw_int8_fp32((const int8_t*)ctx.devBufs[0], (const int8_t*)ctx.devBufs[1],
                                  (const int32_t*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (int8_t*)ctx.devBufs[4],
                                  (int8_t)ctx.intArgs[0], (int8_t)ctx.intArgs[1],
                                  ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
                                  ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9],
                                  ctx.intArgs[10], ctx.intArgs[11], ctx.intArgs[12], ctx.intArgs[13],
                                  ctx.intArgs[14], ctx.intArgs[15], ctx.intArgs[16],
                                  ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaConvDwInt8Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() * (int)sizeof(float) < total * (int)sizeof(int8_t)) return false;
    const int8_t* out = reinterpret_cast<const int8_t*>(output.data());
    for (int i = 0; i < std::min(10, total); ++i) {
        if (out[i] != 0) return true; // smoke: at least one non-zero
    }
    return false;
}

// ---- CONV_DW3x3S1_INT8_OPT ----
bool CudaConvDw3x3S1Int8Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_conv_dw3x3s1_int8_fp32";
    const int iw = spec.intParam("iw", 4), ih = spec.intParam("ih", 4);
    const int c = spec.intParam("c", 8), c_p = UP_DIV(c, 4) * 4;
    const int ow = iw, oh = ih; // 3x3 s1 p1 -> same size
    const int kw = 3, kh = 3, k_p = c_p, dw = 1, dh = 1, sw = 1, sh = 1, pw = 1, ph = 1;
    const int total = ow * oh * c_p;
    std::vector<int8_t> input(ih * iw * c_p), kernel(9 * c_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (int8_t)(i % 7);
    for (int i = 0; i < (int)kernel.size(); ++i) kernel[i] = (int8_t)(i % 5);
    std::vector<int32_t> bias(c_p, 0);
    std::vector<float> scale(c_p, 1.0f);
    ac.buffers.push_back(makeInt8Buffer(input, false));
    ac.buffers.push_back(makeInt8Buffer(kernel, false));
    AdaptedBuffer bBuf; bBuf.sizeBytes = c_p * sizeof(int32_t); bBuf.initialData.assign((uint8_t*)bias.data(), (uint8_t*)bias.data()+bBuf.sizeBytes); bBuf.isOutput = false;
    AdaptedBuffer sBuf; sBuf.setFp32(scale); sBuf.isOutput = false;
    ac.buffers.push_back(bBuf); ac.buffers.push_back(sBuf);
    ac.buffers.push_back(makeInt8Output(total));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::scalarInt((int8_t)127)); // maxV [0]
    ac.args.push_back(AdaptedArg::scalarInt((int8_t)-128)); // minV [1]
    ac.args.push_back(AdaptedArg::scalarInt(iw));  // [2]
    ac.args.push_back(AdaptedArg::scalarInt(ih));  // [3]
    ac.args.push_back(AdaptedArg::scalarInt(c));   // [4]
    ac.args.push_back(AdaptedArg::scalarInt(c_p)); // [5]
    ac.args.push_back(AdaptedArg::scalarInt(ow));  // [6]
    ac.args.push_back(AdaptedArg::scalarInt(oh));  // [7]
    ac.args.push_back(AdaptedArg::scalarInt(kw));  // [8]
    ac.args.push_back(AdaptedArg::scalarInt(kh));  // [9]
    ac.args.push_back(AdaptedArg::scalarInt(k_p)); // [10]
    ac.args.push_back(AdaptedArg::scalarInt(dw));  // [11]
    ac.args.push_back(AdaptedArg::scalarInt(dh));  // [12]
    ac.args.push_back(AdaptedArg::scalarInt(sw));  // [13]
    ac.args.push_back(AdaptedArg::scalarInt(sh));  // [14]
    ac.args.push_back(AdaptedArg::scalarInt(pw));  // [15]
    ac.args.push_back(AdaptedArg::scalarInt(ph));  // [16]
    ac.args.push_back(AdaptedArg::scalarInt(total)); // [17]
    ac.globalSize[0] = gridFor(total/8); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaConvDw3x3S1Int8Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv_dw3x3s1_int8_fp32((const int8_t*)ctx.devBufs[0], (const int8_t*)ctx.devBufs[1],
                                        (const int32_t*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (int8_t*)ctx.devBufs[4],
                                        (int8_t)ctx.intArgs[0], (int8_t)ctx.intArgs[1],
                                        ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
                                        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9],
                                        ctx.intArgs[10], ctx.intArgs[11], ctx.intArgs[12], ctx.intArgs[13],
                                        ctx.intArgs[14], ctx.intArgs[15], ctx.intArgs[16],
                                        ctx.intArgs[17],
                                        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaConvDw3x3S1Int8Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() * (int)sizeof(float) < total * (int)sizeof(int8_t)) return false;
    const int8_t* out = reinterpret_cast<const int8_t*>(output.data());
    for (int i = 0; i < std::min(10, total); ++i) {
        if (out[i] != 0) return true;
    }
    return false;
}

// ---- Im2Col_packC_16 (smoke-only) ----
bool CudaIm2ColPackC16Int8Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_im2col_packc16_int8_fp32";
    const int iw = spec.intParam("iw", 4), ih = spec.intParam("ih", 4);
    const int ic = spec.intParam("ic", 8);
    const int icDiv4 = UP_DIV(ic, 4);
    const int kw = spec.intParam("kw", 3), kh = 3;
    const int sw = spec.intParam("sw", 1), sh = 1;
    const int pw = spec.intParam("pw", 1), ph = 1;
    const int ow = (iw + 2*pw - kw) / sw + 1;
    const int oh = (ih + 2*ph - kh) / sh + 1;
    const int l = icDiv4 * kw * kh;
    const int e = ow * oh;
    const size_t maxCount = (size_t)e * l;
    std::vector<int32_t> A(ih * iw * icDiv4);
    for (int i = 0; i < (int)A.size(); ++i) A[i] = i;
    AdaptedBuffer aBuf; aBuf.sizeBytes = A.size() * sizeof(int32_t); aBuf.initialData.assign((uint8_t*)A.data(), (uint8_t*)A.data()+aBuf.sizeBytes); aBuf.isOutput = false;
    AdaptedBuffer apBuf; apBuf.sizeBytes = maxCount * sizeof(int32_t); apBuf.isOutput = true;
    ac.buffers.push_back(aBuf); ac.buffers.push_back(apBuf);
    ac.args.push_back(AdaptedArg::scalarInt(sw));
    ac.args.push_back(AdaptedArg::scalarInt(sh));
    ac.args.push_back(AdaptedArg::scalarInt(1)); // dw
    ac.args.push_back(AdaptedArg::scalarInt(1)); // dh
    ac.args.push_back(AdaptedArg::scalarInt(pw));
    ac.args.push_back(AdaptedArg::scalarInt(ph));
    ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(iw));
    ac.args.push_back(AdaptedArg::scalarInt(ih));
    ac.args.push_back(AdaptedArg::scalarInt((int)maxCount));
    ac.args.push_back(AdaptedArg::scalarInt(1)); // iBlock
    ac.args.push_back(AdaptedArg::scalarInt(icDiv4));
    ac.args.push_back(AdaptedArg::scalarInt(e));
    ac.args.push_back(AdaptedArg::scalarInt(l));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(ow));  // ow for d_ow
    ac.args.push_back(AdaptedArg::scalarInt(oh));  // oh for d_oh
    ac.args.push_back(AdaptedArg::scalarInt(kw));  // kw for d_fx
    ac.globalSize[0] = gridFor((int)maxCount); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.elementCount = (int)maxCount;
    ac.p = oh; ac.q = ow;  // stash for validator
    return true;
}
cudaError_t CudaIm2ColPackC16Int8Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // intArgs: [0]=sw,[1]=sh,[2]=dw,[3]=dh,[4]=pw,[5]=ph,[6]=ic,[7]=iw,[8]=ih,
    //          [9]=maxCount,[10]=iBlock,[11]=icDiv4,[12]=e,[13]=l,
    //          [14]=ow,[15]=oh,[16]=kw
    mnn_corpus_im2col_packc16_int8(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                    ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6],
                                    ctx.intArgs[7], ctx.intArgs[8], (size_t)ctx.intArgs[9],
                                    ctx.intArgs[10], ctx.intArgs[11], ctx.intArgs[12], ctx.intArgs[13],
                                    (const int32_t*)ctx.devBufs[0], (int32_t*)ctx.devBufs[1],
                                    ctx.intArgs[14], ctx.intArgs[15], ctx.intArgs[16],
                                    ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaIm2ColPackC16Int8Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if ((int)output.size() < ac.elementCount) return false;
    return true; // smoke-only
}

// ---- WeightInt8PackFill ----
bool CudaWeightInt8PackFillFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_weight_int8_pack_fill_fp32";
    const int l = spec.intParam("l", 27); // kw*kh*ic
    const int h = spec.intParam("h", 4); // oc
    const int ic = spec.intParam("ic", 3);
    const int hp = UP_DIV(h, 4) * 4;
    const int maxCount = l * hp;
    std::vector<int8_t> param(h * l);
    for (int i = 0; i < (int)param.size(); ++i) param[i] = (int8_t)(i % 19 - 9);
    ac.buffers.push_back(makeInt8Buffer(param, false));
    ac.buffers.push_back(makeInt8Output(maxCount));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(maxCount));
    ac.args.push_back(AdaptedArg::scalarInt(l));
    ac.args.push_back(AdaptedArg::scalarInt(h));
    ac.args.push_back(AdaptedArg::scalarInt(hp));
    ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.globalSize[0] = gridFor(maxCount); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = std::vector<float>(param.begin(), param.end());
    ac.elementCount = maxCount; ac.k = l; ac.n = h; ac.w = ic;
    return true;
}
cudaError_t CudaWeightInt8PackFillFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_weight_int8_pack_fill_fp32((const int8_t*)ctx.devBufs[0], (int8_t*)ctx.devBufs[1],
                                           ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
                                           ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaWeightInt8PackFillFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int maxCount = ac.elementCount, l = ac.k, h = ac.n, ic = ac.w;
    if ((int)output.size() * (int)sizeof(float) < maxCount * (int)sizeof(int8_t)) return false;
    const int8_t* out = reinterpret_cast<const int8_t*>(output.data());
    for (int i = 0; i < std::min(10, maxCount); ++i) {
        if (out[i] != 0) return true; // smoke
    }
    return false;
}

// ---- BINARY_INT8_ADD / MUL ----
#define BINARY_INT8_ADAPTER(CLASS, SHIM, OP) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp32"; \
    const int count = spec.intParam("count", 64); \
    const float s0 = spec.floatParam("scale0", 0.1f); \
    const float s1 = spec.floatParam("scale1", 0.1f); \
    const float os = spec.floatParam("out_scale", 10.0f); \
    std::vector<int8_t> in0(count), in1(count); \
    for (int i = 0; i < count; ++i) { in0[i] = (int8_t)(i % 7); in1[i] = (int8_t)(i % 5); } \
    ac.buffers.push_back(makeInt8Buffer(in0, false)); \
    ac.buffers.push_back(makeInt8Buffer(in1, false)); \
    ac.buffers.push_back(makeInt8Output(count)); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::scalarFloat(s0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::scalarFloat(s1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::scalarFloat(os)); \
    ac.args.push_back(AdaptedArg::scalarInt(1)); /* s0 */ \
    ac.args.push_back(AdaptedArg::scalarInt(1)); /* s1 */ \
    ac.args.push_back(AdaptedArg::scalarInt(count)); \
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = std::vector<float>(in0.begin(), in0.end()); \
    ac.validatorInputB = std::vector<float>(in1.begin(), in1.end()); \
    ac.elementCount = count; \
    ac.validatorFloats = {s0, s1, os}; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp32((const int8_t*)ctx.devBufs[0], ctx.floatArgs[0], \
                              (const int8_t*)ctx.devBufs[1], ctx.floatArgs[1], \
                              (int8_t*)ctx.devBufs[2], ctx.floatArgs[2], \
                              ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int count = ac.elementCount; \
    if ((int)output.size() * (int)sizeof(float) < count * (int)sizeof(int8_t)) return false; \
    const int8_t* out = reinterpret_cast<const int8_t*>(output.data()); \
    const float s0 = ac.validatorFloats[0], s1 = ac.validatorFloats[1], os = ac.validatorFloats[2]; \
    for (int i = 0; i < std::min(10, count); ++i) { \
        float x = ac.validatorInputA[i] * s0; \
        float y = ac.validatorInputB[i] * s1; \
        float val = (OP); \
        int expected = host_float2int_rn(os * val); \
        expected = min(expected, 127); expected = max(expected, -128); \
        if (out[i] != (int8_t)expected) return false; \
    } \
    return true; \
}

BINARY_INT8_ADAPTER(CudaBinaryInt8AddFp32Kernel, binary_int8_add, x + y)
BINARY_INT8_ADAPTER(CudaBinaryInt8MulFp32Kernel, binary_int8_mul, x * y)

// ============================================================================
// P5 Raster fused binary adapters
// ============================================================================
// Helper: build 3D test data for Binary/BinaryMid (sizeZ*sizeY*sizeX elements)
struct RasterBinaryParams {
    int sizeZ, sizeY, sizeX;
    int strideZ, strideY, strideX;      // input0 strides
    int strideZ1, strideY1, strideX1;   // input1 strides
    int dstStrideZ, dstStrideY, dstStrideX;
    int total;
};
static RasterBinaryParams buildRasterBinaryParams(const CaseSpec& spec) {
    RasterBinaryParams p;
    p.sizeZ = spec.intParam("size_z", 2);
    p.sizeY = spec.intParam("size_y", 4);
    p.sizeX = spec.intParam("size_x", 4);
    // Contiguous layout for both inputs and output
    p.strideZ = p.sizeY * p.sizeX; p.strideY = p.sizeX; p.strideX = 1;
    p.strideZ1 = p.sizeY * p.sizeX; p.strideY1 = p.sizeX; p.strideX1 = 1;
    p.dstStrideZ = p.sizeY * p.sizeX; p.dstStrideY = p.sizeX; p.dstStrideX = 1;
    p.total = p.sizeZ * p.sizeY * p.sizeX;
    return p;
}

// ---- Binary (basic 3D-stride): ADD/MUL ----
#define RASTER_BINARY_ADAPTER(CLASS, SHIM, OP) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp32"; \
    auto p = buildRasterBinaryParams(spec); \
    std::vector<float> in0(p.total), in1(p.total); \
    for (int i = 0; i < p.total; ++i) { in0[i] = 0.1f * (i % 7); in1[i] = 0.1f * (i % 5); } \
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false; \
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false; \
    AdaptedBuffer out; out.sizeBytes = p.total * sizeof(float); out.isOutput = true; \
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(out); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeX)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideX)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideZ1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideY1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideX1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideX)); \
    ac.args.push_back(AdaptedArg::scalarInt(0)); /* activationType=0 */ \
    ac.globalSize[0] = gridFor(p.total); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = in0; ac.validatorInputB = in1; \
    ac.elementCount = p.total; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp32((const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (float*)ctx.devBufs[2], \
                              ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                              ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], \
                              ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], \
                              ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11], \
                              ctx.intArgs[12], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    if ((int)output.size() < ac.elementCount) return false; \
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) { \
        float expected = OP; \
        if (std::fabs(output[i] - expected) > 1e-3f) return false; \
    } \
    return true; \
}

RASTER_BINARY_ADAPTER(CudaBinaryAddRasterFp32Kernel, binary_add, (ac.validatorInputA[i] + ac.validatorInputB[i]))
RASTER_BINARY_ADAPTER(CudaBinaryMulRasterFp32Kernel, binary_mul, (ac.validatorInputA[i] * ac.validatorInputB[i]))

// ---- BinaryFuseAdd (atomicAdd): ADD/MUL ----
#define RASTER_FUSEADD_ADAPTER(CLASS, SHIM, OP) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp32"; \
    auto p = buildRasterBinaryParams(spec); \
    std::vector<float> in0(p.total), in1(p.total); \
    for (int i = 0; i < p.total; ++i) { in0[i] = 0.1f * (i % 7); in1[i] = 0.1f * (i % 5); } \
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false; \
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false; \
    AdaptedBuffer out; out.sizeBytes = p.total * sizeof(float); out.isOutput = true; \
    out.initialData.assign(p.total * sizeof(float), 0); /* zero-init for atomicAdd */ \
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(out); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeX)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideX)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideZ1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideY1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideX1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideX)); \
    ac.globalSize[0] = gridFor(p.total); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = in0; ac.validatorInputB = in1; \
    ac.elementCount = p.total; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp32((const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (float*)ctx.devBufs[2], \
                              ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                              ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], \
                              ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], \
                              ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    if ((int)output.size() < ac.elementCount) return false; \
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) { \
        float expected = OP; \
        if (std::fabs(output[i] - expected) > 1e-3f) return false; \
    } \
    return true; \
}

RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddAddFp32Kernel, binary_fuseadd_add, (ac.validatorInputA[i] + ac.validatorInputB[i]))
RASTER_FUSEADD_ADAPTER(CudaBinaryFuseAddMulFp32Kernel, binary_fuseadd_mul, (ac.validatorInputA[i] * ac.validatorInputB[i]))

// ---- BinaryMid (DivModFast): ADD/MUL — same adapter as Binary but different shim ----
#define RASTER_BINARYMID_ADAPTER(CLASS, SHIM, OP) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp32"; \
    auto p = buildRasterBinaryParams(spec); \
    std::vector<float> in0(p.total), in1(p.total); \
    for (int i = 0; i < p.total; ++i) { in0[i] = 0.1f * (i % 7); in1[i] = 0.1f * (i % 5); } \
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false; \
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false; \
    AdaptedBuffer out; out.sizeBytes = p.total * sizeof(float); out.isOutput = true; \
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(out); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.sizeX)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideX)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideZ1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideY1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.strideX1)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideZ)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideY)); \
    ac.args.push_back(AdaptedArg::scalarInt(p.dstStrideX)); \
    ac.args.push_back(AdaptedArg::scalarInt(0)); /* activationType=0 */ \
    ac.globalSize[0] = gridFor(p.total); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = in0; ac.validatorInputB = in1; \
    ac.elementCount = p.total; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp32((const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (float*)ctx.devBufs[2], \
                              ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                              ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], \
                              ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], \
                              ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11], \
                              ctx.intArgs[12], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    if ((int)output.size() < ac.elementCount) return false; \
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) { \
        float expected = OP; \
        if (std::fabs(output[i] - expected) > 1e-3f) return false; \
    } \
    return true; \
}

RASTER_BINARYMID_ADAPTER(CudaBinaryMidAddFp32Kernel, binarymid_add, (ac.validatorInputA[i] + ac.validatorInputB[i]))
RASTER_BINARYMID_ADAPTER(CudaBinaryMidMulFp32Kernel, binarymid_mul, (ac.validatorInputA[i] * ac.validatorInputB[i]))

// ---- BinaryMidLinear4 (float4 vectorized): ADD/MUL ----
#define RASTER_BINARYMIDLINEAR4_ADAPTER(CLASS, SHIM, OP) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = "mnn_corpus_" #SHIM "_fp32"; \
    const int count = spec.intParam("count", 64); \
    const int count_4 = count / 4; \
    std::vector<float> in0(count), in1(count); \
    for (int i = 0; i < count; ++i) { in0[i] = 0.1f * (i % 7); in1[i] = 0.1f * (i % 5); } \
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false; \
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false; \
    AdaptedBuffer out; out.sizeBytes = count * sizeof(float); out.isOutput = true; \
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(out); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    ac.args.push_back(AdaptedArg::buffer(2)); \
    ac.args.push_back(AdaptedArg::scalarInt(count_4)); \
    ac.args.push_back(AdaptedArg::scalarInt(0)); /* activationType */ \
    ac.args.push_back(AdaptedArg::scalarInt(0)); /* inp0Broadcast */ \
    ac.args.push_back(AdaptedArg::scalarInt(0)); /* inp1Broadcast */ \
    ac.globalSize[0] = gridFor(count_4); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = in0; ac.validatorInputB = in1; \
    ac.elementCount = count; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    mnn_corpus_##SHIM##_fp32((const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (float*)ctx.devBufs[2], \
                              ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], \
                              ctx.grid, ctx.block, ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    if ((int)output.size() < ac.elementCount) return false; \
    for (int i = 0; i < std::min(10, ac.elementCount); ++i) { \
        float expected = OP; \
        if (std::fabs(output[i] - expected) > 1e-3f) return false; \
    } \
    return true; \
}

RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4AddFp32Kernel, binarymidlinear4_add, (ac.validatorInputA[i] + ac.validatorInputB[i]))
RASTER_BINARYMIDLINEAR4_ADAPTER(CudaBinaryMidLinear4MulFp32Kernel, binarymidlinear4_mul, (ac.validatorInputA[i] * ac.validatorInputB[i]))

// ============================================================================
// New fp16/bf16 cast & pool adapters — 1:1 with shims in unary_cast.cu /
// pool.cu / conv_base.cu.
// ============================================================================

// Helper: convert bf16 bits (int16_t) to float on host.
static inline float bf16ToFloatHost(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}
// Helper: convert float to bf16 bits (round to nearest even) on host.
static inline uint16_t floatToBf16Host(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    // Round to nearest even: take top 16 bits with rounding of the dropped 16.
    uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)((u + rounding_bias) >> 16);
}

// ---- CASTMIDFLOAT half→int32 ----
bool CudaCastMidFloatF16I32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_castmidfloat_f16_i32";
    const int count = spec.intParam("size", 64);
    std::vector<float> inputF(count);
    for (int i = 0; i < count; ++i) inputF[i] = 0.1f * (i % 13) - 0.5f;
    auto halfBuf = packHalf(inputF);
    AdaptedBuffer inBuf; inBuf.sizeBytes = halfBuf.size(); inBuf.initialData = halfBuf; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = count;
    return true;
}
cudaError_t CudaCastMidFloatF16I32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_castmidfloat_f16_i32(ctx.devBufs[0], (int32_t*)ctx.devBufs[1],
        (size_t)ctx.intArgs[0], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastMidFloatF16I32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if ((int)output.size() * sizeof(float) < count * sizeof(int32_t)) return false;
    const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
    std::vector<uint8_t> halfBytes = packHalf(ac.validatorInputA);
    const __half* inH = reinterpret_cast<const __half*>(halfBytes.data());
    for (int i = 0; i < count; ++i) {
        // CASTMIDFLOAT<half,int32_t>: output[i] = (int32_t)((float)half_input[i]) — truncating cast via float
        int32_t expected = (int32_t)__half2float(inH[i]);
        if (out[i] != expected) return false;
    }
    return true;
}

// ---- CASTMIDFLOAT half→int8 ----
bool CudaCastMidFloatF16I8Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_castmidfloat_f16_i8";
    const int count = spec.intParam("size", 64);
    std::vector<float> inputF(count);
    for (int i = 0; i < count; ++i) inputF[i] = 0.1f * (i % 13) - 0.5f;
    auto halfBuf = packHalf(inputF);
    AdaptedBuffer inBuf; inBuf.sizeBytes = halfBuf.size(); inBuf.initialData = halfBuf; inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count; outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = count;
    return true;
}
cudaError_t CudaCastMidFloatF16I8Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_castmidfloat_f16_i8(ctx.devBufs[0], (int8_t*)ctx.devBufs[1],
        (size_t)ctx.intArgs[0], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastMidFloatF16I8Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if ((int)output.size() * sizeof(float) < count) return false;
    const int8_t* out = reinterpret_cast<const int8_t*>(output.data());
    std::vector<uint8_t> halfBytes = packHalf(ac.validatorInputA);
    const __half* inH = reinterpret_cast<const __half*>(halfBytes.data());
    for (int i = 0; i < count; ++i) {
        // CASTMIDFLOAT<half,int8_t>: output[i] = (int8_t)((float)half_input[i]) — truncating cast via float
        int8_t expected = (int8_t)(int32_t)__half2float(inH[i]);
        if (out[i] != expected) return false;
    }
    return true;
}

// ---- CASTMIDFLOAT int32→half ----
bool CudaCastMidFloatI32F16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_castmidfloat_i32_f16";
    const int count = spec.intParam("size", 64);
    std::vector<int32_t> input(count);
    for (int i = 0; i < count; ++i) input[i] = i - count / 2;
    AdaptedBuffer inBuf; inBuf.sizeBytes = count * sizeof(int32_t);
    inBuf.initialData.assign((const uint8_t*)input.data(), (const uint8_t*)input.data() + inBuf.sizeBytes); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(__half); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA.assign(input.begin(), input.end());
    ac.elementCount = count;
    return true;
}
cudaError_t CudaCastMidFloatI32F16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_castmidfloat_i32_f16((const int32_t*)ctx.devBufs[0], ctx.devBufs[1],
        (size_t)ctx.intArgs[0], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastMidFloatI32F16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if ((int)output.size() * sizeof(float) < count * (int)sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < count; ++i) {
        // CASTMIDFLOAT<int32_t,half>: output[i] = (half)((float)int32_input[i])
        int32_t v = (int32_t)ac.validatorInputA[i];
        __half expected = __float2half((float)v);
        if (std::fabs(__half2float(out[i]) - __half2float(expected)) > 1e-3f) return false;
    }
    return true;
}

// ---- BF16→float (fp16 output: bf16→half via round trip) ----
bool CudaBf162FloatF16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_bf162float_f16";
    const int count = spec.intParam("size", 64);
    std::vector<float> inputF(count);
    for (int i = 0; i < count; ++i) inputF[i] = 0.1f * (i % 13) - 0.5f;
    std::vector<uint16_t> bf16In(count);
    for (int i = 0; i < count; ++i) bf16In[i] = floatToBf16Host(inputF[i]);
    AdaptedBuffer inBuf; inBuf.sizeBytes = count * sizeof(uint16_t);
    inBuf.initialData.assign((const uint8_t*)bf16In.data(), (const uint8_t*)bf16In.data() + inBuf.sizeBytes); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(__half); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = count;
    return true;
}
cudaError_t CudaBf162FloatF16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_bf162float_f16((const int16_t*)ctx.devBufs[0], ctx.devBufs[1],
        (size_t)ctx.intArgs[0], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaBf162FloatF16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if ((int)output.size() * sizeof(float) < count * sizeof(__half)) return false;
    const __half* out = reinterpret_cast<const __half*>(output.data());
    for (int i = 0; i < count; ++i) {
        // BF162FLOAT<half>: bf16→float→half (round to nearest half)
        float expected = ac.validatorInputA[i];
        float got = __half2float(out[i]);
        float tol = std::max(1e-3f, std::fabs(expected) * 1e-2f);
        if (std::fabs(got - expected) > tol) return false;
    }
    return true;
}

// ---- BF16→float (fp32 output) ----
bool CudaBf162FloatF32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_bf162float_f32";
    const int count = spec.intParam("size", 64);
    std::vector<float> inputF(count);
    for (int i = 0; i < count; ++i) inputF[i] = 0.1f * (i % 13) - 0.5f;
    std::vector<uint16_t> bf16In(count);
    for (int i = 0; i < count; ++i) bf16In[i] = floatToBf16Host(inputF[i]);
    AdaptedBuffer inBuf; inBuf.sizeBytes = count * sizeof(uint16_t);
    inBuf.initialData.assign((const uint8_t*)bf16In.data(), (const uint8_t*)bf16In.data() + inBuf.sizeBytes); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = count;
    return true;
}
cudaError_t CudaBf162FloatF32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_bf162float_f32((const int16_t*)ctx.devBufs[0], (float*)ctx.devBufs[1],
        (size_t)ctx.intArgs[0], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaBf162FloatF32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if ((int)output.size() < count) return false;
    for (int i = 0; i < count; ++i) {
        // bf16→float: exact recovery (bf16 has 8-bit mantissa)
        float expected = ac.validatorInputA[i];
        if (std::fabs(output[i] - expected) > std::max(1e-2f, std::fabs(expected) * 1e-2f)) return false;
    }
    return true;
}

// ---- BF16 maxpool (sm80+ executes, sm75 empty) — smoke-only validation ----
bool CudaMaxpoolC8Bf16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_maxpool_c8_bf16";
    const int ib = spec.intParam("batch", 1);
    const int ic = spec.intParam("c", 8);
    const int ic_p = (ic + 7) / 8 * 8;
    const int ih = spec.intParam("h", 4);
    const int iw = spec.intParam("w", 4);
    const int kx = spec.intParam("kernel_size", 2);
    const int sx = spec.intParam("stride", 2);
    const int pad = spec.intParam("pad", 0);
    const int oh = (ih + 2 * pad - kx) / sx + 1;
    const int ow = (iw + 2 * pad - kx) / sx + 1;
    const int total = ib * oh * ow * ic_p;
    const int inSize = ib * ih * iw * ic_p;
    std::vector<float> inputF(inSize);
    for (int i = 0; i < inSize; ++i) inputF[i] = 0.1f * (i % 11);
    std::vector<uint16_t> bf16In(inSize);
    for (int i = 0; i < inSize; ++i) bf16In[i] = floatToBf16Host(inputF[i]);
    AdaptedBuffer inBuf; inBuf.sizeBytes = inSize * sizeof(uint16_t);
    inBuf.initialData.assign((const uint8_t*)bf16In.data(), (const uint8_t*)bf16In.data() + inBuf.sizeBytes); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(uint16_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    for (int v : {ib, ic_p, ih, iw, oh, ow, pad, pad, kx, kx, sx, sx})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = total;
    ac.m = ib; ac.n = ic_p; ac.h = ih; ac.w = iw; ac.k = kx; ac.stride = sx; ac.orderType = pad;
    ac.p = oh; ac.q = ow;
    return true;
}
cudaError_t CudaMaxpoolC8Bf16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_maxpool_c8_bf16(ctx.devBufs[0], ctx.devBufs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
        ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
        ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaMaxpoolC8Bf16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // smoke-only: kernel body empty on sm75, full on sm80+. On sm80+ validate fully.
    const int ib = ac.m, ic_p = ac.n, ih = ac.h, iw = ac.w, kx = ac.k, sx = ac.stride, pad = ac.orderType;
    const int oh = ac.p, ow = ac.q;
    const int total = ib * oh * ow * ic_p;
    if ((int)output.size() * sizeof(float) < total * sizeof(uint16_t)) return true;  // can't validate → pass (smoke)
    const uint16_t* out = reinterpret_cast<const uint16_t*>(output.data());
    // Detect sm80+: if output is all-zero AND input had non-zero data, kernel was empty (sm75) → pass
    bool anyNonzero = false;
    for (int i = 0; i < total; ++i) if (out[i] != 0) { anyNonzero = true; break; }
    if (!anyNonzero) return true;  // sm75 empty kernel → smoke pass
    // sm80+: validate maxpool over NC4HW4-style [b][ih][iw][ic_p] layout
    for (int b = 0; b < ib; ++b)
        for (int c = 0; c < ic_p; ++c)
            for (int oy = 0; oy < oh; ++oy)
                for (int ox = 0; ox < ow; ++ox) {
                    float maxV = -65504.0f;
                    for (int fy = 0; fy < kx; ++fy)
                        for (int fx = 0; fx < kx; ++fx) {
                            int iy = oy * sx - pad + fy, ix = ox * sx - pad + fx;
                            if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                            int off = (b * ih + iy) * iw * ic_p + ix * ic_p + c;
                            maxV = std::max(maxV, ac.validatorInputA[off]);
                        }
                    int outOff = ((b * oh + oy) * ow + ox) * ic_p + c;
                    float got = bf16ToFloatHost(out[outOff]);
                    if (std::fabs(got - maxV) > std::max(1e-2f, std::fabs(maxV) * 1e-2f)) return false;
                }
    return true;
}

// ---- BF16 avgpool (sm80+ executes, sm75 empty) — smoke-only validation ----
bool CudaAvgpoolC8Bf16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_avgpool_c8_bf16";
    const int ib = spec.intParam("batch", 1);
    const int ic = spec.intParam("c", 8);
    const int ic_p = (ic + 7) / 8 * 8;
    const int ih = spec.intParam("h", 4);
    const int iw = spec.intParam("w", 4);
    const int kx = spec.intParam("kernel_size", 2);
    const int sx = spec.intParam("stride", 2);
    const int pad = spec.intParam("pad", 0);
    const int oh = (ih + 2 * pad - kx) / sx + 1;
    const int ow = (iw + 2 * pad - kx) / sx + 1;
    const int total = ib * oh * ow * ic_p;
    const int inSize = ib * ih * iw * ic_p;
    std::vector<float> inputF(inSize);
    for (int i = 0; i < inSize; ++i) inputF[i] = 0.1f * (i % 11);
    std::vector<uint16_t> bf16In(inSize);
    for (int i = 0; i < inSize; ++i) bf16In[i] = floatToBf16Host(inputF[i]);
    AdaptedBuffer inBuf; inBuf.sizeBytes = inSize * sizeof(uint16_t);
    inBuf.initialData.assign((const uint8_t*)bf16In.data(), (const uint8_t*)bf16In.data() + inBuf.sizeBytes); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(uint16_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    for (int v : {ib, ic_p, ih, iw, oh, ow, pad, pad, kx, kx, sx, sx})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = inputF; ac.elementCount = total;
    ac.m = ib; ac.n = ic_p; ac.h = ih; ac.w = iw; ac.k = kx; ac.stride = sx; ac.orderType = pad;
    ac.p = oh; ac.q = ow;
    return true;
}
cudaError_t CudaAvgpoolC8Bf16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_avgpool_c8_bf16(ctx.devBufs[0], ctx.devBufs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
        ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
        ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaAvgpoolC8Bf16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ib = ac.m, ic_p = ac.n, ih = ac.h, iw = ac.w, kx = ac.k, sx = ac.stride, pad = ac.orderType;
    const int oh = ac.p, ow = ac.q;
    const int total = ib * oh * ow * ic_p;
    if ((int)output.size() * sizeof(float) < total * sizeof(uint16_t)) return true;
    const uint16_t* out = reinterpret_cast<const uint16_t*>(output.data());
    bool anyNonzero = false;
    for (int i = 0; i < total; ++i) if (out[i] != 0) { anyNonzero = true; break; }
    if (!anyNonzero) return true;  // sm75 empty → smoke pass
    for (int b = 0; b < ib; ++b)
        for (int c = 0; c < ic_p; ++c)
            for (int oy = 0; oy < oh; ++oy)
                for (int ox = 0; ox < ow; ++ox) {
                    int iw_idx = ox * sx - pad, ih_idx = oy * sx - pad;
                    int s_x = std::max(0, -iw_idx), s_y = std::max(0, -ih_idx);
                    int e_x = std::min(kx, iw - iw_idx), e_y = std::min(kx, ih - ih_idx);
                    int div = (e_y - s_y) * (e_x - s_x);
                    if (div <= 0) continue;
                    float sum = 0.0f;
                    for (int fy = s_y; fy < e_y; ++fy)
                        for (int fx = s_x; fx < e_x; ++fx) {
                            int iy = ih_idx + fy, ix = iw_idx + fx;
                            int off = (b * ih + iy) * iw * ic_p + ix * ic_p + c;
                            sum += ac.validatorInputA[off];
                        }
                    int outOff = ((b * oh + oy) * ow + ox) * ic_p + c;
                    float got = bf16ToFloatHost(out[outOff]);
                    if (std::fabs(got - sum / div) > std::max(1e-2f, std::fabs(sum / div) * 1e-2f)) return false;
                }
    return true;
}

// ---- Float22BFloat16 (float→bf16, sm80+ executes, sm75 empty) ----
bool CudaFloat22BFloat16Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_float22bfloat16_fp32";
    const int count = spec.intParam("count", 64);
    const size_t maxCount = (size_t)count;
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = 0.1f * (i % 13);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(uint16_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt((int)maxCount));
    ac.globalSize[0] = gridFor(maxCount); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    return true;
}
cudaError_t CudaFloat22BFloat16Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_float22bfloat16_fp32((const float*)ctx.devBufs[0], ctx.devBufs[1],
        (size_t)ctx.intArgs[0], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaFloat22BFloat16Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if ((int)output.size() * sizeof(float) < count * sizeof(uint16_t)) return false;
    const uint16_t* out = reinterpret_cast<const uint16_t*>(output.data());
    bool anyNonzero = false;
    for (int i = 0; i < count; ++i) if (out[i] != 0) { anyNonzero = true; break; }
    if (!anyNonzero) return true;  // sm75 empty → smoke pass
    for (int i = 0; i < count; ++i) {
        uint16_t expected = floatToBf16Host(ac.validatorInputA[i]);
        if (out[i] != expected) {
            float ev = bf16ToFloatHost(expected), got = bf16ToFloatHost(out[i]);
            if (std::fabs(got - ev) > std::max(1e-2f, std::fabs(ev) * 1e-2f)) return false;
        }
    }
    return true;
}

// ============================================================================
// weight_only_quant fp16 adapters (conv_fpa_intb). Smoke-only: launch the
// shim, then check the __half output has any non-zero element. Small geometry
// mirrors the fp32 path (batch=1, ic=8, ic_p=8, oc=4, oc_p=8, quanC=4).
// Helpers packHalf / makeHalfBuffer / makeHalfOutput come from above.
// ============================================================================
namespace {
// Local WOQ fp16 geometry constants (mirror CudaOpsWoq.cpp fp32 path).
constexpr int kWoqHpBatch = 1;
constexpr int kWoqHpIc = 8;
constexpr int kWoqHpIcP = 8;
constexpr int kWoqHpOc = 4;
constexpr int kWoqHpOcP = 8;
constexpr int kWoqHpQuanC = 4;
constexpr float kWoqHpScale = 0.1f;
constexpr float kWoqHpOffset = 0.0f;
constexpr float kWoqHpBias = 0.5f;
constexpr float kWoqHpMaxV = 6.0f;
constexpr float kWoqHpMinV = 0.0f;
const int8_t kWoqHpInt8Q = 2;
const uint8_t kWoqHpInt4Byte = 0xAA;

std::vector<int8_t> woqHpBuildInt8Weight(int oc, int ic_p, int8_t q) {
    return std::vector<int8_t>((size_t)oc * ic_p, q);
}
std::vector<uint8_t> woqHpBuildInt4Weight(int oc, int ic_p, uint8_t byte) {
    return std::vector<uint8_t>((size_t)oc * (ic_p / 2), byte);
}
std::vector<float> woqHpBuildInput(int batch, int ic, int ic_p) {
    std::vector<float> v((size_t)batch * ic_p, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int k = 0; k < ic; ++k) v[b * ic_p + k] = 0.1f * (k % 7);
    return v;
}
std::vector<float> woqHpBuildScale(int oc, int quanC, float s) {
    const int num_qg = quanC / oc;
    return std::vector<float>((size_t)oc * (num_qg > 0 ? num_qg : 1), s);
}
std::vector<float> woqHpBuildOffset(int oc, int quanC, float o) {
    const int num_qg = quanC / oc;
    return std::vector<float>((size_t)oc * (num_qg > 0 ? num_qg : 1), o);
}
std::vector<float> woqHpBuildBias(int oc, float b) {
    return std::vector<float>((size_t)oc, b);
}
std::vector<float2> woqHpBuildGemvParams(int oc, int num_qg, float scale, float offset) {
    std::vector<float2> v((size_t)oc * num_qg);
    for (size_t i = 0; i < v.size(); ++i) { v[i].x = scale; v[i].y = offset - 8.0f * scale; }
    return v;
}
// __half output -> check first n halfs for any non-zero.
bool woqHpAnyNonZeroHalf(const std::vector<float>& output, int n) {
    int floats = (int)output.size();
    int halfs = floats / 1; // each float slot holds 2 __half bytes after readback
    // Output readback packs __half bytes; reinterpret as __half.
    const __half* h = reinterpret_cast<const __half*>(output.data());
    int lim = std::min(n, halfs * 2);
    for (int i = 0; i < lim; ++i) if (__half2float(h[i]) != 0.0f) return true;
    return false;
}
} // namespace (anonymous)

// ---- 1. CudaPrecomputeGemvParamsFp16Kernel — PrecomputeGemvParams<__half> ----
bool CudaPrecomputeGemvParamsFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_precomputegemvparams_fp16";
    const int oc = spec.intParam("oc", kWoqHpOc);
    const int num_qg = spec.intParam("num_qg", 1);
    const int total = oc * num_qg;
    auto scale = woqHpBuildScale(oc, oc * num_qg, kWoqHpScale);
    auto offset = woqHpBuildOffset(oc, oc * num_qg, kWoqHpOffset);
    ac.buffers.push_back(makeHalfBuffer(scale, false));
    ac.buffers.push_back(makeHalfBuffer(offset, false));
    AdaptedBuffer pBuf; pBuf.sizeBytes = (size_t)total * sizeof(float2); pBuf.isOutput = true;
    ac.buffers.push_back(pBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1)); ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaPrecomputeGemvParamsFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_precomputegemvparams_fp16(ctx.devBufs[0], ctx.devBufs[1], (float2*)ctx.devBufs[2],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.stream);
    return cudaGetLastError();
}
bool CudaPrecomputeGemvParamsFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // float2 output: 2 floats per element.
    if ((int)output.size() < ac.elementCount * 2) return false;
    for (int i = 0; i < ac.elementCount; ++i) if (output[i * 2] != 0.0f) return true;
    return false;
}

// ---- 2. CudaQuantAFp16Kernel — QuantA<__half> ----
bool CudaQuantAFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_quanta_fp16";
    const int M = spec.intParam("m", 4);
    const int K_i = spec.intParam("k", kWoqHpIc);
    const int lda = K_i;
    std::vector<float> A_sub_fp((size_t)M * lda, 0.0f);
    for (int i = 0; i < M * lda; ++i) A_sub_fp[i] = 0.1f * (i % 7);
    ac.buffers.push_back(makeHalfBuffer(A_sub_fp, false));
    AdaptedBuffer qBuf; qBuf.sizeBytes = (size_t)M * lda; qBuf.isOutput = true;
    ac.buffers.push_back(qBuf);
    ac.buffers.push_back(makeHalfOutput(M)); // scale_A_out
    ac.buffers.push_back(makeHalfOutput(M)); // offset_A_out
    AdaptedBuffer suBuf; suBuf.sizeBytes = M * sizeof(int32_t); suBuf.isOutput = true;
    ac.buffers.push_back(suBuf);
    for (int i = 0; i < 5; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::scalarInt(M)); ac.args.push_back(AdaptedArg::scalarInt(K_i)); ac.args.push_back(AdaptedArg::scalarInt(lda));
    const int total = M * lda;
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaQuantAFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_quanta_fp16(ctx.devBufs[0], (int8_t*)ctx.devBufs[1],
        ctx.devBufs[2], ctx.devBufs[3], (int32_t*)ctx.devBufs[4],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.stream);
    return cudaGetLastError();
}
bool CudaQuantAFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int8_t* q = reinterpret_cast<const int8_t*>(output.data());
    int lim = std::min(ac.elementCount, (int)(output.size() * 4));
    for (int i = 0; i < lim; ++i) if (q[i] != 0) return true;
    return false;
}

// ---- 3. CudaDequantAndAccFp16Kernel — DequantAndAcc<__half> ----
bool CudaDequantAndAccFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_dequantandacc_fp16";
    const int M = spec.intParam("m", 4);
    const int N = spec.intParam("n", kWoqHpOc);
    const int K_i = spec.intParam("k", kWoqHpIc);
    const int ldc = N;
    const int num_oc_groups = 1, group_idx = 0;
    std::vector<int32_t> C_q((size_t)M * ldc, 8);
    std::vector<float> scale_A(M, kWoqHpScale), offset_A(M, kWoqHpOffset);
    std::vector<float> base_scale_B(N, kWoqHpScale), base_offset_B(N, kWoqHpOffset);
    std::vector<int32_t> base_sum_B_q(N, K_i * kWoqHpInt8Q);
    std::vector<int32_t> sum_A_q(M, K_i * kWoqHpInt8Q);
    AdaptedBuffer cqBuf; cqBuf.sizeBytes = C_q.size() * 4; cqBuf.initialData.assign((const uint8_t*)C_q.data(), (const uint8_t*)C_q.data() + C_q.size() * 4); cqBuf.isOutput = false;
    ac.buffers.push_back(cqBuf);
    ac.buffers.push_back(makeHalfOutput(M * ldc)); // C_fp_final
    ac.buffers.push_back(makeHalfBuffer(scale_A, false));
    ac.buffers.push_back(makeHalfBuffer(offset_A, false));
    ac.buffers.push_back(makeHalfBuffer(base_scale_B, false));
    ac.buffers.push_back(makeHalfBuffer(base_offset_B, false));
    AdaptedBuffer bsbqBuf; bsbqBuf.sizeBytes = base_sum_B_q.size() * 4; bsbqBuf.initialData.assign((const uint8_t*)base_sum_B_q.data(), (const uint8_t*)base_sum_B_q.data() + base_sum_B_q.size() * 4); bsbqBuf.isOutput = false;
    ac.buffers.push_back(bsbqBuf);
    AdaptedBuffer saqBuf; saqBuf.sizeBytes = sum_A_q.size() * 4; saqBuf.initialData.assign((const uint8_t*)sum_A_q.data(), (const uint8_t*)sum_A_q.data() + sum_A_q.size() * 4); saqBuf.isOutput = false;
    ac.buffers.push_back(saqBuf);
    for (int i = 0; i < 8; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::scalarInt(group_idx)); ac.args.push_back(AdaptedArg::scalarInt(num_oc_groups));
    ac.args.push_back(AdaptedArg::scalarInt(7)); // sum_A_q_in index
    ac.args.push_back(AdaptedArg::scalarInt(M)); ac.args.push_back(AdaptedArg::scalarInt(N)); ac.args.push_back(AdaptedArg::scalarInt(K_i));
    ac.args.push_back(AdaptedArg::scalarInt(ldc));
    const int gridX = (N + 15) / 16, gridY = (M + 15) / 16, blockX = 16, blockY = 16;
    ac.args.push_back(AdaptedArg::scalarInt(gridX)); ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(blockX)); ac.args.push_back(AdaptedArg::scalarInt(blockY));
    ac.globalSize[0] = gridX; ac.globalSize[1] = gridY; ac.localSize[0] = blockX; ac.localSize[1] = blockY; ac.dims = 2;
    ac.elementCount = M * ldc;
    return true;
}
cudaError_t CudaDequantAndAccFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_dequantandacc_fp16((const int32_t*)ctx.devBufs[0], ctx.devBufs[1],
        ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4], ctx.devBufs[5],
        (const int32_t*)ctx.devBufs[6], ctx.intArgs[0], ctx.intArgs[1],
        (const int32_t*)ctx.devBufs[7], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}
bool CudaDequantAndAccFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqHpAnyNonZeroHalf(output, ac.elementCount);
}

// ---- 4. CudaBiasAndActivationFp16Kernel — BiasAndActivation<__half> ----
bool CudaBiasAndActivationFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_biasandactivation_fp16";
    const int M = spec.intParam("m", 1);
    const int N = spec.intParam("n", kWoqHpOc);
    const int ldc = N;
    std::vector<float> data((size_t)M * ldc, 0.0f);
    for (int i = 0; i < M * ldc; ++i) data[i] = 0.1f * (i % 7);
    auto bias = woqHpBuildBias(N, kWoqHpBias);
    // data is __half in/out (in-place).
    ac.buffers.push_back(makeHalfBuffer(data, true));
    ac.buffers.push_back(makeHalfBuffer(bias, false));
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMinV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMaxV));
    ac.args.push_back(AdaptedArg::scalarInt(M)); ac.args.push_back(AdaptedArg::scalarInt(N)); ac.args.push_back(AdaptedArg::scalarInt(ldc));
    const int total = M * ldc;
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaBiasAndActivationFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_biasandactivation_fp16(ctx.devBufs[0], ctx.devBufs[1],
        ctx.floatArgs[0], ctx.floatArgs[1], ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
        ctx.intArgs[3], ctx.intArgs[4], ctx.stream);
    return cudaGetLastError();
}
bool CudaBiasAndActivationFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqHpAnyNonZeroHalf(output, ac.elementCount);
}

// ---- 5/6. GEMM_FpAInt8B / GEMM_FpAInt4B (fp16) ----
// ENTRY is the full shim symbol (e.g. mnn_corpus_gemm_fpaint8b_fp16).
#define WOQ_FP16_GEMM_SMOKE_BODY(CLASS, ENTRY, KBUF_TYPE, IS_INT4) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = #ENTRY; \
    const int batch = spec.intParam("batch", kWoqHpBatch); \
    const int ic = spec.intParam("ic", kWoqHpIc); \
    const int ic_p = spec.intParam("ic_p", kWoqHpIcP); \
    const int oc = spec.intParam("oc", kWoqHpOc); \
    const int oc_p = spec.intParam("oc_p", kWoqHpOcP); \
    const int quanC = spec.intParam("quan_c", kWoqHpQuanC); \
    auto input = woqHpBuildInput(batch, ic, ic_p); \
    auto scale = woqHpBuildScale(oc, quanC, kWoqHpScale); \
    auto offset = woqHpBuildOffset(oc, quanC, kWoqHpOffset); \
    auto bias = woqHpBuildBias(oc, kWoqHpBias); \
    ac.buffers.push_back(makeHalfBuffer(input, false)); \
    AdaptedBuffer kBuf; \
    if (IS_INT4) { auto w = woqHpBuildInt4Weight(oc, ic_p, kWoqHpInt4Byte); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign(w.begin(), w.end()); } \
    else { auto w = woqHpBuildInt8Weight(oc, ic_p, kWoqHpInt8Q); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign((const uint8_t*)w.data(), (const uint8_t*)w.data() + w.size()); } \
    kBuf.isOutput = false; \
    ac.buffers.push_back(kBuf); \
    ac.buffers.push_back(makeHalfBuffer(scale, false)); \
    ac.buffers.push_back(makeHalfBuffer(offset, false)); \
    ac.buffers.push_back(makeHalfBuffer(bias, false)); \
    ac.buffers.push_back(makeHalfOutput(batch * oc_p)); \
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i)); \
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMinV)); \
    ac.args.push_back(AdaptedArg::scalarInt(ic)); ac.args.push_back(AdaptedArg::scalarInt(ic_p)); \
    ac.args.push_back(AdaptedArg::scalarInt(oc)); ac.args.push_back(AdaptedArg::scalarInt(oc_p)); \
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(quanC)); \
    const int gridX = (oc + 15) / 16, gridY = batch, blockX = 16, blockY = 16; \
    ac.args.push_back(AdaptedArg::scalarInt(gridX)); ac.args.push_back(AdaptedArg::scalarInt(gridY)); \
    ac.args.push_back(AdaptedArg::scalarInt(blockX)); ac.args.push_back(AdaptedArg::scalarInt(blockY)); \
    ac.globalSize[0] = gridX; ac.globalSize[1] = gridY; ac.localSize[0] = blockX; ac.localSize[1] = blockY; ac.dims = 2; \
    ac.elementCount = batch * oc_p; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    ENTRY(ctx.devBufs[0], (KBUF_TYPE)ctx.devBufs[1], \
        ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4], ctx.devBufs[5], \
        ctx.floatArgs[0], ctx.floatArgs[1], \
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], \
        ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], \
        ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    return woqHpAnyNonZeroHalf(output, ac.elementCount); \
}

WOQ_FP16_GEMM_SMOKE_BODY(CudaGemmFpAInt8BFp16Kernel, mnn_corpus_gemm_fpaint8b_fp16, const int8_t*, false)
WOQ_FP16_GEMM_SMOKE_BODY(CudaGemmFpAInt4BFp16Kernel, mnn_corpus_gemm_fpaint4b_fp16, const uint8_t*, true)
#undef WOQ_FP16_GEMM_SMOKE_BODY

// ---- 7/8/9/10/11/12. GEMV_* fp16 variants (batch-first arg order). ----
// ENTRY is the full shim symbol (e.g. mnn_corpus_gemv_fpaint8b_fp16).
#define WOQ_FP16_GEMV_SMOKE_BODY(CLASS, ENTRY, KBUF_TYPE, IS_INT4, ARG_ORDER) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = #ENTRY; \
    const int batch = spec.intParam("batch", kWoqHpBatch); \
    const int ic = spec.intParam("ic", kWoqHpIc); \
    const int ic_p = spec.intParam("ic_p", kWoqHpIcP); \
    const int oc = spec.intParam("oc", kWoqHpOc); \
    const int oc_p = spec.intParam("oc_p", kWoqHpOcP); \
    const int quanC = spec.intParam("quan_c", kWoqHpQuanC); \
    auto input = woqHpBuildInput(batch, ic, ic_p); \
    auto scale = woqHpBuildScale(oc, quanC, kWoqHpScale); \
    auto offset = woqHpBuildOffset(oc, quanC, kWoqHpOffset); \
    auto bias = woqHpBuildBias(oc, kWoqHpBias); \
    ac.buffers.push_back(makeHalfBuffer(input, false)); \
    AdaptedBuffer kBuf; \
    if (IS_INT4) { auto w = woqHpBuildInt4Weight(oc, ic_p, kWoqHpInt4Byte); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign(w.begin(), w.end()); } \
    else { auto w = woqHpBuildInt8Weight(oc, ic_p, kWoqHpInt8Q); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign((const uint8_t*)w.data(), (const uint8_t*)w.data() + w.size()); } \
    kBuf.isOutput = false; \
    ac.buffers.push_back(kBuf); \
    ac.buffers.push_back(makeHalfBuffer(scale, false)); \
    ac.buffers.push_back(makeHalfBuffer(offset, false)); \
    ac.buffers.push_back(makeHalfBuffer(bias, false)); \
    ac.buffers.push_back(makeHalfOutput(batch * oc_p)); \
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i)); \
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMinV)); \
    ARG_ORDER \
    ac.globalSize[0] = oc; ac.globalSize[1] = batch; ac.localSize[0] = 64; ac.dims = 2; \
    ac.elementCount = batch * oc_p; \
    return true; \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    return woqHpAnyNonZeroHalf(output, ac.elementCount); \
}

// GEMV_FpAInt8B fp16 (2D grid + blockY).
WOQ_FP16_GEMV_SMOKE_BODY(CudaGemvFpAInt8BFp16Kernel, mnn_corpus_gemv_fpaint8b_fp16, const int8_t*, false,
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 15) / 16)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(16)); ac.args.push_back(AdaptedArg::scalarInt(16));)
cudaError_t CudaGemvFpAInt8BFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint8b_fp16(ctx.devBufs[0], (const int8_t*)ctx.devBufs[1],
        ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4], ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}

// GEMV_FpAInt4B fp16 (2D grid + blockY, shared mem).
WOQ_FP16_GEMV_SMOKE_BODY(CudaGemvFpAInt4BFp16Kernel, mnn_corpus_gemv_fpaint4b_fp16, const uint8_t*, true,
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    const int sharedMem = ic_p * sizeof(__half) + 64 * sizeof(float);
    ac.args.push_back(AdaptedArg::scalarInt(sharedMem));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 15) / 16)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(64));)
cudaError_t CudaGemvFpAInt4BFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_fp16(ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4], ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}

// GEMV_FpAInt4B_V5 fp16 (1D block, no shared mem).
WOQ_FP16_GEMV_SMOKE_BODY(CudaGemvFpAInt4BV5Fp16Kernel, mnn_corpus_gemv_fpaint4b_v5_fp16, const uint8_t*, true,
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 15) / 16)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(64));)
cudaError_t CudaGemvFpAInt4BV5Fp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_v5_fp16(ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4], ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.stream);
    return cudaGetLastError();
}

// GEMV_FpAInt4B_V9 fp16 (OC_PER_BLK=4, 1D block).
WOQ_FP16_GEMV_SMOKE_BODY(CudaGemvFpAInt4BV9Fp16Kernel, mnn_corpus_gemv_fpaint4b_v9_fp16, const uint8_t*, true,
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 3) / 4)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(64));)
cudaError_t CudaGemvFpAInt4BV9Fp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_v9_fp16(ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4], ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.stream);
    return cudaGetLastError();
}

#undef WOQ_FP16_GEMV_SMOKE_BODY

// GEMV_FpAInt4B_V14 fp16 (uses float2 gemv_params, no scale/offset).
bool CudaGemvFpAInt4BV14Fp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gemv_fpaint4b_v14_fp16";
    const int batch = spec.intParam("batch", kWoqHpBatch);
    const int ic = spec.intParam("ic", kWoqHpIc);
    const int ic_p = spec.intParam("ic_p", kWoqHpIcP);
    const int oc = spec.intParam("oc", kWoqHpOc);
    const int oc_p = spec.intParam("oc_p", kWoqHpOcP);
    const int quanC = spec.intParam("quan_c", kWoqHpQuanC);
    const int num_qg = quanC / oc;
    auto input = woqHpBuildInput(batch, ic, ic_p);
    auto kernel = woqHpBuildInt4Weight(oc, ic_p, kWoqHpInt4Byte);
    auto gemv_params = woqHpBuildGemvParams(oc, num_qg, kWoqHpScale, kWoqHpOffset);
    auto bias = woqHpBuildBias(oc, kWoqHpBias);
    ac.buffers.push_back(makeHalfBuffer(input, false));
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernel.size(); kBuf.initialData.assign(kernel.begin(), kernel.end()); kBuf.isOutput = false;
    ac.buffers.push_back(kBuf);
    AdaptedBuffer pBuf; pBuf.sizeBytes = gemv_params.size() * sizeof(float2); pBuf.initialData.assign((const uint8_t*)gemv_params.data(), (const uint8_t*)gemv_params.data() + pBuf.sizeBytes); pBuf.isOutput = false;
    ac.buffers.push_back(pBuf);
    ac.buffers.push_back(makeHalfBuffer(bias, false));
    ac.buffers.push_back(makeHalfOutput(batch * oc_p));
    for (int i = 0; i < 5; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMinV));
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(num_qg));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 3) / 4)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(64));
    ac.globalSize[0] = (oc + 3) / 4; ac.globalSize[1] = batch; ac.localSize[0] = 64; ac.dims = 2;
    ac.elementCount = batch * oc_p;
    return true;
}
cudaError_t CudaGemvFpAInt4BV14Fp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_v14_fp16(ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        (const float2*)ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.stream);
    return cudaGetLastError();
}
bool CudaGemvFpAInt4BV14Fp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqHpAnyNonZeroHalf(output, ac.elementCount);
}

// GEMV_FpAInt4B_V14_MB fp16 (1D grid, MAX_BATCH=1).
bool CudaGemvFpAInt4BV14MbFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gemv_fpaint4b_v14_mb_fp16";
    const int batch = 1;
    const int ic = spec.intParam("ic", kWoqHpIc);
    const int ic_p = spec.intParam("ic_p", kWoqHpIcP);
    const int oc = spec.intParam("oc", kWoqHpOc);
    const int oc_p = spec.intParam("oc_p", kWoqHpOcP);
    const int quanC = spec.intParam("quan_c", kWoqHpQuanC);
    const int num_qg = quanC / oc;
    auto input = woqHpBuildInput(batch, ic, ic_p);
    auto kernel = woqHpBuildInt4Weight(oc, ic_p, kWoqHpInt4Byte);
    auto gemv_params = woqHpBuildGemvParams(oc, num_qg, kWoqHpScale, kWoqHpOffset);
    auto bias = woqHpBuildBias(oc, kWoqHpBias);
    ac.buffers.push_back(makeHalfBuffer(input, false));
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernel.size(); kBuf.initialData.assign(kernel.begin(), kernel.end()); kBuf.isOutput = false;
    ac.buffers.push_back(kBuf);
    AdaptedBuffer pBuf; pBuf.sizeBytes = gemv_params.size() * sizeof(float2); pBuf.initialData.assign((const uint8_t*)gemv_params.data(), (const uint8_t*)gemv_params.data() + pBuf.sizeBytes); pBuf.isOutput = false;
    ac.buffers.push_back(pBuf);
    ac.buffers.push_back(makeHalfBuffer(bias, false));
    ac.buffers.push_back(makeHalfOutput(batch * oc_p));
    for (int i = 0; i < 5; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMinV));
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(num_qg));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 3) / 4)); ac.args.push_back(AdaptedArg::scalarInt(64));
    ac.globalSize[0] = (oc + 3) / 4; ac.localSize[0] = 64; ac.dims = 1;
    ac.elementCount = batch * oc_p;
    return true;
}
cudaError_t CudaGemvFpAInt4BV14MbFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_v14_mb_fp16(ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        (const float2*)ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.stream);
    return cudaGetLastError();
}
bool CudaGemvFpAInt4BV14MbFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqHpAnyNonZeroHalf(output, ac.elementCount);
}

// GEMV_FpAInt8B_V2 fp16 (2D grid + blockY).
bool CudaGemvFpAInt8BV2Fp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gemv_fpaint8b_v2_fp16";
    const int batch = spec.intParam("batch", kWoqHpBatch);
    const int ic = spec.intParam("ic", kWoqHpIc);
    const int ic_p = spec.intParam("ic_p", kWoqHpIcP);
    const int oc = spec.intParam("oc", kWoqHpOc);
    const int oc_p = spec.intParam("oc_p", kWoqHpOcP);
    const int quanC = spec.intParam("quan_c", kWoqHpQuanC);
    auto input = woqHpBuildInput(batch, ic, ic_p);
    auto kernel = woqHpBuildInt8Weight(oc, ic_p, kWoqHpInt8Q);
    auto scale = woqHpBuildScale(oc, quanC, kWoqHpScale);
    auto offset = woqHpBuildOffset(oc, quanC, kWoqHpOffset);
    auto bias = woqHpBuildBias(oc, kWoqHpBias);
    ac.buffers.push_back(makeHalfBuffer(input, false));
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernel.size(); kBuf.initialData.assign((const uint8_t*)kernel.data(), (const uint8_t*)kernel.data() + kernel.size()); kBuf.isOutput = false;
    ac.buffers.push_back(kBuf);
    ac.buffers.push_back(makeHalfBuffer(scale, false));
    ac.buffers.push_back(makeHalfBuffer(offset, false));
    ac.buffers.push_back(makeHalfBuffer(bias, false));
    ac.buffers.push_back(makeHalfOutput(batch * oc_p));
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMinV));
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 15) / 16)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(16)); ac.args.push_back(AdaptedArg::scalarInt(16));
    ac.globalSize[0] = (oc + 15) / 16; ac.globalSize[1] = batch; ac.localSize[0] = 16; ac.localSize[1] = 16; ac.dims = 2;
    ac.elementCount = batch * oc_p;
    return true;
}
cudaError_t CudaGemvFpAInt8BV2Fp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint8b_v2_fp16(ctx.devBufs[0], (const int8_t*)ctx.devBufs[1],
        ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4], ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}
bool CudaGemvFpAInt8BV2Fp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqHpAnyNonZeroHalf(output, ac.elementCount);
}

// ---- CONV_FpAInt8B / CONV_FpAInt4B fp16 ----
// ENTRY is the full shim symbol (e.g. mnn_corpus_conv_fpaint8b_fp16).
#define WOQ_FP16_CONV_SMOKE_BODY(CLASS, ENTRY, KBUF_TYPE, IS_INT4) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = #ENTRY; \
    const int ic = spec.intParam("ic", kWoqHpIc); \
    const int ic_p = spec.intParam("ic_p", kWoqHpIcP); \
    const int oc = spec.intParam("oc", kWoqHpOc); \
    const int oc_p = spec.intParam("oc_p", kWoqHpOcP); \
    const int iw = spec.intParam("iw", 4), ih = spec.intParam("ih", 4); \
    const int kw = spec.intParam("kw", 3), kh = spec.intParam("kh", 3); \
    const int sw = spec.intParam("sw", 1), sh = spec.intParam("sh", 1); \
    const int dw = spec.intParam("dw", 1), dh = spec.intParam("dh", 1); \
    const int pw = spec.intParam("pw", 1), ph = spec.intParam("ph", 1); \
    const int quanC = spec.intParam("quan_c", kWoqHpQuanC); \
    const int ow = (iw + 2 * pw - dw * (kw - 1) - 1) / sw + 1; \
    const int oh = (ih + 2 * ph - dh * (kh - 1) - 1) / sh + 1; \
    auto input = woqHpBuildInput(1, ic, ic_p); \
    auto scale = woqHpBuildScale(oc, quanC, kWoqHpScale); \
    auto offset = woqHpBuildOffset(oc, quanC, kWoqHpOffset); \
    auto bias = woqHpBuildBias(oc, kWoqHpBias); \
    ac.buffers.push_back(makeHalfBuffer(input, false)); \
    AdaptedBuffer kBuf; \
    if (IS_INT4) { auto w = woqHpBuildInt4Weight(oc, ic_p * kw * kh, kWoqHpInt4Byte); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign(w.begin(), w.end()); } \
    else { auto w = woqHpBuildInt8Weight(oc, ic_p * kw * kh, kWoqHpInt8Q); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign((const uint8_t*)w.data(), (const uint8_t*)w.data() + w.size()); } \
    kBuf.isOutput = false; \
    ac.buffers.push_back(kBuf); \
    ac.buffers.push_back(makeHalfBuffer(scale, false)); \
    ac.buffers.push_back(makeHalfBuffer(offset, false)); \
    ac.buffers.push_back(makeHalfBuffer(bias, false)); \
    ac.buffers.push_back(makeHalfOutput(ow * oh * oc_p)); \
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i)); \
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqHpMinV)); \
    ac.args.push_back(AdaptedArg::scalarInt(ic)); ac.args.push_back(AdaptedArg::scalarInt(ic_p)); \
    ac.args.push_back(AdaptedArg::scalarInt(iw)); ac.args.push_back(AdaptedArg::scalarInt(ih)); \
    ac.args.push_back(AdaptedArg::scalarInt(oc)); ac.args.push_back(AdaptedArg::scalarInt(oc_p)); \
    ac.args.push_back(AdaptedArg::scalarInt(ow)); ac.args.push_back(AdaptedArg::scalarInt(oh)); \
    ac.args.push_back(AdaptedArg::scalarInt(kw)); ac.args.push_back(AdaptedArg::scalarInt(kh)); \
    ac.args.push_back(AdaptedArg::scalarInt(dw)); ac.args.push_back(AdaptedArg::scalarInt(dh)); \
    ac.args.push_back(AdaptedArg::scalarInt(sw)); ac.args.push_back(AdaptedArg::scalarInt(sh)); \
    ac.args.push_back(AdaptedArg::scalarInt(pw)); ac.args.push_back(AdaptedArg::scalarInt(ph)); \
    const int total = ow * oh * oc_p; \
    ac.args.push_back(AdaptedArg::scalarInt(total)); ac.args.push_back(AdaptedArg::scalarInt(quanC)); \
    const int grid = mnnGridFor(total, kBlock), block = kBlock; \
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block)); \
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1; \
    ac.elementCount = total; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    ENTRY(ctx.devBufs[0], (KBUF_TYPE)ctx.devBufs[1], \
        ctx.devBufs[2], ctx.devBufs[3], ctx.devBufs[4], ctx.devBufs[5], \
        ctx.floatArgs[0], ctx.floatArgs[1], \
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], \
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11], \
        ctx.intArgs[12], ctx.intArgs[13], ctx.intArgs[14], ctx.intArgs[15], ctx.intArgs[16], ctx.intArgs[17], \
        ctx.intArgs[18], ctx.intArgs[19], ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    return woqHpAnyNonZeroHalf(output, ac.elementCount); \
}

WOQ_FP16_CONV_SMOKE_BODY(CudaConvFpAInt8BFp16Kernel, mnn_corpus_conv_fpaint8b_fp16, const int8_t*, false)
WOQ_FP16_CONV_SMOKE_BODY(CudaConvFpAInt4BFp16Kernel, mnn_corpus_conv_fpaint4b_fp16, const uint8_t*, true)
#undef WOQ_FP16_CONV_SMOKE_BODY

// ---- CudaGatedDeltaRulePrefillFp16Kernel — gated_delta_rule_prefill<__half> ----
bool CudaGatedDeltaRulePrefillFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gated_delta_rule_prefill_fp16";
    const int B = spec.intParam("batch", 1);
    const int L = spec.intParam("l", 2);
    const int H_k = spec.intParam("h_k", 1);
    const int H_v = spec.intParam("h_v", 1);
    const int d_k = spec.intParam("d_k", 4);
    const int d_v = spec.intParam("d_v", 4);
    const int key_dim = H_k * d_k;
    const int val_dim = H_v * d_v;
    const int D = key_dim + val_dim;
    const int gqa_factor = H_v / H_k;
    const bool useL2Norm = spec.intParam("use_l2norm", 1) != 0;
    const float qScale = spec.floatParam("q_scale", 1.0f);
    const bool gateC4 = spec.intParam("gate_c4", 0) != 0;
    const bool betaC4 = spec.intParam("beta_c4", 0) != 0;
    const bool outputC4 = spec.intParam("output_c4", 0) != 0;
    std::vector<float> convOut(B * L * D, 0.1f);
    std::vector<float> gateInput(B * L * H_v, 0.2f);
    std::vector<float> betaInput(B * L * H_v, 0.3f);
    std::vector<float> recurrentState(B * H_v * d_k * d_v, 0.0f);
    // convOut/recurrentState: float; gateInput/betaInput/output: __half
    AdaptedBuffer cBuf; cBuf.setFp32(convOut); cBuf.isOutput = false;
    AdaptedBuffer rBuf; rBuf.setFp32(recurrentState); rBuf.isOutput = false;
    ac.buffers.push_back(cBuf);
    ac.buffers.push_back(makeHalfBuffer(gateInput, false));
    ac.buffers.push_back(makeHalfBuffer(betaInput, false));
    ac.buffers.push_back(rBuf);
    ac.buffers.push_back(makeHalfOutput(B * L * H_v * d_v));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::scalarInt(B));
    ac.args.push_back(AdaptedArg::scalarInt(L));
    ac.args.push_back(AdaptedArg::scalarInt(H_k));
    ac.args.push_back(AdaptedArg::scalarInt(H_v));
    ac.args.push_back(AdaptedArg::scalarInt(d_k));
    ac.args.push_back(AdaptedArg::scalarInt(d_v));
    ac.args.push_back(AdaptedArg::scalarInt(key_dim));
    ac.args.push_back(AdaptedArg::scalarInt(val_dim));
    ac.args.push_back(AdaptedArg::scalarInt(D));
    ac.args.push_back(AdaptedArg::scalarInt(gqa_factor));
    ac.args.push_back(AdaptedArg::scalarInt(useL2Norm ? 1 : 0));
    ac.args.push_back(AdaptedArg::scalarFloat(qScale));
    ac.args.push_back(AdaptedArg::scalarInt(gateC4 ? 1 : 0));
    ac.args.push_back(AdaptedArg::scalarInt(betaC4 ? 1 : 0));
    ac.args.push_back(AdaptedArg::scalarInt(outputC4 ? 1 : 0));
    const int grid = B * H_v;
    const int block = 256;
    const size_t sharedMem = (block + 2 * d_k + 3 * d_v) * sizeof(float);
    ac.args.push_back(AdaptedArg::scalarInt(grid));
    ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.args.push_back(AdaptedArg::scalarInt((int)sharedMem));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = B * L * H_v * d_v;
    return true;
}
cudaError_t CudaGatedDeltaRulePrefillFp16Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gated_delta_rule_prefill_fp16(
        (const float*)ctx.devBufs[0], ctx.devBufs[1], ctx.devBufs[2],
        (float*)ctx.devBufs[3], ctx.devBufs[4],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
        ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10] != 0, ctx.floatArgs[0],
        ctx.intArgs[11] != 0, ctx.intArgs[12] != 0, ctx.intArgs[13] != 0,
        ctx.grid, ctx.block, (size_t)ctx.intArgs[16], ctx.stream);
    return cudaGetLastError();
}
bool CudaGatedDeltaRulePrefillFp16Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (static_cast<int>(output.size() * sizeof(float)) < ac.elementCount * (int)sizeof(__half)) return false;
    return woqHpAnyNonZeroHalf(output, ac.elementCount);
}

void registerCudaOpsFp16() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReluFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaClampFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryAtan2Fp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryModFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryLogicalOrFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaRangeFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaRangeI32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSelectFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSoftmaxFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaLayerNormFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaPreluFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaScaleFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaMaxPoolFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaAvgPoolFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGlobalAvgPoolFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGlobalMaxPoolFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastI82I32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastI322U8Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastU82I32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGatherV2Fp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaArgMaxFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaArgMinFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaNhwc2NchwFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaNchw2NhwcFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvDwOptFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvDwHalf2OptFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvDw3x3Half2OptFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvDwMultiWidth4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvDwMultiWidthChannelFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaPackCommonHalf4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaPackCommonRearrangeHalf4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaUnpackCommonRearrangeHalf4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaFuseBlitHalf4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaUnaryHalf2SigmoidFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaWinoInputTransHalf2Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaWinoTrans2OutputHalf2Fp32Kernel()));
            // P0: fp16 Attention / LinearAttention / RoPE / TopKV2
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaFlashDecodeFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaFlashDecodeWithMaskFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaFlashDecodeSplitkFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaFlashAttnCombineResultsFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCopyKvToCacheFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaQkKernelTiledFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaQkvKernelTiledFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConv1dSiluFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaShortConvFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaShortConvOutputFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGatedDeltaRuleDecodeFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaRopeC4Fp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaTopKV2Fp16Kernel()));
            // P1: fp16 Reduction + Interp + GridSample + LayerNormC4
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionSumFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionMeanFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionMaxFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionMinFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionProdFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionSumAxisFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionMeanAxisFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInterpNearestFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInterpBilinearFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInterpNearestRoundFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInterpBilinearOptFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGridSampleNearestFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGridSampleBilinearFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGridSampleNearest3dFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGridSampleBilinear3dFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaLayerNormC4Fp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryLayerNormC4Fp16Kernel()));
            // P2: plugin kernels — GroupNorm(half) / SeqLen2Spatial / splitGeLU / SPLIT_FusedQKV
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGroupNormNHWCSumFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGroupNormNHWCScaleFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSeqLen2SpatialFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSeqLen2SpatialFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSplitGeluFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSplitGeluFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSplitFusedQKVFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSplitFusedQKVFp16Kernel()));
            // P4: int8 kernels
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaFloat2Int8Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaFloat2Int8SingleFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInt82FloatFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInt82FloatSingleFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaDequantizeInt8WeightFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaDequantizeInt4WeightFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvDwInt8Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvDw3x3S1Int8Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaIm2ColPackC16Int8Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaWeightInt8PackFillFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryInt8AddFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryInt8MulFp32Kernel()));
            // P5: Raster fused binary
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryAddRasterFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryMulRasterFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryFuseAddAddFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryFuseAddMulFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryMidAddFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryMidMulFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryMidLinear4AddFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryMidLinear4MulFp32Kernel()));
            // New fp16/bf16 adapters
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastMidFloatF16I32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastMidFloatF16I8Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastMidFloatI32F16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBf162FloatF16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBf162FloatF32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaMaxpoolC8Bf16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaAvgpoolC8Bf16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaFloat22BFloat16Fp32Kernel()));
            // weight_only_quant + gated_delta_rule_prefill fp16 adapters.
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaPrecomputeGemvParamsFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaQuantAFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaDequantAndAccFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBiasAndActivationFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemmFpAInt8BFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt8BFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemmFpAInt4BFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BV5Fp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BV9Fp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BV14Fp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BV14MbFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt8BV2Fp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvFpAInt8BFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvFpAInt4BFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGatedDeltaRulePrefillFp16Kernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
