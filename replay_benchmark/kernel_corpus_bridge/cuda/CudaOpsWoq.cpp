// CudaOpsWoq.cpp - weight_only_quant (conv_fpa_intb) fp32 adapters + the
// gated_delta_rule_prefill fp32 adapter. Compiled by g++ (alongside CudaOps.cpp).
// The fp16 counterparts live in CudaOpsFp16.cu (nvcc). The standalone
// CudaGemmInt8Fp32Kernel is implemented in CudaOpsMisc.cpp; this file only
// registers it (plus the 20 new fp32 adapter implementations defined here).
//
// All adapters are smoke-only: launch the shim, then validate the output is
// not all-zero. Small geometry: batch=1, ic=8, ic_p=8, oc=4, oc_p=8, quanC=4.
// Weight int8 value=2; int4 packed=0xAA (both nibbles = q+8 = 2+8 = 10);
// scale=0.1f, offset=0.0f, bias=0.5f; input 0.1f*(i%7) pattern.
#include "CudaOps.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Shim forward declarations (defined in weight_only_quant.cu / attention.cu,
// compiled into replay_cuda_corpus and linked at device-link time).
extern "C" {
void mnn_corpus_gemm_int8_fp32(const int8_t*, const int8_t*, int32_t*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_precompute_sumbq_fp32(const int8_t*, int32_t*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_rearrange_packed_weight_int4_fp32(const uint8_t*, uint8_t*, int, size_t, int, int, int, int, cudaStream_t);
void mnn_corpus_rearrange_weight_int4_fp32(const int8_t*, uint8_t*, int, size_t, int, int, int, int, cudaStream_t);
void mnn_corpus_rearrange_weight_int8_fp32(const int8_t*, int8_t*, int, size_t, int, int, int, int, cudaStream_t);
void mnn_corpus_precomputegemvparams_fp32(const float*, const float*, float2*, int, int, int, cudaStream_t);
void mnn_corpus_quanta_fp32(const float*, int8_t*, float*, float*, int32_t*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_dequantandacc_fp32(const int32_t*, float*, const float*, const float*, const float*, const float*, const int32_t*, int, int, const int32_t*, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_biasandactivation_fp32(float*, const float*, float, float, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemm_fpaint8b_fp32(const float*, const int8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint8b_fp32(const float*, const int8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemm_fpaint4b_fp32(const float*, const uint8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_fp32(const float*, const uint8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_v5_fp32(const float*, const uint8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_v9_fp32(const float*, const uint8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_v14_fp32(const float*, const uint8_t*, const float2*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint4b_v14_mb_fp32(const float*, const uint8_t*, const float2*, const float*, float*, float, float, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gemv_fpaint8b_v2_fp32(const float*, const int8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_fpaint8b_fp32(const float*, const int8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_fpaint4b_fp32(const float*, const uint8_t*, const float*, const float*, const float*, float*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_gated_delta_rule_prefill_fp32(const float*, const void*, const void*, float*, void*, int, int, int, int, int, int, int, int, int, int, bool, float, bool, bool, bool, int, int, size_t, cudaStream_t);
}

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

namespace {
// Local copies of the WOQ helpers from CudaOpsMisc.cpp's anonymous namespace.
// (Cannot share anonymous-namespace symbols across TUs.)
constexpr int kWoqBatch = 1;
constexpr int kWoqIc = 8;
constexpr int kWoqIcP = 8;        // ic padded to PACK_NUMBER (4) multiple
constexpr int kWoqOc = 4;
constexpr int kWoqOcP = 8;        // oc padded to 8 (oc_p >= oc, multiple of 4)
constexpr int kWoqQuanC = 4;      // = oc * num_qg (num_qg=1)
constexpr float kWoqScale = 0.1f;
constexpr float kWoqOffset = 0.0f;
constexpr float kWoqBias = 0.5f;
constexpr float kWoqMaxV = 6.0f;
constexpr float kWoqMinV = 0.0f;
const int8_t kWoqInt8Q = 2;       // weight int8 value
const uint8_t kWoqInt4Byte = 0xAA;// both nibbles = q+8 = 2+8 = 10

std::vector<int8_t> woqBuildInt8Weight(int oc, int ic_p, int8_t q) {
    return std::vector<int8_t>((size_t)oc * ic_p, q);
}
std::vector<uint8_t> woqBuildInt4Weight(int oc, int ic_p, uint8_t byte) {
    return std::vector<uint8_t>((size_t)oc * (ic_p / 2), byte);
}
std::vector<float> woqBuildInput(int batch, int ic, int ic_p) {
    std::vector<float> v((size_t)batch * ic_p, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int k = 0; k < ic; ++k) v[b * ic_p + k] = 0.1f * (k % 7);
    return v;
}
std::vector<float> woqBuildScale(int oc, int quanC, float s) {
    const int num_qg = quanC / oc;
    return std::vector<float>((size_t)oc * (num_qg > 0 ? num_qg : 1), s);
}
std::vector<float> woqBuildOffset(int oc, int quanC, float o) {
    const int num_qg = quanC / oc;
    return std::vector<float>((size_t)oc * (num_qg > 0 ? num_qg : 1), o);
}
std::vector<float> woqBuildBias(int oc_p, float b) {
    return std::vector<float>((size_t)oc_p, b);
}
// Build float2 params for V14: .x=scale, .y=offset-8*scale.
std::vector<float2> woqBuildGemvParams(int oc, int num_qg, float scale, float offset) {
    std::vector<float2> v((size_t)oc * num_qg);
    for (size_t i = 0; i < v.size(); ++i) { v[i].x = scale; v[i].y = offset - 8.0f * scale; }
    return v;
}
// True if `output` (reinterpreted as float[n]) has any non-zero among first n.
bool woqAnyNonZeroFp32(const std::vector<float>& output, int n) {
    int lim = std::min(n, (int)output.size());
    for (int i = 0; i < lim; ++i) if (output[i] != 0.0f) return true;
    return false;
}
} // namespace (anonymous)

// ============================================================================
// 1. CudaPrecomputeSumbqFp32Kernel — Precompute_SumBq (non-template, fp32).
//    sum_B_q_out[g*oc + n] = sum over ic_per_group of B_q[n*ic_p + g*icpg + k].
// ============================================================================
bool CudaPrecomputeSumbqFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_precompute_sumbq_fp32";
    const int num_groups = spec.intParam("num_groups", 1);
    const int ic_per_group = spec.intParam("ic_per_group", kWoqIc);
    const int oc = spec.intParam("oc", kWoqOc);
    const int ic_p = spec.intParam("ic_p", kWoqIcP);
    auto B_q = woqBuildInt8Weight(oc, ic_p, kWoqInt8Q);
    AdaptedBuffer bBuf; bBuf.sizeBytes = B_q.size(); bBuf.initialData.assign((const uint8_t*)B_q.data(), (const uint8_t*)B_q.data() + B_q.size()); bBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.sizeBytes = (size_t)num_groups * oc * sizeof(int32_t); oBuf.isOutput = true;
    ac.buffers.push_back(bBuf); ac.buffers.push_back(oBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(num_groups)); ac.args.push_back(AdaptedArg::scalarInt(ic_per_group));
    ac.args.push_back(AdaptedArg::scalarInt(oc)); ac.args.push_back(AdaptedArg::scalarInt(ic_p));
    const int total = num_groups * oc * ic_p;
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = num_groups * oc;
    return true;
}
cudaError_t CudaPrecomputeSumbqFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_precompute_sumbq_fp32((const int8_t*)ctx.devBufs[0], (int32_t*)ctx.devBufs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
        ctx.intArgs[4], ctx.intArgs[5], ctx.stream);
    return cudaGetLastError();
}
bool CudaPrecomputeSumbqFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if ((int)output.size() * 4 < ac.elementCount * 4) return false;
    return woqAnyNonZeroFp32(output, ac.elementCount);
}

// ============================================================================
// 2. CudaRearrangePackedWeightInt4Fp32Kernel — packed-int4 weight reorder.
// ============================================================================
bool CudaRearrangePackedWeightInt4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_rearrange_packed_weight_int4_fp32";
    const int oc = spec.intParam("oc", kWoqOc);
    const int ic = spec.intParam("ic", kWoqIc);
    const int khw = spec.intParam("khw", 9);
    const int icp2 = (ic + 1) / 2;
    std::vector<uint8_t> param((size_t)oc * khw * icp2, kWoqInt4Byte);
    AdaptedBuffer pBuf; pBuf.sizeBytes = param.size(); pBuf.initialData.assign(param.begin(), param.end()); pBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.sizeBytes = param.size(); oBuf.isOutput = true;
    ac.buffers.push_back(pBuf); ac.buffers.push_back(oBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(khw));
    ac.args.push_back(AdaptedArg::scalarInt((int)param.size())); // maxCount (bytes)
    ac.args.push_back(AdaptedArg::scalarInt(oc)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    const int total = oc * khw * icp2;
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaRearrangePackedWeightInt4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_rearrange_packed_weight_int4_fp32((const uint8_t*)ctx.devBufs[0], (uint8_t*)ctx.devBufs[1],
        ctx.intArgs[0], (size_t)ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
        ctx.intArgs[4], ctx.intArgs[5], ctx.stream);
    return cudaGetLastError();
}
bool CudaRearrangePackedWeightInt4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if ((int)output.size() * 4 < ac.elementCount) return false;
    // output is raw bytes; reinterpret as uint8_t and check non-zero.
    const uint8_t* p = reinterpret_cast<const uint8_t*>(output.data());
    int lim = std::min(ac.elementCount, (int)(output.size() * 4));
    for (int i = 0; i < lim; ++i) if (p[i] != 0) return true;
    return false;
}

// ============================================================================
// 3. CudaRearrangeWeightInt4Fp32Kernel — int8→packed-int4 weight reorder.
// ============================================================================
bool CudaRearrangeWeightInt4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_rearrange_weight_int4_fp32";
    const int oc = spec.intParam("oc", kWoqOc);
    const int ic = spec.intParam("ic", kWoqIc);
    const int khw = spec.intParam("khw", 9);
    const int icp2 = (ic + 1) / 2;
    std::vector<int8_t> param((size_t)oc * khw * ic, kWoqInt8Q);
    AdaptedBuffer pBuf; pBuf.sizeBytes = param.size(); pBuf.initialData.assign((const uint8_t*)param.data(), (const uint8_t*)param.data() + param.size()); pBuf.isOutput = false;
    const size_t outBytes = (size_t)oc * khw * icp2;
    AdaptedBuffer oBuf; oBuf.sizeBytes = outBytes; oBuf.isOutput = true;
    ac.buffers.push_back(pBuf); ac.buffers.push_back(oBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(khw));
    ac.args.push_back(AdaptedArg::scalarInt((int)outBytes)); // maxCount
    ac.args.push_back(AdaptedArg::scalarInt(oc)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    const int total = oc * khw * icp2;
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = (int)outBytes;
    return true;
}
cudaError_t CudaRearrangeWeightInt4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_rearrange_weight_int4_fp32((const int8_t*)ctx.devBufs[0], (uint8_t*)ctx.devBufs[1],
        ctx.intArgs[0], (size_t)ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
        ctx.intArgs[4], ctx.intArgs[5], ctx.stream);
    return cudaGetLastError();
}
bool CudaRearrangeWeightInt4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if ((int)output.size() * 4 < ac.elementCount) return false;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(output.data());
    int lim = std::min(ac.elementCount, (int)(output.size() * 4));
    for (int i = 0; i < lim; ++i) if (p[i] != 0) return true;
    return false;
}

// ============================================================================
// 4. CudaRearrangeWeightInt8Fp32Kernel — int8 weight reorder [oc,ic]→[oc,ic_p].
// ============================================================================
bool CudaRearrangeWeightInt8Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_rearrange_weight_int8_fp32";
    const int oc = spec.intParam("oc", kWoqOc);
    const int ic = spec.intParam("ic", kWoqIc);
    const int khw = spec.intParam("khw", 9);
    std::vector<int8_t> param((size_t)oc * khw * ic, kWoqInt8Q);
    AdaptedBuffer pBuf; pBuf.sizeBytes = param.size(); pBuf.initialData.assign((const uint8_t*)param.data(), (const uint8_t*)param.data() + param.size()); pBuf.isOutput = false;
    const size_t outBytes = param.size();
    AdaptedBuffer oBuf; oBuf.sizeBytes = outBytes; oBuf.isOutput = true;
    ac.buffers.push_back(pBuf); ac.buffers.push_back(oBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(khw));
    ac.args.push_back(AdaptedArg::scalarInt((int)outBytes)); // maxCount
    ac.args.push_back(AdaptedArg::scalarInt(oc)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    const int total = oc * khw * ic;
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = (int)outBytes;
    return true;
}
cudaError_t CudaRearrangeWeightInt8Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_rearrange_weight_int8_fp32((const int8_t*)ctx.devBufs[0], (int8_t*)ctx.devBufs[1],
        ctx.intArgs[0], (size_t)ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
        ctx.intArgs[4], ctx.intArgs[5], ctx.stream);
    return cudaGetLastError();
}
bool CudaRearrangeWeightInt8Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if ((int)output.size() * 4 < ac.elementCount) return false;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(output.data());
    int lim = std::min(ac.elementCount, (int)(output.size() * 4));
    for (int i = 0; i < lim; ++i) if (p[i] != 0) return true;
    return false;
}

// ============================================================================
// 5. CudaPrecomputeGemvParamsFp32Kernel — PrecomputeGemvParams<float>.
//    params[i] = make_float2(scale[i], offset[i]).
// ============================================================================
bool CudaPrecomputeGemvParamsFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_precomputegemvparams_fp32";
    const int oc = spec.intParam("oc", kWoqOc);
    const int num_qg = spec.intParam("num_qg", 1);
    const int total = oc * num_qg;
    auto scale = woqBuildScale(oc, oc * num_qg, kWoqScale);
    auto offset = woqBuildOffset(oc, oc * num_qg, kWoqOffset);
    AdaptedBuffer sBuf; sBuf.setFp32(scale); sBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.setFp32(offset); oBuf.isOutput = false;
    AdaptedBuffer pBuf; pBuf.sizeBytes = (size_t)total * sizeof(float2); pBuf.isOutput = true;
    ac.buffers.push_back(sBuf); ac.buffers.push_back(oBuf); ac.buffers.push_back(pBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1)); ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaPrecomputeGemvParamsFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_precomputegemvparams_fp32((const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (float2*)ctx.devBufs[2],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.stream);
    return cudaGetLastError();
}
bool CudaPrecomputeGemvParamsFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // float2 output: 2 floats per element.
    if ((int)output.size() < ac.elementCount * 2) return false;
    for (int i = 0; i < ac.elementCount; ++i) if (output[i * 2] != 0.0f) return true;
    return false;
}

// ============================================================================
// 6. CudaQuantAFp32Kernel — QuantA<float>. Quantize A_sub_fp[M,K] row-wise.
// ============================================================================
bool CudaQuantAFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_quanta_fp32";
    const int M = spec.intParam("m", 4);
    const int K_i = spec.intParam("k", kWoqIc);
    const int lda = K_i;
    std::vector<float> A_sub_fp((size_t)M * lda, 0.0f);
    for (int i = 0; i < M * lda; ++i) A_sub_fp[i] = 0.1f * (i % 7);
    AdaptedBuffer aBuf; aBuf.setFp32(A_sub_fp); aBuf.isOutput = false;
    AdaptedBuffer qBuf; qBuf.sizeBytes = (size_t)M * lda; qBuf.isOutput = true;
    AdaptedBuffer scBuf; scBuf.sizeBytes = M * sizeof(float); scBuf.isOutput = true;
    AdaptedBuffer ofBuf; ofBuf.sizeBytes = M * sizeof(float); ofBuf.isOutput = true;
    AdaptedBuffer suBuf; suBuf.sizeBytes = M * sizeof(int32_t); suBuf.isOutput = true;
    ac.buffers.push_back(aBuf); ac.buffers.push_back(qBuf); ac.buffers.push_back(scBuf); ac.buffers.push_back(ofBuf); ac.buffers.push_back(suBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2)); ac.args.push_back(AdaptedArg::buffer(3)); ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::scalarInt(M)); ac.args.push_back(AdaptedArg::scalarInt(K_i)); ac.args.push_back(AdaptedArg::scalarInt(lda));
    const int total = M * lda;
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaQuantAFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_quanta_fp32((const float*)ctx.devBufs[0], (int8_t*)ctx.devBufs[1],
        (float*)ctx.devBufs[2], (float*)ctx.devBufs[3], (int32_t*)ctx.devBufs[4],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.stream);
    return cudaGetLastError();
}
bool CudaQuantAFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // qBuf (int8) is the first output; check non-zero among first elementCount.
    const int8_t* q = reinterpret_cast<const int8_t*>(output.data());
    int lim = std::min(ac.elementCount, (int)(output.size() * 4));
    for (int i = 0; i < lim; ++i) if (q[i] != 0) return true;
    return false;
}

// ============================================================================
// 7. CudaDequantAndAccFp32Kernel — DequantAndAcc<float>. Output [M, ldc] float.
//    Uses precomputed C_q[M,N] (int32), scale/offset A and B, sum_A_q.
// ============================================================================
bool CudaDequantAndAccFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_dequantandacc_fp32";
    const int M = spec.intParam("m", 4);
    const int N = spec.intParam("n", kWoqOc);
    const int K_i = spec.intParam("k", kWoqIc);
    const int ldc = N;
    const int num_oc_groups = 1, group_idx = 0;
    std::vector<int32_t> C_q((size_t)M * ldc, 8); // some int32 acc
    std::vector<float> scale_A(M, kWoqScale), offset_A(M, kWoqOffset);
    std::vector<float> base_scale_B(N, kWoqScale), base_offset_B(N, kWoqOffset);
    std::vector<int32_t> base_sum_B_q(N, K_i * kWoqInt8Q);
    std::vector<int32_t> sum_A_q(M, K_i * kWoqInt8Q);
    AdaptedBuffer cqBuf; cqBuf.sizeBytes = C_q.size() * sizeof(int32_t); cqBuf.initialData.assign((const uint8_t*)C_q.data(), (const uint8_t*)C_q.data() + C_q.size() * 4); cqBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.sizeBytes = (size_t)M * ldc * sizeof(float); oBuf.isOutput = true;
    AdaptedBuffer saBuf; saBuf.setFp32(scale_A); saBuf.isOutput = false;
    AdaptedBuffer oaBuf; oaBuf.setFp32(offset_A); oaBuf.isOutput = false;
    AdaptedBuffer bsbBuf; bsbBuf.setFp32(base_scale_B); bsbBuf.isOutput = false;
    AdaptedBuffer bobBuf; bobBuf.setFp32(base_offset_B); bobBuf.isOutput = false;
    AdaptedBuffer bsbqBuf; bsbqBuf.sizeBytes = base_sum_B_q.size() * 4; bsbqBuf.initialData.assign((const uint8_t*)base_sum_B_q.data(), (const uint8_t*)base_sum_B_q.data() + base_sum_B_q.size() * 4); bsbqBuf.isOutput = false;
    AdaptedBuffer saqBuf; saqBuf.sizeBytes = sum_A_q.size() * 4; saqBuf.initialData.assign((const uint8_t*)sum_A_q.data(), (const uint8_t*)sum_A_q.data() + sum_A_q.size() * 4); saqBuf.isOutput = false;
    ac.buffers.push_back(cqBuf); ac.buffers.push_back(oBuf); ac.buffers.push_back(saBuf); ac.buffers.push_back(oaBuf);
    ac.buffers.push_back(bsbBuf); ac.buffers.push_back(bobBuf); ac.buffers.push_back(bsbqBuf); ac.buffers.push_back(saqBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2)); ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4)); ac.args.push_back(AdaptedArg::buffer(5));
    ac.args.push_back(AdaptedArg::buffer(6)); ac.args.push_back(AdaptedArg::buffer(7));
    ac.args.push_back(AdaptedArg::scalarInt(group_idx)); ac.args.push_back(AdaptedArg::scalarInt(num_oc_groups));
    ac.args.push_back(AdaptedArg::scalarInt(7)); // sum_A_q_in index (buffer 7)
    ac.args.push_back(AdaptedArg::scalarInt(M)); ac.args.push_back(AdaptedArg::scalarInt(N)); ac.args.push_back(AdaptedArg::scalarInt(K_i));
    ac.args.push_back(AdaptedArg::scalarInt(ldc));
    const int gridX = (N + 15) / 16, gridY = (M + 15) / 16, blockX = 16, blockY = 16;
    ac.args.push_back(AdaptedArg::scalarInt(gridX)); ac.args.push_back(AdaptedArg::scalarInt(gridY));
    ac.args.push_back(AdaptedArg::scalarInt(blockX)); ac.args.push_back(AdaptedArg::scalarInt(blockY));
    ac.globalSize[0] = gridX; ac.globalSize[1] = gridY; ac.localSize[0] = blockX; ac.localSize[1] = blockY; ac.dims = 2;
    ac.elementCount = M * ldc;
    return true;
}
cudaError_t CudaDequantAndAccFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // sum_A_q_in is passed as a pointer (buffer 7); the int arg 10 is unused by
    // this adapter's layout but the shim expects it — pass buffer 7 directly.
    mnn_corpus_dequantandacc_fp32((const int32_t*)ctx.devBufs[0], (float*)ctx.devBufs[1],
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3],
        (const float*)ctx.devBufs[4], (const float*)ctx.devBufs[5],
        (const int32_t*)ctx.devBufs[6], ctx.intArgs[0], ctx.intArgs[1],
        (const int32_t*)ctx.devBufs[7], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}
bool CudaDequantAndAccFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqAnyNonZeroFp32(output, ac.elementCount);
}

// ============================================================================
// 8. CudaBiasAndActivationFp32Kernel — BiasAndActivation<float>.
//    data[M, ldc] += bias[N]; clamp to [minV, maxV].
// ============================================================================
bool CudaBiasAndActivationFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_biasandactivation_fp32";
    const int M = spec.intParam("m", 1);
    const int N = spec.intParam("n", kWoqOc);
    const int ldc = N;
    std::vector<float> data((size_t)M * ldc, 0.0f);
    for (int i = 0; i < M * ldc; ++i) data[i] = 0.1f * (i % 7);
    auto bias = woqBuildBias(N, kWoqBias);
    AdaptedBuffer dBuf; dBuf.setFp32(data); dBuf.isOutput = true; // in-place
    AdaptedBuffer bBuf; bBuf.setFp32(bias); bBuf.isOutput = false;
    ac.buffers.push_back(dBuf); ac.buffers.push_back(bBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqMinV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqMaxV));
    ac.args.push_back(AdaptedArg::scalarInt(M)); ac.args.push_back(AdaptedArg::scalarInt(N)); ac.args.push_back(AdaptedArg::scalarInt(ldc));
    const int total = M * ldc;
    const int grid = mnnGridFor(total, kBlock), block = kBlock;
    ac.args.push_back(AdaptedArg::scalarInt(grid)); ac.args.push_back(AdaptedArg::scalarInt(block));
    ac.globalSize[0] = grid; ac.localSize[0] = block; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaBiasAndActivationFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_biasandactivation_fp32((float*)ctx.devBufs[0], (const float*)ctx.devBufs[1],
        ctx.floatArgs[0], ctx.floatArgs[1], ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
        ctx.intArgs[3], ctx.intArgs[4], ctx.stream);
    return cudaGetLastError();
}
bool CudaBiasAndActivationFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqAnyNonZeroFp32(output, ac.elementCount);
}

// ----------------------------------------------------------------------------
// Helper macro to reduce boilerplate for the FpA*(int8/int4)B GEMM/GEMV/CONV
// fp32 adapters. ENTRY is the full shim symbol (e.g. mnn_corpus_gemm_fpaint8b_fp32).
// Each launches a 2D-grid shim with the standard argument ordering: input,
// kernel, scale, offset, bias, output, maxV, minV, dims..., gridX, gridY,
// blockX, blockY (GEMM/GEMV_V2) or gridX, gridY, blockX (V5/V9/V14/V14_MB).
// ----------------------------------------------------------------------------
#define WOQ_FP32_SMOKE_BODY(CLASS, ENTRY, KBUF_TYPE, IS_INT4, EXTRA_GRID) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = #ENTRY; \
    const int batch = spec.intParam("batch", kWoqBatch); \
    const int ic = spec.intParam("ic", kWoqIc); \
    const int ic_p = spec.intParam("ic_p", kWoqIcP); \
    const int oc = spec.intParam("oc", kWoqOc); \
    const int oc_p = spec.intParam("oc_p", kWoqOcP); \
    const int quanC = spec.intParam("quan_c", kWoqQuanC); \
    auto input = woqBuildInput(batch, ic, ic_p); \
    auto scale = woqBuildScale(oc, quanC, kWoqScale); \
    auto offset = woqBuildOffset(oc, quanC, kWoqOffset); \
    auto bias = woqBuildBias(oc, kWoqBias); \
    AdaptedBuffer iBuf; iBuf.setFp32(input); iBuf.isOutput = false; \
    AdaptedBuffer kBuf; \
    if (IS_INT4) { auto w = woqBuildInt4Weight(oc, ic_p, kWoqInt4Byte); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign(w.begin(), w.end()); } \
    else { auto w = woqBuildInt8Weight(oc, ic_p, kWoqInt8Q); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign((const uint8_t*)w.data(), (const uint8_t*)w.data() + w.size()); } \
    kBuf.isOutput = false; \
    AdaptedBuffer sBuf; sBuf.setFp32(scale); sBuf.isOutput = false; \
    AdaptedBuffer oBuf_; oBuf_.setFp32(offset); oBuf_.isOutput = false; \
    AdaptedBuffer bBuf; bBuf.setFp32(bias); bBuf.isOutput = false; \
    AdaptedBuffer outBuf; outBuf.sizeBytes = (size_t)batch * oc_p * sizeof(float); outBuf.isOutput = true; \
    ac.buffers.push_back(iBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(sBuf); \
    ac.buffers.push_back(oBuf_); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf); \
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i)); \
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqMinV)); \
    ac.args.push_back(AdaptedArg::scalarInt(ic)); ac.args.push_back(AdaptedArg::scalarInt(ic_p)); \
    ac.args.push_back(AdaptedArg::scalarInt(oc)); ac.args.push_back(AdaptedArg::scalarInt(oc_p)); \
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(quanC)); \
    const int gridX = (oc + 15) / 16, gridY = batch, blockX = 16, blockY = 16; \
    ac.args.push_back(AdaptedArg::scalarInt(gridX)); ac.args.push_back(AdaptedArg::scalarInt(gridY)); \
    EXTRA_GRID \
    ac.globalSize[0] = gridX; ac.globalSize[1] = gridY; ac.localSize[0] = blockX; ac.localSize[1] = blockY; ac.dims = 2; \
    ac.elementCount = batch * oc_p; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const { \
    ENTRY((const float*)ctx.devBufs[0], (KBUF_TYPE)ctx.devBufs[1], \
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (const float*)ctx.devBufs[4], (float*)ctx.devBufs[5], \
        ctx.floatArgs[0], ctx.floatArgs[1], \
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], \
        ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], \
        ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    return woqAnyNonZeroFp32(output, ac.elementCount); \
}

// 9. CudaGemmFpAInt8BFp32Kernel — GEMM_FpAInt8B<float> (2D grid, blockY present).
WOQ_FP32_SMOKE_BODY(CudaGemmFpAInt8BFp32Kernel, mnn_corpus_gemm_fpaint8b_fp32, const int8_t*, false,
    ac.args.push_back(AdaptedArg::scalarInt(blockX)); ac.args.push_back(AdaptedArg::scalarInt(blockY));)

// 10. CudaGemmFpAInt4BFp32Kernel — GEMM_FpAInt4B<float> (2D grid, blockY present).
WOQ_FP32_SMOKE_BODY(CudaGemmFpAInt4BFp32Kernel, mnn_corpus_gemm_fpaint4b_fp32, const uint8_t*, true,
    ac.args.push_back(AdaptedArg::scalarInt(blockX)); ac.args.push_back(AdaptedArg::scalarInt(blockY));)

#undef WOQ_FP32_SMOKE_BODY

// ----------------------------------------------------------------------------
// GEMV_* adapters (batch-first arg order; 1D-block V5/V9/V14 variants).
// ENTRY is the full shim symbol (e.g. mnn_corpus_gemv_fpaint8b_fp32).
// ----------------------------------------------------------------------------
#define WOQ_GEMV_FP32_SMOKE_BODY(CLASS, ENTRY, KBUF_TYPE, IS_INT4, ARG_ORDER) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = #ENTRY; \
    const int batch = spec.intParam("batch", kWoqBatch); \
    const int ic = spec.intParam("ic", kWoqIc); \
    const int ic_p = spec.intParam("ic_p", kWoqIcP); \
    const int oc = spec.intParam("oc", kWoqOc); \
    const int oc_p = spec.intParam("oc_p", kWoqOcP); \
    const int quanC = spec.intParam("quan_c", kWoqQuanC); \
    auto input = woqBuildInput(batch, ic, ic_p); \
    auto scale = woqBuildScale(oc, quanC, kWoqScale); \
    auto offset = woqBuildOffset(oc, quanC, kWoqOffset); \
    auto bias = woqBuildBias(oc, kWoqBias); \
    AdaptedBuffer iBuf; iBuf.setFp32(input); iBuf.isOutput = false; \
    AdaptedBuffer kBuf; \
    if (IS_INT4) { auto w = woqBuildInt4Weight(oc, ic_p, kWoqInt4Byte); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign(w.begin(), w.end()); } \
    else { auto w = woqBuildInt8Weight(oc, ic_p, kWoqInt8Q); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign((const uint8_t*)w.data(), (const uint8_t*)w.data() + w.size()); } \
    kBuf.isOutput = false; \
    AdaptedBuffer sBuf; sBuf.setFp32(scale); sBuf.isOutput = false; \
    AdaptedBuffer oBuf_; oBuf_.setFp32(offset); oBuf_.isOutput = false; \
    AdaptedBuffer bBuf; bBuf.setFp32(bias); bBuf.isOutput = false; \
    AdaptedBuffer outBuf; outBuf.sizeBytes = (size_t)batch * oc_p * sizeof(float); outBuf.isOutput = true; \
    ac.buffers.push_back(iBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(sBuf); \
    ac.buffers.push_back(oBuf_); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf); \
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i)); \
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqMinV)); \
    ARG_ORDER \
    ac.globalSize[0] = oc; ac.globalSize[1] = batch; ac.localSize[0] = 64; ac.dims = 2; \
    ac.elementCount = batch * oc_p; \
    return true; \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    return woqAnyNonZeroFp32(output, ac.elementCount); \
}

// 11. CudaGemvFpAInt8BFp32Kernel — GEMV_FpAInt8B (2D grid + blockY).
WOQ_GEMV_FP32_SMOKE_BODY(CudaGemvFpAInt8BFp32Kernel, mnn_corpus_gemv_fpaint8b_fp32, const int8_t*, false,
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 15) / 16)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(16)); ac.args.push_back(AdaptedArg::scalarInt(16));)
cudaError_t CudaGemvFpAInt8BFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint8b_fp32((const float*)ctx.devBufs[0], (const int8_t*)ctx.devBufs[1],
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (const float*)ctx.devBufs[4], (float*)ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}

// 12. CudaGemvFpAInt4BFp32Kernel — GEMV_FpAInt4B (2D grid + blockY, shared mem).
WOQ_GEMV_FP32_SMOKE_BODY(CudaGemvFpAInt4BFp32Kernel, mnn_corpus_gemv_fpaint4b_fp32, const uint8_t*, true,
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    const int sharedMem = ic_p * sizeof(float) + 64 * sizeof(float);
    ac.args.push_back(AdaptedArg::scalarInt(sharedMem));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 15) / 16)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(64));)
cudaError_t CudaGemvFpAInt4BFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_fp32((const float*)ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (const float*)ctx.devBufs[4], (float*)ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}

// 13. CudaGemvFpAInt4BV5Fp32Kernel — GEMV_FpAInt4B_V5 (1D block, no shared mem).
WOQ_GEMV_FP32_SMOKE_BODY(CudaGemvFpAInt4BV5Fp32Kernel, mnn_corpus_gemv_fpaint4b_v5_fp32, const uint8_t*, true,
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 15) / 16)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(64));)
cudaError_t CudaGemvFpAInt4BV5Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_v5_fp32((const float*)ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (const float*)ctx.devBufs[4], (float*)ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.stream);
    return cudaGetLastError();
}

// 14. CudaGemvFpAInt4BV9Fp32Kernel — GEMV_FpAInt4B_V9 (OC_PER_BLK=4, 1D block).
WOQ_GEMV_FP32_SMOKE_BODY(CudaGemvFpAInt4BV9Fp32Kernel, mnn_corpus_gemv_fpaint4b_v9_fp32, const uint8_t*, true,
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 3) / 4)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(64));)
cudaError_t CudaGemvFpAInt4BV9Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_v9_fp32((const float*)ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (const float*)ctx.devBufs[4], (float*)ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.stream);
    return cudaGetLastError();
}

#undef WOQ_GEMV_FP32_SMOKE_BODY

// 15. CudaGemvFpAInt4BV14Fp32Kernel — GEMV_FpAInt4B_V14 (uses float2 gemv_params,
//     no scale/offset; args: input, kernel, gemv_params, bias, output, ...).
bool CudaGemvFpAInt4BV14Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gemv_fpaint4b_v14_fp32";
    const int batch = spec.intParam("batch", kWoqBatch);
    const int ic = spec.intParam("ic", kWoqIc);
    const int ic_p = spec.intParam("ic_p", kWoqIcP);
    const int oc = spec.intParam("oc", kWoqOc);
    const int oc_p = spec.intParam("oc_p", kWoqOcP);
    const int quanC = spec.intParam("quan_c", kWoqQuanC);
    const int num_qg = quanC / oc;
    auto input = woqBuildInput(batch, ic, ic_p);
    auto kernel = woqBuildInt4Weight(oc, ic_p, kWoqInt4Byte);
    auto gemv_params = woqBuildGemvParams(oc, num_qg, kWoqScale, kWoqOffset);
    auto bias = woqBuildBias(oc, kWoqBias);
    AdaptedBuffer iBuf; iBuf.setFp32(input); iBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernel.size(); kBuf.initialData.assign(kernel.begin(), kernel.end()); kBuf.isOutput = false;
    AdaptedBuffer pBuf; pBuf.sizeBytes = gemv_params.size() * sizeof(float2); pBuf.initialData.assign((const uint8_t*)gemv_params.data(), (const uint8_t*)gemv_params.data() + pBuf.sizeBytes); pBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(bias); bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = (size_t)batch * oc_p * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(iBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(pBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);
    for (int i = 0; i < 5; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqMinV));
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(num_qg));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 3) / 4)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(64));
    ac.globalSize[0] = (oc + 3) / 4; ac.globalSize[1] = batch; ac.localSize[0] = 64; ac.dims = 2;
    ac.elementCount = batch * oc_p;
    return true;
}
cudaError_t CudaGemvFpAInt4BV14Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_v14_fp32((const float*)ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        (const float2*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (float*)ctx.devBufs[4],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.stream);
    return cudaGetLastError();
}
bool CudaGemvFpAInt4BV14Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqAnyNonZeroFp32(output, ac.elementCount);
}

// 16. CudaGemvFpAInt4BV14MbFp32Kernel — GEMV_FpAInt4B_V14_MB (1D grid, MAX_BATCH=1).
bool CudaGemvFpAInt4BV14MbFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gemv_fpaint4b_v14_mb_fp32";
    const int batch = 1; // MAX_BATCH=1
    const int ic = spec.intParam("ic", kWoqIc);
    const int ic_p = spec.intParam("ic_p", kWoqIcP);
    const int oc = spec.intParam("oc", kWoqOc);
    const int oc_p = spec.intParam("oc_p", kWoqOcP);
    const int quanC = spec.intParam("quan_c", kWoqQuanC);
    const int num_qg = quanC / oc;
    auto input = woqBuildInput(batch, ic, ic_p);
    auto kernel = woqBuildInt4Weight(oc, ic_p, kWoqInt4Byte);
    auto gemv_params = woqBuildGemvParams(oc, num_qg, kWoqScale, kWoqOffset);
    auto bias = woqBuildBias(oc, kWoqBias);
    AdaptedBuffer iBuf; iBuf.setFp32(input); iBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernel.size(); kBuf.initialData.assign(kernel.begin(), kernel.end()); kBuf.isOutput = false;
    AdaptedBuffer pBuf; pBuf.sizeBytes = gemv_params.size() * sizeof(float2); pBuf.initialData.assign((const uint8_t*)gemv_params.data(), (const uint8_t*)gemv_params.data() + pBuf.sizeBytes); pBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(bias); bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = (size_t)batch * oc_p * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(iBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(pBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);
    for (int i = 0; i < 5; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqMinV));
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(num_qg));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 3) / 4)); ac.args.push_back(AdaptedArg::scalarInt(64));
    ac.globalSize[0] = (oc + 3) / 4; ac.localSize[0] = 64; ac.dims = 1;
    ac.elementCount = batch * oc_p;
    return true;
}
cudaError_t CudaGemvFpAInt4BV14MbFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint4b_v14_mb_fp32((const float*)ctx.devBufs[0], (const uint8_t*)ctx.devBufs[1],
        (const float2*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (float*)ctx.devBufs[4],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.stream);
    return cudaGetLastError();
}
bool CudaGemvFpAInt4BV14MbFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqAnyNonZeroFp32(output, ac.elementCount);
}

// 17. CudaGemvFpAInt8BV2Fp32Kernel — GEMV_FpAInt8B_V2 (2D grid + blockY).
bool CudaGemvFpAInt8BV2Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gemv_fpaint8b_v2_fp32";
    const int batch = spec.intParam("batch", kWoqBatch);
    const int ic = spec.intParam("ic", kWoqIc);
    const int ic_p = spec.intParam("ic_p", kWoqIcP);
    const int oc = spec.intParam("oc", kWoqOc);
    const int oc_p = spec.intParam("oc_p", kWoqOcP);
    const int quanC = spec.intParam("quan_c", kWoqQuanC);
    auto input = woqBuildInput(batch, ic, ic_p);
    auto kernel = woqBuildInt8Weight(oc, ic_p, kWoqInt8Q);
    auto scale = woqBuildScale(oc, quanC, kWoqScale);
    auto offset = woqBuildOffset(oc, quanC, kWoqOffset);
    auto bias = woqBuildBias(oc, kWoqBias);
    AdaptedBuffer iBuf; iBuf.setFp32(input); iBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernel.size(); kBuf.initialData.assign((const uint8_t*)kernel.data(), (const uint8_t*)kernel.data() + kernel.size()); kBuf.isOutput = false;
    AdaptedBuffer sBuf; sBuf.setFp32(scale); sBuf.isOutput = false;
    AdaptedBuffer oBuf_; oBuf_.setFp32(offset); oBuf_.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(bias); bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = (size_t)batch * oc_p * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(iBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(sBuf); ac.buffers.push_back(oBuf_); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqMinV));
    ac.args.push_back(AdaptedArg::scalarInt(batch)); ac.args.push_back(AdaptedArg::scalarInt(ic));
    ac.args.push_back(AdaptedArg::scalarInt(ic_p)); ac.args.push_back(AdaptedArg::scalarInt(oc));
    ac.args.push_back(AdaptedArg::scalarInt(oc_p)); ac.args.push_back(AdaptedArg::scalarInt(quanC));
    ac.args.push_back(AdaptedArg::scalarInt((oc + 15) / 16)); ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(16)); ac.args.push_back(AdaptedArg::scalarInt(16));
    ac.globalSize[0] = (oc + 15) / 16; ac.globalSize[1] = batch; ac.localSize[0] = 16; ac.localSize[1] = 16; ac.dims = 2;
    ac.elementCount = batch * oc_p;
    return true;
}
cudaError_t CudaGemvFpAInt8BV2Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gemv_fpaint8b_v2_fp32((const float*)ctx.devBufs[0], (const int8_t*)ctx.devBufs[1],
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (const float*)ctx.devBufs[4], (float*)ctx.devBufs[5],
        ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5],
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.stream);
    return cudaGetLastError();
}
bool CudaGemvFpAInt8BV2Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqAnyNonZeroFp32(output, ac.elementCount);
}

// ----------------------------------------------------------------------------
// CONV_* adapters (FpAInt8B / FpAInt4B). 1D grid over total = ow*oh*c_p.
// ENTRY is the full shim symbol (e.g. mnn_corpus_conv_fpaint8b_fp32).
// ----------------------------------------------------------------------------
#define WOQ_CONV_FP32_SMOKE_BODY(CLASS, ENTRY, KBUF_TYPE, IS_INT4) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "3.6.0") return false; \
    ac.entry = #ENTRY; \
    const int ic = spec.intParam("ic", kWoqIc); \
    const int ic_p = spec.intParam("ic_p", kWoqIcP); \
    const int oc = spec.intParam("oc", kWoqOc); \
    const int oc_p = spec.intParam("oc_p", kWoqOcP); \
    const int iw = spec.intParam("iw", 4), ih = spec.intParam("ih", 4); \
    const int kw = spec.intParam("kw", 3), kh = spec.intParam("kh", 3); \
    const int sw = spec.intParam("sw", 1), sh = spec.intParam("sh", 1); \
    const int dw = spec.intParam("dw", 1), dh = spec.intParam("dh", 1); \
    const int pw = spec.intParam("pw", 1), ph = spec.intParam("ph", 1); \
    const int quanC = spec.intParam("quan_c", kWoqQuanC); \
    const int ow = (iw + 2 * pw - dw * (kw - 1) - 1) / sw + 1; \
    const int oh = (ih + 2 * ph - dh * (kh - 1) - 1) / sh + 1; \
    auto input = woqBuildInput(1, ic, ic_p); \
    auto scale = woqBuildScale(oc, quanC, kWoqScale); \
    auto offset = woqBuildOffset(oc, quanC, kWoqOffset); \
    auto bias = woqBuildBias(oc, kWoqBias); \
    AdaptedBuffer iBuf; iBuf.setFp32(input); iBuf.isOutput = false; \
    AdaptedBuffer kBuf; \
    if (IS_INT4) { auto w = woqBuildInt4Weight(oc, ic_p * kw * kh, kWoqInt4Byte); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign(w.begin(), w.end()); } \
    else { auto w = woqBuildInt8Weight(oc, ic_p * kw * kh, kWoqInt8Q); \
        kBuf.sizeBytes = w.size(); kBuf.initialData.assign((const uint8_t*)w.data(), (const uint8_t*)w.data() + w.size()); } \
    kBuf.isOutput = false; \
    AdaptedBuffer sBuf; sBuf.setFp32(scale); sBuf.isOutput = false; \
    AdaptedBuffer oBuf_; oBuf_.setFp32(offset); oBuf_.isOutput = false; \
    AdaptedBuffer bBuf; bBuf.setFp32(bias); bBuf.isOutput = false; \
    AdaptedBuffer outBuf; outBuf.sizeBytes = (size_t)ow * oh * oc_p * sizeof(float); outBuf.isOutput = true; \
    ac.buffers.push_back(iBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(sBuf); \
    ac.buffers.push_back(oBuf_); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf); \
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i)); \
    ac.args.push_back(AdaptedArg::scalarFloat(kWoqMaxV)); ac.args.push_back(AdaptedArg::scalarFloat(kWoqMinV)); \
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
    ENTRY((const float*)ctx.devBufs[0], (KBUF_TYPE)ctx.devBufs[1], \
        (const float*)ctx.devBufs[2], (const float*)ctx.devBufs[3], (const float*)ctx.devBufs[4], (float*)ctx.devBufs[5], \
        ctx.floatArgs[0], ctx.floatArgs[1], \
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], \
        ctx.intArgs[6], ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11], \
        ctx.intArgs[12], ctx.intArgs[13], ctx.intArgs[14], ctx.intArgs[15], ctx.intArgs[16], ctx.intArgs[17], \
        ctx.intArgs[18], ctx.intArgs[19], ctx.stream); \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    return woqAnyNonZeroFp32(output, ac.elementCount); \
}

// 18. CudaConvFpAInt8BFp32Kernel — CONV_FpAInt8B<float>.
WOQ_CONV_FP32_SMOKE_BODY(CudaConvFpAInt8BFp32Kernel, mnn_corpus_conv_fpaint8b_fp32, const int8_t*, false)

// 19. CudaConvFpAInt4BFp32Kernel — CONV_FpAInt4B<float>.
WOQ_CONV_FP32_SMOKE_BODY(CudaConvFpAInt4BFp32Kernel, mnn_corpus_conv_fpaint4b_fp32, const uint8_t*, true)

#undef WOQ_CONV_FP32_SMOKE_BODY

// ============================================================================
// 20. CudaGatedDeltaRulePrefillFp32Kernel — gated_delta_rule_prefill_kernel<float>.
//     L>1 register-tiled variant. Smoke-only: launch + non-zero check.
// ============================================================================
bool CudaGatedDeltaRulePrefillFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "mnn_corpus_gated_delta_rule_prefill_fp32";
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
    AdaptedBuffer cBuf; cBuf.setFp32(convOut); cBuf.isOutput = false;
    AdaptedBuffer gBuf; gBuf.setFp32(gateInput); gBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(betaInput); bBuf.isOutput = false;
    AdaptedBuffer rBuf; rBuf.setFp32(recurrentState); rBuf.isOutput = false;
    AdaptedBuffer oBuf; oBuf.sizeBytes = (size_t)B * L * H_v * d_v * sizeof(float); oBuf.isOutput = true;
    ac.buffers.push_back(cBuf); ac.buffers.push_back(gBuf); ac.buffers.push_back(bBuf);
    ac.buffers.push_back(rBuf); ac.buffers.push_back(oBuf);
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
cudaError_t CudaGatedDeltaRulePrefillFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gated_delta_rule_prefill_fp32(
        (const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (const float*)ctx.devBufs[2],
        (float*)ctx.devBufs[3], (float*)ctx.devBufs[4],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
        ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
        ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10] != 0, ctx.floatArgs[0],
        ctx.intArgs[11] != 0, ctx.intArgs[12] != 0, ctx.intArgs[13] != 0,
        ctx.grid, ctx.block, (size_t)ctx.intArgs[16], ctx.stream);
    return cudaGetLastError();
}
bool CudaGatedDeltaRulePrefillFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return woqAnyNonZeroFp32(output, ac.elementCount);
}

// ============================================================================
// Registration. CudaGemmInt8Fp32Kernel is implemented in CudaOpsMisc.cpp; the
// other 20 fp32 adapters are implemented above.
// ============================================================================
void registerWoqAdapters(OpAdapterRegistry& r) {
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemmInt8Fp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaPrecomputeSumbqFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaRearrangePackedWeightInt4Fp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaRearrangeWeightInt4Fp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaRearrangeWeightInt8Fp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaPrecomputeGemvParamsFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaQuantAFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaDequantAndAccFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBiasAndActivationFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemmFpAInt8BFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt8BFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemmFpAInt4BFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BV5Fp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BV9Fp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BV14Fp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt4BV14MbFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGemvFpAInt8BV2Fp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvFpAInt8BFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvFpAInt4BFp32Kernel()));
    r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGatedDeltaRulePrefillFp32Kernel()));
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
