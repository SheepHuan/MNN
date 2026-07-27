// CudaOpsFp16.cu - fp16/int8/bf16 variants of CUDA corpus adapters.
// Compiled by nvcc (via replay_cuda_corpus) because __half/__nv_bfloat16 are
// not available in plain g++ compilation. The fp32 variants live in CudaOps.cpp.
#include "CudaOps.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>

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
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
