#include "CudaOps.hpp"
#include <cuda_fp16.h>
#include <cmath>
#include <cstring>

// extern "C" launch shims (defined in kernel_corpus_bridge/cuda/CorpusKernels.cu)
extern "C" {
void mnn_corpus_relu_fp32(const float*, float*, size_t, float, int, int, cudaStream_t);
void mnn_corpus_relu_fp16(const void*, void*, size_t, float, int, int, cudaStream_t);
void mnn_corpus_relu_int8(const int8_t*, int8_t*, size_t, int8_t, int, int, cudaStream_t);
void mnn_corpus_clamp_fp32(const float*, float*, size_t, float, float, int, int, cudaStream_t);
void mnn_corpus_clamp_fp16(const void*, void*, size_t, float, float, int, int, cudaStream_t);
void mnn_corpus_castbool_i32(const int32_t*, int32_t*, size_t, int, int, cudaStream_t);
void mnn_corpus_cast_f32_i32(const float*, int32_t*, size_t, int, int, cudaStream_t);
void mnn_corpus_cast_i32_f32(const int32_t*, float*, size_t, int, int, cudaStream_t);
void mnn_corpus_atan2_fp32(const float*, const float*, float*, size_t, size_t, size_t, int, int, cudaStream_t);
void mnn_corpus_mod_fp32(const float*, const float*, float*, size_t, size_t, size_t, int, int, cudaStream_t);
void mnn_corpus_logicalor_fp32(const float*, const float*, float*, size_t, size_t, size_t, int, int, cudaStream_t);
void mnn_corpus_range_fp32(const int, const float*, const float*, float*, int, int, cudaStream_t);
void mnn_corpus_select_fp32(const int, const int*, const float*, const float*, int, int, float*, int, int, cudaStream_t);
void mnn_corpus_softmax_fp32(const float*, float*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_layernorm_fp32(const int, const int, const int, const float, const float*, float*, const float*, const float*, bool, int, int, cudaStream_t);
void mnn_corpus_prelu_fp32(const int, const int, const int, const float*, float*, const float*, int, int, int, cudaStream_t);
void mnn_corpus_scale_fp32(const int, const int, const int, const float*, float*, const float*, const float*, int, int, cudaStream_t);
void mnn_corpus_maxpool_fp32(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_avgpool_fp32(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_global_avgpool_fp32(const float*, float*, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_global_maxpool_fp32(const float*, float*, int, int, int, int, int, int, int, cudaStream_t);
// 1.2.0 tag shims (different kernel implementations)
void mnn_corpus_maxpool_120_fp32(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_avgpool_120_fp32(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_scale_120_fp32(const int, const int, const int, const float*, float*, const float*, const float*, int, int, cudaStream_t);
void mnn_corpus_layernorm_120_fp32(const int, const int, const int, const float, const float*, float*, const float*, const float*, int, int, cudaStream_t);
void mnn_corpus_layernorm_127_fp32(const int, const int, const int, const float, const float*, float*, const float*, const float*, int, int, cudaStream_t);
void mnn_corpus_layernorm_222_fp32(const int, const int, const int, const float, const float*, float*, const float*, const float*, int, int, cudaStream_t);
void mnn_corpus_prelu_120_fp32(const int, const int, const int, const float*, float*, const float*, int, int, int, cudaStream_t);
void mnn_corpus_prelu_127_fp32(const int, const int, const int, const float*, float*, const float*, int, int, int, cudaStream_t);
void mnn_corpus_prelu_128_fp32(const int, const int, const int, const float*, float*, const float*, int, int, int, cudaStream_t);
}

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// ============================================================================
// Unary: ReLU fp32
// ============================================================================
bool CudaReluFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_relu_fp32";
    const int count = spec.intParam("size", 1024);
    const float slope = spec.floatParam("slope", 0.0f);
    std::vector<float> input; fillInputAlternating(input, count);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarFloat(slope));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    return true;
}
cudaError_t CudaReluFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_relu_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                          (size_t)ctx.intArgs[0], ctx.floatArgs[0], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaReluFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.validatorInputA.size() != output.size()) return false;
    const float slope = ac.args.size() >= 4 ? ac.args[3].floatVal : 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        const float x = ac.validatorInputA[i];
        const float expected = x > 0.0f ? x : x * slope;
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Unary: ReLU int8
// ============================================================================
bool CudaReluInt8Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_relu_int8";
    const int count = spec.intParam("size", 1024);
    const int8_t zeroPoint = (int8_t)spec.intParam("zero_point", 0);
    std::vector<int8_t> input(count);
    for (int i = 0; i < count; ++i) input[i] = (int8_t)(i % 256 - 128);
    AdaptedBuffer inBuf; inBuf.sizeBytes = count; inBuf.initialData.assign((const uint8_t*)input.data(), (const uint8_t*)input.data() + count);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count; outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(zeroPoint));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    // Store int8 input bytes in validatorInputA (reinterpret as float buffer).
    // Round up to float boundary to avoid OOB read in validate().
    const int floatSlots = (count + (int)sizeof(float) - 1) / (int)sizeof(float);
    ac.validatorInputA.resize(floatSlots, 0.0f);
    std::memcpy(ac.validatorInputA.data(), input.data(), count);
    ac.elementCount = count;
    return true;
}
cudaError_t CudaReluInt8Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_relu_int8((const int8_t*)ctx.devBufs[0], (int8_t*)ctx.devBufs[1], (size_t)ctx.intArgs[0],
                          (int8_t)ctx.intArgs[1], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaReluInt8Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size() * sizeof(float)) < count) return false;
    const int8_t zeroPoint = ac.args.size() >= 4 ? (int8_t)ac.args[3].intVal : 0;
    const int8_t* in = reinterpret_cast<const int8_t*>(ac.validatorInputA.data());
    const int8_t* out = reinterpret_cast<const int8_t*>(output.data());
    for (int i = 0; i < count; ++i) {
        const int8_t expected = in[i] > zeroPoint ? in[i] : zeroPoint;
        if (out[i] != expected) return false;
    }
    return true;
}

// ============================================================================
// Unary: CLAMP fp32 (relu6: min=0, max=6)
// ============================================================================
bool CudaClampFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_clamp_fp32";
    const int count = spec.intParam("size", 1024);
    const float minV = spec.floatParam("min_v", 0.0f);
    const float maxV = spec.floatParam("max_v", 6.0f);
    std::vector<float> input; fillInputRamp(input, count);
    for (int i = 0; i < count; ++i) input[i] = input[i] * 10.0f - 5.0f; // spread across clamp range
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
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
cudaError_t CudaClampFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_clamp_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1], (size_t)ctx.intArgs[0],
                           ctx.floatArgs[0], ctx.floatArgs[1], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaClampFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.validatorInputA.size() != output.size()) return false;
    const float minV = ac.args[3].floatVal, maxV = ac.args[4].floatVal;
    for (size_t i = 0; i < output.size(); ++i) {
        const float x = ac.validatorInputA[i];
        const float expected = std::min(std::max(x, minV), maxV);
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Cast: CASTBOOL
// ============================================================================
bool CudaCastBoolKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_castbool_i32";
    const int count = spec.intParam("size", 1024);
    std::vector<int32_t> input(count);
    for (int i = 0; i < count; ++i) input[i] = (i % 3 == 0) ? 0 : ((i % 2 == 0) ? i : -i);
    AdaptedBuffer inBuf; inBuf.sizeBytes = count * sizeof(int32_t);
    inBuf.initialData.assign((const uint8_t*)input.data(), (const uint8_t*)input.data() + inBuf.sizeBytes); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.elementCount = count;
    return true;
}
cudaError_t CudaCastBoolKernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_castbool_i32((const int32_t*)ctx.devBufs[0], (int32_t*)ctx.devBufs[1], (size_t)ctx.intArgs[0],
                             ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastBoolKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size()) < count) return false;
    const uint32_t* bits = reinterpret_cast<const uint32_t*>(output.data());
    for (int i = 0; i < count; ++i) {
        const int32_t in = (i % 3 == 0) ? 0 : ((i % 2 == 0) ? i : -i);
        const uint32_t expected = in > 0 ? 1u : 0u;
        if (bits[i] != expected) return false;
    }
    return true;
}

// ============================================================================
// Cast: f32 -> i32
// ============================================================================
bool CudaCastF322I32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_cast_f32_i32";
    const int count = spec.intParam("size", 1024);
    std::vector<float> input; fillInputRamp(input, count);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    return true;
}
cudaError_t CudaCastF322I32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_cast_f32_i32((const float*)ctx.devBufs[0], (int32_t*)ctx.devBufs[1], (size_t)ctx.intArgs[0],
                             ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastF322I32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size()) < count) return false;
    const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
    for (int i = 0; i < count; ++i) {
        const int32_t expected = (int32_t)ac.validatorInputA[i];
        if (out[i] != expected) return false;
    }
    return true;
}

// ============================================================================
// Cast: i32 -> f32
// ============================================================================
bool CudaCastI322F32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_cast_i32_f32";
    const int count = spec.intParam("size", 1024);
    std::vector<int32_t> input(count);
    for (int i = 0; i < count; ++i) input[i] = i - count / 2;
    AdaptedBuffer inBuf; inBuf.sizeBytes = count * sizeof(int32_t);
    inBuf.initialData.assign((const uint8_t*)input.data(), (const uint8_t*)input.data() + inBuf.sizeBytes); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA.assign(input.begin(), input.end());
    ac.elementCount = count;
    return true;
}
cudaError_t CudaCastI322F32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_cast_i32_f32((const int32_t*)ctx.devBufs[0], (float*)ctx.devBufs[1], (size_t)ctx.intArgs[0],
                             ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaCastI322F32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size()) < count) return false;
    for (int i = 0; i < count; ++i) {
        const float expected = (float)((int32_t)ac.validatorInputA[i]);
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Binary: ATAN2
// ============================================================================
bool CudaBinaryAtan2Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_atan2_fp32";
    const int count = spec.intParam("size", 1024);
    std::vector<float> in0, in1; fillInputRamp(in0, count); fillInputRamp(in1, count);
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(1));  // s0
    ac.args.push_back(AdaptedArg::scalarInt(1));  // s1
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = in0; ac.validatorInputB = in1; ac.elementCount = count;
    return true;
}
cudaError_t CudaBinaryAtan2Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_atan2_fp32((const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (float*)ctx.devBufs[2],
                          (size_t)ctx.intArgs[0], (size_t)ctx.intArgs[1], (size_t)ctx.intArgs[2],
                          ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaBinaryAtan2Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float expected = atan2f(ac.validatorInputA[i], ac.validatorInputB[i]);
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Binary: MOD
// ============================================================================
bool CudaBinaryModKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_mod_fp32";
    const int count = spec.intParam("size", 1024);
    std::vector<float> in0, in1; fillInputRamp(in0, count); fillInputRamp(in1, count);
    for (int i = 0; i < count; ++i) in1[i] = in1[i] + 1.0f; // avoid div by 0
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(1));
    ac.args.push_back(AdaptedArg::scalarInt(1));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = in0; ac.validatorInputB = in1; ac.elementCount = count;
    return true;
}
cudaError_t CudaBinaryModKernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_mod_fp32((const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (float*)ctx.devBufs[2],
                        (size_t)ctx.intArgs[0], (size_t)ctx.intArgs[1], (size_t)ctx.intArgs[2],
                        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaBinaryModKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float x = ac.validatorInputA[i], y = ac.validatorInputB[i];
        const float expected = x - x / y;
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Binary: LOGICALOR
// ============================================================================
bool CudaBinaryLogicalOrKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_logicalor_fp32";
    const int count = spec.intParam("size", 1024);
    std::vector<float> in0, in1; fillInputAlternating(in0, count); fillInputAlternating(in1, count);
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(1));
    ac.args.push_back(AdaptedArg::scalarInt(1));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = in0; ac.validatorInputB = in1; ac.elementCount = count;
    return true;
}
cudaError_t CudaBinaryLogicalOrKernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_logicalor_fp32((const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1], (float*)ctx.devBufs[2],
                              (size_t)ctx.intArgs[0], (size_t)ctx.intArgs[1], (size_t)ctx.intArgs[2],
                              ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaBinaryLogicalOrKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float expected = (ac.validatorInputA[i] || ac.validatorInputB[i]) ? 1.0f : 0.0f;
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Range
// ============================================================================
bool CudaRangeFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_range_fp32";
    const int count = spec.intParam("size", 1024);
    std::vector<float> start(1, spec.floatParam("start", 0.0f));
    std::vector<float> step(1, spec.floatParam("step", 1.0f));
    AdaptedBuffer startBuf; startBuf.setFp32(start); startBuf.isOutput = false;
    AdaptedBuffer stepBuf; stepBuf.setFp32(step); stepBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(startBuf); ac.buffers.push_back(stepBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = start; ac.validatorInputB = step; ac.elementCount = count;
    return true;
}
cudaError_t CudaRangeFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_range_fp32(ctx.intArgs[0], (const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1],
                          (float*)ctx.devBufs[2], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaRangeFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size()) < count) return false;
    const float start = ac.validatorInputA[0], step = ac.validatorInputB[0];
    for (int i = 0; i < count; ++i) {
        if (std::fabs(output[i] - (start + i * step)) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Select
// ============================================================================
bool CudaSelectFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_select_fp32";
    const int count = spec.intParam("size", 1024);
    std::vector<int32_t> sel(count);
    for (int i = 0; i < count; ++i) sel[i] = i % 2;
    std::vector<float> in1, in2; fillInputRamp(in1, count); fillInputRamp(in2, count);
    for (int i = 0; i < count; ++i) in2[i] += 100.0f;
    AdaptedBuffer selBuf; selBuf.sizeBytes = count * sizeof(int32_t);
    selBuf.initialData.assign((const uint8_t*)sel.data(), (const uint8_t*)sel.data() + selBuf.sizeBytes); selBuf.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer b2; b2.setFp32(in2); b2.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(selBuf); ac.buffers.push_back(b1); ac.buffers.push_back(b2); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(1));  // s1
    ac.args.push_back(AdaptedArg::scalarInt(1));  // s2
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = in1; ac.validatorInputB = in2; ac.elementCount = count;
    return true;
}
cudaError_t CudaSelectFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_select_fp32(ctx.intArgs[0], (const int*)ctx.devBufs[0], (const float*)ctx.devBufs[1],
                            (const float*)ctx.devBufs[2], ctx.intArgs[1], ctx.intArgs[2],
                            (float*)ctx.devBufs[3], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaSelectFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int count = ac.elementCount;
    if (static_cast<int>(output.size()) < count) return false;
    for (int i = 0; i < count; ++i) {
        const float expected = (i % 2 > 0) ? ac.validatorInputA[i] : ac.validatorInputB[i];
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Softmax
// ============================================================================
bool CudaSoftmaxFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_softmax_fp32";
    const int outside = spec.intParam("outside", 4);
    const int axis = spec.intParam("axis", 16);
    const int inside = spec.intParam("inside", 1);
    const int count = outside * inside;
    std::vector<float> input(outside * axis * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 7);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = input.size() * sizeof(float); outBuf.isOutput = true;
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
cudaError_t CudaSoftmaxFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_softmax_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                              ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                              ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaSoftmaxFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    if ((int)output.size() < outside * axis * inside) return false;
    for (int o = 0; o < outside; ++o) {
        for (int x = 0; x < inside; ++x) {
            const float* src = ac.validatorInputA.data() + o * axis * inside + x;
            float maxV = src[0];
            for (int z = 1; z < axis; ++z) maxV = std::max(maxV, src[z * inside]);
            float sum = 0.0f;
            for (int z = 0; z < axis; ++z) {
                float t = src[z * inside] - maxV;
                if (t < -87.0f) t = -87.0f;
                sum += expf(t);
            }
            sum = 1.0f / sum;
            for (int z = 0; z < axis; ++z) {
                float t = src[z * inside] - maxV;
                if (t < -87.0f) t = -87.0f;
                const float expected = expf(t) * sum;
                const float got = output[(o * axis + z) * inside + x];
                if (std::fabs(got - expected) > 1e-3f) return false;
            }
        }
    }
    return true;
}

// ============================================================================
// LayerNorm
// ============================================================================
bool CudaLayerNormFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "1.2.0") {
        ac.entry = "mnn_corpus_layernorm_120_fp32";
    } else if (spec.tag == "1.2.7") {
        ac.entry = "mnn_corpus_layernorm_127_fp32";
    } else if (spec.tag == "2.2.2") {
        ac.entry = "mnn_corpus_layernorm_222_fp32";
    } else {
        // 2.8.4 and 3.6.0: RMSNorm param (bodies identical, shared shim)
        ac.entry = "mnn_corpus_layernorm_fp32";
    }
    const int outside = spec.intParam("outside", 4);
    const int inside = spec.intParam("inside", 32);
    const float eps = spec.floatParam("epsilon", 1e-5f);
    const int count = outside * inside;
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = 0.1f * (i % 13) - 0.5f;
    std::vector<float> gamma(inside, 1.0f), beta(inside, 0.0f);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer gammaBuf; gammaBuf.setFp32(gamma); gammaBuf.isOutput = false;
    AdaptedBuffer betaBuf; betaBuf.setFp32(beta); betaBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
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
    ac.args.push_back(AdaptedArg::scalarInt(0));  // RMSNorm = false
    if (spec.tag == "1.2.0") {
        const int blk = mnnBlock120(count);
        ac.globalSize[0] = mnnGridFor(count, blk); ac.localSize[0] = blk;
    } else {
        ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock;
    }
    ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = outside; ac.k = inside;
    return true;
}
cudaError_t CudaLayerNormFp32Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    if (ac.tag == "1.2.0" || ac.tag == "1.2.7" || ac.tag == "2.2.2") {
        // 1.2.0/1.2.7/2.2.2: no RMSNorm flag in signature
        const float* gin = (const float*)ctx.devBufs[0];
        float* gout = (float*)ctx.devBufs[3];
        const float* gg = (const float*)ctx.devBufs[1];
        const float* gb = (const float*)ctx.devBufs[2];
        if (ac.tag == "1.2.0")
            mnn_corpus_layernorm_120_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.floatArgs[0], gin, gout, gg, gb, ctx.grid, ctx.block, ctx.stream);
        else if (ac.tag == "1.2.7")
            mnn_corpus_layernorm_127_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.floatArgs[0], gin, gout, gg, gb, ctx.grid, ctx.block, ctx.stream);
        else
            mnn_corpus_layernorm_222_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.floatArgs[0], gin, gout, gg, gb, ctx.grid, ctx.block, ctx.stream);
    } else {
        mnn_corpus_layernorm_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.floatArgs[0],
                                   (const float*)ctx.devBufs[0], (float*)ctx.devBufs[3],
                                   (const float*)ctx.devBufs[1], (const float*)ctx.devBufs[2],
                                   /*RMSNorm=*/false, ctx.grid, ctx.block, ctx.stream);
    }
    return cudaGetLastError();
}
bool CudaLayerNormFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, inside = ac.k;
    const float eps = ac.args[3].floatVal;  // epsilon stored in args[3]
    if ((int)output.size() < outside * inside) return false;
    for (int o = 0; o < outside; ++o) {
        const float* in = ac.validatorInputA.data() + o * inside;
        float mean = 0.0f;
        for (int j = 0; j < inside; ++j) mean += in[j];
        mean /= inside;
        float sqsum = 0.0f;
        for (int j = 0; j < inside; ++j) sqsum += (in[j] - mean) * (in[j] - mean);
        float invStd = 1.0f / sqrtf(sqsum / inside + eps);
        for (int j = 0; j < inside; ++j) {
            const float expected = (in[j] - mean) * invStd; // gamma=1, beta=0
            if (std::fabs(output[o * inside + j] - expected) > 1e-2f) return false;
        }
    }
    return true;
}

// ============================================================================
// PReLU
// ============================================================================
bool CudaPreluFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int total = spec.intParam("size", 1024);
    int channelsPack = spec.intParam("channels", 8);
    const int dim = total / channelsPack;
    std::vector<float> input(total); fillInputAlternating(input, total);

    if (spec.tag == "1.2.0") {
        ac.entry = "mnn_corpus_prelu_120_fp32";
        std::vector<float> slope(channelsPack, 0.1f);
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
        AdaptedBuffer slopeBuf; slopeBuf.setFp32(slope); slopeBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(slopeBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(total));
        ac.args.push_back(AdaptedArg::scalarInt(channelsPack));
        ac.args.push_back(AdaptedArg::scalarInt(dim));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(1));  // div_factor=1
        const int blk = mnnBlock120(total);
        ac.globalSize[0] = mnnGridFor(total, blk); ac.localSize[0] = blk;
    } else if (spec.tag == "1.2.7" || spec.tag == "1.2.8") {
        // 1.2.7/1.2.8: PACK_NUMBER=4 channel packing.
        // mChannel = UP_DIV(channels, PACK), mArea = total / (mChannel * PACK)
        // mCount = mChannel * mArea * PACK = total
        const int PACK = 4;
        const int mChannel = (channelsPack + PACK - 1) / PACK;  // packed channel groups
        const int mArea = total / (mChannel * PACK);            // spatial elements
        const int slopeSize = mChannel * PACK;                  // slope packed
        std::vector<float> slope(slopeSize, 0.1f);
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
        AdaptedBuffer slopeBuf; slopeBuf.setFp32(slope); slopeBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(slopeBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(total));       // n = mCount
        ac.args.push_back(AdaptedArg::scalarInt(mChannel));     // channels (packed groups)
        ac.args.push_back(AdaptedArg::scalarInt(mArea));        // dim = mArea
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::buffer(1));
        if (spec.tag == "1.2.7") {
            ac.entry = "mnn_corpus_prelu_127_fp32";
            ac.args.push_back(AdaptedArg::scalarInt(1));  // div_factor=1
        } else {
            ac.entry = "mnn_corpus_prelu_128_fp32";
            ac.args.push_back(AdaptedArg::scalarInt(0));  // share_factor=0
        }
        ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock;
    } else {
        // 2.0.4 / 3.6.0
        ac.entry = "mnn_corpus_prelu_fp32";
        std::vector<float> slope(channelsPack, 0.1f);
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
        AdaptedBuffer slopeBuf; slopeBuf.setFp32(slope); slopeBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(slopeBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(total));
        ac.args.push_back(AdaptedArg::scalarInt(channelsPack));
        ac.args.push_back(AdaptedArg::scalarInt(dim));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(0));  // share_factor=0
        ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock;
    }
    ac.dims = 1;
    ac.validatorInputA = input;
    ac.elementCount = total;
    // Set ac.n and validatorInputB per-tag for validate()
    if (spec.tag == "1.2.7" || spec.tag == "1.2.8") {
        const int PACK = 4;
        const int mChannel = (channelsPack + PACK - 1) / PACK;
        ac.n = mChannel;
        ac.validatorInputB = std::vector<float>(mChannel * PACK, 0.1f);
    } else {
        ac.n = channelsPack;
        ac.validatorInputB = std::vector<float>(channelsPack, 0.1f);
    }
    return true;
}
cudaError_t CudaPreluFp32Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    if (ac.tag == "1.2.0") {
        mnn_corpus_prelu_120_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                                   (const float*)ctx.devBufs[0], (float*)ctx.devBufs[2],
                                   (const float*)ctx.devBufs[1], ctx.intArgs[3],
                                   ctx.grid, ctx.block, ctx.stream);
    } else if (ac.tag == "1.2.7") {
        mnn_corpus_prelu_127_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                                   (const float*)ctx.devBufs[0], (float*)ctx.devBufs[2],
                                   (const float*)ctx.devBufs[1], ctx.intArgs[3],
                                   ctx.grid, ctx.block, ctx.stream);
    } else if (ac.tag == "1.2.8") {
        mnn_corpus_prelu_128_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                                   (const float*)ctx.devBufs[0], (float*)ctx.devBufs[2],
                                   (const float*)ctx.devBufs[1], ctx.intArgs[3],
                                   ctx.grid, ctx.block, ctx.stream);
    } else {
        mnn_corpus_prelu_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                               (const float*)ctx.devBufs[0], (float*)ctx.devBufs[2],
                               (const float*)ctx.devBufs[1], ctx.intArgs[3],
                               ctx.grid, ctx.block, ctx.stream);
    }
    return cudaGetLastError();
}
bool CudaPreluFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount, channelsPack = ac.n;
    if ((int)output.size() < total) return false;
    const int PACK = 4;
    if (ac.tag == "1.2.0") {
        const int dim = ac.args[2].intVal;
        const int div_factor = ac.args[6].intVal;
        for (int i = 0; i < total; ++i) {
            const int c_idx = (i / dim) % channelsPack / div_factor;
            const float x = ac.validatorInputA[i];
            const float expected = x > 0.0f ? x : x * ac.validatorInputB[c_idx];
            if (std::fabs(output[i] - expected) > 1e-3f) return false;
        }
    } else if (ac.tag == "1.2.7") {
        // 1.2.7: mChannel = UP_DIV(channels,PACK), c = (index/mArea) % mChannel / div_factor
        const int PACK = 4;
        const int mChannel = ac.n;
        const int mArea = ac.args[2].intVal;
        const int div_factor = ac.args[6].intVal;
        for (int t = 0; t < total; ++t) {
            int index = t / PACK;
            int r = t % PACK;
            int c = (index / mArea) % mChannel / div_factor;
            int c_idx = c * PACK + r;
            const float x = ac.validatorInputA[t];
            const float expected = x > 0.0f ? x : x * ac.validatorInputB[c_idx];
            if (std::fabs(output[t] - expected) > 1e-3f) return false;
        }
    } else if (ac.tag == "1.2.8") {
        // 1.2.8: mChannel = UP_DIV(channels,PACK), c = (index/mArea) % mChannel
        const int PACK = 4;
        const int mChannel = ac.n;
        const int mArea = ac.args[2].intVal;
        const int share_factor = ac.args[6].intVal;
        for (int t = 0; t < total; ++t) {
            int index = t / PACK;
            int r = t % PACK;
            int c = (index / mArea) % mChannel;
            int c_idx = share_factor ? 0 : (c * PACK + r);
            const float x = ac.validatorInputA[t];
            const float expected = x > 0.0f ? x : x * ac.validatorInputB[c_idx];
            if (std::fabs(output[t] - expected) > 1e-3f) return false;
        }
    } else {
        // 2.0.4 / 3.6.0: c_idx = index % channelsPack
        const int share_factor = ac.args[6].intVal;
        for (int i = 0; i < total; ++i) {
            int c_idx = i % channelsPack;
            c_idx = share_factor ? 0 : c_idx;
            const float x = ac.validatorInputA[i];
            const float expected = x > 0.0f ? x : x * ac.validatorInputB[c_idx];
            if (std::fabs(output[i] - expected) > 1e-3f) return false;
        }
    }
    return true;
}

// ============================================================================
// Scale
// ============================================================================
bool CudaScaleFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "1.2.0") {
        ac.entry = "mnn_corpus_scale_120_fp32";
    } else {
        ac.entry = "mnn_corpus_scale_fp32";
    }
    const int total = spec.intParam("size", 1024);
    const int channelsPack = spec.intParam("channels", 8);
    std::vector<float> input(total); fillInputRamp(input, total);
    std::vector<float> scale(channelsPack, 2.0f), bias(channelsPack, 1.0f);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer scaleBuf; scaleBuf.setFp32(scale); scaleBuf.isOutput = false;
    AdaptedBuffer biasBuf; biasBuf.setFp32(bias); biasBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(scaleBuf); ac.buffers.push_back(biasBuf);
    ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(channelsPack));
    ac.args.push_back(AdaptedArg::scalarInt(total / channelsPack));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    if (spec.tag == "1.2.0") {
        const int blk = mnnBlock120(total);
        ac.globalSize[0] = mnnGridFor(total, blk); ac.localSize[0] = blk;
    } else {
        ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock;
    }
    ac.dims = 1;
    ac.validatorInputA = input; ac.validatorInputB = scale; ac.elementCount = total;
    ac.n = channelsPack; ac.validatorInputA.resize(total); // will use validatorInputB for bias
    return true;
}
cudaError_t CudaScaleFp32Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    if (ac.tag == "1.2.0") {
        mnn_corpus_scale_120_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                                   (const float*)ctx.devBufs[0], (float*)ctx.devBufs[3],
                                   (const float*)ctx.devBufs[1], (const float*)ctx.devBufs[2],
                                   ctx.grid, ctx.block, ctx.stream);
    } else {
        mnn_corpus_scale_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                               (const float*)ctx.devBufs[0], (float*)ctx.devBufs[3],
                               (const float*)ctx.devBufs[1], (const float*)ctx.devBufs[2],
                               ctx.grid, ctx.block, ctx.stream);
    }
    return cudaGetLastError();
}
bool CudaScaleFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount, channelsPack = ac.n;
    if ((int)output.size() < total) return false;
    // validatorInputB holds scale; reconstruct bias as 1.0 (we set it so)
    for (int i = 0; i < total; ++i) {
        const int c = i % channelsPack;
        const float expected = ac.validatorInputA[i] * ac.validatorInputB[c] + 1.0f;
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// Pool: maxpool
// ============================================================================
bool CudaMaxPoolFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_maxpool_fp32";
    const int ib = spec.intParam("batch", 1);
    const int ic_p = spec.intParam("channels", 8);
    const int ih = spec.intParam("h", 8);
    const int iw = spec.intParam("w", 8);
    const int kx = spec.intParam("kernel_size", 3);
    const int ky = kx;
    const int sx = spec.intParam("stride", 2);
    const int sy = sx;
    const int padX = spec.intParam("pad", 1);
    const int padY = padX;
    const int oh = (ih + 2 * padY - ky) / sy + 1;
    const int ow = (iw + 2 * padX - kx) / sx + 1;
    const int inSize = ib * ic_p * ih * iw;
    std::vector<float> input(inSize);
    for (int i = 0; i < inSize; ++i) input[i] = 0.1f * (i % 11);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = ib * ic_p * oh * ow * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    if (spec.tag == "1.2.0") {
        // 1.2.0: NCHW layout, bc = ib*ic_p
        int bc = ib * ic_p;
        for (int v : {bc, ih, iw, oh, ow, padX, padY, kx, ky, sx, sy})
            ac.args.push_back(AdaptedArg::scalarInt(v));
        ac.entry = "mnn_corpus_maxpool_120_fp32";
    } else {
        for (int v : {ib, ic_p, ih, iw, oh, ow, padX, padY, kx, ky, sx, sy})
            ac.args.push_back(AdaptedArg::scalarInt(v));
        ac.entry = "mnn_corpus_maxpool_fp32";
    }
    const int total = ib * oh * ow * ic_p;
    if (spec.tag == "1.2.0") {
        const int blk = mnnBlock120(total);
        ac.globalSize[0] = mnnGridFor(total, blk); ac.localSize[0] = blk;
    } else {
        ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock;
    }
    ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.m = ib; ac.n = ic_p; ac.h = ih; ac.w = iw; ac.k = kx; ac.stride = sx; ac.orderType = padX;
    return true;
}
cudaError_t CudaMaxPoolFp32Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    if (ac.tag == "1.2.0") {

        mnn_corpus_maxpool_120_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                    ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                    ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                    ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10],
                                    ctx.grid, ctx.block, ctx.stream);
    } else {
        mnn_corpus_maxpool_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                 ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                 ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                 ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                                 ctx.grid, ctx.block, ctx.stream);
    }
    return cudaGetLastError();
}
bool CudaMaxPoolFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ib = ac.m, ic_p = ac.n, ih = ac.h, iw = ac.w;
    const int kx = ac.k, ky = kx, sx = ac.stride, padX = ac.orderType, padY = padX;
    const int oh = (ih + 2 * padY - ky) / sx + 1;
    const int ow = (iw + 2 * padX - kx) / sx + 1;
    const int bc = ib * ic_p;
    for (int b = 0; b < ib; ++b)
      for (int c = 0; c < ic_p; ++c)
        for (int oy = 0; oy < oh; ++oy)
          for (int ox = 0; ox < ow; ++ox) {
            float maxV = -65504.0f;
            for (int fy = 0; fy < ky; ++fy)
              for (int fx = 0; fx < kx; ++fx) {
                int iy = oy * sx - padY + fy, ix = ox * sx - padX + fx;
                if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                int off;
                if (ac.tag == "1.2.0") {
                    // NCHW: [bc, ih, iw], bc = b*ic_p+c
                    int z = b * ic_p + c;
                    off = z * ih * iw + iy * iw + ix;
                } else {
                    // NC4HW4: [b, ih, iw, ic_p]
                    off = ((b * ih + iy) * iw + ix) * ic_p + c;
                }
                maxV = std::max(maxV, ac.validatorInputA[off]);
              }
            int outOff;
            if (ac.tag == "1.2.0") {
                int z = b * ic_p + c;
                outOff = z * oh * ow + oy * ow + ox;
            } else {
                outOff = ((b * oh + oy) * ow + ox) * ic_p + c;
            }
            if (std::fabs(output[outOff] - maxV) > 1e-3f) return false;
          }
    return true;
}

// ============================================================================
// Pool: avgpool
// ============================================================================
bool CudaAvgPoolFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_avgpool_fp32";
    const int ib = spec.intParam("batch", 1);
    const int ic_p = spec.intParam("channels", 8);
    const int ih = spec.intParam("h", 8);
    const int iw = spec.intParam("w", 8);
    const int kx = spec.intParam("kernel_size", 3);
    const int sx = spec.intParam("stride", 2);
    const int padX = spec.intParam("pad", 1);
    const int oh = (ih + 2 * padX - kx) / sx + 1;
    const int ow = oh;
    const int inSize = ib * ic_p * ih * iw;
    std::vector<float> input(inSize);
    for (int i = 0; i < inSize; ++i) input[i] = 0.1f * (i % 11);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = ib * ic_p * oh * ow * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    if (spec.tag == "1.2.0") {
        int bc = ib * ic_p;
        for (int v : {bc, ih, iw, oh, ow, padX, padX, kx, kx, sx, sx})
            ac.args.push_back(AdaptedArg::scalarInt(v));
        ac.entry = "mnn_corpus_avgpool_120_fp32";
    } else {
        for (int v : {ib, ic_p, ih, iw, oh, ow, padX, padX, kx, kx, sx, sx})
            ac.args.push_back(AdaptedArg::scalarInt(v));
        ac.entry = "mnn_corpus_avgpool_fp32";
    }
    const int total = ib * oh * ow * ic_p;
    if (spec.tag == "1.2.0") {
        const int blk = mnnBlock120(total);
        ac.globalSize[0] = mnnGridFor(total, blk); ac.localSize[0] = blk;
    } else {
        ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock;
    }
    ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.m = ib; ac.n = ic_p; ac.h = ih; ac.w = iw; ac.k = kx; ac.stride = sx; ac.orderType = padX;
    return true;
}
cudaError_t CudaAvgPoolFp32Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    if (ac.tag == "1.2.0") {

        mnn_corpus_avgpool_120_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                    ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                    ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                    ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10],
                                    ctx.grid, ctx.block, ctx.stream);
    } else {
        mnn_corpus_avgpool_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                 ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                 ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                 ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
                                 ctx.grid, ctx.block, ctx.stream);
    }
    return cudaGetLastError();
}
bool CudaAvgPoolFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ib = ac.m, ic_p = ac.n, ih = ac.h, iw = ac.w;
    const int kx = ac.k, sx = ac.stride, padX = ac.orderType;
    const int oh = (ih + 2 * padX - kx) / sx + 1, ow = oh;
    for (int b = 0; b < ib; ++b)
      for (int c = 0; c < ic_p; ++c)
        for (int oy = 0; oy < oh; ++oy)
          for (int ox = 0; ox < ow; ++ox) {
            float sum = 0.0f; int cnt = 0;
            for (int fy = 0; fy < kx; ++fy)
              for (int fx = 0; fx < kx; ++fx) {
                int iy = oy * sx - padX + fy, ix = ox * sx - padX + fx;
                if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                int off;
                if (ac.tag == "1.2.0") {
                    int z = b * ic_p + c;
                    off = z * ih * iw + iy * iw + ix;
                } else {
                    off = ((b * ih + iy) * iw + ix) * ic_p + c;
                }
                sum += ac.validatorInputA[off]; ++cnt;
              }
            if (cnt == 0) cnt = 1;
            int outOff;
            if (ac.tag == "1.2.0") {
                int z = b * ic_p + c;
                outOff = z * oh * ow + oy * ow + ox;
            } else {
                outOff = ((b * oh + oy) * ow + ox) * ic_p + c;
            }
            if (std::fabs(output[outOff] - sum / cnt) > 1e-3f) return false;
          }
    return true;
}

// ============================================================================
// Pool: global avgpool / global maxpool
// ============================================================================
bool CudaGlobalAvgPoolFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_global_avgpool_fp32";
    const int outside = spec.intParam("outside", 4);
    const int axis = spec.intParam("axis", 16);
    const int inside = spec.intParam("inside", 1);
    const int per_block_size = 128;
    const int calc_multi_num = (axis + per_block_size - 1) / per_block_size;
    std::vector<float> input(outside * axis * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outside * inside * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    for (int v : {outside, axis, inside, per_block_size, calc_multi_num})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    // grid = outside * inside (one block per output element)
    ac.globalSize[0] = outside * inside; ac.localSize[0] = per_block_size; ac.dims = 1;
    ac.validatorInputA = input;
    ac.m = outside; ac.n = axis; ac.k = inside; ac.elementCount = outside * inside;
    return true;
}
cudaError_t CudaGlobalAvgPoolFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_global_avgpool_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                    ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                                    ctx.intArgs[3], ctx.intArgs[4], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaGlobalAvgPoolFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    for (int o = 0; o < outside; ++o)
      for (int x = 0; x < inside; ++x) {
        float sum = 0.0f;
        for (int a = 0; a < axis; ++a) sum += ac.validatorInputA[(o * axis + a) * inside + x];
        if (std::fabs(output[o * inside + x] - sum / axis) > 1e-3f) return false;
      }
    return true;
}

bool CudaGlobalMaxPoolFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_global_maxpool_fp32";
    const int outside = spec.intParam("outside", 4);
    const int axis = spec.intParam("axis", 16);
    const int inside = spec.intParam("inside", 1);
    const int per_block_size = 128;
    const int calc_multi_num = (axis + per_block_size - 1) / per_block_size;
    std::vector<float> input(outside * axis * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outside * inside * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    for (int v : {outside, axis, inside, per_block_size, calc_multi_num})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = outside * inside; ac.localSize[0] = per_block_size; ac.dims = 1;
    ac.validatorInputA = input;
    ac.m = outside; ac.n = axis; ac.k = inside; ac.elementCount = outside * inside;
    return true;
}
cudaError_t CudaGlobalMaxPoolFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_global_maxpool_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                    ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2],
                                    ctx.intArgs[3], ctx.intArgs[4], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaGlobalMaxPoolFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    for (int o = 0; o < outside; ++o)
      for (int x = 0; x < inside; ++x) {
        float mx = ac.validatorInputA[o * axis * inside + x];
        for (int a = 1; a < axis; ++a) mx = std::max(mx, ac.validatorInputA[(o * axis + a) * inside + x]);
        if (std::fabs(output[o * inside + x] - mx) > 1e-3f) return false;
      }
    return true;
}

void registerCudaOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReluFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReluInt8Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaClampFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastBoolKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastF322I32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaCastI322F32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryAtan2Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryModKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBinaryLogicalOrKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaRangeFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSelectFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSoftmaxFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaLayerNormFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaPreluFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaScaleFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaMaxPoolFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaAvgPoolFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGlobalAvgPoolFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGlobalMaxPoolFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGatherV2Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaArgMaxFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaArgMinFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInterpNearestFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaNhwc2NchwFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaNchw2NhwcFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGridSampleNearestFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionSumFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionMeanFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionMaxFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionMinFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaReductionProdFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInterpBilinearFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaInterpNearestRoundFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGridSampleBilinearFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaTopKV2Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaBlitRegionFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGridSampleNearest3dFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaGridSampleBilinear3dFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaConvDwFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaPackC4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaUnpackC4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaSetZeroFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CudaAddBiasFp32Kernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
