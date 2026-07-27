#include "NcnnElementwiseOps.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// Helper: fill 20190611 scalar push_constant {dims,w,h,c,cstep}
static void fillPushConst5(std::vector<uint8_t>& pc, int w, int h, int c) {
    struct PP { int dims, w, h, c, cstep; } p;
    p.dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
    p.w = w; p.h = h; p.c = c; p.cstep = w * h;
    pc.resize(sizeof(p));
    std::memcpy(pc.data(), &p, sizeof(p));
}

// Helper: fill 20260526 pack4 push_constant {uint n}
static void fillPushConstN(std::vector<uint8_t>& pc, uint32_t n) {
    struct PP { uint32_t n; } p;
    p.n = n;
    pc.resize(sizeof(p));
    std::memcpy(pc.data(), &p, sizeof(p));
}

static void fillInput(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % 13);
}

// ---- 20190611 scalar in-place elementwise ----
// All share: binding=0 buffer, push_constant {dims,w,h,c,cstep}, dispatch (w,h,c)

#define DEFINE_2019_INPLACE(OpName, op_type_str, variant_str) \
bool OpName::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "20190611") return false; \
    ac.entry = "main"; \
    const int w = spec.w(), h = spec.h(), c = spec.c(); \
    const int total = w * h * c; \
    std::vector<float> in; fillInput(in, total); \
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true; \
    ac.buffers.push_back(buf); \
    ac.vulkanBindings = {0}; \
    ac.validatorInputA = in; \
    fillPushConst5(ac.pushConstants, w, h, c); \
    const uint32_t lx = 64; \
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx; \
    ac.globalSize[1] = h; ac.globalSize[2] = c; \
    ac.localSize[0] = lx; \
    return true; \
}

DEFINE_2019_INPLACE(VulkanClipKernel, "clip", "clip")
DEFINE_2019_INPLACE(VulkanClipPack4Kernel, "clip", "clip_pack4")
DEFINE_2019_INPLACE(VulkanDropoutKernel, "dropout", "dropout")
DEFINE_2019_INPLACE(VulkanDropoutPack4Kernel, "dropout", "dropout_pack4")
DEFINE_2019_INPLACE(VulkanUnaryopKernel, "unaryop", "unaryop")
DEFINE_2019_INPLACE(VulkanUnaryopPack4Kernel, "unaryop", "unaryop_pack4")

#undef DEFINE_2019_INPLACE

// ---- 20190611 multi-buffer (prelu/batchnorm/eltwise have extra bindings) ----
// These have custom adapt implementations below — skip macro versions.

// prelu 20190611: binding=0 in/out, binding=1 slope, push {dims,w,h,c,cstep}
// Already handled by DEFINE_2019_INPLACE above — but prelu has 2 bindings.
// Override: remove the macro version, use custom.
bool VulkanPreluKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20190611") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    const int slopeN = (c + 3) / 4 * 4;
    std::vector<float> in, slope; fillInput(in, total); fillInput(slope, slopeN);
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true;
    AdaptedBuffer slopeBuf; slopeBuf.setFp32(slope); slopeBuf.isOutput = false;
    ac.buffers.push_back(buf); ac.buffers.push_back(slopeBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    fillPushConst5(ac.pushConstants, w, h, c);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}

bool VulkanPreluPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20190611") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    const int slopeN = (c + 3) / 4 * 4;
    std::vector<float> in, slope; fillInput(in, total); fillInput(slope, slopeN);
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true;
    AdaptedBuffer slopeBuf; slopeBuf.setFp32(slope); slopeBuf.isOutput = false;
    ac.buffers.push_back(buf); ac.buffers.push_back(slopeBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    fillPushConst5(ac.pushConstants, w, h, c);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}

// eltwise 20190611: binding=0 in0, binding=1 in1, binding=2 out (or in-place)
// Actually eltwise has 3 buffers: bottom_blob, bottom_blob1, top_blob
bool VulkanEltwiseKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20190611") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in0, in1; fillInput(in0, total); fillInput(in1, total);
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in0;
    fillPushConst5(ac.pushConstants, w, h, c);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}

bool VulkanEltwisePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20190611") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in0, in1; fillInput(in0, total); fillInput(in1, total);
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in0;
    fillPushConst5(ac.pushConstants, w, h, c);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}

// ---- 20260526 pack4 in-place elementwise ----
// All share: binding=0 vec4 buffer, push_constant {uint n}, spec_const id=0 = n

#define DEFINE_2026_PACK4(OpName, op_type_str, variant_str, num_spec_consts) \
bool OpName::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag != "20260526") return false; \
    ac.entry = "main"; \
    const int elemCount = spec.intParam("size", 64); \
    const int n = (elemCount + 3) / 4; \
    const int totalFloats = n * 4; \
    std::vector<float> in; fillInput(in, totalFloats); \
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true; \
    ac.buffers.push_back(buf); \
    ac.vulkanBindings = {0}; \
    ac.validatorInputA = in; \
    fillPushConstN(ac.pushConstants, static_cast<uint32_t>(n)); \
    for (int i = 0; i < (num_spec_consts); ++i) \
        ac.specConstants.push_back({i, static_cast<uint32_t>(n)}); \
    const uint32_t lx = 64; \
    ac.globalSize[0] = ((n + lx - 1) / lx) * lx; \
    ac.globalSize[1] = 1; ac.globalSize[2] = 1; \
    ac.localSize[0] = lx; \
    return true; \
}

DEFINE_2026_PACK4(VulkanClip2026Kernel, "clip", "clip", 3)
DEFINE_2026_PACK4(VulkanCeluKernel, "celu", "celu", 2)
DEFINE_2026_PACK4(VulkanEluKernel, "elu", "elu", 2)
DEFINE_2026_PACK4(VulkanErfKernel, "erf", "erf", 1)
DEFINE_2026_PACK4(VulkanGeluKernel, "gelu", "gelu", 1)
DEFINE_2026_PACK4(VulkanHardsigmoidKernel, "hardsigmoid", "hardsigmoid", 2)
DEFINE_2026_PACK4(VulkanHardswishKernel, "hardswish", "hardswish", 2)
DEFINE_2026_PACK4(VulkanMishKernel, "mish", "mish", 1)
DEFINE_2026_PACK4(VulkanSeluKernel, "selu", "selu", 2)
DEFINE_2026_PACK4(VulkanShrinkKernel, "shrink", "shrink", 2)
DEFINE_2026_PACK4(VulkanSoftplusKernel, "softplus", "softplus", 1)
DEFINE_2026_PACK4(VulkanSwishKernel, "swish", "swish", 1)
DEFINE_2026_PACK4(VulkanSigmoid2019Kernel, "sigmoid", "sigmoid", 1)
DEFINE_2026_PACK4(VulkanSigmoidPack4Kernel2, "sigmoid", "sigmoid_pack4", 1)
DEFINE_2026_PACK4(VulkanTanh2019Kernel, "tanh", "tanh", 1)
DEFINE_2026_PACK4(VulkanTanhPack4Kernel2, "tanh", "tanh_pack4", 1)
DEFINE_2026_PACK4(VulkanAbsval2019Kernel, "absval", "absval", 1)
DEFINE_2026_PACK4(VulkanAbsvalPack4Kernel2, "absval", "absval_pack4", 1)
DEFINE_2026_PACK4(VulkanRelu2019Kernel, "relu", "relu", 1)
DEFINE_2026_PACK4(VulkanReluPack4Kernel2, "relu", "relu_pack4", 1)
DEFINE_2026_PACK4(VulkanDropout2026Kernel, "dropout", "dropout", 1)
DEFINE_2026_PACK4(VulkanEltwise2026Kernel, "eltwise", "eltwise", 1)
DEFINE_2026_PACK4(VulkanUnaryop2026Kernel, "unaryop", "unaryop", 1)

#undef DEFINE_2026_PACK4

// binaryop 20190611: binding=0 in0, binding=1 in1, binding=2 out
bool VulkanBinaryopKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20190611") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in0, in1; fillInput(in0, total); fillInput(in1, total);
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in0;
    fillPushConst5(ac.pushConstants, w, h, c);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}

bool VulkanBinaryopPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20190611") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in0, in1; fillInput(in0, total); fillInput(in1, total);
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in0;
    fillPushConst5(ac.pushConstants, w, h, c);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}

// deepcopy: binding=0 in, binding=1 out, push {dims,w,h,c,cstep}
bool VulkanDeepcopyKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    fillPushConst5(ac.pushConstants, w, h, c);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}

bool VulkanDeepcopyPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    fillPushConst5(ac.pushConstants, w, h, c);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}

void registerNcnnElementwiseOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            // 20190611
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanClipKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanClipPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPreluKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPreluPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanDropoutKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanDropoutPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanEltwiseKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanEltwisePack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanUnaryopKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanUnaryopPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanBinaryopKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanBinaryopPack4Kernel()));
            // 20260526
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanClip2026Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCeluKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanEluKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanErfKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGeluKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanHardsigmoidKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanHardswishKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanMishKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSeluKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanShrinkKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSoftplusKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSwishKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSigmoid2019Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSigmoidPack4Kernel2()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanTanh2019Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanTanhPack4Kernel2()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanAbsval2019Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanAbsvalPack4Kernel2()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanRelu2019Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReluPack4Kernel2()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanDropout2026Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanEltwise2026Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanUnaryop2026Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanDeepcopyKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanDeepcopyPack4Kernel()));
        }
    } r;
    (void)r;
}

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
