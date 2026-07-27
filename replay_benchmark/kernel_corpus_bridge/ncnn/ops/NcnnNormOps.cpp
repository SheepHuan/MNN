#include "NcnnNormOps.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// All norm ops are 20260526 only.

static void fillInput(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % 13);
}

static uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

template <typename T>
static void setPC(std::vector<uint8_t>& pc, const T& v) {
    pc.resize(sizeof(v));
    std::memcpy(pc.data(), &v, sizeof(v));
}

// ============================================================
// groupnorm_coeffs: 5 bindings (coeffs, mean, var, gamma, beta), push {int w}, eps=id0, affine=id1, w=id2, channels_g=id3
// dispatch: (w, 1, 1)
// ============================================================
static bool adaptGroupnormCoeffs(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w();
    const int c = spec.c();
    const int groups = spec.groups();
    const int channelsG = c / std::max(1, groups);
    const int coeffsN = c * w;
    std::vector<float> dummyC, mean, var, gamma, beta;
    fillInput(dummyC, coeffsN);
    fillInput(mean, groups);
    fillInput(var, groups);
    fillInput(gamma, c);
    fillInput(beta, c);
    AdaptedBuffer coeffs; coeffs.setFp32(dummyC); coeffs.isOutput = true;
    AdaptedBuffer meanB; meanB.setFp32(mean); meanB.isOutput = false;
    AdaptedBuffer varB; varB.setFp32(var); varB.isOutput = false;
    AdaptedBuffer gammaB; gammaB.setFp32(gamma); gammaB.isOutput = false;
    AdaptedBuffer betaB; betaB.setFp32(beta); betaB.isOutput = false;
    ac.buffers.push_back(coeffs);
    ac.buffers.push_back(meanB);
    ac.buffers.push_back(varB);
    ac.buffers.push_back(gammaB);
    ac.buffers.push_back(betaB);
    ac.vulkanBindings = {0, 1, 2, 3, 4};
    ac.validatorInputA = dummyC;
    ac.w = w; ac.h = 1; ac.c = c;
    struct PC { int w; } pc; pc.w = w;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, floatBits(spec.epsilon())}); // eps
    ac.specConstants.push_back({1, 1u}); // affine
    ac.specConstants.push_back({2, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({3, static_cast<uint32_t>(channelsG)});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanGroupnormCoeffsKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptGroupnormCoeffs(spec, ac); }
bool VulkanGroupnormCoeffsPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptGroupnormCoeffs(spec, ac); }

// groupnorm_norm: 2 bindings (in/out, coeffs), push {dims,w,h,c,cstep}, shape spec_constants offset 0
// dispatch (w,h,c)
static bool adaptGroupnormNorm(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;
    std::vector<float> in, coeffs;
    fillInput(in, total);
    fillInput(coeffs, c * w);
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true;
    AdaptedBuffer cB; cB.setFp32(coeffs); cB.isOutput = false;
    ac.buffers.push_back(buf);
    ac.buffers.push_back(cB);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
    struct PC { int dims, w, h, c, cstep; } pc;
    pc.dims = dims; pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, dims});
    ac.specConstants.push_back({1, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({2, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({3, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({4, static_cast<uint32_t>(cstep)});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanGroupnormNormKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptGroupnormNorm(spec, ac); }
bool VulkanGroupnormNormPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptGroupnormNorm(spec, ac); }

// groupnorm_reduce_mean: 2 bindings (in, out mean), push {int w,h,c,cstep, float area, int group, int channels_g}, shape spec offset 0
// dispatch (group, 1, 1)
static bool adaptGroupnormReduceMean(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int groups = std::max(1, spec.groups());
    const int channelsG = c / groups;
    const int area = w * h;
    std::vector<float> in, mean;
    fillInput(in, c * cstep);
    fillInput(mean, groups);
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer meanB; meanB.setFp32(mean); meanB.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(meanB);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep; float area; int group; int channels_g; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    pc.area = static_cast<float>(area * channelsG);
    pc.group = groups; pc.channels_g = channelsG;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({1, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({2, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({3, static_cast<uint32_t>(cstep)});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((groups + lx - 1) / lx) * lx;
    ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanGroupnormReduceMeanKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptGroupnormReduceMean(spec, ac); }
bool VulkanGroupnormReduceMeanPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptGroupnormReduceMean(spec, ac); }

// groupnorm_reduce_sum4_fp32: 2 bindings (in, out), push {w,h,c,cstep, outw,outh,outc,outcstep}, no spec_constants
// dispatch (outw, outh, outc)
static bool adaptReduceSum4(const CaseSpec& spec, AdaptedCase& ac, const char* opType) {
    if (spec.tag != "20260526") return false;
    (void)opType;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    // outw = c/4 rounded, outh=1, outc=groups
    const int groups = std::max(1, spec.groups());
    const int outw = (c + 3) / 4;
    const int outh = 1;
    const int outc = groups;
    const int outcstep = outw * outh;
    const int inTotal = c * cstep;
    const int outTotal = outc * outcstep;
    std::vector<float> in; fillInput(in, std::max(inTotal, outTotal));
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer outB; outB.sizeBytes = outTotal * sizeof(float); outB.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(outB);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep, outw, outh, outc, outcstep; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    pc.outw = outw; pc.outh = outh; pc.outc = outc; pc.outcstep = outcstep;
    setPC(ac.pushConstants, pc);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((outw + lx - 1) / lx) * lx;
    ac.globalSize[1] = outh; ac.globalSize[2] = outc;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanGroupnormReduceSum4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptReduceSum4(spec, ac, "groupnorm"); }
bool VulkanGroupnormReduceSum4Fp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptReduceSum4(spec, ac, "groupnorm"); }
bool VulkanGroupnormReduceSum4Fp16ToFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptReduceSum4(spec, ac, "groupnorm"); }
bool VulkanGroupnormReduceSum4Fp16ToFp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptReduceSum4(spec, ac, "groupnorm"); }

// groupnorm_sub_mean_square: 1 binding (in) + 1 (mean) + 1 (out square), push {dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep, channels_g}, shape spec offset 0
// dispatch (w,h,c)
static bool adaptGroupnormSubMeanSquare(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int groups = std::max(1, spec.groups());
    const int channelsG = c / groups;
    const int total = c * cstep;
    std::vector<float> in, mean, sq;
    fillInput(in, total);
    fillInput(mean, groups);
    fillInput(sq, total);
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer meanB; meanB.setFp32(mean); meanB.isOutput = false;
    AdaptedBuffer sqB; sqB.setFp32(sq); sqB.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(meanB);
    ac.buffers.push_back(sqB);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
    struct PC { int dims, w, h, c, cstep, outdims, outw, outh, outc, outcstep, channels_g; } pc;
    pc.dims = dims; pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    pc.outdims = dims; pc.outw = w; pc.outh = h; pc.outc = c; pc.outcstep = cstep;
    pc.channels_g = channelsG;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, dims});
    ac.specConstants.push_back({1, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({2, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({3, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({4, static_cast<uint32_t>(cstep)});
    ac.specConstants.push_back({5, dims});
    ac.specConstants.push_back({6, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({7, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({8, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({9, static_cast<uint32_t>(cstep)});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanGroupnormSubMeanSquareKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptGroupnormSubMeanSquare(spec, ac); }
bool VulkanGroupnormSubMeanSquarePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptGroupnormSubMeanSquare(spec, ac); }

// ============================================================
// layernorm_coeffs: 3 bindings (coeffs, mean, var), push {int num_groups_per_channel, c}, eps=id0
// dispatch (1, num_groups_per_channel, c)
// ============================================================
static bool adaptLayernormCoeffs(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int c = spec.c();
    const int numGroups = spec.intParam("num_groups_per_channel", 1);
    std::vector<float> coeffs, mean, var;
    fillInput(coeffs, c);
    fillInput(mean, c);
    fillInput(var, c);
    AdaptedBuffer cb; cb.setFp32(coeffs); cb.isOutput = true;
    AdaptedBuffer mb; mb.setFp32(mean); mb.isOutput = false;
    AdaptedBuffer vb; vb.setFp32(var); vb.isOutput = false;
    ac.buffers.push_back(cb);
    ac.buffers.push_back(mb);
    ac.buffers.push_back(vb);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = coeffs;
    ac.c = c;
    struct PC { int num_groups_per_channel; int c; } pc;
    pc.num_groups_per_channel = numGroups; pc.c = c;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, floatBits(spec.epsilon())});
    ac.globalSize[0] = 1; ac.globalSize[1] = numGroups; ac.globalSize[2] = c;
    return true;
}
bool VulkanLayernormCoeffsKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormCoeffs(spec, ac); }
bool VulkanLayernormCoeffsPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormCoeffs(spec, ac); }

// layernorm_norm: 4 bindings (in/out, coeffs, gamma, beta), push {w,h,c,cstep}, affine=id0, affine_size=id1
// dispatch (w,h,c)
static bool adaptLayernormNorm(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;
    std::vector<float> in, coeffs, gamma, beta;
    fillInput(in, total);
    fillInput(coeffs, c * w);
    fillInput(gamma, c);
    fillInput(beta, c);
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true;
    AdaptedBuffer cB; cB.setFp32(coeffs); cB.isOutput = false;
    AdaptedBuffer gB; gB.setFp32(gamma); gB.isOutput = false;
    AdaptedBuffer bB; bB.setFp32(beta); bB.isOutput = false;
    ac.buffers.push_back(buf);
    ac.buffers.push_back(cB);
    ac.buffers.push_back(gB);
    ac.buffers.push_back(bB);
    ac.vulkanBindings = {0, 1, 2, 3};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, 1u}); // affine
    ac.specConstants.push_back({1, static_cast<uint32_t>(c)}); // affine_size
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanLayernormNormKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormNorm(spec, ac); }
bool VulkanLayernormNormPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormNorm(spec, ac); }

// layernorm_reduce_mean: 2 bindings (in sum, out mean), push {w,h,c,cstep, float group_size}, no spec_constants
// dispatch (1, h, c)
static bool adaptLayernormReduceMean(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    std::vector<float> sumV, mean;
    fillInput(sumV, c);
    fillInput(mean, c);
    AdaptedBuffer sb; sb.setFp32(sumV); sb.isOutput = false;
    AdaptedBuffer mb; mb.setFp32(mean); mb.isOutput = true;
    ac.buffers.push_back(sb);
    ac.buffers.push_back(mb);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = sumV;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep; float group_size; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    pc.group_size = static_cast<float>(w);
    setPC(ac.pushConstants, pc);
    ac.globalSize[0] = 1; ac.globalSize[1] = h; ac.globalSize[2] = c;
    return true;
}
bool VulkanLayernormReduceMeanKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormReduceMean(spec, ac); }
bool VulkanLayernormReduceMeanPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormReduceMean(spec, ac); }

// layernorm_reduce_sum4_fp32 / _fp16_to_fp32: 2 buffers, push {w,h,c,cstep, outw,outh,outc,outcstep}
// dispatch (outw, outh, outc)
static bool adaptLayernormReduceSum4(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int outw = (w + 3) / 4;
    const int outh = h;
    const int outc = c;
    const int outcstep = outw * outh;
    std::vector<float> in; fillInput(in, c * cstep);
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer outB; outB.sizeBytes = outc * outcstep * sizeof(float); outB.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(outB);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep, outw, outh, outc, outcstep; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    pc.outw = outw; pc.outh = outh; pc.outc = outc; pc.outcstep = outcstep;
    setPC(ac.pushConstants, pc);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((outw + lx - 1) / lx) * lx;
    ac.globalSize[1] = outh; ac.globalSize[2] = outc;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanLayernormReduceSum4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormReduceSum4(spec, ac); }
bool VulkanLayernormReduceSum4Fp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormReduceSum4(spec, ac); }
bool VulkanLayernormReduceSum4Fp16ToFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormReduceSum4(spec, ac); }
bool VulkanLayernormReduceSum4Fp16ToFp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormReduceSum4(spec, ac); }

// layernorm_sub_mean_square: 3 buffers (in, mean, out square), push {w,h,c,cstep}, affine_size=id0
// dispatch (w,h,c)
static bool adaptLayernormSubMeanSquare(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;
    std::vector<float> in, mean, sq;
    fillInput(in, total);
    fillInput(mean, c);
    fillInput(sq, total);
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer mb; mb.setFp32(mean); mb.isOutput = false;
    AdaptedBuffer sqB; sqB.setFp32(sq); sqB.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(mb);
    ac.buffers.push_back(sqB);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, static_cast<uint32_t>(w)}); // affine_size
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanLayernormSubMeanSquareKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormSubMeanSquare(spec, ac); }
bool VulkanLayernormSubMeanSquarePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptLayernormSubMeanSquare(spec, ac); }

// ============================================================
// instancenorm — same structure as groupnorm but per-channel (group=c, channels_g=1)
// instancenorm_coeffs: 5 bindings, push {int w}, eps=id0, affine=id1, w=id2; dispatch (w,1,1)
// ============================================================
static bool adaptInstancenormCoeffs(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w();
    const int c = spec.c();
    const int coeffsN = c * w;
    std::vector<float> coeffs, mean, var, gamma, beta;
    fillInput(coeffs, coeffsN);
    fillInput(mean, c);
    fillInput(var, c);
    fillInput(gamma, c);
    fillInput(beta, c);
    AdaptedBuffer cb; cb.setFp32(coeffs); cb.isOutput = true;
    AdaptedBuffer mb; mb.setFp32(mean); mb.isOutput = false;
    AdaptedBuffer vb; vb.setFp32(var); vb.isOutput = false;
    AdaptedBuffer gb; gb.setFp32(gamma); gb.isOutput = false;
    AdaptedBuffer bb; bb.setFp32(beta); bb.isOutput = false;
    ac.buffers.push_back(cb);
    ac.buffers.push_back(mb);
    ac.buffers.push_back(vb);
    ac.buffers.push_back(gb);
    ac.buffers.push_back(bb);
    ac.vulkanBindings = {0, 1, 2, 3, 4};
    ac.validatorInputA = coeffs;
    ac.w = w; ac.h = 1; ac.c = c;
    struct PC { int w; } pc; pc.w = w;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, floatBits(spec.epsilon())});
    ac.specConstants.push_back({1, 1u});
    ac.specConstants.push_back({2, static_cast<uint32_t>(w)});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanInstancenormCoeffsKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormCoeffs(spec, ac); }
bool VulkanInstancenormCoeffsPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormCoeffs(spec, ac); }

// instancenorm_norm: same as groupnorm_norm (2 bindings, push {dims,w,h,c,cstep}, shape spec offset 0, dispatch w,h,c)
static bool adaptInstancenormNorm(const CaseSpec& spec, AdaptedCase& ac) {
    return adaptGroupnormNorm(spec, ac);
}
bool VulkanInstancenormNormKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormNorm(spec, ac); }
bool VulkanInstancenormNormPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormNorm(spec, ac); }

// instancenorm_reduce_mean: 2 buffers (in, out mean), push {w,h,c,cstep, float area}, shape spec offset 0
// dispatch (1,1,1) — actually per-channel; gx>=1, gy>=1, gz>=1
static bool adaptInstancenormReduceMean(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int area = w * h;
    std::vector<float> in, mean;
    fillInput(in, c);
    fillInput(mean, c);
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer mb; mb.setFp32(mean); mb.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(mb);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep; float area; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    pc.area = static_cast<float>(area);
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({1, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({2, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({3, static_cast<uint32_t>(cstep)});
    ac.globalSize[0] = 1; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    return true;
}
bool VulkanInstancenormReduceMeanKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormReduceMean(spec, ac); }
bool VulkanInstancenormReduceMeanPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormReduceMean(spec, ac); }

// instancenorm_reduce_sum4_fp32 / _fp16_to_fp32: same structure as groupnorm (group=c)
static bool adaptInstancenormReduceSum4(const CaseSpec& spec, AdaptedCase& ac) {
    // Use groupnorm variant with groups=c so channels_g=1, outc=c
    CaseSpec s = spec;
    s.intParams["groups"] = spec.c();
    return adaptReduceSum4(s, ac, "instancenorm");
}
bool VulkanInstancenormReduceSum4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormReduceSum4(spec, ac); }
bool VulkanInstancenormReduceSum4Fp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormReduceSum4(spec, ac); }
bool VulkanInstancenormReduceSum4Fp16ToFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormReduceSum4(spec, ac); }
bool VulkanInstancenormReduceSum4Fp16ToFp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormReduceSum4(spec, ac); }

// instancenorm_sub_mean_square: like groupnorm_sub_mean_square but channels_g=1, no outdims block (uses 10-int push with shape spec offset 0)
// Actually the kernel uses shape_constant_id_offset 0 with 10 spec constants (dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep)
// and push_constant {dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep}
static bool adaptInstancenormSubMeanSquare(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;
    std::vector<float> in, sq;
    fillInput(in, total);
    fillInput(sq, total);
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer sqB; sqB.setFp32(sq); sqB.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(sqB);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
    struct PC { int dims, w, h, c, cstep, outdims, outw, outh, outc, outcstep; } pc;
    pc.dims = dims; pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    pc.outdims = dims; pc.outw = w; pc.outh = h; pc.outc = c; pc.outcstep = cstep;
    setPC(ac.pushConstants, pc);
    for (int i = 0; i < 5; ++i) {
        uint32_t v = 0;
        switch (i) { case 0: v = dims; break; case 1: v = w; break; case 2: v = h; break; case 3: v = c; break; case 4: v = cstep; break; }
        ac.specConstants.push_back({i, v});
    }
    for (int i = 0; i < 5; ++i) {
        uint32_t v = 0;
        switch (i) { case 0: v = dims; break; case 1: v = w; break; case 2: v = h; break; case 3: v = c; break; case 4: v = cstep; break; }
        ac.specConstants.push_back({5 + i, v});
    }
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanInstancenormSubMeanSquareKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormSubMeanSquare(spec, ac); }
bool VulkanInstancenormSubMeanSquarePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptInstancenormSubMeanSquare(spec, ac); }

// ============================================================
// rmsnorm (20260526)
// rmsnorm_coeffs: 2 bindings (coeffs, rms), push {int num_groups_per_channel, channels}, eps=id0
// dispatch (1, num_groups_per_channel, channels)
// rmsnorm_norm: 3 bindings (in/out, coeffs, gamma), push {w,h,c,cstep}, affine=id0, affine_size=id1; dispatch (w,h,c)
// rmsnorm_square: 2 bindings (in, out square), push {w,h,c,cstep}; dispatch (w,h,c)
// ============================================================
static bool adaptRmsnormCoeffs(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int c = spec.c();
    const int numGroups = spec.intParam("num_groups_per_channel", 1);
    std::vector<float> coeffs, rms;
    fillInput(coeffs, c);
    fillInput(rms, c);
    AdaptedBuffer cb; cb.setFp32(coeffs); cb.isOutput = true;
    AdaptedBuffer rb; rb.setFp32(rms); rb.isOutput = false;
    ac.buffers.push_back(cb);
    ac.buffers.push_back(rb);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = coeffs;
    ac.c = c;
    struct PC { int num_groups_per_channel; int channels; } pc;
    pc.num_groups_per_channel = numGroups; pc.channels = c;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, floatBits(spec.epsilon())});
    ac.globalSize[0] = 1; ac.globalSize[1] = numGroups; ac.globalSize[2] = c;
    return true;
}
bool VulkanRmsnormCoeffsKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptRmsnormCoeffs(spec, ac); }
bool VulkanRmsnormCoeffsPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptRmsnormCoeffs(spec, ac); }

static bool adaptRmsnormNorm(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;
    std::vector<float> in, coeffs, gamma;
    fillInput(in, total);
    fillInput(coeffs, c);
    fillInput(gamma, c);
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true;
    AdaptedBuffer cb; cb.setFp32(coeffs); cb.isOutput = false;
    AdaptedBuffer gb; gb.setFp32(gamma); gb.isOutput = false;
    ac.buffers.push_back(buf);
    ac.buffers.push_back(cb);
    ac.buffers.push_back(gb);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, 1u});
    ac.specConstants.push_back({1, static_cast<uint32_t>(c)});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanRmsnormNormKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptRmsnormNorm(spec, ac); }
bool VulkanRmsnormNormPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptRmsnormNorm(spec, ac); }

static bool adaptRmsnormSquare(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;
    std::vector<float> in, sq;
    fillInput(in, total);
    fillInput(sq, total);
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer sqB; sqB.setFp32(sq); sqB.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(sqB);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    setPC(ac.pushConstants, pc);
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanRmsnormSquareKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptRmsnormSquare(spec, ac); }
bool VulkanRmsnormSquarePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptRmsnormSquare(spec, ac); }

// ============================================================
// normalize (20260526)
// normalize_coeffs: 2 bindings (sqsum, coeffs), push {w,h,c,cstep}, across_spatial=id0, across_channel=id1, eps=id2, eps_mode=id3
// dispatch (1, 1, c)
// normalize_norm: 2 bindings (in/out), push {dims,w,h,c,cstep}, spec_constants 0-4 + shape offset 5; dispatch (w,h,c)
// normalize_reduce_sum4_fp32: 2 buffers, push {w,h,c,cstep,outw,outh,outc,outcstep}, across_spatial=id0, across_channel=id1
// normalize_reduce_sum4_fp16_to_fp32: same
// ============================================================
static bool adaptNormalizeCoeffs(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    std::vector<float> sqsum, coeffs;
    fillInput(sqsum, c);
    fillInput(coeffs, c);
    AdaptedBuffer sb; sb.setFp32(sqsum); sb.isOutput = false;
    AdaptedBuffer cb; cb.setFp32(coeffs); cb.isOutput = true;
    ac.buffers.push_back(sb);
    ac.buffers.push_back(cb);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = coeffs;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, 0u}); // across_spatial
    ac.specConstants.push_back({1, 1u}); // across_channel
    ac.specConstants.push_back({2, floatBits(spec.epsilon())});
    ac.specConstants.push_back({3, 0u}); // eps_mode
    ac.globalSize[0] = 1; ac.globalSize[1] = 1; ac.globalSize[2] = c;
    return true;
}
bool VulkanNormalizeCoeffsKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptNormalizeCoeffs(spec, ac); }
bool VulkanNormalizeCoeffsPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptNormalizeCoeffs(spec, ac); }

static bool adaptNormalizeNorm(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true;
    ac.buffers.push_back(buf);
    ac.vulkanBindings = {0};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
    struct PC { int dims, w, h, c, cstep; } pc;
    pc.dims = dims; pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, 0u}); // across_spatial
    ac.specConstants.push_back({1, 1u}); // across_channel
    ac.specConstants.push_back({2, 0u}); // channel_shared
    ac.specConstants.push_back({3, 0u}); // scale_term
    float one = 1.0f;
    ac.specConstants.push_back({4, floatBits(one)});
    // shape_constant_id_offset = 5
    ac.specConstants.push_back({5, dims});
    ac.specConstants.push_back({6, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({7, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({8, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({9, static_cast<uint32_t>(cstep)});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h; ac.globalSize[2] = c;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanNormalizeNormKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptNormalizeNorm(spec, ac); }
bool VulkanNormalizeNormPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptNormalizeNorm(spec, ac); }

static bool adaptNormalizeReduceSum4(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int outw = (w + 3) / 4;
    const int outh = h;
    const int outc = c;
    const int outcstep = outw * outh;
    std::vector<float> in; fillInput(in, c * cstep);
    AdaptedBuffer inB; inB.setFp32(in); inB.isOutput = false;
    AdaptedBuffer outB; outB.sizeBytes = outc * outcstep * sizeof(float); outB.isOutput = true;
    ac.buffers.push_back(inB);
    ac.buffers.push_back(outB);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    struct PC { int w, h, c, cstep, outw, outh, outc, outcstep; } pc;
    pc.w = w; pc.h = h; pc.c = c; pc.cstep = cstep;
    pc.outw = outw; pc.outh = outh; pc.outc = outc; pc.outcstep = outcstep;
    setPC(ac.pushConstants, pc);
    ac.specConstants.push_back({0, 0u});
    ac.specConstants.push_back({1, 1u});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((outw + lx - 1) / lx) * lx;
    ac.globalSize[1] = outh; ac.globalSize[2] = outc;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanNormalizeReduceSum4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptNormalizeReduceSum4(spec, ac); }
bool VulkanNormalizeReduceSum4Fp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptNormalizeReduceSum4(spec, ac); }
bool VulkanNormalizeReduceSum4Fp16ToFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptNormalizeReduceSum4(spec, ac); }
bool VulkanNormalizeReduceSum4Fp16ToFp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptNormalizeReduceSum4(spec, ac); }

// ============================================================
// Registration
// ============================================================
void registerNcnnNormOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            // groupnorm
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormCoeffsKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormCoeffsPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormNormKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormNormPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormReduceMeanKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormReduceMeanPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormReduceSum4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormReduceSum4Fp32Pack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormReduceSum4Fp16ToFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormReduceSum4Fp16ToFp32Pack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormSubMeanSquareKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGroupnormSubMeanSquarePack4Kernel()));
            // layernorm
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormCoeffsKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormCoeffsPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormNormKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormNormPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormReduceMeanKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormReduceMeanPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormReduceSum4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormReduceSum4Fp32Pack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormReduceSum4Fp16ToFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormReduceSum4Fp16ToFp32Pack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormSubMeanSquareKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanLayernormSubMeanSquarePack4Kernel()));
            // instancenorm
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormCoeffsKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormCoeffsPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormNormKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormNormPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormReduceMeanKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormReduceMeanPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormReduceSum4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormReduceSum4Fp32Pack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormReduceSum4Fp16ToFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormReduceSum4Fp16ToFp32Pack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormSubMeanSquareKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanInstancenormSubMeanSquarePack4Kernel()));
            // rmsnorm
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanRmsnormCoeffsKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanRmsnormCoeffsPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanRmsnormNormKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanRmsnormNormPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanRmsnormSquareKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanRmsnormSquarePack4Kernel()));
            // normalize
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormalizeCoeffsKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormalizeCoeffsPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormalizeNormKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormalizeNormPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormalizeReduceSum4Fp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormalizeReduceSum4Fp32Pack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormalizeReduceSum4Fp16ToFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormalizeReduceSum4Fp16ToFp32Pack4Kernel()));
        }
    } r;
    (void)r;
}

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
