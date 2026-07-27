#include "NcnnShapeOps.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// ============================================================
// Helpers
// ============================================================

// 20190611 scalar push_constant {int dims, w, h, c, cstep}  (5 ints, 20 bytes)
struct Push5 { int dims, w, h, c, cstep; };
static void fillPC5(std::vector<uint8_t>& pc, int w, int h, int c) {
    Push5 p;
    p.dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
    p.w = w; p.h = h; p.c = c; p.cstep = w * h;
    pc.resize(sizeof(p));
    std::memcpy(pc.data(), &p, sizeof(p));
}

// 20190611 10-int push_constant {dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep}
struct Push10 { int dims, w, h, c, cstep, outdims, outw, outh, outc, outcstep; };
static void fillPC10(std::vector<uint8_t>& pc, int w, int h, int c) {
    Push10 p;
    int dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
    int cstep = w * h;
    p.dims = dims; p.w = w; p.h = h; p.c = c; p.cstep = cstep;
    p.outdims = dims; p.outw = w; p.outh = h; p.outc = c; p.outcstep = cstep;
    pc.resize(sizeof(p));
    std::memcpy(pc.data(), &p, sizeof(p));
}

// 20260526 12-int push_constant {dims,w,h,d,c,cstep, outdims,outw,outh,outd,outc,outcstep}
struct Push12 { int dims, w, h, d, c, cstep, outdims, outw, outh, outd, outc, outcstep; };
static void fillPC12(std::vector<uint8_t>& pc, int w, int h, int c) {
    Push12 p;
    int dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
    int cstep = w * h;
    p.dims = dims; p.w = w; p.h = h; p.d = 1; p.c = c; p.cstep = cstep;
    p.outdims = dims; p.outw = w; p.outh = h; p.outd = 1; p.outc = c; p.outcstep = cstep;
    pc.resize(sizeof(p));
    std::memcpy(pc.data(), &p, sizeof(p));
}

// 20260526 14-int push_constant for padding/crop: input{6} + output{6} + 2 extra (left,top or woffset,hoffset)
struct Push14 { int a[14]; };
static void fillPC14(std::vector<uint8_t>& pc, int w, int h, int c, int e1, int e2) {
    Push14 p;
    int dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
    int cstep = w * h;
    int a[14] = {dims, w, h, c, cstep, dims, w, h, c, cstep, e1, e2, 0, 0};
    std::memcpy(p.a, a, sizeof(a));
    pc.resize(sizeof(p));
    std::memcpy(pc.data(), &p, sizeof(p));
}

// packing 20260526 push_constant {uint n, c, stride}
struct PushPacking { uint32_t n, c, stride; };
static void fillPCPacking(std::vector<uint8_t>& pc, uint32_t n, uint32_t c, uint32_t stride) {
    PushPacking p;
    p.n = n; p.c = c; p.stride = stride;
    pc.resize(sizeof(p));
    std::memcpy(pc.data(), &p, sizeof(p));
}

static void fillInput(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % 13);
}

// Dispatch helper for 20190611 pack variants: align globalSize to localSize 64 on x.
static void setDispatchWHC(AdaptedCase& ac, int w, int h, int c) {
    const uint32_t lx = 64;
    ac.globalSize[0] = ((w + lx - 1) / lx) * lx;
    ac.globalSize[1] = h;
    ac.globalSize[2] = c;
    ac.localSize[0] = lx;
}

// ============================================================
// batchnorm: 3 bindings (in/out, a, b), push {dims,w,h,c,cstep}
// 20190611: scalar/pack4, dispatch (w,h,c)
// 20260526: scalar/pack4 + spec_constants for shape (offset 0)
// ============================================================
static bool adaptBatchnorm(const CaseSpec& spec, AdaptedCase& ac, bool pack4) {
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    const int cP = (c + 3) / 4 * 4;
    std::vector<float> in, a, b;
    fillInput(in, total);
    fillInput(a, cP);
    fillInput(b, cP);
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true;
    AdaptedBuffer aBuf; aBuf.setFp32(a); aBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(b); bBuf.isOutput = false;
    ac.buffers.push_back(buf);
    ac.buffers.push_back(aBuf);
    ac.buffers.push_back(bBuf);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    if (spec.tag == "20190611") {
        fillPC5(ac.pushConstants, w, h, c);
        setDispatchWHC(ac, w, h, c);
    } else if (spec.tag == "20260526") {
        // 20260526 batchnorm uses shape_constant_id_offset 0
        fillPC5(ac.pushConstants, w, h, c);
        const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
        ac.specConstants.push_back({0, dims});
        ac.specConstants.push_back({1, static_cast<uint32_t>(w)});
        ac.specConstants.push_back({2, static_cast<uint32_t>(h)});
        ac.specConstants.push_back({3, static_cast<uint32_t>(c)});
        ac.specConstants.push_back({4, static_cast<uint32_t>(w * h)});
        setDispatchWHC(ac, w, h, c);
    } else {
        return false;
    }
    (void)pack4;
    return true;
}
bool VulkanBatchnormKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptBatchnorm(spec, ac, false); }
bool VulkanBatchnormPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptBatchnorm(spec, ac, true); }

// ============================================================
// cast_fp16_to_fp32 / cast_fp32_to_fp16: 2 bindings (in, out), push 10-int (20190611)
// 20260526: spec_constants for shape
// ============================================================
static bool adaptCast(const CaseSpec& spec, AdaptedCase& ac) {
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    if (spec.tag == "20190611") {
        fillPC10(ac.pushConstants, w, h, c);
        setDispatchWHC(ac, w, h, c);
    } else if (spec.tag == "20260526") {
        fillPC10(ac.pushConstants, w, h, c);
        const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
        ac.specConstants.push_back({0, dims});
        ac.specConstants.push_back({1, static_cast<uint32_t>(w)});
        ac.specConstants.push_back({2, static_cast<uint32_t>(h)});
        ac.specConstants.push_back({3, static_cast<uint32_t>(c)});
        ac.specConstants.push_back({4, static_cast<uint32_t>(w * h)});
        ac.specConstants.push_back({5, dims});
        ac.specConstants.push_back({6, static_cast<uint32_t>(w)});
        ac.specConstants.push_back({7, static_cast<uint32_t>(h)});
        ac.specConstants.push_back({8, static_cast<uint32_t>(c)});
        ac.specConstants.push_back({9, static_cast<uint32_t>(w * h)});
        setDispatchWHC(ac, w, h, c);
    } else {
        return false;
    }
    return true;
}
bool VulkanCastFp16ToFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptCast(spec, ac); }
bool VulkanCastFp16ToFp32Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptCast(spec, ac); }
bool VulkanCastFp32ToFp16Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptCast(spec, ac); }
bool VulkanCastFp32ToFp16Pack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptCast(spec, ac); }

// ============================================================
// scale: in/out + scale + bias (bias_term spec_constant), push {dims,w,h,c,cstep}
// 20190611: bias_term=id0, scalar/pack4
// 20260526: bias_term=id0, shape spec_constants offset 1
// ============================================================
static bool adaptScale(const CaseSpec& spec, AdaptedCase& ac) {
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    const int cP = (c + 3) / 4 * 4;
    std::vector<float> in, scaleData, biasData;
    fillInput(in, total);
    fillInput(scaleData, cP);
    fillInput(biasData, cP);
    AdaptedBuffer buf; buf.setFp32(in); buf.isOutput = true;
    AdaptedBuffer sBuf; sBuf.setFp32(scaleData); sBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(biasData); bBuf.isOutput = false;
    ac.buffers.push_back(buf);
    ac.buffers.push_back(sBuf);
    ac.buffers.push_back(bBuf);
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    if (spec.tag == "20190611") {
        fillPC5(ac.pushConstants, w, h, c);
        ac.specConstants.push_back({0, 1}); // bias_term = 1
        setDispatchWHC(ac, w, h, c);
    } else if (spec.tag == "20260526") {
        fillPC5(ac.pushConstants, w, h, c);
        ac.specConstants.push_back({0, 1}); // bias_term = 1
        const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
        ac.specConstants.push_back({1, dims});
        ac.specConstants.push_back({2, static_cast<uint32_t>(w)});
        ac.specConstants.push_back({3, static_cast<uint32_t>(h)});
        ac.specConstants.push_back({4, static_cast<uint32_t>(c)});
        ac.specConstants.push_back({5, static_cast<uint32_t>(w * h)});
        setDispatchWHC(ac, w, h, c);
    } else {
        return false;
    }
    return true;
}
bool VulkanScaleKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptScale(spec, ac); }
bool VulkanScalePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptScale(spec, ac); }

// ============================================================
// reshape / flatten / reorg / shufflechannel / pixelshuffle / permute
// All: 2 bindings (in, out), dispatch (outw, outh, outc)
// 20190611: push_constant 10 ints, + own spec_constant id=0
// 20260526: push_constant 12 ints (with d), + spec_constants for shape
// ============================================================

// Generic reshape-like adapter for 20190611 tag.
// extraSpecConst0: the value of spec_constant id=0 (e.g. ndim, stride, group, order_type).
static bool adaptReshapeLike2019(const CaseSpec& spec, AdaptedCase& ac, uint32_t specConst0) {
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    fillPC10(ac.pushConstants, w, h, c);
    ac.specConstants.push_back({0, specConst0});
    setDispatchWHC(ac, w, h, c);
    return true;
}

// Generic reshape-like adapter for 20260526 tag.
// preShapeConsts: spec_constants before shape_constant_id_offset
// shapeOffset: shape_constant_id_offset value
static bool adaptReshapeLike2026(const CaseSpec& spec, AdaptedCase& ac,
                                  const std::vector<std::pair<int, uint32_t>>& preShapeConsts,
                                  int shapeOffset) {
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    fillPC12(ac.pushConstants, w, h, c);
    for (const auto& sc : preShapeConsts) {
        ac.specConstants.push_back(sc);
    }
    const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
    // shape_constant_id_offset + 0..11
    for (int i = 0; i < 6; ++i) {
        uint32_t v = 0;
        switch (i) {
            case 0: v = dims; break;
            case 1: v = static_cast<uint32_t>(w); break;
            case 2: v = static_cast<uint32_t>(h); break;
            case 3: v = 1u; break; // d
            case 4: v = static_cast<uint32_t>(c); break;
            case 5: v = static_cast<uint32_t>(w * h); break;
        }
        ac.specConstants.push_back({shapeOffset + i, v});
    }
    for (int i = 0; i < 6; ++i) {
        uint32_t v = 0;
        switch (i) {
            case 0: v = dims; break;
            case 1: v = static_cast<uint32_t>(w); break;
            case 2: v = static_cast<uint32_t>(h); break;
            case 3: v = 1u; break; // outd
            case 4: v = static_cast<uint32_t>(c); break;
            case 5: v = static_cast<uint32_t>(w * h); break;
        }
        ac.specConstants.push_back({shapeOffset + 6 + i, v});
    }
    setDispatchWHC(ac, w, h, c);
    return true;
}

// reshape: ndim=3 spec_const id=0 (20190611); ndim id=0, shape offset=1 (20260526)
bool VulkanReshapeKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, 3);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, 3}}, 1);
    return false;
}
bool VulkanReshapePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, 3);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, 3}}, 1);
    return false;
}
bool VulkanReshapePack1to4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, 3);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, 3}}, 1);
    return false;
}
bool VulkanReshapePack4to1Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, 3);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, 3}}, 1);
    return false;
}

// flatten: no spec_const on 20190611 (uses push_constant only); 20260526: shape offset 0
bool VulkanFlattenKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, 0);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {}, 0);
    return false;
}
bool VulkanFlattenPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, 0);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {}, 0);
    return false;
}
bool VulkanFlattenPack1to4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, 0);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {}, 0);
    return false;
}

// reorg: stride=id0 (20190611); stride=id0, mode=id1, shape offset=2 (20260526)
bool VulkanReorgKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t stride = static_cast<uint32_t>(spec.stride());
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, stride);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, stride}, {1, 0}}, 2);
    return false;
}
bool VulkanReorgPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t stride = static_cast<uint32_t>(spec.stride());
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, stride);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, stride}, {1, 0}}, 2);
    return false;
}
bool VulkanReorgPack1to4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t stride = static_cast<uint32_t>(spec.stride());
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, stride);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, stride}, {1, 0}}, 2);
    return false;
}

// shufflechannel: group=id0 (20190611 default 1); group=id0, bugihfa=id1, shape offset=2 (20260526)
bool VulkanShufflechannelKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t group = static_cast<uint32_t>(spec.groups());
    if (group == 0) return false; // avoid div-by-zero in shader
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, group);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, group}, {1, 0}}, 2);
    return false;
}
bool VulkanShufflechannelPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t group = static_cast<uint32_t>(spec.groups());
    if (group == 0) return false;
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, group);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, group}, {1, 0}}, 2);
    return false;
}

// pixelshuffle: only 20260526. upscale_factor=id0, mode=id1, shape offset=2
bool VulkanPixelshuffleKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20260526") return false;
    const uint32_t upscale = static_cast<uint32_t>(spec.intParam("upscale_factor", 2));
    return adaptReshapeLike2026(spec, ac, {{0, upscale}, {1, 0}}, 2);
}
bool VulkanPixelshufflePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20260526") return false;
    const uint32_t upscale = static_cast<uint32_t>(spec.intParam("upscale_factor", 2));
    return adaptReshapeLike2026(spec, ac, {{0, upscale}, {1, 0}}, 2);
}
bool VulkanPixelshufflePack4to1Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20260526") return false;
    const uint32_t upscale = static_cast<uint32_t>(spec.intParam("upscale_factor", 2));
    return adaptReshapeLike2026(spec, ac, {{0, upscale}, {1, 0}}, 2);
}

// permute: order_type=id0 (20190611); order_type=id0, bugihfa=id1, shape offset=2 (20260526)
bool VulkanPermuteKernel2::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t orderType = static_cast<uint32_t>(spec.orderType());
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, orderType);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, orderType}, {1, 0}}, 2);
    return false;
}
bool VulkanPermutePack4Kernel2::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t orderType = static_cast<uint32_t>(spec.orderType());
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, orderType);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, orderType}, {1, 0}}, 2);
    return false;
}
bool VulkanPermutePack1to4Kernel2::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t orderType = static_cast<uint32_t>(spec.orderType());
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, orderType);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, orderType}, {1, 0}}, 2);
    return false;
}
bool VulkanPermutePack4to1Kernel2::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const uint32_t orderType = static_cast<uint32_t>(spec.orderType());
    if (spec.tag == "20190611") return adaptReshapeLike2019(spec, ac, orderType);
    if (spec.tag == "20260526") return adaptReshapeLike2026(spec, ac, {{0, orderType}, {1, 0}}, 2);
    return false;
}

// ============================================================
// padding: 2 bindings (in, out), push 14 ints {dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep, left,top, front,0}
// 20190611: type=id0, value=id1 (float)
// 20260526: type=id0, value=id1, per_channel_pad=id2, shape offset=3
// ============================================================
static bool adaptPadding(const CaseSpec& spec, AdaptedCase& ac) {
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    const int left = spec.intParam("pad_left", 0);
    const int top = spec.intParam("pad_top", 0);
    if (spec.tag == "20190611") {
        fillPC14(ac.pushConstants, w, h, c, left, top);
        // type=id0 (1=replicate), value=id1 (0.0f as uint bits)
        ac.specConstants.push_back({0, 1});
        // float spec constant: encode 0.0f as uint32
        float zero = 0.0f;
        uint32_t zeroBits;
        std::memcpy(&zeroBits, &zero, sizeof(zero));
        ac.specConstants.push_back({1, zeroBits});
        setDispatchWHC(ac, w, h, c);
    } else if (spec.tag == "20260526") {
        fillPC14(ac.pushConstants, w, h, c, left, top);
        ac.specConstants.push_back({0, 1}); // type
        float zero = 0.0f;
        uint32_t zeroBits;
        std::memcpy(&zeroBits, &zero, sizeof(zero));
        ac.specConstants.push_back({1, zeroBits}); // value
        ac.specConstants.push_back({2, 0}); // per_channel_pad
        const uint32_t dims = (c == 1 && h == 1) ? 1u : (h == 1 ? 2u : 3u);
        // shape_constant_id_offset = 3
        ac.specConstants.push_back({3, dims});
        ac.specConstants.push_back({4, static_cast<uint32_t>(w)});
        ac.specConstants.push_back({5, static_cast<uint32_t>(h)});
        ac.specConstants.push_back({6, static_cast<uint32_t>(c)});
        ac.specConstants.push_back({7, static_cast<uint32_t>(w * h)});
        ac.specConstants.push_back({8, dims});
        ac.specConstants.push_back({9, static_cast<uint32_t>(w)});
        ac.specConstants.push_back({10, static_cast<uint32_t>(h)});
        ac.specConstants.push_back({11, static_cast<uint32_t>(c)});
        ac.specConstants.push_back({12, static_cast<uint32_t>(w * h)});
        setDispatchWHC(ac, w, h, c);
    } else {
        return false;
    }
    return true;
}
bool VulkanPaddingKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptPadding(spec, ac); }
bool VulkanPaddingPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptPadding(spec, ac); }
bool VulkanPaddingPack1to4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptPadding(spec, ac); }
bool VulkanPaddingPack4to1Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptPadding(spec, ac); }

// ============================================================
// crop: 2 bindings (in, out), push {dims,w,h,d,c,cstep, outdims,outw,outh,outd,outc,outcstep, woffset,hoffset,doffset,coffset}
// 20190611: 11 ints {dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep, woffset,hoffset,coffset} — actually uses 13 ints
// 20260526: bugihfa=id0, shape offset=1
// ============================================================
static bool adaptCrop(const CaseSpec& spec, AdaptedCase& ac) {
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    const int dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
    const int cstep = w * h;
    const int woffset = spec.intParam("woffset", 0);
    const int hoffset = spec.intParam("hoffset", 0);
    const int coffset = spec.intParam("coffset", 0);
    if (spec.tag == "20190611") {
        // 13 ints: dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep, woffset,hoffset,coffset
        int p[13] = {dims, w, h, c, cstep, dims, w, h, c, cstep, woffset, hoffset, coffset};
        ac.pushConstants.resize(sizeof(p));
        std::memcpy(ac.pushConstants.data(), p, sizeof(p));
        setDispatchWHC(ac, w, h, c);
    } else if (spec.tag == "20260526") {
        // 16 ints: dims,w,h,d,c,cstep, outdims,outw,outh,outd,outc,outcstep, woffset,hoffset,doffset,coffset
        int p[16] = {dims, w, h, 1, c, cstep, dims, w, h, 1, c, cstep, woffset, hoffset, 0, coffset};
        ac.pushConstants.resize(sizeof(p));
        std::memcpy(ac.pushConstants.data(), p, sizeof(p));
        ac.specConstants.push_back({0, 0}); // bugihfa
        // shape_constant_id_offset = 1
        ac.specConstants.push_back({1, static_cast<uint32_t>(dims)});
        ac.specConstants.push_back({2, static_cast<uint32_t>(w)});
        ac.specConstants.push_back({3, static_cast<uint32_t>(h)});
        ac.specConstants.push_back({4, 1u}); // d
        ac.specConstants.push_back({5, static_cast<uint32_t>(c)});
        ac.specConstants.push_back({6, static_cast<uint32_t>(cstep)});
        ac.specConstants.push_back({7, static_cast<uint32_t>(dims)});
        ac.specConstants.push_back({8, static_cast<uint32_t>(w)});
        ac.specConstants.push_back({9, static_cast<uint32_t>(h)});
        ac.specConstants.push_back({10, 1u}); // outd
        ac.specConstants.push_back({11, static_cast<uint32_t>(c)});
        ac.specConstants.push_back({12, static_cast<uint32_t>(cstep)});
        setDispatchWHC(ac, w, h, c);
    } else {
        return false;
    }
    return true;
}
bool VulkanCropKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptCrop(spec, ac); }
bool VulkanCropPack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptCrop(spec, ac); }
bool VulkanCropPack1to4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptCrop(spec, ac); }
bool VulkanCropPack4to1Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptCrop(spec, ac); }

// ============================================================
// slice: 2 bindings (in, out), push {dims,w,h,d,c,cstep, outdims,outw,outh,outd,outc,outcstep, offset}
// 20260526 only: axis=id0, shape offset=1
// ============================================================
static bool adaptSlice(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int total = w * h * c;
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    const int dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
    const int cstep = w * h;
    // 13 ints: dims,w,h,d,c,cstep, outdims,outw,outh,outd,outc,outcstep, offset
    int p[13] = {dims, w, h, 1, c, cstep, dims, w, h, 1, c, cstep, 0};
    ac.pushConstants.resize(sizeof(p));
    std::memcpy(ac.pushConstants.data(), p, sizeof(p));
    ac.specConstants.push_back({0, 0}); // axis
    // shape_constant_id_offset = 1
    ac.specConstants.push_back({1, static_cast<uint32_t>(dims)});
    ac.specConstants.push_back({2, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({3, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({4, 1u}); // d
    ac.specConstants.push_back({5, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({6, static_cast<uint32_t>(cstep)});
    ac.specConstants.push_back({7, static_cast<uint32_t>(dims)});
    ac.specConstants.push_back({8, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({9, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({10, 1u}); // outd
    ac.specConstants.push_back({11, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({12, static_cast<uint32_t>(cstep)});
    setDispatchWHC(ac, w, h, c);
    return true;
}
bool VulkanSliceKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptSlice(spec, ac); }
bool VulkanSlicePack1to4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptSlice(spec, ac); }
bool VulkanSlicePack4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptSlice(spec, ac); }

// ============================================================
// packing: 4 bindings (in, in_fp32, out, out_fp32), push {uint n, c, stride}
// 20260526 only: cast_type_from=id0, cast_type_to=id1, shape offset=2 (n, c, stride as uint)
// ============================================================
static bool adaptPacking(const CaseSpec& spec, AdaptedCase& ac, int castFrom, int castTo) {
    if (spec.tag != "20260526") return false;
    ac.entry = "main";
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;
    // pack4 storage: n = total/4 rounded, but the kernel indexes by (gx, gy) where gx in [0,n), gy in [0,c)
    // For packing, n = cstep (elems per channel), c = c
    const uint32_t n = static_cast<uint32_t>(cstep);
    const uint32_t cu = static_cast<uint32_t>(c);
    const uint32_t stride = n; // stride = cstep
    std::vector<float> in; fillInput(in, total);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer inFp32Buf; inFp32Buf.sizeBytes = total * sizeof(float); inFp32Buf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer outFp32Buf; outFp32Buf.sizeBytes = total * sizeof(float); outFp32Buf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(inFp32Buf);
    ac.buffers.push_back(outBuf);
    ac.buffers.push_back(outFp32Buf);
    ac.vulkanBindings = {0, 1, 2, 3};
    ac.validatorInputA = in;
    ac.w = w; ac.h = h; ac.c = c;
    fillPCPacking(ac.pushConstants, n, cu, stride);
    ac.specConstants.push_back({0, static_cast<uint32_t>(castFrom)}); // cast_type_from
    ac.specConstants.push_back({1, static_cast<uint32_t>(castTo)});   // cast_type_to
    // shape_constant_id_offset = 2
    ac.specConstants.push_back({2, n});
    ac.specConstants.push_back({3, cu});
    ac.specConstants.push_back({4, stride});
    const uint32_t lx = 64;
    ac.globalSize[0] = ((n + lx - 1) / lx) * lx;
    ac.globalSize[1] = cu;
    ac.globalSize[2] = 1;
    ac.localSize[0] = lx;
    return true;
}
bool VulkanPackingKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptPacking(spec, ac, 1, 1); }
bool VulkanPacking1to4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptPacking(spec, ac, 1, 1); }
bool VulkanPacking4to1Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const { return adaptPacking(spec, ac, 1, 1); }

// ============================================================
// Registration
// ============================================================
void registerNcnnShapeOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            // batchnorm
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanBatchnormKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanBatchnormPack4Kernel()));
            // cast
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCastFp16ToFp32Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCastFp16ToFp32Pack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCastFp32ToFp16Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCastFp32ToFp16Pack4Kernel()));
            // scale
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanScaleKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanScalePack4Kernel()));
            // reshape
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReshapeKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReshapePack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReshapePack1to4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReshapePack4to1Kernel()));
            // flatten
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanFlattenKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanFlattenPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanFlattenPack1to4Kernel()));
            // reorg
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReorgKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReorgPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReorgPack1to4Kernel()));
            // shufflechannel
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanShufflechannelKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanShufflechannelPack4Kernel()));
            // pixelshuffle
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPixelshuffleKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPixelshufflePack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPixelshufflePack4to1Kernel()));
            // permute (override the placeholder permute; keep order_type=0 identity)
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPermuteKernel2()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPermutePack4Kernel2()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPermutePack1to4Kernel2()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPermutePack4to1Kernel2()));
            // padding
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPaddingKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPaddingPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPaddingPack1to4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPaddingPack4to1Kernel()));
            // crop
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCropKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCropPack4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCropPack1to4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCropPack4to1Kernel()));
            // slice
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSliceKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSlicePack1to4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSlicePack4Kernel()));
            // packing
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPackingKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPacking1to4Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPacking4to1Kernel()));
        }
    } r;
    (void)r;
}

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
