#include "VulkanTag120Ops.hpp"
#include <cmath>
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// ============================================================================
// VulkanBlit120Kernel — 1.2.0 blit.comp (identity copy via stride/extent)
// ============================================================================
bool VulkanBlit120Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "1.2.0") return false;
    ac.entry = "vulkan_blit_120_fp32";
    const int n = spec.intParam("size", 256);

    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = 0.1f * (i % 13);
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = n;

    // For an identity copy of n elements viewed as (size.x=1, size.y=1, size.z=1, size.w=n):
    //   pos.x = i / 1 = i, pos.z/y = 0
    //   srcOffset = stride.w + stride.z*0 + stride.y*0 + stride.x*i = stride.x*i
    //   dstOffset = extent.w + extent.x*i
    // Choose stride.x = 1, extent.x = 1, others 0 → srcOffset = dstOffset = i.
    struct PushParam { int32_t stride[4]; int32_t size[4]; int32_t extent[4]; };
    PushParam pp;
    pp.stride[0] = 1; pp.stride[1] = 0; pp.stride[2] = 0; pp.stride[3] = 0;
    pp.size[0]   = 1; pp.size[1]   = 1; pp.size[2]   = 1; pp.size[3]   = n;
    pp.extent[0] = 1; pp.extent[1] = 0; pp.extent[2] = 0; pp.extent[3] = 0;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t gx = ((static_cast<uint32_t>(n) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanBlit120Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - ac.validatorInputA[i]) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// VulkanNc4hw4ToNchw120Kernel — 1.2.0 nc4hw4Tonchw.comp
// ============================================================================
bool VulkanNc4hw4ToNchw120Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "1.2.0") return false;
    ac.entry = "vulkan_nc4hw4_to_nchw_120_fp32";
    const int batch = spec.intParam("batch", 1);
    const int channel = spec.intParam("channel", 8);
    const int h = spec.intParam("h", 4);
    const int w = spec.intParam("w", 4);
    const int c4 = (channel + 3) / 4;

    const int inSize = batch * c4 * h * w * 4;
    const int outSize = batch * channel * h * w;
    std::vector<float> in(inSize);
    for (int i = 0; i < inSize; ++i) in[i] = 0.1f * (i % 11);

    AdaptedBuffer outBuf; outBuf.sizeBytes = outSize * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = outSize;

    struct PushParam { int32_t size[4]; int32_t stride[4]; };
    PushParam pp;
    pp.size[0] = batch; pp.size[1] = channel; pp.size[2] = h; pp.size[3] = w;
    pp.stride[0] = channel * h * w; pp.stride[1] = h * w; pp.stride[2] = w; pp.stride[3] = 1;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t total = static_cast<uint32_t>(batch * c4 * h * w);
    const uint32_t localX = 256;
    const uint32_t gx = ((total + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanNc4hw4ToNchw120Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 8 * sizeof(int32_t)) return false;
    int32_t sz[4], st[4];
    std::memcpy(sz, ac.pushConstants.data(), sizeof(sz));
    std::memcpy(st, ac.pushConstants.data() + sizeof(sz), sizeof(st));
    const int batch = sz[0], channel = sz[1], h = sz[2], w = sz[3];
    const int c4 = (channel + 3) / 4;
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channel; ++c) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int nchwIdx = b * st[0] + c * st[1] + y * st[2] + x * st[3];
                    const int nc4hw4Idx = ((b * c4 + c / 4) * h * w + y * w + x) * 4 + (c % 4);
                    if (std::fabs(output[nchwIdx] - ac.validatorInputA[nc4hw4Idx]) > 1e-3f) return false;
                }
            }
        }
    }
    return true;
}

// ============================================================================
// VulkanNchwToNc4hw4120Kernel — 1.2.0 nchwTonc4hw4.comp
// ============================================================================
bool VulkanNchwToNc4hw4120Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "1.2.0") return false;
    ac.entry = "vulkan_nchw_to_nc4hw4_120_fp32";
    const int batch = spec.intParam("batch", 1);
    const int channel = spec.intParam("channel", 8);
    const int h = spec.intParam("h", 4);
    const int w = spec.intParam("w", 4);
    const int c4 = (channel + 3) / 4;

    // NCHW input
    const int inSize = batch * channel * h * w;
    // NC4HW4 output
    const int outSize = batch * c4 * h * w * 4;
    std::vector<float> in(inSize);
    for (int i = 0; i < inSize; ++i) in[i] = 0.1f * (i % 13);

    // Note: shader binding=0 is sourceBuffer (input), binding=1 is destBuffer (output)
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outSize * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);    // binding=0
    ac.buffers.push_back(outBuf);   // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = inSize;

    struct PushParam { int32_t size[4]; int32_t stride[4]; };
    PushParam pp;
    pp.size[0] = batch; pp.size[1] = channel; pp.size[2] = h; pp.size[3] = w;
    pp.stride[0] = channel * h * w; pp.stride[1] = h * w; pp.stride[2] = w; pp.stride[3] = 1;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t total = static_cast<uint32_t>(batch * c4 * h * w);
    const uint32_t localX = 256;
    const uint32_t gx = ((total + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanNchwToNc4hw4120Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 8 * sizeof(int32_t)) return false;
    int32_t sz[4], st[4];
    std::memcpy(sz, ac.pushConstants.data(), sizeof(sz));
    std::memcpy(st, ac.pushConstants.data() + sizeof(sz), sizeof(st));
    const int batch = sz[0], channel = sz[1], h = sz[2], w = sz[3];
    const int c4 = (channel + 3) / 4;
    // NCHW -> NC4HW4: out[b][c4][y][x][c%4] = in[b][c][y][x] for c in range, else 0
    for (int b = 0; b < batch; ++b) {
        for (int zd4 = 0; zd4 < c4; ++zd4) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    for (int c4idx = 0; c4idx < 4; ++c4idx) {
                        const int c = zd4 * 4 + c4idx;
                        const int outIdx = ((b * c4 + zd4) * h * w + y * w + x) * 4 + c4idx;
                        float expected = 0.0f;
                        if (c < channel) {
                            const int inIdx = b * st[0] + c * st[1] + y * st[2] + x * st[3];
                            expected = ac.validatorInputA[inIdx];
                        }
                        if (std::fabs(output[outIdx] - expected) > 1e-3f) return false;
                    }
                }
            }
        }
    }
    return true;
}

// ============================================================================
// VulkanReduceSum120Kernel — 1.2.0 reduce.comp (SUM variant)
// ============================================================================
bool VulkanReduceSum120Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "1.2.0") return false;
    ac.entry = "vulkan_reduce_sum_120_fp32";
    const int w = spec.intParam("w", 4);   // inside
    const int h = spec.intParam("h", 8);   // axis (reduce dim)
    const int c = spec.intParam("c", 2);   // outside

    const int inSize = w * h * c;
    const int outSize = w * c;
    std::vector<float> in(inSize);
    for (int i = 0; i < inSize; ++i) in[i] = 0.1f * ((i * 3) % 17);

    AdaptedBuffer outBuf; outBuf.sizeBytes = outSize * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = outSize;

    struct PushParam { int32_t w; int32_t h; int32_t c; float k; };
    PushParam pp;
    pp.w = w; pp.h = h; pp.c = c; pp.k = 1.0f;  // k=1.0 for SUM (no scaling)
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t total = static_cast<uint32_t>(outSize);
    const uint32_t gx = ((total + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanReduceSum120Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 4 * sizeof(int32_t)) return false;
    int32_t p[3];
    std::memcpy(p, ac.pushConstants.data(), sizeof(p));
    const int w = p[0], h = p[1], c = p[2];
    if (output.size() != static_cast<size_t>(w * c)) return false;
    for (int oc = 0; oc < c; ++oc) {
        for (int iw = 0; iw < w; ++iw) {
            const int base = oc * h * w + iw;
            float sum = 0.0f;
            for (int i = 0; i < h; ++i) sum += ac.validatorInputA[base + i * w];
            if (std::fabs(output[oc * w + iw] - sum) > 1e-3f) return false;
        }
    }
    return true;
}

// ============================================================================
// VulkanSoftmaxHeight120Kernel — 1.2.0 softmaxHeight_NHWC.comp
// ============================================================================
bool VulkanSoftmaxHeight120Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "1.2.0") return false;
    ac.entry = "vulkan_softmax_height_120_fp32";
    const int w = spec.intParam("w", 4);
    const int h = spec.intParam("h", 16);
    const int c = spec.intParam("c", 2);

    const int n = w * h * c;
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = 0.1f * ((i * 3) % 17);

    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = n;

    struct PushParam { int32_t w; int32_t h; int32_t c; };
    PushParam pp;
    pp.w = w; pp.h = h; pp.c = c;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t lx = 8, ly = 8;
    const uint32_t gx = ((static_cast<uint32_t>(c) + lx - 1) / lx) * lx;
    const uint32_t gy = ((static_cast<uint32_t>(w) + ly - 1) / ly) * ly;
    ac.globalSize[0] = gx; ac.globalSize[1] = gy; ac.globalSize[2] = 1;
    ac.localSize[0] = lx; ac.localSize[1] = ly; ac.localSize[2] = 1;
    ac.dims = 2;
    return true;
}

bool VulkanSoftmaxHeight120Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 3 * sizeof(int32_t)) return false;
    int32_t p[3];
    std::memcpy(p, ac.pushConstants.data(), sizeof(p));
    const int w = p[0], h = p[1], c = p[2];
    if (output.size() != static_cast<size_t>(w * h * c)) return false;
    for (int oc = 0; oc < c; ++oc) {
        for (int iw = 0; iw < w; ++iw) {
            const int base = oc * h * w + iw;
            // 1.2.0 initializes maxValue = -1000.0 then loops i=0..h-1.
            // For h>=1 the result is the same as initializing with the first element.
            float maxV = -1000.0f;
            for (int i = 0; i < h; ++i) {
                maxV = std::max(maxV, ac.validatorInputA[base + i * w]);
            }
            float sum = 0.0f;
            for (int i = 0; i < h; ++i) sum += std::exp(ac.validatorInputA[base + i * w] - maxV);
            for (int i = 0; i < h; ++i) {
                const float expected = std::exp(ac.validatorInputA[base + i * w] - maxV) / sum;
                if (std::fabs(output[base + i * w] - expected) > 1e-3f) return false;
            }
        }
    }
    return true;
}

void registerVulkanTag120Ops() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanBlit120Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNc4hw4ToNchw120Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNchwToNc4hw4120Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanReduceSum120Kernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSoftmaxHeight120Kernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
