#include "PoolingOp.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool OpenCLPoolingMaxKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "pooling";
    const int ih = spec.h(), iw = spec.w(), channel = spec.c();
    const int channel_block = (channel + 3) / 4;
    const int kh = spec.kernelSize(), kw = spec.kernelSize();
    const int stride = spec.stride();
    const int oh = (ih - kh) / stride + 1;
    const int ow = (iw - kw) / stride + 1;

    const int inputCount = ih * iw * channel_block * 4;
    const int outputCount = oh * ow * channel_block * 4;
    std::vector<float> input(inputCount);
    for (int i = 0; i < inputCount; ++i) input[i] = 0.1f * (i % 13);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = outputCount * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = input;
    ac.h = ih; ac.w = iw; ac.c = channel; ac.k = kh;
    ac.elementCount = outputCount;

    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::int2(ih, iw));
    ac.args.push_back(AdaptedArg::int2(oh, ow));
    ac.args.push_back(AdaptedArg::int2(0, 0));
    ac.args.push_back(AdaptedArg::int2(stride, stride));
    ac.args.push_back(AdaptedArg::int2(kh, kh));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(channel_block));
    ac.globalSize[0] = ow; ac.globalSize[1] = oh; ac.globalSize[2] = channel_block;
    ac.dims = 3;
    return true;
}

// Shared pooling validate helper. Both OpenCL and Vulkan use NC4HW4 (FLOAT4)
// storage with the same layout: input[((cb*ih+iy)*iw+ix)*4], output[((cb*oh+oy)*ow+ox)*4].
namespace {
bool validatePoolingMaxImpl(const AdaptedCase& ac, const std::vector<float>& output) {
    const int ih = ac.h, iw = ac.w, channel = ac.c, kh = ac.k, kw = ac.k;
    const int stride = 2;
    const int channel_block = (channel + 3) / 4;
    const int oh = (ih - kh) / stride + 1;
    const int ow = (iw - kw) / stride + 1;
    for (int cb = 0; cb < channel_block; ++cb) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                float expected[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {
                        const int iy = oy * stride + ky;
                        const int ix = ox * stride + kx;
                        if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                        const int in_off = ((cb * ih + iy) * iw + ix) * 4;
                        for (int j = 0; j < 4; ++j) {
                            if (in_off + j < static_cast<int>(ac.validatorInputA.size()))
                                expected[j] = std::max(expected[j], ac.validatorInputA[in_off + j]);
                        }
                    }
                }
                const int out_off = ((cb * oh + oy) * ow + ox) * 4;
                for (int j = 0; j < 4; ++j) {
                    if (out_off + j >= static_cast<int>(output.size())) continue;
                    if (std::fabs(output[out_off + j] - expected[j]) > 1e-3f) return false;
                }
            }
        }
    }
    return true;
}

bool validatePoolingAvgImpl(const AdaptedCase& ac, const std::vector<float>& output) {
    const int ih = ac.h, iw = ac.w, channel = ac.c, kh = ac.k, kw = ac.k;
    const int stride = 2;
    const int channel_block = (channel + 3) / 4;
    const int oh = (ih - kh) / stride + 1;
    const int ow = (iw - kw) / stride + 1;
    for (int cb = 0; cb < channel_block; ++cb) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                float sum[4] = {0, 0, 0, 0};
                int count = 0;
                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {
                        const int iy = oy * stride + ky;
                        const int ix = ox * stride + kx;
                        if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                        const int in_off = ((cb * ih + iy) * iw + ix) * 4;
                        for (int j = 0; j < 4; ++j) {
                            if (in_off + j < static_cast<int>(ac.validatorInputA.size()))
                                sum[j] += ac.validatorInputA[in_off + j];
                        }
                        ++count;
                    }
                }
                if (count == 0) count = 1;
                const int out_off = ((cb * oh + oy) * ow + ox) * 4;
                for (int j = 0; j < 4; ++j) {
                    if (out_off + j < static_cast<int>(output.size())) {
                        if (std::fabs(output[out_off + j] - sum[j] / count) > 1e-3f) return false;
                    }
                }
            }
        }
    }
    return true;
}
} // namespace

bool OpenCLPoolingMaxKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return validatePoolingMaxImpl(ac, output);
}

bool VulkanPoolingMaxKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return validatePoolingMaxImpl(ac, output);
}

bool VulkanPoolingAvgKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return validatePoolingAvgImpl(ac, output);
}
// { ivec4 inputSize; ivec4 outputSize; ivec2 pad; ivec2 kernelSize;
//   ivec2 stride; [ivec2 count;] }  (count only for avgpool)
static void fillVulkanPoolingPushConstants(AdaptedCase& ac, int ih, int iw,
                                           int oh, int ow, int channel_block,
                                           int kh, int kw, int stride,
                                           bool avgPool) {
    struct MaxParam {
        int32_t inputSize[4]; int32_t outputSize[4];
        int32_t pad[2]; int32_t kernelSize[2]; int32_t stride[2];
    };
    struct AvgParam {
        int32_t inputSize[4]; int32_t outputSize[4];
        int32_t pad[2]; int32_t kernelSize[2]; int32_t stride[2]; int32_t count[2];
    };
    if (avgPool) {
        AvgParam pp;
        pp.inputSize[0] = iw; pp.inputSize[1] = ih; pp.inputSize[2] = channel_block; pp.inputSize[3] = 0;
        pp.outputSize[0] = ow; pp.outputSize[1] = oh; pp.outputSize[2] = channel_block; pp.outputSize[3] = 0;
        pp.pad[0] = 0; pp.pad[1] = 0;
        pp.kernelSize[0] = kw; pp.kernelSize[1] = kh;
        pp.stride[0] = stride; pp.stride[1] = stride;
        pp.count[0] = 0; pp.count[1] = 0;
        ac.pushConstants.resize(sizeof(pp));
        std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));
    } else {
        MaxParam pp;
        pp.inputSize[0] = iw; pp.inputSize[1] = ih; pp.inputSize[2] = channel_block; pp.inputSize[3] = 0;
        pp.outputSize[0] = ow; pp.outputSize[1] = oh; pp.outputSize[2] = channel_block; pp.outputSize[3] = 0;
        pp.pad[0] = 0; pp.pad[1] = 0;
        pp.kernelSize[0] = kw; pp.kernelSize[1] = kh;
        pp.stride[0] = stride; pp.stride[1] = stride;
        ac.pushConstants.resize(sizeof(pp));
        std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));
    }
}

bool VulkanPoolingMaxKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_maxpool_fp32";
    const int ih = spec.h(), iw = spec.w(), channel = spec.c();
    const int channel_block = (channel + 3) / 4;
    const int kh = spec.kernelSize(), kw = spec.kernelSize();
    const int stride = spec.stride();
    const int oh = (ih - kh) / stride + 1;
    const int ow = (iw - kw) / stride + 1;

    const int inputCount = ih * iw * channel_block * 4;
    const int outputCount = oh * ow * channel_block * 4;
    std::vector<float> input(inputCount);
    for (int i = 0; i < inputCount; ++i) input[i] = 0.1f * (i % 13);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = outputCount * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {1, 0};  // input->binding=1, output->binding=0
    ac.validatorInputA = input;
    ac.h = ih; ac.w = iw; ac.c = channel; ac.k = kh;
    ac.elementCount = outputCount;

    fillVulkanPoolingPushConstants(ac, ih, iw, oh, ow, channel_block, kh, kh, stride, false);
    ac.globalSize[0] = static_cast<uint32_t>(ow);
    ac.globalSize[1] = static_cast<uint32_t>(oh);
    ac.globalSize[2] = static_cast<uint32_t>(channel_block);
    ac.localSize[0] = 8; ac.localSize[1] = 8; ac.localSize[2] = 1;
    ac.dims = 3;
    return true;
}

bool VulkanPoolingAvgKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_avgpool_fp32";
    const int ih = spec.h(), iw = spec.w(), channel = spec.c();
    const int channel_block = (channel + 3) / 4;
    const int kh = spec.kernelSize(), kw = spec.kernelSize();
    const int stride = spec.stride();
    const int oh = (ih - kh) / stride + 1;
    const int ow = (iw - kw) / stride + 1;

    const int inputCount = ih * iw * channel_block * 4;
    const int outputCount = oh * ow * channel_block * 4;
    std::vector<float> input(inputCount);
    for (int i = 0; i < inputCount; ++i) input[i] = 0.1f * (i % 13);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = outputCount * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {1, 0};
    ac.validatorInputA = input;
    ac.h = ih; ac.w = iw; ac.c = channel; ac.k = kh;
    ac.elementCount = outputCount;

    fillVulkanPoolingPushConstants(ac, ih, iw, oh, ow, channel_block, kh, kh, stride, true);
    ac.globalSize[0] = static_cast<uint32_t>(ow);
    ac.globalSize[1] = static_cast<uint32_t>(oh);
    ac.globalSize[2] = static_cast<uint32_t>(channel_block);
    ac.localSize[0] = 8; ac.localSize[1] = 8; ac.localSize[2] = 1;
    ac.dims = 3;
    return true;
}

void registerPoolingOp() {
    static struct Reg {
        Reg() {
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new OpenCLPoolingMaxKernel()));
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new VulkanPoolingMaxKernel()));
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new VulkanPoolingAvgKernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
