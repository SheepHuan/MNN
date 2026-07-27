#include "PoolingOp.hpp"
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

// Vulkan pooling uniform layout:
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
