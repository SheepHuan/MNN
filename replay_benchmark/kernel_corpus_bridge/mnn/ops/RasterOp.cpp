#include "RasterOp.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool OpenCLRasterSetZeroKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "buffer_set_zero";
    const int count = spec.intParam("size", 1024);
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = count * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.globalSize[0] = count;
    ac.globalSize[1] = 1;
    ac.dims = 2;
    ac.elementCount = count;
    return true;
}

bool VulkanRasterBlitC4Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    // MNN Vulkan blit.comp (C4 baked: TYPE=FLOAT4). Identity copy.
    // layout(binding=0) output FLOAT4[], binding=1 input FLOAT4[],
    // binding=2 uniform { ivec4 stride; ivec4 size; ivec4 extent; }
    ac.entry = "vulkan_blit_c4_fp32";
    const int count = spec.intParam("size", 1024);
    const int n4 = (count + 3) / 4;
    const int totalFloats = n4 * 4;

    std::vector<float> input(totalFloats);
    for (int i = 0; i < totalFloats; ++i) input[i] = 0.1f * (i % 13);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = totalFloats * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    // buffers: [0]=input->binding=1, [1]=output->binding=0
    ac.vulkanBindings = {1, 0};
    ac.validatorInputA = input;
    ac.elementCount = totalFloats;

    struct PushParam { int32_t stride[4]; int32_t size[4]; int32_t extent[4]; };
    PushParam pp;
    pp.stride[0] = 1; pp.stride[1] = 0; pp.stride[2] = 0; pp.stride[3] = 0;
    pp.size[0] = 1;   pp.size[1] = 1;   pp.size[2] = 1;   pp.size[3] = n4;
    pp.extent[0] = 1; pp.extent[1] = 0; pp.extent[2] = 0; pp.extent[3] = 0;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    ac.globalSize[0] = ((static_cast<uint32_t>(n4) + localX - 1) / localX) * localX;
    ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

void registerRasterOp() {
    static struct Reg {
        Reg() {
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new OpenCLRasterSetZeroKernel()));
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new VulkanRasterBlitC4Kernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
