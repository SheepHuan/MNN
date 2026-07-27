#include "BinaryOp.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool OpenCLBinaryAddKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "binary_buf";
    const int elemCount = spec.intParam("size", 256);
    const int cb = (elemCount + 3) / 4;

    std::vector<float> in0(elemCount), in1(elemCount);
    for (int i = 0; i < elemCount; ++i) {
        in0[i] = 0.1f * (i % 7);
        in1[i] = 0.1f * (i % 5);
    }
    AdaptedBuffer buf0; buf0.setFp32(in0); buf0.isOutput = false;
    AdaptedBuffer buf1; buf1.setFp32(in1); buf1.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = elemCount * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(buf0);
    ac.buffers.push_back(buf1);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = in0;
    ac.validatorInputB = in1;
    ac.elementCount = elemCount;

    ac.compileMacros.push_back("-DOPERATOR=(in0+in1)");

    if (spec.tag == "1.2.0") {
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::int4(1, 1, 1, cb));
        ac.args.push_back(AdaptedArg::int2(1, 1));
        ac.globalSize[0] = cb; ac.globalSize[1] = 1;
        ac.dims = 2;
    } else {
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::scalarInt(elemCount));
        ac.args.push_back(AdaptedArg::scalarInt(0));
        ac.globalSize[0] = cb; ac.globalSize[1] = 1;
        ac.dims = 2;
    }
    return true;
}

bool VulkanBinaryAddKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    // MNN Vulkan binary.comp (ADD baked into SPIR-V).
    // layout(binding=0) output FLOAT4[], binding=1 input0 FLOAT4[],
    // binding=2 input1 FLOAT4[], binding=3 uniform
    // { ivec4 stride00; int activationType; }
    ac.entry = "vulkan_binary_buf_add_fp32";
    const int elemCount = spec.intParam("size", 256);
    const int n4 = (elemCount + 3) / 4;
    const int totalFloats = n4 * 4;

    std::vector<float> in0(totalFloats), in1(totalFloats);
    for (int i = 0; i < elemCount; ++i) {
        in0[i] = 0.1f * (i % 7);
        in1[i] = 0.1f * (i % 5);
    }
    for (int i = elemCount; i < totalFloats; ++i) { in0[i] = 0.0f; in1[i] = 0.0f; }
    AdaptedBuffer buf0; buf0.setFp32(in0); buf0.isOutput = false;
    AdaptedBuffer buf1; buf1.setFp32(in1); buf1.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = totalFloats * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(buf0);
    ac.buffers.push_back(buf1);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {1, 2, 0};  // input0->binding=1, input1->binding=2, output->binding=0
    ac.validatorInputA = in0;
    ac.validatorInputB = in1;
    ac.elementCount = totalFloats;

    struct PushParam { int32_t stride00[4]; int32_t activationType; };
    PushParam pp;
    pp.stride00[0] = 1; pp.stride00[1] = 1; pp.stride00[2] = n4; pp.stride00[3] = n4;
    pp.activationType = 0;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t gx = ((static_cast<uint32_t>(n4) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

void registerBinaryOp() {
    static struct Reg {
        Reg() {
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new OpenCLBinaryAddKernel()));
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new VulkanBinaryAddKernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
