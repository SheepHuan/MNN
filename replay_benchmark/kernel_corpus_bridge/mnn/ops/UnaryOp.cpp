#include "UnaryOp.hpp"
#include <cmath>
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

namespace {
// Shared exp validator: output[i] == exp(input[i]).
bool validateExpImpl(const AdaptedCase& ac, const std::vector<float>& output) {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - std::exp(ac.validatorInputA[i])) > 1e-3f) return false;
    }
    return true;
}
} // namespace

bool OpenCLUnaryExpKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "unary_buf";
    const int count = spec.intParam("size", 1024);
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = 0.1f * (i % 10);

    AdaptedBuffer inBuf;
    inBuf.setFp32(input);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = count * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = input;
    ac.elementCount = count;

    const int cb = (count + 3) / 4;
    if (spec.tag == "1.2.0") {
        // args: dim0=cb, dim1=1, dim2=1, input, output, height=1
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::sizeConst(2));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(1));
        ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
        ac.dims = 3;
    } else {
        // 3.6.0: dim0=cb, dim1=1, input, output, size=count
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(count));
        ac.globalSize[0] = cb; ac.globalSize[1] = 1;
        ac.dims = 2;
    }
    return true;
}

bool VulkanUnaryExpKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    // MNN Vulkan unary.comp (EXP baked into SPIR-V).
    // layout(binding=0) output FLOAT4[], binding=1 input FLOAT4[],
    // binding=2 uniform { ivec4 size; vec4 slope; }
    // size.x = element count in FLOAT4 units. dispatch globalSize.x = size.x, local 256.
    ac.entry = "vulkan_unary_buf_exp_fp32";
    const int count = spec.intParam("size", 1024);
    const int n4 = (count + 3) / 4;
    const int totalFloats = n4 * 4;

    std::vector<float> input(totalFloats);
    for (int i = 0; i < totalFloats; ++i) input[i] = 0.1f * (i % 10);

    AdaptedBuffer inBuf;
    inBuf.setFp32(input);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = totalFloats * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {1, 0};  // input->binding=1, output->binding=0
    ac.validatorInputA = input;
    ac.elementCount = totalFloats;

    struct PushParam { int32_t size[4]; float slope[4]; };
    PushParam pp;
    pp.size[0] = n4; pp.size[1] = 0; pp.size[2] = 0; pp.size[3] = 0;
    pp.slope[0] = 0.0f; pp.slope[1] = 0.0f; pp.slope[2] = 0.0f; pp.slope[3] = 0.0f;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    ac.globalSize[0] = ((static_cast<uint32_t>(n4) + localX - 1) / localX) * localX;
    ac.globalSize[1] = 1;
    ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool OpenCLUnaryExpKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return validateExpImpl(ac, output);
}

bool VulkanUnaryExpKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    return validateExpImpl(ac, output);
}

void registerUnaryOp() {
    static struct Reg {
        Reg() {
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new OpenCLUnaryExpKernel()));
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new VulkanUnaryExpKernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
