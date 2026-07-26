#include "OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {

OpAdapterRegistry& OpAdapterRegistry::instance() {
    static OpAdapterRegistry reg;
    return reg;
}

void OpAdapterRegistry::registerAdapter(std::unique_ptr<OpAdapter> adapter) {
    mAdapters.push_back(std::move(adapter));
}

bool FallbackAdapter::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    // Minimal fallback: single in/out buffer, 1D dispatch, no validation.
    const int elemCount = spec.elementCount > 0 ? spec.elementCount : 64;
    std::vector<float> input(elemCount);
    for (int i = 0; i < elemCount; ++i) input[i] = 0.1f * (i % 13);
    AdaptedBuffer buf;
    buf.setFp32(input);
    buf.isOutput = true;
    ac.buffers.push_back(buf);
    ac.validatorInputA = input;
    ac.validator = "identity_fp32";
    // For OpenCL: assume 1 hidden dim + 1 buffer arg
    // For Vulkan: 1 binding
    if (spec.backend == "opencl") {
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.globalSize[0] = (elemCount + 3) / 4;
        ac.globalSize[1] = 1;
        ac.dims = 2;
    } else {
        ac.vulkanBindings = {0};
        ac.globalSize[0] = (elemCount + 3) / 4;
        ac.globalSize[1] = 1;
        ac.globalSize[2] = 1;
        ac.localSize[0] = 64;
    }
    return true;
}

void registerFallbackAdapter() {
    static struct Reg {
        Reg() { OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new FallbackAdapter())); }
    } r;
    (void)r;
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
