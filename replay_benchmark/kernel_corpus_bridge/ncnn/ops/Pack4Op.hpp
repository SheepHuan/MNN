#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_PACK4_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_PACK4_OP_HPP

#include "../../OpAdapter.hpp"
#include <cmath>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// Pack4 elementwise adapters for ncnn 20260526 shaders.
// These shaders use sfpvec4 (vec4 packed), single in-place storage buffer,
// push_constant { uint n }, spec constant constant_id=0 = n (element count / 4).
// Each variant is a separate class so multiple adapters share the "sigmoid"
// opType without conflict (distinguished by variant name).

#define PACK4_ELEMENTWISE_OPS(X) \
    X(sigmoid, "sigmoid_pack4_fp32") \
    X(tanh, "tanh_pack4_fp32") \
    X(absval, "absval_pack4_fp32") \
    X(relu, "relu_pack4_fp32_slope0")

#define DECLARE_PACK4_OP(name, vname) \
    class Vulkan##name##Pack4Kernel : public OpAdapter { \
    public: \
        const char* opType() const override { return #name; } \
        const char* variant() const override { return vname; } \
        bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override; \
    };

PACK4_ELEMENTWISE_OPS(DECLARE_PACK4_OP)
#undef DECLARE_PACK4_OP

// Concat adapter for ncnn 20260526 (axis=0, dims=3, scalar sfp).
class VulkanConcatKernel : public OpAdapter {
public:
    const char* opType() const override { return "concat"; }
    const char* variant() const override { return "concat_axis0_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - ac.validatorInputA[i]) > 1e-5f) return false;
    }
    return true;
    }
};

void registerPack4Ops();
void registerConcatOp();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_PACK4_OP_HPP
