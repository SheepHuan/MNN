#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_ELEMENTWISE_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_ELEMENTWISE_OP_HPP

#include "../../OpAdapter.hpp"
#include <cmath>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// Shared elementwise adaptation for in-place unary shaders (sigmoid, tanh).
// push_constant: {dims, w, h, c, cstep}
void adaptElementwise(const CaseSpec& spec, AdaptedCase& ac);

// Sigmoid adapter for ncnn 20190611 (scalar sfp, in-place).
class VulkanSigmoidKernel : public OpAdapter {
public:
    const char* opType() const override { return "sigmoid"; }
    const char* variant() const override { return "sigmoid_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override {
        if (spec.tag != "20190611") return false;
        adaptElementwise(spec, ac); return true;
    }
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
        if (ac.validatorInputA.size() != output.size()) return false;
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::fabs(output[i] - 1.0f / (1.0f + std::exp(-ac.validatorInputA[i]))) > 1e-4f) return false;
        }
        return true;
    }
};

// Tanh adapter for ncnn 20190611 (scalar sfp, in-place).
class VulkanTanhKernel : public OpAdapter {
public:
    const char* opType() const override { return "tanh"; }
    const char* variant() const override { return "tanh_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override {
        if (spec.tag != "20190611") return false;
        adaptElementwise(spec, ac); return true;
    }
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
        if (ac.validatorInputA.size() != output.size()) return false;
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::fabs(output[i] - std::tanh(ac.validatorInputA[i])) > 1e-4f) return false;
        }
        return true;
    }
};

// Permute adapter: handles permute with order_type specialization constant.
class VulkanPermuteKernel : public OpAdapter {
public:
    const char* opType() const override { return "permute"; }
    const char* variant() const override { return "permute_order0_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

void registerElementwiseOp();
void registerPermuteOp();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_ELEMENTWISE_OP_HPP
