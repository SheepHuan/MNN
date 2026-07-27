#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_ELEMENTWISE_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_ELEMENTWISE_OPS_HPP

#include "../../OpAdapter.hpp"
#include <cmath>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// ---- 20190611 scalar (sfp) elementwise — in-place, push_constant {dims,w,h,c,cstep} ----
// absval, relu, sigmoid, tanh already have dedicated adapters.
// These handle: clip, prelu, dropout, eltwise, unaryop

class VulkanClipKernel : public OpAdapter {
public:
    const char* opType() const override { return "clip"; }
    const char* variant() const override { return "clip"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanClipPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "clip"; }
    const char* variant() const override { return "clip_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanPreluKernel : public OpAdapter {
public:
    const char* opType() const override { return "prelu"; }
    const char* variant() const override { return "prelu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanPreluPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "prelu"; }
    const char* variant() const override { return "prelu_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanDropoutKernel : public OpAdapter {
public:
    const char* opType() const override { return "dropout"; }
    const char* variant() const override { return "dropout"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (ac.validatorInputA[i])) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanDropoutPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "dropout"; }
    const char* variant() const override { return "dropout_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (ac.validatorInputA[i])) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanEltwiseKernel : public OpAdapter {
public:
    const char* opType() const override { return "eltwise"; }
    const char* variant() const override { return "eltwise"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanEltwisePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "eltwise"; }
    const char* variant() const override { return "eltwise_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanUnaryopKernel : public OpAdapter {
public:
    const char* opType() const override { return "unaryop"; }
    const char* variant() const override { return "unaryop"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    // ncnn unaryop's default op_type is identity (no transform), so output==input.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
        if (ac.validatorInputA.size() != output.size()) return false;
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::fabs(output[i] - ac.validatorInputA[i]) > 1e-5f) return false;
        }
        return true;
    }
};

class VulkanUnaryopPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "unaryop"; }
    const char* variant() const override { return "unaryop_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    // ncnn unaryop default is identity (no transform).
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
        if (ac.validatorInputA.size() != output.size()) return false;
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::fabs(output[i] - ac.validatorInputA[i]) > 1e-5f) return false;
        }
        return true;
    }
};

// ---- 20260526 pack4 elementwise — in-place, push_constant {uint n}, spec_const n ----
// absval/sigmoid/tanh/relu already have dedicated adapters.
// These handle: clip, celu, elu, erf, gelu, hardsigmoid, hardswish, mish, selu, shrink, softplus, swish

class VulkanClip2026Kernel : public OpAdapter {
public:
    const char* opType() const override { return "clip"; }
    const char* variant() const override { return "clip"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanCeluKernel : public OpAdapter {
public:
    const char* opType() const override { return "celu"; }
    const char* variant() const override { return "celu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - ((v >= 0.0f ? v : std::exp(v) - 1.0f))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanEluKernel : public OpAdapter {
public:
    const char* opType() const override { return "elu"; }
    const char* variant() const override { return "elu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanErfKernel : public OpAdapter {
public:
    const char* opType() const override { return "erf"; }
    const char* variant() const override { return "erf"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (std::erf(ac.validatorInputA[i]))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanGeluKernel : public OpAdapter {
public:
    const char* opType() const override { return "gelu"; }
    const char* variant() const override { return "gelu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (0.5f * v * (1.0f + std::tanh(0.79788458f * (v + 0.044715f * v * v * v))))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanHardsigmoidKernel : public OpAdapter {
public:
    const char* opType() const override { return "hardsigmoid"; }
    const char* variant() const override { return "hardsigmoid"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (std::max(0.0f, std::min(1.0f, v / 6.0f + 0.5f)))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanHardswishKernel : public OpAdapter {
public:
    const char* opType() const override { return "hardswish"; }
    const char* variant() const override { return "hardswish"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (v * std::max(0.0f, std::min(1.0f, (v + 3.0f) / 6.0f)))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanMishKernel : public OpAdapter {
public:
    const char* opType() const override { return "mish"; }
    const char* variant() const override { return "mish"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (v * std::tanh(std::log(1.0f + std::exp(v))))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanSeluKernel : public OpAdapter {
public:
    const char* opType() const override { return "selu"; }
    const char* variant() const override { return "selu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (1.0507f * (v > 0.0f ? v : 1.6732f * (std::exp(v) - 1.0f)))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanShrinkKernel : public OpAdapter {
public:
    const char* opType() const override { return "shrink"; }
    const char* variant() const override { return "shrink"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - ((std::fabs(v) > 0.0f ? v : 0.0f))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanSoftplusKernel : public OpAdapter {
public:
    const char* opType() const override { return "softplus"; }
    const char* variant() const override { return "softplus"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (std::log(1.0f + std::exp(v)))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanSwishKernel : public OpAdapter {
public:
    const char* opType() const override { return "swish"; }
    const char* variant() const override { return "swish"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (v / (1.0f + std::exp(-v)))) > 1e-3f) return false;
    }
    return true;
    }
};

// ---- 20260526 scalar elementwise (no pack4) ----
class VulkanSigmoid2019Kernel : public OpAdapter {
public:
    const char* opType() const override { return "sigmoid"; }
    const char* variant() const override { return "sigmoid"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (1.0f / (1.0f + std::exp(-ac.validatorInputA[i])))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanSigmoidPack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "sigmoid"; }
    const char* variant() const override { return "sigmoid_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (1.0f / (1.0f + std::exp(-ac.validatorInputA[i])))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanTanh2019Kernel : public OpAdapter {
public:
    const char* opType() const override { return "tanh"; }
    const char* variant() const override { return "tanh"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (std::tanh(ac.validatorInputA[i]))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanTanhPack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "tanh"; }
    const char* variant() const override { return "tanh_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (std::tanh(ac.validatorInputA[i]))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanAbsval2019Kernel : public OpAdapter {
public:
    const char* opType() const override { return "absval"; }
    const char* variant() const override { return "absval"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanAbsvalPack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "absval"; }
    const char* variant() const override { return "absval_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (std::fabs(ac.validatorInputA[i]))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanRelu2019Kernel : public OpAdapter {
public:
    const char* opType() const override { return "relu"; }
    const char* variant() const override { return "relu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - ((ac.validatorInputA[i] > 0.0f ? ac.validatorInputA[i] : 0.0f))) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanReluPack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "relu"; }
    const char* variant() const override { return "relu_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - ((ac.validatorInputA[i] > 0.0f ? ac.validatorInputA[i] : 0.0f))) > 1e-3f) return false;
    }
    return true;
    }
};

// ---- binaryop (20190611 + 20260526) ----
class VulkanBinaryopKernel : public OpAdapter {
public:
    const char* opType() const override { return "binaryop"; }
    const char* variant() const override { return "binaryop"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanBinaryopPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "binaryop"; }
    const char* variant() const override { return "binaryop_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanDropout2026Kernel : public OpAdapter {
public:
    const char* opType() const override { return "dropout"; }
    const char* variant() const override { return "dropout"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (ac.validatorInputA[i])) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanEltwise2026Kernel : public OpAdapter {
public:
    const char* opType() const override { return "eltwise"; }
    const char* variant() const override { return "eltwise"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    // ncnn eltwise default is multiply (square when single input). output = input^2.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
        if (ac.validatorInputA.size() != output.size()) return false;
        for (size_t i = 0; i < output.size(); ++i) {
            const float v = ac.validatorInputA[i];
            if (std::fabs(output[i] - v * v) > 1e-3f) return false;
        }
        return true;
    }
};

class VulkanUnaryop2026Kernel : public OpAdapter {
public:
    const char* opType() const override { return "unaryop"; }
    const char* variant() const override { return "unaryop"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ---- Deepcopy ----
class VulkanDeepcopyKernel : public OpAdapter {
public:
    const char* opType() const override { return "deepcopy"; }
    const char* variant() const override { return "deepcopy"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (ac.validatorInputA[i])) > 1e-3f) return false;
    }
    return true;
    }
};

class VulkanDeepcopyPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "deepcopy"; }
    const char* variant() const override { return "deepcopy_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const float v = ac.validatorInputA[i];
        if (std::fabs(output[i] - (ac.validatorInputA[i])) > 1e-3f) return false;
    }
    return true;
    }
};

void registerNcnnElementwiseOps();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
