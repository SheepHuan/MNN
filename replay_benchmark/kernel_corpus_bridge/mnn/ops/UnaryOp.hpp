#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_UNARY_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_UNARY_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// UnaryKernelBase: pure interface contract for the "unary" op's kernel
// variants. The unary op has one logical computation (elementwise transform)
// but multiple kernel implementations (OpenCL unary_buf, Vulkan unary_buf_exp).
// Each subclass fully implements adapt() — data preparation and backend-
// specific argument layout are the subclass's own concern; the base holds
// no shared state or shared adapt() body.
class UnaryKernelBase : public OpAdapter {
public:
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override = 0;
};

// OpenCL unary_buf_exp kernel: 1.2.0 (3 dims) and 3.6.0 (2 dims) layouts.
class OpenCLUnaryExpKernel : public UnaryKernelBase {
public:
    const char* opType() const override { return "unary"; }
    const char* variant() const override { return "unary_buf_exp_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// MNN Vulkan unary.comp kernel (EXP baked into SPIR-V).
class VulkanUnaryExpKernel : public UnaryKernelBase {
public:
    const char* opType() const override { return "unary"; }
    const char* variant() const override { return "vulkan_unary_buf_exp_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerUnaryOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_UNARY_OP_HPP
