#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_BINARY_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_BINARY_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// BinaryKernelBase: pure interface contract for the "binary" op's kernel
// variants. The binary op has one logical computation (elementwise binary)
// but multiple kernel implementations (OpenCL binary_buf, Vulkan
// binary_buf_add). Each subclass fully implements adapt() — data preparation
// and backend-specific argument layout are the subclass's own concern; the
// base holds no shared state or shared adapt() body.
class BinaryKernelBase : public OpAdapter {
public:
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override = 0;
};

// OpenCL binary_buf kernel: 1.2.0 (int4 shape, int2 isFull) and 3.6.0 (int size, int actType).
class OpenCLBinaryAddKernel : public BinaryKernelBase {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "binary_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// MNN Vulkan binary.comp kernel (ADD baked into SPIR-V).
class VulkanBinaryAddKernel : public BinaryKernelBase {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "vulkan_binary_buf_add_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

void registerBinaryOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_BINARY_OP_HPP
