#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_POOLING_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_POOLING_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// PoolingKernelBase: pure interface contract for the "pooling" op's kernel
// variants. Input/output use NC4HW4 (FLOAT4) storage on both OpenCL and
// Vulkan paths. Each subclass fully implements adapt() — data preparation
// and backend-specific argument layout are the subclass's own concern; the
// base holds no shared state or shared adapt() body.
class PoolingKernelBase : public OpAdapter {
public:
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override = 0;
};

// OpenCL pooling kernel (max variant, 2x2 stride 2 no pad).
class OpenCLPoolingMaxKernel : public PoolingKernelBase {
public:
    const char* opType() const override { return "pooling"; }
    const char* variant() const override { return "pooling_max_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// MNN Vulkan maxpool.comp kernel.
class VulkanPoolingMaxKernel : public PoolingKernelBase {
public:
    const char* opType() const override { return "pooling"; }
    const char* variant() const override { return "vulkan_maxpool_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// MNN Vulkan avgpool.comp kernel.
class VulkanPoolingAvgKernel : public PoolingKernelBase {
public:
    const char* opType() const override { return "pooling"; }
    const char* variant() const override { return "vulkan_avgpool_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

void registerPoolingOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_POOLING_OP_HPP
