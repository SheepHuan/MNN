#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_REDUCTION_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_REDUCTION_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// ReductionKernelBase: pure interface contract for the "reduction" op's
// kernel variants. The reduction op has OpenCL (NC4HW4 FLOAT4) and Vulkan
// (scalar FLOAT) kernel variants with different storage layouts. Each
// subclass fully implements adapt() — data preparation and backend-specific
// argument layout are the subclass's own concern; the base holds no shared
// state or shared adapt() body.
class ReductionKernelBase : public OpAdapter {
public:
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override = 0;
};

// OpenCL reduct_buf_sum kernel: (GLOBAL_SIZE_2_DIMS, input, output, batch, height, width).
class OpenCLReductionSumKernel : public ReductionKernelBase {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "reduct_buf_sum_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// MNN Vulkan reduce.comp kernel (SUM baked into SPIR-V).
class VulkanReductionSumKernel : public ReductionKernelBase {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "vulkan_reduce_buf_sum_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerReductionOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_REDUCTION_OP_HPP
