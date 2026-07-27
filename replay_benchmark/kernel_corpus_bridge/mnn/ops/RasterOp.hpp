#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_RASTER_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_RASTER_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// RasterKernelBase: pure interface contract for the "raster" op's kernel
// variants. The raster op covers distinct kernels: OpenCL buffer_set_zero
// (output-only zero fill) and Vulkan blit C4 (identity copy with an input
// buffer). Each subclass fully implements adapt() — data preparation and
// backend-specific argument layout are the subclass's own concern; the base
// holds no shared state or shared adapt() body.
class RasterKernelBase : public OpAdapter {
public:
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override = 0;
};

// OpenCL buffer_set_zero kernel: (GLOBAL_SIZE_2_DIMS, output).
class OpenCLRasterSetZeroKernel : public RasterKernelBase {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "buffer_set_zero_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// MNN Vulkan blit.comp kernel (C4 baked: TYPE=FLOAT4). Identity copy.
class VulkanRasterBlitC4Kernel : public RasterKernelBase {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "vulkan_blit_c4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerRasterOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_RASTER_OP_HPP
