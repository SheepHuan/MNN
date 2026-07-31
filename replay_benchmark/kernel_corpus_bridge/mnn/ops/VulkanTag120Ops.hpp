#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_VULKAN_TAG120_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_VULKAN_TAG120_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// ============================================================================
// Vulkan blit.comp kernel (1.2.0) — scalar float blit (stride/extent copy)
//   layout: binding=0 dst float[], binding=1 src float[],
//           binding=2 push_constant { ivec4 stride; ivec4 size; ivec4 extent }
// ============================================================================
class VulkanBlit120Kernel : public OpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "vulkan_blit_120_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan nc4hw4Tonchw.comp kernel (1.2.0) — NC4HW4 -> NCHW
//   Same layout as 3.6.0 but float/vec4 directly (no FLOAT macro).
// ============================================================================
class VulkanNc4hw4ToNchw120Kernel : public OpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "vulkan_nc4hw4_to_nchw_120_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan nchwTonc4hw4.comp kernel (1.2.0) — NCHW -> NC4HW4
// ============================================================================
class VulkanNchwToNc4hw4120Kernel : public OpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "vulkan_nchw_to_nc4hw4_120_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan reduce.comp kernel (1.2.0, SUM variant) — reduce along H axis
//   layout: binding=0 dst float[], binding=1 src float[],
//           binding=2 push_constant { int w; int h; int c; float k }
// ============================================================================
class VulkanReduceSum120Kernel : public OpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "vulkan_reduce_sum_120_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan softmaxHeight_NHWC.comp kernel (1.2.0)
//   Same layout as 3.6.0 but uses float directly (no FLOAT macro) and
//   initial maxValue = -1000.0 (no first-element init).
// ============================================================================
class VulkanSoftmaxHeight120Kernel : public OpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "vulkan_softmax_height_120_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

void registerVulkanTag120Ops();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_VULKAN_TAG120_OPS_HPP
