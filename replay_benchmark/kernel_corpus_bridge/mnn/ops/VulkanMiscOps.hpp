#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_VULKAN_MISC_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_VULKAN_MISC_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// ============================================================================
// Vulkan select.comp kernel (3.6.0)
//   layout: binding=0 dest float[], binding=1 in0 float[], binding=2 in1 float[],
//           binding=3 select int[], binding=4 push_constant { ivec4 size }
//   select[i] > 0 ? in0[i] : in1[i]
// ============================================================================
class VulkanSelectKernel : public OpAdapter {
public:
    const char* opType() const override { return "select"; }
    const char* variant() const override { return "vulkan_select_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan range.comp kernel (3.6.0)
//   layout: binding=0 dest float[], binding=1 start float[], binding=2 delta float[],
//           binding=3 push_constant { ivec4 size }
//   out[i] = float(i) * delta[0] + start[0]
// ============================================================================
class VulkanRangeKernel : public OpAdapter {
public:
    const char* opType() const override { return "range"; }
    const char* variant() const override { return "vulkan_range_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan cast_float_int.comp kernel (3.6.0)
//   layout: binding=0 dest OUTPUT_TYPE[], binding=1 input INPUT_TYPE[],
//           binding=2 push_constant { ivec4 size; vec4 slope }
//   The shader uses INPUT_TYPE/OUTPUT_TYPE macros; for the FP32 path we use
//   float->float cast (identity), since the FP32 baked preamble defines both
//   INPUT_TYPE and OUTPUT_TYPE as float.
// ============================================================================
class VulkanCastFloatIntKernel : public OpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "vulkan_cast_float_int_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan scale.comp kernel (3.6.0)
//   layout: binding=0 out FLOAT4[], binding=1 in FLOAT4[], binding=2 scale FLOAT4[],
//           binding=3 bias FLOAT4[], binding=4 push_constant { ivec4 imgSize }
//   NC4HW4 layout. out[i] = in[i] * scale[i/imgSize.x] + bias[i/imgSize.x]
// ============================================================================
class VulkanScaleKernel : public OpAdapter {
public:
    const char* opType() const override { return "scale"; }
    const char* variant() const override { return "vulkan_scale_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan preluWithChannel.comp kernel (3.6.0)
//   layout: binding=0 dest FLOAT4[], binding=1 in FLOAT4[], binding=2 slope FLOAT4[],
//           binding=3 push_constant { ivec4 imgSize }
//   NC4HW4 layout. out[i] = in[i] > 0 ? in[i] : in[i] * slope[i/imgSize.x]
// ============================================================================
class VulkanPreluKernel : public OpAdapter {
public:
    const char* opType() const override { return "prelu"; }
    const char* variant() const override { return "vulkan_prelu_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan argmax.comp kernel (3.6.0)
//   layout: binding=0 dest int[], binding=1 in float[],
//           binding=2 push_constant { ivec4 size } // inside, axis, outside, reduceAxis
//   output[index] = argmax over axis of input[outside, axis, inside]
// ============================================================================
class VulkanArgmaxKernel : public OpAdapter {
public:
    const char* opType() const override { return "argmax"; }
    const char* variant() const override { return "vulkan_argmax_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    // Argmax output is int indices; runner reads back as float. We validate by
    // checking that each output value is a valid non-negative integer index
    // within the axis range (smoke-level validation).
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan softmaxHeight_NHWC.comp kernel (3.6.0)
//   layout: binding=0 dest float[], binding=1 in float[],
//           binding=2 push_constant { int w; int h; int c }
//   softmax along H axis on NHWC layout (no channel pack)
// ============================================================================
class VulkanSoftmaxHeightKernel : public OpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "vulkan_softmax_height_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan norm.comp kernel (3.6.0) — LayerNorm, no gamma/beta (LAYERNORM_SCALE off)
//   layout: binding=0 dest float[], binding=1 in float[],
//           binding=2 push_constant { ivec4 size; float eps },
//           constant_id=3 USE_RMS = 0 (LayerNorm, not RMSNorm)
//   output[y, j] = (in[y, j] - mean(y)) / sqrt(var(y) + eps)
// ============================================================================
class VulkanNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "vulkan_norm_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan resizeNearest.comp kernel (3.6.0)
//   layout: binding=0 out FLOAT4[], binding=1 in FLOAT4[],
//           binding=2 push_constant { ivec4 inImgSize; ivec4 outImgSize; vec4 scale }
//   NC4HW4 nearest-neighbour resize
// ============================================================================
class VulkanResizeNearestKernel : public OpAdapter {
public:
    const char* opType() const override { return "interp"; }
    const char* variant() const override { return "vulkan_resize_nearest_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan resizeBilinear.comp kernel (3.6.0)
//   Same layout as resizeNearest, bilinear interpolation.
// ============================================================================
class VulkanResizeBilinearKernel : public OpAdapter {
public:
    const char* opType() const override { return "interp"; }
    const char* variant() const override { return "vulkan_resize_bilinear_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan gridSampleNearest.comp kernel (3.6.0)
//   layout: binding=0 out FLOAT4[], binding=1 in FLOAT4[], binding=2 grid float[],
//           binding=3 push_constant { ivec4 inShape; ivec4 outShape; bool alignCorners }
// ============================================================================
class VulkanGridSampleNearestKernel : public OpAdapter {
public:
    const char* opType() const override { return "grid_sample"; }
    const char* variant() const override { return "vulkan_grid_sample_nearest_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

// ============================================================================
// Vulkan nc4hw4Tonchw.comp kernel (3.6.0) — layout conversion raster
//   layout: binding=0 dest float[], binding=1 src FLOAT4[],
//           binding=2 push_constant { ivec4 size; ivec4 stride } // NCHW, NCHW stride
// ============================================================================
class VulkanNc4hw4ToNchwKernel : public OpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "vulkan_nc4hw4_to_nchw_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override;
};

void registerVulkanMiscOps();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_VULKAN_MISC_OPS_HPP
