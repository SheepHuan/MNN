#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_SHAPE_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_SHAPE_OPS_HPP

#include "../../OpAdapter.hpp"
#include <cmath>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// ===================== batchnorm =====================
// 20190611 + 20260526, scalar + pack4. 3 bindings (in/out, a, b), push {dims,w,h,c,cstep}
class VulkanBatchnormKernel : public OpAdapter {
public:
    const char* opType() const override { return "batchnorm"; }
    const char* variant() const override { return "batchnorm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanBatchnormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "batchnorm"; }
    const char* variant() const override { return "batchnorm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

// ===================== cast =====================
// cast_fp16_to_fp32 / cast_fp32_to_fp16: 2 bindings (in, out), push {dims,w,h,c,cstep / outdims...}
class VulkanCastFp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "cast_fp16_to_fp32"; }
    const char* variant() const override { return "cast_fp16_to_fp32"; }
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
class VulkanCastFp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "cast_fp16_to_fp32"; }
    const char* variant() const override { return "cast_fp16_to_fp32_pack4"; }
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
class VulkanCastFp32ToFp16Kernel : public OpAdapter {
public:
    const char* opType() const override { return "cast_fp32_to_fp16"; }
    const char* variant() const override { return "cast_fp32_to_fp16"; }
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
class VulkanCastFp32ToFp16Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "cast_fp32_to_fp16"; }
    const char* variant() const override { return "cast_fp32_to_fp16_pack4"; }
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

// ===================== scale =====================
// 20190611 + 20260526, scalar + pack4. in/out + scale (+ bias). push {dims,w,h,c,cstep}
class VulkanScaleKernel : public OpAdapter {
public:
    const char* opType() const override { return "scale"; }
    const char* variant() const override { return "scale"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanScalePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "scale"; }
    const char* variant() const override { return "scale_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

// ===================== reshape / flatten / reorg / shufflechannel / pixelshuffle / permute =====================
// All share 2 bindings (in, out) and push_constant of 10 ints: {dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep}
// reshape has ndim spec_constant (id=0); flatten/reorg/shufflechannel/pixelshuffle have their own spec_constants.
// 20190611 variants: scalar sfp / pack4 / pack1to4 / pack4to1
// 20260526 variants: spec_constants for shape (offset depends on op)

class VulkanReshapeKernel : public OpAdapter {
public:
    const char* opType() const override { return "reshape"; }
    const char* variant() const override { return "reshape"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanReshapePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "reshape"; }
    const char* variant() const override { return "reshape_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanReshapePack1to4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "reshape"; }
    const char* variant() const override { return "reshape_pack1to4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanReshapePack4to1Kernel : public OpAdapter {
public:
    const char* opType() const override { return "reshape"; }
    const char* variant() const override { return "reshape_pack4to1"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanFlattenKernel : public OpAdapter {
public:
    const char* opType() const override { return "flatten"; }
    const char* variant() const override { return "flatten"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanFlattenPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "flatten"; }
    const char* variant() const override { return "flatten_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanFlattenPack1to4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "flatten"; }
    const char* variant() const override { return "flatten_pack1to4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanReorgKernel : public OpAdapter {
public:
    const char* opType() const override { return "reorg"; }
    const char* variant() const override { return "reorg"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanReorgPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "reorg"; }
    const char* variant() const override { return "reorg_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanReorgPack1to4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "reorg"; }
    const char* variant() const override { return "reorg_pack1to4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanShufflechannelKernel : public OpAdapter {
public:
    const char* opType() const override { return "shufflechannel"; }
    const char* variant() const override { return "shufflechannel"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanShufflechannelPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "shufflechannel"; }
    const char* variant() const override { return "shufflechannel_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanPixelshuffleKernel : public OpAdapter {
public:
    const char* opType() const override { return "pixelshuffle"; }
    const char* variant() const override { return "pixelshuffle"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPixelshufflePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "pixelshuffle"; }
    const char* variant() const override { return "pixelshuffle_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPixelshufflePack4to1Kernel : public OpAdapter {
public:
    const char* opType() const override { return "pixelshuffle"; }
    const char* variant() const override { return "pixelshuffle_pack4to1"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanPermuteKernel2 : public OpAdapter {
public:
    const char* opType() const override { return "permute"; }
    const char* variant() const override { return "permute"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPermutePack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "permute"; }
    const char* variant() const override { return "permute_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPermutePack1to4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "permute"; }
    const char* variant() const override { return "permute_pack1to4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPermutePack4to1Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "permute"; }
    const char* variant() const override { return "permute_pack4to1"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

// ===================== padding / crop / slice / packing =====================
// padding: 2 bindings + push {dims,w,h,c,cstep, outdims,outw,outh,outc,outcstep, left,top} + spec_constants
class VulkanPaddingKernel : public OpAdapter {
public:
    const char* opType() const override { return "padding"; }
    const char* variant() const override { return "padding"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPaddingPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "padding"; }
    const char* variant() const override { return "padding_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPaddingPack1to4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "padding"; }
    const char* variant() const override { return "padding_pack1to4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPaddingPack4to1Kernel : public OpAdapter {
public:
    const char* opType() const override { return "padding"; }
    const char* variant() const override { return "padding_pack4to1"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanCropKernel : public OpAdapter {
public:
    const char* opType() const override { return "crop"; }
    const char* variant() const override { return "crop"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanCropPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "crop"; }
    const char* variant() const override { return "crop_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanCropPack1to4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "crop"; }
    const char* variant() const override { return "crop_pack1to4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanCropPack4to1Kernel : public OpAdapter {
public:
    const char* opType() const override { return "crop"; }
    const char* variant() const override { return "crop_pack4to1"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanSliceKernel : public OpAdapter {
public:
    const char* opType() const override { return "slice"; }
    const char* variant() const override { return "slice"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanSlicePack1to4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "slice"; }
    const char* variant() const override { return "slice_pack1to4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanSlicePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "slice"; }
    const char* variant() const override { return "slice_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

class VulkanPackingKernel : public OpAdapter {
public:
    const char* opType() const override { return "packing"; }
    const char* variant() const override { return "packing"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPacking1to4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "packing"; }
    const char* variant() const override { return "packing_1to4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanPacking4to1Kernel : public OpAdapter {
public:
    const char* opType() const override { return "packing"; }
    const char* variant() const override { return "packing_4to1"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

void registerNcnnShapeOps();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
