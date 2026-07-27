#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_NORM_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_NORM_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// ===================== groupnorm (6 variants, 20260526) =====================
class VulkanGroupnormCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_coeffs"; }
    const char* variant() const override { return "groupnorm_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_coeffs"; }
    const char* variant() const override { return "groupnorm_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_norm"; }
    const char* variant() const override { return "groupnorm_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_norm"; }
    const char* variant() const override { return "groupnorm_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormReduceMeanKernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_mean"; }
    const char* variant() const override { return "groupnorm_reduce_mean"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormReduceMeanPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_mean"; }
    const char* variant() const override { return "groupnorm_reduce_mean_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormReduceSum4Fp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "groupnorm_reduce_sum4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormReduceSum4Fp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "groupnorm_reduce_sum4_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormReduceSum4Fp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "groupnorm_reduce_sum4_fp16_to_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormReduceSum4Fp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "groupnorm_reduce_sum4_fp16_to_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormSubMeanSquareKernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_sub_mean_square"; }
    const char* variant() const override { return "groupnorm_sub_mean_square"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanGroupnormSubMeanSquarePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_sub_mean_square"; }
    const char* variant() const override { return "groupnorm_sub_mean_square_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ===================== layernorm (6 variants, 20260526) =====================
class VulkanLayernormCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_coeffs"; }
    const char* variant() const override { return "layernorm_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_coeffs"; }
    const char* variant() const override { return "layernorm_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_norm"; }
    const char* variant() const override { return "layernorm_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_norm"; }
    const char* variant() const override { return "layernorm_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormReduceMeanKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_mean"; }
    const char* variant() const override { return "layernorm_reduce_mean"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormReduceMeanPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_mean"; }
    const char* variant() const override { return "layernorm_reduce_mean_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormReduceSum4Fp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "layernorm_reduce_sum4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormReduceSum4Fp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "layernorm_reduce_sum4_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormReduceSum4Fp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "layernorm_reduce_sum4_fp16_to_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormReduceSum4Fp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "layernorm_reduce_sum4_fp16_to_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormSubMeanSquareKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_sub_mean_square"; }
    const char* variant() const override { return "layernorm_sub_mean_square"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanLayernormSubMeanSquarePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_sub_mean_square"; }
    const char* variant() const override { return "layernorm_sub_mean_square_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ===================== instancenorm (6 variants, 20260526) =====================
class VulkanInstancenormCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_coeffs"; }
    const char* variant() const override { return "instancenorm_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_coeffs"; }
    const char* variant() const override { return "instancenorm_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_norm"; }
    const char* variant() const override { return "instancenorm_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_norm"; }
    const char* variant() const override { return "instancenorm_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormReduceMeanKernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_mean"; }
    const char* variant() const override { return "instancenorm_reduce_mean"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormReduceMeanPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_mean"; }
    const char* variant() const override { return "instancenorm_reduce_mean_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormReduceSum4Fp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "instancenorm_reduce_sum4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormReduceSum4Fp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "instancenorm_reduce_sum4_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormReduceSum4Fp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "instancenorm_reduce_sum4_fp16_to_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormReduceSum4Fp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "instancenorm_reduce_sum4_fp16_to_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormSubMeanSquareKernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_sub_mean_square"; }
    const char* variant() const override { return "instancenorm_sub_mean_square"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanInstancenormSubMeanSquarePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_sub_mean_square"; }
    const char* variant() const override { return "instancenorm_sub_mean_square_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ===================== rmsnorm (3 variants, 20260526) =====================
class VulkanRmsnormCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_coeffs"; }
    const char* variant() const override { return "rmsnorm_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanRmsnormCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_coeffs"; }
    const char* variant() const override { return "rmsnorm_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanRmsnormNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_norm"; }
    const char* variant() const override { return "rmsnorm_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanRmsnormNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_norm"; }
    const char* variant() const override { return "rmsnorm_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanRmsnormSquareKernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_square"; }
    const char* variant() const override { return "rmsnorm_square"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanRmsnormSquarePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_square"; }
    const char* variant() const override { return "rmsnorm_square_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ===================== normalize (4 variants, 20260526) =====================
class VulkanNormalizeCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_coeffs"; }
    const char* variant() const override { return "normalize_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanNormalizeCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_coeffs"; }
    const char* variant() const override { return "normalize_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanNormalizeNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_norm"; }
    const char* variant() const override { return "normalize_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanNormalizeNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_norm"; }
    const char* variant() const override { return "normalize_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanNormalizeReduceSum4Fp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_reduce_sum4_fp32"; }
    const char* variant() const override { return "normalize_reduce_sum4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanNormalizeReduceSum4Fp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_reduce_sum4_fp32"; }
    const char* variant() const override { return "normalize_reduce_sum4_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanNormalizeReduceSum4Fp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "normalize_reduce_sum4_fp16_to_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
class VulkanNormalizeReduceSum4Fp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "normalize_reduce_sum4_fp16_to_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerNcnnNormOps();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
