#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_NORM_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_NORM_OPS_HPP

#include "../../OpAdapter.hpp"
#include <cmath>

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

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_coeffs"; }
    const char* variant() const override { return "groupnorm_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_norm"; }
    const char* variant() const override { return "groupnorm_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_norm"; }
    const char* variant() const override { return "groupnorm_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormReduceMeanKernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_mean"; }
    const char* variant() const override { return "groupnorm_reduce_mean"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormReduceMeanPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_mean"; }
    const char* variant() const override { return "groupnorm_reduce_mean_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormReduceSum4Fp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "groupnorm_reduce_sum4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormReduceSum4Fp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "groupnorm_reduce_sum4_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormReduceSum4Fp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "groupnorm_reduce_sum4_fp16_to_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormReduceSum4Fp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "groupnorm_reduce_sum4_fp16_to_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormSubMeanSquareKernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_sub_mean_square"; }
    const char* variant() const override { return "groupnorm_sub_mean_square"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanGroupnormSubMeanSquarePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm_sub_mean_square"; }
    const char* variant() const override { return "groupnorm_sub_mean_square_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

// ===================== layernorm (6 variants, 20260526) =====================
class VulkanLayernormCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_coeffs"; }
    const char* variant() const override { return "layernorm_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_coeffs"; }
    const char* variant() const override { return "layernorm_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_norm"; }
    const char* variant() const override { return "layernorm_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_norm"; }
    const char* variant() const override { return "layernorm_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormReduceMeanKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_mean"; }
    const char* variant() const override { return "layernorm_reduce_mean"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormReduceMeanPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_mean"; }
    const char* variant() const override { return "layernorm_reduce_mean_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormReduceSum4Fp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "layernorm_reduce_sum4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormReduceSum4Fp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "layernorm_reduce_sum4_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormReduceSum4Fp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "layernorm_reduce_sum4_fp16_to_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormReduceSum4Fp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "layernorm_reduce_sum4_fp16_to_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormSubMeanSquareKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_sub_mean_square"; }
    const char* variant() const override { return "layernorm_sub_mean_square"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanLayernormSubMeanSquarePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm_sub_mean_square"; }
    const char* variant() const override { return "layernorm_sub_mean_square_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

// ===================== instancenorm (6 variants, 20260526) =====================
class VulkanInstancenormCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_coeffs"; }
    const char* variant() const override { return "instancenorm_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_coeffs"; }
    const char* variant() const override { return "instancenorm_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_norm"; }
    const char* variant() const override { return "instancenorm_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_norm"; }
    const char* variant() const override { return "instancenorm_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormReduceMeanKernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_mean"; }
    const char* variant() const override { return "instancenorm_reduce_mean"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormReduceMeanPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_mean"; }
    const char* variant() const override { return "instancenorm_reduce_mean_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormReduceSum4Fp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "instancenorm_reduce_sum4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormReduceSum4Fp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_sum4_fp32"; }
    const char* variant() const override { return "instancenorm_reduce_sum4_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormReduceSum4Fp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "instancenorm_reduce_sum4_fp16_to_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormReduceSum4Fp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "instancenorm_reduce_sum4_fp16_to_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormSubMeanSquareKernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_sub_mean_square"; }
    const char* variant() const override { return "instancenorm_sub_mean_square"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanInstancenormSubMeanSquarePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "instancenorm_sub_mean_square"; }
    const char* variant() const override { return "instancenorm_sub_mean_square_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

// ===================== rmsnorm (3 variants, 20260526) =====================
class VulkanRmsnormCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_coeffs"; }
    const char* variant() const override { return "rmsnorm_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanRmsnormCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_coeffs"; }
    const char* variant() const override { return "rmsnorm_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanRmsnormNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_norm"; }
    const char* variant() const override { return "rmsnorm_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanRmsnormNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_norm"; }
    const char* variant() const override { return "rmsnorm_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanRmsnormSquareKernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_square"; }
    const char* variant() const override { return "rmsnorm_square"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanRmsnormSquarePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "rmsnorm_square"; }
    const char* variant() const override { return "rmsnorm_square_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

// ===================== normalize (4 variants, 20260526) =====================
class VulkanNormalizeCoeffsKernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_coeffs"; }
    const char* variant() const override { return "normalize_coeffs"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanNormalizeCoeffsPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_coeffs"; }
    const char* variant() const override { return "normalize_coeffs_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanNormalizeNormKernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_norm"; }
    const char* variant() const override { return "normalize_norm"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanNormalizeNormPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_norm"; }
    const char* variant() const override { return "normalize_norm_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanNormalizeReduceSum4Fp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_reduce_sum4_fp32"; }
    const char* variant() const override { return "normalize_reduce_sum4_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanNormalizeReduceSum4Fp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_reduce_sum4_fp32"; }
    const char* variant() const override { return "normalize_reduce_sum4_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanNormalizeReduceSum4Fp16ToFp32Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "normalize_reduce_sum4_fp16_to_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};
class VulkanNormalizeReduceSum4Fp16ToFp32Pack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "normalize_reduce_sum4_fp16_to_fp32"; }
    const char* variant() const override { return "normalize_reduce_sum4_fp16_to_fp32_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override {
    // Smoke test: kernel dispatched successfully.
    (void)ac; (void)output; return true;
    }
};

void registerNcnnNormOps();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
