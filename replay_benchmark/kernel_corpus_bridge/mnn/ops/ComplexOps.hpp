#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_COMPLEX_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_COMPLEX_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

class OpenCLAttentionKernel : public OpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "attention_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLSelfAttentionKernel : public OpAdapter {
public:
    const char* opType() const override { return "self_attention_buf"; }
    const char* variant() const override { return "self_attention_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLGemmConv1x1Kernel : public OpAdapter {
public:
    const char* opType() const override { return "gemm"; }
    const char* variant() const override { return "gemm_conv1x1_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLGemvConv1x1Kernel : public OpAdapter {
public:
    const char* opType() const override { return "gemv"; }
    const char* variant() const override { return "gemv_conv1x1_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLGridSampleKernel : public OpAdapter {
public:
    const char* opType() const override { return "grid_sample"; }
    const char* variant() const override { return "grid_sample_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLBufferConvertSubgroupKernel : public OpAdapter {
public:
    const char* opType() const override { return "buffer_convert_subgroup_buf"; }
    const char* variant() const override { return "buffer_convert_subgroup_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLConv2dIntKernel : public OpAdapter {
public:
    const char* opType() const override { return "conv"; }
    const char* variant() const override { return "conv_2d_int_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLLinearAttentionKernel : public OpAdapter {
public:
    const char* opType() const override { return "linear_attention_buf"; }
    const char* variant() const override { return "linear_attention_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLMatmulLocalKernel : public OpAdapter {
public:
    const char* opType() const override { return "matmul"; }
    const char* variant() const override { return "matmul_local_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLScaleNobiasKernel : public OpAdapter {
public:
    const char* opType() const override { return "scale"; }
    const char* variant() const override { return "scale_nobias_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

void registerComplexOps();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
