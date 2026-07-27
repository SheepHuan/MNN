#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_CONV_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_CONV_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

class OpenCLConv2dKernel : public OpAdapter {
public:
    const char* opType() const override { return "conv"; }
    const char* variant() const override { return "conv_2d_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLDepthwiseConv2dKernel : public OpAdapter {
public:
    const char* opType() const override { return "depthwise_conv"; }
    const char* variant() const override { return "depthwise_conv2d_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLMatmulBufKernel : public OpAdapter {
public:
    const char* opType() const override { return "matmul"; }
    const char* variant() const override { return "matmul_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLGemmKernel : public OpAdapter {
public:
    const char* opType() const override { return "gemm"; }
    const char* variant() const override { return "gemm_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLWinogradTransformKernel : public OpAdapter {
public:
    const char* opType() const override { return "winograd"; }
    const char* variant() const override { return "winogradTransform_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLInterpKernel : public OpAdapter {
public:
    const char* opType() const override { return "interp"; }
    const char* variant() const override { return "interp_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLInputTranseKernel : public OpAdapter {
public:
    const char* opType() const override { return "input_transe"; }
    const char* variant() const override { return "input_transe_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

class OpenCLBufferConvertKernel : public OpAdapter {
public:
    const char* opType() const override { return "buffer_convert_buf"; }
    const char* variant() const override { return "buffer_convert_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

void registerConvOps();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
