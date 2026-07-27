#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_NORM_CONV_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_NORM_CONV_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

class OpenCLPoolingBufKernel : public OpAdapter {
public:
    const char* opType() const override { return "pooling"; }
    const char* variant() const override { return "pooling_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class OpenCLScaleKernel : public OpAdapter {
public:
    const char* opType() const override { return "scale"; }
    const char* variant() const override { return "scale_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class OpenCLGroupnormKernel : public OpAdapter {
public:
    const char* opType() const override { return "groupnorm"; }
    const char* variant() const override { return "groupnorm_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class OpenCLLayernormKernel : public OpAdapter {
public:
    const char* opType() const override { return "layernorm"; }
    const char* variant() const override { return "layernorm_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class OpenCLSplitgeluKernel : public OpAdapter {
public:
    const char* opType() const override { return "splitgelu"; }
    const char* variant() const override { return "splitgelu_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class OpenCLTopkv2Kernel : public OpAdapter {
public:
    const char* opType() const override { return "topkv2"; }
    const char* variant() const override { return "topkv2_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class OpenCLStrassenBinaryKernel : public OpAdapter {
public:
    const char* opType() const override { return "strassen"; }
    const char* variant() const override { return "strassen_binary_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerNormConvOps();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
