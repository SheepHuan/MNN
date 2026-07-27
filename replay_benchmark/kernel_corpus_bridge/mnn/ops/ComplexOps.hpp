#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_COMPLEX_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_COMPLEX_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

class AttentionBufOp : public OpAdapter {
public:
    const char* opType() const override { return "attention"; }
    const char* variant() const override { return "attention_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class SelfAttentionBufOp : public OpAdapter {
public:
    const char* opType() const override { return "self_attention_buf"; }
    const char* variant() const override { return "self_attention_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class GemmConv1x1BufOp : public OpAdapter {
public:
    const char* opType() const override { return "gemm"; }
    const char* variant() const override { return "gemm_conv1x1_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class GemvConv1x1BufOp : public OpAdapter {
public:
    const char* opType() const override { return "gemv"; }
    const char* variant() const override { return "gemv_conv1x1_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class GridSampleBufOp2 : public OpAdapter {
public:
    const char* opType() const override { return "grid_sample"; }
    const char* variant() const override { return "grid_sample_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class BufferConvertSubgroupBufOp : public OpAdapter {
public:
    const char* opType() const override { return "buffer_convert_subgroup_buf"; }
    const char* variant() const override { return "buffer_convert_subgroup_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class Conv2dIntBufOp : public OpAdapter {
public:
    const char* opType() const override { return "conv"; }
    const char* variant() const override { return "conv_2d_int_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class LinearAttentionBufOp : public OpAdapter {
public:
    const char* opType() const override { return "linear_attention_buf"; }
    const char* variant() const override { return "linear_attention_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class MatmulLocalBufOp : public OpAdapter {
public:
    const char* opType() const override { return "matmul"; }
    const char* variant() const override { return "matmul_local_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class ScaleNobiasOp : public OpAdapter {
public:
    const char* opType() const override { return "scale"; }
    const char* variant() const override { return "scale_nobias_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerComplexOps();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
