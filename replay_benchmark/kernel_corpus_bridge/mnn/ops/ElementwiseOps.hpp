#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_ELEMENTWISE_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_ELEMENTWISE_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// cast_buf 3.6.0: (dim0, dim1, INPUT* in, OUTPUT* out, int size)
class CastBufOp : public OpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cast_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// select_buf 3.6.0: (dim0, dim1, int* select, FLOAT* in0, FLOAT* in1, FLOAT* out)
class SelectBufOp : public OpAdapter {
public:
    const char* opType() const override { return "select"; }
    const char* variant() const override { return "select_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// range_buf 3.6.0: (dim0, dim1, INPUT* start, INPUT* step, OUTPUT* out, int size)
class RangeBufOp : public OpAdapter {
public:
    const char* opType() const override { return "range"; }
    const char* variant() const override { return "range_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// unary_buf_fp32 1.2.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int height)
// unary_buf_fp32 3.6.0: (dim0, dim1, INPUT* in, OUTPUT* out, int size)
class UnaryBufOp : public OpAdapter {
public:
    const char* opType() const override { return "unary"; }
    const char* variant() const override { return "unary_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// reduction_buf_fp32 1.2.0: (dim0, dim1, FLOAT* in, FLOAT* out, int batch, int height, int width)
// reduction_buf_fp32 3.6.0: (dim0, dim1, dim2, INPUT* in, OUTPUT* out, int inside, int outside, int dim)
class ReductionBufOp : public OpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "reduction_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// raster_buf_fp32 1.2.0+3.6.0: (dim0, dim1, FLOAT* output) — same as buffer_set_zero
class RasterBufOp : public OpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "raster_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// softmax_buf_fp32 1.2.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int outCh, int remainCh, int4 shape)
// softmax_buf_fp32 3.6.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int inside, int outside, int dim)
class SoftmaxBufOp : public OpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "softmax_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerElementwiseOps();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
