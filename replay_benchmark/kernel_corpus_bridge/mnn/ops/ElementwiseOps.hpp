#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_ELEMENTWISE_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_ELEMENTWISE_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// cast_buf 3.6.0: (dim0, dim1, INPUT* in, OUTPUT* out, int size)
class OpenCLCastKernel : public OpAdapter {
public:
    const char* opType() const override { return "cast"; }
    const char* variant() const override { return "cast_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

// select_buf 3.6.0: (dim0, dim1, int* select, FLOAT* in0, FLOAT* in1, FLOAT* out)
class OpenCLSelectKernel : public OpAdapter {
public:
    const char* opType() const override { return "select"; }
    const char* variant() const override { return "select_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

// range_buf 3.6.0: (dim0, dim1, INPUT* start, INPUT* step, OUTPUT* out, int size)
class OpenCLRangeKernel : public OpAdapter {
public:
    const char* opType() const override { return "range"; }
    const char* variant() const override { return "range_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

// unary_buf_fp32 1.2.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int height)
// unary_buf_fp32 3.6.0: (dim0, dim1, INPUT* in, OUTPUT* out, int size)
class OpenCLUnaryKernel : public OpAdapter {
public:
    const char* opType() const override { return "unary"; }
    const char* variant() const override { return "unary_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

// reduction_buf_fp32 1.2.0: (dim0, dim1, FLOAT* in, FLOAT* out, int batch, int height, int width)
// reduction_buf_fp32 3.6.0: (dim0, dim1, dim2, INPUT* in, OUTPUT* out, int inside, int outside, int dim)
class OpenCLReductionKernel : public OpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "reduction_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

// raster_buf_fp32 1.2.0+3.6.0: (dim0, dim1, FLOAT* output) — same as buffer_set_zero
class OpenCLRasterKernel : public OpAdapter {
public:
    const char* opType() const override { return "raster"; }
    const char* variant() const override { return "raster_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

// softmax_buf_fp32 1.2.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int outCh, int remainCh, int4 shape)
// softmax_buf_fp32 3.6.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int inside, int outside, int dim)
class OpenCLSoftmaxKernel : public OpAdapter {
public:
    const char* opType() const override { return "softmax"; }
    const char* variant() const override { return "softmax_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

void registerElementwiseOps();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
