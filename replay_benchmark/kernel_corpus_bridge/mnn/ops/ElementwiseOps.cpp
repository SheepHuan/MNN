#include "ElementwiseOps.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

static void fillInput(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % 13);
}

// cast_buf 3.6.0: (dim0, dim1, INPUT* in, OUTPUT* out, int size)
bool CastBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "cast_buf";
    const int n = spec.intParam("size", 256);
    const int cb = (n + 3) / 4;
    std::vector<float> in; fillInput(in, n);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));   // dim0 = cb
    ac.args.push_back(AdaptedArg::sizeConst(1));   // dim1 = 1
    ac.args.push_back(AdaptedArg::buffer(0));      // in
    ac.args.push_back(AdaptedArg::buffer(1));      // out
    ac.args.push_back(AdaptedArg::scalarInt(n));   // size
    ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.dims = 2;
    ac.validatorInputA = in;
    return true;
}

// select_buf 3.6.0: (dim0, dim1, int* select, FLOAT* in0, FLOAT* in1, FLOAT* out)
bool SelectBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "select_buf";
    const int n = spec.intParam("size", 256);
    const int cb = (n + 3) / 4;
    std::vector<float> in0, in1; fillInput(in0, n); fillInput(in1, n);
    std::vector<int> sel(n, 0);
    AdaptedBuffer selBuf; selBuf.setFp32(std::vector<float>(n, 0)); selBuf.isOutput = false;
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(selBuf); ac.buffers.push_back(b0); ac.buffers.push_back(b1); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));  // select
    ac.args.push_back(AdaptedArg::buffer(1));  // in0
    ac.args.push_back(AdaptedArg::buffer(2));  // in1
    ac.args.push_back(AdaptedArg::buffer(3));  // out
    ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.dims = 2;
    ac.validatorInputA = in0;
    return true;
}

// range_buf 3.6.0: (dim0, dim1, INPUT* start, INPUT* step, OUTPUT* out, int size)
bool RangeBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "range_buf";
    const int n = spec.intParam("size", 256);
    const int cb = (n + 3) / 4;
    std::vector<float> start(n, 0.0f), step(n, 1.0f);
    AdaptedBuffer startBuf; startBuf.setFp32(start); startBuf.isOutput = false;
    AdaptedBuffer stepBuf; stepBuf.setFp32(step); stepBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(startBuf); ac.buffers.push_back(stepBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));  // start
    ac.args.push_back(AdaptedArg::buffer(1));  // step
    ac.args.push_back(AdaptedArg::buffer(2));  // out
    ac.args.push_back(AdaptedArg::scalarInt(n));
    ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.dims = 2;
    ac.validatorInputA = start;
    return true;
}

// unary_buf_fp32 1.2.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int height)
// unary_buf_fp32 3.6.0: (dim0, dim1, INPUT* in, OUTPUT* out, int size)
bool UnaryBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "unary_buf";
    const int n = spec.intParam("size", 256);
    const int cb = (n + 3) / 4;
    std::vector<float> in; fillInput(in, n);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.compileMacros.push_back("-DOPERATOR=in");
    if (spec.tag == "1.2.0") {
        ac.args.push_back(AdaptedArg::sizeConst(0));   // dim0 = cb
        ac.args.push_back(AdaptedArg::sizeConst(1));   // dim1 = 1
        ac.args.push_back(AdaptedArg::sizeConst(2));   // dim2 = 1
        ac.args.push_back(AdaptedArg::buffer(0));      // in
        ac.args.push_back(AdaptedArg::buffer(1));      // out
        ac.args.push_back(AdaptedArg::scalarInt(1));   // height
        ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.globalSize[2] = 1; ac.dims = 3;
    } else {
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(n));
        ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.dims = 2;
    }
    ac.validatorInputA = in;
    return true;
}

// reduction_buf_fp32 1.2.0: (dim0, dim1, FLOAT* in, FLOAT* out, int batch, int height, int width)
// reduction_buf_fp32 3.6.0: (dim0, dim1, dim2, INPUT* in, OUTPUT* out, int inside, int outside, int dim)
bool ReductionBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "reduct_buf";
    const int batch = spec.batch(), height = spec.h(), width = spec.w();
    const int n = batch * height * width * 4;
    std::vector<float> in; fillInput(in, n);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = batch * width * 4 * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.compileMacros.push_back("-DOPERATE=(num+in)");
    if (spec.tag == "1.2.0") {
        ac.args.push_back(AdaptedArg::sizeConst(0));   // dim0 = batch
        ac.args.push_back(AdaptedArg::sizeConst(1));   // dim1 = width
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(batch));
        ac.args.push_back(AdaptedArg::scalarInt(height));
        ac.args.push_back(AdaptedArg::scalarInt(width));
        ac.globalSize[0] = batch; ac.globalSize[1] = width; ac.dims = 2;
    } else {
        const int inside = width, outside = batch, dim = height;
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::sizeConst(2));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(inside));
        ac.args.push_back(AdaptedArg::scalarInt(outside));
        ac.args.push_back(AdaptedArg::scalarInt(dim));
        ac.globalSize[0] = outside; ac.globalSize[1] = 1; ac.globalSize[2] = 1; ac.dims = 3;
    }
    ac.validatorInputA = in;
    return true;
}

// raster_buf_fp32: (dim0, dim1, FLOAT* output) — same as buffer_set_zero
bool RasterBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "buffer_set_zero";
    const int n = spec.intParam("size", 1024);
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));   // dim0 = n
    ac.args.push_back(AdaptedArg::sizeConst(1));   // dim1 = 1
    ac.args.push_back(AdaptedArg::buffer(0));      // output
    ac.globalSize[0] = n; ac.globalSize[1] = 1; ac.dims = 2;
    return true;
}

// softmax_buf_fp32 1.2.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int outCh, int remainCh, int4 shape)
// softmax_buf_fp32 3.6.0: (dim0, dim1, dim2, FLOAT* in, FLOAT* out, int inside, int outside, int dim)
bool SoftmaxBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int channels = spec.c(), w = spec.w(), h = spec.h();
    const int cb = (channels + 3) / 4;
    const int n = cb * w * h * 4;
    std::vector<float> in; fillInput(in, n);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    if (spec.tag == "1.2.0") {
        ac.entry = "softmax_channel";
        ac.args.push_back(AdaptedArg::sizeConst(0));   // dim0 = cb
        ac.args.push_back(AdaptedArg::sizeConst(1));   // dim1 = w
        ac.args.push_back(AdaptedArg::sizeConst(2));   // dim2 = h
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(channels));   // output_channels
        ac.args.push_back(AdaptedArg::scalarInt(0));            // remain_channels
        ac.args.push_back(AdaptedArg::int4(1, h, w, cb));      // shape NCHW
        ac.globalSize[0] = cb; ac.globalSize[1] = w; ac.globalSize[2] = h; ac.dims = 3;
    } else {
        ac.entry = "softmax_in1_buf";
        const int inside = w, outside = h * cb, dim = 1;
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::sizeConst(2));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(inside));
        ac.args.push_back(AdaptedArg::scalarInt(outside));
        ac.args.push_back(AdaptedArg::scalarInt(dim));
        ac.globalSize[0] = outside; ac.globalSize[1] = 1; ac.globalSize[2] = 1; ac.dims = 3;
    }
    ac.validatorInputA = in;
    return true;
}

void registerElementwiseOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            r.registerAdapter(std::unique_ptr<OpAdapter>(new CastBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new SelectBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new RangeBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new UnaryBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new ReductionBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new RasterBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new SoftmaxBufOp()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
