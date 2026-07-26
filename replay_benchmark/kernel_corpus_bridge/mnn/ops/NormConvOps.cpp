#include "NormConvOps.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

static void fillInput(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % 13);
}

// pooling 1.2.0: (dim0,dim1,dim2, FLOAT* in, int2 input_shape, int2 output_shape, int2 pad, int2 stride, int2 kernel, FLOAT* out, int channel_block)
// pooling 3.6.0: same but + FLOAT* rediceOutput
bool PoolingBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "pooling";
    const int ih = 8, iw = 8, channel = 4, kh = 2, kw = 2, stride = 2;
    const int cb = (channel + 3) / 4;
    const int oh = (ih - kh) / stride + 1;
    const int ow = (iw - kw) / stride + 1;
    const int batch = 1;
    const int inN = batch * cb * ih * iw * 4;
    const int outN = batch * cb * oh * ow * 4;
    std::vector<float> in; fillInput(in, inN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));  // dim0 = ow
    ac.args.push_back(AdaptedArg::sizeConst(1));  // dim1 = batch*oh
    ac.args.push_back(AdaptedArg::sizeConst(2));  // dim2 = cb
    ac.args.push_back(AdaptedArg::buffer(0));     // in
    ac.args.push_back(AdaptedArg::int2(ih, iw));  // input_shape
    ac.args.push_back(AdaptedArg::int2(oh, ow));  // output_shape
    ac.args.push_back(AdaptedArg::int2(0, 0));    // pad
    ac.args.push_back(AdaptedArg::int2(stride, stride)); // stride
    ac.args.push_back(AdaptedArg::int2(kh, kw));  // kernel
    ac.args.push_back(AdaptedArg::buffer(1));     // out
    if (spec.tag == "1.2.0") {
        ac.args.push_back(AdaptedArg::scalarInt(cb));
    } else {
        // 3.6.0 has extra rediceOutput buffer
        AdaptedBuffer rediceBuf; rediceBuf.sizeBytes = outN * sizeof(float); rediceBuf.isOutput = false;
        ac.buffers.push_back(rediceBuf);
        ac.args.push_back(AdaptedArg::buffer(2));  // rediceOutput
        ac.args.push_back(AdaptedArg::scalarInt(cb));
    }
    ac.globalSize[0] = ow; ac.globalSize[1] = batch * oh; ac.globalSize[2] = cb; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// scale_buf 1.2.0: (dim0,dim1, FLOAT* in, FLOAT* scale, FLOAT* out, int4 shape)
// scale_buf 3.6.0: (dim0,dim1, FLOAT* in, FLOAT* scale, FLOAT* out, int channelBlock, int ohw, int offset)
bool ScaleBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "scale_buf";
    const int cb = 4, hw = 16;
    const int n = cb * hw * 4;
    std::vector<float> in, scale; fillInput(in, n); fillInput(scale, cb * 4);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer scaleBuf; scaleBuf.setFp32(scale); scaleBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(scaleBuf); ac.buffers.push_back(outBuf);
    if (spec.tag == "1.2.0") {
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::int4(1, hw, 1, cb));
        ac.globalSize[0] = cb; ac.globalSize[1] = hw; ac.dims = 2;
    } else {
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::scalarInt(cb));
        ac.args.push_back(AdaptedArg::scalarInt(hw));
        ac.args.push_back(AdaptedArg::scalarInt(0));
        ac.globalSize[0] = cb; ac.globalSize[1] = hw; ac.dims = 2;
    }
    ac.validatorInputA = in;
    return true;
}

// grid_sample 3.6.0: (dim0,dim1,dim2, FLOAT* in, FLOAT* grid, FLOAT* out, int inH, int inW, int outH, int outW, int batch, int channel, ...)
bool GridSampleBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "nearest_buf";
    const int inH = 8, inW = 8, outH = 4, outW = 4, batch = 1, channel = 4;
    const int cb = (channel + 3) / 4;
    const int inN = batch * cb * inH * inW * 4;
    const int outN = batch * cb * outH * outW * 4;
    const int gridN = batch * outH * outW * 2 * 4;  // grid has x,y per pixel
    std::vector<float> in, grid; fillInput(in, inN); fillInput(grid, gridN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer gridBuf; gridBuf.setFp32(grid); gridBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(gridBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));  // in
    ac.args.push_back(AdaptedArg::buffer(1));  // grid
    ac.args.push_back(AdaptedArg::buffer(2));  // out
    ac.args.push_back(AdaptedArg::scalarInt(inH));
    ac.args.push_back(AdaptedArg::scalarInt(inW));
    ac.args.push_back(AdaptedArg::scalarInt(outH));
    ac.args.push_back(AdaptedArg::scalarInt(outW));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(channel));
    ac.globalSize[0] = outW; ac.globalSize[1] = outH; ac.globalSize[2] = cb; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// groupnorm 3.6.0: (dim0,dim1,dim2, FLOAT* in, FLOAT* out, FLOAT* group, FLOAT* gamma, int4 shape, float eps, ...)
// Simplified — exact layout depends on #ifdef
bool GroupnormBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "groupnorm_plain_buf";
    const int hw = 16, channel = 4, groups = 2;
    const int cb = (channel + 3) / 4;
    const int n = cb * hw * 4;
    std::vector<float> in; fillInput(in, n);
    std::vector<float> group(groups * cb * 4), gamma(groups * cb * 4);
    fillInput(group, groups * cb * 4); fillInput(gamma, groups * cb * 4);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer groupBuf; groupBuf.setFp32(group); groupBuf.isOutput = false;
    AdaptedBuffer gammaBuf; gammaBuf.setFp32(gamma); gammaBuf.isOutput = false;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.buffers.push_back(groupBuf); ac.buffers.push_back(gammaBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));  // in
    ac.args.push_back(AdaptedArg::buffer(1));  // out
    ac.args.push_back(AdaptedArg::buffer(2));  // group
    ac.args.push_back(AdaptedArg::buffer(3));  // gamma
    ac.args.push_back(AdaptedArg::scalarInt(hw));
    ac.args.push_back(AdaptedArg::scalarInt(groups));
    ac.args.push_back(AdaptedArg::scalarInt(cb));
    ac.args.push_back(AdaptedArg::scalarInt(0));
    ac.args.push_back(AdaptedArg::scalarFloat(1e-5f));
    ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.globalSize[2] = 1; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// layernorm 3.6.0: (dim0,dim1, FLOAT* in, FLOAT* out, int inside, FLOAT* gamma, FLOAT* beta, float eps)
bool LayernormBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "layernorm_buf";
    const int inside = 64, outside = 4;
    const int n = inside * outside;
    std::vector<float> in; fillInput(in, n);
    std::vector<float> gamma(inside), beta(inside);
    fillInput(gamma, inside); fillInput(beta, inside);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer gammaBuf; gammaBuf.setFp32(gamma); gammaBuf.isOutput = false;
    AdaptedBuffer betaBuf; betaBuf.setFp32(beta); betaBuf.isOutput = false;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.buffers.push_back(gammaBuf); ac.buffers.push_back(betaBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::buffer(2));  // gamma
    ac.args.push_back(AdaptedArg::buffer(3));  // beta
    ac.args.push_back(AdaptedArg::scalarFloat(1e-5f));
    ac.globalSize[0] = outside; ac.globalSize[1] = 1; ac.dims = 2;
    ac.validatorInputA = in;
    return true;
}

// splitgelu 3.6.0: (dim0,dim1, FLOAT* in, FLOAT* out, int4 shape)
bool SplitgeluBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "splitgelu_buf";
    const int hw = 16, cb = 4;
    const int n = cb * hw * 4;
    std::vector<float> in; fillInput(in, n);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::int4(1, hw, 1, cb));
    ac.globalSize[0] = cb; ac.globalSize[1] = hw; ac.dims = 2;
    ac.validatorInputA = in;
    return true;
}

// topkv2 3.6.0: (dim0,dim1,dim2, DTYPE* outValue, int* outIndex, DTYPE* inValue, int rowSize, int k, int numRows)
bool Topkv2BufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "topkv2_buf";
    const int rowSize = 16, k = 4, numRows = 4;
    std::vector<float> in; fillInput(in, rowSize * numRows);
    AdaptedBuffer outValBuf; outValBuf.sizeBytes = k * numRows * sizeof(float); outValBuf.isOutput = true;
    AdaptedBuffer outIdxBuf; outIdxBuf.sizeBytes = k * numRows * sizeof(float); outIdxBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outValBuf); ac.buffers.push_back(outIdxBuf); ac.buffers.push_back(inBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));  // outValue
    ac.args.push_back(AdaptedArg::buffer(1));  // outIndex
    ac.args.push_back(AdaptedArg::buffer(2));  // inValue
    ac.args.push_back(AdaptedArg::scalarInt(rowSize));
    ac.args.push_back(AdaptedArg::scalarInt(k));
    ac.args.push_back(AdaptedArg::scalarInt(numRows));
    ac.globalSize[0] = numRows; ac.globalSize[1] = 1; ac.globalSize[2] = 1; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// strassen 3.6.0: (dim0,dim1, FLOAT* in0, int offsetC, int strideC, FLOAT* in1, FLOAT* out, int width, int height)
bool StrassenBinaryBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "binary_cfunction_buf";
    const int width = 4, height = 4;
    const int n = width * height * 4;
    std::vector<float> in0, in1; fillInput(in0, n); fillInput(in1, n);
    AdaptedBuffer in0Buf; in0Buf.setFp32(in0); in0Buf.isOutput = false;
    AdaptedBuffer in1Buf; in1Buf.setFp32(in1); in1Buf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(in0Buf); ac.buffers.push_back(in1Buf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));  // in0
    ac.args.push_back(AdaptedArg::scalarInt(0));   // offsetC
    ac.args.push_back(AdaptedArg::scalarInt(width)); // strideC
    ac.args.push_back(AdaptedArg::buffer(1));  // in1
    ac.args.push_back(AdaptedArg::buffer(2));  // out
    ac.args.push_back(AdaptedArg::scalarInt(width));
    ac.args.push_back(AdaptedArg::scalarInt(height));
    ac.globalSize[0] = width; ac.globalSize[1] = height; ac.dims = 2;
    ac.validatorInputA = in0;
    return true;
}

void registerNormConvOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            r.registerAdapter(std::unique_ptr<OpAdapter>(new PoolingBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new ScaleBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new GridSampleBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new GroupnormBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new LayernormBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new SplitgeluBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new Topkv2BufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new StrassenBinaryBufOp()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
