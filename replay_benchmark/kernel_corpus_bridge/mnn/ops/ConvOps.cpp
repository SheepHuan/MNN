#include "ConvOps.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

static void fillInput(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % 13);
}

// conv_2d_c4h1w1 1.2.0: (dim0,dim1, FLOAT* in, FLOAT* weight, FLOAT* bias, FLOAT* out,
//   int2 in_hw, int inChannel, int in_c_blocks, int2 out_hw, int2 filter_hw, int2 stride_hw, int2 pad_hw, int2 dilate_hw, int out_w_blocks, int out_c_blocks)
// conv_2d_1x1_local 3.6.0: (int out_w_blocks, FLOAT* in, FLOAT* kernel, FLOAT* bias, FLOAT* out,
//   int in_c_block, int batch, int out_h, int out_w, int out_c_block, int out_c_pack)
bool Conv2dBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int inC = spec.c(), outC = spec.outChannels(), ih = spec.h(), iw = spec.w();
    const int kh = spec.kernelSize(), kw = spec.kernelSize(), stride = spec.stride();
    const int inCB = (inC + 3) / 4, outCB = (outC + 3) / 4;
    const int outH = ih, outW = iw;
    const int outWB = (outW + 3) / 4;
    if (spec.tag == "1.2.0") {
        ac.entry = "conv_2d_c4h1w1";
        const int inN = inCB * ih * iw * 4;
        const int wN = outCB * inCB * kh * kw * 4;
        const int bN = outCB * 4;
        const int outN = outCB * outH * outW * 4;
        std::vector<float> in, w, b; fillInput(in, inN); fillInput(w, wN); fillInput(b, bN);
        AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
        AdaptedBuffer wBuf; wBuf.setFp32(w); wBuf.isOutput = false;
        AdaptedBuffer bBuf; bBuf.setFp32(b); bBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
        ac.buffers = {inBuf, wBuf, bBuf, outBuf};
        ac.args.push_back(AdaptedArg::sizeConst(0));  // dim0 = outWB
        ac.args.push_back(AdaptedArg::sizeConst(1));  // dim1 = outH * outCB
        ac.args.push_back(AdaptedArg::buffer(0));     // in
        ac.args.push_back(AdaptedArg::buffer(1));     // weight
        ac.args.push_back(AdaptedArg::buffer(2));     // bias
        ac.args.push_back(AdaptedArg::buffer(3));     // out
        ac.args.push_back(AdaptedArg::int2(ih, iw));   // in_hw
        ac.args.push_back(AdaptedArg::scalarInt(inC));
        ac.args.push_back(AdaptedArg::scalarInt(inCB));
        ac.args.push_back(AdaptedArg::int2(outH, outW));  // out_hw
        ac.args.push_back(AdaptedArg::int2(kh, kw));     // filter_hw
        ac.args.push_back(AdaptedArg::int2(stride, stride)); // stride
        ac.args.push_back(AdaptedArg::int2(1, 1));        // pad
        ac.args.push_back(AdaptedArg::int2(1, 1));        // dilate
        ac.args.push_back(AdaptedArg::scalarInt(outWB));
        ac.args.push_back(AdaptedArg::scalarInt(outCB));
        ac.globalSize[0] = outWB; ac.globalSize[1] = outH * outCB; ac.dims = 2;
        ac.validatorInputA = in;
    } else {
        ac.entry = "conv_2d_1x1_local";
        const int inN = inCB * ih * iw * 4;
        const int wN = outCB * inCB * 4;
        const int bN = outCB * 4;
        const int outN = outCB * outH * outW * 4;
        std::vector<float> in, w, b; fillInput(in, inN); fillInput(w, wN); fillInput(b, bN);
        AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
        AdaptedBuffer wBuf; wBuf.setFp32(w); wBuf.isOutput = false;
        AdaptedBuffer bBuf; bBuf.setFp32(b); bBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
        ac.buffers = {inBuf, wBuf, bBuf, outBuf};
        ac.args.push_back(AdaptedArg::scalarInt(outWB));  // out_w_blocks (first arg, not sizeConst)
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::buffer(3));
        ac.args.push_back(AdaptedArg::scalarInt(inCB));
        ac.args.push_back(AdaptedArg::scalarInt(1));       // batch
        ac.args.push_back(AdaptedArg::scalarInt(outH));
        ac.args.push_back(AdaptedArg::scalarInt(outW));
        ac.args.push_back(AdaptedArg::scalarInt(outCB));
        ac.args.push_back(AdaptedArg::scalarInt(4));       // out_c_pack
        ac.globalSize[0] = outWB; ac.globalSize[1] = outH; ac.globalSize[2] = outCB; ac.dims = 3;
        ac.validatorInputA = in;
    }
    return true;
}

// depthwise_conv2d_c4h1w4: (dim0,dim1, FLOAT* in, FLOAT* filter, FLOAT* bias, FLOAT* out,
//   int2 in_hw, int channel, int2 out_hw, int2 filter_hw, int2 pad_hw, int2 dilate_hw, int2 stride_hw, int out_w_blocks, int c_blocks)
bool DepthwiseConv2dBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "depthwise_conv2d_c4h1w4";
    const int channel = spec.c(), ih = spec.h(), iw = spec.w();
    const int kh = spec.kernelSize(), kw = spec.kernelSize(), stride = spec.stride();
    const int cb = (channel + 3) / 4;
    const int outH = ih, outW = iw;
    const int outWB = (outW + 3) / 4;
    const int inN = cb * ih * iw * 4;
    const int fN = cb * kh * kw * 4;
    const int bN = cb * 4;
    const int outN = cb * outH * outW * 4;
    std::vector<float> in, f, b; fillInput(in, inN); fillInput(f, fN); fillInput(b, bN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer fBuf; fBuf.setFp32(f); fBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(b); bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {inBuf, fBuf, bBuf, outBuf};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::int2(ih, iw));
    ac.args.push_back(AdaptedArg::scalarInt(channel));
    ac.args.push_back(AdaptedArg::int2(outH, outW));
    ac.args.push_back(AdaptedArg::int2(kh, kw));
    ac.args.push_back(AdaptedArg::int2(1, 1));  // pad
    ac.args.push_back(AdaptedArg::int2(1, 1));  // dilate
    ac.args.push_back(AdaptedArg::int2(stride, stride));
    ac.args.push_back(AdaptedArg::scalarInt(outWB));
    if (spec.tag == "1.2.0") {
        ac.args.push_back(AdaptedArg::scalarInt(cb));  // c_blocks
    } else {
        // 3.6.0 has batch instead of c_blocks at same position
        ac.args.push_back(AdaptedArg::scalarInt(1));  // batch
    }
    ac.globalSize[0] = outWB; ac.globalSize[1] = cb * outH; ac.dims = 2;
    ac.validatorInputA = in;
    return true;
}

// matmul_buf 1.2.0: (dim0,dim1, FLOAT* a, FLOAT* b, FLOAT* out, int channels, int channel_blocks, int width_blocks)
// matmul_buf 3.6.0: (dim0,dim1, FLOAT* a, FLOAT* b, FLOAT* out, int M, int N, int K)
bool MatmulBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "matmul_buf";
    const int M = spec.m(), N = spec.n(), K = spec.k();
    const int M4 = (M + 3) / 4, N4 = (N + 3) / 4, K4 = (K + 3) / 4;
    const int aN = M4 * K4 * 4, bN = K4 * N4 * 4, outN = M4 * N4 * 4;
    std::vector<float> a, b; fillInput(a, aN); fillInput(b, bN);
    AdaptedBuffer aBuf; aBuf.setFp32(a); aBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(b); bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {aBuf, bBuf, outBuf};
    ac.args.push_back(AdaptedArg::sizeConst(0));  // dim0 = N4
    ac.args.push_back(AdaptedArg::sizeConst(1));  // dim1 = M4
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    if (spec.tag == "1.2.0") {
        ac.args.push_back(AdaptedArg::scalarInt(K));
        ac.args.push_back(AdaptedArg::scalarInt(K4));
        ac.args.push_back(AdaptedArg::scalarInt(N4));
    } else {
        ac.args.push_back(AdaptedArg::scalarInt(M));
        ac.args.push_back(AdaptedArg::scalarInt(N));
        ac.args.push_back(AdaptedArg::scalarInt(K));
    }
    ac.globalSize[0] = N4; ac.globalSize[1] = M4; ac.dims = 2;
    ac.validatorInputA = a;
    return true;
}

// gemm_buf 1.2.0: (dim0,dim1, FLOAT* in0, FLOAT* in1, FLOAT* out, int width, int height, int srcChannelC4, int alpha2)
// gemm_buf 3.6.0: (dim0,dim1, int alignM, int alignK, int M, int K, int area, FLOAT* in, FLOAT* out)
bool GemmBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int M = spec.m(), K = spec.k(), area = spec.w() * spec.h();
    const int M4 = (M + 3) / 4, K4 = (K + 3) / 4;
    if (spec.tag == "1.2.0") {
        ac.entry = "gemm_buf";
        const int inN0 = M4 * K4 * 4, inN1 = K4 * area * 4, outN = M4 * area * 4;
        std::vector<float> in0, in1; fillInput(in0, inN0); fillInput(in1, inN1);
        AdaptedBuffer in0Buf; in0Buf.setFp32(in0); in0Buf.isOutput = false;
        AdaptedBuffer in1Buf; in1Buf.setFp32(in1); in1Buf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
        ac.buffers = {in0Buf, in1Buf, outBuf};
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::scalarInt(area));
        ac.args.push_back(AdaptedArg::scalarInt(M4));
        ac.args.push_back(AdaptedArg::scalarInt(K4));
        ac.args.push_back(AdaptedArg::scalarInt(0));  // alpha2
        ac.globalSize[0] = area; ac.globalSize[1] = M4; ac.dims = 2;
        ac.validatorInputA = in0;
    } else {
        ac.entry = "transpose_pad";
        const int inN = M * K, outN = M4 * K4 * 4;
        std::vector<float> in; fillInput(in, inN);
        AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
        ac.buffers = {inBuf, outBuf};
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::scalarInt(M4));
        ac.args.push_back(AdaptedArg::scalarInt(K4));
        ac.args.push_back(AdaptedArg::scalarInt(M));
        ac.args.push_back(AdaptedArg::scalarInt(K));
        ac.args.push_back(AdaptedArg::scalarInt(M * K));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.globalSize[0] = M4; ac.globalSize[1] = K4; ac.dims = 2;
        ac.validatorInputA = in;
    }
    return true;
}

// winogradTransform_buf 1.2.0: winoTransSrcBuf2_3_1 (dim0,dim1, FLOAT* in, FLOAT* out, int unitW, int unitH, int padX, int padY, int srcW, int srcH, int srcChannelC4, int batchOffset)
// winogradTransform_buf 3.6.0: winoTransWeightBuf2_3_1 (dim0,dim1, FLOAT* in, FLOAT* out, int srcChannel, int dstChannel, int srcChannelPad, int dstChannelPad)
bool WinogradTransformBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int cb = spec.c(), hw = spec.w();
    if (spec.tag == "1.2.0") {
        ac.entry = "winoTransSrcBuf2_3_1";
        const int unitW = (hw + 2) / 3, unitH = (hw + 2) / 3;
        const int inN = cb * hw * hw * 4;
        const int outN = cb * unitW * unitH * 4;
        std::vector<float> in; fillInput(in, inN);
        AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
        ac.buffers = {inBuf, outBuf};
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(unitW));
        ac.args.push_back(AdaptedArg::scalarInt(unitH));
        ac.args.push_back(AdaptedArg::scalarInt(0));  // padX
        ac.args.push_back(AdaptedArg::scalarInt(0));  // padY
        ac.args.push_back(AdaptedArg::scalarInt(hw));  // srcW
        ac.args.push_back(AdaptedArg::scalarInt(hw));  // srcH
        ac.args.push_back(AdaptedArg::scalarInt(cb));  // srcChannelC4
        ac.args.push_back(AdaptedArg::scalarInt(0));   // batchOffset
        ac.globalSize[0] = unitW; ac.globalSize[1] = unitH * cb; ac.dims = 2;
        ac.validatorInputA = in;
    } else {
        ac.entry = "winoTransWeightBuf2_3_1";
        const int srcC = 4, dstC = 4;
        const int srcCP = (srcC + 3) / 4, dstCP = (dstC + 3) / 4;
        const int inN = dstCP * srcC * 4, outN = dstCP * srcCP * 4;
        std::vector<float> in; fillInput(in, inN);
        AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
        ac.buffers = {inBuf, outBuf};
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(srcC));
        ac.args.push_back(AdaptedArg::scalarInt(dstC));
        ac.args.push_back(AdaptedArg::scalarInt(srcCP));
        ac.args.push_back(AdaptedArg::scalarInt(dstCP));
        ac.globalSize[0] = srcCP; ac.globalSize[1] = dstCP; ac.dims = 2;
        ac.validatorInputA = in;
    }
    return true;
}

// interp_buf: (dim0,dim1,dim2, FLOAT* in, FLOAT* out, float h_scale, float w_scale, float h_offset, float w_offset, int inH, int inW, int outH, int outW, int channelBlocks)
bool InterpBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "nearest_buf";
    const int inH = spec.h(), inW = spec.w(), cb = spec.c();
    const int outH = spec.intParam("out_h", inH * 2), outW = spec.intParam("out_w", inW * 2);
    const int inN = cb * inH * inW * 4;
    const int outN = cb * outH * outW * 4;
    std::vector<float> in; fillInput(in, inN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {inBuf, outBuf};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarFloat(2.0f));   // height_scale
    ac.args.push_back(AdaptedArg::scalarFloat(2.0f));   // width_scale
    ac.args.push_back(AdaptedArg::scalarFloat(0.0f));   // height_offset
    ac.args.push_back(AdaptedArg::scalarFloat(0.0f));   // width_offset
    ac.args.push_back(AdaptedArg::scalarInt(inH));
    ac.args.push_back(AdaptedArg::scalarInt(inW));
    ac.args.push_back(AdaptedArg::scalarInt(outH));
    ac.args.push_back(AdaptedArg::scalarInt(outW));
    ac.args.push_back(AdaptedArg::scalarInt(cb));
    ac.globalSize[0] = cb; ac.globalSize[1] = outW; ac.globalSize[2] = outH; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// input_transe 3.6.0: (dim0,dim1,dim2, FLOAT* in, FLOAT* out, int inW, int inH, int inC, int batch, int channel_blocks, int pad_left, int pad_right)
bool InputTranseBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "conv_transe_c4_c1";
    const int inW = spec.w(), inH = spec.h(), inC = spec.c(), batch = spec.batch();
    const int cb = (inC + 3) / 4;
    const int inN = cb * inH * inW * 4;
    const int outN = inC * inH * inW;
    std::vector<float> in; fillInput(in, inN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {inBuf, outBuf};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(inW));
    ac.args.push_back(AdaptedArg::scalarInt(inH));
    ac.args.push_back(AdaptedArg::scalarInt(inC));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(cb));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // pad_left
    ac.args.push_back(AdaptedArg::scalarInt(0));  // pad_right
    ac.globalSize[0] = inW; ac.globalSize[1] = inH; ac.globalSize[2] = cb; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// buffer_convert_buf 1.2.0: nhwc_buffer_to_nc4hw4_buffer (dim0,dim1, FLOAT* in, int height, int width, int channels, FLOAT* out)
// buffer_convert_buf 3.6.0: buffer_convert_to_buffer (dim0,dim1,dim2, INPUT* in, int4 shape, OUTPUT* out)
bool BufferConvertBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int hw = spec.w(), channel = spec.c();
    const int cb = (channel + 3) / 4;
    if (spec.tag == "1.2.0") {
        ac.entry = "nhwc_buffer_to_nc4hw4_buffer";
        const int inN = hw * hw * channel;
        const int outN = cb * hw * hw * 4;
        std::vector<float> in; fillInput(in, inN);
        AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
        ac.buffers = {inBuf, outBuf};
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::scalarInt(hw));
        ac.args.push_back(AdaptedArg::scalarInt(hw));
        ac.args.push_back(AdaptedArg::scalarInt(channel));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.globalSize[0] = hw; ac.globalSize[1] = hw * cb; ac.dims = 2;
        ac.validatorInputA = in;
    } else {
        ac.entry = "buffer_convert_to_buffer";
        const int n = cb * hw * hw * 4;
        std::vector<float> in; fillInput(in, n);
        AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
        ac.buffers = {inBuf, outBuf};
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::sizeConst(2));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::int4(1, cb, hw, hw));  // shape NC4HW
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.globalSize[0] = hw; ac.globalSize[1] = cb; ac.globalSize[2] = 1; ac.dims = 3;
        ac.validatorInputA = in;
    }
    return true;
}

void registerConvOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            r.registerAdapter(std::unique_ptr<OpAdapter>(new Conv2dBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new DepthwiseConv2dBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new MatmulBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new GemmBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new WinogradTransformBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new InterpBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new InputTranseBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new BufferConvertBufOp()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
