#include "ComplexOps.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

static void fillInput(std::vector<float>& v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % 13);
}

// attention: rearrange_qkv(dim0,dim1,dim2, in_q, in_k, in_v, out_q, out_k, out_v, int4 tile, int4 shape, int4 param, int maxLenKV)
// No SAVE_KV
bool AttentionBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "rearrange_qkv";
    const int seqLenQ = 16, seqLenKV = 16, headNum = 4, headDim = 16, group = 2, batch = 1;
    const int tileQ = 16, tileKV = 16, tileHDK = 16, tileHDN = 16;
    const int inN = batch * headNum * headDim * seqLenQ * 4;
    const int outN = batch * headNum * headDim * seqLenQ * 4;
    std::vector<float> inQ, inK, inV;
    fillInput(inQ, inN); fillInput(inK, inN); fillInput(inV, inN);
    AdaptedBuffer bq; bq.setFp32(inQ); bq.isOutput = false;
    AdaptedBuffer bk; bk.setFp32(inK); bk.isOutput = false;
    AdaptedBuffer bv; bv.setFp32(inV); bv.isOutput = false;
    AdaptedBuffer oq; oq.sizeBytes = outN * sizeof(float); oq.isOutput = true;
    AdaptedBuffer ok; ok.sizeBytes = outN * sizeof(float); ok.isOutput = true;
    AdaptedBuffer ov; ov.sizeBytes = outN * sizeof(float); ov.isOutput = true;
    ac.buffers = {bq, bk, bv, oq, ok, ov};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    for (int i = 0; i < 6; ++i) ac.args.push_back(AdaptedArg::buffer(i));
    ac.args.push_back(AdaptedArg::int4(tileQ, tileKV, tileHDK, tileHDN));  // tile
    ac.args.push_back(AdaptedArg::int4(seqLenQ, seqLenKV, headNum, headDim));  // shape
    ac.args.push_back(AdaptedArg::int4(group, batch, 0, 0));  // param
    ac.args.push_back(AdaptedArg::scalarInt(seqLenKV));  // maxLenKV
    ac.globalSize[0] = tileQ; ac.globalSize[1] = headNum; ac.globalSize[2] = batch; ac.dims = 3;
    ac.validatorInputA = inQ;
    return true;
}

// self_attention: split_transpose_qkv(dim0,dim1,dim2, in, out_q, out_k, out_v, int×7)
bool SelfAttentionBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "split_transpose_qkv";
    const int seqLen = 16, headNum = 4, headDim = 16, batch = 1;
    const int inN = batch * (seqLen / 4) * headNum * 3 * headDim * 4;
    const int outN = batch * headNum * headDim * seqLen * 4;
    std::vector<float> in; fillInput(in, inN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer oq; oq.sizeBytes = outN * sizeof(float); oq.isOutput = true;
    AdaptedBuffer ok; ok.sizeBytes = outN * sizeof(float); ok.isOutput = true;
    AdaptedBuffer ov; ov.sizeBytes = outN * sizeof(float); ov.isOutput = true;
    ac.buffers = {inBuf, oq, ok, ov};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarInt(seqLen));     // seq_len_pack_mn
    ac.args.push_back(AdaptedArg::scalarInt(seqLen));     // seq_len_piece
    ac.args.push_back(AdaptedArg::scalarInt(headDim));    // head_dim_pack_mn
    ac.args.push_back(AdaptedArg::scalarInt(headDim));    // head_dim_pack_k
    ac.args.push_back(AdaptedArg::scalarInt(seqLen));     // seq_len
    ac.args.push_back(AdaptedArg::scalarInt(headNum));    // head_num
    ac.args.push_back(AdaptedArg::scalarInt(headDim));    // head_dim
    ac.args.push_back(AdaptedArg::scalarInt(batch));      // batch
    ac.args.push_back(AdaptedArg::scalarInt(0));          // seq_index
    ac.globalSize[0] = seqLen; ac.globalSize[1] = 1; ac.globalSize[2] = 1; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// gemm_conv1x1: inverse_quant_weight(dim0,dim1, FLOAT* dequantScaleOffset, FLOAT* out, int×4, float coef)
// Use buffer path (no USE_IMAGE, QUANT_BIT=8 → char* weight, but we use FLOAT* via -DQUANT_BIT=0)
bool GemmConv1x1BufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "inverse_quant_weight";
    const int inC = 4, outC = 4, blockDim = 4;
    const int inC4 = (inC + 3) / 4, outC4 = (outC + 3) / 4;
    const int wN = outC * inC;
    const int dN = outC4 * 4 * 2;
    const int outN = outC4 * inC4 * 4;
    std::vector<float> w, d; fillInput(w, wN); fillInput(d, dN);
    AdaptedBuffer wBuf; wBuf.setFp32(w); wBuf.isOutput = false;
    AdaptedBuffer dBuf; dBuf.setFp32(d); dBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {wBuf, dBuf, outBuf};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));   // weight (as FLOAT*)
    ac.args.push_back(AdaptedArg::buffer(1));   // dequantScaleOffset
    ac.args.push_back(AdaptedArg::buffer(2));   // output
    ac.args.push_back(AdaptedArg::scalarInt(inC));
    ac.args.push_back(AdaptedArg::scalarInt(inC4));
    ac.args.push_back(AdaptedArg::scalarInt(outC));
    ac.args.push_back(AdaptedArg::scalarInt(outC4));
    ac.args.push_back(AdaptedArg::scalarInt(blockDim));
    ac.args.push_back(AdaptedArg::scalarFloat(1.0f));  // coef
    ac.globalSize[0] = outC4; ac.globalSize[1] = inC4; ac.dims = 2;
    ac.validatorInputA = w;
    return true;
}

// gemv_conv1x1: gemv_conv_c8_buf(dim0,dim1,dim2, FLOAT* in, FLOAT* weight, FLOAT* dequantScaleOffset, FLOAT* bias, FLOAT* out, int×6, float coef)
bool GemvConv1x1BufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "gemv_conv_c8_buf";
    const int inC = 4, outC = 4, blockDim = 4;
    const int inC4 = (inC + 3) / 4, outC4 = (outC + 3) / 4;
    const int inN = inC4 * 4;
    const int wN = outC * inC;
    const int dN = outC4 * 4 * 2;
    const int bN = outC4 * 4;
    const int outN = outC4 * 4;
    std::vector<float> in, w, d, b;
    fillInput(in, inN); fillInput(w, wN); fillInput(d, dN); fillInput(b, bN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer wBuf; wBuf.setFp32(w); wBuf.isOutput = false;
    AdaptedBuffer dBuf; dBuf.setFp32(d); dBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(b); bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {inBuf, wBuf, dBuf, bBuf, outBuf};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));  // input
    ac.args.push_back(AdaptedArg::buffer(1));  // weight
    ac.args.push_back(AdaptedArg::buffer(2));  // dequantScaleOffset
    ac.args.push_back(AdaptedArg::buffer(3));  // bias
    ac.args.push_back(AdaptedArg::buffer(4));  // output
    ac.args.push_back(AdaptedArg::scalarInt(outC));
    ac.args.push_back(AdaptedArg::scalarInt(inC));
    ac.args.push_back(AdaptedArg::scalarInt(outC4));
    ac.args.push_back(AdaptedArg::scalarInt(inC4));
    ac.args.push_back(AdaptedArg::scalarInt(inC));
    ac.args.push_back(AdaptedArg::scalarInt(1));  // blockNum
    ac.args.push_back(AdaptedArg::scalarInt(blockDim));
    ac.args.push_back(AdaptedArg::scalarFloat(1.0f));
    ac.globalSize[0] = outC4; ac.globalSize[1] = 1; ac.globalSize[2] = 1; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// grid_sample: nearest_buf has enum BorderMode param — use int
bool GridSampleBufOp2::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "nearest_buf";
    const int inH = spec.h(), inW = spec.w(), batch = spec.batch(), channel = spec.c();
    const int cb = (channel + 3) / 4;
    const int outH = spec.intParam("out_h", inH / 2), outW = spec.intParam("out_w", inW / 2);
    const int inN = batch * cb * inH * inW * 4;
    const int gridN = batch * outH * outW * 2 * 4;
    const int outN = batch * cb * outH * outW * 4;
    std::vector<float> in, grid; fillInput(in, inN); fillInput(grid, gridN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer gridBuf; gridBuf.setFp32(grid); gridBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {inBuf, gridBuf, outBuf};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(inH));
    ac.args.push_back(AdaptedArg::scalarInt(inW));
    ac.args.push_back(AdaptedArg::scalarInt(outH));
    ac.args.push_back(AdaptedArg::scalarInt(outW));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // paddingMode (enum int)
    ac.args.push_back(AdaptedArg::scalarInt(0));  // alignCorners
    ac.globalSize[0] = cb; ac.globalSize[1] = outW; ac.globalSize[2] = outH * batch; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

// buffer_convert_subgroup: nhwc_buffer_to_nc16hw16_buffer(dim0,dim1, INPUT* in, int height, int width, int channels, OUTPUT* out, int pad_l, int pad_r, int out_pad_l, int out_pad_r)
bool BufferConvertSubgroupBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "nhwc_buffer_to_nc16hw16_buffer";
    const int hw = spec.w() * spec.h(), channel = spec.c();
    const int cb16 = (channel + 15) / 16;
    const int inN = hw * channel;
    const int outN = cb16 * 16 * hw;
    std::vector<float> in; fillInput(in, inN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {inBuf, outBuf};
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::scalarInt(spec.h()));
    ac.args.push_back(AdaptedArg::scalarInt(spec.w()));
    ac.args.push_back(AdaptedArg::scalarInt(channel));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(0));  // input_pad_left
    ac.args.push_back(AdaptedArg::scalarInt(0));  // input_pad_right
    ac.args.push_back(AdaptedArg::scalarInt(0));  // output_pad_left
    ac.args.push_back(AdaptedArg::scalarInt(0));  // output_pad_right
    ac.globalSize[0] = spec.w(); ac.globalSize[1] = spec.h() * cb16; ac.dims = 2;
    ac.validatorInputA = in;
    return true;
}

// conv_2d_int: conv_2d_int_c4h1w1(dim0,dim1, FLOAT* in, char* weight, FLOAT* dequant, FLOAT* bias, FLOAT* out, int2 in_hw, int inC, int inCB, int batch, int2 out_hw, int2 filter_hw, int2 stride, int2 pad, int2 dilate, int outWB, int outCB, int outHB, int blockDim, float coef)
bool Conv2dIntBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "conv_2d_int_c4h1w1";
    const int inC = spec.c(), outC = spec.outChannels(), ih = spec.h(), iw = spec.w();
    const int kh = spec.kernelSize(), stride = spec.stride();
    const int inCB = (inC + 3) / 4, outCB = (outC + 3) / 4;
    const int outH = ih, outW = iw, outWB = (outW + 3) / 4;
    const int inN = inCB * ih * iw * 4;
    const int wN = outCB * inCB * kh * kh * 4;
    const int dN = outCB * 4 * 2;
    const int bN = outCB * 4;
    const int outN = outCB * outH * outW * 4;
    std::vector<float> in, w, d, b; fillInput(in, inN); fillInput(w, wN); fillInput(d, dN); fillInput(b, bN);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer wBuf; wBuf.setFp32(w); wBuf.isOutput = false;
    AdaptedBuffer dBuf; dBuf.setFp32(d); dBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(b); bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {inBuf, wBuf, dBuf, bBuf, outBuf};
    ac.compileMacros.push_back("-DQUANT_BIT=8");
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::buffer(4));
    ac.args.push_back(AdaptedArg::int2(ih, iw));
    ac.args.push_back(AdaptedArg::scalarInt(inC));
    ac.args.push_back(AdaptedArg::scalarInt(inCB));
    ac.args.push_back(AdaptedArg::scalarInt(1));   // batch
    ac.args.push_back(AdaptedArg::int2(outH, outW));
    ac.args.push_back(AdaptedArg::int2(kh, kh));
    ac.args.push_back(AdaptedArg::int2(stride, stride));
    ac.args.push_back(AdaptedArg::int2(1, 1));    // pad
    ac.args.push_back(AdaptedArg::int2(1, 1));    // dilate
    ac.args.push_back(AdaptedArg::scalarInt(outWB));
    ac.args.push_back(AdaptedArg::scalarInt(outCB));
    ac.args.push_back(AdaptedArg::scalarInt(outH));
    ac.args.push_back(AdaptedArg::scalarInt(4));   // blockDim
    ac.args.push_back(AdaptedArg::scalarFloat(1.0f));  // coef
    ac.globalSize[0] = outWB; ac.globalSize[1] = outH * outCB; ac.dims = 2;
    ac.validatorInputA = in;
    return true;
}

// linear_attention: linear_attn_conv_silu(dim0, qkv, conv_state, conv_weight, conv_out, batch, conv_dim, seq_len, kernel_size, conv_state_size)
bool LinearAttentionBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "linear_attn_conv_silu";
    const int batch = spec.batch(), convDim = 16, seqLen = 16, ks = spec.kernelSize(), convStateSize = 32;
    const int qkvN = batch * convDim * seqLen * 4;
    const int csN = batch * convDim * convStateSize * 4;
    const int cwN = convDim * convDim * ks * 4;
    const int outN = batch * convDim * seqLen * 4;
    std::vector<float> qkv, cs, cw; fillInput(qkv, qkvN); fillInput(cs, csN); fillInput(cw, cwN);
    AdaptedBuffer qkvBuf; qkvBuf.setFp32(qkv); qkvBuf.isOutput = false;
    AdaptedBuffer csBuf; csBuf.setFp32(cs); csBuf.isOutput = false;
    AdaptedBuffer cwBuf; cwBuf.setFp32(cw); cwBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers = {qkvBuf, csBuf, cwBuf, outBuf};
    ac.compileMacros.push_back("-DLOCAL_SIZE=64");
    ac.compileMacros.push_back("-DK_SIZE=32");
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(convDim));
    ac.args.push_back(AdaptedArg::scalarInt(seqLen));
    ac.args.push_back(AdaptedArg::scalarInt(ks));
    ac.args.push_back(AdaptedArg::scalarInt(convStateSize));
    ac.globalSize[0] = batch * seqLen; ac.globalSize[1] = 1; ac.dims = 2;
    ac.validatorInputA = qkv;
    return true;
}

// matmul_local: matmul_local_buf(int M, int N, int K, FLOAT* A, FLOAT* B, FLOAT* C)
// No BIAS, no LOW_BIT_WEIGHT
bool MatmulLocalBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "matmul_local_buf";
    const int M = spec.m(), N = spec.n(), K = spec.k();
    const int M4 = (M + 3) / 4, N4 = (N + 3) / 4, K4 = (K + 3) / 4;
    const int aN = M4 * K4 * 4, bN = K4 * N4 * 4, cN = M4 * N4 * 4;
    std::vector<float> a, b; fillInput(a, aN); fillInput(b, bN);
    AdaptedBuffer aBuf; aBuf.setFp32(a); aBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(b); bBuf.isOutput = false;
    AdaptedBuffer cBuf; cBuf.sizeBytes = cN * sizeof(float); cBuf.isOutput = true;
    ac.buffers = {aBuf, bBuf, cBuf};
    ac.compileMacros.push_back("-DOPWM=64");
    ac.compileMacros.push_back("-DOPWN=128");
    ac.compileMacros.push_back("-DCPWK=8");
    ac.compileMacros.push_back("-DOPTM=4");
    ac.compileMacros.push_back("-DOPTN=8");
    ac.args.push_back(AdaptedArg::scalarInt(M));
    ac.args.push_back(AdaptedArg::scalarInt(N));
    ac.args.push_back(AdaptedArg::scalarInt(K));
    ac.args.push_back(AdaptedArg::buffer(0));      // A
    ac.args.push_back(AdaptedArg::buffer(1));      // B
    ac.args.push_back(AdaptedArg::buffer(2));      // C
    ac.globalSize[0] = N4; ac.globalSize[1] = M4; ac.dims = 2;
    ac.validatorInputA = a;
    return true;
}

// scale_nobias: scale(dim0,dim1,dim2, image2d_t input, image2d_t scale, image2d_t output)
// Uses image2d_t — unsupported on buffer-only runner
bool ScaleNobiasOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    // scale_nobias uses image2d_t — runner does not support image args.
    // Return false to let FallbackAdapter handle (will compile but may fail dispatch).
    return false;
}

void registerComplexOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            r.registerAdapter(std::unique_ptr<OpAdapter>(new AttentionBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new SelfAttentionBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new GemmConv1x1BufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new GemvConv1x1BufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new GridSampleBufOp2()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new BufferConvertSubgroupBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new Conv2dIntBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new LinearAttentionBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new MatmulLocalBufOp()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new ScaleNobiasOp()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
