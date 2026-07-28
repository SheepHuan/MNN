// convdw.cu - CONV_DW (3.6.0) + ConvDwConstBuffer_120 struct + CONV_DW_120 (1.2.0)
//   + CONV_DW_204 (2.0.4) kernels + shims
//   source/backend/cuda/execution/ConvDepthWiseExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// Conv DepthWise: source/backend/cuda/execution/ConvDepthWiseExecution.cu
// Simplified fp32 re-implementation (processes 2 channels per thread like original).
// ============================================================================
template <typename T>
__global__ void CONV_DW(const T* input, const half* kernel, const half* bias, T* output,
                        float maxV, float minV, int iw, int ih, int c, int c_p,
                        int ow, int oh, int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                        int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total / 2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);
        int oz = oz_2 << 1;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        float color1 = bias[oz + 1];
        int fxSta = max(0, (int)ceil(-(float)ix / dw));
        int fySta = max(0, (int)ceil(-(float)iy / dh));
        int fxEnd = min(kw, (int)ceil((float)(iw - ix) / dw));
        int fyEnd = min(kh, (int)ceil((float)(ih - iy) / dh));
        for (int fy = fySta; fy < fyEnd; ++fy) {
            for (int fx = fxSta; fx < fxEnd; ++fx) {
                int currentX = ix + fx * dw;
                int currentY = iy + fy * dh;
                const half* k0 = kernel + (oz * kh + fy) * kw + fx;
                const half* k1 = kernel + ((oz + 1) * kh + fy) * kw + fx;
                const T* inp = (const T*)(input + (ob * ih * iw + currentY * iw + currentX) * c_p + oz);
                float v0 = (float)inp[0];
                float v1 = (float)inp[1];
                color0 += v0 * (float)k0[0];
                color1 += v1 * (float)k1[0];
            }
        }
        color0 = max(minV, min(maxV, color0));
        color1 = max(minV, min(maxV, color1));
        T* dst0 = (T*)(output + (ob * oh * ow + oy * ow + ox) * c_p + oz);
        dst0[0] = (T)color0;
        dst0[1] = (T)color1;
    }
}

// ============================================================================
// Conv DepthWise 1.2.0: constBuffer struct param, NCHW layout, float kernel/bias
// (source/backend/cuda/execution/ConvDepthWiseExecution.cu @ tag 1.2.0)
// ============================================================================
struct ConvDwConstBuffer_120 {
    int pad[2];
    int kernelSize[2];
    int stride[2];
    int dilate[2];
    int inputSize[2];
    int outputSize[2];
    int channel;
    int subChannel;
    int total;
    int activationType;
};
__global__ void CONV_DW_120(const float* input, const float* kernel, const float* bias,
                             float* output, const ConvDwConstBuffer_120* u) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)u->total; i += blockDim.x * gridDim.x) {
        int iw = u->inputSize[0], ih = u->inputSize[1];
        int ow = u->outputSize[0], oh = u->outputSize[1];
        int kw = u->kernelSize[0], kh = u->kernelSize[1];
        int dw = u->dilate[0], dh = u->dilate[1];
        int sw = u->stride[0], sh = u->stride[1];
        int pw = u->pad[0], ph = u->pad[1];
        int acttype = u->activationType;
        int oz = i / (ow * oh);
        int tmp = i % (ow * oh);
        int oy = tmp / ow;
        int ox = tmp % ow;
        int kz = oz % u->subChannel;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color = (bias != nullptr) ? bias[kz] : 0.0f;
        for (int fy = 0; fy < kh; ++fy) {
            int sy = fy * dh + iy;
            if (sy >= ih || sy < 0) continue;
            for (int fx = 0; fx < kw; ++fx) {
                int sx = fx * dw + ix;
                if (sx >= iw || sx < 0) continue;
                float inputValue = input[sx + sy * iw + oz * iw * ih];
                float k = kernel[fx + fy * kw + kz * kw * kh];
                color += k * inputValue;
            }
        }
        color = (acttype == 1) ? max(0.0f, color) : (acttype == 2 ? (min(max(0.0f, color), 6.0f)) : color);
        output[ox + oy * ow + oz * ow * oh] = color;
    }
}

// ============================================================================
// Conv DepthWise 2.0.4: c_p-based indexing (signature same as 2.2.3 but
// single-channel-per-thread loop, no DivModFast). Reproduced from
// source/backend/cuda/execution/ConvDepthWiseExecution.cu @ tag 2.0.4.
// ============================================================================
template <typename T>
__global__ void CONV_DW_204(const T* input, const half* kernel, const half* bias, T* output,
                             float maxV, float minV, int iw, int ih, int c, int c_p,
                             int ow, int oh, int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                             int total) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)total; index += blockDim.x * gridDim.x) {
        int oz = index % c_p;
        int tmp = index / c_p;
        int ox = tmp % ow;
        int oy = tmp / ow;
        int ob = oy / oh;
        int iy = oy % oh;
        int ix = ox * sw - pw;
        int iyv = iy * sh - ph;
        float color = (float)bias[oz];
        int fxSta = max(0, (int)ceil(-(float)ix / dw));
        int fySta = max(0, (int)ceil(-(float)iyv / dh));
        int fxEnd = min(kw, (int)ceil((float)(iw - ix) / dw));
        int fyEnd = min(kh, (int)ceil((float)(ih - iyv) / dh));
        for (int fy = fySta; fy < fyEnd; ++fy) {
            for (int fx = fxSta; fx < fxEnd; ++fx) {
                int currentX = ix + fx * dw;
                int currentY = iyv + fy * dh;
                const half* k = kernel + (oz * kh + fy) * kw + fx;
                const T* inp = (const T*)(input + (ob * ih * iw + currentY * iw + currentX) * c_p + oz);
                color += (float)inp[0] * (float)k[0];
            }
        }
        color = max(minV, min(maxV, color));
        T* dst = (T*)(output + (ob * oh * ow + oy * ow + ox) * c_p + oz);
        dst[0] = (T)color;
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Conv DepthWise fp32 (3.6.0 / 2.2.3+) ----
void mnn_corpus_conv_dw_fp32(const float* input, const half* kernel, const half* bias, float* output,
                             float maxV, float minV, int iw, int ih, int c, int c_p,
                             int ow, int oh, int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                             int total, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(c_p / 2);
    MNN::Corpus::DivModFast d_ow(ow);
    MNN::Corpus::DivModFast d_oh(1);
    MNN::Corpus::CONV_DW<float><<<grid, block, 0, stream>>>(
        input, kernel, bias, output, maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph,
        total, d_oc, d_ow, d_oh);
}

// ---- Conv DepthWise 1.2.0: constBuffer struct, NCHW, float kernel/bias ----
void mnn_corpus_conv_dw_120_fp32(const float* input, const float* kernel, const float* bias, float* output,
                                  const MNN::Corpus::ConvDwConstBuffer_120* uConst, int grid, int block,
                                  cudaStream_t stream) {
    MNN::Corpus::CONV_DW_120<<<grid, block, 0, stream>>>(input, kernel, bias, output, uConst);
}

// ---- Conv DepthWise 2.0.4: c_p indexing, single-channel-per-thread ----
void mnn_corpus_conv_dw_204_fp32(const float* input, const half* kernel, const half* bias, float* output,
                                  float maxV, float minV, int iw, int ih, int c, int c_p,
                                  int ow, int oh, int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                                  int total, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::CONV_DW_204<float><<<grid, block, 0, stream>>>(
        input, kernel, bias, output, maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total);
}

} // extern "C"
