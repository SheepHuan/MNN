// convdw_extra.cu - ConvDepthWise fp16/half2 kernel variants for replay benchmark corpus
//   Kernels copied verbatim from source/backend/cuda/execution/ConvDepthWiseExecution.cu:
//     - CONV_DW_OPT             (fp32 input + half weight/bias)
//     - CONV_DW_HALF2_OPT       (all half2)
//     - CONV_DW3x3_HALF2_OPT    (all half2, 3x3 specialization)
//     - CONV_DW_MULTI_WIDTH4<T> (template T, 1D multi-width-4)
//     - CONV_DW_MULTI_WIDTH_CHANNEL (fp32 input + half weight/bias)
#include "corpus_common.cuh"
#include <cuda_fp16.h>

namespace MNN {
namespace Corpus {

static const int PACK_NUMBER = 8;

__global__ void CONV_DW_HALF2_OPT(const half2* input,
    const half2* kernel,
    const half2* bias,
    half2 *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        half2 color = bias[oz];

        int fxSta = max(0, -ix);
        int fySta = max(0, -iy);
        int fxEnd = min(kw, iw - ix);
        int fyEnd = min(kh, ih - iy);
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx + ix;
                int src_offset = ((ob * ih + sy) * iw + sx) * c_p + oz;
                half2 inp = input[src_offset];
                half2 ker = kernel[(fy * kw + fx) * c_p + oz];

                color = __hfma2(inp, ker, color);
            }
        }
        color.x = max(color.x, minV);
        color.x = min(color.x, maxV);

        color.y = max(color.y, minV);
        color.y = min(color.y, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;
        output[dst_offset] = color;
    }
}

__global__ void CONV_DW3x3_HALF2_OPT(const half2* input,
    const half2* kernel,
    const half2* bias,
    half2 *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/4; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox_2, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox_2);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ox = ox_2 << 1;
        int ix = ox - 1;
        int iy = oy - 1;
        half2 color0 = bias[oz];
        half2 color1 = color0;

        half2 zero;
        zero.x = (half)0.0;
        zero.y = (half)0.0;

        half2 inp[12];
        half2 ker[3][3];
        for(int j=0; j<3; j++) {
            if(iy < 0 && j==0) {
                for(int i=0; i<4; i++) {
                    inp[i] = zero;
                }
                continue;
            }
            if(iy+2 > ih-1 && j==2) {
                for(int i=0; i<4; i++) {
                    inp[8+i] = zero;
                }
                continue;
            }

            for(int i=0; i<4; i++) {
                if(ix < 0 && i==0) {
                    for(int j=0; j<3; j++) {
                        inp[4*j+0] = zero;
                    }
                    continue;
                }
                if(ix+3 > iw-1 && i==3) {
                    for(int j=0; j<3; j++) {
                        inp[4*j+3] = zero;
                    }
                    continue;
                }
                int src_offset = ((ob * ih + iy+j) * iw + ix+i) * c_p + oz;
                inp[4*j+i] = input[src_offset];
            }
        }

        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                ker[j][i] = kernel[(j * 3 + i) * c_p + oz];
            }
        }

        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                color0 = __hfma2(inp[4*j+i], ker[j][i], color0);
                color1 = __hfma2(inp[4*j+i+1], ker[j][i], color1);
            }
        }

        color0.x = max(color0.x, minV);
        color0.x = min(color0.x, maxV);
        color0.y = max(color0.y, minV);
        color0.y = min(color0.y, maxV);

        color1.x = max(color1.x, minV);
        color1.x = min(color1.x, maxV);
        color1.y = max(color1.y, minV);
        color1.y = min(color1.y, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;
        output[dst_offset] = color0;
        output[dst_offset+c_p] = color1;
    }
}

__global__ void CONV_DW_OPT(const float* input, const half* kernel, const half* bias, float *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
    ) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total / 2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2 << 1;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        float color1 = bias[oz+1];

        int fxSta = max(0, -ix);
        int fySta = max(0, -iy);
        int fxEnd = min(kw, iw - ix);
        int fyEnd = min(kh, ih - iy);
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx + ix;
                int src_offset = ((ob * ih + sy) * iw + sx) * c_p + oz;
                float inp0 = input[src_offset];
                float inp1 = input[src_offset+1];

                float ker0 = kernel[(fy * kw + fx) * c_p + oz];
                float ker1 = kernel[(fy * kw + fx) * c_p + oz + 1];

                color0 = color0 + inp0 * ker0;
                color1 = color1 + inp1 * ker1;
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        color1 = max(color1, minV);
        color1 = min(color1, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
        output[dst_offset+1] = color1;
    }
}

template<typename T>
__global__ void CONV_DW_MULTI_WIDTH4(const T* input, const half* kernel, const half* bias, T *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow_4,
    DivModFast d_oh
    ) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total / 4; index += blockDim.x * gridDim.x) {
        int oz, tmp2, oy, ox_4, tmp1, ob;
        d_oc.divmod(index, tmp1, oz);
        d_ow_4.divmod(tmp1, tmp2, ox_4);
        d_oh.divmod(tmp2, ob, oy);

        float color0 = bias[oz];
        float color1 = color0;
        float color2 = color0;
        float color3 = color0;

        // Parallel pipelining read and calculate
        float src;
        float filter0, filter1, filter2, filter3;
        int src_offset = ((ob * ih + oy) * iw + (ox_4 << 2)) * c_p + oz;
        int filter_offset = 0 * c_p + oz;

        src    = input[src_offset + 0 * c_p];
        filter0 = kernel[filter_offset + 0 * c_p];
        color0 += (src * filter0);

        filter1 = kernel[filter_offset + 1 * c_p];
        src    = input[src_offset + 1 * c_p];
        color0 += (src * filter1);
        color1 += (src * filter0);

        filter2 = kernel[filter_offset + 2 * c_p];
        src    = input[src_offset + 2 * c_p];
        color0 += (src * filter2);
        color1 += (src * filter1);
        color2 += (src * filter0);

        filter3 = kernel[filter_offset + 3 * c_p];



        for (int fx=3; fx<kw; ++fx) {
            src    = input[src_offset + fx * c_p];
            color0 += (src * filter3);
            color1 += (src * filter2);
            color2 += (src * filter1);
            color3 += (src * filter0);

            filter0 = filter1;
            filter1 = filter2;
            filter2 = filter3;
            filter3 = kernel[filter_offset + (fx+1) * c_p];
        }

        src    = input[src_offset + kw * c_p];
        color1 += (src * filter2);
        color2 += (src * filter1);
        color3 += (src * filter0);

        src    = input[src_offset + (kw+1) * c_p];
        color2 += (src * filter2);
        color3 += (src * filter1);

        src    = input[src_offset + (kw+2) * c_p];
        color3 += (src * filter2);


        color0 = max(color0, minV);
        color0 = min(color0, maxV);
        color1 = max(color1, minV);
        color1 = min(color1, maxV);

        color2 = max(color2, minV);
        color2 = min(color2, maxV);
        color3 = max(color3, minV);
        color3 = min(color3, maxV);

        int dst_offset = ((ob * oh + oy) * ow + (ox_4 << 2)) * c_p + oz;

        output[dst_offset] = color0;
        output[dst_offset+c_p] = color1;
        output[dst_offset+2*c_p] = color2;
        output[dst_offset+3*c_p] = color3;
    }
}

__global__ void CONV_DW_MULTI_WIDTH_CHANNEL(const float* input, const half* kernel, const half* bias, float *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int total,
    DivModFast d_oc_2,
    DivModFast d_ow_2,
    DivModFast d_oh
    ) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total / 4; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox_2, tmp1, ob;
        d_oc_2.divmod(index, tmp1, oz_2);
        d_ow_2.divmod(tmp1, tmp2, ox_2);
        d_oh.divmod(tmp2, ob, oy);

        float2 color0 =  __half22float2((( half2 *)(bias + (oz_2 << 1)))[0]);
        float2 color1 = color0;

        // Parallel pipelining read and calculate
        float src0, src2, filter0, filter2;
        int src_offset = ((ob * ih + oy) * iw + (ox_2 << 1)) * c_p + (oz_2 << 1);
        int filter_offset = 0 * c_p + (oz_2 << 1);

        float2 src    = ((float2 *)(input + src_offset + 0 * c_p))[0];
        float2 filter = __half22float2(((half2 *)(kernel + filter_offset + 0 * c_p))[0]);

        color0.x += (src.x * filter.x);
        color0.y += (src.y * filter.y);

        for (int fx=1; fx<kw; ++fx) {
            src    = ((float2 *)(input + src_offset + fx * c_p))[0];
            color1.x += (src.x * filter.x);
            color1.y += (src.y * filter.y);

            filter = __half22float2(((half2 *)(void *)(kernel + filter_offset + fx * c_p))[0]);
            color0.x += (src.x * filter.x);
            color0.y += (src.y * filter.y);
        }

        src    = ((float2 *)(input + src_offset + kw * c_p))[0];
        color1.x += (src.x * filter.x);
        color1.y += (src.y * filter.y);

        color0.x = max(color0.x, minV);
        color0.x = min(color0.x, maxV);
        color1.x = max(color1.x, minV);
        color1.x = min(color1.x, maxV);

        color0.y = max(color0.y, minV);
        color0.y = min(color0.y, maxV);
        color1.y = max(color1.y, minV);
        color1.y = min(color1.y, maxV);

        int dst_offset = ((ob * oh + oy) * ow + (ox_2 << 1)) * c_p + (oz_2 << 1);

        ((float2 *)(output + dst_offset))[0] = color0;
        ((float2 *)(output + dst_offset + c_p))[0] = color1;
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- CONV_DW_OPT: fp32 input + half weight/bias -> fp32 output ----
void mnn_corpus_conv_dw_opt_fp32(const float* input, const void* kernel, const void* bias, float* output,
    float maxV, float minV, int iw, int ih, int c, int c_p, int ow, int oh,
    int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph, int total,
    int d_oc, int d_ow, int d_oh, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_(d_oc);
    MNN::Corpus::DivModFast d_ow_(d_ow);
    MNN::Corpus::DivModFast d_oh_(d_oh);
    MNN::Corpus::CONV_DW_OPT<<<grid, block, 0, stream>>>(
        input, (const half*)kernel, (const half*)bias, output,
        maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
        d_oc_, d_ow_, d_oh_);
}

// ---- CONV_DW_HALF2_OPT: all half2 ----
void mnn_corpus_conv_dw_half2_opt_fp32(const void* input, const void* kernel, const void* bias, void* output,
    float maxV, float minV, int iw, int ih, int c, int c_p, int ow, int oh,
    int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph, int total,
    int d_oc, int d_ow, int d_oh, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_(d_oc);
    MNN::Corpus::DivModFast d_ow_(d_ow);
    MNN::Corpus::DivModFast d_oh_(d_oh);
    MNN::Corpus::CONV_DW_HALF2_OPT<<<grid, block, 0, stream>>>(
        (const half2*)input, (const half2*)kernel, (const half2*)bias, (half2*)output,
        maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
        d_oc_, d_ow_, d_oh_);
}

// ---- CONV_DW3x3_HALF2_OPT: all half2, 3x3 specialization ----
// Note: d_ow here uses ow/2 (from MNN launch site).
void mnn_corpus_conv_dw3x3_half2_opt_fp32(const void* input, const void* kernel, const void* bias, void* output,
    float maxV, float minV, int iw, int ih, int c, int c_p, int ow, int oh,
    int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph, int total,
    int d_oc, int d_ow, int d_oh, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_(d_oc);
    MNN::Corpus::DivModFast d_ow_(d_ow);
    MNN::Corpus::DivModFast d_oh_(d_oh);
    MNN::Corpus::CONV_DW3x3_HALF2_OPT<<<grid, block, 0, stream>>>(
        (const half2*)input, (const half2*)kernel, (const half2*)bias, (half2*)output,
        maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
        d_oc_, d_ow_, d_oh_);
}

// ---- CONV_DW_MULTI_WIDTH4<float>: fp32 input + half weight/bias ----
void mnn_corpus_conv_dw_multi_width4_fp32(const float* input, const void* kernel, const void* bias, float* output,
    float maxV, float minV, int iw, int ih, int c, int c_p, int ow, int oh, int kw, int kh, int total,
    int d_oc, int d_ow_4, int d_oh, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_(d_oc);
    MNN::Corpus::DivModFast d_ow_4_(d_ow_4);
    MNN::Corpus::DivModFast d_oh_(d_oh);
    MNN::Corpus::CONV_DW_MULTI_WIDTH4<float><<<grid, block, 0, stream>>>(
        input, (const half*)kernel, (const half*)bias, output,
        maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, total,
        d_oc_, d_ow_4_, d_oh_);
}

// ---- CONV_DW_MULTI_WIDTH_CHANNEL: fp32 input + half weight/bias ----
void mnn_corpus_conv_dw_multi_width_channel_fp32(const float* input, const void* kernel, const void* bias, float* output,
    float maxV, float minV, int iw, int ih, int c, int c_p, int ow, int oh, int kw, int kh, int total,
    int d_oc_2, int d_ow_2, int d_oh, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_2_(d_oc_2);
    MNN::Corpus::DivModFast d_ow_2_(d_ow_2);
    MNN::Corpus::DivModFast d_oh_(d_oh);
    MNN::Corpus::CONV_DW_MULTI_WIDTH_CHANNEL<<<grid, block, 0, stream>>>(
        input, (const half*)kernel, (const half*)bias, output,
        maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, total,
        d_oc_2_, d_ow_2_, d_oh_);
}

} // extern "C"
