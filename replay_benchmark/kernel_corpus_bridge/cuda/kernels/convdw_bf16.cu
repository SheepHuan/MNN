// convdw_bf16.cu - BF16 depthwise conv kernels + weight/bias transpose.
//   source/backend/cuda/execution/bf16/ConvDepthWiseBf16.cuh
//
// Faithful copy of MNN's 6 BF16 kernels. The 4 depthwise conv kernels
// (CONV_DW_BF16/BF162_OPT/3x3_BF162_OPT/MULTI_WIDTH4) use
// `#if (__CUDA_ARCH__ >= 800)` guards: on sm75 the kernel body compiles to an
// empty function (bf16 compute instructions unavailable), but the type
// definitions (__nv_bfloat16/__nv_bfloat162) are available via cuda_bf16.h.
// On sm80+ the body executes. WeightTransToBf16/BiasTransToBf16 have no arch
// guard (pure type conversion, no bf16 compute) and run on any arch.
//
// Corpus defines ENABLE_CUDA_BF16 (matching MNN's MNN_CUDA_BF16 option) so
// the kernel symbols are always present in the binary.
#define ENABLE_CUDA_BF16
#include "corpus_common.cuh"
#include <cuda_bf16.h>

#ifndef PACK_NUMBER
#define PACK_NUMBER 8
#endif

namespace MNN {
namespace Corpus {

__global__ void CONV_DW_BF16(const __nv_bfloat16* input,
    const __nv_bfloat16* kernel,
    const __nv_bfloat16* bias,
    __nv_bfloat16 *output,
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
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2 << 1;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        __nv_bfloat16 color0 = bias[oz];
        __nv_bfloat16 color1 = bias[oz+1];

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                int src_offset = ((ob * ih + sy) * iw + sx) * c_p + oz;
                __nv_bfloat16 inp0 = input[src_offset];
                __nv_bfloat16 inp1 = input[src_offset+1];

                __nv_bfloat16 ker0 = kernel[(fy * kw + fx) * c_p + oz];
                __nv_bfloat16 ker1 = kernel[(fy * kw + fx) * c_p + oz + 1];

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
    #endif
}

__global__ void CONV_DW_BF162_OPT(const __nv_bfloat162* input,
    const __nv_bfloat162* kernel,
    const __nv_bfloat162* bias,
    __nv_bfloat162 *output,
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
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        __nv_bfloat162 color = bias[oz];

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
                __nv_bfloat162 inp = input[src_offset];
                __nv_bfloat162 ker = kernel[(fy * kw + fx) * c_p + oz];

                color = __hfma2(inp, ker, color);
            }
        }

        float2 maxV2, minV2;
        maxV2.x = maxV;
        maxV2.y = maxV;
        minV2.x = minV;
        minV2.y = minV;

        color = __hmax2(color, __float22bfloat162_rn(minV2));
        color = __hmin2(color, __float22bfloat162_rn(maxV2));

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;
        output[dst_offset] = color;
    }
    #endif
}


__global__ void CONV_DW3x3_BF162_OPT(const __nv_bfloat162* input,
    const __nv_bfloat162* kernel,
    const __nv_bfloat162* bias,
    __nv_bfloat162 *output,
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
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/4; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox_2, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox_2);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ox = ox_2 << 1;
        int ix = ox - 1;
        int iy = oy - 1;
        __nv_bfloat162 color0 = bias[oz];
        __nv_bfloat162 color1 = color0;

        __nv_bfloat162 zero;
        zero.x = (__nv_bfloat16)0.0;
        zero.y = (__nv_bfloat16)0.0;

        __nv_bfloat162 inp[12];
        __nv_bfloat162 ker[3][3];
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
    #endif
}

template<typename T>
__global__ void CONV_DW_BF16_MULTI_WIDTH4(const T* input, const __nv_bfloat16* kernel, const __nv_bfloat16* bias, T *output,
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
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
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
    #endif
}

template<typename T0, typename T>
__global__ void WeightTransToBf16(const T0* param,
    T* output,
    const size_t maxCount,
    const int khw,
    const int oc,
    DivModFast d_cp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int kIndex, cpIndex;
        d_cp.divmod(index, kIndex, cpIndex);

        if(cpIndex >= oc) {
            output[index] = (T)0.0f;
            continue;
        }
        output[index] = param[cpIndex * khw + kIndex];
    }
}

template<typename T0, typename T>
__global__ void BiasTransToBf16(const T0* param,
    T* output,
    const size_t maxCount,
    const int oc
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        if(index >= oc) {
            output[index] = (T)0.0f;
            continue;
        }
        output[index] = param[index];
    }
}

} //namespace Corpus
} //namespace MNN

// ============================================================================
// extern "C" shims — thin launch wrappers matching MNN's onExecute geometry.
// Shim names are tag-independent (tag branching is in the adapter).
// ============================================================================
extern "C" {

// CONV_DW_BF16: MNN launch is blockNum/threadNum (runtime adaptive). Corpus
// uses fixed kBlock=128 grid like the fp32 CONV_DW path. c_p passed as-is
// (half-unit channel count, matches MNN call site line 544-547).
void mnn_corpus_conv_dw_bf16_fp32(const __nv_bfloat16* input, const __nv_bfloat16* kernel,
    const __nv_bfloat16* bias, __nv_bfloat16* output, float maxV, float minV,
    int iw, int ih, int c, int c_p, int ow, int oh, int kw, int kh,
    int dw, int dh, int sw, int sh, int pw, int ph, int total,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_(c_p);
    MNN::Corpus::DivModFast d_ow_(ow);
    MNN::Corpus::DivModFast d_oh_(oh);
    MNN::Corpus::CONV_DW_BF16<<<grid, block, 0, stream>>>(
        input, kernel, bias, output, maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh,
        dw, dh, sw, sh, pw, ph, total, d_oc_, d_ow_, d_oh_);
}

// CONV_DW_BF162_OPT: MNN passes c_p/2 (half2-unit). Adapter passes c_p/2.
void mnn_corpus_conv_dw_bf162_opt_fp32(const __nv_bfloat162* input, const __nv_bfloat162* kernel,
    const __nv_bfloat162* bias, __nv_bfloat162* output, float maxV, float minV,
    int iw, int ih, int c, int c_p_half, int ow, int oh, int kw, int kh,
    int dw, int dh, int sw, int sh, int pw, int ph, int total,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_(c_p_half);
    MNN::Corpus::DivModFast d_ow_(ow);
    MNN::Corpus::DivModFast d_oh_(oh);
    MNN::Corpus::CONV_DW_BF162_OPT<<<grid, block, 0, stream>>>(
        input, kernel, bias, output, maxV, minV, iw, ih, c, c_p_half, ow, oh, kw, kh,
        dw, dh, sw, sh, pw, ph, total, d_oc_, d_ow_, d_oh_);
}

// CONV_DW3x3_BF162_OPT: MNN passes c_p/2 + d_ow2(ow/2). Adapter passes same.
void mnn_corpus_conv_dw3x3_bf162_opt_fp32(const __nv_bfloat162* input, const __nv_bfloat162* kernel,
    const __nv_bfloat162* bias, __nv_bfloat162* output, float maxV, float minV,
    int iw, int ih, int c, int c_p_half, int ow, int oh, int kw, int kh,
    int dw, int dh, int sw, int sh, int pw, int ph, int total,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_(c_p_half);
    MNN::Corpus::DivModFast d_ow_(ow / 2);
    MNN::Corpus::DivModFast d_oh_(oh);
    MNN::Corpus::CONV_DW3x3_BF162_OPT<<<grid, block, 0, stream>>>(
        input, kernel, bias, output, maxV, minV, iw, ih, c, c_p_half, ow, oh, kw, kh,
        dw, dh, sw, sh, pw, ph, total, d_oc_, d_ow_, d_oh_);
}

// CONV_DW_BF16_MULTI_WIDTH4: template instantiated with __nv_bfloat16.
// MNN passes c_p (half-unit) + d_ow(ow/4). Adapter passes same.
void mnn_corpus_conv_dw_bf16_multi_width4_fp32(const __nv_bfloat16* input, const __nv_bfloat16* kernel,
    const __nv_bfloat16* bias, __nv_bfloat16* output, float maxV, float minV,
    int iw, int ih, int c, int c_p, int ow, int oh, int kw, int kh, int total,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc_(c_p);
    MNN::Corpus::DivModFast d_ow_(ow / 4);
    MNN::Corpus::DivModFast d_oh_(oh);
    MNN::Corpus::CONV_DW_BF16_MULTI_WIDTH4<__nv_bfloat16><<<grid, block, 0, stream>>>(
        input, kernel, bias, output, maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh,
        total, d_oc_, d_ow_, d_oh_);
}

// WeightTransToBf16<float, __nv_bfloat16>: transpose weight [oc, khw] ->
// [khw, oc_p] layout. MNN launch: block_num/threads_num (runtime adaptive).
// Corpus uses kBlock=128 grid. maxCount = depthC*PACK_NUMBER*khw.
void mnn_corpus_weight_trans_to_bf16_fp32(const float* param, __nv_bfloat16* output,
    size_t maxCount, int khw, int oc, int oc_p,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_cp_(oc_p);
    MNN::Corpus::WeightTransToBf16<float, __nv_bfloat16><<<grid, block, 0, stream>>>(
        param, output, maxCount, khw, oc, d_cp_);
}

// BiasTransToBf16<float, __nv_bfloat16>: copy bias, zero-pad to oc_p.
void mnn_corpus_bias_trans_to_bf16_fp32(const float* param, __nv_bfloat16* output,
    size_t maxCount, int oc,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BiasTransToBf16<float, __nv_bfloat16><<<grid, block, 0, stream>>>(
        param, output, maxCount, oc);
}

} // extern "C"
