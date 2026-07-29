// deconv.cu - DeconvKernelReorder / Col2Im / Col2Im_Vec4 (3.6.0)
//   source/backend/cuda/execution/DeconvBaseKernel.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// 1.2.0-era deconv parameter structs (from legacy_kernels.cu)
// ============================================================================
struct ConstBuffer_120 {
    int pad[2]; int kernelSize[2]; int stride[2]; int dilate[2];
    int inputSize[2]; int outputSize[2];
    int channel; int subChannel; int total; int activationType;
};
struct InputReorderParameter {
    int ic_stride; int ib_stride; int oc_stride; int ob_stride;
    int hw_size; int l_size; int h_size; int lpack_size; int hpack_size;
};

// ============================================================================
// DeconvKernelReorder: [Ci,Co,KhKw] → [KhKw,Co,Cip] weight reordering
// ============================================================================
template<typename T0, typename T1>
__global__ void DeconvKernelReorder(const T0* B, T1* BP, int kw, int kh, int ic, int oc, int icPack, int ocPack) {
    int kernelCount = kw * kh;
    int maxCount = kernelCount * icPack * oc;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int l_idx = index % icPack;
        int h_idx = index / icPack;
        int oc_idx = h_idx % oc;
        int khw_idx = h_idx / oc;
        if (l_idx >= ic) {
            BP[index] = (T1)0.0f;
            continue;
        }
        BP[index] = (T1)(B[(l_idx * oc + oc_idx) * kernelCount + khw_idx]);
    }
}

// ============================================================================
// Col2Im: scalar version, deconvolution col→im transform
// ============================================================================
template <typename Stype, typename Dtype>
__global__ void Col2Im(const int n, const Stype* data_col,
    const int batch, const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int activationType,
    const int height_col, const int width_col,
    const Dtype* bias, Dtype* data_im,
    DivModFast d_ocp, DivModFast d_ow, DivModFast d_oh, DivModFast d_ob
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)n; index += blockDim.x * gridDim.x) {
        Dtype val = 0;
        int idx_ocp, idx_tmp, idx_bh, b_im, idx_w, idx_h;
        d_ocp.divmod(index, idx_tmp, idx_ocp);
        const int c_im = idx_ocp;
        if (c_im >= channels) {
            data_im[index] = val;
            continue;
        }
        d_ow.divmod(idx_tmp, idx_bh, idx_w);
        d_oh.divmod(idx_bh, b_im, idx_h);
        const int w_im = idx_w + pad_w;
        const int h_im = idx_h + pad_h;
        if (nullptr != bias) {
            val += (Dtype)bias[c_im];
        }
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        const int w_col_start = (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start = (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    const int data_col_index = ((((b_im * height_col + h_col) * width_col + w_col) * kernel_h + h_k) * kernel_w + w_k) * channels + c_im;
                    val += (Dtype)data_col[data_col_index];
                }
            }
        }
        if (activationType == 1) {
            val = (Dtype)max((float)val, 0.0f);
        }
        if (activationType == 2) {
            val = (Dtype)max((float)val, 0.0f);
            val = (Dtype)min((float)val, 6.0f);
        }
        data_im[index] = val;
    }
}

// ============================================================================
// Col2Im_Vec4: vec4 version (oc % 4 == 0), processes 4 channels per thread.
// Faithful to MNN DeconvBaseKernel.cu: takes a `precision` argument and routes
// the final store through DATA_CONVERT_COPY (precision==1 float4, precision==2
// float4→half4 via two half2 conversions; precision==0/3 bf16 skipped on sm75).
// Bias load also branches on precision (precision==2 loads 4 scalar half).
// ============================================================================
#define DECONV_DATA_CONVERT_COPY(precision) \
    if (precision == 1) { *((float4*)((float*)data_im + dst_offset)) = val; } \
    else if (precision == 2) { \
        float2 t0; t0.x = val.x; t0.y = val.y; *(half2*)((half*)(data_im + dst_offset)) = __float22half2_rn(t0); \
        float2 t1; t1.x = val.z; t1.y = val.w; *(half2*)((half*)(data_im + dst_offset + 2)) = __float22half2_rn(t1); }

template <typename Stype, typename Dtype>
__global__ void Col2Im_Vec4(const int n, const Stype* data_col,
    const int batch, const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int activationType,
    const int height_col, const int width_col,
    const Dtype* bias, Dtype* data_im,
    const int precision,
    DivModFast d_ocp, DivModFast d_ow, DivModFast d_oh, DivModFast d_ob
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)n; index += blockDim.x * gridDim.x) {
        float4 val;
        val.x = 0.0f; val.y = 0.0f; val.z = 0.0f; val.w = 0.0f;
        int idx_ocp, idx_bhw, idx_bh, b_im, idx_w, idx_h;
        d_ocp.divmod(index, idx_bhw, idx_ocp);
        const int c_im = idx_ocp << 2;
        if (c_im >= channels) { continue; }
        d_ow.divmod(idx_bhw, idx_bh, idx_w);
        d_oh.divmod(idx_bh, b_im, idx_h);
        const int w_im = idx_w + pad_w;
        const int h_im = idx_h + pad_h;
        if (nullptr != bias) {
            if (precision == 2) {
                val.x += (float)bias[c_im];
                val.y += (float)bias[c_im + 1];
                val.z += (float)bias[c_im + 2];
                val.w += (float)bias[c_im + 3];
            } else {
                val = *((float4*)((float*)bias + c_im));
            }
        }
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        const int w_col_start = (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start = (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    const int data_col_index = ((((b_im * height_col + h_col) * width_col + w_col) * kernel_h + h_k) * kernel_w + w_k) * channels + c_im;
                    val.x += (float)data_col[data_col_index];
                    val.y += (float)data_col[data_col_index + 1];
                    val.z += (float)data_col[data_col_index + 2];
                    val.w += (float)data_col[data_col_index + 3];
                }
            }
        }
        if (activationType == 1) {
            val.x = max(val.x, 0.0f); val.y = max(val.y, 0.0f);
            val.z = max(val.z, 0.0f); val.w = max(val.w, 0.0f);
        }
        if (activationType == 2) {
            val.x = min(max(val.x, 0.0f), 6.0f); val.y = min(max(val.y, 0.0f), 6.0f);
            val.z = min(max(val.z, 0.0f), 6.0f); val.w = min(max(val.w, 0.0f), 6.0f);
        }
        int dst_offset = index << 2;
        DECONV_DATA_CONVERT_COPY(precision);
    }
}

// ============================================================================
// 1.2.0-era legacy deconv kernels (from legacy_kernels.cu)
// ============================================================================
template <typename T>
__global__ void cutPad(const size_t size, const T* input, const int old_height,
                    const int old_width, const int height, const int width, const int pad_top,
                    const int pad_left, T* output) {
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
        int block_num = pos / (width*height);
        int left = pos % (width*height);
        const int out_w = left % width;
        const int out_h = left / width % height;

        output[pos] = input[(block_num * old_height + out_h + pad_top) * old_width + out_w + pad_left];
    }
    return;
}

__global__ void DECONV_DW(const float* input, const float* kernel, const float* bias, float *output, const ConstBuffer_120* uConstant) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < uConstant->total; i += blockDim.x * gridDim.x) {
        {
            int iw = uConstant->inputSize[0];
            int ih = uConstant->inputSize[1];
            int c = uConstant->channel;
            int ow = uConstant->outputSize[0];
            int oh = uConstant->outputSize[1];
            int kw = uConstant->kernelSize[0];
            int kh = uConstant->kernelSize[1];
            int dw = uConstant->dilate[0];
            int dh = uConstant->dilate[1];
            int sw = uConstant->stride[0];
            int sh = uConstant->stride[1];
            int pw = uConstant->pad[0];
            int ph = uConstant->pad[1];

            int oz = i / (ow * oh);
            int tmp = i % (ow * oh);
            int oy = tmp / ow;
            int ox = tmp % ow;
            int kz = oz % uConstant->subChannel;
            
            int ix = ox + pw;
            int iy = oy + ph;
            float color = 0.0;
            if (bias != nullptr) {
                color = bias[kz];
            }

            int fx, fy, fz;
            for (fy=0; fy<kh; ++fy) {
                int sy = iy - fy*dh;
                int y = sy / sh;
                if (sy % sh == 0 && y >= 0 && y < ih) {
                    for (int fx=0; fx<kw; ++fx) {
                        int sx = ix - fx*dw;
                        int x = sx / sw;
                        if (sx % sw == 0 && x >= 0 && x < iw) {
                            float inputValue = input[0
                                + x
                                + y * iw
                                + oz * iw * ih
                            ];
                            float k = kernel[0
                                + fx
                                + fy * kw
                                + kz * kw * kh
                            ];
                            color  += k*inputValue;                            
                        }
                    }
                }
            }
            output[0
                + ox
                + oy * ow
                + oz * ow * oh
            ] = color;
        }
    }
    return;
}

__global__ void DeconvInputRerange(const int count,
        const InputReorderParameter* param,
        const float* Inp,
        __half* InpRe
        ) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int l = param->l_size;
        int h = param->h_size;
        int lIndex = i % l;
        int hIndex = i / l;
        int lU = lIndex / 16;
        int lR = lIndex % 16;
        int hU = hIndex / 16;
        int hR = hIndex % 16;

        int bIndex = hIndex / param->hw_size;
        int hwIndex = hIndex % param->hw_size;

        float value = Inp[bIndex * param->ib_stride + lIndex * param->ic_stride + hwIndex];
        //inpRe[lIndex * param->oc_stride + bIndex * param->ob_stride + hwIndex] = value;

        //__half* dst = InpRe + lU * param->hpack_size * 16 * 16 + hU * 16 * 16 + hR + lR * 16;
        __half* dst = InpRe + hU * param->lpack_size * 16 * 16 + lU * 16 * 16 + lR + hR * 16;
        dst[0] = value;
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- DeconvKernelReorder (fp32) ----
void mnn_corpus_deconv_kernel_reorder_fp32(const float* B, float* BP, int kw, int kh, int ic, int oc, int icPack, int ocPack,
                                           int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DeconvKernelReorder<float, float><<<grid, block, 0, stream>>>(B, BP, kw, kh, ic, oc, icPack, ocPack);
}

// ---- Col2Im (fp32, scalar) ----
void mnn_corpus_col2im_fp32(const int n, const float* data_col,
                            int batch, int height, int width, int channels,
                            int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w, int activationType,
                            int height_col, int width_col,
                            const float* bias, float* data_im,
                            int d_ocp_val, int d_ow_val, int d_oh_val, int d_ob_val,
                            int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_ocp(d_ocp_val), d_ow(d_ow_val), d_oh(d_oh_val), d_ob(d_ob_val);
    MNN::Corpus::Col2Im<float, float><<<grid, block, 0, stream>>>(
        n, data_col, batch, height, width, channels, kh, kw, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w, activationType,
        height_col, width_col, bias, data_im, d_ocp, d_ow, d_oh, d_ob);
}

// ---- Col2Im_Vec4 (fp32, vec4, precision=1) ----
void mnn_corpus_col2im_vec4_fp32(const int n, const float* data_col,
                                  int batch, int height, int width, int channels,
                                  int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w,
                                  int dilation_h, int dilation_w, int activationType,
                                  int height_col, int width_col,
                                  const float* bias, float* data_im,
                                  int precision,
                                  int d_ocp_val, int d_ow_val, int d_oh_val, int d_ob_val,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_ocp(d_ocp_val), d_ow(d_ow_val), d_oh(d_oh_val), d_ob(d_ob_val);
    MNN::Corpus::Col2Im_Vec4<float, float><<<grid, block, 0, stream>>>(
        n, data_col, batch, height, width, channels, kh, kw, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w, activationType,
        height_col, width_col, bias, data_im, precision, d_ocp, d_ow, d_oh, d_ob);
}

} // extern "C"
