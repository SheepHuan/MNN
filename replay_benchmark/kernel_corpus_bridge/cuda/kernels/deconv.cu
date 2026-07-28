// deconv.cu - DeconvKernelReorder / Col2Im / Col2Im_Vec4 (3.6.0)
//   source/backend/cuda/execution/DeconvBaseKernel.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

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
// Col2Im_Vec4: vec4 version (oc % 4 == 0), processes 4 channels per thread
// ============================================================================
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
            val = *((float4*)((float*)bias + c_im));
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
        *((float4*)((float*)data_im + dst_offset)) = val;
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

// ---- Col2Im_Vec4 (fp32, vec4) ----
void mnn_corpus_col2im_vec4_fp32(const int n, const float* data_col,
                                  int batch, int height, int width, int channels,
                                  int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w,
                                  int dilation_h, int dilation_w, int activationType,
                                  int height_col, int width_col,
                                  const float* bias, float* data_im,
                                  int d_ocp_val, int d_ow_val, int d_oh_val, int d_ob_val,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_ocp(d_ocp_val), d_ow(d_ow_val), d_oh(d_oh_val), d_ob(d_ob_val);
    MNN::Corpus::Col2Im_Vec4<float, float><<<grid, block, 0, stream>>>(
        n, data_col, batch, height, width, channels, kh, kw, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w, activationType,
        height_col, width_col, bias, data_im, d_ocp, d_ow, d_oh, d_ob);
}

} // extern "C"
