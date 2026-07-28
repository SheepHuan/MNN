// interp.cu - INTERP_NERAEST / INTERP_BILINEAR / INTERP_NERAEST_ROUND (3.6.0)
//   + INTERP_120 / INTERP_BILINEAR_120 (1.2.0) kernels + shims
//   source/backend/cuda/execution/InterpExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// Interp nearest: source/backend/cuda/execution/InterpExecution.cu
// ============================================================================
template <typename T>
__global__ void INTERP_NERAEST(const int total, const int c_p,
                                const int ih, const int iw, const int oh, const int ow,
                                const float scaleh, const float scalew, const float offseth, const float offsetw,
                                const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw)), iw - 1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth)), ih - 1);
        out[((z * oh + y) * ow + x) * c_p + c_idx] = in[((z * ih + iy) * iw + ix) * c_p + c_idx];
    }
}

// ============================================================================
// Interp bilinear / round: source/backend/cuda/execution/InterpExecution.cu
// ============================================================================
template <typename T>
__global__ void INTERP_BILINEAR(const int total, const int c_p,
                                const int ih, const int iw, const int oh, const int ow,
                                const float scaleh, const float scalew, const float offseth, const float offsetw,
                                const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        float fx = x * scalew + offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw - 1);
        int ix_1 = min((int)ceil(fx), iw - 1);
        float fy = y * scaleh + offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih - 1);
        int iy_1 = min((int)ceil(fy), ih - 1);
        int index_00 = ((z * ih + iy_0) * iw + ix_0) * c_p + c_idx;
        int index_01 = ((z * ih + iy_0) * iw + ix_1) * c_p + c_idx;
        int index_10 = ((z * ih + iy_1) * iw + ix_0) * c_p + c_idx;
        int index_11 = ((z * ih + iy_1) * iw + ix_1) * c_p + c_idx;
        float factor_x = fx - ix_0;
        float factor_y = fy - iy_0;
        out[((z * oh + y) * ow + x) * c_p + c_idx] = (T)(
            (1.0f - factor_x) * (1.0f - factor_y) * (float)in[index_00]
            + factor_x * (1.0f - factor_y) * (float)in[index_01]
            + (1.0f - factor_x) * factor_y * (float)in[index_10]
            + factor_x * factor_y * (float)in[index_11]);
    }
}
template <typename T>
__global__ void INTERP_NERAEST_ROUND(const int total, const int c_p,
                                     const int ih, const int iw, const int oh, const int ow,
                                     const float scaleh, const float scalew, const float offseth, const float offsetw,
                                     const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw + 0.499f)), iw - 1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth + 0.499f)), ih - 1);
        out[((z * oh + y) * ow + x) * c_p + c_idx] = in[((z * ih + iy) * iw + ix) * c_p + c_idx];
    }
}

// ============================================================================
// 1.2.0 tag: INTERP (nearest, different signature from 3.6.0 INTERP_NERAEST)
// ============================================================================
template <typename T>
__global__ void INTERP_120(const int n, const int ih, const int iw, const int oh, const int ow,
                            const float scaleh, const float scalew, const float offseth, const float offsetw,
                            const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw)), iw - 1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth)), ih - 1);
        out[(z * oh + y) * ow + x] = in[(z * ih + iy) * iw + ix];
    }
}

// ============================================================================
// 1.2.0 tag: INTERP_BILINEAR (different from 3.6.0: no c_p, uses n not total)
// ============================================================================
template <typename T>
__global__ void INTERP_BILINEAR_120(const int n, const int ih, const int iw, const int oh, const int ow,
                                     const float scaleh, const float scalew, const float offseth, const float offsetw,
                                     const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        float fx = x * scalew + offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw - 1);
        int ix_1 = min((int)ceil(fx), iw - 1);
        float fy = y * scaleh + offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih - 1);
        int iy_1 = min((int)ceil(fy), ih - 1);
        float factor_x = fx - ix_0;
        float factor_y = fy - iy_0;
        out[(z * oh + y) * ow + x] = (T)(
            (1.0f - factor_x) * (1.0f - factor_y) * (float)in[(z * ih + iy_0) * iw + ix_0]
            + factor_x * (1.0f - factor_y) * (float)in[(z * ih + iy_0) * iw + ix_1]
            + (1.0f - factor_x) * factor_y * (float)in[(z * ih + iy_1) * iw + ix_0]
            + factor_x * factor_y * (float)in[(z * ih + iy_1) * iw + ix_1]);
    }
}

// ---- INTERP_NERAEST 1.2.7: PACK_NUMBER, no c_p ----
template <typename T>
__global__ void INTERP_NERAEST_127(const int n, const int ih, const int iw, const int oh, const int ow,
                                    const float scaleh, const float scalew, const float offseth, const float offsetw,
                                    const T* in, T* out) {
    const int PACK_NUMBER = 4;
    CUDA_KERNEL_LOOP(total, n) {
        int index = total / PACK_NUMBER;
        int remain = total % PACK_NUMBER;
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw)), iw - 1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth)), ih - 1);
        out[(z * oh * ow + y * ow + x) * PACK_NUMBER + remain]
            = in[(z * ih * iw + iy * iw + ix) * PACK_NUMBER + remain];
    }
}

// ---- INTERP_BILINEAR 1.2.7: PACK_NUMBER, no c_p ----
template <typename T>
__global__ void INTERP_BILINEAR_127(const int n, const int ih, const int iw, const int oh, const int ow,
                                     const float scaleh, const float scalew, const float offseth, const float offsetw,
                                     const T* in, T* out) {
    const int PACK_NUMBER = 4;
    CUDA_KERNEL_LOOP(total, n) {
        int index = total / PACK_NUMBER;
        int remain = total % PACK_NUMBER;
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        float fx = x * scalew + offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw - 1);
        int ix_1 = min((int)ceil(fx), iw - 1);
        float fy = y * scaleh + offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih - 1);
        int iy_1 = min((int)ceil(fy), ih - 1);
        int i00 = (z * ih * iw + iy_0 * iw + ix_0) * PACK_NUMBER + remain;
        int i01 = (z * ih * iw + iy_0 * iw + ix_1) * PACK_NUMBER + remain;
        int i10 = (z * ih * iw + iy_1 * iw + ix_0) * PACK_NUMBER + remain;
        int i11 = (z * ih * iw + iy_1 * iw + ix_1) * PACK_NUMBER + remain;
        float fx_w = fx - ix_0, fy_w = fy - iy_0;
        out[(z * oh * ow + y * ow + x) * PACK_NUMBER + remain] = (T)(
            (1.0 - fx_w) * (1.0 - fy_w) * (float)in[i00] + fx_w * (1.0 - fy_w) * (float)in[i01] +
            (1.0 - fx_w) * fy_w * (float)in[i10] + fx_w * fy_w * (float)in[i11]);
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Interp nearest ----
void mnn_corpus_interp_nearest_fp32(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                     float sh, float sw, float ohf, float owf,
                                     const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_NERAEST<float><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}

// ---- Interp bilinear / round ----
void mnn_corpus_interp_bilinear_fp32(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                      float sh, float sw, float ohf, float owf,
                                      const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_BILINEAR<float><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}
void mnn_corpus_interp_nearest_round_fp32(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                           float sh, float sw, float ohf, float owf,
                                           const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_NERAEST_ROUND<float><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}

// ---- 1.2.0 tag: INTERP ----
void mnn_corpus_interp_120_fp32(const int n, int ih, int iw, int oh, int ow,
                                float sh, float sw, float ohf, float owf,
                                const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_120<float><<<grid, block, 0, stream>>>(n, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}
void mnn_corpus_interp_bilinear_120_fp32(const int n, int ih, int iw, int oh, int ow,
                                         float sh, float sw, float ohf, float owf,
                                         const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_BILINEAR_120<float><<<grid, block, 0, stream>>>(n, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}

// ---- INTERP_NERAEST 1.2.7 (PACK_NUMBER, no c_p) ----
void mnn_corpus_interp_nearest_127_fp32(const int n, int ih, int iw, int oh, int ow,
                                         float sh, float sw, float ohf, float owf,
                                         const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_NERAEST_127<float><<<grid, block, 0, stream>>>(n, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}
// ---- INTERP_BILINEAR 1.2.7 (PACK_NUMBER, no c_p) ----
void mnn_corpus_interp_bilinear_127_fp32(const int n, int ih, int iw, int oh, int ow,
                                          float sh, float sw, float ohf, float owf,
                                          const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_BILINEAR_127<float><<<grid, block, 0, stream>>>(n, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}

} // extern "C"
