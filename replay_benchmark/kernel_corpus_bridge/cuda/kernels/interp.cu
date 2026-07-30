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

// ============================================================================
// INTERP_BILINEAR_OPT: optimized bilinear (2 output x-pixels per thread)
//   source/backend/cuda/execution/InterpExecution.cu (lines 86-164)
//   Disabled in MNN with if(0), but kernel is complete and usable.
//   Uses NC4HW4 layout: index = (spatial_flat << 4) + remain, where remain
//   indexes the PACK_NUMBER=4 channel lane within a C4 group.
// ============================================================================
static const int PACK_NUMBER = 4;  // MNN InterpExecution uses PACK_NUMBER=4

template<typename T>
__global__ void INTERP_BILINEAR_OPT(const int n, const int ih, const int iw, const int oh, const int ow,
    const float scaleh, const float scalew, const float offseth, const float offsetw, const T* in, T* out,
    DivModFast d_ow, DivModFast d_oh) {
    CUDA_KERNEL_LOOP(total, n) {
        size_t index = total >> 4;
        size_t remain = total & 15;

        int tmp, x_idx, y, z;
        d_ow.divmod(index, tmp, x_idx);
        d_oh.divmod(tmp, z, y);

        size_t x = x_idx << 1;
        float fx = x*scalew+offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw-1);
        int ix_1 = min((int)ceil(fx), iw-1);

        float fx_1 = fx + scalew;
        int ix_2 = min(max(0, (int)floor(fx_1)), iw-1);
        int ix_3 = min((int)ceil(fx_1), iw-1);

        float fy = y*scaleh+offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih-1);
        int iy_1 = min((int)ceil(fy), ih-1);

        int index_00 = (z*ih+ iy_0)*iw + ix_0;
        int index_01 = index_00 - ix_0 + ix_1;
        int index_10 = (z*ih+ iy_1)*iw + ix_0;
        int index_11 = index_10 - ix_0 + ix_1;
        index_00 = (index_00 << 4) + remain;
        index_01 = (index_01 << 4) + remain;
        index_10 = (index_10 << 4) + remain;
        index_11 = (index_11 << 4) + remain;

        float factor_x = fx-ix_0;
        float factor_y = fy-iy_0;
        float in_00 = (float)in[index_00];
        float in_01 = (float)in[index_01];
        float in_10 = (float)in[index_10];
        float in_11 = (float)in[index_11];

        float factor_00 = (1.0-factor_x)*(1.0-factor_y);
        float factor_01 = factor_x*(1.0-factor_y);
        float factor_10 = (1.0-factor_x)*factor_y;
        float factor_11 = factor_x*factor_y;

        size_t dstOffset = (((z*oh+ y)*ow + x) << 4) + remain;
        out[dstOffset] = \
            factor_00* in_00 + factor_01*in_01 + \
            factor_10* in_10 + factor_11*in_11;

        if(x+1 >= ow) {
            continue;
        }

        if(ix_2 != ix_0) {
            index_00 = index_00 + ((ix_2-ix_0) << 4);
            index_10 = index_10 + ((ix_2-ix_0) << 4);
            in_00 = (float)in[index_00];
            in_10 = (float)in[index_10];
        }
        if(ix_3 != ix_1) {
            index_01 = index_01 + ((ix_3-ix_1) << 4);
            index_11 = index_11 + ((ix_3-ix_1) << 4);
            in_01 = (float)in[index_01];
            in_11 = (float)in[index_11];
        }

        if(factor_x != fx_1-ix_2) {
            factor_x = fx_1-ix_2;
            factor_00 = (1.0-factor_x)*(1.0-factor_y);
            factor_01 = factor_x*(1.0-factor_y);
            factor_10 = (1.0-factor_x)*factor_y;
            factor_11 = factor_x*factor_y;
        }
        out[dstOffset+ PACK_NUMBER] = \
            factor_00* in_00 + factor_01*in_01 + \
            factor_10* in_10 + factor_11*in_11;
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

// ---- INTERP_BILINEAR_OPT (2x-pixel-per-thread optimized bilinear, NC4HW4) ----
void mnn_corpus_interp_bilinear_opt_fp32(const int n, int ih, int iw, int oh, int ow,
                                         float sh, float sw, float ohf, float owf,
                                         const float* in, float* out,
                                         int d_ow, int d_oh, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast owD(d_ow), ohD(d_oh);
    MNN::Corpus::INTERP_BILINEAR_OPT<float><<<grid, block, 0, stream>>>(
        n, ih, iw, oh, ow, sh, sw, ohf, owf, in, out, owD, ohD);
}

// ============================================================================
// fp16 (<half>) variants — only 3.6.0 (nearest, bilinear, nearest_round, opt)
// ============================================================================
void mnn_corpus_interp_nearest_fp16(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                     float sh, float sw, float ohf, float owf,
                                     const void* in, void* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_NERAEST<__half><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf,
                                                                     (const __half*)in, (__half*)out);
}
void mnn_corpus_interp_bilinear_fp16(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                      float sh, float sw, float ohf, float owf,
                                      const void* in, void* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_BILINEAR<__half><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf,
                                                                      (const __half*)in, (__half*)out);
}
void mnn_corpus_interp_nearest_round_fp16(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                           float sh, float sw, float ohf, float owf,
                                           const void* in, void* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_NERAEST_ROUND<__half><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf,
                                                                           (const __half*)in, (__half*)out);
}
void mnn_corpus_interp_bilinear_opt_fp16(const int n, int ih, int iw, int oh, int ow,
                                          float sh, float sw, float ohf, float owf,
                                          const void* in, void* out,
                                          int d_ow, int d_oh, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast owD(d_ow), ohD(d_oh);
    MNN::Corpus::INTERP_BILINEAR_OPT<__half><<<grid, block, 0, stream>>>(
        n, ih, iw, oh, ow, sh, sw, ohf, owf, (const __half*)in, (__half*)out, owD, ohD);
}

} // extern "C"
