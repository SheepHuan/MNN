// prelu.cu - PRELU 1.2.7/1.2.8/2.0.4-3.6.0 kernels + shims
//   source/backend/cuda/execution/PReLUExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// PReLU 1.2.7: float* slope, div_factor, PACK_NUMBER channel packing
// (source/backend/cuda/execution/PReLUExecution.cu @ tag 1.2.7)
// ============================================================================
template <typename T>
__global__ void PRELU_127(const int n, const int channels, const int dim, const T* in, T* out,
                          const float* slopeData, int div_factor) {
    const int PACK_NUMBER = 4;
    CUDA_KERNEL_LOOP(t, n) {
        int index = t / PACK_NUMBER;
        int r = t % PACK_NUMBER;
        int c = (index / dim) % channels / div_factor;
        float iv = (float)in[t];
        float ov = iv > 0.0f ? iv : iv * slopeData[c * PACK_NUMBER + r];
        out[t] = (T)ov;
    }
}

// ============================================================================
// PReLU 1.2.8: float* slope, share_factor, PACK_NUMBER + share_factor gating
// (source/backend/cuda/execution/PReLUExecution.cu @ tag 1.2.8)
// ============================================================================
template <typename T>
__global__ void PRELU_128(const int n, const int channels, const int dim, const T* in, T* out,
                          const float* slopeData, int share_factor) {
    const int PACK_NUMBER = 4;
    CUDA_KERNEL_LOOP(t, n) {
        int index = t / PACK_NUMBER;
        int r = t % PACK_NUMBER;
        int c = (index / dim) % channels;
        float iv = (float)in[t];
        const int c_idx = share_factor ? 0 : (c * PACK_NUMBER + r);
        float ov = iv > 0.0f ? iv : iv * slopeData[c_idx];
        out[t] = (T)ov;
    }
}

// ============================================================================
// PReLU 2.0.4 / 3.6.0: float* slope, share_factor, channelsPack (no PACK_NUMBER)
// (source/backend/cuda/execution/PReLUExecution.cu @ tag 2.0.4 / HEAD)
// ============================================================================
template <typename T>
__global__ void PRELU(const int total, const int channelsPack, const int dim, const T* in, T* out,
                       const float* slopeData, int share_factor) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx = index / channelsPack;
        int c_idx = index % channelsPack;
        float iv = (float)in[index];
        c_idx = share_factor ? 0 : c_idx;
        float ov = iv > 0.0 ? iv : iv * slopeData[c_idx];
        out[index] = (T)ov;
    }
}

// ============================================================================
// 1.2.0 tag: PRELU (slope is T* not float*; div_factor instead of share_factor)
// ============================================================================
template <typename T>
__global__ void PRELU_120(const int n, const int channels, const int dim, const T* in, T* out,
                          const T* slopeData, int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c = (index / dim) % channels / div_factor;
        out[index] = in[index] > 0 ? in[index] : in[index] * slopeData[c];
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- PReLU ----
void mnn_corpus_prelu_fp32(const int total, const int channelsPack, const int dim, const float* in, float* out,
                            const float* slopeData, int share_factor, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PRELU<float><<<grid, block, 0, stream>>>(total, channelsPack, dim, in, out, slopeData, share_factor);
}
void mnn_corpus_prelu_fp16(const int total, const int channelsPack, const int dim, const void* in, void* out,
                            const float* slopeData, int share_factor, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PRELU<half><<<grid, block, 0, stream>>>(total, channelsPack, dim, (const half*)in, (half*)out,
                                                           slopeData, share_factor);
}
// ---- PReLU 1.2.7: PACK_NUMBER, div_factor ----
void mnn_corpus_prelu_127_fp32(const int n, const int channels, const int dim, const float* in, float* out,
                                const float* slopeData, int div_factor, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PRELU_127<float><<<grid, block, 0, stream>>>(n, channels, dim, in, out, slopeData, div_factor);
}
// ---- PReLU 1.2.8: PACK_NUMBER, share_factor ----
void mnn_corpus_prelu_128_fp32(const int n, const int channels, const int dim, const float* in, float* out,
                                const float* slopeData, int share_factor, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PRELU_128<float><<<grid, block, 0, stream>>>(n, channels, dim, in, out, slopeData, share_factor);
}

// ---- 1.2.0 tag: PRELU ----
void mnn_corpus_prelu_120_fp32(const int n, const int channels, const int dim, const float* in, float* out,
                               const float* slopeData, int div_factor, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PRELU_120<float><<<grid, block, 0, stream>>>(n, channels, dim, in, out, slopeData, div_factor);
}

} // extern "C"
