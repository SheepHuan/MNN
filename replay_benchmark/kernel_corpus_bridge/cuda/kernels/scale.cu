// scale.cu - SCALE 2.0.4/3.6.0 + 1.2.7 kernels + shims
//   source/backend/cuda/execution/ScaleExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// Scale: source/backend/cuda/execution/ScaleExecution.cu
// ============================================================================
template <typename T>
__global__ void SCALE(const int total, const int channelsPack, const int dim, const T* in, T* out,
                       const float* scaleData, const float* biasData) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx = index / channelsPack;
        int c_idx = index % channelsPack;
        out[index] = (T)((float)in[index] * scaleData[c_idx] + biasData[c_idx]);
    }
}

// ---- SCALE 1.2.7: float* scale/bias, PACK_NUMBER channel packing ----
template <typename T>
__global__ void SCALE_127(const int n, const int channels, const int dim, const T* in, T* out,
                          const float* scaleData, const float* biasData) {
    const int PACK_NUMBER = 4;
    CUDA_KERNEL_LOOP(count, n) {
        int index = count / PACK_NUMBER;
        int r = count % PACK_NUMBER;
        int c = (index / dim) * PACK_NUMBER + r;
        out[count] = (T)((float)in[count] * scaleData[c] + biasData[c]);
    }
}

// ============================================================================
// 1.2.0 tag: SCALE (scale/bias are T* not float*)
// ============================================================================
template <typename T>
__global__ void SCALE_120(const int n, const int channels, const int dim, const T* in, T* out,
                           const T* scaleData, const T* biasData) {
    CUDA_KERNEL_LOOP(index, n) {
        int c = (index / dim) % channels;
        out[index] = in[index] * scaleData[c] + biasData[c];
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Scale ----
void mnn_corpus_scale_fp32(const int total, const int channelsPack, const int dim, const float* in, float* out,
                            const float* scaleData, const float* biasData, int grid, int block,
                            cudaStream_t stream) {
    MNN::Corpus::SCALE<float><<<grid, block, 0, stream>>>(total, channelsPack, dim, in, out, scaleData, biasData);
}
void mnn_corpus_scale_fp16(const int total, const int channelsPack, const int dim, const void* in, void* out,
                            const float* scaleData, const float* biasData, int grid, int block,
                            cudaStream_t stream) {
    MNN::Corpus::SCALE<half><<<grid, block, 0, stream>>>(total, channelsPack, dim, (const half*)in, (half*)out,
                                                          scaleData, biasData);
}

// ---- SCALE 1.2.7 ----
void mnn_corpus_scale_127_fp32(const int n, const int channels, const int dim, const float* in, float* out,
                                const float* scaleData, const float* biasData, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SCALE_127<float><<<grid, block, 0, stream>>>(n, channels, dim, in, out, scaleData, biasData);
}

// ---- 1.2.0 tag: SCALE ----
void mnn_corpus_scale_120_fp32(const int n, const int channels, const int dim, const float* in, float* out,
                                const float* scaleData, const float* biasData, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SCALE_120<float><<<grid, block, 0, stream>>>(n, channels, dim, in, out, scaleData, biasData);
}

} // extern "C"
