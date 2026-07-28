// binary.cu - ATAN2/MOD/LOGICALOR kernels + shims
//   source/backend/cuda/execution/BinaryExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// Binary: source/backend/cuda/execution/BinaryExecution.cu
// ============================================================================
template <typename T>
__global__ void ATAN2(const T* input0, const T* input1, T* output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = atan2f(x, y);
    }
}
template <typename T>
__global__ void MOD(const T* input0, const T* input1, T* output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = x - x / y;
    }
}
template <typename T>
__global__ void LOGICALOR(const T* input0, const T* input1, T* output, size_t count, size_t s0, size_t s1) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        T x = input0[i * s0];
        T y = input1[i * s1];
        output[i] = (x || y) ? 1 : 0;
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Binary ----
void mnn_corpus_atan2_fp32(const float* in0, const float* in1, float* out, size_t count, size_t s0, size_t s1,
                           int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ATAN2<float><<<grid, block, 0, stream>>>(in0, in1, out, count, s0, s1);
}
void mnn_corpus_atan2_fp16(const void* in0, const void* in1, void* out, size_t count, size_t s0, size_t s1,
                           int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ATAN2<half><<<grid, block, 0, stream>>>((const half*)in0, (const half*)in1, (half*)out, count, s0, s1);
}
void mnn_corpus_mod_fp32(const float* in0, const float* in1, float* out, size_t count, size_t s0, size_t s1,
                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MOD<float><<<grid, block, 0, stream>>>(in0, in1, out, count, s0, s1);
}
void mnn_corpus_mod_fp16(const void* in0, const void* in1, void* out, size_t count, size_t s0, size_t s1,
                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MOD<half><<<grid, block, 0, stream>>>((const half*)in0, (const half*)in1, (half*)out, count, s0, s1);
}
void mnn_corpus_logicalor_fp32(const float* in0, const float* in1, float* out, size_t count, size_t s0, size_t s1,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::LOGICALOR<float><<<grid, block, 0, stream>>>(in0, in1, out, count, s0, s1);
}
void mnn_corpus_logicalor_fp16(const void* in0, const void* in1, void* out, size_t count, size_t s0, size_t s1,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::LOGICALOR<half><<<grid, block, 0, stream>>>((const half*)in0, (const half*)in1, (half*)out, count, s0, s1);
}

} // extern "C"
