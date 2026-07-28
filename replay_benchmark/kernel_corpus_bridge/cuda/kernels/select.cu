// select.cu - SELECT (3.6.0) + SELECT_272 (2.7.2) kernels + shims
//   source/backend/cuda/execution/SelectExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// Select: source/backend/cuda/execution/SelectExecution.cu
// ============================================================================
template <typename T>
__global__ void SELECT(const int size, const int* input0, const T* input1, const T* input2,
                        int s1, int s2, T* output) {
    CUDA_KERNEL_LOOP(i, size) {
        if (input0[i] > 0) {
            output[i] = input1[i * s1];
        } else {
            output[i] = input2[i * s2];
        }
    }
}

// ---- SELECT 2.7.2: no stride params ----
template <typename T>
__global__ void SELECT_272(const int size, const int* input0, const T* input1, const T* input2, T* output) {
    CUDA_KERNEL_LOOP(i, size) {
        output[i] = (input0[i] > 0) ? input1[i] : input2[i];
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Select ----
void mnn_corpus_select_fp32(const int size, const int* sel, const float* in1, const float* in2, int s1, int s2,
                             float* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SELECT<float><<<grid, block, 0, stream>>>(size, sel, in1, in2, s1, s2, output);
}
void mnn_corpus_select_fp16(const int size, const int* sel, const void* in1, const void* in2, int s1, int s2,
                             void* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SELECT<half><<<grid, block, 0, stream>>>(size, sel, (const half*)in1, (const half*)in2, s1, s2,
                                                            (half*)output);
}

// ---- SELECT 2.7.2 (no stride) ----
void mnn_corpus_select_272_fp32(const int size, const int* sel, const float* in1, const float* in2,
                                 float* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SELECT_272<float><<<grid, block, 0, stream>>>(size, sel, in1, in2, output);
}

} // extern "C"
