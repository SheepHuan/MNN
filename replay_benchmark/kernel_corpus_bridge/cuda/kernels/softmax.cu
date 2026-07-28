// softmax.cu - SOFTMAX (3.6.0) + SOFTMAX_222 (2.2.2) kernels + shims
//   source/backend/cuda/execution/SoftmaxExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// Softmax: source/backend/cuda/execution/SoftmaxExecution.cu
// ============================================================================
template <typename T>
__global__ void SOFTMAX(const T* input, T* output, const int inside, const int axis, const int outside,
                        const int count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int y = i / inside;
        int x = i % inside;
        const T* src = input + y * axis * inside + x;
        T* dst = output + y * axis * inside + x;
        float maxValue = (float)src[0];
        for (int z = 1; z < axis; ++z) { maxValue = max(maxValue, (float)src[z * inside]); }
        float sumValue = 0.0;
        for (int z = 0; z < axis; ++z) {
            float tmpSub = (float)src[z * inside] - maxValue;
            tmpSub = ((tmpSub < -87.0) ? -87.0 : tmpSub);
            sumValue = sumValue + exp(tmpSub);
        }
        sumValue = 1.0 / sumValue;
        for (int z = 0; z < axis; ++z) {
            float tmpSub = (float)src[z * inside] - maxValue;
            tmpSub = ((tmpSub < -87.0) ? -87.0 : tmpSub);
            dst[z * inside] = (T)(exp(tmpSub) * sumValue);
        }
    }
}

// ---- SOFTMAX 2.2.2: ReduceParam struct ----
template <typename T>
__global__ void SOFTMAX_222(const T* input, T* output, const ReduceParam_127* param) {
    int inside = param->inside, axis = param->axis, outside = param->outside;
    int count = inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        int y = i / inside, x = i % inside;
        const T* src = input + y * axis * inside + x;
        T* dst = output + y * axis * inside + x;
        float maxValue = (float)src[0];
        for (int z = 1; z < axis; ++z) maxValue = max(maxValue, (float)src[z * inside]);
        float sumValue = 0.0;
        for (int z = 0; z < axis; ++z) sumValue += exp((float)src[z * inside] - maxValue);
        sumValue = 1.0 / sumValue;
        for (int z = 0; z < axis; ++z) dst[z * inside] = (T)(exp((float)src[z * inside] - maxValue) * sumValue);
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Softmax ----
void mnn_corpus_softmax_fp32(const float* input, float* output, int inside, int axis, int outside, int count,
                              int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SOFTMAX<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside, count);
}
void mnn_corpus_softmax_fp16(const void* input, void* output, int inside, int axis, int outside, int count,
                              int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SOFTMAX<half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, inside, axis, outside,
                                                             count);
}

// ---- SOFTMAX 2.2.2 (ReduceParam) ----
void mnn_corpus_softmax_222_fp32(const float* input, float* output, const MNN::Corpus::ReduceParam_127* param,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SOFTMAX_222<float><<<grid, block, 0, stream>>>(input, output, param);
}

} // extern "C"
