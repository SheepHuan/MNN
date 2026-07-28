// layernorm.cu - LAYERNORM 1.2.7/2.2.2/2.8.4-3.6.0 kernels + shims
//   source/backend/cuda/execution/LayerNormExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// LayerNorm 1.2.7: float* gamma/beta, float accumulation + (float) casts, no null check
// (source/backend/cuda/execution/LayerNormExecution.cu @ tag 1.2.7)
// ============================================================================
template <typename T>
__global__ void LAYERNORM_127(const int count, const int outside, const int inside, const float epsilon,
                               const T* in, T* out, const float* gamma_data, const float* beta_data) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int index = i % inside;
        const T* inner_input = in + o * inside;
        T* inner_output = out + o * inside;
        float sum = 0.f;
        for (int j = 0; j < inside; ++j) { sum += (float)inner_input[j]; }
        float mean = sum / inside;
        float square_sum = 0.f;
        for (int j = 0; j < inside; ++j) {
            square_sum += ((float)inner_input[j] - mean) * ((float)inner_input[j] - mean);
        }
        float variable = square_sum / inside;
        variable = 1.f / sqrt(variable + epsilon);
        inner_output[index] = (T)(((float)inner_input[index] - mean) * variable * gamma_data[index] + beta_data[index]);
    }
}

// ============================================================================
// LayerNorm 2.2.2: 1.2.7 + null check for gamma/beta
// (source/backend/cuda/execution/LayerNormExecution.cu @ tag 2.2.2)
// ============================================================================
template <typename T>
__global__ void LAYERNORM_222(const int count, const int outside, const int inside, const float epsilon,
                               const T* in, T* out, const float* gamma_data, const float* beta_data) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int index = i % inside;
        const T* inner_input = in + o * inside;
        T* inner_output = out + o * inside;
        float sum = 0.f;
        for (int j = 0; j < inside; ++j) { sum += (float)inner_input[j]; }
        float mean = sum / inside;
        float square_sum = 0.f;
        for (int j = 0; j < inside; ++j) {
            square_sum += ((float)inner_input[j] - mean) * ((float)inner_input[j] - mean);
        }
        float variable = square_sum / inside;
        variable = 1.f / sqrt(variable + epsilon);
        float res = ((float)inner_input[index] - mean) * variable;
        if (gamma_data != nullptr && beta_data != nullptr) {
            res = res * gamma_data[index] + beta_data[index];
        }
        inner_output[index] = (T)res;
    }
}

// ============================================================================
// LayerNorm 2.8.4 / 3.6.0: 2.2.2 + bool RMSNorm param + branch
// (source/backend/cuda/execution/LayerNormExecution.cu @ tag 2.8.4 / HEAD)
// ============================================================================
template <typename T>
__global__ void LAYERNORM(const int count, const int outside, const int inside, const float epsilon,
                          const T* in, T* out, const float* gamma_data, const float* beta_data, bool RMSNorm) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int index = i % inside;
        const T* inner_input = in + o * inside;
        T* inner_output = out + o * inside;
        float mean = 0.0f;
        if (!RMSNorm) {
            float sum = 0.f;
            for (int j = 0; j < inside; ++j) { sum += (float)inner_input[j]; }
            mean = sum / inside;
        }
        float square_sum = 0.f;
        for (int j = 0; j < inside; ++j) {
            square_sum += ((float)inner_input[j] - mean) * ((float)inner_input[j] - mean);
        }
        float variable = square_sum / inside;
        variable = 1.f / sqrt(variable + epsilon);
        float res = ((float)inner_input[index] - mean) * variable;
        if (gamma_data != nullptr && beta_data != nullptr) {
            res = res * gamma_data[index] + beta_data[index];
        }
        inner_output[index] = (T)res;
    }
}

// ============================================================================
// 1.2.0 tag: LAYERNORM (gamma/beta are T* not float*; no RMSNorm)
// ============================================================================
template <typename T>
__global__ void LAYERNORM_120(const int count, const int outside, const int inside, const float epsilon,
                               const T* in, T* out, const T* gamma_data, const T* beta_data) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int index = i % inside;
        const T* inner_input = in + o * inside;
        T* inner_output = out + o * inside;
        T mean = (T)0;
        T sum = (T)0;
        for (int j = 0; j < inside; ++j) sum += inner_input[j];
        mean = sum / (T)inside;
        T square_sum = (T)0;
        for (int j = 0; j < inside; ++j) square_sum += (inner_input[j] - mean) * (inner_input[j] - mean);
        T variable = square_sum / (T)inside;
        variable = (T)1 / sqrt(variable + (T)epsilon);
        T res = (inner_input[index] - mean) * variable;
        if (gamma_data != nullptr && beta_data != nullptr) {
            res = res * gamma_data[index] + beta_data[index];
        }
        inner_output[index] = res;
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- LayerNorm ----
void mnn_corpus_layernorm_fp32(const int count, const int outside, const int inside, const float epsilon,
                               const float* in, float* out, const float* gamma, const float* beta, bool RMSNorm,
                               int grid, int block, cudaStream_t stream) {
    MNN::Corpus::LAYERNORM<float><<<grid, block, 0, stream>>>(count, outside, inside, epsilon, in, out, gamma, beta,
                                                                RMSNorm);
}
void mnn_corpus_layernorm_fp16(const int count, const int outside, const int inside, const float epsilon,
                               const void* in, void* out, const float* gamma, const float* beta, bool RMSNorm,
                               int grid, int block, cudaStream_t stream) {
    MNN::Corpus::LAYERNORM<half><<<grid, block, 0, stream>>>(count, outside, inside, epsilon, (const half*)in,
                                                              (half*)out, gamma, beta, RMSNorm);
}
// ---- LayerNorm 1.2.7: float* gamma/beta, float acc, no null check ----
void mnn_corpus_layernorm_127_fp32(const int count, const int outside, const int inside, const float epsilon,
                                    const float* in, float* out, const float* gamma, const float* beta,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::LAYERNORM_127<float><<<grid, block, 0, stream>>>(count, outside, inside, epsilon, in, out, gamma, beta);
}
// ---- LayerNorm 2.2.2: float* gamma/beta, float acc, null check ----
void mnn_corpus_layernorm_222_fp32(const int count, const int outside, const int inside, const float epsilon,
                                    const float* in, float* out, const float* gamma, const float* beta,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::LAYERNORM_222<float><<<grid, block, 0, stream>>>(count, outside, inside, epsilon, in, out, gamma, beta);
}

// ---- 1.2.0 tag: LAYERNORM ----
void mnn_corpus_layernorm_120_fp32(const int count, const int outside, const int inside, const float epsilon,
                                   const float* in, float* out, const float* gamma, const float* beta,
                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::LAYERNORM_120<float><<<grid, block, 0, stream>>>(count, outside, inside, epsilon, in, out, gamma, beta);
}

} // extern "C"
