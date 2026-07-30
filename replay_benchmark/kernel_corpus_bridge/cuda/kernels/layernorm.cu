// layernorm.cu - LAYERNORM 1.2.7/2.2.2/2.8.4-3.6.0 kernels + shims
//   source/backend/cuda/execution/LayerNormExecution.cu
#include "corpus_common.cuh"
#include <cuda_fp16.h>

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

// ---- layernorm_c4 (3.6.0, C4 packed, block-per-row with blockReduceSum) ----
template <typename T>
__global__ void layernorm_c4(T* output, const T* input, const float* gamma, const float* beta, int inside,
                             int rowStride, float epsilon, bool RMSNorm) {
    const int tid = threadIdx.x;
    const int base = blockIdx.x * rowStride;
    __shared__ float sMean;
    __shared__ float sVariance;

    float localSum = 0.0f;
    float localSquareSum = 0.0f;
    for (int i = tid; i < inside; i += blockDim.x) {
        float value = (float)input[base + i];
        localSum += value;
        localSquareSum += value * value;
    }
    float sum = blockReduceSum<float>(localSum);
    if (tid == 0) {
        sMean = RMSNorm ? 0.0f : sum / inside;
    }
    __syncthreads();
    float squareSum = blockReduceSum<float>(localSquareSum);
    if (tid == 0) {
        float squareMean = squareSum / inside;
        sVariance = (RMSNorm ? squareMean : fmaxf(squareMean - sMean * sMean, 0.0f)) + epsilon;
    }
    __syncthreads();

    float invStd = rsqrtf(sVariance);
    for (int i = tid; i < inside; i += blockDim.x) {
        float result = ((float)input[base + i] - sMean) * invStd;
        if (gamma != nullptr && beta != nullptr) {
            result = result * __ldg(gamma + i) + __ldg(beta + i);
        }
        output[base + i] = (T)result;
    }
}

// ---- binary_layernorm_c4 (3.6.0, fuses binary add + layernorm C4) ----
template <typename T>
__global__ void binary_layernorm_c4(T* sumOut, T* normOut, const T* input0, const T* input1, const float* gamma,
                                    const float* beta, int inside, int rowStride, float epsilon, bool RMSNorm) {
    const int tid = threadIdx.x;
    const int base = blockIdx.x * rowStride;
    __shared__ float sMean;
    __shared__ float sVariance;

    float localSum = 0.0f;
    float localSquareSum = 0.0f;
    for (int i = tid; i < inside; i += blockDim.x) {
        float value = (float)input0[base + i] + (float)input1[base + i];
        localSum += value;
        localSquareSum += value * value;
    }
    float sum = blockReduceSum<float>(localSum);
    if (tid == 0) {
        sMean = RMSNorm ? 0.0f : sum / inside;
    }
    __syncthreads();
    float squareSum = blockReduceSum<float>(localSquareSum);
    if (tid == 0) {
        float squareMean = squareSum / inside;
        sVariance = (RMSNorm ? squareMean : fmaxf(squareMean - sMean * sMean, 0.0f)) + epsilon;
    }
    __syncthreads();

    float invStd = rsqrtf(sVariance);
    for (int i = tid; i < inside; i += blockDim.x) {
        float value = (float)input0[base + i] + (float)input1[base + i];
        float result = (value - sMean) * invStd;
        if (gamma != nullptr && beta != nullptr) {
            result = result * __ldg(gamma + i) + __ldg(beta + i);
        }
        sumOut[base + i] = (T)value;
        normOut[base + i] = (T)result;
    }
}

// ============================================================================
// input_layernorm_<size>: size-specialized LayerNorm/RMSNorm kernels.
// Faithful copies from source/backend/cuda/execution/LayerNormExecution.cu.
// Each block handles one row; gamma/beta are float; T is float or half.
// NOTE: MNN uses `#pragma unroll(N)` which nvcc accepts; the standard form
// `#pragma unroll N` is used here for portability.
// ============================================================================

template <typename T>
__global__ void input_layernorm_320(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm) {
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out = 0.0f;
    s_mean = 0;
    float value_tmp[5];
    value_tmp[0] = input[blockIdx.x * n + 0*64 + tid];
    value_tmp[1] = input[blockIdx.x * n + 1*64 + tid];
    value_tmp[2] = input[blockIdx.x * n + 2*64 + tid];
    value_tmp[3] = input[blockIdx.x * n + 3*64 + tid];
    value_tmp[4] = input[blockIdx.x * n + 4*64 + tid];
    if (!RMSNorm) {
        for (int idx = 0; idx < 5; idx++) local_out += value_tmp[idx];
        mean = blockReduceSum<float>(local_out);
        if (threadIdx.x == 0) s_mean = mean / n;
        __syncthreads();
    }
    mean = s_mean;
    float var_tmp = 0.0f;
    for (int idx = 0; idx < 5; idx++) var_tmp += ((value_tmp[idx] - mean) * (value_tmp[idx] - mean));
    variance = blockReduceSum<float>(var_tmp);
    if (threadIdx.x == 0) s_variance = variance / n + epsilon;
    __syncthreads();
    for (int idx = 0; idx < 5; idx++) {
        float res = ((value_tmp[idx] - mean) * rsqrtf(s_variance));
        if (gamma != nullptr && beta != nullptr) {
            res = res * (float)(__ldg(&gamma[idx*64 + tid])) + (float)(__ldg(&beta[idx*64 + tid]));
        }
        out[blockIdx.x * n + idx*64 + tid] = (T)res;
    }
}

template <typename T>
__global__ void input_layernorm_2048(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm) {
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out = 0.0f;
    s_mean = 0;
    float value_tmp[8];
    value_tmp[0] = input[blockIdx.x * 2048 + 0*256 + tid];
    value_tmp[1] = input[blockIdx.x * 2048 + 1*256 + tid];
    value_tmp[2] = input[blockIdx.x * 2048 + 2*256 + tid];
    value_tmp[3] = input[blockIdx.x * 2048 + 3*256 + tid];
    value_tmp[4] = input[blockIdx.x * 2048 + 4*256 + tid];
    value_tmp[5] = input[blockIdx.x * 2048 + 5*256 + tid];
    value_tmp[6] = input[blockIdx.x * 2048 + 6*256 + tid];
    value_tmp[7] = input[blockIdx.x * 2048 + 7*256 + tid];
    if (!RMSNorm) {
        #pragma unroll 8
        for (int idx = 0; idx < 8; idx++) local_out += (float)value_tmp[idx];
        mean = blockReduceSum<float>(local_out);
        if (threadIdx.x == 0) s_mean = mean / n;
        __syncthreads();
    }
    mean = s_mean;
    float var_tmp = 0.0f;
    #pragma unroll 8
    for (int idx = 0; idx < 8; idx++) var_tmp += ((value_tmp[idx] - mean) * (value_tmp[idx] - mean));
    variance = blockReduceSum<float>(var_tmp);
    if (threadIdx.x == 0) s_variance = variance / n + epsilon;
    __syncthreads();
    #pragma unroll 8
    for (int idx = 0; idx < 8; idx++) {
        float res = ((value_tmp[idx] - mean) * rsqrtf(s_variance));
        if (gamma != nullptr && beta != nullptr) {
            res = res * (float)(__ldg(&gamma[idx*256 + tid])) + (float)(__ldg(&beta[idx*256 + tid]));
        }
        out[blockIdx.x * 2048 + idx*256 + tid] = (T)res;
    }
}

template <typename T>
__global__ void input_layernorm_1024(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm) {
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out = 0.0f;
    s_mean = 0;
    float value_tmp[4];
    value_tmp[0] = input[blockIdx.x * 1024 + 0*256 + tid];
    value_tmp[1] = input[blockIdx.x * 1024 + 1*256 + tid];
    value_tmp[2] = input[blockIdx.x * 1024 + 2*256 + tid];
    value_tmp[3] = input[blockIdx.x * 1024 + 3*256 + tid];
    if (!RMSNorm) {
        #pragma unroll 4
        for (int idx = 0; idx < 4; idx++) local_out += (float)value_tmp[idx];
        mean = blockReduceSum<float>(local_out);
        if (threadIdx.x == 0) s_mean = mean / n;
        __syncthreads();
    }
    mean = s_mean;
    float var_tmp = 0.0f;
    #pragma unroll 4
    for (int idx = 0; idx < 4; idx++) var_tmp += ((value_tmp[idx] - mean) * (value_tmp[idx] - mean));
    variance = blockReduceSum<float>(var_tmp);
    if (threadIdx.x == 0) s_variance = variance / n + epsilon;
    __syncthreads();
    #pragma unroll 4
    for (int idx = 0; idx < 4; idx++) {
        float res = ((value_tmp[idx] - mean) * rsqrtf(s_variance));
        if (gamma != nullptr && beta != nullptr) {
            res = res * (float)(__ldg(&gamma[idx*256 + tid])) + (float)(__ldg(&beta[idx*256 + tid]));
        }
        out[blockIdx.x * 1024 + idx*256 + tid] = (T)res;
    }
}

template <typename T>
__global__ void input_layernorm_512(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm) {
    int tid = threadIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float local_out = 0.0f;
    s_mean = 0;
    float value_tmp[2];
    value_tmp[0] = input[blockIdx.x * 512 + 0*256 + tid];
    value_tmp[1] = input[blockIdx.x * 512 + 1*256 + tid];
    if (!RMSNorm) {
        local_out += (float)value_tmp[0];
        local_out += (float)value_tmp[1];
        mean = blockReduceSum<float>(local_out);
        if (threadIdx.x == 0) s_mean = mean / n;
        __syncthreads();
    }
    mean = s_mean;
    float var_tmp = 0.0f;
    var_tmp += ((value_tmp[0] - mean) * (value_tmp[0] - mean));
    var_tmp += ((value_tmp[1] - mean) * (value_tmp[1] - mean));
    variance = blockReduceSum<float>(var_tmp);
    if (threadIdx.x == 0) s_variance = variance / n + epsilon;
    __syncthreads();
    float res0 = ((value_tmp[0] - mean) * rsqrtf(s_variance));
    float res1 = ((value_tmp[1] - mean) * rsqrtf(s_variance));
    if (gamma != nullptr && beta != nullptr) {
        res0 = res0 * (float)(__ldg(&gamma[0*256 + tid])) + (float)(__ldg(&beta[0*256 + tid]));
        res1 = res1 * (float)(__ldg(&gamma[1*256 + tid])) + (float)(__ldg(&beta[1*256 + tid]));
    }
    out[blockIdx.x * 512 + 0*256 + tid] = (T)res0;
    out[blockIdx.x * 512 + 1*256 + tid] = (T)res1;
}

template <typename T>
__global__ void input_layernorm_adaptive(T* out, const T* input, const float* gamma, const float* beta, int m, int n, const float epsilon, bool RMSNorm) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int row = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    s_mean = 0.0f;
    if (!RMSNorm) {
        float local_sum = 0.0f;
        for (int i = tid; i < n; i += num_threads) local_sum += (float)input[row * n + i];
        mean = blockReduceSum<float>(local_sum);
        if (tid == 0) s_mean = mean / n;
        __syncthreads();
    }
    mean = s_mean;
    float var_sum = 0.0f;
    for (int i = tid; i < n; i += num_threads) {
        float val = (float)input[row * n + i] - mean;
        var_sum += val * val;
    }
    float variance = blockReduceSum<float>(var_sum);
    if (tid == 0) s_variance = variance / n + epsilon;
    __syncthreads();
    float inv_std = rsqrtf(s_variance);
    for (int i = tid; i < n; i += num_threads) {
        float res = ((float)input[row * n + i] - mean) * inv_std;
        if (gamma != nullptr && beta != nullptr) {
            res = res * (float)__ldg(&gamma[i]) + (float)__ldg(&beta[i]);
        }
        out[row * n + i] = (T)res;
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

// ---- layernorm_c4 (C4 variant, 3.6.0) ----
void mnn_corpus_layernorm_c4_fp32(float* output, const float* input, const float* gamma, const float* beta,
                                  int inside, int rowStride, float epsilon, bool RMSNorm,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::layernorm_c4<float><<<grid, block, 0, stream>>>(output, input, gamma, beta,
                                                                  inside, rowStride, epsilon, RMSNorm);
}

// ---- binary_layernorm_c4 (fused binary add + layernorm C4, 3.6.0) ----
void mnn_corpus_binary_layernorm_c4_fp32(float* sumOut, float* normOut,
                                         const float* input0, const float* input1,
                                         const float* gamma, const float* beta,
                                         int inside, int rowStride, float epsilon, bool RMSNorm,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::binary_layernorm_c4<float><<<grid, block, 0, stream>>>(
        sumOut, normOut, input0, input1, gamma, beta, inside, rowStride, epsilon, RMSNorm);
}

// ============================================================================
// fp16 (<half>) variants — layernorm_c4 / binary_layernorm_c4 (3.6.0)
// gamma/beta stay float (kernel signature uses const float*).
// ============================================================================
void mnn_corpus_layernorm_c4_fp16(void* output, const void* input, const float* gamma, const float* beta,
                                   int inside, int rowStride, float epsilon, bool RMSNorm,
                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::layernorm_c4<__half><<<grid, block, 0, stream>>>((__half*)output, (const __half*)input, gamma, beta,
                                                                   inside, rowStride, epsilon, RMSNorm);
}
void mnn_corpus_binary_layernorm_c4_fp16(void* sumOut, void* normOut,
                                         const void* input0, const void* input1,
                                         const float* gamma, const float* beta,
                                         int inside, int rowStride, float epsilon, bool RMSNorm,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::binary_layernorm_c4<__half><<<grid, block, 0, stream>>>(
        (__half*)sumOut, (__half*)normOut, (const __half*)input0, (const __half*)input1, gamma, beta,
        inside, rowStride, epsilon, RMSNorm);
}

// ---- input_layernorm_<size> (size-specialized, fp32; faithful to MNN) ----
// grid = m (rows), block per MNN dispatch: 320→64, 512/1024/2048→256, adaptive→caller-chosen.
void mnn_corpus_input_layernorm_320_fp32(float* out, const float* input, const float* gamma, const float* beta,
                                         int m, int n, float epsilon, bool RMSNorm,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::input_layernorm_320<float><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n, epsilon, RMSNorm);
}
void mnn_corpus_input_layernorm_512_fp32(float* out, const float* input, const float* gamma, const float* beta,
                                         int m, int n, float epsilon, bool RMSNorm,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::input_layernorm_512<float><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n, epsilon, RMSNorm);
}
void mnn_corpus_input_layernorm_1024_fp32(float* out, const float* input, const float* gamma, const float* beta,
                                          int m, int n, float epsilon, bool RMSNorm,
                                          int grid, int block, cudaStream_t stream) {
    MNN::Corpus::input_layernorm_1024<float><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n, epsilon, RMSNorm);
}
void mnn_corpus_input_layernorm_2048_fp32(float* out, const float* input, const float* gamma, const float* beta,
                                          int m, int n, float epsilon, bool RMSNorm,
                                          int grid, int block, cudaStream_t stream) {
    MNN::Corpus::input_layernorm_2048<float><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n, epsilon, RMSNorm);
}
void mnn_corpus_input_layernorm_adaptive_fp32(float* out, const float* input, const float* gamma, const float* beta,
                                              int m, int n, float epsilon, bool RMSNorm,
                                              int grid, int block, cudaStream_t stream) {
    MNN::Corpus::input_layernorm_adaptive<float><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n, epsilon, RMSNorm);
}

} // extern "C"
