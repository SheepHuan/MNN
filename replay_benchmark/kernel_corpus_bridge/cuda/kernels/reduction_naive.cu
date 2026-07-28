// reduction_naive.cu - SUM_NAIVE / MEAN_NAIVE / MINIMUM / MAXIMUM / PROD (3.6.0)
//   + SUM_120 / MEAN_120 (1.2.0) kernels + shims
//   source/backend/cuda/execution/ReductionTemplate.cuh
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// Reduction: source/backend/cuda/execution/ReductionTemplate.cuh
// ============================================================================
template <typename T>
__global__ void SUM_NAIVE(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        float sumValue = 0.0f;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue += (float)basicInput[v * inside];
        output[y * inside + x] = (T)sumValue;
    }
}
template <typename T>
__global__ void MEAN_NAIVE(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        float sumValue = 0.0f;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue += (float)basicInput[v * inside];
        output[y * inside + x] = (T)(sumValue / (float)axis);
    }
}
template <typename T>
__global__ void MINIMUM(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        const T* basicInput = input + y * axis * inside + x;
        float res = (float)basicInput[0];
        for (int v = 1; v < axis; ++v) res = min((float)basicInput[v * inside], res);
        output[y * inside + x] = (T)res;
    }
}
template <typename T>
__global__ void MAXIMUM(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        const T* basicInput = input + y * axis * inside + x;
        float res = (float)basicInput[0];
        for (int v = 1; v < axis; ++v) res = max((float)basicInput[v * inside], res);
        output[y * inside + x] = (T)res;
    }
}
template <typename T>
__global__ void PROD(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        float sumValue = 1.0f;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue *= (float)basicInput[v * inside];
        output[y * inside + x] = (T)sumValue;
    }
}

// ============================================================================
// 1.2.0 tag: Reduction (T accumulation, param order: inside, axis, outside)
// ============================================================================
template <typename T>
__global__ void SUM_120(const T* input, T* output, int inside, int axis, int outside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        T sumValue = (T)0;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue += basicInput[v * inside];
        output[y * inside + x] = sumValue;
    }
}
template <typename T>
__global__ void MEAN_120(const T* input, T* output, int inside, int axis, int outside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        T sumValue = (T)0;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue += basicInput[v * inside];
        output[y * inside + x] = sumValue / (T)axis;
    }
}

// ---- Reduction 1.2.7: ReduceParam struct, float accumulator ----
template <typename T>
__global__ void SUM_127(const T* input, T* output, const ReduceParam_127* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        int y = i / param->inside, x = i % param->inside;
        float sumValue = 0.0;
        const T* basicInput = input + y * param->axis * param->inside + x;
        for (int v = 0; v < param->axis; ++v) sumValue += (float)basicInput[v * param->inside];
        output[y * param->inside + x] = (T)sumValue;
    }
}
template <typename T>
__global__ void MEAN_127(const T* input, T* output, const ReduceParam_127* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        int y = i / param->inside, x = i % param->inside;
        float sumValue = 0.0;
        const T* basicInput = input + y * param->axis * param->inside + x;
        for (int v = 0; v < param->axis; ++v) sumValue += (float)basicInput[v * param->inside];
        output[y * param->inside + x] = (T)(sumValue / (float)param->axis);
    }
}
template <typename T>
__global__ void MAXIMUM_127(const T* input, T* output, const ReduceParam_127* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        int y = i / param->inside, x = i % param->inside;
        const T* basicInput = input + y * param->axis * param->inside + x;
        float res = (float)basicInput[0];
        for (int v = 1; v < param->axis; ++v) res = max((float)basicInput[v * param->inside], res);
        output[y * param->inside + x] = (T)res;
    }
}
template <typename T>
__global__ void MINIMUM_127(const T* input, T* output, const ReduceParam_127* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        int y = i / param->inside, x = i % param->inside;
        const T* basicInput = input + y * param->axis * param->inside + x;
        float res = (float)basicInput[0];
        for (int v = 1; v < param->axis; ++v) res = min((float)basicInput[v * param->inside], res);
        output[y * param->inside + x] = (T)res;
    }
}
template <typename T>
__global__ void PROD_127(const T* input, T* output, const ReduceParam_127* param) {
    int count = param->inside * param->outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        int y = i / param->inside, x = i % param->inside;
        float sumValue = 1.0;
        const T* basicInput = input + y * param->axis * param->inside + x;
        for (int v = 0; v < param->axis; ++v) sumValue *= (float)basicInput[v * param->inside];
        output[y * param->inside + x] = (T)sumValue;
    }
}

// ---- SUM_REDUCE_AXIS / MEAN_REDUCE_AXIS (2.5.1+, axis >= 32 block reduce) ----
// Distinct variant: blockReduceSum to collapse the axis dim in one block per
// (outside, inside). Body identical 2.5.1 through 3.6.0 (one shim, no tag diff).
template <typename T>
__global__ void SUM_REDUCE_AXIS(const T* input, T* output, const int outside, const int axis, const int inside,
                                const int per_block_size, const int calc_multi_num) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;
    float local_src = 0.0f;
    __shared__ float sumValue;
    for (int i = 0; i < calc_multi_num; ++i) {
        if (tid + i * per_block_size < axis) {
            local_src += (float)src[(tid + i * per_block_size) * inside];
        }
    }
    float maxRes = blockReduceSum<float>(local_src);
    if (tid == 0) sumValue = maxRes;
    __syncthreads();
    output[idx_outside * inside + idx_inside] = (T)sumValue;
}
template <typename T>
__global__ void MEAN_REDUCE_AXIS(const T* input, T* output, const int outside, const int axis, const int inside,
                                 const int per_block_size, const int calc_multi_num) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;
    float local_src = 0.0f;
    __shared__ float sumValue;
    for (int i = 0; i < calc_multi_num; ++i) {
        if (tid + i * per_block_size < axis) {
            local_src += (float)src[(tid + i * per_block_size) * inside];
        }
    }
    float maxRes = blockReduceSum<float>(local_src);
    if (tid == 0) sumValue = maxRes;
    __syncthreads();
    output[idx_outside * inside + idx_inside] = (T)(sumValue / (float)axis);
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Reduction (SUM/MAX/MIN/MEAN/PROD naive) ----
void mnn_corpus_reduction_sum_fp32(const float* input, float* output, int outside, int axis, int inside,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SUM_NAIVE<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}
void mnn_corpus_reduction_mean_fp32(const float* input, float* output, int outside, int axis, int inside,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MEAN_NAIVE<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}
void mnn_corpus_reduction_max_fp32(const float* input, float* output, int outside, int axis, int inside,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MAXIMUM<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}
void mnn_corpus_reduction_min_fp32(const float* input, float* output, int outside, int axis, int inside,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MINIMUM<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}
void mnn_corpus_reduction_prod_fp32(const float* input, float* output, int outside, int axis, int inside,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PROD<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}

// ---- 1.2.0 tag: Reduction (T accumulation, param order: inside, axis, outside) ----
void mnn_corpus_reduction_sum_120_fp32(const float* input, float* output, int inside, int axis, int outside,
                                       int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SUM_120<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside);
}
void mnn_corpus_reduction_mean_120_fp32(const float* input, float* output, int inside, int axis, int outside,
                                        int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MEAN_120<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside);
}

// ---- Reduction 1.2.7 (ReduceParam struct) ----
void mnn_corpus_reduction_sum_127_fp32(const float* input, float* output, const MNN::Corpus::ReduceParam_127* param,
                                       int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SUM_127<float><<<grid, block, 0, stream>>>(input, output, param);
}
void mnn_corpus_reduction_mean_127_fp32(const float* input, float* output, const MNN::Corpus::ReduceParam_127* param,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MEAN_127<float><<<grid, block, 0, stream>>>(input, output, param);
}
void mnn_corpus_reduction_max_127_fp32(const float* input, float* output, const MNN::Corpus::ReduceParam_127* param,
                                        int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MAXIMUM_127<float><<<grid, block, 0, stream>>>(input, output, param);
}
void mnn_corpus_reduction_min_127_fp32(const float* input, float* output, const MNN::Corpus::ReduceParam_127* param,
                                        int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MINIMUM_127<float><<<grid, block, 0, stream>>>(input, output, param);
}
void mnn_corpus_reduction_prod_127_fp32(const float* input, float* output, const MNN::Corpus::ReduceParam_127* param,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PROD_127<float><<<grid, block, 0, stream>>>(input, output, param);
}

// ---- SUM_REDUCE_AXIS / MEAN_REDUCE_AXIS (axis variant, 2.5.1+) ----
void mnn_corpus_reduction_sum_axis_fp32(const float* input, float* output, int outside, int axis, int inside,
                                        int per_block_size, int calc_multi_num, int grid, int block,
                                        cudaStream_t stream) {
    MNN::Corpus::SUM_REDUCE_AXIS<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside,
                                                                    per_block_size, calc_multi_num);
}
void mnn_corpus_reduction_mean_axis_fp32(const float* input, float* output, int outside, int axis, int inside,
                                         int per_block_size, int calc_multi_num, int grid, int block,
                                         cudaStream_t stream) {
    MNN::Corpus::MEAN_REDUCE_AXIS<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside,
                                                                     per_block_size, calc_multi_num);
}

} // extern "C"
