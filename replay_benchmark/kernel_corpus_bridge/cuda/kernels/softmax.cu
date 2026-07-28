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

// ---- SOFTMAX_WARP_32 3.6.0 (2.4.2+ with exp cutoff) ----
// Used when axis <= 32. One warp processes one (outside, inside) pair.
template <typename T>
__global__ void SOFTMAX_WARP_32(const T *input, T *output,
    const int inside, const int axis, const int outside, const int count) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    auto src = input + idx_outside * axis * inside + idx_inside;
    float local_src = -FLT_MAX;
    __shared__ float maxValue;
    __shared__ float sumValue;
    int tid = threadIdx.x;
    if(tid < axis) { local_src = (float)(src[tid * inside]); }
    float maxRes = warpReduceMax<float>(local_src);
    if(tid == 0) maxValue = maxRes;
    __syncthreads();
    float local_exp = 0.0f;
    if(tid < axis) {
        float tmpSub = local_src - maxValue;
        tmpSub = ((tmpSub < -87.0) ? -87.0 : tmpSub);
        local_exp = exp(tmpSub);
    }
    float sumRes = warpReduceSum<float>(local_exp);
    if(tid == 0) sumValue = sumRes;
    __syncthreads();
    float divSumValue = 1.0 / sumValue;
    if(tid < axis) {
        output[(idx_outside * axis + tid) * inside + idx_inside] = (T)(local_exp * divSumValue);
    }
}

// ---- SOFTMAX_WARP_32 2.4.2 (no exp cutoff) ----
template <typename T>
__global__ void SOFTMAX_WARP_32_242(const T *input, T *output,
    const int inside, const int axis, const int outside, const int count) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    auto src = input + idx_outside * axis * inside + idx_inside;
    float local_src = -FLT_MAX;
    __shared__ float maxValue;
    __shared__ float sumValue;
    int tid = threadIdx.x;
    if(tid < axis) { local_src = (float)(src[tid * inside]); }
    float maxRes = warpReduceMax<float>(local_src);
    if(tid == 0) maxValue = maxRes;
    __syncthreads();
    float local_exp = 0.0f;
    if(tid < axis) { local_exp = exp(local_src - maxValue); }
    float sumRes = warpReduceSum<float>(local_exp);
    if(tid == 0) sumValue = sumRes;
    __syncthreads();
    float divSumValue = 1.0 / sumValue;
    if(tid < axis) {
        output[(idx_outside * axis + tid) * inside + idx_inside] = (T)(local_exp * divSumValue);
    }
}

// ---- SOFTMAX_AXIS_REDUCE 3.6.0 (2.4.2+ with exp cutoff) ----
// Used when axis % 256 == 0 || axis >= 768 (block=256) or axis % 64 == 0 || axis > 32 (block=64).
template <typename T>
__global__ void SOFTMAX_AXIS_REDUCE(const T *input, T *output,
    const int inside, const int axis, const int per_block_size,
    const int calc_multi_num, const int outside, const int count) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    auto src = input + idx_outside * axis * inside + idx_inside;
    auto dst = output + idx_outside * axis * inside + idx_inside;
    float local_src = -FLT_MAX;
    __shared__ float maxValue;
    __shared__ float sumValue;
    int tid = threadIdx.x;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            local_src = max(local_src, (float)(src[(tid + i * per_block_size) * inside]));
        }
    }
    float maxRes = blockReduceMax<float>(local_src);
    if(tid == 0) maxValue = maxRes;
    __syncthreads();
    float local_exp = 0.0f;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            float tmpSub = (float)(src[(tid + i * per_block_size) * inside]) - maxValue;
            tmpSub = ((tmpSub < -87.0) ? -87.0 : tmpSub);
            local_exp += exp(tmpSub);
        }
    }
    float sumRes = blockReduceSum<float>(local_exp);
    if(tid == 0) sumValue = sumRes;
    __syncthreads();
    float divSumValue = 1.0 / sumValue;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            float tmpSub = (float)(src[(tid + i * per_block_size) * inside]) - maxValue;
            tmpSub = ((tmpSub < -87.0) ? -87.0 : tmpSub);
            float tmp_exp = exp(tmpSub);
            dst[(tid + i * per_block_size) * inside] = (T)(tmp_exp * divSumValue);
        }
    }
}

// ---- SOFTMAX_AXIS_REDUCE 2.4.2 (no exp cutoff) ----
template <typename T>
__global__ void SOFTMAX_AXIS_REDUCE_242(const T *input, T *output,
    const int inside, const int axis, const int per_block_size,
    const int calc_multi_num, const int outside, const int count) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    auto src = input + idx_outside * axis * inside + idx_inside;
    auto dst = output + idx_outside * axis * inside + idx_inside;
    float local_src = -FLT_MAX;
    __shared__ float maxValue;
    __shared__ float sumValue;
    int tid = threadIdx.x;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            local_src = max(local_src, (float)(src[(tid + i * per_block_size) * inside]));
        }
    }
    float maxRes = blockReduceMax<float>(local_src);
    if(tid == 0) maxValue = maxRes;
    __syncthreads();
    float local_exp = 0.0f;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            local_exp += exp((float)(src[(tid + i * per_block_size) * inside]) - maxValue);
        }
    }
    float sumRes = blockReduceSum<float>(local_exp);
    if(tid == 0) sumValue = sumRes;
    __syncthreads();
    float divSumValue = 1.0 / sumValue;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            float tmp_exp = exp((float)(src[(tid + i * per_block_size) * inside]) - maxValue);
            dst[(tid + i * per_block_size) * inside] = (T)(tmp_exp * divSumValue);
        }
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

// ---- SOFTMAX_WARP_32 3.6.0 (axis <= 32, exp cutoff) ----
void mnn_corpus_softmax_warp32_fp32(const float* input, float* output, int inside, int axis, int outside, int count,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SOFTMAX_WARP_32<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside, count);
}
// ---- SOFTMAX_WARP_32 2.4.2 (no exp cutoff) ----
void mnn_corpus_softmax_warp32_242_fp32(const float* input, float* output, int inside, int axis, int outside, int count,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SOFTMAX_WARP_32_242<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside, count);
}

// ---- SOFTMAX_AXIS_REDUCE 3.6.0 (exp cutoff) ----
void mnn_corpus_softmax_axis_reduce_fp32(const float* input, float* output, int inside, int axis,
                                           int per_block_size, int calc_multi_num, int outside, int count,
                                           int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SOFTMAX_AXIS_REDUCE<float><<<grid, block, 0, stream>>>(input, output, inside, axis,
                                                                         per_block_size, calc_multi_num, outside, count);
}
// ---- SOFTMAX_AXIS_REDUCE 2.4.2 (no exp cutoff) ----
void mnn_corpus_softmax_axis_reduce_242_fp32(const float* input, float* output, int inside, int axis,
                                               int per_block_size, int calc_multi_num, int outside, int count,
                                               int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SOFTMAX_AXIS_REDUCE_242<float><<<grid, block, 0, stream>>>(input, output, inside, axis,
                                                                              per_block_size, calc_multi_num, outside, count);
}

} // extern "C"
