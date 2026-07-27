//
//  CorpusKernels.cu
//  MNN replay_benchmark
//
//  Self-contained CUDA kernel corpus for replay_benchmark. Each kernel here
//  is a verbatim re-implementation of the corresponding MNN __global__
//  kernel (same formula, same parameter layout) so the corpus runner can
//  launch it without depending on MNN_Cuda_Main's internal symbol
//  visibility (CUDA __global__ host stubs are always internal and cannot be
//  referenced across translation units). This file is compiled by nvcc
//  (via cuda_add_library in replay_benchmark/CMakeLists.txt) and linked
//  into replay_benchmark.out directly.
//
//  Keeping a second copy here is intentional: the corpus is an isolated
//  single-operator test body, not a participant in MNN inference. Any
//  semantic change to the original kernel must be mirrored here manually.
//  The source file of record is noted in each kernel's comment.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <float.h>

namespace MNN {
namespace Corpus {

// ============================================================================
// Device helpers (inlined from source/backend/cuda/execution/MNNCUDAFunction.cuh)
// ============================================================================
#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum<T>(val);
    if (lane == 0) { shared[wid] = val; }
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = warpReduceSum(val);
    return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;
    val = warpReduceMax<T>(val);
    if (lane == 0) { shared[wid] = val; }
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(-1.0e30f);
    val = warpReduceMax(val);
    return val;
}

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#define HALF_MIN half(-65504)

// ============================================================================
// Unary: source/backend/cuda/execution/UnaryExecution.cu
// ============================================================================

// UnaryExecution.cu:55  out[i] = x > 0 ? x : x * slope
__global__ void RELU(const float* input, float* output, size_t count, float slope) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        float x = input[i];
        output[i] = x > 0 ? x : x * slope;
    }
}
__global__ void RELU_Half(const half* input, half* output, size_t count, float slope) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        float x = input[i];
        output[i] = (half)(x > 0 ? x : x * slope);
    }
}
__global__ void RELU_INT8(const int8_t* input, int8_t* output, size_t count, int8_t zeroPoint) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int8_t x = input[i];
        output[i] = x > zeroPoint ? x : zeroPoint;
    }
}
// UnaryExecution.cu:125  out[i] = clamp(x, minV, maxV)
template <typename T>
__global__ void CLAMP(const T* input, T* output, size_t count, float minV, float maxV) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        float x = input[i];
        output[i] = min(max(x, minV), maxV);
    }
}

// ============================================================================
// Cast: source/backend/cuda/execution/CastExecution.cu
// ============================================================================
template <typename T1, typename T2>
__global__ void CAST(T1* input, T2* output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        output[i] = (T2)(input[i]);
    }
}
template <typename T1, typename T2>
__global__ void CASTMIDFLOAT(T1* input, T2* output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        output[i] = (T2)((float)input[i]);
    }
}
template <typename T>
__global__ void BF162FLOAT(int16_t* input, T* output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        float tmp;
        ((int16_t*)&tmp)[0] = 0;
        ((int16_t*)&tmp)[1] = input[i];
        output[i] = (T)tmp;
    }
}
__global__ void CASTBOOL(int32_t* input, int32_t* output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        output[i] = input[i] > 0 ? 1 : 0;
    }
}
template <typename T>
__global__ void FLOAT_2_INT8_CAST(const int count, const T* in, int8_t* out, const float scaleData,
                                   const int8_t zeroPoint, const int8_t clampMax, const int8_t clampMin) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        float inp_0 = in[index];
        int res = __float2int_rn(inp_0 * scaleData) + zeroPoint;
        res = min(res, (int)clampMax);
        res = max(res, (int)clampMin);
        out[index] = res;
    }
}
template <typename T>
__global__ void INT8_2_FLOAT_CAST(const int count, const int8_t* in, T* out, const float scaleData,
                                   const int8_t zeroPoint) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        char inp_0 = in[index];
        out[index] = (T)((inp_0 - zeroPoint) * scaleData);
    }
}

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

// ============================================================================
// Range: source/backend/cuda/execution/RangeExecution.cu
// ============================================================================
template <typename T>
__global__ void RANGE(const int size, const T* input0, const T* input2, T* output) {
    CUDA_KERNEL_LOOP(i, size) {
        T start = input0[0];
        T step = input2[0];
        output[i] = start + (T)i * step;
    }
}

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

// ============================================================================
// LayerNorm: source/backend/cuda/execution/LayerNormExecution.cu (simple version)
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
// PReLU: source/backend/cuda/execution/PReLUExecution.cu
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

// ============================================================================
// Pool: source/backend/cuda/execution/PoolExecution.cu
// ============================================================================
template <typename T>
__global__ void maxpool_C8(const T* uInput, T* uOutput, const int ib, const int ic_p, const int ih, const int iw,
                           const int oh, const int ow, const int padX, const int padY, const int kernelX,
                           const int kernelY, const int strideX, const int strideY) {
    int total = ib * oh * ow * ic_p;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int ic_idx = i % ic_p;
        int tmp0 = i / ic_p;
        int ow_idx = tmp0 % ow;
        int tmp1 = tmp0 / ow;
        int ib_idx = tmp1 / oh;
        int oh_idx = tmp1 % oh;
        int iw_idx = ow_idx * strideX - padX;
        int ih_idx = oh_idx * strideY - padY;
        int sx = max(0, -iw_idx);
        int sy = max(0, -ih_idx);
        int ex = min(kernelX, iw - iw_idx);
        int ey = min(kernelY, ih - ih_idx);
        T maxValue = HALF_MIN;
        for (int fy = sy; fy < ey; ++fy) {
            for (int fx = sx; fx < ex; ++fx) {
                int currentX = iw_idx + fx;
                int currentY = ih_idx + fy;
                const T* input = (const T*)(uInput + ib_idx * ih * iw * ic_p + currentY * iw * ic_p + currentX * ic_p + ic_idx);
                T val = *input;
                maxValue = maxValue > val ? maxValue : val;
            }
        }
        T* dst = (T*)(uOutput + ib_idx * oh * ow * ic_p + oh_idx * ow * ic_p + ow_idx * ic_p + ic_idx);
        *dst = maxValue;
    }
}
template <typename T>
__global__ void avgpool_C8(const T* uInput, T* uOutput, const int ib, const int ic_p, const int ih, const int iw,
                           const int oh, const int ow, const int padX, const int padY, const int kernelX,
                           const int kernelY, const int strideX, const int strideY) {
    int total = ib * oh * ow * ic_p;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int ic_idx = i % ic_p;
        int tmp0 = i / ic_p;
        int ow_idx = tmp0 % ow;
        int tmp1 = tmp0 / ow;
        int ib_idx = tmp1 / oh;
        int oh_idx = tmp1 % oh;
        int iw_idx = ow_idx * strideX - padX;
        int ih_idx = oh_idx * strideY - padY;
        int sx = max(0, -iw_idx);
        int sy = max(0, -ih_idx);
        int ex = min(kernelX, iw - iw_idx);
        int ey = min(kernelY, ih - ih_idx);
        T div = (float)(ey - sy) * (float)(ex - sx);
        T sumValue = (T)0.0f;
        for (int fy = sy; fy < ey; ++fy) {
            for (int fx = sx; fx < ex; ++fx) {
                int currentX = iw_idx + fx;
                int currentY = ih_idx + fy;
                const T* input = (const T*)(uInput + ib_idx * ih * iw * ic_p + currentY * iw * ic_p + currentX * ic_p + ic_idx);
                T val = *input;
                sumValue += val;
            }
        }
        sumValue /= div;
        T* dst = (T*)(uOutput + ib_idx * oh * ow * ic_p + oh_idx * ow * ic_p + ow_idx * ic_p + ic_idx);
        *dst = sumValue;
    }
}
template <typename T>
__global__ void global_avgpool_C8(const T* input, T* output, const int outside, const int axis, const int inside,
                                  const int per_block_size, const int calc_multi_num) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;
    float local_src = 0.0;
    __shared__ float sumValue;
    for (int i = 0; i < calc_multi_num; i++) {
        if (tid + i * per_block_size < axis) {
            local_src += (float)(src[(tid + i * per_block_size) * inside]);
        }
    }
    float maxRes = blockReduceSum<float>(local_src);
    if (tid == 0) sumValue = maxRes;
    __syncthreads();
    output[idx_outside * inside + idx_inside] = (T)(sumValue / (float)axis);
}
template <typename T>
__global__ void global_maxpool_C8(const T* input, T* output, const int outside, const int axis, const int inside,
                                   const int per_block_size, const int calc_multi_num) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;
    float local_src = -FLT_MAX;
    __shared__ float maxValue;
    for (int i = 0; i < calc_multi_num; i++) {
        if (tid + i * per_block_size < axis) {
            local_src = max(local_src, (float)(src[(tid + i * per_block_size) * inside]));
        }
    }
    float maxRes = blockReduceMax<float>(local_src);
    if (tid == 0) maxValue = maxRes;
    __syncthreads();
    output[idx_outside * inside + idx_inside] = (T)maxValue;
}

} // namespace Corpus
} // namespace MNN

// ============================================================================
// Explicit template instantiations + extern "C" launch shims
// ============================================================================
extern "C" {

// ---- Unary ----
void mnn_corpus_relu_fp32(const float* input, float* output, size_t count, float slope, int grid, int block,
                          cudaStream_t stream) {
    MNN::Corpus::RELU<<<grid, block, 0, stream>>>(input, output, count, slope);
}
void mnn_corpus_relu_fp16(const void* input, void* output, size_t count, float slope, int grid, int block,
                          cudaStream_t stream) {
    MNN::Corpus::RELU_Half<<<grid, block, 0, stream>>>((const half*)input, (half*)output, count, slope);
}
void mnn_corpus_relu_int8(const int8_t* input, int8_t* output, size_t count, int8_t zeroPoint, int grid, int block,
                          cudaStream_t stream) {
    MNN::Corpus::RELU_INT8<<<grid, block, 0, stream>>>(input, output, count, zeroPoint);
}
void mnn_corpus_clamp_fp32(const float* input, float* output, size_t count, float minV, float maxV, int grid,
                           int block, cudaStream_t stream) {
    MNN::Corpus::CLAMP<float><<<grid, block, 0, stream>>>(input, output, count, minV, maxV);
}
void mnn_corpus_clamp_fp16(const void* input, void* output, size_t count, float minV, float maxV, int grid,
                           int block, cudaStream_t stream) {
    MNN::Corpus::CLAMP<half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, count, minV, maxV);
}

// ---- Cast ----
void mnn_corpus_castbool_i32(const int32_t* input, int32_t* output, size_t count, int grid, int block,
                             cudaStream_t stream) {
    MNN::Corpus::CASTBOOL<<<grid, block, 0, stream>>>((int32_t*)input, (int32_t*)output, count);
}
void mnn_corpus_cast_f32_i32(const float* input, int32_t* output, size_t count, int grid, int block,
                             cudaStream_t stream) {
    MNN::Corpus::CAST<float, int32_t><<<grid, block, 0, stream>>>((float*)input, (int32_t*)output, count);
}
void mnn_corpus_cast_i32_f32(const int32_t* input, float* output, size_t count, int grid, int block,
                             cudaStream_t stream) {
    MNN::Corpus::CAST<int32_t, float><<<grid, block, 0, stream>>>((int32_t*)input, (float*)output, count);
}
void mnn_corpus_cast_i8_i32(const int8_t* input, int32_t* output, size_t count, int grid, int block,
                            cudaStream_t stream) {
    MNN::Corpus::CAST<int8_t, int32_t><<<grid, block, 0, stream>>>((int8_t*)input, (int32_t*)output, count);
}
void mnn_corpus_cast_i32_u8(const int32_t* input, uint8_t* output, size_t count, int grid, int block,
                            cudaStream_t stream) {
    MNN::Corpus::CAST<int32_t, uint8_t><<<grid, block, 0, stream>>>((int32_t*)input, (uint8_t*)output, count);
}
void mnn_corpus_cast_u8_i32(const uint8_t* input, int32_t* output, size_t count, int grid, int block,
                            cudaStream_t stream) {
    MNN::Corpus::CAST<uint8_t, int32_t><<<grid, block, 0, stream>>>((uint8_t*)input, (int32_t*)output, count);
}
void mnn_corpus_castmidfloat_f16_i32(const void* input, int32_t* output, size_t count, int grid, int block,
                                     cudaStream_t stream) {
    MNN::Corpus::CASTMIDFLOAT<half, int32_t><<<grid, block, 0, stream>>>((half*)input, (int32_t*)output, count);
}
void mnn_corpus_castmidfloat_i32_f16(const int32_t* input, void* output, size_t count, int grid, int block,
                                     cudaStream_t stream) {
    MNN::Corpus::CASTMIDFLOAT<int32_t, half><<<grid, block, 0, stream>>>((int32_t*)input, (half*)output, count);
}
void mnn_corpus_castmidfloat_f16_i8(const void* input, int8_t* output, size_t count, int grid, int block,
                                    cudaStream_t stream) {
    MNN::Corpus::CASTMIDFLOAT<half, int8_t><<<grid, block, 0, stream>>>((half*)input, (int8_t*)output, count);
}
void mnn_corpus_bf162float_f32(const int16_t* input, float* output, size_t count, int grid, int block,
                               cudaStream_t stream) {
    MNN::Corpus::BF162FLOAT<float><<<grid, block, 0, stream>>>((int16_t*)input, (float*)output, count);
}
void mnn_corpus_bf162float_f16(const int16_t* input, void* output, size_t count, int grid, int block,
                               cudaStream_t stream) {
    MNN::Corpus::BF162FLOAT<half><<<grid, block, 0, stream>>>((int16_t*)input, (half*)output, count);
}
void mnn_corpus_castmidfloat_f32_i32(const float* input, int32_t* output, size_t count, int grid, int block,
                                     cudaStream_t stream) {
    MNN::Corpus::CASTMIDFLOAT<float, int32_t><<<grid, block, 0, stream>>>((float*)input, (int32_t*)output, count);
}
void mnn_corpus_float2int8_fp32(const float* in, int8_t* out, size_t count, float scaleData, int8_t zeroPoint,
                                 int8_t clampMax, int8_t clampMin, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::FLOAT_2_INT8_CAST<float><<<grid, block, 0, stream>>>((int)count, in, out, scaleData, zeroPoint,
                                                                       clampMax, clampMin);
}
void mnn_corpus_int82float_fp32(const int8_t* in, float* out, size_t count, float scaleData, int8_t zeroPoint,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INT8_2_FLOAT_CAST<float><<<grid, block, 0, stream>>>((int)count, in, out, scaleData, zeroPoint);
}

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

// ---- Range ----
void mnn_corpus_range_fp32(const int size, const float* start, const float* step, float* output, int grid, int block,
                           cudaStream_t stream) {
    MNN::Corpus::RANGE<float><<<grid, block, 0, stream>>>(size, start, step, output);
}
void mnn_corpus_range_i32(const int size, const int* start, const int* step, int* output, int grid, int block,
                          cudaStream_t stream) {
    MNN::Corpus::RANGE<int><<<grid, block, 0, stream>>>(size, start, step, output);
}
void mnn_corpus_range_fp16(const int size, const void* start, const void* step, void* output, int grid, int block,
                           cudaStream_t stream) {
    MNN::Corpus::RANGE<half><<<grid, block, 0, stream>>>(size, (const half*)start, (const half*)step, (half*)output);
}

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

// ---- Pool ----
void mnn_corpus_maxpool_fp32(const float* in, float* out, int ib, int ic_p, int ih, int iw, int oh, int ow,
                              int padX, int padY, int kx, int ky, int sx, int sy, int grid, int block,
                              cudaStream_t stream) {
    MNN::Corpus::maxpool_C8<float><<<grid, block, 0, stream>>>(in, out, ib, ic_p, ih, iw, oh, ow, padX, padY, kx, ky,
                                                                 sx, sy);
}
void mnn_corpus_maxpool_fp16(const void* in, void* out, int ib, int ic_p, int ih, int iw, int oh, int ow,
                              int padX, int padY, int kx, int ky, int sx, int sy, int grid, int block,
                              cudaStream_t stream) {
    MNN::Corpus::maxpool_C8<half><<<grid, block, 0, stream>>>((const half*)in, (half*)out, ib, ic_p, ih, iw, oh, ow,
                                                                 padX, padY, kx, ky, sx, sy);
}
void mnn_corpus_avgpool_fp32(const float* in, float* out, int ib, int ic_p, int ih, int iw, int oh, int ow,
                              int padX, int padY, int kx, int ky, int sx, int sy, int grid, int block,
                              cudaStream_t stream) {
    MNN::Corpus::avgpool_C8<float><<<grid, block, 0, stream>>>(in, out, ib, ic_p, ih, iw, oh, ow, padX, padY, kx, ky,
                                                                 sx, sy);
}
void mnn_corpus_avgpool_fp16(const void* in, void* out, int ib, int ic_p, int ih, int iw, int oh, int ow,
                              int padX, int padY, int kx, int ky, int sx, int sy, int grid, int block,
                              cudaStream_t stream) {
    MNN::Corpus::avgpool_C8<half><<<grid, block, 0, stream>>>((const half*)in, (half*)out, ib, ic_p, ih, iw, oh, ow,
                                                                 padX, padY, kx, ky, sx, sy);
}
void mnn_corpus_global_avgpool_fp32(const float* input, float* output, int outside, int axis, int inside,
                                    int per_block_size, int calc_multi_num, int grid, int block,
                                    cudaStream_t stream) {
    MNN::Corpus::global_avgpool_C8<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside,
                                                                       per_block_size, calc_multi_num);
}
void mnn_corpus_global_avgpool_fp16(const void* input, void* output, int outside, int axis, int inside,
                                    int per_block_size, int calc_multi_num, int grid, int block,
                                    cudaStream_t stream) {
    MNN::Corpus::global_avgpool_C8<half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, outside, axis,
                                                                       inside, per_block_size, calc_multi_num);
}
void mnn_corpus_global_maxpool_fp32(const float* input, float* output, int outside, int axis, int inside,
                                     int per_block_size, int calc_multi_num, int grid, int block,
                                     cudaStream_t stream) {
    MNN::Corpus::global_maxpool_C8<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside,
                                                                       per_block_size, calc_multi_num);
}
void mnn_corpus_global_maxpool_fp16(const void* input, void* output, int outside, int axis, int inside,
                                     int per_block_size, int calc_multi_num, int grid, int block,
                                     cudaStream_t stream) {
    MNN::Corpus::global_maxpool_C8<half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, outside, axis,
                                                                       inside, per_block_size, calc_multi_num);
}

} // extern "C"

