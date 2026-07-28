// unary_cast.cu - RELU/CLAMP + CAST-family kernels + shims
//   source/backend/cuda/execution/UnaryExecution.cu
//   source/backend/cuda/execution/CastExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

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

// ---- FLOAT_2_INT8_CAST_PACK (packed quantization cast, 2.5.3+) ----
template<typename T>
__global__ void FLOAT_2_INT8_CAST_PACK(const int count, const T* in, int8_t* out,
                                       const float scaleData, const int8_t zeroPoint,
                                       const int8_t clampMax, const int8_t clampMin,
                                       const int channelPackFloat, const int channels,
                                       DivModFast d_cp) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)count; index += blockDim.x * gridDim.x) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);
        if (c_idx >= channels) {
            out[index] = 0;
            return;
        }
        float inp_0 = in[nhw_idx * channelPackFloat + c_idx];
        int res = __float2int_rn(inp_0 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);
        out[index] = res;
    }
}

// ---- INT8_2_FLOAT_CAST_PACK (packed dequantization cast, 2.5.3+) ----
template<typename T>
__global__ void INT8_2_FLOAT_CAST_PACK(const int count, const int8_t* in, T* out,
                                       const float scaleData, const int8_t zeroPoint,
                                       const int channelPackInt8, const int channels,
                                       DivModFast d_cp) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)count; index += blockDim.x * gridDim.x) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);
        char inp_0 = in[nhw_idx * channelPackInt8 + c_idx];
        out[index] = (T)((inp_0 - zeroPoint) * scaleData);
    }
}

} // namespace Corpus
} // namespace MNN

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

// ---- CLAMP 1.2.7 (templatized, fp32 = same as 1.2.0 body) ----
void mnn_corpus_clamp_127_fp32(const float* input, float* output, size_t count, float minV, float maxV,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::CLAMP<float><<<grid, block, 0, stream>>>(input, output, count, minV, maxV);
}

// ---- FLOAT_2_INT8_CAST_PACK (packed quantization cast, 2.5.3+) ----
void mnn_corpus_float2int8_cast_pack_fp32(const int count, const float* in, int8_t* out,
                                          float scaleData, int8_t zeroPoint, int8_t clampMax, int8_t clampMin,
                                          int channelPackFloat, int channels, int d_cp_val,
                                          int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_cp(d_cp_val);
    MNN::Corpus::FLOAT_2_INT8_CAST_PACK<float><<<grid, block, 0, stream>>>(
        count, in, out, scaleData, zeroPoint, clampMax, clampMin, channelPackFloat, channels, d_cp);
}

// ---- INT8_2_FLOAT_CAST_PACK (packed dequantization cast, 2.5.3+) ----
void mnn_corpus_int82float_cast_pack_fp32(const int count, const int8_t* in, float* out,
                                          float scaleData, int8_t zeroPoint,
                                          int channelPackInt8, int channels, int d_cp_val,
                                          int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_cp(d_cp_val);
    MNN::Corpus::INT8_2_FLOAT_CAST_PACK<float><<<grid, block, 0, stream>>>(
        count, in, out, scaleData, zeroPoint, channelPackInt8, channels, d_cp);
}

} // extern "C"
