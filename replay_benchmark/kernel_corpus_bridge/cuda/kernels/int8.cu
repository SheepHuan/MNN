// int8.cu - Int8 quantization kernels for replay benchmark corpus
//   source/backend/cuda/execution/int8/FloatToInt8Execution.cu (FLOAT_2_INT8[_SINGLE])
//   source/backend/cuda/execution/int8/Int8ToFloatExecution.cu (INT8_2_FLOAT[_SINGLE])
//   source/backend/cuda/execution/int8/DepthwiseConvInt8Execution.cu (CONV_DW_INT8_, CONV_DW3x3S1_INT8_OPT)
//   source/backend/cuda/execution/int8/ConvInt8CutlassExecution.cu (Im2Col_packC_16, WeightInt8PackFill)
//   source/backend/cuda/execution/int8/BinaryInt8Execution.cu (BINARY_INT8_ADD/MUL)
//   source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu (DequantizeInt8Weight/Int4Weight)
//
// Reimplemented without #ifdef ENABLE_CUDA_QUANT guard (corpus is standalone).
// INT8_PACK_NUMBER=4 (packs 4 int8 per thread, matching MNN's INT8_PACK_NUMBER=4).
#include "corpus_common.cuh"
#include <cuda_fp16.h>

namespace MNN {
namespace Corpus {

// ============================================================================
// FLOAT_2_INT8: quantize float4 -> int8x4 with per-channel scale.
// Template T = float (fp32 path). half path uses separate __half instantiation.
// ============================================================================
template<typename T>
__global__ void FLOAT_2_INT8(const int total, const int channelsPackInt8,
                              const int channelsPackFloat, const int channels,
                              const T* in, int8_t* out,
                              const float* scaleData, const int8_t zeroPoint,
                              const int8_t clampMax, const int8_t clampMin,
                              DivModFast d_cp) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        int out_idx = index << 2;
        if(4 * c_idx >= channels) {
            ((char4 *)(out + out_idx))[0] = make_char4(0, 0, 0, 0);
            continue;
        }

        float4 scale_0 = ((float4 *)(scaleData + (c_idx << 2)))[0];

        int idx_inp = nhw_idx * channelsPackFloat + 4*c_idx;

        float inp_0 = in[idx_inp];
        float inp_1 = in[idx_inp+1];
        float inp_2 = in[idx_inp+2];
        float inp_3 = in[idx_inp+3];

        int res = __float2int_rn(inp_0 * scale_0.x) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx] = res;

        res = __float2int_rn(inp_1 * scale_0.y) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 1] = res;

        res = __float2int_rn(inp_2 * scale_0.z) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 2] = res;


        res = __float2int_rn(inp_3 * scale_0.w) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 3] = res;
    }
}

template<typename T>
__global__ void FLOAT_2_INT8_SINGLE(const int total, const int channelsPackInt8,
                                     const int channelsPackFloat, const int channels,
                                     const T* in, int8_t* out,
                                     const float scaleData, const int8_t zeroPoint,
                                     const int8_t clampMax, const int8_t clampMin,
                                     DivModFast d_cp) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        int out_idx = index << 2;
        if(4 * c_idx >= channels) {
            ((char4 *)(out + out_idx))[0] = make_char4(0, 0, 0, 0);
            continue;
        }

        int idx_inp = nhw_idx * channelsPackFloat + 4*c_idx;

        float inp_0 = in[idx_inp];
        float inp_1 = in[idx_inp+1];
        float inp_2 = in[idx_inp+2];
        float inp_3 = in[idx_inp+3];

        int res = __float2int_rn(inp_0 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx] = res;

        res = __float2int_rn(inp_1 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 1] = res;

        res = __float2int_rn(inp_2 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 2] = res;


        res = __float2int_rn(inp_3 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 3] = res;
    }
}

// ============================================================================
// INT8_2_FLOAT: dequantize int8x4 -> float4 with per-channel scale.
// ============================================================================
template<typename T>
__global__ void INT8_2_FLOAT(const int total, const int channelsPackInt8,
                              const int channelsPackFloat, const int channels,
                              const int8_t* in, T* out,
                              const float* scaleData, const int8_t zeroPoint,
                              DivModFast d_cp) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        int idx_inp = nhw_idx * channelsPackInt8 + 4*c_idx;
        char4 inp_0 = ((char4 *)(in + idx_inp))[0];
        float4 scale_0 = ((float4 *)(scaleData + (c_idx << 2)))[0];

        const int idx_out = index << 2;

        out[idx_out+0] = (T)((inp_0.x - zeroPoint) * scale_0.x);
        out[idx_out+1] = (T)((inp_0.y - zeroPoint) * scale_0.y);
        out[idx_out+2] = (T)((inp_0.z - zeroPoint) * scale_0.z);
        out[idx_out+3] = (T)((inp_0.w - zeroPoint) * scale_0.w);
    }
}

template<typename T>
__global__ void INT8_2_FLOAT_SINGLE(const int total, const int channelsPackInt8,
                                     const int channelsPackFloat, const int channels,
                                     const int8_t* in, T* out,
                                     const float scaleData, const int8_t zeroPoint,
                                     DivModFast d_cp) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        int idx_inp = nhw_idx * channelsPackInt8 + 4*c_idx;
        char4 inp_0 = ((char4 *)(in + idx_inp))[0];

        const int idx_out = index << 2;
        out[idx_out+0] = (T)((inp_0.x - zeroPoint) * scaleData);
        out[idx_out+1] = (T)((inp_0.y - zeroPoint) * scaleData);
        out[idx_out+2] = (T)((inp_0.z - zeroPoint) * scaleData);
        out[idx_out+3] = (T)((inp_0.w - zeroPoint) * scaleData);
    }
}

// ============================================================================
// DequantizeInt8Weight: int8 weight -> half, per-group scale/offset.
// ============================================================================
template<typename T, typename dT>
__global__ void DequantizeInt8Weight(const int8_t* quantized_kernel, dT* dequantized_kernel,
                                      const T* scale, const T* offset,
                                      const int oc, const int ic, const int ic_p, const int quanC) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < oc && col < ic) {
        const int num_quan_groups = (quanC > 0) ? (quanC / oc) : 1;
        const int ic_per_group = (num_quan_groups > 0) ? (ic / num_quan_groups) : ic;
        const int group_idx = col / ic_per_group;
        const int quan_param_index = row * num_quan_groups + group_idx;
        const float x_scale = (float)scale[quan_param_index];
        const float x_offset = (float)offset[quan_param_index];
        const float qval = (float)quantized_kernel[row * ic_p + col];
        dequantized_kernel[row * ic_p + col] = (dT)(qval * x_scale + x_offset);
    }
}

// ============================================================================
// DequantizeInt4Weight: packed int4 (2 per byte) -> half, per-group scale.
// Faithful copy of MNN source: 8-byte vectorized load, 16 half values per thread.
// ============================================================================
template<typename T, typename dT>
__global__ void DequantizeInt4Weight(
    const uint8_t* __restrict__ packed_weight,
    dT* __restrict__ output,
    const T* __restrict__ scale,
    const T* __restrict__ offset,
    const int oc, const int ic, const int ic_p, const int quanC
) {
    // Each thread handles 8 packed bytes = 16 int4 values
    const int tid = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;  // OC index

    if (row >= oc) return;

    const int half_ic = ic_p / 2;  // bytes per row
    const int col_byte = tid * 8;  // starting byte within row
    if (col_byte >= half_ic) return;

    const int num_qg = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = max(1, (num_qg > 0) ? (ic / num_qg) : ic);
    const int qparam_base = row * num_qg;

    // Vectorized 8-byte weight load (two uint32s)
    const uint32_t* src = reinterpret_cast<const uint32_t*>(packed_weight + row * half_ic + col_byte);
    const uint32_t packed0 = __ldg(src);
    const uint32_t packed1 = (col_byte + 4 < half_ic) ? __ldg(src + 1) : 0;

    const int col = col_byte * 2;  // starting ic index
    const int out_base = row * ic_p + col;

    // Process 8 bytes = 16 int4 values, write as half2 pairs
    half result[16];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int kk = col + j * 2;
        if (kk >= ic) break;
        const int gidx = min(kk / ic_per_group, num_qg - 1);
        const float s = (float)__ldg(&scale[qparam_base + gidx]);
        const float o = (float)__ldg(&offset[qparam_base + gidx]);
        const uint8_t byte_val = (packed0 >> (j * 8)) & 0xFF;
        result[j * 2] = (half)(((float)(byte_val >> 4) - 8.0f) * s + o);
        result[j * 2 + 1] = (half)(((float)(byte_val & 0x0F) - 8.0f) * s + o);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int kk = col + 8 + j * 2;
        if (kk >= ic) break;
        const int gidx = min(kk / ic_per_group, num_qg - 1);
        const float s = (float)__ldg(&scale[qparam_base + gidx]);
        const float o = (float)__ldg(&offset[qparam_base + gidx]);
        const uint8_t byte_val = (packed1 >> (j * 8)) & 0xFF;
        result[8 + j * 2] = (half)(((float)(byte_val >> 4) - 8.0f) * s + o);
        result[8 + j * 2 + 1] = (half)(((float)(byte_val & 0x0F) - 8.0f) * s + o);
    }

    // Vectorized write: 16 halves = 32 bytes = float4 + float4
    if (col + 8 <= ic) {
        *reinterpret_cast<float4*>(output + out_base) = *reinterpret_cast<float4*>(result);
    } else {
        for (int i = 0; i < 8 && col + i < ic; i++) output[out_base + i] = (dT)result[i];
    }
    if (col + 16 <= ic) {
        *reinterpret_cast<float4*>(output + out_base + 8) = *reinterpret_cast<float4*>(result + 8);
    } else if (col + 8 < ic) {
        for (int i = 8; i < 16 && col + i < ic; i++) output[out_base + i] = (dT)result[i];
    }
}

// ============================================================================
// CONV_DW_INT8_: int8 depthwise conv with int32 accumulator + dp4a.
// ============================================================================
static inline __device__ __host__ int32_t vecDot4(char4 a, char4 b, int32_t val) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
    return __dp4a(a, b, val);
#else
    return val + a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
#endif
}

__global__ void CONV_DW_INT8_(const int8_t* input, const int8_t* kernel, const int32_t* bias,
                               const float* scale, int8_t* output,
                               const int8_t maxV, const int8_t minV,
                               const int iw, const int ih, const int c, const int c_p,
                               const int ow, const int oh, const int kw, const int kh,
                               const int dw, const int dh, const int sw, const int sh,
                               const int pw, const int ph, const int total,
                               DivModFast d_oc, DivModFast d_ow, DivModFast d_oh) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/4; index += blockDim.x * gridDim.x) {
        int oz_4, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_4);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);
        
        int oz = oz_4 << 2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;

        int4 bias4 = ((int4 *)(bias + oz))[0];
        int color0 = bias4.x;
        int color1 = bias4.y;
        int color2 = bias4.z;
        int color3 = bias4.w;

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        for (int fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (int fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                int src_offset = ((ob * ih + sy) * iw + sx) * c_p + oz;

                char4 inp4 = ((char4 *)(input + src_offset))[0];
                char4 ker4 = ((char4 *)(kernel + (fy * kw + fx) * c_p + oz))[0];

                color0 = color0 + (int)inp4.x * (int)ker4.x;
                color1 = color1 + (int)inp4.y * (int)ker4.y;
                color2 = color2 + (int)inp4.z * (int)ker4.z;
                color3 = color3 + (int)inp4.w * (int)ker4.w;

            }
        }

        float4 scale4 = ((float4 *)(scale + oz))[0];
        color0 = __float2int_rn((float)color0 * scale4.x);
        color1 = __float2int_rn((float)color1 * scale4.y);
        color2 = __float2int_rn((float)color2 * scale4.z);
        color3 = __float2int_rn((float)color3 * scale4.w);

        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        color1 = max(color1, minV);
        color1 = min(color1, maxV);

        color2 = max(color2, minV);
        color2 = min(color2, maxV);

        color3 = max(color3, minV);
        color3 = min(color3, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        ((char4*)(output + dst_offset))[0] = make_char4((color0), (color1), (color2), (color3));
    }
}

// CONV_DW3x3S1_INT8_OPT: specialized for 3x3 stride=1, 2 output x per thread.
// Faithful copy of MNN source (DepthwiseConvInt8Execution.cu:124-302).
__global__ void CONV_DW3x3S1_INT8_OPT(const int8_t* input, const int8_t* kernel, const int32_t* bias,
                                       const float* scale, int8_t* output,
                                       const int8_t maxV, const int8_t minV,
                                       const int iw, const int ih, const int c, const int c_p,
                                       const int ow, const int oh, const int kw, const int kh,
                                       const int k_p, const int dw, const int dh,
                                       const int sw, const int sh, const int pw, const int ph,
                                       const int total,
                                       DivModFast d_oc, DivModFast d_ow, DivModFast d_oh) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/8; index += blockDim.x * gridDim.x) {
        int oz, ix, oy, ox, iy, ob;
        d_oc.divmod(index, iy, oz);
        d_ow.divmod(iy, ix, ox);
        d_oh.divmod(ix, ob, oy);
        
        ox = ox << 1;
        oz = oz << 2;
        ix = ox - 1;
        iy = oy - 1;

        int4 bias4 = ((int4 *)(bias + oz))[0];
        int color0_0 = (int)bias4.x;
        int color0_1 = color0_0;
        int color1_0 = (int)bias4.y;
        int color1_1 = color1_0;
        int color2_0 = (int)bias4.z;
        int color2_1 = color2_0;
        int color3_0 = (int)bias4.w;
        int color3_1 = color3_0;

        char4 zero4 = make_char4(0, 0, 0, 0);
        char4 inp4[12], ker4[3][3];
        #pragma unroll
        for(int j=0; j<3; j++) {
            if(iy < 0 && j==0) {
                for(int i=0; i<4; i++) {
                    inp4[i] = zero4;
                }
                continue;
            }
            if(iy+2 > ih-1 && j==2) {
                for(int i=0; i<4; i++) {
                    inp4[8+i] = zero4;
                }
                continue;
            }

            for(int i=0; i<4; i++) {
                if(ix < 0 && i==0) {
                    for(int j=0; j<3; j++) {
                        inp4[4*j+0] = zero4;
                    }
                    continue;
                }
                if(ix+3 > iw-1 && i==3) {
                    for(int j=0; j<3; j++) {
                        inp4[4*j+3] = zero4;
                    }
                    continue;
                }
                int src_offset = ((ob * ih + iy+j) * iw + ix+i) * c_p + oz;
                inp4[4*j+i] = ((char4 *)(input + src_offset))[0];
            }
        }

        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                ker4[j][i] = ((char4 *)(kernel + (j * 3 + i) * c_p + oz))[0];
            }
        }

        // 1st channel
        char4 tmp_ker4 = make_char4(ker4[0][0].x, ker4[0][1].x, ker4[0][2].x, ker4[1][0].x);
        color0_0 += vecDot4(make_char4(inp4[0].x, inp4[1].x, inp4[2].x, inp4[4].x), tmp_ker4, 0);
        color0_1 += vecDot4(make_char4(inp4[1].x, inp4[2].x, inp4[3].x, inp4[5].x), tmp_ker4, 0);

        tmp_ker4 = make_char4(ker4[1][1].x, ker4[1][2].x, ker4[2][0].x, ker4[2][1].x);
        color0_0 += vecDot4(make_char4(inp4[5].x, inp4[6].x, inp4[8].x, inp4[9].x), tmp_ker4, 0);
        color0_1 += vecDot4(make_char4(inp4[6].x, inp4[7].x, inp4[9].x, inp4[10].x), tmp_ker4, 0);

        color0_0 += inp4[10].x * ker4[2][2].x;
        color0_1 += inp4[11].x * ker4[2][2].x;

        // 2nd channel
        tmp_ker4 = make_char4(ker4[0][0].y, ker4[0][1].y, ker4[0][2].y, ker4[1][0].y);
        color1_0 += vecDot4(make_char4(inp4[0].y, inp4[1].y, inp4[2].y, inp4[4].y), tmp_ker4, 0);
        color1_1 += vecDot4(make_char4(inp4[1].y, inp4[2].y, inp4[3].y, inp4[5].y), tmp_ker4, 0);

        tmp_ker4 = make_char4(ker4[1][1].y, ker4[1][2].y, ker4[2][0].y, ker4[2][1].y);
        color1_0 += vecDot4(make_char4(inp4[5].y, inp4[6].y, inp4[8].y, inp4[9].y), tmp_ker4, 0);
        color1_1 += vecDot4(make_char4(inp4[6].y, inp4[7].y, inp4[9].y, inp4[10].y), tmp_ker4, 0);

        color1_0 += inp4[10].y * ker4[2][2].y;
        color1_1 += inp4[11].y * ker4[2][2].y;

        // 3rd channel
        tmp_ker4 = make_char4(ker4[0][0].z, ker4[0][1].z, ker4[0][2].z, ker4[1][0].z);
        color2_0 += vecDot4(make_char4(inp4[0].z, inp4[1].z, inp4[2].z, inp4[4].z), tmp_ker4, 0);
        color2_1 += vecDot4(make_char4(inp4[1].z, inp4[2].z, inp4[3].z, inp4[5].z), tmp_ker4, 0);

        tmp_ker4 = make_char4(ker4[1][1].z, ker4[1][2].z, ker4[2][0].z, ker4[2][1].z);
        color2_0 += vecDot4(make_char4(inp4[5].z, inp4[6].z, inp4[8].z, inp4[9].z), tmp_ker4, 0);
        color2_1 += vecDot4(make_char4(inp4[6].z, inp4[7].z, inp4[9].z, inp4[10].z), tmp_ker4, 0);

        color2_0 += inp4[10].z * ker4[2][2].z;
        color2_1 += inp4[11].z * ker4[2][2].z;

        // 4th channel
        tmp_ker4 = make_char4(ker4[0][0].w, ker4[0][1].w, ker4[0][2].w, ker4[1][0].w);
        color3_0 += vecDot4(make_char4(inp4[0].w, inp4[1].w, inp4[2].w, inp4[4].w), tmp_ker4, 0);
        color3_1 += vecDot4(make_char4(inp4[1].w, inp4[2].w, inp4[3].w, inp4[5].w), tmp_ker4, 0);

        tmp_ker4 = make_char4(ker4[1][1].w, ker4[1][2].w, ker4[2][0].w, ker4[2][1].w);
        color3_0 += vecDot4(make_char4(inp4[5].w, inp4[6].w, inp4[8].w, inp4[9].w), tmp_ker4, 0);
        color3_1 += vecDot4(make_char4(inp4[6].w, inp4[7].w, inp4[9].w, inp4[10].w), tmp_ker4, 0);

        color3_0 += inp4[10].w * ker4[2][2].w;
        color3_1 += inp4[11].w * ker4[2][2].w;

        // Multiple scale
        float4 scale4 = ((float4 *)(scale + oz))[0];
        color0_0 = __float2int_rn((float)color0_0 * scale4.x);
        color0_1 = __float2int_rn((float)color0_1 * scale4.x);

        color1_0 = __float2int_rn((float)color1_0 * scale4.y);
        color1_1 = __float2int_rn((float)color1_1 * scale4.y);

        color2_0 = __float2int_rn((float)color2_0 * scale4.z);
        color2_1 = __float2int_rn((float)color2_1 * scale4.z);

        color3_0 = __float2int_rn((float)color3_0 * scale4.w);
        color3_1 = __float2int_rn((float)color3_1 * scale4.w);

        // Clamp
        color0_0 = max(color0_0, minV);
        color0_0 = min(color0_0, maxV);
        color0_1 = max(color0_1, minV);
        color0_1 = min(color0_1, maxV);

        color1_0 = max(color1_0, minV);
        color1_0 = min(color1_0, maxV);
        color1_1 = max(color1_1, minV);
        color1_1 = min(color1_1, maxV);

        color2_0 = max(color2_0, minV);
        color2_0 = min(color2_0, maxV);
        color2_1 = max(color2_1, minV);
        color2_1 = min(color2_1, maxV);

        color3_0 = max(color3_0, minV);
        color3_0 = min(color3_0, maxV);
        color3_1 = max(color3_1, minV);
        color3_1 = min(color3_1, maxV);
        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        ((char4*)(output + dst_offset))[0] = make_char4((color0_0), (color1_0), (color2_0), (color3_0));
        ((char4*)(output + dst_offset + c_p))[0] = make_char4((color0_1), (color1_1), (color2_1), (color3_1));

    }
}

// ============================================================================
// Im2Col_packC_16: im2col for int8 conv (packC16 layout, int4 vectorized).
// ============================================================================
__global__ void Im2Col_packC_16(const int sw, const int sh, const int dw, const int dh,
                                 const int pw, const int ph, const int ic,
                                 const int iw, const int ih, const size_t maxCount,
                                 const int iBlock, const int icDiv4, const int e, const int l,
                                 const int32_t* A, int32_t* AP,
                                 DivModFast d_lp, DivModFast d_ow, DivModFast d_oh,
                                 DivModFast d_icp, DivModFast d_fx) {
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int eIndex, lpIndex;
        d_lp.divmod(indexO, eIndex, lpIndex);
        if (eIndex >= e) {
            *(((int4*)AP) + indexO) = make_int4(0, 0, 0, 0);
            continue;
        }
        int ox, oby, ob, oy, sz, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_icp.divmod(lpIndex, kI, sz);
        d_fx.divmod(kI, ksy, ksx);
        size_t sx = ox * sw + ksx * dw - pw;
        size_t sy = oy * sh + ksy * dh - ph;
        if (sx >= 0 && sx < (size_t)iw && sy >= 0 && sy < (size_t)ih) {
            size_t offset = ((ob * ih + sy) * iw + sx) * icDiv4 + sz;
            *(((int4*)AP) + indexO) = (*(((int4*)A) + offset));
            continue;
        }
        *(((int4*)AP) + indexO) = make_int4(0, 0, 0, 0);
    }
}

// ============================================================================
// WeightInt8PackFill: pack int8 weight for cutlass GEMM.
// ============================================================================
template<typename T>
__global__ void WeightInt8PackFill(const int8_t* param, T* output, const int maxCount,
                                    const int l, const int h, const int hp, const int ic,
                                    DivModFast d_lp, DivModFast d_hp, DivModFast d_icp,
                                    const bool ocMajor) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        if (ocMajor) {
            int lIndex, hpIndex;
            d_hp.divmod(index, lIndex, hpIndex);
            if (hpIndex >= h) { output[index] = (T)0; continue; }
            output[index] = (T)param[hpIndex * l + lIndex];
        } else {
            int lpIndex, fxyIndex, icpIndex, hpIndex;
            d_hp.divmod(index, lpIndex, hpIndex);
            if (hpIndex >= h) { output[index] = (T)0; continue; }
            d_lp.divmod(lpIndex, fxyIndex, icpIndex);
            if (icpIndex >= ic) { output[index] = (T)0; continue; }
            output[index] = (T)param[hpIndex * ic * l + fxyIndex * ic + icpIndex];
        }
    }
}

// ============================================================================
// BINARY_INT8_ADD / BINARY_INT8_MUL: element-wise int8 binary ops.
// ============================================================================
__global__ void BINARY_INT8_ADD(const int maxCount, const int8_t* input0_addr, const float input0_scale,
                                  const int8_t* input1_addr, const float input1_scale,
                                  int8_t* output_addr, const float output_scale,
                                  const int s0, const int s1) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        float x = (float)input0_addr[index*s0] * input0_scale;
        float y = (float)input1_addr[index*s1] * input1_scale;
        float val = x + y;
        int res = __float2int_rn(output_scale * val);
        res = min(res, 127); res = max(res, -128);
        output_addr[index] = (int8_t)res;
    }
}

__global__ void BINARY_INT8_MUL(const int maxCount, const int8_t* input0_addr, const float input0_scale,
                                  const int8_t* input1_addr, const float input1_scale,
                                  int8_t* output_addr, const float output_scale,
                                  const int s0, const int s1) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        float x = (float)input0_addr[index*s0] * input0_scale;
        float y = (float)input1_addr[index*s1] * input1_scale;
        float val = x * y;
        int res = __float2int_rn(output_scale * val);
        res = min(res, 127); res = max(res, -128);
        output_addr[index] = (int8_t)res;
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- FLOAT_2_INT8 (fp32 input, per-channel scale) ----
void mnn_corpus_float2int8_packed_fp32(const float* in, int8_t* out, int total,
                                  int channelsPackInt8, int channelsPackFloat, int channels,
                                  const float* scaleData, int8_t zeroPoint, int8_t clampMax, int8_t clampMin,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_cp(channelsPackInt8);
    MNN::Corpus::FLOAT_2_INT8<float><<<grid, block, 0, stream>>>(
        total, channelsPackInt8, channelsPackFloat, channels, in, out,
        scaleData, zeroPoint, clampMax, clampMin, d_cp);
}
// ---- FLOAT_2_INT8_SINGLE (fp32 input, single scale) ----
void mnn_corpus_float2int8_single_packed_fp32(const float* in, int8_t* out, int total,
                                         int channelsPackInt8, int channelsPackFloat, int channels,
                                         float scaleData, int8_t zeroPoint, int8_t clampMax, int8_t clampMin,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_cp(channelsPackInt8);
    MNN::Corpus::FLOAT_2_INT8_SINGLE<float><<<grid, block, 0, stream>>>(
        total, channelsPackInt8, channelsPackFloat, channels, in, out,
        scaleData, zeroPoint, clampMax, clampMin, d_cp);
}

// ---- INT8_2_FLOAT (fp32 output, per-channel scale) ----
void mnn_corpus_int82float_packed_fp32(const int8_t* in, float* out, int total,
                                  int channelsPackInt8, int channelsPackFloat, int channels,
                                  const float* scaleData, int8_t zeroPoint,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_cp(channelsPackInt8);
    MNN::Corpus::INT8_2_FLOAT<float><<<grid, block, 0, stream>>>(
        total, channelsPackInt8, channelsPackFloat, channels, in, out,
        scaleData, zeroPoint, d_cp);
}
// ---- INT8_2_FLOAT_SINGLE (fp32 output, single scale) ----
void mnn_corpus_int82float_single_packed_fp32(const int8_t* in, float* out, int total,
                                         int channelsPackInt8, int channelsPackFloat, int channels,
                                         float scaleData, int8_t zeroPoint,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_cp(channelsPackInt8);
    MNN::Corpus::INT8_2_FLOAT_SINGLE<float><<<grid, block, 0, stream>>>(
        total, channelsPackInt8, channelsPackFloat, channels, in, out,
        scaleData, zeroPoint, d_cp);
}

// ---- DequantizeInt8Weight (fp16 output) ----
void mnn_corpus_dequantize_int8_weight_fp16(const int8_t* qk, void* dk, const void* scale, const void* offset,
                                             int oc, int ic, int ic_p, int quanC,
                                             int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::DequantizeInt8Weight<__half, __half><<<grid, block, 0, stream>>>(
        qk, (__half*)dk, (const __half*)scale, (const __half*)offset, oc, ic, ic_p, quanC);
}
// ---- DequantizeInt4Weight (fp16 output) ----
void mnn_corpus_dequantize_int4_weight_fp16(const uint8_t* pw, void* output, const void* scale, const void* offset,
                                             int oc, int ic, int ic_p, int quanC,
                                             int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::DequantizeInt4Weight<__half, __half><<<grid, block, 0, stream>>>(
        pw, (__half*)output, (const __half*)scale, (const __half*)offset, oc, ic, ic_p, quanC);
}

// ---- CONV_DW_INT8_ ----
void mnn_corpus_conv_dw_int8_fp32(const int8_t* input, const int8_t* kernel, const int32_t* bias,
                                    const float* scale, int8_t* output,
                                    int8_t maxV, int8_t minV,
                                    int iw, int ih, int c, int c_p, int ow, int oh,
                                    int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                                    int total, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(c_p / 4), d_ow(ow), d_oh(oh);
    MNN::Corpus::CONV_DW_INT8_<<<grid, block, 0, stream>>>(
        input, kernel, bias, scale, output, maxV, minV,
        iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, d_oc, d_ow, d_oh);
}
// ---- CONV_DW3x3S1_INT8_OPT ----
void mnn_corpus_conv_dw3x3s1_int8_fp32(const int8_t* input, const int8_t* kernel, const int32_t* bias,
                                         const float* scale, int8_t* output,
                                         int8_t maxV, int8_t minV,
                                         int iw, int ih, int c, int c_p, int ow, int oh,
                                         int kw, int kh, int k_p, int dw, int dh, int sw, int sh, int pw, int ph,
                                         int total, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(c_p / 4), d_ow(ow/2), d_oh(oh);
    MNN::Corpus::CONV_DW3x3S1_INT8_OPT<<<grid, block, 0, stream>>>(
        input, kernel, bias, scale, output, maxV, minV,
        iw, ih, c, c_p, ow, oh, kw, kh, k_p, dw, dh, sw, sh, pw, ph, total, d_oc, d_ow, d_oh);
}

// ---- Im2Col_packC_16: faithful DivModFast construction from ow/oh/kw params.
// MNN passes d_lp(l), d_ow(ow), d_oh(oh), d_icp(icDiv4), d_fx(kw). The earlier
// smoke-only shim hard-coded d_ow=d_oh=d_fx=1, which only exercised the
// eIndex<e fast path and never the source-read branch. ----
void mnn_corpus_im2col_packc16_int8(const int sw, const int sh, const int dw, const int dh,
                                     const int pw, const int ph, const int ic,
                                     const int iw, const int ih, const size_t maxCount,
                                     const int iBlock, const int icDiv4, const int e, const int l,
                                     const int32_t* A, int32_t* AP,
                                     const int ow, const int oh, const int kw,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_lp(l), d_ow(ow), d_oh(oh), d_icp(icDiv4), d_fx(kw);
    MNN::Corpus::Im2Col_packC_16<<<grid, block, 0, stream>>>(
        sw, sh, dw, dh, pw, ph, ic, iw, ih, maxCount, iBlock, icDiv4, e, l, A, AP,
        d_lp, d_ow, d_oh, d_icp, d_fx);
}

// ---- WeightInt8PackFill (int8 output) ----
void mnn_corpus_weight_int8_pack_fill_fp32(const int8_t* param, int8_t* output, int maxCount,
                                            int l, int h, int hp, int ic,
                                            int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_lp(l), d_hp(hp), d_icp(ic);
    MNN::Corpus::WeightInt8PackFill<int8_t><<<grid, block, 0, stream>>>(
        param, output, maxCount, l, h, hp, ic, d_lp, d_hp, d_icp, false);
}

// ---- BINARY_INT8_ADD ----
void mnn_corpus_binary_int8_add_fp32(const int8_t* in0, float in0_scale,
                                       const int8_t* in1, float in1_scale,
                                       int8_t* out, float out_scale, int s0, int s1,
                                       int maxCount, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BINARY_INT8_ADD<<<grid, block, 0, stream>>>(
        maxCount, in0, in0_scale, in1, in1_scale, out, out_scale, s0, s1);
}
// ---- BINARY_INT8_MUL ----
void mnn_corpus_binary_int8_mul_fp32(const int8_t* in0, float in0_scale,
                                       const int8_t* in1, float in1_scale,
                                       int8_t* out, float out_scale, int s0, int s1,
                                       int maxCount, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BINARY_INT8_MUL<<<grid, block, 0, stream>>>(
        maxCount, in0, in0_scale, in1, in1_scale, out, out_scale, s0, s1);
}

} // extern "C"
