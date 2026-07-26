// Baked from MNN historical kernel. Copyright Alibaba Group Holding Limited.
// FP32 path: FLOAT=float, all macros expanded. See bake_mnn_kernels.py.
#ifndef MNN_BAKED_FP32_PREAMBLE
#define MNN_BAKED_FP32_PREAMBLE
typedef float FLOAT;
typedef float2 FLOAT2;
typedef float4 FLOAT4;
typedef float8 FLOAT8;
typedef float16 FLOAT16;
typedef float INPUT_TYPE;
typedef float OUTPUT_TYPE;
typedef float3 FLOAT3;
typedef float3 INPUT_TYPE3;
typedef float3 OUTPUT_TYPE3;
typedef float4 INPUT_TYPE4;
typedef float4 OUTPUT_TYPE4;
typedef float8 INPUT_TYPE8;
typedef float8 OUTPUT_TYPE8;
typedef float16 INPUT_TYPE16;
typedef float16 OUTPUT_TYPE16;
#define COMPUTE_FLOAT float
#define COMPUTE_FLOAT2 float2
#define COMPUTE_FLOAT3 float3
#define COMPUTE_FLOAT4 float4
#define COMPUTE_FLOAT8 float8
#define COMPUTE_FLOAT16 float16
#define CONVERT_FLOAT4(x) ((FLOAT4)(x))
#define CONVERT_FLOAT(x) ((FLOAT)(x))
#define CONVERT_COMPUTE_FLOAT4(x) ((float4)(x))
#define CONVERT_COMPUTE_FLOAT2(x) ((float2)(x))
#define CONVERT_COMPUTE_FLOAT3(x) ((float3)(x))
#define CONVERT_OUTPUT4(x) ((float4)(x))
#define CONVERT_INPUT4(x) ((float4)(x))
#define CONVERT_OUTPUT8(x) ((float8)(x))
#define CONVERT_INPUT8(x) ((float8)(x))
#define CONVERT_OUTPUT3(x) ((float3)(x))
#define CONVERT_INPUT3(x) ((float3)(x))
#define CONVERT_COMPUTE_FLOAT8(x) ((float8)(x))
#define CONVERT_COMPUTE_FLOAT16(x) ((float16)(x))
#define CONVERT_OUTPUT16(x) ((float16)(x))
#define CONVERT_INPUT16(x) ((float16)(x))
#define CONVERT_FLOAT3(x) ((float3)(x))
#define CONVERT_FLOAT2(x) ((float2)(x))
#define CONVERT_FLOAT8(x) ((float8)(x))
#define CONVERT_FLOAT16(x) ((float16)(x))
#define CONVERT_FLOAT(x) ((FLOAT)(x))
#define RI_F(img, smp, coord) read_imagef(img, smp, coord)
#define WI_F(img, coord, val) write_imagef(img, coord, val)
#define SAMPLER (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST)
#define COMPUTE_FLOAT float
#define COMPUTE_FLOAT4 float4
#define CONVERT_COMPUTE_FLOAT4(x) ((float4)(x))
#define CONVERT_OUTPUT4(x) ((float4)(x))
#define CONVERT_INPUT4(x) ((float4)(x))
#define OUTPUT_TYPE4 float4
#define INPUT_TYPE4 float4
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define GLOBAL_SIZE_DIM2 __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2) if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { return; }
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { return; }
#endif
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void pooling(GLOBAL_SIZE_3_DIMS __global const FLOAT *input,
                      __private const int2 input_shape,
                      __private const int2 output_shape,
                      __private const int2 pad_shape,
                      __private const int2 stride_shape,
                      __private const int2 kernel_shape,
                      __global FLOAT *output,
                      __private const int channel_block) {
                          
    const int ow_idx   = get_global_id(0);
    const int b_oh_idx = get_global_id(1);
    const int c_idx    = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(ow_idx, b_oh_idx, c_idx);
    
    const int b_idx = b_oh_idx / output_shape.x;
    const int oh_idx = b_oh_idx % output_shape.x;
    const int iw_start = mad24(ow_idx, stride_shape.y, -pad_shape.y);
    const int ih_start = mad24(oh_idx, stride_shape.x, -pad_shape.x);
    
    #ifdef POOL_AVG
    FLOAT4 result = (FLOAT4)(0);
    const int inp_offset = (((b_idx*channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start)*4;
    int total_count = 0;
    for(int kh=0; kh<kernel_shape.x; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }
        for(int kw=0; kw<kernel_shape.y; kw++) {
            int iw_cur = iw_start + kw;
            if(iw_cur < 0 || iw_cur >= input_shape.y) {
                continue;
            }
            FLOAT4 inp_data = vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4);
            result += inp_data;
            total_count++;
        }
    }
    result = result / (FLOAT4)(1.0*total_count);
    #else
    FLOAT4 result = (FLOAT4)(-FLT_MAX);
    const int inp_offset = (((b_idx*channel_block+c_idx)*input_shape.x+ih_start)*input_shape.y+iw_start)*4;
    for(int kh=0; kh<kernel_shape.x; kh++) {
        int ih_cur = ih_start + kh;
        if(ih_cur < 0 || ih_cur >= input_shape.x) {
            continue;
        }
        for(int kw=0; kw<kernel_shape.y; kw++) {
            int iw_cur = iw_start + kw;
            if(iw_cur < 0 || iw_cur >= input_shape.y) {
                continue;
            }
            FLOAT4 inp_data = vload4(0, input+inp_offset+(kh*input_shape.y+kw)*4);
            result = fmax(result, inp_data);
        }
    }
    #endif
    
    const int out_offset = (((b_idx*channel_block + c_idx)*output_shape.x + oh_idx)* output_shape.y + ow_idx)*4;
    vstore4(result, 0, output+out_offset);
}
