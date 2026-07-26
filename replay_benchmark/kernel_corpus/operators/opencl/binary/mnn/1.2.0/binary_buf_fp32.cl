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
#define OPERATOR in
__kernel void binary_buf(__private int global_dim0, __private int global_dim1,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NC4, HW
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset = pos.x * (shape.y*shape.z) + pos.y;
        FLOAT4 in0 = vload4(offset*isFull.x, input0);
        FLOAT4 in1 = vload4(offset*isFull.y, input1);
        if(isFull.x == 0) {
            in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
        }
        if(isFull.y == 0) {
            in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
        }
        FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
        vstore4(out, offset, output);
    }
}


__kernel void prelu_buf(__private int global_dim0, __private int global_dim1,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape//[N,H,W,C4]
                         ) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NC4, HW
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset = pos.x * (shape.y*shape.z) + pos.y;
        FLOAT4 in0 = vload4(offset, input0);
        FLOAT4 in1 = vload4(pos.x % shape.w, input1);
        FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
        vstore4(out, offset, output);
    }
}
