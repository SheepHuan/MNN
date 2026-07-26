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
#define AS_INPUT_DATA4(x) ((float4)(x))
#define AS_INPUT_DATA8(x) ((float8)(x))
#define AS_INPUT_DATA16(x) ((float16)(x))
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
__kernel void binary_cfunction_buf(__private int global_dim0, __private int global_dim1,
                         __global FLOAT* input0,
                         __private const int offsetC,
                         __private const int strideC,
                         __global FLOAT* input1, __global FLOAT* output,
                         __private const int width,//[offsetA, offsetB, offsetC, 0]
                         __private const int height//[strideA, strideB, strideC, 0]
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));// [X/16, Y]
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset_11 = offsetC + pos.x * 8 + pos.y * strideC;
        int offset_12 = offset_11 + width;
        int offset_21 = offset_11 + strideC * height;
        int offset_22 = offset_21 + width;

        FLOAT8 in_11 = vload8(0, input0 + offset_11);
        FLOAT8 in_12 = vload8(0, input0 + offset_12);
        FLOAT8 in_21 = vload8(0, input0 + offset_21);
        FLOAT8 in_22 = vload8(0, input0 + offset_22);
        FLOAT8 in_cx = vload8(0, input1 + pos.x * 8 + pos.y * width);

        in_12 = in_12 + in_cx;
        in_21 = in_12 + in_21;
        in_12 = in_22 + in_12;
        in_22 = in_22 + in_21;
        in_12 = in_11 + in_12;

        vstore8(in_21, 0, output + offset_21);
        vstore8(in_22, 0, output + offset_22);
        vstore8(in_12, 0, output + offset_12);
    }
}

#ifndef OPERATOR
#define OPERATOR in0+in1
#endif

__kernel void binary_function_buf(__private int global_dim0, __private int global_dim1,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 baseOffsets,//[offsetA, offsetB, offsetC, 0]
                         __private const int4 strides//[strideA, strideB, strideC, 0]
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));// [X/16, Y]
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int baseOffsetA = baseOffsets.x;
        const int baseOffsetB = baseOffsets.y;
        const int baseOffsetC = baseOffsets.z;
        const int strideA = strides.x;
        const int strideB = strides.y;
        const int strideC = strides.z;
        
        
        int offsetA = pos.x * 8 + pos.y * VEC_H * strideA + baseOffsetA;
        int offsetB = pos.x * 8 + pos.y * VEC_H * strideB + baseOffsetB;
        int offsetC = pos.x * 8 + pos.y * VEC_H * strideC + baseOffsetC;

        {
            FLOAT8 in0 = vload8(0, input0 + offsetA);
            FLOAT8 in1 = vload8(0, input1 + offsetB);
            FLOAT8 out = OPERATOR;
            vstore8(out, 0, output + offsetC);
        }
        #if VEC_H >= 2
        {
            offsetA += strideA;
            offsetB += strideB;
            offsetC += strideC;
            FLOAT8 in0 = vload8(0, input0 + offsetA);
            FLOAT8 in1 = vload8(0, input1 + offsetB);
            FLOAT8 out = OPERATOR;
            vstore8(out, 0, output + offsetC);
        }
        #endif
        #if VEC_H == 4
        {
            offsetA += strideA;
            offsetB += strideB;
            offsetC += strideC;
            FLOAT8 in0 = vload8(0, input0 + offsetA);
            FLOAT8 in1 = vload8(0, input1 + offsetB);
            FLOAT8 out = OPERATOR;
            vstore8(out, 0, output + offsetC);
        }
        {
            offsetA += strideA;
            offsetB += strideB;
            offsetC += strideC;
            FLOAT8 in0 = vload8(0, input0 + offsetA);
            FLOAT8 in1 = vload8(0, input1 + offsetB);
            FLOAT8 out = OPERATOR;
            vstore8(out, 0, output + offsetC);
        }
        #endif
    }
}
