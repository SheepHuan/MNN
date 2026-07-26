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
#define CONVERT_FLOAT4(x) (convert_float4(x))
#define CONVERT_FLOAT(x) (convert_float(x))
#define CONVERT_FLOAT2(x) (convert_float2(x))
#define CONVERT_FLOAT3(x) (convert_float3(x))
#define CONVERT_FLOAT8(x) (convert_float8(x))
#define CONVERT_FLOAT16(x) (convert_float16(x))
#define CONVERT_COMPUTE_FLOAT2(x) (convert_float2(x))
#define CONVERT_COMPUTE_FLOAT3(x) (convert_float3(x))
#define CONVERT_COMPUTE_FLOAT4(x) (convert_float4(x))
#define CONVERT_COMPUTE_FLOAT8(x) (convert_float8(x))
#define CONVERT_COMPUTE_FLOAT16(x) (convert_float16(x))
#define CONVERT_OUTPUT4(x) (convert_float4(x))
#define CONVERT_INPUT4(x) (convert_float4(x))
#define CONVERT_OUTPUT8(x) (convert_float8(x))
#define CONVERT_INPUT8(x) (convert_float8(x))
#define CONVERT_OUTPUT3(x) (convert_float3(x))
#define CONVERT_INPUT3(x) (convert_float3(x))
#define CONVERT_OUTPUT16(x) (convert_float16(x))
#define CONVERT_INPUT16(x) (convert_float16(x))
#define AS_INPUT_DATA4(x) (convert_float4(x))
#define AS_INPUT_DATA8(x) (convert_float8(x))
#define AS_INPUT_DATA16(x) (convert_float16(x))
#define RI_F(img, smp, coord) read_imagef(img, smp, coord)
#define WI_F(img, coord, val) write_imagef(img, coord, val)
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define GLOBAL_SIZE_DIM2 __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2) if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { return; }
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { return; }
#endif
#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                                                   \
    }

__kernel void range_buf(GLOBAL_SIZE_2_DIMS
                            __global const INPUT_TYPE* input0,
                            __global const INPUT_TYPE* input2,
                            __global OUTPUT_TYPE* output,
                            __private const int size
                            ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(x, y);
                                
    int index = x << 2;
    int4 index4 = (int4)(index, index + 1, index + 2, index + 3);
    INPUT_TYPE start = input0[0];
    INPUT_TYPE step = input2[0];
    OUTPUT_TYPE4 value = (OUTPUT_TYPE4)start + CONVERT_OUTPUT4(index4) * (OUTPUT_TYPE4)step;
#ifdef PACK_LEAVE
    if(index + 3 >= size){
        OUTPUT_TYPE* value_ptr = (OUTPUT_TYPE*)&value;
        for(int i = 0; i < size - index; ++i){
            output[index + i] = value_ptr[i];
        }
    }else{
#endif
        vstore4(value, 0, output + index);
#ifdef PACK_LEAVE
    }
#endif
}
