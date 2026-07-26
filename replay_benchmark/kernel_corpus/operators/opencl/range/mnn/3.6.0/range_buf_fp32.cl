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
#define CONVERT_FLOAT4(x) ((FLOAT4)(x))
#define CONVERT_FLOAT(x) ((FLOAT)(x))
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
