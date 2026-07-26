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

__kernel void cast_buf(GLOBAL_SIZE_2_DIMS
                            __global INPUT_TYPE* input,
                            __global OUTPUT_TYPE* output,
                            __private const int size
                            ) {
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(idx, idy);
    const int inp_offset = idx * 4;
#ifdef PACK_LEAVE
    if(inp_offset + 3 >= size){
        int remain = size - inp_offset;
        for(int i = 0; i < remain; ++i){
            #ifdef TO_BOOL
            int value = (int)input[inp_offset + i];
            value = value == 0 ? 0 : 1;
            output[inp_offset + i] = (OUTPUT_TYPE)value;
            #else
            output[inp_offset + i] = (OUTPUT_TYPE)input[inp_offset + i];
            #endif
        }
    }else {
#endif
        #ifdef TO_BOOL
        int4 value = convert_int4(vload4(0, input + inp_offset));
        value = value == (int4)0 ? (int4)0 : (int4)1;
        vstore4(CONVERT_OUTPUT4(value), 0, output + inp_offset);
        #else
        vstore4(CONVERT_OUTPUT4(vload4(0, input + inp_offset)), 0, output + inp_offset);
        #endif
#ifdef PACK_LEAVE
    }
#endif
}
