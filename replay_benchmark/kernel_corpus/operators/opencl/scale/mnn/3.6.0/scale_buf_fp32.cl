// Baked from MNN historical kernel. Copyright Alibaba Group Holding Limited.
// FP32 path: FLOAT=float, all macros expanded. See bake_mnn_kernels.py.
#ifndef MNN_BAKED_FP32_PREAMBLE
#define MNN_BAKED_FP32_PREAMBLE
typedef float FLOAT;
typedef float2 FLOAT2;
typedef float4 FLOAT4;
typedef float8 FLOAT8;
typedef float16 FLOAT16;
#define CONVERT_FLOAT4(x) ((FLOAT4)(x))
#define CONVERT_FLOAT(x) ((FLOAT)(x))
#define COMPUTE_FLOAT float
#define COMPUTE_FLOAT4 float4
#define CONVERT_COMPUTE_FLOAT4(x) ((float4)(x))
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define GLOBAL_SIZE_DIM2 __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2) if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { return; }
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { return; }
#endif
#define GLOBAL_SIZE_2_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

__kernel void scale_buf(GLOBAL_SIZE_2_DIMS
                        __global const FLOAT* input,
                        __global const FLOAT* scale,
#ifdef BIAS
                        __global const FLOAT* bias,
#endif
                        __global FLOAT* output,
                        __private const int channelBlock,
                        __private const int batch,
                        __private const int inside) {

    const int x = get_global_id(0); // inside(width * height)
    const int y = get_global_id(1); // channelBlock * batch
    
    DEAL_NON_UNIFORM_DIM2(x, y);

    const int out_c_idx = y % channelBlock;
    const int out_b_idx = y / channelBlock;
    const int offset = ((out_b_idx + out_c_idx * batch) * inside + x) * 4;
    COMPUTE_FLOAT4 in_value    = CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset));
    COMPUTE_FLOAT4 scale_value = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, scale));
    #ifdef BIAS
    COMPUTE_FLOAT4 bias_value = CONVERT_COMPUTE_FLOAT4(vload4(out_c_idx, bias));
    COMPUTE_FLOAT4 out_value  = in_value * scale_value + bias_value;
    #else
    COMPUTE_FLOAT4 out_value  = in_value * scale_value;
    #endif
    vstore4(CONVERT_FLOAT4(out_value), 0, output+offset);
}
