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

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void reduct_buf(GLOBAL_SIZE_3_DIMS
                              __global const INPUT_TYPE *input,
                              __global OUTPUT_TYPE *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    INPUT_TYPE out = (INPUT_TYPE)VALUE;
    const int offset = z * dim * inside + y;
    
#if REDUCT_LOCAL_SIZE > 4
    const int lid = get_local_id(0);
    INPUT_TYPE local sum_mnn[REDUCT_LOCAL_SIZE];
    for(int i = lid; i < dim; i+=REDUCT_LOCAL_SIZE){
        INPUT_TYPE in = (INPUT_TYPE)input[offset + i * inside];
        out = OPERATE(out, in);
    }
    sum_mnn[lid] = out;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = REDUCT_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = OPERATE(sum_mnn[lid], sum_mnn[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out = sum_mnn[0];
#else
    for(int i = 0; i < dim; ++i){
        INPUT_TYPE in = (INPUT_TYPE)input[offset + i * inside];
        out = OPERATE(out, in);
    }
#endif

#ifdef GET_AVG
    out = out / dim;
#endif
    output[z * inside + y] = (OUTPUT_TYPE)out;
}

__kernel void reduct_v4_buf(GLOBAL_SIZE_3_DIMS
                              __global const INPUT_TYPE *input,
                              __global OUTPUT_TYPE *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    const int offset = z * dim * inside + (y << 2);
    
#if REDUCT_LOCAL_SIZE > 4
    const int lid = get_local_id(0);
    INPUT_TYPE4 local sum_mnn[REDUCT_LOCAL_SIZE];
    for(int i = lid; i < dim; i+=REDUCT_LOCAL_SIZE){
        INPUT_TYPE4 in = vload4(0, input + offset + i * inside);
        out = OPERATE(out, in);
    }
    sum_mnn[lid] = out;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = REDUCT_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = OPERATE(sum_mnn[lid], sum_mnn[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out = sum_mnn[0];
#else
    for(int i = 0; i < dim; ++i){
        INPUT_TYPE4 in = vload4(0, input + offset + i * inside);
        out = OPERATE(out, in);
    }
#endif

#ifdef GET_AVG
    out = out / (INPUT_TYPE4)dim;
#endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + z * inside + (y << 2));
}
