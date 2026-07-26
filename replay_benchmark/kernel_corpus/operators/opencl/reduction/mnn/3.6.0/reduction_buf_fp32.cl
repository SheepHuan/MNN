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
