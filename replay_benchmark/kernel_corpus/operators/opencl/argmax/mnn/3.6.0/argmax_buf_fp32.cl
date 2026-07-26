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
#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define ARGMAX_SELECT(A, B, C, D)          \
    if(A.x < B.x){ A.x = B.x; C.x = D; }    \
    if(A.y < B.y){ A.y = B.y; C.y = D; }    \
    if(A.z < B.z){ A.z = B.z; C.z = D; }    \
    if(A.w < B.w){ A.w = B.w; C.w = D; }    

#define ARGMIN_SELECT(A, B, C, D)    \
    if(A.x > B.x){ A.x = B.x; C.x = D; }    \
    if(A.y > B.y){ A.y = B.y; C.y = D; }    \
    if(A.z > B.z){ A.z = B.z; C.z = D; }    \
    if(A.w > B.w){ A.w = B.w; C.w = D; }    


__kernel void argmax_buf(GLOBAL_SIZE_3_DIMS
                        __global const FLOAT* input,
                        __global int* output,
                        __private const int inside,
                        __private const int outside,
                        __private const int dim){
    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    int index = 0;
#ifdef ARGMAX
    FLOAT maxValue = (FLOAT)-FLT_MAX;
#else
FLOAT maxValue = (FLOAT)FLT_MAX;
#endif
    const int offset = z * dim * inside + y;
#if ARGMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT local reduce[ARGMAX_LOCAL_SIZE];
    int local index_reduce[ARGMAX_LOCAL_SIZE];
        
    for (int i=lid; i < dim; i+=ARGMAX_LOCAL_SIZE) {
        FLOAT value = input[offset + i * inside];
#ifdef ARGMAX
        if(maxValue < value){ maxValue = value; index = i; }
#else
        if(maxValue > value){ maxValue = value; index = i; }
#endif
    }
    reduce[lid] = maxValue;
    index_reduce[lid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ARGMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i){
#ifdef ARGMAX
            if(reduce[lid] < reduce[lid + i]){reduce[lid] = reduce[lid + i]; index_reduce[lid] = index_reduce[lid + i];}
#else
            if(reduce[lid] > reduce[lid + i]){reduce[lid] = reduce[lid + i]; index_reduce[lid] = index_reduce[lid + i];}
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        output[z * inside + y] = index_reduce[0];
    }
#else
    for(int i = 0; i < dim; ++i){
        FLOAT value = input[ + offset + i * inside];
#ifdef ARGMAX
        if(maxValue < value){ maxValue = value; index = i; }
#else
        if(maxValue > value){ maxValue = value; index = i; }
#endif
    }
    output[z * inside + y] = index;
#endif
}


__kernel void argmax_v4_buf(GLOBAL_SIZE_3_DIMS
                        __global const FLOAT* input,
                        __global int* output,
                        __private const int inside,
                        __private const int outside,
                        __private const int dim){
    const int x = get_global_id(0);
    const int y = get_global_id(1) << 2; // inside
    const int z = get_global_id(2); // outside
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    int4 index = 0;
#ifdef ARGMAX
    FLOAT4 maxValue = (FLOAT4)-FLT_MAX;
#else
    FLOAT4 maxValue = (FLOAT4)FLT_MAX;
#endif
    const int offset = z * dim * inside + y;
#if ARGMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT4 local reduce[ARGMAX_LOCAL_SIZE];
    int4 local index_reduce[ARGMAX_LOCAL_SIZE];
        
    for (int i=lid; i < dim; i+=ARGMAX_LOCAL_SIZE) {
        FLOAT4 value = vload4(0, input + offset + i * inside);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    reduce[lid] = maxValue;
    index_reduce[lid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ARGMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i){
#ifdef ARGMAX
            if(reduce[lid].x < reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y < reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z < reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w < reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#else
            if(reduce[lid].x > reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y > reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z > reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w > reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        vstore4(index_reduce[0], 0, output + z * inside + y);
    }
#else
    for(int i = 0; i < dim; ++i){
        FLOAT4 value = vload4(0, input + offset + i * inside);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    vstore4(index, 0, output + z * inside + y);
#endif
}
