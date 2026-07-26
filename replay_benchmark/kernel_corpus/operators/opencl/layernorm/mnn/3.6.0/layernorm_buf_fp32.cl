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
__kernel void layernorm_buf(__private int global_dim0, __private int global_dim1,
                        __global const FLOAT * input,
                        __global FLOAT * output,
                        __private const int inside,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
#if LOCAL_SIZE > 1
    float local sum_mnn[LOCAL_SIZE];
    #ifndef RMSNORM
    float local sum_mean_mnn[LOCAL_SIZE];
    #endif
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int lid = get_local_id(0);
        const int offset = pos.y * inside;
        const int inside_v4 = (inside + 3) >> 2;
        #ifdef PACK_LEAVE
        const int loop = inside_v4 - 1;
        const int inside_remain = inside - ((inside_v4-1) << 2);
        #else
        const int loop = inside_v4;
        #endif
        
        float4 in_sum = 0;
        int index = lid;
        #ifdef RMSNORM
        float4 mean = (float4)0;
        #else
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += in;
        }
        sum_mean_mnn[lid] = in_sum.x + in_sum.y + in_sum.z+ in_sum.w;
        
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i){
                float in = input[offset + index * 4 + i];
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + in;
            }
        }
        #endif
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + sum_mean_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum_mean_mnn[0] / (float4)inside;
        #endif

        in_sum = 0;
        index = lid;
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum_mnn[lid] = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i) {
                float in = input[offset + index * 4 + i];
                in = (in - mean.x) * (in - mean.x);
                sum_mnn[lid] = sum_mnn[lid] + in;
            }
        }
        #endif
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum_mnn[0] / (float4)inside;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        index = lid;
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            #ifdef GAMMA_BETA
            float4 out = (in - mean) * value * convert_float4(vload4(index, gamma)) + convert_float4(vload4(index, beta));
            #else
            float4 out = (in - mean) * value;
            #endif
            vstore4(CONVERT_FLOAT4(out), index, output + offset);
        }
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i){
                float in = input[offset + index * 4 + i];
                #ifdef GAMMA_BETA
                float out = (in - mean.x) * value.x * (float)gamma[index * 4 + i] + (float)beta[index * 4 + i];
                #else
                float out = (in - mean.x) * value.x;
                #endif
                output[offset + index * 4 + i] = out;
            }
        }
        #endif
    }
#else
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int offset = pos.y * inside;
        float in_sum = 0;
        #ifdef RMSNORM
        float mean = 0;
        #else
        for(int index = 0; index < inside; index++){
            in_sum += (float)input[offset + index];
        }
        float mean = in_sum / inside;
        #endif

        in_sum = 0;
        for(int index = 0; index < inside; index++){
            float in = (float)input[offset + index];
            in_sum += (in - mean) * (in - mean);
        }
        float square_sum = in_sum / inside;
        float value = 1.0f / sqrt(square_sum + epsilon);
        for(int i = 0; i < inside; ++i){
            float in = input[offset + i];
            #ifdef GAMMA_BETA
            float out = (in - mean) * value * (float)gamma[i] + (float)beta[i];
            #else
            float out = (in - mean) * value;
            #endif
            output[offset + i] = out;
        }
    }

#endif
}
