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
#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

// [K/4, M, 4] -> [alignK, alignM]
__kernel void transpose_pad(GLOBAL_SIZE_DIM2
                        const int alignM,
                        const int alignK,
                        const int M,
                        const int K,
                        const int area,
                        __global const FLOAT* input,
                        __global FLOAT* output
                        ) {
    const int idx_m4 = get_global_id(0); // idx M
    const int idx_k4 = get_global_id(1); // idx K
    UNIFORM_BOUNDRY_CHECK(idx_m4, idx_k4);

    const int idx_m = idx_m4 << 2;
    const int idx_k = idx_k4 << 2;
    const int K_4 = (K + 3) >> 2;
    const int in_offset_base  = (idx_k4 * M + idx_m) * 4;
    const int out_offset_base = idx_k * alignM + idx_m;
    
    FLOAT4 m0k4 = (idx_k4 >= K_4 || idx_m + 0 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base);
    FLOAT4 m1k4 = (idx_k4 >= K_4 || idx_m + 1 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 4);
    FLOAT4 m2k4 = (idx_k4 >= K_4 || idx_m + 2 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 8);
    FLOAT4 m3k4 = (idx_k4 >= K_4 || idx_m + 3 >= M) ? (FLOAT4)0 : vload4(0, input + in_offset_base + 12);
    
    vstore4((FLOAT4)(m0k4.x, m1k4.x, m2k4.x, m3k4.x), 0, output + out_offset_base);
    vstore4((FLOAT4)(m0k4.y, m1k4.y, m2k4.y, m3k4.y), 0, output + out_offset_base + alignM);
    vstore4((FLOAT4)(m0k4.z, m1k4.z, m2k4.z, m3k4.z), 0, output + out_offset_base + alignM + alignM);
    vstore4((FLOAT4)(m0k4.w, m1k4.w, m2k4.w, m3k4.w), 0, output + out_offset_base + alignM + alignM + alignM);
}

#ifndef M_VEC
#define M_VEC 1
#endif

// [alignM, alignN] -> [N/4, B, area, N4] (M = B * area)
__kernel void transpose_bias(GLOBAL_SIZE_DIM2
                        const int alignM,
                        const int alignN,
                        const int M,
                        const int N,
                        const int area,
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output
                        #ifdef PRELU
                        ,__global const FLOAT *slope_ptr
                        #endif
                        ) {
    int idx_m = get_global_id(0); // idx M
    int idx_n4 = get_global_id(1); // idx N
    UNIFORM_BOUNDRY_CHECK(idx_m, idx_n4);

    const int idx_n = idx_n4 << 2;

    idx_m = idx_m * M_VEC;
    FLOAT4 res1 = vload4(0, input1 + idx_n);
    #ifdef PRELU
    FLOAT4 slope_in = vload4(0, slope_ptr + idx_n);
    #endif
    #pragma unroll
    for(int i = 0; i < M_VEC; i++) {
        FLOAT4 res0 = vload4(0, input0 + (idx_m + i) * alignN + idx_n);
        FLOAT4 res = res0 + res1;
        #ifdef RELU
        res = fmax(res, (FLOAT4)0);
        #endif
        #ifdef RELU6
        res = clamp(res, (FLOAT4)0, (FLOAT4)6);
        #endif
        #ifdef PRELU
        res = select(res * slope_in, res, res >= 0);
        #endif
        vstore4(res, 0, output + ((idx_n4 * M + idx_m + i) << 2));
    }
}
