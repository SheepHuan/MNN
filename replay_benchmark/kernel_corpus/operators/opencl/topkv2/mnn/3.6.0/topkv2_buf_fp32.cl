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
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define THREAD_NUMBER 128
#define LOCAL_K 8
#define CANDIDATE_NUMBER (THREAD_NUMBER * LOCAL_K)

#ifdef IS_INT
typedef int DTYPE;
#else
typedef FLOAT DTYPE;
#endif

inline bool afterAsc(DTYPE aValue, int aIndex, DTYPE bValue, int bIndex) {
    if (aValue > bValue) {
        return true;
    }
    if (aValue < bValue) {
        return false;
    }
    return aIndex > bIndex;
}

inline bool better(DTYPE aValue, int aIndex, DTYPE bValue, int bIndex) {
    if (bIndex < 0) {
        return true;
    }
    if (aIndex < 0) {
        return false;
    }
#ifdef SORT_DESC
    if (aValue > bValue) {
        return true;
    }
    if (aValue < bValue) {
        return false;
    }
#else
    if (aValue < bValue) {
        return true;
    }
    if (aValue > bValue) {
        return false;
    }
#endif
    return aIndex < bIndex;
}

__kernel void topkv2_buf(GLOBAL_SIZE_3_DIMS
                       __global DTYPE *outValue,
                       __global int *outIndex,
                       __global const DTYPE *inValue,
                       __private const int rowSize,
                       __private const int k,
                       __private const int numRows) {
    const int gid0 = get_global_id(0);
    const int gid1 = get_global_id(1);
    const int gid2 = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(gid0, gid1, gid2);
    
    const int tid = get_local_id(0);
    const int row = get_group_id(1);

    if (tid >= THREAD_NUMBER || row >= numRows) {
        return;
    }

#ifdef IS_INT
    const DTYPE initWorst = (DTYPE)(2147483647);
    const DTYPE initBestWorst = (DTYPE)(-2147483648);
#else
    const DTYPE initWorst = (DTYPE)(FLT_MAX);
    const DTYPE initBestWorst = (DTYPE)(-FLT_MAX);
#endif

    DTYPE localValue[LOCAL_K];
    int localIndex[LOCAL_K];
#ifdef SORT_DESC
    for (uint i = 0; i < LOCAL_K; ++i) {
        localValue[i] = initBestWorst;
        localIndex[i] = -1;
    }
#else
    for (uint i = 0; i < LOCAL_K; ++i) {
        localValue[i] = initWorst;
        localIndex[i] = -1;
    }
#endif

    const __global DTYPE *rowIn = inValue + row * rowSize;

    for (int i = tid; i < rowSize; i += THREAD_NUMBER) {
        const DTYPE value = rowIn[i];
        if (!better(value, i, localValue[LOCAL_K - 1], localIndex[LOCAL_K - 1])) {
            continue;
        }

        uint insertPos = LOCAL_K;
        for (uint j = 0; j < LOCAL_K; ++j) {
            if (better(value, i, localValue[j], localIndex[j])) {
                insertPos = j;
                break;
            }
        }
        if (insertPos >= LOCAL_K) {
            continue;
        }
        for (uint j = LOCAL_K - 1; j > insertPos; --j) {
            localValue[j] = localValue[j - 1];
            localIndex[j] = localIndex[j - 1];
        }
        localValue[insertPos] = value;
        localIndex[insertPos] = i;
    }

    __local DTYPE sharedValue[CANDIDATE_NUMBER];
    __local int sharedIndex[CANDIDATE_NUMBER];
    const uint base = tid * LOCAL_K;
    for (uint i = 0; i < LOCAL_K; ++i) {
        sharedValue[base + i] = localValue[i];
        sharedIndex[base + i] = localIndex[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint size = 2; size <= CANDIDATE_NUMBER; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint idx = tid; idx < CANDIDATE_NUMBER; idx += THREAD_NUMBER) {
                const uint ixj = idx ^ stride;
                if (ixj <= idx) {
                    continue;
                }
                bool up = ((idx & size) == 0);
#ifdef SORT_DESC
                up = !up;
#endif

                const bool after = afterAsc(sharedValue[idx], sharedIndex[idx], sharedValue[ixj], sharedIndex[ixj]);
                if (up == after) {
                    const DTYPE tValue = sharedValue[idx];
                    sharedValue[idx] = sharedValue[ixj];
                    sharedValue[ixj] = tValue;
                    const int tIndex = sharedIndex[idx];
                    sharedIndex[idx] = sharedIndex[ixj];
                    sharedIndex[ixj] = tIndex;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (tid == 0) {
        __global DTYPE *rowOut = outValue + row * k;
        __global int *rowIdx = outIndex + row * k;
        const int realK = min(k, rowSize);
        for (int i = 0; i < realK; ++i) {
            rowOut[i] = sharedValue[i];
            rowIdx[i] = sharedIndex[i];
        }
    }
}
