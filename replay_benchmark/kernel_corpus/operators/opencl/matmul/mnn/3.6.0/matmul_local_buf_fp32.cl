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
/*
 \
 #define OPWM 64                // The outputsize-per-workgroup in dimension M
 #define OPWN 128                // The outputsize-per-workgroup in dimension N
 #define CPWK 8                 // The cachesize-per-workgroup in dimension K
 #define OPTM 4                 // The outputsize-per-thread in dimension M
 #define OPTN 8                 // The outputsize-per-thread in dimension N
 */
#define TPWM (OPWM/OPTM)        // The threadsize-per-workgroup in dimension M
#define TPWN (OPWN/OPTN)        // The threadsize-per-workgroup in dimension N
#define LPTA ((CPWK*OPWM)/(TPWM*TPWN)) // Loads-num-per-thread for A
#define LPTB ((CPWK*OPWN)/(TPWM*TPWN)) // Loads-num-per-thread for B

// vetorize +  pragma unroll
__kernel void matmul_local_buf(const int M, const int N, const int K,
                      __global const FLOAT* A,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char* B,
                        __global const float* dequantScale,
                        __global const float* dequantOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar* B,
                        __global const float* dequantScale,
                        __global const float* dequantOffset,
#else
                        __global const FLOAT* B,
#endif
#ifdef BIAS
                      __global const FLOAT* bias,
#endif
                      __global FLOAT* C) {

    // Local thread id
    const int lidm = get_local_id(0); // Local row ID
    const int lidn = get_local_id(1); // Local col ID
    // group id
    const int offsetM = get_group_id(0) * OPWM; // Work-group offset M
    const int offsetN = get_group_id(1) * OPWN; // Work-group offset N

    // Local memory for work-group cache of A and B
    __local FLOAT Alocal[CPWK][OPWM];
    __local FLOAT Blocal[OPWN][CPWK+2];

    // Allocate register space
    COMPUTE_FLOAT sum[OPTM][OPTN];

    // Initialise the accumulation registers
    for (int wm=0; wm<OPTM; wm++) {
        for (int wn=0; wn<OPTN; wn++) {
            sum[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numLoops = K/CPWK;
    int lid = lidn*TPWM + lidm;

    for (int t=0; t<numLoops; t++) {
        // Load one work-group of A and B into local memory
        for (int la=0; la<LPTA; la++) {
            int id = la*TPWN*TPWM + lid;
            int row = id % OPWM;
            int col = id / OPWM;
            int tiledIndex = CPWK*t + col;
            #ifdef TRANSPOSE_A
            // [K, M]
            Alocal[col][row] = A[tiledIndex*M + (offsetM + row)];
            #else
            // [M, K]
            Alocal[col][row] = A[(offsetM + row)*K + tiledIndex];
            #endif
        }

        for (int la=0; la<LPTB; la++) {
            int id = la*TPWN*TPWM + lid;
            int row = id % OPWN;
            int col = id / OPWN;
            int tiledIndex = CPWK*t + col;
            #ifdef TRANSPOSE_B
            // [N, K]
            Blocal[row][col] = B[(offsetN + row)*K + tiledIndex];
            #else
            // [K, N]
            Blocal[row][col] = B[tiledIndex*N + offsetN + row];
            #endif
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        
        // Perform the computation
        FLOAT4 A_k0, B_k0[OPTN];
        {
            int row = lidm;
            int col = lidn;
            
            A_k0.s0 = Alocal[0][row];
            A_k0.s1 = Alocal[1][row];
            A_k0.s2 = Alocal[2][row];
            A_k0.s3 = Alocal[3][row];
            
            #pragma unroll
            for (int wn=0; wn<OPTN; wn++) {
                B_k0[wn].s0 = Blocal[col][0];
                B_k0[wn].s1 = Blocal[col][1];
                B_k0[wn].s2 = Blocal[col][2];
                B_k0[wn].s3 = Blocal[col][3];
                sum[0][wn] += dot(A_k0, B_k0[wn]);
                col += TPWN;
            }
            
            #pragma unroll
            for(int wm=1; wm<OPTM; wm++) {
                row += TPWM;
                A_k0.s0 = Alocal[0][row];
                A_k0.s1 = Alocal[1][row];
                A_k0.s2 = Alocal[2][row];
                A_k0.s3 = Alocal[3][row];
                for (int wn=0; wn<OPTN; wn++) {
                    sum[wm][wn] += dot(A_k0, B_k0[wn]);
                }
            }
        }

        {
            int col = lidn;
            for (int wn=0; wn<OPTN; wn++) {
                B_k0[wn].s0 = Blocal[col][4];
                B_k0[wn].s1 = Blocal[col][5];
                B_k0[wn].s2 = Blocal[col][6];
                B_k0[wn].s3 = Blocal[col][7];
                col += TPWN;
            }
            int row = lidm;
            for (int wm=0; wm<OPTM; wm++) {
                A_k0.s0 = Alocal[4][row];
                A_k0.s1 = Alocal[5][row];
                A_k0.s2 = Alocal[6][row];
                A_k0.s3 = Alocal[7][row];
                for (int wn=0; wn<OPTN; wn++) {
                    sum[wm][wn] += dot(A_k0, B_k0[wn]);
                }
                row += TPWM;
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int wm=0; wm<OPTM; wm++) {
        int globalRow = offsetM + lidm + wm*TPWM;
        for (int wn=0; wn<OPTN; wn++) {
            int globalCol = offsetN + lidn + wn*TPWN;
            #ifdef BIAS
            sum[wm][wn] += bias[globalCol];
            #endif
            C[globalRow*N + globalCol] = sum[wm][wn];
        }
    }
}


// double buffer
__kernel void matmul_local_double_buf(const int M, const int N, const int K,
                      __global const FLOAT* A,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                        __global const char* B,
                        __global const float* dequantScale,
                        __global const float* dequantOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                        __global const uchar* B,
                        __global const float* dequantScale,
                        __global const float* dequantOffset,
#else
                        __global const FLOAT* B,
#endif
#ifdef BIAS
                      __global const FLOAT* bias,
#endif
                      __global FLOAT* C) {

    // Local thread id
    const ushort lidm = get_local_id(0); // Local row ID
    const ushort lidn = get_local_id(1); // Local col ID
    // group id
    const ushort offsetM = get_group_id(0) * OPWM; // Work-group offset M
    const ushort offsetN = get_group_id(1) * OPWN; // Work-group offset N

    // Local memory for work-group cache of A and B
    __local FLOAT AlocalR[CPWK][OPWM];
    __local FLOAT BlocalR[OPWN][CPWK+2];
    __local FLOAT AlocalC[CPWK][OPWM];
    __local FLOAT BlocalC[OPWN][CPWK+2];
    
    // Allocate register space
    COMPUTE_FLOAT sum[OPTM][OPTN];

    // Initialise the accumulation registers
    for (ushort wm=0; wm<OPTM; wm++) {
        for (ushort wn=0; wn<OPTN; wn++) {
            sum[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const ushort numLoops = K/CPWK;
    ushort lid = lidn*TPWM + lidm;

    for (ushort t=0; t<numLoops; t++) {
        // Load one work-group of A and B into local memory
        for (ushort la=0; la<LPTA; la++) {
            ushort id = la*TPWN*TPWM + lid;
            ushort row = id % OPWM;
            ushort col = id / OPWM;
            ushort tiledIndex = CPWK*t + col;
            #ifdef TRANSPOSE_A
            // [K, M]
            AlocalR[col][row] = A[tiledIndex*M + (offsetM + row)];
            #else
            // [M, K]
            AlocalR[col][row] = A[(offsetM + row)*K + tiledIndex];
            #endif
        }

        for (ushort la=0; la<LPTB; la++) {
            ushort id = la*TPWN*TPWM + lid;
            ushort row = id % OPWN;
            ushort col = id / OPWN;
            ushort tiledIndex = CPWK*t + col;
            #ifdef TRANSPOSE_B
            // [N, K]
            BlocalR[row][col] = B[(offsetN + row)*K + tiledIndex];
            #else
            // [K, N]
            BlocalR[row][col] = B[tiledIndex*N + offsetN + row];
            #endif
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Loop over the values of a single tile
        
        // Perform the computation
        FLOAT4 A_k0, B_k0[OPTN];
        {
            ushort row = lidm;
            ushort col = lidn;
            
            A_k0.s0 = AlocalR[0][row];
            A_k0.s1 = AlocalR[1][row];
            A_k0.s2 = AlocalR[2][row];
            A_k0.s3 = AlocalR[3][row];
            
            #pragma unroll
            for (ushort wn=0; wn<OPTN; wn++) {
                B_k0[wn].s0 = BlocalR[col][0];
                B_k0[wn].s1 = BlocalR[col][1];
                B_k0[wn].s2 = BlocalR[col][2];
                B_k0[wn].s3 = BlocalR[col][3];
                sum[0][wn] += dot(A_k0, B_k0[wn]);
                col += TPWN;
            }
            
            #pragma unroll
            for(ushort wm=1; wm<OPTM; wm++) {
                row += TPWM;
                A_k0.s0 = AlocalR[0][row];
                A_k0.s1 = AlocalR[1][row];
                A_k0.s2 = AlocalR[2][row];
                A_k0.s3 = AlocalR[3][row];
                for (ushort wn=0; wn<OPTN; wn++) {
                    sum[wm][wn] += dot(A_k0, B_k0[wn]);
                }
            }
        }

        {
            int col = lidn;
            for (ushort wn=0; wn<OPTN; wn++) {
                B_k0[wn].s0 = BlocalR[col][4];
                B_k0[wn].s1 = BlocalR[col][5];
                B_k0[wn].s2 = BlocalR[col][6];
                B_k0[wn].s3 = BlocalR[col][7];
                col += TPWN;
            }
            ushort row = lidm;
            for (ushort wm=0; wm<OPTM; wm++) {
                A_k0.s0 = AlocalR[4][row];
                A_k0.s1 = AlocalR[5][row];
                A_k0.s2 = AlocalR[6][row];
                A_k0.s3 = AlocalR[7][row];
                for (ushort wn=0; wn<OPTN; wn++) {
                    sum[wm][wn] += dot(A_k0, B_k0[wn]);
                }
                row += TPWM;
            }
        }
        
        t++;
        // Loop over the values of a single tile
        // Load one work-group of A and B into local memory
        for (ushort la=0; la<LPTA; la++) {
            ushort id = la*TPWN*TPWM + lid;
            ushort row = id % OPWM;
            ushort col = id / OPWM;
            ushort tiledIndex = CPWK*t + col;
            #ifdef TRANSPOSE_A
            // [K, M]
            AlocalC[col][row] = A[tiledIndex*M + (offsetM + row)];
            #else
            // [M, K]
            AlocalC[col][row] = A[(offsetM + row)*K + tiledIndex];
            #endif
        }

        for (ushort la=0; la<LPTB; la++) {
            ushort id = la*TPWN*TPWM + lid;
            ushort row = id % OPWN;
            ushort col = id / OPWN;
            ushort tiledIndex = CPWK*t + col;
            #ifdef TRANSPOSE_B
            // [N, K]
            BlocalC[row][col] = B[(offsetN + row)*K + tiledIndex];
            #else
            // [K, N]
            BlocalC[row][col] = B[tiledIndex*N + offsetN + row];
            #endif
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Perform the computation
        {
            ushort row = lidm;
            ushort col = lidn;
            
            A_k0.s0 = AlocalC[0][row];
            A_k0.s1 = AlocalC[1][row];
            A_k0.s2 = AlocalC[2][row];
            A_k0.s3 = AlocalC[3][row];
            
            #pragma unroll
            for (ushort wn=0; wn<OPTN; wn++) {
                B_k0[wn].s0 = BlocalC[col][0];
                B_k0[wn].s1 = BlocalC[col][1];
                B_k0[wn].s2 = BlocalC[col][2];
                B_k0[wn].s3 = BlocalC[col][3];
                sum[0][wn] += dot(A_k0, B_k0[wn]);
                col += TPWN;
            }
            
            #pragma unroll
            for(ushort wm=1; wm<OPTM; wm++) {
                row += TPWM;
                A_k0.s0 = AlocalC[0][row];
                A_k0.s1 = AlocalC[1][row];
                A_k0.s2 = AlocalC[2][row];
                A_k0.s3 = AlocalC[3][row];
                for (ushort wn=0; wn<OPTN; wn++) {
                    sum[wm][wn] += dot(A_k0, B_k0[wn]);
                }
            }
        }

        {
            ushort col = lidn;
            for (ushort wn=0; wn<OPTN; wn++) {
                B_k0[wn].s0 = BlocalC[col][4];
                B_k0[wn].s1 = BlocalC[col][5];
                B_k0[wn].s2 = BlocalC[col][6];
                B_k0[wn].s3 = BlocalC[col][7];
                col += TPWN;
            }
            ushort row = lidm;
            for (ushort wm=0; wm<OPTM; wm++) {
                A_k0.s0 = AlocalC[4][row];
                A_k0.s1 = AlocalC[5][row];
                A_k0.s2 = AlocalC[6][row];
                A_k0.s3 = AlocalC[7][row];
                for (ushort wn=0; wn<OPTN; wn++) {
                    sum[wm][wn] += dot(A_k0, B_k0[wn]);
                }
                row += TPWM;
            }
        }
    }

    // Store the final results in C
    for (ushort wm=0; wm<OPTM; wm++) {
        ushort globalRow = offsetM + lidm + wm*TPWM;
        for (ushort wn=0; wn<OPTN; wn++) {
            ushort globalCol = offsetN + lidn + wn*TPWN;
            #ifdef BIAS
            sum[wm][wn] += bias[globalCol];
            #endif
            C[globalRow*N + globalCol] = sum[wm][wn];
        }
    }
}
