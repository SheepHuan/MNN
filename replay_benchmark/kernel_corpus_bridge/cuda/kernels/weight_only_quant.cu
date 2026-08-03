// weight_only_quant.cu - Weight-only quantization GEMV/GEMM/CONV kernels (3.6.0)
//   source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
//   Faithful copy of all 20 __global__ kernels + extern "C" shims.
#include "corpus_common.cuh"
#include <cuda_fp16.h>

namespace MNN {
namespace Corpus {

const int TILE_DIM = 16; // GEMM_FpAIntB
const int GEMV_TILE = 64; // GEMV_FpAIntB (legacy, kept for INT8)
const int GEMV_TILE_V2 = 256; // P1: Optimized GEMV tile size
const int GEMV_OC_PER_BLOCK = 4; // P1: Output channels per block for GEMV
const int SUB_GEMM_TILE_DIM = 16; // GEMM_Int
const int BLOCK_SIZE = 256; // Quant / SumBq / Bias
const int DEQUANT_TILE_DIM = 32; // Dequant
const int WARP_SIZE = 32;

// R6: Pre-compute GEMV params: scale and adj_offset as float2 array
// Eliminates scattered offset loads and runtime computation in GEMV
// Grid: (ceil(total/256)), Block: 256
template<typename T>
__global__ void PrecomputeGemvParams(
    const T* __restrict__ scale,
    const T* __restrict__ offset,
    float2* __restrict__ params,   // [oc * num_qg], .x=scale, .y=adj_offset
    const int total                // oc * num_qg
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const float s = (float)scale[idx];
    const float o = (float)offset[idx];
    params[idx] = make_float2(s, o - 8.0f * s);  // adj_offset = offset - 8*scale
}

// 动态量化 A、计算 sum_A
template<typename T>
__global__ void QuantA(
    const T* A_sub_fp, int8_t* A_sub_q,
    T* scale_A_out, T* offset_A_out, int32_t* sum_A_q_out,
    const int M, const int K_i, const int lda
) {
    // 使用 union 复用共享内存，减少总占用量
    __shared__ union {
        float min_max_vals[2][BLOCK_SIZE / WARP_SIZE]; // 用于 min/max 的 warp 间规约
        int32_t sum_vals[BLOCK_SIZE / WARP_SIZE];      // 用于 sum 的 warp 间规约
        float scale_offset[2];                        // 用于向块内所有线程广播 scale 和 offset
    } smem;

    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    float my_min = FLT_MAX;
    float my_max = -FLT_MAX;

    for (int k = tid; k < K_i; k += BLOCK_SIZE) {
        float val = (float)A_sub_fp[m * lda + k];
        my_min = min(my_min, val);
        my_max = max(my_max, val);
    }

    // Warp 内规约：使用 __shfl_down_sync 在 warp 内无锁计算 min/max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        my_min = min(my_min, __shfl_down_sync(0xFFFFFFFF, my_min, offset));
        my_max = max(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
    }

    // Warp 间规约：每个 warp 的 0 号线程将 warp 结果写入共享内存
    if (lane_id == 0) {
        smem.min_max_vals[0][warp_id] = my_min;
        smem.min_max_vals[1][warp_id] = my_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        // 从共享内存加载各 warp 的结果
        if (lane_id < (BLOCK_SIZE / WARP_SIZE)) {
            my_min = smem.min_max_vals[0][lane_id];
            my_max = smem.min_max_vals[1][lane_id];
        } else {
            my_min = FLT_MAX;
            my_max = -FLT_MAX;
        }

        // 在第一个 warp 内部完成最终规约
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
            my_min = min(my_min, __shfl_down_sync(0xFFFFFFFF, my_min, offset));
            my_max = max(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
        }
    }
    // 线程 0 计算 scale/offset 并写入共享内存进行广播
    if (tid == 0) {
        float scale = (my_max - my_min) / 255.0f;
        float offset;
        if (abs(scale) > 1e-5) {
            offset = my_max - scale * 127.0f; // my_max 映射到 127, my_min 映射到 -128
        } else {
            scale = 1.0f;
            offset = my_max;
        }

        scale_A_out[m] = (T)scale;
        offset_A_out[m] = (T)offset;
        
        smem.scale_offset[0] = scale;
        smem.scale_offset[1] = offset;
    }
    __syncthreads();

    // 所有线程从共享内存读取 scale/offset, 并进行量化和求和
    const float s_scale_A = smem.scale_offset[0];
    const float s_offset_A = smem.scale_offset[1];

    int32_t my_sum_q = 0;
    for (int k = tid; k < K_i; k += BLOCK_SIZE) {
        float val_fp = (float)A_sub_fp[m * lda + k];
        int32_t val_q = roundf((val_fp - s_offset_A) / s_scale_A);
        int8_t a_q = (int8_t)max(-128, min(127, val_q));
        A_sub_q[m * K_i + k] = a_q;
        my_sum_q += a_q;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        my_sum_q += __shfl_down_sync(0xFFFFFFFF, my_sum_q, offset);
    }

    if (lane_id == 0) smem.sum_vals[warp_id] = my_sum_q;
    __syncthreads();

    if (warp_id == 0) {
        my_sum_q = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? smem.sum_vals[lane_id] : 0;
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
             my_sum_q += __shfl_down_sync(0xFFFFFFFF, my_sum_q, offset);
        }
    }

    // 线程 0 将最终的和写入全局内存
    if (tid == 0) sum_A_q_out[m] = my_sum_q;
}

__global__ void GEMM_Int8(
    const int8_t* A_q, const int8_t* B_q, int32_t* C_q,
    const int M, const int N, const int K_i,
    const int lda_q, const int ldb, const int ldc
) {
    __shared__ int8_t A_tile_s8[SUB_GEMM_TILE_DIM][SUB_GEMM_TILE_DIM];
    __shared__ int8_t B_tile_s8[SUB_GEMM_TILE_DIM][SUB_GEMM_TILE_DIM];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = block_row * SUB_GEMM_TILE_DIM + ty;
    const int col = block_col * SUB_GEMM_TILE_DIM + tx;

    int32_t acc = 0;
    const int num_k_tiles = UP_DIV(K_i, SUB_GEMM_TILE_DIM);

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * SUB_GEMM_TILE_DIM;

        A_tile_s8[ty][tx] = (row < M && (k_tile_base + tx) < K_i) ? A_q[row * lda_q + k_tile_base + tx] : 0;
        B_tile_s8[ty][tx] = (col < N && (k_tile_base + ty) < K_i) ? B_q[col * ldb + k_tile_base + ty] : 0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SUB_GEMM_TILE_DIM; ++k) {
            acc += (int32_t)A_tile_s8[ty][k] * (int32_t)B_tile_s8[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) C_q[row * ldc + col] = acc;
}

template<typename T>
__global__ void DequantAndAcc(
    const int32_t* C_q, T* C_fp_final,
    const T* scale_A_in, const T* offset_A_in,
    const T* base_scale_B, const T* base_offset_B,
    const int32_t* base_sum_B_q,
    const int group_idx, const int num_oc_groups,
    const int32_t* sum_A_q_in,
    const int M, const int N, const int K_i, const int ldc
) {
    // 使用共享内存缓存行和列的量化参数
    __shared__ float smem_scale_A[DEQUANT_TILE_DIM];
    __shared__ float smem_offset_A[DEQUANT_TILE_DIM];
    __shared__ int32_t smem_sum_A_q[DEQUANT_TILE_DIM];

    __shared__ float smem_scale_B[DEQUANT_TILE_DIM];
    __shared__ float smem_offset_B[DEQUANT_TILE_DIM];
    __shared__ int32_t smem_sum_B_q[DEQUANT_TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row_start = blockIdx.y * DEQUANT_TILE_DIM;
    const int block_col_start = blockIdx.x * DEQUANT_TILE_DIM;

    // 每个线程加载一个 A 的参数
    int m_load_idx = block_row_start + ty;
    if (tx == 0 && m_load_idx < M) {
        smem_scale_A[ty]   = (float)scale_A_in[m_load_idx];
        smem_offset_A[ty]  = (float)offset_A_in[m_load_idx];
        smem_sum_A_q[ty]   = sum_A_q_in[m_load_idx];
    }

    // 每个线程加载一个 B 的参数
    int n_load_idx = block_col_start + tx;
    if (ty == 0 && n_load_idx < N) {
        const size_t b_param_idx = n_load_idx * num_oc_groups + group_idx;
        smem_scale_B[tx]  = (float)base_scale_B[b_param_idx];
        smem_offset_B[tx] = (float)base_offset_B[b_param_idx];
        smem_sum_B_q[tx]  = base_sum_B_q[b_param_idx];
    }

    __syncthreads();

    // 计算每个线程负责的输出点
    const int m = block_row_start + ty;
    const int n = block_col_start + tx;

    if (m < M && n < N) {
        // 从共享内存中读取参数
        const float scale_A = smem_scale_A[ty];
        const float offset_A = smem_offset_A[ty];
        const float sum_A_q = (float)smem_sum_A_q[ty];

        const float scale_B = smem_scale_B[tx];
        const float offset_B = smem_offset_B[tx];
        const float sum_B_q = (float)smem_sum_B_q[tx];
        
        const float c_q_val = (float)C_q[m * ldc + n];

        float term1 = scale_A * (c_q_val * scale_B + sum_A_q * offset_B);
        float term2 = offset_A * (sum_B_q * scale_B + K_i * offset_B);
        float final_val = term1 + term2;

        #if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA)
            if constexpr (std::is_same_v<T, float>) {
                atomicAdd(&C_fp_final[m * ldc + n], final_val);
            } else if constexpr (std::is_same_v<T, half>) {
                // printf("[final_val]%.4f %.4f\n", __half2float(C_fp_final[m * ldc + n]), final_val);
                atomicAdd(&C_fp_final[m * ldc + n], __float2half(final_val));
                // printf("[C_fp_final]%.4f\n", __half2float(C_fp_final[m * ldc + n]));
            }
        #else
            C_fp_final[m * ldc + n] += final_val;
        #endif
    }
}

// 预计算权重 B 的修正项 (sum_B_q)
__global__ void Precompute_SumBq(
    const int8_t* B_q, int32_t* sum_B_q_out,
    const int num_groups, const int ic_per_group,
    const int oc, const int ic_p
) {
    // 每个线程处理一个 (output_channel, group) 的修正项
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)oc * num_groups; index += blockDim.x * gridDim.x) {
        
        const int i = index % num_groups; // group_idx
        const int n = index / num_groups; // output_channel
        
        const int k_start = i * ic_per_group;
        
        int32_t sum = 0;
        for (int k_offset = 0; k_offset < ic_per_group; ++k_offset) {
            // 访问 B 矩阵的布局是 [oc][ic_p]
            sum += (int32_t)B_q[n * ic_p + k_start + k_offset];
        }
        sum_B_q_out[index] = sum;
    }
}

template<typename T>
__global__ void BiasAndActivation(
    T* data, const T* bias,
    const float minV, const float maxV,
    const int M, const int N, const int ldc
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)M * N; index += blockDim.x * gridDim.x) {
        const int m = index / N;
        const int n = index % N;

        const size_t buffer_idx = m * ldc + n;
        float val = (float)data[buffer_idx];
        val += (float)bias[n];
        val = max(val, minV);
        val = min(val, maxV);
        data[buffer_idx] = (T)val;
    }
}

template<typename T>
__global__ void GEMM_FpAInt8B(
    const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int oc, const int oc_p,
    const int batch, const int quanC
) {
    __shared__ T A_tile[TILE_DIM][TILE_DIM]; // [batch, ic]
    __shared__ int8_t B_tile_s8[TILE_DIM][TILE_DIM]; // [ic, oc] kernel 本身是 B^T [oc, ic]
    __shared__ T B_tile_fp[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y; // batch
    const int block_col = blockIdx.x; // output channel

    // 每个线程负责计算输出Tile中的一个元素
    const int out_row = block_row * TILE_DIM + ty; // M
    const int out_col = block_col * TILE_DIM + tx; // N

    float acc = 0.0f;
    // 沿K维度（输入通道ic）分块循环
    const int num_k_tiles = UP_DIV(ic, TILE_DIM);

    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * TILE_DIM;

        // 合并访问加载 A_tile (input)，线程 (ty, tx) 加载 A_tile[ty][tx]
        int a_col_idx = k_tile_base + tx;
        A_tile[ty][tx] = (out_row < batch && a_col_idx < ic) ? input[out_row * ic_p + a_col_idx] : (T)0.0f;
        
        // 合并访问加载 B_tile (kernel)，线程 (ty, tx) 加载 B_tile[ty][tx]
        // kernel 布局为 [oc, ic]，需要 B(k,n)，即 kernel(n,k)
        int b_load_row = block_col * TILE_DIM + ty;
        int b_col_idx = k_tile_base + tx;
        B_tile_s8[ty][tx] = (b_load_row < oc && b_col_idx < ic) ? kernel[b_load_row * ic_p + b_col_idx] : 0;

        __syncthreads();

        // 反量化 + 转置
        const int K = ty;
        const int N = tx;

        const int global_k = k_tile_base + K;
        const int global_n = block_col * TILE_DIM + N;

        if (global_n < oc && global_k < ic) {
            const int group_idx = global_k / ic_per_group;
            const int quan_param_index = global_n * num_quan_groups_per_channel + group_idx;

            const float x_scale  = (float)scale[quan_param_index];
            const float x_offset = (float)offset[quan_param_index];
            
            // B_tile_s8(n,k), thread(n,k) -> (tx, ty)
            // So we need to read from B_tile_s8[n_dim_in_tile][k_dim_in_tile] -> B_tile_s8[tx][ty]
            const float b_quant = (float)B_tile_s8[N][K];

            // B_tile_fp[k][n] -> B_tile_fp[ty][tx]
            B_tile_fp[K][N] = (T)(b_quant * x_scale + x_offset);
        }
        
        __syncthreads();

        // 在 SMEM 中进行子矩阵乘法
        if (out_col < oc) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                acc += (float)A_tile[ty][k] * (float)B_tile_fp[k][tx];
            }
        }

        __syncthreads();
    }

    // 写回全局内存
    if (out_row < batch && out_col < oc) {
        acc += (float)bias[out_col];
        acc = max(acc, minV);
        acc = min(acc, maxV);
        output[out_row * oc_p + out_col] = (T)acc;
    }
}

template<typename T>
__global__ void GEMV_FpAInt8B(
    const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    __shared__ float partial_sums[GEMV_TILE]; // 每个线程块计算一个输出通道

    const int oz = blockIdx.x;
    const int b  = blockIdx.y;

    // if (oz >= oc || b >= batch) return;

    const int tid = threadIdx.x; // 块内线程ID
    
    // 准备量化参数
    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
    const int quan_param_index_base = oz * num_quan_groups_per_channel;

    // 每个线程计算自己的部分和
    float my_sum = 0.0f;
    for (int k = tid; k < ic; k += blockDim.x) {
        const float inp0 = input[b * ic_p + k];
        const int8_t ker0 = kernel[oz * ic_p + k];
        
        const int group_idx = k / ic_per_group;
        const int quan_param_index = quan_param_index_base + group_idx;
        const float x_scale  = scale[quan_param_index];
        const float x_offset = offset[quan_param_index];
        
        my_sum += inp0 * ((float)ker0 * x_scale + x_offset);
    }

    partial_sums[tid] = my_sum;
    __syncthreads();

    // 在 SMEM 中执行并行规约（蝶式交换）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    if (tid == 0) { // 写回全局内存
        float final_val = partial_sums[0] + (float)bias[oz];
        final_val = max(final_val, minV);
        final_val = min(final_val, maxV);
        output[b * oc_p + oz] = (T)final_val;
    }
}

template<typename T>
__global__ void GEMM_FpAInt4B(
    const T* input,
    const uint8_t* kernel, // kernel 是打包的 int4
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int oc, const int oc_p,
    const int batch, const int quanC
) {
    __shared__ T A_tile[TILE_DIM][TILE_DIM]; // [batch, ic]
    __shared__ uint8_t B_tile_u8[TILE_DIM][TILE_DIM / 2]; // [ic, oc] kernel 本身是 B^T [oc, ic]
    __shared__ T B_tile_fp[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y; // batch
    const int block_col = blockIdx.x; // output channel

    const int out_row = block_row * TILE_DIM + ty; // M
    const int out_col = block_col * TILE_DIM + tx; // N

    float acc = 0.0f;
    const int num_k_tiles = UP_DIV(ic, TILE_DIM);
    
    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * TILE_DIM;

        // 合并访问加载 A_tile (input)
        int a_col_idx = k_tile_base + tx;
        A_tile[ty][tx] = (out_row < batch && a_col_idx < ic) ? input[out_row * ic_p + a_col_idx] : (T)0.0f;
        
        // 合并访问加载 B_tile (kernel)，每个线程加载一个 byte，但 tile 宽度减半
        if (tx < TILE_DIM / 2) {
            int b_load_row = block_col * TILE_DIM + ty;
            // kernel 布局为 [oc, ic/2]，所以列索引要除以2
            int b_col_idx = (k_tile_base / 2) + tx;
            B_tile_u8[ty][tx] = (b_load_row < oc && (k_tile_base + tx*2) < ic) ? kernel[b_load_row * (ic_p / 2) + b_col_idx] : 0;
        }

        __syncthreads();

        // 反量化 + 转置
        const int K = ty;
        const int N = tx;

        const int global_k = k_tile_base + K;
        const int global_n = block_col * TILE_DIM + N;

        if (global_n < oc && global_k < ic) {
            const int group_idx = global_k / ic_per_group;
            const int quan_param_index = global_n * num_quan_groups_per_channel + group_idx;

            const float x_scale  = (float)scale[quan_param_index];
            const float x_offset = (float)offset[quan_param_index];
            
            // B_tile_u8(n,k) -> B_tile_u8[tx][ty/2] (转置后)
            const uint8_t b_packed = B_tile_u8[N][K / 2];
            // 根据 K 的奇偶性解包 int4
            const int8_t b_quant_s8 = (K % 2 == 0) ? ((b_packed >> 4) - 8) : ((b_packed & 0x0F) - 8);
            const float b_quant = (float)b_quant_s8;
            
            // B_tile_fp[k][n] -> B_tile_fp[ty][tx]
            B_tile_fp[K][N] = (T)(b_quant * x_scale + x_offset);
        }
        
        __syncthreads();

        // 在 SMEM 中进行子矩阵乘法
        if (out_col < oc) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                acc += (float)A_tile[ty][k] * (float)B_tile_fp[k][tx];
            }
        }

        __syncthreads();
    }

    // 写回全局内存
    if (out_row < batch && out_col < oc) {
        acc += (float)bias[out_col];
        acc = max(acc, minV);
        acc = min(acc, maxV);
        output[out_row * oc_p + out_col] = (T)acc;
    }
}

template<typename T>
__global__ void GEMV_FpAInt4B(
    const T* input,
    const uint8_t* kernel, // kernel 是打包的 int4
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    extern __shared__ uint8_t smem_buffer[];
    T* smem_input = reinterpret_cast<T*>(smem_buffer);
    float* partial_sums = reinterpret_cast<float*>(smem_buffer + ic_p * sizeof(T));
    
    const int oz = blockIdx.x; // 当前线程块负责计算的输出通道索引
    const int b  = blockIdx.y;

    const int tid = threadIdx.x;
    
    // if (oz >= oc || b >= batch) return;
    
    // 加载 input 到共享内存
    for (int i = tid; i < ic; i += blockDim.x) smem_input[i] = input[b * ic_p + i];
    for (int i = ic + tid; i < ic_p; i += blockDim.x) smem_input[i] = (T)0.0f;
    __syncthreads();

    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
    const int quan_param_index_base = oz * num_quan_groups_per_channel;

    float my_sum = 0.0f;
    for (int k = tid * 2; k < ic; k += blockDim.x * 2) {
        // 加载打包的权重并解包
        const uint8_t ker_packed = kernel[oz * (ic_p / 2) + k / 2];
        const int8_t ker0_s8 = (ker_packed >> 4) - 8;
        const int8_t ker1_s8 = (ker_packed & 0x0F) - 8;

        const int group_idx0 = k / ic_per_group;
        const int quan_param_index0 = quan_param_index_base + group_idx0;
        const float x_scale0  = scale[quan_param_index0];
        const float x_offset0 = offset[quan_param_index0];
        my_sum += (float)smem_input[k] * ((float)ker0_s8 * x_scale0 + x_offset0);
        my_sum += (float)smem_input[k + 1] * ((float)ker1_s8 * x_scale0 + x_offset0);
        
        // if (k + 1 < ic) {
        //     const float inp1 = (float)smem_input[k + 1];
        //     const int group_idx1 = (k + 1) / ic_per_group;

        //     if (group_idx0 == group_idx1) {
        //         my_sum += inp1 * ((float)ker1_s8 * x_scale0 + x_offset0);
        //     } else {
        //         const int quan_param_index1 = quan_param_index_base + group_idx1;
        //         const float x_scale1  = (float)scale[quan_param_index1];
        //         const float x_offset1 = (float)offset[quan_param_index1];
        //         my_sum += inp1 * ((float)ker1_s8 * x_scale1 + x_offset1);
        //     }
        // }
    }

    partial_sums[tid] = my_sum;
    __syncthreads();

    // 并行规约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float final_val = partial_sums[0] + (float)bias[oz];
        final_val = max(final_val, minV);
        final_val = min(final_val, maxV);
        output[b * oc_p + oz] = (T)final_val;
    }
}


// =====================================================================
// P1-V5: 128-thread GEMV — all 128 threads cooperate on 1 OC
// - Grid: (oc, batch), Block: 128
// =====================================================================
template<typename T>
__global__ void GEMV_FpAInt4B_V5(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,  // packed int4 weights [oc, ic_p/2]
    const T* __restrict__ scale, const T* __restrict__ offset, const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    const int oz = blockIdx.x;  // 1 OC per block
    const int b = blockIdx.y;
    if (oz >= oc || b >= batch) return;

    const int tid = threadIdx.x;  // 0..127
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_qg = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_qg > 0) ? (ic / num_qg) : ic;
    const int qparam_base = oz * num_qg;

    const int row_bytes = ic_p / 2;
    const uint8_t* w_row = kernel + oz * row_bytes;
    const T* in_ptr = input + b * ic_p;

    float acc = 0.0f;

    // 128 threads × 8 elements = 1024 per iteration
    for (int k = tid * 8; k < ic; k += 1024) {
        uint32_t w32 = __ldg(reinterpret_cast<const uint32_t*>(w_row + k / 2));

        const int gidx = k / ic_per_group;
        const float s = (float)__ldg(&scale[qparam_base + gidx]);
        const float o = (float)__ldg(&offset[qparam_base + gidx]);
        const float adj_off = o - 8.0f * s;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int kk = k + j * 2;
            if (kk >= ic) break;
            const uint8_t packed = (w32 >> (j * 8)) & 0xFF;
            const float w0 = (float)(packed >> 4) * s + adj_off;
            const float w1 = (float)(packed & 0x0F) * s + adj_off;
            acc += (float)in_ptr[kk] * w0;
            if (kk + 1 < ic) acc += (float)in_ptr[kk + 1] * w1;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }

    // Cross-warp reduction via shared memory (4 warps → 1 result)
    __shared__ float smem[4];
    if (lane_id == 0) smem[warp_id] = acc;
    __syncthreads();

    if (tid == 0) {
        float result = smem[0] + smem[1] + smem[2] + smem[3];
        result += (float)bias[oz];
        result = fmaxf(result, minV);
        result = fminf(result, maxV);
        output[b * oc_p + oz] = (T)result;
    }
}

// =====================================================================
// V9: Multi-OC GEMV — each block processes OC_PER_BLK output channels
// 128 threads cooperatively reduce each OC in sequence (fewer blocks, better latency hiding)
// Grid: (ceil(oc/OC_PER_BLK), batch), Block: 128
// =====================================================================
template<typename T, int OC_PER_BLK>
__global__ void GEMV_FpAInt4B_V9(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,  // packed int4 weights [oc, ic_p/2]
    const T* __restrict__ scale, const T* __restrict__ offset, const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    const int oc_base = blockIdx.x * OC_PER_BLK;
    const int b = blockIdx.y;
    if (oc_base >= oc || b >= batch) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_qg = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_qg > 0) ? (ic / num_qg) : ic;

    const int row_bytes = ic_p / 2;
    const T* in_ptr = input + b * ic_p;

    // blockDim.x threads, (blockDim.x/32) warps
    const int nwarps = blockDim.x / 32;
    __shared__ float smem[8];  // up to 8 warps (256 threads)

    #pragma unroll
    for (int oc_off = 0; oc_off < OC_PER_BLK; oc_off++) {
        const int oz = oc_base + oc_off;
        if (oz >= oc) break;

        const int qparam_base = oz * num_qg;
        const uint8_t* w_row = kernel + oz * row_bytes;

        float acc = 0.0f;

        // Each thread handles 8 IC elements per iteration, stride = nthreads * 8
        const int stride = blockDim.x * 8;
        for (int k = tid * 8; k < ic; k += stride) {
            uint32_t w32 = __ldg(reinterpret_cast<const uint32_t*>(w_row + k / 2));

            const int gidx = k / ic_per_group;
            const float s = (float)__ldg(&scale[qparam_base + gidx]);
            const float o = (float)__ldg(&offset[qparam_base + gidx]);
            const float adj_off = o - 8.0f * s;

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int kk = k + j * 2;
                if (kk >= ic) break;
                const uint8_t packed = (w32 >> (j * 8)) & 0xFF;
                const float w0 = (float)(packed >> 4) * s + adj_off;
                const float w1 = (float)(packed & 0x0F) * s + adj_off;
                acc += (float)in_ptr[kk] * w0;
                if (kk + 1 < ic) acc += (float)in_ptr[kk + 1] * w1;
            }
        }

        // Warp-level reduction
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);

        if (lane_id == 0) smem[warp_id] = acc;
        __syncthreads();

        if (tid == 0) {
            float result = 0.0f;
            for (int w = 0; w < nwarps; w++) result += smem[w];
            result += (float)bias[oz];
            result = fmaxf(result, minV);
            result = fminf(result, maxV);
            output[b * oc_p + oz] = (T)result;
        }
        __syncthreads();
    }
}




// =====================================================================
// V14: Factored-dequant multi-OC GEMV — reduces ALU per weight byte.
// Key insight: acc += in * (s * nibble + adj) = s * sum(in*nibble) + adj * sum(in)
// By factoring out scale/adj, we save 6 FMA per OC per 8-IC iteration.
// Also: explicit load-compute separation for better instruction scheduling.
// Grid: (ceil(oc/OC_PER_BLK), batch), Block: 128
// =====================================================================
template<typename T, int OC_PER_BLK>
__global__ void __launch_bounds__(128, 12) GEMV_FpAInt4B_V14(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,  // packed int4 weights [oc, ic_p/2]
    const float2* __restrict__ gemv_params,  // [oc * num_qg], .x=scale, .y=adj_offset
    const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int num_qg
) {
    // Grid layout: (oc/OC_PER_BLK, batch)
    const int oc_base = blockIdx.x * OC_PER_BLK;
    const int b = blockIdx.y;
    if (oc_base >= oc || b >= batch) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int nwarps = blockDim.x / 32;

    const int ic_per_group = ic / num_qg;
    const int row_bytes = ic_p / 2;
    const T* in_ptr = input + b * ic_p;

    // Compute per-OC base pointers (on-the-fly, no pointer arrays to save registers)
    float accs[OC_PER_BLK];
    #pragma unroll
    for (int i = 0; i < OC_PER_BLK; i++)
        accs[i] = 0.0f;

    // Main IC loop — all OCs processed simultaneously
    const int stride = blockDim.x * 8;
    for (int k = tid * 8; k < ic; k += stride) {
        // Phase 1: Load ALL input values
        float in_vals[8];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            in_vals[j] = (float)in_ptr[k + j];

        // Pre-compute input sum for factored adj_offset term (shared across all OCs)
        float input_sum = in_vals[0] + in_vals[1] + in_vals[2] + in_vals[3]
                        + in_vals[4] + in_vals[5] + in_vals[6] + in_vals[7];

        const int gidx = k / ic_per_group;

        // Phase 2: Load ALL weights and params before any compute
        uint32_t w_data[OC_PER_BLK];
        float s_data[OC_PER_BLK], adj_data[OC_PER_BLK];
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++) {
            const int oc_idx = oc_base + oi;
            if (oc_idx < oc) {
                w_data[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    kernel + oc_idx * row_bytes + k / 2));
                const float2 p = __ldg(gemv_params + oc_idx * num_qg + gidx);
                s_data[oi] = p.x;
                adj_data[oi] = p.y;
            }
        }

        // Phase 3: Compute raw dot products (no scale/adj in the hot loop)
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++) {
            if (oc_base + oi >= oc) break;
            float raw_dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t packed = (w_data[oi] >> (j * 8)) & 0xFF;
                raw_dot += in_vals[j * 2] * (float)(packed >> 4);
                raw_dot += in_vals[j * 2 + 1] * (float)(packed & 0x0F);
            }
            // Apply scale and adj_offset: acc += s * raw_dot + adj * input_sum
            accs[oi] = fmaf(adj_data[oi], input_sum, fmaf(s_data[oi], raw_dot, accs[oi]));
        }
    }

    // Warp-level reduction for all accumulators
    #pragma unroll
    for (int oi = 0; oi < OC_PER_BLK; oi++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accs[oi] += __shfl_xor_sync(0xffffffff, accs[oi], off);
    }

    // Single-sync parallel cross-warp reduction for ALL OCs at once
    __shared__ float smem[OC_PER_BLK][8];  // [oc_idx][warp_idx]

    if (lane_id == 0) {
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++)
            smem[oi][warp_id] = accs[oi];
    }
    __syncthreads();

    // First OC_PER_BLK threads each reduce one OC
    if (tid < OC_PER_BLK && oc_base + tid < oc) {
        float result = 0.0f;
        for (int w = 0; w < nwarps; w++) result += smem[tid][w];
        result += (float)__ldg(&bias[oc_base + tid]);
        result = fmaxf(result, minV);
        result = fminf(result, maxV);
        output[b * oc_p + oc_base + tid] = (T)result;
    }
}

// V14_MB: Multi-batch variant — one block processes ALL batch elements for the same OC group.
// Weights are loaded ONCE and reused across batch elements, reducing memory bandwidth by ~batch×.
template<typename T, int OC_PER_BLK, int MAX_BATCH>
__global__ void __launch_bounds__(128, 8) GEMV_FpAInt4B_V14_MB(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,
    const float2* __restrict__ gemv_params,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int num_qg
) {
    const int oc_base = blockIdx.x * OC_PER_BLK;
    if (oc_base >= oc) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int nwarps = blockDim.x / 32;

    const int ic_per_group = ic / num_qg;
    const int row_bytes = ic_p / 2;

    float accs[MAX_BATCH * OC_PER_BLK];
    #pragma unroll
    for (int i = 0; i < MAX_BATCH * OC_PER_BLK; i++)
        accs[i] = 0.0f;

    const int stride = blockDim.x * 8;
    for (int k = tid * 8; k < ic; k += stride) {
        const int gidx = k / ic_per_group;

        // Load weights ONCE for all batch elements
        uint32_t w_data[OC_PER_BLK];
        float s_data[OC_PER_BLK], adj_data[OC_PER_BLK];
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++) {
            const int oc_idx = oc_base + oi;
            if (oc_idx < oc) {
                w_data[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    kernel + oc_idx * row_bytes + k / 2));
                const float2 p = __ldg(gemv_params + oc_idx * num_qg + gidx);
                s_data[oi] = p.x;
                adj_data[oi] = p.y;
            }
        }

        // Process each batch element with the cached weights
        #pragma unroll
        for (int b = 0; b < MAX_BATCH; b++) {
            if (b >= batch) break;
            const T* in_ptr = input + b * ic_p;
            float in_vals[8];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                in_vals[j] = (float)in_ptr[k + j];

            float input_sum = in_vals[0] + in_vals[1] + in_vals[2] + in_vals[3]
                            + in_vals[4] + in_vals[5] + in_vals[6] + in_vals[7];

            #pragma unroll
            for (int oi = 0; oi < OC_PER_BLK; oi++) {
                if (oc_base + oi >= oc) break;
                float raw_dot = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t packed = (w_data[oi] >> (j * 8)) & 0xFF;
                    raw_dot += in_vals[j * 2] * (float)(packed >> 4);
                    raw_dot += in_vals[j * 2 + 1] * (float)(packed & 0x0F);
                }
                accs[b * OC_PER_BLK + oi] = fmaf(adj_data[oi], input_sum, fmaf(s_data[oi], raw_dot, accs[b * OC_PER_BLK + oi]));
            }
        }
    }

    // Warp-level reduction for all accumulators
    #pragma unroll
    for (int i = 0; i < MAX_BATCH * OC_PER_BLK; i++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accs[i] += __shfl_xor_sync(0xffffffff, accs[i], off);
    }

    // Cross-warp reduction using shared memory
    __shared__ float smem[MAX_BATCH * OC_PER_BLK][4]; // [batch*oc][warp_id], nwarps<=4

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < MAX_BATCH * OC_PER_BLK; i++)
            smem[i][warp_id] = accs[i];
    }
    __syncthreads();

    // First batch*OC_PER_BLK threads each reduce one accumulator
    const int reduce_count = batch * OC_PER_BLK;
    if (tid < reduce_count) {
        const int b = tid / OC_PER_BLK;
        const int oi = tid % OC_PER_BLK;
        if (oc_base + oi < oc) {
            float result = 0.0f;
            for (int w = 0; w < nwarps; w++) result += smem[tid][w];
            result += (float)__ldg(&bias[oc_base + oi]);
            result = fmaxf(result, minV);
            result = fminf(result, maxV);
            output[b * oc_p + oc_base + oi] = (T)result;
        }
    }
}







// P1: Optimized INT8 GEMV with warp shuffle reduction
template<typename T>
__global__ void GEMV_FpAInt8B_V2(
    const T* __restrict__ input,
    const int8_t* __restrict__ kernel,
    const T* __restrict__ scale, const T* __restrict__ offset, const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    const int b = blockIdx.y;
    const int oz_base = blockIdx.x * GEMV_OC_PER_BLOCK;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int oz = oz_base + warp_id;

    if (oz >= oc || b >= batch) return;

    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
    const int quan_param_index_base = oz * num_quan_groups_per_channel;

    float my_sum = 0.0f;
    for (int k = lane_id; k < ic; k += WARP_SIZE) {
        const float inp0 = (float)input[b * ic_p + k];
        const int8_t ker0 = __ldg(&kernel[oz * ic_p + k]);

        const int group_idx = k / ic_per_group;
        const int quan_param_index = quan_param_index_base + group_idx;
        const float x_scale = (float)__ldg(&scale[quan_param_index]);
        const float x_offset = (float)__ldg(&offset[quan_param_index]);

        my_sum += inp0 * ((float)ker0 * x_scale + x_offset);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset_val = WARP_SIZE / 2; offset_val > 0; offset_val >>= 1) {
        my_sum += __shfl_xor_sync(0xffffffff, my_sum, offset_val);
    }

    if (lane_id == 0) {
        float final_val = my_sum + (float)bias[oz];
        final_val = max(final_val, minV);
        final_val = min(final_val, maxV);
        output[b * oc_p + oz] = (T)final_val;
    }
}

template<typename T>
__global__ void CONV_FpAInt8B(const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T *output,
    const float maxV, const float minV,
    const int ic,  const int ic_p, const int iw, const int ih,
    const int c, const int c_p, const int ow, const int oh,
    const int kw, const int kh, const int dw, const int dh,
    const int sw, const int sh, const int pw, const int ph,
    const int total, const int quanC,
    DivModFast d_oc, DivModFast d_ow, DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        
        const int num_quan_groups_per_channel = (c > 0 && quanC > 0) ? (quanC / c) : 1;
        const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
        const int quan_param_index_base = oz * num_quan_groups_per_channel;

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                for (int group_idx = 0; group_idx < num_quan_groups_per_channel; ++group_idx) {
                    const int quan_param_index = quan_param_index_base + group_idx;
                    const float x_scale = scale[quan_param_index];
                    const float x_offset = offset[quan_param_index];

                    const int sz_start = group_idx * ic_per_group;
                    const int sz_end = sz_start + ic_per_group;

                    for (int sz = sz_start; sz < sz_end && sz < ic_p; ++ sz) {
                        int src_offset = ((ob * ih + sy) * iw + sx) * ic_p + sz;
                        float inp0 = input[src_offset];
                        //[Cop, KhKw, Cip]
                        int8_t ker0 = kernel[((oz * kh + fy) * kw + fx) * ic_p + sz];

                        color0 = color0 + inp0 * ((float)ker0 * x_scale + x_offset);
                    }
                }
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
    }
}

template<typename T>
__global__ void CONV_FpAInt4B(const T* input,
    const uint8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T *output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int iw, const int ih,
    const int c, const int c_p, const int ow, const int oh,
    const int kw, const int kh, const int dw, const int dh,
    const int sw, const int sh, const int pw, const int ph,
    const int total, const int quanC,
    DivModFast d_oc, DivModFast d_ow, DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];

        const int num_quan_groups_per_channel = (c > 0 && quanC > 0) ? (quanC / c) : 1;
        const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
        const int quan_param_index_base = oz * num_quan_groups_per_channel;

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                for (int group_idx = 0; group_idx < num_quan_groups_per_channel; ++group_idx) {
                    const int quan_param_index = quan_param_index_base + group_idx;
                    const float x_scale = scale[quan_param_index];
                    const float x_offset = offset[quan_param_index];

                    const int sz_start = group_idx * ic_per_group / 2;
                    const int sz_end = sz_start + ic_per_group / 2;

                    for (int sz = sz_start; sz < sz_end && sz * 2 < ic_p; ++ sz) {
                        int src_offset = ((ob * ih + sy) * iw + sx) * ic_p + 2 * sz;
                        float inp0 = input[src_offset];
                        float inp1 = input[src_offset+1];

                        //[Cop, KhKw, Cip]
                        uint8_t ker = kernel[((oz * kh + fy) * kw + fx) * ic_p / 2 + sz];
                        int8_t ker0 = (ker >> 4) - 8;
                        int8_t ker1 = (ker & 15) - 8;
                        color0 = color0 + inp0 * ((float)ker0 * x_scale + x_offset);
                        color0 = color0 + inp1 * ((float)ker1 * x_scale + x_offset);
                    }
                }
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
    }
}

__global__ void Rearrange_Packed_Weight_Int4(const uint8_t* param,
    uint8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp2
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icp2Index, temp, ocpIndex, khwIndex;
        d_icp2.divmod(index, temp, icp2Index);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(ocpIndex >= oc || 2 * icp2Index >= ic) {
            output[index] = 0;
            continue;
        }

        int src_index = (ocpIndex * UP_DIV(ic, 2) + icp2Index) * khw + khwIndex;
        output[index] = param[src_index];
    }
}

__global__ void Rearrange_Weight_Int4(const int8_t* param,
    uint8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp2
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icp2Index, temp, ocpIndex, khwIndex;
        d_icp2.divmod(index, temp, icp2Index);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(2*icp2Index >= ic || ocpIndex >= oc) {
            output[index] = 0;
            continue;
        }
        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip/2], Ci available for vectorize
        output[index] = ((uint8_t)(param[(ocpIndex * ic + 2*icp2Index+0) * khw + khwIndex] + 8 )) * 16;
        if(2*icp2Index+1 < ic) {
            output[index] += ((uint8_t)(param[(ocpIndex * ic + 2*icp2Index+1) * khw + khwIndex] + 8 ));
        }
    }
}

__global__ void Rearrange_Weight_Int8(const int8_t* param,
    int8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icpIndex, temp, ocpIndex, khwIndex;
        d_icp.divmod(index, temp, icpIndex);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(icpIndex >= ic || ocpIndex >= oc) {
            output[index] = 0;
            continue;
        }
        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip], Ci available for vectorize
        output[index] = param[(ocpIndex * ic + icpIndex) * khw + khwIndex];
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- PrecomputeGemvParams (fp32) ----
void mnn_corpus_precomputegemvparams_fp32(const float* scale, const float* offset, float2* params,
                                          int total, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PrecomputeGemvParams<float><<<grid, block, 0, stream>>>(
        scale, offset, params, total);
}
// ---- PrecomputeGemvParams (fp16) ----
void mnn_corpus_precomputegemvparams_fp16(const void* scale, const void* offset, float2* params,
                                          int total, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PrecomputeGemvParams<__half><<<grid, block, 0, stream>>>(
        (const __half*)scale, (const __half*)offset, params, total);
}

// ---- QuantA (fp32) ----
void mnn_corpus_quanta_fp32(const float* A_sub_fp, int8_t* A_sub_q,
                             float* scale_A_out, float* offset_A_out, int32_t* sum_A_q_out,
                             int M, int K_i, int lda, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::QuantA<float><<<grid, block, 0, stream>>>(
        A_sub_fp, A_sub_q, scale_A_out, offset_A_out, sum_A_q_out, M, K_i, lda);
}
// ---- QuantA (fp16) ----
void mnn_corpus_quanta_fp16(const void* A_sub_fp, int8_t* A_sub_q,
                             void* scale_A_out, void* offset_A_out, int32_t* sum_A_q_out,
                             int M, int K_i, int lda, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::QuantA<__half><<<grid, block, 0, stream>>>(
        (const __half*)A_sub_fp, A_sub_q, (__half*)scale_A_out, (__half*)offset_A_out,
        sum_A_q_out, M, K_i, lda);
}

// ---- GEMM_Int8 (non-template, single fp32 shim) ----
void mnn_corpus_gemm_int8_fp32(const int8_t* A_q, const int8_t* B_q, int32_t* C_q,
                                int M, int N, int K_i, int lda_q, int ldb, int ldc,
                                int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMM_Int8<<<grid, block, 0, stream>>>(
        A_q, B_q, C_q, M, N, K_i, lda_q, ldb, ldc);
}

// ---- DequantAndAcc (fp32) ----
void mnn_corpus_dequantandacc_fp32(const int32_t* C_q, float* C_fp_final,
                                    const float* scale_A_in, const float* offset_A_in,
                                    const float* base_scale_B, const float* base_offset_B,
                                    const int32_t* base_sum_B_q,
                                    int group_idx, int num_oc_groups,
                                    const int32_t* sum_A_q_in,
                                    int M, int N, int K_i, int ldc,
                                    int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::DequantAndAcc<float><<<grid, block, 0, stream>>>(
        C_q, C_fp_final, scale_A_in, offset_A_in, base_scale_B, base_offset_B,
        base_sum_B_q, group_idx, num_oc_groups, sum_A_q_in, M, N, K_i, ldc);
}
// ---- DequantAndAcc (fp16) ----
void mnn_corpus_dequantandacc_fp16(const int32_t* C_q, void* C_fp_final,
                                    const void* scale_A_in, const void* offset_A_in,
                                    const void* base_scale_B, const void* base_offset_B,
                                    const int32_t* base_sum_B_q,
                                    int group_idx, int num_oc_groups,
                                    const int32_t* sum_A_q_in,
                                    int M, int N, int K_i, int ldc,
                                    int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::DequantAndAcc<__half><<<grid, block, 0, stream>>>(
        C_q, (__half*)C_fp_final, (const __half*)scale_A_in, (const __half*)offset_A_in,
        (const __half*)base_scale_B, (const __half*)base_offset_B, base_sum_B_q,
        group_idx, num_oc_groups, sum_A_q_in, M, N, K_i, ldc);
}

// ---- Precompute_SumBq (non-template, single fp32 shim) ----
void mnn_corpus_precompute_sumbq_fp32(const int8_t* B_q, int32_t* sum_B_q_out,
                                       int num_groups, int ic_per_group, int oc, int ic_p,
                                       int grid, int block, cudaStream_t stream) {
    MNN::Corpus::Precompute_SumBq<<<grid, block, 0, stream>>>(
        B_q, sum_B_q_out, num_groups, ic_per_group, oc, ic_p);
}

// ---- BiasAndActivation (fp32) ----
void mnn_corpus_biasandactivation_fp32(float* data, const float* bias,
                                        float minV, float maxV, int M, int N, int ldc,
                                        int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BiasAndActivation<float><<<grid, block, 0, stream>>>(
        data, bias, minV, maxV, M, N, ldc);
}
// ---- BiasAndActivation (fp16) ----
void mnn_corpus_biasandactivation_fp16(void* data, const void* bias,
                                        float minV, float maxV, int M, int N, int ldc,
                                        int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BiasAndActivation<__half><<<grid, block, 0, stream>>>(
        (__half*)data, (const __half*)bias, minV, maxV, M, N, ldc);
}

// ---- GEMM_FpAInt8B (fp32) ----
void mnn_corpus_gemm_fpaint8b_fp32(const float* input, const int8_t* kernel,
                                    const float* scale, const float* offset, const float* bias,
                                    float* output, float maxV, float minV,
                                    int ic, int ic_p, int oc, int oc_p, int batch, int quanC,
                                    int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMM_FpAInt8B<float><<<grid, block, 0, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        ic, ic_p, oc, oc_p, batch, quanC);
}
// ---- GEMM_FpAInt8B (fp16) ----
void mnn_corpus_gemm_fpaint8b_fp16(const void* input, const int8_t* kernel,
                                    const void* scale, const void* offset, const void* bias,
                                    void* output, float maxV, float minV,
                                    int ic, int ic_p, int oc, int oc_p, int batch, int quanC,
                                    int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMM_FpAInt8B<__half><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        ic, ic_p, oc, oc_p, batch, quanC);
}

// ---- GEMV_FpAInt8B (fp32) ----
void mnn_corpus_gemv_fpaint8b_fp32(const float* input, const int8_t* kernel,
                                    const float* scale, const float* offset, const float* bias,
                                    float* output, float maxV, float minV,
                                    int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                    int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMV_FpAInt8B<float><<<grid, block, 0, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}
// ---- GEMV_FpAInt8B (fp16) ----
void mnn_corpus_gemv_fpaint8b_fp16(const void* input, const int8_t* kernel,
                                    const void* scale, const void* offset, const void* bias,
                                    void* output, float maxV, float minV,
                                    int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                    int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMV_FpAInt8B<__half><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}

// ---- GEMM_FpAInt4B (fp32) ----
void mnn_corpus_gemm_fpaint4b_fp32(const float* input, const uint8_t* kernel,
                                    const float* scale, const float* offset, const float* bias,
                                    float* output, float maxV, float minV,
                                    int ic, int ic_p, int oc, int oc_p, int batch, int quanC,
                                    int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMM_FpAInt4B<float><<<grid, block, 0, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        ic, ic_p, oc, oc_p, batch, quanC);
}
// ---- GEMM_FpAInt4B (fp16) ----
void mnn_corpus_gemm_fpaint4b_fp16(const void* input, const uint8_t* kernel,
                                    const void* scale, const void* offset, const void* bias,
                                    void* output, float maxV, float minV,
                                    int ic, int ic_p, int oc, int oc_p, int batch, int quanC,
                                    int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMM_FpAInt4B<__half><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        ic, ic_p, oc, oc_p, batch, quanC);
}

// ---- GEMV_FpAInt4B (fp32) ----
// Uses extern __shared__ memory: caller passes sharedMem bytes (ic_p * sizeof(T) + GEMV_TILE * sizeof(float)).
void mnn_corpus_gemv_fpaint4b_fp32(const float* input, const uint8_t* kernel,
                                    const float* scale, const float* offset, const float* bias,
                                    float* output, float maxV, float minV,
                                    int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                    int sharedMem, int gridX, int gridY, int blockX,
                                    cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B<float><<<grid, block, sharedMem, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}
// ---- GEMV_FpAInt4B (fp16) ----
void mnn_corpus_gemv_fpaint4b_fp16(const void* input, const uint8_t* kernel,
                                    const void* scale, const void* offset, const void* bias,
                                    void* output, float maxV, float minV,
                                    int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                    int sharedMem, int gridX, int gridY, int blockX,
                                    cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B<__half><<<grid, block, sharedMem, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}

// ---- GEMV_FpAInt4B_V5 (fp32) ----
void mnn_corpus_gemv_fpaint4b_v5_fp32(const float* input, const uint8_t* kernel,
                                       const float* scale, const float* offset, const float* bias,
                                       float* output, float maxV, float minV,
                                       int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                       int gridX, int gridY, int blockX, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B_V5<float><<<grid, block, 0, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}
// ---- GEMV_FpAInt4B_V5 (fp16) ----
void mnn_corpus_gemv_fpaint4b_v5_fp16(const void* input, const uint8_t* kernel,
                                       const void* scale, const void* offset, const void* bias,
                                       void* output, float maxV, float minV,
                                       int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                       int gridX, int gridY, int blockX, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B_V5<__half><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}

// ---- GEMV_FpAInt4B_V9 (fp32, OC_PER_BLK=2 — matches MNN HEAD) ----
void mnn_corpus_gemv_fpaint4b_v9_fp32(const float* input, const uint8_t* kernel,
                                       const float* scale, const float* offset, const float* bias,
                                       float* output, float maxV, float minV,
                                       int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                       int gridX, int gridY, int blockX, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B_V9<float, 2><<<grid, block, 0, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}
// ---- GEMV_FpAInt4B_V9 (fp16, OC_PER_BLK=2 — matches MNN HEAD) ----
void mnn_corpus_gemv_fpaint4b_v9_fp16(const void* input, const uint8_t* kernel,
                                       const void* scale, const void* offset, const void* bias,
                                       void* output, float maxV, float minV,
                                       int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                       int gridX, int gridY, int blockX, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B_V9<__half, 2><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}

// ---- GEMV_FpAInt4B_V14 (fp32, OC_PER_BLK=4) ----
void mnn_corpus_gemv_fpaint4b_v14_fp32(const float* input, const uint8_t* kernel,
                                        const float2* gemv_params, const float* bias,
                                        float* output, float maxV, float minV,
                                        int batch, int ic, int ic_p, int oc, int oc_p, int num_qg,
                                        int gridX, int gridY, int blockX, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B_V14<float, 4><<<grid, block, 0, stream>>>(
        input, kernel, gemv_params, bias, output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, num_qg);
}
// ---- GEMV_FpAInt4B_V14 (fp16, OC_PER_BLK=4) ----
void mnn_corpus_gemv_fpaint4b_v14_fp16(const void* input, const uint8_t* kernel,
                                        const float2* gemv_params, const void* bias,
                                        void* output, float maxV, float minV,
                                        int batch, int ic, int ic_p, int oc, int oc_p, int num_qg,
                                        int gridX, int gridY, int blockX, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B_V14<__half, 4><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, gemv_params, (const __half*)bias, (__half*)output,
        maxV, minV, batch, ic, ic_p, oc, oc_p, num_qg);
}

// ---- GEMV_FpAInt4B_V14_MB (fp32, OC_PER_BLK=4, MAX_BATCH=1) ----
void mnn_corpus_gemv_fpaint4b_v14_mb_fp32(const float* input, const uint8_t* kernel,
                                           const float2* gemv_params, const float* bias,
                                           float* output, float maxV, float minV,
                                           int batch, int ic, int ic_p, int oc, int oc_p, int num_qg,
                                           int gridX, int blockX, cudaStream_t stream) {
    dim3 grid(gridX);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B_V14_MB<float, 4, 1><<<grid, block, 0, stream>>>(
        input, kernel, gemv_params, bias, output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, num_qg);
}
// ---- GEMV_FpAInt4B_V14_MB (fp16, OC_PER_BLK=4, MAX_BATCH=1) ----
void mnn_corpus_gemv_fpaint4b_v14_mb_fp16(const void* input, const uint8_t* kernel,
                                           const float2* gemv_params, const void* bias,
                                           void* output, float maxV, float minV,
                                           int batch, int ic, int ic_p, int oc, int oc_p, int num_qg,
                                           int gridX, int blockX, cudaStream_t stream) {
    dim3 grid(gridX);
    dim3 block(blockX);
    MNN::Corpus::GEMV_FpAInt4B_V14_MB<__half, 4, 1><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, gemv_params, (const __half*)bias, (__half*)output,
        maxV, minV, batch, ic, ic_p, oc, oc_p, num_qg);
}

// ---- GEMV_FpAInt8B_V2 (fp32) ----
void mnn_corpus_gemv_fpaint8b_v2_fp32(const float* input, const int8_t* kernel,
                                       const float* scale, const float* offset, const float* bias,
                                       float* output, float maxV, float minV,
                                       int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                       int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMV_FpAInt8B_V2<float><<<grid, block, 0, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}
// ---- GEMV_FpAInt8B_V2 (fp16) ----
void mnn_corpus_gemv_fpaint8b_v2_fp16(const void* input, const int8_t* kernel,
                                       const void* scale, const void* offset, const void* bias,
                                       void* output, float maxV, float minV,
                                       int batch, int ic, int ic_p, int oc, int oc_p, int quanC,
                                       int gridX, int gridY, int blockX, int blockY, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::GEMV_FpAInt8B_V2<__half><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        batch, ic, ic_p, oc, oc_p, quanC);
}

// ---- CONV_FpAInt8B (fp32) ----
void mnn_corpus_conv_fpaint8b_fp32(const float* input, const int8_t* kernel,
                                    const float* scale, const float* offset, const float* bias,
                                    float* output, float maxV, float minV,
                                    int ic, int ic_p, int iw, int ih, int c, int c_p, int ow, int oh,
                                    int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                                    int total, int quanC, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(c_p), d_ow(ow), d_oh(oh);
    MNN::Corpus::CONV_FpAInt8B<float><<<grid, block, 0, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        ic, ic_p, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph,
        total, quanC, d_oc, d_ow, d_oh);
}
// ---- CONV_FpAInt8B (fp16) ----
void mnn_corpus_conv_fpaint8b_fp16(const void* input, const int8_t* kernel,
                                    const void* scale, const void* offset, const void* bias,
                                    void* output, float maxV, float minV,
                                    int ic, int ic_p, int iw, int ih, int c, int c_p, int ow, int oh,
                                    int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                                    int total, int quanC, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(c_p), d_ow(ow), d_oh(oh);
    MNN::Corpus::CONV_FpAInt8B<__half><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        ic, ic_p, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph,
        total, quanC, d_oc, d_ow, d_oh);
}

// ---- CONV_FpAInt4B (fp32) ----
void mnn_corpus_conv_fpaint4b_fp32(const float* input, const uint8_t* kernel,
                                    const float* scale, const float* offset, const float* bias,
                                    float* output, float maxV, float minV,
                                    int ic, int ic_p, int iw, int ih, int c, int c_p, int ow, int oh,
                                    int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                                    int total, int quanC, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(c_p), d_ow(ow), d_oh(oh);
    MNN::Corpus::CONV_FpAInt4B<float><<<grid, block, 0, stream>>>(
        input, kernel, scale, offset, bias, output, maxV, minV,
        ic, ic_p, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph,
        total, quanC, d_oc, d_ow, d_oh);
}
// ---- CONV_FpAInt4B (fp16) ----
void mnn_corpus_conv_fpaint4b_fp16(const void* input, const uint8_t* kernel,
                                    const void* scale, const void* offset, const void* bias,
                                    void* output, float maxV, float minV,
                                    int ic, int ic_p, int iw, int ih, int c, int c_p, int ow, int oh,
                                    int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                                    int total, int quanC, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(c_p), d_ow(ow), d_oh(oh);
    MNN::Corpus::CONV_FpAInt4B<__half><<<grid, block, 0, stream>>>(
        (const __half*)input, kernel, (const __half*)scale, (const __half*)offset,
        (const __half*)bias, (__half*)output, maxV, minV,
        ic, ic_p, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph,
        total, quanC, d_oc, d_ow, d_oh);
}

// ---- Rearrange_Packed_Weight_Int4 (non-template, single fp32 shim) ----
void mnn_corpus_rearrange_packed_weight_int4_fp32(const uint8_t* param, uint8_t* output,
                                                   int khw, size_t maxCount, int oc, int ic,
                                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_khw(khw), d_icp2((ic + 1) / 2);
    MNN::Corpus::Rearrange_Packed_Weight_Int4<<<grid, block, 0, stream>>>(
        param, output, khw, maxCount, oc, ic, d_khw, d_icp2);
}

// ---- Rearrange_Weight_Int4 (non-template, single fp32 shim) ----
void mnn_corpus_rearrange_weight_int4_fp32(const int8_t* param, uint8_t* output,
                                            int khw, size_t maxCount, int oc, int ic,
                                            int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_khw(khw), d_icp2((ic + 1) / 2);
    MNN::Corpus::Rearrange_Weight_Int4<<<grid, block, 0, stream>>>(
        param, output, khw, maxCount, oc, ic, d_khw, d_icp2);
}

// ---- Rearrange_Weight_Int8 (non-template, single fp32 shim) ----
void mnn_corpus_rearrange_weight_int8_fp32(const int8_t* param, int8_t* output,
                                            int khw, size_t maxCount, int oc, int ic,
                                            int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_khw(khw), d_icp(ic);
    MNN::Corpus::Rearrange_Weight_Int8<<<grid, block, 0, stream>>>(
        param, output, khw, maxCount, oc, ic, d_khw, d_icp);
}

} // extern "C"
