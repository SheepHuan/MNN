// attention.cu - Attention + LinearAttention kernels for replay benchmark corpus
//   source/backend/cuda/execution/AttentionExecution.cu
//   source/backend/cuda/execution/LinearAttentionExecution.cu
// Kernels are wrapped unconditionally (no MNN_SUPPORT_TRANSFORMER_FUSE guard).
#include "corpus_common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// MNN CUDA backend uses PACK_NUMBER=8 (source/backend/cuda/execution/MNNCUDADefine.hpp).
// LinearAttention read_qkv/read_token_channel/write_token_channel helpers depend on it.
static const int PACK_NUMBER = 8;

namespace MNN {
namespace Corpus {

// ============================================================================
// AttentionKernelParam (copied from AttentionExecution.hpp)
// ============================================================================
struct AttentionKernelParam {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int kv_head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int current_kv_seq_len_new;
    int past_kv_len;
};

// ============================================================================
// LinearAttention device helpers (copied from LinearAttentionExecution.cu)
// ============================================================================
template <typename T>
__device__ __forceinline__ float read_qkv(const T* input, int b, int d, int l, int D, int L, bool inputC4) {
    const int packedD = ((D + PACK_NUMBER - 1) / PACK_NUMBER) * PACK_NUMBER;
    const int offset = inputC4 ? (b * L + l) * packedD + d : (b * D + d) * L + l;
    return (float)input[offset];
}

template <typename T>
__device__ __forceinline__ float read_token_channel(const T* input, int b, int l, int c, int L, int C,
                                                    bool inputC4) {
    const int packedC = ((C + PACK_NUMBER - 1) / PACK_NUMBER) * PACK_NUMBER;
    const int offset = inputC4 ? (b * L + l) * packedC + c : (b * L + l) * C + c;
    return (float)input[offset];
}

template <typename T>
__device__ __forceinline__ void write_token_channel(T* output, int token, int c, int C, bool outputC4, float value) {
    const int packedC = ((C + PACK_NUMBER - 1) / PACK_NUMBER) * PACK_NUMBER;
    const int offset = outputC4 ? token * packedC + c : token * C + c;
    output[offset] = (T)value;
}

// ============================================================================
// compact_kv_cache_kernel (AttentionExecution.cu:20-69)
// ============================================================================
__global__ void compact_kv_cache_kernel(
    const void* src_key_cache,
    const void* src_value_cache,
    void* dst_key_cache,
    void* dst_value_cache,
    const int* reserve_info,
    const int* reserve_offsets,
    int n_reserve_pairs,
    int past_kv_len_after_remove,
    int b, int h_kv, int d,
    int src_kv_cache_max_len,
    int dst_kv_cache_max_len,
    size_t element_size
) {
    int bhd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int reserve_pair_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (bhd_idx >= b * h_kv * d || reserve_pair_idx >= n_reserve_pairs) {
        return;
    }

    int b_idx = bhd_idx / (h_kv * d);
    int h_kv_idx = (bhd_idx % (h_kv * d)) / d;
    int d_idx = bhd_idx % d;

    int copy_src_begin = reserve_info[reserve_pair_idx * 2];
    int copy_len = reserve_info[reserve_pair_idx * 2 + 1];
    int copy_dst_begin_offset = reserve_offsets[reserve_pair_idx];

    int src_offset = past_kv_len_after_remove + copy_src_begin;
    int dst_offset = past_kv_len_after_remove + copy_dst_begin_offset;

    const uint8_t* src_k_ptr = static_cast<const uint8_t*>(src_key_cache);
    uint8_t* dst_k_ptr = static_cast<uint8_t*>(dst_key_cache);
    const uint8_t* src_v_ptr = static_cast<const uint8_t*>(src_value_cache);
    uint8_t* dst_v_ptr = static_cast<uint8_t*>(dst_value_cache);

    for (int l = 0; l < copy_len; ++l) {
        int k_src_idx = (src_offset + l) * b * h_kv * d + b_idx * h_kv * d + h_kv_idx * d + d_idx;
        int k_dst_idx = (dst_offset + l) * b * h_kv * d + b_idx * h_kv * d + h_kv_idx * d + d_idx;
        memcpy(dst_k_ptr + k_dst_idx * element_size, src_k_ptr + k_src_idx * element_size, element_size);

        int v_src_idx = ((b_idx * h_kv + h_kv_idx) * src_kv_cache_max_len + (src_offset + l)) * d + d_idx;
        int v_dst_idx = ((b_idx * h_kv + h_kv_idx) * dst_kv_cache_max_len + (dst_offset + l)) * d + d_idx;
        memcpy(dst_v_ptr + v_dst_idx * element_size, src_v_ptr + v_src_idx * element_size, element_size);
    }
}

// ============================================================================
// copy_kv_to_cache_kernel<T> (AttentionExecution.cu:72-122)
// ============================================================================
template<typename T>
__global__ void copy_kv_to_cache_kernel(
    const T* key_input,
    const T* value_input,
    T* key_cache_output,
    T* value_cache_output,
    int batch_size,
    int new_kv_seq_len,
    int kv_num_head,
    int head_dim,
    int past_kv_len,
    int allocated_kv_len
) {
    int d_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l_idx_new = blockIdx.y * blockDim.y + threadIdx.y;
    int bh_kv_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (d_idx >= head_dim || l_idx_new >= new_kv_seq_len || bh_kv_idx >= batch_size * kv_num_head) {
        return;
    }

    int b_idx = bh_kv_idx / kv_num_head;
    int h_kv_idx = bh_kv_idx % kv_num_head;

    int input_offset = b_idx * new_kv_seq_len * kv_num_head * head_dim +
                             l_idx_new * kv_num_head * head_dim +
                             h_kv_idx * head_dim;

    T val_to_copy_k = key_input[input_offset + d_idx];
    T val_to_copy_v = value_input[input_offset + d_idx];

    int dest_seq_idx_cache = past_kv_len + l_idx_new;
    if (dest_seq_idx_cache >= allocated_kv_len) return;

    int key_cache_idx = dest_seq_idx_cache * batch_size * kv_num_head * head_dim +
                              b_idx * kv_num_head * head_dim +
                              h_kv_idx * head_dim +
                              d_idx;
    key_cache_output[key_cache_idx] = val_to_copy_k;

    int value_cache_idx = b_idx * kv_num_head * allocated_kv_len * head_dim +
                                h_kv_idx * allocated_kv_len * head_dim +
                                dest_seq_idx_cache * head_dim +
                                d_idx;
    value_cache_output[value_cache_idx] = val_to_copy_v;
}

// ============================================================================
// flash_decode_kernel<T> (AttentionExecution.cu:135-264)
// ============================================================================
template<typename T>
__global__ void flash_decode_kernel(
    const T* __restrict__ query_input,
    const T* __restrict__ key_cache,
    const T* __restrict__ value_cache,
    T* __restrict__ output,
    const int batch,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int key_seq_len,
    const int max_kv_len,
    const float scale
) {
    const int bh_idx = blockIdx.x;
    const int b_idx = bh_idx / head_num;
    const int h_q_idx = bh_idx % head_num;
    const int h_kv_idx = h_q_idx / (head_num / kv_head_num);
    const int tid = threadIdx.x;
    const int WARP_SIZE = 32;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    if (b_idx >= batch) return;

    const T* q_ptr = query_input + b_idx * head_num * head_dim + h_q_idx * head_dim;

    extern __shared__ char smem_raw[];
    float* smem_max = reinterpret_cast<float*>(smem_raw);
    float* smem_sum = smem_max + num_warps;
    float* smem_out = smem_sum + num_warps;

    float thread_max = -1e20f;
    float thread_sum = 0.0f;

    float thread_out[8];
    const int d_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < d_per_thread && i < 8; i++) thread_out[i] = 0.0f;

    const int KV_TILE = 16;
    for (int kv_start = 0; kv_start < key_seq_len; kv_start += KV_TILE) {
        int kv_end = min(kv_start + KV_TILE, key_seq_len);

        for (int k = kv_start; k < kv_end; k++) {
            const T* k_ptr = key_cache + k * batch * kv_head_num * head_dim
                            + b_idx * kv_head_num * head_dim
                            + h_kv_idx * head_dim;

            float qk_partial = 0.0f;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                qk_partial += (float)q_ptr[d] * (float)k_ptr[d];
            }

            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                qk_partial += __shfl_xor_sync(0xffffffff, qk_partial, offset);
            }
            if (lane_id == 0) {
                smem_max[warp_id] = qk_partial;
            }
            __syncthreads();

            float qk_score = 0.0f;
            if (tid == 0) {
                for (int w = 0; w < num_warps; w++) {
                    qk_score += smem_max[w];
                }
                qk_score *= scale;
                smem_max[0] = qk_score;
            }
            __syncthreads();
            qk_score = smem_max[0];

            float old_max = thread_max;
            float new_max = fmaxf(old_max, qk_score);
            float exp_diff = expf(old_max - new_max);
            float exp_score = expf(qk_score - new_max);

            for (int i = 0; i < d_per_thread && i < 8; i++) {
                thread_out[i] *= exp_diff;
            }
            thread_sum = thread_sum * exp_diff + exp_score;
            thread_max = new_max;

            const T* v_base = value_cache + b_idx * kv_head_num * max_kv_len * head_dim
                            + h_kv_idx * max_kv_len * head_dim;

            for (int i = 0; i < d_per_thread && i < 8; i++) {
                int d = tid + i * blockDim.x;
                if (d < head_dim) {
                    float v_val = (float)v_base[k * head_dim + d];
                    thread_out[i] += exp_score * v_val;
                }
            }
        }
    }

    float inv_sum = (thread_sum > 0.0f) ? (1.0f / thread_sum) : 0.0f;

    T* out_ptr = output + b_idx * head_num * head_dim + h_q_idx * head_dim;
    for (int i = 0; i < d_per_thread && i < 8; i++) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            out_ptr[d] = (T)(thread_out[i] * inv_sum);
        }
    }
}

// ============================================================================
// flash_decode_kernel_with_mask<T> (AttentionExecution.cu:274-405)
// ============================================================================
template<typename T>
__global__ void flash_decode_kernel_with_mask(
    const T* __restrict__ query_input,
    const T* __restrict__ key_cache,
    const T* __restrict__ value_cache,
    T* __restrict__ output,
    const T* __restrict__ mask,
    const int batch,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int key_seq_len,
    const int max_kv_len,
    const int query_seq_len,
    const float scale
) {
    const int idx = blockIdx.x;
    const int q_idx = idx % query_seq_len;
    const int bh_idx = idx / query_seq_len;
    const int b_idx = bh_idx / head_num;
    const int h_q_idx = bh_idx % head_num;
    const int h_kv_idx = h_q_idx / (head_num / kv_head_num);
    const int tid = threadIdx.x;
    const int WARP_SIZE = 32;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    if (b_idx >= batch) return;

    const T* q_ptr = query_input + b_idx * query_seq_len * head_num * head_dim
                    + q_idx * head_num * head_dim + h_q_idx * head_dim;

    const int past_kv_len = key_seq_len - query_seq_len;

    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    float thread_max = -1e20f;
    float thread_sum = 0.0f;

    float thread_out[8];
    const int d_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < d_per_thread && i < 8; i++) thread_out[i] = 0.0f;

    const T* v_base = value_cache + b_idx * kv_head_num * max_kv_len * head_dim
                    + h_kv_idx * max_kv_len * head_dim;

    for (int k = 0; k < key_seq_len; k++) {
        const T* k_ptr = key_cache + k * batch * kv_head_num * head_dim
                        + b_idx * kv_head_num * head_dim
                        + h_kv_idx * head_dim;

        float qk_partial = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            qk_partial += (float)q_ptr[d] * (float)k_ptr[d];
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            qk_partial += __shfl_xor_sync(0xffffffff, qk_partial, offset);
        }
        if (lane_id == 0) {
            smem[warp_id] = qk_partial;
        }
        __syncthreads();

        float qk_score = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < num_warps; w++) {
                qk_score += smem[w];
            }
            qk_score *= scale;

            if (k >= past_kv_len) {
                int new_k_idx = k - past_kv_len;
                float mask_val = (float)mask[q_idx * query_seq_len + new_k_idx];
                qk_score += mask_val;
            }

            smem[0] = qk_score;
        }
        __syncthreads();
        qk_score = smem[0];

        float old_max = thread_max;
        float new_max = fmaxf(old_max, qk_score);
        float exp_diff = expf(old_max - new_max);
        float exp_score = expf(qk_score - new_max);

        for (int i = 0; i < d_per_thread && i < 8; i++) {
            thread_out[i] *= exp_diff;
        }
        thread_sum = thread_sum * exp_diff + exp_score;
        thread_max = new_max;

        for (int i = 0; i < d_per_thread && i < 8; i++) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) {
                float v_val = (float)v_base[k * head_dim + d];
                thread_out[i] += exp_score * v_val;
            }
        }
    }

    float inv_sum = (thread_sum > 0.0f) ? (1.0f / thread_sum) : 0.0f;

    T* out_ptr = output + b_idx * query_seq_len * head_num * head_dim
                + q_idx * head_num * head_dim + h_q_idx * head_dim;
    for (int i = 0; i < d_per_thread && i < 8; i++) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            out_ptr[d] = (T)(thread_out[i] * inv_sum);
        }
    }
}

// ============================================================================
// flash_decode_kernel_splitk<T> (AttentionExecution.cu:412-537)
// ============================================================================
template<typename T>
__global__ void flash_decode_kernel_splitk(
    const T* __restrict__ query_input,
    const T* __restrict__ key_cache,
    const T* __restrict__ value_cache,
    float* __restrict__ partial_output,
    float* __restrict__ partial_meta,
    const int batch,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int key_seq_len,
    const int max_kv_len,
    const float scale_factor,
    const int parallel_blocks
) {
    const int bh_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int b_idx = bh_idx / head_num;
    const int h_q_idx = bh_idx % head_num;
    const int h_kv_idx = h_q_idx / (head_num / kv_head_num);
    const int tid = threadIdx.x;

    if (b_idx >= batch) return;

    const int kv_per_block = (key_seq_len + parallel_blocks - 1) / parallel_blocks;
    const int kv_start = split_idx * kv_per_block;
    const int kv_end = min(kv_start + kv_per_block, key_seq_len);
    if (kv_start >= key_seq_len) {
        if (tid == 0) {
            partial_meta[(split_idx * batch * head_num + bh_idx) * 2 + 0] = -1e20f;
            partial_meta[(split_idx * batch * head_num + bh_idx) * 2 + 1] = 0.0f;
        }
        const int d_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
        float* out_ptr = partial_output + (split_idx * batch * head_num + bh_idx) * head_dim;
        for (int i = 0; i < d_per_thread && i < 8; i++) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) out_ptr[d] = 0.0f;
        }
        return;
    }

    const T* q_ptr = query_input + b_idx * head_num * head_dim + h_q_idx * head_dim;

    const int WARP_SIZE_LOCAL = 32;
    const int warp_id = tid / WARP_SIZE_LOCAL;
    const int lane_id = tid % WARP_SIZE_LOCAL;
    const int num_warps = blockDim.x / WARP_SIZE_LOCAL;

    extern __shared__ char smem_raw_splitk[];
    float* smem_qk = reinterpret_cast<float*>(smem_raw_splitk);

    float thread_max = -1e20f;
    float thread_sum = 0.0f;

    const int d_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    float thread_out[8];
    for (int i = 0; i < d_per_thread && i < 8; i++) thread_out[i] = 0.0f;

    for (int k = kv_start; k < kv_end; k++) {
        const T* k_ptr = key_cache + k * batch * kv_head_num * head_dim
                        + b_idx * kv_head_num * head_dim
                        + h_kv_idx * head_dim;

        float qk_partial = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            qk_partial += (float)q_ptr[d] * (float)k_ptr[d];
        }

        for (int offset = WARP_SIZE_LOCAL / 2; offset > 0; offset >>= 1) {
            qk_partial += __shfl_xor_sync(0xffffffff, qk_partial, offset);
        }
        if (lane_id == 0) smem_qk[warp_id] = qk_partial;
        __syncthreads();

        float qk_score = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < num_warps; w++) qk_score += smem_qk[w];
            qk_score *= scale_factor;
            smem_qk[0] = qk_score;
        }
        __syncthreads();
        qk_score = smem_qk[0];

        float old_max = thread_max;
        float new_max = fmaxf(old_max, qk_score);
        float exp_diff = expf(old_max - new_max);
        float exp_score = expf(qk_score - new_max);

        for (int i = 0; i < d_per_thread && i < 8; i++) thread_out[i] *= exp_diff;
        thread_sum = thread_sum * exp_diff + exp_score;
        thread_max = new_max;

        const T* v_base = value_cache + b_idx * kv_head_num * max_kv_len * head_dim
                        + h_kv_idx * max_kv_len * head_dim;
        for (int i = 0; i < d_per_thread && i < 8; i++) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) {
                float v_val = (float)v_base[k * head_dim + d];
                thread_out[i] += exp_score * v_val;
            }
        }
    }

    float* out_ptr = partial_output + (split_idx * batch * head_num + bh_idx) * head_dim;
    for (int i = 0; i < d_per_thread && i < 8; i++) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            out_ptr[d] = thread_out[i];
        }
    }
    if (tid == 0) {
        partial_meta[(split_idx * batch * head_num + bh_idx) * 2 + 0] = thread_max;
        partial_meta[(split_idx * batch * head_num + bh_idx) * 2 + 1] = thread_sum;
    }
}

// ============================================================================
// flash_attn_combine_results<T> (AttentionExecution.cu:543-584)
// ============================================================================
template<typename T>
__global__ void flash_attn_combine_results(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_meta,
    T* __restrict__ final_output,
    const int batch,
    const int head_num,
    const int head_dim,
    const int parallel_blocks
) {
    const int bh_idx = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    const int b_idx = bh_idx / head_num;
    const int h_q_idx = bh_idx % head_num;

    float global_max = -1e20f;
    for (int s = 0; s < parallel_blocks; s++) {
        float local_max = partial_meta[(s * batch * head_num + bh_idx) * 2 + 0];
        global_max = fmaxf(global_max, local_max);
    }

    float combined_output = 0.0f;
    float combined_sum = 0.0f;

    for (int s = 0; s < parallel_blocks; s++) {
        float local_max = partial_meta[(s * batch * head_num + bh_idx) * 2 + 0];
        float local_sum = partial_meta[(s * batch * head_num + bh_idx) * 2 + 1];
        float rescale = expf(local_max - global_max);

        combined_output += partial_output[(s * batch * head_num + bh_idx) * head_dim + d] * rescale;
        combined_sum += local_sum * rescale;
    }

    float inv_sum = (combined_sum > 0.0f) ? (1.0f / combined_sum) : 0.0f;
    T* out_ptr = final_output + b_idx * head_num * head_dim + h_q_idx * head_dim;
    out_ptr[d] = (T)(combined_output * inv_sum);
}

// ============================================================================
// qk_kernel_tiled<T,AccT> (AttentionExecution.cu:589-696)
// ============================================================================
template <typename T, typename AccT = float>
__global__ void qk_kernel_tiled(const T* __restrict__ query_input,
                                const T* __restrict__ key_cache,
                                T* __restrict__ qk_scores_output,
                                const void* mask_tensor_data, const AttentionKernelParam* param, int q_seq_piece_offset,
                                bool has_mask_flag, bool is_add_mask_flag, bool is_causal_mask_flag) {
    const int QK_TILE = 16;
    __shared__ AccT q_tile[QK_TILE][128 + 1];
    __shared__ AccT k_tile[QK_TILE][128 + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bh_q_idx = blockIdx.z;

    if (bh_q_idx >= param->batch * param->head_num) return;

    const int b_idx = bh_q_idx / param->head_num;
    const int h_q_idx = bh_q_idx % param->head_num;
    const int h_kv_idx = h_q_idx / param->group;

    const int q_idx_in_piece = blockIdx.y * QK_TILE + ty;
    const int k_idx = blockIdx.x * QK_TILE + tx;
    const int current_full_q_idx = q_seq_piece_offset + q_idx_in_piece;

    AccT score_sum = 0.0f;

    const int D_TILE = 32;
    for (int d_start = 0; d_start < param->head_dim; d_start += D_TILE) {
        if (current_full_q_idx < param->query_seq_len && q_idx_in_piece < param->q_seq_piece_len) {
            for (int dd = tx; dd < D_TILE && (d_start + dd) < param->head_dim; dd += QK_TILE) {
                int q_offset = b_idx * param->query_seq_len * param->head_num * param->head_dim
                             + current_full_q_idx * param->head_num * param->head_dim
                             + h_q_idx * param->head_dim + d_start + dd;
                q_tile[ty][dd] = (AccT)query_input[q_offset];
            }
        } else {
            for (int dd = tx; dd < D_TILE; dd += QK_TILE) {
                q_tile[ty][dd] = AccT(0.0f);
            }
        }

        if (k_idx < param->key_seq_len) {
            for (int dd = ty; dd < D_TILE && (d_start + dd) < param->head_dim; dd += QK_TILE) {
                int k_offset = k_idx * param->batch * param->kv_head_num * param->head_dim
                             + b_idx * param->kv_head_num * param->head_dim
                             + h_kv_idx * param->head_dim + d_start + dd;
                k_tile[tx][dd] = (AccT)key_cache[k_offset];
            }
        } else {
            for (int dd = ty; dd < D_TILE; dd += QK_TILE) {
                k_tile[tx][dd] = AccT(0.0f);
            }
        }

        __syncthreads();

        int d_end = min(D_TILE, param->head_dim - d_start);
        #pragma unroll 8
        for (int dd = 0; dd < d_end; dd++) {
            score_sum += q_tile[ty][dd] * k_tile[tx][dd];
        }

        __syncthreads();
    }

    if (k_idx >= param->key_seq_len || q_idx_in_piece >= param->q_seq_piece_len || current_full_q_idx >= param->query_seq_len) {
        return;
    }

    score_sum *= param->scale;

    if (is_causal_mask_flag && k_idx > param->key_seq_len - param->query_seq_len + current_full_q_idx) {
        score_sum = (sizeof(T) == sizeof(__half)) ? AccT(-65504.0f) : AccT(-1e9f);
    } else if (has_mask_flag && mask_tensor_data) {
        if (is_add_mask_flag) {
            int mask_idx = current_full_q_idx * param->query_seq_len + k_idx - param->key_seq_len + param->query_seq_len;
            if (k_idx >= param->key_seq_len - param->query_seq_len) {
                if (sizeof(T) == sizeof(__half)) {
                    score_sum += __half2float(((const __half*)mask_tensor_data)[mask_idx]);
                } else {
                    score_sum += static_cast<const AccT*>(mask_tensor_data)[mask_idx];
                }
            }
        } else {
            int mask_idx = current_full_q_idx * param->key_seq_len + k_idx;
            if (static_cast<const int*>(mask_tensor_data)[mask_idx] == 0) {
                score_sum = (sizeof(T) == sizeof(__half)) ? AccT(-65504.0f) : AccT(-1e9f);
            }
        }
    }

    if (sizeof(T) == sizeof(__half)) {
        const AccT max_half_val = AccT(65504.0f);
        score_sum = fminf(fmaxf(score_sum, -max_half_val), max_half_val);
    }

    int out_idx = b_idx * param->head_num * param->q_seq_piece_len * param->key_seq_len +
                  h_q_idx * param->q_seq_piece_len * param->key_seq_len +
                  q_idx_in_piece * param->key_seq_len + k_idx;
    qk_scores_output[out_idx] = static_cast<T>(score_sum);
}

// ============================================================================
// qkv_kernel_tiled<T,AccT> (AttentionExecution.cu:701-752)
// ============================================================================
template<typename T, typename AccT = float>
__global__ void qkv_kernel_tiled(
    const T* __restrict__ softmax_probs,
    const T* __restrict__ value_cache,
    T* __restrict__ attention_output,
    const AttentionKernelParam* param,
    int q_seq_piece_offset
) {
    const int d_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int q_idx_in_piece = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh_q_idx = blockIdx.z;

    if (d_idx >= param->head_dim || q_idx_in_piece >= param->q_seq_piece_len || bh_q_idx >= param->batch * param->head_num) {
        return;
    }

    const int b_idx = bh_q_idx / param->head_num;
    const int h_q_idx = bh_q_idx % param->head_num;
    const int current_full_q_idx = q_seq_piece_offset + q_idx_in_piece;

    if (current_full_q_idx >= param->query_seq_len) return;

    const int h_kv_idx = h_q_idx / param->group;

    AccT weighted_sum = 0.0f;

    const T* prob_ptr = softmax_probs + b_idx * param->head_num * param->q_seq_piece_len * param->key_seq_len +
                        h_q_idx * param->q_seq_piece_len * param->key_seq_len +
                        q_idx_in_piece * param->key_seq_len;

    const T* val_ptr_base = value_cache + b_idx * param->kv_head_num * param->max_kv_len * param->head_dim +
                            h_kv_idx * param->max_kv_len * param->head_dim;

    const int hd = param->head_dim;
    int k_s = 0;
    for (; k_s + 3 < param->key_seq_len; k_s += 4) {
        weighted_sum += (AccT)prob_ptr[k_s]     * (AccT)val_ptr_base[k_s * hd + d_idx];
        weighted_sum += (AccT)prob_ptr[k_s + 1] * (AccT)val_ptr_base[(k_s + 1) * hd + d_idx];
        weighted_sum += (AccT)prob_ptr[k_s + 2] * (AccT)val_ptr_base[(k_s + 2) * hd + d_idx];
        weighted_sum += (AccT)prob_ptr[k_s + 3] * (AccT)val_ptr_base[(k_s + 3) * hd + d_idx];
    }
    for (; k_s < param->key_seq_len; k_s++) {
        weighted_sum += (AccT)prob_ptr[k_s] * (AccT)val_ptr_base[k_s * hd + d_idx];
    }

    int out_idx = b_idx * param->query_seq_len * param->head_num * param->head_dim +
                  current_full_q_idx * param->head_num * param->head_dim +
                  h_q_idx * param->head_dim + d_idx;
    attention_output[out_idx] = static_cast<T>(weighted_sum);
}

// ============================================================================
// conv1d_silu_kernel<T> (LinearAttentionExecution.cu:59-105)
// ============================================================================
template <typename T>
__global__ void conv1d_silu_kernel(const T* __restrict__ qkvInput,
                                   const T* __restrict__ convWeight,
                                   float* __restrict__ convState,
                                   float* __restrict__ convOutFp32,
                                   int B, int D, int L, int K_conv, int convStateSize, bool inputC4) {
    int channelIdx = blockIdx.x;
    if (channelIdx >= B * D) return;

    int d = channelIdx % D;
    int b = channelIdx / D;
    const T* weight = convWeight + d * K_conv;
    float* outFp32 = convOutFp32 + channelIdx * L;

    extern __shared__ float smem[];
    float* wShared = smem;
    float* padded = smem + K_conv;

    for (int i = threadIdx.x; i < K_conv; i += blockDim.x)
        wShared[i] = (float)weight[i];

    int totalLen = convStateSize + L;
    if (convState != nullptr) {
        float* state = convState + channelIdx * convStateSize;
        for (int i = threadIdx.x; i < convStateSize; i += blockDim.x)
            padded[i] = state[i];
    }
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        padded[convStateSize + i] = read_qkv(qkvInput, b, d, i, D, L, inputC4);
    __syncthreads();

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_conv; ++k)
            sum += padded[l + k] * wShared[k];
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        outFp32[l] = sum * sigmoid_val;
    }

    if (convState != nullptr && convStateSize > 0) {
        __syncthreads();
        float* state = convState + channelIdx * convStateSize;
        for (int i = threadIdx.x; i < convStateSize; i += blockDim.x)
            state[i] = padded[totalLen - convStateSize + i];
    }
}

// ============================================================================
// short_conv_kernel<T> (LinearAttentionExecution.cu:107-140)
// ============================================================================
template <typename T>
__global__ void short_conv_kernel(const T* __restrict__ qkvInput, const T* __restrict__ convWeight,
                                  float* __restrict__ convState, float* __restrict__ convOut, int B, int D, int L,
                                  int H, int K, int convStateSize, bool inputC4) {
    const int channelIdx = blockIdx.x;
    if (channelIdx >= B * H)
        return;
    const int b = channelIdx / H;
    const int h = channelIdx % H;
    extern __shared__ float padded[];

    for (int i = threadIdx.x; i < convStateSize; i += blockDim.x) {
        padded[i] = convState[channelIdx * convStateSize + i];
    }
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        const float bValue = read_qkv(qkvInput, b, h, l, D, L, inputC4);
        const float xValue = read_qkv(qkvInput, b, 2 * H + h, l, D, L, inputC4);
        padded[convStateSize + l] = bValue * xValue;
    }
    __syncthreads();

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += padded[l + k] * (float)convWeight[h * K + k];
        }
        convOut[channelIdx * L + l] = sum;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < convStateSize; i += blockDim.x) {
        convState[channelIdx * convStateSize + i] = padded[L + i];
    }
}

// ============================================================================
// short_conv_output_kernel<T> (LinearAttentionExecution.cu:142-156)
// ============================================================================
template <typename T>
__global__ void short_conv_output_kernel(const T* __restrict__ qkvInput, const float* __restrict__ convOut,
                                         T* __restrict__ output, int B, int D, int L, int H, bool inputC4,
                                         bool outputC4) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * L * H;
    if (index >= total)
        return;
    const int h = index % H;
    const int token = index / H;
    const int l = token % L;
    const int b = token / L;
    const float cValue = read_qkv(qkvInput, b, H + h, l, D, L, inputC4);
    write_token_channel(output, token, h, H, outputC4, cValue * convOut[(b * H + h) * L + l]);
}

// ============================================================================
// gated_delta_rule_decode_kernel<T> (LinearAttentionExecution.cu:192-304)
// ============================================================================
template<typename T>
__global__ void gated_delta_rule_decode_kernel(
    const float* __restrict__ convOut,
    const T* __restrict__ gateInput,
    const T* __restrict__ betaInput,
    float* __restrict__ recurrentState,
    T* __restrict__ output,
    int B, int H_k, int H_v, int d_k, int d_v,
    int key_dim, int val_dim, int D,
    int gqa_factor, bool useL2Norm, float qScale,
    bool gateC4, bool betaC4, bool outputC4
) {
    int idx = blockIdx.x;
    if (idx >= B * H_v) return;

    int b = idx / H_v;
    int h = idx % H_v;
    int k_head = h / gqa_factor;

    extern __shared__ float shared[];
    float* q_s = shared;
    float* k_s = q_s + d_k;
    float* v_s = k_s + d_k;
    float* vpred_s = v_s + d_v;
    float* delta_s = vpred_s + d_v;

    const float* convBase = convOut + b * D;
    for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
        q_s[i] = convBase[k_head * d_k + i];
        k_s[i] = convBase[key_dim + k_head * d_k + i];
    }
    for (int i = threadIdx.x; i < d_v; i += blockDim.x)
        v_s[i] = convBase[2 * key_dim + h * d_v + i];
    __syncthreads();

    if (useL2Norm) {
        __shared__ float normQ, normK;
        float sumSqQ = 0.0f, sumSqK = 0.0f;
        for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
            sumSqQ += q_s[i] * q_s[i];
            sumSqK += k_s[i] * k_s[i];
        }
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sumSqQ += __shfl_down_sync(0xffffffff, sumSqQ, offset);
            sumSqK += __shfl_down_sync(0xffffffff, sumSqK, offset);
        }
        __shared__ float warpSumsQ[32], warpSumsK[32];
        int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
        if (lid == 0) { warpSumsQ[wid] = sumSqQ; warpSumsK[wid] = sumSqK; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            float tQ = 0, tK = 0;
            for (int w = 0; w < nw; w++) { tQ += warpSumsQ[w]; tK += warpSumsK[w]; }
            normQ = 1.0f / sqrtf(tQ + 1e-6f);
            normK = 1.0f / sqrtf(tK + 1e-6f);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < d_k; i += blockDim.x) { q_s[i] *= normQ; k_s[i] *= normK; }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < d_k; i += blockDim.x) q_s[i] *= qScale;
    __syncthreads();

    float decay = expf(read_token_channel(gateInput, b, 0, h, 1, H_v, gateC4));
    float beta_t = read_token_channel(betaInput, b, 0, h, 1, H_v, betaC4);
    float* state = recurrentState + (b * H_v + h) * d_k * d_v;
    int stateSize = d_k * d_v;
    int stateSize4 = stateSize / 4;
    int dv4 = d_v / 4;
    float4* state4 = reinterpret_cast<float4*>(state);

    for (int i = threadIdx.x; i < stateSize4; i += blockDim.x) {
        float4 s = state4[i];
        s.x *= decay; s.y *= decay; s.z *= decay; s.w *= decay;
        state4[i] = s;
    }
    for (int i = stateSize4 * 4 + threadIdx.x; i < stateSize; i += blockDim.x)
        state[i] *= decay;
    __syncthreads();

    for (int j = threadIdx.x; j < d_v; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < d_k; i++) sum += state[i * d_v + j] * k_s[i];
        vpred_s[j] = sum;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < d_v; j += blockDim.x)
        delta_s[j] = beta_t * (v_s[j] - vpred_s[j]);
    __syncthreads();

    for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
        float k_val = k_s[i];
        float4* delta4 = reinterpret_cast<float4*>(delta_s);
        float4* row4 = reinterpret_cast<float4*>(state + i * d_v);
        for (int j4 = 0; j4 < dv4; j4++) {
            float4 d4 = delta4[j4], s4 = row4[j4];
            s4.x += k_val * d4.x; s4.y += k_val * d4.y;
            s4.z += k_val * d4.z; s4.w += k_val * d4.w;
            row4[j4] = s4;
        }
        for (int j = dv4 * 4; j < d_v; j++)
            state[i * d_v + j] += k_val * delta_s[j];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < d_v; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < d_k; i++) sum += state[i * d_v + j] * q_s[i];
        write_token_channel(output, b * H_v + h, j, d_v, outputC4, sum);
    }
}

} // namespace Corpus
} // namespace MNN

// ============================================================================
// extern "C" shim functions (fp32 corpus: T=float)
// ============================================================================
extern "C" {

// ---- compact_kv_cache_kernel (non-templated, 2D grid) ----
void mnn_corpus_compact_kv_cache_fp32(
    const void* src_key_cache, const void* src_value_cache,
    void* dst_key_cache, void* dst_value_cache,
    const int* reserve_info, const int* reserve_offsets,
    int n_reserve_pairs, int past_kv_len_after_remove,
    int b, int h_kv, int d,
    int src_kv_cache_max_len, int dst_kv_cache_max_len, size_t element_size,
    int gridX, int gridY, int blockX, int blockY, size_t sharedMem, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    dim3 block(blockX, blockY);
    MNN::Corpus::compact_kv_cache_kernel<<<grid, block, sharedMem, stream>>>(
        src_key_cache, src_value_cache, dst_key_cache, dst_value_cache,
        reserve_info, reserve_offsets, n_reserve_pairs, past_kv_len_after_remove,
        b, h_kv, d, src_kv_cache_max_len, dst_kv_cache_max_len, element_size);
}

// ---- copy_kv_to_cache_kernel<float> (3D grid) ----
void mnn_corpus_copy_kv_to_cache_fp32(
    const float* key_input, const float* value_input,
    float* key_cache_output, float* value_cache_output,
    int batch_size, int new_kv_seq_len, int kv_num_head, int head_dim,
    int past_kv_len, int allocated_kv_len,
    int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ,
    size_t sharedMem, cudaStream_t stream) {
    dim3 grid(gridX, gridY, gridZ);
    dim3 block(blockX, blockY, blockZ);
    MNN::Corpus::copy_kv_to_cache_kernel<float><<<grid, block, sharedMem, stream>>>(
        key_input, value_input, key_cache_output, value_cache_output,
        batch_size, new_kv_seq_len, kv_num_head, head_dim, past_kv_len, allocated_kv_len);
}

// ---- flash_decode_kernel<float> (1D grid, shared memory) ----
void mnn_corpus_flash_decode_fp32(
    const float* query_input, const float* key_cache, const float* value_cache,
    float* output, int batch, int head_num, int kv_head_num, int head_dim,
    int key_seq_len, int max_kv_len, float scale,
    int grid, int block, size_t sharedMem, cudaStream_t stream) {
    MNN::Corpus::flash_decode_kernel<float><<<grid, block, sharedMem, stream>>>(
        query_input, key_cache, value_cache, output, batch, head_num, kv_head_num,
        head_dim, key_seq_len, max_kv_len, scale);
}

// ---- flash_decode_kernel_with_mask<float> (1D grid, shared memory) ----
void mnn_corpus_flash_decode_with_mask_fp32(
    const float* query_input, const float* key_cache, const float* value_cache,
    float* output, const float* mask, int batch, int head_num, int kv_head_num,
    int head_dim, int key_seq_len, int max_kv_len, int query_seq_len, float scale,
    int grid, int block, size_t sharedMem, cudaStream_t stream) {
    MNN::Corpus::flash_decode_kernel_with_mask<float><<<grid, block, sharedMem, stream>>>(
        query_input, key_cache, value_cache, output, mask, batch, head_num, kv_head_num,
        head_dim, key_seq_len, max_kv_len, query_seq_len, scale);
}

// ---- flash_decode_kernel_splitk<float> (2D grid, shared memory) ----
void mnn_corpus_flash_decode_splitk_fp32(
    const float* query_input, const float* key_cache, const float* value_cache,
    float* partial_output, float* partial_meta,
    int batch, int head_num, int kv_head_num, int head_dim,
    int key_seq_len, int max_kv_len, float scale_factor, int parallel_blocks,
    int gridX, int gridY, int block, size_t sharedMem, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    MNN::Corpus::flash_decode_kernel_splitk<float><<<grid, block, sharedMem, stream>>>(
        query_input, key_cache, value_cache, partial_output, partial_meta,
        batch, head_num, kv_head_num, head_dim, key_seq_len, max_kv_len,
        scale_factor, parallel_blocks);
}

// ---- flash_attn_combine_results<float> (1D grid) ----
void mnn_corpus_flash_attn_combine_results_fp32(
    const float* partial_output, const float* partial_meta, float* final_output,
    int batch, int head_num, int head_dim, int parallel_blocks,
    int grid, int block, size_t sharedMem, cudaStream_t stream) {
    MNN::Corpus::flash_attn_combine_results<float><<<grid, block, sharedMem, stream>>>(
        partial_output, partial_meta, final_output, batch, head_num, head_dim, parallel_blocks);
}

// ---- qk_kernel_tiled<float,float> (3D grid, 2D block) ----
void mnn_corpus_qk_kernel_tiled_fp32(
    const float* query_input, const float* key_cache, float* qk_scores_output,
    const void* mask_tensor_data, const MNN::Corpus::AttentionKernelParam* param,
    int q_seq_piece_offset, bool has_mask_flag, bool is_add_mask_flag, bool is_causal_mask_flag,
    int gridX, int gridY, int gridZ, int blockX, int blockY, size_t sharedMem, cudaStream_t stream) {
    dim3 grid(gridX, gridY, gridZ);
    dim3 block(blockX, blockY);
    MNN::Corpus::qk_kernel_tiled<float, float><<<grid, block, sharedMem, stream>>>(
        query_input, key_cache, qk_scores_output, mask_tensor_data, param,
        q_seq_piece_offset, has_mask_flag, is_add_mask_flag, is_causal_mask_flag);
}

// ---- qkv_kernel_tiled<float,float> (3D grid, 2D block) ----
void mnn_corpus_qkv_kernel_tiled_fp32(
    const float* softmax_probs, const float* value_cache, float* attention_output,
    const MNN::Corpus::AttentionKernelParam* param, int q_seq_piece_offset,
    int gridX, int gridY, int gridZ, int blockX, int blockY, size_t sharedMem, cudaStream_t stream) {
    dim3 grid(gridX, gridY, gridZ);
    dim3 block(blockX, blockY);
    MNN::Corpus::qkv_kernel_tiled<float, float><<<grid, block, sharedMem, stream>>>(
        softmax_probs, value_cache, attention_output, param, q_seq_piece_offset);
}

// ---- conv1d_silu_kernel<float> (1D grid, shared memory) ----
void mnn_corpus_conv1d_silu_fp32(
    const float* qkvInput, const float* convWeight, float* convState, float* convOutFp32,
    int B, int D, int L, int K_conv, int convStateSize, bool inputC4,
    int grid, int block, size_t sharedMem, cudaStream_t stream) {
    MNN::Corpus::conv1d_silu_kernel<float><<<grid, block, sharedMem, stream>>>(
        qkvInput, convWeight, convState, convOutFp32, B, D, L, K_conv, convStateSize, inputC4);
}

// ---- short_conv_kernel<float> (1D grid, shared memory) ----
void mnn_corpus_short_conv_fp32(
    const float* qkvInput, const float* convWeight, float* convState, float* convOut,
    int B, int D, int L, int H, int K, int convStateSize, bool inputC4,
    int grid, int block, size_t sharedMem, cudaStream_t stream) {
    MNN::Corpus::short_conv_kernel<float><<<grid, block, sharedMem, stream>>>(
        qkvInput, convWeight, convState, convOut, B, D, L, H, K, convStateSize, inputC4);
}

// ---- short_conv_output_kernel<float> (1D grid) ----
void mnn_corpus_short_conv_output_fp32(
    const float* qkvInput, const float* convOut, float* output,
    int B, int D, int L, int H, bool inputC4, bool outputC4,
    int grid, int block, size_t sharedMem, cudaStream_t stream) {
    MNN::Corpus::short_conv_output_kernel<float><<<grid, block, sharedMem, stream>>>(
        qkvInput, convOut, output, B, D, L, H, inputC4, outputC4);
}

// ---- gated_delta_rule_decode_kernel<float> (1D grid, shared memory) ----
void mnn_corpus_gated_delta_rule_decode_fp32(
    const float* convOut, const float* gateInput, const float* betaInput,
    float* recurrentState, float* output,
    int B, int H_k, int H_v, int d_k, int d_v,
    int key_dim, int val_dim, int D,
    int gqa_factor, bool useL2Norm, float qScale,
    bool gateC4, bool betaC4, bool outputC4,
    int grid, int block, size_t sharedMem, cudaStream_t stream) {
    MNN::Corpus::gated_delta_rule_decode_kernel<float><<<grid, block, sharedMem, stream>>>(
        convOut, gateInput, betaInput, recurrentState, output,
        B, H_k, H_v, d_k, d_v, key_dim, val_dim, D,
        gqa_factor, useL2Norm, qScale, gateC4, betaC4, outputC4);
}

} // extern "C"
