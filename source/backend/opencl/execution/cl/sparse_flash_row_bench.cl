#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#ifndef NUMHEAD_GROUP_SIZE
#define NUMHEAD_GROUP_SIZE 1
#endif

#define SPARSE_FLASH_MAX_HEAD_DIM8 16

__kernel void sparse_flash_attention_row64(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 query_seq_len_4]
                              __global const FLOAT *past_key, // [batch kv_head_num head_dim_4 kv_max_length]
                              __global const FLOAT *past_value, // [batch kv_head_num kv_max_length head_dim]
                              __global const int *sparse_query, // [query_seq_len]
                              __global FLOAT *output, // [batch query_seq_len head_num head_dim]
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int output_seq_len,
                              __private const int query_rows_are_full,
                              __private const int q_start,
                              __private const int q_piece_len,
                              __private const int key_seq_len,
                              __private const int key_max_len,
                              __private const int value_max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
    const int x = get_global_id(0); // fixed 64 lanes
    const int y = get_global_id(1); // q piece token
    int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lid = get_local_id(0);
    const int head_dim8 = head_dim >> 3;
    if (get_local_size(0) != 64 || lid >= 64 || (head_dim != 64 && head_dim != 128) ||
        head_dim8 <= 0 || head_dim8 > SPARSE_FLASH_MAX_HEAD_DIM8) {
        return;
    }
    const int q_index = q_start + y;
    if (y >= q_piece_len || q_index >= output_seq_len) {
        return;
    }

    const int b = z / head_num;
    const int h = z - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }
    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int q_logical = sparse_query[q_index];
    const int q_row = query_rows_are_full ? q_logical : q_index;
    if (q_row < 0 || q_row >= query_seq_len) {
        return;
    }
    const int active_kv_seq_len = clamp(q_logical + 1, 0, key_seq_len);

    COMPUTE_FLOAT local local_m[64];
    COMPUTE_FLOAT local local_l[64];
    COMPUTE_FLOAT8 local local_o[64 * SPARSE_FLASH_MAX_HEAD_DIM8];
    COMPUTE_FLOAT8 o[SPARSE_FLASH_MAX_HEAD_DIM8];
    for (int d8 = 0; d8 < head_dim8; ++d8) {
        o[d8] = (COMPUTE_FLOAT8)0;
    }

    COMPUTE_FLOAT m = (COMPUTE_FLOAT)-FLT_MAX;
    COMPUTE_FLOAT l = (COMPUTE_FLOAT)0;
    const int query_offset = z * head_dim * query_seq_len4 + q_row;
    const int key_base = (b * kv_head_num + kvh) * head_dim * key_max_len;
    const int value_base = (b * kv_head_num + kvh) * value_max_len * head_dim;

    for (int k = lid; k < active_kv_seq_len; k += 64) {
        COMPUTE_FLOAT score = (COMPUTE_FLOAT)0;
        for (int d4 = 0; d4 < head_dim; d4 += 4) {
            COMPUTE_FLOAT4 qv = CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_offset + d4 * query_seq_len4));
            COMPUTE_FLOAT4 kv = CONVERT_COMPUTE_FLOAT4(vload4(0, past_key + key_base + d4 * key_max_len + k));
            score += dot(qv, kv);
        }
        score *= (COMPUTE_FLOAT)scale;
        const COMPUTE_FLOAT new_m = fmax(m, score);
        const COMPUTE_FLOAT alpha = l > (COMPUTE_FLOAT)0 ? exp(m - new_m) : (COMPUTE_FLOAT)0;
        const COMPUTE_FLOAT beta = exp(score - new_m);
        const int value_offset = value_base + k * head_dim;
        const COMPUTE_FLOAT8 beta8 = (COMPUTE_FLOAT8)beta;
        for (int d8 = 0; d8 < head_dim8; ++d8) {
            o[d8] = o[d8] * alpha + beta8 *
                CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + value_offset + (d8 << 3)));
        }
        l = l * alpha + beta;
        m = new_m;
    }
    local_m[lid] = m;
    local_l[lid] = l;
    for (int d8 = 0; d8 < head_dim8; ++d8) {
        local_o[lid * SPARSE_FLASH_MAX_HEAD_DIM8 + d8] = o[d8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            const COMPUTE_FLOAT m0 = local_m[lid];
            const COMPUTE_FLOAT l0 = local_l[lid];
            const COMPUTE_FLOAT m1 = local_m[lid + stride];
            const COMPUTE_FLOAT l1 = local_l[lid + stride];
            const COMPUTE_FLOAT merged_m = fmax(m0, m1);
            const COMPUTE_FLOAT a = l0 > (COMPUTE_FLOAT)0 ? exp(m0 - merged_m) : (COMPUTE_FLOAT)0;
            const COMPUTE_FLOAT b_scale = l1 > (COMPUTE_FLOAT)0 ? exp(m1 - merged_m) : (COMPUTE_FLOAT)0;
            for (int d8 = 0; d8 < head_dim8; ++d8) {
                local_o[lid * SPARSE_FLASH_MAX_HEAD_DIM8 + d8] =
                    local_o[lid * SPARSE_FLASH_MAX_HEAD_DIM8 + d8] * a +
                    local_o[(lid + stride) * SPARSE_FLASH_MAX_HEAD_DIM8 + d8] * b_scale;
            }
            local_l[lid] = l0 * a + l1 * b_scale;
            local_m[lid] = merged_m;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const COMPUTE_FLOAT denom = local_l[0] > (COMPUTE_FLOAT)0 ? local_l[0] : (COMPUTE_FLOAT)1;
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim;
        for (int d8 = 0; d8 < head_dim8; ++d8) {
            vstore8(CONVERT_FLOAT8(local_o[d8] / (COMPUTE_FLOAT8)denom), 0,
                    output + output_offset + (d8 << 3));
        }
    }
}
__kernel void sparse_flash_attention_row32(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 query_seq_len_4]
                              __global const FLOAT *past_key, // [batch kv_head_num head_dim_4 kv_max_length]
                              __global const FLOAT *past_value, // [batch kv_head_num kv_max_length head_dim]
                              __global const int *sparse_query, // [query_seq_len]
                              __global FLOAT *output, // [batch query_seq_len head_num head_dim]
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int output_seq_len,
                              __private const int query_rows_are_full,
                              __private const int q_start,
                              __private const int q_piece_len,
                              __private const int key_seq_len,
                              __private const int key_max_len,
                              __private const int value_max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
    const int x = get_global_id(0); // fixed 32 lanes
    const int y = get_global_id(1); // q piece token
    int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lid = get_local_id(0);
    const int head_dim8 = head_dim >> 3;
    if (get_local_size(0) != 32 || lid >= 32 || (head_dim != 64 && head_dim != 128) ||
        head_dim8 <= 0 || head_dim8 > SPARSE_FLASH_MAX_HEAD_DIM8) {
        return;
    }
    const int q_index = q_start + y;
    if (y >= q_piece_len || q_index >= output_seq_len) {
        return;
    }

    const int b = z / head_num;
    const int h = z - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }
    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int q_logical = sparse_query[q_index];
    const int q_row = query_rows_are_full ? q_logical : q_index;
    if (q_row < 0 || q_row >= query_seq_len) {
        return;
    }
    const int active_kv_seq_len = clamp(q_logical + 1, 0, key_seq_len);

    COMPUTE_FLOAT local local_m[32];
    COMPUTE_FLOAT local local_l[32];
    COMPUTE_FLOAT8 local local_o[32 * SPARSE_FLASH_MAX_HEAD_DIM8];
    COMPUTE_FLOAT8 o[SPARSE_FLASH_MAX_HEAD_DIM8];
    for (int d8 = 0; d8 < head_dim8; ++d8) {
        o[d8] = (COMPUTE_FLOAT8)0;
    }

    COMPUTE_FLOAT m = (COMPUTE_FLOAT)-FLT_MAX;
    COMPUTE_FLOAT l = (COMPUTE_FLOAT)0;
    const int query_offset = z * head_dim * query_seq_len4 + q_row;
    const int key_base = (b * kv_head_num + kvh) * head_dim * key_max_len;
    const int value_base = (b * kv_head_num + kvh) * value_max_len * head_dim;

    for (int k = lid; k < active_kv_seq_len; k += 32) {
        COMPUTE_FLOAT score = (COMPUTE_FLOAT)0;
        for (int d4 = 0; d4 < head_dim; d4 += 4) {
            COMPUTE_FLOAT4 qv = CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_offset + d4 * query_seq_len4));
            COMPUTE_FLOAT4 kv = CONVERT_COMPUTE_FLOAT4(vload4(0, past_key + key_base + d4 * key_max_len + k));
            score += dot(qv, kv);
        }
        score *= (COMPUTE_FLOAT)scale;
        const COMPUTE_FLOAT new_m = fmax(m, score);
        const COMPUTE_FLOAT alpha = l > (COMPUTE_FLOAT)0 ? exp(m - new_m) : (COMPUTE_FLOAT)0;
        const COMPUTE_FLOAT beta = exp(score - new_m);
        const int value_offset = value_base + k * head_dim;
        const COMPUTE_FLOAT8 beta8 = (COMPUTE_FLOAT8)beta;
        for (int d8 = 0; d8 < head_dim8; ++d8) {
            o[d8] = o[d8] * alpha + beta8 *
                CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + value_offset + (d8 << 3)));
        }
        l = l * alpha + beta;
        m = new_m;
    }
    local_m[lid] = m;
    local_l[lid] = l;
    for (int d8 = 0; d8 < head_dim8; ++d8) {
        local_o[lid * SPARSE_FLASH_MAX_HEAD_DIM8 + d8] = o[d8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lid < stride) {
            const COMPUTE_FLOAT m0 = local_m[lid];
            const COMPUTE_FLOAT l0 = local_l[lid];
            const COMPUTE_FLOAT m1 = local_m[lid + stride];
            const COMPUTE_FLOAT l1 = local_l[lid + stride];
            const COMPUTE_FLOAT merged_m = fmax(m0, m1);
            const COMPUTE_FLOAT a = l0 > (COMPUTE_FLOAT)0 ? exp(m0 - merged_m) : (COMPUTE_FLOAT)0;
            const COMPUTE_FLOAT b_scale = l1 > (COMPUTE_FLOAT)0 ? exp(m1 - merged_m) : (COMPUTE_FLOAT)0;
            for (int d8 = 0; d8 < head_dim8; ++d8) {
                local_o[lid * SPARSE_FLASH_MAX_HEAD_DIM8 + d8] =
                    local_o[lid * SPARSE_FLASH_MAX_HEAD_DIM8 + d8] * a +
                    local_o[(lid + stride) * SPARSE_FLASH_MAX_HEAD_DIM8 + d8] * b_scale;
            }
            local_l[lid] = l0 * a + l1 * b_scale;
            local_m[lid] = merged_m;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const COMPUTE_FLOAT denom = local_l[0] > (COMPUTE_FLOAT)0 ? local_l[0] : (COMPUTE_FLOAT)1;
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim;
        for (int d8 = 0; d8 < head_dim8; ++d8) {
            vstore8(CONVERT_FLOAT8(local_o[d8] / (COMPUTE_FLOAT8)denom), 0,
                    output + output_offset + (d8 << 3));
        }
    }
}
