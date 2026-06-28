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

#define MQTILE_SPARSE_FLASH_HD64_QTILE 4
#define MQTILE_SPARSE_FLASH_HD64_KTILE 16
#define MQTILE_SPARSE_FLASH_HD64_DLANES 16

__kernel void mqtile_sparse_flash_hd64_q4k16(GLOBAL_SIZE_3_DIMS
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
    const int x = get_global_id(0); // fixed 16 dim lanes
    const int y = get_global_id(1); // q piece token
    int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lane = get_local_id(0);
    const int q_lane = get_local_id(1);
    if (get_local_size(0) != MQTILE_SPARSE_FLASH_HD64_DLANES ||
        get_local_size(1) != MQTILE_SPARSE_FLASH_HD64_QTILE ||
        lane >= MQTILE_SPARSE_FLASH_HD64_DLANES ||
        q_lane >= MQTILE_SPARSE_FLASH_HD64_QTILE ||
        head_dim != 64) {
        return;
    }

    const int q_tile_base = y - q_lane;
    const int q_tile_valid_count = clamp(q_piece_len - q_tile_base, 0, MQTILE_SPARSE_FLASH_HD64_QTILE);
    if (q_tile_valid_count <= 0) {
        return;
    }
    const int q_index = q_start + y;
    const int q_valid = (q_lane < q_tile_valid_count && y < q_piece_len && q_index < output_seq_len) ? 1 : 0;

    const int b = z / head_num;
    const int h = z - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }

    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int q_logical = q_valid ? sparse_query[q_index] : 0;
    const int q_row = query_rows_are_full ? q_logical : q_index;
    const int q_row_valid = q_valid && q_row >= 0 && q_row < query_seq_len;
    int local q_tile_logical[MQTILE_SPARSE_FLASH_HD64_QTILE];
    int local q_tile_min_logical;
    int local q_tile_max_logical;
    if (lane == 0) {
        q_tile_logical[q_lane] = q_row_valid ? q_logical : -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0 && q_lane == 0) {
        int tile_min_logical = key_seq_len;
        int tile_max_logical = -1;
        for (int qi = 0; qi < MQTILE_SPARSE_FLASH_HD64_QTILE; ++qi) {
            const int logical = q_tile_logical[qi];
            if (logical < 0) {
                continue;
            }
            tile_min_logical = min(tile_min_logical, logical);
            tile_max_logical = max(tile_max_logical, logical);
        }
        q_tile_min_logical = tile_max_logical >= 0 ? tile_min_logical : -1;
        q_tile_max_logical = tile_max_logical;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int active_kv_seq_len = q_tile_max_logical >= 0 ? clamp(q_tile_max_logical + 1, 0, key_seq_len) : 0;

    COMPUTE_FLOAT local row_m[MQTILE_SPARSE_FLASH_HD64_QTILE];
    COMPUTE_FLOAT local row_l[MQTILE_SPARSE_FLASH_HD64_QTILE];
    COMPUTE_FLOAT local row_alpha[MQTILE_SPARSE_FLASH_HD64_QTILE];
    COMPUTE_FLOAT local row_new_m[MQTILE_SPARSE_FLASH_HD64_QTILE];
    COMPUTE_FLOAT local score_tile[MQTILE_SPARSE_FLASH_HD64_QTILE * MQTILE_SPARSE_FLASH_HD64_KTILE];
    COMPUTE_FLOAT local beta_tile[MQTILE_SPARSE_FLASH_HD64_QTILE * MQTILE_SPARSE_FLASH_HD64_KTILE];
    COMPUTE_FLOAT local partial_tile[MQTILE_SPARSE_FLASH_HD64_QTILE *
                                     MQTILE_SPARSE_FLASH_HD64_KTILE *
                                     MQTILE_SPARSE_FLASH_HD64_DLANES];
    FLOAT4 local local_k[MQTILE_SPARSE_FLASH_HD64_KTILE * MQTILE_SPARSE_FLASH_HD64_DLANES];
    FLOAT4 local local_v[MQTILE_SPARSE_FLASH_HD64_KTILE * MQTILE_SPARSE_FLASH_HD64_DLANES];

    if (lane == 0) {
        row_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
        row_l[q_lane] = (COMPUTE_FLOAT)0;
        row_alpha[q_lane] = (COMPUTE_FLOAT)0;
        row_new_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int d_base = lane << 2;
    const int query_base = z * head_dim * query_seq_len4 + q_row;
    const COMPUTE_FLOAT4 qv = q_row_valid
        ? CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_base + d_base * query_seq_len4))
        : (COMPUTE_FLOAT4)0;
    COMPUTE_FLOAT4 out = (COMPUTE_FLOAT4)0;

    const int key_base = (b * kv_head_num + kvh) * head_dim * key_max_len;
    const int value_base = (b * kv_head_num + kvh) * value_max_len * head_dim;
    const int linear_tid = q_lane * MQTILE_SPARSE_FLASH_HD64_DLANES + lane;
    for (int k_base = 0; k_base < active_kv_seq_len; k_base += MQTILE_SPARSE_FLASH_HD64_KTILE) {
        const int k_tile = min(MQTILE_SPARSE_FLASH_HD64_KTILE, active_kv_seq_len - k_base);
        const int k_tile_end = k_base + k_tile - 1;
        const int k_tile_full = (q_tile_min_logical >= 0 && k_tile_end <= q_tile_min_logical) ? 1 : 0;
        for (int load = linear_tid; load < k_tile * MQTILE_SPARSE_FLASH_HD64_DLANES;
             load += MQTILE_SPARSE_FLASH_HD64_QTILE * MQTILE_SPARSE_FLASH_HD64_DLANES) {
            const int k_local = load / MQTILE_SPARSE_FLASH_HD64_DLANES;
            const int d_lane = load - k_local * MQTILE_SPARSE_FLASH_HD64_DLANES;
            const int dim = d_lane << 2;
            const int k_index = k_base + k_local;
            local_k[k_local * MQTILE_SPARSE_FLASH_HD64_DLANES + d_lane] =
                vload4(0, past_key + key_base + dim * key_max_len + k_index);
            local_v[k_local * MQTILE_SPARSE_FLASH_HD64_DLANES + d_lane] =
                vload4(0, past_value + value_base + k_index * head_dim + dim);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_local = 0; k_local < MQTILE_SPARSE_FLASH_HD64_KTILE; ++k_local) {
            const int partial_index = ((q_lane * MQTILE_SPARSE_FLASH_HD64_KTILE + k_local) *
                                       MQTILE_SPARSE_FLASH_HD64_DLANES) + lane;
            COMPUTE_FLOAT partial = (COMPUTE_FLOAT)0;
            if (q_row_valid && k_local < k_tile) {
                const int k_index = k_base + k_local;
                const int k_visible = k_tile_full || k_index <= q_logical;
                if (k_visible) {
                    const COMPUTE_FLOAT4 kv = CONVERT_COMPUTE_FLOAT4(
                        local_k[k_local * MQTILE_SPARSE_FLASH_HD64_DLANES + lane]);
                    partial = dot(qv, kv);
                }
            }
            partial_tile[partial_index] = partial;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < MQTILE_SPARSE_FLASH_HD64_KTILE) {
            COMPUTE_FLOAT score = (COMPUTE_FLOAT)-FLT_MAX;
            if (q_row_valid && lane < k_tile) {
                const int k_index = k_base + lane;
                const int score_index = q_lane * MQTILE_SPARSE_FLASH_HD64_KTILE + lane;
                const int partial_base = score_index * MQTILE_SPARSE_FLASH_HD64_DLANES;
                if (k_tile_full || k_index <= q_logical) {
                    score = (COMPUTE_FLOAT)0;
                    for (int d_lane = 0; d_lane < MQTILE_SPARSE_FLASH_HD64_DLANES; ++d_lane) {
                        score += partial_tile[partial_base + d_lane];
                    }
                    score *= (COMPUTE_FLOAT)scale;
                }
            }
            score_tile[q_lane * MQTILE_SPARSE_FLASH_HD64_KTILE + lane] = score;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            COMPUTE_FLOAT tile_max = (COMPUTE_FLOAT)-FLT_MAX;
            if (q_row_valid) {
                for (int k_local = 0; k_local < k_tile; ++k_local) {
                    tile_max = fmax(tile_max,
                                    score_tile[q_lane * MQTILE_SPARSE_FLASH_HD64_KTILE + k_local]);
                }
                const COMPUTE_FLOAT merged_m = fmax(row_m[q_lane], tile_max);
                row_alpha[q_lane] = row_l[q_lane] > (COMPUTE_FLOAT)0
                    ? exp(row_m[q_lane] - merged_m)
                    : (COMPUTE_FLOAT)0;
                row_new_m[q_lane] = merged_m;
            } else {
                row_alpha[q_lane] = (COMPUTE_FLOAT)0;
                row_new_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < MQTILE_SPARSE_FLASH_HD64_KTILE) {
            COMPUTE_FLOAT beta = (COMPUTE_FLOAT)0;
            if (q_row_valid && lane < k_tile) {
                beta = exp(score_tile[q_lane * MQTILE_SPARSE_FLASH_HD64_KTILE + lane] - row_new_m[q_lane]);
            }
            beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD64_KTILE + lane] = beta;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            COMPUTE_FLOAT tile_sum = (COMPUTE_FLOAT)0;
            if (q_row_valid) {
                for (int k_local = 0; k_local < k_tile; ++k_local) {
                    tile_sum += beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD64_KTILE + k_local];
                }
                row_l[q_lane] = row_l[q_lane] * row_alpha[q_lane] + tile_sum;
                row_m[q_lane] = row_new_m[q_lane];
            } else {
                row_l[q_lane] = (COMPUTE_FLOAT)0;
                row_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (q_row_valid) {
            out *= (COMPUTE_FLOAT4)row_alpha[q_lane];
            for (int k_local = 0; k_local < k_tile; ++k_local) {
                out += (COMPUTE_FLOAT4)beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD64_KTILE + k_local] *
                       CONVERT_COMPUTE_FLOAT4(local_v[k_local * MQTILE_SPARSE_FLASH_HD64_DLANES + lane]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (q_row_valid) {
        const COMPUTE_FLOAT denom = row_l[q_lane] > (COMPUTE_FLOAT)0 ? row_l[q_lane] : (COMPUTE_FLOAT)1;
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim + d_base;
        vstore4(CONVERT_FLOAT4(out / (COMPUTE_FLOAT4)denom), 0, output + output_offset);
    }
}

#define MQTILE_SPARSE_FLASH_HD128_QTILE 4
#define MQTILE_SPARSE_FLASH_HD128_KTILE 16
#define MQTILE_SPARSE_FLASH_HD128_DLANES 16

__kernel void mqtile_sparse_flash_hd128_q4k16(GLOBAL_SIZE_3_DIMS
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
    const int x = get_global_id(0); // fixed 16 dim lanes
    const int y = get_global_id(1); // q piece token
    int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lane = get_local_id(0);
    const int q_lane = get_local_id(1);
    if (get_local_size(0) != MQTILE_SPARSE_FLASH_HD128_DLANES ||
        get_local_size(1) != MQTILE_SPARSE_FLASH_HD128_QTILE ||
        lane >= MQTILE_SPARSE_FLASH_HD128_DLANES ||
        q_lane >= MQTILE_SPARSE_FLASH_HD128_QTILE ||
        head_dim != 128) {
        return;
    }

    const int q_tile_base = y - q_lane;
    const int q_tile_valid_count = clamp(q_piece_len - q_tile_base, 0, MQTILE_SPARSE_FLASH_HD128_QTILE);
    if (q_tile_valid_count <= 0) {
        return;
    }
    const int q_index = q_start + y;
    const int q_valid = (q_lane < q_tile_valid_count && y < q_piece_len && q_index < output_seq_len) ? 1 : 0;

    const int b = z / head_num;
    const int h = z - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }

    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int q_logical = q_valid ? sparse_query[q_index] : 0;
    const int q_row = query_rows_are_full ? q_logical : q_index;
    const int q_row_valid = q_valid && q_row >= 0 && q_row < query_seq_len;
    int local q_tile_logical[MQTILE_SPARSE_FLASH_HD128_QTILE];
    int local q_tile_min_logical;
    int local q_tile_max_logical;
    if (lane == 0) {
        q_tile_logical[q_lane] = q_row_valid ? q_logical : -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0 && q_lane == 0) {
        int tile_min_logical = key_seq_len;
        int tile_max_logical = -1;
        for (int qi = 0; qi < MQTILE_SPARSE_FLASH_HD128_QTILE; ++qi) {
            const int logical = q_tile_logical[qi];
            if (logical < 0) {
                continue;
            }
            tile_min_logical = min(tile_min_logical, logical);
            tile_max_logical = max(tile_max_logical, logical);
        }
        q_tile_min_logical = tile_max_logical >= 0 ? tile_min_logical : -1;
        q_tile_max_logical = tile_max_logical;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int active_kv_seq_len = q_tile_max_logical >= 0 ? clamp(q_tile_max_logical + 1, 0, key_seq_len) : 0;

    COMPUTE_FLOAT local row_m[MQTILE_SPARSE_FLASH_HD128_QTILE];
    COMPUTE_FLOAT local row_l[MQTILE_SPARSE_FLASH_HD128_QTILE];
    COMPUTE_FLOAT local row_alpha[MQTILE_SPARSE_FLASH_HD128_QTILE];
    COMPUTE_FLOAT local row_new_m[MQTILE_SPARSE_FLASH_HD128_QTILE];
    COMPUTE_FLOAT local score_tile[MQTILE_SPARSE_FLASH_HD128_QTILE * MQTILE_SPARSE_FLASH_HD128_KTILE];
    COMPUTE_FLOAT local beta_tile[MQTILE_SPARSE_FLASH_HD128_QTILE * MQTILE_SPARSE_FLASH_HD128_KTILE];
    COMPUTE_FLOAT local partial_tile[MQTILE_SPARSE_FLASH_HD128_QTILE *
                                     MQTILE_SPARSE_FLASH_HD128_KTILE *
                                     MQTILE_SPARSE_FLASH_HD128_DLANES];
    FLOAT4 local local_k0[MQTILE_SPARSE_FLASH_HD128_KTILE * MQTILE_SPARSE_FLASH_HD128_DLANES];
    FLOAT4 local local_k1[MQTILE_SPARSE_FLASH_HD128_KTILE * MQTILE_SPARSE_FLASH_HD128_DLANES];
    FLOAT8 local local_v[MQTILE_SPARSE_FLASH_HD128_KTILE * MQTILE_SPARSE_FLASH_HD128_DLANES];

    if (lane == 0) {
        row_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
        row_l[q_lane] = (COMPUTE_FLOAT)0;
        row_alpha[q_lane] = (COMPUTE_FLOAT)0;
        row_new_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int d_base = lane << 3;
    const int query_base = z * head_dim * query_seq_len4 + q_row;
    const COMPUTE_FLOAT4 qv0 = q_row_valid
        ? CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_base + d_base * query_seq_len4))
        : (COMPUTE_FLOAT4)0;
    const COMPUTE_FLOAT4 qv1 = q_row_valid
        ? CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_base + (d_base + 4) * query_seq_len4))
        : (COMPUTE_FLOAT4)0;
    COMPUTE_FLOAT8 out = (COMPUTE_FLOAT8)0;

    const int key_base = (b * kv_head_num + kvh) * head_dim * key_max_len;
    const int value_base = (b * kv_head_num + kvh) * value_max_len * head_dim;
    const int linear_tid = q_lane * MQTILE_SPARSE_FLASH_HD128_DLANES + lane;
    for (int k_base = 0; k_base < active_kv_seq_len; k_base += MQTILE_SPARSE_FLASH_HD128_KTILE) {
        const int k_tile = min(MQTILE_SPARSE_FLASH_HD128_KTILE, active_kv_seq_len - k_base);
        const int k_tile_end = k_base + k_tile - 1;
        const int k_tile_full = (q_tile_min_logical >= 0 && k_tile_end <= q_tile_min_logical) ? 1 : 0;
        for (int load = linear_tid; load < k_tile * MQTILE_SPARSE_FLASH_HD128_DLANES;
             load += MQTILE_SPARSE_FLASH_HD128_QTILE * MQTILE_SPARSE_FLASH_HD128_DLANES) {
            const int k_local = load / MQTILE_SPARSE_FLASH_HD128_DLANES;
            const int d_lane = load - k_local * MQTILE_SPARSE_FLASH_HD128_DLANES;
            const int dim = d_lane << 3;
            const int k_index = k_base + k_local;
            local_k0[k_local * MQTILE_SPARSE_FLASH_HD128_DLANES + d_lane] =
                vload4(0, past_key + key_base + dim * key_max_len + k_index);
            local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_DLANES + d_lane] =
                vload4(0, past_key + key_base + (dim + 4) * key_max_len + k_index);
            local_v[k_local * MQTILE_SPARSE_FLASH_HD128_DLANES + d_lane] =
                vload8(0, past_value + value_base + k_index * head_dim + dim);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_local = 0; k_local < MQTILE_SPARSE_FLASH_HD128_KTILE; ++k_local) {
            const int partial_index = ((q_lane * MQTILE_SPARSE_FLASH_HD128_KTILE + k_local) *
                                       MQTILE_SPARSE_FLASH_HD128_DLANES) + lane;
            COMPUTE_FLOAT partial = (COMPUTE_FLOAT)0;
            if (q_row_valid && k_local < k_tile) {
                const int k_index = k_base + k_local;
                const int k_visible = k_tile_full || k_index <= q_logical;
                if (k_visible) {
                    const COMPUTE_FLOAT4 kv0 = CONVERT_COMPUTE_FLOAT4(
                        local_k0[k_local * MQTILE_SPARSE_FLASH_HD128_DLANES + lane]);
                    const COMPUTE_FLOAT4 kv1 = CONVERT_COMPUTE_FLOAT4(
                        local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_DLANES + lane]);
                    partial = dot(qv0, kv0) + dot(qv1, kv1);
                }
            }
            partial_tile[partial_index] = partial;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < MQTILE_SPARSE_FLASH_HD128_KTILE) {
            COMPUTE_FLOAT score = (COMPUTE_FLOAT)-FLT_MAX;
            if (q_row_valid && lane < k_tile) {
                const int k_index = k_base + lane;
                const int score_index = q_lane * MQTILE_SPARSE_FLASH_HD128_KTILE + lane;
                const int partial_base = score_index * MQTILE_SPARSE_FLASH_HD128_DLANES;
                if (k_tile_full || k_index <= q_logical) {
                    score = (COMPUTE_FLOAT)0;
                    for (int d_lane = 0; d_lane < MQTILE_SPARSE_FLASH_HD128_DLANES; ++d_lane) {
                        score += partial_tile[partial_base + d_lane];
                    }
                    score *= (COMPUTE_FLOAT)scale;
                }
            }
            score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_KTILE + lane] = score;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            COMPUTE_FLOAT tile_max = (COMPUTE_FLOAT)-FLT_MAX;
            if (q_row_valid) {
                for (int k_local = 0; k_local < k_tile; ++k_local) {
                    tile_max = fmax(tile_max,
                                    score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_KTILE + k_local]);
                }
                const COMPUTE_FLOAT merged_m = fmax(row_m[q_lane], tile_max);
                row_alpha[q_lane] = row_l[q_lane] > (COMPUTE_FLOAT)0
                    ? exp(row_m[q_lane] - merged_m)
                    : (COMPUTE_FLOAT)0;
                row_new_m[q_lane] = merged_m;
            } else {
                row_alpha[q_lane] = (COMPUTE_FLOAT)0;
                row_new_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < MQTILE_SPARSE_FLASH_HD128_KTILE) {
            COMPUTE_FLOAT beta = (COMPUTE_FLOAT)0;
            if (q_row_valid && lane < k_tile) {
                beta = exp(score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_KTILE + lane] - row_new_m[q_lane]);
            }
            beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_KTILE + lane] = beta;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            COMPUTE_FLOAT tile_sum = (COMPUTE_FLOAT)0;
            if (q_row_valid) {
                for (int k_local = 0; k_local < k_tile; ++k_local) {
                    tile_sum += beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_KTILE + k_local];
                }
                row_l[q_lane] = row_l[q_lane] * row_alpha[q_lane] + tile_sum;
                row_m[q_lane] = row_new_m[q_lane];
            } else {
                row_l[q_lane] = (COMPUTE_FLOAT)0;
                row_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (q_row_valid) {
            out *= (COMPUTE_FLOAT8)row_alpha[q_lane];
            for (int k_local = 0; k_local < k_tile; ++k_local) {
                out += (COMPUTE_FLOAT8)beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_KTILE + k_local] *
                       CONVERT_COMPUTE_FLOAT8(local_v[k_local * MQTILE_SPARSE_FLASH_HD128_DLANES + lane]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (q_row_valid) {
        const COMPUTE_FLOAT denom = row_l[q_lane] > (COMPUTE_FLOAT)0 ? row_l[q_lane] : (COMPUTE_FLOAT)1;
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim + d_base;
        vstore8(CONVERT_FLOAT8(out / (COMPUTE_FLOAT8)denom), 0, output + output_offset);
    }
}

#define MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE 4
#define MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE 8
#define MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES 16

__kernel void mqtile_sparse_flash_hd128_q4k8(GLOBAL_SIZE_3_DIMS
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
    const int x = get_global_id(0); // fixed 16 dim lanes
    const int y = get_global_id(1); // q piece token
    int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lane = get_local_id(0);
    const int q_lane = get_local_id(1);
    if (get_local_size(0) != MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES ||
        get_local_size(1) != MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE ||
        lane >= MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES ||
        q_lane >= MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE ||
        head_dim != 128) {
        return;
    }

    const int q_tile_base = y - q_lane;
    const int q_tile_valid_count = clamp(q_piece_len - q_tile_base, 0, MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE);
    if (q_tile_valid_count <= 0) {
        return;
    }
    const int q_index = q_start + y;
    const int q_valid = (q_lane < q_tile_valid_count && y < q_piece_len && q_index < output_seq_len) ? 1 : 0;

    const int b = z / head_num;
    const int h = z - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }

    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int q_logical = q_valid ? sparse_query[q_index] : 0;
    const int q_row = query_rows_are_full ? q_logical : q_index;
    const int q_row_valid = q_valid && q_row >= 0 && q_row < query_seq_len;
    int local q_tile_logical[MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE];
    int local q_tile_min_logical;
    int local q_tile_max_logical;
    if (lane == 0) {
        q_tile_logical[q_lane] = q_row_valid ? q_logical : -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0 && q_lane == 0) {
        int tile_min_logical = key_seq_len;
        int tile_max_logical = -1;
        for (int qi = 0; qi < MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE; ++qi) {
            const int logical = q_tile_logical[qi];
            if (logical < 0) {
                continue;
            }
            tile_min_logical = min(tile_min_logical, logical);
            tile_max_logical = max(tile_max_logical, logical);
        }
        q_tile_min_logical = tile_max_logical >= 0 ? tile_min_logical : -1;
        q_tile_max_logical = tile_max_logical;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int active_kv_seq_len = q_tile_max_logical >= 0 ? clamp(q_tile_max_logical + 1, 0, key_seq_len) : 0;

    COMPUTE_FLOAT local row_m[MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE];
    COMPUTE_FLOAT local row_l[MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE];
    COMPUTE_FLOAT local row_alpha[MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE];
    COMPUTE_FLOAT local row_new_m[MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE];
    COMPUTE_FLOAT local score_tile[MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE];
    COMPUTE_FLOAT local beta_tile[MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE];
    COMPUTE_FLOAT local partial_tile[MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE *
                                     MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE *
                                     MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES];
    FLOAT4 local local_k0[MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES];
    FLOAT4 local local_k1[MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES];
    FLOAT8 local local_v[MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES];

    if (lane == 0) {
        row_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
        row_l[q_lane] = (COMPUTE_FLOAT)0;
        row_alpha[q_lane] = (COMPUTE_FLOAT)0;
        row_new_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int d_base = lane << 3;
    const int query_base = z * head_dim * query_seq_len4 + q_row;
    const COMPUTE_FLOAT4 qv0 = q_row_valid
        ? CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_base + d_base * query_seq_len4))
        : (COMPUTE_FLOAT4)0;
    const COMPUTE_FLOAT4 qv1 = q_row_valid
        ? CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_base + (d_base + 4) * query_seq_len4))
        : (COMPUTE_FLOAT4)0;
    COMPUTE_FLOAT8 out = (COMPUTE_FLOAT8)0;

    const int key_base = (b * kv_head_num + kvh) * head_dim * key_max_len;
    const int value_base = (b * kv_head_num + kvh) * value_max_len * head_dim;
    const int linear_tid = q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + lane;
    for (int k_base = 0; k_base < active_kv_seq_len; k_base += MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE) {
        const int k_tile = min(MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE, active_kv_seq_len - k_base);
        const int k_tile_end = k_base + k_tile - 1;
        const int k_tile_full = (q_tile_min_logical >= 0 && k_tile_end <= q_tile_min_logical) ? 1 : 0;
        for (int load = linear_tid; load < k_tile * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES;
             load += MQTILE_SPARSE_FLASH_HD128_Q4K8_QTILE * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES) {
            const int k_local = load / MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES;
            const int d_lane = load - k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES;
            const int dim = d_lane << 3;
            const int k_index = k_base + k_local;
            local_k0[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + d_lane] =
                vload4(0, past_key + key_base + dim * key_max_len + k_index);
            local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + d_lane] =
                vload4(0, past_key + key_base + (dim + 4) * key_max_len + k_index);
            local_v[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + d_lane] =
                vload8(0, past_value + value_base + k_index * head_dim + dim);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_local = 0; k_local < MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE; ++k_local) {
            const int partial_index = ((q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE + k_local) *
                                       MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES) + lane;
            COMPUTE_FLOAT partial = (COMPUTE_FLOAT)0;
            if (q_row_valid && k_local < k_tile) {
                const int k_index = k_base + k_local;
                const int k_visible = k_tile_full || k_index <= q_logical;
                if (k_visible) {
                    const COMPUTE_FLOAT4 kv0 = CONVERT_COMPUTE_FLOAT4(
                        local_k0[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + lane]);
                    const COMPUTE_FLOAT4 kv1 = CONVERT_COMPUTE_FLOAT4(
                        local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + lane]);
                    partial = dot(qv0, kv0) + dot(qv1, kv1);
                }
            }
            partial_tile[partial_index] = partial;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE) {
            COMPUTE_FLOAT score = (COMPUTE_FLOAT)-FLT_MAX;
            if (q_row_valid && lane < k_tile) {
                const int k_index = k_base + lane;
                const int score_index = q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE + lane;
                const int partial_base = score_index * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES;
                if (k_tile_full || k_index <= q_logical) {
                    score = (COMPUTE_FLOAT)0;
                    for (int d_lane = 0; d_lane < MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES; ++d_lane) {
                        score += partial_tile[partial_base + d_lane];
                    }
                    score *= (COMPUTE_FLOAT)scale;
                }
            }
            score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE + lane] = score;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            COMPUTE_FLOAT tile_max = (COMPUTE_FLOAT)-FLT_MAX;
            if (q_row_valid) {
                for (int k_local = 0; k_local < k_tile; ++k_local) {
                    tile_max = fmax(tile_max,
                                    score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE + k_local]);
                }
                const COMPUTE_FLOAT merged_m = fmax(row_m[q_lane], tile_max);
                row_alpha[q_lane] = row_l[q_lane] > (COMPUTE_FLOAT)0
                    ? exp(row_m[q_lane] - merged_m)
                    : (COMPUTE_FLOAT)0;
                row_new_m[q_lane] = merged_m;
            } else {
                row_alpha[q_lane] = (COMPUTE_FLOAT)0;
                row_new_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE) {
            COMPUTE_FLOAT beta = (COMPUTE_FLOAT)0;
            if (q_row_valid && lane < k_tile) {
                beta = exp(score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE + lane] - row_new_m[q_lane]);
            }
            beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE + lane] = beta;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            COMPUTE_FLOAT tile_sum = (COMPUTE_FLOAT)0;
            if (q_row_valid) {
                for (int k_local = 0; k_local < k_tile; ++k_local) {
                    tile_sum += beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE + k_local];
                }
                row_l[q_lane] = row_l[q_lane] * row_alpha[q_lane] + tile_sum;
                row_m[q_lane] = row_new_m[q_lane];
            } else {
                row_l[q_lane] = (COMPUTE_FLOAT)0;
                row_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (q_row_valid) {
            out *= (COMPUTE_FLOAT8)row_alpha[q_lane];
            for (int k_local = 0; k_local < k_tile; ++k_local) {
                out += (COMPUTE_FLOAT8)beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q4K8_KTILE + k_local] *
                       CONVERT_COMPUTE_FLOAT8(local_v[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + lane]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (q_row_valid) {
        const COMPUTE_FLOAT denom = row_l[q_lane] > (COMPUTE_FLOAT)0 ? row_l[q_lane] : (COMPUTE_FLOAT)1;
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim + d_base;
        vstore8(CONVERT_FLOAT8(out / (COMPUTE_FLOAT8)denom), 0, output + output_offset);
    }
}

#define MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE 8
#define MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE 16
#define MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES 16

__kernel void mqtile_sparse_flash_hd128_q8k16(GLOBAL_SIZE_3_DIMS
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
    const int x = get_global_id(0); // fixed 16 dim lanes
    const int y = get_global_id(1); // q piece token
    int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lane = get_local_id(0);
    const int q_lane = get_local_id(1);
    if (get_local_size(0) != MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES ||
        get_local_size(1) != MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE ||
        lane >= MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES ||
        q_lane >= MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE ||
        head_dim != 128) {
        return;
    }

    const int q_tile_base = y - q_lane;
    const int q_tile_valid_count = clamp(q_piece_len - q_tile_base, 0, MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE);
    if (q_tile_valid_count <= 0) {
        return;
    }
    const int q_index = q_start + y;
    const int q_valid = (q_lane < q_tile_valid_count && y < q_piece_len && q_index < output_seq_len) ? 1 : 0;

    const int b = z / head_num;
    const int h = z - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }

    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int q_logical = q_valid ? sparse_query[q_index] : 0;
    const int q_row = query_rows_are_full ? q_logical : q_index;
    const int q_row_valid = q_valid && q_row >= 0 && q_row < query_seq_len;
    int local q_tile_logical[MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE];
    int local q_tile_min_logical;
    int local q_tile_max_logical;
    if (lane == 0) {
        q_tile_logical[q_lane] = q_row_valid ? q_logical : -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0 && q_lane == 0) {
        int tile_min_logical = key_seq_len;
        int tile_max_logical = -1;
        for (int qi = 0; qi < MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE; ++qi) {
            const int logical = q_tile_logical[qi];
            if (logical < 0) {
                continue;
            }
            tile_min_logical = min(tile_min_logical, logical);
            tile_max_logical = max(tile_max_logical, logical);
        }
        q_tile_min_logical = tile_max_logical >= 0 ? tile_min_logical : -1;
        q_tile_max_logical = tile_max_logical;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int active_kv_seq_len = q_tile_max_logical >= 0 ? clamp(q_tile_max_logical + 1, 0, key_seq_len) : 0;

    COMPUTE_FLOAT local row_m[MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE];
    COMPUTE_FLOAT local row_l[MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE];
    COMPUTE_FLOAT local row_alpha[MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE];
    COMPUTE_FLOAT local row_new_m[MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE];
    COMPUTE_FLOAT local score_tile[MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE];
    COMPUTE_FLOAT local beta_tile[MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE];
    COMPUTE_FLOAT local partial_tile[MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE *
                                     MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE *
                                     MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES];
    FLOAT4 local local_k0[MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES];
    FLOAT4 local local_k1[MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES];
    FLOAT8 local local_v[MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES];

    if (lane == 0) {
        row_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
        row_l[q_lane] = (COMPUTE_FLOAT)0;
        row_alpha[q_lane] = (COMPUTE_FLOAT)0;
        row_new_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int d_base = lane << 3;
    const int query_base = z * head_dim * query_seq_len4 + q_row;
    const COMPUTE_FLOAT4 qv0 = q_row_valid
        ? CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_base + d_base * query_seq_len4))
        : (COMPUTE_FLOAT4)0;
    const COMPUTE_FLOAT4 qv1 = q_row_valid
        ? CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_base + (d_base + 4) * query_seq_len4))
        : (COMPUTE_FLOAT4)0;
    COMPUTE_FLOAT8 out = (COMPUTE_FLOAT8)0;

    const int key_base = (b * kv_head_num + kvh) * head_dim * key_max_len;
    const int value_base = (b * kv_head_num + kvh) * value_max_len * head_dim;
    const int linear_tid = q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + lane;
    for (int k_base = 0; k_base < active_kv_seq_len; k_base += MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE) {
        const int k_tile = min(MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE, active_kv_seq_len - k_base);
        const int k_tile_end = k_base + k_tile - 1;
        const int k_tile_full = (q_tile_min_logical >= 0 && k_tile_end <= q_tile_min_logical) ? 1 : 0;
        for (int load = linear_tid; load < k_tile * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES;
             load += MQTILE_SPARSE_FLASH_HD128_Q8K16_QTILE * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES) {
            const int k_local = load / MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES;
            const int d_lane = load - k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES;
            const int dim = d_lane << 3;
            const int k_index = k_base + k_local;
            local_k0[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + d_lane] =
                vload4(0, past_key + key_base + dim * key_max_len + k_index);
            local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + d_lane] =
                vload4(0, past_key + key_base + (dim + 4) * key_max_len + k_index);
            local_v[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + d_lane] =
                vload8(0, past_value + value_base + k_index * head_dim + dim);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k_local = 0; k_local < MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE; ++k_local) {
            const int partial_index = ((q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE + k_local) *
                                       MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES) + lane;
            COMPUTE_FLOAT partial = (COMPUTE_FLOAT)0;
            if (q_row_valid && k_local < k_tile) {
                const int k_index = k_base + k_local;
                const int k_visible = k_tile_full || k_index <= q_logical;
                if (k_visible) {
                    const COMPUTE_FLOAT4 kv0 = CONVERT_COMPUTE_FLOAT4(
                        local_k0[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + lane]);
                    const COMPUTE_FLOAT4 kv1 = CONVERT_COMPUTE_FLOAT4(
                        local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + lane]);
                    partial = dot(qv0, kv0) + dot(qv1, kv1);
                }
            }
            partial_tile[partial_index] = partial;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE) {
            COMPUTE_FLOAT score = (COMPUTE_FLOAT)-FLT_MAX;
            if (q_row_valid && lane < k_tile) {
                const int k_index = k_base + lane;
                const int score_index = q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE + lane;
                const int partial_base = score_index * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES;
                if (k_tile_full || k_index <= q_logical) {
                    score = (COMPUTE_FLOAT)0;
                    for (int d_lane = 0; d_lane < MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES; ++d_lane) {
                        score += partial_tile[partial_base + d_lane];
                    }
                    score *= (COMPUTE_FLOAT)scale;
                }
            }
            score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE + lane] = score;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            COMPUTE_FLOAT tile_max = (COMPUTE_FLOAT)-FLT_MAX;
            if (q_row_valid) {
                for (int k_local = 0; k_local < k_tile; ++k_local) {
                    tile_max = fmax(tile_max,
                                    score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE + k_local]);
                }
                const COMPUTE_FLOAT merged_m = fmax(row_m[q_lane], tile_max);
                row_alpha[q_lane] = row_l[q_lane] > (COMPUTE_FLOAT)0
                    ? exp(row_m[q_lane] - merged_m)
                    : (COMPUTE_FLOAT)0;
                row_new_m[q_lane] = merged_m;
            } else {
                row_alpha[q_lane] = (COMPUTE_FLOAT)0;
                row_new_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane < MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE) {
            COMPUTE_FLOAT beta = (COMPUTE_FLOAT)0;
            if (q_row_valid && lane < k_tile) {
                beta = exp(score_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE + lane] - row_new_m[q_lane]);
            }
            beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE + lane] = beta;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            COMPUTE_FLOAT tile_sum = (COMPUTE_FLOAT)0;
            if (q_row_valid) {
                for (int k_local = 0; k_local < k_tile; ++k_local) {
                    tile_sum += beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE + k_local];
                }
                row_l[q_lane] = row_l[q_lane] * row_alpha[q_lane] + tile_sum;
                row_m[q_lane] = row_new_m[q_lane];
            } else {
                row_l[q_lane] = (COMPUTE_FLOAT)0;
                row_m[q_lane] = (COMPUTE_FLOAT)-FLT_MAX;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (q_row_valid) {
            out *= (COMPUTE_FLOAT8)row_alpha[q_lane];
            for (int k_local = 0; k_local < k_tile; ++k_local) {
                out += (COMPUTE_FLOAT8)beta_tile[q_lane * MQTILE_SPARSE_FLASH_HD128_Q8K16_KTILE + k_local] *
                       CONVERT_COMPUTE_FLOAT8(local_v[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + lane]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (q_row_valid) {
        const COMPUTE_FLOAT denom = row_l[q_lane] > (COMPUTE_FLOAT)0 ? row_l[q_lane] : (COMPUTE_FLOAT)1;
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim + d_base;
        vstore8(CONVERT_FLOAT8(out / (COMPUTE_FLOAT8)denom), 0, output + output_offset);
    }
}
