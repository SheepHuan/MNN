#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void zero_output(
    __global FLOAT* output,
    const int total) {
    int index = get_global_id(0);
    if (index >= total) {
        return;
    }
    output[index] = (FLOAT)0;
}

__kernel void copy_paged_kv(
    __global const FLOAT* key,
    __global const FLOAT* value,
    __global FLOAT* key_cache,
    __global FLOAT* value_cache,
    __global const int* slot_table,
    __global const int* sparse_query,
    const int batch,
    const int new_kv_len,
    const int kv_write_len,
    const int kv_heads,
    const int head_dim,
    const int base_logical,
    const int max_slots,
    const int sparse_active,
    const int total) {
    int index = get_global_id(0);
    if (index >= total) {
        return;
    }
    int d = index % head_dim;
    int t = index / head_dim;
    int h = t % kv_heads;
    t = t / kv_heads;
    int l = t % kv_write_len;
    int b = t / kv_write_len;
    int logical = sparse_active ? sparse_query[l] : (base_logical + l);
    if (logical < 0) {
        return;
    }
    int slot = slot_table[logical];
    if (slot < 0 || slot >= max_slots || l >= new_kv_len) {
        return;
    }
    int in_offset = ((b * new_kv_len + l) * kv_heads + h) * head_dim + d;
    int k_offset = ((slot * batch + b) * kv_heads + h) * head_dim + d;
    int v_offset = ((b * kv_heads + h) * max_slots + slot) * head_dim + d;
    key_cache[k_offset] = key[in_offset];
    value_cache[v_offset] = value[in_offset];
}

__kernel void pack_paged_kv_prefill(GLOBAL_SIZE_3_DIMS
    __global const FLOAT* key_cache,      // [max_slots, batch, kv_heads, head_dim]
    __global const FLOAT* value_cache,    // [batch, kv_heads, max_slots, head_dim]
    __global FLOAT* packed_key,           // [batch * kv_heads, head_dim_pack, kv_len_pack]
    __global FLOAT* packed_value,         // [batch * kv_heads, kv_len_pack, head_dim_pack]
    __global const int* slot_table,
    const int batch,
    const int kv_len,
    const int kv_heads,
    const int head_dim,
    const int max_slots) {
    const int x = get_global_id(0); // kv token / 4
    const int y = get_global_id(1); // head dim / 4
    int z = get_global_id(2);       // batch * kv_heads
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int logical4 = x << 2;
    const int dim4 = y << 2;
    const int head_dim_pack = ((head_dim + 3) / 4) * 4;
    const int kv_len_pack = ((kv_len + 3) / 4) * 4;
    const int b = z / kv_heads;
    const int h = z - b * kv_heads;

    FLOAT4 k0 = (FLOAT4)0;
    FLOAT4 k1 = (FLOAT4)0;
    FLOAT4 k2 = (FLOAT4)0;
    FLOAT4 k3 = (FLOAT4)0;
    FLOAT4 v0 = (FLOAT4)0;
    FLOAT4 v1 = (FLOAT4)0;
    FLOAT4 v2 = (FLOAT4)0;
    FLOAT4 v3 = (FLOAT4)0;

    if (dim4 < head_dim) {
        int slot0 = logical4 < kv_len ? slot_table[logical4] : -1;
        int slot1 = logical4 + 1 < kv_len ? slot_table[logical4 + 1] : -1;
        int slot2 = logical4 + 2 < kv_len ? slot_table[logical4 + 2] : -1;
        int slot3 = logical4 + 3 < kv_len ? slot_table[logical4 + 3] : -1;
        if (slot0 >= 0 && slot0 < max_slots) {
            k0 = vload4(0, key_cache + ((slot0 * batch + b) * kv_heads + h) * head_dim + dim4);
            v0 = vload4(0, value_cache + ((b * kv_heads + h) * max_slots + slot0) * head_dim + dim4);
        }
        if (slot1 >= 0 && slot1 < max_slots) {
            k1 = vload4(0, key_cache + ((slot1 * batch + b) * kv_heads + h) * head_dim + dim4);
            v1 = vload4(0, value_cache + ((b * kv_heads + h) * max_slots + slot1) * head_dim + dim4);
        }
        if (slot2 >= 0 && slot2 < max_slots) {
            k2 = vload4(0, key_cache + ((slot2 * batch + b) * kv_heads + h) * head_dim + dim4);
            v2 = vload4(0, value_cache + ((b * kv_heads + h) * max_slots + slot2) * head_dim + dim4);
        }
        if (slot3 >= 0 && slot3 < max_slots) {
            k3 = vload4(0, key_cache + ((slot3 * batch + b) * kv_heads + h) * head_dim + dim4);
            v3 = vload4(0, value_cache + ((b * kv_heads + h) * max_slots + slot3) * head_dim + dim4);
        }
        if (dim4 + 3 >= head_dim) {
            if (dim4 + 1 >= head_dim) {
                k0.yzw = (FLOAT3)0; k1.yzw = (FLOAT3)0; k2.yzw = (FLOAT3)0; k3.yzw = (FLOAT3)0;
                v0.yzw = (FLOAT3)0; v1.yzw = (FLOAT3)0; v2.yzw = (FLOAT3)0; v3.yzw = (FLOAT3)0;
            } else if (dim4 + 2 >= head_dim) {
                k0.zw = (FLOAT2)0; k1.zw = (FLOAT2)0; k2.zw = (FLOAT2)0; k3.zw = (FLOAT2)0;
                v0.zw = (FLOAT2)0; v1.zw = (FLOAT2)0; v2.zw = (FLOAT2)0; v3.zw = (FLOAT2)0;
            } else {
                k0.w = (FLOAT)0; k1.w = (FLOAT)0; k2.w = (FLOAT)0; k3.w = (FLOAT)0;
                v0.w = (FLOAT)0; v1.w = (FLOAT)0; v2.w = (FLOAT)0; v3.w = (FLOAT)0;
            }
        }
    }

    const int key_offset = (z * head_dim_pack + dim4) * kv_len_pack + logical4;
    vstore4((FLOAT4)(k0.s0, k1.s0, k2.s0, k3.s0), 0, packed_key + key_offset);
    vstore4((FLOAT4)(k0.s1, k1.s1, k2.s1, k3.s1), 0, packed_key + key_offset + kv_len_pack);
    vstore4((FLOAT4)(k0.s2, k1.s2, k2.s2, k3.s2), 0, packed_key + key_offset + kv_len_pack * 2);
    vstore4((FLOAT4)(k0.s3, k1.s3, k2.s3, k3.s3), 0, packed_key + key_offset + kv_len_pack * 3);

    const int value_offset = (z * kv_len_pack + logical4) * head_dim_pack + dim4;
    vstore4(v0, 0, packed_value + value_offset);
    vstore4(v1, 0, packed_value + value_offset + head_dim_pack);
    vstore4(v2, 0, packed_value + value_offset + head_dim_pack * 2);
    vstore4(v3, 0, packed_value + value_offset + head_dim_pack * 3);
}

__kernel void rearrange_paged_q_gemm_prefill(GLOBAL_SIZE_3_DIMS
    __global const FLOAT* query,          // [batch, query_seq_len, head_num, head_dim]
    __global FLOAT* packed_query,         // [batch * head_num, head_dim_pack, query_seq_len_pack]
    const int seq_len,
    const int head_dim,
    const int head_num,
    const int seq_len_pack) {
    const int x = get_global_id(0); // query token / 4
    const int y = get_global_id(1); // head dim / 4
    int z = get_global_id(2);       // batch * head_num
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int x4 = x << 2;
    const int y4 = y << 2;
    const int b = z / head_num;
    const int h = z - b * head_num;
    FLOAT4 q0 = (FLOAT4)0;
    FLOAT4 q1 = (FLOAT4)0;
    FLOAT4 q2 = (FLOAT4)0;
    FLOAT4 q3 = (FLOAT4)0;
    if (x4 < seq_len && y4 < head_dim) {
        const int stride = head_num * head_dim;
        int query_offset = ((b * seq_len + x4) * head_num + h) * head_dim + y4;
        q0 = vload4(0, query + query_offset);
        q1 = x4 + 1 < seq_len ? vload4(0, query + query_offset + stride) : (FLOAT4)0;
        q2 = x4 + 2 < seq_len ? vload4(0, query + query_offset + stride * 2) : (FLOAT4)0;
        q3 = x4 + 3 < seq_len ? vload4(0, query + query_offset + stride * 3) : (FLOAT4)0;
        if (y4 + 3 >= head_dim) {
            if (y4 + 1 >= head_dim) {
                q0.yzw = (FLOAT3)0; q1.yzw = (FLOAT3)0; q2.yzw = (FLOAT3)0; q3.yzw = (FLOAT3)0;
            } else if (y4 + 2 >= head_dim) {
                q0.zw = (FLOAT2)0; q1.zw = (FLOAT2)0; q2.zw = (FLOAT2)0; q3.zw = (FLOAT2)0;
            } else {
                q0.w = (FLOAT)0; q1.w = (FLOAT)0; q2.w = (FLOAT)0; q3.w = (FLOAT)0;
            }
        }
    }
    const int out_offset = ((z * head_dim + y4) * seq_len_pack + x4);
    vstore4((FLOAT4)(q0.s0, q1.s0, q2.s0, q3.s0), 0, packed_query + out_offset);
    vstore4((FLOAT4)(q0.s1, q1.s1, q2.s1, q3.s1), 0, packed_query + out_offset + seq_len_pack);
    vstore4((FLOAT4)(q0.s2, q1.s2, q2.s2, q3.s2), 0, packed_query + out_offset + seq_len_pack * 2);
    vstore4((FLOAT4)(q0.s3, q1.s3, q2.s3, q3.s3), 0, packed_query + out_offset + seq_len_pack * 3);
}

__kernel void pack_paged_kv_prefill_gemm(GLOBAL_SIZE_3_DIMS
    __global const FLOAT* key_cache,      // [max_slots, batch, kv_heads, head_dim]
    __global const FLOAT* value_cache,    // [batch, kv_heads, max_slots, head_dim]
    __global FLOAT* packed_key,           // [batch * kv_heads, head_dim_pack, kv_len_pack]
    __global FLOAT* packed_value,         // [batch * kv_heads, kv_len_pack, head_dim_pack]
    __global const int* slot_table,
    const int batch,
    const int kv_len,
    const int kv_heads,
    const int head_dim,
    const int max_slots,
    const int kv_len_pack,
    const int head_dim_pack) {
    const int x = get_global_id(0); // kv token / 4
    const int y = get_global_id(1); // head dim / 4
    int z = get_global_id(2);       // batch * kv_heads
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int logical4 = x << 2;
    const int dim4 = y << 2;
    const int b = z / kv_heads;
    const int h = z - b * kv_heads;
    FLOAT4 k0 = (FLOAT4)0;
    FLOAT4 k1 = (FLOAT4)0;
    FLOAT4 k2 = (FLOAT4)0;
    FLOAT4 k3 = (FLOAT4)0;
    FLOAT4 v0 = (FLOAT4)0;
    FLOAT4 v1 = (FLOAT4)0;
    FLOAT4 v2 = (FLOAT4)0;
    FLOAT4 v3 = (FLOAT4)0;

    if (dim4 < head_dim) {
        int slot0 = logical4 < kv_len ? slot_table[logical4] : -1;
        int slot1 = logical4 + 1 < kv_len ? slot_table[logical4 + 1] : -1;
        int slot2 = logical4 + 2 < kv_len ? slot_table[logical4 + 2] : -1;
        int slot3 = logical4 + 3 < kv_len ? slot_table[logical4 + 3] : -1;
        if (slot0 >= 0 && slot0 < max_slots) {
            k0 = vload4(0, key_cache + ((slot0 * batch + b) * kv_heads + h) * head_dim + dim4);
            v0 = vload4(0, value_cache + ((b * kv_heads + h) * max_slots + slot0) * head_dim + dim4);
        }
        if (slot1 >= 0 && slot1 < max_slots) {
            k1 = vload4(0, key_cache + ((slot1 * batch + b) * kv_heads + h) * head_dim + dim4);
            v1 = vload4(0, value_cache + ((b * kv_heads + h) * max_slots + slot1) * head_dim + dim4);
        }
        if (slot2 >= 0 && slot2 < max_slots) {
            k2 = vload4(0, key_cache + ((slot2 * batch + b) * kv_heads + h) * head_dim + dim4);
            v2 = vload4(0, value_cache + ((b * kv_heads + h) * max_slots + slot2) * head_dim + dim4);
        }
        if (slot3 >= 0 && slot3 < max_slots) {
            k3 = vload4(0, key_cache + ((slot3 * batch + b) * kv_heads + h) * head_dim + dim4);
            v3 = vload4(0, value_cache + ((b * kv_heads + h) * max_slots + slot3) * head_dim + dim4);
        }
        if (dim4 + 3 >= head_dim) {
            if (dim4 + 1 >= head_dim) {
                k0.yzw = (FLOAT3)0; k1.yzw = (FLOAT3)0; k2.yzw = (FLOAT3)0; k3.yzw = (FLOAT3)0;
                v0.yzw = (FLOAT3)0; v1.yzw = (FLOAT3)0; v2.yzw = (FLOAT3)0; v3.yzw = (FLOAT3)0;
            } else if (dim4 + 2 >= head_dim) {
                k0.zw = (FLOAT2)0; k1.zw = (FLOAT2)0; k2.zw = (FLOAT2)0; k3.zw = (FLOAT2)0;
                v0.zw = (FLOAT2)0; v1.zw = (FLOAT2)0; v2.zw = (FLOAT2)0; v3.zw = (FLOAT2)0;
            } else {
                k0.w = (FLOAT)0; k1.w = (FLOAT)0; k2.w = (FLOAT)0; k3.w = (FLOAT)0;
                v0.w = (FLOAT)0; v1.w = (FLOAT)0; v2.w = (FLOAT)0; v3.w = (FLOAT)0;
            }
        }
    }

    const int key_offset = (z * head_dim_pack + dim4) * kv_len_pack + logical4;
    vstore4((FLOAT4)(k0.s0, k1.s0, k2.s0, k3.s0), 0, packed_key + key_offset);
    vstore4((FLOAT4)(k0.s1, k1.s1, k2.s1, k3.s1), 0, packed_key + key_offset + kv_len_pack);
    vstore4((FLOAT4)(k0.s2, k1.s2, k2.s2, k3.s2), 0, packed_key + key_offset + kv_len_pack * 2);
    vstore4((FLOAT4)(k0.s3, k1.s3, k2.s3, k3.s3), 0, packed_key + key_offset + kv_len_pack * 3);

    const int value_offset = (z * kv_len_pack + logical4) * head_dim_pack + dim4;
    vstore4(v0, 0, packed_value + value_offset);
    vstore4(v1, 0, packed_value + value_offset + head_dim_pack);
    vstore4(v2, 0, packed_value + value_offset + head_dim_pack * 2);
    vstore4(v3, 0, packed_value + value_offset + head_dim_pack * 3);
}

__kernel void pack_paged_k_prefill(GLOBAL_SIZE_3_DIMS
    __global const FLOAT* key_cache,      // [max_slots, batch, kv_heads, head_dim]
    __global FLOAT* packed_key,           // [batch * kv_heads, head_dim_pack, kv_len_pack]
    __global const int* slot_table,
    const int batch,
    const int kv_len,
    const int kv_heads,
    const int head_dim,
    const int max_slots) {
    const int x = get_global_id(0); // kv token / 4
    const int y = get_global_id(1); // head dim / 4
    int z = get_global_id(2);       // batch * kv_heads
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int logical4 = x << 2;
    const int dim4 = y << 2;
    const int head_dim_pack = ((head_dim + 3) / 4) * 4;
    const int kv_len_pack = ((kv_len + 3) / 4) * 4;
    const int b = z / kv_heads;
    const int h = z - b * kv_heads;

    FLOAT4 k0 = (FLOAT4)0;
    FLOAT4 k1 = (FLOAT4)0;
    FLOAT4 k2 = (FLOAT4)0;
    FLOAT4 k3 = (FLOAT4)0;

    if (dim4 < head_dim) {
        int slot0 = logical4 < kv_len ? slot_table[logical4] : -1;
        int slot1 = logical4 + 1 < kv_len ? slot_table[logical4 + 1] : -1;
        int slot2 = logical4 + 2 < kv_len ? slot_table[logical4 + 2] : -1;
        int slot3 = logical4 + 3 < kv_len ? slot_table[logical4 + 3] : -1;
        if (slot0 >= 0 && slot0 < max_slots) {
            k0 = vload4(0, key_cache + ((slot0 * batch + b) * kv_heads + h) * head_dim + dim4);
        }
        if (slot1 >= 0 && slot1 < max_slots) {
            k1 = vload4(0, key_cache + ((slot1 * batch + b) * kv_heads + h) * head_dim + dim4);
        }
        if (slot2 >= 0 && slot2 < max_slots) {
            k2 = vload4(0, key_cache + ((slot2 * batch + b) * kv_heads + h) * head_dim + dim4);
        }
        if (slot3 >= 0 && slot3 < max_slots) {
            k3 = vload4(0, key_cache + ((slot3 * batch + b) * kv_heads + h) * head_dim + dim4);
        }
        if (dim4 + 3 >= head_dim) {
            if (dim4 + 1 >= head_dim) {
                k0.yzw = (FLOAT3)0; k1.yzw = (FLOAT3)0; k2.yzw = (FLOAT3)0; k3.yzw = (FLOAT3)0;
            } else if (dim4 + 2 >= head_dim) {
                k0.zw = (FLOAT2)0; k1.zw = (FLOAT2)0; k2.zw = (FLOAT2)0; k3.zw = (FLOAT2)0;
            } else {
                k0.w = (FLOAT)0; k1.w = (FLOAT)0; k2.w = (FLOAT)0; k3.w = (FLOAT)0;
            }
        }
    }

    const int key_offset = (z * head_dim_pack + dim4) * kv_len_pack + logical4;
    vstore4((FLOAT4)(k0.s0, k1.s0, k2.s0, k3.s0), 0, packed_key + key_offset);
    vstore4((FLOAT4)(k0.s1, k1.s1, k2.s1, k3.s1), 0, packed_key + key_offset + kv_len_pack);
    vstore4((FLOAT4)(k0.s2, k1.s2, k2.s2, k3.s2), 0, packed_key + key_offset + kv_len_pack * 2);
    vstore4((FLOAT4)(k0.s3, k1.s3, k2.s3, k3.s3), 0, packed_key + key_offset + kv_len_pack * 3);
}

static inline float paged_rope_inv_freq(
    const float theta,
    const int rope_type_llama3,
    const float factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const int old_context,
    const int pair_index,
    const int rope_dim) {
    float inv_freq = pow(theta, -((float)(2 * pair_index)) / (float)rope_dim);
    if (rope_type_llama3 == 0 || old_context <= 0 || factor == 1.0f || low_freq_factor == high_freq_factor) {
        return inv_freq;
    }
    const float two_pi = 6.28318530717958647692f;
    const float wavelen = two_pi / inv_freq;
    const float low_freq_wavelen = (float)old_context / low_freq_factor;
    const float high_freq_wavelen = (float)old_context / high_freq_factor;
    float scaled = wavelen > low_freq_wavelen ? inv_freq / factor : inv_freq;
    if (wavelen >= high_freq_wavelen && wavelen <= low_freq_wavelen) {
        const float smooth = ((float)old_context / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        scaled = (1.0f - smooth) * inv_freq / factor + smooth * inv_freq;
    }
    return scaled;
}

__kernel void pic_page_attention_hydrate_kv(
    __global const FLOAT* source_key,      // [token_count, batch, kv_heads, head_dim]
    __global const FLOAT* source_value,    // [batch, kv_heads, token_count, head_dim]
    __global FLOAT* key_cache,             // [max_slots, batch, kv_heads, head_dim]
    __global FLOAT* value_cache,           // [batch, kv_heads, max_slots, head_dim]
    __global const int* slot_table,
    const int batch,
    const int kv_heads,
    const int head_dim,
    const int max_slots,
    const int logical_start,
    const int token_count,
    const int key_source_logical_start,
    const int value_source_start,
    const int value_source_stride,
    const int hydrate_value,
    const int rope_dim_in,
    const float rope_theta,
    const int rope_type_llama3,
    const float rope_scaling_factor,
    const float rope_scaling_low_freq_factor,
    const float rope_scaling_high_freq_factor,
    const int rope_scaling_original_max_position_embeddings,
    const int max_position_embeddings,
    const float rope_attention_scaling,
    const int total) {
    int index = get_global_id(0);
    if (index >= total) {
        return;
    }
    int d = index % head_dim;
    int t = index / head_dim;
    int h = t % kv_heads;
    t = t / kv_heads;
    int b = t % batch;
    int local_index = t / batch;
    if (local_index >= token_count) {
        return;
    }
    int logical = logical_start + local_index;
    int slot = slot_table[logical];
    if (slot < 0 || slot >= max_slots) {
        return;
    }
    int source_token = key_source_logical_start + local_index;
    if (source_token < 0 || source_token >= token_count) {
        if (key_source_logical_start <= 0) {
            return;
        }
        if (source_token >= max_slots) {
            return;
        }
    }
    int value_source_token = value_source_start + local_index;
    if (value_source_token < 0 || value_source_token >= value_source_stride) {
        return;
    }
    int key_src_base = ((source_token * batch + b) * kv_heads + h) * head_dim;
    int key_dst_base = ((slot * batch + b) * kv_heads + h) * head_dim;
    int value_src = ((b * kv_heads + h) * value_source_stride + value_source_token) * head_dim + d;
    int value_dst = ((b * kv_heads + h) * max_slots + slot) * head_dim + d;
    int rope_dim = min(rope_dim_in > 0 ? rope_dim_in : head_dim, head_dim);
    rope_dim = (rope_dim / 2) * 2;
    int rope_half = rope_dim / 2;
    if (d < rope_half) {
        int pair = d;
        float inv_freq = paged_rope_inv_freq(
            rope_theta > 0.0f ? rope_theta : 10000.0f,
            rope_type_llama3,
            max(rope_scaling_factor, 1.0f),
            max(rope_scaling_low_freq_factor, 1.0e-6f),
            max(rope_scaling_high_freq_factor, 1.0e-6f),
            rope_scaling_original_max_position_embeddings > 0 ? rope_scaling_original_max_position_embeddings
                                                              : max_position_embeddings,
            pair,
            rope_dim);
        float angle = (float)logical * inv_freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = (float)source_key[key_src_base + pair];
        float x1 = (float)source_key[key_src_base + pair + rope_half];
        key_cache[key_dst_base + pair] = (FLOAT)((x0 * c - x1 * s) * rope_attention_scaling);
        key_cache[key_dst_base + pair + rope_half] = (FLOAT)((x1 * c + x0 * s) * rope_attention_scaling);
    } else if (d >= rope_dim) {
        key_cache[key_dst_base + d] = source_key[key_src_base + d];
    }
    if (hydrate_value != 0) {
        value_cache[value_dst] = source_value[value_src];
    }
}

__kernel void export_canonical_paged_key(
    __global const FLOAT* key_cache, // [max_slots, batch, kv_heads, head_dim]
    __global FLOAT* key_out,         // [kv_len, batch, kv_heads, head_dim]
    __global const int* slot_table,
    const int batch,
    const int kv_len,
    const int kv_heads,
    const int head_dim,
    const int max_slots,
    const int rope_dim_in,
    const float rope_theta,
    const int rope_type_llama3,
    const float rope_scaling_factor,
    const float rope_scaling_low_freq_factor,
    const float rope_scaling_high_freq_factor,
    const int rope_scaling_original_max_position_embeddings,
    const int max_position_embeddings,
    const float rope_attention_scaling,
    const int total) {
    int index = get_global_id(0);
    if (index >= total) {
        return;
    }
    int d = index % head_dim;
    int t = index / head_dim;
    int h = t % kv_heads;
    t = t / kv_heads;
    int b = t % batch;
    int logical = t / batch;
    if (logical >= kv_len) {
        return;
    }
    int slot = slot_table[logical];
    if (slot < 0 || slot >= max_slots) {
        return;
    }
    int src_base = ((slot * batch + b) * kv_heads + h) * head_dim;
    int dst = ((logical * batch + b) * kv_heads + h) * head_dim + d;
    float out = (float)key_cache[src_base + d];
    int rope_dim = min(rope_dim_in > 0 ? rope_dim_in : head_dim, head_dim);
    rope_dim = (rope_dim / 2) * 2;
    if (rope_dim > 0 && d < rope_dim) {
        int rope_half = rope_dim / 2;
        int pair = d < rope_half ? d : d - rope_half;
        float inv_freq = paged_rope_inv_freq(
            rope_theta > 0.0f ? rope_theta : 10000.0f,
            rope_type_llama3,
            max(rope_scaling_factor, 1.0f),
            max(rope_scaling_low_freq_factor, 1.0e-6f),
            max(rope_scaling_high_freq_factor, 1.0e-6f),
            rope_scaling_original_max_position_embeddings > 0 ? rope_scaling_original_max_position_embeddings
                                                              : max_position_embeddings,
            pair,
            rope_dim);
        float angle = (float)logical * inv_freq;
        float c = cos(angle);
        float s = sin(angle);
        float y0 = (float)key_cache[src_base + pair];
        float y1 = (float)key_cache[src_base + pair + rope_half];
        float inv_scale = rope_attention_scaling > 0.0f ? 1.0f / rope_attention_scaling : 1.0f;
        out = d < rope_half ? (y0 * c + y1 * s) * inv_scale : (y1 * c - y0 * s) * inv_scale;
    }
    key_out[dst] = (FLOAT)out;
}

__kernel void pic_cacheblend_value_score(
    __global const FLOAT* reference_value_cache, // [batch, kv_heads, max_slots, head_dim]
    __global const FLOAT* cached_value_cache,    // [batch, kv_heads, cached_max_slots, head_dim]
    __global const int* slot_table,
    __global float* scores,
    const int batch,
    const int kv_heads,
    const int head_dim,
    const int max_slots,
    const int cached_max_slots,
    const int cached_slot_start,
    const int logical_start,
    const int token_count,
    const int score_offset) {
    int token_local = get_global_id(0);
    if (token_local >= token_count) {
        return;
    }
    int logical = logical_start + token_local;
    if (logical < 0 || logical >= max_slots) {
        scores[score_offset + token_local] = -3.4028234663852886e+38f;
        return;
    }
    int slot = slot_table[logical];
    int cached_slot = cached_slot_start + token_local;
    if (slot < 0 || slot >= max_slots || cached_slot < 0 || cached_slot >= cached_max_slots) {
        scores[score_offset + token_local] = -3.4028234663852886e+38f;
        return;
    }
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kv_heads; ++h) {
            int ref_base = ((b * kv_heads + h) * max_slots + slot) * head_dim;
            int cached_base = ((b * kv_heads + h) * cached_max_slots + cached_slot) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                acc += fabs((float)reference_value_cache[ref_base + d] - (float)cached_value_cache[cached_base + d]);
            }
        }
    }
    int denom_int = batch * kv_heads * head_dim;
    float denom = (float)(denom_int > 0 ? denom_int : 1);
    scores[score_offset + token_local] = acc / denom;
}

__kernel void pic_cacheblend_topk(
    __global const float* scores,
    __global int* selected,
    const int token_count,
    const int top_k) {
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    __local float best_values[1024];
    __local int best_indices[1024];
    if (get_group_id(0) != 0 || lid >= 1024) {
        return;
    }
    if (token_count <= 1024) {
        for (int i = lid; i < 1024; i += local_size) {
            float value = -3.4028234663852886e+38f;
            int index = i;
            if (i < token_count) {
                value = scores[i];
                if (isnan(value)) {
                    value = -3.4028234663852886e+38f;
                }
            }
            best_values[i] = value;
            best_indices[i] = index;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 2; k <= 1024; k <<= 1) {
            for (int stride = k >> 1; stride > 0; stride >>= 1) {
                for (int i = lid; i < 1024; i += local_size) {
                    int other = i ^ stride;
                    if (other > i) {
                        float value_i = best_values[i];
                        float value_o = best_values[other];
                        int index_i = best_indices[i];
                        int index_o = best_indices[other];
                        int other_better = (value_o > value_i) ||
                            (value_o == value_i && index_o >= 0 &&
                             (index_i < 0 || index_o < index_i));
                        int self_better = (value_i > value_o) ||
                            (value_i == value_o && index_i >= 0 &&
                             (index_o < 0 || index_i < index_o));
                        int descending = ((i & k) == 0);
                        if ((descending && other_better) || (!descending && self_better)) {
                            best_values[i] = value_o;
                            best_indices[i] = index_o;
                            best_values[other] = value_i;
                            best_indices[other] = index_i;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        for (int i = lid; i < top_k; i += local_size) {
            selected[i] = best_indices[i];
        }
        return;
    }
    for (int k = 0; k < top_k; ++k) {
        float best = -3.4028234663852886e+38f;
        int best_index = -1;
        for (int i = lid; i < token_count; i += local_size) {
            int used = 0;
            for (int prev = 0; prev < k; ++prev) {
                if (selected[prev] == i) {
                    used = 1;
                    break;
                }
            }
            if (used) {
                continue;
            }
            float value = scores[i];
            if (isnan(value)) {
                value = -3.4028234663852886e+38f;
            }
            if (best_index < 0 || value > best || (value == best && i < best_index)) {
                best = value;
                best_index = i;
            }
        }
        best_values[lid] = best;
        best_indices[lid] = best_index;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                float other_value = best_values[lid + stride];
                int other_index = best_indices[lid + stride];
                if (other_value > best_values[lid] ||
                    (other_value == best_values[lid] && other_index >= 0 &&
                     (best_indices[lid] < 0 || other_index < best_indices[lid]))) {
                    best_values[lid] = other_value;
                    best_indices[lid] = other_index;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            selected[k] = best_indices[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void paged_attention_row(
    __global const FLOAT* query,
    __global const FLOAT* key_cache,
    __global const FLOAT* value_cache,
    __global FLOAT* output,
    __global const FLOAT* mask,
    __global const int* slot_table,
    __global const int* sparse_query,
    const int mask_elements,
    const int batch,
    const int query_len,
    const int output_len,
    const int attn_len,
    const int num_heads,
    const int kv_heads,
    const int head_dim,
    const int base_logical,
    const int kv_len,
    const int max_slots,
    const float scale,
    const int sparse_active,
    const int query_rows_are_full,
    const int total) {
    int index = get_global_id(0);
    if (index >= total) {
        return;
    }
    int h = index % num_heads;
    int t = index / num_heads;
    int q = t % attn_len;
    int b = t / attn_len;
    int group = num_heads / kv_heads;
    int kv_head = h / group;
    int q_logical = sparse_active ? sparse_query[q] : (base_logical + q);
    int q_row = query_rows_are_full ? q_logical : q;
    int valid_len = min(kv_len, q_logical + 1);
    if (valid_len <= 0 || head_dim > 256) {
        return;
    }

    float max_score = -3.4028234663852886e+38f;
    for (int k = 0; k < valid_len; ++k) {
        int slot = slot_table[k];
        if (slot < 0 || slot >= max_slots) {
            continue;
        }
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            int q_offset = ((b * query_len + q_row) * num_heads + h) * head_dim + d;
            int k_offset = ((slot * batch + b) * kv_heads + kv_head) * head_dim + d;
            score += (float)query[q_offset] * (float)key_cache[k_offset];
        }
        score *= scale;
        if (mask_elements > 1) {
            int mask_cols = mask_elements >= attn_len * kv_len ? kv_len : attn_len;
            int gap = kv_len - mask_cols;
            int col = k - gap;
            int mask_row = query_rows_are_full ? q_logical : q;
            int mask_index = mask_row * mask_cols + col;
            if (col >= 0 && col < mask_cols && mask_index >= 0 && mask_index < mask_elements) {
                score += (float)mask[mask_index];
            }
        }
        max_score = fmax(max_score, score);
    }

    float acc[256];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }
    float sum = 0.0f;
    for (int k = 0; k < valid_len; ++k) {
        int slot = slot_table[k];
        if (slot < 0 || slot >= max_slots) {
            continue;
        }
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            int q_offset = ((b * query_len + q_row) * num_heads + h) * head_dim + d;
            int k_offset = ((slot * batch + b) * kv_heads + kv_head) * head_dim + d;
            score += (float)query[q_offset] * (float)key_cache[k_offset];
        }
        score *= scale;
        if (mask_elements > 1) {
            int mask_cols = mask_elements >= attn_len * kv_len ? kv_len : attn_len;
            int gap = kv_len - mask_cols;
            int col = k - gap;
            int mask_row = query_rows_are_full ? q_logical : q;
            int mask_index = mask_row * mask_cols + col;
            if (col >= 0 && col < mask_cols && mask_index >= 0 && mask_index < mask_elements) {
                score += (float)mask[mask_index];
            }
        }
        float weight = exp(score - max_score);
        sum += weight;
        for (int d = 0; d < head_dim; ++d) {
            int v_offset = ((b * kv_heads + kv_head) * max_slots + slot) * head_dim + d;
            acc[d] += weight * (float)value_cache[v_offset];
        }
    }
    float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        output[((b * output_len + q) * num_heads + h) * head_dim + d] = (FLOAT)(acc[d] * inv_sum);
    }
}

__kernel void paged_attention(
    __global const FLOAT* query,
    __global const FLOAT* key_cache,
    __global const FLOAT* value_cache,
    __global FLOAT* output,
    __global const FLOAT* mask,
    __global const int* slot_table,
    __global const int* sparse_query,
    const int mask_elements,
    const int batch,
    const int query_len,
    const int output_len,
    const int attn_len,
    const int num_heads,
    const int kv_heads,
    const int head_dim,
    const int base_logical,
    const int kv_len,
    const int max_slots,
    const float scale,
    const int sparse_active,
    const int query_rows_are_full,
    const int total) {
    int index = get_global_id(0);
    if (index >= total) {
        return;
    }
    int d = index % head_dim;
    int t = index / head_dim;
    int h = t % num_heads;
    t = t / num_heads;
    int q = t % attn_len;
    int b = t / attn_len;
    int group = num_heads / kv_heads;
    int kv_head = h / group;
    int q_logical = sparse_active ? sparse_query[q] : (base_logical + q);
    int q_row = query_rows_are_full ? q_logical : q;
    int valid_len = min(kv_len, q_logical + 1);
    if (valid_len <= 0) {
        output[((b * output_len + q) * num_heads + h) * head_dim + d] = (FLOAT)0;
        return;
    }

    float max_score = -3.4028234663852886e+38f;
    for (int k = 0; k < valid_len; ++k) {
        int slot = slot_table[k];
        float score = 0.0f;
        for (int kd = 0; kd < head_dim; ++kd) {
            int q_offset = ((b * query_len + q_row) * num_heads + h) * head_dim + kd;
            int k_offset = ((slot * batch + b) * kv_heads + kv_head) * head_dim + kd;
            score += (float)query[q_offset] * (float)key_cache[k_offset];
        }
        score *= scale;
        if (mask_elements > 1) {
            int mask_cols = mask_elements >= attn_len * kv_len ? kv_len : attn_len;
            int gap = kv_len - mask_cols;
            int col = k - gap;
            int mask_row = query_rows_are_full ? q_logical : q;
            int mask_index = mask_row * mask_cols + col;
            if (col >= 0 && col < mask_cols && mask_index >= 0 && mask_index < mask_elements) {
                score += (float)mask[mask_index];
            }
        }
        max_score = fmax(max_score, score);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int k = 0; k < valid_len; ++k) {
        int slot = slot_table[k];
        float score = 0.0f;
        for (int kd = 0; kd < head_dim; ++kd) {
            int q_offset = ((b * query_len + q_row) * num_heads + h) * head_dim + kd;
            int k_offset = ((slot * batch + b) * kv_heads + kv_head) * head_dim + kd;
            score += (float)query[q_offset] * (float)key_cache[k_offset];
        }
        score *= scale;
        if (mask_elements > 1) {
            int mask_cols = mask_elements >= attn_len * kv_len ? kv_len : attn_len;
            int gap = kv_len - mask_cols;
            int col = k - gap;
            int mask_row = query_rows_are_full ? q_logical : q;
            int mask_index = mask_row * mask_cols + col;
            if (col >= 0 && col < mask_cols && mask_index >= 0 && mask_index < mask_elements) {
                score += (float)mask[mask_index];
            }
        }
        float weight = exp(score - max_score);
        int v_offset = ((b * kv_heads + kv_head) * max_slots + slot) * head_dim + d;
        acc += weight * (float)value_cache[v_offset];
        sum += weight;
    }
    output[((b * output_len + q) * num_heads + h) * head_dim + d] = (FLOAT)(sum > 0.0f ? acc / sum : 0.0f);
}
