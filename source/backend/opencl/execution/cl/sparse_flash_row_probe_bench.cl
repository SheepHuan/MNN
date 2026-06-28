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

__kernel void sparse_flash_attention_row32_probe(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query,
                              __global const FLOAT *past_key,
                              __global const FLOAT *past_value,
                              __global const int *sparse_query,
                              __global FLOAT *output,
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
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lid = get_local_id(0);
    if (get_local_size(0) != 32 || lid >= 32 || (head_dim != 64 && head_dim != 128)) {
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

    if (lid == 0) {
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim;
        for (int d8 = 0; d8 < head_dim; d8 += 8) {
            vstore8((FLOAT8)0, 0, output + output_offset + d8);
        }
    }
}
