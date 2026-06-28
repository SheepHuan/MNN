#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define GLOBAL_SIZE_2_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                                                   \
    }

#define DEAL_OUTER_SEQLEN_NOT_ALIGN(length) \
    if(4 * sl + 3 >= length) {\
        temp_3 = (FLOAT4)0;\
    }\
    if(4 * sl + 2 >= length) {\
        temp_2 = (FLOAT4)0;\
    }\
    if(4 * sl + 1 >= length) {\
        temp_1 = (FLOAT4)0;\
    }

#define DEAL_INNER_HEADDIM_NOT_ALIGN(length) \
    if(hd * 4 + 3 >= length) {\
        temp_0.w = (FLOAT)0;\
        temp_1.w = (FLOAT)0;\
        temp_2.w = (FLOAT)0;\
        temp_3.w = (FLOAT)0;\
    }\
    if(hd * 4 + 2 >= length) {\
        temp_0.z = (FLOAT)0;\
        temp_1.z = (FLOAT)0;\
        temp_2.z = (FLOAT)0;\
        temp_3.z = (FLOAT)0;\
    }\
    if(hd * 4 + 1 >= length) {\
        temp_0.y = (FLOAT)0;\
        temp_1.y = (FLOAT)0;\
        temp_2.y = (FLOAT)0;\
        temp_3.y = (FLOAT)0;\
    }



__kernel void rearrange_qkv(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input_q, //[batch, seqLenQ/4, headNum, headDim, seqLenQ_4]
                              __global const FLOAT *input_k, // [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4]
                              __global const FLOAT *input_v, // [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4]
                              __global FLOAT *output_q, // [batch*headNum, ROUND_UP(headDim, mTileHDK), ROUND_UP(seqLenQ, mTileQ)]
                              __global FLOAT *output_k, // [batch*headNum/group, ROUND_UP(headDim, mTileHDK), ROUND_UP(seqLenKV, mTileKV)]
                              __global FLOAT *output_v, // [batch*headNum/group, ROUND_UP(seqLenKV, mTileKV), ROUND_UP(headDim, mTileHDN)]
                              #ifdef SAVE_KV
                              __global FLOAT *past_k, // [batch, headNum/group, headDim, seqLenKV_4]
                              __global FLOAT *past_v, // [batch, headNum/group, seqLenKV_4, headDim]
                              #endif
                              __private const int4 tile, // [mTileQ, mTileKV, mTileHDK, mTileHDN]
                              __private const int4 shape,// [seqLenQ, seqLenKV, headNum, headDim]
                              __private const int4 param, // [group, batch, max_len, past_len]
                              __private const int maxLenKV
) {
    const int sl = get_global_id(0); // seqLen/4 : max(seqLenPackQ/4, seqLenPackKV/4)
    const int hd = get_global_id(1); // headDim/4 : max(headDimPackQK/4, headDimPackV/4)
    const int z = get_global_id(2); // batch * headNum
    DEAL_NON_UNIFORM_DIM3(sl, hd, z);
    
    const int seqLenQ = shape.x;
    const int seqLenKV = shape.y;
    const int headNum = shape.z;
    const int headDim = shape.w;
    const int group = param.x;
    const int batch = param.y;

    const int b = z % batch;
    const int hn = z / batch;
    
    const int seqLenQ_4 = (seqLenQ + 3) / 4;
    //const int in_offset_q = (((b * seqLenQ_4 + sl) * headNum + hn) * headDim + 4 * hd) * 4;
    const int in_offset_q = (((b * seqLenQ + sl * 4) * headNum + hn) * headDim + 4 * hd);

    const int seqLenPackQ = ((seqLenQ + tile.x - 1) / tile.x) * tile.x;
    const int headDimPackQK = ((headDim + tile.z - 1) / tile.z) * tile.z;
    const int out_offset_q = (((b * headNum + hn) * headDimPackQK + hd * 4) * seqLenPackQ + sl * 4);
    
    if(sl * 4 < seqLenPackQ && hd * 4 < headDimPackQK) {
        if(sl * 4 >= seqLenQ || hd * 4 >= headDim) {
            vstore4((FLOAT4)0, 0, output_q + out_offset_q);
            vstore4((FLOAT4)0, 0, output_q + out_offset_q + seqLenPackQ);
            vstore4((FLOAT4)0, 0, output_q + out_offset_q + 2 * seqLenPackQ);
            vstore4((FLOAT4)0, 0, output_q + out_offset_q + 3 * seqLenPackQ);
        } else {
            FLOAT4 temp_0 = vload4(0, input_q + in_offset_q);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenQ) ? (FLOAT4)0 : vload4(0, input_q + in_offset_q + headNum*headDim);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenQ) ? (FLOAT4)0 : vload4(0, input_q + in_offset_q + 2*headNum*headDim);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenQ) ? (FLOAT4)0 : vload4(0, input_q + in_offset_q + 3*headNum*headDim);
            #ifdef HEADDIM_LEAVE
            DEAL_INNER_HEADDIM_NOT_ALIGN(headDim)
            #endif
            #ifdef SEQLEN_LEAVE
            DEAL_OUTER_SEQLEN_NOT_ALIGN(seqLenQ)
            #endif
            vstore4((FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0), 0, output_q + out_offset_q);
            vstore4((FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1), 0, output_q + out_offset_q + seqLenPackQ);
            vstore4((FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2), 0, output_q + out_offset_q + 2 * seqLenPackQ);
            vstore4((FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3), 0, output_q + out_offset_q + 3 * seqLenPackQ);
        }
    }
        
    if(hn >= headNum / group) {
        return;
    }
    

    const int seqLenPackKV = ((seqLenKV + tile.y - 1) / tile.y) * tile.y;
    const int headDimPackV = ((headDim + tile.w - 1) / tile.w) * tile.w;
    const int seqLenKV_4 = (seqLenKV + 3) / 4;
    const int in_offset_kv = (((b * seqLenKV + sl*4) * headNum/group + hn) * headDim + 4 * hd);
    const int past_offset_k = (((b * headNum/group + hn) * headDim + hd * 4) * maxLenKV + sl*4);
    const int past_offset_v = (((b * headNum/group + hn) * maxLenKV + sl*4) * headDim + 4 * hd);
    if(sl * 4 < seqLenPackKV && hd * 4 < headDimPackQK) {
        const int out_offset_k = (((b * headNum/group + hn) * headDimPackQK + hd * 4) * seqLenPackKV + sl * 4);

        if(sl * 4 >= seqLenKV || hd * 4 >= headDim) {
            vstore4((FLOAT4)0, 0, output_k + out_offset_k);
            vstore4((FLOAT4)0, 0, output_k + out_offset_k + seqLenPackKV);
            vstore4((FLOAT4)0, 0, output_k + out_offset_k + 2 * seqLenPackKV);
            vstore4((FLOAT4)0, 0, output_k + out_offset_k + 3 * seqLenPackKV);
        } else {
            FLOAT4 temp_0 = vload4(0, input_k + in_offset_kv);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_k + in_offset_kv + headNum*headDim/group);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_k + in_offset_kv + 2*headNum*headDim/group);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_k + in_offset_kv + 3*headNum*headDim/group);
            #ifdef HEADDIM_LEAVE
            DEAL_INNER_HEADDIM_NOT_ALIGN(headDim)
            #endif
            #ifdef SEQLEN_LEAVE
            DEAL_OUTER_SEQLEN_NOT_ALIGN(seqLenKV)
            #endif
            FLOAT4 key0 = (FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0);
            FLOAT4 key1 = (FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1);
            FLOAT4 key2 = (FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2);
            FLOAT4 key3 = (FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3);
            vstore4(key0, 0, output_k + out_offset_k);
            vstore4(key1, 0, output_k + out_offset_k + seqLenPackKV);
            vstore4(key2, 0, output_k + out_offset_k + 2 * seqLenPackKV);
            vstore4(key3, 0, output_k + out_offset_k + 3 * seqLenPackKV);
            
            // pastK
            #ifdef SAVE_KV
            vstore4(key0, 0, past_k + past_offset_k);
            vstore4(key1, 0, past_k + past_offset_k + maxLenKV);
            vstore4(key2, 0, past_k + past_offset_k + 2*maxLenKV);
            vstore4(key3, 0, past_k + past_offset_k + 3*maxLenKV);
            #endif
        }
        
    }
    
    if(sl * 4 < seqLenPackKV && hd * 4 < headDimPackV) {
        const int out_offset_v = (((b * headNum/group + hn) * seqLenPackKV + sl * 4) * headDimPackV + hd * 4);

        if(sl * 4 >= seqLenKV || hd * 4 >= headDim) {
            vstore4((FLOAT4)0, 0, output_v + out_offset_v);
            vstore4((FLOAT4)0, 0, output_v + out_offset_v + headDimPackV);
            vstore4((FLOAT4)0, 0, output_v + out_offset_v + 2 * headDimPackV);
            vstore4((FLOAT4)0, 0, output_v + out_offset_v + 3 * headDimPackV);
        } else {
            FLOAT4 temp_0 = vload4(0, input_v + in_offset_kv);
            FLOAT4 temp_1 = (sl * 4 + 1 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_v + in_offset_kv + headNum*headDim/group);
            FLOAT4 temp_2 = (sl * 4 + 2 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_v + in_offset_kv + 2*headNum*headDim/group);
            FLOAT4 temp_3 = (sl * 4 + 3 >= seqLenKV) ? (FLOAT4)0 : vload4(0, input_v + in_offset_kv + 3*headNum*headDim/group);
            #ifdef HEADDIM_LEAVE
            DEAL_INNER_HEADDIM_NOT_ALIGN(headDim)
            #endif
            #ifdef SEQLEN_LEAVE
            DEAL_OUTER_SEQLEN_NOT_ALIGN(seqLenKV)
            #endif
            vstore4(temp_0, 0, output_v + out_offset_v);
            vstore4(temp_1, 0, output_v + out_offset_v + headDimPackV);
            vstore4(temp_2, 0, output_v + out_offset_v + 2 * headDimPackV);
            vstore4(temp_3, 0, output_v + out_offset_v + 3 * headDimPackV);
            
            // pastV
            #ifdef SAVE_KV
            vstore4(temp_0, 0, past_v + past_offset_v);
            vstore4(temp_1, 0, past_v + past_offset_v + headDim);
            vstore4(temp_2, 0, past_v + past_offset_v + 2*headDim);
            vstore4(temp_3, 0, past_v + past_offset_v + 3*headDim);
            #endif
        }
        
    }
}

#ifndef MASK_DTYPE
#define MASK_DTYPE FLOAT
#define MASK_DTYPE4 FLOAT4
#endif
__kernel void rearrange_mask(GLOBAL_SIZE_3_DIMS
        __global const MASK_DTYPE *input_mask, // [batch, 1, seqLenQ, seqLenKV, 4]
        __global MASK_DTYPE *output_mask, // [batch, ROUND_UP(seqLenQ, mTileQ), ROUND_UP(seqLenKV, mTileKV)]
        const int4 shape // [seqLenQ, seqLenKV, mTileQ, mTileKV]
) {
    const int sl = get_global_id(0); // seqLen_4
    const int sl_kv = get_global_id(1); // seqLenKV_4
    const int b = get_global_id(2); // Batch
    DEAL_NON_UNIFORM_DIM3(sl, sl_kv, b);
        
    const int seq_len_pack = ((shape.x + shape.z - 1) / shape.z) * shape.z;
    const int seq_len_kv_pack = ((shape.y + shape.w - 1) / shape.w) * shape.w;

    int in_offset = ((b * shape.x + sl * 4) * shape.y + sl_kv * 4);
    int out_offset = (b * seq_len_pack + sl * 4) * seq_len_kv_pack + sl_kv * 4;

    if(sl * 4 >= shape.x || sl_kv * 4 >= shape.y) {
        vstore4((MASK_DTYPE4)0, 0, output_mask + out_offset);
        vstore4((MASK_DTYPE4)0, 0, output_mask + out_offset + seq_len_kv_pack);
        vstore4((MASK_DTYPE4)0, 0, output_mask + out_offset + seq_len_kv_pack * 2);
        vstore4((MASK_DTYPE4)0, 0, output_mask + out_offset + seq_len_kv_pack * 3);
    } else {
        int y_down_align4 = (shape.y / 4 * 4);
        MASK_DTYPE4 temp_0, temp_1, temp_2, temp_3;
        
        if(sl_kv * 4 < y_down_align4) {
            temp_0 = vload4(0, input_mask + in_offset);
            temp_1 = (sl * 4 + 1 >= shape.x) ? (MASK_DTYPE4)0 : vload4(0, input_mask + in_offset + shape.y);
            temp_2 = (sl * 4 + 2 >= shape.x) ? (MASK_DTYPE4)0 : vload4(0, input_mask + in_offset + shape.y * 2);
            temp_3 = (sl * 4 + 3 >= shape.x) ? (MASK_DTYPE4)0 : vload4(0, input_mask + in_offset + shape.y * 3);
        } else if(sl_kv * 4 + 1 == shape.y){
            temp_0 = (MASK_DTYPE4)(input_mask[in_offset], 0, 0, 0);
            temp_1 = (sl * 4 + 1 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y], 0, 0, 0);//vload4(0, input_mask + in_offset + shape.y);
            temp_2 = (sl * 4 + 2 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*2], 0, 0, 0);//vload4(0, input_mask + in_offset + shape.y * 2);
            temp_3 = (sl * 4 + 3 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*3], 0, 0, 0);//vload4(0, input_mask + in_offset + shape.y * 3);
        } else if(sl_kv * 4 + 2 == shape.y){
            temp_0 = (MASK_DTYPE4)(input_mask[in_offset], input_mask[in_offset+1], 0, 0);
            temp_1 = (sl * 4 + 1 >= shape.x) ? (MASK_DTYPE4)0 : (FLOAT4)(input_mask[in_offset + shape.y], input_mask[in_offset + shape.y + 1], 0, 0);//vload4(0, input_mask + in_offset + shape.y);
            temp_2 = (sl * 4 + 2 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*2], input_mask[in_offset + shape.y*2 + 1], 0, 0);//vload4(0, input_mask + in_offset + shape.y * 2);
            temp_3 = (sl * 4 + 3 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*3], input_mask[in_offset + shape.y*3 + 1], 0, 0);//vload4(0, input_mask + in_offset + shape.y * 3);
        } else if(sl_kv * 4 + 3 == shape.y){
            temp_0 = (MASK_DTYPE4)(input_mask[in_offset], input_mask[in_offset+1], input_mask[in_offset+2], 0);
            temp_1 = (sl * 4 + 1 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y], input_mask[in_offset + shape.y + 1], input_mask[in_offset + shape.y + 2], 0);//vload4(0, input_mask + in_offset + shape.y);
            temp_2 = (sl * 4 + 2 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*2], input_mask[in_offset + shape.y*2 + 1], input_mask[in_offset + shape.y*2 + 2], 0);//vload4(0, input_mask + in_offset + shape.y * 2);
            temp_3 = (sl * 4 + 3 >= shape.x) ? (MASK_DTYPE4)0 : (MASK_DTYPE4)(input_mask[in_offset + shape.y*3], input_mask[in_offset + shape.y*3 + 1], input_mask[in_offset + shape.y*3 + 2], 0);//vload4(0, input_mask + in_offset + shape.y * 3);
        }

        vstore4(temp_0, 0, output_mask + out_offset);
        vstore4(temp_1, 0, output_mask + out_offset + seq_len_kv_pack);
        vstore4(temp_2, 0, output_mask + out_offset + 2 * seq_len_kv_pack);
        vstore4(temp_3, 0, output_mask + out_offset + 3 * seq_len_kv_pack);
    }

}

__kernel void qkv_transpose_output(GLOBAL_SIZE_3_DIMS
          __global const FLOAT *input, // [Batch * mNumHead, ROUND_UP(mHeadDim, mTileHDN), ROUND_UP(seqLen, mTileQ)]
          __global FLOAT *output, // [Batch, seqLen/4, mNumHead， mHeadDim, 4]
          __private const int tile_q,
          __private const int tile_hdn,
          __private const int seq_len,
          __private const int head_num,
          __private const int head_dim
) {
    
    const int sl = get_global_id(0); // seqLen_4
    const int hd = get_global_id(1); // mHeadDim_4
    const int z = get_global_id(2); // Batch * mNumHead
    DEAL_NON_UNIFORM_DIM3(sl, hd, z);
    
    const int b = z / head_num;
    const int hn = z % head_num;
        
    const int seq_len_pack = ((seq_len + tile_q - 1) / tile_q) * tile_q;
    const int head_dim_pack = ((head_dim + tile_hdn - 1) / tile_hdn) * tile_hdn;
    
    const int offset_inp = ((b * head_num + hn) * head_dim_pack + 4 * hd) * seq_len_pack + 4 * sl;
    
    const int offset_out = (((b * seq_len + sl*4) * head_num + hn) * head_dim + 4 * hd);
    
    // Q
    FLOAT4 temp_0 = vload4(0, input + offset_inp);
    FLOAT4 temp_1 = vload4(0, input + offset_inp + seq_len_pack);
    FLOAT4 temp_2 = vload4(0, input + offset_inp + 2 * seq_len_pack);
    FLOAT4 temp_3 = vload4(0, input + offset_inp + 3 * seq_len_pack);
    
    vstore4((FLOAT4)(temp_0.s0, temp_1.s0, temp_2.s0, temp_3.s0), 0, output + offset_out);
    if(4 * sl + 1 >= seq_len) return;
    vstore4((FLOAT4)(temp_0.s1, temp_1.s1, temp_2.s1, temp_3.s1), 0, output + offset_out + head_num*head_dim);
    if(4 * sl + 2 >= seq_len) return;
    vstore4((FLOAT4)(temp_0.s2, temp_1.s2, temp_2.s2, temp_3.s2), 0, output + offset_out + 2*head_num*head_dim);
    if(4 * sl + 3 >= seq_len) return;
    vstore4((FLOAT4)(temp_0.s3, temp_1.s3, temp_2.s3, temp_3.s3), 0, output + offset_out + 3*head_num*head_dim);

}

#ifndef NUMHEAD_GROUP_SIZE
#define NUMHEAD_GROUP_SIZE 1
#endif

__kernel void rearrange_q(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch query_seq_len head_num head_dim]
                              __global FLOAT *query_tmp, // [batch head_num head_dim_4 query_seq_len_4]
                              __private const int seq_len,
                              __private const int head_dim,
                              __private const int head_num) {
    /*
     the kernel assume head_dim is multiple of 4.
     */
    const int x = get_global_id(0); // query_seq_len/4
    const int y = get_global_id(1); // head_dim/4
    int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int b = z / head_num;// batch
    z = z % head_num;// head_num
    
    const int x4 = x << 2;
    const int y4 = y << 2;
    const int seq_len4 = (seq_len + 3) / 4 * 4;;
    const int stride = head_num * head_dim;
    int query_offset = ((b * seq_len + x4) * head_num + z) * head_dim + y4;
    FLOAT4 query_vec0 = vload4(0, query + query_offset); query_offset += stride;
    FLOAT4 query_vec1 = (x4 + 1 >= seq_len) ? (FLOAT4)0 : vload4(0, query + query_offset); query_offset += stride;
    FLOAT4 query_vec2 = (x4 + 2 >= seq_len) ? (FLOAT4)0 : vload4(0, query + query_offset); query_offset += stride;
    FLOAT4 query_vec3 = (x4 + 3 >= seq_len) ? (FLOAT4)0 : vload4(0, query + query_offset);
    
    const int queryout_offset = ((b * head_num + z) * head_dim + y4) * seq_len4 + x4;
    vstore4((FLOAT4)(query_vec0.s0, query_vec1.s0, query_vec2.s0, query_vec3.s0), 0, query_tmp + queryout_offset);
    vstore4((FLOAT4)(query_vec0.s1, query_vec1.s1, query_vec2.s1, query_vec3.s1), 0, query_tmp + queryout_offset + seq_len4);
    vstore4((FLOAT4)(query_vec0.s2, query_vec1.s2, query_vec2.s2, query_vec3.s2), 0, query_tmp + queryout_offset + seq_len4 + seq_len4);
    vstore4((FLOAT4)(query_vec0.s3, query_vec1.s3, query_vec2.s3, query_vec3.s3), 0, query_tmp + queryout_offset + seq_len4 + seq_len4 + seq_len4);
}

__kernel void rearrange_q_sparse(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch query_seq_len head_num head_dim]
                              __global const int *sparse_query, // [output_seq_len]
                              __global FLOAT *query_tmp, // [batch head_num head_dim_4 output_seq_len_4]
                              __private const int output_seq_len,
                              __private const int input_seq_len,
                              __private const int head_dim,
                              __private const int head_num) {
    /*
     Pack selected logical rows from full query into the same layout as rearrange_q,
     so later sparse q-split kernels can read compact rows contiguously.
     */
    const int x = get_global_id(0); // output_seq_len / 4
    const int y = get_global_id(1); // head_dim / 4
    int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int b = z / head_num;
    z = z % head_num;

    const int x4 = x << 2;
    const int y4 = y << 2;
    const int output_seq_len4 = (output_seq_len + 3) / 4 * 4;
    const int stride = head_num * head_dim;

    const int q_index0 = x4;
    const int q_index1 = x4 + 1;
    const int q_index2 = x4 + 2;
    const int q_index3 = x4 + 3;
    const int q_row0 = q_index0 < output_seq_len ? sparse_query[q_index0] : -1;
    const int q_row1 = q_index1 < output_seq_len ? sparse_query[q_index1] : -1;
    const int q_row2 = q_index2 < output_seq_len ? sparse_query[q_index2] : -1;
    const int q_row3 = q_index3 < output_seq_len ? sparse_query[q_index3] : -1;

    const bool valid0 = q_row0 >= 0 && q_row0 < input_seq_len;
    const bool valid1 = q_row1 >= 0 && q_row1 < input_seq_len;
    const bool valid2 = q_row2 >= 0 && q_row2 < input_seq_len;
    const bool valid3 = q_row3 >= 0 && q_row3 < input_seq_len;

    const int query_offset0 = ((b * input_seq_len + q_row0) * head_num + z) * head_dim + y4;
    const int query_offset1 = ((b * input_seq_len + q_row1) * head_num + z) * head_dim + y4;
    const int query_offset2 = ((b * input_seq_len + q_row2) * head_num + z) * head_dim + y4;
    const int query_offset3 = ((b * input_seq_len + q_row3) * head_num + z) * head_dim + y4;
    const FLOAT4 query_vec0 = valid0 ? vload4(0, query + query_offset0) : (FLOAT4)0;
    const FLOAT4 query_vec1 = valid1 ? vload4(0, query + query_offset1) : (FLOAT4)0;
    const FLOAT4 query_vec2 = valid2 ? vload4(0, query + query_offset2) : (FLOAT4)0;
    const FLOAT4 query_vec3 = valid3 ? vload4(0, query + query_offset3) : (FLOAT4)0;

    const int queryout_offset = ((b * head_num + z) * head_dim + y4) * output_seq_len4 + x4;
    vstore4((FLOAT4)(query_vec0.s0, query_vec1.s0, query_vec2.s0, query_vec3.s0), 0, query_tmp + queryout_offset);
    vstore4((FLOAT4)(query_vec0.s1, query_vec1.s1, query_vec2.s1, query_vec3.s1), 0,
            query_tmp + queryout_offset + output_seq_len4);
    vstore4((FLOAT4)(query_vec0.s2, query_vec1.s2, query_vec2.s2, query_vec3.s2), 0,
            query_tmp + queryout_offset + output_seq_len4 + output_seq_len4);
    vstore4((FLOAT4)(query_vec0.s3, query_vec1.s3, query_vec2.s3, query_vec3.s3), 0,
            query_tmp + queryout_offset + output_seq_len4 + output_seq_len4 + output_seq_len4);
}
__kernel void rearrange_k(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *key, // [batch key_seq_len kv_head_num head_dim]
                              __global FLOAT *past_key, // [batch kv_head_num head_dim max_length]
                              __private const int past_len, // prefill = 0, decode = past_key len
                              __private const int max_len,
                              __private const int seq_len,
                              __private const int kv_head_num,
                              __private const int head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // seq_len decode = 1
    const int y = get_global_id(1); // head_dim
    int z = get_global_id(2); //
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int b = z / kv_head_num;
    z = z % kv_head_num;
    const int y4 = y << 2;
    
#ifdef OPENCL_PREFILL_ATTENTION
    const int x4 = x << 2;
    const int stride = kv_head_num * head_dim;
    int key_offset = ((b * seq_len + x4) * kv_head_num + z) * head_dim + y4;
    FLOAT4 key_vec0 = vload4(0, key + key_offset); key_offset += stride;
    FLOAT4 key_vec1 = (x4 + 1 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset); key_offset += stride;
    FLOAT4 key_vec2 = (x4 + 2 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset); key_offset += stride;
    FLOAT4 key_vec3 = (x4 + 3 >= seq_len) ? (FLOAT4)0 : vload4(0, key + key_offset);
    const int output_offset = ((b * kv_head_num + z) * head_dim + y4) * max_len + past_len + x4;
    vstore4((FLOAT4)(key_vec0.s0, key_vec1.s0, key_vec2.s0, key_vec3.s0), 0, past_key + output_offset);
    vstore4((FLOAT4)(key_vec0.s1, key_vec1.s1, key_vec2.s1, key_vec3.s1), 0, past_key + output_offset + max_len);
    vstore4((FLOAT4)(key_vec0.s2, key_vec1.s2, key_vec2.s2, key_vec3.s2), 0, past_key + output_offset + max_len + max_len);
    vstore4((FLOAT4)(key_vec0.s3, key_vec1.s3, key_vec2.s3, key_vec3.s3), 0, past_key + output_offset + max_len + max_len + max_len);
#else
    FLOAT4 key_vec = vload4(0, key + (b * kv_head_num + z) * head_dim + y4);
    const int output_offset = ((b * kv_head_num + z) * head_dim + y4) * max_len + past_len;
    past_key[output_offset] = key_vec.s0;
    past_key[output_offset + max_len] = key_vec.s1;
    past_key[output_offset + max_len + max_len] = key_vec.s2;
    past_key[output_offset + max_len + max_len + max_len] = key_vec.s3;
#endif
}

__kernel void rearrange_v(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *value, // [batch value_seq_len kv_head_num head_dim]
                              __global FLOAT *past_value, // [batch kv_head_num max_length head_dim]
                              __private const int past_len,
                              __private const int max_len,
                              __private const int seq_len,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // seq_len decode = 1
    int z = get_global_id(2); // kv_head_num
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int b = z / kv_head_num;
    z = z % kv_head_num;
    const int x4 = x << 2;
    
#ifdef OPENCL_PREFILL_ATTENTION
    const int y4 = y << 2;
    const int stride = kv_head_num * head_dim;
    int value_offset = ((b * seq_len + y4) * kv_head_num + z) * head_dim + x4;
    FLOAT4 value_vec0 = vload4(0, value + value_offset); value_offset += stride;
    FLOAT4 value_vec1 = (y4 + 1 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset); value_offset += stride;
    FLOAT4 value_vec2 = (y4 + 2 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset); value_offset += stride;
    FLOAT4 value_vec3 = (y4 + 3 >= seq_len) ? (FLOAT4)0 : vload4(0, value + value_offset);
    const int output_offset = ((b * kv_head_num + z) * max_len + past_len + y4) * head_dim + x4;
    vstore4(value_vec0, 0, past_value + output_offset);
    vstore4(value_vec1, 0, past_value + output_offset + head_dim);
    vstore4(value_vec2, 0, past_value + output_offset + head_dim + head_dim);
    vstore4(value_vec3, 0, past_value + output_offset + head_dim + head_dim + head_dim);
#else
    FLOAT4 value_vec = vload4(0, value + (b * kv_head_num + z) * head_dim + x4);
    const int output_offset = ((b * kv_head_num + z) * max_len + past_len) * head_dim + x4;
    vstore4(value_vec, 0, past_value + output_offset);
#endif
}

__kernel void rearrange_mask_shortprefill(GLOBAL_SIZE_3_DIMS
                                #ifdef ADD_MASK
                                __global const FLOAT* mask,
                                __global FLOAT* maskout,
                                #else
                                __global const int* mask, // [1 1 query_seq_len mask_key_seq_len4]
                                __global int* maskout, // [1 1 mask_key_seq_len4 query_seq_len4]
                                #endif
                                __private const int query_seq_len,
                                __private const int mask_key_seq_len){
    const int x = get_global_id(0); // query_seq_len4
    const int y = get_global_id(1); // mask_key_seq_len4
    const int z = get_global_id(2); // batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int x4 = x << 2;
    const int y4 = y << 2;
    float4 mask_tmp0, mask_tmp1, mask_tmp2, mask_tmp3;
    float4 mask0, mask1, mask2, mask3;
    int mask_offset = x4 * mask_key_seq_len + y4;
    if(x4 + 3 < query_seq_len && y4 + 3 < mask_key_seq_len){
        mask_tmp0 = convert_float4(vload4(0, mask + mask_offset)); mask_offset += mask_key_seq_len;
        mask_tmp1 = convert_float4(vload4(0, mask + mask_offset)); mask_offset += mask_key_seq_len;
        mask_tmp2 = convert_float4(vload4(0, mask + mask_offset)); mask_offset += mask_key_seq_len;
        mask_tmp3 = convert_float4(vload4(0, mask + mask_offset));
    } else{
        if(y4 + 3 < mask_key_seq_len){
            mask_tmp0 = convert_float4(vload4(0, mask + mask_offset)); mask_offset += mask_key_seq_len;
            mask_tmp1 = (x4 + 1 >= query_seq_len) ? (float4)0 : convert_float4(vload4(0, mask + mask_offset)); mask_offset += mask_key_seq_len;
            mask_tmp2 = (x4 + 2 >= query_seq_len) ? (float4)0 : convert_float4(vload4(0, mask + mask_offset)); mask_offset += mask_key_seq_len;
            mask_tmp3 = (x4 + 3 >= query_seq_len) ? (float4)0 : convert_float4(vload4(0, mask + mask_offset));
        } else if(y4 + 1 == mask_key_seq_len){
            mask_tmp0 = (float4)(mask[mask_offset], 0, 0, 0); mask_offset += mask_key_seq_len;
            mask_tmp1 = (x4 + 1 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], 0, 0, 0); mask_offset += mask_key_seq_len;
            mask_tmp2 = (x4 + 2 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], 0, 0, 0); mask_offset += mask_key_seq_len;
            mask_tmp3 = (x4 + 3 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], 0, 0, 0);
        }else if(y4 + 2 == mask_key_seq_len){
            mask_tmp0 = (float4)(mask[mask_offset], mask[mask_offset + 1], 0, 0); mask_offset += mask_key_seq_len;
            mask_tmp1 = (x4 + 1 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], mask[mask_offset + 1], 0, 0); mask_offset += mask_key_seq_len;
            mask_tmp2 = (x4 + 2 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], mask[mask_offset + 1], 0, 0); mask_offset += mask_key_seq_len;
            mask_tmp3 = (x4 + 3 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], mask[mask_offset + 1], 0, 0);
        }else if(y4 + 3 == mask_key_seq_len){
            mask_tmp0 = (float4)(mask[mask_offset], mask[mask_offset + 1], mask[mask_offset + 2], 0); mask_offset += mask_key_seq_len;
            mask_tmp1 = (x4 + 1 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], mask[mask_offset + 1], mask[mask_offset + 2], 0); mask_offset += mask_key_seq_len;
            mask_tmp2 = (x4 + 2 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], mask[mask_offset + 1], mask[mask_offset + 2], 0); mask_offset += mask_key_seq_len;
            mask_tmp3 = (x4 + 3 >= query_seq_len) ? (float4)0 : (float4)(mask[mask_offset], mask[mask_offset + 1], mask[mask_offset + 2], 0);
        }
    }
    mask0 = (float4)(mask_tmp0.s0, mask_tmp1.s0, mask_tmp2.s0, mask_tmp3.s0);
    mask1 = (float4)(mask_tmp0.s1, mask_tmp1.s1, mask_tmp2.s1, mask_tmp3.s1);
    mask2 = (float4)(mask_tmp0.s2, mask_tmp1.s2, mask_tmp2.s2, mask_tmp3.s2);
    mask3 = (float4)(mask_tmp0.s3, mask_tmp1.s3, mask_tmp2.s3, mask_tmp3.s3);
    
    int query_seq_len4 = ((query_seq_len + 3) / 4) * 4;
    int output_offset = y4 * query_seq_len4 + x4;
    #ifdef ADD_MASK
    vstore4(CONVERT_FLOAT4(mask0), 0, maskout + output_offset);
    vstore4(CONVERT_FLOAT4(mask1), 0, maskout + output_offset + query_seq_len4);
    vstore4(CONVERT_FLOAT4(mask2), 0, maskout + output_offset + query_seq_len4 + query_seq_len4);
    vstore4(CONVERT_FLOAT4(mask3), 0, maskout + output_offset + query_seq_len4 + query_seq_len4 + query_seq_len4);
    #else
    vstore4(convert_int4(mask0), 0, maskout + output_offset);
    vstore4(convert_int4(mask1), 0, maskout + output_offset + query_seq_len4);
    vstore4(convert_int4(mask2), 0, maskout + output_offset + query_seq_len4 + query_seq_len4);
    vstore4(convert_int4(mask3), 0, maskout + output_offset + query_seq_len4 + query_seq_len4 + query_seq_len4);
    #endif
}

__kernel void matmul_qk_div_mask_prefill(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 query_seq_len_4]
                              __global const FLOAT *past_key, // [batch kv_head_num head_dim_4 kv_max_length]
                              #ifdef ADD_MASK
                              __global const FLOAT* mask,
                              #elif defined(SET_MASK)
                              __global const int* mask, // [1 1 query_seq_len mask_key_seq_len]
                              #endif
                              __global FLOAT *qk, // [batch head_num kv_seq_length query_seq_len_4]
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int mask_key_seq_len,
                              __private const int key_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // query_seq_len
    const int y = get_global_id(1); // kv_seq_length
    const int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int x4 = x << 2;
    const int y4 = y << 2;
    
    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;;
    const int query_offset = z * head_dim * query_seq_len4 + x4;
    const int past_offset = (z / NUMHEAD_GROUP_SIZE) * head_dim * max_len + y4;
    float4 out0 = 0, out1 = 0, out2 = 0, out3 = 0;
    
    for(int i = 0; i < head_dim / 4; ++i){
        int i4 = i << 2;
        float4 query_vec0 = convert_float4(vload4(0, query + query_offset + i4 * query_seq_len4));
        float4 query_vec1 = convert_float4(vload4(0, query + query_offset + (i4 + 1) * query_seq_len4));
        float4 query_vec2 = convert_float4(vload4(0, query + query_offset + (i4 + 2) * query_seq_len4));
        float4 query_vec3 = convert_float4(vload4(0, query + query_offset + (i4 + 3) * query_seq_len4));
        
        float4 past_vec0 = convert_float4(vload4(0, past_key + past_offset + i4 * max_len));
        float4 past_vec1 = convert_float4(vload4(0, past_key + past_offset + (i4 + 1) * max_len));
        float4 past_vec2 = convert_float4(vload4(0, past_key + past_offset + (i4 + 2) * max_len));
        float4 past_vec3 = convert_float4(vload4(0, past_key + past_offset + (i4 + 3) * max_len));

        out0 = mad((float4)past_vec0.s0, query_vec0, out0);
        out0 = mad((float4)past_vec1.s0, query_vec1, out0);
        out0 = mad((float4)past_vec2.s0, query_vec2, out0);
        out0 = mad((float4)past_vec3.s0, query_vec3, out0);
        
        out1 = mad((float4)past_vec0.s1, query_vec0, out1);
        out1 = mad((float4)past_vec1.s1, query_vec1, out1);
        out1 = mad((float4)past_vec2.s1, query_vec2, out1);
        out1 = mad((float4)past_vec3.s1, query_vec3, out1);
        
        out2 = mad((float4)past_vec0.s2, query_vec0, out2);
        out2 = mad((float4)past_vec1.s2, query_vec1, out2);
        out2 = mad((float4)past_vec2.s2, query_vec2, out2);
        out2 = mad((float4)past_vec3.s2, query_vec3, out2);
        
        out3 = mad((float4)past_vec0.s3, query_vec0, out3);
        out3 = mad((float4)past_vec1.s3, query_vec1, out3);
        out3 = mad((float4)past_vec2.s3, query_vec2, out3);
        out3 = mad((float4)past_vec3.s3, query_vec3, out3);
    }
    out0 *= (float4)scale;
    out1 *= (float4)scale;
    out2 *= (float4)scale;
    out3 *= (float4)scale;
    {
        #if defined(ADD_MASK) || defined(SET_MASK)
        int query_seq_len4 = ((query_seq_len + 3) / 4) * 4;
        int mask_clp = y4 + mask_key_seq_len - key_seq_len;
        int mask_offset = mask_clp * query_seq_len4 + x4;
        float4 mask0 = mask_clp >= 0 && mask_clp < mask_key_seq_len ? convert_float4(vload4(0, mask + mask_offset)) : 0; mask_offset += query_seq_len4;
        float4 mask1 = mask_clp + 1 >= 0 && mask_clp + 1 < mask_key_seq_len? convert_float4(vload4(0, mask + mask_offset)) : 0; mask_offset += query_seq_len4;
        float4 mask2 = mask_clp + 2 >= 0 && mask_clp + 2 < mask_key_seq_len? convert_float4(vload4(0, mask + mask_offset)) : 0; mask_offset += query_seq_len4;
        float4 mask3 = mask_clp + 3 >= 0 && mask_clp + 3 < mask_key_seq_len? convert_float4(vload4(0, mask + mask_offset)) : 0;
        #endif
        
        #ifdef ADD_MASK
        out0 += mask0;
        out1 += mask1;
        out2 += mask2;
        out3 += mask3;
        #elif defined(SET_MASK)
        out0 = (mask0 == (float4)0) ? (float4)(-FLT_MAX) : out0;
        out1 = (mask1 == (float4)0) ? (float4)(-FLT_MAX) : out1;
        out2 = (mask2 == (float4)0) ? (float4)(-FLT_MAX) : out2;
        out3 = (mask3 == (float4)0) ? (float4)(-FLT_MAX) : out3;
        #endif
    }
    
    const int qk_offset = (z * key_seq_len + y4) * query_seq_len4 + x4;
    vstore4(CONVERT_FLOAT4(out0), 0, qk + qk_offset);
    if(y4 + 1 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out1), 0, qk + qk_offset + query_seq_len4);
    if(y4 + 2 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out2), 0, qk + qk_offset + query_seq_len4 + query_seq_len4);
    if(y4 + 3 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out3), 0, qk + qk_offset + query_seq_len4 + query_seq_len4 + query_seq_len4);
}

__kernel void matmul_qk_div_mask_prefill_piece(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 full_query_seq_len_4]
                              __global const FLOAT *past_key, // [batch kv_head_num head_dim_4 kv_max_length]
                              #ifdef ADD_MASK
                              __global const FLOAT* mask,
                              #elif defined(SET_MASK)
                              __global const int* mask, // [1 1 mask_key_seq_len full_query_seq_len_4]
                              #endif
                              __global FLOAT *qk, // [batch head_num kv_seq_length q_piece_len_4]
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int q_start,
                              __private const int q_piece_len,
                              __private const int mask_key_seq_len,
                              __private const int key_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int head_dim) {

    const int x = get_global_id(0); // q piece token / 4
    const int y = get_global_id(1); // kv token / 4
    const int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int x4 = x << 2;
    const int q4 = q_start + x4;
    const int y4 = y << 2;
    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int q_piece_len4 = (q_piece_len + 3) / 4 * 4;
    const int query_offset = z * head_dim * query_seq_len4 + q4;
    const int past_offset = (z / NUMHEAD_GROUP_SIZE) * head_dim * max_len + y4;
    float4 out0 = 0, out1 = 0, out2 = 0, out3 = 0;

    for(int i = 0; i < head_dim / 4; ++i){
        int i4 = i << 2;
        float4 query_vec0 = convert_float4(vload4(0, query + query_offset + i4 * query_seq_len4));
        float4 query_vec1 = convert_float4(vload4(0, query + query_offset + (i4 + 1) * query_seq_len4));
        float4 query_vec2 = convert_float4(vload4(0, query + query_offset + (i4 + 2) * query_seq_len4));
        float4 query_vec3 = convert_float4(vload4(0, query + query_offset + (i4 + 3) * query_seq_len4));

        float4 past_vec0 = convert_float4(vload4(0, past_key + past_offset + i4 * max_len));
        float4 past_vec1 = convert_float4(vload4(0, past_key + past_offset + (i4 + 1) * max_len));
        float4 past_vec2 = convert_float4(vload4(0, past_key + past_offset + (i4 + 2) * max_len));
        float4 past_vec3 = convert_float4(vload4(0, past_key + past_offset + (i4 + 3) * max_len));

        out0 = mad((float4)past_vec0.s0, query_vec0, out0);
        out0 = mad((float4)past_vec1.s0, query_vec1, out0);
        out0 = mad((float4)past_vec2.s0, query_vec2, out0);
        out0 = mad((float4)past_vec3.s0, query_vec3, out0);

        out1 = mad((float4)past_vec0.s1, query_vec0, out1);
        out1 = mad((float4)past_vec1.s1, query_vec1, out1);
        out1 = mad((float4)past_vec2.s1, query_vec2, out1);
        out1 = mad((float4)past_vec3.s1, query_vec3, out1);

        out2 = mad((float4)past_vec0.s2, query_vec0, out2);
        out2 = mad((float4)past_vec1.s2, query_vec1, out2);
        out2 = mad((float4)past_vec2.s2, query_vec2, out2);
        out2 = mad((float4)past_vec3.s2, query_vec3, out2);

        out3 = mad((float4)past_vec0.s3, query_vec0, out3);
        out3 = mad((float4)past_vec1.s3, query_vec1, out3);
        out3 = mad((float4)past_vec2.s3, query_vec2, out3);
        out3 = mad((float4)past_vec3.s3, query_vec3, out3);
    }
    out0 *= (float4)scale;
    out1 *= (float4)scale;
    out2 *= (float4)scale;
    out3 *= (float4)scale;
    {
        #if defined(ADD_MASK) || defined(SET_MASK)
        int mask_clp = y4 + mask_key_seq_len - key_seq_len;
        int mask_offset = mask_clp * query_seq_len4 + q4;
        float4 mask0 = mask_clp >= 0 && mask_clp < mask_key_seq_len ? convert_float4(vload4(0, mask + mask_offset)) : 0; mask_offset += query_seq_len4;
        float4 mask1 = mask_clp + 1 >= 0 && mask_clp + 1 < mask_key_seq_len? convert_float4(vload4(0, mask + mask_offset)) : 0; mask_offset += query_seq_len4;
        float4 mask2 = mask_clp + 2 >= 0 && mask_clp + 2 < mask_key_seq_len? convert_float4(vload4(0, mask + mask_offset)) : 0; mask_offset += query_seq_len4;
        float4 mask3 = mask_clp + 3 >= 0 && mask_clp + 3 < mask_key_seq_len? convert_float4(vload4(0, mask + mask_offset)) : 0;
        #endif

        #ifdef ADD_MASK
        out0 += mask0;
        out1 += mask1;
        out2 += mask2;
        out3 += mask3;
        #elif defined(SET_MASK)
        out0 = (mask0 == (float4)0) ? (float4)(-FLT_MAX) : out0;
        out1 = (mask1 == (float4)0) ? (float4)(-FLT_MAX) : out1;
        out2 = (mask2 == (float4)0) ? (float4)(-FLT_MAX) : out2;
        out3 = (mask3 == (float4)0) ? (float4)(-FLT_MAX) : out3;
        #endif
    }

    const int qk_offset = (z * key_seq_len + y4) * q_piece_len4 + x4;
    vstore4(CONVERT_FLOAT4(out0), 0, qk + qk_offset);
    if(y4 + 1 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out1), 0, qk + qk_offset + q_piece_len4);
    if(y4 + 2 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out2), 0, qk + qk_offset + q_piece_len4 + q_piece_len4);
    if(y4 + 3 >= key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out3), 0, qk + qk_offset + q_piece_len4 + q_piece_len4 + q_piece_len4);
}

__kernel void matmul_qk_div_mask_prefill_piece_sparse(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 query_seq_len_4]
                              __global const FLOAT *past_key, // [batch kv_head_num head_dim_4 kv_max_length]
                              #ifdef ADD_MASK
                              __global const FLOAT* mask,
                              #elif defined(SET_MASK)
                              __global const int* mask,
                              #endif
                              __global const int *sparse_query,
                              __global FLOAT *qk, // [batch head_num active_kv_seq_len q_piece_len_4]
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int output_seq_len,
                              __private const int query_rows_are_full,
                              __private const int q_start,
                              __private const int q_piece_len,
                              __private const int mask_key_seq_len,
                              __private const int active_key_seq_len,
                              __private const int full_key_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int head_dim) {

    const int x = get_global_id(0); // q piece token / 4
    const int y = get_global_id(1); // active kv token / 4
    const int z = get_global_id(2); // head_num * batch
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int x4 = x << 2;
    const int y4 = y << 2;
    const int q_piece_len4 = (q_piece_len + 3) / 4 * 4;
    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int q_index0 = q_start + x4;
    const int q_index1 = q_index0 + 1;
    const int q_index2 = q_index0 + 2;
    const int q_index3 = q_index0 + 3;

    const int q_logical0 = q_index0 < output_seq_len ? sparse_query[q_index0] : -1;
    const int q_logical1 = q_index1 < output_seq_len ? sparse_query[q_index1] : -1;
    const int q_logical2 = q_index2 < output_seq_len ? sparse_query[q_index2] : -1;
    const int q_logical3 = q_index3 < output_seq_len ? sparse_query[q_index3] : -1;

    const int q_row0 = query_rows_are_full ? q_logical0 : q_index0;
    const int q_row1 = query_rows_are_full ? q_logical1 : q_index1;
    const int q_row2 = query_rows_are_full ? q_logical2 : q_index2;
    const int q_row3 = query_rows_are_full ? q_logical3 : q_index3;

    const bool q_valid0 = q_index0 < output_seq_len && x4 < q_piece_len && q_row0 >= 0 && q_row0 < query_seq_len;
    const bool q_valid1 = q_index1 < output_seq_len && x4 + 1 < q_piece_len && q_row1 >= 0 && q_row1 < query_seq_len;
    const bool q_valid2 = q_index2 < output_seq_len && x4 + 2 < q_piece_len && q_row2 >= 0 && q_row2 < query_seq_len;
    const bool q_valid3 = q_index3 < output_seq_len && x4 + 3 < q_piece_len && q_row3 >= 0 && q_row3 < query_seq_len;
    const int q_max_logical = max(
        max(q_valid0 ? q_logical0 : -1, q_valid1 ? q_logical1 : -1),
        max(q_valid2 ? q_logical2 : -1, q_valid3 ? q_logical3 : -1));
    const int q_min_logical = min(
        min(q_valid0 ? q_logical0 : full_key_seq_len, q_valid1 ? q_logical1 : full_key_seq_len),
        min(q_valid2 ? q_logical2 : full_key_seq_len, q_valid3 ? q_logical3 : full_key_seq_len));
    const int k0 = y4;
    const int k1 = y4 + 1;
    const int k2 = y4 + 2;
    const int k3 = y4 + 3;
    const bool compute0 = q_max_logical >= 0 && k0 < active_key_seq_len && k0 <= q_max_logical;
    const bool compute1 = q_max_logical >= 0 && k1 < active_key_seq_len && k1 <= q_max_logical;
    const bool compute2 = q_max_logical >= 0 && k2 < active_key_seq_len && k2 <= q_max_logical;
    const bool compute3 = q_max_logical >= 0 && k3 < active_key_seq_len && k3 <= q_max_logical;
    const bool tile_empty = !(compute0 || compute1 || compute2 || compute3);
    const bool tile_full = q_max_logical >= 0 && k3 < active_key_seq_len && k3 <= q_min_logical;

    const int past_offset = (z / NUMHEAD_GROUP_SIZE) * head_dim * max_len + y4;
    COMPUTE_FLOAT4 out0 = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)-FLT_MAX);
    COMPUTE_FLOAT4 out1 = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)-FLT_MAX);
    COMPUTE_FLOAT4 out2 = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)-FLT_MAX);
    COMPUTE_FLOAT4 out3 = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)-FLT_MAX);

    if (tile_empty) {
        const int qk_offset = (z * active_key_seq_len + y4) * q_piece_len4 + x4;
        vstore4(CONVERT_FLOAT4(out0), 0, qk + qk_offset);
        if (y4 + 1 >= active_key_seq_len) return;
        vstore4(CONVERT_FLOAT4(out1), 0, qk + qk_offset + q_piece_len4);
        if (y4 + 2 >= active_key_seq_len) return;
        vstore4(CONVERT_FLOAT4(out2), 0, qk + qk_offset + q_piece_len4 + q_piece_len4);
        if (y4 + 3 >= active_key_seq_len) return;
        vstore4(CONVERT_FLOAT4(out3), 0, qk + qk_offset + q_piece_len4 + q_piece_len4 + q_piece_len4);
        return;
    }
    if (compute0) out0 = 0;
    if (compute1) out1 = 0;
    if (compute2) out2 = 0;
    if (compute3) out3 = 0;

    for (int i = 0; i < head_dim / 4; ++i) {
        const int i4 = i << 2;
        const int query_base = z * head_dim * query_seq_len4 + i4 * query_seq_len4;

        COMPUTE_FLOAT4 query_vec0;
        COMPUTE_FLOAT4 query_vec1;
        COMPUTE_FLOAT4 query_vec2;
        COMPUTE_FLOAT4 query_vec3;
        if (!query_rows_are_full) {
            const int query_offset = query_base + q_index0;
            query_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_offset));
            query_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_offset + query_seq_len4));
            query_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_offset + query_seq_len4 + query_seq_len4));
            query_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_offset + query_seq_len4 + query_seq_len4 + query_seq_len4));
        } else {
            query_vec0 = (COMPUTE_FLOAT4)(
                q_valid0 ? (COMPUTE_FLOAT)(query[query_base + q_row0]) : (COMPUTE_FLOAT)0,
                q_valid1 ? (COMPUTE_FLOAT)(query[query_base + q_row1]) : (COMPUTE_FLOAT)0,
                q_valid2 ? (COMPUTE_FLOAT)(query[query_base + q_row2]) : (COMPUTE_FLOAT)0,
                q_valid3 ? (COMPUTE_FLOAT)(query[query_base + q_row3]) : (COMPUTE_FLOAT)0);
            query_vec1 = (COMPUTE_FLOAT4)(
                q_valid0 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + q_row0]) : (COMPUTE_FLOAT)0,
                q_valid1 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + q_row1]) : (COMPUTE_FLOAT)0,
                q_valid2 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + q_row2]) : (COMPUTE_FLOAT)0,
                q_valid3 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + q_row3]) : (COMPUTE_FLOAT)0);
            query_vec2 = (COMPUTE_FLOAT4)(
                q_valid0 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + query_seq_len4 + q_row0]) : (COMPUTE_FLOAT)0,
                q_valid1 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + query_seq_len4 + q_row1]) : (COMPUTE_FLOAT)0,
                q_valid2 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + query_seq_len4 + q_row2]) : (COMPUTE_FLOAT)0,
                q_valid3 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + query_seq_len4 + q_row3]) : (COMPUTE_FLOAT)0);
            query_vec3 = (COMPUTE_FLOAT4)(
                q_valid0 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + query_seq_len4 + query_seq_len4 + q_row0]) : (COMPUTE_FLOAT)0,
                q_valid1 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + query_seq_len4 + query_seq_len4 + q_row1]) : (COMPUTE_FLOAT)0,
                q_valid2 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + query_seq_len4 + query_seq_len4 + q_row2]) : (COMPUTE_FLOAT)0,
                q_valid3 ? (COMPUTE_FLOAT)(query[query_base + query_seq_len4 + query_seq_len4 + query_seq_len4 + q_row3]) : (COMPUTE_FLOAT)0);
        }

        const COMPUTE_FLOAT4 past_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_key + past_offset + i4 * max_len));
        const COMPUTE_FLOAT4 past_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_key + past_offset + (i4 + 1) * max_len));
        const COMPUTE_FLOAT4 past_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_key + past_offset + (i4 + 2) * max_len));
        const COMPUTE_FLOAT4 past_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_key + past_offset + (i4 + 3) * max_len));

        if (tile_full) {
            out0 = mad((COMPUTE_FLOAT4)past_vec0.s0, query_vec0, out0);
            out0 = mad((COMPUTE_FLOAT4)past_vec1.s0, query_vec1, out0);
            out0 = mad((COMPUTE_FLOAT4)past_vec2.s0, query_vec2, out0);
            out0 = mad((COMPUTE_FLOAT4)past_vec3.s0, query_vec3, out0);

            out1 = mad((COMPUTE_FLOAT4)past_vec0.s1, query_vec0, out1);
            out1 = mad((COMPUTE_FLOAT4)past_vec1.s1, query_vec1, out1);
            out1 = mad((COMPUTE_FLOAT4)past_vec2.s1, query_vec2, out1);
            out1 = mad((COMPUTE_FLOAT4)past_vec3.s1, query_vec3, out1);

            out2 = mad((COMPUTE_FLOAT4)past_vec0.s2, query_vec0, out2);
            out2 = mad((COMPUTE_FLOAT4)past_vec1.s2, query_vec1, out2);
            out2 = mad((COMPUTE_FLOAT4)past_vec2.s2, query_vec2, out2);
            out2 = mad((COMPUTE_FLOAT4)past_vec3.s2, query_vec3, out2);

            out3 = mad((COMPUTE_FLOAT4)past_vec0.s3, query_vec0, out3);
            out3 = mad((COMPUTE_FLOAT4)past_vec1.s3, query_vec1, out3);
            out3 = mad((COMPUTE_FLOAT4)past_vec2.s3, query_vec2, out3);
            out3 = mad((COMPUTE_FLOAT4)past_vec3.s3, query_vec3, out3);
        } else if (compute0) {
            out0 = mad((COMPUTE_FLOAT4)past_vec0.s0, query_vec0, out0);
            out0 = mad((COMPUTE_FLOAT4)past_vec1.s0, query_vec1, out0);
            out0 = mad((COMPUTE_FLOAT4)past_vec2.s0, query_vec2, out0);
            out0 = mad((COMPUTE_FLOAT4)past_vec3.s0, query_vec3, out0);
        }

        if (!tile_full && compute1) {
            out1 = mad((COMPUTE_FLOAT4)past_vec0.s1, query_vec0, out1);
            out1 = mad((COMPUTE_FLOAT4)past_vec1.s1, query_vec1, out1);
            out1 = mad((COMPUTE_FLOAT4)past_vec2.s1, query_vec2, out1);
            out1 = mad((COMPUTE_FLOAT4)past_vec3.s1, query_vec3, out1);
        }

        if (!tile_full && compute2) {
            out2 = mad((COMPUTE_FLOAT4)past_vec0.s2, query_vec0, out2);
            out2 = mad((COMPUTE_FLOAT4)past_vec1.s2, query_vec1, out2);
            out2 = mad((COMPUTE_FLOAT4)past_vec2.s2, query_vec2, out2);
            out2 = mad((COMPUTE_FLOAT4)past_vec3.s2, query_vec3, out2);
        }

        if (!tile_full && compute3) {
            out3 = mad((COMPUTE_FLOAT4)past_vec0.s3, query_vec0, out3);
            out3 = mad((COMPUTE_FLOAT4)past_vec1.s3, query_vec1, out3);
            out3 = mad((COMPUTE_FLOAT4)past_vec2.s3, query_vec2, out3);
            out3 = mad((COMPUTE_FLOAT4)past_vec3.s3, query_vec3, out3);
        }
    }

    if (tile_full || compute0) out0 *= (COMPUTE_FLOAT4)((COMPUTE_FLOAT)scale);
    if (tile_full || compute1) out1 *= (COMPUTE_FLOAT4)((COMPUTE_FLOAT)scale);
    if (tile_full || compute2) out2 *= (COMPUTE_FLOAT4)((COMPUTE_FLOAT)scale);
    if (tile_full || compute3) out3 *= (COMPUTE_FLOAT4)((COMPUTE_FLOAT)scale);

    #if defined(ADD_MASK) || defined(SET_MASK)
    const int mask_gap = full_key_seq_len - mask_key_seq_len;
    const int mask_col0 = k0 - mask_gap;
    const int mask_col1 = k1 - mask_gap;
    const int mask_col2 = k2 - mask_gap;
    const int mask_col3 = k3 - mask_gap;
    #endif

    #ifdef ADD_MASK
    if (k0 >= 0 && k0 < active_key_seq_len) {
        out0.s0 += (q_valid0 && mask_col0 >= 0 && mask_col0 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row0 * mask_key_seq_len + mask_col0]) : (COMPUTE_FLOAT)0;
        out0.s1 += (q_valid1 && mask_col0 >= 0 && mask_col0 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row1 * mask_key_seq_len + mask_col0]) : (COMPUTE_FLOAT)0;
        out0.s2 += (q_valid2 && mask_col0 >= 0 && mask_col0 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row2 * mask_key_seq_len + mask_col0]) : (COMPUTE_FLOAT)0;
        out0.s3 += (q_valid3 && mask_col0 >= 0 && mask_col0 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row3 * mask_key_seq_len + mask_col0]) : (COMPUTE_FLOAT)0;
    }
    if (k1 >= 0 && k1 < active_key_seq_len) {
        out1.s0 += (q_valid0 && mask_col1 >= 0 && mask_col1 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row0 * mask_key_seq_len + mask_col1]) : (COMPUTE_FLOAT)0;
        out1.s1 += (q_valid1 && mask_col1 >= 0 && mask_col1 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row1 * mask_key_seq_len + mask_col1]) : (COMPUTE_FLOAT)0;
        out1.s2 += (q_valid2 && mask_col1 >= 0 && mask_col1 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row2 * mask_key_seq_len + mask_col1]) : (COMPUTE_FLOAT)0;
        out1.s3 += (q_valid3 && mask_col1 >= 0 && mask_col1 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row3 * mask_key_seq_len + mask_col1]) : (COMPUTE_FLOAT)0;
    }
    if (k2 >= 0 && k2 < active_key_seq_len) {
        out2.s0 += (q_valid0 && mask_col2 >= 0 && mask_col2 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row0 * mask_key_seq_len + mask_col2]) : (COMPUTE_FLOAT)0;
        out2.s1 += (q_valid1 && mask_col2 >= 0 && mask_col2 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row1 * mask_key_seq_len + mask_col2]) : (COMPUTE_FLOAT)0;
        out2.s2 += (q_valid2 && mask_col2 >= 0 && mask_col2 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row2 * mask_key_seq_len + mask_col2]) : (COMPUTE_FLOAT)0;
        out2.s3 += (q_valid3 && mask_col2 >= 0 && mask_col2 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row3 * mask_key_seq_len + mask_col2]) : (COMPUTE_FLOAT)0;
    }
    if (k3 >= 0 && k3 < active_key_seq_len) {
        out3.s0 += (q_valid0 && mask_col3 >= 0 && mask_col3 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row0 * mask_key_seq_len + mask_col3]) : (COMPUTE_FLOAT)0;
        out3.s1 += (q_valid1 && mask_col3 >= 0 && mask_col3 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row1 * mask_key_seq_len + mask_col3]) : (COMPUTE_FLOAT)0;
        out3.s2 += (q_valid2 && mask_col3 >= 0 && mask_col3 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row2 * mask_key_seq_len + mask_col3]) : (COMPUTE_FLOAT)0;
        out3.s3 += (q_valid3 && mask_col3 >= 0 && mask_col3 < mask_key_seq_len) ? (COMPUTE_FLOAT)(mask[q_row3 * mask_key_seq_len + mask_col3]) : (COMPUTE_FLOAT)0;
    }
    #endif

    if (tile_full) {
        out0.s0 = q_valid0 ? out0.s0 : -FLT_MAX;
        out0.s1 = q_valid1 ? out0.s1 : -FLT_MAX;
        out0.s2 = q_valid2 ? out0.s2 : -FLT_MAX;
        out0.s3 = q_valid3 ? out0.s3 : -FLT_MAX;
        out1.s0 = q_valid0 ? out1.s0 : -FLT_MAX;
        out1.s1 = q_valid1 ? out1.s1 : -FLT_MAX;
        out1.s2 = q_valid2 ? out1.s2 : -FLT_MAX;
        out1.s3 = q_valid3 ? out1.s3 : -FLT_MAX;
        out2.s0 = q_valid0 ? out2.s0 : -FLT_MAX;
        out2.s1 = q_valid1 ? out2.s1 : -FLT_MAX;
        out2.s2 = q_valid2 ? out2.s2 : -FLT_MAX;
        out2.s3 = q_valid3 ? out2.s3 : -FLT_MAX;
        out3.s0 = q_valid0 ? out3.s0 : -FLT_MAX;
        out3.s1 = q_valid1 ? out3.s1 : -FLT_MAX;
        out3.s2 = q_valid2 ? out3.s2 : -FLT_MAX;
        out3.s3 = q_valid3 ? out3.s3 : -FLT_MAX;
    } else {
        out0.s0 = (compute0 && q_valid0 && k0 <= q_logical0) ? out0.s0 : -FLT_MAX;
        out0.s1 = (compute0 && q_valid1 && k0 <= q_logical1) ? out0.s1 : -FLT_MAX;
        out0.s2 = (compute0 && q_valid2 && k0 <= q_logical2) ? out0.s2 : -FLT_MAX;
        out0.s3 = (compute0 && q_valid3 && k0 <= q_logical3) ? out0.s3 : -FLT_MAX;
        out1.s0 = (compute1 && q_valid0 && k1 <= q_logical0) ? out1.s0 : -FLT_MAX;
        out1.s1 = (compute1 && q_valid1 && k1 <= q_logical1) ? out1.s1 : -FLT_MAX;
        out1.s2 = (compute1 && q_valid2 && k1 <= q_logical2) ? out1.s2 : -FLT_MAX;
        out1.s3 = (compute1 && q_valid3 && k1 <= q_logical3) ? out1.s3 : -FLT_MAX;
        out2.s0 = (compute2 && q_valid0 && k2 <= q_logical0) ? out2.s0 : -FLT_MAX;
        out2.s1 = (compute2 && q_valid1 && k2 <= q_logical1) ? out2.s1 : -FLT_MAX;
        out2.s2 = (compute2 && q_valid2 && k2 <= q_logical2) ? out2.s2 : -FLT_MAX;
        out2.s3 = (compute2 && q_valid3 && k2 <= q_logical3) ? out2.s3 : -FLT_MAX;
        out3.s0 = (compute3 && q_valid0 && k3 <= q_logical0) ? out3.s0 : -FLT_MAX;
        out3.s1 = (compute3 && q_valid1 && k3 <= q_logical1) ? out3.s1 : -FLT_MAX;
        out3.s2 = (compute3 && q_valid2 && k3 <= q_logical2) ? out3.s2 : -FLT_MAX;
        out3.s3 = (compute3 && q_valid3 && k3 <= q_logical3) ? out3.s3 : -FLT_MAX;
    }

    #ifdef SET_MASK
    if (k0 >= 0 && k0 < active_key_seq_len) {
        out0.s0 = (q_valid0 && mask_col0 >= 0 && mask_col0 < mask_key_seq_len && mask[q_row0 * mask_key_seq_len + mask_col0] != 0) ? out0.s0 : -FLT_MAX;
        out0.s1 = (q_valid1 && mask_col0 >= 0 && mask_col0 < mask_key_seq_len && mask[q_row1 * mask_key_seq_len + mask_col0] != 0) ? out0.s1 : -FLT_MAX;
        out0.s2 = (q_valid2 && mask_col0 >= 0 && mask_col0 < mask_key_seq_len && mask[q_row2 * mask_key_seq_len + mask_col0] != 0) ? out0.s2 : -FLT_MAX;
        out0.s3 = (q_valid3 && mask_col0 >= 0 && mask_col0 < mask_key_seq_len && mask[q_row3 * mask_key_seq_len + mask_col0] != 0) ? out0.s3 : -FLT_MAX;
    }
    if (k1 >= 0 && k1 < active_key_seq_len) {
        out1.s0 = (q_valid0 && mask_col1 >= 0 && mask_col1 < mask_key_seq_len && mask[q_row0 * mask_key_seq_len + mask_col1] != 0) ? out1.s0 : -FLT_MAX;
        out1.s1 = (q_valid1 && mask_col1 >= 0 && mask_col1 < mask_key_seq_len && mask[q_row1 * mask_key_seq_len + mask_col1] != 0) ? out1.s1 : -FLT_MAX;
        out1.s2 = (q_valid2 && mask_col1 >= 0 && mask_col1 < mask_key_seq_len && mask[q_row2 * mask_key_seq_len + mask_col1] != 0) ? out1.s2 : -FLT_MAX;
        out1.s3 = (q_valid3 && mask_col1 >= 0 && mask_col1 < mask_key_seq_len && mask[q_row3 * mask_key_seq_len + mask_col1] != 0) ? out1.s3 : -FLT_MAX;
    }
    if (k2 >= 0 && k2 < active_key_seq_len) {
        out2.s0 = (q_valid0 && mask_col2 >= 0 && mask_col2 < mask_key_seq_len && mask[q_row0 * mask_key_seq_len + mask_col2] != 0) ? out2.s0 : -FLT_MAX;
        out2.s1 = (q_valid1 && mask_col2 >= 0 && mask_col2 < mask_key_seq_len && mask[q_row1 * mask_key_seq_len + mask_col2] != 0) ? out2.s1 : -FLT_MAX;
        out2.s2 = (q_valid2 && mask_col2 >= 0 && mask_col2 < mask_key_seq_len && mask[q_row2 * mask_key_seq_len + mask_col2] != 0) ? out2.s2 : -FLT_MAX;
        out2.s3 = (q_valid3 && mask_col2 >= 0 && mask_col2 < mask_key_seq_len && mask[q_row3 * mask_key_seq_len + mask_col2] != 0) ? out2.s3 : -FLT_MAX;
    }
    if (k3 >= 0 && k3 < active_key_seq_len) {
        out3.s0 = (q_valid0 && mask_col3 >= 0 && mask_col3 < mask_key_seq_len && mask[q_row0 * mask_key_seq_len + mask_col3] != 0) ? out3.s0 : -FLT_MAX;
        out3.s1 = (q_valid1 && mask_col3 >= 0 && mask_col3 < mask_key_seq_len && mask[q_row1 * mask_key_seq_len + mask_col3] != 0) ? out3.s1 : -FLT_MAX;
        out3.s2 = (q_valid2 && mask_col3 >= 0 && mask_col3 < mask_key_seq_len && mask[q_row2 * mask_key_seq_len + mask_col3] != 0) ? out3.s2 : -FLT_MAX;
        out3.s3 = (q_valid3 && mask_col3 >= 0 && mask_col3 < mask_key_seq_len && mask[q_row3 * mask_key_seq_len + mask_col3] != 0) ? out3.s3 : -FLT_MAX;
    }
    #endif

    const int qk_offset = (z * active_key_seq_len + y4) * q_piece_len4 + x4;
    vstore4(CONVERT_FLOAT4(out0), 0, qk + qk_offset);
    if (y4 + 1 >= active_key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out1), 0, qk + qk_offset + q_piece_len4);
    if (y4 + 2 >= active_key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out2), 0, qk + qk_offset + q_piece_len4 + q_piece_len4);
    if (y4 + 3 >= active_key_seq_len) return;
    vstore4(CONVERT_FLOAT4(out3), 0, qk + qk_offset + q_piece_len4 + q_piece_len4 + q_piece_len4);
}
__kernel void matmul_qk_decode(GLOBAL_SIZE_2_DIMS
                              __global const FLOAT *query, // key [1 head_num head_dim]
                              __global const FLOAT *past_key, // [1 head_num head_dim max_length]
                              __global FLOAT *qk, // [1 head_num key_seq_len 1]
                              __private const float scale,
                              __private const int seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // key_seq_len
    const int y = get_global_id(1); // head_num
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int x4 = x << 2;
    
    const int query_offset = y * head_dim;
    const int past_offset = (y / NUMHEAD_GROUP_SIZE) * head_dim * max_len + x4;
    float4 out0 = 0;
    
    for(int i = 0; i < head_dim / 4; ++i){
        int i4 = i << 2;
        float4 query_vec = convert_float4(vload4(0, query + query_offset + i4));
        
        float4 past_vec0 = convert_float4(vload4(0, past_key + past_offset + i4 * max_len));
        float4 past_vec1 = convert_float4(vload4(0, past_key + past_offset + (i4 + 1) * max_len));
        float4 past_vec2 = convert_float4(vload4(0, past_key + past_offset + (i4 + 2) * max_len));
        float4 past_vec3 = convert_float4(vload4(0, past_key + past_offset + (i4 + 3) * max_len));
        
        out0 = mad((float4)query_vec.s0, past_vec0, out0);
        out0 = mad((float4)query_vec.s1, past_vec1, out0);
        out0 = mad((float4)query_vec.s2, past_vec2, out0);
        out0 = mad((float4)query_vec.s3, past_vec3, out0);
    }
    out0 *= (float4)scale;
    const int qk_offset = y * seq_len + x4;
    if(x4 + 3 < seq_len){
        vstore4(CONVERT_FLOAT4(out0), 0, qk + qk_offset);
    }else {
        int remain = seq_len - x4;
        if(remain == 3){
            vstore3(CONVERT_FLOAT3((float3)(out0.s012)), 0, qk + qk_offset);
        } else if(remain == 2){
            vstore2(CONVERT_FLOAT2((float2)(out0.s01)), 0, qk + qk_offset);
        }else if(remain == 1){
            qk[qk_offset] = out0.s0;
        }
    }
}

__kernel void matmul_qkv_prefill(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *qk, // qk prefill [batch head_num kv_seq_length query_seq_len]
                              __global const FLOAT *past_value, // [batch kv_head_num max_len head_dim]
                              __global FLOAT *output, // [batch query_seq_len head_num head_dim]
                              __private const int query_seq_len,
                              __private const int kv_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // query_seq_len
    int z = get_global_id(2); // head_num * batch
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    const int b = z / head_num;
    z = z % head_num;
    const int x8 = x << 3;
    const int y4 = y << 2;
    
    const int query_seq_len4 = (query_seq_len + 3) / 4 * 4;
    const int qk_offset = (b * head_num + z) * kv_seq_len * query_seq_len4 + y4;
    const int past_offset = ((b * kv_head_num + z / NUMHEAD_GROUP_SIZE) * max_len) * head_dim + x8;
    const int loop_end = max(kv_seq_len / 4 - 1, 0);
    COMPUTE_FLOAT8 out0 = 0, out1 = 0, out2 = 0, out3 = 0;
    
    for(int i = 0; i < loop_end; ++i){
        int i4 = i << 2;
        COMPUTE_FLOAT4 qk_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i4 * query_seq_len4));
        COMPUTE_FLOAT4 qk_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 1) * query_seq_len4));
        COMPUTE_FLOAT4 qk_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 2) * query_seq_len4));
        COMPUTE_FLOAT4 qk_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 3) * query_seq_len4));
        
        COMPUTE_FLOAT8 past_vec0 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i4 * head_dim));
        COMPUTE_FLOAT8 past_vec1 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 1) * head_dim));
        COMPUTE_FLOAT8 past_vec2 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 2) * head_dim));
        COMPUTE_FLOAT8 past_vec3 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 3) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT8)qk_vec0.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec1.s0, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec2.s0, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec3.s0, past_vec3, out0);
        
        out1 = mad((COMPUTE_FLOAT8)qk_vec0.s1, past_vec0, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec1.s1, past_vec1, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec2.s1, past_vec2, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec3.s1, past_vec3, out1);
        
        out2 = mad((COMPUTE_FLOAT8)qk_vec0.s2, past_vec0, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec1.s2, past_vec1, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec2.s2, past_vec2, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec3.s2, past_vec3, out2);
        
        out3 = mad((COMPUTE_FLOAT8)qk_vec0.s3, past_vec0, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec1.s3, past_vec1, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec2.s3, past_vec2, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec3.s3, past_vec3, out3);
    }
    for(int i = (loop_end << 2); i < kv_seq_len; ++i){
        COMPUTE_FLOAT4 qk_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i * query_seq_len4));
        COMPUTE_FLOAT8 past_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i * head_dim));
        
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s0, past_vec, out0);
        out1 = mad((COMPUTE_FLOAT8)qk_vec.s1, past_vec, out1);
        out2 = mad((COMPUTE_FLOAT8)qk_vec.s2, past_vec, out2);
        out3 = mad((COMPUTE_FLOAT8)qk_vec.s3, past_vec, out3);
    }
    
    const int output_offset = ((b * query_seq_len + y4) * head_num + z) * head_dim + x8;
    const int stride = head_num * head_dim;
    vstore8(CONVERT_FLOAT8(out0), 0, output + output_offset);
    if(y4 + 1 >= query_seq_len) return;
    vstore8(CONVERT_FLOAT8(out1), 0, output + output_offset + stride);
    if(y4 + 2 >= query_seq_len) return;
    vstore8(CONVERT_FLOAT8(out2), 0, output + output_offset + stride + stride);
    if(y4 + 3 >= query_seq_len) return;
    vstore8(CONVERT_FLOAT8(out3), 0, output + output_offset + stride + stride + stride);
}

__kernel void matmul_qkv_prefill_piece(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *qk, // [batch head_num kv_seq_length q_piece_len_4]
                              __global const FLOAT *past_value, // [batch kv_head_num max_len head_dim]
                              __global const int *sparse_query, // [output_seq_len], used when sparse_query_active != 0
                              __global FLOAT *output, // [batch full_query_seq_len head_num head_dim]
                              __private const int query_seq_len,
                              __private const int output_seq_len,
                              __private const int q_start,
                              __private const int q_piece_len,
                              __private const int kv_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int sparse_query_active,
                              __private const int head_dim) {

    const int x = get_global_id(0); // head_dim / 8
    const int y = get_global_id(1); // q piece token / 4
    int z = get_global_id(2); // head_num * batch

    DEAL_NON_UNIFORM_DIM3(x, y, z);
    const int b = z / head_num;
    z = z % head_num;
    const int x8 = x << 3;
    const int y4 = y << 2;
    const int global_y4 = q_start + y4;

    const int q_piece_len4 = (q_piece_len + 3) / 4 * 4;
    const int qk_offset = (b * head_num + z) * kv_seq_len * q_piece_len4 + y4;
    const int past_offset = ((b * kv_head_num + z / NUMHEAD_GROUP_SIZE) * max_len) * head_dim + x8;
    int q_group_max_logical = kv_seq_len - 1;
    if (sparse_query_active != 0) {
        const int q_index0 = global_y4;
        const int q_index1 = global_y4 + 1;
        const int q_index2 = global_y4 + 2;
        const int q_index3 = global_y4 + 3;
        q_group_max_logical = max(
            max(q_index0 < output_seq_len ? sparse_query[q_index0] : -1,
                q_index1 < output_seq_len ? sparse_query[q_index1] : -1),
            max(q_index2 < output_seq_len ? sparse_query[q_index2] : -1,
                q_index3 < output_seq_len ? sparse_query[q_index3] : -1));
        q_group_max_logical = clamp(q_group_max_logical, -1, kv_seq_len - 1);
    }
    const int visible_kv_seq_len = sparse_query_active != 0 ? clamp(q_group_max_logical + 1, 0, kv_seq_len) : kv_seq_len;
    const int loop_end = max(visible_kv_seq_len / 4 - 1, 0);
    COMPUTE_FLOAT8 out0 = 0, out1 = 0, out2 = 0, out3 = 0;

    for(int i = 0; i < loop_end; ++i){
        int i4 = i << 2;
        COMPUTE_FLOAT4 qk_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i4 * q_piece_len4));
        COMPUTE_FLOAT4 qk_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 1) * q_piece_len4));
        COMPUTE_FLOAT4 qk_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 2) * q_piece_len4));
        COMPUTE_FLOAT4 qk_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + (i4 + 3) * q_piece_len4));

        COMPUTE_FLOAT8 past_vec0 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i4 * head_dim));
        COMPUTE_FLOAT8 past_vec1 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 1) * head_dim));
        COMPUTE_FLOAT8 past_vec2 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 2) * head_dim));
        COMPUTE_FLOAT8 past_vec3 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 3) * head_dim));

        out0 = mad((COMPUTE_FLOAT8)qk_vec0.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec1.s0, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec2.s0, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec3.s0, past_vec3, out0);

        out1 = mad((COMPUTE_FLOAT8)qk_vec0.s1, past_vec0, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec1.s1, past_vec1, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec2.s1, past_vec2, out1);
        out1 = mad((COMPUTE_FLOAT8)qk_vec3.s1, past_vec3, out1);

        out2 = mad((COMPUTE_FLOAT8)qk_vec0.s2, past_vec0, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec1.s2, past_vec1, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec2.s2, past_vec2, out2);
        out2 = mad((COMPUTE_FLOAT8)qk_vec3.s2, past_vec3, out2);

        out3 = mad((COMPUTE_FLOAT8)qk_vec0.s3, past_vec0, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec1.s3, past_vec1, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec2.s3, past_vec2, out3);
        out3 = mad((COMPUTE_FLOAT8)qk_vec3.s3, past_vec3, out3);
    }
    for(int i = (loop_end << 2); i < visible_kv_seq_len; ++i){
        COMPUTE_FLOAT4 qk_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i * q_piece_len4));
        COMPUTE_FLOAT8 past_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i * head_dim));

        out0 = mad((COMPUTE_FLOAT8)qk_vec.s0, past_vec, out0);
        out1 = mad((COMPUTE_FLOAT8)qk_vec.s1, past_vec, out1);
        out2 = mad((COMPUTE_FLOAT8)qk_vec.s2, past_vec, out2);
        out3 = mad((COMPUTE_FLOAT8)qk_vec.s3, past_vec, out3);
    }

    const int output_offset = ((b * query_seq_len + global_y4) * head_num + z) * head_dim + x8;
    const int stride = head_num * head_dim;
    vstore8(CONVERT_FLOAT8(out0), 0, output + output_offset);
    if(y4 + 1 >= q_piece_len || global_y4 + 1 >= query_seq_len) return;
    vstore8(CONVERT_FLOAT8(out1), 0, output + output_offset + stride);
    if(y4 + 2 >= q_piece_len || global_y4 + 2 >= query_seq_len) return;
    vstore8(CONVERT_FLOAT8(out2), 0, output + output_offset + stride + stride);
    if(y4 + 3 >= q_piece_len || global_y4 + 3 >= query_seq_len) return;
    vstore8(CONVERT_FLOAT8(out3), 0, output + output_offset + stride + stride + stride);
}
#define SPARSE_FLASH_MAX_HEAD_DIM8 16
__constant sampler_t SPARSE_FLASH_IMAGE_SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline FLOAT4 sparse_flash_read_linear4_image(__read_only image2d_t image,
                                              const int linear_index,
                                              const int image_width) {
    const int pixel0 = linear_index >> 2;
    const int2 coord0 = (int2)(pixel0 - (pixel0 / image_width) * image_width, pixel0 / image_width);
    const FLOAT4 p0 = RI_F(image, SPARSE_FLASH_IMAGE_SAMPLER, coord0);
    const int lane = linear_index & 3;
    if (lane == 0) {
        return p0;
    }
    const int pixel1 = (linear_index + 3) >> 2;
    const int2 coord1 = (int2)(pixel1 - (pixel1 / image_width) * image_width, pixel1 / image_width);
    const FLOAT4 p1 = RI_F(image, SPARSE_FLASH_IMAGE_SAMPLER, coord1);
    if (lane == 1) {
        return (FLOAT4)(p0.y, p0.z, p0.w, p1.x);
    }
    if (lane == 2) {
        return (FLOAT4)(p0.z, p0.w, p1.x, p1.y);
    }
    return (FLOAT4)(p0.w, p1.x, p1.y, p1.z);
}

inline FLOAT8 sparse_flash_read_linear8_image(__read_only image2d_t image,
                                              const int linear_index,
                                              const int image_width) {
    const FLOAT4 lo = sparse_flash_read_linear4_image(image, linear_index, image_width);
    const FLOAT4 hi = sparse_flash_read_linear4_image(image, linear_index + 4, image_width);
    return (FLOAT8)(lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w);
}

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

__kernel void mqtile_sparse_flash_hd128_q4k8_kvimg(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 query_seq_len_4]
                              __read_only image2d_t past_key_image,
                              __read_only image2d_t past_value_image,
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
                              __private const int key_image_width,
                              __private const int value_image_width,
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
                sparse_flash_read_linear4_image(past_key_image, key_base + dim * key_max_len + k_index,
                                                key_image_width);
            local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + d_lane] =
                sparse_flash_read_linear4_image(past_key_image, key_base + (dim + 4) * key_max_len + k_index,
                                                key_image_width);
            local_v[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + d_lane] =
                sparse_flash_read_linear8_image(past_value_image, value_base + k_index * head_dim + dim,
                                                value_image_width);
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

__kernel void mqtile_sparse_flash_hd128_q4k8_kimg(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 query_seq_len_4]
                              __read_only image2d_t past_key_image,
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
                              __private const int key_image_width,
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
                sparse_flash_read_linear4_image(past_key_image, key_base + dim * key_max_len + k_index,
                                                key_image_width);
            local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_Q4K8_DLANES + d_lane] =
                sparse_flash_read_linear4_image(past_key_image, key_base + (dim + 4) * key_max_len + k_index,
                                                key_image_width);
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

__kernel void mqtile_sparse_flash_hd128_q8k16_kvimg(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 query_seq_len_4]
                              __read_only image2d_t past_key_image,
                              __read_only image2d_t past_value_image,
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
                              __private const int key_image_width,
                              __private const int value_image_width,
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
                sparse_flash_read_linear4_image(past_key_image, key_base + dim * key_max_len + k_index,
                                                key_image_width);
            local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + d_lane] =
                sparse_flash_read_linear4_image(past_key_image, key_base + (dim + 4) * key_max_len + k_index,
                                                key_image_width);
            local_v[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + d_lane] =
                sparse_flash_read_linear8_image(past_value_image, value_base + k_index * head_dim + dim,
                                                value_image_width);
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

__kernel void mqtile_sparse_flash_hd128_q8k16_kimg(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch head_num head_dim_4 query_seq_len_4]
                              __read_only image2d_t past_key_image,
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
                              __private const int key_image_width,
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
                sparse_flash_read_linear4_image(past_key_image, key_base + dim * key_max_len + k_index,
                                                key_image_width);
            local_k1[k_local * MQTILE_SPARSE_FLASH_HD128_Q8K16_DLANES + d_lane] =
                sparse_flash_read_linear4_image(past_key_image, key_base + (dim + 4) * key_max_len + k_index,
                                                key_image_width);
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

__kernel void decode_causal_attention_row64(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch, query_seq_len, head_num, head_dim]
                              __global const FLOAT *key_cache, // [max_slots, batch, kv_head_num, head_dim]
                              __global const FLOAT *value_cache, // [batch, kv_head_num, max_slots, head_dim]
                              __global const int *slot_table, // [key_seq_len]
                              __global const int *sparse_query, // [output_seq_len], used when sparse_query_active != 0
                              __global FLOAT *output, // [batch, output_seq_len, head_num, head_dim]
                              __private const float scale,
                              __private const int batch,
                              __private const int query_seq_len,
                              __private const int output_seq_len,
                              __private const int base_logical,
                              __private const int sparse_query_active,
                              __private const int query_rows_are_full,
                              __private const int key_seq_len,
                              __private const int key_max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lid = get_local_id(0);
    if (get_local_size(0) != 64 || lid >= 64 || head_dim != 64) {
        return;
    }
    const int q_index = y;
    if (q_index >= output_seq_len) {
        return;
    }

    const int b = z / head_num;
    const int h = z - b * head_num;
    if (b >= batch) {
        return;
    }
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }

    const int q_logical = sparse_query_active != 0 ? sparse_query[q_index] : base_logical + q_index;
    const int q_row = query_rows_are_full ? q_logical : q_index;
    if (q_row < 0 || q_row >= query_seq_len) {
        return;
    }
    const int active_kv_seq_len = clamp(q_logical + 1, 0, key_seq_len);

    COMPUTE_FLOAT local local_m[64];
    COMPUTE_FLOAT local local_l[64];
    COMPUTE_FLOAT8 local local_o[512];
    COMPUTE_FLOAT8 o0 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o1 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o2 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o3 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o4 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o5 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o6 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o7 = (COMPUTE_FLOAT8)0;

    COMPUTE_FLOAT m = (COMPUTE_FLOAT)-FLT_MAX;
    COMPUTE_FLOAT l = (COMPUTE_FLOAT)0;
    const int query_offset = ((b * query_seq_len + q_row) * head_num + h) * head_dim;

    for (int k = lid; k < active_kv_seq_len; k += 64) {
        const int slot = slot_table[k];
        if (slot < 0 || slot >= key_max_len) {
            continue;
        }
        COMPUTE_FLOAT score = (COMPUTE_FLOAT)0;
        const int key_offset = ((slot * batch + b) * kv_head_num + kvh) * head_dim;
        for (int d4 = 0; d4 < 64; d4 += 4) {
            COMPUTE_FLOAT4 qv = CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_offset + d4));
            COMPUTE_FLOAT4 kv = CONVERT_COMPUTE_FLOAT4(vload4(0, key_cache + key_offset + d4));
            score += dot(qv, kv);
        }
        score *= (COMPUTE_FLOAT)scale;
        const COMPUTE_FLOAT new_m = fmax(m, score);
        const COMPUTE_FLOAT alpha = l > (COMPUTE_FLOAT)0 ? exp(m - new_m) : (COMPUTE_FLOAT)0;
        const COMPUTE_FLOAT beta = exp(score - new_m);
        const int value_offset = ((b * kv_head_num + kvh) * key_max_len + slot) * head_dim;
        const COMPUTE_FLOAT8 beta8 = (COMPUTE_FLOAT8)beta;
        o0 = o0 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset));
        o1 = o1 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 8));
        o2 = o2 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 16));
        o3 = o3 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 24));
        o4 = o4 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 32));
        o5 = o5 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 40));
        o6 = o6 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 48));
        o7 = o7 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 56));
        l = l * alpha + beta;
        m = new_m;
    }
    local_m[lid] = m;
    local_l[lid] = l;
    local_o[lid * 8] = o0;
    local_o[lid * 8 + 1] = o1;
    local_o[lid * 8 + 2] = o2;
    local_o[lid * 8 + 3] = o3;
    local_o[lid * 8 + 4] = o4;
    local_o[lid * 8 + 5] = o5;
    local_o[lid * 8 + 6] = o6;
    local_o[lid * 8 + 7] = o7;
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
            for (int d8 = 0; d8 < 8; ++d8) {
                local_o[lid * 8 + d8] =
                    local_o[lid * 8 + d8] * a + local_o[(lid + stride) * 8 + d8] * b_scale;
            }
            local_l[lid] = l0 * a + l1 * b_scale;
            local_m[lid] = merged_m;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const COMPUTE_FLOAT denom = local_l[0] > (COMPUTE_FLOAT)0 ? local_l[0] : (COMPUTE_FLOAT)1;
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim;
        for (int d8 = 0; d8 < 8; ++d8) {
            vstore8(CONVERT_FLOAT8(local_o[d8] / (COMPUTE_FLOAT8)denom), 0,
                    output + output_offset + (d8 << 3));
        }
    }
}

__kernel void decode_causal_attention_row32(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query, // [batch, query_seq_len, head_num, head_dim]
                              __global const FLOAT *key_cache, // [max_slots, batch, kv_head_num, head_dim]
                              __global const FLOAT *value_cache, // [batch, kv_head_num, max_slots, head_dim]
                              __global const int *slot_table, // [key_seq_len]
                              __global const int *sparse_query, // [output_seq_len], used when sparse_query_active != 0
                              __global FLOAT *output, // [batch, output_seq_len, head_num, head_dim]
                              __private const float scale,
                              __private const int batch,
                              __private const int query_seq_len,
                              __private const int output_seq_len,
                              __private const int base_logical,
                              __private const int sparse_query_active,
                              __private const int query_rows_are_full,
                              __private const int key_seq_len,
                              __private const int key_max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int lid = get_local_id(0);
    if (get_local_size(0) != 32 || lid >= 32 || head_dim != 64) {
        return;
    }
    const int q_index = y;
    if (q_index >= output_seq_len) {
        return;
    }

    const int b = z / head_num;
    const int h = z - b * head_num;
    if (b >= batch) {
        return;
    }
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }

    const int q_logical = sparse_query_active != 0 ? sparse_query[q_index] : base_logical + q_index;
    const int q_row = query_rows_are_full ? q_logical : q_index;
    if (q_row < 0 || q_row >= query_seq_len) {
        return;
    }
    const int active_kv_seq_len = clamp(q_logical + 1, 0, key_seq_len);

    COMPUTE_FLOAT local local_m[32];
    COMPUTE_FLOAT local local_l[32];
    COMPUTE_FLOAT8 local local_o[256];
    COMPUTE_FLOAT8 o0 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o1 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o2 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o3 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o4 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o5 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o6 = (COMPUTE_FLOAT8)0;
    COMPUTE_FLOAT8 o7 = (COMPUTE_FLOAT8)0;

    COMPUTE_FLOAT m = (COMPUTE_FLOAT)-FLT_MAX;
    COMPUTE_FLOAT l = (COMPUTE_FLOAT)0;
    const int query_offset = ((b * query_seq_len + q_row) * head_num + h) * head_dim;

    for (int k = lid; k < active_kv_seq_len; k += 32) {
        const int slot = slot_table[k];
        if (slot < 0 || slot >= key_max_len) {
            continue;
        }
        COMPUTE_FLOAT score = (COMPUTE_FLOAT)0;
        const int key_offset = ((slot * batch + b) * kv_head_num + kvh) * head_dim;
        for (int d4 = 0; d4 < 64; d4 += 4) {
            COMPUTE_FLOAT4 qv = CONVERT_COMPUTE_FLOAT4(vload4(0, query + query_offset + d4));
            COMPUTE_FLOAT4 kv = CONVERT_COMPUTE_FLOAT4(vload4(0, key_cache + key_offset + d4));
            score += dot(qv, kv);
        }
        score *= (COMPUTE_FLOAT)scale;
        const COMPUTE_FLOAT new_m = fmax(m, score);
        const COMPUTE_FLOAT alpha = l > (COMPUTE_FLOAT)0 ? exp(m - new_m) : (COMPUTE_FLOAT)0;
        const COMPUTE_FLOAT beta = exp(score - new_m);
        const int value_offset = ((b * kv_head_num + kvh) * key_max_len + slot) * head_dim;
        const COMPUTE_FLOAT8 beta8 = (COMPUTE_FLOAT8)beta;
        o0 = o0 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset));
        o1 = o1 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 8));
        o2 = o2 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 16));
        o3 = o3 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 24));
        o4 = o4 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 32));
        o5 = o5 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 40));
        o6 = o6 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 48));
        o7 = o7 * alpha + beta8 * CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_offset + 56));
        l = l * alpha + beta;
        m = new_m;
    }
    local_m[lid] = m;
    local_l[lid] = l;
    local_o[lid * 8] = o0;
    local_o[lid * 8 + 1] = o1;
    local_o[lid * 8 + 2] = o2;
    local_o[lid * 8 + 3] = o3;
    local_o[lid * 8 + 4] = o4;
    local_o[lid * 8 + 5] = o5;
    local_o[lid * 8 + 6] = o6;
    local_o[lid * 8 + 7] = o7;
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
            for (int d8 = 0; d8 < 8; ++d8) {
                local_o[lid * 8 + d8] =
                    local_o[lid * 8 + d8] * a + local_o[(lid + stride) * 8 + d8] * b_scale;
            }
            local_l[lid] = l0 * a + l1 * b_scale;
            local_m[lid] = merged_m;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const COMPUTE_FLOAT denom = local_l[0] > (COMPUTE_FLOAT)0 ? local_l[0] : (COMPUTE_FLOAT)1;
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim;
        for (int d8 = 0; d8 < 8; ++d8) {
            vstore8(CONVERT_FLOAT8(local_o[d8] / (COMPUTE_FLOAT8)denom), 0,
                    output + output_offset + (d8 << 3));
        }
    }
}
__kernel void matmul_qkv_decode_b8(GLOBAL_SIZE_2_DIMS
                              __global const FLOAT *qk, // qk [1 head_num qk_seq_len 1]
                              __global const FLOAT *past_value, // [1 head_num max_len head_dim]
                              __global FLOAT *output, // [1 1 head_num head_dim]
                              __private const int qk_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // head_num
    
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int x8 = x << 3;
    
    const int qk_offset = y * qk_seq_len;
    const int past_offset = ((y / NUMHEAD_GROUP_SIZE) * max_len) * head_dim + x8;
    COMPUTE_FLOAT8 out0 = 0;
    #ifdef LOOP_UNROLL_4
    const int loop_end = max((qk_seq_len + 3) / 4 - 1, 0);
    for(int i = 0; i < loop_end; ++i){
        int i4 = i << 2;
        COMPUTE_FLOAT4 qk_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i4));
        
        COMPUTE_FLOAT8 past_vec0 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i4 * head_dim));
        COMPUTE_FLOAT8 past_vec1 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 1) * head_dim));
        COMPUTE_FLOAT8 past_vec2 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 2) * head_dim));
        COMPUTE_FLOAT8 past_vec3 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i4 + 3) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s1, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s2, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s3, past_vec3, out0);
    }
    for(int i = (loop_end << 2); i < qk_seq_len; ++i){
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT8 past_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i * head_dim));
        out0 = mad((COMPUTE_FLOAT8)qk_vec, past_vec, out0);
    }
    #elif (defined LOOP_UNROLL_8)
    const int loop_end = max((qk_seq_len + 7) / 8 - 1, 0);
    for(int i = 0; i < loop_end; ++i){
        int i8 = i << 3;
        COMPUTE_FLOAT8 qk_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, qk + qk_offset + i8));
        
        COMPUTE_FLOAT8 past_vec0 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i8 * head_dim));
        COMPUTE_FLOAT8 past_vec1 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 1) * head_dim));
        COMPUTE_FLOAT8 past_vec2 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 2) * head_dim));
        COMPUTE_FLOAT8 past_vec3 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 3) * head_dim));
        COMPUTE_FLOAT8 past_vec4 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 4) * head_dim));
        COMPUTE_FLOAT8 past_vec5 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 5) * head_dim));
        COMPUTE_FLOAT8 past_vec6 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 6) * head_dim));
        COMPUTE_FLOAT8 past_vec7 = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + (i8 + 7) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s1, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s2, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s3, past_vec3, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s4, past_vec4, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s5, past_vec5, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s6, past_vec6, out0);
        out0 = mad((COMPUTE_FLOAT8)qk_vec.s7, past_vec7, out0);
    }
    for(int i = (loop_end << 3); i < qk_seq_len; ++i){
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT8 past_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, past_value + past_offset + i * head_dim));
        out0 = mad((COMPUTE_FLOAT8)qk_vec, past_vec, out0);
    }
    #endif
    
    const int output_offset = y * head_dim + x8;
    vstore8(CONVERT_FLOAT8(out0), 0, output + output_offset);
}

__kernel void matmul_qkv_decode_b4(GLOBAL_SIZE_2_DIMS
                              __global const FLOAT *qk, // qk [1 head_num qk_seq_len 1]
                              __global const FLOAT *past_value, // [1 head_num max_len head_dim]
                              __global FLOAT *output, // [1 1 head_num head_dim]
                              __private const int qk_seq_len,
                              __private const int max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
                                  
    const int x = get_global_id(0); // head_dim
    const int y = get_global_id(1); // head_num
    
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int x4 = x << 2;
    
    const int qk_offset = y * qk_seq_len;
    const int past_offset = ((y / NUMHEAD_GROUP_SIZE) * max_len) * head_dim + x4;
    COMPUTE_FLOAT4 out0 = 0;
    #ifdef LOOP_UNROLL_4
    const int loop_end = max((qk_seq_len + 3) / 4 - 1, 0);
    for(int i = 0; i < loop_end; ++i){
        int i4 = i << 2;
        COMPUTE_FLOAT4 qk_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, qk + qk_offset + i4));
        
        COMPUTE_FLOAT4 past_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + i4 * head_dim));
        COMPUTE_FLOAT4 past_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i4 + 1) * head_dim));
        COMPUTE_FLOAT4 past_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i4 + 2) * head_dim));
        COMPUTE_FLOAT4 past_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i4 + 3) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s1, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s2, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s3, past_vec3, out0);
    }
    for(int i = (loop_end << 2); i < qk_seq_len; ++i){
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT4 past_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + i * head_dim));
        out0 = mad((COMPUTE_FLOAT4)qk_vec, past_vec, out0);
    }
    #elif (defined LOOP_UNROLL_8)
    const int loop_end = max((qk_seq_len + 7) / 8 - 1, 0);
    for(int i = 0; i < loop_end; ++i){
        int i8 = i << 3;
        COMPUTE_FLOAT8 qk_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, qk + qk_offset + i8));
        
        COMPUTE_FLOAT4 past_vec0 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + i8 * head_dim));
        COMPUTE_FLOAT4 past_vec1 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 1) * head_dim));
        COMPUTE_FLOAT4 past_vec2 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 2) * head_dim));
        COMPUTE_FLOAT4 past_vec3 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 3) * head_dim));
        COMPUTE_FLOAT4 past_vec4 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 4) * head_dim));
        COMPUTE_FLOAT4 past_vec5 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 5) * head_dim));
        COMPUTE_FLOAT4 past_vec6 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 6) * head_dim));
        COMPUTE_FLOAT4 past_vec7 = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + (i8 + 7) * head_dim));
        
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s0, past_vec0, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s1, past_vec1, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s2, past_vec2, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s3, past_vec3, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s4, past_vec4, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s5, past_vec5, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s6, past_vec6, out0);
        out0 = mad((COMPUTE_FLOAT4)qk_vec.s7, past_vec7, out0);
    }
    for(int i = (loop_end << 3); i < qk_seq_len; ++i){
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT4 past_vec = CONVERT_COMPUTE_FLOAT4(vload4(0, past_value + past_offset + i * head_dim));
        out0 = mad((COMPUTE_FLOAT4)qk_vec, past_vec, out0);
    }
    #endif
    
    const int output_offset = y * head_dim + x4;
    vstore4(CONVERT_FLOAT4(out0), 0, output + output_offset);
}
