//
//  KVMeta.hpp
//  MNN
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef KVMeta_hpp
#define KVMeta_hpp

#include <vector>
#include <string>

namespace MNN {

struct KVMeta {
    enum {
        NoChange,
        PendingWrite,
        PendingRead
    } file_operation;
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::string file_name = "";
    int file_flag = NoChange;
    int seqlen_in_disk = 0;
    int layer_index = 0;
    int layer_nums = 0;
    std::vector<int> reserveHost;
    // Attention scaling override (gemma4 uses 1.0 instead of 1/sqrt(head_dim))
    float attn_scale = 0.0f; // 0 means use default 1/sqrt(head_dim)
    // RoPE metadata used when exporting reusable paged KV cache. PagedAttention
    // stores key cache after forward RoPE; text-cache export writes canonical
    // no-RoPE keys so later PIC requests can reapply RoPE for their slot.
    float rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;
    float rope_scaling_low_freq_factor = 1.0f;
    float rope_scaling_high_freq_factor = 4.0f;
    int rope_scaling_original_max_position_embeddings = 0;
    int max_position_embeddings = 0;
    int rope_dim = 0; // 0 means use head_dim.
    float rope_attention_scaling = 1.0f;
    std::string rope_type = "default";
    int computeReverseSize() const {
        int sum = 0;
        for (int i=0; i<n_reserve; ++i) {
            int reserveUnit = reserve[2*i+1];
            if (reserveUnit <= 0) {
                // Invalid
                return -1;
            }
            sum += reserveUnit;
        }
        return sum;
    }
    void sync() {
        int revertNumber = 0;
        for (int i=0; i<n_reserve; ++i) {
            revertNumber += reserve[2*i+1];
        }
        previous = previous - remove + add + revertNumber;
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
};

} // namespace MNN
#endif // KVMeta_hpp
