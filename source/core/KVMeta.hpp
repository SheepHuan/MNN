//
//  KVMeta.hpp
//  MNN
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef KVMeta_hpp
#define KVMeta_hpp

#include <cstdint>
#include <vector>
#include <string>
#include <MNN/MNNDefine.h>

namespace MNN {

struct KVMeta {
    enum {
        NoChange,
        PendingWrite,
        PendingRead,
        PendingReadSegments
    } file_operation;
    enum {
        KeyRopePositionEncoded,
        KeyRopeCanonicalNoRope
    } key_rope_operation;
    enum {
        RopePairingHalf
    } rope_pairing_type;
    struct PrefixSegment {
        std::string cache_name;
        int token_count = 0;
        // 保存 token ids 方便上层恢复 history；CPU 后端拼 KV 时只使用 token_count。
        std::vector<int> token_ids;
        // prefix key 在磁盘中的 RoPE 状态。canonical_no_rope 表示保存前已逆 RoPE。
        int key_rope_state = KeyRopePositionEncoded;
        int rope_dim = 0;
        float rope_theta = 10000.0f;
        int rope_pairing = RopePairingHalf;
        int source_position_base = 0;
        // Backend-native document cache format. GPU direct_segments must not
        // silently parse another backend's packed KV layout.
        std::string backend;
        std::string layout;
        std::string dtype;
        int page_size = 0;
    };
    size_t block = 4096;
    // previous/remove/add 描述运行时 KV 历史：已有长度、待删除长度、本次新增 token 数。
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    // file_name/file_flag 用于单 prefix cache 的磁盘读写状态。
    std::string file_name = "";
    // Prefix cache directory is runtime metadata for backends that materialize
    // direct_segments themselves, e.g. CUDA PrefixAttention.
    std::string prefix_cache_dir = ".cache/kvshare/prefixcache";
    int file_flag = NoChange;
    // 磁盘 prefix cache 的真实 token 长度，不等于 .k/.v 文件按 pack 对齐后的容量。
    int seqlen_in_disk = 0;
    // 每个 Attention layer 依次读取 <cache_name>_<layer>.k/v。
    int layer_index = 0;
    int layer_nums = 0;
    // 多文档直读 prefix：按 segment 顺序读取真实 token_count 并拼成连续运行时 KV。
    std::vector<PrefixSegment> prefix_segments;
    int segment_total_tokens = 0;
    // direct_segments request-controlled prefetch:
    // false keeps the path where each PrefixAttention layer loads KV just
    // before its attention compute; true submits host file prefetch for all
    // backends. CUDA and OpenCL additionally enqueue layer KV uploads on a
    // backend copy stream/queue and wait by event per layer when available.
    bool prefix_device_prefetch = false;
    uint64_t prefix_request_id = 0;
    int prefix_prompt_token_count = 0;
    // RoPE 元信息由 Llm 从模型 config 注入，CPUKVCacheManager 只按这些参数处理 packed key。
    int key_rope_state = KeyRopePositionEncoded;
    int rope_dim = 0;
    float rope_theta = 10000.0f;
    int rope_pairing = RopePairingHalf;
    int source_position_base = 0;
    std::vector<int> reserveHost;
    // Attention scaling override (gemma4 uses 1.0 instead of 1/sqrt(head_dim))
    float attn_scale = 0.0f; // 0 means use default 1/sqrt(head_dim)
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

using PrefixDevicePrefetchSubmitter = bool (*)(KVMeta*);

MNN_PUBLIC void registerPrefixDevicePrefetchSubmitter(PrefixDevicePrefetchSubmitter submitter);
MNN_PUBLIC bool submitPrefixDevicePrefetch(KVMeta* meta);

} // namespace MNN
#endif // KVMeta_hpp
