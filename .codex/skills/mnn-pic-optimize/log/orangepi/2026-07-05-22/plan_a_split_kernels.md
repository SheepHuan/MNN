# Plan A — 拆独立 QK/softmax/QKV kernel，真·参考 normal decode 做 GQA 复用

> **转交文档。** 本文件为下一会话的完整执行 plan，自包含。新会话先读
> `AGENTS.md`、`.codex/skills/mnn-pic-optimize/SKILL.md`、
> `.codex/skills/mnn-pic-optimize/references/decode_opencl_workflow.md`，再读本文件，
> 然后按 §7 实现顺序执行。

## 1. 背景：为什么做 Plan A

PIC dualgraph x0 decode 当前 artifact（commit `38ba6851` + 工作区改动）上 Qwen3-4B
ctx1024 只比 true normal 慢 ~2.5ms（157.3 vs 154.9），MiniCPM5-1B 上 PIC 已更快
（43.1 vs 50.0）。x0 的剩余 gap 在 attention kernel 内部，且所有"融合"方向的优化
（fused-append、q1-gqa 融合、slot-identity-split、lane-force、q1-identity-fused-kv）
都在 Mali 和/或 Adreno 上 A/B 退化、已硬编码 `return false`。

**根因（上一轮分析结论）**：当前 x0 走融合 flash kernel
`decode_causal_attention_hd128_transposed_k_sparse_row128`，gws z = `batch*num_head`
（每 query-head 一个 workgroup，128 lane 做 online softmax）。同 GQA group 的 4 个
query-head workgroup 各自独立读整条 K stream，没有跨 workgroup 的 K 复用 → 4× 冗余
K 流量（PMC ~2.17 MB/layer）。

**Normal decode 不融合**，拆成 5 个独立 kernel：`rearrange_k`、`matmul_qk_decode`、
`softmax_in1_buf`、`rearrange_v`、`matmul_qkv_decode_b8`。QK 是 2D 细粒度
`{UP_DIV(kvSeqlen,4), numHead}`，每 work-item 算 4 个 K 位置；GQA 复用靠**位置除法
`y/NUMHEAD_GROUP_SIZE`** —— 同 group 的 4 个 query-head 是 dim1 上的 4 个相邻
work-item，读同一份 K，靠 **L2 cache 自然吸收**。每个 kernel 都全占用，没有空闲 lane。

**Plan A 就是把 PIC x0 从"融合 flash"改成"独立 QK/softmax/QKV"，抄 normal 的并行模型，
从而拿到 GQA K 复用和 V 阶段全占用。** 代价是引入中间 `qk`/`softmax` tensor 的写读
往返（每层 ~64KB）。赌的是：GQA K 复用省的 4× K 带宽 + V 阶段全占用，能盖过中间
tensor 往返的开销。

## 2. 硬约束（来自 SKILL / AGENTS / 用户 brief，必须全程满足）

1. **不路由回退**：x=0/1/3/5/7 都走 PIC dualgraph decode-only graph
   （`llm_decode.mnn` / `mDecodeModule` / `mnn_token_id_sparse_decode`）。禁止静默回退
   主图 `llm.mnn` / `mModule` / `mModulePool`。禁止 x0 走 true normal identity path。
2. **P0 profile 判据**（正式 decode profile 必须满足，任一不满足该 run 作废）：
   - `use_decode_graph=1`
   - `pic_decode_repair_decode_graph=1`
   - `ordinary_q1=0`
   - `decode_prepare_inside_decode=0`
   - 日志无 `ERROR` / `target unavailable` / `async persistent PIC cache read failed` / `Cache invalid`
3. **PagedCache 是唯一真实 KV 工作区**。Plan A 不新增第二份 runtime KV cache、scratch
   `.k/.v`、完整 host KV staging。新增的 `qk`/`softmax` tensor 是**单层中间计算
   buffer**（同 normal 的 `mTempQK`/`mTempSoftMax`），不是 KV cache。`decodeKey` 仍是
   PagedCache 的 decode-only 派生视图，prepare 阶段填、decode 阶段只 append 当前 token。
4. **同一 family**：x0 是 active rows=1 的 Plan A 路径；x1/3/5/7 暂时保持现有
   `decode_causal_attention_hd128_transposed_k_sparse_qtile` family（qtile）。Plan A
   先只对 x0/q1 验证，不强行覆盖 x>0。A/B 变体必须用清晰 env 标签单独运行、单独报告，
   不混入默认结果。
5. **无效实现先注释、写明原因、不删上下文**（用户约束 #7 + SKILL `Decode 算子优化约束`）。
6. **修 `.cl` 后必须** `cd source/backend/opencl/execution/cl && python3 opencl_codegen.py .`
   再 `git diff --check`。
7. **Mali 不支持 QCOM record queue**（`cl_recording_qcom` 是 Adreno 专有）。
   `isUseRecordQueue()` 在 OrangePi 恒 false。Plan A 的 kernel 都走逐 enqueue，不依赖
   record queue。这是正常的，normal decode 在 Mali 上也这样。

## 3. 目标 kernel 设计（PIC 版的独立 QK/softmax/QKV）

### 3.1 不变的部分（复用现有）

- **append**：复用 `append_sparse_decode_key_value_hd128`
  （`paged_decode_attention_buf.cl:88-128`），写 `key_cache` + `value_cache` + `decode_key`
  3 store，per-KV-head。host 端 `mDecodeKeyAppendSparseKernel`
  （`PagedAttentionBufExecutionDecode.cpp:412` build、`:2330` dispatch）。**不改**。
- **decodeKey prepare**：复用 `transpose_paged_key_to_decode_key` +
  `ensureDecodeKeyReady`，prepare 阶段做，`decode_prepare_inside_decode=0`。**不改**。

### 3.2 新增的 3 个独立 kernel（写在 `paged_decode_attention_buf.cl`）

直接参考 `attention_buf.cl` 的 `matmul_qk_decode`(:1041) / `softmax_in1_buf`
(`softmax_buf.cl:15`) / `matmul_qkv_decode_b8`(:3707)，但 K 源换成 `decode_key`、
V 源换成 `value_cache`，并保留 PIC 的 sparse_query/causal 语义。

#### kernel 1: `matmul_qk_decode_pic_hd128`（2D，细粒度，GQA 靠 L2）

```c
__kernel void matmul_qk_decode_pic_hd128(GLOBAL_SIZE_2_DIMS
    __global const FLOAT *query,        // [batch, query_seq_len, head_num, 128]
    __global const FLOAT *decode_key,   // [batch, kv_head_num, 128, max_slots]
    __global FLOAT *qk,                 // [batch*head_num, key_seq_len]  (与 normal 一致)
    __global const int *sparse_query,   // [output_seq_len]  逻辑位置
    __private const float scale,
    __private const int batch,
    __private const int query_seq_len,
    __private const int output_seq_len, // = attnLen
    __private const int key_seq_len,    // = kvLen
    __private const int key_max_len,    // = maxSlots
    __private const int head_num,
    __private const int kv_head_num) {
    const int x = get_global_id(0); // UP_DIV(key_seq_len, 4)
    const int y = get_global_id(1); // batch * head_num
    DEAL_NON_UNIFORM_DIM2(x, y);
    if (head_dim != 128) return;     // 硬编码 128，避免传参
    const int x4 = x << 2;
    const int b = y / head_num;
    const int h = y - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;   // GQA 位置除法（关键）
    if (b >= batch || kvh >= kv_head_num) return;
    // 当前 query-head 的 logical position（x0: attnLen=1, q_index=0）
    const int q_index = 0;            // Plan A 先只支持 x0/q1，attnLen==1
    const int q_logical = sparse_query[q_index];
    const int causal_len = min(q_logical + 1, key_seq_len);  // causal mask
    // K base: decode_key[((b*kv_head_num+kvh)*128)] * key_max_len
    const int key_base = ((b * kv_head_num + kvh) * 128) * key_max_len;
    const int query_offset = ((b * query_seq_len + q_index) * head_num + h) * 128;
    COMPUTE_FLOAT4 out0 = (COMPUTE_FLOAT4)0;
    // 4 个 K 位置，stride-1 along max_len（decodeKey 的转置布局）
    if (x4 + 3 < causal_len) {
        for (int d = 0; d < 128; ++d) {
            COMPUTE_FLOAT qv = (COMPUTE_FLOAT)query[query_offset + d];
            COMPUTE_FLOAT4 kv = CONVERT_COMPUTE_FLOAT4(
                vload4(0, decode_key + key_base + d * key_max_len + x4));
            out0 = mad((COMPUTE_FLOAT4)qv, kv, out0);
        }
        out0 *= (COMPUTE_FLOAT4)scale;
        vstore4(CONVERT_FLOAT4(out0), 0, qk + y * key_seq_len + x4);
        return;
    }
    // tail（causal 边界，参考 matmul_qk_decode 的 vstore3/2/1 分支）
    for (int d = 0; d < 128; ++d) {
        COMPUTE_FLOAT qv = (COMPUTE_FLOAT)query[query_offset + d];
        if (x4 < causal_len)     out0.s0 += qv * (COMPUTE_FLOAT)decode_key[key_base + d*key_max_len + x4];
        if (x4+1 < causal_len)   out0.s1 += qv * (COMPUTE_FLOAT)decode_key[key_base + d*key_max_len + x4+1];
        if (x4+2 < causal_len)   out0.s2 += qv * (COMPUTE_FLOAT)decode_key[key_base + d*key_max_len + x4+2];
    }
    out0 *= (COMPUTE_FLOAT4)scale;
    // tail store（参考 attention_buf.cl:1078-1088 的 vstore3/2/1 边界处理）
    ...
}
```

**关键点**：
- 2D gws `{UP_DIV(key_seq_len,4), batch*head_num}`，每 work-item 算 4 个 K 位置。
  Qwen ctx1024: `{257, 32}` = 8224 work-item（vs 融合版 32 workgroup）。
- **GQA 复用靠 L2**：`kvh = h / NUMHEAD_GROUP_SIZE`，同 group 4 个 query-head 是 dim1
  相邻 work-item，读同一份 `decode_key[key_base + ...]`，L2 吸收。
- **causal mask**：`x4 < causal_len`（`causal_len = q_logical+1`），跳过未来 K。
- **不融合 softmax**：输出 `qk` tensor，下个 kernel 做 softmax。

#### kernel 2: softmax —— 直接复用 `softmax_in1_buf`（`softmax_buf.cl:15`）

`softmax_in1_buf` 是通用的，输入 `qk [batch*head_num, key_seq_len]`，输出同 shape。
PIC 的 causal mask 已经在 QK kernel 里用 0-score 体现（future 位置不累加，保持
`-FLT_MAX` 初始？**注意**：QK kernel 必须把 future 位置写成 `-FLT_MAX` 或让 softmax
跳过。最干净的做法：QK kernel 在 `x4+j >= causal_len` 时写 `-FLT_MAX`，softmax 正常算。
**或**：参考 normal 的做法 —— normal 的 `matmul_qk_decode` 不做 causal mask，mask 在
`attention_mask` tensor 里，softmax 之前加 mask。Plan A 简化为：QK kernel 直接对
`x4+j >= causal_len` 的位置写一个极小值（如 `-1e30`），softmax 不用改。验证时确认
softmax_in1_buf 对 `-1e30` 输入的数值稳定性。

**用现成 `softmax_in1_buf`，gws `{64, 1, batch*head_num}`，lws `{64,1,1}`。**

#### kernel 3: `matmul_qkv_decode_pic_hd128_b8`（2D，细粒度，V 全占用）

直接抄 `matmul_qkv_decode_b8`（`attention_buf.cl:3707`），V 源是 `value_cache`
（布局 `[batch, kv_head_num, max_slots, head_dim]`，和 normal 的 `past_value`
`[kv_head_num, max_len, head_dim]` 对 batch=1 等价）：

```c
__kernel void matmul_qkv_decode_pic_hd128_b8(GLOBAL_SIZE_2_DIMS
    __global const FLOAT *qk,          // [batch*head_num, qk_seq_len]  (= softmax 输出)
    __global const FLOAT *value_cache, // [batch, kv_head_num, max_slots, 128]
    __global FLOAT *output,            // [batch, output_seq_len, head_num, 128]
    __global const int *sparse_query,  // [output_seq_len]
    __private const int qk_seq_len,
    __private const int key_max_len,
    __private const int batch,
    __private const int head_num,
    __private const int kv_head_num) {
    const int x = get_global_id(0); // head_dim/8 = 16
    const int y = get_global_id(1); // batch*head_num
    DEAL_NON_UNIFORM_DIM2(x, y);
    const int x8 = x << 3;
    const int b = y / head_num;
    const int h = y - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    const int q_index = 0;                       // Plan A 先只支持 x0
    const int q_logical = sparse_query[q_index];
    const int causal_len = min(q_logical + 1, qk_seq_len);
    const int qk_offset = y * qk_seq_len;
    // value_cache: ((b*kv_head_num+kvh)*key_max_len + slot)*128, slot==logical (slot_identity)
    const int value_base = ((b * kv_head_num + kvh) * key_max_len) * 128 + x8;
    COMPUTE_FLOAT8 out0 = (COMPUTE_FLOAT8)0;
    // LOOP_UNROLL_8（参考 matmul_qkv_decode_b8:3747-3775）
    const int loop_end = max((causal_len + 7) / 8 - 1, 0);
    for (int i = 0; i < loop_end; ++i) {
        int i8 = i << 3;
        COMPUTE_FLOAT8 qk_vec = CONVERT_COMPUTE_FLOAT8(vload8(0, qk + qk_offset + i8));
        COMPUTE_FLOAT8 pv0 = CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_base + (i8+0)*128));
        ... // 8 路 mad
    }
    for (int i = (loop_end<<3); i < causal_len; ++i) {
        COMPUTE_FLOAT qk_vec = qk[qk_offset + i];
        COMPUTE_FLOAT8 pv = CONVERT_COMPUTE_FLOAT8(vload8(0, value_cache + value_base + i*128));
        out0 = mad((COMPUTE_FLOAT8)qk_vec, pv, out0);
    }
    const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * 128 + x8;
    vstore8(CONVERT_FLOAT8(out0), 0, output + output_offset);
}
```

**关键点**：
- 2D gws `{16, batch*head_num}`（head_dim=128, /8）。Qwen: `{16, 32}` = 512 work-item。
- **V 全占用**：每 work-item 算 8 个 head_dim 元素，没有 `if(lid<32)` 空闲 lane。
- **causal**：`i < causal_len`，跳过 future V。
- **GQA**：`kvh = h/NUMHEAD_GROUP_SIZE`，V 靠 L2 复用。
- **slot_identity 假设**：`slot = logical`。PagedCache 在 sparse decode-repair 下
  `slot_identity=1`（profile 已确认）。如果未来 slot≠logical，这个 kernel 会读错 V，
  需要加 slot_table 间接寻址（Plan A 阶段不做，标注 unsupported）。

### 3.3 中间 tensor 分配

参考 normal 的 `mTempQK`/`mTempSoftMax`（`AttentionBufExecution.cpp:310-311`）：

```cpp
// 在 PagedAttentionBufExecution 里新增（按 layer 共享或 per-layer，看 ensureCache 结构）
// qk tensor: [ROUND_UP(key_seq_len, 4) * batch * head_num]   // x0: attnLen=1
// softmax tensor: 同上
// 用 DYNAMIC_IN_EXECUTION 分配/释放，避免常驻显存
```

**注意**：PIC 的 `ensureCache` 是 per-layer `SharedPagedCache`。中间 tensor 可以放
`mCache` 里（和 `decodeKey` 同生命周期），或单独管理。优先放 `mCache` 以复用现有
alloc/zero 机制。大小 = `UP_DIV(maxSlots,4)*4 * batch * numHead` floats（x0 attnLen=1）。
Qwen ctx1024 maxSlots=5120: `5120*1*32*2B(fp16) = 320KB` per layer × 36 = ~11.5MB。
可接受（不是 KV cache，是 transient）。

## 4. host 端 dispatch（新增 `runDecodeCausalAttentionHD128SplitKernels`）

在 `PagedAttentionBufExecutionDecode.cpp` 新增一个方法，参考 normal 的
`decodeResize` + `onExecute` 结构。**Plan A 只在 `attnLen==1`（x0）且 env 开启时走这条路**。

### 4.1 build（`ensureDecodeCausalKernelHD128Split`，参考 `:271-350` 的 build 模式）

```cpp
// 在 ensureDecodeTransposedTemps 或新 ensure 函数里 build 3 个 kernel
mDecodeQKSplitKernel = runtime->buildKernel("paged_decode_attention_buf",
    "matmul_qk_decode_pic_hd128", options + " -DNUMHEAD_GROUP_SIZE=...", ...);
mDecodeQKVSplitKernel = runtime->buildKernel("paged_decode_attention_buf",
    "matmul_qkv_decode_pic_hd128_b8", options + " -DNUMHEAD_GROUP_SIZE=...", ...);
// softmax 直接复用 attention_buf 的 softmax_in1_buf（跨 .cl 文件 build）
mDecodeSoftmaxSplitKernel = runtime->buildKernel("softmax_buf", "softmax_in1_buf",
    std::set<std::string>{"-DSOFTMAX_LOCAL_SIZE=64"}, ...);
```

### 4.2 dispatch（`runDecodeCausalAttentionHD128SplitKernels`）

```cpp
ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128SplitKernels(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
    int kvLen, int attnLen, int layerIndex) {
    // Plan A: 只支持 attnLen==1 (x0)
    if (attnLen != 1) return INVALID_VALUE;
    auto query = inputs[0]; auto output = outputs[0];
    // 1. append（复用现有 mDecodeKeyAppendSparseKernel，或不重复 —— append 已在主路径做？）
    //    注意：当前 runDecodeCausalAttentionHD128TransposedKSparse 内部做了 append。
    //    Plan A 也要先 append，再 QK/softmax/QKV。复用 append 逻辑。
    ...
    // 2. QK
    std::vector<uint32_t> qkGws = {UP_DIV(kvLen,4), batch*head_num};
    auto qkLws = localWS2DDefault(qkGws, maxWg, runtime, "matmul_qk_decode_pic_hd128", ...);
    setArgs(mDecodeQKSplitKernel, query, mCache->decodeKey, mTempQK, mCache->sparseQuery,
            scale, batch, query_seq_len=1, output_seq_len=1, kvLen, maxSlots, numHead, kvNumHead);
    run2DKernelDefault(mDecodeQKSplitKernel, qkGws, qkLws, runtime, ...);
    // 3. softmax（复用 softmax_in1_buf）
    std::vector<uint32_t> smGws = {64, 1, batch*head_num};
    std::vector<uint32_t> smLws = {64, 1, 1};
    setArgs(mDecodeSoftmaxSplitKernel, mTempQK, mTempSoftmax, inside=1, outside=batch*head_num, kvLen);
    run3DKernelDefault(mDecodeSoftmaxSplitKernel, smGws, smLws, runtime, ...);
    // 4. QKV
    std::vector<uint32_t> qkvGws = {UP_DIV(128,8)=16, batch*head_num};
    auto qkvLws = localWS2DDefault(qkvGws, maxWg, runtime, "matmul_qkv_decode_pic_hd128_b8", ...);
    setArgs(mDecodeQKVSplitKernel, mTempSoftmax, mCache->value, output, mCache->sparseQuery,
            kvLen, maxSlots, batch, numHead, kvNumHead);
    run2DKernelDefault(mDecodeQKVSplitKernel, qkvGws, qkvLws, runtime, ...);
    // 5. rank（x0 通常不武装，复用 runDecodeAttentionRankCaptureOpenCL）
    runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    return NO_ERROR;
}
```

### 4.3 dispatch 入口（`PagedAttentionBufExecution.cpp:1626` 附近）

```cpp
// 在 runDecodeCausalAttentionHD128TransposedKSparse 调用之前，加 Plan A 分支
const bool planAEnabled = attnLen == 1 && _decodeRepairSplitKernelsEnabled();  // 新 env
if (planAEnabled) {
    auto splitErr = runDecodeCausalAttentionHD128SplitKernels(inputs, outputs, kvLen, attnLen, layerIndex);
    if (splitErr == NO_ERROR) return NO_ERROR;
    // 失败则 fall through 到原 transposed-K sparse path
}
auto transposedSparseErr = runDecodeCausalAttentionHD128TransposedKSparse(...);
```

### 4.4 env gate

```cpp
// PagedAttentionBufExecution.cpp 顶部，参考 _decodeRepairQ1GqaEnabled 模式
static bool _decodeRepairSplitKernelsEnabled() {
    // Plan A: 拆独立 QK/softmax/QKV，参考 normal decode 的并行模型做 GQA L2 复用。
    // 默认 OFF，显式 A/B。Promotion 需 3 模型 × ctx512/1024 × x0 不退化。
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS", false);
    return enabled;
}
```

## 5. 精度与正确性

- **slot_identity 假设**：Plan A 的 QKV kernel 假设 `slot = logical`（读 `value_cache[(...)*max_len + logical]`）。这在 PIC sparse decode-repair 下成立（profile 确认 `slot_identity=1`）。**注意：现有代码里 `slot_identity` 是硬假设，没有运行时检查函数**（`mLastDecodeKeySlotIdentity` 只是 profile 标记，在 `PagedAttentionBufExecutionDecode.cpp:670` 和 `DecodeRepairSlotIdentity.cpp:226` 无条件设 true；grep `_slotTableIsIdentity` 无结果 —— 不要引用这个不存在的函数）。Plan A 同样硬假设 slot_identity 成立；sparse decode-repair 路径下它成立（append kernel 也硬编码 `slot=logical`，见 `paged_decode_attention_buf.cl:118`）。如果未来 PagedCache 引入真 paging，Plan A 和现有 sparse 路径都会一起坏，不是 Plan A 独有问题。
- **causal mask 数值稳定性**：QK kernel 对 future 位置写 `-FLT_MAX`。`softmax_in1_buf`（`softmax_buf.cl:15`）先算 `max`，future 位置的 `-FLT_MAX` 不会成为 max（除非整行都是 future，不会发生），`exp(-FLT_MAX - max) ≈ 0`，**数值稳定**（已确认 softmax 实现是 max-subtract 形式）。不必用 `-1e30`，直接 `-FLT_MAX`。
- **输出对比**：Plan A 路径的 decode 输出 token 必须和默认 transposed-K sparse 路径**逐 token 一致**（greedy）。用 `pic_llm_demo` 跑同一 prompt、同 seed，对比生成的 token id 序列。这是正确性硬门槛。

## 6. 验证流程（严格按 decode_opencl_workflow.md）

### Step 1: 改 .cl + opencl_codegen + build

```bash
cd /root/code/kvshare-edge/impl/MNN
( cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . )
git diff --check

JOBS=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

file .cache/output/mnn/artifacts/orangepi5plus/bin/pic_server \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

### Step 2: profile smoke（profile ON，确认 Plan A kernel 命中 + P0）

```bash
RUN_ID=planA_profile_smoke_orangepi_qwen_ctx1024_x0_<date>
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS=1 \
MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=200 \
MNN_PAGED_ATTENTION_PROFILE=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_DECODE_REPAIR_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 1024 \
  --pic-repair-tokens 0 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --skip-normal \
  --run-id "${RUN_ID}"
```

**P0 必须全部满足**（见 §2.2）。额外确认：
- 日志出现 `op=matmul_qk_decode_pic_hd128` / `op=matmul_qkv_decode_pic_hd128_b8` / `op=softmax_in1_buf`
- `decode_prepare_inside_decode=0`
- `use_decode_graph=1` `pic_decode_repair_decode_graph=1` `ordinary_q1=0`

```bash
RUN_ID=<上面的>
ssh orangepi@192.168.101.113 \
  "grep -R 'decode_prepare_inside_decode=1' -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/${RUN_ID}/pic_server_*.log || true; \
   grep -R 'ERROR\|target unavailable\|async persistent PIC cache read failed\|Cache invalid' -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/${RUN_ID}/pic_server_*.log || true"
# 两个 grep 都必须无输出
```

### Step 3: 正确性（输出 token 一致性）

跑同一 prompt + greedy，对比 Plan A vs 默认路径生成的 token 序列：

```bash
# 默认路径
PIC_SWEEP_SERVER_ENV_EXTRA='' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 512 \
  --pic-repair-tokens 0 --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 0 --skip-normal \
  --run-id planA_correctness_default_<date>

# Plan A 路径
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 512 \
  --pic-repair-tokens 0 --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 0 --skip-normal \
  --run-id planA_correctness_split_<date>
```

对比两边的生成 token id 序列（从 `pic_decode_*.csv` 或 server log 的
`output_tokens` 提取）。**必须完全一致**。不一致则 Plan A 有精度 bug，先修精度再谈性能。

### Step 4: formal TPOT A/B（profile OFF，这是有效性判据）

```bash
# OLD baseline（默认 transposed-K sparse）
PIC_SWEEP_SERVER_ENV_EXTRA='' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b --contexts 512,1024 \
  --pic-repair-tokens 0,1 --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --skip-normal \
  --run-id decode_ab_old_splitkernels_orangepi_<date>

# NEW Plan A（env 开启）
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b --contexts 512,1024 \
  --pic-repair-tokens 0,1 --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --skip-normal \
  --run-id planA_new_orangepi_<date>
```

**注意**：Plan A 先只对 `repair_tokens=0`（x0/attnLen=1）生效，`repair_tokens=1`（x1）
仍走原 qtile 路径（用于对比 x1 是否受 Plan A env 影响 —— 应该不受，因为 `attnLen!=1`
时 Plan A return INVALID_VALUE fall through）。CSV 里 `repair_tokens=0` 行对比 Plan A
vs old，`repair_tokens=1` 行应基本持平（验证 env gate 不污染 x1）。

### Step 5: 成功判据

- **正确性**：Plan A 路径生成 token 与默认路径逐 token 一致（greedy）。
- **P0**：所有 run 满足 §2.2。
- **有效性**：Plan A 的 `repair_tokens=0` TPOT ≤ old 的 `repair_tokens=0` TPOT，在
  3 模型 × ctx512/1024 上**至少不退化**（`new_ms <= old_ms`）。目标是 Qwen ctx1024
  x0 从 157ms 往 154.9ms（true normal）方向压。
- **x1 不受影响**：`repair_tokens=1` 行 Plan A vs old 持平（证明 env gate 干净）。
- **如果退化**：按硬约束 #5，**不删代码**，把 Plan A 的 dispatch 用 `#if 0` 注释，
  `_decodeRepairSplitKernelsEnabled` 保持 `return false`，在 §8 记录 A/B 数字和退化原因。

## 7. 实现顺序（新会话按此执行）

1. **读文档**：`AGENTS.md`、`SKILL.md`、`decode_opencl_workflow.md`、本文件、
   上一轮 `log/orangepi/2026-07-05-22/README.md` + `context.md`。
2. **git status**：确认工作区干净（除已有 staged 改动），不覆盖用户改动。
3. **写 kernel**：在 `paged_decode_attention_buf.cl` 末尾加 3 个 kernel
   （§3.2 的 kernel 1/3，kernel 2 复用 `softmax_in1_buf`）。注意 `NUMHEAD_GROUP_SIZE`
   宏要通过 build options 传（参考 `:436` 现有 `-DNUMHEAD_GROUP_SIZE=`）。
4. **opencl_codegen + git diff --check**。
5. **host 端**：在 `PagedAttentionBufExecutionDecode.cpp` 加 `ensureDecodeCausalKernelHD128Split`
   （build 3 kernel）+ `runDecodeCausalAttentionHD128SplitKernels`（dispatch）。在
   `PagedAttentionBufExecution.cpp:1626` 加 env-gated 分支。在 `PagedAttentionBufExecution.hpp`
   加 kernel成员 + 中间 tensor 成员。加 `_decodeRepairSplitKernelsEnabled` env reader。
6. **中间 tensor**：在 `SharedPagedCache`（`PagedAttentionBufExecution.hpp`）加
   `mTempQK` / `mTempSoftmax`，在 `ensureCache`（`:762` 附近）按 `maxSlots*batch*numHead`
   分配、zero。用 `DYNAMIC_IN_EXECUTION` 或 STATIC（看哪个不破坏现有 alloc 语义）。
7. **build + sync**（§6 Step 1）。
8. **profile smoke + P0**（§6 Step 2）。如果 Plan A kernel 没命中或 P0 挂，先修。
9. **正确性**（§6 Step 3）。token 不一致就修精度（causal mask、softmax 数值、
   slot_identity 检查）。
10. **formal A/B**（§6 Step 4）。
11. **判定 + 记录**（§6 Step 5 + §8）。有效则保留 env-gated（默认 off，等推广决策），
    无效则 `#if 0` 注释 + 记录。**不要默认开启进生产路径**，必须先 3 模型全 sweep 不退化。

## 8. 记录要求

在 `.codex/skills/mnn-pic-optimize/log/orangepi/<新小时目录>/` 写：
- `README.md`：短摘要（Plan A 是否有效、A/B 数字、是否 promote）。
- `context.md`：完整命令、CSV 路径、profile 输出、正确性对比、退化原因（若退化）。
- 把稳定结论回收到 SKILL.md 的 `Decode 算子优化约束` 段（Plan A 是否成为第 6 个
  tried-and-failed，或成为新的可推广路径）。

## 9. 风险预判（基于 5 个前作的退化模式）

| 风险 | 概率 | 缓解 |
|---|---|---|
| 中间 qk/softmax tensor 往返吃掉 GQA 收益 | 中 | 先测 Qwen ctx1024（GQA group=4，收益最大）；若 Qwen 持平/小赢、MiniCPM（group=8）退化，说明 tensor 往返主导，记录为 tried-and-failed |
| softmax_in1_buf 对 `-1e30` 数值不稳 | 低 | 先 `-1e30` 测；不稳改 `-1e9` 或加 mask kernel |
| slot_identity 不成立导致 V 读错 | 低 | sparse decode-repair 下硬成立（append kernel 也硬编码 slot=logical）；无运行时检查函数，Plan A 同样硬假设。若未来引入真 paging 需整体改 |
| L2 在 Mali-G610 上不吸收 GQA 复用（K ~2.17MB > 1MB per-core L2） | 中高 | 这正是 Plan A 的核心赌点。若 PMC 显示 K external read 没降到 ~0.55MB，说明 L2 没吸收，Plan A 无效，记录 |
| env gate 不干净污染 x1 | 低 | `attnLen!=1` 时 `return INVALID_VALUE` fall through；A/B 对比 `repair_tokens=1` 行应持平 |

## 10. 一句话目标

**把 PIC x0 的 attention 从"融合 flash + 每 query-head 独立读 K"改成"独立 QK/softmax/QKV
+ GQA 位置除法 + L2 复用"，赌 GQA K 复用省的 4× 带宽 + V 全占用能盖过中间 tensor 往返。
用 `MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS=1` 显式 A/B，3 模型不退化才考虑推广。**

---

## 附：关键 file:line 速查

```text
当前 x0 dispatch 入口:
  PagedAttentionBufExecution.cpp:1626  (调用 runDecodeCausalAttentionHD128TransposedKSparse)
  PagedAttentionBufExecutionDecode.cpp:2196  (定义)
  PagedAttentionBufExecutionDecode.cpp:2240-2260  (q1 kernel 选择)
  PagedAttentionBufExecutionDecode.cpp:2330-2367  (append dispatch)
  PagedAttentionBufExecutionDecode.cpp:2374-2418  (attention gws/lws)

append kernel:
  paged_decode_attention_buf.cl:88-128  (append_sparse_decode_key_value_hd128)
  PagedAttentionBufExecutionDecode.cpp:412-413  (build)
  PagedAttentionBufExecution.hpp:187  (mDecodeKeyAppendSparseKernel)

decodeKey prepare (prepare 阶段, 不计 decode):
  PagedAttentionBufExecutionDecode.cpp:653-751  (ensureDecodeKeyReady)
  paged_decode_attention_buf.cl:21-49  (transpose_paged_key_to_decode_key)

参考的 normal decode kernel:
  attention_buf.cl:1041  matmul_qk_decode       (2D, GQA 位置除法)
  attention_buf.cl:3707  matmul_qkv_decode_b8   (2D, V 全占用)
  softmax_buf.cl:15      softmax_in1_buf        (64-lane softmax)
  attention_buf.cl:398   rearrange_k            (PIC 用 decodeKey 替代，不需要)
  AttentionBufExecution.cpp:310-311  mTempQK/mTempSoftMax 分配
  AttentionBufExecution.cpp:1466-1500  softmax build/dispatch

env gate 模式参考:
  PagedAttentionBufExecution.cpp:282-294  (_decodeRepairQ1GqaEnabled 等已禁用的 env)

SharedPagedCache 结构 (加中间 tensor 的地方):
  PagedAttentionBufExecution.hpp:~24  (key/value/decodeKey/sparseQuery 成员)
  PagedAttentionBufExecution.cpp:761-800  (ensureCache 分配 + zero)

slot_identity 检查:
  无运行时检查函数。`mLastDecodeKeySlotIdentity` 是 profile 标记（Decode.cpp:670 无条件 true）。
  Plan A 硬假设 slot_identity（和现有 sparse 路径一致，append kernel :118 硬编码 slot=logical）。

profile/P0/PMC 流程:
  .codex/skills/mnn-pic-optimize/references/decode_opencl_workflow.md
  .codex/skills/mnn-pic-optimize/references/orangepi_opencl_pmc_profile.md
```
