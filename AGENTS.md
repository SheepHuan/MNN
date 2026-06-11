# MNN Agent Guide

## 范围

本目录是本 MNN 仓库。Codex 只使用本仓库 `.codex/skills/` 下的 MNN 专属 skills。

MNN 专属 Codex skills 由本目录维护：

```text
.codex/skills/mnn-add-new-op/SKILL.md  新增或扩展 MNN 算子
.codex/skills/mnn-build-artifacts/SKILL.md  MNN Jetson/x64 CUDA/LLM 产物构建与检查
.codex/skills/mnn-llm-bench/SKILL.md  运行和诊断 MNN llm_bench / demo，比较普通 LLM 与 PIC/PagedAttention 输出
.codex/skills/mnn-llm-export/SKILL.md  导出 MNN LLM 或 prefixllm 模型
.codex/skills/mnn-ops-bench/SKILL.md  CUDA 单算子精度/性能测试
.codex/skills/mnn-opt-ops/SKILL.md  CPU/CUDA 算子优化和 Jetson 验证流程
.codex/skills/mnn-opencl-pic-attention/SKILL.md  OpenCL PIC/PagedAttention attention 计算优化与 score layer/sparse 边界分析
.codex/skills/mnn-pic-benchmark/SKILL.md  本机交叉编译 MNN PIC server 产物、推到 Jetson 启服务，并从本机跑数据集 benchmark
.codex/skills/mnn-support-new-llm/SKILL.md  在 transformers/pic_llm 中适配新 LLM 模型
```

上游 MNN 原生 skills 仍保留在 `skills/`，不要和 Codex skills 混放。Codex 不打开、不阅读、不引用 `skills/` 下的内容；即使任务看起来匹配，也不要读取 `skills/*/SKILL.md` 或把其中内容作为依据。需要新增或调整工作流时，在 `.codex/skills/` 中维护对应 Codex skill。

## 进入本目录后

1. 先读本 MNN 仓库根目录 `AGENTS.md`。
2. 修改 MNN 源码前先在本仓库运行：

```bash
git status --short
```

不要覆盖用户已有改动。

3. 仓库级构建、输出同步和测试命令默认从本 MNN 仓库根目录执行。
4. 构建产物、模型缓存、导出结果和临时文件放在本仓库 `.cache/`、`output/` 或用户指定的本地目录，不要提交到本仓库。
5. Python 导出、模型分析和相关测试默认使用 `kvshare-edge` conda 环境，例如 `conda run -n kvshare-edge python ...`；不要直接用系统 `python`/`python3` 或 base 环境跑 exporter。
6. 本机编译、交叉编译和构建验证如果没有用户显式指定 `JOBS` / `--parallel`，默认只使用当前进程可用 CPU 总数的一半，最少为 1；不要默认跑满全部 CPU。

## 何时读 skill

- MNN 构建、重编 CUDA MNN、CUDA arch、Jetson/x64 MNN CUDA/LLM 产物构建、install/output 检查或残留构建进程诊断：读 `.codex/skills/mnn-build-artifacts/SKILL.md`。
- 新增或扩展 MNN 算子、schema/op type/shape/backend/注册表、`PrefixAttention`：读 `.codex/skills/mnn-add-new-op/SKILL.md`。
- CUDA 单算子 direct-op 精度/性能测试、`test/bench_ops/cuda` 或 CUDA event kernel 耗时统计：读 `.codex/skills/mnn-ops-bench/SKILL.md`。
- CPU/CUDA 算子性能优化、分析算子慢因，或在本机交叉编译产物并推到 Jetson 验证：读 `.codex/skills/mnn-opt-ops/SKILL.md`。
- OpenCL PIC/PagedAttention attention 计算优化、OrangePi OpenCL cacheblend/epic/kvshare sparse prefill、score layer 前后 full compute 与 sparse compute 边界分析、`PagedAttentionBufExecution.cpp` / `attention_buf.cl` / `softmax_buf.cl` 逐层热点定位：读 `.codex/skills/mnn-opencl-pic-attention/SKILL.md`。
- 从 ModelScope 或 Hugging Face/Transformers 格式模型导出 MNN LLM、PIC LLM PagedAttention 或 PrefixLLM/PrefixAttention 模型：读 `.codex/skills/mnn-llm-export/SKILL.md`。
- 运行 `llm_bench` / `llm_demo` / `pic_llm_bench` / `pic_llm_demo`、测试 CUDA/OpenCL/Vulkan/CPU LLM 推理、比较普通 MNN LLM 与 PIC/PagedAttention LLM 输出、确认 GPU 后端注册、检查 execution class 日志、排查后端回退，或围绕 LLM bench 采集 DF power / Jetson `tegrastats` 功耗：读 `.codex/skills/mnn-llm-bench/SKILL.md`。
- 本机交叉编译 MNN `pic_server` / `libpic_llm` / CUDA artifact，推送到 Jetson，远端启动 MNN PIC server，并从本机运行 `impl/pic_bench/cli.py` 数据集 benchmark 向 Jetson 发请求：读 `.codex/skills/mnn-pic-benchmark/SKILL.md`。
- 新增、适配或诊断一个尚未支持的 Hugging Face/ModelScope LLM 或多模态 LLM 模型，并修改 `transformers/pic_llm` 的 mapper/config/model/vision/audio/export 流程：读 `.codex/skills/mnn-support-new-llm/SKILL.md`。

如果任务没有对应的 `.codex/skills/` 覆盖，按本文件、仓库源码和用户上下文处理，不要读取 `skills/` 下的内容。不要批量阅读无关文档；按当前任务打开需要的文件即可。

## LLM 输出对比约定

比较普通 MNN LLM 和 PIC/PagedAttention LLM 的 prompt 输出精度时，必须使用两个模式各自真实导出的模型目录：

```text
普通 MNN LLM:       .cache/mnn-llm-export/<model>/config.json  用 llm_demo 跑
PIC/PagedAttention: .cache/weight/<model>/config.json           用 pic_llm_demo 跑
```

不要把 `.cache/weight/` 下的 PIC 模型复制或改配置当作普通模型对比；普通模型以 `.cache/mnn-llm-export/` 下的正常导出为准。两边用同一个 prompt、相同 backend / precision / memory / sampler 配置，并分别从各自模型目录执行 demo。

`--skip_weight` 导出的模型只用于检查导出流程和图结构，不作为正确性或性能测试产物。尤其 GLM / GLM-Edge 这类 `tie_word_embeddings=false` 的模型，不能因为 skip-weight skeleton 里的空/meta tensor 看起来相等就把 `tie_word_embeddings` / `tie_embeddings` 写成 true；正确性测试必须使用真实 embedding 文件和完整 `llm.mnn.weight`。历史上 OrangePi GLM PIC skiptest 产物曾因为错误 tied embedding 和 EOF 后 lm_head offset 输出 NUL/`APP`，修复后的判据是普通/PIC demo 都能输出自然语言，再进入 PIC server full-compute/full-reuse/cacheblend/epic 测试。

## PIC/PagedAttention 当前实现思路与关键设计

当前实现把 PIC/PagedAttention 明确分成两条边界：持久 text cache 构建边界和请求内推理边界。`POST /v1/prefill/text` 是唯一允许把文档 KV 写成持久 `.k/.v` 的入口；后续 chat、full-reference、scoring、sparse recompute 都围绕当前请求的 PagedCache、slot table 和 backend device buffer 执行，不再生成 chat/scoring scratch `.k/.v`，也不通过读盘临时 reference 来绕过 PagedCache。

持久 KV 使用 backend 无关的 raw 表示：每层 key/value 分文件保存，旁边的 `.json` sidecar 记录真实 `batch/kv_heads/head_dim/dtype_bytes`、layout 和 RoPE 状态。写盘前 key 被规范化成 `canonical_no_rope`，value 保持原布局；hydrate 到请求 PagedCache 时，CPU/CUDA/OpenCL PagedAttention 按当前 logical slot 对 key 重新施加 RoPE，再写入 paged KV slots。这样同一份 text cache 可以安全复用于不同请求、不同 suffix 和不同 logical position。

运行时以 PagedCache 作为唯一真实 KV 交换层。`full-compute` 计算 prelude + PIC tokens + suffix，并把当前请求 K/V 写入 PagedCache 后再由 PagedAttention 从 slot table 读回；`full-reuse` 只 hydrate 持久 PIC KV 并计算 suffix，不做 scoring、不做 sparse recompute，也不默认重算最后一个 PIC token；`epic`、`cacheblend`、`kvshare` 都是 sparse prefill：在请求内完成 reference/scoring/top-k 选择，然后只重算被选中的 PIC token，其余 PIC KV 由持久 cache hydrate 得到。

scoring 设计上坚持“请求内 native”语义。`cacheblend` / `delta-v` 使用 score layer 上 full-reference value 与 cached PIC value 的差异选 top-ratio token；`kvshare` / `delta-a` 复用 sparse prefill plan，但当前 C++ 侧没有 HF autograd，评分使用 full-vs-cached K/V delta influence proxy，并在 metadata 中明确标注和 Python `attention_output` gradient influence 的差异。CPU backend 可以在 host 侧计算 score/top-k，但数据源仍必须来自当前请求 PagedCache/slot table 与持久 cache；CUDA/OpenCL backend 的目标是设备侧完成 scoring 和 top-k，只把 compact indices/少量 metadata 暴露给 CPU 调度。

性能评估也按这条数据流拆解。baseline 是普通 MNN LLM 的 full-compute prefill；PIC full-compute 要先对齐普通 full-compute；full-reuse 的成本应主要剩下磁盘读取、hydrate、PagedCache 写入和 suffix prefill；cacheblend/epic/kvshare 的每个 ratio 都是独立请求，延迟必须包含本次请求自己的 full-reference/scoring/top-k、hydrate、sparse recompute 和 suffix prefill，不能跨 ratio 或跨请求复用 score vector 或 recompute indices。

## PIC Server 约定

`transformers/pic_llm/engine/app/pic_server.cpp` 是 MNN PIC 自维护的独立 HTTP server，不依赖 `mls`。构建产物名固定为 `pic_server`，由 `.codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh` 安装到 artifact root 的 `bin/` 下。

首个协议入口是：

```text
POST /v1/prefill/text
POST /v1/kv/pic_caches
POST /v1/chat/completions
POST /chat/completions
```

它接受内联文本 `{"id":"doc-1","type":"text","content":"...","force":false}`，用 PIC/PagedAttention 模型 prefill 文档，并按 layer 导出 raw KV cache：key 和 value 分开写到 `.k` / `.v` 文件，真实 `batch/kv_heads/head_dim/dtype_bytes` 写到同层 `.json` sidecar。导出前 CPU/CUDA PagedAttention 必须对 key cache 做 inverse RoPE，metadata 标注 `kv_layout.layout=mnn_paged_attention_raw_v1` 和 `kv_layout.key_rope_state=canonical_no_rope`；value cache 保持原布局。

`POST /v1/kv/pic_caches` 接受 `{"id":"pic-1","text_cache_refs":[{"id":"doc-1"}],"selection_algorithm":"full-reuse"}`，解析一个或多个 text cache 并返回 PIC metadata。`POST /v1/chat/completions` / `/chat/completions` 兼容 OpenAI 风格 `messages`，可在 `pic_cache` 中内联同一份 spec；prompt 中的 `{{pic_cache}}` 是 PIC 注入位置。带 `pic_cache` 的 chat 请求必须在 `messages.content` 中显式包含该 placeholder（或 `pic_cache.placeholder` 指定的自定义 placeholder）；不再支持无 placeholder 的 legacy 隐式 PIC 插入。独立正确性/性能实验之间由客户端显式调用 `POST /reset` 清理 LLM 请求状态；不要在 CUDA/OpenCL PagedAttention backend 按请求自动清整块 PagedCache，这会干扰 decode/continuous 状态和性能。Jetson CUDA 服务启动后的首轮 prefix-cache 写盘若用于 warm-up，必须匹配目标 token length / shape bucket，或对首个 text cache 做重建/验证；这些探测不计入报告。当前 MNN 原生执行语义：

- `full-reuse`：特殊 PIC 计算路径，不重算 PIC token。执行时先从磁盘 `.k/.v` hydrate 全量 PIC KV，CPU/CUDA/OpenCL PagedAttention 对 canonical_no_rope key 按当前 logical slot 重新施加 RoPE 并写入 paged KV slots；suffix token 作为真实 query 继续从同一份 PagedCache 读取 PIC KV。
- `full-compute`：不读取磁盘 PIC KV，按 prelude + PIC tokens + suffix 完整计算。
- `epic`：参考 `hf_pic_runtime` 的 EpicPlanner，按 `pic_recompute_ratio` 选择 PIC 开头连续 token；`score_layer_idx` 之前的层保持 token 正常计算语义，走普通 full-prompt PagedAttention 并把当前请求 K/V 写入 PagedCache；从 `score_layer_idx` 开始，选中的 PIC token 和 suffix/非复用 KV token 参与计算，其他 PIC 位置直接复用磁盘 KV。
- `cacheblend` / `delta-v`：参考 Python CacheBlend/DeltaV planner，用 score layer 上 full-prompt reference 与 cached PIC value 的 mean-abs delta 选 top-ratio token；执行模式为 `native-cacheblend-sparse-recompute`。
- `kvshare` / `delta-a`：执行模式为 `native-kvshare-sparse-recompute`，复用同一套 sparse prefill layer plan；当前 MNN C++ 没有 HF autograd，因此评分使用 full-vs-cached K/V delta influence proxy，metadata 中必须标注与 Python `attention_output` gradient influence 的差异。

CPU / CUDA / OpenCL 三个 PagedAttention backend 的功能语义必须对齐：都要支持 full-compute 写入并从 PagedCache 读取、full-reuse 从持久 `.k/.v` hydrate 到 PagedCache 并对 canonical_no_rope key 重新施加 RoPE、sparse recompute 按 logical slot 更新选中 PIC token、任意 `score_layer_idx` 的 cacheblend/kvshare scoring。CPU backend 可以在 host 上计算 delta-v score 和 top-k，但仍必须读取当前请求 PagedCache / slot table 与持久 cached `.v`，通过 `PagedKVMeta::setCacheBlendScoringResult` 返回 local indices；不得新增 chat/scoring scratch `.k/.v`，不得绕过 PagedCache 直接比较临时 tensor。CUDA/OpenCL backend 的 scoring/top-k 仍按设备侧 native 分支要求实现。

PIC cache 磁盘边界是硬约束：只有 `POST /v1/prefill/text` 允许把 text cache 的 K/V 作为持久 `.k/.v` 写入磁盘。`/v1/chat/completions` 中的 `full-compute`、`full-reuse`、`epic`、`cacheblend`、`kvshare` 以及任何 full-reference / scoring 过程，都不得为了实现内部 reference 或评分调用 prefix-cache `PendingWrite`、`setPrefixCacheFile`，也不得生成 scratch reference `.k/.v` 再读回。`cacheblend` / `delta-v` 的正确数据流是：正常 forward 到 score layer，score layer 之前的 Attention 也是普通 full-prompt PagedAttention，当前请求 K/V 只写入 PagedCache；reference K/V 保留在 PagedCache / backend device buffer；CUDA/OpenCL scoring 分支直接读 reference device buffer 与 cached PIC source/PagedCache buffer 计算 score，并在 GPU/CL/CUDA 上完成 top-k 选择。不得把完整 score vector 拷回 CPU 后排序；如当前调度接口需要，最多只把最终 compact top-k logical indices/少量 metadata 暴露给 CPU。若 GPU/CL/CUDA scoring + top-k 分支尚未实现，必须明确标注为 unsupported/fallback，不能用 CPU `std::vector` 读盘扫描冒充 `native-cacheblend-sparse-recompute`。

多 ratio 性能对比也是硬约束：`cacheblend` / `kvshare` 的 `1%/5%/10%/20%/30%` 只能共享同一个已由 `/v1/prefill/text` 构建的 text cache、同一模型和同一 suffix；每个 ratio 必须作为独立 `/v1/chat/completions` 请求执行，并在该请求内独立完成 full-reference / scoring / top-k 选择。不得跨 ratio 或跨请求缓存、复用 score vector 或 recompute logical indices；单个 ratio 的延迟必须包含它自己的 scoring 成本。

历史调试记录：Jetson CUDA 上曾出现 `/v1/prefill/text` 冷启动首轮 prefix-cache 写盘后层 `.k/.v` 变成 `ff7f...`，随后 `full-reuse` / `cacheblend` / `epic` 输出大量 `!`。定位结论是问题集中在 legacy prefix-cache `PendingWrite` 与 PagedAttention/PagedCache 状态交界处，而不是 decode 本身。尝试过在 backend 按 request 自动清整块 PagedCache，这条路被否定，因为会干扰 decode/continuous 状态和性能；尝试过为 prefix-write 临时 clone prefill module，也不是真正隔离 PagedAttention 共享 cache，且会引入 runtime path / session 状态复杂度。正确方向是减少旧 prefix-cache 兼容面：旧逻辑只允许服务 `/v1/prefill/text` 持久导出；chat、full-reference、scoring、sparse recompute 全部围绕当前请求 PagedCache 维护，不写 scratch `.k/.v`，不绕过 slot table，不用 CPU 读盘扫描冒充 native scoring。

PIC/PagedAttention 优化目标按正确语义分层处理，不为了跑分绕过 PagedCache 或省略请求内 scoring：

- 当前性能分析只看 prefill-only，PIC chat 请求使用 `max_tokens=0`；不要把 decode token、采样、decode1 或 next-logits forward 算入 prefill latency。
- `normal LLM full-compute prefill` 是普通模型基线，必须用 `.cache/mnn-llm-export/<model>/config.json` 的真实普通导出模型跑 `llm_bench -n 0`，只读取 `results[type=prefill]`。
- `PIC full-compute prefill` 必须完整计算 prelude + PIC tokens + suffix，不读取磁盘 PIC KV；但仍要把当前 K/V 写入 PagedCache，再从 PagedCache / slot table 读回做 PagedAttention，不能绕过 PagedCache 直接用原始 K/V 伪装普通 Attention。优化目标是接近 normal LLM full-compute；若明显更慢，优先优化 PagedAttention mask fast path、query split、PagedCache 读写和 kernel 选择。
- `PIC full-reuse prefill` 不做 full-reference、scoring 或 sparse recompute，也不默认重算最后一个 PIC token；它只 hydrate 磁盘 cached PIC KV 到 PagedCache、在 GPU/CUDA/OpenCL kernel 内按当前 logical slot 重新施加 RoPE，然后执行 suffix prefill。优化目标是显著快于 PIC full-compute；若慢，优先拆 `disk_read`、`hydrate`、`PagedCache write`、`suffix_prefill`。
- `cacheblend` / `epic` / `kvshare` sparse prefill 的目标不是复用跨请求 scoring，而是在每个独立请求内高效完成 full-reference / scoring / top-k，再 hydrate 大量复用 KV，只 sparse recompute 选中的 PIC token。低 ratio 应接近 full-reuse，高 ratio 延迟应随重算 token 数增加而合理上升；若比 full-compute 更慢，优先拆 `full_reference/scoring`、`disk_read/hydrate`、`sparse_recompute`、`suffix_prefill` 定位瓶颈。
- 报告性能时至少给出 `PIC full-compute / normal LLM full-compute`、`full-reuse / PIC full-compute`、各 ratio `cacheblend / full-reuse`、各 ratio `cacheblend / PIC full-compute`、各 ratio `cacheblend / normal LLM full-compute`。这些倍数是判断优化方向是否正确的主指标。
- 优先级固定为：先让 `PIC full-compute` 对齐 `normal LLM full-compute`，再让 `full-reuse` 只剩 hydrate + suffix prefill 成本，最后再优化 `cacheblend` / `epic` / `kvshare` 的 GPU/CL/CUDA native scoring 和 sparse recompute。
- OpenCL PIC 正式性能测试前必须先 warm 目标 shape / ratio，并调用 `/v1/tune/update_cache` 写回 MNN OpenCL autotune cache；冷启动 kernel build、LWS tuning、cachefile 生成或首轮 prefix-cache 探测不能计入正式 cacheblend/epic latency。正式报告只使用 warm 后的独立请求计时，run log 必须确认没有 `Cache invalid`、`target unavailable`、`async persistent PIC cache read failed` 或 `ERROR`。
- OpenCL PIC 默认生产路径必须使用当前最快且已验证的实现和调度逻辑：`score_layer_idx=1`，`layer=1` 走 full-Q/compact-output `score_flash_attention`，`layer>=2` 走 fused `sparse_flash_attention`，row32/row64 由内置 shape/plan heuristic 选择，cacheblend selected PIC ratio `>=50%` 保持 row64。不要留下 env 开关、旧三段 sparse QK/softmax/QKV fallback、手工强制 direct-value 路径或其它调试分支进入正式路径，避免无意性能回退。

## MNN 修改约束

- 只在必要范围内修改本仓库源码。
- 不要读取或修改 `schema/private/` 和 `source/internal/`。
- 不要把 `3rd_party/cutlass/` 等构建时下载目录加入 git。
- 不要用旧路径 `impl/mnn` 新增说明或脚本默认值。
- 修改 schema 后使用 `bash schema/generate.sh`，并检查生成文件 diff。
- 更新注册表时使用 `python3 tools/script/register.py .`，并确认生成变化合理。
