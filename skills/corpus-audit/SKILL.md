---
name: corpus-audit
description: 审计 replay_benchmark kernel corpus 的忠实性（kernel body 是否忠实复制 MNN 源码）与正确性（adapter 参数转换是否匹配 MNN onExecute、launch geometry 是否一致、validator 是否真正验证输出值）。覆盖 kernel body diff、常量一致性、launch geometry 核对、output 非零诊断、validator 有效性测试、多形状 case 覆盖。与 kernel-adapt skill 配合使用——kernel-adapt 负责"如何添加"，本 skill 负责"验证已添加的是否正确"。
---

# Corpus 忠实性与正确性审计 Skill

> **触发**：当需要验证 corpus 中的 kernel/adapter/case 是否忠实复制 MNN、是否正确执行、validator 是否有效时触发。包括：新增 kernel 后审计、修复 adapter bug 后回归、发现 smoke validator 掩盖问题时排查、扩展 case 覆盖多种 shape/参数时。
>
> **前置**：已完成 kernel-adapt skill 的"添加 kernel"流程，corpus 能编译并通过基础测试。本 skill 在此基础上做深度审计。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。

---

## 审计总览

```
忠 实 性                    正 确 性
(kernel == MNN?)            (adapter 参数对吗? validator 有效吗?)
    │                           │
    ├─§1 kernel body diff      ├─§4 launch geometry 核对
    ├─§2 常量/宏一致性          ├─§5 output 非零诊断
    ├─§3 版本分流确认           ├─§6 validator 有效性测试
    │                           ├─§7 测试数据值域检查
    │                           └─§8 多形状 case 覆盖
    └─§9 审计清单（汇总执行）
```

---

## §1 Kernel Body Diff（忠实性基础）

> **原则**：corpus 的 `__global__` kernel 函数体必须与 MNN 源码**字节级一致**（忽略空白/注释）。任何"简化""适配"都是忠实性违规。

### 方法

```bash
# 对每个 kernel，diff corpus vs MNN（忽略空白）
diff <(sed -n '/__global__ void KERNEL_NAME(/,/^}/p' \
  replay_benchmark/kernel_corpus_bridge/cuda/kernels/OP.cu) \
  <(sed -n '/__global__ void KERNEL_NAME(/,/^}/p' \
  source/backend/cuda/execution/FILE.cu) --ignore-all-space
```

### 检查项

1. **函数体完全一致**：`diff --ignore-all-space` 输出为空
2. **签名一致**：参数类型、顺序、const 限定符
3. **模板实例化一致**：MNN 用 `KERNEL<T>` 实例化，corpus shim 必须用相同模板参数
4. **namespace 隔离**：corpus 用 `MNN::Corpus`，MNN 用 `MNN::CUDA`（这是允许的，只要函数体一致）

### 常见违规

| 违规 | 示例 | 后果 |
|------|------|------|
| 简化 kernel 逻辑 | 删除 `#pragma unroll`、合并循环 | 性能特征不同，PMU 数据失真 |
| 改变参数类型 | `const float*` → `const T*` 且实例化不同 | 编译通过但语义不同 |
| 添加/删除 arch 守卫 | MNN 有 `#if __CUDA_ARCH__>=800`，corpus 删了 | sm75 下执行不该执行的指令，可能 crash |
| 改变 shared memory 用量 | MNN 用 `extern __shared__`，corpus 用固定大小 | 布局不匹配，越界 |

### 批量审计脚本

```bash
# 审计所有 corpus kernel vs MNN
for cu in replay_benchmark/kernel_corpus_bridge/cuda/kernels/*.cu; do
  grep -oP '__global__ void \K\w+' "$cu" | sort -u | while read k; do
    mnn_file=$(grep -rl "__global__ void ${k}(" source/backend/cuda/execution/*.cu source/backend/cuda/execution/*/*.cu source/backend/cuda/execution/*/*.cuh 2>/dev/null | head -1)
    if [ -z "$mnn_file" ]; then
      echo "WARN: $k not found in MNN (corpus-only kernel?)"
      continue
    fi
    result=$(diff \
      <(sed -n "/__global__ void ${k}(/,/^}/p" "$cu" | tr -s '[:space:]' ' ') \
      <(sed -n "/__global__ void ${k}(/,/^}/p" "$mnn_file" | tr -s '[:space:]' ' ') \
      --ignore-all-space)
    if [ -n "$result" ]; then
      echo "DIFF: $k ($cu vs $mnn_file)"
    fi
  done
done
```

---

## §2 常量/宏一致性（忠实性深化）

> **教训**：corpus 的 `INT8_PACK_NUMBER` 曾定义为 4，而 MNN 一直是 16，导致 adapter 计算的 `channelsPackInt8` 与 MNN 不一致，kernel 越界读取 scale 数组。

### 必查常量

| 常量 | MNN 定义位置 | corpus 必须一致 |
|------|-------------|----------------|
| `PACK_NUMBER` | `MNNCUDADefine.hpp` = 8 | 通道打包（float/half） |
| `INT8_PACK_NUMBER` | `MNNCUDADefine.hpp` = 16 | int8 通道打包 |
| `GEMV_TILE` | `ConvFpAIntBExecution.cu` = 64 | GEMV 归约线程数 |
| `TILE_DIM` | `ConvFpAIntBExecution.cu` = 16 | GEMM tile 大小 |
| `WARP_SIZE` | = 32 | warp 大小 |
| `GEMV_OC_PER_BLOCK` | = 4 | V2 GEMV 每 block OC 数 |
| `OC_PER_BLK`（V9/V14） | MNN=2(corpus=4) | 见 §3 版本分流 |

### 检查方法

```bash
# grep corpus 中的常量定义，与 MNN 对比
grep -rn "#define PACK_NUMBER\|#define INT8_PACK\|constexpr int PACK\|constexpr int GEMV\|constexpr int TILE" \
  replay_benchmark/kernel_corpus_bridge/cuda/kernels/*.cu \
  replay_benchmark/kernel_corpus_bridge/cuda/CudaOps*.cpp \
  replay_benchmark/kernel_corpus_bridge/cuda/CudaOpsFp16.cu

# MNN 对应
grep -rn "#define PACK_NUMBER\|#define INT8_PACK\|const int GEMV\|const int TILE\|const int WARP" \
  source/backend/cuda/execution/MNNCUDADefine.hpp \
  source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
```

### adapter 参数计算公式核对

adapter 中的 `channelsPackInt8`、`channelsPackFloat`、`total` 等计算必须与 MNN `onExecute` / `onResize` 一致：

```bash
# MNN 的计算公式
grep -n "channelPackInt8 =\|channelPackFloat =\|mCount =" \
  source/backend/cuda/execution/int8/FloatToInt8Execution.cu

# corpus adapter 的计算
grep -n "channelsPackInt8\|channelsPackFloat\|total =" \
  replay_benchmark/kernel_corpus_bridge/cuda/CudaOpsFp16.cu | grep -v "//"
```

**关键公式**（以 FLOAT_2_INT8 为例）：
- MNN: `channelPackInt8 = UP_DIV(channel, INT8_PACK_NUMBER=16) * 4`
- MNN: `channelPackFloat = UP_DIV(channel, PACK_NUMBER=8) * 8`
- MNN: `mCount = area * UP_DIV(channel, 16) * 4 = area * channelPackInt8`
- corpus adapter 必须用相同公式

---

## §3 版本分流确认（忠实性进阶）

> **原则**：不同 MNN 版本的 kernel 如果函数体有实质差异，必须独立实现 kernel + 独立 shim。不能共用 shim 仅改参数。

### 分流判定规则

用 `git diff <tagA> <tagB> -- <file>.cu` 对比 `__global__` 函数体：
- **函数体字节级一致** → 可共用 shim（不同 entry 名）
- **函数体有任何实质差异** → **必须独立 kernel + 独立 shim**

### 检查方法

```bash
# 扫描所有 tag 的函数体差异
for k in $(grep -oP '__global__ void \K\w+' replay_benchmark/kernel_corpus_bridge/cuda/kernels/*.cu | sort -u); do
  mnn_file=$(grep -rl "__global__ void ${k}(" source/backend/cuda/execution/*.cu source/backend/cuda/execution/*/*.cu source/backend/cuda/execution/*/*.cuh 2>/dev/null | head -1)
  [ -z "$mnn_file" ] && continue
  for tag in 1.2.0 1.2.7 2.0.4 2.2.3 2.4.2 2.5.1 2.8.0 2.8.4; do
    prev_tag=$(git log --format=%H $tag^ 2>/dev/null | head -1)
    [ -z "$prev_tag" ] && continue
    diff_out=$(git diff $tag $prev_tag -- "$mnn_file" 2>/dev/null | grep "^[+-].*__global__\|^[+-].*out\[.*\] =" | head -1)
    [ -n "$diff_out" ] && echo "CHANGED: $k at $tag"
  done
done
```

### 已知分流点速查

详见 `replay_benchmark/kernel_corpus_bridge/cuda/CUDA_KERNEL_VERSIONING.md` 的"逐 kernel 变更详情"和"精确变更点"表。

### 常量版本差异

某些常量在不同版本不同（如 `OC_PER_BLK`：MNN HEAD V9=2，corpus 硬编码=4）。如果这是 corpus 的有意选择，必须在 HANDOFF.md 标注为"corpus 适配"并说明原因。

---

## §4 Launch Geometry 核对（正确性核心）

> **教训**：GEMV adapter 曾用 `gridX=(oc+15)/16, block=16/64`，而 MNN `onExecute` 用 `grid(oc,batch), block(64)`——gridX 错误导致只写 oc=0，被 smoke validator 掩盖。

### 核对方法

对每个 adapter，对比 `adapt()` 设置的 `ac.globalSize/localSize` 与 MNN `onExecute` 的 `<<<grid, block>>>`：

```bash
# MNN launch
grep -B5 "KERNEL_NAME<<<" source/backend/cuda/execution/FILE.cu | grep -E "dim3|<<<|grid|block"

# corpus adapter
grep -A5 "ac.globalSize\|ac.localSize" replay_benchmark/kernel_corpus_bridge/cuda/CudaOps*.cpp | grep -E "globalSize|localSize"
```

### 必查项

1. **gridX/gridY 与 MNN dim3 一致**：`dim3 grid(X, Y)` → `ac.globalSize[0]=X, ac.globalSize[1]=Y`
2. **blockX/blockY 与 MNN dim3 一致**：`dim3 block(X, Y)` → `ac.localSize[0]=X, ac.localSize[1]=Y`
3. **DivModFast 参数与 MNN 一致**：MNN `DivModFast d(val)` → shim 内 `DivModFast(val)`，adapter 传 `val`
4. **shared memory 大小与 MNN 一致**：`<<<g, b, smem>>>` → adapter 传 `sharedMem` 参数
5. **block 选择策略按版本分流**：1.2.0 用 `mnnBlock120()`，1.2.1+ 用 `kBlock=128`

### 逐变体核对表（WOQ GEMV 示例）

| 变体 | MNN launch | corpus adapter 必须 |
|------|-----------|-------------------|
| V1 int8/int4 | `blocks(oc, batch), threads(GEMV_TILE=64)` | `gridX=oc, gridY=batch, blockX=64` |
| V5 | `v5_grid(oc, batch), v5_block(128)` | `gridX=oc, blockX=128` |
| V9 | `v9_grid((oc+3)/4, batch), v5_block(128)` | `gridX=(oc+3)/4, blockX=128` |
| V14 | `v14_grid((oc+3)/4, batch), v14_block(128)` | `gridX=(oc+3)/4, blockX=128` |
| V2 | `blocks((oc+3)/4, batch), threads(32, 4)` | `gridX=(oc+3)/4, blockX=32, blockY=4` |

---

## §5 Output 非零诊断（正确性基础）

> **教训**：cast/float2int8 adapter 用 `0.1f*(i%13)` 作 input，scale=0.1，导致 `input*scale=0.01` → round=0 → output 全零。smoke validator 通过但实际没验证任何东西。

### 诊断方法

临时在 runner 的 validate 前统计 output 非零情况：

```cpp
// 在 KernelCorpusBenchmark.cpp 的 CUDA readback 后、validate 前
int nzCount = 0;
for (size_t i = 0; i < output.size(); ++i)
    if (output[i] != 0.0f) nzCount++;
int nzBytes = 0;
const uint8_t* raw = reinterpret_cast<const uint8_t*>(output.data());
for (size_t i = 0; i < output.size() * sizeof(float); ++i)
    if (raw[i] != 0) nzBytes++;
report.error = "DIAG nzFloat=" + std::to_string(nzCount) + "/" + std::to_string(output.size())
            + " nzBytes=" + std::to_string(nzBytes) + "/" + std::to_string(totalBytes);
```

运行后分析：

```python
import json, re
d = json.load(open('/tmp/kc_diag.json'))
for c in d['cases']:
    if c['backend'] != 'cuda': continue
    m = re.search(r'nzBytes=(\d+)/(\d+)', c.get('error',''))
    if m and int(m.group(1)) == 0:
        print(f'ALL-ZERO: {c["case"]}')
```

### 全零分类与处理

| 类别 | 原因 | 处理 |
|------|------|------|
| BF16 arch-guarded | sm75 空 kernel（`#if __CUDA_ARCH__>=800`） | ✓ 合理，标注 smoke |
| setzero/bias_zero | kernel 语义就是清零 | ✓ 合理 |
| reduction min/prod | input 含 0，结果数学上=0 | ✓ 合理（但建议改 input 避免 0） |
| cast/quantize 全零 | input 值域太小，量化后全 0 | ✗ **bug**，见 §7 |
| kernel 未执行 | launch 参数错误、grid=0 | ✗ **bug**，见 §4 |

---

## §6 Validator 有效性测试（正确性深化）

> **原则**：validator 必须真正比较 output 值与 expected 值。一个 validator 如果无论 output 是什么都返回 true，它就是无效的。

### 测试方法：故意破坏 validator

临时把 validator 改为**总是返回 false**，运行测试：
- 如果 case 变成 `validation_failed` → validator 有效 ✓
- 如果 case 仍然 `validation_passed` → validator 被跳过 ✗

```bash
# 临时破坏：在 validate 函数开头加 return false
sed -i 's/bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const {/bool CLASS::validate(...) const { return false; \/*TEST*\/ if (false)/' FILE
# 编译运行，确认 case fail
# 然后撤销
```

### validator 分类

| 类型 | 检查方式 | 有效性 |
|------|---------|--------|
| **real** | host 重算 expected，`fabs(output-expected) < tol` | ✓ 有效 |
| **smoke (非零)** | 检查 output 非全零 | △ 部分有效（捕获 kernel crash/未执行，但不捕获值错误） |
| **smoke (BF16)** | sm75 空 kernel，output 必然全零，直接 return true | △ 合理（sm75 无法验证） |
| **return true (无检查)** | 直接返回 true | ✗ 无效（必须改为 real 或 smoke） |

### 必改项

任何 `return true;` 的 validator（无任何检查）必须改为至少 smoke（检查非零）或 real（重算对比）。

---

## §7 测试数据值域检查（正确性保障）

> **原则**：adapter 构造的测试 input 必须使 kernel 输出**非零且有变化**，否则 validator 无法捕获 bug。

### 问题模式

| input 模式 | 问题 | 修复 |
|-----------|------|------|
| `0.1f*(i%13)` | 值域 [0, 1.2]，太小 | 改为随机 [-5, 5] |
| `0.1f*(i%13) - 0.5f` | 值域 [-0.5, 0.7]，cast 到 int 全 0 | 改为随机 [-5, 5] |
| scale=0.1 + input=0.1 | `input*scale=0.01` → round=0 | scale 改为 1.0 或 input 更大 |

### 推荐方案：确定性随机输入

```cpp
// 在 CudaOps.hpp 中定义（inline，跨 TU 共享）
inline std::vector<float> fillInputRand(int n, unsigned seed = 0x533d,
                                        float lo = -5.0f, float hi = 5.0f) {
    std::vector<float> v(n);
    unsigned state = seed ? seed : 0x533d;
    for (int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;  // LCG
        float r = (float)((state >> 8) & 0xFFFF) / 65535.0f;
        v[i] = lo + r * (hi - lo);
    }
    return v;
}
```

**要求**：
1. 固定种子（确定性）——validator 能重算相同 expected
2. 值域足够大——cast/量化后非零
3. 有正有负——覆盖 clamp 的正负边界

### 适用范围

- cast 类（CASTMIDFLOAT、FLOAT_2_INT8、INT8_2_FLOAT）——需要大值域使 int 截断非零
- 量化类（weight_only_quant）——scale 与 input 组合后非零
- reduction min/prod——input 不应包含 0（否则 min/prod=0 无法验证）

---

## §8 多形状 Case 覆盖（覆盖度提升）

> **原则**：同类 kernel 的不同变体应配置**相同的基础参数**，测试变体实现是否符合语义。特殊形状适配的变体（如 3x3 特化、MULTI_WIDTH4 要求 ow%4==0）按其约束测试。

### Case 覆盖原则

1. **变体一致性**：同类 kernel（如 GEMV_FpAInt8B V1/V2/V5/V9/V14）用相同的 input/weight/scale/bias，验证不同实现策略输出一致
2. **形状多样性**：每个 op 至少 2-3 种形状（小/中/大 channel、不同 kernel size、不同 stride/pad）
3. **边界覆盖**：channel=1（最小）、channel=PACK_NUMBER 的倍数/非倍数、ow%4==0 vs !=0
4. **版本覆盖**：同一变体的不同 tag（1.2.0/1.2.7/3.6.0）各有 case

### 添加 case 方法

```python
# 为同类变体配置相同参数（变体一致性测试）
import json
d = json.load(open('replay_benchmark/kernel_corpus/operator_cases.json'))
# 所有 GEMV 变体用相同的 ic/oc/batch/quan_c
common_params = {"ic": 16, "oc": 8, "batch": 2, "quan_c": 8, "ic_p": 16, "oc_p": 8}
for variant in ["cuda_gemv_fpaint8b_fp32", "cuda_gemv_fpaint4b_v5_fp32", "cuda_gemv_fpaint4b_v9_fp32"]:
    d['cases'].append({
        "name": f"{variant}_consistency",
        "backend": "cuda", "framework": "mnn", "op_type": "conv_fpa_intb",
        "tag": "3.6.0", "variant": variant, "int_params": common_params
    })
json.dump(d, open('replay_benchmark/kernel_corpus/operator_cases.json','w'), indent=2)
```

### 形状多样性 case 示例

```json
// 卷积：不同 kernel size / pad / stride
{"name":"cuda_conv_dw_fp32_kw3_pad1","variant":"cuda_conv_dw_fp32","int_params":{"iw":8,"ih":8,"channels":4,"kw":3,"kh":3,"pw":1,"ph":1}}
{"name":"cuda_conv_dw_fp32_kw5_pad2","variant":"cuda_conv_dw_fp32","int_params":{"iw":10,"ih":10,"channels":8,"kw":5,"kh":5,"pw":2,"ph":2}}
{"name":"cuda_conv_dw_fp32_stride2","variant":"cuda_conv_dw_fp32","int_params":{"iw":16,"ih":16,"channels":4,"kw":3,"kh":3,"sw":2,"sh":2}}
```

### 特殊形状变体

- `CONV_DW3x3_HALF2_OPT`：要求 kw=kh=3, sw=sh=1, pw=ph=1, ow%2==0
- `CONV_DW_MULTI_WIDTH4`：要求 ow%4==0, kw>3 && kw<12, kh==1, pw=ph=0
- `CONV_DW_BF162_OPT`：dw=dh=1（无 dilation）
- 这些变体**不强求通用参数**，按其约束测试即可

---

## §9 审计清单（汇总执行）

按以下顺序逐项执行，每项 pass 后再进入下一项：

### 清单

| # | 审计项 | 方法 | pass 条件 |
|---|--------|------|----------|
| 1 | kernel body diff | §1 批量脚本 | 所有 kernel `diff --ignore-all-space` 为空 |
| 2 | 常量一致性 | §2 grep 对比 | corpus 常量与 MNN `MNNCUDADefine.hpp` 一致 |
| 3 | 版本分流 | §3 git diff 扫描 | 函数体变更点有独立 shim |
| 4 | launch geometry | §4 逐变体核对 | `globalSize/localSize` 与 MNN `<<<>>>` 一致 |
| 5 | output 非零 | §5 诊断 patch | 无"意外全零"（BF16/setzero/数学全零除外） |
| 6 | validator 有效 | §6 破坏测试 | 改 return false 后 case fail |
| 7 | 测试数据值域 | §7 检查 input 模式 | 无"input 太小导致全零" |
| 8 | 多形状覆盖 | §8 case 审计 | 每 op 至少 2-3 种形状 |

### 执行流程

```
1. kernel body diff（§1）
   ├─ 全部一致 → 继续 §2
   └─ 有差异 → 修复 kernel body（忠实复制 MNN）
2. 常量一致性（§2）
   ├─ 一致 → 继续 §3
   └─ 不一致 → 修正 corpus 常量定义
3. 版本分流（§3）
   ├─ 正确 → 继续 §4
   └─ 缺独立 shim → 按 kernel-adapt skill 添加
4. launch geometry（§4）
   ├─ 一致 → 继续 §5
   └─ 不一致 → 修复 adapter 参数
5. output 非零（§5）
   ├─ 无意外全零 → 继续 §6
   └─ 有意外全零 → §7 排查 input 数据
6. validator 有效（§6）
   ├─ 有效 → 继续 §7
   └─ 无效 → 改为 real 或 smoke validator
7. 测试数据值域（§7）
   ├─ 合理 → 继续 §8
   └─ 太小 → 改为 fillInputRand（确定性随机）
8. 多形状覆盖（§8）
   ├─ 覆盖充分 → 审计完成
   └─ 不足 → 添加 case（变体一致性 + 形状多样性）
```

### 审计报告

审计完成后，在 `replay_benchmark/kernel_corpus_bridge/cuda/HANDOFF.md` 的 §8（Validator 质量分析）更新：
- real validator 数量变化
- smoke validator 数量变化
- 新发现并修复的问题列表
- 剩余待改进项

---

## 已知问题分类及修复方向

| 问题 | 涉及 | 修复方式 | 优先级 |
|------|------|---------|--------|
| input 值域太小→全零 | cast/quantize adapter | 改为 `fillInputRand(-5,5)` | 🔴 高 |
| smoke validator 掩盖 bug | 所有 `return true` 的 validator | 改为 real（host 重算） | 🔴 高 |
| launch geometry 与 MNN 不一致 | GEMV/cast adapter | 对照 MNN `<<<>>>` 修正 | 🔴 高 |
| 常量与 MNN 不一致（INT8_PACK_NUMBER） | int8 adapter | 改为 MNN 的值 | 🔴 高 |
| 缺少版本独立 shim | 跨版本 kernel | 按 kernel-adapt skill 添加 | 🟡 中 |
| case 形状单一 | 所有 op | 添加多形状 case | 🟡 中 |
| BF16 sm75 空 kernel | convdw_bf16 | smoke 合理，标注 | 🟢 低 |

## 参考案例

### 案例 1：GEMV launch geometry bug（§4）

**症状**：WOQ GEMV adapter 用 `gridX=(oc+15)/16`，MNN 用 `gridX=oc`，导致只写 oc=0。
**发现方法**：升级 smoke→real validator 后，output[1..3]=0 而 expected≠0，validation_failed。
**根因**：smoke validator 只检查非零，output[0] 非零就 pass，掩盖了 gridX 错误。
**修复**：对照 MNN `onExecute` 的 `<<<blocks(oc, batch), threads(64)>>>` 修正所有 GEMV 变体的 gridX/blockX。

### 案例 2：INT8_PACK_NUMBER 不一致（§2）

**症状**：`FLOAT_2_INT8_PACKED` adapter 的 output 全零（compute-sanitizer 无错误）。
**发现方法**：§5 output 非零诊断发现 cast/int8 系列 case 全零；printf kernel 确认执行但 res=0。
**根因**：corpus `INT8_PACK_NUMBER=4`，MNN=16，导致 `channelsPackInt8` 计算错误，scale 数组越界读取。
**修复**：corpus `INT8_PACK_NUMBER` 改为 16，与 MNN `MNNCUDADefine.hpp` 一致。

### 案例 3：cast input 值域太小（§7）

**症状**：`CASTMIDFLOAT` 用 `0.1f*(i%13)-0.5f`（[-0.5,0.7]），cast 到 int32 全为 0。
**发现方法**：§5 诊断发现 cast 系列 case 全零；分析 input 值域发现全在 (-1,1)。
**修复**：改用 `fillInputRand(-5,5)`，使 cast 后有非零且有变化的输出。

---

## 与其他 skill 的配合

| skill | 关系 |
|-------|------|
| `kernel-adapt` | 负责"如何添加 kernel/adapter/case"（bake、shim、adapter 实现）。本 skill 在其完成后做"验证已添加的是否正确" |
| `general-debug` | 当审计发现 bug 时，用 general-debug 的方法论排查根因 |
| `test-ci` | 审计修复后，用 test-ci 跑回归测试确认无回归 |
| `retrospective` | 审计发现的 reusable lessons 加入 retrospective |
