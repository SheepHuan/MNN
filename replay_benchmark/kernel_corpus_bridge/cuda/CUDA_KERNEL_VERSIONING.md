# CUDA Kernel 版本演进 (1.2.0 → 3.6.0)

> 本文档记录 MNN CUDA 后端各 `__global__` kernel 在 1.2.0 到 3.6.0(工作树)
> 之间的签名/公式变更点,用于指导 kernel corpus 的多版本适配。
>
> **数据来源**:git tag `1.2.0` → `2.8.4`(最后 tag)+ 工作树(3.6.0)。
> 3.x 无 git tag,工作树即 3.6.0。
>
> **变更判定标准**:仅记录 `__global__` 函数的参数列表(个数/类型/顺序)
> 或核心计算公式发生实质性变化的版本点。纯空白/注释/不含 `__global__`
> 的 host 代码重构不计。

## 关键背景

- **1.2.0 的 block 自适应**:1.2.0 的 `CUDARuntime::blocks_num()` 基于
  `maxThreadsPerBlock` 自适应分档(maxNum / maxNum/2 / ... /128);
  1.2.1 起该逻辑被注释,固定 `mThreadPerBlock = 128`。
  corpus 通过 `mnnBlock120()` 复现 1.2.0 的自适应规则。
- **文件迁移**:
  - 1.2.0:CAST/CASTBOOL 在 `UnaryExecution.cu`;SUM/MEAN/MINIMUM/MAXIMUM/PROD
    在 `ReductionExecution.cu`;SETZERO 在 `ScatterNdExecution.cu`;
    add_bias 在 `MatMulExecution.cu`。
  - 1.2.7:Reductions 迁移到 `ReductionTemplate.cuh`。
  - 2.5.3:CAST 系列迁移到独立的 `CastExecution.cu`。
- **已删除的 kernel**:`pack_c4`/`unpack_c4`(1.2.0 独有,1.2.7 前删除)、
  `SETZERO`(2.2.2 前删除)、`add_bias`(1.2.6 前删除)。

## 变更点汇总表

| Kernel | 变更 tag | 变更性质 |
|---|---|---|
| RELU_Half | 1.2.7(新增) | 新 kernel(half 精度) |
| RELU_INT8 | 2.5.3(新增) | 新 kernel(int8 量化) |
| CLAMP | 1.2.7 | input/output 类型 `float*`→`T*` |
| SELECT | 2.8.0 | 新增 `s1,s2` stride 参数 + 公式改 |
| SOFTMAX | 2.2.3 | `ReduceParam*`→4 个显式 int 参数 |
| LAYERNORM | 2.2.2 | gamma/beta `T*`→`float*`,float 累加 |
| LAYERNORM | 2.8.4 | 新增 `bool RMSNorm` 参数 + 分支 |
| PRELU | 2.0.4 | slopeData `T*`→`float*`,公式重写(nhw/c_idx) |
| SCALE | 2.0.4 | scale/bias `T*`→`float*`,公式重写(nhw/c_idx) |
| maxpool_C8/avgpool_C8 | 2.0.4(新增) | 新 kernel(C8 通道打包) |
| global_avgpool_C8/global_maxpool_C8 | 2.8.3(新增) | 新 kernel(warp reduce) |
| ARGMAX | 2.0.4 | 输出 `T*`→`int*` + 索引改 |
| ARGMAX | 2.5.0 | 指针算术公式重写(签名不变) |
| INTERP_NERAEST/ROUND/BILINEAR | 1.2.7 | 重命名+PACK_NUMBER 重打包 |
| INTERP_NERAEST/ROUND/BILINEAR | 2.0.4 | 新增 `c_p` 参数 + per-element 循环 |
| NHWC_2_NCHW | 2.7.1(新增) | 新 kernel |
| NCHW_2_NHWC | 2.1.2(新增) | 新 kernel |
| NCHW_2_NHWC | 2.7.1 | src_offset 用 inChannelPack 代替 channel |
| GRID_SAMPLE_NEAREST/BILINEAR | 2.8.0 | 公式:`% channel_pack`→`% channel`,dst_offset 写入 |
| SUM | 1.2.7 | 迁移+结构体参数+float 累加 |
| SUM | 2.5.1 | 拆分为 SUM_NAIVE + SUM_REDUCE_AXIS,显式 int 参数 |
| MEAN | 1.2.7 | 迁移+结构体参数+float 累加 |
| MEAN | 2.5.1 | 结构体→显式 int 参数 |
| MEAN | 2.8.2 | 拆分为 MEAN_NAIVE + MEAN_REDUCE_AXIS |
| MINIMUM/MAXIMUM | 1.2.7 | 迁移+结构体参数+float 累加 |
| MINIMUM/MAXIMUM | 2.5.1 | 结构体→显式 int 参数 |
| PROD | 1.2.7 | 迁移+结构体参数+float 累加 |
| PROD | 2.5.1 | 结构体→显式 int + 循环起点 0/`sumValue=1.0` |
| blitRegion | 2.4.2 | 新增 `count` 参数 + fuseIndex/x/y/z 循环 |
| CONV_DW | 1.2.7 | kernel/bias `float*`→`half*`,NC4HW4 重打包 |
| CONV_DW | 2.0.2 | input/output `float`→`T`(模板化) |
| CONV_DW | 2.0.4 | 公式:c_p 索引重写(签名不变) |
| CONV_DW | 2.2.3 | `constBuffer*`→~20 个显式标量 + DivModFast |
| TopKAllRows/GetResultAllRows | 2.6.3(新增) | 新 kernel;`__global__` 签名 2.6.3→3.6.0 稳定 |
| RANGE | 2.2.1(新增) | 新 kernel;稳定至 3.6.0 |
| GATHERV2 | (无变化) | 1.2.0→3.6.0 稳定 |
| ATAN2/MOD/LOGICALOR | (无变化) | 1.2.0→3.6.0 稳定 |
| ARGMIN | 2.1.2(新增) | 新 kernel;稳定至 3.6.0 |

## 逐 kernel 变更详情

### RELU / RELU_Half / RELU_INT8 (UnaryExecution.cu)

- **1.2.0**: `RELU(const float*, float*, size_t, float)` — float 专用,稳定至 3.6.0
- **1.2.7**: 新增 `RELU_Half(const half*, half*, size_t, float)`
- **2.5.3**: 新增 `RELU_INT8(const int8_t*, int8_t*, size_t, int8_t)` — 公式 `y = x > zeroPoint ? x : zeroPoint`

### CLAMP (UnaryExecution.cu)

- **1.2.0**: `CLAMP(const float*, float*, size_t, float, float)` — float 专用
- **1.2.7**: input/output 类型 `const float*`→`const T*`(支持 half),公式不变

### CAST / CASTMIDFLOAT / CASTBOOL (1.2.7 起在 UnaryExecution.cu,2.5.3 迁移到 CastExecution.cu)

- **1.2.7**: 新增 `CAST<T1,T2>`、`CASTMIDFLOAT<T1,T2>`、`CASTBOOL(int32_t*, int32_t*, size_t)`
- **2.5.3**: 迁移到 CastExecution.cu,签名不变
- **2.5.3**: 新增 `FLOAT_2_INT8_CAST<T>`、`INT8_2_FLOAT_CAST<T>`(量化 cast)

### ATAN2 / MOD / LOGICALOR (BinaryExecution.cu)

- **1.2.0→3.6.0**: 签名/公式无变化。`ATAN2(const T*, const T*, T*, size_t, size_t, size_t)`
  (MOD/LOGICALOR 同形)

### RANGE (RangeExecution.cu)

- **2.2.1**: 新增 `RANGE(const int, const T*, const T*, T*)`,稳定至 3.6.0

### SELECT (SelectExecution.cu)

- **1.2.6**: 首次出现(旧签名,无 stride)
- **2.8.0**: 新增 `s1, s2` stride 参数(插在 output 前);公式改为按 stride 索引
  `output[i] = input1[i*s1] vs input2[i*s2]`。稳定至 3.6.0

### SOFTMAX (SoftmaxExecution.cu)

- **1.2.0**: 无 `__global__` SOFTMAX(用 cuDNN)
- **1.2.7**: 自定义 `SOFTMAX(const T*, T*, const ReduceParam*)` 首次出现
- **2.2.3**: `ReduceParam*`→4 个显式 int `(inside, axis, outside, count)`,公式不变
- **2.3.1**: 新增 `SOFTMAX_WARP_32`
- **2.4.1**: 新增 `SOFTMAX_AXIS_REDUCE`

### LAYERNORM (LayerNormExecution.cu)

- **1.2.0→2.0.4**: 稳定,`LAYERNORM(int, int, int, float, const T*, T*, const T*, const T*)`
- **2.2.2**: gamma_data/beta_data 类型 `const T*`→`const float*`;累加器 `T sum`→`float sum`
- **2.8.4**: 新增尾部 `bool RMSNorm` 参数;RMSNorm=true 时跳过 mean 计算。稳定至 3.6.0

### PRELU (PReLUExecution.cu)

- **1.2.0→1.2.8**: 稳定,`PRELU(int, int, int, const T*, T*, const T*, int div_factor)`
  — channel 索引 `c = (index/dim) % channels / div_factor`
- **2.0.4**: slopeData `const T*`→`const float*`;末参数 `div_factor`→`share_factor`;
  公式重写为 `nhw_idx/c_idx` 布局 + float 累加 + share_factor 门控。稳定至 3.6.0

### SCALE (ScaleExecution.cu)

- **1.2.0→1.2.7**: 稳定,`SCALE(int, int, int, const T*, T*, const T*, const T*)`
  — channel 索引 `c = (index/dim) % channels`
- **2.0.4**: scaleData/biasData `const T*`→`const float*`;公式重写为 `nhw_idx/c_idx`
  索引 + 显式 float cast。稳定至 3.6.0

### maxpool_C8 / avgpool_C8 (PoolExecution.cu)

- **1.2.0**: `maxpool/avgpool(const T*, T*, int bc, int ih, int iw, int oh, int ow, ...)` — NCHW
- **1.2.7**: `_halfC16/_floatC16` 变体(中间过渡)
- **2.0.4**: 新增 `maxpool_C8/avgpool_C8(const T*, T*, int ib, int ic_p, int ih, int iw, int oh, int ow, ...)`
  — C8 通道打包(NC4HW4)。稳定至 3.6.0

### global_avgpool_C8 / global_maxpool_C8 (PoolExecution.cu)

- **2.8.3**: 新增 `global_avgpool_C8/global_maxpool_C8(const T*, T*, int, int, int, int, int)`
  — warp reduce 实现。launch 固定 `<<<count, 128>>>`。稳定至 3.6.0

### GATHERV2 (GatherV2Execution.cu)

- **1.2.0→3.6.0**: 签名/公式无变化。`GATHERV2(int, int, int, int, int, ...)`
  (2.0.4 仅 host 侧加 fp16 dispatch)

### ARGMAX (ArgMaxExecution.cu)

- **1.2.0→1.2.8**: 稳定,`ARGMAX(int, int, int, int, const T*, T*)` — 输出 `T*`
- **2.0.4**: 输出类型 `T*`→`int*`;输入索引从 `inpPtr[0]`/`inpPtr[j*inside]`
  改为 `inpPtr[n+0*inside]`/`inpPtr[n+j*inside]`
- **2.5.0**: 签名不变,但指针算术公式重写(`inpPtr = input + idx_out*inside*dim + idx_in`;
  输出写 `outPtr[n]`→`outPtr[0]`)。新增 ARGMAX_FIRST_STEP/ARGMAX_SECOND_STEP 辅助 kernel。
  稳定至 3.6.0

### ARGMIN (ArgMinExecution.cu)

- **2.1.2**: 新增 `ARGMIN(int, int, int, int, const T*, int*)` — 保持旧的 `outPtr[n]` 索引。
  稳定至 3.6.0

### INTERP_NERAEST / INTERP_NERAEST_ROUND / INTERP_BILINEAR (InterpExecution.cu)

- **1.2.0**: `INTERP/INTERP_BILINEAR(int n, int ih, int iw, int oh, int ow, float, float, float, float, const T*, T*)`
  — 无 channel 维
- **1.2.7**: 重命名 INTERP→INTERP_NERAEST;新增 INTERP_NERAEST_ROUND;
  body 改为 PACK_NUMBER 通道打包(`* PACK_NUMBER + remain` 索引)
- **2.0.4**: 新增 `c_p` channel 参数(第 2 位);PACK_NUMBER 常量替换为运行时 c_p;
  循环从 pack-stride 改为 per-element。稳定至 3.6.0

### NHWC_2_NCHW / NCHW_2_NHWC (Transpose.cu)

- **2.0.4**: 仅 `NCHW_2_NHWC8` 存在
- **2.1.2**: 新增 `NCHW_2_NHWC(T0*, T1*, int, int, int, int, DivModFast, DivModFast)`
- **2.7.1**: 新增 `NHWC_2_NCHW`(同签名);NCHW_2_NHWC 公式改:
  src_offset 从 `(batch_idx * channel + chnl_idx) * area` 改为
  `(batch_idx * inChannelPack + chnl_idx) * area`(用输入 channel-pack 而非输出 channel)。
  稳定至 3.6.0

### GRID_SAMPLE_NEAREST / GRID_SAMPLE_BILINEAR (GridSampleExecution.cu)

- **2.4.1**: 首次出现
- **2.8.0**: 签名不变,但公式实质改:`idx_cp = index % channel_pack`→`index % channel`;
  新增 `dst_offset` 写入 `output[dst_offset]` 而非 `output[index]`;
  移除 `if(idx_cp >= channel)` 提前 continue。新增 3D 变体。
  稳定至 3.6.0

### Reductions: SUM / MEAN / MINIMUM / MAXIMUM / PROD

文件迁移:1.2.0 在 `ReductionExecution.cu`,1.2.7 起在 `ReductionTemplate.cuh`。

- **1.2.0**: `SUM/MEAN/MINIMUM/MAXIMUM/PROD(const T*, T*, int inside, int axis, int outside)`
  — T 累加,参数顺序 `(inside, axis, outside)`
- **1.2.7**: 迁移到 ReductionTemplate.cuh;签名改为 `(const T*, T*, const ReduceParam*)`;
  累加器 `T`→`float` + 显式 `(float)` cast
- **2.5.1**: 结构体参数→显式 int,但顺序变为 `(outside, axis, inside)`(与 1.2.0 相反);
  SUM 拆分为 SUM_NAIVE + SUM_REDUCE_AXIS;PROD 公式改(起点 0/`sumValue=1.0`);
  MINIMUM/MAXIMUM 签名同 2.5.1。3.6.0 的 corpus 用此签名
- **2.8.2**: MEAN 拆分为 MEAN_NAIVE + MEAN_REDUCE_AXIS。稳定至 3.6.0

> **1.2.0 vs 3.6.0 差异**(corpus 已适配):
> - SUM/MEAN:1.2.0 用 T 累积 + `(inside, axis, outside)`;
>   3.6.0 用 float 累积 + `(outside, axis, inside)` → **独立 kernel + shim**
> - MAX/MIN/PROD:逻辑相同,仅参数顺序不同 → 复用 3.6.0 shim,launch 里 swap 参数

### blitRegion (Raster.cu)

- **1.2.0→2.4.1**: 稳定,`blitRegion(const T*, T*, int loopCount, const int32_t*, ...)`
- **2.4.2**: 新增 `int count` 参数(第 2 位);循环变量 `i`→`fuseIndex`,
  `total=loopCount`→`total=count`,新增 x/y/z 分解。稳定至 3.6.0

### pack_c4 / unpack_c4 (Raster.cu)

- **1.2.0**: 存在;1.2.7 前删除(被 Transpose.cu 的 PACKCOMMON 替代)。
  corpus 仅保留 1.2.0 版本

### SETZERO (1.2.0 在 ScatterNdExecution.cu)

- **1.2.0→2.2.0**: `SETZERO(const int, T*)`;2.2.2 前文件删除。
  corpus 仅保留 1.2.0 版本

### add_bias (1.2.0 在 MatMulExecution.cu)

- **1.2.0→1.2.5**: `add_bias(T*, T*, const T*, int e, int h)`;1.2.6 前删除。
  corpus 仅保留 1.2.0 版本

### TopKAllRows / GetResultAllRows (TopKV2Execution.cu)

- **2.6.3**: 首次出现。`__global__` 签名 2.6.3→3.6.0 字节级稳定
- **2.8.4→3.6.0**(工作树):委托的 `__device__ TopKOneRow` helper 从插入排序改为堆排序
  (commit 67657bbd6,未发布,仅工作树)。`__global__` shell 本身不变

### CONV_DW (ConvDepthWiseExecution.cu)

- **1.2.0**: `CONV_DW(const float*, const float* kernel, const float* bias, float*, const constBuffer*)`
- **1.2.7**: kernel/bias 类型 `const float*`→`const half*`;body 重打包为 NC4HW4(PACK_NUMBER)
- **2.0.2**: input/output `float`→`T`(模板化,支持 half)
- **2.0.4**: 签名不变,但 body 索引重写(`i*PACK_NUMBER`/`zR` → c_p 索引 + `ob=i/(ow*oh)`)
- **2.2.3**: `const constBuffer*`→~20 个显式标量参数 + 3 个 DivModFast helper;
  循环改为每线程处理 2 通道(`total/2`, `oz = oz_2 << 1`)。稳定至 3.6.0

> CONV_DW 是演进最丰富的 kernel(4 个变更点),适合作为多版本适配的重点样例。

## block/grid 配置演进

| 版本 | block 选择策略 | corpus 适配方式 |
|---|---|---|
| 1.2.0 | 自适应分档(`maxThreadsPerBlock`-based: maxNum / maxNum/2 / maxNum/4 / maxNum/8 / 128) | `mnnBlock120(total)` |
| 1.2.1+ ~ 3.6.0 | 固定 `mThreadPerBlock = 128` | `kBlock = 128` |
| GlobalPool(2.8.3+) | 固定 `<<<count, 128>>>` | `localSize = per_block_size(128)` |
| TopKV2(2.6.3+) | 自定义 grid1/block1/grid2/block2(动态计算) | adapter 内计算,放进 intArgs |

源码:`source/backend/cuda/core/runtime/CUDARuntime.cpp`
- 1.2.0: `blocks_num()` 内自适应分档(未注释)
- 1.2.1+: 自适应逻辑被注释,固定 `mThreadPerBlock = 128`

## 建议加入 corpus 的中间版本变体

按变更价值排序(签名 + 公式都变的最有价值):

1. **CONV_DW** — 4 个变更点(1.2.7, 2.0.2, 2.0.4, 2.2.3),演进最丰富
2. **Reductions(SUM/MEAN)** — 3 个过渡点(1.2.7 结构体, 2.5.1 显式 int+拆分, 2.8.2 MEAN 拆分)
3. **INTERP_NERAEST/ROUND/BILINEAR** — 2 个点(1.2.7 重打包, 2.0.4 加 c_p)
4. **ARGMAX** — 2 个点(2.0.4 输出类型+索引, 2.5.0 指针算术重写)
5. **LAYERNORM** — 2 个点(2.2.2 类型/累加器, 2.8.4 加 RMSNorm)
6. **PRELU / SCALE** — 各 1 个点(2.0.4 类型+公式重写)
7. **NCHW_2_NHWC** — 1 个点(2.7.1 src_offset inChannelPack)
8. **GRID_SAMPLE_NEAREST/BILINEAR** — 1 个点(2.8.0 dst_offset 公式)
9. **SELECT** — 1 个点(2.8.0 加 s1/s2 stride)
10. **SOFTMAX** — 1 个点(2.2.3 ReduceParam→显式 int)
11. **blitRegion** — 1 个点(2.4.2 加 count + fuseIndex)
12. **CLAMP** — 1 个点(1.2.7 float→T)
