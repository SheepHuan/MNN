# Context

## Scope

Only these target models are considered:

```text
MiniCPM5-1B
Qwen3-4B
Llama3.2-3B
```

Llama3.2-1B was intentionally excluded from optimization decisions after the scope was tightened.

## Command Shape

All runs used the Jetson CUDA artifact:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda
```

Server env:

```text
MNN_PIC_GRAPH_PROFILE=1
MNN_PIC_GRAPH_PROFILE_TOP=1000
LD_LIBRARY_PATH=<artifact>/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib
```

Benchmark client:

```text
NO_PROXY=192.168.101.192,localhost,127.0.0.1
no_proxy=192.168.101.192,localhost,127.0.0.1
conda run -n kvshare-edge python .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py
  --base-url http://192.168.101.192:18131
  --device jetson --device-display "Jetson AGX Xavier" --backend cuda
  --mode full-reuse --contexts 1024 --budgets 0.00
  --repair-tokens 1,3 --decode-selectors top_hkvd
  --max-tokens 32 --warm-repeats 0 --repeats 1
  --require-exact-context --timeout 1200/1500
```

The server was stopped after profiling; port `18131` had no remaining listener.

## Results

### MiniCPM5-1B

Model:

```text
.cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary
score_layer_idx=1
```

TPOT:

```text
x1 decode_tpot_ms=86.7627
x3 decode_tpot_ms=97.5569
TPOT delta=+10.7942 ms/token
```

Per-TPOT type contribution:

```text
Convolution          +11.51 ms/token
PagedAttention        +0.26 ms/token
PicSparseAttention    -0.15 ms/token
PicScoreAttention     -0.03 ms/token
```

Per-TPOT category contribution:

```text
MLP gate/up/down total     +5.04 ms/token
attention q/k/v/o total    +6.53 ms/token
attention op total          +0.08 ms/token
```

Kernel route evidence:

```text
x1: conv_fpa_intb_1x1_tiny_gemv batch=2
x3: conv_fpa_intb_1x1_rows45_cublas batch=4, plus generic batch=4 Conv for some shapes
```

### Qwen3-4B

Model:

```text
.cache/weight/Qwen__Qwen3-4B-pic-boundary
score_layer_idx=1
hidden_size=2560
layer_nums=36
num_attention_heads=32
num_key_value_heads=8
head_dim=128
```

TPOT:

```text
x1 decode_tpot_ms=170.4681
x3 decode_tpot_ms=225.5854
TPOT delta=+55.1173 ms/token
```

Per-TPOT type contribution:

```text
Convolution          +51.67 ms/token
PagedAttention        +1.00 ms/token
PicSparseAttention    -0.22 ms/token
PicScoreAttention     -0.01 ms/token
```

Per-TPOT category contribution:

```text
MLP gate/up/down total     +28.93 ms/token
attention q/k/v/o total    +22.87 ms/token
attention op total          +0.78 ms/token
```

Kernel route evidence:

```text
x1: conv_fpa_intb_1x1_tiny_gemv batch=2
x3: conv_fpa_intb_1x1_rows45_cublas batch=4, plus generic batch=4 Conv
```

Qwen3-4B-specific note: server log contains generic `conv_fpa_intb_1x1 batch=4 ... runtime_dequant=1 static_dequant=0` rows. This means the rows4 cliff is not only cublas route overhead; part of the model falls out of static FP16 dequant cache and pays runtime dequant in the generic Conv path.

### Llama3.2-3B

Model:

```text
.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-silumul
score_layer_idx=1
hidden_size=3072
layer_nums=28
num_attention_heads=24
num_key_value_heads=8
head_dim=128
```

TPOT:

```text
x1 decode_tpot_ms=128.8164
x3 decode_tpot_ms=160.0733
TPOT delta=+31.2569 ms/token
```

Per-TPOT type contribution:

```text
Convolution          +28.20 ms/token
PagedAttention        +0.70 ms/token
PicSparseAttention    -0.23 ms/token
PicScoreAttention     -0.04 ms/token
```

Per-TPOT category contribution:

```text
MLP gate/up/down total     +17.72 ms/token
attention q/k/v/o total    +10.59 ms/token
attention op total          +0.43 ms/token
```

Kernel route evidence:

```text
x1: conv_fpa_intb_1x1_tiny_gemv batch=2
x3: conv_fpa_intb_1x1_rows45_cublas batch=4
```

## Interpretation

`repair_tokens=x` produces `active_tokens_per_decode_step=x+1`.

- `x=1` means active rows `2`, and target models hit the tiny GEMV route.
- `x=3` means active rows `4`, and target models hit batch-4 dense routes.
- This is the real cliff. `PicSparseAttention` is not the extra cost in the `x=1 -> x=3` step.
- `x=3/5/7` being much flatter than `x=1 -> x=3` is consistent with prior direct-op rows4/6/8 observations: once the path has entered the rows4/6/8 dense family, more rows do not create another route cliff of the same size.

Relevant source thresholds:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  picRows45CublasMatches: batch must be 4..8
  int4GemvBatchLimit: decode repair sparse uses tiny GEMV only for batch <= 3
```

## Next Engineering Target

The right target for this cliff is CUDA active-row weight-only dense:

1. Improve rows4/6/8 Linear for MLP gate/up/down and attention q/k/v/o projection.
2. For Qwen3-4B, inspect static dequant cache coverage and runtime-dequant fallback before designing new kernels.
3. Keep gate/up packed as a graph/export experiment only; it cannot fix the q/k/v/o projection part and did not beat endpoint TPOT.
4. Do not spend the next pass tuning `PicSparseAttention` for this specific cliff; it was flat or negative in all target models.
