# Jetson CUDA Gate/Up Re-export Decode Check

## Scope

Task: re-export CUDA gate/up fused Llama3.2-1B decode-repair models and run real end-to-end decode speed for x=`1/3/5/7`.

Device and runtime:

- Device: Jetson Orin NX
- Backend: CUDA
- Artifact: `.cache/output/mnn/artifacts/jetson_cross_cuda`
- Server port: remote `18131`, local tunnel `19131`
- Benchmark client: `.codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py`
- Matrix: contexts `512,1024,1536`, mode `full-reuse`, selector `top_hkvd`, max_tokens `32`, repeats `3`, warm repeats `1`

## Fresh Exports Tried

1. `AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-gateup_packed_cuda_fresh_20260701`
   - Flags: `--pic_decode_fusion_backend cuda --pic_decode_tiny_fusion --pic_decode_gateup_fusion --pic_decode_nhwc_linear_fusion --pic_decode_nhwc_linear_scope mlp_gateup`
   - Graph contained `PicLinearNhwcWeightOnly=16`, `PicPackedSiluMul=16`.
   - Failed `/v1/prefill/text`.

2. `AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-gateup_concat_cuda_fresh_20260701`
   - Flags: CUDA tiny + gateup fusion, no NHWC linear.
   - Graph contained `PicPackedSiluMul=16`.
   - Failed `/v1/prefill/text`.

3. `AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-gateup_concat_cuda_fixed_20260701`
   - Exporter adjusted to emit historical `concat_conv -> ConvertTensor -> Reshape -> PicPackedSiluMul` CUDA layout instead of direct `packed_input_nc4`.
   - Graph contained `PicPackedSiluMul=16`, no `PicLinearNhwcWeightOnly`, no `PicBench`.
   - Failed `/v1/prefill/text`.

4. `AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-gateup_concat_cuda_fixed_score10_20260701`
   - Same fixed concat export, but `pic_recompute_score_layer_idx=10` to match current `silumul-score10`.
   - Failed `/v1/prefill/text`.

Common failure:

```text
Failed to build persistent text cache [status=4, current=-1, all_seq=0, prompt=512, gen_seq=0, output_tokens=0, last_error=forwardRaw outputs empty seq_len=512 add=512 all_seq=0 gen_seq=0 paged_max=4096 request_capacity=4096 logical_length=0 previous=0 remove=0]
```

The historical `AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-gateup-packed-silu-patch` still passes smoke on the same artifact, so this is fresh export / graph layout compatibility, not a blanket runtime outage.

## End-to-End Decode Results

Fresh exports were not timed because they fail text-cache prefill. For a usable packed-gate/up signal, the historical patch was retested against current production baseline `silumul-score10`.

TPOT ms/token:

```text
model                                           context  x=1     x=3     x=5     x=7
Llama3.2-1B gateup-packed-silu-patch            512      29.975  37.247  37.594  39.238
Llama3.2-1B gateup-packed-silu-patch            1024     34.816  42.920  41.592  43.907
Llama3.2-1B gateup-packed-silu-patch            1536     39.715  48.054  45.891  48.810
Llama3.2-1B silumul-score10 baseline            512      27.615  35.130  35.166  36.858
Llama3.2-1B silumul-score10 baseline            1024     32.247  40.426  39.387  41.719
Llama3.2-1B silumul-score10 baseline            1536     37.611  46.007  43.469  46.449
```

Packed patch minus baseline:

```text
context  x  delta_ms  delta_pct
512      1  +2.360    +8.55%
512      3  +2.116    +6.02%
512      5  +2.428    +6.90%
512      7  +2.381    +6.46%
1024     1  +2.569    +7.97%
1024     3  +2.494    +6.17%
1024     5  +2.206    +5.60%
1024     7  +2.188    +5.24%
1536     1  +2.104    +5.59%
1536     3  +2.047    +4.45%
1536     5  +2.422    +5.57%
1536     7  +2.361    +5.08%
```

All 24 timed rows were `benchmark_status=ok`; the client required `mnn_token_id_sparse_decode`.

## Decision

- Current production decode path should stay on `silumul-score10` for this matrix.
- Do not use fresh CUDA gate/up exports in production; they fail before decode at persistent text-cache build.
- Do not promote the historical packed patch either; even though it runs, it is slower than current baseline in this end-to-end decode check.
- Next debugging target, if revisiting gate/up export, is fresh export compatibility around `forwardRaw` returning empty outputs; old patched graph success shows the runtime can execute a packed gate/up graph, but current exporter output is not yet equivalent enough for `/v1/prefill/text`.
