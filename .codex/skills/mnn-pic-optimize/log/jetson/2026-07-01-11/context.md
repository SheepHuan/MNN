# Jetson PIC Export Contract And Decode Smoke

## Device Contract

Implemented a centralized PIC export contract:

- `jetson` -> CUDA family, backend `jetson`.
- `rhinopi` -> Adreno family, backend `rhinopi`.
- `orangpi` / `orangepi` -> generic/Mali family, backend `generic`.

Fail-fast checks reject:

- OrangePi/OrangPi with backend-specific decode fusion flags.
- Jetson with RhinoPi/Adreno-only flags such as `pic_decode_tiny_mlp_fusion` or `pic_decode_silu_nhwc_down_fusion`.
- gate/up dependent flags without `pic_decode_gateup_fusion`.
- conflicting `pic_export_device` and `pic_decode_fusion_backend`.

The exporter writes contract fields to `export_args.json` and `llm_config.json`:

- `pic_export_contract_version`
- `pic_export_device`
- `pic_decode_fusion_backend`
- `pic_decode_fusion_family`

## Jetson Fresh Export Tested

Model:

```text
.cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-jetson-contract-silumul-score10-20260701
```

Command shape:

```text
MNN_LLM_EXPORTER=pic
MNN_PIC_EXPORT_DEVICE=jetson
--paged_kv_max_tokens 4096
--pic_decode_repair_outputs
--pic_recompute_score_layer_idx 10
--pic_decode_tiny_fusion
```

Export succeeded and produced the expected contract:

```text
pic_export_device=jetson
pic_decode_fusion_backend=jetson
pic_decode_fusion_family=cuda
pic_decode_tiny_fusion=true
pic_decode_gateup_fusion=false
pic_decode_nhwc_linear_fusion=false
```

Graph check: the graph contains `PagedAttention` + `PicSiluMul`; it does not contain `PicGateUp*`, `PicPackedSiluMul`, or `PicAdreno*`.

## Runtime Result

Fresh export fails on Jetson CUDA at persistent text cache construction:

```text
POST /v1/prefill/text
HTTP 400
forwardRaw outputs empty seq_len=512 add=512 all_seq=0 gen_seq=0 paged_max=4096 request_capacity=4096 logical_length=0 previous=0 remove=0
```

This is the `/v1/prefill/text` prefill stage that builds the persistent text cache, not decode.

Removing the newly added contract fields from `llm_config.json` / `export_args.json` did not fix it, so the failure is not caused by runtime parsing those fields.

## Isolation Tests

Old verified baseline:

```text
.cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-silumul-score10
```

Old baseline passed `/v1/prefill/text` and completed decode x=0/1/3/5/7 at context 512:

```text
.cache/bench_ops/mlp_fusion_20260701/old_silumul_score10_decode_fullreuse_20260701_retry/benchmark_decode.csv
```

Results:

```text
x=0  tpot=28.688 ms  ok
x=1  tpot=27.681 ms  ok
x=3  tpot=35.173 ms  ok
x=5  tpot=35.211 ms  ok
x=7  tpot=37.014 ms  ok
```

Graph comparison:

- old and fresh `llm.mnn.json` have identical `oplists`, `tensorName`, and `outputName`.
- external offsets are self-consistent in both models.
- old `llm.mnn` + fresh `llm.mnn.weight` passes `/v1/prefill/text`.
- fresh `llm.mnn` + old `llm.mnn.weight` still fails `/v1/prefill/text`.
- current Jetson `MNNConvert -f JSON` roundtrip of old JSON also fails `/v1/prefill/text`.

Conclusion: the fresh/roundtripped `llm.mnn` binary is the failing artifact, even when the JSON view is structurally identical to the old working graph. The old binary remains required for production until the MNN binary/JSON roundtrip issue is fixed.

## Production Decision

Do not promote fresh Jetson exports from the current converter path. Production must stay on the old verified `silumul-score10` baseline for Jetson decode. Gate/up packed CUDA remains unsupported for production because fresh exports fail before decode, during text-cache prefill.
