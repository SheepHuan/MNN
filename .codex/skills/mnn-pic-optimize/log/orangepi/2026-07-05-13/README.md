# OrangePi OpenCL decode-repair fused append negative result

This hour tested a fused append + attention record path for PIC dualgraph
decode-repair on OrangePi OpenCL, targeting the same x=0/1/3/5/7 family.

Result: do not promote this path. On Qwen3-4B ctx1024, warm A/B showed:

```text
default append+attention record:
  x0 162.869 ms
  x1 237.751 ms

fused append + single record:
  x0 177.585 ms
  x1 255.331 ms
```

The fused path regressed x0 by about 14.7 ms and x1 by about 17.6 ms. The
likely cause is that fusing append into the attention kernel saves one dispatch
but adds global write-after-read ordering, barrier/fence cost, extra register
pressure, and duplicate append-side work inside the attention workgroup shape.
On Mali this cost is larger than the saved append launch/record overhead.

Code status:

- The new fused-record C++ route, method, and state are commented out with
  `#if 0`.
- `_decodeRepairFusedAppendEnabled()` now returns `false`; the env-gated branch
  is kept inside `#if 0` as negative evidence.
- The default path remains the existing two-kernel append + transposed-K qtile
  attention record path.
- The dormant fused kernel output-offset correctness fix remains active because
  it does not enable the failed route.

Artifact status:

- Rebuilt OrangePi `pic_server` artifact after disabling the fused path.
- Verified outputs are ARM aarch64.
- Synced artifacts to
  `/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/`.

See `context.md` for commands, CSV paths, and the forward plan.

## Follow-up profile

After disabling the failed fused path, default-path diagnostics were run on
Qwen3-4B ctx1024 x0/x1.

Warm software detail profile, measure segment:

```text
x0: PagedAttention sub-ops 20.959 ms
    append 2.742 ms, attention scan 18.128 ms, rank 0.034 ms
x1: PagedAttention sub-ops 67.836 ms
    append 4.117 ms, attention scan 62.052 ms, rank 1.600 ms
```

PMC was enabled in a separate diagnostic artifact and then the normal no-PMC
artifact was restored. Attention counters were usable; append counters were not
stable on the very short append scopes. Median attention records:

```text
q1/x0 attention: wall 644 us/layer, external read ~2.17 MB/layer
q2/x1 attention: wall 1622 us/layer, external read ~4.28 MB/layer
```

Current conclusion: x0's remaining PIC-over-normal delta is mainly the
transposed-K qtile attention scan over PagedCache/decodeKey, not rank and not
append. x1 grows roughly with query rows because the attention kernel repeats
the K/V scan work.

Lane-force A/B was also tested through the existing explicit env controls. It
did not produce a stable fix:

```text
single-run x0/x1 ms:
  default repeat2 159.776 / 235.736
  lane64          161.176 / 237.411
  lane32          155.950 / 237.902
  lane128         159.766 / 232.871

x0 repeat=3:
  default         156.940
  lane32          159.115
```

Do not promote lane32/64/128 forcing as a default. The next useful change needs
to reduce real K/V scan work or improve K/V reuse, not just retune lane width.

## True Normal x0 Follow-up

After the fused/lane experiments, true normal Qwen3-4B ctx1024 was rerun with
the ordinary non-PIC export:

```text
true normal x0, n=1: 154.962 ms/token
true normal x0, n=8: 155.264 ms/token
```

The latest PIC x0 formal-style repeat from this hour was `156.940 ms`, so the
current default path is not consistently +20 ms/token versus true normal. The
~20 ms number observed in detail profile is the PIC PagedAttention sub-path
cost inside a synchronized debug/profile run:

```text
PIC x0 PagedAttention measure segment:
  total 20.959 ms = append 2.742 ms + attention 18.128 ms + rank 0.034 ms
```

True normal OpenCL decode does not use `PagedAttentionBufExecution`; it uses the
ordinary `Attention` op / `AttentionBufExecution` path. Therefore
`MNN_PAGED_ATTENTION_PROFILE` prints no normal attention rows. A whole-response
`llm_bench --profile` confirms normal Qwen3-4B uses ordinary `Attention` plus
`Convolution`, but that profile includes the 1024-token prefill and cannot
isolate decode-only attention.

Implication: the remaining work is not to remove a simple extra 20 ms wall-time
delay. It is to reduce the PIC/PagedAttention decode-repair subpath so its
PagedCache semantics approach the ordinary `Attention` decode implementation.
The likely target is a q1/q2 attention rewrite that keeps the decode-repair
family but avoids the current PagedAttention append + transposed `decodeKey`
scan overhead where possible.

## Code Organization Follow-up

Started splitting device-specific policy away from the large execution files:

- OpenCL Mali policy moved into `PagedAttentionMaliUtils.{hpp,cpp}` for
  decode-repair qtile/lane choices and sparse-flash variant cache rejection.
- OpenCL Adreno sparse-flash reject/schedule/static-workspace policy moved into
  `PagedAttentionAdrenoUtils.{hpp,cpp}`; the execution body now delegates those
  decisions instead of expanding Adreno heuristics inline.
- CUDA Jetson sparse qtile policy moved into
  `PagedAttentionJetsonPolicy.{hpp,cpp}`. `PagedAttentionExecution.cu` keeps the
  kernel launch body, while qtile variant names, decode-repair variant choice,
  large sparse tile thresholds, and prefill q-split thresholds are centralized.

This is a behavior-preserving cleanup. It does not change default OpenCL
Mali/Adreno or CUDA Jetson routing, and it does not promote the previously
negative fused-append or split-QK/QKV experiments.
