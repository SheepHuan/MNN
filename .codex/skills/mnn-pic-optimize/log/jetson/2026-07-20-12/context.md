# Jetson CUDA 3072 Prefill Workspace Context

The original Jetson Qwen3-4B 3072 sweep returned `curl 52` because `pic_server`
was killed during `/v1/prefill/text`. Jetson kernel logs showed:

```text
Out of memory: Killed process ... (pic_server) total-vm=33297212kB
```

`PagedAttentionExecution::ensurePrefillTemp` allocates both QK and softmax
buffers as `batch * heads * max_q_piece * kv_len` floats for every layer. The
previous Jetson policy used `max_q_piece=1024` at context 3072, which is about
384 MiB per buffer and about 768 MiB per layer. Qwen3-4B has 36 layers, so the
per-layer execution workspaces exceeded the Xavier unified-memory budget.

The policy now uses `ceil(attnLen / 256)` pieces for all `attnLen > 256` and
keeps one piece for shorter queries. This only changes prefill workspace sizing;
decode `attnLen=1` remains unchanged.

Build and validation:

```text
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

The rebuilt `libMNN_Cuda_Main.so` was synced atomically to the remote artifact
root. The fixed sweep built the 3058-token persistent PIC cache successfully,
then measured full-reuse and six independent cacheblend requests at ratios
0.05, 0.10, 0.20, 0.30, 0.40, and 0.50. No energy capture was enabled.

