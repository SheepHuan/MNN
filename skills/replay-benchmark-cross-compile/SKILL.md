---
name: replay-benchmark-cross-compile
description: Cross-compile MNN's replay_benchmark for AArch64 with the ARM GNU Toolchain 11.3 archive under /home/yanghuan/code/eperf-devlop/prebuilts/archives, using the repository's project/cross-compile/build.sh workflow. Use when the user asks to build replay_benchmark with the 11.3 cross-compiler or to produce the target executable from this repository.
---

# Replay Benchmark Cross-Compile

Use this workflow from the MNN repository root. It intentionally uses the
existing `project/cross-compile/build.sh` entry point and does not inspect or
modify `schema/private/` or `source/internal/`.

## Critical Constraints

- The 11.3 archive's `aarch64-none-linux-gnu-*` prefix is not the prefix used
  by MNN's AArch64 toolchain branch; aliases are required.
- Extract the toolchain once into its persistent directory. Do not unpack the
  archive into a new temporary directory on every build.
- `build.sh` alone does not enable `replay_benchmark`: `MNN_BUILD_BENCHMARK=ON`
  must be present in the build cache before invoking it.
- With OpenCL enabled, static MNN plus `replay_benchmark` duplicates
  `OpenCLWrapper.cpp` at link time. Use shared MNN libraries with
  `MNN_SEP_BUILD=ON`.
- Never put the root password in this Skill, a command line, or a log. Use an
  SSH key or interactive password entry.
- Verify the final ELF architecture; a successful CMake configure or host-side
  link is not evidence that the cross-compiled artifact is usable.

## Workflow

### 1. Check and persist the 11.3 toolchain

Use the 11.3 archive, not the older GCC 9.2 archive:

```bash
repo_root=$(pwd)
toolchain_root=/home/yanghuan/code/eperf-devlop/prebuilts/archives/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu
toolchain_archive="${toolchain_root}.tar.xz"
test -f "$repo_root/project/cross-compile/build.sh"
test -f "$toolchain_archive"

if test -e "$toolchain_root"; then
    test -x "$toolchain_root/bin/aarch64-none-linux-gnu-gcc"
else
    tar -xf "$toolchain_archive" -C "$(dirname "$toolchain_root")"
fi
toolchain_bin="$toolchain_root/bin"
compiler="$toolchain_bin/aarch64-none-linux-gnu-gcc"
test -x "$compiler"
"$compiler" --version | head -n 1
test "$("$compiler" -dumpversion)" = "11.3.1"
```

The extracted directory is persistent and must be reused by later builds. Do
not unpack the archive into a new temporary directory on every run. If the
directory already exists but the compiler is missing, stop and report the
incomplete installation instead of silently overlaying it.

### 2. Bridge the compiler prefix expected by MNN

The archive names its binaries `aarch64-none-linux-gnu-*`, while
`project/cross-compile/arm.toolchain.cmake` expects
`aarch64-linux-gnueabihf-*` for the supported `aarch64-gnueabihf` build name.
Create aliases under the build directory; do not modify the repository
toolchain file or the archive:

```bash
alias_bin="$repo_root/build-aarch64-gnueabihf/toolchain-alias"
mkdir -p "$alias_bin"
for tool in gcc g++ c++ cpp ar ranlib nm objcopy objdump strip readelf; do
    source="$toolchain_bin/aarch64-none-linux-gnu-$tool"
    if test -x "$source"; then
        ln -sfn "$source" "$alias_bin/aarch64-linux-gnueabihf-$tool"
    fi
done
export PATH="$alias_bin:$toolchain_bin:$PATH"
command -v aarch64-linux-gnueabihf-gcc
test "$(aarch64-linux-gnueabihf-gcc -dumpversion)" = "11.3.1"
```

Use the `aarch64-gnueabihf` argument below because it is the AArch64 option
implemented by `build.sh`; the compiler itself remains the 11.3
`aarch64-none-linux-gnu` compiler.

### 3. Enable the benchmark and invoke the project script

`MNN_BUILD_BENCHMARK` defaults to `OFF`, so seed the fixed build directory with
the required cache values before invoking the script. Setting an in-repository
install prefix also avoids an unintended write to `/usr/local` during the
script's `make install` step:

```bash
cmake -S "$repo_root" -B "$repo_root/build-aarch64-gnueabihf" \
    -DCMAKE_TOOLCHAIN_FILE="$repo_root/project/cross-compile/arm.toolchain.cmake" \
    -DARM_NAME=aarch64-gnueabihf \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$repo_root/build-aarch64-gnueabihf/install" \
    -DMNN_BUILD_BENCHMARK=ON \
    -DMNN_BUILD_TOOLS=ON \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_SEP_BUILD=ON \
    -DMNN_OPENCL=ON \
    -DMNN_USE_SYSTEM_LIB=OFF \
    -DMNN_KLEIDIAI=OFF

cd "$repo_root"
MAKEFLAGS="-j$(nproc)" ./project/cross-compile/build.sh aarch64-gnueabihf
```

The shared-library configuration avoids the OpenCL wrapper duplicate-symbol
failure. Do not use the previous static configuration for this OpenCL target.

If the build directory already has a cache configured for a different compiler
or toolchain, preserve it and use a fresh build directory only after adapting
the script's fixed `build-aarch64-gnueabihf` directory handling; never delete a
user build directory without explicit approval. A pre-existing compatible cache
may be reused.

### 4. Verify the output

The executable and shared libraries required by the OpenCL target are:

```bash
artifact="$repo_root/build-aarch64-gnueabihf/replay_benchmark.out"
test -x "$artifact"
test -f "$repo_root/build-aarch64-gnueabihf/libMNN.so"
test -f "$repo_root/build-aarch64-gnueabihf/express/libMNN_Express.so"
test -f "$repo_root/build-aarch64-gnueabihf/source/backend/opencl/libMNN_CL.so"
file "$artifact"
readelf -h "$artifact" | rg 'Class|Machine'
readelf -d "$artifact" | rg 'NEEDED|RPATH|RUNPATH'
```

The `file`/`readelf` output must identify an ELF 64-bit AArch64 executable. A
successful host-side link is not enough if the artifact is missing or has the
wrong architecture. Keep all three MNN shared libraries when deploying; the
build-tree RPATH is host-specific, so set `LD_LIBRARY_PATH` on the device.

### 5. Deploy to the Rhino Pi-X1

The device is `${RHINO_PI_USER}@${RHINO_PI_HOST}` (see `.env`), architecture
`aarch64`, and the required workspace is `$RHINO_PI_WORKSPACE`. Use an SSH
key for non-interactive runs:

```bash
device="${RHINO_PI_USER:-root}@${RHINO_PI_HOST:?source .env first}"
remote_root="${RHINO_PI_WORKSPACE:-/mnt/nvme/workspace}/replay-benchmark"
ssh "$device" "mkdir -p $remote_root/bin $remote_root/lib $remote_root/models $remote_root/records"
scp "$artifact" "$device:$remote_root/bin/"
scp "$repo_root/build-aarch64-gnueabihf/libMNN.so" \
    "$repo_root/build-aarch64-gnueabihf/express/libMNN_Express.so" \
    "$repo_root/build-aarch64-gnueabihf/source/backend/opencl/libMNN_CL.so" \
    "$device:$remote_root/lib/"
scp "$repo_root"/benchmark/models/*.mnn "$device:$remote_root/models/"
ssh "$device" "cd $remote_root && chmod 755 bin/replay_benchmark.out"
```

### 6. Record and replay all benchmark models

Record all models with Adreno OpenCL. The positional values are model, loop,
warmup, forward (`3`), GPU mode (`4`), and precision (`2`):

```bash
ssh "$device" "cd $remote_root && \
    LD_LIBRARY_PATH=\$PWD/lib bin/replay_benchmark.out models 2 2 3 4 2 \
    --record records/opencl"
```

Replay every recorded op and count failures:

```bash
ssh "$device" 'sh -s' <<'REMOTE'
set -eu
cd /mnt/nvme/workspace/replay-benchmark
export LD_LIBRARY_PATH="$PWD/lib"
pass=0
fail=0
for model in models/*.mnn; do
    name=${model##*/}
    record="records/opencl/$name"
    for op_file in "$record"/ops/*.op.fb; do
        test -f "$op_file"
        op=${op_file##*/}
        op=${op%%.op.fb}
        op=$(printf '%s\n' "$op" | sed 's/^0*//')
        test -n "$op" || op=0
        if output=$(bin/replay_benchmark.out --model "$model" --record "$record" \
            --op-id "$op" 2>&1); then
            pass=$((pass + 1))
        else
            fail=$((fail + 1))
            printf 'FAIL model=%s op_id=%s\n%s\n' "$name" "$op" "$output" | tail -n 8
        fi
    done
done
printf 'opencl_replay_pass=%s opencl_replay_fail=%s\n' "$pass" "$fail"
test "$fail" -eq 0
REMOTE
```

For CPU validation on Rhino Pi-X1, keep `MNN_ARM82=ON` (the default): the
device selects the ARMv8/ARM82 backend and reports
`MNN_FORWARD_CPU_EXTENSION`. The replay benchmark accepts this backend when
the requested forward is CPU, records `forward=13`, and maps it back to the
CPU runtime creator during replay. If ARM82 cannot create an isolated
Execution, replay falls back to the generic CPU backend, matching the normal
Session backup backend used for unsupported ARM82 operators. Do not disable
ARM82 for the default CPU test; `MNN_ARM82=OFF` is only an optional diagnostic
variant.

The ARM82 CPU validation recorded all 8 models and 1,152 ops, then replayed
1,151 successfully. The only remaining mismatch is
`resnet-v2-50.mnn` op 108 (`Reduction`, `MEAN`, axis 1): the record snapshot
contains the spatial sum (`1.8623047`) while isolated CPU replay correctly
computes the mean (`0.0380062`, exactly sum / 49). The same snapshot mismatch
is reproduced with `MNN_ARM82=OFF`, so classify it as a record/tensor snapshot
issue rather than an ARM82 execution failure.

Use this command to record the default ARMv8 CPU backend:

```bash
ssh "$device" "cd $remote_root && \
    LD_LIBRARY_PATH=\$PWD/lib bin/replay_benchmark.out models 2 2 0 4 2 \
    --record records/cpu-extension"
```

Replay CPU records with the same `ops/*.op.fb` enumeration shown above, using
`records/cpu-extension` instead of `records/opencl`. Preserve the full failure
log under `/mnt/nvme/workspace/replay-benchmark/records/`; do not convert the
one resnet snapshot mismatch into a pass by widening numeric tolerances.

### 7. Record the validation result

Keep the device-side records and failure output under
`/mnt/nvme/workspace/replay-benchmark/records/`. Report model counts, operation
counts, pass/fail counts, and the first diagnostic for each failure. Do not call
a partial replay result a full correctness pass.

The validated Rhino Pi-X1 run recorded all 8 models and 1,152 OpenCL ops. It
replayed 1,137 successfully and failed 15. Failures were concentrated in
`mobilenetV3` early BinaryOp/Loop cases, plus one SqueezeNet backend-creation
failure, one NASNet execution mismatch, and one ResNet output mismatch.

