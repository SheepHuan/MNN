#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_opencl_remote_op_bench.sh --device rhino|orangepi [options]

Options:
  --device <name>          Target device: rhino or orangepi.
  --test <name>            run_test.out case name.
                           Default: bench_ops/opencl/perf/WeightOnlyConv
  --case <name>            Value for MNN_BENCH_OPENCL_WEIGHT_ONLY_CASE.
  --rows <spec>            Value for MNN_BENCH_OPENCL_WEIGHT_ONLY_ROWS.
  --family <name>          Value for MNN_BENCH_OPENCL_COMPACT_DENSE_FORCE_FAMILY.
  --storage <name>         Value for MNN_BENCH_OPENCL_WEIGHT_ONLY_FORCE_STORAGE.
  --tune-level <name>      none|fast|normal|heavy|wide. Default: heavy
  --warmup <n>             Bench warmup count. Default: 10
  --repeat <n>             Bench repeat count. Default: 40
  --extra-env 'K=V ...'    Extra environment entries appended remotely.
  --                         Remaining args passed through to run_test.out.

Examples:
  run_opencl_remote_op_bench.sh --device rhino \
    --case minicpm_hidden_to_ffn --rows 519

  run_opencl_remote_op_bench.sh --device rhino \
    --case minicpm_hidden_to_attn --rows 269 \
    --family pic_quant_wg4x32 --storage image
EOF
}

device=""
test_name="bench_ops/opencl/perf/WeightOnlyConv"
case_name=""
rows_spec=""
family_name=""
storage_name=""
tune_level="heavy"
warmup="10"
repeat="40"
extra_env=""
pass_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      device="${2:-}"; shift 2 ;;
    --test)
      test_name="${2:-}"; shift 2 ;;
    --case)
      case_name="${2:-}"; shift 2 ;;
    --rows)
      rows_spec="${2:-}"; shift 2 ;;
    --family)
      family_name="${2:-}"; shift 2 ;;
    --storage)
      storage_name="${2:-}"; shift 2 ;;
    --tune-level)
      tune_level="${2:-}"; shift 2 ;;
    --warmup)
      warmup="${2:-}"; shift 2 ;;
    --repeat)
      repeat="${2:-}"; shift 2 ;;
    --extra-env)
      extra_env="${2:-}"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      pass_args+=("$@")
      break ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$device" ]]; then
  echo "--device is required" >&2
  usage >&2
  exit 2
fi

ssh_target=""
remote_root=""
remote_run_test=""
remote_lib_dir=""
remote_preload=""

case "$device" in
  rhino)
    ssh_target="aidlux@192.168.101.227"
    remote_root="/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl_testbench"
    remote_run_test="./bin/run_test.out"
    remote_lib_dir="$remote_root/lib"
    remote_preload="/usr/lib/libOpenCL_adreno.so"
    ;;
  orangepi)
    ssh_target="orangepi@192.168.101.113"
    remote_root="/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus_testbench"
    remote_run_test="./bin/run_test.out"
    remote_lib_dir="$remote_root/lib"
    remote_preload=""
    ;;
  *)
    echo "Unsupported device: $device" >&2
    exit 2
    ;;
esac

remote_env=(
  "LD_LIBRARY_PATH=$remote_lib_dir:/usr/lib:/usr/lib/aarch64-linux-gnu"
  "MNN_OPENCL_TUNE_LEVEL=$tune_level"
  "MNN_BENCH_OPENCL_WEIGHT_ONLY_WARMUP=$warmup"
  "MNN_BENCH_OPENCL_WEIGHT_ONLY_REPEAT=$repeat"
)
if [[ -n "$remote_preload" ]]; then
  remote_env+=("LD_PRELOAD=$remote_preload")
fi
if [[ -n "$case_name" ]]; then
  remote_env+=("MNN_BENCH_OPENCL_WEIGHT_ONLY_CASE=$case_name")
fi
if [[ -n "$rows_spec" ]]; then
  remote_env+=("MNN_BENCH_OPENCL_WEIGHT_ONLY_ROWS=$rows_spec")
fi
if [[ -n "$family_name" ]]; then
  remote_env+=("MNN_BENCH_OPENCL_COMPACT_DENSE_FORCE_FAMILY=$family_name")
fi
if [[ -n "$storage_name" ]]; then
  remote_env+=("MNN_BENCH_OPENCL_WEIGHT_ONLY_FORCE_STORAGE=$storage_name")
fi
if [[ -n "$extra_env" ]]; then
  remote_env+=("$extra_env")
fi

printf -v env_prefix 'env %q ' "${remote_env[@]}"
printf -v pass_tail ' %q' "${pass_args[@]}"

remote_cmd=$(
  cat <<EOF
set -e
cd '$remote_root'
$env_prefix $remote_run_test '$test_name' 3 0 1 '' 2$pass_tail
EOF
)

exec ssh -o StrictHostKeyChecking=accept-new "$ssh_target" "bash -lc $(printf '%q' "$remote_cmd")"
