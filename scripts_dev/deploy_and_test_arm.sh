#!/usr/bin/env bash
# Deploy MNN source to ARM device (rhinopi/orangepi), cross-build natively on
# device, run OpenCL kernel latency + PMU availability tests, collect results.
#
# Prerequisites:
#   - .env with RHINO_PI_PASSWORD / ORANGE_PI_PASSWORD
#   - sshpass installed on host
#   - device has gcc/g++/cmake/make + enough disk
#
# Usage:
#   ./scripts_dev/deploy_and_test_arm.sh rhinopi
#   ./scripts_dev/deploy_and_test_arm.sh orangepi
#   ./scripts_dev/deploy_and_test_arm.sh both

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a && source .env 2>/dev/null && set +a || true

DEVICE="${1:-both}"

DEVICES_TO_RUN=()
if [ "$DEVICE" = "rhinopi" ] || [ "$DEVICE" = "both" ]; then
    DEVICES_TO_RUN+=("rhinopi")
fi
if [ "$DEVICE" = "orangepi" ] || [ "$DEVICE" = "both" ]; then
    DEVICES_TO_RUN+=("orangepi")
fi
if [ ${#DEVICES_TO_RUN[@]} -eq 0 ]; then
    echo "usage: $0 rhinopi|orangepi|both"
    exit 1
fi

run_on_device() {
    local name="$1" host="$2" pass="$3" workspace="$4"
    local remote="$workspace/mnn-arm-build"
    echo ""
    echo "=================================================================="
    echo "=== $name ($host) ==="
    echo "=================================================================="

    # 1. rsync source (skip .git, build dirs, kernel corpus sources except .spv, models)
    echo "--- [1/6] rsync source to $name:$remote ---"
    SSHPASS="$pass" sshpass -e rsync -az --delete \
        -e "ssh -o StrictHostKeyChecking=no" \
        --exclude='.git/' \
        --exclude='build*/' \
        --exclude='*.deb' \
        --exclude='corpus_perf_*.json' \
        --exclude='replay_benchmark/kernel_corpus/__pycache__/' \
        --exclude='records/' \
        --exclude='docs/' \
        --exclude='test/' \
        --exclude='apps/' \
        --exclude='demo/' \
        --exclude='transformers/' \
        --exclude='pymnn/' \
        --exclude='package/' \
        ./ "$SSH_OPT_USER@$host:$remote/"
    # Sync only the pre-compiled .spv files (1.1MB total) — needed by ncnn
    # Vulkan shaders when the device lacks glslangValidator.
    SSHPASS="$pass" sshpass -e rsync -az \
        -e "ssh -o StrictHostKeyChecking=no" \
        --include='*/' \
        --include='*.spv' \
        --exclude='*' \
        ./replay_benchmark/kernel_corpus/sources/ \
        "$SSH_OPT_USER@$host:$remote/replay_benchmark/kernel_corpus/sources/"
    echo "  rsync done"

    # 2. CMake configure (if not cached)
    echo "--- [2/6] cmake configure on $name ---"
    SSHPASS="$pass" sshpass -e ssh -o StrictHostKeyChecking=no "$SSH_OPT_USER@$host" bash <<EOF
set -euo pipefail
cd $remote
mkdir -p build && cd build
if [ ! -f CMakeCache.txt ]; then
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_SHARED_LIBS=ON \
        -DMNN_SEP_BUILD=ON \
        -DMNN_BUILD_BENCHMARK=ON \
        -DMNN_BUILD_TEST=OFF \
        -DMNN_BUILD_DEMO=OFF \
        -DMNN_BUILD_TOOLS=OFF \
        -DMNN_BUILD_CONVERTER=OFF \
        -DMNN_BUILD_QUANTOOLS=OFF \
        -DMNN_BUILD_TRAIN=OFF \
        -DMNN_BUILD_LLM=OFF \
        -DMNN_BUILD_AUDIO=OFF \
        -DMNN_BUILD_EVALUATION=OFF \
        -DMNN_OPENCL=ON \
        -DMNN_VULKAN=ON \
        -DMNN_USE_SYSTEM_LIB=OFF \
        -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON \
        -DMNN_ARM82=ON
fi
EOF
    echo "  cmake configure done"

    # 3. Build
    echo "--- [3/6] build on $name ---"
    SSHPASS="$pass" sshpass -e ssh -o StrictHostKeyChecking=no "$SSH_OPT_USER@$host" bash <<EOF
set -euo pipefail
cd $remote/build
cmake --build . --target replay_benchmark.out replay_opencl_kernel_latency replay_opencl_pmu_availability -j\$(nproc) 2>&1 | tail -20
EOF
    echo "  build done"

    # 4. Copy corpus manifest files into build dir (so the test can find them)
    echo "--- [4/6] deploy corpus manifests ---"
    SSHPASS="$pass" sshpass -e ssh -o StrictHostKeyChecking=no "$SSH_OPT_USER@$host" bash <<EOF
set -euo pipefail
mkdir -p $remote/build/corpus
cp $remote/replay_benchmark/kernel_corpus/operator_cases.json $remote/build/corpus/
cp $remote/replay_benchmark/kernel_corpus/operators.json $remote/build/corpus/ 2>/dev/null || true
cp $remote/replay_benchmark/kernel_corpus/arg_layouts.json $remote/build/corpus/ 2>/dev/null || true
EOF
    echo "  corpus manifests deployed"

    # 5. Run latency test
    echo "--- [5/6] run replay_opencl_kernel_latency on $name ---"
    SSHPASS="$pass" sshpass -e ssh -o StrictHostKeyChecking=no "$SSH_OPT_USER@$host" bash <<EOF
set -euo pipefail
cd $remote/build
export LD_LIBRARY_PATH=.:\$PWD
export REPLAY_KERNEL_CORPUS_ROOT=$remote/replay_benchmark/kernel_corpus
export REPLAY_KERNEL_LATENCY_RUNS=5
export REPLAY_OPENCL_KERNEL_LATENCY_OUTPUT=$remote/build/opencl_kernel_latency.json
export REPLAY_VULKAN_KERNEL_LATENCY_OUTPUT=$remote/build/vulkan_kernel_latency.json
export REPLAY_MNN_OPENCL_KERNEL_LATENCY_OUTPUT=$remote/build/mnn_opencl_kernel_latency.json
export REPLAY_NCNN_VULKAN_KERNEL_LATENCY_OUTPUT=$remote/build/ncnn_vulkan_kernel_latency.json
./replay_opencl_kernel_latency 2>&1 | tail -30
EOF
    echo "  latency test done"

    # 6. Run PMU availability test (discovery only on first pass)
    echo "--- [6/6] run replay_opencl_pmu_availability on $name ---"
    SSHPASS="$pass" sshpass -e ssh -o StrictHostKeyChecking=no "$SSH_OPT_USER@$host" bash <<EOF
set -euo pipefail
cd $remote/build
export LD_LIBRARY_PATH=.:\$PWD
export REPLAY_KERNEL_CORPUS_ROOT=$remote/replay_benchmark/kernel_corpus
./replay_opencl_pmu_availability --filter OpenclPmu.Discovery 2>&1 | tail -20
EOF
    echo "  PMU discovery done"

    # 7. Pull result JSONs back
    echo "--- collect results from $name ---"
    local result_dir="arm-test-results/$name"
    mkdir -p "$result_dir"
    SSHPASS="$pass" sshpass -e rsync -az \
        -e "ssh -o StrictHostKeyChecking=no" \
        "$SSH_OPT_USER@$host:$remote/build/*.json" "$result_dir/" 2>/dev/null || true
    ls -la "$result_dir/" | tail -10
}

# Set SSH user
SSH_OPT_USER="${SSH_USER:-root}"

for dev in "${DEVICES_TO_RUN[@]}"; do
    case "$dev" in
        rhinopi)
            run_on_device "rhinopi" "$RHINO_PI_HOST" "$RHINO_PI_PASSWORD" "${RHINO_PI_WORKSPACE:-/mnt/nvme/workspace}"
            ;;
        orangepi)
            run_on_device "orangepi" "$ORANGE_PI_HOST" "$ORANGE_PI_PASSWORD" "${ORANGE_PI_WORKSPACE:-/mnt/ssd/workspace}"
            ;;
    esac
done

echo ""
echo "=== All done. Results in ./arm-test-results/ ==="
