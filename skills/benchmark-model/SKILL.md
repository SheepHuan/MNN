---
name: benchmark-model
description: 在 Rhino Pi-X1（AArch64，OpenCL/Vulkan）和 x86 CUDA 主机上跑 MNN benchmark 测延迟。用 tools/download_archives.sh 准备 ARM 11.3 交叉编译工具链，开启 OpenCL+Vulkan 后端交叉编译 benchmark.out；用 modelscope CLI 下载私有数据集 sheephuan/edgedploy-modelzoo/artifacts 到 /mnt/hdd_4tb/kernflow-models；用 rsync over SSH 把单个 .mnn 模型同步到设备（sshpass 密码登录）；调用 benchmark.out 测量 forward=3/7/2 的推理延迟。参考 skills/replay-benchmark-cross-compile 的工具链别名方案。
---

# Benchmark Model on RhinoPi / OrangePi / x86 CUDA

> **触发**：把 MNN 模型下发到 ARM 设备（Rhino Pi-X1 / Orange Pi）测 OpenCL/Vulkan
> 延迟，或在 x86 主机测 CUDA 延迟，使用 `benchmark.out`。
>
> **工具链来源**：`tools/download_archives.sh` 会把 ARM 11.3 工具链 archive
> 下载到 `prebuilts/archives/` 并校验 sha256；解压后路径为
> `prebuilts/archives/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/`。
> 该目录与 archive 已在 `.gitignore` 里，不会被 git 跟踪。
>
> **边界**：不读不改 `schema/private/`、`source/internal/`。设备口令/密码/token
> 不得写入本文件、命令行或日志；SSH 密码通过环境变量 `RHINO_PI_PASSWORD` /
> `ORANGE_PI_PASSWORD` 传入，ModelScope token 通过 `MODELSCOPE_API_TOKEN`
> 传入。两台设备均默认 `root` 登录。

## 关键约束

- **forward 数值**：CPU=0，OpenCL=3，Vulkan=7，CUDA=2。`benchmark.out` 调用
  形式为 `benchmark.out <models_dir> <loop> <warmup> <forward> <thread> <precision>`。
- **ARM 交叉编译产物**在 `build-aarch64-gnueabihf/`，工具链别名目录挂在该
  构建目录下；该目录已在 `.gitignore` 里。复用 `replay-benchmark-cross-compile`
  的别名方案（`aarch64-linux-gnueabihf-*` → `aarch64-none-linux-gnu-*`）。
- **Vulkan 后端在交叉编译时**：`MNN_USE_SYSTEM_LIB=OFF`，使用 lib wrapper（运行时
  `dlopen` 设备自带 `libvulkan.so`），不链接 host Vulkan；`MNN_VULKAN=ON`。
- **x86 CUDA 编译**：本机 gcc + `/usr/local/cuda`，`MNN_CUDA=ON`，
  `MNN_BUILD_BENCHMARK=ON`，单独的 `build-x86-cuda/` 目录。
- **模型下发用 rsync over SSH**（不是 scp），单模型同步：只同步用户指定的那一个
  `.mnn`，避免整目录覆盖。设备上工作目录与 lib 目录必须已就位（见 step 4）。
- **延迟报告**：`benchmark.out` 的 `Avg` 与各分位时间，单位 ms。OpenCL/Vulkan
  必须分别报；CPU 结果作为对照。任一后端创建失败会自动回退 CPU，必须在日志里
  确认 `Forward type:` 与 `MNN_FORWARD_*` 一致，否则视为假通过。
- **token/密码来源**：用户提供的 ModelScope 访问 token 从环境变量
  `MODELSCOPE_API_TOKEN` 读取；设备 SSH 密码从 `RHINO_PI_PASSWORD` /
  `ORANGE_PI_PASSWORD` 读取；均不得硬编码、不得落入日志。
- **密码登录用 `sshpass`**：两台设备均默认 `root` 登录且无 SSH key，所有
  `ssh`/`rsync` 调用前置 `sshpass -p "$<DEVICE>_PASSWORD"`；`sshpass` 缺失时
  先 `apt-get install -y sshpass`。Orange Pi 当前不可达，命令会超时失败属预期，
  不要重试到卡死。

## 设备与路径清单

| 角色 | 主机 | 用户 | 密码环境变量 | 远端工作目录 | 状态 |
|------|------|------|--------------|---------------|------|
| Rhino Pi-X1（Adreno OpenCL/Vulkan） | `192.168.101.227` | `root` | `RHINO_PI_PASSWORD`（`aidlux`） | `/mnt/nvme/workspace/benchmark-model` | 在线；Ubuntu，主机名 `kalama`，Adreno GPU，预装 libvulkan.so.1 + libvulkan_adreno.so |
| Orange Pi 5 Plus（OpenCL） | `192.168.101.113` | `root` | `ORANGE_PI_PASSWORD`（`orangepi`） | `/root/benchmark-model` | 当前不可达；可达后同流程 |
| x86 CUDA | `localhost` | — | — | `<repo>/build-x86-cuda` | CUDA toolkit 在 `/usr/local/cuda` |
| 模型本地缓存 | `localhost` | — | — | `/mnt/hdd_4tb/kernflow-models` | modelscope 下载目标 |

> 两台设备均默认 `root` 登录，密码登录（无 SSH key）。所有 `ssh`/`rsync`
> 调用必须前置 `sshpass -p "$<DEVICE>_PASSWORD"`。Orange Pi 不可达时跳过该设备，
> 只跑 Rhino + x86，并在结果里标注「Orange Pi 未跑（不可达）」。
>
> 下文示例用 `device=root@192.168.101.227` + `sshpass` 演示；Orange Pi 把
> IP/密码变量/工作目录替换即可。`sshpass` 缺失先 `apt-get install -y sshpass`。

---

## 工作流

### 0. 准备 ARM 11.3 交叉编译工具链

用仓库自带的 `tools/download_archives.sh` 下载并校验 ARM 11.3 工具链 archive
（sha256 内嵌在脚本 manifest 里）。该脚本会下载到 `prebuilts/archives/`，
archive 与解压目录均已在 `.gitignore` 里，不入库。

```bash
repo_root=$(pwd)
cd "$repo_root"

# 下载 + 校验 archive（幂等：已存在且校验通过则跳过）
bash tools/download_archives.sh --jobs 2

# 解压（若尚未解压）
toolchain_root="$repo_root/prebuilts/archives/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu"
toolchain_archive="${toolchain_root}.tar.xz"
if ! test -x "$toolchain_root/bin/aarch64-none-linux-gnu-gcc"; then
    tar -xf "$toolchain_archive" -C "$(dirname "$toolchain_root")"
fi
test -x "$toolchain_root/bin/aarch64-none-linux-gnu-gcc"
"$toolchain_root/bin/aarch64-none-linux-gnu-gcc" --version | head -n 1
test "$("$toolchain_root/bin/aarch64-none-linux-gnu-gcc" -dumpversion)" = "11.3.1"
```

#### 0.1 桥接编译器前缀（alias）

11.3 archive 的二进制是 `aarch64-none-linux-gnu-*`，而
`project/cross-compile/arm.toolchain.cmake` 期望 `aarch64-linux-gnueabihf-*`
（对应 `build.sh` 的 `aarch64-gnueabihf` 构建名）。在构建目录下建符号链接
别名，不改工具链文件、不改 archive：

```bash
alias_bin="$repo_root/build-aarch64-gnueabihf/toolchain-alias"
mkdir -p "$alias_bin"
for tool in gcc g++ c++ cpp ar ranlib nm objcopy objdump strip readelf; do
    source="$toolchain_root/bin/aarch64-none-linux-gnu-$tool"
    if test -x "$source"; then
        ln -sfn "$source" "$alias_bin/aarch64-linux-gnueabihf-$tool"
    fi
done
export PATH="$alias_bin:$toolchain_root/bin:$PATH"
command -v aarch64-linux-gnueabihf-gcc
test "$(aarch64-linux-gnueabihf-gcc -dumpversion)" = "11.3.1"
```

> 用 `aarch64-gnueabihf` 作为 `build.sh` 参数，因为这是该脚本实现的 AArch64
> 构建名；编译器本身仍是 11.3 的 `aarch64-none-linux-gnu`。
> `build-aarch64-gnueabihf/` 已在 `.gitignore` 里。

### 1. 交叉编译 ARM benchmark（OpenCL + Vulkan）

Vulkan 用 lib wrapper（`MNN_USE_SYSTEM_LIB=OFF`），避免链接 host Vulkan；运行
时 `dlopen` 设备自带的 `libvulkan.so`。沿用共享库配置以规避 OpenCL wrapper
重复符号问题，并新增 `libMNN_Vulkan.so`。

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
    -DMNN_VULKAN=ON \
    -DMNN_USE_SYSTEM_LIB=OFF \
    -DMNN_KLEIDIAI=OFF \
    -DMNN_ARM82=ON

cd "$repo_root"
MAKEFLAGS="-j$(nproc)" ./project/cross-compile/build.sh aarch64-gnueabihf
```

> 若缓存里已有 OpenCL-only 配置，cmake 会增量更新；Vulkan 源文件首次编译较慢
> 属正常。不要删 `build-aarch64-gnueabihf/` 重新配置，除非 step 0 的别名检查
> 失败。`PATH` 必须保留 `$alias_bin`（见 step 0.1），否则 `build.sh` 找不到
> `aarch64-linux-gnueabihf-gcc`。

### 2. 交叉编译产物校验

`benchmark.out` 与四个 `.so` 是必选项。`replay_benchmark.out` 是可选项：它的
CUDA bridge（`replay_benchmark/kernel_corpus_bridge/cuda/`）强依赖
`cuda_runtime.h`，交叉编译环境没有 CUDA toolkit 时会编译失败，**这是预期**，
不影响本 skill。`replay_benchmark.out` 仅 `replay-benchmark-cross-compile` skill
需要，本 skill 用 `benchmark.out` 即可。

```bash
b="$repo_root/build-aarch64-gnueabihf"

# 必选项
test -x "$b/benchmark.out"
test -f "$b/libMNN.so"
test -f "$b/express/libMNN_Express.so"
test -f "$b/source/backend/opencl/libMNN_CL.so"
test -f "$b/source/backend/vulkan/libMNN_Vulkan.so"
file "$b/benchmark.out" | rg 'ELF 64-bit.*aarch64'
readelf -d "$b/benchmark.out" | rg 'NEEDED|RPATH|RUNPATH'

# 可选项（缺 CUDA toolkit 时会失败，可跳过）
test -x "$b/replay_benchmark.out" 2>/dev/null \
    && echo "replay_benchmark.out: ok" \
    || echo "replay_benchmark.out: missing (optional, skip)"
```

`benchmark.out` NEEDED 会出现 `libMNN.so`、`libMNN_CL.so`、`libMNN_Vulkan.so`、
`libMNN_Express.so`；OpenCL/Vulkan `.so` 必须随 benchmark 一起部署到设备。

### 3. x86 CUDA 本机编译（可选，并行）

独立目录，避免污染 ARM 构建缓存：

```bash
cmake -S "$repo_root" -B "$repo_root/build-x86-cuda" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$repo_root/build-x86-cuda/install" \
    -DMNN_BUILD_BENCHMARK=ON \
    -DMNN_BUILD_TOOLS=ON \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_SEP_BUILD=ON \
    -DMNN_CUDA=ON \
    -DMNN_KLEIDIAI=OFF
cmake --build "$repo_root/build-x86-cuda" -j"$(nproc)"
```

> CUDA toolkit 默认在 `/usr/local/cuda`，cmake 的 `find_package(CUDA)` 会自动
> 找到。若装在别处，设 `CUDA_HOME` 或 `-DCUDA_TOOLKIT_ROOT_DIR=<path>`。

校验：

```bash
b="$repo_root/build-x86-cuda"
test -x "$b/benchmark.out"
test -f "$b/libMNN.so"
test -f "$b/source/backend/cuda/libMNN_CUDA.so"   # MNN_SEP_BUILD=ON 时
ldd "$b/benchmark.out" | rg 'libMNN|libcudart'
```

### 4. 部署 benchmark 二进制与库到 ARM 设备（一次性）

首次部署用 rsync 把 bin/lib 同步到设备工作目录（幂等，重复执行安全）：

```bash
device_user=root
device_host=192.168.101.227
device="${device_user}@${device_host}"
remote_root=/mnt/nvme/workspace/benchmark-model
b="$repo_root/build-aarch64-gnueabihf"

command -v sshpass >/dev/null || sudo apt-get install -y sshpass
: "${RHINO_PI_PASSWORD:?set RHINO_PI_PASSWORD first}"
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no "$device" \
    "mkdir -p $remote_root/bin $remote_root/lib $remote_root/models $remote_root/result"

SSHPASS="$RHINO_PI_PASSWORD" sshpass -e rsync -azP --inplace \
    "$b/benchmark.out" "$b/replay_benchmark.out" \
    "$device:$remote_root/bin/"

SSHPASS="$RHINO_PI_PASSWORD" sshpass -e rsync -azP --inplace \
    "$b/libMNN.so" \
    "$b/express/libMNN_Express.so" \
    "$b/source/backend/opencl/libMNN_CL.so" \
    "$b/source/backend/vulkan/libMNN_Vulkan.so" \
    "$device:$remote_root/lib/"

SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no "$device" \
    "chmod 755 $remote_root/bin/benchmark.out $remote_root/bin/replay_benchmark.out"
```

> 用 `sshpass -e` 从环境变量 `SSHPASS` 读密码，避免密码出现在 `ps`/命令行。
> Orange Pi 部署同形：`device_host=192.168.101.113`、
> `remote_root=/root/benchmark-model`、`SSHPASS="$ORANGE_PI_PASSWORD"`。
> 二进制与库变化后重跑此步即可，rsync `--inplace` 避免重传整文件。

### 5. 下载模型到本地缓存

数据集 `sheephuan/edgedploy-modelzoo` 为私有，需要 token。**token 从环境变量
`MODELSCOPE_API_TOKEN` 读取**，不写入本文件、不进命令历史、不进日志。
用 uv 管理的仓库 venv（`.venv/`，已在 `.gitignore` 里），装 modelscope CLI。

#### 5.1 准备 venv 并安装 modelscope（首次）

```bash
cd "$(git rev-parse --show-toplevel)"   # 仓库根
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "modelscope>=1.18.0"
# CLI 入口：.venv/bin/modelscope （别名 .venv/bin/ms）
```

#### 5.2 下载 artifacts 子目录到本地缓存

用 `modelscope download` 的 `--include "artifacts/*"` 只拉该子目录，
`--local-dir` 直写到目标路径（绕过 cache，幂等增量）：

```bash
: "${MODELSCOPE_API_TOKEN:?set MODELSCOPE_API_TOKEN env var first}"
MODELSCOPE_API_TOKEN="$MODELSCOPE_API_TOKEN" \
.venv/bin/modelscope download \
    --repo-type dataset sheephuan/edgedploy-modelzoo \
    --revision master \
    --local-dir /mnt/hdd_4tb/kernflow-models \
    --include "artifacts/*"
```

下载后用 `ls -la /mnt/hdd_4tb/kernflow-models/artifacts/` 确认 `.mnn` 文件就位。
重跑该命令会增量同步（已存在且未变的文件跳过）。

### 6. 下发单个模型到设备（rsync，按需）

**只同步用户点名的那一个 `.mnn`**，避免整目录覆盖设备上其他文件：

```bash
device_user=root
device_host=192.168.101.227
device="${device_user}@${device_host}"
remote_root=/mnt/nvme/workspace/benchmark-model
local_models=/mnt/hdd_4tb/kernflow-models/artifacts
model=${1:?usage: model=<name>.mnn}

: "${RHINO_PI_PASSWORD:?set RHINO_PI_PASSWORD first}"
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e rsync -azP --inplace \
    "$local_models/$model" \
    "$device:$remote_root/models/"
```

封装成 `skills/benchmark-model/scripts/push_model.sh <name>.mnn`，用户调用即可。
不要 `rsync` 整个 `local_models/` 目录——磁盘与带宽都浪费，且会清掉设备上
手动放的对比模型。

### 7. 在设备上跑 benchmark 测延迟

`benchmark.out` 参数顺序：`models_dir loop warmup forward thread precision`。
默认 `loop=20 warmup=10 thread=4 precision=2`（fp16，设备支持时）。

#### 7.1 Rhino Pi-X1：OpenCL

```bash
: "${RHINO_PI_PASSWORD:?set RHINO_PI_PASSWORD first}"
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no "$device" \
    "cd $remote_root && \
    LD_LIBRARY_PATH=\$PWD/lib \
    bin/benchmark.out models 20 10 3 4 2 2>&1 | tee result/opencl_\$(date +%s).log"
```

#### 7.2 Rhino Pi-X1：Vulkan

> Rhino Pi-X1 跑的是 Ubuntu（不是 Android），Adreno Vulkan 驱动
> `libvulkan_adreno.so` + `libvulkan.so.1` 已预装在 `/lib/aarch64-linux-gnu/`。
> 但系统只有 `libvulkan.so.1`，没有 `libvulkan.so`（无版本号软链），而 MNN
> wrapper 模式 `dlopen("libvulkan.so")` 找不到它。运行前需在 `lib/` 里建软链：
> `ln -sf /usr/lib/aarch64-linux-gnu/libvulkan.so.1 lib/libvulkan.so`

```bash
: "${RHINO_PI_PASSWORD:?set RHINO_PI_PASSWORD first}"
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no "$device" \
    "ln -sf /usr/lib/aarch64-linux-gnu/libvulkan.so.1 $remote_root/lib/libvulkan.so 2>/dev/null; \
     cd $remote_root && \
     LD_LIBRARY_PATH=\$PWD/lib \
     bin/benchmark.out models 20 10 7 4 2 2>&1 | tee result/vulkan_\$(date +%s).log"
```

#### 7.3 Rhino Pi-X1：CPU（对照）

```bash
: "${RHINO_PI_PASSWORD:?set RHINO_PI_PASSWORD first}"
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no "$device" \
    "cd $remote_root && \
    LD_LIBRARY_PATH=\$PWD/lib \
    bin/benchmark.out models 20 10 0 4 2 2>&1 | tee result/cpu_\$(date +%s).log"
```

> Orange Pi 5 Plus：可达时把 `device_host=192.168.101.113`、
> `remote_root=/root/benchmark-model`、`SSHPASS="$ORANGE_PI_PASSWORD"` 替换进
> 7.1/7.3（Orange Pi 只测 OpenCL + CPU，不测 Vulkan）。当前 113 不可达，跳过。

#### 7.4 x86 CUDA（本机）

```bash
b="$repo_root/build-x86-cuda"
local_models=/mnt/hdd_4tb/kernflow-models/artifacts
mkdir -p "$b/result"
LD_LIBRARY_PATH="$b:$b/source/backend/cuda" \
"$b/benchmark.out" "$local_models" 20 10 2 4 2 2>&1 \
    | tee "$b/result/cuda_$(date +%s).log"
```

> 若 `libMNN_CUDA.so` 在 `build-x86-cuda/source/backend/cuda/`，把它也加进
> `LD_LIBRARY_PATH`。CUDA 默认 RPATH 通常已 OK，加 path 是为了 MNN_SEP_BUILD。

### 8. 校验不是假通过

每份日志必须确认两点，否则不算数：

1. `Forward type:` 行打印的后端与请求一致（`CPU`/`OPENCL`/`VULKAN`/`CUDA`）。
2. 没有 `Create backend failed` / `fallback to CPU` 警告。

```bash
# 例：检查 OpenCL 日志没有回退
: "${RHINO_PI_PASSWORD:?set RHINO_PI_PASSWORD first}"
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no "$device" \
    "grep -E 'Forward type|backend failed|fallback|FALLBACK' \
    $remote_root/result/opencl_*.log | tail -50"
```

回退发生时，先查 `LD_LIBRARY_PATH` 是否包含 `libMNN_CL.so` / `libMNN_Vulkan.so`
所在目录，再查设备是否有 `/usr/lib/libOpenCL.so` / `libvulkan.so`。

### 9. 收集结果

```bash
: "${RHINO_PI_PASSWORD:?set RHINO_PI_PASSWORD first}"
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no "$device" \
    "cd $remote_root && \
    for f in result/*.log; do
        echo '=== '\$(basename \$f)' ==='
        grep -E 'Forward type|Avg|Model|backend failed|fallback' \$f
    done"
```

把每个模型的 `Avg (ms)`、`Min`、`Max` 按 OpenCL/Vulkan/CPU/CUDA 整理成一张表
回给用户。Orange Pi 可达后按 7.1/7.3 同形跑一次，结果并表。

### 10. 后续 / 排错

| 现象 | 排查 |
|------|------|
| OpenCL `Create backend failed` | 设备缺 `libOpenCL.so`；或 `LD_LIBRARY_PATH` 没含 `libMNN_CL.so` 目录 |
| Vulkan 回退 CPU | 设备缺 `libvulkan.so`；或 `libMNN_Vulkan.so` 没同步 |
| CUDA `CUDA backend not compiled` | x86 build 缓存没开 `MNN_CUDA=ON`；删 `build-x86-cuda` 重配 |
| 模型下发慢 | 确认 rsync 走的是局域网 IP，不是公网；`--inplace` 已开 |
| modelscope 401/403 | token 过期或无数据集权限；让用户重新设 `MODELSCOPE_API_TOKEN` |
| `modelscope: command not found` | 没用 `.venv/bin/modelscope`；或没装：`uv pip install --python .venv/bin/python "modelscope>=1.18.0"` |
| Orange Pi 不可达 | 当前 `192.168.101.113` 离线；不重试到卡死，跳过并在结果标注「未跑」 |
| SSH 密码认证失败 | 确认 `RHINO_PI_PASSWORD`/`ORANGE_PI_PASSWORD` 已设；`sshpass` 已装 |
| `sshpass: command not found` | `sudo apt-get install -y sshpass` |

---

## 与 `replay-benchmark-cross-compile` 的关系

- **参考**：工具链别名方案（`aarch64-linux-gnueabihf-*` → `aarch64-none-linux-gnu-*`）
  源自 `replay-benchmark-cross-compile` skill；该 skill 把工具链放在仓库外，
  本 skill 改为用 `tools/download_archives.sh` 下载到仓库 `prebuilts/archives/`，
  路径自洽、可复现。
- **扩展**：追加 `MNN_VULKAN=ON`（lib wrapper 模式），新增 `benchmark.out` 编译
  目标（`replay-benchmark-cross-compile` 只编 `replay_benchmark.out`），新增
  x86 CUDA 本机编译分支，新增 modelscope CLI 数据集下载与单模型 rsync 下发。
- **不重复**：不重写工具链别名逻辑的原理（只把路径内联进本 skill）、
  不改 `arm.toolchain.cmake`、不改 `tools/download_archives.sh`。
