# GPU kernel source corpus

该目录保存多个 MNN 和 ncnn 发布 tag 中与 GPU 执行有关的源码快照，供后续
`replay_benchmark` 的 OpenCL/Vulkan workload 选择和 PMU 对比使用。本阶段只归档源码，
不编译这些快照，也不把它们直接链接进 MNN。

## 当前覆盖

- MNN：73 个 tag，`1.2.0` 到 `3.6.0`；归档 `source/backend/opencl` 和
  `source/backend/vulkan`。
- ncnn：37 个有 Vulkan 源码的 tag，`20190611` 到 `20260526`；归档
  `src/layer/vulkan`、`src/command.{cpp,h}` 和 `src/gpu.{cpp,h}`。
- ncnn 的 OpenCL backend 标记为 `unavailable`，因为这些 tag 没有可归档的主线
  OpenCL kernel 实现。

每个版本的文件、commit、backend 状态和 SHA-256 都记录在 `manifest.json` 中；每个
版本同时保留上游 license 文件。不会收录模型、构建产物、库文件、PMU 记录，且遵守
仓库的 `schema/private/` 和 `source/internal/` 禁止访问约束。

## 目录和维护

```text
kernel_corpus/
  config.json                 # tag、selector、backend 状态
  manifest.json               # 快照索引和 hash
  sources/<framework>/<tag>/  # 源码快照
  collect_kernel_sources.py   # 从 git tag 生成快照
  validate_kernel_sources.py  # 校验文件、hash 和禁止内容
```

使用本地 git cache 重抓：

```bash
python3 replay_benchmark/kernel_corpus/collect_kernel_sources.py \
  --config replay_benchmark/kernel_corpus/config.json \
  --output replay_benchmark/kernel_corpus \
  --cache /tmp/mnn-ncnn-kernel-source-cache

python3 replay_benchmark/kernel_corpus/validate_kernel_sources.py \
  --manifest replay_benchmark/kernel_corpus/manifest.json \
  --root replay_benchmark/kernel_corpus
```

新增 tag 时先修改 `config.json`，再运行 collector；提交前运行 validator、单元测试和
`git diff --check`。后续若要测试某个 kernel，应在 replay benchmark 中重新封装成
独立 workload，避免直接把不同版本的完整 backend 混入当前 MNN 构建。
