# GPU Kernel Source Corpus Design

## Goal

Collect the GPU kernel implementation source used by multiple tagged MNN and
ncnn releases under `replay_benchmark/kernel_corpus/`. This phase is source
archival only: it does not compile, link, execute, or change any framework
implementation.

## Repository layout

```text
replay_benchmark/kernel_corpus/
├── manifest.json
├── mnn/2.8.4/...
└── ncnn/20260113/...
```

Each version directory preserves the original repository-relative paths for
the selected GPU implementation files. Generated binaries, build directories,
models, caches, and PMU records are excluded.

## Source selection

For MNN, archive the OpenCL and Vulkan backend implementation files, embedded
OpenCL kernel sources, Vulkan shader sources, and directly referenced public
GPU headers needed to understand those implementations. Do not archive
`schema/private/` or `source/internal/`.

For ncnn, archive the Vulkan layer/command/runtime implementation and shader
source files, plus directly referenced public GPU headers. Mainline ncnn does
not provide a standard OpenCL backend; this is recorded as an explicit
`unavailable` backend entry rather than creating an empty or synthetic source
tree.

## Manifest contract

`manifest.json` contains one record per version:

```json
{
  "framework": "mnn",
  "tag": "2.8.4",
  "commit": "b0565d3402167fc7f85f8f8f5d4a109b6293b5f8",
  "source_url": "https://github.com/alibaba/MNN.git",
  "backends": {"opencl": "archived", "vulkan": "archived"},
  "files": [],
  "licenses": ["LICENSE"]
}
```

The manifest records the exact tag/commit, source URL, backend availability,
relative source paths, file hashes, and license files. It is generated after
copying the source and must be deterministic for the same source commit.

## Version policy

Use the tags available from the upstream repositories and keep one snapshot
per tag. Store source commits rather than relying on a moving branch. Do not
modify the original framework history or vendor a complete build tree.

## Acceptance criteria

- MNN and ncnn GPU source snapshots exist under separate tag directories.
- Every snapshot has a manifest record with tag, commit, backend status, file
  list, hashes, and license metadata.
- No binaries, models, build products, or PMU outputs are included.
- The archive preserves source-relative paths and is independent of the host
  or target device.
- `git diff --check` passes and the corpus validation script can detect missing
  manifest entries or hash mismatches.

The later PMU runner integration and cross-version performance analysis are
separate phases and are intentionally out of scope here.
