# GPU Kernel Source Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Collect GPU implementation source from selected MNN and ncnn tags under `replay_benchmark/kernel_corpus/`, with deterministic manifests, hashes, license metadata, and no build or PMU integration.

**Architecture:** A Python collector resolves framework tags in a temporary git cache, copies only configured GPU source paths into `sources/<framework>/<tag>/`, and writes one deterministic manifest. A validator checks copied files, hashes, licenses, backend status, and forbidden-path exclusions. ncnn Vulkan is archived; upstream ncnn OpenCL is recorded as unavailable.

**Tech Stack:** Python 3 standard library, git, JSON, SHA-256, and unittest.

---

### Task 1: Configuration and path-policy tests

**Files:**
- Create: `replay_benchmark/kernel_corpus/config.json`
- Create: `replay_benchmark/kernel_corpus/tests/test_kernel_corpus.py`

- [ ] Add tests requiring MNN tags `2.4.0`, `2.6.0`, `2.8.0`, `2.8.4`, ncnn tags `20240102`, `20240820`, `20250916`, `20260113`, explicit OpenCL/Vulkan backend states, and no `schema/private` or `source/internal` selector.
- [ ] Run `python3 -m unittest replay_benchmark.kernel_corpus.tests.test_kernel_corpus` and observe the expected missing-config failure.
- [ ] Add the configuration with repository URLs, exact tags, GPU selectors, license names, and ncnn OpenCL `unavailable` status.
- [ ] Re-run the focused unittest and require PASS.

### Task 2: Source collector

**Files:**
- Create: `replay_benchmark/kernel_corpus/collect_kernel_sources.py`
- Modify: `replay_benchmark/kernel_corpus/tests/test_kernel_corpus.py`

- [ ] Add tests using a temporary source tree to require selected GPU files and licenses to copy, forbidden paths to be rejected, and manifest file hashes to be 64-character SHA-256 values.
- [ ] Run the focused unittest and observe the expected collector-missing failure.
- [ ] Implement `select_source_files`, `copy_snapshot`, `build_manifest_record`, and `collect`. Resolve tags with a temporary/cache clone using `git clone --filter=blob:none --no-checkout` and `git fetch --tags`; copy sorted source-relative paths; hash copied files; exclude binaries, models, build directories, caches, PMU records, `schema/private`, and `source/internal`.
- [ ] Re-run the focused unittest and require PASS.

### Task 3: Manifest validator

**Files:**
- Create: `replay_benchmark/kernel_corpus/validate_kernel_sources.py`
- Modify: `replay_benchmark/kernel_corpus/tests/test_kernel_corpus.py`

- [ ] Add tests for valid manifests and rejection of missing files, changed hashes, missing licenses, forbidden paths, and backend entries without an explicit status.
- [ ] Run the focused unittest and observe the expected validator-missing failure.
- [ ] Implement `validate_manifest(manifest_path, corpus_root)` with nonzero exit status and readable errors; validation must not mutate the corpus.
- [ ] Run the focused unittest and require PASS.

### Task 4: Collect the source snapshots

**Files:**
- Create: `replay_benchmark/kernel_corpus/manifest.json`
- Create: `replay_benchmark/kernel_corpus/sources/`
- Create: `replay_benchmark/kernel_corpus/licenses/`

- [ ] Run:

```bash
python3 replay_benchmark/kernel_corpus/collect_kernel_sources.py \
  --config replay_benchmark/kernel_corpus/config.json \
  --output replay_benchmark/kernel_corpus \
  --cache /tmp/mnn-ncnn-kernel-source-cache
```

- [ ] Validate with:

```bash
python3 replay_benchmark/kernel_corpus/validate_kernel_sources.py \
  --manifest replay_benchmark/kernel_corpus/manifest.json \
  --root replay_benchmark/kernel_corpus
```

- [ ] Confirm eight framework/tag records, explicit ncnn OpenCL unavailability, no `.so`, `.a`, `.mnn`, `.bin`, `schema/private`, or `source/internal` files.

### Task 5: Documentation and final verification

**Files:**
- Create: `replay_benchmark/kernel_corpus/README.md`
- Modify: `replay_benchmark/README.md`

- [ ] Document tag list, backend availability, source-only scope, license handling, collector command, validator command, and PMU integration as a later phase.
- [ ] Run the focused unittest, the validator, and `git diff --check`; require all commands to exit 0.
