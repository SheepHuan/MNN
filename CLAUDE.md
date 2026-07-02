# MNN Project Instructions

Claude Code / cc agents working in this repository must read and follow
`AGENTS.md` first.

`AGENTS.md` is the source of truth for:

- MNN repository scope and Codex skill locations (`.codex/skills/` vs upstream `skills/`)
- model cache, build artifact, and output path rules
- required conda environment (`kvshare-edge`) for Python commands
- when to read repository skills under `.codex/skills/`
- MNN build, export, benchmark, and PIC server conventions
- LLM output comparison and PIC server protocol conventions
- MNN source modification constraints

Do not duplicate or reinterpret those rules here. After reading `AGENTS.md`,
open only the task-relevant skill or document that it names.

Common examples:

- MNN build / CUDA / Jetson artifact work: read `.codex/skills/mnn-build-artifacts/SKILL.md`
- New or extended MNN op: read `.codex/skills/mnn-add-new-op/SKILL.md`
- CUDA single-op benchmark: read `.codex/skills/mnn-ops-bench/SKILL.md`
- CPU/CUDA op optimization: read `.codex/skills/mnn-opt-ops/SKILL.md`
- LLM export (MNN / PIC / PrefixLLM): read `.codex/skills/mnn-llm-export/SKILL.md`
- LLM benchmark / demo / PIC comparison: read `.codex/skills/mnn-llm-bench/SKILL.md`
- PIC server cross-compile / deploy / benchmark: read `.codex/skills/mnn-pic-benchmark/SKILL.md`
- PIC/PagedAttention OpenCL/CUDA/Adreno sparse prefill optimization, score-layer boundary, kernel hot spots, cross-device regression / optimization logs: read `.codex/skills/mnn-pic-optimize/SKILL.md`
- Jetson/OrangePi/AidLux storage (NVMe/SSD/UFS) and DRAM read bandwidth baselines (`fio`, `lmbench bw_mem rd`, STREAM): read `.codex/skills/mnn-device-io-bench/SKILL.md`
- New LLM model adaptation in `transformers/pic_llm`: read `.codex/skills/mnn-support-new-llm/SKILL.md`
- Drafting an MNN commit message from the current working-tree diff: read `.codex/skills/mnn-git-commit-message/SKILL.md`

If any instruction in another document seems to conflict with `AGENTS.md`, treat
`AGENTS.md` as authoritative for this repository and ask the user before taking
an action that could violate it.
