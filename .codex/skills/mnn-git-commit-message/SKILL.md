---
name: mnn-git-commit-message
description: Draft accurate Git commit messages for the current MNN workspace by inspecting staged and unstaged diffs. Use when the user asks for a commit message, commit summary, `git commit -m` text, staged-change summary, or help preparing an MNN repository commit from current working tree changes.
---

# MNN Git Commit Message

## Overview

Generate a commit message from the actual repository state. Prefer a concise subject plus a body that captures the main behavioral changes, touched subsystems, and validation status.

## Workflow

1. From the MNN repository root, read `AGENTS.md` if it has not already been read in the turn.
2. Run `git status --short` before inspecting diffs. Do not remove, stage, unstage, or revert user changes unless explicitly asked.
3. Inspect staged changes first:
   - `git diff --cached --stat`
   - `git diff --cached --name-status`
   - targeted `git diff --cached -- <path>` for important files
4. Inspect unstaged changes when the user asks for the current workspace rather than staged-only:
   - `git diff --stat`
   - `git diff --name-status`
   - targeted `git diff -- <path>` for important files
5. If staged and unstaged changes differ, say which scope the message covers. For a real commit message, prefer staged changes when anything is staged; include unstaged work only when the user explicitly asks for all workspace changes.
6. If the workspace contains unrelated changes, either propose separate commit messages or clearly mark the message as broad/mixed.
7. Do not claim tests, benchmarks, deployment, or validation were run unless there is direct evidence in the conversation or tool output.

## Message Shape

Use this format by default:

```text
<area>: <imperative summary>

- <main change>
- <supporting change>
- <validation or benchmark note, only if evidenced>
```

Guidelines:

- Keep the subject under roughly 72 characters when practical.
- Use an MNN subsystem as the area, such as `opencl`, `cuda`, `pic`, `llm`, `build`, `bench`, `docs`, or `skills`.
- Use imperative voice in the subject: `add`, `fix`, `tune`, `document`, `align`, `enable`.
- Mention benchmark data files only when they are part of the intended commit.
- Avoid vague subjects such as `update files`, `fix bug`, or `misc changes`.
- For mixed MNN PIC/PagedAttention work, make the subject reflect the dominant behavior change, not every file touched.

## Output Rules

- Provide a ready-to-use message in a fenced `text` block.
- Add a short note if the message covers staged changes only, all workspace changes, or a proposed split.
- If the user asks to commit, run the commit only after confirming the exact scope if staged and unstaged changes do not match.
