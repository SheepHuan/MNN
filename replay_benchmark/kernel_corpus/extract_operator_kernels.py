#!/usr/bin/env python3
"""Extract operator metadata from the curated operators/ corpus.

The extractor scans ``operators/<backend>/<op_type>/<framework>/<tag>/``
for ``.cl`` (OpenCL) and ``.comp`` (Vulkan GLSL) files, discovers kernel
entry points and emits a stable-sorted ``operators.json``.

Each operator file is a self-contained, baked compilation unit: all macros
(``FLOAT``, ``OUTPUT_TYPE``, ``sfp``, ``afp``, ...) are already defined
inside the file. The extractor only records facts; it never compiles,
links, or reads ``schema/private/`` / ``source/internal/`` content.
"""

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath


FORBIDDEN_PARTS = ("schema/private", "source/internal")
OPENCL_SUFFIX = ".cl"
VULKAN_SUFFIX = ".comp"

_OPENCL_KERNEL_RE = re.compile(r"__kernel\s+void\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


class ExtractorError(Exception):
    """Raised when the corpus is inconsistent or malformed."""


def _is_forbidden(relative):
    return any(part in relative for part in FORBIDDEN_PARTS)


def _scan_opencl_entry(source_text):
    match = _OPENCL_KERNEL_RE.search(source_text)
    if match is None:
        return None
    return match.group(1)


def _backend_for_suffix(suffix):
    if suffix == OPENCL_SUFFIX:
        return "opencl", "opencl"
    if suffix == VULKAN_SUFFIX:
        return "vulkan", "glsl"
    return None, None


def _extract_one(file_path, corpus_root):
    relative = file_path.relative_to(corpus_root).as_posix()
    if _is_forbidden(relative):
        raise ExtractorError("forbidden path in operators corpus: {}".format(relative))
    suffix = PurePosixPath(relative).suffix.lower()
    backend, language = _backend_for_suffix(suffix)
    if backend is None:
        return None
    variant = file_path.stem
    parts = relative.split("/")
    # expected: operators/<backend>/<op_type>/<framework>/<tag>/<variant>.<suffix>
    if len(parts) < 6:
        raise ExtractorError("unexpected operators path layout: {}".format(relative))
    op_type = parts[2]
    framework = parts[3]
    tag = parts[4]
    try:
        source_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ExtractorError("cannot read {}: {}".format(relative, exc))
    if backend == "opencl":
        entry = _scan_opencl_entry(source_text)
        if entry is None:
            return None
    else:
        if "void main" not in source_text and "void main(" not in source_text:
            return None
        entry = variant
    return {
        "backend": backend,
        "op_type": op_type,
        "framework": framework,
        "tag": tag,
        "variant": variant,
        "entry": entry,
        "language": language,
        "file": relative,
        "execution_spec": "manual",
    }


def extract_operators(corpus_root):
    """Scan ``operators/`` under ``corpus_root`` and return sorted records."""
    corpus_root = Path(corpus_root)
    operators_dir = corpus_root / "operators"
    if not operators_dir.is_dir():
        return []
    records = []
    seen_keys = set()
    for path in sorted(operators_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (OPENCL_SUFFIX, VULKAN_SUFFIX):
            continue
        record = _extract_one(path, corpus_root)
        if record is None:
            continue
        key = (record["backend"], record["op_type"], record["framework"],
               record["tag"], record["variant"])
        if key in seen_keys:
            raise ExtractorError("duplicate operator key: {}".format(key))
        seen_keys.add(key)
        records.append(record)
    records.sort(key=lambda r: (r["backend"], r["op_type"], r["framework"],
                                r["tag"], r["variant"]))
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="Corpus root directory (contains operators/).")
    parser.add_argument("--output", required=True, help="Output operators.json path.")
    parser.add_argument("--backend", help="Optional backend filter (opencl/vulkan).")
    parser.add_argument("--framework", help="Optional framework filter (mnn/ncnn).")
    parser.add_argument("--tag", help="Optional tag filter.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = extract_operators(args.root)
    if args.backend:
        records = [r for r in records if r["backend"] == args.backend]
    if args.framework:
        records = [r for r in records if r["framework"] == args.framework]
    if args.tag:
        records = [r for r in records if r["tag"] == args.tag]
    document = {
        "format": "mnn-kernel-operator-corpus",
        "version": 1,
        "operators": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
