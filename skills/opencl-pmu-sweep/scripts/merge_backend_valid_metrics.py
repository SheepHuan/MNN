#!/usr/bin/env python3
"""Merge valid OpenCL/Vulkan PMU metrics for one device."""

import argparse
import csv
from pathlib import Path


INPUT_COLUMNS = ["pmu_metric_name", "valid", "valid_cases"]
OUTPUT_COLUMNS = [
    "pmu_metric_name", "category", "opencl_valid", "opencl_valid_cases",
    "vulkan_valid", "vulkan_valid_cases", "valid_backends",
]

CACHE_MARKERS = (
    "cache", "l1", "l2", "uche", "cche", "tph", "icl1", "flag_cache", "texture",
)
MEMORY_MARKERS = (
    "memory", "gmem", "vbif", "external", "read", "write", "load", "store",
    "bandwidth", "beat", "transaction", "request", "latency", "outstanding",
    "evict", "snoop", "ubwc", "decomp",
)
COMPUTE_MARKERS = (
    "compute", "arithmetic", "instruction", "alu", "fma", "sfu", "cvt", "efu",
    "active", "busy", "working", "issue", "wave", "gpr", "task", "starve",
)


def classify_metric(metric_name):
    """Classify by name; cache wins over memory, then compute, then other."""
    name = metric_name.lower()
    if any(marker in name for marker in CACHE_MARKERS):
        return "cache"
    if any(marker in name for marker in MEMORY_MARKERS):
        return "memory"
    if any(marker in name for marker in COMPUTE_MARKERS):
        return "compute"
    return "other"


def read_valid_summary(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != INPUT_COLUMNS:
            raise ValueError("{} must have columns {}".format(path, ",".join(INPUT_COLUMNS)))
        for row in reader:
            if row["valid"].strip().lower() != "true":
                continue
            name = row["pmu_metric_name"].strip()
            if name:
                result[name] = row["valid_cases"].strip()
    return result


def merge_metric_summaries(opencl_path, vulkan_path, output_path):
    opencl = read_valid_summary(opencl_path)
    vulkan = read_valid_summary(vulkan_path)
    names = list(opencl)
    names.extend(name for name in vulkan if name not in opencl)
    rows = []
    for name in names:
        opencl_valid = name in opencl
        vulkan_valid = name in vulkan
        rows.append({
            "pmu_metric_name": name,
            "category": classify_metric(name),
            "opencl_valid": "true" if opencl_valid else "false",
            "opencl_valid_cases": opencl.get(name, ""),
            "vulkan_valid": "true" if vulkan_valid else "false",
            "vulkan_valid_cases": vulkan.get(name, ""),
            "valid_backends": ";".join(
                backend for backend, valid in (("opencl", opencl_valid), ("vulkan", vulkan_valid)) if valid
            ),
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencl-metrics", required=True)
    parser.add_argument("--vulkan-metrics", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    merge_metric_summaries(args.opencl_metrics, args.vulkan_metrics, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
