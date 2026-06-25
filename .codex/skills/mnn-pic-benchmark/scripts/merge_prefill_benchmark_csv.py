#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEYS = [
    "device",
    "device_display",
    "model",
    "backend",
    "frequency_profile",
    "context_tokens",
    "mode",
    "budget",
]

FIELDNAMES = KEYS + ["prefill_latency_s", "prefill_tps"]

DEVICE_ORDER = {"jetson": 0, "orangepi": 1, "rhino": 2}
MODEL_ORDER = {
    "Llama3.2 1B": 0,
    "MiniCPM5-1B": 1,
    "Llama3.2 3B": 2,
    "Qwen3-8B": 3,
}
MODE_ORDER = {
    "normal-full-recompute": 0,
    "pic-full-recompute": 1,
    "full-reuse": 2,
    "cacheblend": 3,
    "epic": 4,
}


def parse_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            parsed = {key: row.get(key, "") for key in FIELDNAMES}
            if not parsed["prefill_latency_s"] or not parsed["prefill_tps"]:
                continue
            rows.append(parsed)
        return rows


def key_for(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(key, "") for key in KEYS)


def sort_key(row: dict[str, str]) -> tuple[object, ...]:
    device = row.get("device", "")
    model = row.get("model", "")
    context = int(row.get("context_tokens", "0") or 0)
    mode = row.get("mode", "")
    budget_raw = row.get("budget", "")
    try:
        budget_num = float(budget_raw)
    except ValueError:
        budget_num = -1.0 if budget_raw == "full" else 0.0
    return (
        DEVICE_ORDER.get(device, 99),
        MODEL_ORDER.get(model, 99),
        context,
        MODE_ORDER.get(mode, 99),
        budget_num,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_csv")
    parser.add_argument("summary_csv", nargs="+")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_csv)
    merged: dict[tuple[str, ...], dict[str, str]] = {}
    for row in parse_rows(benchmark_path):
        merged[key_for(row)] = row
    for summary in args.summary_csv:
        for row in parse_rows(Path(summary)):
            merged[key_for(row)] = row

    rows = sorted(merged.values(), key=sort_key)
    with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
