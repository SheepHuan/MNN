#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEYS = [
    "device",
    "device_display",
    "model",
    "backend",
    "frequency_note",
    "context_tokens",
    "mode",
    "budget",
]

FIELDNAMES = KEYS + ["prefill_latency_s", "prefill_tps"]
FORMAL_REQUIRED_DEVICES = ("jetson", "orangepi")

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


def summary_devices(path: Path) -> set[str]:
    return {str(row.get("device", "")).strip() for row in parse_rows(path) if str(row.get("device", "")).strip()}


def load_run_config(summary_path: Path) -> dict[str, object]:
    run_config = summary_path.parent / "run_config.json"
    if not run_config.is_file():
        return {}
    try:
        with run_config.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def validate_formal_inputs(summary_paths: list[Path], *, allow_nonformal: bool) -> None:
    if allow_nonformal:
        return
    covered: set[str] = set()
    run_config_flags: list[str] = []
    for summary_path in summary_paths:
        covered.update(summary_devices(summary_path))
        config = load_run_config(summary_path)
        if config:
            device_roles = config.get("device_roles")
            if isinstance(device_roles, dict):
                extra = sorted(
                    str(device).strip()
                    for device, role in device_roles.items()
                    if str(role).strip() == "extra-profile"
                )
                if extra:
                    run_config_flags.append(
                        f"{summary_path}: extra-profile devices in run_config={','.join(extra)}"
                    )
            if bool(config.get("allow_subset_devices")):
                run_config_flags.append(f"{summary_path}: run_config allow_subset_devices=true")
            if bool(config.get("allow_extra_device")):
                run_config_flags.append(f"{summary_path}: run_config allow_extra_device=true")
    missing = [device for device in FORMAL_REQUIRED_DEVICES if device not in covered]
    if not missing:
        return
    detail = f"summary devices={','.join(sorted(covered)) or '(none)'}"
    if run_config_flags:
        detail += "; " + "; ".join(run_config_flags)
    required = ",".join(FORMAL_REQUIRED_DEVICES)
    raise SystemExit(
        "refusing to merge non-formal prefill summaries: "
        f"new inputs must cover {required}, missing={','.join(missing)}; "
        f"{detail}. Use --allow-nonformal for targeted single-device or rhino-only merges."
    )


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
    parser.add_argument(
        "--allow-nonformal",
        action="store_true",
        help="Allow merging targeted single-device or extra-profile summaries that do not cover jetson,orangepi together.",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_csv)
    summary_paths = [Path(item) for item in args.summary_csv]
    validate_formal_inputs(summary_paths, allow_nonformal=bool(args.allow_nonformal))
    merged: dict[tuple[str, ...], dict[str, str]] = {}
    for row in parse_rows(benchmark_path):
        merged[key_for(row)] = row
    for summary in summary_paths:
        for row in parse_rows(summary):
            merged[key_for(row)] = row

    rows = sorted(merged.values(), key=sort_key)
    with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
