#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "analysis_status",
    "power_uuid",
    "device",
    "device_display",
    "backend",
    "frequency_note",
    "model_key",
    "model",
    "context_tokens",
    "algorithm",
    "case_label",
    "budget",
    "pic_recompute_ratio",
    "pic_recompute_score_layer_idx",
    "power_rep_total",
    "selected_rep_count",
    "kept_rep_indices",
    "drop_first",
    "drop_last",
    "detected_rep_count",
    "raw_active_span_count",
    "segment_grouping",
    "capture_duration_s",
    "base_power_w",
    "base_windows_s",
    "base_method",
    "work_power_avg_w",
    "work_minus_base_avg_w",
    "initial_base_power_w",
    "active_power_p95_w",
    "threshold_power_w",
    "threshold_ratio",
    "selected_duration_s",
    "duration_per_inference_s",
    "selected_total_energy_j",
    "selected_baseline_energy_j",
    "selected_incremental_avg_power_w",
    "selected_incremental_energy_j",
    "energy_per_inference_j",
    "energy_per_inference_mj",
    "mj_per_token",
    "prefill_latency_s",
    "kept_prefill_latency_s",
    "kept_prefill_tps",
    "detected_rep_windows_s",
    "selected_rep_windows_s",
    "power_csv",
    "power_txt",
    "power_png",
    "power_run_id",
    "manifest",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-summarize PIC power captures by detecting repeated inference reps, "
            "dropping the first and last rep by default, subtracting base power, "
            "and reporting incremental energy plus mJ/token. Run this once at "
            "the end of a power batch over power directories or manifest.tsv files."
        ),
        epilog=(
            "Example: summarize_pic_power_reps.py "
            ".codex/skills/mnn-pic-benchmark/power/orangepi/2026-06-30 "
            ".codex/skills/mnn-pic-benchmark/power/rhino/2026-06-30 "
            "-o .codex/skills/mnn-pic-benchmark/power/middle_rep_energy_20260630.csv "
            "--plot-output .codex/skills/mnn-pic-benchmark/power/middle_rep_energy_20260630.png"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Power CSV/TXT files, manifest.tsv files, or power directories.",
    )
    parser.add_argument("-o", "--output", help="Write summary to this file. Defaults to stdout.")
    parser.add_argument(
        "--plot-output",
        help=(
            "Write one summary PNG with one subplot per case. The plot shows full power, "
            "dropped rep windows, and kept middle-rep windows."
        ),
    )
    parser.add_argument(
        "--plot-max-points-per-case",
        type=int,
        default=2000,
        help="Maximum plotted points per case after visualization-only downsampling. Defaults to 2000.",
    )
    parser.add_argument(
        "--plot-width",
        type=float,
        default=13.0,
        help="Summary PNG width in inches. Defaults to 13.",
    )
    parser.add_argument(
        "--plot-row-height",
        type=float,
        default=1.8,
        help="Summary PNG height per case in inches. Defaults to 1.8.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=140,
        help="Summary PNG DPI. Defaults to 140.",
    )
    parser.add_argument(
        "--delimiter",
        choices=["csv", "tsv"],
        default="csv",
        help="Output delimiter. Defaults to csv.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="When a directory does not contain manifest.tsv directly, do not search it recursively.",
    )
    parser.add_argument(
        "--drop-first",
        type=int,
        default=1,
        help="Number of leading reps to remove before energy averaging. Defaults to 1.",
    )
    parser.add_argument(
        "--drop-last",
        type=int,
        default=1,
        help="Number of trailing reps to remove before energy averaging. Defaults to 1.",
    )
    parser.add_argument(
        "--base-percentile",
        type=float,
        default=5.0,
        help="Initial base-power percentile used before rep detection. Defaults to 5.",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.25,
        help=(
            "Active-span threshold as base + ratio * (p95 - base). "
            "Defaults to 0.25."
        ),
    )
    parser.add_argument(
        "--min-dynamic-power-w",
        type=float,
        default=0.05,
        help="Minimum p95-base delta required to detect active spans. Defaults to 0.05 W.",
    )
    parser.add_argument(
        "--bin-sec",
        type=float,
        default=0.05,
        help="Power bin width before detection. Defaults to 0.05 s.",
    )
    parser.add_argument(
        "--smooth-sec",
        type=float,
        default=0.25,
        help="Moving-average smoothing window for active-span detection. Defaults to 0.25 s.",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=1.0,
        help="Merge active spans separated by gaps no larger than this. Defaults to 1 s.",
    )
    parser.add_argument(
        "--min-active-sec",
        type=float,
        default=0.25,
        help="Drop active spans shorter than this. Defaults to 0.25 s.",
    )
    parser.add_argument(
        "--window-pad-sec",
        type=float,
        default=0.0,
        help="Pad detected rep windows before integration. Defaults to 0.",
    )
    parser.add_argument(
        "--min-window-latency-ratio",
        type=float,
        default=0.25,
        help=(
            "When per-rep prefill latency is available, mark a kept window invalid if "
            "its detected duration is shorter than this fraction of that latency. "
            "Defaults to 0.25."
        ),
    )
    parser.add_argument(
        "--token-field",
        default="context_tokens",
        help="Metadata field used as the mJ/token denominator. Defaults to context_tokens.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_column(fieldnames: list[str] | None, candidates: list[str]) -> str | None:
    normalized = {normalize_name(name): name for name in (fieldnames or [])}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt_float(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    q = min(max(q, 0.0), 1.0)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            item = dict(row)
            item["manifest"] = str(path)
            rows.append(item)
    return rows


def parse_power_txt(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {"power_txt": str(path)}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return meta
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            meta[key] = value.strip()
    return meta


def has_glob_magic(path: str) -> bool:
    return any(char in path for char in "*?[")


def case_from_csv(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "power_csv": str(path),
        "power_uuid": path.stem,
    }
    txt_path = path.with_suffix(".txt")
    if txt_path.exists():
        row["power_txt"] = str(txt_path)
    return row


def case_from_txt(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = parse_power_txt(path)
    row.setdefault("power_uuid", path.stem)
    csv_path = path.with_suffix(".csv")
    if "power_csv" not in row and csv_path.exists():
        row["power_csv"] = str(csv_path)
    return row


def expand_directory(path: Path, recursive: bool) -> list[dict[str, Any]]:
    manifest = path / "manifest.tsv"
    if manifest.exists():
        return read_manifest(manifest)
    if recursive:
        manifests = sorted(path.rglob("manifest.tsv"))
        if manifests:
            rows: list[dict[str, Any]] = []
            for item in manifests:
                rows.extend(read_manifest(item))
            return rows
    return [case_from_csv(item) for item in sorted(path.glob("*.csv"))]


def expand_inputs(paths: list[str], recursive: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in paths:
        expanded = glob.glob(item) if has_glob_magic(item) else [item]
        for expanded_item in expanded:
            path = Path(expanded_item)
            if path.is_dir():
                rows.extend(expand_directory(path, recursive))
            elif path.name == "manifest.tsv":
                rows.extend(read_manifest(path))
            elif path.suffix.lower() == ".csv":
                rows.append(case_from_csv(path))
            elif path.suffix.lower() == ".txt":
                rows.append(case_from_txt(path))
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("power_csv") or row.get("power_txt") or row.get("power_uuid") or "")
        if not key:
            continue
        key = str(Path(key).resolve()) if not key.startswith("http") else key
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def load_case_metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = dict(row)
    txt_path_text = str(meta.get("power_txt") or "").strip()
    csv_path_text = str(meta.get("power_csv") or "").strip()
    if not txt_path_text and csv_path_text:
        candidate = Path(csv_path_text).with_suffix(".txt")
        if candidate.exists():
            txt_path_text = str(candidate)
    if txt_path_text:
        txt_meta = parse_power_txt(Path(txt_path_text))
        for key, value in txt_meta.items():
            if value != "":
                meta[key] = value
    if not meta.get("power_csv") and meta.get("power_txt"):
        candidate = Path(str(meta["power_txt"])).with_suffix(".csv")
        if candidate.exists():
            meta["power_csv"] = str(candidate)
    if not meta.get("power_uuid"):
        source = str(meta.get("power_csv") or meta.get("power_txt") or "")
        if source:
            meta["power_uuid"] = Path(source).stem
    return meta


def parse_repetition_results(meta: dict[str, Any]) -> list[dict[str, Any]]:
    raw = meta.get("power_repetition_results", "")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if not str(raw).strip():
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def read_power_samples(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        time_col = find_column(
            reader.fieldnames,
            ["timestamp [us]", "timestamp_us", "time_us", "timestamp", "time"],
        )
        power_col = find_column(reader.fieldnames, ["power [W]", "power_w", "power"])
        if not time_col or not power_col:
            raise ValueError(f"expected timestamp and power columns, got {reader.fieldnames}")

        raw: list[tuple[float, float]] = []
        for row in reader:
            timestamp = to_float(row.get(time_col))
            power_w = to_float(row.get(power_col))
            if math.isfinite(timestamp) and math.isfinite(power_w):
                raw.append((timestamp, power_w))
    if not raw:
        raise ValueError("no numeric power samples")

    time_key = normalize_name(time_col)
    if "us" in time_key:
        scale = 1.0 / 1_000_000.0
    elif "ms" in time_key:
        scale = 1.0 / 1_000.0
    else:
        max_timestamp = max(value for value, _ in raw)
        scale = 1.0 / 1_000_000.0 if max_timestamp > 100_000 else 1.0

    origin = raw[0][0]
    samples = [((timestamp - origin) * scale, power_w) for timestamp, power_w in raw]
    samples.sort(key=lambda item: item[0])
    return samples


def downsample(samples: list[tuple[float, float]], bin_sec: float) -> list[tuple[float, float]]:
    if not samples:
        return []
    bin_sec = max(bin_sec, 0.001)
    sums: list[float] = []
    counts: list[int] = []
    for time_sec, power_w in samples:
        if time_sec < 0:
            continue
        index = int(time_sec / bin_sec)
        while len(sums) <= index:
            sums.append(0.0)
            counts.append(0)
        sums[index] += power_w
        counts[index] += 1
    bins: list[tuple[float, float]] = []
    for index, count in enumerate(counts):
        if count:
            bins.append(((index + 0.5) * bin_sec, sums[index] / count))
    return bins


def smooth_bins(bins: list[tuple[float, float]], bin_sec: float, smooth_sec: float) -> list[tuple[float, float]]:
    if not bins or smooth_sec <= bin_sec:
        return bins
    width = max(int(round(smooth_sec / max(bin_sec, 0.001))), 1)
    radius = max(width // 2, 0)
    powers = [power for _, power in bins]
    prefix = [0.0]
    for power in powers:
        prefix.append(prefix[-1] + power)
    smoothed: list[tuple[float, float]] = []
    for index, (time_sec, _) in enumerate(bins):
        lo = max(index - radius, 0)
        hi = min(index + radius + 1, len(bins))
        avg = (prefix[hi] - prefix[lo]) / (hi - lo)
        smoothed.append((time_sec, avg))
    return smoothed


def merge_segments(
    segments: list[tuple[float, float]],
    *,
    merge_gap_sec: float,
    min_active_sec: float,
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in segments:
        if end <= start:
            continue
        if not merged or start - merged[-1][1] > merge_gap_sec:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [
        (start, end)
        for start, end in (tuple(item) for item in merged)
        if end - start >= min_active_sec
    ]


def collapse_to_rep_count(
    segments: list[tuple[float, float]],
    rep_total: int,
) -> tuple[list[tuple[float, float]], str]:
    if rep_total <= 0 or len(segments) <= rep_total:
        return segments, "direct"
    gaps = [
        (segments[index + 1][0] - segments[index][1], index)
        for index in range(len(segments) - 1)
    ]
    separators = {
        index
        for _, index in sorted(gaps, key=lambda item: item[0], reverse=True)[: max(rep_total - 1, 0)]
    }
    grouped: list[tuple[float, float]] = []
    group_start = 0
    for index in range(len(segments) - 1):
        if index in separators:
            grouped.append((segments[group_start][0], segments[index][1]))
            group_start = index + 1
    grouped.append((segments[group_start][0], segments[-1][1]))
    return grouped, "largest-gap-collapse"


def pad_windows(
    windows: list[tuple[float, float]],
    *,
    pad_sec: float,
    capture_end_sec: float,
) -> list[tuple[float, float]]:
    if pad_sec <= 0:
        return windows
    return [
        (max(0.0, start - pad_sec), min(capture_end_sec, end + pad_sec))
        for start, end in windows
    ]


def detect_rep_windows(
    samples: list[tuple[float, float]],
    rep_total: int,
    args: argparse.Namespace,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    positive = [power for _, power in samples if power > 0.0]
    if not positive:
        return [], {"status": "no_positive_power_samples"}

    base_q = min(max(args.base_percentile / 100.0, 0.0), 1.0)
    initial_base = quantile(positive, base_q)
    active_p95 = quantile(positive, 0.95)
    dynamic_power = active_p95 - initial_base
    if dynamic_power < args.min_dynamic_power_w:
        return [], {
            "status": "no_active_power_span",
            "initial_base_power_w": initial_base,
            "active_power_p95_w": active_p95,
            "threshold_power_w": math.nan,
            "raw_active_span_count": 0,
            "segment_grouping": "none",
        }

    threshold = initial_base + args.threshold_ratio * dynamic_power
    bins = smooth_bins(downsample(samples, args.bin_sec), args.bin_sec, args.smooth_sec)

    raw_segments: list[tuple[float, float]] = []
    start: float | None = None
    end: float | None = None
    half_bin = max(args.bin_sec, 0.001) * 0.5
    for time_sec, power_w in bins:
        if power_w >= threshold:
            if start is None:
                start = max(0.0, time_sec - half_bin)
            end = time_sec + half_bin
        elif start is not None and end is not None:
            raw_segments.append((start, end))
            start = None
            end = None
    if start is not None and end is not None:
        raw_segments.append((start, end))

    merged = merge_segments(
        raw_segments,
        merge_gap_sec=max(args.merge_gap_sec, 0.0),
        min_active_sec=max(args.min_active_sec, 0.0),
    )
    grouped, grouping = collapse_to_rep_count(merged, rep_total)
    grouped = pad_windows(
        grouped,
        pad_sec=max(args.window_pad_sec, 0.0),
        capture_end_sec=samples[-1][0],
    )
    return grouped, {
        "status": "ok",
        "initial_base_power_w": initial_base,
        "active_power_p95_w": active_p95,
        "threshold_power_w": threshold,
        "raw_active_span_count": len(merged),
        "segment_grouping": grouping,
    }


def outside_power_values(
    samples: list[tuple[float, float]],
    windows: list[tuple[float, float]],
) -> list[float]:
    if not windows:
        return [power for _, power in samples if power > 0.0]
    windows = sorted(windows)
    out: list[float] = []
    window_index = 0
    for time_sec, power_w in samples:
        if power_w <= 0.0:
            continue
        while window_index < len(windows) and time_sec > windows[window_index][1]:
            window_index += 1
        inside = (
            window_index < len(windows)
            and windows[window_index][0] <= time_sec <= windows[window_index][1]
        )
        if not inside:
            out.append(power_w)
    return out


def identify_base_power(
    samples: list[tuple[float, float]],
    rep_windows: list[tuple[float, float]],
    initial_base: float,
) -> tuple[float, str]:
    inactive = outside_power_values(samples, rep_windows)
    if len(inactive) >= 10:
        return statistics.median(inactive), "inactive-window-median"
    positive = [power for _, power in samples if power > 0.0]
    if positive:
        return initial_base, "all-sample-low-percentile"
    return math.nan, "none"


def interpolate_power(times: list[float], powers: list[float], time_sec: float) -> float:
    if not times:
        return math.nan
    if time_sec <= times[0]:
        return powers[0]
    if time_sec >= times[-1]:
        return powers[-1]
    index = bisect.bisect_left(times, time_sec)
    if index < len(times) and times[index] == time_sec:
        return powers[index]
    t0 = times[index - 1]
    t1 = times[index]
    p0 = powers[index - 1]
    p1 = powers[index]
    if t1 <= t0:
        return p0
    ratio = (time_sec - t0) / (t1 - t0)
    return p0 + ratio * (p1 - p0)


def integrate_windows(
    samples: list[tuple[float, float]],
    windows: list[tuple[float, float]],
    base_power_w: float,
) -> tuple[float, float, float]:
    if not samples or not windows:
        return 0.0, 0.0, 0.0
    times = [time_sec for time_sec, _ in samples]
    powers = [power_w for _, power_w in samples]
    total_duration = 0.0
    total_energy = 0.0
    incremental_energy = 0.0

    for start, end in windows:
        start = max(start, times[0])
        end = min(end, times[-1])
        if end <= start:
            continue
        left = bisect.bisect_right(times, start)
        right = bisect.bisect_left(times, end)
        points: list[tuple[float, float]] = [(start, interpolate_power(times, powers, start))]
        points.extend(samples[left:right])
        points.append((end, interpolate_power(times, powers, end)))
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            total_energy += dt * (p0 + p1) * 0.5
            incremental_energy += dt * (
                max(p0 - base_power_w, 0.0) + max(p1 - base_power_w, 0.0)
            ) * 0.5
        total_duration += end - start
    return total_duration, total_energy, incremental_energy


def json_windows(windows: list[tuple[float, float]]) -> str:
    return json.dumps(
        [
            {
                "rep_index": index + 1,
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "duration_s": round(end - start, 6),
            }
            for index, (start, end) in enumerate(windows)
        ],
        separators=(",", ":"),
    )


def json_selected_windows(
    rep_windows: list[tuple[float, float]],
    selected_indices: list[int],
) -> str:
    selected = []
    for rep_index in selected_indices:
        if 1 <= rep_index <= len(rep_windows):
            start, end = rep_windows[rep_index - 1]
            selected.append(
                {
                    "rep_index": rep_index,
                    "start_s": round(start, 6),
                    "end_s": round(end, 6),
                    "duration_s": round(end - start, 6),
                }
            )
    return json.dumps(selected, separators=(",", ":"))


def json_base_windows(
    rep_windows: list[tuple[float, float]],
    capture_duration_s: float,
) -> str:
    if capture_duration_s <= 0:
        return "[]"
    base_windows: list[dict[str, float]] = []
    previous_end = 0.0
    for start, end in sorted(rep_windows):
        if start > previous_end:
            base_windows.append(
                {
                    "start_s": round(previous_end, 6),
                    "end_s": round(start, 6),
                    "duration_s": round(start - previous_end, 6),
                }
            )
        previous_end = max(previous_end, end)
    if capture_duration_s > previous_end:
        base_windows.append(
            {
                "start_s": round(previous_end, 6),
                "end_s": round(capture_duration_s, 6),
                "duration_s": round(capture_duration_s - previous_end, 6),
            }
        )
    return json.dumps(base_windows, separators=(",", ":"))


def selected_rep_indices(rep_total: int, drop_first: int, drop_last: int) -> list[int]:
    first = max(drop_first, 0) + 1
    last = rep_total - max(drop_last, 0)
    if last < first:
        return []
    return list(range(first, last + 1))


def summarize_repetition_results(
    repetition_results: list[dict[str, Any]],
    selected_indices: list[int],
) -> tuple[float, float]:
    if not repetition_results or not selected_indices:
        return math.nan, math.nan
    selected = []
    selected_set = set(selected_indices)
    for item in repetition_results:
        rep_index = to_int(item.get("rep_index"), 0)
        if rep_index in selected_set:
            selected.append(item)
    latencies = [to_float(item.get("prefill_latency_s")) for item in selected]
    tps_values = [to_float(item.get("prefill_tps")) for item in selected]
    latencies = [item for item in latencies if math.isfinite(item)]
    tps_values = [item for item in tps_values if math.isfinite(item)]
    latency = sum(latencies) / len(latencies) if latencies else math.nan
    tps = sum(tps_values) / len(tps_values) if tps_values else math.nan
    return latency, tps


def base_output(meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "analysis_status": "",
        "power_uuid": meta.get("power_uuid", ""),
        "device": meta.get("device", ""),
        "device_display": meta.get("device_display", ""),
        "backend": meta.get("backend", ""),
        "frequency_note": meta.get("frequency_note", ""),
        "model_key": meta.get("model_key", ""),
        "model": meta.get("model", ""),
        "context_tokens": meta.get("context_tokens", ""),
        "algorithm": meta.get("algorithm", ""),
        "case_label": meta.get("case_label", ""),
        "budget": meta.get("budget", ""),
        "pic_recompute_ratio": meta.get("pic_recompute_ratio", ""),
        "pic_recompute_score_layer_idx": meta.get("pic_recompute_score_layer_idx", ""),
        "power_rep_total": meta.get("power_rep_total", ""),
        "drop_first": str(max(args.drop_first, 0)),
        "drop_last": str(max(args.drop_last, 0)),
        "prefill_latency_s": meta.get("prefill_latency_s", ""),
        "power_csv": meta.get("power_csv", ""),
        "power_txt": meta.get("power_txt", ""),
        "power_png": meta.get("power_png", ""),
        "power_run_id": meta.get("power_run_id", ""),
        "manifest": meta.get("manifest", ""),
        "error": "",
    }


def summarize_case(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    meta = load_case_metadata(row)
    out = base_output(meta, args)
    csv_path_text = str(meta.get("power_csv") or "").strip()
    if not csv_path_text:
        out["analysis_status"] = "missing_power_csv"
        out["error"] = "missing power_csv metadata"
        return out

    csv_path = Path(csv_path_text)
    if not csv_path.exists():
        out["analysis_status"] = "missing_power_csv"
        out["error"] = f"{csv_path} does not exist"
        return out

    rep_total = to_int(meta.get("power_rep_total"), 0)
    repetition_results = parse_repetition_results(meta)
    if rep_total <= 0:
        rep_total = max((to_int(item.get("rep_index"), 0) for item in repetition_results), default=0)
    if rep_total <= 0:
        rep_total = 1
    selected_indices = selected_rep_indices(rep_total, args.drop_first, args.drop_last)
    out["power_rep_total"] = str(rep_total)
    out["kept_rep_indices"] = ",".join(str(item) for item in selected_indices)
    out["selected_rep_count"] = str(len(selected_indices))
    if not selected_indices:
        out["analysis_status"] = "no_middle_reps"
        out["error"] = "drop-first/drop-last removes every rep"
        return out

    try:
        samples = read_power_samples(csv_path)
    except Exception as exc:
        out["analysis_status"] = "read_power_csv_failed"
        out["error"] = str(exc)
        return out

    capture_duration = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
    rep_windows, detection = detect_rep_windows(samples, rep_total, args)
    out["capture_duration_s"] = fmt_float(capture_duration, 6)
    out["initial_base_power_w"] = fmt_float(to_float(detection.get("initial_base_power_w")), 6)
    out["active_power_p95_w"] = fmt_float(to_float(detection.get("active_power_p95_w")), 6)
    out["threshold_power_w"] = fmt_float(to_float(detection.get("threshold_power_w")), 6)
    out["threshold_ratio"] = fmt_float(float(args.threshold_ratio), 6)
    out["raw_active_span_count"] = str(detection.get("raw_active_span_count", ""))
    out["segment_grouping"] = str(detection.get("segment_grouping", ""))
    out["detected_rep_count"] = str(len(rep_windows))
    out["detected_rep_windows_s"] = json_windows(rep_windows)

    if detection.get("status") != "ok":
        out["analysis_status"] = str(detection.get("status") or "detect_rep_windows_failed")
        out["error"] = out["analysis_status"]
        return out
    if len(rep_windows) < rep_total:
        out["analysis_status"] = "insufficient_detected_reps"
        out["error"] = f"detected {len(rep_windows)} rep windows for expected {rep_total}"
        return out

    rep_windows = rep_windows[:rep_total]
    selected_windows = [rep_windows[index - 1] for index in selected_indices]
    base_power, base_method = identify_base_power(
        samples,
        rep_windows,
        to_float(detection.get("initial_base_power_w")),
    )
    duration_s, total_energy_j, incremental_energy_j = integrate_windows(
        samples,
        selected_windows,
        base_power,
    )

    selected_count = len(selected_windows)
    work_power_avg_w = total_energy_j / duration_s if duration_s > 0 else math.nan
    work_minus_base_avg_w = (
        work_power_avg_w - base_power
        if math.isfinite(work_power_avg_w) and math.isfinite(base_power)
        else math.nan
    )
    energy_per_inference_j = incremental_energy_j / selected_count if selected_count else math.nan
    duration_per_inference_s = duration_s / selected_count if selected_count else math.nan
    token_count = to_float(meta.get(args.token_field))
    mj_per_token = (
        energy_per_inference_j * 1000.0 / token_count
        if math.isfinite(energy_per_inference_j) and token_count > 0
        else math.nan
    )
    kept_latency_s, kept_tps = summarize_repetition_results(repetition_results, selected_indices)

    out.update(
        {
            "analysis_status": "ok",
            "base_power_w": fmt_float(base_power, 6),
            "base_windows_s": json_base_windows(rep_windows, capture_duration),
            "base_method": base_method,
            "work_power_avg_w": fmt_float(work_power_avg_w, 6),
            "work_minus_base_avg_w": fmt_float(work_minus_base_avg_w, 6),
            "selected_duration_s": fmt_float(duration_s, 6),
            "duration_per_inference_s": fmt_float(duration_per_inference_s, 6),
            "selected_total_energy_j": fmt_float(total_energy_j, 6),
            "selected_baseline_energy_j": fmt_float(base_power * duration_s, 6),
            "selected_incremental_avg_power_w": fmt_float(
                incremental_energy_j / duration_s if duration_s > 0 else math.nan,
                6,
            ),
            "selected_incremental_energy_j": fmt_float(incremental_energy_j, 6),
            "energy_per_inference_j": fmt_float(energy_per_inference_j, 6),
            "energy_per_inference_mj": fmt_float(energy_per_inference_j * 1000.0, 6),
            "mj_per_token": fmt_float(mj_per_token, 6),
            "kept_prefill_latency_s": fmt_float(kept_latency_s, 6),
            "kept_prefill_tps": fmt_float(kept_tps, 6),
            "selected_rep_windows_s": json_selected_windows(rep_windows, selected_indices),
        }
    )
    return out


def parse_plot_windows(text: Any) -> list[dict[str, float | int]]:
    if not str(text or "").strip():
        return []
    try:
        raw = json.loads(str(text))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    windows: list[dict[str, float | int]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rep_index = to_int(item.get("rep_index"), 0)
        start = to_float(item.get("start_s"))
        end = to_float(item.get("end_s"))
        if rep_index > 0 and math.isfinite(start) and math.isfinite(end) and end > start:
            windows.append({"rep_index": rep_index, "start_s": start, "end_s": end})
    return windows


def downsample_for_plot(
    samples: list[tuple[float, float]],
    max_points: int,
) -> list[tuple[float, float]]:
    if max_points <= 0 or len(samples) <= max_points:
        return samples
    step = max(int(math.ceil(len(samples) / max_points)), 1)
    out: list[tuple[float, float]] = []
    for start in range(0, len(samples), step):
        chunk = samples[start : start + step]
        if not chunk:
            continue
        time_sec = sum(item[0] for item in chunk) / len(chunk)
        power_w = sum(item[1] for item in chunk) / len(chunk)
        out.append((time_sec, power_w))
    return out


def shorten(text: Any, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."


def plot_title(summary: dict[str, Any]) -> str:
    parts = [
        str(summary.get("device", "") or "device?"),
        str(summary.get("model_key", "") or summary.get("model", "") or "model?"),
        f"ctx{summary.get('context_tokens', '')}",
        str(summary.get("algorithm", "") or "mode?"),
    ]
    budget = str(summary.get("budget", "") or "")
    if budget:
        parts.append(f"b={budget}")
    status = str(summary.get("analysis_status", ""))
    if status == "ok":
        energy_mj = str(summary.get("energy_per_inference_mj", ""))
        mj_token = str(summary.get("mj_per_token", ""))
        kept = str(summary.get("kept_rep_indices", ""))
        parts.append(f"keep rep {kept}")
        if energy_mj:
            parts.append(f"{energy_mj} mJ")
        if mj_token:
            parts.append(f"{mj_token} mJ/token")
    else:
        parts.append(status or "not analyzed")
    return shorten(" | ".join(item for item in parts if item), 180)


def annotate_windows(ax: Any, summary: dict[str, Any]) -> None:
    detected = parse_plot_windows(summary.get("detected_rep_windows_s"))
    selected = parse_plot_windows(summary.get("selected_rep_windows_s"))
    selected_indices = {int(item["rep_index"]) for item in selected}
    if not detected:
        ax.text(
            0.01,
            0.9,
            str(summary.get("analysis_status") or "no detected rep windows"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#991b1b",
        )
        return

    for item in detected:
        rep_index = int(item["rep_index"])
        start = float(item["start_s"])
        end = float(item["end_s"])
        kept = rep_index in selected_indices
        color = "#16a34a" if kept else "#dc2626"
        fill_alpha = 0.14 if kept else 0.08
        label = f"keep r{rep_index}" if kept else f"drop r{rep_index}"
        ax.axvspan(start, end, color=color, alpha=fill_alpha, linewidth=0)
        ax.axvline(start, color=color, linestyle="--", linewidth=0.9, alpha=0.95)
        ax.axvline(end, color=color, linestyle="--", linewidth=0.9, alpha=0.95)
        ax.text(
            (start + end) * 0.5,
            0.96,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
            color=color,
        )


def render_summary_plot(summaries: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    if not summaries:
        raise ValueError("no summaries to plot")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for --plot-output. Install it in the active Python environment."
        ) from exc

    count = len(summaries)
    dpi = max(int(args.plot_dpi), 40)
    max_height_in = 60000.0 / dpi
    requested_height = max(float(args.plot_row_height), 0.35) * count + 1.0
    fig_height = min(max(requested_height, 2.8), max_height_in)
    fig_width = max(float(args.plot_width), 6.0)
    fig, axes = plt.subplots(
        count,
        1,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        sharex=False,
        squeeze=False,
    )

    for ax, summary in zip(axes[:, 0], summaries):
        csv_path_text = str(summary.get("power_csv") or "").strip()
        samples: list[tuple[float, float]] = []
        error_text = ""
        if csv_path_text:
            try:
                samples = read_power_samples(Path(csv_path_text))
            except Exception as exc:
                error_text = str(exc)
        else:
            error_text = "missing power_csv"

        if samples:
            plot_samples = downsample_for_plot(samples, int(args.plot_max_points_per_case))
            times = [time_sec for time_sec, _ in plot_samples]
            powers = [power_w for _, power_w in plot_samples]
            ax.plot(times, powers, linewidth=0.9, color="#2563eb")
            base_power = to_float(summary.get("base_power_w"))
            if math.isfinite(base_power):
                ax.axhline(base_power, color="#52525b", linestyle="--", linewidth=0.8, alpha=0.8)
            annotate_windows(ax, summary)
            ax.set_xlim(samples[0][0], samples[-1][0])
        else:
            ax.text(
                0.01,
                0.78,
                error_text or str(summary.get("analysis_status") or "no samples"),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#991b1b",
            )

        ax.set_title(plot_title(summary), loc="left", fontsize=8)
        ax.set_ylabel("W", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, color="#d4d4d8", linewidth=0.5, alpha=0.75)

    axes[-1, 0].set_xlabel("Time since power capture start (s)", fontsize=9)
    legend_items = [
        Line2D([0], [0], color="#2563eb", linewidth=1.0, label="power"),
        Line2D([0], [0], color="#52525b", linestyle="--", linewidth=0.8, label="base power"),
        Patch(facecolor="#16a34a", alpha=0.14, label="kept middle rep"),
        Patch(facecolor="#dc2626", alpha=0.08, label="dropped edge rep"),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=8,
        frameon=False,
        ncol=4,
    )
    fig.suptitle(
        f"PIC power rep selection summary ({count} cases)",
        x=0.01,
        y=0.997,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output_path = Path(str(args.plot_output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    recursive = not args.no_recursive
    rows = expand_inputs(args.paths, recursive)
    if not rows:
        raise SystemExit("no power inputs found")

    delimiter = "," if args.delimiter == "csv" else "\t"
    output = open(args.output, "w", encoding="utf-8", newline="") if args.output else sys.stdout
    failed = 0
    total = 0
    summaries: list[dict[str, Any]] = []
    try:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            summary = summarize_case(row, args)
            summaries.append(summary)
            total += 1
            if summary.get("analysis_status") != "ok":
                failed += 1
            writer.writerow(summary)
    except BrokenPipeError:
        return 0
    finally:
        if output is not sys.stdout:
            output.close()
    if args.plot_output:
        render_summary_plot(summaries, args)
    return 1 if total > 0 and failed == total else 0


if __name__ == "__main__":
    raise SystemExit(main())
