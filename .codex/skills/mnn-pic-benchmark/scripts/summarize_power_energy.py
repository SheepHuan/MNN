#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize eperf power CSV energy after trimming non-benchmark warmup/cooldown."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Power CSV files, a manifest.tsv, or directories containing *_power.csv files.",
    )
    parser.add_argument(
        "--trim-sec",
        type=float,
        default=10.0,
        help="Seconds to trim from both beginning and end. Defaults to 10.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write TSV summary to this path. Defaults to stdout.",
    )
    return parser.parse_args()


def read_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            csv_path = row.get("power_csv") or row.get("csv")
            if not csv_path:
                continue
            rows.append(
                {
                    "model_ref": row.get("model_ref", ""),
                    "model_dir": row.get("model_dir", ""),
                    "prompt": row.get("prompt", ""),
                    "decode": row.get("decode", ""),
                    "rep": row.get("rep", ""),
                    "csv": csv_path,
                    "log": row.get("log_path", "") or row.get("log", ""),
                }
            )
    return rows


def infer_from_csv_path(path):
    name = Path(path).name
    match = re.search(r"_p(?P<prompt>\d+)_n(?P<decode>\d+)(?:_rep(?P<rep>\d+))?_power\.csv$", name)
    return {
        "model_ref": "",
        "model_dir": re.sub(r"_p\d+_n\d+(?:_rep\d+)?_power\.csv$", "", name),
        "prompt": match.group("prompt") if match else "",
        "decode": match.group("decode") if match else "",
        "rep": match.group("rep") if match and match.group("rep") else "1",
        "csv": str(path),
        "log": "",
    }


def expand_inputs(paths):
    rows = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            for csv_path in sorted(path.glob("*_power.csv")):
                rows.append(infer_from_csv_path(csv_path))
        elif path.name == "manifest.tsv":
            rows.extend(read_manifest(path))
        elif path.suffix.lower() == ".csv":
            rows.append(infer_from_csv_path(path))
        else:
            for csv_path in sorted(glob.glob(item)):
                rows.append(infer_from_csv_path(csv_path))
    return rows


def find_column(fieldnames, candidates):
    normalized = {
        re.sub(r"[^a-z0-9]+", "", name.lower()): name
        for name in (fieldnames or [])
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def read_power_samples(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        time_col = find_column(reader.fieldnames, ["timestamp [us]", "timestamp_us", "time_us", "timestamp"])
        power_col = find_column(reader.fieldnames, ["power [W]", "power_w", "power"])
        if not time_col or not power_col:
            raise ValueError(f"{path}: expected timestamp and power columns, got {reader.fieldnames}")

        samples = []
        for row in reader:
            try:
                timestamp_sec = float(row[time_col]) / 1_000_000.0
                power_w = float(row[power_col])
            except (TypeError, ValueError):
                continue
            if math.isfinite(timestamp_sec) and math.isfinite(power_w):
                samples.append((timestamp_sec, power_w))
    if not samples:
        raise ValueError(f"{path}: no numeric samples")
    origin = samples[0][0]
    return [(time_sec - origin, power_w) for time_sec, power_w in samples]


def trim_samples(samples, trim_sec):
    if trim_sec <= 0:
        return samples
    end_time = samples[-1][0] - trim_sec
    trimmed = [(t - trim_sec, p) for t, p in samples if trim_sec <= t <= end_time]
    return trimmed or samples


def integrate_joules(samples):
    if len(samples) < 2:
        return 0.0
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt > 0:
            energy += dt * (p0 + p1) * 0.5
    return energy


def parse_int(value, default=1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def summarize(row, trim_sec):
    csv_path = row["csv"]
    samples = read_power_samples(csv_path)
    total_capture_sec = samples[-1][0] - samples[0][0]
    bench_samples = trim_samples(samples, trim_sec)
    bench_duration_sec = bench_samples[-1][0] - bench_samples[0][0] if len(bench_samples) >= 2 else 0.0
    energy_j = integrate_joules(bench_samples)
    avg_power_w = energy_j / bench_duration_sec if bench_duration_sec > 0 else 0.0
    max_power_w = max(power for _, power in bench_samples)
    rep = max(parse_int(row.get("rep"), 1), 1)
    out = dict(row)
    out.update(
        {
            "rep": str(rep),
            "trim_sec_each_side": f"{trim_sec:.3f}",
            "capture_duration_s": f"{total_capture_sec:.3f}",
            "bench_duration_s": f"{bench_duration_sec:.3f}",
            "avg_power_w": f"{avg_power_w:.6f}",
            "max_power_w": f"{max_power_w:.6f}",
            "bench_energy_j": f"{energy_j:.6f}",
            "energy_per_iter_j": f"{energy_j / rep:.6f}",
            "energy_per_iter_wh": f"{energy_j / rep / 3600.0:.9f}",
        }
    )
    return out


def main():
    args = parse_args()
    rows = expand_inputs(args.paths)
    if not rows:
        raise SystemExit("no power CSV inputs found")

    output_fields = [
        "model_ref",
        "model_dir",
        "prompt",
        "decode",
        "rep",
        "trim_sec_each_side",
        "capture_duration_s",
        "bench_duration_s",
        "avg_power_w",
        "max_power_w",
        "bench_energy_j",
        "energy_per_iter_j",
        "energy_per_iter_wh",
        "csv",
        "log",
    ]
    summaries = [summarize(row, args.trim_sec) for row in rows]

    fp = open(args.output, "w", encoding="utf-8", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(fp, fieldnames=output_fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    finally:
        if fp is not sys.stdout:
            fp.close()


if __name__ == "__main__":
    main()
