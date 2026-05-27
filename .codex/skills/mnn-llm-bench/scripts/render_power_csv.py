#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render DF power CSV files to PNG charts with time on X and power on Y."
    )
    parser.add_argument("csv", nargs="+", help="Input CSV path(s).")
    parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path. Only valid with one CSV unless --overlay is used.",
    )
    parser.add_argument(
        "--out-dir",
        help="Directory for rendered PNG files. Defaults to each CSV directory.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Render all CSV files into one PNG with one vertical subplot per CSV.",
    )
    parser.add_argument(
        "--trim-sec",
        type=float,
        default=0.0,
        help="Trim this many seconds from both the beginning and end before plotting.",
    )
    parser.add_argument(
        "--title",
        help="Chart title. Defaults to the CSV filename.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG DPI. Defaults to 160.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=12.0,
        help="Figure width in inches. Defaults to 12.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.8,
        help="Figure height in inches. Defaults to 4.8.",
    )
    return parser.parse_args()


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


def read_power_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        time_col = find_column(
            reader.fieldnames,
            ["timestamp [us]", "timestamp_us", "time_us", "timestamp"],
        )
        power_col = find_column(
            reader.fieldnames,
            ["power [W]", "power_w", "power"],
        )
        if not time_col or not power_col:
            raise ValueError(
                f"{path}: expected timestamp and power columns, got {reader.fieldnames}"
            )

        times = []
        powers = []
        for row in reader:
            try:
                time_value = float(row[time_col])
                power_value = float(row[power_col])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(time_value) or not math.isfinite(power_value):
                continue
            times.append(time_value / 1_000_000.0)
            powers.append(power_value)

    if not times:
        raise ValueError(f"{path}: no numeric power samples found")

    origin = times[0]
    times = [item - origin for item in times]
    return times, powers


def trim_series(times, powers, trim_sec):
    if trim_sec <= 0:
        return times, powers
    end = times[-1] - trim_sec
    trimmed = [
        (time_value - trim_sec, power_value)
        for time_value, power_value in zip(times, powers)
        if trim_sec <= time_value <= end
    ]
    if not trimmed:
        return times, powers
    trim_times, trim_powers = zip(*trimmed)
    return list(trim_times), list(trim_powers)


def default_output_path(csv_path, out_dir):
    csv_path = Path(csv_path)
    directory = Path(out_dir) if out_dir else csv_path.parent
    return directory / f"{csv_path.stem}.png"


def render_one(csv_path, output_path, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, powers = read_power_csv(csv_path)
    times, powers = trim_series(times, powers, args.trim_sec)

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    ax.plot(times, powers, linewidth=1.1, color="#2563eb")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(args.title or Path(csv_path).name)
    ax.grid(True, color="#d4d4d8", linewidth=0.6, alpha=0.8)

    avg_power = sum(powers) / len(powers)
    ax.axhline(avg_power, color="#dc2626", linewidth=0.9, linestyle="--")
    ax.text(
        0.99,
        0.96,
        f"avg {avg_power:.3f} W",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#991b1b",
    )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_overlay(csv_paths, output_path, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subplot_count = len(csv_paths)
    fig_height = max(args.height, args.height * subplot_count * 0.78)
    fig, axes = plt.subplots(
        subplot_count,
        1,
        figsize=(args.width, fig_height),
        dpi=args.dpi,
        sharex=False,
        squeeze=False,
    )

    for ax, csv_path in zip(axes[:, 0], csv_paths):
        times, powers = read_power_csv(csv_path)
        times, powers = trim_series(times, powers, args.trim_sec)
        label = Path(csv_path).stem
        avg_power = sum(powers) / len(powers)
        ax.plot(times, powers, linewidth=1.0, color="#2563eb")
        ax.axhline(avg_power, color="#dc2626", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Power (W)")
        ax.set_xlabel("Time (s)")
        ax.set_title(label, fontsize=10, loc="left")
        ax.text(
            0.99,
            0.9,
            f"avg {avg_power:.3f} W",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#991b1b",
        )
        ax.grid(True, color="#d4d4d8", linewidth=0.6, alpha=0.8)

    fig.suptitle(args.title or "Power", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()

    if args.output and len(args.csv) > 1 and not args.overlay:
        raise SystemExit("--output with multiple CSV files requires --overlay")

    try:
        if args.overlay:
            output = args.output
            if not output:
                output_dir = Path(args.out_dir) if args.out_dir else Path(args.csv[0]).parent
                output = output_dir / "power_overlay.png"
            rendered = [render_overlay(args.csv, output, args)]
        else:
            rendered = []
            for csv_path in args.csv:
                output = args.output or default_output_path(csv_path, args.out_dir)
                rendered.append(render_one(csv_path, output, args))
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to render PNG files. Install it in the active Python environment."
        ) from exc

    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
