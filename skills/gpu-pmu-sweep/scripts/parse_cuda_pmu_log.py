#!/usr/bin/env python3
"""Parse CudaMetric.AllMetrics output into complete and valid-only CSV files."""

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path


ALL_COLUMNS = ["index", "total", "metric", "status", "value"]
VALID_COLUMNS = ["metric", "value"]
STATUSES = {"VALID", "NOT_FOUND", "OVERFLOW", "COMMAND_FAILED", "MALFORMED_OUTPUT"}
LINE_RE = re.compile(
    r"^\[\s*(?P<index>\d+)\s*/\s*(?P<total>\d+)\]\s+"
    r"(?P<metric>\S+)\s+(?P<status>[A-Z_]+)\s+value=\s*(?P<value>\S+)\s*$"
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUMMARY_RE = re.compile(
    r"^all metrics:\s*"
    r"(?P<valid>\d+) valid,\s*"
    r"(?P<not_found>\d+) not_found,\s*"
    r"(?P<overflow>\d+) overflow,\s*"
    r"(?P<command_failed>\d+) command_failed\s*"
    r"\(of (?P<total>\d+)\)\s*$"
)
ALL_METRICS_OK_RE = re.compile(r"^\[\s*OK\s*\]\s+CudaMetric\.AllMetrics\s*$")
SUITE_PASSED_RE = re.compile(r"^\[\s*PASSED\s*\]\s+\d+\s+tests?\.\s*$")


def _plain_line(line):
    return ANSI_ESCAPE_RE.sub("", line).strip()


def _validate_complete_log(records, summaries, all_metrics_ok, suite_passed):
    if not records:
        raise ValueError("availability log contains no metric result lines")

    totals = {record["total"] for record in records}
    if len(totals) != 1:
        raise ValueError("availability log uses inconsistent total metric counts")
    total = totals.pop()
    if total <= 0:
        raise ValueError("availability log total metric count must be positive")

    actual_indices = [record["index"] for record in records]
    expected_indices = list(range(1, total + 1))
    if actual_indices != expected_indices:
        raise ValueError(
            "availability log is incomplete or out of order: expected indices 1..{}, "
            "found {} complete result lines".format(total, len(records))
        )

    valid_metrics = set()
    for record in records:
        if record["status"] != "VALID":
            continue
        metric = record["metric"]
        if metric in valid_metrics:
            raise ValueError(
                "availability log contains duplicate VALID metric: {}".format(metric)
            )
        valid_metrics.add(metric)
        try:
            value = float(record["value"])
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "availability VALID metric {} has invalid value {}; expected a nonnegative "
                "finite number".format(metric, record["value"])
            )

    if len(summaries) != 1:
        raise ValueError(
            "availability log must contain exactly one completed all-metrics summary"
        )
    summary = summaries[0]
    if summary["total"] != total:
        raise ValueError("availability summary total does not match metric result total")

    counts = Counter(record["status"] for record in records)
    expected_counts = {
        "VALID": summary["valid"],
        "NOT_FOUND": summary["not_found"],
        "OVERFLOW": summary["overflow"],
        "COMMAND_FAILED": summary["command_failed"],
    }
    for status, expected in expected_counts.items():
        if counts.get(status, 0) != expected:
            raise ValueError(
                "availability summary {} count does not match metric results".format(status)
            )
    unexpected = sorted(set(counts) - set(expected_counts))
    if unexpected:
        raise ValueError(
            "availability summary cannot account for statuses: {}".format(
                ", ".join(unexpected)
            )
        )
    if sum(expected_counts.values()) != total:
        raise ValueError("availability summary status counts do not add up to total")
    if not all_metrics_ok:
        raise ValueError("CudaMetric.AllMetrics did not complete with OK status")
    if not suite_passed:
        raise ValueError("availability test suite did not complete with PASSED status")


def parse_sweep_log(text):
    """Return records only when *text* is one complete, passing AllMetrics run."""
    records = []
    summaries = []
    all_metrics_ok = False
    suite_passed = False
    for line_number, line in enumerate(text.splitlines(), 1):
        plain = _plain_line(line)
        summary_match = SUMMARY_RE.match(plain)
        if summary_match:
            summaries.append({key: int(value) for key, value in summary_match.groupdict().items()})
        if ALL_METRICS_OK_RE.match(plain):
            all_metrics_ok = True
        if SUITE_PASSED_RE.match(plain):
            suite_passed = True

        match = LINE_RE.match(plain)
        if not match:
            continue
        record = match.groupdict()
        if record["status"] not in STATUSES:
            raise ValueError("unknown PMU status on line {}: {}".format(line_number, record["status"]))
        records.append({
            "index": int(record["index"]),
            "total": int(record["total"]),
            "metric": record["metric"],
            "status": record["status"],
            "value": record["value"],
        })
    _validate_complete_log(records, summaries, all_metrics_ok, suite_passed)
    return records


def write_csvs(records, all_path, valid_path):
    all_path = Path(all_path)
    valid_path = Path(valid_path)
    all_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    with all_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ALL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    with valid_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=VALID_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            if record["status"] == "VALID":
                writer.writerow({"metric": record["metric"], "value": record["value"]})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CudaMetric.AllMetrics log")
    parser.add_argument("--all-csv", required=True, help="CSV containing every status")
    parser.add_argument("--valid-csv", required=True, help="CSV containing VALID metrics only")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = parse_sweep_log(Path(args.input).read_text(encoding="utf-8"))
    write_csvs(records, args.all_csv, args.valid_csv)
    counts = Counter(record["status"] for record in records)
    print("parsed {} metrics: {}".format(len(records), ", ".join(
        "{} {}".format(status.lower(), counts.get(status, 0)) for status in sorted(STATUSES)
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
