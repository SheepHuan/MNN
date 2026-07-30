#!/usr/bin/env python3
"""Parse CudaMetric.AllMetrics output into complete and valid-only CSV files."""

import argparse
import csv
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


def parse_sweep_log(text):
    """Return one record for every metric result line in *text*."""
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = LINE_RE.match(line.strip())
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
