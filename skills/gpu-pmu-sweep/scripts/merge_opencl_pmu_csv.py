#!/usr/bin/env python3
"""Merge synthetic PMU cases and model workload cases into one device report."""

import argparse
import csv
from pathlib import Path


CSV_COLUMNS = ["case_name", "case_args", "case_runs", "pmu_metric_name", "delta_metric"]
SUMMARY_COLUMNS = ["pmu_metric_name", "valid", "valid_cases"]


def _is_changed(value):
    if value is None or value == "":
        return False
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return False


def _normalise_row(row, source):
    """Convert either the synthetic or model CSV schema to the all.csv schema."""
    if "model_path" not in row:
        return {column: row.get(column, "") for column in CSV_COLUMNS}

    case_args = row.get("case_args", "")
    model_path = row.get("model_path", "")
    if model_path and "--model-path=" not in case_args:
        case_args = "{} --model-path={}".format(case_args, model_path).strip()
    status = row.get("status", "")
    error = row.get("error", "")
    if status and status != "ok" and "--status=" not in case_args:
        case_args = "{} --status={}".format(case_args, status).strip()
    if error and "--error=" not in case_args:
        case_args = "{} --error={}".format(case_args, error).strip()
    return {
        "case_name": row.get("case_name", ""),
        "case_args": case_args,
        "case_runs": row.get("case_runs", ""),
        "pmu_metric_name": row.get("pmu_metric_name", ""),
        "delta_metric": row.get("delta_metric", ""),
    }


def read_rows(paths):
    rows = []
    seen = set()
    for value in paths:
        source = Path(value)
        with source.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row = _normalise_row(raw, source)
                key = tuple(row[column] for column in CSV_COLUMNS)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def _ordered_unique(values):
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def write_report(rows, output_csv, summary_csv):
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    events = _ordered_unique(row["pmu_metric_name"] for row in rows)
    cases = _ordered_unique(row["case_name"] for row in rows)
    changed_cases = {event: [] for event in events}
    for row in rows:
        event = row["pmu_metric_name"]
        case = row["case_name"]
        if _is_changed(row["delta_metric"]) and case not in changed_cases[event]:
            changed_cases[event].append(case)

    summary = Path(summary_csv)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for event in events:
            writer.writerow({
                "pmu_metric_name": event,
                "valid": "true" if changed_cases[event] else "false",
                "valid_cases": ";".join(case for case in cases if case in changed_cases[event]),
            })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", action="append", required=True,
                        help="input synthetic or model CSV; repeat for more workloads")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    write_report(read_rows(args.input_csv), args.output_csv, args.summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
