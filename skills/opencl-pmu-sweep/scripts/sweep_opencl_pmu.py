#!/usr/bin/env python3
"""Run one OpenCL PMU event per benchmark process and emit normalized CSV."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEVICES = {
    "rhinopi": ("root@192.168.101.227", "/mnt/nvme/workspace/replay-benchmark"),
    "orangepi": ("root@192.168.101.113", "/mnt/ssd/workspace"),
}
CSV_COLUMNS = ["case_name", "case_args", "case_runs", "pmu_metric_name", "delta_metric"]


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def extract_delta(report, case_name, metric_name):
    for case in report.get("cases", []):
        if case.get("name") != case_name:
            continue
        for counter in case.get("counters", []):
            if counter.get("name") == metric_name:
                if "control_delta" not in counter or "workload_delta" not in counter:
                    return ""
                return str(int(counter["workload_delta"]) - int(counter["control_delta"]))
    return ""


def make_case_args(args, measurement):
    values = [
        "--opencl-pmu-size=" + str(args.size),
        "--opencl-pmu-local-size=" + str(args.local_size),
        "--opencl-pmu-iterations=" + str(args.iterations),
        "--opencl-pmu-workload-runs=" + str(args.workload_runs),
        "--opencl-pmu-warmup-runs=" + str(args.warmup_runs),
        "--measurement=" + str(measurement),
    ]
    return " ".join(values)


def build_benchmark_argv(binary, case_name, event_name, args):
    if "," in event_name or not event_name.strip():
        raise ValueError("each PMU invocation must contain exactly one event")
    return [
        binary,
        "--opencl-pmu-bench",
        "--opencl-pmu-case",
        case_name,
        "--opencl-pmu-size",
        str(args.size),
        "--opencl-pmu-local-size",
        str(args.local_size),
        "--opencl-pmu-iterations",
        str(args.iterations),
        "--opencl-pmu-workload-runs",
        str(args.workload_runs),
        "--opencl-pmu-warmup-runs",
        str(args.warmup_runs),
        "--perf-counter-events",
        event_name,
    ]


def run_process(argv, env=None, timeout=None):
    completed = subprocess.run(argv, capture_output=True, text=True, env=env, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError("benchmark failed (rc={}): {}".format(completed.returncode, completed.stderr.strip()))
    output = completed.stdout.strip()
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("benchmark did not emit a JSON report: {}".format(output[-400:]))
    return json.loads(output[start : end + 1])


def execute(binary, case_name, event_name, args):
    argv = build_benchmark_argv(binary, case_name, event_name, args)
    if args.device is None:
        env = os.environ.copy()
        if args.lib_dir:
            env["LD_LIBRARY_PATH"] = args.lib_dir
        return run_process(argv, env, args.timeout)

    host, default_root = DEVICES[args.device]
    remote_root = args.remote_root or default_root
    remote_argv = list(argv)
    remote_argv[0] = remote_root + "/bin/replay_benchmark.out"
    command = "cd {} && env LD_LIBRARY_PATH={} {}".format(
        shlex.quote(remote_root),
        shlex.quote(remote_root + "/lib"),
        " ".join(shlex.quote(item) for item in remote_argv),
    )
    return run_process(["ssh", "-o", "BatchMode=yes", host, command], timeout=args.timeout)


def run_sweep(args, executor=execute):
    cases = parse_csv_list(args.cases)
    events = parse_csv_list(args.events)
    if not cases or not events:
        raise ValueError("cases and events must not be empty")
    if args.measurements < 1 or args.workload_runs < 1 or args.warmup_runs < 0:
        raise ValueError("measurements and workload-runs must be positive; warmup-runs cannot be negative")

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output.parent / (output.stem + "-raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_name in cases:
        for event_name in events:
            for measurement in range(1, args.measurements + 1):
                try:
                    report = executor(args.binary, case_name, event_name, args)
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                    print("{} / {} / measurement {}: {}".format(case_name, event_name, measurement, error), file=sys.stderr)
                    report = {"cases": [], "error": str(error)}
                raw_path = raw_dir / "{}__{}__{:03d}.json".format(case_name, event_name, measurement)
                raw_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                rows.append(
                    {
                        "case_name": case_name,
                        "case_args": make_case_args(args, measurement),
                        "case_runs": str(args.workload_runs),
                        "pmu_metric_name": event_name,
                        "delta_metric": extract_delta(report, case_name, event_name),
                    }
                )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="buffer_fp32", help="comma-separated case names")
    parser.add_argument("--events", default="gpu_active_cycles", help="comma-separated PMU events; one is passed per process")
    parser.add_argument("--output-csv", "--output", required=True)
    parser.add_argument("--raw-dir", "--keep-json-dir")
    parser.add_argument("--device", choices=sorted(DEVICES))
    parser.add_argument("--remote-root")
    parser.add_argument("--binary", default="./replay_benchmark.out")
    parser.add_argument("--lib-dir")
    parser.add_argument("--size", type=int, default=262144)
    parser.add_argument("--local-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workload-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        run_sweep(parse_args(argv))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print("opencl-pmu-sweep: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
