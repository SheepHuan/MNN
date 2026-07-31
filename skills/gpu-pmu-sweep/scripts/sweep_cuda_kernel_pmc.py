#!/usr/bin/env python3
"""Sweep selected CUDA corpus cases against every globally valid CUDA PMC.

Each invocation contains exactly one case and a bounded group of metrics. CUPTI
owns any replay passes inside that session. The detailed CSV is the resume
ledger; the compact JSON contains PMC values only:
``{case: {"pmc": {metric: value}}}``.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROW_COLUMNS = ["case", "metric", "status", "value", "pmu_status", "num_passes", "returncode", "error"]
VALID_STATUSES = {"VALID"}
OVERFLOW_SENTINEL = 1 << 63


def discover_cuda_cases(corpus_root, selection="op-type"):
    if selection not in {"op-type", "all"}:
        raise ValueError("unknown CUDA case selection: {}".format(selection))
    data = json.loads((Path(corpus_root) / "operator_cases.json").read_text(encoding="utf-8"))
    cases = []
    seen = set()
    representatives = {}
    for case in data.get("cases", []):
        name = case.get("name")
        if case.get("backend") == "cuda" and isinstance(name, str) and name and name not in seen:
            seen.add(name)
            if selection == "all":
                cases.append(name)
                continue
            op_type = case.get("op_type")
            if not isinstance(op_type, str) or not op_type:
                raise ValueError("CUDA case {} has no op_type".format(name))
            # Keep the first corpus entry for each op type. Corpus generation
            # orders the representative variants before shape-specific cases.
            representatives.setdefault(op_type, name)
    if selection == "all":
        return cases
    return [representatives[op_type] for op_type in sorted(representatives)]


def read_valid_metrics(path):
    metrics = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "metric" not in (reader.fieldnames or []):
            raise ValueError("{} must contain a metric column".format(path))
        for row in reader:
            metric = row["metric"].strip()
            if metric and metric not in seen:
                metrics.append(metric)
                seen.add(metric)
    return metrics


def _normalize_metrics(metrics):
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = list(metrics)
    if not metrics or any(not metric or "," in metric for metric in metrics):
        raise ValueError("metrics must be non-empty names without commas")
    return metrics


def build_benchmark_argv(binary, corpus_root, case_name, metrics, output_path, runs=1):
    metrics = _normalize_metrics(metrics)
    if not case_name:
        raise ValueError("case_name must be non-empty")
    return [
        str(binary), "--kernel-corpus-bench", "--kernel-corpus-root", str(corpus_root),
        "--kernel-corpus-case", case_name, "--kernel-corpus-runs", str(runs),
        "--kernel-corpus-no-latency",
        "--perf-counter-events", ",".join(metrics), "--perf-counter-output", str(output_path),
    ]


def _find_case(report, case_name):
    for case in report.get("cases", []):
        if case.get("case") == case_name:
            return case
    return None


def classify_report(report, case_name, metric, returncode=0, error=""):
    case = _find_case(report, case_name)
    if returncode != 0:
        return {"status": "COMMAND_FAILED", "value": "", "pmu_status": "", "num_passes": "",
                "returncode": returncode, "error": error}
    if not isinstance(case, dict):
        return {"status": "MALFORMED_OUTPUT", "value": "", "pmu_status": "", "num_passes": "",
                "returncode": returncode, "error": "case report missing"}
    status = str(case.get("pmu_status", ""))
    raw = case.get("pmu_metrics", {}).get(metric) if isinstance(case.get("pmu_metrics"), dict) else None
    num_passes = case.get("num_passes", "")
    common = {"pmu_status": status, "num_passes": num_passes, "returncode": returncode}
    if "start_failed" in status or "stop_failed" in status or "cuda_event_fallback(" in status:
        return {"status": "COMMAND_FAILED", "value": "" if raw is None else str(raw), **common,
                "error": case.get("error", "") or status}
    if raw is not None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = None
        if value == OVERFLOW_SENTINEL:
            return {"status": "OVERFLOW", "value": str(raw), **common,
                    "error": case.get("error", "")}
        if status == "sampled" and value is not None and value >= 0:
            return {"status": "VALID", "value": str(value), **common,
                    "error": case.get("error", "")}
        if value is not None:
            return {"status": "NOT_FOUND", "value": str(value), **common,
                    "error": case.get("error", "")}
    if status in {"unavailable", "unsupported", ""}:
        result_status = "NOT_FOUND"
    else:
        result_status = "MALFORMED_OUTPUT"
    return {"status": result_status, "value": "", **common,
            "error": case.get("error", "") or error or status}


def run_batch(binary, corpus_root, case_name, metrics, workdir, lib_dir=None, use_sudo=False,
              runs=1, timeout=300, runner=subprocess.run):
    metrics = _normalize_metrics(metrics)
    temp_dir = None
    output_path = None
    try:
        # Keep the path absent because replay_benchmark creates the output file.
        # A private user-owned directory also lets the caller remove files created
        # by a root benchmark when --sudo is enabled.
        temp_dir = tempfile.mkdtemp(prefix="replay_cuda_pmu_")
        output_path = str(Path(temp_dir) / "report.json")
        argv = build_benchmark_argv(binary, corpus_root, case_name, metrics, output_path, runs)
        env = os.environ.copy()
        if lib_dir:
            env["LD_LIBRARY_PATH"] = str(lib_dir)
        if use_sudo:
            argv = ["sudo", "env", "LD_LIBRARY_PATH={}".format(env.get("LD_LIBRARY_PATH", ""))] + argv
        completed = runner(argv, cwd=str(workdir), env=env, capture_output=True, text=True,
                           timeout=timeout, check=False)
        report = {}
        if Path(output_path).is_file() and Path(output_path).stat().st_size:
            try:
                report = json.loads(Path(output_path).read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {metric: classify_report({}, case_name, metric, completed.returncode, "invalid JSON output")
                        for metric in metrics}
        return {metric: classify_report(report, case_name, metric, completed.returncode,
                                        completed.stderr.strip()) for metric in metrics}
    except subprocess.TimeoutExpired:
        return {metric: classify_report({}, case_name, metric, -1, "timeout") for metric in metrics}
    finally:
        if output_path:
            try:
                Path(output_path).unlink()
            except FileNotFoundError:
                pass
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_one(binary, corpus_root, case_name, metric, workdir, lib_dir=None, use_sudo=False,
            runs=1, timeout=300, runner=subprocess.run):
    """Compatibility wrapper for a one-metric invocation."""
    return run_batch(binary, corpus_root, case_name, [metric], workdir, lib_dir, use_sudo,
                     runs, timeout, runner)[metric]


def run_batch_with_fallback(binary, corpus_root, case_name, metrics, workdir, lib_dir=None,
                            use_sudo=False, runs=1, timeout=300, runner=subprocess.run):
    """Run a batch, splitting it when CUPTI rejects the metric configuration."""
    metrics = _normalize_metrics(metrics)
    result = run_batch(binary, corpus_root, case_name, metrics, workdir, lib_dir, use_sudo,
                       runs, timeout, runner)
    failure_statuses = {"COMMAND_FAILED", "MALFORMED_OUTPUT"}
    if len(metrics) <= 1 or not all(result[metric]["status"] in failure_statuses for metric in metrics):
        return result
    middle = len(metrics) // 2
    left = run_batch_with_fallback(binary, corpus_root, case_name, metrics[:middle], workdir,
                                   lib_dir, use_sudo, runs, timeout, runner)
    right = run_batch_with_fallback(binary, corpus_root, case_name, metrics[middle:], workdir,
                                    lib_dir, use_sudo, runs, timeout, runner)
    left.update(right)
    return left


def _read_rows(path, cases=None, metrics=None):
    completed = set()
    if not Path(path).is_file():
        return completed
    case_set = set(cases) if cases is not None else None
    metric_set = set(metrics) if metrics is not None else None
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROW_COLUMNS:
            raise ValueError("{} must have columns {}".format(path, ",".join(ROW_COLUMNS)))
        for row in reader:
            if case_set is not None and row["case"] not in case_set:
                continue
            if metric_set is not None and row["metric"] not in metric_set:
                continue
            completed.add((row["case"], row["metric"]))
    return completed


def build_result_json_from_csv(cases, path):
    result = {case: {"pmc": {}} for case in cases}
    if not Path(path).is_file():
        return result
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROW_COLUMNS:
            raise ValueError("{} must have columns {}".format(path, ",".join(ROW_COLUMNS)))
        for row in reader:
            case = row["case"]
            if case not in result:
                continue
            if row["status"] in VALID_STATUSES and row["value"]:
                result[case]["pmc"][row["metric"]] = int(row["value"])
    return result


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def build_result_json(cases, rows):
    result = {case: {"pmc": {}} for case in cases}
    for row in rows:
        case = row["case"]
        if case not in result:
            result[case] = {"pmc": {}}
        if row["status"] in VALID_STATUSES and row["value"]:
            result[case]["pmc"][row["metric"]] = int(row["value"])
    return result


def metric_batches(metrics, batch_size):
    if batch_size <= 0:
        raise ValueError("metrics-per-session must be positive")
    for start in range(0, len(metrics), batch_size):
        yield metrics[start:start + batch_size]


def run_sweep(args, executor=run_batch_with_fallback):
    corpus_root = Path(args.corpus_root).resolve()
    cases = discover_cuda_cases(corpus_root, args.case_selection)
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case]
    if args.max_cases:
        cases = cases[:args.max_cases]
    metrics = read_valid_metrics(args.valid_csv)
    if args.max_metrics:
        metrics = metrics[:args.max_metrics]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_rows(output_csv, cases, metrics) if args.resume else set()
    mode = "a" if args.resume and output_csv.is_file() else "w"
    with output_csv.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_COLUMNS, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        total = len(cases) * len(metrics)
        done = len(completed)
        for case in cases:
            for batch in metric_batches(metrics, args.metrics_per_session):
                pending = [metric for metric in batch if (case, metric) not in completed]
                if not pending:
                    continue
                batch_rows = executor(args.binary, corpus_root, case, pending, args.workdir,
                                      args.lib_dir, args.sudo, args.runs, args.timeout)
                for metric in pending:
                    row = {"case": case, "metric": metric, **batch_rows[metric]}
                    writer.writerow(row)
                    completed.add((case, metric))
                    done += 1
                    print("[{}/{}] {} {} {}".format(done, total, case, metric,
                                                     row["status"]), flush=True)
                stream.flush()
    payload = build_result_json_from_csv(cases, output_csv)
    _write_json_atomic(args.output_json, payload)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--lib-dir")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sudo", action="store_true", help="run each benchmark through sudo")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--metrics-per-session", type=int, default=32,
                        help="metrics passed to one CUPTI session (default: 32)")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-metrics", type=int)
    parser.add_argument("--case-filter")
    parser.add_argument("--case-selection", choices=("op-type", "all"), default="op-type",
                        help="select one corpus case per op_type (default) or scan all cases")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
