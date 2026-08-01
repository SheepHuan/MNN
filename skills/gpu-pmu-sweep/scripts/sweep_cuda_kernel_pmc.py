#!/usr/bin/env python3
"""Sweep selected CUDA corpus cases against every globally valid CUDA PMC.

Each invocation contains exactly one case and a bounded group of metrics. CUPTI
owns any replay passes inside that session. The detailed CSV is the resume
ledger; the compact JSON contains PMC values only:
``{case: {"pmc": {metric: value}}}``.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernel_agent.pmc_interpreter.dataset.selection import (  # noqa: E402
    ALL_CASES_POLICY,
    CONDITION_BALANCED_POLICY,
    build_case_selection_plan,
    load_case_selection_plan,
    validate_case_selection_plan_source,
    write_case_selection_plan,
)


ENVIRONMENT_COLUMNS = [
    "collection_session_id",
    "order_index",
    "gpu_clock_hz_before",
    "gpu_clock_hz_after",
    "temperature_c_before",
    "temperature_c_after",
    "environment_status",
    "environment_source",
]
ROW_COLUMNS = [
    "sweep_plan_id",
    "sweep_config_id",
    "case",
    "metric",
    "status",
    "value",
    "pmu_status",
    "num_passes",
    "returncode",
    "error",
] + ENVIRONMENT_COLUMNS
VALID_STATUSES = {"VALID"}
OVERFLOW_SENTINEL = 1 << 63


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


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path) if path.is_file() else "missing",
    }


def _default_selection_plan_path(output_csv):
    return Path("{}.selection.json".format(output_csv))


def _resolve_selection_plan(args, corpus_root, output_csv):
    manifest_path = corpus_root / "operator_cases.json"
    plan_output_path = Path(
        getattr(args, "selection_plan_output", None)
        or _default_selection_plan_path(output_csv)
    )
    plan_input_raw = getattr(args, "selection_plan_input", None)
    if args.resume and not plan_input_raw:
        plan_input_raw = str(plan_output_path)

    if plan_input_raw:
        plan_input_path = Path(plan_input_raw)
        if not plan_input_path.is_file():
            raise ValueError("case selection plan does not exist: {}".format(plan_input_path))
        plan = load_case_selection_plan(plan_input_path)
        validate_case_selection_plan_source(plan, manifest_path)
        rebuilt = build_case_selection_plan(
            manifest_path,
            backend=plan.backend,
            policy=plan.policy,
            minimum_conditions_per_op_type=plan.minimum_conditions_per_op_type,
            target_op_types=plan.target_op_types,
        )
        if rebuilt.model_dump(mode="json") != plan.model_dump(mode="json"):
            raise ValueError("case selection plan no longer matches the current selection algorithm")
    else:
        plan = build_case_selection_plan(
            manifest_path,
            backend="cuda",
            policy=getattr(args, "case_selection", CONDITION_BALANCED_POLICY),
            minimum_conditions_per_op_type=getattr(args, "conditions_per_op_type", 5),
            target_op_types=getattr(args, "target_op_type", None),
        )
    if plan.backend != "cuda":
        raise ValueError("CUDA sweep requires a cuda case selection plan")

    if args.resume and plan_output_path.is_file():
        associated = load_case_selection_plan(plan_output_path)
        if associated.plan_id != plan.plan_id:
            raise ValueError(
                "resume selection plan mismatch: expected {}, found {}".format(
                    associated.plan_id, plan.plan_id
                )
            )
    write_case_selection_plan(plan, plan_output_path)
    return plan


def _sweep_config_id(args, plan, metrics):
    workdir = Path(args.workdir).resolve()
    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = workdir / binary
    payload = {
        "schema": "mnn-cuda-pmc-sweep-config/v1",
        "selection_plan_id": plan.plan_id,
        "valid_metric_source": _file_identity(args.valid_csv),
        "ordered_metrics": list(metrics),
        "binary": _file_identity(binary),
        "workdir": str(workdir),
        "lib_dir": str(args.lib_dir or ""),
        "runs": args.runs,
        "timeout": args.timeout,
        "metrics_per_session": args.metrics_per_session,
        "sudo": bool(args.sudo),
        "device_fingerprint": str(getattr(args, "device_fingerprint", "unspecified")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def _empty_environment(collection_session_id=""):
    return {
        "collection_session_id": collection_session_id,
        "order_index": "",
        "gpu_clock_hz_before": "",
        "gpu_clock_hz_after": "",
        "temperature_c_before": "",
        "temperature_c_after": "",
        "environment_status": "not_collected",
        "environment_source": "",
    }


def classify_report(
    report,
    case_name,
    metric,
    returncode=0,
    error="",
    collection_session_id="",
):
    case = _find_case(report, case_name)
    environment = _empty_environment(collection_session_id)
    if returncode != 0:
        return {"status": "COMMAND_FAILED", "value": "", "pmu_status": "", "num_passes": "",
                "returncode": returncode, "error": error, **environment}
    if not isinstance(case, dict):
        return {"status": "MALFORMED_OUTPUT", "value": "", "pmu_status": "", "num_passes": "",
                "returncode": returncode, "error": "case report missing", **environment}
    status = str(case.get("pmu_status", ""))
    raw = case.get("pmu_metrics", {}).get(metric) if isinstance(case.get("pmu_metrics"), dict) else None
    num_passes = case.get("num_passes", "")
    common = {"pmu_status": status, "num_passes": num_passes, "returncode": returncode}
    if "start_failed" in status or "stop_failed" in status or "cuda_event_fallback(" in status:
        return {"status": "COMMAND_FAILED", "value": "" if raw is None else str(raw), **common,
                "error": case.get("error", "") or status, **environment}
    if raw is not None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = None
        if value == OVERFLOW_SENTINEL:
            return {"status": "OVERFLOW", "value": str(raw), **common,
                    "error": case.get("error", ""), **environment}
        if status == "sampled" and value is not None and value >= 0:
            return {"status": "VALID", "value": str(value), **common,
                    "error": case.get("error", ""), **environment}
        if value is not None:
            return {"status": "NOT_FOUND", "value": str(value), **common,
                    "error": case.get("error", ""), **environment}
    if status in {"unavailable", "unsupported", ""}:
        result_status = "NOT_FOUND"
    else:
        result_status = "MALFORMED_OUTPUT"
    return {"status": result_status, "value": "", **common,
            "error": case.get("error", "") or error or status, **environment}


def run_batch(binary, corpus_root, case_name, metrics, workdir, lib_dir=None, use_sudo=False,
              runs=1, timeout=300, runner=subprocess.run):
    metrics = _normalize_metrics(metrics)
    collection_session_id = "cuda-pmc-{}".format(uuid.uuid4().hex)
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
                return {
                    metric: classify_report(
                        {}, case_name, metric, completed.returncode,
                        "invalid JSON output", collection_session_id,
                    )
                    for metric in metrics
                }
        return {metric: classify_report(report, case_name, metric, completed.returncode,
                                        completed.stderr.strip(), collection_session_id)
                for metric in metrics}
    except subprocess.TimeoutExpired:
        return {
            metric: classify_report(
                {}, case_name, metric, -1, "timeout", collection_session_id
            )
            for metric in metrics
        }
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


def _expected_pairs(cases, metrics):
    return [(case, metric) for case in cases for metric in metrics]


def _read_rows(path, cases, metrics, sweep_plan_id, sweep_config_id, require_file=False):
    completed = set()
    if not Path(path).is_file():
        if require_file:
            raise ValueError("resume ledger does not exist: {}".format(path))
        return completed
    expected = _expected_pairs(cases, metrics)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROW_COLUMNS:
            raise ValueError("{} must have columns {}".format(path, ",".join(ROW_COLUMNS)))
        for index, row in enumerate(reader):
            if index >= len(expected):
                raise ValueError("{} has rows outside the current sweep plan".format(path))
            pair = (row["case"], row["metric"])
            if pair != expected[index]:
                raise ValueError(
                    "{} is not a canonical sweep prefix at row {}: expected {}, found {}".format(
                        path, index + 2, expected[index], pair
                    )
                )
            if row["sweep_plan_id"] != sweep_plan_id:
                raise ValueError(
                    "{} row {} uses a different case selection plan".format(path, index + 2)
                )
            if row["sweep_config_id"] != sweep_config_id:
                raise ValueError(
                    "{} row {} uses a different sweep configuration".format(path, index + 2)
                )
            if pair in completed:
                raise ValueError("{} contains duplicate pair {}".format(path, pair))
            completed.add(pair)
    return completed


def build_result_json_from_csv(cases, metrics, path, sweep_plan_id, sweep_config_id):
    result = {case: {"pmc": {}} for case in cases}
    if not Path(path).is_file():
        return result
    _read_rows(path, cases, metrics, sweep_plan_id, sweep_config_id)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROW_COLUMNS:
            raise ValueError("{} must have columns {}".format(path, ",".join(ROW_COLUMNS)))
        for row in reader:
            case = row["case"]
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
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selection_plan = _resolve_selection_plan(args, corpus_root, output_csv)
    cases = [case.case_id for case in selection_plan.selected_cases]
    metrics = read_valid_metrics(args.valid_csv)
    if args.max_metrics:
        metrics = metrics[:args.max_metrics]
    if not metrics:
        raise ValueError("no valid metrics selected")
    sweep_config_id = _sweep_config_id(args, selection_plan, metrics)

    completed = (
        _read_rows(
            output_csv,
            cases,
            metrics,
            selection_plan.plan_id,
            sweep_config_id,
            require_file=True,
        )
        if args.resume
        else set()
    )
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
                    measurement = dict(batch_rows[metric])
                    for column in ENVIRONMENT_COLUMNS:
                        measurement.setdefault(
                            column, "not_collected" if column == "environment_status" else ""
                        )
                    measurement["order_index"] = str(done + 1)
                    row = {
                        "sweep_plan_id": selection_plan.plan_id,
                        "sweep_config_id": sweep_config_id,
                        "case": case,
                        "metric": metric,
                        **measurement,
                    }
                    writer.writerow(row)
                    completed.add((case, metric))
                    done += 1
                    print("[{}/{}] {} {} {}".format(done, total, case, metric,
                                                     row["status"]), flush=True)
                stream.flush()
    payload = build_result_json_from_csv(
        cases,
        metrics,
        output_csv,
        selection_plan.plan_id,
        sweep_config_id,
    )
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
    parser.add_argument("--max-metrics", type=int)
    parser.add_argument(
        "--case-selection",
        choices=(CONDITION_BALANCED_POLICY, ALL_CASES_POLICY),
        default=CONDITION_BALANCED_POLICY,
        help="select diverse conditions per op_type (default) or scan all cases",
    )
    parser.add_argument(
        "--conditions-per-op-type",
        type=int,
        default=5,
        help="minimum distinct workload conditions selected per op_type (default: 5)",
    )
    parser.add_argument(
        "--target-op-type",
        action="append",
        help="limit the plan to an exact op_type; repeat for multiple types",
    )
    parser.add_argument("--selection-plan-input")
    parser.add_argument("--selection-plan-output")
    parser.add_argument(
        "--device-fingerprint",
        default="unspecified",
        help="stable device/driver/CUDA identity included in resume validation",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
