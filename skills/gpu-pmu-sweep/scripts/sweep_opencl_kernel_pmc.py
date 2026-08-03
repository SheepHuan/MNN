#!/usr/bin/env python3
"""Sweep every kernel corpus case against every valid OpenCL PMU metric.

Unlike sweep_cuda_kernel_pmc.py (which groups metrics into one CUPTI Session
per case), OpenCL on Adreno/Mali accepts exactly one metric per benchmark
invocation. This script therefore iterates (case, metric) and writes a row
per pair. The detailed CSV is the resume ledger; the compact JSON contains
only VALID PMC values: {case: {"pmc": {metric: value}}}.

Usage:
    python3 sweep_opencl_kernel_pmc.py \
        --valid-csv opencl_pmc_valid.csv \
        --corpus-root ../replay_benchmark/kernel_corpus \
        --binary ./replay_benchmark.out \
        --workdir . \
        --lib-dir .:source/backend/opencl:source/backend/vulkan \
        --output-csv rows.csv --output-json pmc.json --resume
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROW_COLUMNS = [
    "collection_session_id",  # UUID shared by all rows in this sweep run
    "manifest_sha256",        # sha256 of operator_cases.json (first 16 chars)
    "device_fingerprint",     # GPU device name (e.g. "QUALCOMM Adreno(TM) 740")
    "case",
    "metric",
    "status",       # VALID / NOT_FOUND / COMMAND_FAILED / TIMEOUT
    "value",        # numeric PMC value (workload delta; empty when not VALID)
    "control_value",  # baseline PMC value from control case (same metric)
    "delta",        # value - control_value (signed; can be negative for noisy counters)
    "pmu_status",   # sampled / start_failed / disabled / unavailable / ...
    "returncode",
    "error",
    "framework",    # mnn / ncnn
    "backend",      # opencl / vulkan
    "tag",          # 1.2.0 / 3.6.0 / 20190611 / 20260526
    "variant",
]
VALID_STATUSES = {"VALID"}

# Default control case: a small, fast kernel that's known to execute successfully
# on all backends. Its PMC reading represents the "session open/close + minimal
# kernel dispatch" baseline. For corpus kernels, the delta (workload - control)
# isolates the PMC increment attributable to the kernel's actual work.
DEFAULT_CONTROL_CASE = "argmax_buf_fp32_smoke"


def _sha256_of_file(path, chunk_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _detect_device_fingerprint(binary, workdir, lib_dir):
    """Ask replay_benchmark.out for the OpenCL device name; fall back to 'unknown'."""
    cmd = [binary, "--opencl-pmu-list-events"]
    env = os.environ.copy()
    if lib_dir:
        env["LD_LIBRARY_PATH"] = lib_dir
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                    cwd=workdir, timeout=30, check=False)
        if completed.returncode != 0:
            return "unknown"
        payload = _strip_banner(completed.stdout)
        doc = json.loads(payload)
        return doc.get("pmu_product_name") or doc.get("device") or "unknown"
    except Exception:
        return "unknown"


def _is_nonnegative_finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
    )


def _is_finite_number(value):
    """Finite numeric (int or float, not bool, not NaN/inf). Allows negative deltas."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


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


def load_cases(corpus_root, backend_filter=None, framework_filter=None):
    cases_path = Path(corpus_root) / "operator_cases.json"
    with cases_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    cases = []
    for item in document.get("cases", []):
        backend = item.get("backend", "")
        if backend_filter and backend != backend_filter:
            continue
        framework = item.get("framework", "")
        if framework_filter and framework != framework_filter:
            continue
        cases.append({
            "name": item.get("name", ""),
            "backend": backend,
            "framework": framework,
            "tag": item.get("tag", ""),
            "variant": item.get("variant", ""),
            "op_type": item.get("op_type", ""),
        })
    return cases


def _read_completed_pairs(output_csv):
    """Read existing rows CSV; return dict {(case, tag, metric): row} for resume."""
    completed = {}
    if not output_csv.exists():
        return completed
    with output_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = (row.get("case", ""), row.get("tag", ""), row.get("metric", ""))
            completed[key] = row
    return completed


def _write_json(pmc_json_path, valid_data, fingerprint=None):
    tmp = pmc_json_path.with_suffix(".tmp")
    document = dict(valid_data)
    if fingerprint:
        # Wrap cases inside a top-level object that also carries session metadata.
        # Consumers that only care about PMC values can read ["cases"] directly.
        document = {
            "collection_session_id": fingerprint.get("collection_session_id", ""),
            "manifest_sha256": fingerprint.get("manifest_sha256", ""),
            "device_fingerprint": fingerprint.get("device_fingerprint", ""),
            "cases": valid_data,
        }
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    tmp.replace(pmc_json_path)


def _append_row(csv_file, csv_writer, row):
    csv_writer.writerow(row)
    csv_file.flush()


def _strip_banner(text):
    """Strip non-JSON banner (e.g. Rockchip "arm_release_ver: ...")."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return ""
    return text[start:end + 1]


def run_one(binary, workdir, lib_dir, corpus_root, case_name, metric, runs, timeout_s, sudo,
            keep_json_dir=None):
    """Run one (case, metric) benchmark; return dict with status/value/pmu_status/error.

    If keep_json_dir is given, the benchmark's raw JSON output is preserved at
    <keep_json_dir>/<case>__<metric>.json for audit; otherwise it's deleted.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        output_path = tmp.name
    try:
        cmd = []
        if sudo:
            cmd.append("sudo")
        cmd.extend([
            binary,
            "--kernel-corpus-bench",
            "--kernel-corpus-root", corpus_root,
            "--kernel-corpus-case", case_name,
            "--kernel-corpus-runs", str(runs),
            "--perf-counter-events", metric,
            "--perf-counter-output", output_path,
        ])
        env = os.environ.copy()
        if lib_dir:
            env["LD_LIBRARY_PATH"] = lib_dir
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, env=env,
                cwd=workdir, timeout=timeout_s, check=False,
            )
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "value": "", "pmu_status": "",
                    "returncode": -1, "error": "timeout"}
        if rc != 0 or not Path(output_path).exists():
            return {"status": "COMMAND_FAILED", "value": "", "pmu_status": "",
                    "returncode": rc, "error": completed.stderr[-200:] if completed.stderr else ""}
        # Preserve raw JSON before any further processing.
        if keep_json_dir:
            keep_dir = Path(keep_json_dir)
            keep_dir.mkdir(parents=True, exist_ok=True)
            safe_case = case_name.replace("/", "_").replace(":", "_")
            safe_metric = metric.replace("/", "_").replace(":", "_")
            keep_path = keep_dir / "{}__{}.json".format(safe_case, safe_metric)
            try:
                keep_path.write_text(Path(output_path).read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                pass
        try:
            payload = _strip_banner(Path(output_path).read_text(encoding="utf-8"))
            document = json.loads(payload)
        except (ValueError, OSError) as exc:
            return {"status": "COMMAND_FAILED", "value": "", "pmu_status": "",
                    "returncode": rc, "error": "json parse: {}".format(exc)}
        cases = document.get("cases", [])
        if not cases:
            return {"status": "COMMAND_FAILED", "value": "", "pmu_status": "",
                    "returncode": rc, "error": "no case in report"}
        report = cases[0]
        pmu_status = report.get("pmu_status", "")
        pmu_metrics = report.get("pmu_metrics", {}) or {}
        value = pmu_metrics.get(metric)
        # VALID requires a numeric value. 0 is a valid counter reading.
        if pmu_status == "sampled" and _is_nonnegative_finite_number(value):
            return {"status": "VALID", "value": value, "pmu_status": pmu_status,
                    "returncode": rc, "error": ""}
        return {"status": "NOT_FOUND", "value": "", "pmu_status": pmu_status,
                "returncode": rc, "error": report.get("error", "")}
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-csv", required=True,
                        help="CSV with a 'metric' column listing valid PMU metrics")
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--lib-dir")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60,
                        help="per-benchmark timeout in seconds (default: 60)")
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--backend", choices=("opencl", "vulkan"),
                        help="limit to one backend (default: all corpus cases)")
    parser.add_argument("--framework", choices=("mnn", "ncnn"),
                        help="limit to one framework")
    parser.add_argument("--case-filter",
                        help="only run cases whose name contains this substring")
    parser.add_argument("--max-cases", type=int,
                        help="limit number of cases (for smoke tests)")
    parser.add_argument("--max-metrics", type=int,
                        help="limit number of metrics (for smoke tests)")
    parser.add_argument("--control-case", default=DEFAULT_CONTROL_CASE,
                        help="case used as the empty-control baseline (default: %(default)s)")
    parser.add_argument("--no-control", action="store_true",
                        help="skip control measurement; emit raw workload values only")
    parser.add_argument("--keep-json-dir",
                        help="preserve the raw benchmark JSON of every (case, metric) pair "
                             "into this directory for later audit")
    args = parser.parse_args(argv)

    metrics = read_valid_metrics(args.valid_csv)
    if args.max_metrics:
        metrics = metrics[:args.max_metrics]
    cases = load_cases(args.corpus_root, args.backend, args.framework)
    if args.case_filter:
        cases = [c for c in cases if args.case_filter in c["name"]]
    if args.max_cases:
        cases = cases[:args.max_cases]

    if not metrics:
        print("no valid metrics to sweep", file=sys.stderr)
        return 1
    if not cases:
        print("no cases to sweep", file=sys.stderr)
        return 1

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Fingerprint: ties every row to this specific (manifest, device) pair.
    manifest_path = Path(args.corpus_root) / "operator_cases.json"
    manifest_sha = _sha256_of_file(manifest_path)[:16] if manifest_path.exists() else "unknown"
    device_fp = _detect_device_fingerprint(args.binary, args.workdir, args.lib_dir)
    # Resume: keep the original session id; new run: mint a fresh one.
    session_id = None
    completed = _read_completed_pairs(output_csv) if args.resume else {}
    if args.resume and completed:
        for row in completed.values():
            session_id = row.get("collection_session_id")
            if session_id:
                break
    if not session_id:
        session_id = str(uuid.uuid4())
    print("collection_session_id={} manifest_sha256={} device={}".format(
        session_id, manifest_sha, device_fp), flush=True)
    fingerprint = {
        "collection_session_id": session_id,
        "manifest_sha256": manifest_sha,
        "device_fingerprint": device_fp,
    }
    valid_data = {}
    # Restore previously VALID values into the in-memory JSON mirror.
    if args.resume:
        for (case_name, tag, metric), row in completed.items():
            if row.get("status") == "VALID":
                try:
                    value = json.loads(row.get("delta", "null"))
                except ValueError:
                    continue
                if _is_finite_number(value):
                    json_key = case_name if tag == "" else "{}@{}".format(case_name, tag)
                    valid_data.setdefault(json_key, {"pmc": {}})["pmc"][metric] = value

    write_header = not output_csv.exists() or not args.resume
    csv_file = output_csv.open("a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=ROW_COLUMNS, lineterminator="\n")
    if write_header:
        csv_writer.writeheader()
        csv_file.flush()

    # Phase 1: collect control baseline per metric (unless --no-control).
    control_values = {}
    if not args.no_control:
        print("collecting control baseline using case: {}".format(args.control_case), flush=True)
        for metric in metrics:
            row_data = run_one(args.binary, args.workdir, args.lib_dir, args.corpus_root,
                               args.control_case, metric, args.runs, args.timeout, args.sudo,
                               keep_json_dir=args.keep_json_dir)
            if row_data["status"] == "VALID" and _is_nonnegative_finite_number(row_data["value"]):
                control_values[metric] = row_data["value"]
            else:
                # Control failed for this metric — record as 0 so delta = value.
                control_values[metric] = 0
        print("  control baseline collected for {}/{} metrics".format(
            sum(1 for v in control_values.values() if v > 0), len(metrics)), flush=True)

    # Phase 2: sweep each (case, metric) and compute delta = value - control.
    total = len(cases) * len(metrics)
    done = 0
    t_start = time.time()
    # Fast-fail: if a case produces TIMEOUT for several consecutive metrics,
    # skip the remaining metrics for that case. This avoids spending hours
    # on kernels that hang on the target device (e.g. gemm_buf_fp32_smoke
    # on Mali-G610).
    TIMEOUT_SKIP_THRESHOLD = 3
    for case in cases:
        case_name = case["name"]
        case_tag = case["tag"]
        consecutive_timeouts = 0
        case_aborted = False
        for metric in metrics:
            key = (case_name, case_tag, metric)
            done += 1
            if case_aborted:
                # Already recorded during a previous run — don't duplicate rows.
                if key in completed:
                    continue
                # Emit a TIMEOUT row without re-running the benchmark.
                row = {
                    "collection_session_id": session_id,
                    "manifest_sha256": manifest_sha,
                    "device_fingerprint": device_fp,
                    "case": case_name,
                    "metric": metric,
                    "status": "TIMEOUT",
                    "value": "",
                    "control_value": json.dumps(control_values.get(metric, 0), allow_nan=False),
                    "delta": "",
                    "pmu_status": "",
                    "returncode": -1,
                    "error": "case aborted after {} consecutive timeouts".format(TIMEOUT_SKIP_THRESHOLD),
                    "framework": case["framework"],
                    "backend": case["backend"],
                    "tag": case["tag"],
                    "variant": case["variant"],
                }
                _append_row(csv_file, csv_writer, row)
                continue
            if key in completed and completed[key].get("status") in VALID_STATUSES | {"NOT_FOUND", "TIMEOUT"}:
                if completed[key].get("status") == "TIMEOUT":
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= TIMEOUT_SKIP_THRESHOLD:
                        case_aborted = True
                else:
                    consecutive_timeouts = 0
                continue
            row_data = run_one(args.binary, args.workdir, args.lib_dir, args.corpus_root,
                               case_name, metric, args.runs, args.timeout, args.sudo,
                               keep_json_dir=args.keep_json_dir)
            if row_data["status"] == "TIMEOUT":
                consecutive_timeouts += 1
                if consecutive_timeouts >= TIMEOUT_SKIP_THRESHOLD:
                    case_aborted = True
            else:
                consecutive_timeouts = 0
            control_value = control_values.get(metric, 0)
            value = row_data.get("value", "")
            delta = ""
            if _is_nonnegative_finite_number(value):
                delta = value - control_value
            row = {
                "collection_session_id": session_id,
                "manifest_sha256": manifest_sha,
                "device_fingerprint": device_fp,
                "case": case_name,
                "metric": metric,
                "status": row_data["status"],
                "value": json.dumps(value, allow_nan=False) if value != "" else "",
                "control_value": json.dumps(control_value, allow_nan=False),
                "delta": json.dumps(delta, allow_nan=False) if delta != "" else "",
                "pmu_status": row_data.get("pmu_status", ""),
                "returncode": row_data.get("returncode", ""),
                "error": (row_data.get("error") or "").replace("\n", " ")[:200],
                "framework": case["framework"],
                "backend": case["backend"],
                "tag": case["tag"],
                "variant": case["variant"],
            }
            _append_row(csv_file, csv_writer, row)
            if row_data["status"] == "VALID":
                # Store delta (workload - control) in JSON, not raw value.
                # Delta can be negative for noisy counters; preserve the sign.
                # JSON key uses case@tag to disambiguate same case name under
                # multiple tags (e.g. binary_buf_fp32_smoke @ 1.2.0 vs 3.6.0).
                json_key = case_name if case_tag == "" else "{}@{}".format(case_name, case_tag)
                valid_data.setdefault(json_key, {"pmc": {}})["pmc"][metric] = delta
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (total - done) / rate if rate > 0 else 0
                print("[{}/{}] case={} metric={} status={} rate={:.1f}/s eta={:.0f}s".format(
                    done, total, case_name, metric[:50], row_data["status"], rate, eta_s), flush=True)
                _write_json(output_json, valid_data, fingerprint)
    csv_file.close()
    _write_json(output_json, valid_data, fingerprint)
    valid_count = sum(len(v["pmc"]) for v in valid_data.values())
    print("done: {} case-metric pairs, {} VALID deltas in {}".format(
        total, valid_count, output_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
