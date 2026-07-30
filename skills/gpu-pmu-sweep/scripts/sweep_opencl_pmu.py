#!/usr/bin/env python3
"""Run one GPU PMU event per benchmark process and emit normalized CSV.

Synthetic cases use the OpenCL kernels. Complete MNN model sessions can use
either the OpenCL or Vulkan backend via ``--backend``.
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from urllib.parse import quote
from pathlib import Path


DEVICES = {
    "rhinopi": (
        f"{os.environ.get('RHINO_PI_USER', 'root')}@{os.environ.get('RHINO_PI_HOST', '192.168.101.227')}",
        f"{os.environ.get('RHINO_PI_WORKSPACE', '/mnt/nvme/workspace')}/replay-benchmark",
    ),
    "orangepi": (
        f"{os.environ.get('ORANGE_PI_USER', 'root')}@{os.environ.get('ORANGE_PI_HOST', '192.168.101.113')}",
        os.environ.get('ORANGE_PI_WORKSPACE', '/mnt/ssd/workspace'),
    ),
}
MODEL_BACKENDS = {"opencl": 3, "vulkan": 7}
CSV_COLUMNS = ["case_name", "case_args", "case_runs", "pmu_metric_name", "delta_metric"]
MODEL_CSV_COLUMNS = [
    "case_name", "model_path", "case_args", "case_runs", "pmu_metric_name", "delta_metric", "status", "error",
]
SUMMARY_COLUMNS = ["pmu_metric_name", "valid", "valid_cases"]
DEFAULT_CASES = [
    "buffer_reuse_small", "buffer_reuse_l2", "buffer_stride", "buffer_pchase", "buffer_vec4",
    "buffer_stream_large", "buffer_write", "buffer_int32", "buffer_fp32", "fp32_throughput",
    "buffer_fp16", "fp16_throughput", "constant_memory", "constant_bandwidth", "local_memory",
    "local_bandwidth", "local_barrier", "global_atomic", "atomic_contended", "atomic_distributed",
    "image_reuse", "image_write", "texture_nearest", "texture_linear", "image_stride_x", "image_stride_y",
]
DEFAULT_EVENTS = ["all"]


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def model_forward_for_backend(backend):
    try:
        return MODEL_BACKENDS[str(backend).strip().lower()]
    except KeyError as error:
        raise ValueError("model backend must be opencl or vulkan") from error


def raw_filename_component(value):
    """Encode event/case names so valid PMU names cannot escape raw_dir."""
    return quote(str(value), safe="._-")


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


def extract_model_delta(report, model_name, metric_name):
    """Extract the signed delta from a complete MNN Session workload report."""
    return extract_delta(report, model_name, metric_name)


def discover_models(roots):
    paths = set()
    for root in roots:
        path = Path(root)
        if path.is_file() and path.suffix == ".mnn":
            paths.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*.mnn"):
                if not candidate.is_file():
                    continue
                package_dir = candidate.parent
                if (package_dir / "llm_config.json").is_file():
                    if candidate.name != "llm.mnn" and not candidate.with_name(candidate.name + ".weight").is_file():
                        continue
                paths.add(candidate)
    return sorted(paths, key=lambda item: (item.name, str(item)))


def summarize_rows(rows, events, cases):
    changed_cases = {event: [] for event in events}
    for row in rows:
        delta = row.get("delta_metric", "")
        if delta == "":
            continue
        try:
            changed = int(delta) != 0
        except ValueError:
            changed = False
        if changed and row["case_name"] not in changed_cases[row["pmu_metric_name"]]:
            changed_cases[row["pmu_metric_name"]].append(row["case_name"])
    return [
        {
            "pmu_metric_name": event,
            "valid": "true" if changed_cases[event] else "false",
            "valid_cases": ";".join(case for case in cases if case in changed_cases[event]),
        }
        for event in events
    ]


def write_summary_csv(path, rows, events, cases):
    summary_rows = summarize_rows(rows, events, cases)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


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


def build_model_benchmark_argv(binary, model_path, event_name, args):
    if "," in event_name or not event_name.strip():
        raise ValueError("each PMU invocation must contain exactly one event")
    argv = [
        binary,
        "--model-pmu-bench",
        "--model",
        str(model_path),
        "--model-pmu-workload-runs",
        str(args.workload_runs),
        "--model-pmu-warmup-runs",
        str(args.warmup_runs),
        "--model-pmu-control-runs",
        str(getattr(args, "control_runs", 1)),
    ]
    model_forward = getattr(args, "model_forward", None)
    if model_forward is None and getattr(args, "backend", None) is not None:
        model_forward = model_forward_for_backend(args.backend)
    if model_forward is not None:
        argv.extend(["--model-pmu-forward", str(model_forward)])
    argv.extend([
        "--perf-counter-events",
        event_name,
        "--perf-counter-output",
        "/tmp/mnn-model-pmu.json",
    ])
    if getattr(args, "precision", None) is not None:
        argv.extend(["--model-pmu-precision", str(args.precision)])
    return argv


def build_discovery_argv(binary):
    return [binary, "--opencl-pmu-list-events"]


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


def discover_device_events(args, runner=run_process):
    """Discover only the events exposed by the selected local/remote device."""
    if args.device is None:
        env = os.environ.copy()
        if args.lib_dir:
            env["LD_LIBRARY_PATH"] = args.lib_dir
        report = runner(build_discovery_argv(args.binary), env=env, timeout=args.timeout)
    else:
        host, default_root = DEVICES[args.device]
        remote_root = args.remote_root or default_root
        binary = remote_root + "/bin/replay_benchmark.out"
        command = "cd {} && env LD_LIBRARY_PATH={} {}".format(
            shlex.quote(remote_root),
            shlex.quote(remote_root + "/lib"),
            " ".join(shlex.quote(item) for item in build_discovery_argv(binary)),
        )
        report = runner(["ssh", "-o", "BatchMode=yes", host, command], timeout=args.timeout)
    events = report.get("pmu_events")
    if report.get("pmu_status") != "available" or not isinstance(events, list):
        raise RuntimeError(report.get("pmu_error") or "device did not report PMU events")
    result = []
    for event in events:
        if isinstance(event, str) and event and event not in result:
            result.append(event)
    if not result:
        raise RuntimeError("device reported no supported PMU events")
    return result


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


def execute_event_batch(binary, case_names, event_name, args):
    """Run one PMU event while the benchmark gives each selected case its own session."""
    argv = build_benchmark_argv(binary, case_names, event_name, args)
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


def execute_model(binary, model_path, event_name, args):
    argv = build_model_benchmark_argv(binary, model_path, event_name, args)
    if args.device is None:
        env = os.environ.copy()
        if args.lib_dir:
            env["LD_LIBRARY_PATH"] = args.lib_dir
        return run_process(argv, env, args.timeout)

    host, default_root = DEVICES[args.device]
    remote_root = args.remote_root or default_root
    remote_model = str(model_path)
    local_path = Path(model_path)
    is_llm_package = (local_path.parent / "llm_config.json").is_file()
    if is_llm_package:
        model_root = getattr(args, "remote_llm_root", None) or getattr(args, "remote_model_root", None)
    else:
        model_root = getattr(args, "remote_dnn_root", None) or getattr(args, "remote_model_root", None)
    if model_root:
        if is_llm_package:
            remote_model = str(Path(model_root) / local_path.parent.name / local_path.name)
        else:
            remote_model = str(Path(model_root) / local_path.name)
    remote_argv = build_model_benchmark_argv(remote_root + "/bin/replay_benchmark.out", remote_model, event_name, args)
    command = "cd {} && env LD_LIBRARY_PATH={} {}".format(
        shlex.quote(remote_root),
        shlex.quote(remote_root + "/lib"),
        " ".join(shlex.quote(item) for item in remote_argv),
    )
    return run_process(["ssh", "-o", "BatchMode=yes", host, command], timeout=args.timeout)


def _model_status(report, model_name):
    for case in report.get("cases", []):
        if case.get("name") == model_name:
            return case.get("status", ""), case.get("error", "")
    return "error", report.get("error", "model report did not contain the requested model")


def run_model_sweep(args, model_executor=execute_model, discover=discover_device_events):
    if getattr(args, "models", ""):
        models = [Path(item) for item in parse_csv_list(args.models)]
    else:
        models = discover_models(parse_csv_list(args.model_roots))
    event_text = getattr(args, "events", "all")
    events = discover(args) if event_text.strip().lower() == "all" else parse_csv_list(event_text)
    if not models or not events:
        raise ValueError("models and events must not be empty")
    precision_text = getattr(args, "precisions", "")
    precisions = parse_csv_list(precision_text) if precision_text else [None]
    if any(value is not None and (not value.lstrip("-").isdigit() or int(value) < 0 or int(value) > 3)
           for value in precisions):
        raise ValueError("precisions must be comma-separated values in the range 0..3")
    if args.measurements < 1 or args.workload_runs < 1 or args.warmup_runs < 0:
        raise ValueError("measurements and workload-runs must be positive; warmup-runs cannot be negative")

    output = Path(args.model_output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.model_raw_dir) if args.model_raw_dir else output.parent / (output.stem + "-raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_path in models:
        model_name = model_path.name
        for precision in precisions:
            run_args = args
            if precision is not None:
                import copy
                run_args = copy.copy(args)
                run_args.precision = int(precision)
            case_name = model_name if precision is None else "{}@precision{}".format(model_name, precision)
            for event_name in events:
                for measurement in range(1, args.measurements + 1):
                    try:
                        report = model_executor(args.binary, str(model_path), event_name, run_args)
                    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                        print("{} / {} / measurement {}: {}".format(model_name, event_name, measurement, error), file=sys.stderr)
                        report = {"cases": [], "error": str(error)}
                    raw_path = raw_dir / "{}__{}__{:03d}.json".format(
                        raw_filename_component(case_name), raw_filename_component(event_name), measurement,
                    )
                    raw_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                    status, error = _model_status(report, model_name)
                    precision_arg = " --model-pmu-precision={}".format(precision) if precision is not None else ""
                    forward_arg = " --model-pmu-forward={}".format(
                        getattr(
                            run_args, "model_forward",
                            model_forward_for_backend(getattr(run_args, "backend", "opencl")),
                        )
                    )
                    rows.append({
                        "case_name": case_name,
                        "model_path": str(model_path),
                        "case_args": "--model-pmu-workload-runs={} --model-pmu-warmup-runs={} --model-pmu-control-runs={}{}{} --measurement={}".format(
                            args.workload_runs, args.warmup_runs, getattr(args, "control_runs", 1), forward_arg,
                            precision_arg, measurement,
                        ),
                        "case_runs": str(args.workload_runs),
                        "pmu_metric_name": event_name,
                        "delta_metric": extract_model_delta(report, model_name, event_name),
                        "status": status,
                        "error": error,
                    })
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MODEL_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_value = getattr(args, "model_summary_csv", None)
    summary = Path(summary_value) if summary_value else output.with_name(output.stem + "-metrics.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary_cases = [
        model.name if precision is None else "{}@precision{}".format(model.name, precision)
        for model in models for precision in precisions
    ]
    write_summary_csv(summary, rows, events, summary_cases)
    return rows


def run_sweep(args, executor=execute, discover=discover_device_events, batch_executor=execute_event_batch):
    cases = parse_csv_list(args.cases)
    event_text = getattr(args, "events", "all")
    events = discover(args) if event_text.strip().lower() == "all" else parse_csv_list(event_text)
    if not cases or not events:
        raise ValueError("cases and events must not be empty")
    if args.measurements < 1 or args.workload_runs < 1 or args.warmup_runs < 0:
        raise ValueError("measurements and workload-runs must be positive; warmup-runs cannot be negative")

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output.parent / (output.stem + "-raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if getattr(args, "batch_cases", False):
        case_selection = ",".join(cases)
        for event_name in events:
            for measurement in range(1, args.measurements + 1):
                try:
                    report = batch_executor(args.binary, case_selection, event_name, args)
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                    print("{} / measurement {}: {}".format(event_name, measurement, error), file=sys.stderr)
                    report = {"cases": [], "error": str(error)}
                raw_path = raw_dir / "batch__{}__{:03d}.json".format(
                    raw_filename_component(event_name), measurement
                )
                raw_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                for case_name in cases:
                    rows.append(
                        {
                            "case_name": case_name,
                            "case_args": make_case_args(args, measurement),
                            "case_runs": str(args.workload_runs),
                            "pmu_metric_name": event_name,
                            "delta_metric": extract_delta(report, case_name, event_name),
                        }
                    )
    else:
        for case_name in cases:
            for event_name in events:
                for measurement in range(1, args.measurements + 1):
                    try:
                        report = executor(args.binary, case_name, event_name, args)
                    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                        print("{} / {} / measurement {}: {}".format(case_name, event_name, measurement, error), file=sys.stderr)
                        report = {"cases": [], "error": str(error)}
                    raw_path = raw_dir / "{}__{}__{:03d}.json".format(
                        raw_filename_component(case_name), raw_filename_component(event_name), measurement
                    )
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
    summary_value = getattr(args, "summary_csv", None)
    summary = Path(summary_value) if summary_value else output.with_name(output.stem + "-metrics.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary, rows, events, cases)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="comma-separated case names")
    parser.add_argument("--events", default=",".join(DEFAULT_EVENTS), help="comma-separated events or 'all' for device discovery")
    parser.add_argument("--output-csv", "--output")
    parser.add_argument("--summary-csv", help="per-device metric summary; defaults to <output>-metrics.csv")
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
    parser.add_argument(
        "--batch-cases", dest="batch_cases", action="store_true", default=True,
        help="run selected cases in one process while keeping one PMU session per case (default)",
    )
    parser.add_argument(
        "--no-batch-cases", dest="batch_cases", action="store_false",
        help="launch a separate process for every case/event combination",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model-sweep", action="store_true", help="sweep complete MNN model Sessions")
    parser.add_argument("--models", help="comma-separated .mnn model paths")
    parser.add_argument(
        "--model-roots",
        default="benchmark/models,/home/yanghuan/code/eperf-devlop/prebuilts/models/DNNs,/home/yanghuan/code/eperf-devlop/models/bench_genais/mnn",
        help="comma-separated roots searched recursively for .mnn files",
    )
    parser.add_argument("--model-output-csv", default="model-all.csv")
    parser.add_argument("--model-summary-csv")
    parser.add_argument("--model-raw-dir")
    parser.add_argument("--remote-model-root")
    parser.add_argument("--remote-dnn-root", help="remote root for ordinary .mnn models")
    parser.add_argument("--remote-llm-root", help="remote root containing LLM package directories")
    parser.add_argument("--control-runs", type=int, default=1)
    parser.add_argument("--precisions", default="0,2", help="model runtime precision modes: 0 normal, 2 FP16, comma-separated")
    parser.add_argument(
        "--backend", choices=sorted(MODEL_BACKENDS), default="opencl",
        help="backend for complete model Session sweeps; synthetic cases remain OpenCL",
    )
    args = parser.parse_args(argv)
    args.model_forward = model_forward_for_backend(args.backend)
    if not args.model_sweep and not args.output_csv:
        parser.error("--output-csv/--output is required unless --model-sweep is used")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.model_sweep:
            run_model_sweep(args)
        else:
            run_sweep(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print("opencl-pmu-sweep: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
