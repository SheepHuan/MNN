#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MNN_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = MNN_ROOT / ".cache/latency_budget_20260625"

sys.path.insert(0, str(SCRIPT_DIR))
import run_pic_prefill_latency_sweep as sweep  # noqa: E402


DEVICE_ALIASES = {
    "orangepi5plus": "orangepi",
    "orangepi-mali": "orangepi",
    "rhinopi": "rhino",
    "rhino-adreno": "rhino",
}


def canonical_device(name: str) -> str:
    key = str(name).strip().lower()
    return DEVICE_ALIASES.get(key, key)


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def model_key_by_name() -> dict[str, str]:
    return {str(info["name"]): key for key, info in sweep.MODELS.items()}


def selected_model_keys(text: str) -> set[str] | None:
    if not str(text or "").strip():
        return None
    by_name = model_key_by_name()
    out: set[str] = set()
    for item in parse_csv_list(text):
        if item in sweep.MODELS:
            out.add(item)
        elif item in by_name:
            out.add(by_name[item])
        else:
            raise SystemExit(f"unknown model {item}; expected one of: {', '.join(sorted(sweep.MODELS))}")
    return out


def read_benchmark_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    devices = {canonical_device(item) for item in parse_csv_list(args.devices)}
    unknown_devices = sorted(devices - set(sweep.DEVICES))
    if unknown_devices:
        raise SystemExit(f"unknown devices: {', '.join(unknown_devices)}")
    model_filter = selected_model_keys(args.models)
    name_to_key = model_key_by_name()
    frequency_profile = str(args.frequency_profile)
    max_context = int(args.max_context or 0)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []

    with Path(args.benchmark_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            device = canonical_device(str(row.get("device", "")))
            if device not in devices:
                continue
            model_name = str(row.get("model", ""))
            model_key = name_to_key.get(model_name)
            if not model_key:
                excluded.append({**row, "exclude_reason": "model not known to runner"})
                continue
            if model_key in sweep.POWER_EXCLUDED_MODEL_KEYS:
                excluded.append({**row, "exclude_reason": "model excluded from power capture"})
                continue
            if model_filter is not None and model_key not in model_filter:
                continue
            mode = str(row.get("mode", ""))
            if mode not in sweep.MODE_CHOICES:
                excluded.append({**row, "exclude_reason": "mode disabled or unsupported by runner"})
                continue
            try:
                context = int(row.get("context_tokens", ""))
            except ValueError:
                excluded.append({**row, "exclude_reason": "invalid context_tokens"})
                continue
            if max_context > 0 and context > max_context:
                excluded.append({**row, "exclude_reason": f"context_tokens > max_context {max_context}"})
                continue
            high_risk = sweep.high_risk_context_error(device, frequency_profile, context)
            if high_risk and not args.allow_high_risk_contexts:
                excluded.append({**row, "exclude_reason": high_risk})
                continue
            group = groups.setdefault(
                (device, model_key),
                {
                    "device": device,
                    "model_key": model_key,
                    "model": model_name,
                    "contexts": set(),
                    "modes": set(),
                    "ratios": set(),
                    "row_count": 0,
                },
            )
            group["contexts"].add(context)
            group["modes"].add(mode)
            if mode in sweep.RATIO_MODES:
                group["ratios"].add(float(row.get("budget", "")))
            group["row_count"] += 1

    jobs: list[dict[str, Any]] = []
    mode_order = {mode: index for index, mode in enumerate(sweep.MODE_CHOICES)}
    for group in groups.values():
        jobs.append(
            {
                **group,
                "contexts": sorted(group["contexts"]),
                "modes": sorted(group["modes"], key=lambda item: mode_order.get(item, 999)),
                "ratios": sorted(group["ratios"]),
            }
        )
    jobs.sort(key=lambda item: (item["device"], item["model_key"]))
    return jobs, excluded


def build_job_command(args: argparse.Namespace, job: dict[str, Any], run_prefix: str) -> list[str]:
    device = str(job["device"])
    wrapper = SCRIPT_DIR / f"run_pic_prefill_latency_sweep_{device}.sh"
    if not wrapper.exists():
        raise SystemExit(f"missing wrapper for {device}: {wrapper}")
    run_id = f"{run_prefix}_{device}_{job['model_key']}"
    cmd = [
        "bash",
        str(wrapper),
        "--model-key",
        str(job["model_key"]),
        "--benchmark-csv",
        str(Path(args.benchmark_csv).resolve()),
        "--only-benchmark-csv-rows",
        "--contexts",
        ",".join(str(item) for item in job["contexts"]),
        "--modes",
        ",".join(str(item) for item in job["modes"]),
        "--run-id",
        run_id,
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--frequency-profile",
        str(args.frequency_profile),
        "--power-capture",
        "--power-rep",
        str(int(args.power_rep)),
        "--power-warmup-sec",
        str(float(args.power_warmup_sec)),
        "--power-cooldown-sec",
        str(float(args.power_cooldown_sec)),
        "--remote-memory-limit-percent",
        str(int(args.remote_memory_limit_percent)),
    ]
    if job["ratios"]:
        cmd.extend(["--ratios", ",".join(f"{float(item):.2f}" for item in job["ratios"])])
    if args.power_output_root:
        cmd.extend(["--power-output-root", str(Path(args.power_output_root).resolve())])
    if args.allow_high_risk_contexts:
        cmd.append("--allow-high-risk-contexts")
    if args.no_warm:
        cmd.append("--no-warm")
    if args.restart_server_each_spec:
        cmd.append("--restart-server-each-spec")
    if args.force_cache_build:
        cmd.append("--force-cache-build")
    for item in args.extra_runner_arg:
        cmd.append(item)
    return cmd


def jsonable_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in job.items()
    }


def write_plan_files(
    run_dir: Path,
    jobs: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    commands: dict[tuple[str, str], list[str]],
    args: argparse.Namespace,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "benchmark_csv": str(Path(args.benchmark_csv).resolve()),
        "devices": [canonical_device(item) for item in parse_csv_list(args.devices)],
        "models": str(args.models or ""),
        "frequency_profile": str(args.frequency_profile),
        "max_context": int(args.max_context or 0),
        "power_rep": int(args.power_rep),
        "power_warmup_sec": float(args.power_warmup_sec),
        "power_cooldown_sec": float(args.power_cooldown_sec),
        "allow_high_risk_contexts": bool(args.allow_high_risk_contexts),
        "jobs": [jsonable_job(job) for job in jobs],
        "excluded_count": len(excluded),
    }
    (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    with (run_dir / "commands.sh").open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for job in jobs:
            handle.write(shlex.join(commands[(job["device"], job["model_key"])]) + "\n")
    if excluded:
        fieldnames = sorted({key for row in excluded for key in row})
        with (run_dir / "excluded_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(excluded)


def append_event(run_dir: Path, event: dict[str, Any]) -> None:
    event = dict(event)
    event["time_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_jobs(
    run_dir: Path,
    jobs: list[dict[str, Any]],
    commands: dict[tuple[str, str], list[str]],
    *,
    sequential: bool,
) -> int:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending_by_device: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        pending_by_device.setdefault(str(job["device"]), []).append(job)
    active: dict[str, tuple[dict[str, Any], subprocess.Popen[str], Any]] = {}
    failures = 0
    try:
        while pending_by_device or active:
            can_start_devices = sorted(pending_by_device)
            if sequential and active:
                can_start_devices = []
            for device in can_start_devices:
                if device in active:
                    continue
                queue = pending_by_device.get(device, [])
                if not queue:
                    pending_by_device.pop(device, None)
                    continue
                job = queue.pop(0)
                if not queue:
                    pending_by_device.pop(device, None)
                log_path = logs_dir / f"{device}_{job['model_key']}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                cmd = commands[(device, job["model_key"])]
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(MNN_ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                active[device] = (job, proc, log_handle)
                append_event(
                    run_dir,
                    {
                        "event": "start",
                        "device": device,
                        "model_key": job["model_key"],
                        "pid": proc.pid,
                        "log": str(log_path),
                        "command": cmd,
                    },
                )
            time.sleep(5)
            for device, (job, proc, log_handle) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log_handle.close()
                if code != 0:
                    failures += 1
                append_event(
                    run_dir,
                    {
                        "event": "finish",
                        "device": device,
                        "model_key": job["model_key"],
                        "pid": proc.pid,
                        "returncode": code,
                    },
                )
                del active[device]
    except KeyboardInterrupt:
        append_event(run_dir, {"event": "interrupt", "active": list(active)})
        for _device, (_job, proc, log_handle) in active.items():
            proc.terminate()
            log_handle.close()
        raise
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", default=str(MNN_ROOT / "benchmark.csv"))
    parser.add_argument("--devices", default="orangepi,rhino")
    parser.add_argument("--models", default="")
    parser.add_argument("--run-prefix", default="pic_power_from_benchmark_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--power-output-root", default="")
    parser.add_argument("--power-rep", type=int, default=int(os.environ.get("MNN_POWER_REP", "3")))
    parser.add_argument("--power-warmup-sec", type=float, default=float(os.environ.get("MNN_POWER_WARMUP_SEC", "5")))
    parser.add_argument("--power-cooldown-sec", type=float, default=float(os.environ.get("MNN_POWER_COOLDOWN_SEC", "5")))
    parser.add_argument("--frequency-profile", default="max")
    parser.add_argument("--max-context", type=int, default=0, help="Only schedule benchmark.csv rows with context_tokens <= this value. 0 means no limit.")
    parser.add_argument("--remote-memory-limit-percent", type=int, default=int(os.environ.get("PIC_SWEEP_REMOTE_MEMORY_LIMIT_PERCENT", "95")))
    parser.add_argument("--allow-high-risk-contexts", action="store_true")
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--restart-server-each-spec", action="store_true")
    parser.add_argument("--force-cache-build", action="store_true")
    parser.add_argument("--sequential", action="store_true", help="Run one device/model process at a time. Default runs one process per device concurrently.")
    parser.add_argument("--execute", action="store_true", help="Start the planned benchmark processes. Without this, only write the plan.")
    parser.add_argument("--extra-runner-arg", action="append", default=[])
    args = parser.parse_args()
    if args.power_warmup_sec < 0:
        raise SystemExit("--power-warmup-sec must be non-negative")
    if args.power_cooldown_sec < 0:
        raise SystemExit("--power-cooldown-sec must be non-negative")

    jobs, excluded = read_benchmark_jobs(args)
    run_dir = Path(args.output_dir) / args.run_prefix / "power_from_benchmark"
    commands = {(job["device"], job["model_key"]): build_job_command(args, job, args.run_prefix) for job in jobs}
    write_plan_files(run_dir, jobs, excluded, commands, args)
    print(run_dir)
    print(f"planned_jobs={len(jobs)} excluded_rows={len(excluded)}")
    for job in jobs:
        print(
            f"{job['device']} {job['model_key']} rows={job['row_count']} "
            f"contexts={','.join(map(str, job['contexts']))} modes={','.join(job['modes'])}"
        )
    if not args.execute:
        print(f"dry-run only; commands written to {run_dir / 'commands.sh'}")
        return 0
    if not jobs:
        print("no jobs to run")
        return 0
    return run_jobs(run_dir, jobs, commands, sequential=bool(args.sequential))


if __name__ == "__main__":
    raise SystemExit(main())
