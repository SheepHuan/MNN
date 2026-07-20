#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set


MNN_ROOT = Path(__file__).resolve().parents[4]
SWEEP_SCRIPT = MNN_ROOT / ".codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py"
TARGET_MODE = "fusionrag_online"
POWER_EXCLUDED_MODEL_KEYS = {"llama3.2-1b"}
MODEL_BY_DISPLAY = {
    "Llama3.2 1B": "llama3.2-1b",
    "Llama3.2 3B": "llama3.2-3b",
    "MiniCPM5-1B": "minicpm5-1b",
    "Qwen3-4B": "qwen3-4b",
    "Qwen3-8B": "qwen3-8b",
}
FIELDNAMES = [
    "device",
    "device_display",
    "model",
    "backend",
    "frequency_note",
    "context_tokens",
    "mode",
    "budget",
    "prefill_latency_s",
    "prefill_tps",
]


def load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("mnn_pic_prefill_sweep", SWEEP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load sweep script: {SWEEP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_csv_set(text: Optional[str]) -> Optional[Set[str]]:
    if text is None or not str(text).strip():
        return None
    return {item.strip() for item in str(text).split(",") if item.strip()}


def budget_key(value: Any) -> str:
    return "%.2f" % float(value)


def load_cacheblend_plan(
    csv_path: Path,
    devices: Optional[Set[str]],
    models: Optional[Set[str]],
    contexts: Optional[Set[str]],
    budgets: Optional[Set[str]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    plan: Dict[str, Dict[str, Dict[str, Any]]] = {}
    budget_norm = {budget_key(item) for item in budgets} if budgets is not None else None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") != "cacheblend":
                continue
            device = str(row.get("device", "")).strip()
            model_display = str(row.get("model", "")).strip()
            model_key = MODEL_BY_DISPLAY.get(model_display, "")
            if not device or not model_key:
                continue
            if devices is not None and device not in devices:
                continue
            if models is not None and model_key not in models and model_display not in models:
                continue
            context = int(row["context_tokens"])
            budget = budget_key(row["budget"])
            if contexts is not None and str(context) not in contexts:
                continue
            if budget_norm is not None and budget not in budget_norm:
                continue
            entry = plan.setdefault(model_key, {}).setdefault(
                device,
                {
                    "display_model": model_display,
                    "contexts": set(),
                    "budgets": set(),
                    "rows": [],
                },
            )
            entry["contexts"].add(context)
            entry["budgets"].add(budget)
            entry["rows"].append(dict(row))
    for model_devices in plan.values():
        for entry in model_devices.values():
            entry["contexts"] = sorted(entry["contexts"])
            entry["budgets"] = sorted(entry["budgets"], key=float)
    return plan


def seen_fusion_rows(paths: Sequence[Path], model_name: str, device_name: str) -> Set[tuple[int, str]]:
    seen: Set[tuple[int, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("model") != model_name or row.get("device") != device_name:
                    continue
                if row.get("mode") != TARGET_MODE:
                    continue
                try:
                    latency = float(row.get("prefill_latency_s", "") or 0.0)
                except Exception:
                    latency = 0.0
                if latency <= 0.0:
                    continue
                try:
                    context = int(row["context_tokens"])
                except Exception:
                    continue
                seen.add((context, budget_key(row["budget"])))
    return seen


def missing_rows_by_context(
    benchmark_csv: Path,
    summary_csv: Path,
    model_name: str,
    device_name: str,
    contexts: Sequence[int],
    budgets: Sequence[str],
) -> Dict[int, List[Dict[str, str]]]:
    seen = seen_fusion_rows([benchmark_csv, summary_csv], model_name, device_name)
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for context in contexts:
        for budget in budgets:
            key = (int(context), budget_key(budget))
            if key in seen:
                continue
            grouped.setdefault(int(context), []).append(
                {
                    "context_tokens": str(int(context)),
                    "mode": TARGET_MODE,
                    "budget": budget_key(budget),
                }
            )
    return grouped


def patch_sweep_module(sweep: Any) -> None:
    sweep.RATIO_MODES.add(TARGET_MODE)
    if TARGET_MODE not in sweep.MODE_CHOICES:
        sweep.MODE_CHOICES.append(TARGET_MODE)

    original_row_to_mode_spec = sweep.row_to_mode_spec

    def patched_row_to_mode_spec(row: Dict[str, Any]) -> tuple[str, str, float | None]:
        mode = str(row.get("mode", ""))
        if mode == TARGET_MODE:
            ratio = float(row["budget"])
            return (TARGET_MODE, f"{ratio:.2f}", ratio)
        return original_row_to_mode_spec(row)

    sweep.row_to_mode_spec = patched_row_to_mode_spec


def make_args(
    base: argparse.Namespace,
    sweep: Any,
    model_key: str,
    run_id: str,
    output_dir: Path,
    budgets: Sequence[str],
) -> SimpleNamespace:
    return SimpleNamespace(
        model_key=model_key,
        selected_modes=[TARGET_MODE],
        selected_ratios=[float(item) for item in budgets],
        base_tokens=str(base.base_tokens),
        force_cache_build=bool(base.force_cache_build),
        restart_server_each_spec=bool(base.restart_server_each_spec),
        no_warm=bool(base.no_warm),
        server_env=list(base.server_env or []),
        remote_memory_limit_percent=int(base.remote_memory_limit_percent),
        ssh_command_timeout_sec=float(base.ssh_command_timeout_sec),
        remote_bench_timeout_sec=float(base.remote_bench_timeout_sec),
        server_runtime_timeout_sec=float(base.server_runtime_timeout_sec),
        output_dir=str(output_dir),
        run_id=run_id,
        frequency_profile=str(base.frequency_profile),
        normal_rep=1,
        power_capture=bool(base.power_capture),
        power_api_url=str(getattr(base, "power_api_url", sweep.DEFAULT_POWER_API_URL)),
        power_output_root=str(getattr(base, "power_output_root", "") or (sweep.SKILL_ROOT / "power")),
        power_device=str(base.power_device),
        power_monitor_type=str(base.power_monitor_type),
        power_serial=str(base.power_serial),
        power_port=str(base.power_port),
        power_sample_rate_hz=int(base.power_sample_rate_hz),
        power_voltage_mv=int(base.power_voltage_mv),
        power_max_duration_sec=float(base.power_max_duration_sec),
        power_stop_timeout_sec=float(base.power_stop_timeout_sec),
        power_warmup_sec=float(base.power_warmup_sec),
        power_cooldown_sec=float(base.power_cooldown_sec),
        power_rep=max(1, int(base.power_rep)),
        power_day=str(base.power_day),
        power_hour=str(base.power_hour),
        power_minute=str(base.power_minute),
        power_process_id=str(base.power_process_id),
        benchmark_csv=str(base.benchmark_csv),
        only_benchmark_csv_rows=True,
        evict_document_kv_before_measure=False,
        allow_high_risk_contexts=bool(base.allow_high_risk_contexts),
    )


def write_run_config(
    path: Path,
    sweep: Any,
    plan: Dict[str, Dict[str, Dict[str, Any]]],
    requested: Dict[str, Dict[int, List[Dict[str, str]]]],
    args: argparse.Namespace,
) -> None:
    device_contexts = {
        device: sorted(int(ctx) for ctx in rows_by_ctx)
        for device, rows_by_ctx in requested.items()
        if rows_by_ctx
    }
    run_config = {
        "devices": sorted(device_contexts),
        "device_roles": {
            device: ("formal" if device in getattr(sweep, "FORMAL_REQUIRED_DEVICES", ()) else "extra-profile")
            for device in device_contexts
        },
        "formal_required_devices": list(getattr(sweep, "FORMAL_REQUIRED_DEVICES", ())),
        "extra_profile_devices": list(getattr(sweep, "EXTRA_PROFILE_DEVICES", ())),
        "formal_matrix_complete": all(
            device in device_contexts for device in getattr(sweep, "FORMAL_REQUIRED_DEVICES", ())
        ),
        "model_keys": sorted(plan),
        "models": {model_key: str(sweep.MODELS[model_key]["name"]) for model_key in sorted(plan)},
        "contexts": device_contexts,
        "modes": [TARGET_MODE],
        "ratios": sorted(
            {
                float(budget)
                for model_devices in plan.values()
                for entry in model_devices.values()
                for budget in entry["budgets"]
            }
        ),
        "frequency_profile": str(args.frequency_profile),
        "shared_cache": True,
        "benchmark_csv": str(args.benchmark_csv),
        "base_tokens": str(args.base_tokens),
        "server_env": list(args.server_env),
        "remote_memory_limit_percent": int(args.remote_memory_limit_percent),
        "allow_high_risk_contexts": bool(args.allow_high_risk_contexts),
        "power_capture": bool(args.power_capture),
        "power_rep": int(args.power_rep),
    }
    path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", default=str(MNN_ROOT / "benchmark.csv"))
    parser.add_argument("--devices", default="jetson,orangepi,rhino")
    parser.add_argument("--model-keys", default="")
    parser.add_argument("--contexts", default="")
    parser.add_argument("--budgets", default="")
    parser.add_argument("--run-id", default="fusionrag_online_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", default=str(MNN_ROOT / ".cache/latency_budget_20260625"))
    parser.add_argument("--base-tokens", default="")
    parser.add_argument("--frequency-profile", default="max")
    parser.add_argument("--server-env", action="append", default=[])
    parser.add_argument("--remote-memory-limit-percent", type=int, default=int(os.environ.get("PIC_SWEEP_REMOTE_MEMORY_LIMIT_PERCENT", "95")))
    parser.add_argument("--ssh-command-timeout-sec", type=float, default=120.0)
    parser.add_argument("--remote-bench-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--server-runtime-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--force-cache-build", action="store_true")
    parser.add_argument("--restart-server-each-spec", action="store_true")
    parser.add_argument("--allow-high-risk-contexts", action="store_true")
    parser.add_argument("--power-capture", action="store_true")
    parser.add_argument("--power-api-url", default=os.environ.get("MNN_POWER_API_URL", ""))
    parser.add_argument("--power-output-root", default="")
    parser.add_argument("--power-device", default=os.environ.get("MNN_POWER_DEVICE", ""))
    parser.add_argument("--power-monitor-type", default=os.environ.get("MNN_POWER_MONITOR_TYPE", ""))
    parser.add_argument("--power-serial", default=os.environ.get("MNN_POWER_SERIAL", ""))
    parser.add_argument("--power-port", default=os.environ.get("MNN_POWER_PORT", ""))
    parser.add_argument("--power-sample-rate-hz", type=int, default=int(os.environ["MNN_POWER_SAMPLE_RATE_HZ"]) if os.environ.get("MNN_POWER_SAMPLE_RATE_HZ") else 0)
    parser.add_argument("--power-voltage-mv", type=int, default=int(os.environ["MNN_POWER_VOLTAGE_MV"]) if os.environ.get("MNN_POWER_VOLTAGE_MV") else 0)
    parser.add_argument("--power-max-duration-sec", type=float, default=float(os.environ.get("MNN_POWER_MAX_DURATION_SEC", "1800")))
    parser.add_argument("--power-stop-timeout-sec", type=float, default=float(os.environ.get("MNN_POWER_STOP_TIMEOUT_SEC", "30")))
    parser.add_argument("--power-warmup-sec", type=float, default=float(os.environ.get("MNN_POWER_WARMUP_SEC", "5")))
    parser.add_argument("--power-cooldown-sec", type=float, default=float(os.environ.get("MNN_POWER_COOLDOWN_SEC", "5")))
    parser.add_argument("--power-rep", type=int, default=int(os.environ.get("MNN_POWER_REP", "3")))
    args = parser.parse_args()

    sweep = load_sweep_module()
    patch_sweep_module(sweep)
    if not args.base_tokens:
        args.base_tokens = str(sweep.DEFAULT_BASE_TOKENS)
    if args.power_rep <= 0:
        raise SystemExit("--power-rep must be positive")
    if not args.power_api_url:
        args.power_api_url = str(getattr(sweep, "DEFAULT_POWER_API_URL", ""))
    if not args.power_output_root:
        args.power_output_root = str(sweep.SKILL_ROOT / "power")
    power_now = time.localtime()
    args.power_day = time.strftime("%Y-%m-%d", power_now)
    args.power_hour = time.strftime("%H", power_now)
    args.power_minute = time.strftime("%M", power_now)
    args.power_process_id = "%s_%d_%s" % (args.run_id, os.getpid(), uuid.uuid4().hex[:8])

    devices = parse_csv_set(args.devices)
    model_filter = parse_csv_set(args.model_keys)
    context_filter = parse_csv_set(args.contexts)
    budget_filter = parse_csv_set(args.budgets)
    benchmark_csv = Path(args.benchmark_csv)
    plan = load_cacheblend_plan(benchmark_csv, devices, model_filter, context_filter, budget_filter)
    if args.power_capture:
        excluded = sorted(model_key for model_key in plan if model_key in POWER_EXCLUDED_MODEL_KEYS)
        for model_key in excluded:
            print(f"[power] skip excluded model {model_key}", flush=True)
            plan.pop(model_key, None)
    if not plan:
        raise SystemExit("no cacheblend rows matched the requested filters")

    output_dir = Path(args.output_dir)
    out_dir = output_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    failures_jsonl = out_dir / "failures.jsonl"
    (out_dir / "fusionrag_online_run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, default=list) + "\n",
        encoding="utf-8",
    )

    requested: Dict[str, Dict[int, List[Dict[str, str]]]] = {}
    for model_key in sorted(plan):
        model_name = str(sweep.MODELS[model_key]["name"])
        for device in sorted(plan[model_key]):
            entry = plan[model_key][device]
            grouped = missing_rows_by_context(
                benchmark_csv,
                summary_csv,
                model_name,
                device,
                list(entry["contexts"]),
                list(entry["budgets"]),
            )
            if grouped:
                requested.setdefault(device, {})
                requested[device].update(grouped)
    write_run_config(out_dir / "run_config.json", sweep, plan, requested, args)

    for model_key in sorted(plan):
        for device in sorted(plan[model_key]):
            entry = plan[model_key][device]
            grouped = missing_rows_by_context(
                benchmark_csv,
                summary_csv,
                str(sweep.MODELS[model_key]["name"]),
                device,
                list(entry["contexts"]),
                list(entry["budgets"]),
            )
            contexts = sorted(grouped)
            if not contexts:
                print(f"[skip] {device} {model_key} no fusionrag_online gaps", flush=True)
                continue
            helper_args = make_args(args, sweep, model_key, args.run_id, output_dir, list(entry["budgets"]))
            print(
                "[run] device=%s model=%s contexts=%s budgets=%s"
                % (device, model_key, contexts, list(entry["budgets"])),
                flush=True,
            )
            sweep.run_device(device, contexts, helper_args, FIELDNAMES, summary_csv, failures_jsonl, grouped)

    print(f"[done] summary={summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
