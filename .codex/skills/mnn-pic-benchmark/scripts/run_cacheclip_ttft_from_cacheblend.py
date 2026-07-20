#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


MNN_ROOT = Path(__file__).resolve().parents[4]
SWEEP_SCRIPT = MNN_ROOT / ".codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py"
CACHECLIP_MODE = "cacheclip"
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
    "ttft_s",
    "ttft_ms",
    "chat_latency_s",
    "chat_latency_ms",
    "server_prefill_latency_s",
    "execution_mode",
    "recompute_token_count",
    "reuse_token_count",
    "selected_count",
    "selector_note",
    "selector_latency_s",
    "selector_wall_s",
    "selector_cache_hit",
]


def load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("mnn_pic_prefill_sweep", SWEEP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load sweep script: %s" % SWEEP_SCRIPT)
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


def seen_mode_rows(paths: Sequence[Path], model_name: str, device_name: str, mode: str) -> Set[Tuple[int, str]]:
    seen: Set[Tuple[int, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("model") != model_name or row.get("device") != device_name:
                    continue
                if row.get("mode") != mode:
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
    paths: Sequence[Path],
    model_name: str,
    device_name: str,
    contexts: Sequence[int],
    budgets: Sequence[str],
) -> Dict[int, List[Dict[str, str]]]:
    seen = seen_mode_rows(paths, model_name, device_name, CACHECLIP_MODE)
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for context in contexts:
        for budget in budgets:
            key = (int(context), budget_key(budget))
            if key in seen:
                continue
            grouped.setdefault(int(context), []).append(
                {
                    "context_tokens": str(int(context)),
                    "mode": CACHECLIP_MODE,
                    "budget": budget_key(budget),
                }
            )
    return grouped


def _format_value(template: str, mapping: Dict[str, Any]) -> str:
    if not template:
        return ""
    try:
        return str(template).format_map(mapping)
    except KeyError as exc:
        raise RuntimeError("unknown placeholder %r in selector path/template %r" % (exc.args[0], template))


def _shell_export_lines(entries: Sequence[str], mapping: Dict[str, Any]) -> List[str]:
    exports = []
    for entry in entries:
        rendered = _format_value(str(entry), mapping).strip()
        if not rendered:
            continue
        if "=" not in rendered:
            raise RuntimeError("selector env must be KEY=VALUE, got %r" % (entry,))
        key, value = rendered.split("=", 1)
        key = key.strip()
        if not key:
            raise RuntimeError("selector env has empty key: %r" % (entry,))
        exports.append("export %s=%s" % (key, shlex.quote(value)))
    return exports


def _parse_last_json(stdout: str) -> Dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        text = line.strip()
        if not text.startswith("{") or not text.endswith("}"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("remote selector did not emit a JSON object")


def patch_sweep_module(sweep: Any) -> None:
    sweep.RATIO_MODES.add(CACHECLIP_MODE)
    if CACHECLIP_MODE not in sweep.MODE_CHOICES:
        sweep.MODE_CHOICES.append(CACHECLIP_MODE)

    original_run_chat = sweep.run_chat
    original_run_device = sweep.run_device
    original_summarize = sweep.summarize_chat
    original_row_to_mode_spec = sweep.row_to_mode_spec

    def patched_row_to_mode_spec(row: Dict[str, Any]) -> tuple[str, str, float | None]:
        mode = str(row.get("mode", ""))
        if mode == CACHECLIP_MODE:
            ratio = float(row["budget"])
            return (CACHECLIP_MODE, f"{ratio:.2f}", ratio)
        return original_row_to_mode_spec(row)

    selector_state: Dict[str, Any] = {"device_name": None, "args": None}
    def current_context() -> Tuple[str, argparse.Namespace]:
        device_name = selector_state.get("device_name")
        args = selector_state.get("args")
        if not device_name or args is None:
            raise RuntimeError("cacheclip helper lost current device context")
        return str(device_name), args

    def resolve_selector_device() -> Dict[str, Any]:
        device_name, args = current_context()
        device = sweep.resolve_device_model(device_name, args.model_key)
        device["device_key"] = device_name
        device["ssh_command_timeout_sec"] = float(args.ssh_command_timeout_sec)
        device["remote_bench_timeout_sec"] = float(args.remote_bench_timeout_sec)
        device["server_runtime_timeout_sec"] = float(args.server_runtime_timeout_sec)
        return device

    def selector_mapping(device: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        model_key = str(args.model_key)
        model_name = str(sweep.MODELS[model_key]["name"])
        return {
            "device_key": str(device["device_key"]),
            "device_display": str(device["display"]),
            "remote_work": str(device["remote_work"]),
            "remote_cache_root": str(device["remote_cache_root"]),
            "artifact": str(device["artifact"]),
            "model_key": model_key,
            "model_name": model_name,
            "served_model": str(device["served_model"]),
        }

    def selector_config(device: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        mapping = selector_mapping(device, args)
        primary_tokenizer = _format_value(str(args.cacheclip_primary_tokenizer or ""), mapping).strip()
        aux_model = _format_value(str(args.cacheclip_aux_model or ""), mapping).strip()
        aux_tokenizer = _format_value(str(args.cacheclip_aux_tokenizer or ""), mapping).strip()
        selector_binary = _format_value(str(args.cacheclip_selector_binary or ""), mapping).strip()
        if not primary_tokenizer:
            raise RuntimeError("--cacheclip-primary-tokenizer is required")
        if not aux_model:
            raise RuntimeError("--cacheclip-aux-model is required")
        if not selector_binary:
            raise RuntimeError("--cacheclip-selector-binary is required")
        cache_root_template = str(args.cacheclip_selector_cache_root or "")
        if cache_root_template.strip():
            cache_root = _format_value(cache_root_template, mapping).strip()
        else:
            cache_root = "%s/shared_aux_cache/cacheclip/%s" % (
                str(device["remote_cache_root"]),
                str(args.model_key),
            )
        return {
            "selector_binary": selector_binary,
            "primary_tokenizer": primary_tokenizer,
            "aux_model": aux_model,
            "aux_tokenizer": aux_tokenizer,
            "aux_max_length": int(args.cacheclip_aux_max_length),
            "query_tail_tokens": int(args.cacheclip_query_tail_tokens),
            "group_window_tokens": int(args.cacheclip_group_window_tokens),
            "group_min_candidates": int(args.cacheclip_group_min_candidates),
            "min_tokens": int(args.cacheclip_min_tokens),
            "score_layer_idx": int(args.cacheclip_score_layer_idx),
            "cache_root": cache_root,
            "env_exports": _shell_export_lines(list(args.cacheclip_selector_env or []), mapping),
        }

    def run_remote_selector(
        device: Dict[str, Any],
        args: argparse.Namespace,
        tokens: Dict[str, Any],
        ratio: float,
    ) -> Tuple[Dict[str, Any], float]:
        config = selector_config(device, args)
        prelude_count = int(tokens["prelude_token_count"])
        suffix_count = int(tokens["suffix_token_count"])
        full_prompt_token_ids = list(map(int, tokens["full_prompt_token_ids"]))
        payload = {
            "selector_algorithm": CACHECLIP_MODE,
            "primary_tokenizer": config["primary_tokenizer"],
            "aux_model": config["aux_model"],
            "aux_tokenizer": config["aux_tokenizer"],
            "aux_max_length": int(config["aux_max_length"]),
            "query_tail_tokens": int(config["query_tail_tokens"]),
            "group_window_tokens": int(config["group_window_tokens"]),
            "group_min_candidates": int(config["group_min_candidates"]),
            "min_tokens": int(config["min_tokens"]),
            "ratio": float(ratio),
            "aux_cache_root": str(config["cache_root"]),
            "prefix_token_ids": full_prompt_token_ids[:prelude_count],
            "doc_token_ids": list(map(int, tokens["doc_token_ids"])),
            "query_token_ids": full_prompt_token_ids[-suffix_count:] if suffix_count > 0 else [],
        }
        ld_library_path = str(device.get("ld_library_path", "")).strip()
        env_lines = []
        if ld_library_path:
            env_lines.append("export LD_LIBRARY_PATH=%s:${LD_LIBRARY_PATH:-}" % shlex.quote(ld_library_path))
        env_lines.extend(list(config["env_exports"]))
        env_block = "\n".join(env_lines)
        if env_block:
            env_block += "\n"
        remote_script_body = """set -e
mkdir -p '{cache_root}'
cd '{remote_work}'
if [[ ! -x '{selector_binary}' ]]; then
  printf '%s\\n' {missing_json}
  exit 0
fi
{env_block}'{selector_binary}' <<'JSONEOF'
{payload}
JSONEOF
""".format(
            cache_root=str(config["cache_root"]),
            remote_work=str(device["remote_work"]),
            selector_binary=str(config["selector_binary"]),
            missing_json=shlex.quote(json.dumps({"error": "missing selector binary: %s" % str(config["selector_binary"])})),
            env_block=env_block,
            payload=json.dumps(payload, ensure_ascii=False),
        )
        started = time.perf_counter()
        result = sweep.ssh(
            device,
            remote_script_body,
            timeout=float(args.cacheclip_selector_timeout_sec),
        )
        wall_elapsed = time.perf_counter() - started
        parsed = _parse_last_json(result.stdout)
        if parsed.get("error"):
            raise RuntimeError(str(parsed["error"]))
        return parsed, float(wall_elapsed)

    def cacheclip_chat_payload(
        tokens: Dict[str, Any],
        doc_id: str,
        ratio: float,
        model: str,
        selected_local_indices: Sequence[int],
        score_layer_idx: int,
    ) -> Dict[str, Any]:
        pic_cache = {
            "id": "%s-%s-%s-%s" % (doc_id, CACHECLIP_MODE, ratio, time.time_ns()),
            "text_cache_refs": [{"id": doc_id}],
            "selection_algorithm": CACHECLIP_MODE,
            "pic_recompute_score_layer_idx": int(score_layer_idx),
            "pic_recompute_ratio": float(ratio),
            "pic_recompute_pic_local_indices": [int(item) for item in selected_local_indices],
        }
        return {
            "model": model,
            "stream": False,
            "max_tokens": 0,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "full_prompt_token_ids": tokens["full_prompt_token_ids"],
            "doc_cache_spans": [
                {
                    "prompt_start": tokens["prompt_start"],
                    "source_start": tokens["source_start"],
                    "token_count": tokens["token_count"],
                }
            ],
            "pic_cache": pic_cache,
        }

    def cacheclip_failure_row(tokens: Dict[str, Any], budget: str, error: str) -> Dict[str, Any]:
        return {
            "context_tokens": int(tokens["full_prompt_token_count"]),
            "mode": CACHECLIP_MODE,
            "budget": budget,
            "prefill_latency_s": "",
            "prefill_tps": "",
            "status": 599,
            "execution_mode": "",
            "recompute_token_count": "",
            "reuse_token_count": "",
            "selected_count": "",
            "error": error,
            "ttft_s": "",
            "ttft_ms": "",
            "chat_latency_s": "",
            "chat_latency_ms": "",
            "server_prefill_latency_s": "",
            "selector_note": "",
            "selector_latency_s": "",
            "selector_wall_s": "",
            "selector_cache_hit": "",
        }

    def patched_run_chat(
        base_url: str,
        local_raw: Path,
        tokens: Dict[str, Any],
        doc_id: str,
        mode: str,
        budget: str,
        ratio: Optional[float],
        model: str,
        prefix: str,
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        if mode != CACHECLIP_MODE:
            return original_run_chat(base_url, local_raw, tokens, doc_id, mode, budget, ratio, model, prefix)
        if ratio is None:
            raise RuntimeError("cacheclip requires an explicit recompute ratio")

        device = resolve_selector_device()
        args = current_context()[1]
        score_layer_idx = int(args.cacheclip_score_layer_idx)
        try:
            sweep.post_json(base_url, "/reset", {"reset": True}, timeout=30)
            selector_result, selector_wall = run_remote_selector(device, args, tokens, float(ratio))
            selected = [int(item) for item in selector_result.get("selected_pic_local_indices", [])]
            payload = cacheclip_chat_payload(tokens, doc_id, float(ratio), model, selected, score_layer_idx)
            status, response, elapsed = sweep.post_json(base_url, "/v1/chat/completions", payload)
        except Exception as exc:
            raw = {
                "status": 599,
                "latency_s": 0.0,
                "payload_meta": {"mode": CACHECLIP_MODE, "budget": budget},
                "selector_error": str(exc),
                "response": {"error": str(exc)},
            }
            sweep.write_json(local_raw / ("%s_chat_%s_%s.json" % (prefix, CACHECLIP_MODE, budget.replace(".", "p"))), raw)
            return 599, {"error": str(exc)}, cacheclip_failure_row(tokens, budget, str(exc))

        raw = {
            "status": status,
            "latency_s": elapsed,
            "payload_meta": {"mode": CACHECLIP_MODE, "budget": budget},
            "selector": {
                "wall_s": selector_wall,
                "device_selector": selector_result,
            },
            "response": response,
        }
        sweep.write_json(local_raw / ("%s_chat_%s_%s.json" % (prefix, CACHECLIP_MODE, budget.replace(".", "p"))), raw)

        row = original_summarize(CACHECLIP_MODE, budget, status, elapsed, response, int(tokens["full_prompt_token_count"]))
        selector_latency = float(selector_result.get("selector_latency_s", 0.0) or 0.0)
        selector_wall = float(selector_wall)
        chat_elapsed = float(elapsed or 0.0)
        server_perf = 0.0
        try:
            server_perf = float(sweep.extract_perf(response).get("prefill_latency_s", 0.0) or 0.0)
        except Exception:
            server_perf = 0.0
        total = selector_wall + chat_elapsed if status == 200 else 0.0
        context = int(tokens["full_prompt_token_count"])
        if total > 0.0:
            row["prefill_latency_s"] = total
            row["prefill_tps"] = float(context) / float(total)
            row["ttft_s"] = total
            row["ttft_ms"] = float(total) * 1000.0
        else:
            row["ttft_s"] = ""
            row["ttft_ms"] = ""
        row["chat_latency_s"] = chat_elapsed if status == 200 else ""
        row["chat_latency_ms"] = (chat_elapsed * 1000.0) if status == 200 else ""
        row["server_prefill_latency_s"] = server_perf if server_perf > 0.0 else ""
        row["mode"] = CACHECLIP_MODE
        row["selected_count"] = selector_result.get("selected_count", row.get("selected_count", ""))
        row["selector_note"] = "remote_cpu_mnn_aux_attention_grouped_no_aux_kv_reuse;score_layer=%d;w=%d;tau=%d" % (
            score_layer_idx,
            int(args.cacheclip_group_window_tokens),
            int(args.cacheclip_group_min_candidates),
        )
        row["selector_latency_s"] = selector_latency
        row["selector_wall_s"] = selector_wall
        row["selector_cache_hit"] = bool(selector_result.get("selector_cache_hit", False))
        return status, response, row

    def patched_run_device(
        device_name: str,
        contexts: List[int],
        args: argparse.Namespace,
        fieldnames: List[str],
        summary_csv: Path,
        failures_jsonl: Path,
        missing_rows_by_context: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> None:
        selector_state["device_name"] = device_name
        selector_state["args"] = args
        try:
            return original_run_device(device_name, contexts, args, fieldnames, summary_csv, failures_jsonl, missing_rows_by_context)
        finally:
            selector_state["device_name"] = None
            selector_state["args"] = None

    sweep.run_chat = patched_run_chat
    sweep.run_device = patched_run_device
    sweep.row_to_mode_spec = patched_row_to_mode_spec


def make_args(base: argparse.Namespace, sweep: Any, model_key: str, run_id: str, output_dir: Path, budgets: Sequence[str]) -> SimpleNamespace:
    return SimpleNamespace(
        model_key=model_key,
        selected_modes=[CACHECLIP_MODE],
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
        only_missing_contexts=False,
        evict_document_kv_before_measure=False,
        allow_high_risk_contexts=bool(base.allow_high_risk_contexts),
        allow_subset_devices=True,
        allow_extra_device=True,
        cacheclip_primary_tokenizer=str(base.cacheclip_primary_tokenizer),
        cacheclip_aux_model=str(base.cacheclip_aux_model),
        cacheclip_aux_tokenizer=str(base.cacheclip_aux_tokenizer),
        cacheclip_aux_max_length=int(base.cacheclip_aux_max_length),
        cacheclip_query_tail_tokens=int(base.cacheclip_query_tail_tokens),
        cacheclip_group_window_tokens=int(base.cacheclip_group_window_tokens),
        cacheclip_group_min_candidates=int(base.cacheclip_group_min_candidates),
        cacheclip_min_tokens=int(base.cacheclip_min_tokens),
        cacheclip_score_layer_idx=int(base.cacheclip_score_layer_idx),
        cacheclip_selector_binary=str(base.cacheclip_selector_binary),
        cacheclip_selector_cache_root=str(base.cacheclip_selector_cache_root),
        cacheclip_selector_env=list(base.cacheclip_selector_env or []),
        cacheclip_selector_timeout_sec=float(base.cacheclip_selector_timeout_sec),
    )


def write_run_config(
    path: Path,
    sweep: Any,
    plan: Dict[str, Dict[str, Dict[str, Any]]],
    args: argparse.Namespace,
    requested: Dict[str, Dict[int, List[Dict[str, str]]]],
) -> None:
    normalized_contexts = {
        device: sorted(int(ctx) for ctx in rows_by_ctx)
        for device, rows_by_ctx in requested.items()
        if rows_by_ctx
    }
    model_keys = sorted(plan)
    run_config = {
        "devices": sorted(normalized_contexts),
        "device_roles": {
            device: ("formal" if device in getattr(sweep, "FORMAL_REQUIRED_DEVICES", ()) else "extra-profile")
            for device in normalized_contexts
        },
        "formal_required_devices": list(getattr(sweep, "FORMAL_REQUIRED_DEVICES", ())),
        "extra_profile_devices": list(getattr(sweep, "EXTRA_PROFILE_DEVICES", ())),
        "formal_matrix_complete": all(
            device in normalized_contexts for device in getattr(sweep, "FORMAL_REQUIRED_DEVICES", ())
        ),
        "model_keys": model_keys,
        "models": {model_key: str(sweep.MODELS[model_key]["name"]) for model_key in model_keys},
        "contexts": normalized_contexts,
        "modes": [CACHECLIP_MODE],
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
        "ports": {
            device: {
                "remote_port": sweep.DEVICES[device]["remote_port"],
                "local_port": sweep.DEVICES[device]["local_port"],
                "port_strategy": "fixed-kill-old",
            }
            for device in normalized_contexts
        },
        "force_cache_build": bool(args.force_cache_build),
        "restart_server_each_spec": bool(args.restart_server_each_spec),
        "benchmark_csv": str(args.benchmark_csv),
        "base_tokens": str(args.base_tokens),
        "server_env": list(args.server_env),
        "remote_memory_limit_percent": int(args.remote_memory_limit_percent),
        "allow_high_risk_contexts": bool(args.allow_high_risk_contexts),
        "allow_subset_devices": True,
        "allow_extra_device": True,
        "power_capture": bool(args.power_capture),
        "power_rep": int(args.power_rep),
        "selector": {
            "algorithm": CACHECLIP_MODE,
            "location": "remote-device-cpu",
            "ttft_selector_latency_source": "remote_selector_reported",
            "implementation": "mnn-cacheclip-selector-no-aux-kv-reuse-by-default",
            "ttft_selector_latency_note": "selector_latency_s is device-reported native MNN selector time; selector_wall_s includes ssh/process overhead; aux prefix-cache reuse is disabled unless CACHECLIP_ENABLE_AUX_PREFIX_CACHE=1",
            "primary_tokenizer": str(args.cacheclip_primary_tokenizer),
            "aux_model": str(args.cacheclip_aux_model),
            "aux_tokenizer": str(args.cacheclip_aux_tokenizer),
            "aux_max_length": int(args.cacheclip_aux_max_length),
            "query_tail_tokens": int(args.cacheclip_query_tail_tokens),
            "group_window_tokens": int(args.cacheclip_group_window_tokens),
            "group_min_candidates": int(args.cacheclip_group_min_candidates),
            "min_tokens": int(args.cacheclip_min_tokens),
            "score_layer_idx": int(args.cacheclip_score_layer_idx),
            "selector_binary": str(args.cacheclip_selector_binary),
            "selector_cache_root": str(args.cacheclip_selector_cache_root),
            "remote_env": list(args.cacheclip_selector_env or []),
        },
    }
    path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", default=str(MNN_ROOT / "benchmark.csv"))
    parser.add_argument("--devices", default="jetson,orangepi,rhino")
    parser.add_argument("--model-keys", default="")
    parser.add_argument("--contexts", default="")
    parser.add_argument("--budgets", default="")
    parser.add_argument("--run-id", default="cacheclip_ttft_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", default=str(MNN_ROOT / ".cache/mnn-pic-benchmark"))
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
    parser.add_argument("--no-resume", action="store_true")
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
    parser.add_argument("--cacheclip-primary-tokenizer", default="")
    parser.add_argument("--cacheclip-aux-model", default="")
    parser.add_argument("--cacheclip-aux-tokenizer", default="")
    parser.add_argument("--cacheclip-aux-max-length", type=int, default=0)
    parser.add_argument("--cacheclip-query-tail-tokens", type=int, default=0)
    parser.add_argument("--cacheclip-group-window-tokens", type=int, default=8)
    parser.add_argument("--cacheclip-group-min-candidates", type=int, default=5)
    parser.add_argument("--cacheclip-min-tokens", type=int, default=1)
    parser.add_argument("--cacheclip-score-layer-idx", type=int, default=1)
    parser.add_argument("--cacheclip-selector-binary", default="{artifact}/bin/cacheclip_selector")
    parser.add_argument("--cacheclip-selector-cache-root", default="{remote_cache_root}/shared_aux_cache/cacheclip/{model_key}")
    parser.add_argument("--cacheclip-selector-env", "--remote-selector-env", dest="cacheclip_selector_env", action="append", default=[])
    parser.add_argument("--cacheclip-selector-timeout-sec", "--remote-selector-timeout-sec", dest="cacheclip_selector_timeout_sec", type=float, default=3600.0)
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
            print("[power] skip excluded model %s" % model_key, flush=True)
            plan.pop(model_key, None)
    if not plan:
        raise SystemExit("no cacheblend rows matched the requested filters")

    output_dir = Path(args.output_dir)
    out_dir = output_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    failures_jsonl = out_dir / "failures.jsonl"
    run_plan_path = out_dir / "cacheclip_run_plan.json"
    run_plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
    requested: Dict[str, Dict[int, List[Dict[str, str]]]] = {}
    seen_paths = [benchmark_csv]
    if not args.no_resume:
        seen_paths.append(summary_csv)
    for model_key in sorted(plan):
        model_name = str(sweep.MODELS[model_key]["name"])
        for device in sorted(plan[model_key]):
            entry = plan[model_key][device]
            grouped = missing_rows_by_context(
                seen_paths,
                model_name,
                device,
                list(entry["contexts"]),
                list(entry["budgets"]),
            )
            if not grouped:
                continue
            requested.setdefault(device, {})
            for ctx, rows in grouped.items():
                requested[device].setdefault(int(ctx), []).extend(rows)
    write_run_config(out_dir / "run_config.json", sweep, plan, args, requested)

    for model_key in sorted(plan):
        model_name = str(sweep.MODELS[model_key]["name"])
        for device in sorted(plan[model_key]):
            entry = plan[model_key][device]
            budgets = list(entry["budgets"])
            grouped = missing_rows_by_context(
                seen_paths,
                model_name,
                device,
                list(entry["contexts"]),
                budgets,
            )
            contexts = sorted(grouped)
            if not contexts:
                print("[skip] %s %s no cacheclip gaps" % (device, model_key), flush=True)
                continue
            helper_args = make_args(args, sweep, model_key, args.run_id, output_dir, budgets)
            print(
                "[run] device=%s model=%s contexts=%s budgets=%s"
                % (device, model_key, contexts, budgets),
                flush=True,
            )
            sweep.run_device(device, contexts, helper_args, FIELDNAMES, summary_csv, failures_jsonl, grouped)

    print("[done] summary=%s" % summary_csv, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
