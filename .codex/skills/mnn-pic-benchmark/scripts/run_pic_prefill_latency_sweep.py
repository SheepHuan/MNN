#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MNN_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_TOKENS = MNN_ROOT / ".cache/latency_budget_20260611/opencl_pic_1024_current_20260622_191527/tokens.json"
TEMPLATE_MODEL = "Llama3.2 1B"
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
MODE_CHOICES = ["normal-full-recompute", "full-compute", "full-reuse", "cacheblend", "epic"]
RATIO_MODES = {"cacheblend", "epic"}
CSV_KEYS = [
    "device",
    "device_display",
    "model",
    "backend",
    "frequency_profile",
    "context_tokens",
    "mode",
    "budget",
]
GAP_COMPARE_KEYS = [
    "backend",
    "frequency_profile",
    "context_tokens",
    "mode",
    "budget",
]

DEVICES: dict[str, dict[str, Any]] = {
    "jetson": {
        "display": "Jetson AGX Xavier",
        "ssh": "jetson@192.168.101.192",
        "remote_work": "/home/jetson/code/kvshare-edge/impl/MNN",
        "artifact": "/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda",
        "ld_library_path": "/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib",
        "server_env": "",
        "bench_env": "",
        "backend": "cuda",
        "remote_port": 18131,
        "local_port": 19131,
        "remote_cache_root": "/home/jetson/code/kvshare-edge/impl/MNN/.cache/pic_prefill_latency_sweep",
    },
    "orangepi": {
        "display": "Orange Pi 5 Plus",
        "ssh": "orangepi@192.168.101.113",
        "remote_work": "/mnt/ssd/code/.cache/mnn_opencl_pic",
        "artifact": "/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus",
        "ld_library_path": "/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/lib",
        "server_env": "",
        "bench_env": "LD_PRELOAD=/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so",
        "backend": "opencl",
        "remote_port": 18132,
        "local_port": 19132,
        "remote_cache_root": "/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep",
    },
    "rhino": {
        "display": "Rhino Pi X1",
        "ssh": "aidlux@192.168.101.227",
        "remote_work": "/mnt/nvme/mnn_pic_opencl",
        "artifact": "/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl",
        "ld_library_path": "/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/lib:/usr/lib:/usr/lib/aarch64-linux-gnu",
        "server_env": "LD_PRELOAD=/usr/lib/libOpenCL_adreno.so",
        "bench_env": "LD_PRELOAD=/usr/lib/libOpenCL_adreno.so:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so",
        "backend": "opencl",
        "remote_port": 18133,
        "local_port": 19133,
        "remote_cache_root": "/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep",
    },
}

MODELS: dict[str, dict[str, Any]] = {
    "llama3.2-1b": {
        "name": "Llama3.2 1B",
        "served_model": "llama32-1b-pic",
        "paths": {
            "jetson": {
                "normal_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-llm-export/AI-ModelScope__Llama-3.2-1B-Instruct/config_cuda_greedy.json",
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-1B-Instruct/config_cuda_greedy.json",
                ],
                "pic_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-pic-boundary/config_cuda_greedy.json",
                ],
            },
            "orangepi": {
                "normal_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/AI-ModelScope__Llama-3.2-1B-Instruct/config_opencl_greedy.json",
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/AI-ModelScope__Llama-3___2-1B-Instruct/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/AI-ModelScope__Llama-3___2-1B-Instruct-pic-boundary/config_opencl_greedy.json",
                ],
            },
            "rhino": {
                "normal_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/normal/AI-ModelScope__Llama-3.2-1B-Instruct/config_opencl_greedy.json",
                    "/mnt/nvme/mnn_pic_opencl/models/normal/AI-ModelScope__Llama-3___2-1B-Instruct/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/pic/AI-ModelScope__Llama-3___2-1B-Instruct-pic-boundary/config_opencl_greedy.json",
                ],
            },
        },
    },
    "llama3.2-3b": {
        "name": "Llama3.2 3B",
        "served_model": "llama32-3b-pic",
        "paths": {
            "jetson": {
                "normal_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json",
                ],
                "pic_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json",
                ],
            },
            "orangepi": {
                "normal_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/AI-ModelScope__Llama-3___2-3B-Instruct/config_opencl_greedy.json",
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/AI-ModelScope__Llama-3.2-3B-Instruct/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_opencl_greedy.json",
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/AI-ModelScope__Llama-3___2-3B-Instruct/config_opencl_greedy.json",
                ],
            },
            "rhino": {
                "normal_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/normal/AI-ModelScope__Llama-3___2-3B-Instruct/config_opencl_greedy.json",
                    "/mnt/nvme/mnn_pic_opencl/models/normal/AI-ModelScope__Llama-3.2-3B-Instruct/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/pic/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_opencl_greedy.json",
                    "/mnt/nvme/mnn_pic_opencl/models/pic/AI-ModelScope__Llama-3___2-3B-Instruct/config_opencl_greedy.json",
                ],
            },
        },
    },
    "minicpm5-1b": {
        "name": "MiniCPM5-1B",
        "served_model": "minicpm5-1b-pic",
        "paths": {
            "jetson": {
                "normal_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-llm-export/OpenBMB__MiniCPM5-1B/config_cuda_greedy.json",
                ],
                "pic_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary/config_cuda_greedy.json",
                ],
            },
            "orangepi": {
                "normal_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/OpenBMB__MiniCPM5-1B/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary/config_opencl_greedy.json",
                ],
            },
            "rhino": {
                "normal_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/normal/OpenBMB__MiniCPM5-1B/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary/config_opencl_greedy.json",
                ],
            },
        },
    },
    "qwen3-8b": {
        "name": "Qwen3-8B",
        "served_model": "qwen3-8b-pic",
        "paths": {
            "jetson": {
                "normal_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-llm-export/Qwen__Qwen3-8B/config_cuda_greedy.json",
                ],
                "pic_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/weight/Qwen__Qwen3-8B-pic-boundary/config_cuda_greedy.json",
                ],
            },
            "orangepi": {
                "normal_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/Qwen__Qwen3-8B/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/Qwen__Qwen3-8B-pic-boundary/config_opencl_greedy.json",
                ],
            },
            "rhino": {
                "normal_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/normal/Qwen__Qwen3-8B/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/pic/Qwen__Qwen3-8B-pic-boundary/config_opencl_greedy.json",
                ],
            },
        },
    },
}


def backend_label(name: str) -> str:
    return "CUDA" if str(name).lower() == "cuda" else "OpenCL"


def local_port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
    )


def ssh(
    device: dict[str, Any],
    script: str,
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "StrictHostKeyChecking=accept-new",
            device["ssh"],
            "bash",
            "-lc",
            script,
        ],
        check=check,
        timeout=timeout,
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def resolve_device_model(device_name: str, model_key: str) -> dict[str, Any]:
    if model_key not in MODELS:
        raise SystemExit(f"unknown model key {model_key}; expected one of: {', '.join(sorted(MODELS))}")
    model = MODELS[model_key]
    paths = model["paths"].get(device_name)
    if not paths:
        raise SystemExit(f"model {model_key} has no paths for device {device_name}")
    device = dict(DEVICES[device_name])
    device["model_key"] = model_key
    device["model_name"] = model["name"]
    device["served_model"] = f"{model['served_model']}-{device_name}"
    device["normal_config_candidates"] = list(paths["normal_config"])
    device["pic_config_candidates"] = list(paths["pic_config"])
    return device


def resolve_remote_file(device: dict[str, Any], candidates: list[str], label: str) -> str:
    quoted = " ".join(shlex.quote(path) for path in candidates)
    script = f"""
set -e
for path in {quoted}; do
  if [ -f "$path" ]; then
    printf '%s\\n' "$path"
    exit 0
  fi
done
exit 1
"""
    result = ssh(device, script, check=False, timeout=30)
    output = result.stdout or ""
    for candidate in candidates:
        if candidate in output:
            return candidate
    resolved = [line.strip() for line in output.splitlines() if line.strip()]
    if result.returncode == 0 and resolved:
        return resolved[-1]
    raise RuntimeError(f"{device['display']} {label} missing; candidates={candidates}")


def _parse_curl_json(output: str) -> tuple[int, dict[str, Any]]:
    marker = "\nHTTP_STATUS:"
    if marker not in output:
        return 0, {"error_body": output}
    body, status_text = output.rsplit(marker, 1)
    try:
        status = int(status_text.strip().splitlines()[-1])
    except Exception:
        status = 0
    try:
        parsed = json.loads(body) if body.strip() else {}
    except Exception:
        parsed = {"error_body": body}
    return status, parsed


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float = 600.0) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "--max-time",
            str(int(timeout)),
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            "-w",
            "\nHTTP_STATUS:%{http_code}",
            base_url.rstrip("/") + path,
        ],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    status, parsed = _parse_curl_json(proc.stdout)
    if proc.returncode != 0 and status == 0:
        parsed = {"error_body": proc.stdout, "curl_returncode": proc.returncode}
    return status, parsed, elapsed


def get_json(base_url: str, path: str, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "--max-time",
            str(int(timeout)),
            "-w",
            "\nHTTP_STATUS:%{http_code}",
            base_url.rstrip("/") + path,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return _parse_curl_json(proc.stdout)


def load_base_tokens(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    for key in ["full_prompt_token_ids", "doc_token_ids", "prelude_token_count", "suffix_token_count"]:
        if key not in obj:
            raise RuntimeError(f"missing {key} in {path}")
    return obj


def make_tokens(base: dict[str, Any], context: int) -> dict[str, Any]:
    prelude_n = int(base["prelude_token_count"])
    suffix_n = int(base["suffix_token_count"])
    if context <= prelude_n + suffix_n:
        raise ValueError(f"context {context} too small")
    full = list(map(int, base["full_prompt_token_ids"]))
    base_doc = list(map(int, base["doc_token_ids"]))
    prelude = full[:prelude_n]
    suffix = full[-suffix_n:]
    doc_n = context - prelude_n - suffix_n
    repeated: list[int] = []
    while len(repeated) < doc_n:
        repeated.extend(base_doc)
    doc = repeated[:doc_n]
    return {
        "full_prompt_token_ids": prelude + doc + suffix,
        "doc_token_ids": doc,
        "prompt_start": prelude_n,
        "source_start": 0,
        "token_count": doc_n,
        "full_prompt_token_count": context,
        "prelude_token_count": prelude_n,
        "suffix_token_count": suffix_n,
    }


def stable_doc_id(device: dict[str, Any], context: int, tokens: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(tokens["doc_token_ids"], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"bench-{device['model_key']}-{device['backend']}-ctx{context}-{digest}"


def shared_kv_dir(device: dict[str, Any]) -> str:
    return f"{device['remote_cache_root']}/shared_kv/{device['model_key']}"


def extract_pic_cache(response: dict[str, Any]) -> dict[str, Any]:
    if isinstance(response.get("pic_cache"), dict):
        return response["pic_cache"]
    data = response.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        pic_cache = data[0].get("pic_cache")
        if isinstance(pic_cache, dict):
            return pic_cache
    return {}


def extract_perf(response: dict[str, Any]) -> dict[str, Any]:
    if isinstance(response.get("performance"), dict):
        return response["performance"]
    data = response.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        performance = data[0].get("performance")
        if isinstance(performance, dict):
            return performance
    return {}


def summarize_chat(
    mode: str,
    budget: str,
    status: int,
    elapsed: float,
    response: dict[str, Any],
    context: int,
) -> dict[str, Any]:
    pic_cache = extract_pic_cache(response)
    precision = pic_cache.get("precision_recovery") if isinstance(pic_cache.get("precision_recovery"), dict) else {}
    metadata = precision.get("metadata") if isinstance(precision.get("metadata"), dict) else {}
    score_metadata = metadata.get("score_metadata") if isinstance(metadata.get("score_metadata"), dict) else {}
    perf = extract_perf(response)
    measured = float(perf.get("prefill_latency_s", 0.0) or 0.0) or float(elapsed)
    tps = (context / measured) if measured > 0 else ""
    output_mode = "pic-full-recompute" if mode == "full-compute" else mode
    return {
        "context_tokens": context,
        "mode": output_mode,
        "budget": budget,
        "prefill_latency_s": measured if measured > 0 else "",
        "prefill_tps": tps,
        "status": status,
        "execution_mode": precision.get("execution_mode", ""),
        "recompute_token_count": precision.get("recompute_token_count", ""),
        "reuse_token_count": metadata.get("reuse_token_count", ""),
        "selected_count": metadata.get("selected_count", score_metadata.get("selected_count", "")),
        "error": response.get("error", response.get("error_body", "")),
    }


def row_is_success(row: dict[str, Any]) -> bool:
    try:
        return float(row.get("prefill_latency_s", 0.0) or 0.0) > 0 and float(row.get("prefill_tps", 0.0) or 0.0) > 0
    except Exception:
        return False


def failure_row(
    device: dict[str, Any],
    context: int,
    phase: str,
    mode: str,
    budget: str,
    *,
    status: Any = "",
    error: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "device": device["ssh"],
        "device_name": device["display"],
        "model": device["model_name"],
        "backend": backend_label(device["backend"]),
        "context_tokens": context,
        "phase": phase,
        "mode": mode,
        "budget": budget,
        "status": status,
        "error": error,
    }
    if extra:
        row.update(extra)
    return row


def csv_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(key, "") for key in CSV_KEYS)


def gap_compare_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(key, "") for key in GAP_COMPARE_KEYS)


def cli_mode_to_summary_mode(mode: str) -> str:
    return "pic-full-recompute" if mode == "full-compute" else mode


def parse_modes(text: str | None) -> list[str]:
    if text is None or not str(text).strip():
        return list(MODE_CHOICES)
    modes: list[str] = []
    seen: set[str] = set()
    for item in str(text).split(","):
        mode = item.strip()
        if not mode:
            continue
        if mode not in MODE_CHOICES:
            raise SystemExit(f"unknown mode {mode}; expected one of: {', '.join(MODE_CHOICES)}")
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    if not modes:
        raise SystemExit("no modes selected")
    return modes


def parse_ratios(text: str | None) -> list[float]:
    if text is None or not str(text).strip():
        return list(RATIOS)
    ratios: list[float] = []
    seen: set[str] = set()
    allowed = {f"{item:.2f}" for item in RATIOS}
    for item in str(text).split(","):
        raw = item.strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise SystemExit(f"invalid ratio {raw}") from exc
        key = f"{value:.2f}"
        if key not in allowed:
            raise SystemExit(f"unsupported ratio {raw}; expected subset of: {', '.join(sorted(allowed))}")
        if key not in seen:
            ratios.append(value)
            seen.add(key)
    if not ratios:
        raise SystemExit("no ratios selected")
    return ratios


def selected_mode_specs(selected_modes: list[str], selected_ratios: list[float]) -> list[tuple[str, str, float | None]]:
    specs: list[tuple[str, str, float | None]] = []
    for mode in selected_modes:
        if mode in {"normal-full-recompute"}:
            continue
        if mode == "full-compute":
            specs.append(("full-compute", "full", None))
            continue
        if mode == "full-reuse":
            specs.append(("full-reuse", "0", None))
            continue
        if mode in RATIO_MODES:
            for ratio in selected_ratios:
                specs.append((mode, f"{ratio:.2f}", ratio))
            continue
        raise ValueError(f"unsupported selected mode {mode}")
    return specs


def row_matches_filters(row: dict[str, Any], selected_modes: list[str], selected_ratios: list[float]) -> bool:
    row_mode = str(row.get("mode", ""))
    allowed_modes = {cli_mode_to_summary_mode(mode) for mode in selected_modes}
    if row_mode not in allowed_modes:
        return False
    if row_mode in RATIO_MODES:
        try:
            budget = float(row.get("budget", ""))
        except (TypeError, ValueError):
            return False
        return any(abs(budget - ratio) < 1e-9 for ratio in selected_ratios)
    return True


def row_to_mode_spec(row: dict[str, Any]) -> tuple[str, str, float | None]:
    mode = str(row.get("mode", ""))
    budget = str(row.get("budget", ""))
    if mode == "pic-full-recompute":
        return ("full-compute", "full", None)
    if mode == "full-reuse":
        return ("full-reuse", "0", None)
    if mode in {"cacheblend", "epic"}:
        ratio = float(budget)
        return (mode, f"{ratio:.2f}", ratio)
    raise ValueError(f"unsupported mode row for chat spec: {row}")


def compute_missing_report(
    benchmark_csv: Path,
    target_model: str,
    requested_devices: list[str],
    selected_modes: list[str],
    selected_ratios: list[float],
) -> dict[str, Any]:
    with benchmark_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report: dict[str, Any] = {
        "benchmark_csv": str(benchmark_csv),
        "template_model": TEMPLATE_MODEL,
        "target_model": target_model,
        "selected_modes": selected_modes,
        "selected_ratios": selected_ratios,
        "devices": {},
    }
    for device in requested_devices:
        template_rows = [
            row
            for row in rows
            if row.get("model") == TEMPLATE_MODEL and row.get("device") == device
            and row_matches_filters(row, selected_modes, selected_ratios)
        ]
        template_by_key = {gap_compare_key(row): row for row in template_rows}
        existing = {
            gap_compare_key(row)
            for row in rows
            if row.get("model") == target_model and row.get("device") == device
            and row_matches_filters(row, selected_modes, selected_ratios)
        }
        missing_keys = sorted(template_by_key.keys() - existing)
        missing_contexts = sorted({int(item[2]) for item in missing_keys})
        report["devices"][device] = {
            "missing_count": len(missing_keys),
            "missing_contexts": missing_contexts,
            "missing_rows": [
                {
                    "device": device,
                    "device_display": template_by_key[item].get("device_display", ""),
                    "model": target_model,
                    "backend": item[0],
                    "frequency_profile": item[1],
                    "context_tokens": item[2],
                    "mode": item[3],
                    "budget": item[4],
                }
                for item in missing_keys
            ],
        }
    return report


def _port_pids_local(port: int) -> list[int]:
    commands = [
        ["lsof", f"-tiTCP:{int(port)}", "-sTCP:LISTEN"],
        ["fuser", "-n", "tcp", str(int(port))],
        ["sh", "-lc", f"ss -ltnpH 'sport = :{int(port)}' | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p'"],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            continue
        pids = []
        for token in result.stdout.replace("\n", " ").split():
            if token.strip().isdigit():
                pids.append(int(token))
        if pids:
            return sorted(set(pids))
    return []


def kill_local_port_users(port: int) -> None:
    pids = _port_pids_local(port)
    if not pids:
        return
    print(f"[port-cleanup] local tcp/{port} -> {pids}", flush=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in list(pids):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
            except PermissionError:
                pass
        time.sleep(1.0)
        if local_port_is_free(port):
            return
        pids = _port_pids_local(port)
        if not pids:
            return
    if not local_port_is_free(port):
        raise RuntimeError(f"failed to free local tcp/{port}; remaining listeners: {_port_pids_local(port)}")


def kill_remote_port_users(device: dict[str, Any], port: int) -> None:
    script = f"""
set +e
port='{int(port)}'
collect_pids() {{
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null
    return 0
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | tr ' ' '\\n' | sed '/^$/d'
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnpH "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p'
    return 0
  fi
  pgrep -f "pic_server.*--port $port" 2>/dev/null
}}
pids="$(collect_pids | sort -u | tr '\\n' ' ')"
if [ -n "$pids" ]; then
  kill $pids >/dev/null 2>&1 || true
  sleep 1
  pids="$(collect_pids | sort -u | tr '\\n' ' ')"
  if [ -n "$pids" ]; then
    kill -9 $pids >/dev/null 2>&1 || true
    sleep 1
  fi
fi
"""
    ssh(device, script, check=False)


def start_server(device: dict[str, Any], run_id: str, local_raw: Path, label: str) -> subprocess.Popen[str]:
    remote_run = f"{device['remote_cache_root']}/{run_id}"
    remote_kv = shared_kv_dir(device)
    remote_log = f"{remote_run}/pic_server_{label}.log"
    remote_pid = f"{remote_run}/pic_server_{label}.pid"
    kill_local_port_users(int(device["local_port"]))
    kill_remote_port_users(device, int(device["remote_port"]))
    env_parts = []
    if device["server_env"]:
        env_parts.append(device["server_env"])
    env_parts.extend(device.get("server_env_extra", []))
    extra_env = os.environ.get("PIC_SWEEP_SERVER_ENV_EXTRA", "").strip()
    if extra_env:
        env_parts.append(extra_env)
    env_prefix = (" ".join(env_parts) + " ") if env_parts else ""
    script = f"""
set -e
mkdir -p '{remote_run}' '{remote_kv}'
cd '{device["remote_work"]}'
nohup stdbuf -oL -eL env {env_prefix}LD_LIBRARY_PATH='{device["ld_library_path"]}':${{LD_LIBRARY_PATH:-}} \\
  '{device["artifact"]}/bin/pic_server' \\
  --config '{device["pic_config"]}' \\
  --host 127.0.0.1 --port {device["remote_port"]} \\
  --kv-cache-dir '{remote_kv}' --model '{device["served_model"]}' \\
  > '{remote_log}' 2>&1 &
echo $! > '{remote_pid}'
"""
    ssh(device, script)
    tunnel_log = local_raw / f"ssh_tunnel_{label}.log"
    base_url = f"http://127.0.0.1:{device['local_port']}"
    for attempt in range(2):
        tunnel = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-L",
                f"127.0.0.1:{device['local_port']}:127.0.0.1:{device['remote_port']}",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                device["ssh"],
            ],
            stdout=tunnel_log.open("w"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(0.5)
        if tunnel.poll() is not None:
            log_text = tunnel_log.read_text(errors="replace") if tunnel_log.exists() else ""
            if attempt == 0 and "Address already in use" in log_text:
                kill_local_port_users(int(device["local_port"]))
                continue
            raise RuntimeError(
                f"ssh tunnel exited before server became ready on local port {device['local_port']}\n{log_text}"
            )
        for _ in range(180):
            if tunnel.poll() is not None:
                log_text = tunnel_log.read_text(errors="replace") if tunnel_log.exists() else ""
                raise RuntimeError(
                    f"ssh tunnel exited before server became ready on local port {device['local_port']}\n{log_text}"
                )
            try:
                status, _ = get_json(base_url, "/healthz")
                if status == 200:
                    return tunnel
            except Exception:
                pass
            time.sleep(1)
    log_tail = ssh(device, f"tail -n 120 '{remote_log}' || true", check=False).stdout
    tunnel.terminate()
    raise RuntimeError(f"{device['display']} server not ready\n{log_tail}")


def stop_server(device: dict[str, Any], tunnel: subprocess.Popen[str] | None) -> None:
    if tunnel is not None:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel.kill()
    kill_local_port_users(int(device["local_port"]))
    kill_remote_port_users(device, int(device["remote_port"]))


def run_normal(device: dict[str, Any], ctx: int, run_id: str, local_raw: Path, rep: int) -> dict[str, Any]:
    remote_run = f"{device['remote_cache_root']}/{run_id}/normal"
    remote_json = f"{remote_run}/normal_ctx{ctx}.json"
    remote_log = f"{remote_run}/normal_ctx{ctx}.log"
    bench_env = (device.get("bench_env", "").strip() + " ") if device.get("bench_env", "").strip() else ""
    script = f"""
set -e
mkdir -p '{remote_run}'
cd "$(dirname '{device["normal_config"]}')"
env {bench_env}LD_LIBRARY_PATH='{device["ld_library_path"]}':${{LD_LIBRARY_PATH:-}} \\
  '{device["artifact"]}/bin/llm_bench' -m '{device["normal_config"]}' \\
  -a '{device["backend"]}' -p {ctx} -n 0 -rep {rep} -load false -j '{remote_json}' > '{remote_log}' 2>&1
cat '{remote_json}'
"""
    result = ssh(device, script)
    try:
        text = result.stdout[result.stdout.find("{"):]
        obj = json.loads(text)
    except Exception:
        obj = {"parse_error": result.stdout}
    write_json(local_raw / f"normal_ctx{ctx}.json", obj)
    prefill = None
    for item in obj.get("results", []):
        if isinstance(item, dict) and item.get("type") == "prefill":
            prefill = item
            break
    tps = float(prefill.get("tps")) if prefill and prefill.get("tps") else 0.0
    latency = (ctx / tps) if tps > 0 else ""
    return {
        "device": device["device_key"],
        "device_display": device["display"],
        "model": device["model_name"],
        "backend": backend_label(device["backend"]),
        "frequency_profile": "max",
        "context_tokens": ctx,
        "mode": "normal-full-recompute",
        "budget": "full",
        "prefill_latency_s": latency,
        "prefill_tps": tps if tps > 0 else "",
    }


def chat_payload(tokens: dict[str, Any], doc_id: str, mode: str, ratio: float | None, model: str) -> dict[str, Any]:
    pic_cache: dict[str, Any] = {
        "id": f"{doc_id}-{mode}-{ratio if ratio is not None else 'full'}-{time.time_ns()}",
        "text_cache_refs": [{"id": doc_id}],
        "selection_algorithm": mode,
        "pic_recompute_score_layer_idx": 1,
    }
    if ratio is not None:
        pic_cache["pic_recompute_ratio"] = ratio
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


def run_chat(
    base_url: str,
    local_raw: Path,
    tokens: dict[str, Any],
    doc_id: str,
    mode: str,
    budget: str,
    ratio: float | None,
    model: str,
    prefix: str,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    post_json(base_url, "/reset", {"reset": True}, timeout=30)
    payload = chat_payload(tokens, doc_id, mode, ratio, model)
    status, response, elapsed = post_json(base_url, "/v1/chat/completions", payload)
    raw = {
        "status": status,
        "latency_s": elapsed,
        "payload_meta": {"mode": mode, "budget": budget},
        "response": response,
    }
    write_json(local_raw / f"{prefix}_chat_{mode}_{budget.replace('.', 'p')}.json", raw)
    summary = summarize_chat(mode, budget, status, elapsed, response, int(tokens["full_prompt_token_count"]))
    return status, response, summary


def tail_server_log(device: dict[str, Any], run_id: str, label: str, local_raw: Path) -> str:
    remote_run = f"{device['remote_cache_root']}/{run_id}"
    log = ""
    try:
        log = ssh(
            device,
            f"tail -n 400 '{remote_run}/pic_server_{label}.log' 2>/dev/null || true",
            check=False,
            timeout=30,
        ).stdout
    except subprocess.TimeoutExpired:
        log = f"[timeout] tail remote log {remote_run}/pic_server_{label}.log"
    (local_raw / f"pic_server_{label}.log").write_text(log)
    return log


def run_device(
    device_name: str,
    contexts: list[int],
    args: argparse.Namespace,
    fieldnames: list[str],
    summary_csv: Path,
    failures_jsonl: Path,
    missing_rows_by_context: dict[int, list[dict[str, Any]]] | None = None,
) -> None:
    device = resolve_device_model(device_name, args.model_key)
    device["device_key"] = device_name
    device["normal_config"] = resolve_remote_file(device, device["normal_config_candidates"], "normal_config")
    device["pic_config"] = resolve_remote_file(device, device["pic_config_candidates"], "pic_config")
    device["server_env_extra"] = list(args.server_env)
    out_dir = Path(args.output_dir) / args.run_id
    local_raw = out_dir / "raw" / device_name
    local_raw.mkdir(parents=True, exist_ok=True)
    write_json(local_raw / "device.json", device)
    base_tokens = load_base_tokens(Path(args.base_tokens))

    for ctx in contexts:
        label = f"ctx{ctx}"
        tunnel: subprocess.Popen[str] | None = None
        requested_rows = missing_rows_by_context.get(int(ctx), []) if missing_rows_by_context is not None else []
        if missing_rows_by_context is not None:
            requested_rows = [row for row in requested_rows if row_matches_filters(row, args.selected_modes, args.selected_ratios)]
        normal_needed = (
            "normal-full-recompute" in args.selected_modes
            if missing_rows_by_context is None
            else any(str(row.get("mode", "")) == "normal-full-recompute" for row in requested_rows)
        )
        requested_mode_specs = []
        if missing_rows_by_context is None:
            requested_mode_specs = selected_mode_specs(args.selected_modes, args.selected_ratios)
        else:
            seen_specs: set[tuple[str, str, float | None]] = set()
            for row in requested_rows:
                if str(row.get("mode", "")) == "normal-full-recompute":
                    continue
                spec = row_to_mode_spec(row)
                if spec not in seen_specs:
                    requested_mode_specs.append(spec)
                    seen_specs.add(spec)

        if not normal_needed and not requested_mode_specs:
            print(f"[skip] {device_name} ctx={ctx} no missing rows", flush=True)
            continue
        try:
            if normal_needed:
                normal_row = run_normal(device, int(ctx), args.run_id, local_raw, int(args.normal_rep))
                if row_is_success(normal_row):
                    append_csv(summary_csv, normal_row, fieldnames)
                else:
                    append_jsonl(
                        failures_jsonl,
                        failure_row(device, ctx, "normal", "normal-full-recompute", "full", error="normal llm_bench parse failed"),
                    )
                    continue

            if not requested_mode_specs:
                continue

            tokens = make_tokens(base_tokens, int(ctx))
            doc_id = stable_doc_id(device, int(ctx), tokens)
            write_json(local_raw / f"tokens_ctx{ctx}.json", tokens)
            write_json(local_raw / f"doc_id_ctx{ctx}.json", {"id": doc_id, "shared_cache_dir": shared_kv_dir(device)})

            tunnel = start_server(device, args.run_id, local_raw, label)
            base_url = f"http://127.0.0.1:{device['local_port']}"

            post_json(base_url, "/reset", {"reset": True}, timeout=30)
            status, response, elapsed = post_json(
                base_url,
                "/v1/prefill/text",
                {
                    "id": doc_id,
                    "type": "text",
                    "token_ids": tokens["doc_token_ids"],
                    "force": bool(args.force_cache_build),
                },
                timeout=1800,
            )
            write_json(
                local_raw / f"prefill_text_ctx{ctx}.json",
                {
                    "status": status,
                    "latency_s": elapsed,
                    "request_meta": {
                        "doc_id": doc_id,
                        "shared_cache_dir": shared_kv_dir(device),
                        "force_cache_build": bool(args.force_cache_build),
                    },
                    "cache_status": response.get("cache_status", "") if isinstance(response, dict) else "",
                    "cache_hit": response.get("cache_hit", "") if isinstance(response, dict) else "",
                    "response": response,
                },
            )
            if status != 200:
                append_jsonl(
                    failures_jsonl,
                    failure_row(
                        device,
                        ctx,
                        "prefill_text",
                        "prefill_text",
                        "full",
                        status=status,
                        error=response.get("error", response.get("error_body", "")),
                    ),
                )
                continue

            measure_specs = list(requested_mode_specs)
            if not args.no_warm:
                warm_success: list[tuple[str, str, float | None]] = []
                for mode, budget, ratio in requested_mode_specs:
                    print(f"[warm] {device_name} ctx={ctx} {mode} {budget}", flush=True)
                    warm_status, warm_response, _ = run_chat(
                        base_url,
                        local_raw,
                        tokens,
                        doc_id,
                        mode,
                        budget,
                        ratio,
                        device["served_model"],
                        f"warm_ctx{ctx}",
                    )
                    if warm_status == 200:
                        warm_success.append((mode, budget, ratio))
                    else:
                        append_jsonl(
                            failures_jsonl,
                            failure_row(
                                device,
                                ctx,
                                "warm",
                                "pic-full-recompute" if mode == "full-compute" else mode,
                                budget,
                                status=warm_status,
                                error=warm_response.get("error", warm_response.get("error_body", "")),
                            ),
                        )
                measure_specs = warm_success
                if str(device.get("backend", "")).lower() == "opencl" and warm_success:
                    tune_status, tune_response, tune_elapsed = post_json(base_url, "/v1/tune/update_cache", {}, timeout=120)
                    write_json(
                        local_raw / f"update_runtime_cache_ctx{ctx}.json",
                        {"status": tune_status, "latency_s": tune_elapsed, "response": tune_response},
                    )

            for mode, budget, ratio in measure_specs:
                print(f"[measure] {device_name} ctx={ctx} {mode} {budget}", flush=True)
                measure_status, measure_response, row = run_chat(
                    base_url,
                    local_raw,
                    tokens,
                    doc_id,
                    mode,
                    budget,
                    ratio,
                    device["served_model"],
                    f"ctx{ctx}",
                )
                row.update(
                    {
                        "device": device_name,
                        "device_display": device["display"],
                        "model": device["model_name"],
                        "backend": backend_label(device["backend"]),
                        "frequency_profile": "max",
                    }
                )
                if row_is_success(row) and measure_status == 200:
                    append_csv(summary_csv, row, fieldnames)
                else:
                    append_jsonl(
                        failures_jsonl,
                        failure_row(
                            device,
                            ctx,
                            "measure",
                            row["mode"],
                            budget,
                            status=measure_status,
                            error=measure_response.get("error", measure_response.get("error_body", "")),
                            extra={"execution_mode": row.get("execution_mode", "")},
                        ),
                    )
        except Exception as exc:
            append_jsonl(
                failures_jsonl,
                failure_row(device, ctx, "exception", "context", "full", error=str(exc)),
            )
        finally:
            tail_server_log(device, args.run_id, label, local_raw)
            stop_server(device, tunnel)


def parse_contexts(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",") if item.strip()]


def print_gap_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="jetson,orangepi,rhino")
    parser.add_argument("--contexts", default="512,1024,1536,2048,2560,3072")
    parser.add_argument("--model-key", default="llama3.2-1b", choices=sorted(MODELS))
    parser.add_argument("--modes", default=",".join(MODE_CHOICES))
    parser.add_argument("--ratios", default=",".join(f"{ratio:.2f}" for ratio in RATIOS))
    parser.add_argument(
        "--server-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE env assignment for remote pic_server; may be repeated.",
    )
    parser.add_argument("--run-id", default="prefill_latency_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", default=str(MNN_ROOT / ".cache/latency_budget_20260625"))
    parser.add_argument("--base-tokens", default=str(DEFAULT_BASE_TOKENS))
    parser.add_argument("--benchmark-csv", default="")
    parser.add_argument("--only-missing-contexts", action="store_true")
    parser.add_argument("--print-gap-only", action="store_true")
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--force-cache-build", action="store_true")
    parser.add_argument("--normal-rep", type=int, default=3)
    args = parser.parse_args()

    devices = [item.strip() for item in str(args.devices).split(",") if item.strip()]
    base_contexts = parse_contexts(args.contexts)
    args.selected_modes = parse_modes(args.modes)
    args.selected_ratios = parse_ratios(args.ratios)
    fieldnames = CSV_KEYS + ["prefill_latency_s", "prefill_tps"]

    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    failures_jsonl = out_dir / "failures.jsonl"

    target_model = MODELS[args.model_key]["name"]
    gap_report = None
    device_contexts: dict[str, list[int]] = {device: list(base_contexts) for device in devices}
    device_missing_rows: dict[str, dict[int, list[dict[str, Any]]]] = {}
    if args.benchmark_csv:
        gap_report = compute_missing_report(
            Path(args.benchmark_csv),
            target_model,
            devices,
            args.selected_modes,
            args.selected_ratios,
        )
        write_json(out_dir / "gap_report.json", gap_report)
        if args.only_missing_contexts:
            device_contexts = {}
            for device in devices:
                rows = gap_report["devices"][device]["missing_rows"]
                grouped: dict[int, list[dict[str, Any]]] = {}
                for row in rows:
                    ctx = int(row["context_tokens"])
                    grouped.setdefault(ctx, []).append(row)
                if grouped:
                    device_contexts[device] = sorted(grouped)
                    device_missing_rows[device] = grouped
        if args.print_gap_only:
            print_gap_report(gap_report)
            return 0

    write_json(
        out_dir / "run_config.json",
        {
            "devices": list(device_contexts),
            "model_key": args.model_key,
            "model": target_model,
            "contexts": device_contexts,
            "modes": args.selected_modes,
            "ratios": args.selected_ratios,
            "shared_cache": True,
            "ports": {
                device: {
                    "remote_port": DEVICES[device]["remote_port"],
                    "local_port": DEVICES[device]["local_port"],
                    "port_strategy": "fixed-kill-old",
                }
                for device in device_contexts
            },
            "force_cache_build": bool(args.force_cache_build),
            "benchmark_csv": args.benchmark_csv,
            "only_missing_contexts": bool(args.only_missing_contexts),
            "base_tokens": str(args.base_tokens),
            "server_env": list(args.server_env),
        },
    )

    for device, contexts in device_contexts.items():
        if not contexts:
            print(f"[skip] {device} no missing contexts for {target_model}", flush=True)
            continue
        run_device(
            device,
            contexts,
            args,
            fieldnames,
            summary_csv,
            failures_jsonl,
            device_missing_rows.get(device),
        )

    print(summary_csv)
    if failures_jsonl.exists():
        print(failures_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
