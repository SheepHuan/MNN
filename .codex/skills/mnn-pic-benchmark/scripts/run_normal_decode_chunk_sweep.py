#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
NO_PROXY_TARGETS = "192.168.101.113,192.168.101.227,127.0.0.1,localhost"

DEVICES: dict[str, dict[str, Any]] = {
    "orangepi": {
        "display": "Orange Pi 5 Plus",
        "ssh": "orangepi@192.168.101.113",
        "artifact": "/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus",
        "ld_library_path": "/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/lib",
        "bench_env": "LD_PRELOAD=/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/lib/libMNN_CL.so",
        "backend": "opencl",
        "remote_cache_root": "/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep",
        "remote_run_root": "/mnt/ssd/code/.cache/mnn_opencl_pic/decode_chunk",
    },
    "rhino": {
        "display": "Rhino Pi X1",
        "ssh": "aidlux@192.168.101.227",
        "artifact": "/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl",
        "ld_library_path": "/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/lib:/usr/lib:/usr/lib/aarch64-linux-gnu",
        "bench_env": "LD_PRELOAD=/usr/lib/libOpenCL_adreno.so",
        "backend": "opencl",
        "remote_cache_root": "/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep",
        "remote_run_root": "/mnt/nvme/mnn_pic_opencl/decode_chunk",
    },
}

MODELS: dict[str, dict[str, Any]] = {
    "llama3.2-3b": {
        "name": "Llama3.2 3B",
        "normal_config": {
            "orangepi": [
                "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/AI-ModelScope__Llama-3___2-3B-Instruct/config_opencl_greedy.json",
                "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/AI-ModelScope__Llama-3.2-3B-Instruct/config_opencl_greedy.json",
            ],
            "rhino": [
                "/mnt/nvme/mnn_pic_opencl/models/normal/AI-ModelScope__Llama-3___2-3B-Instruct/config_opencl_greedy.json",
                "/mnt/nvme/mnn_pic_opencl/models/normal/AI-ModelScope__Llama-3.2-3B-Instruct/config_opencl_greedy.json",
            ],
        },
    },
    "minicpm5-1b": {
        "name": "MiniCPM5-1B",
        "normal_config": {
            "orangepi": ["/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/OpenBMB__MiniCPM5-1B/config_opencl_greedy.json"],
            "rhino": ["/mnt/nvme/mnn_pic_opencl/models/normal/OpenBMB__MiniCPM5-1B/config_opencl_greedy.json"],
        },
    },
    "qwen3-4b": {
        "name": "Qwen3-4B",
        "normal_config": {
            "orangepi": ["/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/Qwen__Qwen3-4B/config_opencl_greedy.json"],
            "rhino": ["/mnt/nvme/mnn_pic_opencl/models/normal/Qwen__Qwen3-4B/config_opencl_greedy.json"],
        },
    },
}

FIELDS = [
    "device",
    "device_display",
    "model",
    "backend",
    "frequency_profile",
    "context_tokens",
    "mode",
    "budget",
    "decode_selector",
    "repair_tokens",
    "generated_tokens",
    "decode_latency_s",
    "decode_tpot_ms",
    "decode_tps",
    "benchmark_status",
]


def csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def run_local(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NO_PROXY"] = NO_PROXY_TARGETS
    env["no_proxy"] = NO_PROXY_TARGETS
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=check, timeout=timeout)


def ssh(device: dict[str, Any], script: str, *, timeout: int) -> subprocess.CompletedProcess[str]:
    return run_local(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", str(device["ssh"]), script],
        timeout=timeout,
    )


def first_existing_config(device: dict[str, Any], model_key: str) -> str:
    candidates = MODELS[model_key]["normal_config"][device["device_key"]]
    test = " || ".join(f"test -f {shlex.quote(path)} && echo {shlex.quote(path)}" for path in candidates)
    proc = ssh(device, f"set -e; ({test}) | head -1", timeout=20)
    config = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not config:
        raise RuntimeError(f"no config found for {device['device_key']} {model_key}: {candidates}")
    return config


def remote_env(device: dict[str, Any]) -> str:
    runtime_cache = f"{device['remote_cache_root']}/runtime_cache/opencl"
    parts = []
    if device.get("bench_env"):
        parts.append(str(device["bench_env"]))
    parts.append(f"MNN_LLM_RUNTIME_CACHE_DIR={shlex.quote(runtime_cache)}")
    parts.append(f"LD_LIBRARY_PATH={shlex.quote(str(device['ld_library_path']))}:${{LD_LIBRARY_PATH:-}}")
    return " ".join(parts)


def remote_script(device: dict[str, Any], config: str, ctx: int, generated: int, step_tokens: int, run_id: str) -> str:
    art = str(device["artifact"])
    runtime_cache = f"{device['remote_cache_root']}/runtime_cache/opencl"
    run_dir = f"{device['remote_run_root']}/{run_id}/{device['model_key']}/ctx{ctx}/step{step_tokens}"
    env = remote_env(device)
    return f"""
set -e
ART={shlex.quote(art)}
CFG={shlex.quote(config)}
RUN={shlex.quote(run_dir)}
RUNTIME_CACHE={shlex.quote(runtime_cache)}
mkdir -p "$RUN" "$RUNTIME_CACHE"
cd "$(dirname "$CFG")"
if [ -e tmp ] && [ ! -L tmp ]; then
  mv tmp "tmp.decode_chunk_bak_$(date +%Y%m%d%H%M%S)"
fi
ln -sfn "$RUNTIME_CACHE" tmp
env {env} "$ART/bin/normal_decode_kv_bench" -m "$CFG" -a {shlex.quote(str(device['backend']))} \\
  -p {ctx} -n {generated} --step-tokens {step_tokens} -j "$RUN/result.json" \\
  > "$RUN/run.log" 2>&1
cat "$RUN/result.json"
""".strip()


def parse_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"missing JSON in output: {text[-400:]}")
    return json.loads(text[start:])


def run_case(args: argparse.Namespace, device: dict[str, Any], model_key: str, ctx: int, step_tokens: int) -> dict[str, str]:
    device["model_key"] = model_key
    config = first_existing_config(device, model_key)
    print(f"[measure] {device['device_key']} {model_key} ctx={ctx} step={step_tokens}", flush=True)
    proc = ssh(
        device,
        remote_script(device, config, ctx, args.generated_tokens, step_tokens, args.run_id),
        timeout=args.remote_timeout_sec,
    )
    result = parse_json(proc.stdout)
    generated = int(result.get("generated_tokens") or args.generated_tokens)
    tpot = float(result.get("decode_tpot_ms") or 0.0)
    tps = float(result.get("decode_tps") or 0.0)
    return {
        "device": str(device["device_key"]),
        "device_display": str(device["display"]),
        "model": str(MODELS[model_key]["name"]),
        "backend": str(device["backend"]).upper(),
        "frequency_profile": "cpu=max,gpu=max,ddr=max",
        "context_tokens": str(ctx),
        "mode": "normal-decode-load-kv-chunk-input",
        "budget": str(args.generated_tokens),
        "decode_selector": f"true-normal-llm-input{step_tokens}",
        "repair_tokens": str(max(0, step_tokens - 1)),
        "generated_tokens": str(generated),
        "decode_latency_s": f"{tpot * generated / 1000.0:.6f}",
        "decode_tpot_ms": f"{tpot:.6f}",
        "decode_tps": f"{tps:.6f}",
        "benchmark_status": "ok",
    }


def merge_csv(path: Path, new_rows: list[dict[str, str]]) -> None:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
    key_fields = ["device", "model", "context_tokens", "mode", "budget", "decode_selector", "repair_tokens", "generated_tokens"]
    by_key = {tuple(row.get(field, "") for field in key_fields): row for row in rows}
    for row in new_rows:
        by_key[tuple(row.get(field, "") for field in key_fields)] = row
    merged = sorted(
        by_key.values(),
        key=lambda r: (
            r.get("device", ""),
            r.get("model", ""),
            int(float(r.get("context_tokens") or 0)),
            r.get("mode", ""),
            int(float(r.get("repair_tokens") or 0)),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run normal LLM chunk-input decode TPS sweep.")
    parser.add_argument("--devices", default="orangepi,rhino")
    parser.add_argument("--model-keys", default="llama3.2-3b,minicpm5-1b,qwen3-4b")
    parser.add_argument("--contexts", default="512,1024,1536,2048")
    parser.add_argument("--step-tokens", default="2,4,6,8")
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--remote-timeout-sec", type=int, default=900)
    parser.add_argument("--run-id", default="normal_decode_chunk_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--benchmark-csv", default=str(REPO / "benchmark_decode.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    devices = csv_list(args.devices)
    models = csv_list(args.model_keys)
    contexts = [int(item) for item in csv_list(args.contexts)]
    steps = [int(item) for item in csv_list(args.step_tokens)]
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for device_key in devices:
        device = dict(DEVICES[device_key])
        device["device_key"] = device_key
        for model_key in models:
            for ctx in contexts:
                for step in steps:
                    try:
                        rows.append(run_case(args, device, model_key, ctx, step))
                    except Exception as exc:
                        message = f"{device_key} {model_key} ctx={ctx} step={step}: {exc}"
                        print(f"[failure] {message}", flush=True)
                        failures.append(message)
    merge_csv(Path(args.benchmark_csv), rows)
    print(f"[benchmark] {args.benchmark_csv} merged_rows={len(rows)} failures={len(failures)}")
    if failures:
        for item in failures:
            print(item)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
