#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import ProxyHandler, Request, build_opener


REPO = Path(__file__).resolve().parents[4]
POWER_ROOT = REPO / ".codex/skills/mnn-pic-benchmark/power"
POWER_API_URL = "http://192.168.101.14:8766"
POWER_REP_INTERVAL_SEC = 3.0
NO_PROXY_TARGETS = "192.168.101.14,192.168.101.113,192.168.101.227,127.0.0.1,localhost"
POWER_OPENER = build_opener(ProxyHandler({}))

DEVICES: dict[str, dict[str, Any]] = {
    "orangepi": {
        "display": "Orange Pi 5 Plus",
        "ssh": "orangepi@192.168.101.113",
        "artifact": "/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus",
        "ld_library_path": "/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/lib",
        "bench_env": "LD_PRELOAD=/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/lib/libMNN_CL.so",
        "backend": "opencl",
        "remote_cache_root": "/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep",
        "remote_run_root": "/mnt/ssd/code/.cache/mnn_opencl_pic/decode_energy",
        "power_start": {
            "device": "orangepi5plus",
            "monitor_type": "blu",
            "serial": "F96FBDBA05B0",
            "sample_rate_hz": 10000,
            "voltage_mv": 4800,
        },
    },
    "rhino": {
        "display": "Rhino Pi X1",
        "ssh": "aidlux@192.168.101.227",
        "artifact": "/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl",
        "ld_library_path": "/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/lib:/usr/lib:/usr/lib/aarch64-linux-gnu",
        "bench_env": "LD_PRELOAD=/usr/lib/libOpenCL_adreno.so",
        "backend": "opencl",
        "remote_cache_root": "/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep",
        "remote_run_root": "/mnt/nvme/mnn_pic_opencl/decode_energy",
        "power_start": {
            "monitor_type": "df",
            "serial": "1A5D43",
        },
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

BENCHMARK_FIELDS = [
    "device",
    "device_display",
    "model",
    "backend",
    "frequency_note",
    "context_tokens",
    "mode",
    "budget",
    "generated_tokens",
    "test_date",
    "mj_per_token",
    "decode_tpot_ms",
    "decode_tps",
    "energy_per_inference_mj",
    "duration_per_inference_s",
    "analysis_status",
    "error",
]

DECODE_BENCHMARK_FIELDS = [
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


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def run_local(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NO_PROXY"] = NO_PROXY_TARGETS
    env["no_proxy"] = NO_PROXY_TARGETS
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=check)


def ssh(device: dict[str, Any], script: str, *, timeout: int) -> subprocess.CompletedProcess[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", str(device["ssh"]), script]
    env = os.environ.copy()
    env["NO_PROXY"] = NO_PROXY_TARGETS
    env["no_proxy"] = NO_PROXY_TARGETS
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=True,
        timeout=timeout,
    )


def post_json(api_url: str, path: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    os.environ["NO_PROXY"] = NO_PROXY_TARGETS
    os.environ["no_proxy"] = NO_PROXY_TARGETS
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        urljoin(api_url.rstrip("/") + "/", path.lstrip("/")),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with POWER_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def download_power_csv(api_url: str, raw_url: str, output: Path) -> None:
    os.environ["NO_PROXY"] = NO_PROXY_TARGETS
    os.environ["no_proxy"] = NO_PROXY_TARGETS
    if raw_url.startswith("/"):
        url = urljoin(api_url.rstrip("/") + "/", raw_url.lstrip("/"))
    else:
        url = raw_url
    req = Request(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    with POWER_OPENER.open(req, timeout=180) as resp:
        output.write_bytes(resp.read())


def render_power_csv(path: Path) -> Path:
    png = path.with_suffix(".png")
    run_local([
        "python3",
        str(REPO / ".codex/skills/mnn-pic-benchmark/scripts/render_power_csv.py"),
        str(path),
        "-o",
        str(png),
    ])
    return png


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


def chunk_mode(step_tokens: int) -> str:
    if step_tokens <= 1:
        return "normal-decode-load-kv"
    return f"normal-decode-load-kv-input{step_tokens}"


def decode_csv_mode(step_tokens: int) -> str:
    if step_tokens <= 1:
        return "normal-decode-load-kv"
    return "normal-decode-load-kv-chunk-input"


def remote_start_helper_script(device: dict[str, Any], config: str, ctx: int, generated: int, step_tokens: int, run_id: str) -> str:
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
  mv tmp "tmp.decode_energy_bak_$(date +%Y%m%d%H%M%S)"
fi
ln -sfn "$RUNTIME_CACHE" tmp
rm -f "$RUN/ready" "$RUN/go" "$RUN/result.json" "$RUN/helper.log"
env {env} "$ART/bin/normal_decode_kv_bench" -m "$CFG" -a {shlex.quote(str(device['backend']))} \\
  -p {ctx} -n {generated} --step-tokens {step_tokens} --ready-file "$RUN/ready" --go-file "$RUN/go" \\
  -j "$RUN/result.json" > "$RUN/helper.log" 2>&1 &
echo $!
""".strip()


def remote_wait_ready_script(device: dict[str, Any], ctx: int, step_tokens: int, run_id: str, pid: str, timeout_sec: int) -> str:
    run_dir = f"{device['remote_run_root']}/{run_id}/{device['model_key']}/ctx{ctx}/step{step_tokens}"
    return f"""
set -e
RUN={shlex.quote(run_dir)}
PID={shlex.quote(pid)}
deadline=$(( $(date +%s) + {timeout_sec} ))
while [ ! -f "$RUN/ready" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    cat "$RUN/helper.log" >&2 || true
    exit 10
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    cat "$RUN/helper.log" >&2 || true
    exit 11
  fi
  sleep 0.1
done
echo ready
""".strip()


def remote_trigger_decode_script(device: dict[str, Any], ctx: int, step_tokens: int, run_id: str, pid: str) -> str:
    run_dir = f"{device['remote_run_root']}/{run_id}/{device['model_key']}/ctx{ctx}/step{step_tokens}"
    return f"""
set -e
RUN={shlex.quote(run_dir)}
PID={shlex.quote(pid)}
touch "$RUN/go"
while kill -0 "$PID" 2>/dev/null; do
  sleep 0.05
done
if [ ! -f "$RUN/result.json" ]; then
  cat "$RUN/helper.log" >&2 || true
  exit 12
fi
cat "$RUN/result.json"
""".strip()


def parse_decode_result(text: str, generated: int) -> dict[str, Any]:
    start = text.find("{")
    obj = json.loads(text[start:] if start >= 0 else text)
    item = {}
    for candidate in obj.get("results", []):
        if isinstance(candidate, dict) and int(candidate.get("generate_len", 0) or 0) == generated:
            item = candidate
            break
    if not item and obj.get("results"):
        item = obj["results"][0]
    decode_tps = float(item.get("decode_tps") or 0.0)
    return {
        "prefill_tps": float(item.get("prefill_tps") or 0.0),
        "prefill_latency_s": (float(item.get("prompt_len") or 0.0) / float(item.get("prefill_tps") or 1.0)),
        "decode_tps": decode_tps,
        "decode_tpot_ms": (1000.0 / decode_tps) if decode_tps > 0 else "",
        "load_kv_time_s": float(item.get("loading_time") or 0.0),
    }


def parse_helper_result(text: str) -> dict[str, Any]:
    start = text.find("{")
    obj = json.loads(text[start:] if start >= 0 else text)
    return {
        "prefill_tps": "",
        "prefill_latency_s": float(obj.get("prefill_wall_s") or 0.0),
        "decode_tps": float(obj.get("decode_tps") or 0.0),
        "decode_tpot_ms": float(obj.get("decode_tpot_ms") or 0.0),
        "load_kv_time_s": 0.0,
        "decode_wall_s": float(obj.get("decode_wall_s") or 0.0),
        "generated_tokens_actual": int(obj.get("generated_tokens") or 0),
    }


def write_kv_txt(path: Path, metadata: dict[str, Any]) -> None:
    lines = []
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_manifest(process_dir: Path, metadata: dict[str, Any]) -> None:
    manifest = process_dir / "manifest.tsv"
    fields = [
        "power_uuid",
        "power_rep_total",
        "power_rep_interval_sec",
        "power_warmup_sec",
        "power_cooldown_sec",
        "device",
        "device_display",
        "backend",
        "frequency_note",
        "model_key",
        "model",
        "context_tokens",
        "generated_tokens",
        "algorithm",
        "case_label",
        "budget",
        "prefill_latency_s",
        "prefill_tps",
        "decode_tpot_ms",
        "decode_tps",
        "load_kv_time_s",
        "power_csv",
        "power_png",
        "power_txt",
        "power_run_id",
        "status",
        "error",
    ]
    exists = manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: metadata.get(key, "") for key in fields})


def process_dir(device_key: str, run_id: str) -> Path:
    now = time.localtime()
    return POWER_ROOT / device_key / time.strftime("%Y-%m-%d/%H/%M", now) / run_id


def run_case(args: argparse.Namespace, device: dict[str, Any], model_key: str, ctx: int, step_tokens: int) -> dict[str, Any]:
    device["model_key"] = model_key
    config = first_existing_config(device, model_key)
    print(f"[prepare] {device['device_key']} {model_key} ctx={ctx} step={step_tokens}", flush=True)
    start_helper = ssh(
        device,
        remote_start_helper_script(device, config, ctx, args.generated_tokens, step_tokens, args.run_id),
        timeout=30,
    )
    helper_pid = start_helper.stdout.strip().splitlines()[-1]
    ssh(device, remote_wait_ready_script(device, ctx, step_tokens, args.run_id, helper_pid, args.remote_timeout_sec), timeout=args.remote_timeout_sec + 10)

    out_dir = process_dir(str(device["device_key"]), args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_uuid = uuid.uuid4().hex
    csv_path = out_dir / f"{case_uuid}.csv"
    txt_path = out_dir / f"{case_uuid}.txt"
    png_path = out_dir / f"{case_uuid}.png"
    start_json = out_dir / f"{case_uuid}.start.json"
    stop_json = out_dir / f"{case_uuid}.stop.json"
    start_resp_json = out_dir / f"{case_uuid}.start_response.json"
    stop_resp_json = out_dir / f"{case_uuid}.stop_response.json"

    metadata: dict[str, Any] = {
        "power_uuid": case_uuid,
        "power_rep_total": 1,
        "power_rep_interval_sec": POWER_REP_INTERVAL_SEC,
        "power_warmup_sec": args.power_warmup_sec,
        "power_cooldown_sec": args.power_cooldown_sec,
        "device": device["device_key"],
        "device_display": device["display"],
        "backend": str(device["backend"]).upper(),
        "frequency_note": "cpu=max,gpu=max,ddr=max",
        "run_id": args.run_id,
        "model_key": model_key,
        "model": MODELS[model_key]["name"],
        "context_tokens": ctx,
        "generated_tokens": args.generated_tokens,
        "step_tokens": step_tokens,
        "algorithm": chunk_mode(step_tokens),
        "case_label": f"{model_key}_ctx{ctx}_input{step_tokens}_normal_decode_load_kv",
        "budget": str(args.generated_tokens),
        "power_txt": str(txt_path),
        "power_csv": str(csv_path),
        "power_png": str(png_path),
        "normal_config": config,
        "helper_pid": helper_pid,
    }
    write_kv_txt(txt_path, metadata)

    start_payload = dict(device["power_start"])
    start_payload["max_duration_sec"] = args.power_max_duration_sec
    start_json.write_text(json.dumps(start_payload, indent=2), encoding="utf-8")
    print(f"[power] start {device['device_key']} {model_key} ctx={ctx} step={step_tokens} uuid={case_uuid}", flush=True)
    start_resp = post_json(args.power_api_url, "/v1/power/start", start_payload)
    start_resp_json.write_text(json.dumps(start_resp, indent=2), encoding="utf-8")
    session_id = str(start_resp.get("session_id") or "")
    if not session_id:
        raise RuntimeError(f"power start missing session_id: {start_resp}")
    metadata["power_run_id"] = str(start_resp.get("run_id") or "")
    metadata["power_monitor"] = start_resp.get("monitor", {})
    write_kv_txt(txt_path, metadata)

    reps: list[dict[str, Any]] = []
    stop_resp: dict[str, Any] = {}
    try:
        if args.power_warmup_sec > 0:
            time.sleep(args.power_warmup_sec)
        print(f"[measure] {device['device_key']} {model_key} ctx={ctx} step={step_tokens} decode", flush=True)
        proc = ssh(
            device,
            remote_trigger_decode_script(device, ctx, step_tokens, args.run_id, helper_pid),
            timeout=args.remote_timeout_sec,
        )
        rep_meta = parse_helper_result(proc.stdout)
        rep_meta["rep_index"] = 1
        reps.append(rep_meta)
        metadata["power_repetition_results"] = reps
        metadata.update(rep_meta)
        write_kv_txt(txt_path, metadata)
    finally:
        if args.power_cooldown_sec > 0:
            time.sleep(args.power_cooldown_sec)
        stop_payload = {"session_id": session_id, "timeout_sec": args.power_stop_timeout_sec}
        stop_json.write_text(json.dumps(stop_payload, indent=2), encoding="utf-8")
        stop_resp = post_json(args.power_api_url, "/v1/power/stop", stop_payload, timeout=args.power_stop_timeout_sec + 30)
        stop_resp_json.write_text(json.dumps(stop_resp, indent=2), encoding="utf-8")

    artifacts = stop_resp.get("artifacts") if isinstance(stop_resp, dict) else {}
    csv_ref = artifacts.get("csv") if isinstance(artifacts, dict) and isinstance(artifacts.get("csv"), dict) else {}
    raw_url = str(csv_ref.get("raw_url") or csv_ref.get("download_url") or "")
    if not raw_url and stop_resp.get("run_id"):
        raw_url = f"/v1/artifact/raw?run_id={stop_resp['run_id']}&path=power.csv"
    if not raw_url:
        raise RuntimeError(f"power stop missing CSV URL: {stop_resp}")
    download_power_csv(args.power_api_url, raw_url, csv_path)
    png_path = render_power_csv(csv_path)
    metadata["power_png"] = str(png_path)
    metadata["status"] = 200
    metadata["error"] = ""
    write_kv_txt(txt_path, metadata)
    append_manifest(out_dir, metadata)
    return metadata


def summarize_power(manifests: list[Path], output: Path, plot: Path) -> None:
    cmd = [
        "python3",
        str(REPO / ".codex/skills/mnn-pic-benchmark/scripts/summarize_pic_power_reps.py"),
        *[str(path) for path in manifests],
        "-o",
        str(output),
        "--plot-output",
        str(plot),
        "--token-field",
        "generated_tokens",
        "--drop-first",
        "0",
        "--drop-last",
        "0",
    ]
    run_local(cmd)


def update_benchmark(summary_csv: Path, benchmark_csv: Path, test_date: str) -> None:
    existing: list[dict[str, Any]] = []
    if benchmark_csv.exists():
        with benchmark_csv.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            existing = list(reader)
    key_fields = ["device", "model", "context_tokens", "mode", "budget", "generated_tokens"]
    rows_by_key = {tuple(row.get(field, "") for field in key_fields): row for row in existing}
    with summary_csv.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            out = {
                "device": row.get("device", ""),
                "device_display": row.get("device_display", ""),
                "model": row.get("model", ""),
                "backend": row.get("backend", ""),
                "frequency_note": row.get("frequency_note", ""),
                "context_tokens": row.get("context_tokens", ""),
                "mode": row.get("algorithm", ""),
                "budget": row.get("budget", ""),
                "generated_tokens": row.get("generated_tokens", ""),
                "test_date": test_date,
                "mj_per_token": row.get("mj_per_token", ""),
                "decode_tpot_ms": row.get("kept_decode_tpot_ms") or row.get("decode_tpot_ms", ""),
                "decode_tps": row.get("kept_decode_tps") or row.get("decode_tps", ""),
                "energy_per_inference_mj": row.get("energy_per_inference_mj", ""),
                "duration_per_inference_s": row.get("duration_per_inference_s", ""),
                "analysis_status": row.get("analysis_status", ""),
                "error": row.get("error", ""),
            }
            key = tuple(out.get(field, "") for field in key_fields)
            rows_by_key[key] = out
    rows = sorted(rows_by_key.values(), key=lambda r: (r.get("device", ""), r.get("model", ""), int(float(r.get("context_tokens") or 0))))
    with benchmark_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def update_decode_benchmark(summary_csv: Path, benchmark_csv: Path) -> None:
    existing: list[dict[str, Any]] = []
    if benchmark_csv.exists():
        with benchmark_csv.open("r", encoding="utf-8", newline="") as fp:
            existing = list(csv.DictReader(fp))
    key_fields = ["device", "model", "context_tokens", "mode", "budget", "decode_selector", "repair_tokens", "generated_tokens"]
    rows_by_key = {tuple(row.get(field, "") for field in key_fields): row for row in existing}
    with summary_csv.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            step_tokens = int(float(row.get("step_tokens") or 1))
            generated_tokens = row.get("generated_tokens", "")
            decode_tpot_ms = row.get("kept_decode_tpot_ms") or row.get("decode_tpot_ms", "")
            decode_tps = row.get("kept_decode_tps") or row.get("decode_tps", "")
            latency_s = ""
            try:
                latency_s = f"{float(decode_tpot_ms) * float(generated_tokens) / 1000.0:.6f}"
            except (TypeError, ValueError):
                pass
            out = {
                "device": row.get("device", ""),
                "device_display": row.get("device_display", ""),
                "model": row.get("model", ""),
                "backend": row.get("backend", ""),
                "frequency_profile": row.get("frequency_note", ""),
                "context_tokens": row.get("context_tokens", ""),
                "mode": decode_csv_mode(step_tokens),
                "budget": row.get("budget", ""),
                "decode_selector": f"true-normal-llm-input{step_tokens}",
                "repair_tokens": str(max(0, step_tokens - 1)),
                "generated_tokens": generated_tokens,
                "decode_latency_s": latency_s,
                "decode_tpot_ms": decode_tpot_ms,
                "decode_tps": decode_tps,
                "benchmark_status": "ok" if row.get("analysis_status", "") == "ok" else row.get("analysis_status", ""),
            }
            key = tuple(out.get(field, "") for field in key_fields)
            rows_by_key[key] = out
    rows = sorted(
        rows_by_key.values(),
        key=lambda r: (
            r.get("device", ""),
            r.get("model", ""),
            int(float(r.get("context_tokens") or 0)),
            r.get("mode", ""),
            int(float(r.get("repair_tokens") or 0)),
        ),
    )
    with benchmark_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=DECODE_BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run normal LLM load-KV decode energy sweep on OrangePi/Rhino.")
    parser.add_argument("--devices", default="orangepi,rhino")
    parser.add_argument("--model-keys", default="llama3.2-3b,minicpm5-1b,qwen3-4b")
    parser.add_argument("--contexts", default="512,1024,1536,2048")
    parser.add_argument("--step-tokens", default="1")
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--power-rep", type=int, default=1)
    parser.add_argument("--power-warmup-sec", type=float, default=5.0)
    parser.add_argument("--power-cooldown-sec", type=float, default=5.0)
    parser.add_argument("--power-max-duration-sec", type=float, default=1800.0)
    parser.add_argument("--power-stop-timeout-sec", type=float, default=180.0)
    parser.add_argument("--power-api-url", default=POWER_API_URL)
    parser.add_argument("--remote-timeout-sec", type=int, default=900)
    parser.add_argument("--run-id", default="normal_decode_energy_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--benchmark-csv", default=str(REPO / "benchmark_decode-energy.csv"))
    parser.add_argument("--decode-benchmark-csv", default=str(REPO / "benchmark_decode.csv"))
    parser.add_argument("--summary-csv", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    devices = parse_csv_list(args.devices)
    model_keys = parse_csv_list(args.model_keys)
    contexts = [int(item) for item in parse_csv_list(args.contexts)]
    step_tokens_list = [int(item) for item in parse_csv_list(args.step_tokens)]
    manifests: set[Path] = set()
    failures: list[str] = []
    for device_key in devices:
        if device_key not in DEVICES:
            raise SystemExit(f"unknown device {device_key}")
        device = dict(DEVICES[device_key])
        device["device_key"] = device_key
        for model_key in model_keys:
            if model_key not in MODELS:
                raise SystemExit(f"unknown model {model_key}")
            for ctx in contexts:
                for step_tokens in step_tokens_list:
                    try:
                        metadata = run_case(args, device, model_key, ctx, step_tokens)
                        manifests.add(Path(metadata["power_txt"]).with_name("manifest.tsv"))
                    except Exception as exc:
                        msg = f"{device_key} {model_key} ctx={ctx} step={step_tokens}: {exc}"
                        print(f"[failure] {msg}", flush=True)
                        failures.append(msg)
    if not manifests:
        raise SystemExit("no successful power captures")
    summary_csv = Path(args.summary_csv) if args.summary_csv else POWER_ROOT / f"normal_decode_energy_summary_{args.run_id}.csv"
    summary_png = summary_csv.with_suffix(".png")
    summarize_power(sorted(manifests), summary_csv, summary_png)
    update_benchmark(summary_csv, Path(args.benchmark_csv), time.strftime("%Y-%m-%d"))
    update_decode_benchmark(summary_csv, Path(args.decode_benchmark_csv))
    print(f"[summary] {summary_csv}")
    print(f"[benchmark] {args.benchmark_csv}")
    print(f"[decode_benchmark] {args.decode_benchmark_csv}")
    if failures:
        print("[failures]")
        for item in failures:
            print(item)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
