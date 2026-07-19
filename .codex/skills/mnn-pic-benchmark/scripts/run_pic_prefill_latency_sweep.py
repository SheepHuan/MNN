#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import shlex
import signal
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


MNN_ROOT = Path(__file__).resolve().parents[4]
SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_TOKENS = MNN_ROOT / ".cache/latency_budget_20260611/opencl_pic_1024_current_20260622_191527/tokens.json"
TEMPLATE_MODEL = "Llama3.2 1B"
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
MODE_CHOICES = ["normal-full-recompute", "full-reuse", "cacheblend", "epic"]
DISABLED_MODE_CHOICES = {
    "full-compute": "PIC full-compute / pic-full-recompute is no longer a benchmark target",
    "pic-full-recompute": "PIC full-compute / pic-full-recompute is no longer a benchmark target",
}
RATIO_MODES = {"cacheblend", "epic"}
POWER_EXCLUDED_MODEL_KEYS = {"llama3.2-1b"}
DEFAULT_CONTEXTS = "512,1024,1536,2048,2560"
DEFAULT_CHAT_TIMEOUT_SECONDS = float(os.environ.get("PIC_SWEEP_CHAT_TIMEOUT_SECONDS", "600"))
DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("PIC_SWEEP_SSH_CONNECT_TIMEOUT_SECONDS", "10"))
DEFAULT_SSH_COMMAND_TIMEOUT_SECONDS = float(os.environ.get("PIC_SWEEP_SSH_COMMAND_TIMEOUT_SECONDS", "120"))
DEFAULT_REMOTE_BENCH_TIMEOUT_SECONDS = float(os.environ.get("PIC_SWEEP_REMOTE_BENCH_TIMEOUT_SECONDS", "3600"))
DEFAULT_SERVER_RUNTIME_TIMEOUT_SECONDS = float(os.environ.get("PIC_SWEEP_SERVER_RUNTIME_TIMEOUT_SECONDS", "3600"))
DEFAULT_TIMEOUT_KILL_AFTER_SECONDS = float(os.environ.get("PIC_SWEEP_TIMEOUT_KILL_AFTER_SECONDS", "10"))
DEFAULT_POWER_API_URL = "http://192.168.101.14:8766"
POWER_REP_INTERVAL_SEC = 3.0
CSV_KEYS = [
    "device",
    "device_display",
    "model",
    "backend",
    "frequency_note",
    "context_tokens",
    "mode",
    "budget",
]
GAP_COMPARE_KEYS = [
    "backend",
    "frequency_note",
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
        "root_ssh": "root@192.168.101.113",
        "remote_work": "/mnt/ssd/code/.cache/mnn_opencl_pic",
        "artifact": "/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus",
        "ld_library_path": "/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/lib",
        "server_env": "",
        "bench_env": "LD_PRELOAD=/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/lib/libMNN_CL.so",
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
        "bench_env": "LD_PRELOAD=/usr/lib/libOpenCL_adreno.so",
        "backend": "opencl",
        "remote_port": 18133,
        "local_port": 19133,
        "remote_cache_root": "/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep",
    },
}

FORMAL_REQUIRED_DEVICES = ("jetson", "orangepi")
EXTRA_PROFILE_DEVICES = ("rhino",)

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
    "qwen3-4b": {
        "name": "Qwen3-4B",
        "served_model": "qwen3-4b-pic",
        "paths": {
            "jetson": {
                "normal_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-llm-export/Qwen__Qwen3-4B/config_cuda_greedy.json",
                ],
                "pic_config": [
                    "/home/jetson/code/kvshare-edge/impl/MNN/.cache/weight/Qwen__Qwen3-4B-pic-boundary/config_cuda_greedy.json",
                ],
            },
            "orangepi": {
                "normal_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/Qwen__Qwen3-4B/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/Qwen__Qwen3-4B-pic-boundary/config_opencl_greedy.json",
                ],
            },
            "rhino": {
                "normal_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/normal/Qwen__Qwen3-4B/config_opencl_greedy.json",
                ],
                "pic_config": [
                    "/mnt/nvme/mnn_pic_opencl/models/pic/Qwen__Qwen3-4B-pic-boundary/config_opencl_greedy.json",
                ],
            },
        },
    },
}

DEFAULT_MAX_FREQUENCY_NOTES: dict[str, str] = {
    "jetson": "cpu=max,gpu=max,ddr=max",
    "orangepi": "cpu=max,gpu=max,ddr=max",
    "rhino": "cpu=max,gpu=max,ddr=max",
}

POWER_DEVICE_DEFAULTS: dict[str, dict[str, Any]] = {
    "jetson": {
        "monitor_type": "df",
        "serial": "1A5D43",
    },
    "rhino": {
        "monitor_type": "df",
        "serial": "1A5D43",
    },
    "orangepi": {
        "device": "orangepi5plus",
        "monitor_type": "blu",
        "serial": "F96FBDBA05B0",
        "sample_rate_hz": 10000,
        "voltage_mv": 4800,
    },
}

HIGH_RISK_CONTEXT_POLICIES: dict[str, dict[str, Any]] = {
    "jetson": {
        "profiles": {"max"},
        "max_context": 2560,
        "reason": "jetson max-frequency sweep blocks context > 2560 by default to avoid OOM/shutdown risk",
    },
    "orangepi": {
        "profiles": {"max"},
        "max_context": 2560,
        "reason": "orangepi max-frequency sweep blocks context > 2560 by default to avoid OOM/shutdown risk",
    },
    "rhino": {
        "profiles": {"max", "cpu-high-gpu-max"},
        "max_context": 2560,
        "reason": "rhino high-frequency sweep blocks context > 2560 by default to avoid shutdown risk",
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


def timeout_prefix(timeout: float | None, kill_after: float | None = None) -> list[str]:
    if timeout is None or float(timeout) <= 0:
        return []
    grace = max(1, int(kill_after if kill_after is not None else DEFAULT_TIMEOUT_KILL_AFTER_SECONDS))
    seconds = max(1, int(float(timeout)))
    return ["timeout", "-k", f"{grace}s", f"{seconds}s"]


def timeout_command(cmd: list[str], timeout: float | None, kill_after: float | None = None) -> list[str]:
    return timeout_prefix(timeout, kill_after) + list(cmd)


def subprocess_timeout(timeout: float | None, kill_after: float | None = None) -> float | None:
    if timeout is None or float(timeout) <= 0:
        return None
    grace = max(1.0, float(kill_after if kill_after is not None else DEFAULT_TIMEOUT_KILL_AFTER_SECONDS))
    return float(timeout) + grace + 5.0


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    effective_cmd = timeout_command(cmd, timeout)
    print("+", " ".join(effective_cmd), flush=True)
    return subprocess.run(
        effective_cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=subprocess_timeout(timeout),
    )


def ssh(
    device: dict[str, Any],
    script: str,
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    wrapped = f"bash -lc {shlex.quote(script)}"
    effective_timeout = timeout
    if effective_timeout is None:
        effective_timeout = float(device.get("ssh_command_timeout_sec", DEFAULT_SSH_COMMAND_TIMEOUT_SECONDS))
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS}",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=1",
            "-o",
            "StrictHostKeyChecking=accept-new",
            device["ssh"],
            wrapped,
        ],
        check=check,
        timeout=effective_timeout,
    )


def sudo_prefix(device: dict[str, Any]) -> str:
    password_env = ""
    if device.get("device_key") == "rhino":
        password_env = str(os.environ.get("MNN_RHINO_SUDO_PASSWORD", "")).strip()
    elif device.get("device_key") == "orangepi":
        password_env = str(os.environ.get("MNN_ORANGEPI_SUDO_PASSWORD", "")).strip()
    elif device.get("device_key") == "jetson":
        password_env = str(os.environ.get("MNN_JETSON_SUDO_PASSWORD", "")).strip()
    if password_env:
        return f"echo {shlex.quote(password_env)} | sudo -S -p ''"
    return "sudo -n"


def ssh_root(
    device: dict[str, Any],
    script: str,
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    root_ssh = str(device.get("root_ssh", "")).strip()
    if root_ssh:
        wrapped = f"sh -c {shlex.quote(script)}"
        ssh_target = root_ssh
    else:
        wrapped = f"{sudo_prefix(device)} sh -c {shlex.quote(script)}"
        ssh_target = device["ssh"]
    effective_timeout = timeout
    if effective_timeout is None:
        effective_timeout = float(device.get("ssh_command_timeout_sec", DEFAULT_SSH_COMMAND_TIMEOUT_SECONDS))
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS}",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=1",
            "-o",
            "StrictHostKeyChecking=accept-new",
            ssh_target,
            wrapped,
        ],
        check=check,
        timeout=effective_timeout,
    )


def query_frequency_state(device: dict[str, Any]) -> dict[str, Any]:
    script = r"""
python3 - <<'PY'
import json
import os
import re
import shutil
import subprocess

GPU_PATHS = [
    "/sys/class/devfreq/fb000000.gpu",
    "/sys/class/devfreq/mali0",
    "/sys/class/devfreq/3d00000.qcom,kgsl-3d0",
    "/sys/class/kgsl/kgsl-3d0",
]
DDR_PATHS = [
    "/sys/class/devfreq/dmc",
    "/sys/class/devfreq/dmc_ondemand",
]

def read(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except Exception:
        return ""

def run_cmd(cmd):
    try:
        return subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    except Exception:
        return ""

def policy_sort_key(path):
    name = os.path.basename(path)
    try:
        return int(name.replace("policy", ""))
    except Exception:
        return 10**9

state = {"cpu": {}, "gpu": {}, "ddr": {}}
cpu_policy_root = "/sys/devices/system/cpu/cpufreq"
cpu_policy_paths = []
if os.path.isdir(cpu_policy_root):
    cpu_policy_paths = [
        os.path.join(cpu_policy_root, item)
        for item in os.listdir(cpu_policy_root)
        if item.startswith("policy") and os.path.isdir(os.path.join(cpu_policy_root, item))
    ]
cpu_policy_paths = sorted(cpu_policy_paths, key=policy_sort_key)
cpu_names = ["little", "mid", "big"]
for index, base in enumerate(cpu_policy_paths):
    name = cpu_names[index] if index < len(cpu_names) else os.path.basename(base)
    info = {}
    info["path"] = base
    for key in [
        "related_cpus",
        "affected_cpus",
        "scaling_available_governors",
        "scaling_governor",
        "scaling_cur_freq",
        "scaling_min_freq",
        "scaling_max_freq",
        "cpuinfo_min_freq",
        "cpuinfo_max_freq",
    ]:
        path = os.path.join(base, key)
        if os.path.isfile(path):
            info[key] = read(path)
    if info:
        state["cpu"][name] = info

for base in GPU_PATHS:
    if not os.path.exists(base):
        continue
    info = {"path": base}
    for key in [
        "governor",
        "available_governors",
        "cur_freq",
        "min_freq",
        "max_freq",
        "available_frequencies",
        "gpuclk",
        "gpu_available_frequencies",
        "min_pwrlevel",
        "max_pwrlevel",
        "default_pwrlevel",
        "thermal_pwrlevel",
    ]:
        path = os.path.join(base, key)
        if os.path.isfile(path):
            info[key] = read(path)
    if len(info) > 1:
        state["gpu"] = info
        break

for base in DDR_PATHS:
    if not os.path.exists(base):
        continue
    info = {"path": base}
    for key in [
        "governor",
        "available_governors",
        "cur_freq",
        "min_freq",
        "max_freq",
        "available_frequencies",
    ]:
        path = os.path.join(base, key)
        if os.path.isfile(path):
            info[key] = read(path)
    if len(info) > 1:
        state["ddr"] = info
        break

jetson_clocks = shutil.which("jetson_clocks")
if jetson_clocks:
    show = run_cmd([jetson_clocks, "--show"])
    if "root user" in show or "Permission denied" in show:
        sudo = shutil.which("sudo")
        if sudo:
            sudo_show = run_cmd([sudo, "-n", jetson_clocks, "--show"])
            if sudo_show:
                show = sudo_show
    jetson = {"jetson_clocks_show": show}
    for line in show.splitlines():
        line = line.strip()
        match = re.match(r"GPU MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)", line)
        if match:
            gpu = {
                "path": "jetson_clocks",
                "min_freq": match.group(1),
                "max_freq": match.group(2),
                "cur_freq": match.group(3),
            }
            jetson["gpu"] = gpu
            if not state.get("gpu"):
                state["gpu"] = gpu
            continue
        match = re.match(r"EMC MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)(?: FreqOverride=(\d+))?", line)
        if match:
            ddr = {
                "path": "jetson_clocks",
                "min_freq": match.group(1),
                "max_freq": match.group(2),
                "cur_freq": match.group(3),
            }
            if match.group(4) is not None:
                ddr["freq_override"] = match.group(4)
            jetson["emc"] = ddr
            if not state.get("ddr"):
                state["ddr"] = ddr
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel:
        nvpmodel_output = run_cmd([nvpmodel, "-q"])
        if "NVPM ERROR" in nvpmodel_output or "Permission denied" in nvpmodel_output:
            sudo = shutil.which("sudo")
            if sudo:
                sudo_nvpmodel_output = run_cmd([sudo, "-n", nvpmodel, "-q"])
                if sudo_nvpmodel_output:
                    nvpmodel_output = sudo_nvpmodel_output
        jetson["nvpmodel"] = nvpmodel_output
    state["jetson"] = jetson

print(json.dumps(state, ensure_ascii=False))
PY
"""
    result = ssh(device, script, check=False, timeout=30)
    output = (result.stdout or "").strip().splitlines()
    for line in reversed(output):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"error": output[-1] if output else "query_frequency_state_failed"}


def apply_frequency_profile(device: dict[str, Any], profile: str) -> dict[str, Any]:
    profile = str(profile or "max").strip()
    if device.get("device_key") == "jetson":
        if profile not in {"", "max"}:
            raise RuntimeError(f"frequency profile {profile!r} is not implemented for device jetson")
        current_state = query_frequency_state(device)
        if jetson_frequency_state_is_max(current_state):
            return current_state
        script = """
set -e
if command -v nvpmodel >/dev/null 2>&1; then
  nvpmodel -m 0
fi
if command -v jetson_clocks >/dev/null 2>&1; then
  jetson_clocks
else
  echo "missing jetson_clocks" >&2
  exit 1
fi
"""
        result = ssh_root(device, script, check=False, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                "failed to apply jetson max frequency profile; "
                "set MNN_JETSON_SUDO_PASSWORD or enable passwordless sudo on Jetson\n"
                + (result.stdout or "")
                + (result.stderr or "")
            )
        frequency_state = query_frequency_state(device)
        if not jetson_frequency_state_is_max(frequency_state):
            raise RuntimeError(
                "jetson frequency profile=max did not reach cpu=max,gpu=max,ddr=max\n"
                + json.dumps(frequency_state, ensure_ascii=False, indent=2)
            )
        return frequency_state
    if device.get("device_key") == "orangepi":
        if profile not in {"", "max"}:
            raise RuntimeError(f"frequency profile {profile!r} is not implemented for device orangepi")
        current_state = query_frequency_state(device)
        if orangepi_frequency_state_is_max(current_state):
            return current_state
        script = """
set -e
set_cpu_policy_max() {
  policy="$1"
  [ -d "$policy" ] || return 0
  [ -f "$policy/cpuinfo_max_freq" ] || return 0
  max_freq="$(cat "$policy/cpuinfo_max_freq")"
  [ -n "$max_freq" ] || return 0
  [ -f "$policy/scaling_max_freq" ] && echo "$max_freq" > "$policy/scaling_max_freq"
  [ -f "$policy/scaling_min_freq" ] && echo "$max_freq" > "$policy/scaling_min_freq"
  if [ -f "$policy/scaling_governor" ]; then
    if [ -f "$policy/scaling_available_governors" ] && grep -qw performance "$policy/scaling_available_governors"; then
      echo performance > "$policy/scaling_governor" || true
    else
      echo performance > "$policy/scaling_governor" 2>/dev/null || true
    fi
  fi
}

set_devfreq_max() {
  path="$1"
  label="$2"
  [ -d "$path" ] || {
    echo "missing $label devfreq path: $path" >&2
    return 1
  }
  freq=""
  if [ -f "$path/available_frequencies" ]; then
    freq="$(tr ' ' '\\n' < "$path/available_frequencies" | awk 'NF {print}' | sort -n | tail -1)"
  fi
  if [ -z "$freq" ] && [ -f "$path/max_freq" ]; then
    freq="$(cat "$path/max_freq")"
  fi
  [ -n "$freq" ] || {
    echo "missing $label max frequency under $path" >&2
    return 1
  }
  [ -f "$path/max_freq" ] && echo "$freq" > "$path/max_freq"
  [ -f "$path/min_freq" ] && echo "$freq" > "$path/min_freq"
  if [ -f "$path/governor" ]; then
    if [ -f "$path/available_governors" ] && grep -qw performance "$path/available_governors"; then
      echo performance > "$path/governor" || true
    elif [ -f "$path/available_governors" ] && grep -qw userspace "$path/available_governors"; then
      echo userspace > "$path/governor" || true
    fi
  fi
}

for policy in /sys/devices/system/cpu/cpufreq/policy*; do
  set_cpu_policy_max "$policy"
done
set_devfreq_max /sys/class/devfreq/fb000000.gpu gpu
set_devfreq_max /sys/class/devfreq/dmc ddr
"""
        result = ssh_root(device, script, check=False, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                "failed to apply orangepi max frequency profile; "
                "ensure root SSH works or set MNN_ORANGEPI_SUDO_PASSWORD / passwordless sudo\n"
                + (result.stdout or "")
                + (result.stderr or "")
            )
        frequency_state = query_frequency_state(device)
        if not orangepi_frequency_state_is_max(frequency_state):
            raise RuntimeError(
                "orangepi frequency profile=max did not reach cpu=max,gpu=max,ddr=max\n"
                + json.dumps(frequency_state, ensure_ascii=False, indent=2)
            )
        return frequency_state
    if device.get("device_key") != "rhino":
        if profile in {"", "max"}:
            return query_frequency_state(device)
        raise RuntimeError(f"frequency profile {profile!r} is not implemented for device {device.get('device_key')}")
    scripts = {
        "max": """
set -e
echo 556800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo 2016000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 614400 > /sys/devices/system/cpu/cpufreq/policy3/scaling_min_freq
echo 2803200 > /sys/devices/system/cpu/cpufreq/policy3/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy3/scaling_governor
echo 864000 > /sys/devices/system/cpu/cpufreq/policy5/scaling_min_freq
echo 3187200 > /sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy5/scaling_governor
echo 124800000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/min_freq
echo 680000000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq
""",
        "cpu-high-gpu-max": """
set -e
echo 556800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo 2016000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 614400 > /sys/devices/system/cpu/cpufreq/policy3/scaling_min_freq
echo 2803200 > /sys/devices/system/cpu/cpufreq/policy3/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy3/scaling_governor
echo 864000 > /sys/devices/system/cpu/cpufreq/policy5/scaling_min_freq
echo 3187200 > /sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy5/scaling_governor
echo 124800000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/min_freq
echo 680000000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq
""",
        "safe-low": """
set -e
echo 556800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo 1670400 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 614400 > /sys/devices/system/cpu/cpufreq/policy3/scaling_min_freq
echo 2323200 > /sys/devices/system/cpu/cpufreq/policy3/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy3/scaling_governor
echo 864000 > /sys/devices/system/cpu/cpufreq/policy5/scaling_min_freq
echo 2592000 > /sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy5/scaling_governor
echo 124800000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/min_freq
echo 475000000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq
""",
        # Keep CPU clusters below the top bin while leaving GPU and DDR at max.
        "cpu-low-gpu-max": """
set -e
echo 556800 > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq
echo 1670400 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 614400 > /sys/devices/system/cpu/cpufreq/policy3/scaling_min_freq
echo 2323200 > /sys/devices/system/cpu/cpufreq/policy3/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy3/scaling_governor
echo 864000 > /sys/devices/system/cpu/cpufreq/policy5/scaling_min_freq
echo 2592000 > /sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq
echo performance > /sys/devices/system/cpu/cpufreq/policy5/scaling_governor
echo 124800000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/min_freq
echo 680000000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq
""",
    }
    if profile not in scripts:
        allowed = ", ".join(["max"] + sorted(scripts))
        raise RuntimeError(f"unsupported rhino frequency profile {profile!r}; expected one of: {allowed}")
    script = scripts[profile]
    if profile == "max":
        current_state = query_frequency_state(device)
        if rhino_frequency_state_is_max(current_state):
            return current_state
    result = ssh_root(device, script, check=False, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to apply rhino {profile} frequency profile; "
            "set MNN_RHINO_SUDO_PASSWORD or enable passwordless sudo\n"
            + (result.stdout or "")
        )
    frequency_state = query_frequency_state(device)
    if profile == "max" and not rhino_frequency_state_is_max(frequency_state):
        raise RuntimeError(
            "rhino frequency profile=max did not reach cpu=max,gpu=max,ddr=max\n"
            + json.dumps(frequency_state, ensure_ascii=False, indent=2)
        )
    return frequency_state


def _max_available_freq(info: dict[str, Any] | None) -> str:
    if not isinstance(info, dict):
        return ""
    values: list[int] = []
    for token in str(info.get("available_frequencies", "")).replace(",", " ").split():
        try:
            values.append(int(token))
        except ValueError:
            pass
    for key in ("cpuinfo_max_freq", "max_freq"):
        try:
            value = int(str(info.get(key, "")).strip())
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return str(max(values)) if values else ""


def _devfreq_is_locked_to_max(info: dict[str, Any] | None) -> bool:
    if not isinstance(info, dict) or not info:
        return False
    target = _max_available_freq(info)
    if not target:
        return False
    return str(info.get("min_freq", "")).strip() == target and str(info.get("max_freq", "")).strip() == target


def orangepi_frequency_state_is_max(frequency_state: dict[str, Any] | None) -> bool:
    if not isinstance(frequency_state, dict):
        return False
    cpu = frequency_state.get("cpu") if isinstance(frequency_state.get("cpu"), dict) else {}
    if not cpu:
        return False
    for info in cpu.values():
        if not isinstance(info, dict):
            return False
        target = _max_available_freq(info)
        if not target:
            return False
        if str(info.get("scaling_min_freq", "")).strip() != target:
            return False
        if str(info.get("scaling_max_freq", "")).strip() != target:
            return False
    gpu = frequency_state.get("gpu") if isinstance(frequency_state.get("gpu"), dict) else {}
    ddr = frequency_state.get("ddr") if isinstance(frequency_state.get("ddr"), dict) else {}
    return _devfreq_is_locked_to_max(gpu) and _devfreq_is_locked_to_max(ddr)


def jetson_frequency_state_is_max(frequency_state: dict[str, Any] | None) -> bool:
    if not isinstance(frequency_state, dict):
        return False
    cpu = frequency_state.get("cpu") if isinstance(frequency_state.get("cpu"), dict) else {}
    if not cpu:
        return False
    for info in cpu.values():
        if not isinstance(info, dict):
            return False
        target = _max_available_freq(info)
        if not target:
            return False
        if str(info.get("scaling_min_freq", "")).strip() != target:
            return False
        if str(info.get("scaling_max_freq", "")).strip() != target:
            return False
    gpu = frequency_state.get("gpu") if isinstance(frequency_state.get("gpu"), dict) else {}
    ddr = frequency_state.get("ddr") if isinstance(frequency_state.get("ddr"), dict) else {}
    for info in (gpu, ddr):
        if not isinstance(info, dict) or not info:
            return False
        max_freq = str(info.get("max_freq", "")).strip()
        cur_freq = str(info.get("cur_freq", "")).strip()
        if not max_freq or cur_freq != max_freq:
            return False
    gpu_min = str(gpu.get("min_freq", "")).strip()
    if gpu_min and gpu_min != str(gpu.get("max_freq", "")).strip():
        return False
    override = str(ddr.get("freq_override", "")).strip()
    if override and override != "1":
        return False
    jetson = frequency_state.get("jetson") if isinstance(frequency_state.get("jetson"), dict) else {}
    nvpmodel = str(jetson.get("nvpmodel", ""))
    if nvpmodel and "MAXN" not in nvpmodel:
        return False
    return True


def rhino_frequency_state_is_max(frequency_state: dict[str, Any] | None) -> bool:
    if not isinstance(frequency_state, dict):
        return False
    cpu = frequency_state.get("cpu") if isinstance(frequency_state.get("cpu"), dict) else {}
    gpu = frequency_state.get("gpu") if isinstance(frequency_state.get("gpu"), dict) else {}
    for cluster in ("little", "mid", "big"):
        info = cpu.get(cluster)
        if not isinstance(info, dict):
            return False
        scaling_max = str(info.get("scaling_max_freq", "")).strip()
        cpuinfo_max = str(info.get("cpuinfo_max_freq", "")).strip()
        if not scaling_max or not cpuinfo_max or scaling_max != cpuinfo_max:
            return False
    gpu_max = str(gpu.get("max_freq", "")).strip()
    if gpu_max != "680000000":
        return False
    return True


def format_frequency_note(device: dict[str, Any], frequency_state: dict[str, Any] | None) -> str:
    device_key = str(device.get("device_key", ""))
    default = DEFAULT_MAX_FREQUENCY_NOTES.get(device_key, "cpu=max,gpu=max,ddr=max")
    if device_key == "jetson" and str(device.get("frequency_profile", "")).strip() == "max":
        if jetson_frequency_state_is_max(frequency_state):
            return default
    if device_key == "orangepi" and str(device.get("frequency_profile", "")).strip() == "max":
        if orangepi_frequency_state_is_max(frequency_state):
            return default
    if device_key == "rhino" and str(device.get("frequency_profile", "")).strip() == "max":
        if rhino_frequency_state_is_max(frequency_state):
            return default
    if device_key not in {"jetson", "orangepi", "rhino"}:
        return default
    if not isinstance(frequency_state, dict):
        return "frequency=unknown"
    cpu = frequency_state.get("cpu") if isinstance(frequency_state.get("cpu"), dict) else {}
    gpu = frequency_state.get("gpu") if isinstance(frequency_state.get("gpu"), dict) else {}
    ddr = frequency_state.get("ddr") if isinstance(frequency_state.get("ddr"), dict) else {}
    little = str(cpu.get("little", {}).get("scaling_max_freq", "")).strip() if isinstance(cpu.get("little"), dict) else ""
    mid = str(cpu.get("mid", {}).get("scaling_max_freq", "")).strip() if isinstance(cpu.get("mid"), dict) else ""
    big = str(cpu.get("big", {}).get("scaling_max_freq", "")).strip() if isinstance(cpu.get("big"), dict) else ""
    cpu_parts = [item for item in (little, mid, big) if item]
    if not cpu_parts:
        for item in cpu.values():
            if isinstance(item, dict):
                freq = str(item.get("scaling_max_freq", "")).strip()
                if freq:
                    cpu_parts.append(freq)
    cpu_note = "/".join(cpu_parts) if cpu_parts else "unknown"
    gpu_freq = str(gpu.get("max_freq", "") or gpu.get("cur_freq", "") or "unknown").strip()
    ddr_note = str(ddr.get("max_freq", "") or device.get("ddr_note", "") or "unknown").strip()
    if not (cpu_note and gpu_freq):
        return "frequency=unknown"
    return f"cpu={cpu_note},gpu={gpu_freq},ddr={ddr_note}"


def high_risk_context_error(device_name: str, frequency_profile: str, context: int) -> str | None:
    policy = HIGH_RISK_CONTEXT_POLICIES.get(str(device_name))
    if not policy:
        return None
    if int(context) <= int(policy.get("max_context", 0) or 0):
        return None
    if str(frequency_profile or "") not in set(policy.get("profiles", set())):
        return None
    reason = str(policy.get("reason", "")).strip() or "high-risk context is blocked by default"
    return f"{reason}; use --allow-high-risk-contexts to override"


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


def resolve_remote_file(
    device: dict[str, Any],
    candidates: list[str],
    label: str,
) -> str:
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
""".strip()
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


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float = DEFAULT_CHAT_TIMEOUT_SECONDS) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    cmd = [
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
    ]
    proc = subprocess.run(
        timeout_command(cmd, float(timeout) + 5.0, kill_after=5),
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=subprocess_timeout(float(timeout) + 5.0, kill_after=5),
    )
    elapsed = time.perf_counter() - started
    status, parsed = _parse_curl_json(proc.stdout)
    if proc.returncode != 0 and status == 0:
        parsed = {"error_body": proc.stdout, "curl_returncode": proc.returncode}
    return status, parsed, elapsed


def get_json(base_url: str, path: str, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        str(int(timeout)),
        "-w",
        "\nHTTP_STATUS:%{http_code}",
        base_url.rstrip("/") + path,
    ]
    proc = subprocess.run(
        timeout_command(cmd, float(timeout) + 5.0, kill_after=5),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=subprocess_timeout(float(timeout) + 5.0, kill_after=5),
    )
    return _parse_curl_json(proc.stdout)


def post_power_json(api_url: str, path: str, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    cmd = [
        "curl",
        "--noproxy",
        "*",
        "-sS",
        "--connect-timeout",
        "5",
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
        api_url.rstrip("/") + path,
    ]
    proc = subprocess.run(
        timeout_command(cmd, float(timeout) + 5.0, kill_after=5),
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=subprocess_timeout(float(timeout) + 5.0, kill_after=5),
    )
    elapsed = time.perf_counter() - started
    status, parsed = _parse_curl_json(proc.stdout)
    if proc.returncode != 0 and status == 0:
        parsed = {"error_body": proc.stdout, "curl_returncode": proc.returncode}
    return status, parsed, elapsed


def get_power_raw(api_url: str, url_or_path: str, output_path: Path, timeout: float = 120.0) -> None:
    url = urljoin(api_url.rstrip("/") + "/", str(url_or_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "--noproxy",
        "*",
        "-fsS",
        "--connect-timeout",
        "5",
        "--max-time",
        str(int(timeout)),
        url,
        "-o",
        str(output_path),
    ]
    proc = subprocess.run(
        timeout_command(cmd, float(timeout) + 5.0, kill_after=5),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=subprocess_timeout(float(timeout) + 5.0, kill_after=5),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"power CSV download failed from {url}: {proc.stdout}")


def render_power_png(csv_path: Path) -> tuple[Path | None, str]:
    png_path = csv_path.with_suffix(".png")
    error_path = csv_path.with_suffix(".render_error.txt")
    timeout = 120.0
    cmd = [
        sys.executable,
        str(SKILL_ROOT / "scripts" / "render_power_csv.py"),
        str(csv_path),
    ]
    proc = subprocess.run(
        timeout_command(cmd, timeout, kill_after=5),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=subprocess_timeout(timeout, kill_after=5),
    )
    if proc.returncode == 0 and png_path.exists():
        if error_path.exists():
            error_path.unlink()
        return png_path, ""
    message = proc.stdout.strip() or f"render_power_csv.py exited with {proc.returncode}"
    error_path.write_text(message + "\n")
    return None, message


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("+", "plus")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown"


def power_start_payload(args: argparse.Namespace, device: dict[str, Any]) -> dict[str, Any]:
    device_key = str(device.get("device_key", ""))
    requested_device = str(args.power_device or device_key).strip().lower().replace("-", "_")
    if requested_device in {"rhinopi", "rhino_pi", "rhino_pi_x1", "aidlux", "adreno"}:
        requested_device = "rhino"
    if requested_device in {"orangepi_mali", "orangepi5plus", "orange_pi_5_plus", "orange_pi5_plus", "opi5plus", "opi_5_plus"}:
        requested_device = "orangepi"
    defaults = dict(POWER_DEVICE_DEFAULTS.get(requested_device, {}))

    payload: dict[str, Any] = {}
    explicit_serial = str(args.power_serial or "").strip()
    explicit_type = str(args.power_monitor_type or "").strip()
    if explicit_serial:
        defaults.pop("device", None)
        defaults["serial"] = explicit_serial
    if explicit_type:
        defaults["monitor_type"] = explicit_type

    if defaults.get("device") and not explicit_serial:
        payload["device"] = defaults["device"]
    else:
        if defaults.get("serial"):
            payload["serial"] = defaults["serial"]
        if defaults.get("monitor_type"):
            payload["monitor_type"] = defaults["monitor_type"]
        if args.power_port:
            payload["port"] = args.power_port

    sample_rate = args.power_sample_rate_hz or defaults.get("sample_rate_hz")
    voltage = args.power_voltage_mv or defaults.get("voltage_mv")
    if sample_rate:
        payload["sample_rate_hz"] = int(sample_rate)
    if voltage:
        payload["voltage_mv"] = int(voltage)
    if args.power_max_duration_sec:
        payload["max_duration_sec"] = float(args.power_max_duration_sec)
    return payload


def power_process_dir(args: argparse.Namespace, device: dict[str, Any]) -> Path:
    root = Path(args.power_output_root)
    process_id = str(getattr(args, "power_process_id", "") or args.run_id)
    process_dir = root / str(device["device_key"]) / str(args.power_day) / str(args.power_hour) / str(args.power_minute) / slugify(process_id)
    process_dir.mkdir(parents=True, exist_ok=True)
    return process_dir


def power_case_label(device: dict[str, Any], ctx: int, mode: str, budget: str) -> str:
    budget_slug = "full" if str(budget) == "full" else str(budget).replace(".", "p")
    return "_".join(
        [
            slugify(device.get("model_name", "")),
            f"ctx{ctx}",
            slugify(mode),
            f"b{slugify(budget_slug)}",
        ]
    )


def write_power_config(path: Path, metadata: dict[str, Any]) -> None:
    keys = [
        "device",
        "device_display",
        "backend",
        "frequency_profile",
        "frequency_note",
        "run_id",
        "benchmark_csv",
        "benchmark_csv_row_mode",
        "power_process_dir",
        "power_uuid",
        "power_rep_total",
        "power_rep_interval_sec",
        "power_warmup_sec",
        "power_cooldown_sec",
        "power_repetition_results",
        "model_key",
        "model",
        "context_tokens",
        "algorithm",
        "case_label",
        "budget",
        "pic_recompute_ratio",
        "pic_recompute_score_layer_idx",
        "power_api_url",
        "power_start_request",
        "power_monitor",
        "power_session_id",
        "power_run_id",
        "power_txt",
        "power_csv",
        "power_png",
        "power_start_json",
        "power_start_response_json",
        "power_stop_json",
        "power_stop_response_json",
        "prefill_latency_s",
        "prefill_tps",
        "status",
        "error",
        "started_at_local",
        "stopped_at_local",
    ]
    lines: list[str] = []
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            value_text = str(value)
        lines.append(f"{key}={value_text}")
    path.write_text("\n".join(lines) + "\n")


def append_power_manifest(process_dir: Path, metadata: dict[str, Any]) -> None:
    manifest_path = process_dir / "manifest.tsv"
    fieldnames = [
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
        "algorithm",
        "case_label",
        "budget",
        "pic_recompute_ratio",
        "pic_recompute_score_layer_idx",
        "prefill_latency_s",
        "prefill_tps",
        "power_csv",
        "power_png",
        "power_txt",
        "power_run_id",
        "status",
        "error",
    ]
    exists = manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        row: dict[str, Any] = {}
        for key in fieldnames:
            value = metadata.get(key, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            row[key] = value
        writer.writerow(row)


def extract_power_result_metadata(result: Any) -> dict[str, Any]:
    row: dict[str, Any] | None = None
    status: Any = None
    if isinstance(result, dict):
        row = result
    elif isinstance(result, tuple):
        if len(result) >= 1:
            status = result[0]
        if len(result) >= 3 and isinstance(result[2], dict):
            row = result[2]
    out: dict[str, Any] = {}
    if status is not None:
        out["status"] = status
    if row:
        for key in ["prefill_latency_s", "prefill_tps", "execution_mode", "recompute_token_count"]:
            if key in row:
                out[key] = row[key]
    return out


def combine_power_operation_results(results: list[Any]) -> Any:
    if not results:
        return None
    latencies: list[float] = []
    tps_values: list[float] = []
    for item in results:
        meta = extract_power_result_metadata(item)
        try:
            latencies.append(float(meta.get("prefill_latency_s", "")))
        except Exception:
            pass
        try:
            tps_values.append(float(meta.get("prefill_tps", "")))
        except Exception:
            pass
    last = results[-1]
    if isinstance(last, dict):
        row = dict(last)
        if latencies:
            row["prefill_latency_s"] = sum(latencies) / len(latencies)
        if tps_values:
            row["prefill_tps"] = sum(tps_values) / len(tps_values)
        return row
    if isinstance(last, tuple) and len(last) >= 3 and isinstance(last[2], dict):
        row = dict(last[2])
        if latencies:
            row["prefill_latency_s"] = sum(latencies) / len(latencies)
        if tps_values:
            row["prefill_tps"] = sum(tps_values) / len(tps_values)
        return (*last[:2], row, *last[3:])
    return last


def run_with_power_capture(
    args: argparse.Namespace,
    device: dict[str, Any],
    ctx: int,
    mode: str,
    budget: str,
    ratio: float | None,
    operation,
) -> Any:
    if not bool(getattr(args, "power_capture", False)):
        return operation()

    result: Any = None
    rep_total = max(int(getattr(args, "power_rep", 1) or 1), 1)
    result = run_one_power_capture(
        args,
        device,
        ctx,
        mode,
        budget,
        ratio,
        rep_total,
        operation,
    )
    return result


def run_one_power_capture(
    args: argparse.Namespace,
    device: dict[str, Any],
    ctx: int,
    mode: str,
    budget: str,
    ratio: float | None,
    rep_total: int,
    operation,
) -> Any:
    api_url = str(args.power_api_url or DEFAULT_POWER_API_URL)
    process_dir = power_process_dir(args, device)
    case_uuid = uuid.uuid4().hex
    case_label = power_case_label(device, int(ctx), mode, budget)
    txt_path = process_dir / f"{case_uuid}.txt"
    start_json_path = process_dir / f"{case_uuid}.start.json"
    stop_json_path = process_dir / f"{case_uuid}.stop.json"
    start_response_path = process_dir / f"{case_uuid}.start_response.json"
    stop_response_path = process_dir / f"{case_uuid}.stop_response.json"
    start_request = power_start_payload(args, device)
    metadata: dict[str, Any] = {
        "device": device["device_key"],
        "device_display": device["display"],
        "backend": backend_label(device["backend"]),
        "frequency_profile": str(device.get("frequency_profile", "")),
        "frequency_note": str(device.get("frequency_note", DEFAULT_MAX_FREQUENCY_NOTES.get(device["device_key"], ""))),
        "run_id": str(args.run_id),
        "benchmark_csv": str(getattr(args, "benchmark_csv", "") or ""),
        "benchmark_csv_row_mode": "existing" if bool(getattr(args, "only_benchmark_csv_rows", False)) else "",
        "power_process_dir": str(process_dir),
        "power_uuid": case_uuid,
        "power_rep_total": int(rep_total),
        "power_rep_interval_sec": POWER_REP_INTERVAL_SEC,
        "power_warmup_sec": float(args.power_warmup_sec or 0),
        "power_cooldown_sec": float(args.power_cooldown_sec or 0),
        "model_key": str(args.model_key),
        "model": device["model_name"],
        "context_tokens": int(ctx),
        "algorithm": mode,
        "case_label": case_label,
        "budget": budget,
        "pic_recompute_ratio": "" if ratio is None else ratio,
        "pic_recompute_score_layer_idx": 1 if ratio is not None else "",
        "power_api_url": api_url,
        "power_start_request": start_request,
        "power_txt": str(txt_path),
        "power_start_json": str(start_json_path),
        "power_start_response_json": str(start_response_path),
        "power_stop_json": str(stop_json_path),
        "power_stop_response_json": str(stop_response_path),
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_power_config(txt_path, metadata)
    write_json(start_json_path, start_request)

    print(
        f"[power] start {device['device_key']} ctx={ctx} {mode} {budget} "
        f"rep_total={rep_total} interval={POWER_REP_INTERVAL_SEC:g}s uuid={case_uuid} -> {process_dir}",
        flush=True,
    )
    start_status, start_response, start_elapsed = post_power_json(api_url, "/v1/power/start", start_request, timeout=60)
    write_json(start_response_path, {"status": start_status, "elapsed_s": start_elapsed, "response": start_response})
    if start_status != 200:
        metadata["status"] = start_status
        metadata["error"] = start_response.get("error", start_response.get("error_body", "power start failed"))
        write_power_config(txt_path, metadata)
        append_power_manifest(process_dir, metadata)
        raise RuntimeError(f"power start failed: status={start_status} response={start_response}")

    session_id = str(start_response.get("session_id", ""))
    if not session_id:
        metadata["error"] = "power start response missing session_id"
        write_power_config(txt_path, metadata)
        append_power_manifest(process_dir, metadata)
        raise RuntimeError("power start response missing session_id")
    metadata["power_session_id"] = session_id
    metadata["power_run_id"] = str(start_response.get("run_id", ""))
    metadata["power_monitor"] = start_response.get("monitor", {})
    write_power_config(txt_path, metadata)

    results: list[Any] = []
    op_error: BaseException | None = None
    try:
        if float(args.power_warmup_sec or 0) > 0:
            time.sleep(float(args.power_warmup_sec))
        repetition_results: list[dict[str, Any]] = []
        for rep_index in range(1, int(rep_total) + 1):
            if rep_index > 1:
                time.sleep(POWER_REP_INTERVAL_SEC)
            print(
                f"[power] measure {device['device_key']} ctx={ctx} {mode} {budget} "
                f"rep={rep_index}/{rep_total} uuid={case_uuid}",
                flush=True,
            )
            result = operation()
            results.append(result)
            rep_meta = extract_power_result_metadata(result)
            rep_meta["rep_index"] = rep_index
            repetition_results.append(rep_meta)
            metadata["power_repetition_results"] = repetition_results
            write_power_config(txt_path, metadata)
        combined_result = combine_power_operation_results(results)
        metadata.update(extract_power_result_metadata(combined_result))
        metadata["power_repetition_results"] = repetition_results
        return combined_result
    except BaseException as exc:
        op_error = exc
        metadata["error"] = repr(exc)
        raise
    finally:
        if float(args.power_cooldown_sec or 0) > 0:
            time.sleep(float(args.power_cooldown_sec))
        stop_error: RuntimeError | None = None
        stop_request = {"session_id": session_id, "timeout_sec": float(args.power_stop_timeout_sec)}
        write_json(stop_json_path, stop_request)
        stop_status, stop_response, stop_elapsed = post_power_json(
            api_url,
            "/v1/power/stop",
            stop_request,
            timeout=max(float(args.power_stop_timeout_sec) + 30.0, 60.0),
        )
        write_json(stop_response_path, {"status": stop_status, "elapsed_s": stop_elapsed, "response": stop_response})
        metadata["stopped_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if stop_status == 200:
            metadata.setdefault("status", 200)
            raw_url = ""
            artifacts = stop_response.get("artifacts") if isinstance(stop_response, dict) else {}
            if isinstance(artifacts, dict):
                csv_ref = artifacts.get("csv") if isinstance(artifacts.get("csv"), dict) else {}
                raw_url = str(csv_ref.get("raw_url") or csv_ref.get("download_url") or "")
            if not raw_url and stop_response.get("run_id"):
                raw_url = f"/v1/artifact/raw?run_id={stop_response['run_id']}&path=power.csv"
            if raw_url:
                csv_path = process_dir / f"{case_uuid}.csv"
                try:
                    get_power_raw(api_url, raw_url, csv_path, timeout=120)
                    metadata["power_csv"] = str(csv_path)
                    png_path, render_error = render_power_png(csv_path)
                    if png_path:
                        metadata["power_png"] = str(png_path)
                    elif render_error:
                        metadata["error"] = f"power PNG render failed: {render_error}"
                    print(f"[power] saved {csv_path}", flush=True)
                    if png_path:
                        print(f"[power] rendered {png_path}", flush=True)
                except Exception as exc:
                    metadata["error"] = f"power CSV download failed: {exc}"
                    if op_error is None:
                        stop_error = RuntimeError(metadata["error"])
            else:
                metadata["error"] = "power stop response missing CSV artifact URL"
                if op_error is None:
                    stop_error = RuntimeError(metadata["error"])
        else:
            metadata["status"] = stop_status
            metadata["error"] = stop_response.get("error", stop_response.get("error_body", "power stop failed"))
            if op_error is None:
                stop_error = RuntimeError(f"power stop failed: status={stop_status} response={stop_response}")
        write_power_config(txt_path, metadata)
        append_power_manifest(process_dir, metadata)
        if stop_error is not None:
            raise stop_error


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


def runtime_cache_dir(device: dict[str, Any]) -> str:
    if str(device.get("backend", "")).lower() == "opencl":
        return f"{device['remote_cache_root']}/runtime_cache/opencl"
    return f"{shared_kv_dir(device)}/runtime_cache"


def prepare_normal_runtime_cache(device: dict[str, Any]) -> dict[str, Any]:
    if str(device.get("backend", "")).lower() != "opencl":
        return {"skipped": True, "reason": "backend is not opencl"}
    cache_dir = runtime_cache_dir(device)
    script = f"""
set -e
normal_config={shlex.quote(device["normal_config"])}
cache_dir={shlex.quote(cache_dir)}
normal_dir="$(dirname "$normal_config")"
tmp_path="$normal_dir/tmp"
mkdir -p "$cache_dir"
cache_real="$(readlink -f "$cache_dir")"
status="ready"
backup_path=""
if [ -L "$tmp_path" ]; then
  tmp_real="$(readlink -f "$tmp_path" 2>/dev/null || true)"
  if [ "$tmp_real" != "$cache_real" ]; then
    if [ -f "$tmp_path/mnn_cachefile.bin" ] && [ ! -f "$cache_dir/mnn_cachefile.bin" ]; then
      cp -a "$tmp_path/mnn_cachefile.bin" "$cache_dir/mnn_cachefile.bin"
    fi
    rm "$tmp_path"
    ln -s "$cache_dir" "$tmp_path"
    status="retargeted_symlink"
  else
    status="already_bound"
  fi
elif [ -e "$tmp_path" ]; then
  if [ -f "$tmp_path/mnn_cachefile.bin" ] && [ ! -f "$cache_dir/mnn_cachefile.bin" ]; then
    cp -a "$tmp_path/mnn_cachefile.bin" "$cache_dir/mnn_cachefile.bin"
  fi
  backup_path="$tmp_path.model_cache_$(date -u +%Y%m%d%H%M%S)"
  mv "$tmp_path" "$backup_path"
  ln -s "$cache_dir" "$tmp_path"
  status="backed_up_and_bound"
else
  ln -s "$cache_dir" "$tmp_path"
  status="created_symlink"
fi
cache_file="$cache_dir/mnn_cachefile.bin"
printf '{{"status":"%s","normal_dir":"%s","tmp_path":"%s","runtime_cache_dir":"%s","cache_file":"%s","cache_file_exists":%s,"backup_path":"%s"}}\\n' \\
  "$status" "$normal_dir" "$tmp_path" "$cache_dir" "$cache_file" \\
  "$(test -f "$cache_file" && echo true || echo false)" "$backup_path"
""".strip()
    result = ssh(device, script, timeout=60)
    try:
        return json.loads((result.stdout or "{}").strip().splitlines()[-1])
    except Exception:
        return {"status": "unknown", "stdout": result.stdout}


def sanitize_systemd_unit_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name).strip())
    text = text.strip(".-")
    return text or "mnn-pic-bench"


def wrap_remote_command_with_memory_guard(
    command: str,
    *,
    limit_percent: int,
    unit_name: str | None = None,
    background: bool = False,
    timeout_sec: float | None = None,
) -> str:
    timeout_value = float(timeout_sec) if timeout_sec is not None else 0.0
    timeout_kill_after = max(1, int(DEFAULT_TIMEOUT_KILL_AFTER_SECONDS))

    def remote_timeout_command(inner: str) -> str:
        if timeout_value <= 0:
            return inner
        return (
            f"timeout -k {timeout_kill_after}s {max(1, int(timeout_value))}s "
            f"bash -lc {shlex.quote(inner)}"
        )

    if limit_percent <= 0:
        guarded = remote_timeout_command(command)
        if background:
            return f"nohup {guarded} >/dev/null 2>&1 &"
        return guarded
    if background and not unit_name:
        raise ValueError("background remote memory guard requires unit_name")
    unit_flag = f" --unit {shlex.quote(unit_name)} --collect" if unit_name else ""
    scope_flag = "" if background else " --scope"
    runtime_props = ""
    if timeout_value > 0:
        runtime_props = f""" \\
  -p RuntimeMaxSec={max(1, int(timeout_value))}s \\
  -p TimeoutStopSec={timeout_kill_after}s \\
  -p KillMode=control-group \\
  -p SendSIGKILL=yes"""
    guarded_command = command if background else remote_timeout_command(command)
    return f"""
command -v systemd-run >/dev/null 2>&1 || {{
  echo "systemd-run missing on remote device" >&2
  exit 127
}}
limit_mb="$(awk '/MemTotal/ {{printf "%d", ($2 * {int(limit_percent)} / 100) / 1024}}' /proc/meminfo)"
if [ -z "$limit_mb" ] || [ "$limit_mb" -le 0 ]; then
  echo "failed to compute remote memory limit" >&2
  exit 1
fi
systemd-run --user{scope_flag}{unit_flag} \\
  -p MemoryAccounting=yes \\
  -p MemoryMax="${{limit_mb}}M" \\
  -p MemorySwapMax=0{runtime_props} \\
  bash -lc {shlex.quote(guarded_command)}
""".strip()


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
        if mode in DISABLED_MODE_CHOICES:
            raise SystemExit(
                f"disabled mode {mode}: {DISABLED_MODE_CHOICES[mode]}; "
                f"expected one of: {', '.join(MODE_CHOICES)}"
            )
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
    requested_contexts: list[int],
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
        device_info = DEVICES[device]
        frequency_note = DEFAULT_MAX_FREQUENCY_NOTES.get(device, "cpu=max,gpu=max,ddr=max")
        expected_rows: list[dict[str, str]] = []
        for context in requested_contexts:
            for mode, budget, _ratio in selected_mode_specs(selected_modes, selected_ratios):
                expected_rows.append(
                    {
                        "backend": backend_label(str(device_info["backend"])),
                        "frequency_note": frequency_note,
                        "context_tokens": str(int(context)),
                        "mode": cli_mode_to_summary_mode(mode),
                        "budget": budget,
                    }
                )
            if "normal-full-recompute" in selected_modes:
                expected_rows.append(
                    {
                        "backend": backend_label(str(device_info["backend"])),
                        "frequency_note": frequency_note,
                        "context_tokens": str(int(context)),
                        "mode": "normal-full-recompute",
                        "budget": "full",
                    }
                )
        expected_by_key = {gap_compare_key(row): row for row in expected_rows}
        existing = {
            gap_compare_key(row)
            for row in rows
            if row.get("model") == target_model and row.get("device") == device
            and row_matches_filters(row, selected_modes, selected_ratios)
        }
        missing_keys = sorted(expected_by_key.keys() - existing)
        missing_contexts = sorted({int(item[2]) for item in missing_keys})
        report["devices"][device] = {
            "missing_count": len(missing_keys),
            "missing_contexts": missing_contexts,
            "missing_rows": [
                {
                    "device": device,
                    "device_display": str(device_info["display"]),
                    "model": target_model,
                    "backend": item[0],
                    "frequency_note": item[1],
                    "context_tokens": item[2],
                    "mode": item[3],
                    "budget": item[4],
                }
                for item in missing_keys
            ],
        }
    return report


def compute_existing_rows_report(
    benchmark_csv: Path,
    target_model: str,
    requested_devices: list[str],
    requested_contexts: list[int],
    selected_modes: list[str],
    selected_ratios: list[float],
) -> dict[str, Any]:
    with benchmark_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    requested_context_set = {str(int(context)) for context in requested_contexts}
    report: dict[str, Any] = {
        "benchmark_csv": str(benchmark_csv),
        "template_model": TEMPLATE_MODEL,
        "target_model": target_model,
        "selected_modes": selected_modes,
        "selected_ratios": selected_ratios,
        "devices": {},
    }
    for device in requested_devices:
        device_rows: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for row in rows:
            if row.get("model") != target_model or row.get("device") != device:
                continue
            if str(row.get("context_tokens", "")) not in requested_context_set:
                continue
            if not row_matches_filters(row, selected_modes, selected_ratios):
                continue
            key = csv_key(row)
            if key in seen:
                continue
            seen.add(key)
            device_rows.append({name: row.get(name, "") for name in CSV_KEYS})
        device_rows.sort(key=lambda item: (int(item["context_tokens"]), item["mode"], str(item["budget"])))
        report["devices"][device] = {
            "existing_count": len(device_rows),
            "existing_contexts": sorted({int(row["context_tokens"]) for row in device_rows}),
            "existing_rows": device_rows,
        }
    return report


def _port_pids_local(port: int) -> list[int]:
    commands = [
        ["lsof", f"-tiTCP:{int(port)}", "-sTCP:LISTEN"],
        ["fuser", "-n", "tcp", str(int(port))],
        ["sh", "-lc", f"ss -ltnpH 'sport = :{int(port)}' | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p'"],
        ["pgrep", "-f", f"ssh -N -L 127.0.0.1:{int(port)}:127.0.0.1:"],
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
    remote_runtime_cache = runtime_cache_dir(device)
    remote_log = f"{remote_run}/pic_server_{label}.log"
    remote_unit_path = f"{remote_run}/pic_server_{label}.unit"
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
    unit_name = sanitize_systemd_unit_name(f"mnn-pic-server-{device['device_key']}-{run_id}-{label}")
    server_command = f"""
cd '{device["remote_work"]}'
exec env {env_prefix}LD_LIBRARY_PATH='{device["ld_library_path"]}':${{LD_LIBRARY_PATH:-}} \\
  stdbuf -oL -eL '{device["artifact"]}/bin/pic_server' \\
  --config '{device["pic_config"]}' \\
  --host 127.0.0.1 --port {device["remote_port"]} \\
  --kv-cache-dir '{remote_kv}' --model '{device["served_model"]}' \\
  --runtime-cache-dir '{remote_runtime_cache}' \\
  > '{remote_log}' 2>&1
""".strip()
    server_command = wrap_remote_command_with_memory_guard(
        server_command,
        limit_percent=int(device.get("remote_memory_limit_percent", 0) or 0),
        unit_name=unit_name,
        background=True,
        timeout_sec=float(device.get("server_runtime_timeout_sec", DEFAULT_SERVER_RUNTIME_TIMEOUT_SECONDS)),
    )
    script = f"""
set -e
mkdir -p '{remote_run}' '{remote_kv}' '{remote_runtime_cache}'
{server_command}
printf '%s\\n' {shlex.quote(unit_name)} > '{remote_unit_path}'
"""
    ssh(device, script, timeout=120)
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
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS}",
                "-o",
                "ConnectionAttempts=1",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=1",
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
    log_tail = ssh(device, f"tail -n 120 '{remote_log}' || true", check=False, timeout=30).stdout
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


def build_normal_bench_command(
    device: dict[str, Any],
    ctx: int,
    rep: int,
    remote_json: str,
    remote_log: str,
    timeout_sec: float | None = None,
) -> str:
    env_parts = []
    if device.get("bench_env", "").strip():
        env_parts.append(device["bench_env"].strip())
    if str(device.get("backend", "")).lower() == "opencl":
        env_parts.append(f"MNN_LLM_RUNTIME_CACHE_DIR={shlex.quote(runtime_cache_dir(device))}")
    bench_env = (" ".join(env_parts) + " ") if env_parts else ""
    normal_command = f"""
cd "$(dirname '{device["normal_config"]}')"
env {bench_env}LD_LIBRARY_PATH='{device["ld_library_path"]}':${{LD_LIBRARY_PATH:-}} \\
  '{device["artifact"]}/bin/llm_bench' -m '{device["normal_config"]}' \\
  -a '{device["backend"]}' -p {ctx} -n 0 -rep {rep} -load false -j '{remote_json}' > '{remote_log}' 2>&1
""".strip()
    return wrap_remote_command_with_memory_guard(
        normal_command,
        limit_percent=int(device.get("remote_memory_limit_percent", 0) or 0),
        timeout_sec=timeout_sec if timeout_sec is not None else float(device.get("remote_bench_timeout_sec", DEFAULT_REMOTE_BENCH_TIMEOUT_SECONDS)),
    )


def parse_normal_result(ctx: int, stdout: str) -> tuple[dict[str, Any], float, str | float]:
    try:
        text = stdout[stdout.find("{"):]
        obj = json.loads(text)
    except Exception:
        obj = {"parse_error": stdout}
    prefill = None
    for item in obj.get("results", []):
        if isinstance(item, dict) and item.get("type") == "prefill":
            prefill = item
            break
    tps = float(prefill.get("tps")) if prefill and prefill.get("tps") else 0.0
    latency = (ctx / tps) if tps > 0 else ""
    return obj, tps, latency


def run_normal_warm(device: dict[str, Any], ctx: int, run_id: str, local_raw: Path) -> None:
    remote_run = f"{device['remote_cache_root']}/{run_id}/normal"
    remote_json = f"{remote_run}/warm_normal_ctx{ctx}.json"
    remote_log = f"{remote_run}/warm_normal_ctx{ctx}.log"
    bench_timeout = float(device.get("remote_bench_timeout_sec", DEFAULT_REMOTE_BENCH_TIMEOUT_SECONDS))
    normal_command = build_normal_bench_command(device, ctx, 1, remote_json, remote_log, timeout_sec=bench_timeout)
    script = f"""
set -e
mkdir -p '{remote_run}'
{normal_command}
cat '{remote_json}'
"""
    result = ssh(device, script, timeout=bench_timeout + 60.0)
    obj, tps, _ = parse_normal_result(ctx, result.stdout)
    write_json(local_raw / f"warm_normal_ctx{ctx}.json", obj)
    if tps <= 0:
        raise RuntimeError(f"normal warmup failed for ctx={ctx}")


def run_normal(device: dict[str, Any], ctx: int, run_id: str, local_raw: Path, rep: int) -> dict[str, Any]:
    remote_run = f"{device['remote_cache_root']}/{run_id}/normal"
    remote_json = f"{remote_run}/normal_ctx{ctx}.json"
    remote_log = f"{remote_run}/normal_ctx{ctx}.log"
    bench_timeout = float(device.get("remote_bench_timeout_sec", DEFAULT_REMOTE_BENCH_TIMEOUT_SECONDS))
    normal_command = build_normal_bench_command(device, ctx, rep, remote_json, remote_log, timeout_sec=bench_timeout)
    script = f"""
set -e
mkdir -p '{remote_run}'
{normal_command}
cat '{remote_json}'
"""
    result = ssh(device, script, timeout=bench_timeout + 60.0)
    obj, tps, latency = parse_normal_result(ctx, result.stdout)
    write_json(local_raw / f"normal_ctx{ctx}.json", obj)
    return {
        "device": device["device_key"],
        "device_display": device["display"],
        "model": device["model_name"],
        "backend": backend_label(device["backend"]),
        "frequency_note": str(device.get("frequency_note", DEFAULT_MAX_FREQUENCY_NOTES.get(device["device_key"], "cpu=max,gpu=max,ddr=max"))),
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


def prefill_text_cache(
    base_url: str,
    local_raw: Path,
    device: dict[str, Any],
    ctx: int,
    tokens: dict[str, Any],
    doc_id: str,
    force_cache_build: bool,
    *,
    suffix: str = "",
) -> tuple[int, dict[str, Any], float]:
    post_json(base_url, "/reset", {"reset": True}, timeout=30)
    status, response, elapsed = post_json(
        base_url,
        "/v1/prefill/text",
        {
            "id": doc_id,
            "type": "text",
            "token_ids": tokens["doc_token_ids"],
            "force": bool(force_cache_build),
        },
        timeout=1800,
    )
    filename = f"prefill_text_ctx{ctx}{suffix}.json"
    write_json(
        local_raw / filename,
        {
            "status": status,
            "latency_s": elapsed,
            "request_meta": {
                "doc_id": doc_id,
                "shared_cache_dir": shared_kv_dir(device),
                "runtime_cache_dir": runtime_cache_dir(device),
                "force_cache_build": bool(force_cache_build),
            },
            "cache_status": response.get("cache_status", "") if isinstance(response, dict) else "",
            "cache_hit": response.get("cache_hit", "") if isinstance(response, dict) else "",
            "response": response,
        },
    )
    return status, response, elapsed


def evict_document_kv_from_page_cache(device: dict[str, Any], doc_id: str) -> dict[str, Any]:
    """Evict only this request's persistent .k/.v files after warmup."""
    cache_root = shared_kv_dir(device)
    script = f"""
set -e
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path({cache_root!r})
doc_id = {doc_id!r}
cache_object_id = "doc_" + doc_id
paths = sorted(
    path for path in root.glob(f"objects/**/{{cache_object_id}}/layers/*")
    if path.is_file() and path.suffix in {{".k", ".v"}}
)
if not paths:
    raise SystemExit(f"no persistent KV files found for {{cache_object_id}} under {{root}}")
os.sync()
evicted_bytes = 0
for path in paths:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    evicted_bytes += path.stat().st_size
print(json.dumps({{"doc_id": doc_id, "cache_object_id": cache_object_id, "file_count": len(paths), "evicted_bytes": evicted_bytes}}))
PY
""".strip()
    result = ssh(device, script, timeout=120)
    for line in reversed((result.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"failed to parse persistent KV eviction result: {result.stdout!r}")


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
    device["normal_config"] = resolve_remote_file(
        device,
        device["normal_config_candidates"],
        "normal_config",
    )
    device["pic_config"] = resolve_remote_file(
        device,
        device["pic_config_candidates"],
        "pic_config",
    )
    device["server_env_extra"] = list(args.server_env)
    device["remote_memory_limit_percent"] = int(args.remote_memory_limit_percent)
    device["ssh_command_timeout_sec"] = float(args.ssh_command_timeout_sec)
    device["remote_bench_timeout_sec"] = float(args.remote_bench_timeout_sec)
    device["server_runtime_timeout_sec"] = float(args.server_runtime_timeout_sec)
    out_dir = Path(args.output_dir) / args.run_id
    local_raw = out_dir / "raw" / device_name
    local_raw.mkdir(parents=True, exist_ok=True)
    device["frequency_profile"] = str(args.frequency_profile)
    frequency_state = apply_frequency_profile(device, device["frequency_profile"])
    device["frequency_state"] = frequency_state
    device["frequency_note"] = format_frequency_note(device, frequency_state)
    device["runtime_cache_dir"] = runtime_cache_dir(device)
    runtime_cache_preparation = prepare_normal_runtime_cache(device)
    device["runtime_cache_preparation"] = runtime_cache_preparation
    write_json(local_raw / "device.json", device)
    write_json(local_raw / "frequency_state.json", frequency_state)
    write_json(local_raw / "runtime_cache_preparation.json", runtime_cache_preparation)
    base_tokens = load_base_tokens(Path(args.base_tokens))

    for ctx in contexts:
        high_risk_error = None if bool(args.allow_high_risk_contexts) else high_risk_context_error(
            device_name,
            str(device.get("frequency_profile", "")),
            int(ctx),
        )
        if high_risk_error:
            print(f"[guard] skip {device_name} ctx={ctx}: {high_risk_error}", flush=True)
            append_jsonl(
                failures_jsonl,
                failure_row(
                    device,
                    int(ctx),
                    "guard",
                    "context",
                    "full",
                    error=high_risk_error,
                ),
            )
            continue
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
                if str(device.get("backend", "")).lower() == "opencl" and not args.no_warm:
                    print(f"[warm] {device_name} ctx={ctx} normal full", flush=True)
                    run_normal_warm(device, int(ctx), args.run_id, local_raw)
                normal_row = run_with_power_capture(
                    args,
                    device,
                    int(ctx),
                    "normal-full-recompute",
                    "full",
                    None,
                    lambda: run_normal(
                        device,
                        int(ctx),
                        args.run_id,
                        local_raw,
                        1 if bool(getattr(args, "power_capture", False)) else int(args.normal_rep),
                    ),
                )
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
            write_json(
                local_raw / f"doc_id_ctx{ctx}.json",
                {
                    "id": doc_id,
                    "shared_cache_dir": shared_kv_dir(device),
                    "runtime_cache_dir": runtime_cache_dir(device),
                },
            )

            if args.restart_server_each_spec:
                for mode, budget, ratio in requested_mode_specs:
                    spec_label = f"{label}_{mode}_{budget.replace('.', 'p')}"
                    tunnel = None
                    try:
                        tunnel = start_server(device, args.run_id, local_raw, spec_label)
                        base_url = f"http://127.0.0.1:{device['local_port']}"
                        status, response, _ = prefill_text_cache(
                            base_url,
                            local_raw,
                            device,
                            int(ctx),
                            tokens,
                            doc_id,
                            bool(args.force_cache_build),
                            suffix=f"_{mode}_{budget.replace('.', 'p')}",
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

                        warm_ok = True
                        if not args.no_warm:
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
                            if warm_status != 200:
                                warm_ok = False
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
                            else:
                                tune_status, tune_response, tune_elapsed = post_json(base_url, "/v1/tune/update_cache", {}, timeout=120)
                                write_json(
                                    local_raw / f"update_runtime_cache_ctx{ctx}_{mode}_{budget.replace('.', 'p')}.json",
                                    {"status": tune_status, "latency_s": tune_elapsed, "response": tune_response},
                                )
                        if not warm_ok:
                            continue

                        if args.evict_document_kv_before_measure:
                            write_json(
                                local_raw / f"evict_kv_ctx{ctx}_{mode}_{budget.replace('.', 'p')}.json",
                                evict_document_kv_from_page_cache(device, doc_id),
                            )
                        print(f"[measure] {device_name} ctx={ctx} {mode} {budget}", flush=True)
                        measure_status, measure_response, row = run_with_power_capture(
                            args,
                            device,
                            int(ctx),
                            mode,
                            budget,
                            ratio,
                            lambda: run_chat(
                                base_url,
                                local_raw,
                                tokens,
                                doc_id,
                                mode,
                                budget,
                                ratio,
                                device["served_model"],
                                f"ctx{ctx}",
                            ),
                        )
                        row.update(
                            {
                                "device": device_name,
                                "device_display": device["display"],
                                "model": device["model_name"],
                                "backend": backend_label(device["backend"]),
                                "frequency_note": str(device.get("frequency_note", DEFAULT_MAX_FREQUENCY_NOTES.get(device_name, "cpu=max,gpu=max,ddr=max"))),
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
                    finally:
                        tail_server_log(device, args.run_id, spec_label, local_raw)
                        stop_server(device, tunnel)
            else:
                tunnel = start_server(device, args.run_id, local_raw, label)
                base_url = f"http://127.0.0.1:{device['local_port']}"

                status, response, _ = prefill_text_cache(
                    base_url,
                    local_raw,
                    device,
                    int(ctx),
                    tokens,
                    doc_id,
                    bool(args.force_cache_build),
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
                    if warm_success:
                        tune_status, tune_response, tune_elapsed = post_json(base_url, "/v1/tune/update_cache", {}, timeout=120)
                        write_json(
                            local_raw / f"update_runtime_cache_ctx{ctx}.json",
                            {"status": tune_status, "latency_s": tune_elapsed, "response": tune_response},
                        )

                for mode, budget, ratio in measure_specs:
                    if args.evict_document_kv_before_measure:
                        write_json(
                            local_raw / f"evict_kv_ctx{ctx}_{mode}_{budget.replace('.', 'p')}.json",
                            evict_document_kv_from_page_cache(device, doc_id),
                        )
                    print(f"[measure] {device_name} ctx={ctx} {mode} {budget}", flush=True)
                    measure_status, measure_response, row = run_with_power_capture(
                        args,
                        device,
                        int(ctx),
                        mode,
                        budget,
                        ratio,
                        lambda: run_chat(
                            base_url,
                            local_raw,
                            tokens,
                            doc_id,
                            mode,
                            budget,
                            ratio,
                            device["served_model"],
                            f"ctx{ctx}",
                        ),
                    )
                    row.update(
                        {
                            "device": device_name,
                            "device_display": device["display"],
                            "model": device["model_name"],
                            "backend": backend_label(device["backend"]),
                            "frequency_note": str(device.get("frequency_note", DEFAULT_MAX_FREQUENCY_NOTES.get(device_name, "cpu=max,gpu=max,ddr=max"))),
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


def parse_devices(text: str, *, allow_subset_devices: bool, allow_extra_device: bool) -> list[str]:
    devices: list[str] = []
    seen: set[str] = set()
    for item in str(text).split(","):
        device = item.strip()
        if not device:
            continue
        if device not in DEVICES:
            raise SystemExit(f"unknown device {device}; expected one of: {', '.join(sorted(DEVICES))}")
        if device not in seen:
            devices.append(device)
            seen.add(device)
    if not devices:
        raise SystemExit("no devices selected")
    extra_selected = [device for device in devices if device in EXTRA_PROFILE_DEVICES]
    if extra_selected and not allow_extra_device:
        raise SystemExit(
            "extra-profile devices require --allow-extra-device: "
            + ",".join(extra_selected)
            + ". Formal regression still uses jetson,orangepi as the required pair."
        )
    if not allow_subset_devices:
        missing = [device for device in FORMAL_REQUIRED_DEVICES if device not in seen]
        if missing:
            required = ",".join(FORMAL_REQUIRED_DEVICES)
            raise SystemExit(
                f"formal prefill sweep requires --devices to include {required}; "
                "use --allow-subset-devices only for targeted single-device debug/profile runs"
            )
    return devices


def print_gap_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        default=",".join(FORMAL_REQUIRED_DEVICES),
        help="Comma-separated device list. Formal sweep default is jetson,orangepi; add rhino explicitly if needed.",
    )
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--model-key", default="llama3.2-1b", choices=sorted(MODELS))
    parser.add_argument(
        "--modes",
        default=",".join(MODE_CHOICES),
        help="Comma-separated modes. PIC full-compute/pic-full-recompute is intentionally disabled.",
    )
    parser.add_argument("--ratios", default=",".join(f"{ratio:.2f}" for ratio in RATIOS))
    parser.add_argument(
        "--server-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE env assignment for remote pic_server; may be repeated.",
    )
    parser.add_argument(
        "--frequency-profile",
        default="max",
        help="Device frequency setup profile before benchmark. max locks OrangePi/Rhino where supported; other named profiles are Rhino-only.",
    )
    parser.add_argument("--run-id", default="prefill_latency_" + time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", default=str(MNN_ROOT / ".cache/latency_budget_20260625"))
    parser.add_argument("--base-tokens", default=str(DEFAULT_BASE_TOKENS))
    parser.add_argument("--benchmark-csv", default="")
    parser.add_argument(
        "--only-benchmark-csv-rows",
        action="store_true",
        help="Run only rows already present in --benchmark-csv for the selected device/model/contexts/modes/ratios.",
    )
    parser.add_argument("--only-missing-contexts", action="store_true")
    parser.add_argument("--print-gap-only", action="store_true")
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--restart-server-each-spec", action="store_true")
    parser.add_argument("--force-cache-build", action="store_true")
    parser.add_argument(
        "--evict-document-kv-before-measure",
        action="store_true",
        help="After warmup, evict only the selected persistent .k/.v files from the OS page cache before each measured PIC request.",
    )
    parser.add_argument("--normal-rep", type=int, default=3)
    parser.add_argument(
        "--power-capture",
        action="store_true",
        help="Capture external power for each measured benchmark row. Warm/cache/tune steps are not captured.",
    )
    parser.add_argument(
        "--power-api-url",
        default=os.environ.get("MNN_POWER_API_URL", DEFAULT_POWER_API_URL),
        help=f"eperf power API URL. Defaults to {DEFAULT_POWER_API_URL}.",
    )
    parser.add_argument(
        "--power-output-root",
        default="",
        help="Power output root. Defaults to .codex/skills/mnn-pic-benchmark/power.",
    )
    parser.add_argument(
        "--power-device",
        default=os.environ.get("MNN_POWER_DEVICE", ""),
        help="Power device alias override. Defaults to the benchmark device.",
    )
    parser.add_argument("--power-monitor-type", default=os.environ.get("MNN_POWER_MONITOR_TYPE", ""))
    parser.add_argument("--power-serial", default=os.environ.get("MNN_POWER_SERIAL", ""))
    parser.add_argument("--power-port", default=os.environ.get("MNN_POWER_PORT", ""))
    parser.add_argument("--power-sample-rate-hz", type=int, default=int(os.environ["MNN_POWER_SAMPLE_RATE_HZ"]) if os.environ.get("MNN_POWER_SAMPLE_RATE_HZ") else 0)
    parser.add_argument("--power-voltage-mv", type=int, default=int(os.environ["MNN_POWER_VOLTAGE_MV"]) if os.environ.get("MNN_POWER_VOLTAGE_MV") else 0)
    parser.add_argument("--power-max-duration-sec", type=float, default=float(os.environ.get("MNN_POWER_MAX_DURATION_SEC", "1800")))
    parser.add_argument("--power-stop-timeout-sec", type=float, default=float(os.environ.get("MNN_POWER_STOP_TIMEOUT_SEC", "30")))
    parser.add_argument(
        "--power-warmup-sec",
        type=float,
        default=float(os.environ.get("MNN_POWER_WARMUP_SEC", "5")),
        help="Idle seconds after Power API start and before the first measured rep. Defaults to 5.",
    )
    parser.add_argument(
        "--power-cooldown-sec",
        type=float,
        default=float(os.environ.get("MNN_POWER_COOLDOWN_SEC", "5")),
        help="Idle seconds after the last measured rep and before Power API stop. Defaults to 5.",
    )
    parser.add_argument("--power-rep", type=int, default=int(os.environ.get("MNN_POWER_REP", "3")))
    parser.add_argument(
        "--remote-memory-limit-percent",
        type=int,
        default=int(os.environ.get("PIC_SWEEP_REMOTE_MEMORY_LIMIT_PERCENT", "95")),
        help="Wrap remote llm_bench/pic_server in systemd-run with MemoryMax set to this percent of MemTotal; 0 disables.",
    )
    parser.add_argument(
        "--ssh-command-timeout-sec",
        type=float,
        default=DEFAULT_SSH_COMMAND_TIMEOUT_SECONDS,
        help="Outer timeout for each SSH command. A timeout -k prefix is used so hung banner/session commands are killed.",
    )
    parser.add_argument(
        "--remote-bench-timeout-sec",
        type=float,
        default=DEFAULT_REMOTE_BENCH_TIMEOUT_SECONDS,
        help="Remote timeout for llm_bench commands, also mirrored by the local SSH command timeout.",
    )
    parser.add_argument(
        "--server-runtime-timeout-sec",
        type=float,
        default=DEFAULT_SERVER_RUNTIME_TIMEOUT_SECONDS,
        help="systemd RuntimeMaxSec for remote pic_server units; prevents orphaned servers after host-side failures.",
    )
    parser.add_argument(
        "--allow-high-risk-contexts",
        action="store_true",
        help="Allow device high-frequency sweeps to run contexts above the default safety limit.",
    )
    parser.add_argument(
        "--allow-subset-devices",
        action="store_true",
        help="Allow targeted runs that omit formal required devices jetson,orangepi. Use only for focused debug/profile.",
    )
    parser.add_argument(
        "--allow-extra-device",
        action="store_true",
        help="Allow extra-profile devices such as rhino. Formal regression still requires jetson,orangepi unless --allow-subset-devices is also set.",
    )
    args = parser.parse_args()
    if args.remote_memory_limit_percent < 0 or args.remote_memory_limit_percent >= 100:
        raise SystemExit("--remote-memory-limit-percent must be in [0, 99]")
    if args.ssh_command_timeout_sec <= 0:
        raise SystemExit("--ssh-command-timeout-sec must be positive")
    if args.remote_bench_timeout_sec <= 0:
        raise SystemExit("--remote-bench-timeout-sec must be positive")
    if args.server_runtime_timeout_sec <= 0:
        raise SystemExit("--server-runtime-timeout-sec must be positive")
    if args.power_rep <= 0:
        raise SystemExit("--power-rep must be positive")
    if args.power_capture and args.model_key in POWER_EXCLUDED_MODEL_KEYS:
        raise SystemExit(
            f"power capture excludes model-key {args.model_key}; "
            "do not run Llama3.2 1B power tests"
        )

    devices = parse_devices(
        str(args.devices),
        allow_subset_devices=bool(args.allow_subset_devices),
        allow_extra_device=bool(args.allow_extra_device),
    )
    base_contexts = parse_contexts(args.contexts)
    args.selected_modes = parse_modes(args.modes)
    args.selected_ratios = parse_ratios(args.ratios)
    fieldnames = CSV_KEYS + ["prefill_latency_s", "prefill_tps"]

    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.power_output_root:
        args.power_output_root = str(SKILL_ROOT / "power")
    power_now = time.localtime()
    args.power_day = time.strftime("%Y-%m-%d", power_now)
    args.power_hour = time.strftime("%H", power_now)
    args.power_minute = time.strftime("%M", power_now)
    args.power_process_id = f"{args.run_id}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    summary_csv = out_dir / "summary.csv"
    failures_jsonl = out_dir / "failures.jsonl"

    target_model = MODELS[args.model_key]["name"]
    gap_report = None
    device_contexts: dict[str, list[int]] = {device: list(base_contexts) for device in devices}
    device_missing_rows: dict[str, dict[int, list[dict[str, Any]]]] = {}
    if args.benchmark_csv:
        if args.only_benchmark_csv_rows:
            existing_report = compute_existing_rows_report(
                Path(args.benchmark_csv),
                target_model,
                devices,
                base_contexts,
                args.selected_modes,
                args.selected_ratios,
            )
            write_json(out_dir / "benchmark_rows_report.json", existing_report)
            device_contexts = {}
            for device in devices:
                rows = existing_report["devices"][device]["existing_rows"]
                grouped: dict[int, list[dict[str, Any]]] = {}
                for row in rows:
                    ctx = int(row["context_tokens"])
                    grouped.setdefault(ctx, []).append(row)
                if grouped:
                    device_contexts[device] = sorted(grouped)
                    device_missing_rows[device] = grouped
        else:
            gap_report = compute_missing_report(
                Path(args.benchmark_csv),
                target_model,
                devices,
                base_contexts,
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
            print_gap_report(gap_report if gap_report is not None else existing_report)
            return 0

    write_json(
        out_dir / "run_config.json",
        {
            "devices": list(device_contexts),
            "device_roles": {
                device: ("formal" if device in FORMAL_REQUIRED_DEVICES else "extra-profile")
                for device in device_contexts
            },
            "formal_required_devices": list(FORMAL_REQUIRED_DEVICES),
            "extra_profile_devices": list(EXTRA_PROFILE_DEVICES),
            "formal_matrix_complete": all(device in device_contexts for device in FORMAL_REQUIRED_DEVICES),
            "model_key": args.model_key,
            "model": target_model,
            "contexts": device_contexts,
            "modes": args.selected_modes,
            "ratios": args.selected_ratios,
            "frequency_profile": args.frequency_profile,
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
            "evict_document_kv_before_measure": bool(args.evict_document_kv_before_measure),
            "restart_server_each_spec": bool(args.restart_server_each_spec),
            "benchmark_csv": args.benchmark_csv,
            "only_benchmark_csv_rows": bool(args.only_benchmark_csv_rows),
            "only_missing_contexts": bool(args.only_missing_contexts),
            "base_tokens": str(args.base_tokens),
            "server_env": list(args.server_env),
            "remote_memory_limit_percent": int(args.remote_memory_limit_percent),
            "power_capture": bool(args.power_capture),
            "power_api_url": str(args.power_api_url),
            "power_output_root": str(args.power_output_root),
            "power_device": str(args.power_device),
            "power_rep": int(args.power_rep),
            "power_process_id": str(args.power_process_id),
            "power_defaults": POWER_DEVICE_DEFAULTS,
            "power_excluded_model_keys": sorted(POWER_EXCLUDED_MODEL_KEYS),
            "allow_high_risk_contexts": bool(args.allow_high_risk_contexts),
            "allow_subset_devices": bool(args.allow_subset_devices),
            "allow_extra_device": bool(args.allow_extra_device),
            "high_risk_context_policies": {
                device: {
                    "profiles": sorted(str(item) for item in HIGH_RISK_CONTEXT_POLICIES.get(device, {}).get("profiles", set())),
                    "max_context": int(HIGH_RISK_CONTEXT_POLICIES.get(device, {}).get("max_context", 0) or 0),
                }
                for device in device_contexts
                if device in HIGH_RISK_CONTEXT_POLICIES
            },
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
