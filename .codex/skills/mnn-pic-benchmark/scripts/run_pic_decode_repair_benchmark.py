#!/usr/bin/env python3
"""Collect MNN PIC decode-repair TPOT/TPS into benchmark_decode.csv.

This client assumes a pic_server is already running for exactly one target
device. Run it once per device, appending to the same CSV if desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CONTEXTS = "512,1024,1536,2048,2560"
DEFAULT_BUDGETS = "0.00"
DEFAULT_REPAIR_TOKENS = "0,1,3,5,7"
DEFAULT_DECODE_SELECTORS = "top_hkvd,lagged_attention_hkvd"
MNN_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TOKEN_IDS_JSON = MNN_ROOT / ".cache/latency_budget_20260611/opencl_pic_1024_current_20260622_191527/tokens.json"
FORMAL_REQUIRED_DEVICES = ("jetson", "orangepi")
EXTRA_PROFILE_DEVICES = ("rhino",)
DEVICE_CHOICES = tuple(sorted(FORMAL_REQUIRED_DEVICES + EXTRA_PROFILE_DEVICES))
HIGH_RISK_CONTEXT_POLICIES: dict[str, dict[str, Any]] = {
    "jetson": {
        "profiles": {"max"},
        "max_context": 2560,
        "reason": "jetson max-frequency decode sweep blocks context > 2560 by default to avoid OOM/shutdown risk",
    },
    "orangepi": {
        "profiles": {"max"},
        "max_context": 2560,
        "reason": "orangepi max-frequency decode sweep blocks context > 2560 by default to avoid OOM/shutdown risk",
    },
    "rhino": {
        "profiles": {"max", "cpu-high-gpu-max"},
        "max_context": 2560,
        "reason": "rhino high-frequency decode sweep blocks context > 2560 by default to avoid shutdown risk",
    },
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_head_ids(value: str) -> list[int]:
    if not str(value).strip():
        return []
    return parse_csv_ints(value)


def validate_device(device: str, *, allow_extra_device: bool) -> None:
    if device not in DEVICE_CHOICES:
        raise SystemExit(f"unknown device {device}; expected one of: {', '.join(DEVICE_CHOICES)}")
    if device in EXTRA_PROFILE_DEVICES and not allow_extra_device:
        formal = ",".join(FORMAL_REQUIRED_DEVICES)
        raise SystemExit(
            f"formal decode repair matrix requires one of {formal} per run; "
            f"{device} is extra-profile only, use --allow-extra-device for targeted debug/profile runs"
        )


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


def json_safe_policy(policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(policy)
    profiles = out.get("profiles")
    if isinstance(profiles, set):
        out["profiles"] = sorted(profiles)
    return out


def is_unsupported_error(message: str) -> bool:
    lowered = message.lower()
    return "unsupported" in lowered or "not implemented" in lowered


def post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + endpoint
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {endpoint} failed HTTP {exc.code}: {detail}") from exc
    if status < 200 or status >= 300:
        detail = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {endpoint} failed HTTP {status}: {detail}")
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8", errors="replace"))


def unwrap_chat_response(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return response


def make_doc_content(target_tokens: int, seed: str) -> str:
    # Repeated " token" is close to one token on common BPE tokenizers. The
    # server-reported token_count remains authoritative in the CSV.
    n = max(1, int(target_tokens))
    return (f"{seed} " + " token" * n).strip()


def load_doc_token_ids(path: str) -> list[int]:
    if not path:
        return []
    token_path = Path(path)
    if not token_path.exists():
        return []
    obj = json.loads(token_path.read_text(encoding="utf-8"))
    source = obj.get("doc_token_ids") or obj.get("token_ids") or obj.get("full_prompt_token_ids") or []
    if not isinstance(source, list):
        return []
    return [int(item) for item in source if isinstance(item, int)]


def make_doc_token_ids(target_tokens: int, base_doc_token_ids: list[int]) -> list[int]:
    if target_tokens <= 0 or not base_doc_token_ids:
        return []
    token_ids: list[int] = []
    while len(token_ids) < target_tokens:
        token_ids.extend(base_doc_token_ids)
    return token_ids[:target_tokens]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt_budget(value: float) -> str:
    return f"{value:.2f}"


def build_pic_spec(
    *,
    pic_id: str,
    doc_id: str,
    mode: str,
    budget: float,
    score_layer_idx: int,
    token_ids: list[int],
    suffix_tokens: int,
    repair_tokens: int,
    selector: str,
    attention_layer_idx: int,
    attention_head_ids: list[int],
    top_m: int,
) -> dict[str, Any]:
    suffix = token_ids[-suffix_tokens:] if suffix_tokens > 0 else []
    full_prompt = list(token_ids) + list(suffix)
    spec: dict[str, Any] = {
        "id": pic_id,
        "text_cache_refs": [{"id": doc_id}],
        "selection_algorithm": mode,
        "pic_recompute_ratio": float(budget),
        "pic_recompute_score_layer_idx": int(score_layer_idx),
        "full_prompt_token_ids": full_prompt,
        "pic_token_start": 0,
        "token_count": len(token_ids),
    }
    if repair_tokens > 0:
        spec["decode_refine"] = {
            "enabled": True,
            "tokens_per_decode_step": int(repair_tokens),
            "top_m": int(top_m),
            "selector": selector,
            "attention_layer_idx": int(attention_layer_idx),
            "attention_head_ids": attention_head_ids,
        }
    return spec


def extract_metrics(response: dict[str, Any], requested_repair_tokens: int) -> dict[str, Any]:
    item = unwrap_chat_response(response)
    perf = item.get("performance") if isinstance(item.get("performance"), dict) else {}
    usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
    pic = item.get("pic_cache") if isinstance(item.get("pic_cache"), dict) else {}
    pr = pic.get("precision_recovery") if isinstance(pic.get("precision_recovery"), dict) else {}
    decode = pic.get("decode_refine") if isinstance(pic.get("decode_refine"), dict) else {}
    pr_decode = pr.get("decode_refine") if isinstance(pr.get("decode_refine"), dict) else {}
    metadata = pr.get("metadata") if isinstance(pr.get("metadata"), dict) else {}

    runtime = (
        metadata.get("decode_refine_runtime")
        or pr_decode.get("runtime")
        or decode.get("runtime")
        or ""
    )
    selector = decode.get("selector", pr_decode.get("selector", ""))
    selection_source = metadata.get("decode_refine_selection_source") or metadata.get("decode_refine_ranking_source", "")
    active = perf.get("active_tokens_per_decode_step")
    if active is None:
        active = int(requested_repair_tokens) + 1
    return {
        "benchmark_status": "ok",
        "error_message": "",
        "prefill_latency_s": float(perf.get("prefill_latency_s", 0.0) or 0.0),
        "decode_latency_s": float(perf.get("decode_latency_s", 0.0) or 0.0),
        "decode_tpot_ms": float(perf.get("decode_tpot_ms", 0.0) or 0.0),
        "decode_tps": float(perf.get("decode_tps", 0.0) or 0.0),
        "request_wall_s": float(perf.get("request_wall_s", 0.0) or 0.0),
        "decode_measured_tokens": int(perf.get("decode_measured_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", perf.get("completion_tokens", 0)) or 0),
        "active_tokens_per_decode_step": int(active or 0),
        "execution_mode": pr.get("execution_mode", ""),
        "decode_runtime": runtime,
        "decode_selector": selector,
        "decode_selection_source": selection_source,
    }


def unsupported_metrics(message: str, requested_repair_tokens: int) -> dict[str, Any]:
    return {
        "benchmark_status": "unsupported",
        "error_message": message,
        "prefill_latency_s": "",
        "decode_latency_s": "",
        "decode_tpot_ms": "",
        "decode_tps": "",
        "request_wall_s": "",
        "decode_measured_tokens": "",
        "completion_tokens": "",
        "active_tokens_per_decode_step": int(requested_repair_tokens) + 1,
        "execution_mode": "",
        "decode_runtime": "unsupported",
        "decode_selector": "",
        "decode_selection_source": "unsupported",
    }


def average_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if samples and all(sample.get("benchmark_status") != "ok" for sample in samples):
        return samples[-1]
    keys_float = [
        "prefill_latency_s",
        "decode_latency_s",
        "decode_tpot_ms",
        "decode_tps",
        "request_wall_s",
    ]
    out: dict[str, Any] = {key: mean([float(sample.get(key, 0.0) or 0.0) for sample in samples]) for key in keys_float}
    out["decode_measured_tokens"] = int(round(mean([float(sample.get("decode_measured_tokens", 0) or 0) for sample in samples])))
    out["completion_tokens"] = int(round(mean([float(sample.get("completion_tokens", 0) or 0) for sample in samples])))
    out["active_tokens_per_decode_step"] = int(round(mean([float(sample.get("active_tokens_per_decode_step", 0) or 0) for sample in samples])))
    out["execution_mode"] = samples[-1].get("execution_mode", "") if samples else ""
    out["decode_runtime"] = samples[-1].get("decode_runtime", "") if samples else ""
    out["decode_selector"] = samples[-1].get("decode_selector", "") if samples else ""
    out["decode_selection_source"] = samples[-1].get("decode_selection_source", "") if samples else ""
    out["benchmark_status"] = samples[-1].get("benchmark_status", "") if samples else ""
    out["error_message"] = samples[-1].get("error_message", "") if samples else ""
    return out


def ensure_header(path: Path, fields: list[str], append: bool) -> None:
    if append and path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != fields:
            raise RuntimeError(f"{path} header does not match current decode benchmark schema")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def append_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_chat_once(args: argparse.Namespace, pic_spec: dict[str, Any], repeat_dir: Path) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "{{pic_cache}}\n" + args.question}],
        "max_tokens": int(args.max_tokens),
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "pic_cache": pic_spec,
    }
    if args.reset_before_each:
        post_json(args.base_url, "/reset", {}, args.timeout)
    response = post_json(args.base_url, "/v1/chat/completions", payload, args.timeout)
    write_json(repeat_dir / "payload.json", payload)
    write_json(repeat_dir / "response.json", response)
    return response


def run_chat_metrics(args: argparse.Namespace, pic_spec: dict[str, Any], repeat_dir: Path,
                     repair_tokens: int) -> dict[str, Any]:
    try:
        response = run_chat_once(args, pic_spec, repeat_dir)
    except RuntimeError as exc:
        message = str(exc)
        if args.continue_on_unsupported and is_unsupported_error(message):
            write_json(repeat_dir / "error.json", {"status": "unsupported", "error": message})
            return unsupported_metrics(message, repair_tokens)
        raise
    return extract_metrics(response, repair_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Running pic_server base URL")
    parser.add_argument("--output-csv", default="benchmark_decode.csv")
    parser.add_argument("--output-dir", default=".cache/mnn-pic-benchmark/decode_repair")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--device",
        required=True,
        choices=DEVICE_CHOICES,
        help="Formal decode matrix runs once per device for jetson/orangepi. Rhino requires --allow-extra-device.",
    )
    parser.add_argument("--device-display", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--frequency-profile", default="max")
    parser.add_argument("--model", default="llama-pic")
    parser.add_argument("--model-name", default="Llama3.2 3B")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--mode", default="full-reuse")
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--repair-tokens", default=DEFAULT_REPAIR_TOKENS)
    parser.add_argument("--decode-selector", default="",
                        help="Single decode selector. Ignored when --decode-selectors is set.")
    parser.add_argument("--decode-selectors", default=DEFAULT_DECODE_SELECTORS,
                        help="Comma-separated selectors, e.g. top_hkvd,lagged_attention_hkvd")
    parser.add_argument("--attention-layer-idx", type=int, default=1)
    parser.add_argument("--attention-head-ids", default="")
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--score-layer-idx", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--suffix-from-cache-tokens", type=int, default=1)
    parser.add_argument("--question", default="Answer briefly using the cached text.")
    parser.add_argument("--doc-seed", default="MNN PIC decode repair benchmark document.")
    parser.add_argument("--token-ids-json", default=str(DEFAULT_TOKEN_IDS_JSON))
    parser.add_argument("--text-contexts", action="store_true",
                        help="Use generated text instead of exact token_ids for /v1/prefill/text")
    parser.add_argument("--force-text-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-before-each", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-exact-context", action="store_true")
    parser.add_argument("--require-decode-runtime", default="mnn_token_id_sparse_decode")
    parser.add_argument("--continue-on-unsupported", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-high-risk-contexts",
        action="store_true",
        help="Allow contexts above the default safety limit, for example 3072.",
    )
    parser.add_argument(
        "--allow-extra-device",
        action="store_true",
        help="Allow extra-profile devices such as rhino. Formal regression still requires separate jetson+orangepi runs.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    validate_device(str(args.device), allow_extra_device=bool(args.allow_extra_device))
    decode_selectors = parse_csv_strings(args.decode_selectors)
    if not decode_selectors and str(args.decode_selector).strip():
        decode_selectors = [str(args.decode_selector).strip()]
    if not decode_selectors:
        decode_selectors = ["top_hkvd"]
    for selector in decode_selectors:
        if selector not in {"top_hkvd", "lagged_attention_hkvd"}:
            raise SystemExit(f"unsupported --decode-selectors value: {selector}")
    if "lagged_attention_hkvd" in decode_selectors and int(args.attention_layer_idx) < 0:
        raise SystemExit("--decode-selector lagged_attention_hkvd requires --attention-layer-idx")

    contexts = parse_csv_ints(args.contexts)
    budgets = parse_csv_floats(args.budgets)
    repair_tokens = parse_csv_ints(args.repair_tokens)
    head_ids = parse_head_ids(args.attention_head_ids)
    base_doc_token_ids = [] if args.text_contexts else load_doc_token_ids(args.token_ids_json)
    output_csv = Path(args.output_csv)
    output_root = Path(args.output_dir)
    fields = [
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
    ensure_header(output_csv, fields, args.append)

    all_rows: list[dict[str, Any]] = []
    run_id = time.strftime("%Y%m%d_%H%M%S")
    write_json(
        output_root / run_id / "run_config.json",
        {
            "device": args.device,
            "device_display": args.device_display,
            "device_role": "formal" if args.device in FORMAL_REQUIRED_DEVICES else "extra-profile",
            "formal_required_devices": list(FORMAL_REQUIRED_DEVICES),
            "extra_profile_devices": list(EXTRA_PROFILE_DEVICES),
            "formal_matrix_device_covered": args.device in FORMAL_REQUIRED_DEVICES,
            "formal_matrix_complete": False,
            "model_name": args.model_name,
            "model_config": args.model_config,
            "backend": args.backend,
            "frequency_profile": args.frequency_profile,
            "contexts": contexts,
            "budgets": [fmt_budget(budget) for budget in budgets],
            "repair_tokens": repair_tokens,
            "decode_selectors": decode_selectors,
            "allow_high_risk_contexts": bool(args.allow_high_risk_contexts),
            "allow_extra_device": bool(args.allow_extra_device),
            "high_risk_context_policy": json_safe_policy(
                HIGH_RISK_CONTEXT_POLICIES.get(str(args.device), {})
            ),
        },
    )
    for target_context in contexts:
        high_risk_error = None if bool(args.allow_high_risk_contexts) else high_risk_context_error(
            str(args.device),
            str(args.frequency_profile),
            int(target_context),
        )
        if high_risk_error:
            raise SystemExit(f"context {target_context} blocked for device {args.device}: {high_risk_error}")
        doc_id = f"decode_bench_doc_ctx{target_context}_{run_id}"
        doc_token_ids = make_doc_token_ids(target_context, base_doc_token_ids)
        if doc_token_ids:
            prefill_payload = {
                "id": doc_id,
                "type": "text",
                "token_ids": doc_token_ids,
                "force": bool(args.force_text_cache),
            }
        else:
            content = make_doc_content(target_context, args.doc_seed)
            prefill_payload = {
                "id": doc_id,
                "type": "text",
                "content": content,
                "force": bool(args.force_text_cache),
            }
        prefill_response = post_json(args.base_url, "/v1/prefill/text", prefill_payload, args.timeout)
        write_json(output_root / run_id / f"context_{target_context}" / "prefill_text.response.json", prefill_response)
        actual_context = int(prefill_response.get("token_count", 0) or 0)
        if args.require_exact_context and actual_context != target_context:
            raise RuntimeError(f"context target {target_context} produced token_count={actual_context}")

        for budget in budgets:
            pic_id = f"decode_bench_pic_ctx{target_context}_b{fmt_budget(budget)}_{run_id}"
            pic_payload = {
                "id": pic_id,
                "text_cache_refs": [{"id": doc_id}],
                "selection_algorithm": args.mode,
                "pic_recompute_ratio": float(budget),
                "pic_recompute_score_layer_idx": int(args.score_layer_idx),
            }
            pic_response = post_json(args.base_url, "/v1/kv/pic_caches", pic_payload, args.timeout)
            token_ids = pic_response.get("token_ids") or []
            if not isinstance(token_ids, list) or not token_ids:
                raise RuntimeError(f"/v1/kv/pic_caches did not return token_ids for {pic_id}")
            write_json(
                output_root / run_id / f"context_{target_context}" / f"budget_{fmt_budget(budget)}" / "pic_cache.response.json",
                pic_response,
            )

            for selector in decode_selectors:
                selector_dir = selector.replace("/", "_")
                selector_repair_tokens = repair_tokens
                if selector != decode_selectors[0]:
                    selector_repair_tokens = [repair for repair in repair_tokens if repair > 0]
                # Warm all requested repair shapes before formal timing, then persist
                # the MNN runtime cache for OpenCL backends.
                for _ in range(max(0, int(args.warm_repeats))):
                    for repair in selector_repair_tokens:
                        spec = build_pic_spec(
                            pic_id=pic_id,
                            doc_id=doc_id,
                            mode=args.mode,
                            budget=budget,
                            score_layer_idx=args.score_layer_idx,
                            token_ids=token_ids,
                            suffix_tokens=args.suffix_from_cache_tokens,
                            repair_tokens=repair,
                            selector=selector,
                            attention_layer_idx=args.attention_layer_idx,
                            attention_head_ids=head_ids,
                            top_m=args.top_m,
                        )
                        metrics = run_chat_metrics(
                            args, spec,
                            output_root / run_id / "warm" / f"ctx{target_context}_b{fmt_budget(budget)}"
                            / selector_dir / f"r{repair}",
                            repair,
                        )
                        if metrics.get("benchmark_status") == "unsupported":
                            print(
                                f"warm unsupported: device={args.device} context={actual_context or target_context} "
                                f"budget={fmt_budget(budget)} selector={selector} repair={repair}: "
                                f"{metrics.get('error_message')}",
                                flush=True,
                            )
                if args.update_cache:
                    update_response = post_json(args.base_url, "/v1/tune/update_cache", {}, args.timeout)
                    write_json(
                        output_root / run_id / f"context_{target_context}" / f"budget_{fmt_budget(budget)}"
                        / selector_dir / "update_cache.response.json",
                        update_response,
                    )

                by_repair: dict[int, dict[str, Any]] = {}
                for repair in selector_repair_tokens:
                    samples: list[dict[str, Any]] = []
                    for repeat in range(1, max(1, int(args.repeats)) + 1):
                        spec = build_pic_spec(
                            pic_id=pic_id,
                            doc_id=doc_id,
                            mode=args.mode,
                            budget=budget,
                            score_layer_idx=args.score_layer_idx,
                            token_ids=token_ids,
                            suffix_tokens=args.suffix_from_cache_tokens,
                            repair_tokens=repair,
                            selector=selector,
                            attention_layer_idx=args.attention_layer_idx,
                            attention_head_ids=head_ids,
                            top_m=args.top_m,
                        )
                        repeat_dir = (
                            output_root
                            / run_id
                            / f"context_{target_context}"
                            / f"budget_{fmt_budget(budget)}"
                            / selector_dir
                            / f"repair_{repair}"
                            / f"repeat_{repeat}"
                        )
                        metrics = run_chat_metrics(args, spec, repeat_dir, repair)
                        if metrics.get("benchmark_status") == "unsupported":
                            samples.append(metrics)
                            break
                        if repair > 0 and args.require_decode_runtime:
                            if metrics.get("decode_runtime") != args.require_decode_runtime:
                                raise RuntimeError(
                                    "decode repair did not enter required runtime: "
                                    f"selector={selector} repair={repair} runtime={metrics.get('decode_runtime')!r}"
                                )
                        samples.append(metrics)
                    by_repair[repair] = average_metrics(samples)

                rows: list[dict[str, Any]] = []
                for repair in selector_repair_tokens:
                    metrics = by_repair[repair]
                    row = {
                        "device": args.device,
                        "device_display": args.device_display,
                        "model": args.model_name,
                        "backend": args.backend,
                        "frequency_profile": args.frequency_profile,
                        "context_tokens": actual_context or target_context,
                        "mode": args.mode,
                        "budget": fmt_budget(budget),
                        "decode_selector": selector if repair > 0 else "none",
                        "repair_tokens": repair,
                        "generated_tokens": int(args.max_tokens),
                        "decode_latency_s": metrics.get("decode_latency_s", ""),
                        "decode_tpot_ms": metrics.get("decode_tpot_ms", ""),
                        "decode_tps": metrics.get("decode_tps", ""),
                        "benchmark_status": metrics.get("benchmark_status", ""),
                    }
                    rows.append(row)
                append_rows(output_csv, fields, rows)
                all_rows.extend(rows)
                print(
                    f"wrote {len(rows)} rows for device={args.device} context={actual_context or target_context} "
                    f"budget={fmt_budget(budget)} selector={selector}",
                    flush=True,
                )

    print(f"benchmark rows written: {len(all_rows)} -> {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
