#!/usr/bin/env python3
"""Collect MNN PIC decode-repair TPOT/TPS into benchmark_decode.csv.

This client assumes a pic_server is already running for exactly one target
device. Run it once per device, appending to the same CSV if desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CONTEXTS = "512,1024,1536,2048,2560,3072"
DEFAULT_BUDGETS = "0.05,0.10,0.20"
DEFAULT_REPAIR_TOKENS = "0,1,2,3,4,5,6,7"


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_head_ids(value: str) -> list[int]:
    if not str(value).strip():
        return []
    return parse_csv_ints(value)


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
    parser.add_argument("--device", required=True)
    parser.add_argument("--device-display", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--frequency-profile", default="max")
    parser.add_argument("--model", default="llama-pic")
    parser.add_argument("--model-name", default="Llama3.2 3B")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--mode", default="epic")
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--repair-tokens", default=DEFAULT_REPAIR_TOKENS)
    parser.add_argument("--decode-selector", default="lagged_attention_hkvd")
    parser.add_argument("--attention-layer-idx", type=int, default=-1)
    parser.add_argument("--attention-head-ids", default="")
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--score-layer-idx", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--suffix-from-cache-tokens", type=int, default=1)
    parser.add_argument("--question", default="Answer briefly using the cached text.")
    parser.add_argument("--doc-seed", default="MNN PIC decode repair benchmark document.")
    parser.add_argument("--force-text-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-before-each", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-exact-context", action="store_true")
    parser.add_argument("--require-decode-runtime", default="mnn_token_id_sparse_decode")
    parser.add_argument("--continue-on-unsupported", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if args.decode_selector == "lagged_attention_hkvd" and int(args.attention_layer_idx) < 0:
        raise SystemExit("--decode-selector lagged_attention_hkvd requires --attention-layer-idx")

    contexts = parse_csv_ints(args.contexts)
    budgets = parse_csv_floats(args.budgets)
    repair_tokens = parse_csv_ints(args.repair_tokens)
    head_ids = parse_head_ids(args.attention_head_ids)
    output_csv = Path(args.output_csv)
    output_root = Path(args.output_dir)
    fields = [
        "device",
        "device_display",
        "model",
        "model_config",
        "backend",
        "frequency_profile",
        "target_context_tokens",
        "context_tokens",
        "mode",
        "budget",
        "decode_selector",
        "attention_layer_idx",
        "attention_head_ids",
        "repair_tokens",
        "active_tokens_per_decode_step",
        "generated_tokens",
        "decode_measured_tokens",
        "repeat_count",
        "prefill_latency_s",
        "decode_latency_s",
        "decode_tpot_ms",
        "decode_tps",
        "request_wall_s",
        "baseline_decode_tpot_ms",
        "overhead_vs_normal_decode",
        "decode_tps_vs_normal_decode",
        "execution_mode",
        "decode_runtime",
        "decode_selection_source",
        "benchmark_status",
        "error_message",
    ]
    ensure_header(output_csv, fields, args.append)

    all_rows: list[dict[str, Any]] = []
    run_id = time.strftime("%Y%m%d_%H%M%S")
    for target_context in contexts:
        doc_id = f"decode_bench_doc_ctx{target_context}_{run_id}"
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

            # Warm all requested repair shapes before formal timing, then persist
            # the MNN runtime cache for OpenCL backends.
            for _ in range(max(0, int(args.warm_repeats))):
                for repair in repair_tokens:
                    spec = build_pic_spec(
                        pic_id=pic_id,
                        doc_id=doc_id,
                        mode=args.mode,
                        budget=budget,
                        score_layer_idx=args.score_layer_idx,
                        token_ids=token_ids,
                        suffix_tokens=args.suffix_from_cache_tokens,
                        repair_tokens=repair,
                        selector=args.decode_selector,
                        attention_layer_idx=args.attention_layer_idx,
                        attention_head_ids=head_ids,
                        top_m=args.top_m,
                    )
                    metrics = run_chat_metrics(
                        args, spec,
                        output_root / run_id / "warm" / f"ctx{target_context}_b{fmt_budget(budget)}_r{repair}",
                        repair,
                    )
                    if metrics.get("benchmark_status") == "unsupported":
                        print(
                            f"warm unsupported: device={args.device} context={actual_context or target_context} "
                            f"budget={fmt_budget(budget)} repair={repair}: {metrics.get('error_message')}",
                            flush=True,
                        )
            if args.update_cache:
                update_response = post_json(args.base_url, "/v1/tune/update_cache", {}, args.timeout)
                write_json(
                    output_root / run_id / f"context_{target_context}" / f"budget_{fmt_budget(budget)}" / "update_cache.response.json",
                    update_response,
                )

            by_repair: dict[int, dict[str, Any]] = {}
            for repair in repair_tokens:
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
                        selector=args.decode_selector,
                        attention_layer_idx=args.attention_layer_idx,
                        attention_head_ids=head_ids,
                        top_m=args.top_m,
                    )
                    repeat_dir = (
                        output_root
                        / run_id
                        / f"context_{target_context}"
                        / f"budget_{fmt_budget(budget)}"
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
                                f"repair={repair} runtime={metrics.get('decode_runtime')!r}"
                            )
                    samples.append(metrics)
                by_repair[repair] = average_metrics(samples)

            baseline = by_repair.get(0, {})
            baseline_tpot = float(baseline.get("decode_tpot_ms", 0.0) or 0.0)
            baseline_tps = float(baseline.get("decode_tps", 0.0) or 0.0)
            rows: list[dict[str, Any]] = []
            for repair in repair_tokens:
                metrics = by_repair[repair]
                tpot = float(metrics.get("decode_tpot_ms", 0.0) or 0.0)
                tps = float(metrics.get("decode_tps", 0.0) or 0.0)
                overhead = tpot / baseline_tpot if baseline_tpot > 0.0 and tpot > 0.0 else math.nan
                tps_ratio = tps / baseline_tps if baseline_tps > 0.0 and tps > 0.0 else math.nan
                row = {
                    "device": args.device,
                    "device_display": args.device_display,
                    "model": args.model_name,
                    "model_config": args.model_config,
                    "backend": args.backend,
                    "frequency_profile": args.frequency_profile,
                    "target_context_tokens": target_context,
                    "context_tokens": actual_context or target_context,
                    "mode": args.mode,
                    "budget": fmt_budget(budget),
                    "decode_selector": args.decode_selector if repair > 0 else "none",
                    "attention_layer_idx": args.attention_layer_idx if repair > 0 else "",
                    "attention_head_ids": ",".join(str(item) for item in head_ids) if repair > 0 else "",
                    "repair_tokens": repair,
                    "active_tokens_per_decode_step": metrics.get("active_tokens_per_decode_step", repair + 1),
                    "generated_tokens": int(args.max_tokens),
                    "decode_measured_tokens": metrics.get("decode_measured_tokens", ""),
                    "repeat_count": int(args.repeats),
                    "prefill_latency_s": metrics.get("prefill_latency_s", ""),
                    "decode_latency_s": metrics.get("decode_latency_s", ""),
                    "decode_tpot_ms": metrics.get("decode_tpot_ms", ""),
                    "decode_tps": metrics.get("decode_tps", ""),
                    "request_wall_s": metrics.get("request_wall_s", ""),
                    "baseline_decode_tpot_ms": baseline_tpot,
                    "overhead_vs_normal_decode": overhead,
                    "decode_tps_vs_normal_decode": tps_ratio,
                    "execution_mode": metrics.get("execution_mode", ""),
                    "decode_runtime": metrics.get("decode_runtime", ""),
                    "decode_selection_source": metrics.get("decode_selection_source", ""),
                    "benchmark_status": metrics.get("benchmark_status", ""),
                    "error_message": metrics.get("error_message", ""),
                }
                rows.append(row)
            append_rows(output_csv, fields, rows)
            all_rows.extend(rows)
            print(
                f"wrote {len(rows)} rows for device={args.device} context={actual_context or target_context} "
                f"budget={fmt_budget(budget)}",
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
