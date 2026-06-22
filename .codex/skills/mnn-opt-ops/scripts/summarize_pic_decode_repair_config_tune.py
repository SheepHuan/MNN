#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


CHAT_RE = re.compile(r"chat_tpd(?P<tpd>\d+)_r(?P<repeat>\d+)\.response\.json$")


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None, "missing"
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None, "empty"
    try:
        return json.loads(text), "ok"
    except json.JSONDecodeError as exc:
        return None, f"json_error:{exc}"


def get_path(obj, path, default=None):
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        elif isinstance(cur, list):
            try:
                cur = cur[int(key)]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return cur


def unwrap_chat(data):
    if not isinstance(data, dict):
        return {}, {}
    if isinstance(data.get("data"), list) and data["data"]:
        item = data["data"][0] if isinstance(data["data"][0], dict) else {}
        return data, item
    return data, data


def chat_info(path):
    match = CHAT_RE.search(Path(path).name)
    tpd = int(match.group("tpd")) if match else -1
    repeat = int(match.group("repeat")) if match else -1
    data, state = load_json(path)
    if state != "ok":
        verdict = "dry_run" if state == "empty" else "fail"
        return {
            "tpd": tpd,
            "repeat": repeat,
            "state": state,
            "execution_mode": "",
            "decode_runtime": "",
            "decode_enabled": "",
            "refined_token_count": "",
            "completion_tokens": "",
            "content_preview": "",
            "verdict": verdict,
            "reason": state,
        }
    if isinstance(data, dict) and "error" in data:
        return {
            "tpd": tpd,
            "repeat": repeat,
            "state": state,
            "execution_mode": "",
            "decode_runtime": "",
            "decode_enabled": "",
            "refined_token_count": "",
            "completion_tokens": "",
            "content_preview": "",
            "verdict": "fail",
            "reason": f"error:{data.get('error')}",
        }

    root, item = unwrap_chat(data)
    pic = item.get("pic_cache") or root.get("pic_cache") or {}
    pr = pic.get("precision_recovery") if isinstance(pic, dict) else {}
    pr = pr if isinstance(pr, dict) else {}
    decode = pr.get("decode_refine") if isinstance(pr.get("decode_refine"), dict) else {}
    pic_decode = pic.get("decode_refine") if isinstance(pic.get("decode_refine"), dict) else {}
    metadata = pr.get("metadata") if isinstance(pr.get("metadata"), dict) else {}

    execution_mode = pr.get("execution_mode", "")
    decode_runtime = (
        metadata.get("decode_refine_runtime")
        or decode.get("runtime")
        or pic_decode.get("runtime")
        or ""
    )
    decode_enabled = decode.get("enabled", pic_decode.get("enabled", ""))
    refined_token_count = decode.get(
        "refined_token_count", pic_decode.get("refined_token_count", "")
    )
    completion_tokens = (
        get_path(item, ["usage", "completion_tokens"], None)
        if item is not root
        else get_path(root, ["usage", "completion_tokens"], None)
    )
    if completion_tokens is None:
        completion_tokens = get_path(root, ["usage", "completion_tokens"], "")
    content = (
        get_path(item, ["choices", "0", "message", "content"], "")
        if item
        else get_path(root, ["choices", "0", "message", "content"], "")
    )
    content = str(content).replace("\n", "\\n")[:80]

    verdict = "ok"
    reason = ""
    if tpd > 0 and decode_runtime != "mnn_token_id_sparse_decode":
        verdict = "fail"
        reason = f"decode_runtime={decode_runtime or 'missing'}"
    elif tpd == 0 and not execution_mode:
        verdict = "warn"
        reason = "missing_execution_mode"

    return {
        "tpd": tpd,
        "repeat": repeat,
        "state": state,
        "execution_mode": execution_mode,
        "decode_runtime": decode_runtime,
        "decode_enabled": decode_enabled,
        "refined_token_count": refined_token_count,
        "completion_tokens": completion_tokens,
        "content_preview": content,
        "verdict": verdict,
        "reason": reason,
    }


def update_info(path):
    data, state = load_json(path)
    if state == "empty":
        return "dry_run", state
    if state != "ok":
        return "fail", state
    if data.get("status") == "ok" and data.get("scope") == "mnn_runtime_cache":
        return "ok", ""
    return "fail", f"status={data.get('status')} scope={data.get('scope')}"


def read_summary(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_manifest(path):
    values = {}
    path = Path(path)
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def effective_cache_policy(requested, candidate, output_root):
    if requested != "auto":
        return requested
    manifest = read_manifest(output_root / candidate.get("candidate", "") / "manifest.env")
    text = " ".join(
        [
            manifest.get("artifact", ""),
            manifest.get("config", ""),
            candidate.get("cache_file", ""),
            candidate.get("work_dir", ""),
        ]
    ).lower()
    if "opencl" in text or "orangepi" in text or "adreno" in text:
        return "required"
    return "optional"


def write_tsv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_markdown(rows, fields):
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        print("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize PIC decode-repair config tune warm/update_cache runs."
    )
    parser.add_argument("output_root", help="run_pic_decode_repair_config_tune.sh output root")
    parser.add_argument("--summary", default=None, help="summary.tsv path override")
    parser.add_argument("--output-tsv", default=None, help="write detailed TSV")
    parser.add_argument("--fail-on-bad", action="store_true", help="exit non-zero on fail verdicts")
    parser.add_argument(
        "--cache-file-policy",
        choices=("auto", "required", "optional"),
        default="auto",
        help=(
            "How to treat a missing tmp/mnn_cachefile.bin. auto requires it for "
            "OpenCL/OrangePi-like candidates and only warns for CUDA-like candidates."
        ),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary_path = Path(args.summary) if args.summary else output_root / "summary.tsv"
    candidates = read_summary(summary_path)
    rows = []
    failed = False

    for candidate in candidates:
        warm_dir = Path(candidate.get("warm_dir", ""))
        cache_file = Path(candidate.get("cache_file", ""))
        cache_field = candidate.get("cache_exists", "")
        cache_policy = effective_cache_policy(args.cache_file_policy, candidate, output_root)
        cache_status = "dry_run" if cache_field == "dry_run" else ("ok" if cache_file.exists() or cache_field == "1" else "missing")
        cache_verdict = cache_status
        if cache_status == "missing" and cache_policy == "optional":
            cache_verdict = "missing_optional"
        update_status, update_reason = update_info(warm_dir / "update_cache.response.json")
        chat_files = sorted(warm_dir.glob("chat_tpd*_r*.response.json"))
        if not chat_files:
            rows.append({
                "candidate": candidate.get("candidate", ""),
                "candidate_status": "fail",
                "cache_policy": cache_policy,
                "cache_status": cache_verdict,
                "update_status": update_status,
                "tpd": "",
                "repeat": "",
                "execution_mode": "",
                "decode_runtime": "",
                "decode_enabled": "",
                "refined_token_count": "",
                "completion_tokens": "",
                "chat_verdict": "fail",
                "reason": "missing_chat_responses",
                "response": "",
            })
            failed = True
            continue
        for chat_file in chat_files:
            info = chat_info(chat_file)
            reasons = [r for r in [update_reason, info["reason"]] if r]
            if cache_verdict == "missing_optional":
                reasons.append("cache_file_missing_optional")
            candidate_status = "ok"
            if update_status == "fail" or cache_verdict == "missing" or info["verdict"] == "fail":
                candidate_status = "fail"
                failed = True
            elif update_status == "dry_run" or cache_verdict == "dry_run" or info["verdict"] == "dry_run":
                candidate_status = "dry_run"
            elif cache_verdict == "missing_optional" or info["verdict"] == "warn":
                candidate_status = "warn"
            rows.append({
                "candidate": candidate.get("candidate", ""),
                "candidate_status": candidate_status,
                "cache_policy": cache_policy,
                "cache_status": cache_verdict,
                "update_status": update_status,
                "tpd": info["tpd"],
                "repeat": info["repeat"],
                "execution_mode": info["execution_mode"],
                "decode_runtime": info["decode_runtime"],
                "decode_enabled": info["decode_enabled"],
                "refined_token_count": info["refined_token_count"],
                "completion_tokens": info["completion_tokens"],
                "chat_verdict": info["verdict"],
                "reason": ";".join(reasons),
                "response": str(chat_file),
            })

    fields = [
        "candidate",
        "candidate_status",
        "cache_policy",
        "cache_status",
        "update_status",
        "tpd",
        "repeat",
        "execution_mode",
        "decode_runtime",
        "decode_enabled",
        "refined_token_count",
        "completion_tokens",
        "chat_verdict",
        "reason",
        "response",
    ]
    if args.output_tsv:
        write_tsv(args.output_tsv, rows, fields)
    print_markdown(rows, fields[:-1])
    if failed and args.fail_on_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
