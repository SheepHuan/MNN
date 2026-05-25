#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


PERF_RE = re.compile(
    r"^\[bench_ops/cuda/perf/(?P<op>[^\]]+)\]\s+"
    r"(?P<case>\S+)\s+B=(?P<batch>\d+)\s+qH=(?P<qh>\d+)\s+kvH=(?P<kvh>\d+)\s+"
    r"D=(?P<dim>\d+)\s+past=(?P<past>\d+)\s+add=(?P<add>\d+)\s+"
    r"avg=(?P<avg>[0-9.eE+-]+)\s+ms(?:\s+(?P<us>[0-9.eE+-]+)\s+us/token)?"
)
CASE_RE = re.compile(r"^(?P<model>llama3\.2-[^_]+)_(?P<stage>prefill|decode)_ctx(?P<ctx>\d+)$")
ACC_RE = re.compile(
    r"^\[bench_ops/cuda/accuracy\]\s+(?P<name>\S+)\s+"
    r"max_abs=(?P<max_abs>[0-9.eE+-]+)\s+max_rel=(?P<max_rel>[0-9.eE+-]+)\s+"
    r"bad=(?P<bad>\d+/\d+)"
)
COMPARE_ACC_RE = re.compile(
    r"^PagedAttention_vs_Attention/(?P<model>llama3\.2-[^/]+)/ctx(?P<ctx>\d+)/(?P<stage>prefill|decode)$"
)
PAGED_CPU_ACC_RE = re.compile(
    r"^PagedAttention/(?P<model>llama3\.2-[^/]+)/ctx(?P<ctx>\d+)/(?P<stage>prefill|decode)$"
)

SHAPES = {
    ("llama3.2-1B", "512"): ("32", "8", "64"),
    ("llama3.2-1B", "1024"): ("32", "8", "64"),
    ("llama3.2-1B", "2048"): ("32", "8", "64"),
    ("llama3.2-3B", "512"): ("24", "8", "128"),
    ("llama3.2-3B", "1024"): ("24", "8", "128"),
    ("llama3.2-3B", "2048"): ("24", "8", "128"),
    ("llama3.2-8B", "512"): ("32", "8", "128"),
    ("llama3.2-8B", "1024"): ("32", "8", "128"),
    ("llama3.2-8B", "2048"): ("32", "8", "128"),
}

STAGE_ORDER = {"prefill": 0, "decode": 1}
MODEL_ORDER = {"llama3.2-1B": 0, "llama3.2-3B": 1, "llama3.2-8B": 2}
OP_ORDER = {"Attention": 0, "PagedAttention": 1, "LinearAttention": 2}


def read_logs(log_dir):
    for path in sorted(log_dir.glob("*.log")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield line.rstrip("\n")


def parse_perf(lines):
    rows = {}
    for line in lines:
        match = PERF_RE.match(line)
        if not match:
            continue
        case = CASE_RE.match(match.group("case"))
        if not case:
            continue
        row = {
            "stage": case.group("stage"),
            "model": case.group("model"),
            "ctx": case.group("ctx"),
            "qh": match.group("qh"),
            "kvh": match.group("kvh"),
            "dim": match.group("dim"),
            "op": match.group("op"),
            "avg": float(match.group("avg")),
            "us": float(match.group("us")) if match.group("us") else None,
        }
        rows[(row["stage"], row["model"], row["ctx"], row["op"])] = row
    return list(rows.values())


def parse_accuracy(lines):
    rows = {}
    for line in lines:
        match = ACC_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        compare = COMPARE_ACC_RE.match(name)
        paged_cpu = PAGED_CPU_ACC_RE.match(name)
        if compare:
            model = compare.group("model")
            ctx = compare.group("ctx")
            qh, kvh, dim = SHAPES[(model, ctx)]
            row = {
                "op": "PagedAttention",
                "ref": "Attention",
                "stage": compare.group("stage"),
                "model": model,
                "ctx": ctx,
                "qh": qh,
                "kvh": kvh,
                "dim": dim,
                "max_abs": match.group("max_abs"),
                "max_rel": match.group("max_rel"),
                "bad": match.group("bad"),
            }
        elif paged_cpu:
            model = paged_cpu.group("model")
            ctx = paged_cpu.group("ctx")
            qh, kvh, dim = SHAPES[(model, ctx)]
            row = {
                "op": "PagedAttention",
                "ref": "CPU",
                "stage": paged_cpu.group("stage"),
                "model": model,
                "ctx": ctx,
                "qh": qh,
                "kvh": kvh,
                "dim": dim,
                "max_abs": match.group("max_abs"),
                "max_rel": match.group("max_rel"),
                "bad": match.group("bad"),
            }
        elif name in ("LinearAttention/prefill", "LinearAttention/decode"):
            row = {
                "op": "LinearAttention",
                "ref": "CPU",
                "stage": name.split("/")[-1],
                "model": "-",
                "ctx": "-",
                "qh": "2",
                "kvh": "2",
                "dim": "64",
                "max_abs": match.group("max_abs"),
                "max_rel": match.group("max_rel"),
                "bad": match.group("bad"),
            }
        elif name in ("PagedAttention/prefill", "PagedAttention/decode"):
            row = {
                "op": "PagedAttention",
                "ref": "CPU",
                "stage": name.split("/")[-1],
                "model": "-",
                "ctx": "-",
                "qh": "4",
                "kvh": "2",
                "dim": "16",
                "max_abs": match.group("max_abs"),
                "max_rel": match.group("max_rel"),
                "bad": match.group("bad"),
            }
        else:
            continue
        rows[(row["op"], row["ref"], row["stage"], row["model"], row["ctx"])] = row
    return list(rows.values())


def markdown(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")


def print_perf(rows):
    attention = {
        (row["stage"], row["model"], row["ctx"]): row["avg"]
        for row in rows
        if row["op"] == "Attention"
    }
    out = []
    for row in sorted(
        rows,
        key=lambda r: (
            STAGE_ORDER.get(r["stage"], 9),
            MODEL_ORDER.get(r["model"], 9),
            int(r["ctx"]),
            OP_ORDER.get(r["op"], 9),
        ),
    ):
        base = attention.get((row["stage"], row["model"], row["ctx"]))
        ratio = "-"
        if base and row["op"] == "Attention":
            ratio = "1.00x"
        elif base:
            ratio = f"{base / row['avg']:.2f}x"
        out.append(
            {
                "stage": row["stage"],
                "model": row["model"],
                "ctx": row["ctx"],
                "qH": row["qh"],
                "kvH": row["kvh"],
                "D": row["dim"],
                "op": row["op"],
                "latency_ms": f"{row['avg']:.4f}",
                "Attention/PagedAttention": ratio,
            }
        )
    print("## Performance")
    markdown(["stage", "model", "ctx", "qH", "kvH", "D", "op", "latency_ms", "Attention/PagedAttention"], out)


def print_accuracy(rows):
    out = []
    for row in sorted(
        rows,
        key=lambda r: (
            0 if r["ref"] == "Attention" else 1,
            OP_ORDER.get(r["op"], 9),
            STAGE_ORDER.get(r["stage"], 9),
            MODEL_ORDER.get(r["model"], 9),
            0 if r["ctx"] == "-" else int(r["ctx"]),
        ),
    ):
        out.append(
            {
                "op": row["op"],
                "ref_op": row["ref"],
                "stage": row["stage"],
                "model": row["model"],
                "ctx": row["ctx"],
                "qH": row["qh"],
                "kvH": row["kvh"],
                "D": row["dim"],
                "max_abs": row["max_abs"],
                "max_rel": row["max_rel"],
                "bad": row["bad"],
            }
        )
    print("\n## Accuracy")
    markdown(["op", "ref_op", "stage", "model", "ctx", "qH", "kvH", "D", "max_abs", "max_rel", "bad"], out)


def main():
    parser = argparse.ArgumentParser(description="Summarize MNN CUDA attention direct-op bench logs.")
    parser.add_argument("log_dir", nargs="?", default=".cache/bench_ops", type=Path)
    args = parser.parse_args()

    lines = list(read_logs(args.log_dir))
    perf_rows = parse_perf(lines)
    acc_rows = parse_accuracy(lines)
    if perf_rows:
        print_perf(perf_rows)
    if acc_rows:
        print_accuracy(acc_rows)


if __name__ == "__main__":
    main()
