#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict


LINE_RE = re.compile(
    r"\[bench_ops/cuda/perf/WeightOnlyConv\]\s+"
    r"(?P<case>\S+)\s+rows=(?P<rows>\d+)\s+ic=(?P<ic>\d+)\s+"
    r"oc=(?P<oc>\d+)\s+qblock=(?P<qblock>\d+)\s+avg=(?P<avg>[0-9.]+)\s+ms"
)


def parse_logs(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = LINE_RE.search(line)
                if not m:
                    continue
                item = m.groupdict()
                item["rows"] = int(item["rows"])
                item["ic"] = int(item["ic"])
                item["oc"] = int(item["oc"])
                item["qblock"] = int(item["qblock"])
                item["avg"] = float(item["avg"])
                item["source"] = path
                rows.append(item)
    return rows


def best_by_case_row(items):
    best = {}
    for item in items:
        key = (item["case"], item["rows"])
        if key not in best or item["avg"] < best[key]["avg"]:
            best[key] = item
    return best


def print_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


def fmt_ms(value):
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize CUDA WeightOnlyConv tune logs for PIC decode repair."
    )
    parser.add_argument("logs", nargs="+", help="WeightOnlyConv log files")
    parser.add_argument("--layers", type=int, default=16, help="decoder layer count for rough estimates")
    parser.add_argument("--gate-case", default="hidden_to_inter")
    parser.add_argument("--down-case", default="inter_to_hidden")
    parser.add_argument("--concat-case", default="hidden_to_gateup_concat")
    parser.add_argument(
        "--target-gap-ms",
        type=float,
        default=5.0,
        help="End-to-end TPOT gap this candidate would need to cover. Default: 5ms.",
    )
    parser.add_argument(
        "--tpot-entry-ms",
        type=float,
        default=1.0,
        help="Minimum estimated total saving to consider a candidate worth strict TPOT. Default: 1ms.",
    )
    args = parser.parse_args()

    items = parse_logs(args.logs)
    if not items:
        raise SystemExit("no WeightOnlyConv rows found")

    best = best_by_case_row(items)
    sorted_items = sorted(best.values(), key=lambda x: (x["rows"], x["case"]))

    print("## WeightOnlyConv Best Rows")
    print_table(
        ["case", "rows", "ic", "oc", "qblock", "avg_ms"],
        [
            [item["case"], item["rows"], item["ic"], item["oc"], item["qblock"], fmt_ms(item["avg"])]
            for item in sorted_items
        ],
    )

    by_row = defaultdict(dict)
    for (case, row), item in best.items():
        by_row[row][case] = item["avg"]

    mlp_rows = []
    gateup_rows = []
    for row in sorted(by_row):
        cases = by_row[row]
        gate = cases.get(args.gate_case)
        down = cases.get(args.down_case)
        concat = cases.get(args.concat_case)
        if gate is not None and down is not None:
            mlp = 2.0 * gate + down
            mlp_rows.append([
                row,
                fmt_ms(gate),
                fmt_ms(down),
                fmt_ms(mlp),
                fmt_ms(mlp * args.layers),
            ])
        if gate is not None and concat is not None:
            independent = 2.0 * gate
            delta = independent - concat
            gateup_rows.append([
                row,
                fmt_ms(independent),
                fmt_ms(concat),
                fmt_ms(delta),
                fmt_ms(delta * args.layers),
                "positive" if delta > 0 else "negative",
            ])

    if mlp_rows:
        print()
        print("## Rough MLP Projection Cost")
        print_table(
            ["rows", "gate_or_up_ms", "down_ms", "gate+up+down_ms_per_layer", f"x{args.layers}_layers_ms"],
            mlp_rows,
        )

    if gateup_rows:
        print()
        print("## Gate/Up Concat Delta")
        print_table(
            [
                "rows",
                "2x_gate_up_ms",
                "concat_ms",
                "saved_ms_per_layer",
                f"saved_ms_x{args.layers}_layers",
                "direction",
            ],
            gateup_rows,
        )
        best_saved = max(float(row[4]) for row in gateup_rows)
        required_per_layer = args.target_gap_ms / max(1, args.layers)
        print()
        print("## Verdict")
        print(f"- target_gap_ms: {args.target_gap_ms:.4f}")
        print(f"- required_saving_per_layer_ms: {required_per_layer:.4f}")
        print(f"- best_estimated_total_saving_ms: {best_saved:.4f}")
        print(f"- strict_tpot_entry_threshold_ms: {args.tpot_entry_ms:.4f}")
        if best_saved >= args.target_gap_ms:
            print("- decision: strong_candidate_for_strict_tpot")
        elif best_saved >= args.tpot_entry_ms:
            print("- decision: partial_candidate_needs_end_to_end_check")
        else:
            print("- decision: cleanup_only_do_not_use_as_main_tpot_candidate")


if __name__ == "__main__":
    main()
