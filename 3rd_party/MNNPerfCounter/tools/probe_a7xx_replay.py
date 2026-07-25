#!/usr/bin/env python3
"""Probe Mesa A7xx events through replay_benchmark on a remote device.

The script batches events by Mesa's physical slot count. A batch is sampled
around one isolated replayed operator; the JSON delta is copied back and
flattened into a CSV for event-by-event inspection.
"""

import argparse
import csv
import json
import shlex
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


KGSL_GROUPS = {
    "CP": 0, "RBBM": 1, "PC": 2, "VFD": 3, "HLSQ": 4,
    "VPC": 5, "TSE": 6, "RAS": 7, "UCHE": 8, "TP": 9,
    "SP": 10, "RB": 11, "VSC": 12, "CCU": 13, "LRZ": 14,
    "CMP": 15, "UFC": 16,
}


def events(xml_path, json_path):
    groups = {item["name"]: item["num"] for item in json.loads(Path(json_path).read_text())["groups"]}
    root = ET.parse(xml_path).getroot()
    result = []
    for enum in root.iter():
        if enum.tag.rsplit("}", 1)[-1] != "enum":
            continue
        name = enum.attrib.get("name", "")
        prefix = "a7xx_"
        suffix = "_perfcounter_select"
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        group = name[len(prefix):-len(suffix)].upper()
        if group not in KGSL_GROUPS:
            continue
        for value in enum:
            if value.tag.rsplit("}", 1)[-1] != "value":
                continue
            event = value.attrib.get("name", "")
            if event.startswith("A7XX_PERF_"):
                result.append((group, int(value.attrib["value"], 0), event[10:].lower(), groups[group]))
    return sorted(set(result), key=lambda item: (KGSL_GROUPS[item[0]], item[1], item[2]))


def run_remote(device, remote_command):
    return subprocess.run(["ssh", device, remote_command], text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="SSH target, e.g. root@192.168.101.227")
    parser.add_argument("--remote-root", default="/mnt/nvme/workspace/replay-benchmark")
    parser.add_argument("--model", required=True, help="Remote model path relative to remote-root")
    parser.add_argument("--record", required=True, help="Remote record directory relative to remote-root")
    parser.add_argument("--op-id", required=True, type=int)
    parser.add_argument("--binary", default="bin/replay_benchmark.out")
    parser.add_argument("--xml", default="upstream/mesa/freedreno/registers/adreno/a7xx_perfcntrs.xml")
    parser.add_argument("--json", default="upstream/mesa/freedreno/registers/adreno/a7xx_perfcntrs.json")
    parser.add_argument("--csv", default="a7xx_event_probe.csv")
    args = parser.parse_args()

    all_events = events(args.xml, args.json)
    rows = []
    with tempfile.TemporaryDirectory(prefix="mnn-a7xx-probe-") as temp_dir:
        for group in sorted({item[0] for item in all_events}, key=KGSL_GROUPS.get):
            group_events = [item for item in all_events if item[0] == group]
            slots = group_events[0][3]
            for batch_index in range(0, len(group_events), slots):
                batch = group_events[batch_index:batch_index + slots]
                names = [item[2] for item in batch]
                remote_output = f"{args.remote_root}/perf/a7xx-{group.lower()}-{batch_index // slots}.json"
                command = (
                    f"cd {shlex.quote(args.remote_root)} && mkdir -p perf && "
                    f"LD_LIBRARY_PATH=$PWD/lib {shlex.quote(args.binary)} "
                    f"--model {shlex.quote(args.model)} --record {shlex.quote(args.record)} "
                    f"--op-id {args.op_id} --perf-counter-output {shlex.quote(remote_output)} "
                    f"--perf-counter-events {shlex.quote(','.join(names))}"
                )
                completed = run_remote(args.device, command)
                local_json = Path(temp_dir) / f"{group}-{batch_index}.json"
                fetch = subprocess.run(["scp", f"{args.device}:{remote_output}", str(local_json)],
                                       text=True, capture_output=True)
                if fetch.returncode != 0:
                    for item in batch:
                        rows.append({"group": item[0], "selector": item[1], "event": item[2],
                                     "slots": item[3], "status": "no-report", "delta": "", 
                                     "replay_rc": completed.returncode, "error": fetch.stderr.strip()})
                    continue
                report = json.loads(local_json.read_text())
                values = {item["name"]: item.get("value", 0) for item in report.get("counters", [])}
                for item in batch:
                    rows.append({"group": item[0], "selector": item[1], "event": item[2],
                                 "slots": item[3], "status": report.get("status", "missing"),
                                 "delta": values.get(item[2], ""), "replay_rc": completed.returncode,
                                 "error": report.get("error", "")})

    with Path(args.csv).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["group", "selector", "event", "slots", "status", "delta", "replay_rc", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} event results to {args.csv}")


if __name__ == "__main__":
    main()
