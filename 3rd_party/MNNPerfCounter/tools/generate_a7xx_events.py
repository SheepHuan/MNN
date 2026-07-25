#!/usr/bin/env python3
"""Generate the compact A7xx event table from the Mesa register database."""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <xml> <json> <output>", file=sys.stderr)
        return 1
    xml_path, json_path, output_path = map(Path, sys.argv[1:])
    # KGSL group ids are a kernel UAPI numbering, not the order of Mesa's
    # register JSON. Keep this mapping explicit and skip BV groups, which are
    # separate binning pipes and are not exposed by the ordinary counter ioctl.
    kgsl_groups = {
        "CP": 0, "RBBM": 1, "PC": 2, "VFD": 3, "HLSQ": 4,
        "VPC": 5, "TSE": 6, "RAS": 7, "UCHE": 8, "TP": 9,
        "SP": 10, "RB": 11, "VSC": 12, "CCU": 13, "LRZ": 14,
        "CMP": 15, "UFC": 16,
    }
    mesa_groups = {group["name"]: group["num"] for group in json.loads(json_path.read_text())["groups"]}
    root = ET.parse(xml_path).getroot()
    rows = []
    pattern = re.compile(r"^a7xx_(.+)_perfcounter_select$")
    for enum in root.iter():
        if enum.tag.rsplit("}", 1)[-1] != "enum":
            continue
        match = pattern.match(enum.attrib.get("name", ""))
        if not match:
            continue
        group = match.group(1).upper()
        if group not in kgsl_groups:
            continue
        group_id, slots = kgsl_groups[group], mesa_groups[group]
        for value in enum:
            if value.tag.rsplit("}", 1)[-1] != "value":
                continue
            event = value.attrib.get("name", "")
            if not event.startswith("A7XX_PERF_"):
                continue
            name = event[len("A7XX_PERF_"):].lower()
            rows.append((name, group_id, int(value.attrib["value"], 0), slots))
    rows.sort(key=lambda row: (row[1], row[2], row[0]))
    with output_path.open("w", encoding="utf-8") as output:
        output.write("// Generated from Mesa a7xx_perfcntrs.xml/json. Do not edit.\n")
        output.write('#include "A7xxPerfCounters.hpp"\n\n')
        output.write("namespace MNN { namespace PerfCounter { namespace detail {\n")
        output.write("static const A7xxEvent kEvents[] = {\n")
        for name, group, selector, slots in rows:
            output.write(f'    {{"{name}", {group}u, {selector}u, {slots}u}},\n')
        output.write("};\n")
        output.write("const A7xxEvent* a7xxEvents(size_t* count) {\n")
        output.write("    *count = sizeof(kEvents) / sizeof(kEvents[0]);\n")
        output.write("    return kEvents;\n}\n")
        output.write("} } } // namespace MNN::PerfCounter::detail\n")
    print(f"generated {len(rows)} A7xx events", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
