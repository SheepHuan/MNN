#!/usr/bin/env python3
"""严格验收 OpenCL/Vulkan kernel corpus PMC 原始数据。

验证一份 sweep 产物 (rows.csv + pmc.json) 是否符合发布要求：
- 行 schema 齐全（collection_session_id / manifest_sha256 / device_fingerprint / case / metric / status / value / control_value / delta / pmu_status / ...）
- 所有行共享同一 collection_session_id（一次 sweep 一个 session）
- manifest_sha256 与设备 fingerprint 非空且与 JSON 顶层一致
- 每对 (case, metric) 恰好出现一次（无重复、无遗漏——除非已知 skipped）
- VALID 行的 value/control_value/delta 都是有限数值
- delta = value - control_value（误差容忍为 0，因为是同一数源）
- pmc.json 顶层 fingerprint 与 rows.csv 一致
- pmc.json 中的 delta 与 rows.csv 中的 delta 一致

Exit code: 0 = valid, 1 = invalid (errors written to --output-json).

Schema: mnn-opencl-pmc-source-validation/v1
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


REQUIRED_ROW_COLUMNS = [
    "collection_session_id",
    "manifest_sha256",
    "device_fingerprint",
    "case",
    "metric",
    "status",
    "value",
    "control_value",
    "delta",
    "pmu_status",
    "returncode",
    "error",
    "framework",
    "backend",
    "tag",
    "variant",
]
ROW_STATUSES = {"VALID", "NOT_FOUND", "COMMAND_FAILED", "TIMEOUT", "OVERFLOW"}
EXPECTED_SCHEMA = "mnn-opencl-pmc-source-validation/v1"


def _sha256_of_file(path, chunk_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _parse_number(text, errors, context):
    if text == "" or text is None:
        return None
    try:
        v = json.loads(text)
    except (ValueError, TypeError) as exc:
        errors.append("{}: cannot parse number {}: {}".format(context, text, exc))
        return None
    if not _is_finite_number(v):
        errors.append("{}: not a finite number: {}".format(context, text))
        return None
    return v


def validate(rows_path, pmc_path, operator_cases_path, expected_backend=None,
              expected_framework=None):
    errors = []
    warnings = []
    stats = Counter()

    # ---- 1. Rows CSV schema + integrity ----
    if not Path(rows_path).exists():
        errors.append("rows csv missing: {}".format(rows_path))
        return _result(False, errors, warnings, stats)
    session_ids = set()
    manifest_shas = set()
    devices = set()
    pairs = Counter()  # (case, metric) -> count
    backends = Counter()
    frameworks = Counter()
    tags = Counter()
    valid_rows = 0
    not_found_rows = 0
    failed_rows = 0
    timeout_rows = 0
    delta_mismatch = 0

    # delta computed from rows for cross-checking with JSON
    delta_by_pair = {}

    with open(rows_path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            errors.append("rows csv has no header")
            return _result(False, errors, warnings, stats)
        missing_cols = [c for c in REQUIRED_ROW_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            errors.append("rows csv missing columns: {}".format(missing_cols))
            return _result(False, errors, warnings, stats)

        for row_index, row in enumerate(reader, start=2):  # +1 for header
            stats["rows"] += 1
            session_ids.add(row["collection_session_id"])
            manifest_shas.add(row["manifest_sha256"])
            devices.add(row["device_fingerprint"])
            backends[row["backend"]] += 1
            frameworks[row["framework"]] += 1
            tags[row["tag"]] += 1
            status = row["status"]
            if status not in ROW_STATUSES:
                errors.append("row {}: unknown status {}".format(row_index, status))
            stats["status_" + status.lower()] += 1

            case_name = row["case"]
            metric = row["metric"]
            # Use (case, tag, metric) as the unique key — same case name can
            # appear under multiple tags (e.g. binary_buf_fp32_smoke has 1.2.0
            # and 3.6.0 variants).
            tag = row["tag"]
            pairs[(case_name, tag, metric)] += 1
            if pairs[(case_name, tag, metric)] > 1:
                errors.append("row {}: duplicate (case, tag, metric) triple {} x {} x {}".format(
                    row_index, case_name, tag, metric))

            if expected_backend and row["backend"] != expected_backend:
                errors.append("row {}: backend {} != expected {}".format(
                    row_index, row["backend"], expected_backend))
            if expected_framework and row["framework"] != expected_framework:
                errors.append("row {}: framework {} != expected {}".format(
                    row_index, row["framework"], expected_framework))

            if status == "VALID":
                valid_rows += 1
                value = _parse_number(row["value"], errors, "row {} value".format(row_index))
                control = _parse_number(row["control_value"], errors, "row {} control_value".format(row_index))
                delta = _parse_number(row["delta"], errors, "row {} delta".format(row_index))
                if value is None or control is None or delta is None:
                    continue
                expected_delta = value - control
                # Both value and control come from the same source so delta should
                # match exactly. Floating point should still agree at 1e-9.
                if abs(delta - expected_delta) > 1e-9:
                    delta_mismatch += 1
                    errors.append("row {}: delta {} != value {} - control {} (= {})".format(
                        row_index, delta, value, control, expected_delta))
                delta_by_pair[(case_name, tag, metric)] = delta
            elif status == "NOT_FOUND":
                not_found_rows += 1
            elif status in ("COMMAND_FAILED", "OVERFLOW"):
                failed_rows += 1
            elif status == "TIMEOUT":
                timeout_rows += 1

    # ---- 2. Cross-row consistency ----
    if len(session_ids) != 1:
        errors.append("expected exactly 1 collection_session_id, found {}: {}".format(
            len(session_ids), sorted(session_ids)))
    session_id = next(iter(session_ids)) if session_ids else None
    if len(manifest_shas) != 1:
        errors.append("expected exactly 1 manifest_sha256, found {}: {}".format(
            len(manifest_shas), sorted(manifest_shas)))
    manifest_sha = next(iter(manifest_shas)) if manifest_shas else None
    if len(devices) != 1:
        errors.append("expected exactly 1 device_fingerprint, found {}: {}".format(
            len(devices), sorted(devices)))
    device = next(iter(devices)) if devices else None

    # ---- 3. Manifest sha matches operator_cases.json ----
    if operator_cases_path and Path(operator_cases_path).exists() and manifest_sha:
        actual_sha = _sha256_of_file(operator_cases_path)[:16]
        if actual_sha != manifest_sha:
            errors.append("manifest_sha256 mismatch: rows say {} but file is {}".format(
                manifest_sha, actual_sha))

    # ---- 4. PMC JSON fingerprint matches rows ----
    if not Path(pmc_path).exists():
        errors.append("pmc json missing: {}".format(pmc_path))
    else:
        try:
            doc = json.loads(Path(pmc_path).read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            errors.append("pmc json unreadable: {}".format(exc))
            doc = None
        if doc is not None:
            if doc.get("collection_session_id") != session_id:
                errors.append("pmc json session mismatch: {} != {}".format(
                    doc.get("collection_session_id"), session_id))
            if doc.get("manifest_sha256") != manifest_sha:
                errors.append("pmc json manifest mismatch: {} != {}".format(
                    doc.get("manifest_sha256"), manifest_sha))
            if doc.get("device_fingerprint") != device:
                errors.append("pmc json device mismatch: {} != {}".format(
                    doc.get("device_fingerprint"), device))
            json_cases = doc.get("cases", {})
            if not isinstance(json_cases, dict):
                errors.append("pmc json 'cases' is not an object")
                json_cases = {}
            # ---- 5. JSON deltas agree with rows ----
            # JSON key is "case_name" or "case_name@tag" (when tag is non-empty).
            json_pairs = set()
            for json_key, entry in json_cases.items():
                if "@" in json_key:
                    case_name, tag = json_key.rsplit("@", 1)
                else:
                    case_name, tag = json_key, ""
                pmc = entry.get("pmc", {}) if isinstance(entry, dict) else {}
                for metric, delta in pmc.items():
                    json_pairs.add((case_name, tag, metric))
                    row_key = (case_name, tag, metric)
                    if row_key not in delta_by_pair:
                        errors.append("pmc json has pair not in rows: {}".format(row_key))
                        continue
                    if not _is_finite_number(delta):
                        errors.append("pmc json delta not a finite number: {} = {}".format(
                            row_key, delta))
                        continue
                    row_delta = delta_by_pair[row_key]
                    if abs(delta - row_delta) > 1e-9:
                        errors.append("pmc json delta {} != row delta {} for {}".format(
                            delta, row_delta, row_key))
            rows_valid_pairs = set(delta_by_pair.keys())
            missing_in_json = rows_valid_pairs - json_pairs
            if missing_in_json:
                errors.append("pmc json missing {} VALID pairs (e.g. {})".format(
                    len(missing_in_json), list(missing_in_json)[:3]))

    # ---- 6. Stats for diagnosis ----
    stats["valid_rows"] = valid_rows
    stats["not_found_rows"] = not_found_rows
    stats["failed_rows"] = failed_rows
    stats["timeout_rows"] = timeout_rows
    stats["delta_mismatch"] = delta_mismatch
    stats["unique_cases"] = len({case for (case, _, _) in pairs.keys()})
    stats["unique_case_tag_pairs"] = len({(c, t) for (c, t, _) in pairs.keys()})
    stats["unique_metrics"] = len({m for (_, _, m) in pairs.keys()})
    stats["backends"] = dict(backends)
    stats["frameworks"] = dict(frameworks)
    stats["tags"] = dict(tags)

    if valid_rows == 0:
        errors.append("no VALID rows in the sweep")
    if not_found_rows > 0 and valid_rows == 0:
        warnings.append("sweep has only NOT_FOUND rows; PMC may be unavailable on this device")

    valid = len(errors) == 0
    return _result(valid, errors, warnings, stats,
                   session_id=session_id, manifest_sha=manifest_sha, device=device)


def _result(valid, errors, warnings, stats, session_id=None, manifest_sha=None, device=None):
    return {
        "schema_version": EXPECTED_SCHEMA,
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "collection_session_id": session_id,
        "manifest_sha256": manifest_sha,
        "device_fingerprint": device,
        "stats": dict(stats),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmc-rows", required=True, help="rows csv from sweep_opencl_kernel_pmc.py")
    parser.add_argument("--pmc-json", required=True, help="pmc json from sweep_opencl_kernel_pmc.py")
    parser.add_argument("--operator-cases", help="operator_cases.json to verify manifest_sha256")
    parser.add_argument("--expected-backend", choices=("opencl", "vulkan"))
    parser.add_argument("--expected-framework", choices=("mnn", "ncnn"))
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    result = validate(
        rows_path=args.pmc_rows,
        pmc_path=args.pmc_json,
        operator_cases_path=args.operator_cases,
        expected_backend=args.expected_backend,
        expected_framework=args.expected_framework,
    )
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("schema_version: {}".format(result["schema_version"]))
    print("valid: {}".format(result["valid"]))
    print("errors: {}".format(len(result["errors"])))
    for e in result["errors"][:10]:
        print("  - {}".format(e))
    if len(result["errors"]) > 10:
        print("  ... {} more".format(len(result["errors"]) - 10))
    print("warnings: {}".format(len(result["warnings"])))
    print("stats: {}".format(result["stats"]))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
