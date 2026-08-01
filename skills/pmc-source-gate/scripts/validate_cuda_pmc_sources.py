#!/usr/bin/env python3
"""严格验收 CUDA PMC 原始 manifest、latency、plan 与 rows。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernel_agent.pmc_interpreter.dataset.selection import (  # noqa: E402
    CASE_SELECTION_POLICY_VERSION,
    CONDITION_BALANCED_POLICY,
    load_case_selection_plan,
    validate_case_selection_plan_source,
)
from kernel_agent.pmc_interpreter.dataset.model import (  # noqa: E402
    SemanticFactProvenance,
)


ROW_COLUMNS = [
    "sweep_plan_id",
    "sweep_config_id",
    "case",
    "metric",
    "status",
    "value",
    "pmu_status",
    "num_passes",
    "returncode",
    "error",
    "collection_session_id",
    "order_index",
    "gpu_clock_hz_before",
    "gpu_clock_hz_after",
    "temperature_c_before",
    "temperature_c_after",
    "environment_status",
    "environment_source",
]
ROW_STATUSES = {
    "VALID",
    "NOT_FOUND",
    "OVERFLOW",
    "COMMAND_FAILED",
    "MALFORMED_OUTPUT",
}
SAMPLED_METRIC_STATUSES = {
    "VALID",
    "NOT_FOUND",
    "OVERFLOW",
}
LOWER_HEX_DIGITS = frozenset("0123456789abcdef")
RESOURCE_FIELDS = (
    "registers_per_thread",
    "static_shared_memory_bytes",
    "dynamic_shared_memory_bytes",
    "local_memory_per_thread_bytes",
    "local_memory_total_bytes",
)
ENVIRONMENT_FIELDS = (
    "gpu_clock_hz_before",
    "gpu_clock_hz_after",
    "temperature_c_before",
    "temperature_c_after",
)
P0_TARGET_OP_TYPES = (
    "avgpool",
    "conv_dw",
    "matmul",
    "maxpool",
    "reduction",
    "softmax",
    "transpose",
)
P0_CONDITIONS_PER_OP_TYPE = 5
LATENCY_TOP_LEVEL_FIELDS = (
    "format",
    "version",
    "source_manifest_sha256",
    "cases",
)
VALID_METRIC_COLUMNS = ["metric", "value"]
PROVENANCE_FIELDS = {
    "status",
    "source",
    "method",
    "formula",
    "confidence",
    "notes",
}
PROVENANCE_STATUSES = {
    "declared",
    "not_applicable",
    "theoretical_estimate",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if positive and number <= 0:
        return False
    if nonnegative and number < 0:
        return False
    return True


def _integer(value, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    if positive and value <= 0:
        return False
    if nonnegative and value < 0:
        return False
    return True


def _parse_csv_number(value, *, positive=False, nonnegative=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if positive and number <= 0:
        return None
    if nonnegative and number < 0:
        return None
    return number


def _is_lower_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in LOWER_HEX_DIGITS for character in value)
    )


def _launch_status_kind(raw_status):
    status = str(raw_status or "").strip()
    exact = {
        "sampled": "sampled",
        "no_launch": "no_launch",
        "not_collected": "not_collected",
        "not_requested": "not_collected",
        "unavailable": "unavailable",
        "partial": "error",
        "error": "error",
    }
    if status in exact:
        return exact[status]
    if status.startswith((
        "metadata launch failed:",
        "unavailable:metadata launch failed:",
        "partial:",
        "error:",
    )):
        return "error"
    if status.startswith("unavailable:"):
        return "unavailable"
    return None


def _environment_status_kind(raw_status):
    status = str(raw_status or "").strip()
    exact = {
        "sampled": "sampled",
        "partial": "partial",
        "unavailable": "unavailable",
        "not_collected": "not_collected",
        "error": "error",
    }
    if status in exact:
        return exact[status]
    if status.startswith("partial:"):
        return "partial"
    if status.startswith("unavailable:"):
        return "unavailable"
    if status.startswith("error:"):
        return "error"
    return None


def _validate_provenance(case, field, key, raw, errors, *, value_present):
    label = "{} 的 {}.{} provenance".format(case, field, key)
    if not isinstance(raw, dict) or not raw:
        errors.append("{} 为空或不是对象".format(label))
        return None
    missing = sorted(PROVENANCE_FIELDS - set(raw))
    extra = sorted(set(raw) - PROVENANCE_FIELDS)
    if missing:
        errors.append("{} 缺少字段：{}".format(label, ",".join(missing)))
    if extra:
        errors.append("{} 包含多余字段：{}".format(label, ",".join(extra)))
    provenance = None
    if not missing and not extra:
        try:
            provenance = SemanticFactProvenance.model_validate(raw)
        except ValidationError as error:
            errors.append("{} 不满足结构化 schema：{}".format(label, error))
    if set(raw) != PROVENANCE_FIELDS:
        errors.append(
            "{} 字段必须精确为 {}".format(
                label, ",".join(sorted(PROVENANCE_FIELDS))
            )
        )
    status = raw.get("status")
    if status not in PROVENANCE_STATUSES:
        errors.append("{} status 非法：{}".format(label, status))
    for name in ("source", "method"):
        if not isinstance(raw.get(name), str) or not raw[name].strip():
            errors.append("{} {} 必须是非空字符串".format(label, name))
    formula = raw.get("formula")
    if not isinstance(formula, str):
        errors.append("{} formula 必须是字符串".format(label))
    elif status != "not_applicable" and not formula.strip():
        errors.append("{} 的适用事实必须提供非空 formula".format(label))
    if not isinstance(raw.get("notes"), str):
        errors.append("{} notes 必须是字符串".format(label))
    confidence = raw.get("confidence")
    if (
        not _finite_number(confidence)
        or float(confidence) < 0.0
        or float(confidence) > 1.0
    ):
        errors.append("{} confidence 必须是 [0, 1] 内有限数".format(label))
    if value_present and status == "not_applicable":
        errors.append("{} 同时有数值和 not_applicable 状态".format(label))
    if not value_present and status != "not_applicable":
        errors.append("{} 没有数值但未标为 not_applicable".format(label))
    return provenance


def _load_manifest(path, backend, errors):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "mnn-kernel-operator-cases":
        errors.append("operator_cases.json format 不是 mnn-kernel-operator-cases")
    if payload.get("version") != 1:
        errors.append("operator_cases.json version 不是 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        errors.append("operator_cases.json 缺少 cases 数组")
        return {}, ()
    cases = {}
    order = []
    for index, entry in enumerate(raw_cases):
        if not isinstance(entry, dict) or entry.get("backend") != backend:
            continue
        case = str(entry.get("name") or "")
        if not case:
            errors.append("第 {} 个 {} case 缺少 name".format(index, backend))
            continue
        if case in cases:
            errors.append("{} case name 重复：{}".format(backend, case))
            continue
        cases[case] = entry
        order.append(case)
    if not cases:
        errors.append("manifest 中没有 {} case".format(backend))
    return cases, tuple(order)


def _validate_selected_semantics(selected_cases, manifest_cases, errors):
    for selected in selected_cases:
        case = selected.case_id
        entry = manifest_cases.get(case)
        if entry is None:
            errors.append("selection plan case 不在 manifest：{}".format(case))
            continue
        if selected.metadata_level != "explicit":
            errors.append("P0 selected case 不是 explicit metadata：{}".format(case))
        expected_dtype = str(entry.get("dtype") or "unknown")
        expected_shapes = entry.get("shapes") or {}
        expected_workload = dict(entry.get("workload") or {})
        for key in ("algorithmic_flops", "algorithmic_bytes", "output_elements"):
            if key in entry:
                expected_workload[key] = entry[key]
        expected_params = dict(entry.get("int_params") or {})
        expected_params.update(entry.get("float_params") or {})
        selected_fields = {
            "op_type": selected.op_type,
            "dtype": selected.dtype,
            "shapes": selected.shapes,
            "workload": selected.workload,
            "params": selected.params,
        }
        expected_fields = {
            "op_type": str(entry.get("op_type") or ""),
            "dtype": expected_dtype,
            "shapes": expected_shapes,
            "workload": expected_workload,
            "params": expected_params,
        }
        for field, expected in expected_fields.items():
            if selected_fields[field] != expected:
                errors.append(
                    "selection plan case {} 的 {} 与 manifest 不一致".format(
                        case, field
                    )
                )
        shapes = entry.get("shapes")
        shape_provenance = entry.get("shape_provenance")
        if not shapes or not isinstance(shapes, dict):
            errors.append("P0 selected case 缺少 shapes：{}".format(case))
        elif not isinstance(shape_provenance, dict):
            errors.append("P0 selected case 缺少 shape provenance：{}".format(case))
        else:
            for key in shapes:
                _validate_provenance(
                    case,
                    "shape",
                    key,
                    shape_provenance.get(key),
                    errors,
                    value_present=True,
                )
        workload = entry.get("workload")
        provenance = entry.get("workload_provenance")
        if not isinstance(workload, dict) or not isinstance(provenance, dict):
            errors.append("P0 selected case 缺少 workload/provenance：{}".format(case))
            continue
        for key in ("output_elements", "algorithmic_bytes"):
            value = workload.get(key)
            if not _finite_number(value, positive=True):
                errors.append("{} 的 {} 不是正有限数".format(case, key))
            _validate_provenance(
                case,
                "workload",
                key,
                provenance.get(key),
                errors,
                value_present=value is not None,
            )
        flops = workload.get("algorithmic_flops")
        _validate_provenance(
            case,
            "workload",
            "algorithmic_flops",
            provenance.get("algorithmic_flops"),
            errors,
            value_present=flops is not None,
        )
        if flops is not None and not _finite_number(flops, positive=True):
            errors.append("{} 的 algorithmic_flops 不是正有限数".format(case))
        if "launch_records" in entry or "environment" in entry:
            errors.append("{} 把设备实测字段写入了理论 manifest".format(case))


def _load_latency(path, manifest_sha256, errors):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        errors.append("latency JSON 顶层必须是对象")
        return {}
    if set(payload) != set(LATENCY_TOP_LEVEL_FIELDS):
        errors.append(
            "latency JSON 顶层字段必须精确为 {}".format(
                ",".join(LATENCY_TOP_LEVEL_FIELDS)
            )
        )
    if payload.get("format") != "mnn-kernel-latency":
        errors.append("latency JSON format 不是 mnn-kernel-latency")
    if payload.get("version") != 1:
        errors.append("latency JSON version 不是 1")
    if payload.get("source_manifest_sha256") != manifest_sha256:
        errors.append("latency JSON source_manifest_sha256 与当前 manifest 不一致")
    records = payload.get("cases")
    if not isinstance(records, dict):
        errors.append("latency JSON 缺少 cases 对象")
        return {}
    return records


def _validate_launch_record(case, record, index, errors):
    if not isinstance(record, dict):
        errors.append("{} launch_records[{}] 不是对象".format(case, index))
        return
    if not str(record.get("kernel_name") or ""):
        errors.append("{} launch_records[{}] 缺少 kernel_name".format(case, index))
    for field in ("grid", "block"):
        vector = record.get(field)
        if (
            not isinstance(vector, list)
            or len(vector) != 3
            or any(not _integer(value, positive=True) for value in vector)
        ):
            errors.append(
                "{} launch_records[{}].{} 不是正三维整数向量".format(
                    case, index, field
                )
            )
    for field in RESOURCE_FIELDS:
        if not _integer(record.get(field), nonnegative=True):
            errors.append(
                "{} launch_records[{}].{} 不是非负整数实测值".format(
                    case, index, field
                )
            )


def _validate_latency_environment(case, environment, errors):
    if not isinstance(environment, dict):
        errors.append("latency case {} 缺少 environment 对象".format(case))
        return None, False

    raw_status = environment.get("sampling_status")
    status_kind = _environment_status_kind(raw_status)
    if status_kind is None:
        errors.append(
            "latency case {} 的 environment sampling_status 未知：{}".format(
                case, raw_status
            )
        )

    source = environment.get("sampling_source")
    source_available = isinstance(source, str) and bool(source.strip())
    measured_fields = {}
    for field in ENVIRONMENT_FIELDS:
        value = environment.get(field)
        if value is None or value == "":
            measured_fields[field] = False
            continue
        measured_fields[field] = _finite_number(value, positive=True)
        if not measured_fields[field]:
            errors.append(
                "latency case {} 的 environment.{} 不是正有限实测值".format(
                    case, field
                )
            )
    complete_measurement = all(measured_fields.values())

    if status_kind == "sampled":
        if not source_available:
            errors.append(
                "latency case {} 已实测但 environment sampling_source 为空".format(
                    case
                )
            )
        for field, available in measured_fields.items():
            if not available:
                errors.append(
                    "latency case {} sampled environment 缺少实测 {}".format(
                        case, field
                    )
                )
    elif status_kind is not None and complete_measurement:
        errors.append(
            "latency case {} 的 environment 状态 {} 与完整实测值矛盾".format(
                case, raw_status
            )
        )
    return status_kind, complete_measurement


def _validate_latency_records(records, manifest_cases, selected_ids, errors):
    manifest_ids = set(manifest_cases)
    record_ids = set(records)
    missing = sorted(manifest_ids - record_ids)
    extra = sorted(record_ids - manifest_ids)
    if missing:
        errors.append("latency 缺少 {} 个 manifest case".format(len(missing)))
    if extra:
        errors.append("latency 包含 {} 个额外 case".format(len(extra)))
    for case in sorted(manifest_ids & record_ids):
        record = records[case]
        if not isinstance(record, dict):
            errors.append("latency case {} 不是对象".format(case))
            continue
        if not _finite_number(record.get("latency_us"), positive=True):
            errors.append("latency case {} 的 latency_us 无效".format(case))
        raw_launch_status = record.get("launch_sampling_status")
        launch_status = _launch_status_kind(raw_launch_status)
        if launch_status is None:
            errors.append(
                "latency case {} 的 launch_sampling_status 未知：{}".format(
                    case, raw_launch_status
                )
            )
        launches = record.get("launch_records")
        if not isinstance(launches, list):
            errors.append("latency case {} 的 launch_records 不是数组".format(case))
            launches = []
        for index, launch in enumerate(launches):
            _validate_launch_record(case, launch, index, errors)
        if launch_status == "sampled" and not launches:
            errors.append("latency case {} 标记 sampled 但没有 launch record".format(case))
        elif launch_status is not None and launch_status != "sampled" and launches:
            errors.append(
                "latency case {} 的 launch 状态 {} 不得携带 launch records".format(
                    case, raw_launch_status
                )
            )
        environment_status, environment_complete = _validate_latency_environment(
            case, record.get("environment"), errors
        )
        if case in selected_ids:
            if launch_status != "sampled" or not launches:
                errors.append("P0 selected case {} 没有成功实测 launch/resource".format(case))
            if environment_status != "sampled" or not environment_complete:
                errors.append("P0 selected case {} 没有成功实测环境".format(case))


def _read_valid_metrics(path, errors):
    metrics = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != VALID_METRIC_COLUMNS:
            errors.append(
                "cuda_pmc_valid.csv 表头必须精确为 {}".format(
                    ",".join(VALID_METRIC_COLUMNS)
                )
            )
            return ()
        for line, row in enumerate(reader, 2):
            if None in row:
                errors.append("cuda_pmc_valid.csv 第 {} 行包含额外列".format(line))
            metric = str(row.get("metric") or "").strip()
            if not metric:
                errors.append("cuda_pmc_valid.csv 第 {} 行 metric 为空".format(line))
                continue
            if metric in seen:
                errors.append("cuda_pmc_valid.csv 有重复有效 metric：{}".format(metric))
                continue
            seen.add(metric)
            metrics.append(metric)
            if _parse_csv_number(row.get("value"), nonnegative=True) is None:
                errors.append(
                    "cuda_pmc_valid.csv 第 {} 行 value 不是非负有限数".format(
                        line
                    )
                )
    if not metrics:
        errors.append("cuda_pmc_valid.csv 没有有效 metric")
    return tuple(metrics)


def _validate_rows(
    path,
    plan,
    metrics,
    require_environment,
    reject_infrastructure_failures,
    errors,
):
    selected_ids = tuple(case.case_id for case in plan.selected_cases)
    expected_count = len(selected_ids) * len(metrics)
    status_counts = Counter()
    config_ids = set()
    sessions = {}
    seen_session_order = {}
    last_new_session_order = 0
    active_session_id = None
    closed_session_ids = set()
    row_count = 0
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROW_COLUMNS:
            errors.append("PMC rows 表头与 ROW_COLUMNS 不一致")
            return {
                "row_count": 0,
                "expected_row_count": expected_count,
                "status_counts": {},
                "session_count": 0,
                "sweep_config_ids": [],
            }
        for row_index, row in enumerate(reader):
            row_count += 1
            if None in row:
                errors.append(
                    "PMC rows 第 {} 行包含额外列".format(row_index + 2)
                )
            if row_index >= expected_count:
                errors.append("PMC rows 超出 plan × metric 笛卡尔积")
                continue
            expected_case = selected_ids[row_index // len(metrics)]
            expected_metric = metrics[row_index % len(metrics)]
            if row["case"] != expected_case or row["metric"] != expected_metric:
                errors.append(
                    "PMC rows 第 {} 行顺序错误：应为 ({}, {})，实际为 ({}, {})".format(
                        row_index + 2,
                        expected_case,
                        expected_metric,
                        row["case"],
                        row["metric"],
                    )
                )
            if row["sweep_plan_id"] != plan.plan_id:
                errors.append("PMC rows 第 {} 行 plan ID 不一致".format(row_index + 2))
            raw_config_id = row["sweep_config_id"]
            config_id = raw_config_id.strip()
            if not config_id:
                errors.append("PMC rows 第 {} 行 sweep_config_id 为空".format(row_index + 2))
            else:
                config_ids.add(config_id)
                if not _is_lower_sha256(raw_config_id):
                    errors.append(
                        "PMC rows 第 {} 行 sweep_config_id 不是 64 位小写 SHA256".format(
                            row_index + 2
                        )
                    )
            status = row["status"].strip()
            status_counts[status] += 1
            if status not in ROW_STATUSES:
                errors.append("PMC rows 第 {} 行状态未知：{}".format(row_index + 2, status))
            elif reject_infrastructure_failures and status in {
                "COMMAND_FAILED",
                "MALFORMED_OUTPUT",
            }:
                errors.append(
                    "正式 P0 PMC rows 第 {} 行不允许基础设施失败状态：{}".format(
                        row_index + 2, status
                    )
                )
            if status == "VALID":
                if _parse_csv_number(row["value"], nonnegative=True) is None:
                    errors.append(
                        "PMC rows 第 {} 行 VALID value 不是非负有限数".format(
                            row_index + 2
                        )
                    )
            if status in SAMPLED_METRIC_STATUSES:
                if row["pmu_status"] != "sampled":
                    errors.append(
                        "PMC rows 第 {} 行 {} 但 pmu_status 非 sampled".format(
                            row_index + 2, status
                        )
                    )
                if row["returncode"] != "0":
                    errors.append(
                        "PMC rows 第 {} 行 {} 但 returncode 非 0".format(
                            row_index + 2, status
                        )
                    )
                try:
                    if int(row["num_passes"]) <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(
                        "PMC rows 第 {} 行 {} 但 num_passes 无效".format(
                            row_index + 2, status
                        )
                    )
            session_id = row["collection_session_id"].strip()
            if not session_id:
                errors.append("PMC rows 第 {} 行 collection_session_id 为空".format(row_index + 2))
                continue
            if session_id != active_session_id:
                if active_session_id is not None:
                    closed_session_ids.add(active_session_id)
                if session_id in closed_session_ids:
                    errors.append("PMC session {} 在 rows 中非连续重开".format(session_id))
                active_session_id = session_id
            try:
                order_index = int(row["order_index"])
                if order_index <= 0:
                    raise ValueError
            except ValueError:
                errors.append("PMC rows 第 {} 行 order_index 无效".format(row_index + 2))
                continue
            session_fact = (
                row["case"],
                order_index,
                row["pmu_status"],
                row["num_passes"],
                row["returncode"],
                *(row[field] for field in ENVIRONMENT_FIELDS),
                row["environment_status"],
                row["environment_source"],
            )
            previous = sessions.get(session_id)
            if previous is not None and previous != session_fact:
                errors.append(
                    "PMC session {} 的 case/order/result/environment 不一致".format(
                        session_id
                    )
                )
            elif previous is None:
                sessions[session_id] = session_fact
                if order_index in seen_session_order:
                    errors.append(
                        "PMC session {} 与 {} 重复使用 order_index {}".format(
                            session_id, seen_session_order[order_index], order_index
                        )
                    )
                seen_session_order[order_index] = session_id
                if order_index <= last_new_session_order:
                    errors.append("PMC session order_index 不是严格递增")
                last_new_session_order = max(last_new_session_order, order_index)
            if row["environment_status"] == "sampled" and not row["environment_source"].strip():
                errors.append("PMC session {} 已实测但 environment_source 为空".format(session_id))
            if require_environment:
                if row["environment_status"] != "sampled":
                    errors.append("PMC session {} 没有成功实测环境".format(session_id))
                for field in ENVIRONMENT_FIELDS:
                    if _parse_csv_number(row[field], positive=True) is None:
                        errors.append("PMC session {} 缺少实测 {}".format(session_id, field))
    if row_count != expected_count:
        errors.append(
            "PMC rows 行数为 {}，应为 {} × {} = {}".format(
                row_count, len(selected_ids), len(metrics), expected_count
            )
        )
    if len(config_ids) != 1:
        errors.append("PMC rows 必须且只能包含一个 sweep_config_id")
    return {
        "row_count": row_count,
        "expected_row_count": expected_count,
        "status_counts": dict(sorted(status_counts.items())),
        "session_count": len(sessions),
        "sweep_config_ids": sorted(config_ids),
    }


def validate_sources(
    *,
    operator_cases,
    latency_json,
    selection_plan,
    valid_csv,
    pmc_rows,
    backend="cuda",
    require_complete_targets=False,
    require_measured_environment=False,
):
    errors = []
    manifest_sha256 = _sha256(operator_cases)
    manifest_cases, manifest_order = _load_manifest(operator_cases, backend, errors)
    try:
        plan = load_case_selection_plan(selection_plan)
        validate_case_selection_plan_source(plan, operator_cases)
    except (OSError, ValueError) as error:
        errors.append("selection plan 无效：{}".format(error))
        plan = None

    selected_ids = set()
    if plan is not None:
        if plan.backend != backend:
            errors.append("selection plan backend 与验收 backend 不一致")
        selected_ids = {case.case_id for case in plan.selected_cases}
        _validate_selected_semantics(plan.selected_cases, manifest_cases, errors)
        if require_complete_targets:
            expected_targets = set(P0_TARGET_OP_TYPES)
            actual_targets = set(plan.target_op_types)
            missing_targets = sorted(expected_targets - actual_targets)
            extra_targets = sorted(actual_targets - expected_targets)
            if missing_targets:
                errors.append(
                    "P0 selection plan 缺少 target：{}".format(
                        ",".join(missing_targets)
                    )
                )
            if extra_targets:
                errors.append(
                    "P0 selection plan 包含多余 target：{}".format(
                        ",".join(extra_targets)
                    )
                )
            if plan.policy != CONDITION_BALANCED_POLICY:
                errors.append(
                    "P0 selection plan policy 必须是 {}".format(
                        CONDITION_BALANCED_POLICY
                    )
                )
            if plan.policy_version != CASE_SELECTION_POLICY_VERSION:
                errors.append(
                    "P0 selection plan policy_version 必须是 {}".format(
                        CASE_SELECTION_POLICY_VERSION
                    )
                )
            if plan.minimum_conditions_per_op_type != P0_CONDITIONS_PER_OP_TYPE:
                errors.append(
                    "P0 selection plan 每类 condition 目标必须是 {}".format(
                        P0_CONDITIONS_PER_OP_TYPE
                    )
                )
            for op_type in P0_TARGET_OP_TYPES:
                group = plan.groups.get(op_type)
                if group is None:
                    continue
                if group.status != "satisfied":
                    errors.append("P0 target {} 未由完整显式 condition 满足".format(op_type))
                if group.requested_condition_count != P0_CONDITIONS_PER_OP_TYPE:
                    errors.append(
                        "P0 target {} 的 requested condition 数不是 {}".format(
                            op_type, P0_CONDITIONS_PER_OP_TYPE
                        )
                    )
                if group.selected_case_count < P0_CONDITIONS_PER_OP_TYPE:
                    errors.append("P0 target {} selected condition 数不足".format(op_type))

    latency_records = _load_latency(latency_json, manifest_sha256, errors)
    _validate_latency_records(
        latency_records, manifest_cases, selected_ids, errors
    )
    metrics = _read_valid_metrics(valid_csv, errors)
    row_summary = (
        _validate_rows(
            pmc_rows,
            plan,
            metrics,
            require_measured_environment,
            require_complete_targets,
            errors,
        )
        if plan is not None and metrics
        else {
            "row_count": 0,
            "expected_row_count": 0,
            "status_counts": {},
            "session_count": 0,
            "sweep_config_ids": [],
        }
    )
    summary = {
        "schema": "mnn-cuda-pmc-source-validation/v1",
        "valid": not errors,
        "errors": errors,
        "manifest_sha256": manifest_sha256,
        "manifest_case_count": len(manifest_order),
        "selection_plan_id": plan.plan_id if plan is not None else "",
        "target_op_types": list(plan.target_op_types) if plan is not None else [],
        "selected_case_count": len(selected_ids),
        "latency_case_count": len(latency_records),
        "metric_count": len(metrics),
        **row_summary,
    }
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-cases", required=True)
    parser.add_argument("--latency-json", required=True)
    parser.add_argument("--selection-plan", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--pmc-rows", required=True)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--require-complete-targets", action="store_true")
    parser.add_argument("--require-measured-environment", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = validate_sources(
        operator_cases=args.operator_cases,
        latency_json=args.latency_json,
        selection_plan=args.selection_plan,
        valid_csv=args.valid_csv,
        pmc_rows=args.pmc_rows,
        backend=args.backend,
        require_complete_targets=args.require_complete_targets,
        require_measured_environment=args.require_measured_environment,
    )
    text = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
