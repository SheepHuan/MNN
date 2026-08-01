"""平台无关的 kernel condition 采样计划。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, JsonValue, model_validator

from .model import PmcModel


CASE_SELECTION_SCHEMA_VERSION = "mnn-pmc-case-selection/v1"
CONDITION_BALANCED_POLICY = "condition-balanced"
ALL_CASES_POLICY = "all"
CASE_SELECTION_POLICY_VERSION = 1


class SelectedKernelCase(PmcModel):
    """采样计划中的一个可执行 case 及其 workload condition 身份。"""

    case_id: str
    op_type: str
    dtype: str
    shapes: dict[str, JsonValue] = Field(default_factory=dict)
    workload: dict[str, JsonValue] = Field(default_factory=dict)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    shape_fingerprint: str
    workload_fingerprint: str
    condition_fingerprint: str
    metadata_level: Literal["explicit", "params_proxy", "missing"]
    metadata_issues: tuple[str, ...] = ()
    selection_rank: int = Field(ge=1)
    selection_reason: str


class OpTypeCaseSelection(PmcModel):
    """一个 op_type 的 condition 多样性和最终选择状态。"""

    op_type: str
    available_case_count: int = Field(ge=0)
    distinct_condition_count: int = Field(ge=0)
    requested_condition_count: int = Field(ge=1)
    selected_case_count: int = Field(ge=0)
    status: Literal[
        "satisfied",
        "satisfied_with_proxy",
        "insufficient_unique_conditions",
        "all_selected",
    ]
    metadata_levels: tuple[str, ...]
    selected_case_ids: tuple[str, ...]


class CaseSelectionPlan(PmcModel):
    """从 operator case manifest 确定性产生的 PMC case 采样计划。"""

    schema_version: str = CASE_SELECTION_SCHEMA_VERSION
    plan_id: str
    source_manifest_sha256: str
    backend: str
    policy: Literal["condition-balanced", "all"]
    policy_version: int = CASE_SELECTION_POLICY_VERSION
    minimum_conditions_per_op_type: int = Field(ge=1)
    target_op_types: tuple[str, ...]
    groups: dict[str, OpTypeCaseSelection]
    selected_cases: tuple[SelectedKernelCase, ...]
    issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        errors: list[str] = []
        if self.schema_version != CASE_SELECTION_SCHEMA_VERSION:
            errors.append("unsupported case selection schema {}".format(self.schema_version))
        if self.policy_version != CASE_SELECTION_POLICY_VERSION:
            errors.append("unsupported case selection policy version {}".format(self.policy_version))
        if tuple(sorted(set(self.target_op_types))) != self.target_op_types:
            errors.append("target_op_types must be sorted and unique")
        if set(self.groups) != set(self.target_op_types):
            errors.append("groups must exactly match target_op_types")

        selected_ids = [case.case_id for case in self.selected_cases]
        if len(selected_ids) != len(set(selected_ids)):
            errors.append("selected case IDs must be unique")
        grouped_ids: list[str] = []
        for op_type in self.target_op_types:
            group = self.groups.get(op_type)
            if group is None:
                continue
            if group.op_type != op_type:
                errors.append("group key/op_type mismatch for {}".format(op_type))
            if group.selected_case_count != len(group.selected_case_ids):
                errors.append("selected case count mismatch for {}".format(op_type))
            grouped_ids.extend(group.selected_case_ids)
        if grouped_ids != selected_ids:
            errors.append("selected_cases must follow target op_type and rank order")
        if errors:
            raise ValueError("invalid case selection plan: {}".format("; ".join(errors)))
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _mapping(raw: Any, field: str, case_id: str) -> dict[str, JsonValue]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("case {} field {} must be an object".format(case_id, field))
    return dict(raw)


def _case_record(entry: dict[str, Any]) -> dict[str, Any]:
    case_id = str(entry.get("name") or "")
    op_type = str(entry.get("op_type") or "")
    if not case_id:
        raise ValueError("eligible case has no name")
    if not op_type:
        raise ValueError("case {} has no op_type".format(case_id))

    dtype = str(entry.get("dtype") or "unknown")
    shapes = _mapping(entry.get("shapes"), "shapes", case_id)
    workload = _mapping(entry.get("workload"), "workload", case_id)
    for key in ("algorithmic_flops", "algorithmic_bytes", "output_elements"):
        if key in entry:
            workload[key] = entry[key]
    params = _mapping(entry.get("int_params"), "int_params", case_id)
    params.update(_mapping(entry.get("float_params"), "float_params", case_id))

    if shapes and workload:
        metadata_level = "explicit"
    elif shapes or workload or params:
        metadata_level = "params_proxy"
    else:
        metadata_level = "missing"
    metadata_issues = []
    if dtype == "unknown":
        metadata_issues.append("missing_dtype")
    if not shapes:
        metadata_issues.append("missing_shapes")
    if not workload:
        metadata_issues.append("missing_workload")

    condition_payload = {
        "dtype": dtype,
        "shapes": shapes,
        "workload": workload,
        "params": params,
    }
    workload_payload = workload if workload else {"params_proxy": params}
    return {
        "case_id": case_id,
        "op_type": op_type,
        "dtype": dtype,
        "shapes": shapes,
        "workload": workload,
        "params": params,
        "shape_fingerprint": _fingerprint(shapes),
        "workload_fingerprint": _fingerprint(workload_payload),
        "condition_fingerprint": _fingerprint(condition_payload),
        "metadata_level": metadata_level,
        "metadata_issues": tuple(metadata_issues),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            result.update(_flatten(value[key], child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = "{}.{}".format(prefix, index) if prefix else str(index)
            result.update(_flatten(item, child))
    else:
        result[prefix or "value"] = value
    return result


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ranges(records: list[dict[str, Any]], field: str) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, raw in _flatten(record[field]).items():
            value = _numeric(raw)
            if value is not None:
                values[key].append(value)
    return {key: (min(items), max(items)) for key, items in values.items()}


def _mapping_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    ranges: dict[str, tuple[float, float]],
) -> float:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    keys = sorted(set(left_flat) | set(right_flat))
    if not keys:
        return 0.0
    distances = []
    for key in keys:
        if key not in left_flat or key not in right_flat:
            distances.append(1.0)
            continue
        left_number = _numeric(left_flat[key])
        right_number = _numeric(right_flat[key])
        if left_number is not None and right_number is not None:
            low, high = ranges.get(key, (left_number, right_number))
            distances.append(abs(left_number - right_number) / (high - low) if high > low else 0.0)
        else:
            distances.append(0.0 if left_flat[key] == right_flat[key] else 1.0)
    return sum(distances) / len(distances)


def _condition_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    field_ranges: dict[str, dict[str, tuple[float, float]]],
) -> float:
    distances = [0.0 if left["dtype"] == right["dtype"] else 1.0]
    for field in ("shapes", "workload", "params"):
        if left[field] or right[field]:
            distances.append(
                _mapping_distance(left[field], right[field], field_ranges[field])
            )
    return sum(distances) / len(distances)


def _balanced_condition_records(
    records: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[record["condition_fingerprint"]].append(record)
    representatives = [
        sorted(items, key=lambda item: item["case_id"])[0]
        for _, items in sorted(by_condition.items())
    ]
    if len(representatives) <= count:
        return representatives

    field_ranges = {
        field: _ranges(representatives, field)
        for field in ("shapes", "workload", "params")
    }
    distance_cache: dict[tuple[str, str], float] = {}

    def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        key = tuple(sorted((left["condition_fingerprint"], right["condition_fingerprint"])))
        if key not in distance_cache:
            distance_cache[key] = _condition_distance(left, right, field_ranges)
        return distance_cache[key]

    ordered = sorted(representatives, key=lambda item: item["condition_fingerprint"])
    first = ordered[0]
    best_average = -1.0
    for candidate in ordered:
        average = sum(distance(candidate, other) for other in ordered) / len(ordered)
        if average > best_average:
            first = candidate
            best_average = average

    selected = [first]
    remaining = [item for item in ordered if item is not first]
    while len(selected) < count:
        best = remaining[0]
        best_distance = -1.0
        for candidate in remaining:
            minimum = min(distance(candidate, chosen) for chosen in selected)
            if minimum > best_distance:
                best = candidate
                best_distance = minimum
        selected.append(best)
        remaining.remove(best)
    return selected


def _balanced_records_prefer_explicit(
    records: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """先覆盖完整 shapes+workload，再使用 proxy 或缺失元数据补位。"""

    selected: list[dict[str, Any]] = []
    selected_fingerprints: set[str] = set()
    for level in ("explicit", "params_proxy", "missing"):
        if len(selected) >= count:
            break
        candidates = [
            record
            for record in records
            if record["metadata_level"] == level
            and record["condition_fingerprint"] not in selected_fingerprints
        ]
        if not candidates:
            continue
        chosen = _balanced_condition_records(candidates, count - len(selected))
        selected.extend(chosen)
        selected_fingerprints.update(record["condition_fingerprint"] for record in chosen)
    return selected


def _selected_case(record: dict[str, Any], rank: int, reason: str) -> SelectedKernelCase:
    return SelectedKernelCase(
        case_id=record["case_id"],
        op_type=record["op_type"],
        dtype=record["dtype"],
        shapes=record["shapes"],
        workload=record["workload"],
        params=record["params"],
        shape_fingerprint=record["shape_fingerprint"],
        workload_fingerprint=record["workload_fingerprint"],
        condition_fingerprint=record["condition_fingerprint"],
        metadata_level=record["metadata_level"],
        metadata_issues=record["metadata_issues"],
        selection_rank=rank,
        selection_reason=reason,
    )


def _plan_id(payload: dict[str, Any]) -> str:
    identity = dict(payload)
    identity.pop("plan_id", None)
    return _fingerprint(identity)


def build_case_selection_plan(
    operator_cases_json: str | Path,
    *,
    backend: str,
    policy: Literal["condition-balanced", "all"] = CONDITION_BALANCED_POLICY,
    minimum_conditions_per_op_type: int = 5,
    target_op_types: tuple[str, ...] | list[str] | None = None,
) -> CaseSelectionPlan:
    """从平台无关 operator case manifest 构造确定性采样计划。"""

    if policy not in {CONDITION_BALANCED_POLICY, ALL_CASES_POLICY}:
        raise ValueError("unknown case selection policy: {}".format(policy))
    if minimum_conditions_per_op_type <= 0:
        raise ValueError("minimum_conditions_per_op_type must be positive")
    path = Path(operator_cases_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("operator case manifest must contain a cases array")

    records_by_op_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_case_ids: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict) or raw.get("backend") != backend:
            continue
        record = _case_record(raw)
        if record["case_id"] in seen_case_ids:
            raise ValueError("duplicate {} case {}".format(backend, record["case_id"]))
        seen_case_ids.add(record["case_id"])
        records_by_op_type[record["op_type"]].append(record)

    if target_op_types is None:
        targets = tuple(sorted(records_by_op_type))
    else:
        targets = tuple(sorted(set(str(item) for item in target_op_types if str(item))))
        missing = sorted(set(targets) - set(records_by_op_type))
        if missing:
            raise ValueError("target op_type has no {} cases: {}".format(backend, ",".join(missing)))
    if not targets:
        raise ValueError("no target op_type has eligible {} cases".format(backend))

    groups: dict[str, OpTypeCaseSelection] = {}
    selected_cases: list[SelectedKernelCase] = []
    issues: list[str] = []
    metadata_priority = {"explicit": 0, "params_proxy": 1, "missing": 2}
    for op_type in targets:
        records = sorted(
            records_by_op_type[op_type],
            key=lambda item: (
                metadata_priority[item["metadata_level"]],
                item["condition_fingerprint"],
                item["case_id"],
            ),
        )
        distinct_count = len({item["condition_fingerprint"] for item in records})
        if policy == ALL_CASES_POLICY:
            chosen = records
            status = "all_selected"
            reason = "all eligible cases requested"
        elif distinct_count < minimum_conditions_per_op_type:
            chosen = records
            status = "insufficient_unique_conditions"
            reason = "all cases selected because unique workload conditions are below target"
            issues.append(
                "op_type {} has {} distinct workload conditions; requested {} and selected all {} cases".format(
                    op_type,
                    distinct_count,
                    minimum_conditions_per_op_type,
                    len(records),
                )
            )
        else:
            chosen = _balanced_records_prefer_explicit(
                records, minimum_conditions_per_op_type
            )
            uses_proxy = any(record["metadata_level"] != "explicit" for record in chosen)
            status = "satisfied_with_proxy" if uses_proxy else "satisfied"
            reason = "selected by deterministic mixed-condition farthest-first coverage"
            if uses_proxy:
                issues.append(
                    (
                        "op_type {} needed proxy conditions because fewer than {} "
                        "complete shapes+workload conditions exist"
                    ).format(
                        op_type, minimum_conditions_per_op_type
                    )
                )

        non_explicit_count = sum(
            record["metadata_level"] != "explicit" for record in chosen
        )
        if non_explicit_count and status != "satisfied_with_proxy":
            issues.append(
                "op_type {} selected {} cases without complete shapes+workload metadata under status {}".format(
                    op_type, non_explicit_count, status
                )
            )

        selected = [
            _selected_case(
                record,
                rank,
                reason
                if record["metadata_level"] == "explicit"
                else "{}; {} metadata used".format(reason, record["metadata_level"]),
            )
            for rank, record in enumerate(chosen, 1)
        ]
        selected_cases.extend(selected)
        levels = tuple(sorted({record["metadata_level"] for record in records}))
        groups[op_type] = OpTypeCaseSelection(
            op_type=op_type,
            available_case_count=len(records),
            distinct_condition_count=distinct_count,
            requested_condition_count=minimum_conditions_per_op_type,
            selected_case_count=len(selected),
            status=status,
            metadata_levels=levels,
            selected_case_ids=tuple(case.case_id for case in selected),
        )

    plan_payload = {
        "schema_version": CASE_SELECTION_SCHEMA_VERSION,
        "plan_id": "",
        "source_manifest_sha256": _sha256_file(path),
        "backend": backend,
        "policy": policy,
        "policy_version": CASE_SELECTION_POLICY_VERSION,
        "minimum_conditions_per_op_type": minimum_conditions_per_op_type,
        "target_op_types": targets,
        "groups": groups,
        "selected_cases": tuple(selected_cases),
        "issues": tuple(issues),
    }
    serializable = {
        key: value.model_dump(mode="json") if isinstance(value, PmcModel) else value
        for key, value in plan_payload.items()
    }
    serializable["groups"] = {
        key: value.model_dump(mode="json") for key, value in groups.items()
    }
    serializable["selected_cases"] = [
        value.model_dump(mode="json") for value in selected_cases
    ]
    plan_payload["plan_id"] = _plan_id(serializable)
    return CaseSelectionPlan(**plan_payload)


def load_case_selection_plan(path: str | Path) -> CaseSelectionPlan:
    plan = CaseSelectionPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))
    payload = plan.model_dump(mode="json")
    expected = _plan_id(payload)
    if plan.plan_id != expected:
        raise ValueError(
            "case selection plan ID mismatch: expected {}, found {}".format(expected, plan.plan_id)
        )
    return plan


def write_case_selection_plan(plan: CaseSelectionPlan, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=str(destination.parent), delete=False, encoding="utf-8"
    ) as stream:
        stream.write(plan.model_dump_json(indent=2))
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, destination)


def validate_case_selection_plan_source(
    plan: CaseSelectionPlan, operator_cases_json: str | Path
) -> None:
    manifest_path = Path(operator_cases_json)
    actual = _sha256_file(manifest_path)
    if actual != plan.source_manifest_sha256:
        raise ValueError(
            "case selection plan manifest mismatch: expected {}, found {}".format(
                plan.source_manifest_sha256, actual
            )
        )
    rebuilt = build_case_selection_plan(
        manifest_path,
        backend=plan.backend,
        policy=plan.policy,
        minimum_conditions_per_op_type=plan.minimum_conditions_per_op_type,
        target_op_types=plan.target_op_types,
    )
    if rebuilt.model_dump(mode="json") != plan.model_dump(mode="json"):
        raise ValueError(
            "case selection plan no longer matches the current selection algorithm"
        )
