"""Build a PMC dataset from MNN kernel-replay rows and latency JSON."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..model import (
    DATASET_SCHEMA_VERSION,
    ComparisonPair,
    DatasetManifest,
    DeviceSpec,
    KernelCondition,
    LatencyObservation,
    MeasurementRun,
    MetricObservation,
    PmcDataset,
)
from ..catalog import KernelTaxonomy, MetricRegistry
from ..parsing import parse_number, parse_optional_int, semantic_equivalence_key
from .comparison_pairs import read_comparison_pairs_csv


def _load_metadata(path, backend):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for entry in payload.get("cases", []):
        if backend and entry.get("backend") != backend:
            continue
        name = entry.get("name")
        if not name:
            continue
        if name in result:
            raise ValueError("duplicate corpus case {}".format(name))
        result[name] = entry
    return result


def _load_latency(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for case, raw in payload.items():
        if isinstance(raw, dict):
            raw = raw.get("latency_us")
        value = parse_number(raw, allow_negative=False)
        if value is not None and value > 0:
            result[case] = value
    return result, set(payload)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_condition(case, entry, device_id, taxonomy):
    op_type = str(entry.get("op_type") or "unknown")
    variant = str(entry.get("variant") or case)
    classification = taxonomy.classify(op_type, variant)
    params = {}
    params.update(entry.get("int_params", {}))
    params.update(entry.get("float_params", {}))
    shapes = dict(entry.get("shapes", {}))
    workload = dict(entry.get("workload", {}))
    for key in ("algorithmic_flops", "algorithmic_bytes", "output_elements"):
        if key in entry:
            workload[key] = entry[key]
    launch = dict(entry.get("launch", {}))
    dtype = str(entry.get("dtype") or "unknown")
    validator = str(entry.get("validator") or "")
    tags = {
        "framework": str(entry.get("framework") or ""),
        "tag": str(entry.get("tag") or ""),
        "variant": variant,
        "semantic_family_source": classification["semantic_family_source"],
        "execution_role_source": classification["execution_role_source"],
        "semantic_family_confidence": classification["semantic_family_confidence"],
        "execution_role_confidence": classification["execution_role_confidence"],
    }
    return KernelCondition(
        condition_id=case,
        device_id=device_id,
        case_id=case,
        semantic_family=classification["semantic_family"],
        op_type=op_type,
        execution_role=classification["execution_role"],
        expected_mechanisms=tuple(classification["expected_mechanisms"]),
        performance_regime=str(entry.get("performance_regime") or "unknown"),
        implementation_id="{}:{}".format(tags["tag"], variant),
        semantic_equivalence_key=semantic_equivalence_key(op_type, dtype, validator, shapes, params),
        validator=validator,
        dtype=dtype,
        shapes=shapes,
        params=params,
        workload=workload,
        launch=launch,
        tags=tags,
    )


def _discover_pairs(conditions, case_order):
    groups = defaultdict(list)
    for case in case_order:
        condition = conditions[case]
        groups[condition.semantic_equivalence_key].append(case)
    comparisons = []
    for cases in groups.values():
        implementations = {conditions[case].implementation_id for case in cases}
        if len(cases) < 2 or len(implementations) < 2:
            continue
        baseline = cases[0]
        for index, candidate in enumerate(cases[1:], 1):
            comparisons.append(ComparisonPair(
                pair_id="auto:{}:{}".format(baseline, index),
                baseline_condition_id=baseline,
                candidate_condition_id=candidate,
                intervention_id="",
                controlled_mechanism="",
                design="auto_semantic_match",
                matched_fields=("op_type", "dtype", "validator", "shapes", "params"),
                environment_matched=None,
                randomized_order=None,
                notes="auto pair; verify semantic and environment equivalence",
            ))
    return tuple(comparisons)


def load_mnn_kernelreplay_dataset(
    pmc_csv,
    latency_json,
    operator_cases_json,
    platform="cuda",
    backend="cuda",
    namespace="cupti",
    collector="mnn-kernel-replay",
    dataset_id="mnn-kernel-replay",
    device=None,
    taxonomy=None,
    registry=None,
    pairs_csv=None,
    pmc_workload_runs=1,
    latency_workload_runs=1,
):
    taxonomy = taxonomy or KernelTaxonomy.load()
    registry = registry or MetricRegistry()
    device = device or DeviceSpec(device_id="{}:default".format(platform), vendor=platform)
    metadata = _load_metadata(operator_cases_json, backend)
    latencies, latency_keys = _load_latency(latency_json)
    rows = []
    case_order = []
    seen_cases = set()
    metric_names = set()
    metrics_by_case = defaultdict(set)
    duplicate_pair_count = 0
    status_counts = Counter()
    with Path(pmc_csv).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"case", "metric", "status", "value"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PMC CSV must contain {}".format(",".join(sorted(required))))
        for row in reader:
            case = row["case"].strip()
            metric = row["metric"].strip()
            if not case or not metric:
                continue
            if case not in seen_cases:
                seen_cases.add(case)
                case_order.append(case)
            if metric in metrics_by_case[case]:
                duplicate_pair_count += 1
            metrics_by_case[case].add(metric)
            metric_names.add(metric)
            status_counts[row["status"].strip()] += 1
            rows.append(row)

    issues = []
    manifest_cases = set(metadata)
    missing_latency_cases = sorted(manifest_cases - latency_keys)
    extra_latency_cases = sorted(latency_keys - manifest_cases)
    invalid_latency_cases = sorted(latency_keys - set(latencies))
    if missing_latency_cases or extra_latency_cases or invalid_latency_cases:
        issues.append(
            "latency/manifest mismatch: {} missing, {} extra, {} invalid".format(
                len(missing_latency_cases),
                len(extra_latency_cases),
                len(invalid_latency_cases),
            )
        )
    representative_by_op_type = {}
    for case, entry in metadata.items():
        op_type = str(entry.get("op_type") or "unknown")
        representative_by_op_type.setdefault(op_type, case)
    expected_representatives = set(representative_by_op_type.values())
    observed_row_cases = set(case_order)
    if observed_row_cases != expected_representatives:
        issues.append(
            "PMC rows/manifest representative mismatch: {} expected op-type representatives, {} observed"
            .format(len(expected_representatives), len(observed_row_cases))
        )
    if duplicate_pair_count:
        issues.append(
            "PMC rows contain {} duplicate (case, metric) records".format(
                duplicate_pair_count
            )
        )
    incomplete_cases = sorted(
        case
        for case in observed_row_cases
        if len(metrics_by_case[case]) != len(metric_names)
    )
    if incomplete_cases:
        issues.append(
            "PMC rows are not a complete case x metric grid for {} cases".format(
                len(incomplete_cases)
            )
        )
    conditions = {}
    for case in case_order:
        if case not in metadata:
            issues.append("condition {} has no operator metadata".format(case))
            continue
        conditions[case] = _build_condition(case, metadata[case], device.device_id, taxonomy)
        if case not in latencies:
            issues.append("condition {} has no latency observation".format(case))
    case_order = [case for case in case_order if case in conditions]

    metric_catalog = {}
    runs = {}
    metric_observations = []
    observation_index = 0
    for case in case_order:
        run_id = "pmc::{}".format(case)
        entry = metadata[case]
        runs[run_id] = MeasurementRun(
            run_id=run_id,
            condition_id=case,
            collection_kind="pmc",
            collector=collector,
            repeat_id="",
            collection_id="",
            paired_run_group_id="",
            warmup_runs=parse_optional_int(entry.get("warmup_runs")),
            workload_runs=pmc_workload_runs,
            order_index=None,
            gpu_clock_hz=None,
            temperature_c=None,
            cache_policy="unknown",
            source_ref=str(pmc_csv),
        )
    for row in rows:
        case = row["case"].strip()
        if case not in conditions:
            continue
        native_name = row["metric"].strip()
        metric_id = "{}::{}".format(namespace, native_name)
        if metric_id not in metric_catalog:
            metric_catalog[metric_id] = registry.describe(native_name, namespace, metric_id)
        status = row["status"].strip()
        value = parse_number(row["value"], allow_negative=True) if status == "VALID" else None
        flags = () if value is not None else ("invalid_or_missing_value",)
        observation_index += 1
        metric_observations.append(MetricObservation(
            observation_id="pmc-observation-{:09d}".format(observation_index),
            run_id="pmc::{}".format(case),
            metric_id=metric_id,
            value=value,
            value_semantics="absolute_workload",
            control_value=None,
            workload_value=value,
            status=status,
            profiler_pass_count=parse_optional_int(row.get("num_passes")),
            quality_flags=flags,
            raw_status=row.get("pmu_status", ""),
            error=row.get("error", ""),
        ))

    latency_observations = []
    for case in case_order:
        if case not in latencies:
            continue
        run_id = "latency::{}".format(case)
        entry = metadata[case]
        runs[run_id] = MeasurementRun(
            run_id=run_id,
            condition_id=case,
            collection_kind="latency",
            collector=collector,
            repeat_id="",
            collection_id="",
            paired_run_group_id="",
            warmup_runs=parse_optional_int(entry.get("warmup_runs")),
            workload_runs=latency_workload_runs,
            order_index=None,
            gpu_clock_hz=None,
            temperature_c=None,
            cache_policy="unknown",
            source_ref=str(latency_json),
        )
        latency_observations.append(LatencyObservation(
            observation_id="latency::{}".format(case),
            run_id=run_id,
            latency_us=latencies[case],
            status="VALID",
            quality_flags=(),
        ))

    if pairs_csv:
        comparisons, pair_issues = read_comparison_pairs_csv(
            pairs_csv, conditions
        )
        issues.extend(pair_issues)
    else:
        comparisons = _discover_pairs(conditions, case_order)
    if not any(run.repeat_id for run in runs.values()):
        issues.append("no independent repeat_id; repeatability cannot be estimated")
    issues.append(
        "legacy rows have no collection_session_id; per-case PMC run grouping is synthetic and must not be used "
        "to assume metrics were co-collected"
    )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        platform=platform,
        backend=backend,
        collector=collector,
        metadata={
            "pmc_csv": str(pmc_csv),
            "latency_json": str(latency_json),
            "operator_cases_json": str(operator_cases_json),
            "source_sha256": {
                "pmc_csv": _sha256(pmc_csv),
                "latency_json": _sha256(latency_json),
                "operator_cases_json": _sha256(operator_cases_json),
            },
            "source_consistency": {
                "manifest_case_count": len(manifest_cases),
                "latency_key_count": len(latency_keys),
                "valid_latency_count": len(latencies),
                "expected_representative_count": len(expected_representatives),
                "observed_pmc_case_count": len(observed_row_cases),
                "metric_count": len(metric_names),
                "duplicate_pair_count": duplicate_pair_count,
                "incomplete_case_count": len(incomplete_cases),
            },
            "status_counts": dict(status_counts),
            "profiler_pass_note": "pass count belongs to the collection session, not an independent repeat",
        },
    )
    return PmcDataset(
        schema_version=DATASET_SCHEMA_VERSION,
        manifest=manifest,
        devices={device.device_id: device},
        conditions=conditions,
        runs=runs,
        metric_catalog=metric_catalog,
        metric_observations=tuple(metric_observations),
        latency_observations=tuple(latency_observations),
        comparisons=comparisons,
        issues=tuple(issues),
    )
