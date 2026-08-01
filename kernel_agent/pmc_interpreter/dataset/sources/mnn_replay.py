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
    EnvironmentSample,
    KernelCondition,
    KernelLaunchRecord,
    LatencyObservation,
    MeasurementRun,
    MetricObservation,
    PmcDataset,
    SemanticFactProvenance,
)
from ..catalog import KernelTaxonomy, MetricRegistry
from ..parsing import parse_number, parse_optional_int, semantic_equivalence_key
from ..selection import load_case_selection_plan, validate_case_selection_plan_source
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
    source = {
        "structured": False,
        "format": "legacy-flat",
        "version": 0,
        "source_manifest_sha256": "",
    }
    if isinstance(payload, dict) and "cases" in payload:
        expected_fields = {
            "format",
            "version",
            "source_manifest_sha256",
            "cases",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "structured latency JSON fields must be exactly {}".format(
                    ",".join(sorted(expected_fields))
                )
            )
        if payload.get("format") != "mnn-kernel-latency":
            raise ValueError("structured latency JSON has unsupported format")
        if type(payload.get("version")) is not int or payload["version"] != 1:
            raise ValueError("structured latency JSON has unsupported version")
        source_manifest_sha256 = payload.get("source_manifest_sha256")
        if (
            not isinstance(source_manifest_sha256, str)
            or len(source_manifest_sha256) != 64
            or source_manifest_sha256 != source_manifest_sha256.lower()
            or any(character not in "0123456789abcdef" for character in source_manifest_sha256)
        ):
            raise ValueError(
                "structured latency JSON requires a lowercase SHA256 source_manifest_sha256"
            )
        if not isinstance(payload["cases"], dict):
            raise ValueError("structured latency JSON cases must be an object")
        source = {
            "structured": True,
            "format": payload["format"],
            "version": payload["version"],
            "source_manifest_sha256": source_manifest_sha256,
        }
        payload = payload["cases"]
    if not isinstance(payload, dict):
        raise ValueError("latency JSON must be an object keyed by case name")
    result = {}
    for case, raw in payload.items():
        record = raw if isinstance(raw, dict) else {"latency_us": raw}
        if isinstance(raw, dict):
            raw = raw.get("latency_us")
        value = parse_number(raw, allow_negative=False)
        if value is not None and value > 0:
            result[case] = {**record, "latency_us": value}
    return result, set(payload), source


def _canonical_launch_sampling_status(raw_status):
    status = str(raw_status or "").strip()
    if status == "sampled":
        return "sampled"
    if status in {"", "not_requested", "not_collected"}:
        return "not_collected"
    if status == "no_launch":
        return "no_launch"
    if (
        status in {"partial", "error"}
        or status.startswith("partial:")
        or status.startswith("error:")
        or status.startswith("metadata launch failed:")
        or status.startswith("unavailable:metadata launch failed:")
    ):
        return "error"
    if status == "unavailable" or status.startswith("unavailable:"):
        return "unavailable"
    return "error"


def _load_fact_provenance(entry, field):
    raw = entry.get(field, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("case {} field {} must be an object".format(entry.get("name", ""), field))
    result = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(
                "case {} provenance {}.{} must be an object".format(
                    entry.get("name", ""), field, key
                )
            )
        result[str(key)] = SemanticFactProvenance.model_validate(value)
    return result


def _vector3(raw, field, case):
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("latency case {} {} must contain three integers".format(case, field))
    values = tuple(int(value) for value in raw)
    if any(value <= 0 for value in values):
        raise ValueError("latency case {} {} dimensions must be positive".format(case, field))
    return values


def _parse_launch_records(case, record):
    raw_records = record.get("launch_records", [])
    if raw_records is None:
        return ()
    if not isinstance(raw_records, list):
        raise ValueError("latency case {} launch_records must be an array".format(case))
    records = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError("latency case {} launch record {} must be an object".format(case, index))
        native = raw.get("native", {})
        if not isinstance(native, dict):
            raise ValueError("latency case {} launch record native must be an object".format(case))
        records.append(KernelLaunchRecord(
            stage_index=parse_optional_int(raw.get("stage_index")) if raw.get("stage_index") is not None else index,
            repeat_index=parse_optional_int(raw.get("repeat_index")),
            kernel_name=str(raw.get("kernel_name") or ""),
            grid=_vector3(raw.get("grid"), "grid", case),
            block=_vector3(raw.get("block"), "block", case),
            registers_per_thread=parse_optional_int(raw.get("registers_per_thread")),
            static_shared_memory_bytes=parse_optional_int(raw.get("static_shared_memory_bytes")),
            dynamic_shared_memory_bytes=parse_optional_int(raw.get("dynamic_shared_memory_bytes")),
            local_memory_per_thread_bytes=parse_optional_int(raw.get("local_memory_per_thread_bytes")),
            local_memory_total_bytes=parse_optional_int(raw.get("local_memory_total_bytes")),
            native=native,
        ))
    return tuple(records)


def _parse_environment(record):
    raw = record.get("environment", {})
    if not isinstance(raw, dict):
        return (), None, None
    source = str(raw.get("sampling_source") or raw.get("source") or "unknown")
    status = str(raw.get("sampling_status") or "unknown")
    samples = []
    for phase in ("before", "after"):
        clock = parse_number(raw.get("gpu_clock_hz_{}".format(phase)), allow_negative=False)
        temperature = parse_number(raw.get("temperature_c_{}".format(phase)), allow_negative=True)
        if clock is None and temperature is None:
            continue
        samples.append(EnvironmentSample(
            phase=phase,
            gpu_clock_hz=clock if clock and clock > 0 else None,
            temperature_c=temperature,
            source=source,
            status=status,
        ))
    clocks = [sample.gpu_clock_hz for sample in samples if sample.gpu_clock_hz is not None]
    temperatures = [sample.temperature_c for sample in samples if sample.temperature_c is not None]
    clock = sum(clocks) / len(clocks) if clocks else None
    temperature = sum(temperatures) / len(temperatures) if temperatures else None
    return tuple(samples), clock, temperature


def _single_row_value(rows, key):
    values = {row.get(key, "").strip() for row in rows if row.get(key, "").strip()}
    if len(values) > 1:
        raise ValueError(
            "PMC collection session has inconsistent {} values: {}".format(
                key, ",".join(sorted(values))
            )
        )
    return next(iter(values), "")


def _row_environment(rows):
    source = _single_row_value(rows, "environment_source") or "unknown"
    status = _single_row_value(rows, "environment_status") or "unknown"
    samples = []
    for phase in ("before", "after"):
        clock = parse_number(
            _single_row_value(rows, "gpu_clock_hz_{}".format(phase)),
            allow_negative=False,
        )
        temperature = parse_number(
            _single_row_value(rows, "temperature_c_{}".format(phase)),
            allow_negative=True,
        )
        if clock is None and temperature is None:
            continue
        samples.append(EnvironmentSample(
            phase=phase,
            gpu_clock_hz=clock if clock and clock > 0 else None,
            temperature_c=temperature,
            source=source,
            status=status,
        ))
    clocks = [sample.gpu_clock_hz for sample in samples if sample.gpu_clock_hz is not None]
    temperatures = [sample.temperature_c for sample in samples if sample.temperature_c is not None]
    return (
        tuple(samples),
        sum(clocks) / len(clocks) if clocks else None,
        sum(temperatures) / len(temperatures) if temperatures else None,
    )


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
        shape_provenance=_load_fact_provenance(entry, "shape_provenance"),
        workload_provenance=_load_fact_provenance(entry, "workload_provenance"),
        launch_provenance=_load_fact_provenance(entry, "launch_provenance"),
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
    case_selection_plan=None,
    pmc_workload_runs=1,
    latency_workload_runs=1,
):
    taxonomy = taxonomy or KernelTaxonomy.load()
    registry = registry or MetricRegistry()
    device = device or DeviceSpec(device_id="{}:default".format(platform), vendor=platform)
    metadata = _load_metadata(operator_cases_json, backend)
    latencies, latency_keys, latency_source = _load_latency(latency_json)
    operator_cases_sha256 = _sha256(operator_cases_json)
    if (
        latency_source["structured"]
        and latency_source["source_manifest_sha256"] != operator_cases_sha256
    ):
        raise ValueError(
            "latency source manifest SHA256 does not match operator_cases.json"
        )
    selection_plan = None
    if case_selection_plan:
        selection_plan = load_case_selection_plan(case_selection_plan)
        validate_case_selection_plan_source(selection_plan, operator_cases_json)
        if selection_plan.backend != backend:
            raise ValueError(
                "case selection plan backend {} does not match dataset backend {}".format(
                    selection_plan.backend, backend
                )
            )
    rows = []
    case_order = []
    seen_cases = set()
    metric_names = set()
    metrics_by_case = defaultdict(set)
    sweep_plan_ids = set()
    sweep_config_ids = set()
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
            if row.get("sweep_plan_id", "").strip():
                sweep_plan_ids.add(row["sweep_plan_id"].strip())
            if row.get("sweep_config_id", "").strip():
                sweep_config_ids.add(row["sweep_config_id"].strip())
            rows.append(row)

    issues = []
    manifest_cases = set(metadata)
    missing_latency_cases = sorted(manifest_cases - latency_keys)
    extra_latency_cases = sorted(latency_keys - manifest_cases)
    invalid_latency_cases = sorted(latency_keys - set(latencies))
    if missing_latency_cases or extra_latency_cases or invalid_latency_cases:
        message = "latency/manifest mismatch: {} missing, {} extra, {} invalid".format(
            len(missing_latency_cases),
            len(extra_latency_cases),
            len(invalid_latency_cases),
        )
        if latency_source["structured"]:
            raise ValueError(message)
        issues.append(message)
    if not latency_source["structured"]:
        issues.append(
            "legacy flat latency JSON has no canonical format/version/source_manifest_sha256; "
            "it is exploratory input, not a formal source-consistency gate"
        )
    observed_row_cases = set(case_order)
    expected_selected_cases = (
        {case.case_id for case in selection_plan.selected_cases}
        if selection_plan is not None
        else set()
    )
    if selection_plan is not None and observed_row_cases != expected_selected_cases:
        issues.append(
            "PMC rows/case selection plan mismatch: {} expected selected cases, {} observed"
            .format(len(expected_selected_cases), len(observed_row_cases))
        )
    elif selection_plan is None:
        issues.append(
            "no case selection plan; PMC case coverage cannot be verified against the collection policy"
        )
    if len(sweep_plan_ids) > 1:
        issues.append("PMC rows contain multiple sweep_plan_id values")
    if len(sweep_config_ids) > 1:
        issues.append("PMC rows contain multiple sweep_config_id values")
    if selection_plan is not None and sweep_plan_ids and selection_plan.plan_id not in sweep_plan_ids:
        issues.append("PMC rows sweep_plan_id does not match the supplied case selection plan")
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
    pmc_groups = defaultdict(list)
    for row in rows:
        case = row["case"].strip()
        if case not in conditions:
            continue
        session_id = row.get("collection_session_id", "").strip()
        group_key = (case, session_id or "legacy")
        pmc_groups[group_key].append(row)
    pmc_run_ids = {}
    for (case, session_id), session_rows in sorted(pmc_groups.items()):
        run_id = (
            "pmc-session::{}".format(session_id)
            if session_id != "legacy"
            else "pmc::{}".format(case)
        )
        if run_id in runs:
            raise ValueError("duplicate PMC collection session {}".format(run_id))
        entry = metadata[case]
        environment_samples, gpu_clock_hz, temperature_c = _row_environment(session_rows)
        repeat_id = _single_row_value(session_rows, "repeat_id")
        paired_run_group_id = _single_row_value(session_rows, "paired_run_group_id")
        order_index = parse_optional_int(_single_row_value(session_rows, "order_index"))
        runs[run_id] = MeasurementRun(
            run_id=run_id,
            condition_id=case,
            collection_kind="pmc",
            collector=collector,
            repeat_id=repeat_id,
            collection_id="" if session_id == "legacy" else session_id,
            paired_run_group_id=paired_run_group_id,
            warmup_runs=parse_optional_int(entry.get("warmup_runs")),
            workload_runs=pmc_workload_runs,
            order_index=order_index,
            gpu_clock_hz=gpu_clock_hz,
            temperature_c=temperature_c,
            cache_policy="unknown",
            source_ref=str(pmc_csv),
            environment_samples=environment_samples,
        )
        pmc_run_ids[(case, session_id)] = run_id
    for row in rows:
        case = row["case"].strip()
        if case not in conditions:
            continue
        session_id = row.get("collection_session_id", "").strip() or "legacy"
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
            run_id=pmc_run_ids[(case, session_id)],
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
        latency_record = latencies[case]
        launch_records = _parse_launch_records(case, latency_record)
        launch_sampling_status = _canonical_launch_sampling_status(
            latency_record.get("launch_sampling_status")
        )
        environment_samples, gpu_clock_hz, temperature_c = _parse_environment(
            latency_record
        )
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
            gpu_clock_hz=gpu_clock_hz,
            temperature_c=temperature_c,
            cache_policy="unknown",
            source_ref=str(latency_json),
            launch_sampling_status=launch_sampling_status,
            launch_records=launch_records,
            environment_samples=environment_samples,
        )
        latency_observations.append(LatencyObservation(
            observation_id="latency::{}".format(case),
            run_id=run_id,
            latency_us=latency_record["latency_us"],
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
    if any(not row.get("collection_session_id", "").strip() for row in rows):
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
            "case_selection_plan": str(case_selection_plan or ""),
            "source_sha256": {
                "pmc_csv": _sha256(pmc_csv),
                "latency_json": _sha256(latency_json),
                "operator_cases_json": _sha256(operator_cases_json),
                **(
                    {"case_selection_plan": _sha256(case_selection_plan)}
                    if case_selection_plan
                    else {}
                ),
            },
            "source_consistency": {
                "manifest_case_count": len(manifest_cases),
                "latency_key_count": len(latency_keys),
                "valid_latency_count": len(latencies),
                "expected_selected_case_count": len(expected_selected_cases),
                "observed_pmc_case_count": len(observed_row_cases),
                "metric_count": len(metric_names),
                "duplicate_pair_count": duplicate_pair_count,
                "incomplete_case_count": len(incomplete_cases),
                "sweep_plan_ids": sorted(sweep_plan_ids),
                "sweep_config_ids": sorted(sweep_config_ids),
                "latency_format": latency_source["format"],
                "latency_version": latency_source["version"],
                "latency_source_manifest_sha256": latency_source[
                    "source_manifest_sha256"
                ],
                "latency_source_manifest_matches": (
                    latency_source["structured"]
                    and latency_source["source_manifest_sha256"]
                    == operator_cases_sha256
                ),
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
