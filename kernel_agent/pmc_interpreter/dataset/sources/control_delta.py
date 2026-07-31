"""Load control-adjusted Adreno/Mali/OpenCL/Vulkan PMC observations."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

from ..catalog import MetricRegistry
from ..model import MeasurementRun, MetricObservation, PmcDataset
from ..parsing import parse_number, parse_optional_int


def append_control_delta_csv(
    dataset,
    csv_path,
    namespace,
    collector="control-delta-csv",
    registry=None,
):
    registry = registry or MetricRegistry()
    source_digest = hashlib.sha256()
    source_digest.update(namespace.encode("utf-8"))
    source_digest.update(b"\0")
    source_digest.update(Path(csv_path).read_bytes())
    source_token = source_digest.hexdigest()[:16]
    existing_observation_ids = {
        observation.observation_id for observation in dataset.metric_observations
    }
    condition_by_case = {condition.case_id: condition_id for condition_id, condition in dataset.conditions.items()}
    runs = dict(dataset.runs)
    metric_catalog = dict(dataset.metric_catalog)
    observations = list(dataset.metric_observations)
    issues = list(dataset.issues)
    occurrence = Counter()
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"case_name", "pmu_metric_name", "delta_metric"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("control-delta CSV must contain {}".format(",".join(sorted(required))))
        for row_index, row in enumerate(reader, 1):
            case_name = row["case_name"].strip()
            native_name = row["pmu_metric_name"].strip()
            if case_name not in condition_by_case:
                issues.append("control-delta row {} has unknown case {}".format(row_index, case_name))
                continue
            condition_id = condition_by_case[case_name]
            occurrence[(condition_id, native_name)] += 1
            repeat_index = occurrence[(condition_id, native_name)]
            run_id = "pmc-delta::{}::{}::{}::{}".format(
                source_token, condition_id, native_name, repeat_index
            )
            observation_id = "control-delta::{}::{:09d}".format(
                source_token, row_index
            )
            if observation_id in existing_observation_ids:
                raise ValueError(
                    "control-delta source has already been appended: {}".format(
                        csv_path
                    )
                )
            runs[run_id] = MeasurementRun(
                run_id=run_id,
                condition_id=condition_id,
                collection_kind="pmc",
                collector=collector,
                repeat_id="{}::measurement_{:04d}".format(
                    source_token, repeat_index
                ),
                collection_id="",
                paired_run_group_id="",
                warmup_runs=None,
                workload_runs=parse_optional_int(row.get("case_runs")),
                order_index=row_index,
                gpu_clock_hz=None,
                temperature_c=None,
                cache_policy="unknown",
                source_ref=row.get("case_args", "") or str(csv_path),
            )
            metric_id = "{}::{}".format(namespace, native_name)
            if metric_id not in metric_catalog:
                metric_catalog[metric_id] = registry.describe(native_name, namespace, metric_id)
            value = parse_number(row.get("delta_metric"), allow_negative=True)
            observations.append(MetricObservation(
                observation_id=observation_id,
                run_id=run_id,
                metric_id=metric_id,
                value=value,
                value_semantics="workload_minus_control",
                control_value=parse_number(row.get("control_value"), allow_negative=True),
                workload_value=parse_number(row.get("workload_value"), allow_negative=True),
                status="VALID" if value is not None else "INVALID",
                profiler_pass_count=None,
                quality_flags=() if value is not None else ("invalid_or_missing_value",),
                raw_status=row.get("status", ""),
                error=row.get("error", ""),
            ))
    payload = dataset.model_dump(mode="python")
    payload.update({
        "runs": runs,
        "metric_catalog": metric_catalog,
        "metric_observations": tuple(observations),
        "issues": tuple(issues),
    })
    return PmcDataset.model_validate(payload)
