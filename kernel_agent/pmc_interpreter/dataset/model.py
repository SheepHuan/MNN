"""PMC Interpreter 的平台无关实验事实模型。"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


DATASET_SCHEMA_VERSION = "mnn-pmc-dataset/v1"


class PmcModel(BaseModel):
    """PMC Interpreter 所有持久化结构的共同严格基类。"""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
    )

class DatasetManifest(PmcModel):
    dataset_id: str
    platform: str
    backend: str
    collector: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class DeviceSpec(PmcModel):
    device_id: str
    vendor: str = "unknown"
    architecture: str = "unknown"
    model: str = "unknown"
    driver: str = "unknown"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SemanticFactProvenance(PmcModel):
    """shape/workload 声明的来源，避免把估算值解释成设备实测值。"""

    status: str
    source: str
    method: str
    formula: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str = ""


class KernelCondition(PmcModel):
    condition_id: str
    device_id: str
    case_id: str
    semantic_family: str
    op_type: str
    execution_role: str
    expected_mechanisms: tuple[str, ...]
    performance_regime: str
    implementation_id: str
    semantic_equivalence_key: str
    validator: str
    dtype: str
    shapes: dict[str, JsonValue] = Field(default_factory=dict)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    workload: dict[str, JsonValue] = Field(default_factory=dict)
    launch: dict[str, JsonValue] = Field(default_factory=dict)
    shape_provenance: dict[str, SemanticFactProvenance] = Field(default_factory=dict)
    workload_provenance: dict[str, SemanticFactProvenance] = Field(default_factory=dict)
    launch_provenance: dict[str, SemanticFactProvenance] = Field(default_factory=dict)
    tags: dict[str, JsonValue] = Field(default_factory=dict)


class KernelLaunchRecord(PmcModel):
    """一次真实 device-kernel launch 的平台无关资源记录。"""

    stage_index: int = Field(ge=0)
    repeat_index: int | None = Field(default=None, ge=0)
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    registers_per_thread: int | None = Field(default=None, ge=0)
    static_shared_memory_bytes: int | None = Field(default=None, ge=0)
    dynamic_shared_memory_bytes: int | None = Field(default=None, ge=0)
    local_memory_per_thread_bytes: int | None = Field(default=None, ge=0)
    local_memory_total_bytes: int | None = Field(default=None, ge=0)
    native: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        if any(value <= 0 for value in self.grid):
            raise ValueError("kernel launch grid dimensions must be positive")
        if any(value <= 0 for value in self.block):
            raise ValueError("kernel launch block dimensions must be positive")
        return self


class EnvironmentSample(PmcModel):
    """采集区间外读取的设备环境快照，不宣称是逐 kernel 精确状态。"""

    phase: str
    gpu_clock_hz: float | None = Field(default=None, gt=0.0)
    temperature_c: float | None
    source: str
    status: str


class MeasurementRun(PmcModel):
    run_id: str
    condition_id: str
    collection_kind: str
    collector: str
    repeat_id: str
    collection_id: str
    paired_run_group_id: str
    warmup_runs: int | None
    workload_runs: int | None
    order_index: int | None
    gpu_clock_hz: float | None
    temperature_c: float | None
    cache_policy: str
    source_ref: str
    launch_sampling_status: Literal[
        "sampled",
        "no_launch",
        "unavailable",
        "error",
        "not_collected",
    ] = "not_collected"
    launch_records: tuple[KernelLaunchRecord, ...] = ()
    environment_samples: tuple[EnvironmentSample, ...] = ()

    @model_validator(mode="after")
    def validate_launch_sampling(self) -> Self:
        if self.launch_sampling_status == "sampled" and not self.launch_records:
            raise ValueError("sampled launch collection must contain launch records")
        if (
            self.launch_sampling_status
            in {"no_launch", "unavailable", "not_collected"}
            and self.launch_records
        ):
            raise ValueError(
                "{} launch collection must not contain launch records".format(
                    self.launch_sampling_status
                )
            )
        return self


class MetricDescriptor(PmcModel):
    metric_id: str
    namespace: str
    native_name: str
    native_domain: str
    basename: str
    concept_id: str
    hardware_scope: str
    mechanism: str
    phenomenon_role: str
    quantity_kind: str
    unit: str
    aggregation: str
    normalizer: str
    bounded_range: tuple[float, float] | None
    duration_coupled: bool
    target_equivalent: bool
    dependency_metric_ids: tuple[str, ...]
    collection_cost: float | None
    analysis_scope: str
    mapping_level: str
    mapping_confidence: float = Field(ge=0.0, le=1.0)
    mapping_source: str


class MetricObservation(PmcModel):
    observation_id: str
    run_id: str
    metric_id: str
    value: float | None
    value_semantics: str
    control_value: float | None
    workload_value: float | None
    status: str
    profiler_pass_count: int | None
    quality_flags: tuple[str, ...]
    raw_status: str
    error: str

    @model_validator(mode="after")
    def validate_status_value(self) -> Self:
        if self.status == "VALID" and self.value is None:
            raise ValueError("valid metric observation must contain a finite value")
        return self


class LatencyObservation(PmcModel):
    observation_id: str
    run_id: str
    latency_us: float | None
    status: str
    quality_flags: tuple[str, ...]


class ComparisonPair(PmcModel):
    pair_id: str
    baseline_condition_id: str
    candidate_condition_id: str
    intervention_id: str
    controlled_mechanism: str
    design: str
    matched_fields: tuple[str, ...]
    environment_matched: bool | None
    randomized_order: bool | None
    notes: str


class PmcDataset(PmcModel):
    schema_version: str = DATASET_SCHEMA_VERSION
    manifest: DatasetManifest
    devices: dict[str, DeviceSpec]
    conditions: dict[str, KernelCondition]
    runs: dict[str, MeasurementRun]
    metric_catalog: dict[str, MetricDescriptor]
    metric_observations: tuple[MetricObservation, ...]
    latency_observations: tuple[LatencyObservation, ...]
    comparisons: tuple[ComparisonPair, ...]
    issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        errors: list[str] = []
        if self.schema_version != DATASET_SCHEMA_VERSION:
            errors.append("unsupported dataset schema {}".format(self.schema_version))
        for device_id, device in self.devices.items():
            if device.device_id != device_id:
                errors.append("device key/id mismatch: {}".format(device_id))
        for condition_id, condition in self.conditions.items():
            if condition.condition_id != condition_id:
                errors.append("condition key/id mismatch: {}".format(condition_id))
            if condition.device_id not in self.devices:
                errors.append(
                    "condition {} references missing device {}".format(
                        condition_id, condition.device_id
                    )
                )
        for run_id, run in self.runs.items():
            if run.run_id != run_id:
                errors.append("run key/id mismatch: {}".format(run_id))
            if run.condition_id not in self.conditions:
                errors.append(
                    "run {} references missing condition {}".format(
                        run_id, run.condition_id
                    )
                )
        for metric_id, descriptor in self.metric_catalog.items():
            if descriptor.metric_id != metric_id:
                errors.append("metric key/id mismatch: {}".format(metric_id))
        metric_observation_ids: set[str] = set()
        for observation in self.metric_observations:
            if observation.observation_id in metric_observation_ids:
                errors.append(
                    "duplicate metric observation {}".format(observation.observation_id)
                )
            metric_observation_ids.add(observation.observation_id)
            if observation.run_id not in self.runs:
                errors.append(
                    "metric observation {} references missing run".format(
                        observation.observation_id
                    )
                )
            if observation.metric_id not in self.metric_catalog:
                errors.append(
                    "metric observation {} references missing metric".format(
                        observation.observation_id
                    )
                )
            if (
                observation.run_id in self.runs
                and self.runs[observation.run_id].collection_kind != "pmc"
            ):
                errors.append(
                    "metric observation {} references a non-PMC run".format(
                        observation.observation_id
                    )
                )
        latency_observation_ids: set[str] = set()
        for observation in self.latency_observations:
            if observation.observation_id in latency_observation_ids:
                errors.append(
                    "duplicate latency observation {}".format(observation.observation_id)
                )
            latency_observation_ids.add(observation.observation_id)
            if observation.run_id not in self.runs:
                errors.append(
                    "latency observation {} references missing run".format(
                        observation.observation_id
                    )
                )
            elif self.runs[observation.run_id].collection_kind != "latency":
                errors.append(
                    "latency observation {} references a non-latency run".format(
                        observation.observation_id
                    )
                )
            if (
                observation.status == "VALID"
                and (observation.latency_us is None or observation.latency_us <= 0)
            ):
                errors.append(
                    "valid latency observation {} must be positive".format(
                        observation.observation_id
                    )
                )
        pair_ids: set[str] = set()
        for pair in self.comparisons:
            if pair.pair_id in pair_ids:
                errors.append("duplicate comparison pair {}".format(pair.pair_id))
            pair_ids.add(pair.pair_id)
            if pair.baseline_condition_id not in self.conditions:
                errors.append("pair {} baseline is missing".format(pair.pair_id))
            if pair.candidate_condition_id not in self.conditions:
                errors.append("pair {} candidate is missing".format(pair.pair_id))
            if pair.baseline_condition_id == pair.candidate_condition_id:
                errors.append("pair {} compares a condition with itself".format(pair.pair_id))
            if (
                pair.baseline_condition_id in self.conditions
                and pair.candidate_condition_id in self.conditions
                and self.conditions[pair.baseline_condition_id].implementation_id
                == self.conditions[pair.candidate_condition_id].implementation_id
            ):
                errors.append(
                    "pair {} does not change implementation_id".format(pair.pair_id)
                )
        if errors:
            raise ValueError("invalid PMC dataset: {}".format("; ".join(errors)))
        return self
