"""严格串行分析层基类、Projection 层与共享投影工具。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..dataset.reports import (
    AnalysisContext,
    DataReadiness,
    FeatureSpec,
    FeatureStore,
    LayerRecord,
    ProjectionInput,
    ProjectionOutput,
    ProjectionReport,
    QuestionReadiness,
)
from .statistics import (
    choose_transform,
    choose_transform_scale,
    median,
    transform_vector,
)


class AnalysisLayer:
    """强制每层只接收直接上一层的具体输出类型。"""

    name = "base"
    input_type = object

    def require_input(self, value):
        if type(value) is not self.input_type:
            raise TypeError(
                "{} expects {}, got {}".format(
                    type(self).__name__,
                    self.input_type.__name__,
                    type(value).__name__,
                )
            )


def append_layer_record(output, layer, output_feature_ids, notes=()):
    return output.layer_history + (
        LayerRecord(
            layer=layer,
            input_feature_count=len(output.active_feature_ids),
            output_feature_count=len(output_feature_ids),
            notes=tuple(notes),
        ),
    )


def latency_values_by_condition(dataset):
    values = defaultdict(list)
    for observation in dataset.latency_observations:
        if (
            observation.status != "VALID"
            or observation.latency_us is None
            or observation.latency_us <= 0
        ):
            continue
        run = dataset.runs[observation.run_id]
        values[run.condition_id].append(float(observation.latency_us))
    return {condition_id: tuple(items) for condition_id, items in values.items()}


def log_latency_map(dataset):
    return {
        condition_id: math.log(median(values))
        for condition_id, values in latency_values_by_condition(dataset).items()
    }


def _valid_pair_ids(dataset, latency):
    valid = []
    groups = set()
    for pair in dataset.comparisons:
        baseline = dataset.conditions.get(pair.baseline_condition_id)
        candidate = dataset.conditions.get(pair.candidate_condition_id)
        if baseline is None or candidate is None:
            continue
        if baseline.device_id != candidate.device_id:
            continue
        if baseline.semantic_equivalence_key != candidate.semantic_equivalence_key:
            continue
        if baseline.condition_id not in latency or candidate.condition_id not in latency:
            continue
        valid.append(pair.pair_id)
        groups.add(baseline.semantic_equivalence_key)
    return tuple(valid), tuple(sorted(groups))


def build_data_readiness(dataset, config):
    latency = latency_values_by_condition(dataset)
    condition_count = len(dataset.conditions)
    used_device_ids = {
        condition.device_id for condition in dataset.conditions.values()
    }
    op_counts = Counter(condition.op_type for condition in dataset.conditions.values())
    family_counts = Counter(
        condition.semantic_family for condition in dataset.conditions.values()
    )
    role_counts = Counter(
        condition.execution_role for condition in dataset.conditions.values()
    )

    workload_presence = Counter()
    launch_presence = Counter()
    for condition in dataset.conditions.values():
        workload_presence.update(condition.workload.keys())
        launch_presence.update(condition.launch.keys())
    workload_coverage = {
        key: count / condition_count if condition_count else 0.0
        for key, count in sorted(workload_presence.items())
    }
    launch_coverage = {
        key: count / condition_count if condition_count else 0.0
        for key, count in sorted(launch_presence.items())
    }

    metric_repeats = defaultdict(set)
    for observation in dataset.metric_observations:
        if observation.status != "VALID" or observation.value is None:
            continue
        run = dataset.runs[observation.run_id]
        if run.collection_kind == "pmc" and run.repeat_id:
            metric_repeats[(run.condition_id, observation.metric_id)].add(run.repeat_id)
    pmc_repeat_conditions = {
        condition_id
        for (condition_id, _), repeat_ids in metric_repeats.items()
        if len(repeat_ids) >= 2
    }

    latency_repeats = defaultdict(set)
    for observation in dataset.latency_observations:
        if observation.status != "VALID" or observation.latency_us is None:
            continue
        run = dataset.runs[observation.run_id]
        if run.repeat_id:
            latency_repeats[run.condition_id].add(run.repeat_id)
    latency_repeat_conditions = {
        condition_id
        for condition_id, repeat_ids in latency_repeats.items()
        if len(repeat_ids) >= 2
    }

    clock_conditions = {
        run.condition_id for run in dataset.runs.values() if run.gpu_clock_hz is not None
    }
    temperature_conditions = {
        run.condition_id for run in dataset.runs.values() if run.temperature_c is not None
    }
    clock_coverage = len(clock_conditions) / condition_count if condition_count else 0.0
    temperature_coverage = (
        len(temperature_conditions) / condition_count if condition_count else 0.0
    )

    missing_controls = []
    for key in ("algorithmic_flops", "algorithmic_bytes", "output_elements"):
        if workload_coverage.get(key, 0.0) < config.min_coverage:
            missing_controls.append(key)
    if not launch_coverage or max(launch_coverage.values(), default=0.0) < config.min_coverage:
        missing_controls.append("launch_geometry")
    resource_tokens = ("register", "shared", "occupancy", "local_memory")
    if not any(any(token in key.lower() for token in resource_tokens) for key in launch_coverage):
        missing_controls.append("resource_usage")
    if clock_coverage < config.min_coverage:
        missing_controls.append("gpu_clock")
    if temperature_coverage < config.min_coverage:
        missing_controls.append("temperature")
    if not pmc_repeat_conditions:
        missing_controls.append("pmc_repeat_id")
    if not latency_repeat_conditions:
        missing_controls.append("latency_repeats")

    valid_pair_ids, valid_pair_groups = _valid_pair_ids(dataset, latency)
    source_incomplete = any(
        "mismatch" in issue or "not a complete" in issue for issue in dataset.issues
    )
    if len(latency) < 3:
        correlation_readiness = "not_estimable_without_latency"
    elif source_incomplete:
        correlation_readiness = "exploratory_incomplete_sources"
    else:
        correlation_readiness = "exploratory"
    has_valid_pmc = any(
        observation.status == "VALID" and observation.value is not None
        for observation in dataset.metric_observations
    )
    if not latency or not has_valid_pmc:
        family_readiness = "not_estimable_without_pmc_and_latency"
    elif not any(count >= config.min_group_samples for count in family_counts.values()):
        family_readiness = "insufficient_group_samples"
    elif source_incomplete:
        family_readiness = "family_level_exploratory_incomplete_sources"
    else:
        family_readiness = "family_level_exploratory"
    delta_readiness = (
        "paired_analysis_available"
        if len(valid_pair_groups) >= config.min_delta_pairs
        else "insufficient_paired_variants"
    )
    return DataReadiness(
        device_count=len(used_device_ids),
        condition_count=condition_count,
        metric_count=len(dataset.metric_catalog),
        metric_observation_count=len(dataset.metric_observations),
        latency_condition_count=len(latency),
        comparison_pair_count=len(dataset.comparisons),
        valid_comparison_pair_count=len(valid_pair_ids),
        valid_comparison_group_count=len(valid_pair_groups),
        op_type_count=len(op_counts),
        op_types_with_multiple_conditions=sum(count > 1 for count in op_counts.values()),
        semantic_family_counts=dict(sorted(family_counts.items())),
        execution_role_counts=dict(sorted(role_counts.items())),
        pmc_conditions_with_independent_repeats=len(pmc_repeat_conditions),
        latency_conditions_with_independent_repeats=len(latency_repeat_conditions),
        available_workload_fields=tuple(sorted(workload_presence)),
        available_launch_fields=tuple(sorted(launch_presence)),
        workload_field_coverage=workload_coverage,
        launch_field_coverage=launch_coverage,
        gpu_clock_condition_coverage=clock_coverage,
        temperature_condition_coverage=temperature_coverage,
        missing_high_value_fields=tuple(missing_controls),
        question_readiness=QuestionReadiness(
            correlation_and_classification=correlation_readiness,
            kernel_type_metric_sets=family_readiness,
            delta_pmc_delta_latency=delta_readiness,
        ),
    )


class ProjectionLayer(AnalysisLayer):
    name = "projection"
    input_type = ProjectionInput

    def run(self, layer_input):
        self.require_input(layer_input)
        dataset = layer_input.dataset
        used_device_ids = {
            condition.device_id for condition in dataset.conditions.values()
        }
        if len(used_device_ids) > 1:
            raise ValueError(
                "PMC analysis is device-scoped; split the dataset by device before running the interpreter: {}"
                .format(", ".join(sorted(used_device_ids)))
            )
        raw_values = defaultdict(lambda: defaultdict(list))
        semantics = defaultdict(set)
        valid_count = 0
        ignored_count = 0
        for observation in dataset.metric_observations:
            if observation.status != "VALID" or observation.value is None:
                ignored_count += 1
                continue
            run = dataset.runs[observation.run_id]
            raw_values[observation.metric_id][run.condition_id].append(
                float(observation.value)
            )
            semantics[observation.metric_id].add(observation.value_semantics)
            valid_count += 1

        specs = {}
        values = {}
        transform_counts = Counter()
        issues = list(dataset.issues)
        for metric_id, per_condition in raw_values.items():
            descriptor = dataset.metric_catalog[metric_id]
            flat_values = [value for items in per_condition.values() for value in items]
            value_semantics = tuple(sorted(semantics[metric_id]))
            transform = choose_transform(
                descriptor, flat_values, ",".join(value_semantics)
            )
            transform_counts[transform] += 1
            if len(value_semantics) > 1:
                issues.append(
                    "feature {} mixes value semantics: {}".format(
                        metric_id, ", ".join(value_semantics)
                    )
                )
            specs[metric_id] = FeatureSpec(
                feature_id=metric_id,
                source_metric_ids=(metric_id,),
                concept_id=descriptor.concept_id,
                expression="raw_or_control_adjusted",
                transform=transform,
                transform_scale=choose_transform_scale(flat_values, transform),
                duration_coupled=descriptor.duration_coupled,
                comparability="within_device; cross-platform by concept only",
                value_semantics=value_semantics,
            )
            values[metric_id] = {
                condition_id: tuple(items)
                for condition_id, items in per_condition.items()
            }

        active_feature_ids = tuple(sorted(specs))
        context = AnalysisContext(
            dataset=dataset,
            config=layer_input.config,
            features=FeatureStore(specs=specs, values=values),
            active_condition_ids=tuple(sorted(dataset.conditions)),
            issues=tuple(dict.fromkeys(issues)),
        )
        report = ProjectionReport(
            input_metric_count=len(dataset.metric_catalog),
            projected_feature_count=len(specs),
            valid_metric_observation_count=valid_count,
            ignored_metric_observation_count=ignored_count,
            transform_counts=dict(sorted(transform_counts.items())),
            data_readiness=build_data_readiness(dataset, layer_input.config),
            notes=(
                "only VALID finite PMC observations are projected",
                "all values are retained; later layers use robust condition medians",
                "signed-asinh scale is fixed at projection time for cross-layer comparability",
            ),
        )
        return ProjectionOutput(
            context=context,
            active_feature_ids=active_feature_ids,
            projection_report=report,
            layer_history=(
                LayerRecord(
                    layer=self.name,
                    input_feature_count=len(dataset.metric_catalog),
                    output_feature_count=len(active_feature_ids),
                    notes=report.notes,
                ),
            ),
        )


def descriptor_for_feature(output, feature_id):
    metric_id = output.context.features.specs[feature_id].source_metric_ids[0]
    return output.context.dataset.metric_catalog[metric_id]


def feature_condition_medians(output, feature_id, condition_ids=None):
    condition_ids = (
        output.context.active_condition_ids
        if condition_ids is None
        else tuple(condition_ids)
    )
    per_condition = output.context.features.values.get(feature_id, {})
    return {
        condition_id: median(per_condition[condition_id])
        for condition_id in condition_ids
        if condition_id in per_condition and per_condition[condition_id]
    }


def independent_repeat_values_by_metric(output):
    """只按非空、明确的 repeat_id 归并独立 PMC 重复。"""
    grouped = defaultdict(lambda: defaultdict(list))
    for observation in output.context.dataset.metric_observations:
        if observation.status != "VALID" or observation.value is None:
            continue
        run = output.context.dataset.runs[observation.run_id]
        if run.collection_kind != "pmc" or not run.repeat_id:
            continue
        grouped[(observation.metric_id, run.condition_id)][run.repeat_id].append(
            float(observation.value)
        )
    result = defaultdict(dict)
    for (metric_id, condition_id), repeats in grouped.items():
        result[metric_id][condition_id] = tuple(
            median(values) for _, values in sorted(repeats.items()) if values
        )
    return {
        metric_id: dict(per_condition)
        for metric_id, per_condition in result.items()
    }


def transformed_feature_map(output, feature_id, condition_ids=None):
    medians = feature_condition_medians(output, feature_id, condition_ids)
    condition_order = sorted(medians)
    descriptor = descriptor_for_feature(output, feature_id)
    spec = output.context.features.specs[feature_id]
    transformed = transform_vector(
        [medians[condition_id] for condition_id in condition_order],
        spec.transform,
        descriptor.bounded_range,
        spec.transform_scale,
    )
    return dict(zip(condition_order, transformed))


def repeat_relative_mad(per_condition):
    ratios = []
    for values in per_condition.values():
        if len(values) < 2:
            continue
        center = median(values)
        mad = median([abs(value - center) for value in values])
        ratios.append(mad / (abs(center) + 1e-12))
    return median(ratios) if ratios else None


def aligned_values(left_map, right_map):
    condition_ids = sorted(set(left_map) & set(right_map))
    return (
        condition_ids,
        [left_map[condition_id] for condition_id in condition_ids],
        [right_map[condition_id] for condition_id in condition_ids],
    )
