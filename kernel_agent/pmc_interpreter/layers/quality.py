"""PMC 测量质量、激活性与 canonical rollup 过滤层。"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from ..dataset.reports import (
    MetricQuality,
    MetricQualityDecision,
    ProjectionOutput,
    QualityOutput,
    QualityReport,
)
from .base import (
    AnalysisLayer,
    append_layer_record,
    descriptor_for_feature,
    independent_repeat_values_by_metric,
    repeat_relative_mad,
)
from .statistics import median


def _quality(output, feature_id, repeat_values_by_metric):
    per_condition = output.context.features.values.get(feature_id, {})
    source_metric_ids = output.context.features.specs[feature_id].source_metric_ids
    independent_repeats = {}
    for metric_id in source_metric_ids:
        for condition_id, values in repeat_values_by_metric.get(metric_id, {}).items():
            independent_repeats.setdefault(condition_id, []).extend(values)
    values = [median(items) for items in per_condition.values() if items]
    condition_count = len(output.context.active_condition_ids)
    mean = statistics.fmean(values) if values else 0.0
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    repeat_values = {
        condition_id: tuple(items)
        for condition_id, items in independent_repeats.items()
    }
    return MetricQuality(
        observed_conditions=len(values),
        coverage=len(values) / condition_count if condition_count else 0.0,
        unique_values=len(set(values)),
        nonzero_conditions=sum(value != 0 for value in values),
        nonzero_rate=(
            sum(value != 0 for value in values) / len(values) if values else 0.0
        ),
        cross_condition_mean=mean,
        cross_condition_stddev=stddev,
        cross_condition_cv=(stddev / (abs(mean) + 1e-12) if values else None),
        conditions_with_repeats=sum(len(items) > 1 for items in repeat_values.values()),
        repeat_relative_mad=repeat_relative_mad(repeat_values),
        repeat_semantics="distinct non-empty repeat_id only",
    )


def _canonical_representative(output, feature_ids):
    descriptors = {
        feature_id: descriptor_for_feature(output, feature_id)
        for feature_id in feature_ids
    }
    any_target = any(
        descriptor.target_equivalent for descriptor in descriptors.values()
    )
    any_ratio = any(
        descriptor.quantity_kind in {"ratio", "throughput"}
        for descriptor in descriptors.values()
    )
    preferred = (
        ("avg", "none", "sum", "max", "min")
        if any_target or any_ratio
        else ("sum", "none", "avg", "max", "min")
    )
    by_aggregation = {
        descriptor.aggregation: feature_id
        for feature_id, descriptor in descriptors.items()
    }
    for aggregation in preferred:
        if aggregation in by_aggregation:
            return by_aggregation[aggregation]
    return sorted(feature_ids)[0]


class QualityLayer(AnalysisLayer):
    name = "quality"
    input_type = ProjectionOutput

    def run(self, layer_input):
        self.require_input(layer_input)
        config = layer_input.context.config
        groups = defaultdict(list)
        qualities = {}
        repeat_values_by_metric = independent_repeat_values_by_metric(layer_input)
        status_by_metric = defaultdict(Counter)
        for observation in layer_input.context.dataset.metric_observations:
            status_by_metric[observation.metric_id][observation.status] += 1
        for feature_id in layer_input.active_feature_ids:
            descriptor = descriptor_for_feature(layer_input, feature_id)
            groups[
                (
                    descriptor.namespace,
                    descriptor.basename,
                    descriptor.phenomenon_role,
                    descriptor.normalizer,
                )
            ].append(feature_id)
            qualities[feature_id] = _quality(
                layer_input, feature_id, repeat_values_by_metric
            )
        base_reasons = {}
        for feature_id in layer_input.active_feature_ids:
            descriptor = descriptor_for_feature(layer_input, feature_id)
            quality = qualities[feature_id]
            reasons = []
            if quality.coverage < config.min_coverage:
                reasons.append("low_coverage")
            if quality.nonzero_conditions == 0:
                reasons.append("inactive_all_zero")
            elif quality.unique_values < config.min_unique_values:
                reasons.append("constant_or_near_constant")
            if descriptor.target_equivalent:
                reasons.append("target_equivalent")
            if descriptor.analysis_scope != "kernel_core":
                reasons.append("semantic_scope:{}".format(descriptor.analysis_scope))
            if len(layer_input.context.features.specs[feature_id].value_semantics) > 1:
                reasons.append("mixed_value_semantics")
            if (
                quality.repeat_relative_mad is not None
                and quality.conditions_with_repeats >= 3
                and quality.repeat_relative_mad > config.max_repeat_relative_mad
            ):
                reasons.append("unstable_repeats")
            base_reasons[feature_id] = reasons

        representatives = {}
        for group_key, feature_ids in groups.items():
            eligible = [
                feature_id
                for feature_id in feature_ids
                if not base_reasons[feature_id]
            ]
            representatives[group_key] = _canonical_representative(
                layer_input, eligible or feature_ids
            )

        active = []
        decisions = {}
        exclusion_counts = Counter()
        for feature_id in layer_input.active_feature_ids:
            descriptor = descriptor_for_feature(layer_input, feature_id)
            quality = qualities[feature_id]
            key = (
                descriptor.namespace,
                descriptor.basename,
                descriptor.phenomenon_role,
                descriptor.normalizer,
            )
            representative = representatives[key]
            reasons = list(base_reasons[feature_id])
            if feature_id != representative and not base_reasons[representative]:
                reasons.append("redundant_rollup:{}".format(representative))
            if not reasons:
                active.append(feature_id)
            for reason in reasons:
                exclusion_counts[reason.split(":", 1)[0]] += 1
            observation_status_counts = Counter()
            for metric_id in layer_input.context.features.specs[
                feature_id
            ].source_metric_ids:
                observation_status_counts.update(status_by_metric[metric_id])
            decisions[feature_id] = MetricQualityDecision(
                feature_id=feature_id,
                quality=quality,
                observation_status_counts=dict(observation_status_counts),
                canonical_rollup_representative=representative,
                disposition="candidate" if not reasons else "excluded",
                exclusion_reasons=tuple(reasons),
            )

        active_ids = tuple(sorted(active))
        report = QualityReport(
            input_feature_count=len(layer_input.active_feature_ids),
            active_feature_count=len(active_ids),
            native_basename_expression_groups=len(groups),
            exclusion_counts=dict(sorted(exclusion_counts.items())),
            descriptive_canonical_candidates=active_ids,
            metrics=decisions,
            notes=(
                "canonicalization is semantic/quality based and does not use latency relevance",
                "profiler pass count is retained on observations but is not treated as a repeat or intrinsic metric cost",
            ),
        )
        return QualityOutput(
            context=layer_input.context,
            active_feature_ids=active_ids,
            projection_report=layer_input.projection_report,
            layer_history=append_layer_record(
                layer_input, self.name, active_ids, report.notes
            ),
            quality_report=report,
        )
