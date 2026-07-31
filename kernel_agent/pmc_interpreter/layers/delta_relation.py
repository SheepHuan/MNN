"""同语义实现配对中的 ΔPMC–Δlatency 条件关系层。"""

from __future__ import annotations

import math
from collections import defaultdict

from ..dataset.reports import (
    DeltaCollinearGroup,
    DeltaObservedSupport,
    DeltaRelationOutput,
    DeltaRelationReport,
    DeltaRelationStatistic,
    DeltaRule,
    DeltaScopeCondition,
    DeltaScopeSelector,
    FamilyDeltaReadiness,
    KernelSignatureOutput,
    ObservedNumericRange,
    RejectedComparison,
)
from .base import (
    AnalysisLayer,
    append_layer_record,
    descriptor_for_feature,
    feature_condition_medians,
    latency_values_by_condition,
    transformed_feature_map,
)
from .statistics import (
    approximate_correlation_pvalue,
    benjamini_hochberg,
    median,
    rankdata,
    spearman,
    through_origin_effect,
)


def _valid_pairs(output, pairs):
    latency = latency_values_by_condition(output.context.dataset)
    valid = []
    rejected = []
    for pair in pairs:
        baseline = output.context.dataset.conditions.get(
            pair.baseline_condition_id
        )
        candidate = output.context.dataset.conditions.get(
            pair.candidate_condition_id
        )
        reason = None
        if baseline is None or candidate is None:
            reason = "missing_condition"
        elif baseline.device_id != candidate.device_id:
            reason = "cross_device_pair"
        elif baseline.semantic_equivalence_key != candidate.semantic_equivalence_key:
            reason = "semantic_equivalence_mismatch"
        elif baseline.implementation_id == candidate.implementation_id:
            reason = "implementation_not_changed"
        elif (
            baseline.condition_id not in latency
            or candidate.condition_id not in latency
        ):
            reason = "missing_latency"
        if reason:
            rejected.append(
                RejectedComparison(pair_id=pair.pair_id, reason=reason)
            )
        else:
            valid.append(pair)
    return tuple(valid), tuple(rejected)


def _pair_deltas(output, feature_id, pairs, latency):
    feature_map = transformed_feature_map(output, feature_id)
    result = []
    for pair in pairs:
        baseline = pair.baseline_condition_id
        candidate = pair.candidate_condition_id
        if baseline not in feature_map or candidate not in feature_map:
            continue
        result.append(
            {
                "pair_id": pair.pair_id,
                "pair_group_id": output.context.dataset.conditions[
                    baseline
                ].semantic_equivalence_key,
                "delta_feature": feature_map[candidate] - feature_map[baseline],
                "delta_log_latency": math.log(
                    median(latency[candidate]) / median(latency[baseline])
                ),
            }
        )
    return result


def _collapse_pair_groups(deltas):
    grouped = defaultdict(list)
    for item in deltas:
        grouped[item["pair_group_id"]].append(item)
    return tuple(
        {
            "pair_group_id": group_id,
            "pair_ids": tuple(item["pair_id"] for item in items),
            "delta_feature": median([item["delta_feature"] for item in items]),
            "delta_log_latency": median(
                [item["delta_log_latency"] for item in items]
            ),
        }
        for group_id, items in sorted(grouped.items())
    )


def _independent_pair_group_count(output, pairs):
    return len(
        {
            output.context.dataset.conditions[
                pair.baseline_condition_id
            ].semantic_equivalence_key
            for pair in pairs
        }
    )


def _delta_relation_stats(output, pairs):
    config = output.context.config
    latency = latency_values_by_condition(output.context.dataset)
    partial = {}
    pvalues = {}
    delta_vectors = {}
    for feature_id in output.active_feature_ids:
        deltas = _pair_deltas(output, feature_id, pairs, latency)
        grouped_deltas = _collapse_pair_groups(deltas)
        delta_x = [item["delta_feature"] for item in grouped_deltas]
        delta_y = [item["delta_log_latency"] for item in grouped_deltas]
        contributing_pair_ids = tuple(item["pair_id"] for item in deltas)
        usage_class = output.correlation_report.associations[
            feature_id
        ].usage_class
        if (
            len(grouped_deltas) < config.min_delta_pairs
            or len(set(delta_x)) < 3
        ):
            partial[feature_id] = DeltaRelationStatistic(
                feature_id=feature_id,
                pair_count=len(deltas),
                independent_pair_group_count=len(grouped_deltas),
                contributing_pair_ids=contributing_pair_ids,
                zero_delta_pair_count=sum(
                    abs(item["delta_feature"]) <= 1e-12 for item in deltas
                ),
                relation_class="insufficient_paired_variation",
                usage_class=usage_class,
            )
            continue
        rho = spearman(delta_x, delta_y)
        pvalue = approximate_correlation_pvalue(rho, len(grouped_deltas))
        slope, interval = through_origin_effect(delta_x, delta_y)
        directional = [
            (x, y)
            for x, y in zip(delta_x, delta_y)
            if abs(x) > 1e-12 and abs(y) > 1e-12
        ]
        positive = (
            sum((x > 0) == (y > 0) for x, y in directional)
            / len(directional)
            if directional
            else 0.0
        )
        partial[feature_id] = DeltaRelationStatistic(
            feature_id=feature_id,
            pair_count=len(deltas),
            independent_pair_group_count=len(grouped_deltas),
            contributing_pair_ids=contributing_pair_ids,
            zero_delta_pair_count=sum(
                abs(item["delta_feature"]) <= 1e-12 for item in deltas
            ),
            relation_class="pending_multiple_testing",
            usage_class=usage_class,
            approximate_pvalue=pvalue,
            delta_spearman=rho,
            transformed_slope=slope,
            transformed_slope_ci95=interval,
            positive_direction_concordance=positive,
            negative_direction_concordance=1.0 - positive,
        )
        pvalues[feature_id] = pvalue
        delta_vectors[feature_id] = (
            tuple(item["pair_group_id"] for item in grouped_deltas),
            tuple(delta_x),
        )

    qvalues = benjamini_hochberg(pvalues)
    results = {}
    for feature_id, statistic in partial.items():
        if feature_id not in qvalues:
            results[feature_id] = statistic
            continue
        rho = statistic.delta_spearman or 0.0
        concordance = (
            statistic.positive_direction_concordance
            if rho >= 0
            else statistic.negative_direction_concordance
        ) or 0.0
        relation_class = "no_stable_relation"
        if (
            qvalues[feature_id] <= config.fdr_threshold
            and abs(rho) >= 0.4
            and concordance >= 0.65
        ):
            relation_class = (
                "increase_associated_with_higher_latency"
                if rho > 0
                else "increase_associated_with_lower_latency"
            )
        exclusion_reason = ""
        descriptor = descriptor_for_feature(output, feature_id)
        if (
            statistic.usage_class == "diagnostic_only"
            or descriptor.duration_coupled
            or descriptor.target_equivalent
        ):
            exclusion_reason = "diagnostic_or_elapsed_time_coupled"
        results[feature_id] = statistic.model_copy(
            update={
                "fdr_qvalue": qvalues[feature_id],
                "directional_concordance": concordance,
                "effect_score": abs(rho)
                * concordance
                * math.sqrt(statistic.independent_pair_group_count),
                "relation_class": relation_class,
                "exclusion_reason": exclusion_reason,
            }
        )
    return results, delta_vectors


def _normalized_delta_fingerprint(pair_ids, values):
    ranks = rankdata(values)
    center = sum(ranks) / len(ranks) if ranks else 0.0
    normalized = [value - center for value in ranks]
    norm = math.sqrt(sum(value * value for value in normalized))
    if norm <= 1e-12:
        normalized_values = tuple(0.0 for _ in values)
    else:
        normalized = [value / norm for value in normalized]
        first = next(
            (value for value in normalized if abs(value) > 1e-12), 1.0
        )
        if first < 0:
            normalized = [-value for value in normalized]
        normalized_values = tuple(round(value, 10) for value in normalized)
    return tuple(pair_ids), normalized_values


def _repeat_ids_for_feature(
    output, feature_id, condition_id, paired_group_ids=None
):
    source_metric_ids = set(
        output.context.features.specs[feature_id].source_metric_ids
    )
    paired_group_ids = (
        None if paired_group_ids is None else set(paired_group_ids)
    )
    by_metric = {metric_id: set() for metric_id in source_metric_ids}
    for observation in output.context.dataset.metric_observations:
        if (
            observation.metric_id not in source_metric_ids
            or observation.status != "VALID"
            or observation.value is None
        ):
            continue
        run = output.context.dataset.runs[observation.run_id]
        if (
            run.condition_id == condition_id
            and run.repeat_id
            and (
                paired_group_ids is None
                or run.paired_run_group_id in paired_group_ids
            )
        ):
            by_metric[observation.metric_id].add(run.repeat_id)
    return by_metric


def _latency_repeat_ids(output, condition_id, paired_group_ids=None):
    paired_group_ids = (
        None if paired_group_ids is None else set(paired_group_ids)
    )
    result = set()
    for observation in output.context.dataset.latency_observations:
        if observation.status != "VALID" or observation.latency_us is None:
            continue
        run = output.context.dataset.runs[observation.run_id]
        if (
            run.condition_id == condition_id
            and run.repeat_id
            and (
                paired_group_ids is None
                or run.paired_run_group_id in paired_group_ids
            )
        ):
            result.add(run.repeat_id)
    return result


def _paired_evidence_groups(output, feature_id, condition_id):
    source_metric_ids = set(
        output.context.features.specs[feature_id].source_metric_ids
    )
    pmc_groups_by_metric = {
        metric_id: set() for metric_id in source_metric_ids
    }
    for observation in output.context.dataset.metric_observations:
        if (
            observation.metric_id not in source_metric_ids
            or observation.status != "VALID"
            or observation.value is None
        ):
            continue
        run = output.context.dataset.runs[observation.run_id]
        if (
            run.condition_id == condition_id
            and run.repeat_id
            and run.paired_run_group_id
        ):
            pmc_groups_by_metric[observation.metric_id].add(
                run.paired_run_group_id
            )
    latency_groups = set()
    for observation in output.context.dataset.latency_observations:
        if observation.status != "VALID" or observation.latency_us is None:
            continue
        run = output.context.dataset.runs[observation.run_id]
        if (
            run.condition_id == condition_id
            and run.repeat_id
            and run.paired_run_group_id
        ):
            latency_groups.add(run.paired_run_group_id)
    if not pmc_groups_by_metric:
        return set()
    shared_pmc_groups = set.intersection(
        *(set(groups) for groups in pmc_groups_by_metric.values())
    )
    return shared_pmc_groups & latency_groups


def _environment_values(
    output, condition_id, field, paired_run_group_id
):
    return [
        getattr(run, field)
        for run in output.context.dataset.runs.values()
        if (
            run.condition_id == condition_id
            and run.paired_run_group_id == paired_run_group_id
            and getattr(run, field) is not None
        )
    ]


def _measured_environment_compatible(output, pair, paired_group_ids):
    for paired_group_id in paired_group_ids:
        for field, tolerance in (
            ("gpu_clock_hz", 0.01),
            ("temperature_c", 5.0),
        ):
            baseline_values = _environment_values(
                output,
                pair.baseline_condition_id,
                field,
                paired_group_id,
            )
            candidate_values = _environment_values(
                output,
                pair.candidate_condition_id,
                field,
                paired_group_id,
            )
            if not baseline_values or not candidate_values:
                return False
            baseline = median(baseline_values)
            candidate = median(candidate_values)
            if field == "gpu_clock_hz":
                if (
                    abs(candidate - baseline) / max(abs(baseline), 1.0)
                    > tolerance
                ):
                    return False
            elif abs(candidate - baseline) > tolerance:
                return False
    return True


def _causal_claim_allowed(output, feature_id, pairs, collinear):
    if collinear:
        return False
    if not pairs or any(pair.design != "controlled_intervention" for pair in pairs):
        return False
    if any(
        pair.environment_matched is not True or pair.randomized_order is not True
        for pair in pairs
    ):
        return False
    controlled_mechanism_values = [
        pair.controlled_mechanism.strip() for pair in pairs
    ]
    if any(not mechanism for mechanism in controlled_mechanism_values):
        return False
    controlled_mechanisms = set(controlled_mechanism_values)
    if len(controlled_mechanisms) != 1:
        return False
    descriptor = descriptor_for_feature(output, feature_id)
    if next(iter(controlled_mechanisms)) not in {
        descriptor.mechanism,
        descriptor.concept_id,
    }:
        return False
    if descriptor.duration_coupled or descriptor.target_equivalent:
        return False
    if any(not pair.intervention_id.strip() for pair in pairs):
        return False
    for pair in pairs:
        baseline_groups = _paired_evidence_groups(
            output, feature_id, pair.baseline_condition_id
        )
        candidate_groups = _paired_evidence_groups(
            output, feature_id, pair.candidate_condition_id
        )
        shared_groups = baseline_groups & candidate_groups
        if len(shared_groups) < 2:
            return False
        if not _measured_environment_compatible(
            output, pair, shared_groups
        ):
            return False
        for condition_id in (
            pair.baseline_condition_id,
            pair.candidate_condition_id,
        ):
            metric_repeat_ids = _repeat_ids_for_feature(
                output, feature_id, condition_id, shared_groups
            )
            if not metric_repeat_ids or any(
                len(repeat_ids) < 2 for repeat_ids in metric_repeat_ids.values()
            ):
                return False
            if len(
                _latency_repeat_ids(output, condition_id, shared_groups)
            ) < 2:
                return False
    return True


def _interpretation(output, feature_id, relation_class, slope, causal):
    spec = output.context.features.specs[feature_id]
    if spec.transform == "log1p":
        slope_text = "slope is d(log latency)/d(log(PMC+1))"
    elif spec.transform == "logit":
        slope_text = "slope is per one log-odds unit of the bounded PMC"
    elif spec.transform == "signed_asinh":
        slope_text = (
            "slope is per one signed-asinh unit with fixed scale {:.6g}".format(
                spec.transform_scale or 1.0
            )
        )
    else:
        slope_text = "slope is d(log latency)/d(log PMC)"
    direction = {
        "increase_associated_with_higher_latency": (
            "feature increases align with higher latency"
        ),
        "increase_associated_with_lower_latency": (
            "feature increases align with lower latency"
        ),
    }.get(relation_class, "no stable direction is established")
    claim = (
        "controlled intervention evidence permits a scoped causal interpretation"
        if causal
        else "this is a paired conditional association, not a causal direction"
    )
    return "{}; {}; {} (estimated slope {:.4g}).".format(
        direction, claim, slope_text, slope
    )


def _observed_range(values):
    return ObservedNumericRange(minimum=min(values), maximum=max(values))


def _observed_support(output, feature_id, pairs):
    raw_feature = feature_condition_medians(output, feature_id)
    transformed_feature = transformed_feature_map(output, feature_id)
    latency = latency_values_by_condition(output.context.dataset)
    raw_baseline = []
    raw_candidate = []
    raw_delta = []
    transformed_baseline = []
    transformed_candidate = []
    transformed_delta = []
    latency_ratio = []
    delta_log_latency = []
    for pair in pairs:
        baseline = pair.baseline_condition_id
        candidate = pair.candidate_condition_id
        if (
            baseline not in raw_feature
            or candidate not in raw_feature
            or baseline not in transformed_feature
            or candidate not in transformed_feature
            or baseline not in latency
            or candidate not in latency
        ):
            continue
        baseline_raw_value = raw_feature[baseline]
        candidate_raw_value = raw_feature[candidate]
        baseline_transformed_value = transformed_feature[baseline]
        candidate_transformed_value = transformed_feature[candidate]
        ratio = median(latency[candidate]) / median(latency[baseline])
        raw_baseline.append(baseline_raw_value)
        raw_candidate.append(candidate_raw_value)
        raw_delta.append(candidate_raw_value - baseline_raw_value)
        transformed_baseline.append(baseline_transformed_value)
        transformed_candidate.append(candidate_transformed_value)
        transformed_delta.append(
            candidate_transformed_value - baseline_transformed_value
        )
        latency_ratio.append(ratio)
        delta_log_latency.append(math.log(ratio))
    if not raw_baseline:
        raise ValueError(
            "delta rule {} has no contributing observations".format(feature_id)
        )
    return DeltaObservedSupport(
        raw_baseline_feature=_observed_range(raw_baseline),
        raw_candidate_feature=_observed_range(raw_candidate),
        raw_feature_delta=_observed_range(raw_delta),
        transformed_baseline_feature=_observed_range(transformed_baseline),
        transformed_candidate_feature=_observed_range(transformed_candidate),
        transformed_feature_delta=_observed_range(transformed_delta),
        latency_ratio=_observed_range(latency_ratio),
        delta_log_latency=_observed_range(delta_log_latency),
    )


def _scope_conditions(output, pairs):
    scoped = {}
    for pair in pairs:
        for comparison_role, condition_id in (
            ("baseline", pair.baseline_condition_id),
            ("candidate", pair.candidate_condition_id),
        ):
            condition = output.context.dataset.conditions[condition_id]
            key = (comparison_role, condition_id)
            scoped[key] = DeltaScopeCondition(
                condition_id=condition.condition_id,
                comparison_role=comparison_role,
                device_id=condition.device_id,
                semantic_equivalence_key=condition.semantic_equivalence_key,
                semantic_family=condition.semantic_family,
                op_type=condition.op_type,
                execution_role=condition.execution_role,
                performance_regime=condition.performance_regime,
                dtype=condition.dtype,
                implementation_id=condition.implementation_id,
                shapes=condition.shapes,
                params=condition.params,
                workload=condition.workload,
                launch=condition.launch,
            )
    return tuple(scoped[key] for key in sorted(scoped))


def _make_rule(output, feature_id, statistic, pairs, rule_id, families, collinear):
    descriptor = descriptor_for_feature(output, feature_id)
    causal = _causal_claim_allowed(output, feature_id, pairs, collinear)
    return DeltaRule(
        rule_id=rule_id,
        scope_selector=DeltaScopeSelector(
            device_ids=tuple(
                sorted(
                    {
                        output.context.dataset.conditions[
                            pair.baseline_condition_id
                        ].device_id
                        for pair in pairs
                    }
                )
            ),
            semantic_families=tuple(sorted(families)),
            conditions=_scope_conditions(output, pairs),
            intervention_ids=tuple(
                sorted(
                    {
                        pair.intervention_id.strip()
                        for pair in pairs
                        if pair.intervention_id.strip()
                    }
                )
            ),
            controlled_mechanisms=tuple(
                sorted(
                    {
                        pair.controlled_mechanism.strip()
                        for pair in pairs
                        if pair.controlled_mechanism.strip()
                    }
                )
            ),
        ),
        feature_id=feature_id,
        metric_id=descriptor.metric_id,
        native_name=descriptor.native_name,
        concept_id=descriptor.concept_id,
        mechanism=descriptor.mechanism,
        metric_delta_definition=(
            "transform(candidate PMC) - transform(baseline PMC), transform={}".format(
                output.context.features.specs[feature_id].transform
            )
        ),
        latency_delta_definition=(
            "log(candidate median latency / baseline median latency)"
        ),
        observed_support=_observed_support(output, feature_id, pairs),
        statistics=statistic,
        evidence_level=(
            "controlled_intervention" if causal else "paired_observational"
        ),
        causal_claim_allowed=causal,
        interpretation=_interpretation(
            output,
            feature_id,
            statistic.relation_class,
            statistic.transformed_slope or 0.0,
            causal,
        ),
    )


def _contributing_pairs(pairs, statistic):
    pair_ids = set(statistic.contributing_pair_ids)
    return tuple(pair for pair in pairs if pair.pair_id in pair_ids)


def _empty_report(output, status, pairs, rejected, minimum_pairs):
    return DeltaRelationReport(
        status=status,
        valid_pair_count=len(pairs),
        valid_pair_ids=tuple(pair.pair_id for pair in pairs),
        independent_pair_group_count=_independent_pair_group_count(output, pairs),
        minimum_pairs=minimum_pairs,
        rejected_pairs=rejected,
        metric_relations={},
        candidate_relations=(),
        effect_rules=(),
        global_effect_rules=(),
        semantic_family_effect_rules={},
        delta_collinear_groups=(),
        semantic_family_readiness={},
        required_design=(
            "same device and semantic_equivalence_key",
            "explicit baseline/candidate implementation IDs",
            "independent latency and PMC repeat_id values inside shared paired-run groups",
            "at least two shared paired_run_group_id values linking valid PMC and latency",
            "measured clock/temperature compatibility inside every shared paired-run group",
            "controlled intervention with randomized order, intervention ID, and one non-empty matching mechanism",
        ),
        warning="cross-kernel association is not converted into a delta rule",
        multiple_testing="not_estimable",
        limitations=(
            "implementation pairs are required for delta analysis",
        ),
    )


class DeltaRelationLayer(AnalysisLayer):
    name = "delta_relations"
    input_type = KernelSignatureOutput

    def run(self, layer_input):
        self.require_input(layer_input)
        config = layer_input.context.config
        pairs, rejected = _valid_pairs(
            layer_input, layer_input.context.dataset.comparisons
        )
        independent_pair_group_count = _independent_pair_group_count(
            layer_input, pairs
        )
        if independent_pair_group_count < config.min_delta_pairs:
            report = _empty_report(
                layer_input,
                "insufficient_paired_variant_evidence",
                pairs,
                rejected,
                config.min_delta_pairs,
            )
            return DeltaRelationOutput(
                context=layer_input.context,
                active_feature_ids=layer_input.active_feature_ids,
                projection_report=layer_input.projection_report,
                layer_history=append_layer_record(
                    layer_input,
                    self.name,
                    layer_input.active_feature_ids,
                    (report.warning,),
                ),
                quality_report=layer_input.quality_report,
                redundancy_report=layer_input.redundancy_report,
                correlation_report=layer_input.correlation_report,
                signature_report=layer_input.signature_report,
                delta_report=report,
            )

        statistics_by_feature, delta_vectors = _delta_relation_stats(
            layer_input, pairs
        )
        collinear_groups = defaultdict(list)
        for feature_id, (pair_ids, values) in delta_vectors.items():
            collinear_groups[
                _normalized_delta_fingerprint(pair_ids, values)
            ].append(feature_id)
        collinear_for = {}
        group_payload = []
        for index, feature_ids in enumerate(collinear_groups.values(), 1):
            group_id = "delta_collinear_{:04d}".format(index)
            for feature_id in feature_ids:
                collinear_for[feature_id] = group_id if len(feature_ids) > 1 else ""
            if len(feature_ids) > 1:
                group_payload.append(
                    DeltaCollinearGroup(
                        group_id=group_id,
                        features=tuple(sorted(feature_ids)),
                        warning=(
                            "univariate paired data cannot identify which collinear feature carries the mechanism"
                        ),
                    )
                )

        enriched_statistics = {}
        candidates = []
        for feature_id, statistic in statistics_by_feature.items():
            collinear_group = collinear_for.get(feature_id, "")
            statistic = statistic.model_copy(
                update={"delta_collinear_group": collinear_group}
            )
            enriched_statistics[feature_id] = statistic
            if (
                statistic.relation_class
                in {
                    "increase_associated_with_higher_latency",
                    "increase_associated_with_lower_latency",
                }
                and not statistic.exclusion_reason
            ):
                contributing_pairs = _contributing_pairs(pairs, statistic)
                contributing_families = {
                    layer_input.context.dataset.conditions[
                        pair.baseline_condition_id
                    ].semantic_family
                    for pair in contributing_pairs
                }
                candidates.append(
                    _make_rule(
                        layer_input,
                        feature_id,
                        statistic,
                        contributing_pairs,
                        "delta::{}".format(feature_id),
                        contributing_families,
                        bool(collinear_group),
                    )
                )

        candidates.sort(
            key=lambda item: (
                -(item.statistics.effect_score or 0.0),
                item.feature_id,
            )
        )
        selected_rules = []
        used_collinear = set()
        for candidate in candidates:
            group_id = candidate.statistics.delta_collinear_group
            if group_id and group_id in used_collinear:
                continue
            selected_rules.append(candidate)
            if group_id:
                used_collinear.add(group_id)
            if len(selected_rules) >= config.delta_top_k:
                break

        family_pairs = defaultdict(list)
        for pair in pairs:
            family = layer_input.context.dataset.conditions[
                pair.baseline_condition_id
            ].semantic_family
            family_pairs[family].append(pair)
        scoped_rule_index = {}
        scoped_effect_rules = []
        for family, selected_pairs in sorted(family_pairs.items()):
            family_group_count = _independent_pair_group_count(
                layer_input, selected_pairs
            )
            if (
                len(family_pairs) <= 1
                or family_group_count < config.min_delta_pairs
            ):
                scoped_rule_index[family] = ()
                continue
            family_stats, family_vectors = _delta_relation_stats(
                layer_input, tuple(selected_pairs)
            )
            family_collinear = defaultdict(list)
            for feature_id, (pair_ids, values) in family_vectors.items():
                family_collinear[
                    _normalized_delta_fingerprint(pair_ids, values)
                ].append(feature_id)
            family_collinear_for = {}
            for index, feature_ids in enumerate(
                family_collinear.values(), 1
            ):
                if len(feature_ids) <= 1:
                    continue
                group_id = "family::{}::delta_collinear_{:04d}".format(
                    family, index
                )
                for feature_id in feature_ids:
                    family_collinear_for[feature_id] = group_id
                group_payload.append(
                    DeltaCollinearGroup(
                        group_id=group_id,
                        features=tuple(sorted(feature_ids)),
                        warning=(
                            "family-scoped paired data cannot identify which collinear feature carries the mechanism"
                        ),
                    )
                )
            family_candidates = []
            for feature_id, statistic in family_stats.items():
                if (
                    statistic.relation_class
                    not in {
                        "increase_associated_with_higher_latency",
                        "increase_associated_with_lower_latency",
                    }
                    or statistic.exclusion_reason
                ):
                    continue
                family_collinear_group = family_collinear_for.get(
                    feature_id, ""
                )
                if family_collinear_group:
                    statistic = statistic.model_copy(
                        update={
                            "delta_collinear_group": family_collinear_group
                        }
                    )
                family_candidates.append(
                    _make_rule(
                        layer_input,
                        feature_id,
                        statistic,
                        _contributing_pairs(selected_pairs, statistic),
                        "delta::family::{}::{}".format(family, feature_id),
                        {family},
                        bool(family_collinear_group),
                    )
                )
            family_candidates.sort(
                key=lambda item: (
                    -(item.statistics.effect_score or 0.0),
                    item.feature_id,
                )
            )
            selected_family_rules = []
            used_family_collinear = set()
            for candidate in family_candidates:
                group_id = candidate.statistics.delta_collinear_group
                if group_id and group_id in used_family_collinear:
                    continue
                selected_family_rules.append(candidate)
                if group_id:
                    used_family_collinear.add(group_id)
                if len(selected_family_rules) >= config.delta_top_k:
                    break
            scoped_rule_index[family] = tuple(selected_family_rules)
            scoped_effect_rules.extend(scoped_rule_index[family])

        family_readiness = {
            family: FamilyDeltaReadiness(
                pair_count=len(items),
                independent_pair_group_count=_independent_pair_group_count(
                    layer_input, items
                ),
                status=(
                    "ready"
                    if _independent_pair_group_count(layer_input, items)
                    >= config.min_delta_pairs
                    else "insufficient_pairs"
                ),
            )
            for family, items in sorted(family_pairs.items())
        }
        all_effect_rules = tuple(selected_rules + scoped_effect_rules)
        report = DeltaRelationReport(
            status=(
                "paired_association_available"
                if all_effect_rules
                else "paired_data_no_stable_rules"
            ),
            valid_pair_count=len(pairs),
            valid_pair_ids=tuple(pair.pair_id for pair in pairs),
            independent_pair_group_count=independent_pair_group_count,
            minimum_pairs=config.min_delta_pairs,
            rejected_pairs=rejected,
            metric_relations=enriched_statistics,
            candidate_relations=tuple(candidates),
            effect_rules=all_effect_rules,
            global_effect_rules=tuple(selected_rules),
            semantic_family_effect_rules=scoped_rule_index,
            delta_collinear_groups=tuple(group_payload),
            semantic_family_readiness=family_readiness,
            required_design=(
                "same device and semantic_equivalence_key",
                "explicit baseline/candidate implementation IDs",
                "independent latency and PMC repeat_id values inside shared paired-run groups",
                "at least two shared paired_run_group_id values linking valid PMC and latency",
                "measured clock/temperature compatibility inside every shared paired-run group",
                "controlled intervention with randomized order, intervention ID, and one non-empty matching mechanism",
            ),
            warning=(
                "effect rules are paired conditional associations unless causal_claim_allowed is true"
            ),
            multiple_testing=(
                "Benjamini-Hochberg FDR across active canonical features"
            ),
            limitations=(
                "zero-delta pairs are retained",
                "univariate relations cannot separate simultaneously changing non-collinear mechanisms",
                "duration-coupled and diagnostic-only metrics cannot enter optimization effect rules",
                "causal claims require controlled intervention, environment matching, randomization, and independent repeats",
                "the current slope is a single-variable transformed linear approximation",
                "rules are valid only for their listed conditions, interventions, and observed value ranges",
            ),
        )
        return DeltaRelationOutput(
            context=layer_input.context,
            active_feature_ids=layer_input.active_feature_ids,
            projection_report=layer_input.projection_report,
            layer_history=append_layer_record(
                layer_input,
                self.name,
                layer_input.active_feature_ids,
                (report.warning,),
            ),
            quality_report=layer_input.quality_report,
            redundancy_report=layer_input.redundancy_report,
            correlation_report=layer_input.correlation_report,
            signature_report=layer_input.signature_report,
            delta_report=report,
        )
