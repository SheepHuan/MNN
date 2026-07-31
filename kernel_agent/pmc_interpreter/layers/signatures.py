"""Kernel 类别到平台原生 PMC 诊断签名的解析层。"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from ..dataset.reports import (
    CorrelationOutput,
    KernelSignature,
    KernelSignatureOutput,
    KernelSignatureReport,
    MetricKernelClassLink,
    SignatureMetric,
    SignatureSelectionMethod,
)
from .base import (
    AnalysisLayer,
    aligned_values,
    append_layer_record,
    descriptor_for_feature,
)
from .statistics import (
    approximate_correlation_pvalue,
    benjamini_hochberg,
    spearman,
)


EXPLANATION_USAGE = {"latency_informative_candidate", "descriptive_association"}
DIAGNOSTIC_USAGE = {"diagnostic_only", "workload_indicator"}


def _expected_mechanisms(output, condition_ids):
    mechanisms = set()
    for condition_id in condition_ids:
        mechanisms.update(
            output.context.dataset.conditions[condition_id].expected_mechanisms
        )
    return mechanisms


def _aligned_spearman(left_map, right_map):
    _, left, right = aligned_values(left_map, right_map)
    return spearman(left, right) if len(left) >= 3 else 0.0


def _mrmr_select(
    output,
    condition_ids,
    relevance,
    top_k,
    expected_mechanisms=None,
    allowed_usage=None,
    usage_by_feature=None,
):
    config = output.context.config
    expected_mechanisms = set(expected_mechanisms or ())
    allowed_usage = set(allowed_usage or EXPLANATION_USAGE)
    usage_by_feature = usage_by_feature or {
        feature_id: association.usage_class
        for feature_id, association in output.correlation_report.associations.items()
    }
    candidates = [
        feature_id
        for feature_id in sorted(
            relevance, key=lambda item: (-relevance[item], item)
        )
        if relevance[feature_id] > 0
        and usage_by_feature.get(feature_id) in allowed_usage
    ][:160]
    condition_id_set = set(condition_ids)
    vectors = {
        feature_id: {
            condition_id: value
            for condition_id, value in output.correlation_report.residual_feature_values[
                feature_id
            ].items()
            if condition_id in condition_id_set
        }
        for feature_id in candidates
    }
    selected = []
    mechanism_counts = Counter()
    concept_counts = Counter()
    while candidates and len(selected) < top_k:
        best = None
        best_details = None
        for feature_id in candidates:
            descriptor = descriptor_for_feature(output, feature_id)
            if (
                mechanism_counts[descriptor.mechanism] >= 4
                or concept_counts[descriptor.concept_id] >= 2
            ):
                continue
            redundancies = [
                abs(
                    _aligned_spearman(
                        vectors[feature_id], vectors[item["feature_id"]]
                    )
                )
                for item in selected
            ]
            redundancy = statistics.fmean(redundancies) if redundancies else 0.0
            semantic_bonus = (
                0.04 if descriptor.mechanism in expected_mechanisms else 0.0
            )
            diversity_bonus = (
                0.03 if mechanism_counts[descriptor.mechanism] == 0 else 0.0
            )
            score = (
                relevance[feature_id]
                - config.redundancy_weight * redundancy
                + semantic_bonus
                + diversity_bonus
            )
            details = {
                "feature_id": feature_id,
                "metric_id": descriptor.metric_id,
                "native_name": descriptor.native_name,
                "concept_id": descriptor.concept_id,
                "mechanism": descriptor.mechanism,
                "usage_class": usage_by_feature[feature_id],
                "relevance": relevance[feature_id],
                "mean_selected_redundancy": redundancy,
                "mrmr_score": score,
                "semantic_mechanism_match": (
                    descriptor.mechanism in expected_mechanisms
                ),
            }
            if best_details is None or (
                score,
                relevance[feature_id],
                feature_id,
            ) > (
                best_details["mrmr_score"],
                best_details["relevance"],
                best_details["feature_id"],
            ):
                best = feature_id
                best_details = details
        if best is None:
            break
        selected.append(best_details)
        mechanism_counts[best_details["mechanism"]] += 1
        concept_counts[best_details["concept_id"]] += 1
        candidates.remove(best)
    return selected


def _group_stats(output, condition_ids, target_map):
    stats = {}
    pvalues = {}
    condition_id_set = set(condition_ids)
    scoped_target_map = {
        condition_id: value
        for condition_id, value in target_map.items()
        if condition_id in condition_id_set
    }
    for feature_id in output.active_feature_ids:
        feature_map = {
            condition_id: value
            for condition_id, value in output.correlation_report.residual_feature_values[
                feature_id
            ].items()
            if condition_id in condition_id_set
        }
        aligned_ids, feature_values, target_values = aligned_values(
            feature_map, scoped_target_map
        )
        if len(aligned_ids) < max(4, int(math.ceil(len(condition_ids) * 0.8))):
            continue
        if len(set(feature_values)) < 3:
            continue
        rho = spearman(feature_values, target_values)
        pvalue = approximate_correlation_pvalue(rho, len(aligned_ids))
        quality = output.quality_report.metrics[feature_id].quality
        relevance = abs(rho) * math.sqrt(max(0.0, quality.nonzero_rate))
        descriptor = descriptor_for_feature(output, feature_id)
        global_usage = output.correlation_report.associations[
            feature_id
        ].usage_class
        if descriptor.duration_coupled or descriptor.target_equivalent:
            usage_class = "diagnostic_only"
        elif global_usage in {"diagnostic_only", "workload_indicator"}:
            usage_class = global_usage
        elif abs(rho) >= output.context.config.moderate_rho:
            usage_class = "latency_informative_candidate"
        else:
            usage_class = "descriptive_only"
        stats[feature_id] = {
            "sample_count": len(aligned_ids),
            "residual_latency_spearman": rho,
            "approximate_pvalue": pvalue,
            "relevance_score": relevance,
            "usage_class": usage_class,
        }
        pvalues[feature_id] = pvalue
    qvalues = benjamini_hochberg(pvalues)
    for feature_id, values in stats.items():
        values["fdr_qvalue"] = qvalues[feature_id]
    return stats


def _resolved_metric(selection, association, config, small_sample):
    if small_sample:
        evidence = "small_n_descriptive"
    elif (
        association["fdr_qvalue"] <= config.fdr_threshold
        and abs(association["residual_latency_spearman"]) >= config.strong_rho
    ):
        evidence = "fdr_supported_group_association"
    else:
        evidence = "descriptive_group_association"
    return SignatureMetric(
        **selection,
        sample_count=association["sample_count"],
        residual_latency_spearman=association["residual_latency_spearman"],
        approximate_pvalue=association["approximate_pvalue"],
        fdr_qvalue=association["fdr_qvalue"],
        direction=(
            "higher_feature_with_higher_latency"
            if association["residual_latency_spearman"] > 0
            else "higher_feature_with_lower_latency"
        ),
        evidence=evidence,
        selection_reason=(
            "relevance minus redundancy with mechanism/concept diversity"
        ),
    )


def _build_signature(
    output,
    selector,
    condition_ids,
    target_map,
    top_k,
    expected_mechanisms,
):
    config = output.context.config
    if len(condition_ids) < config.min_group_samples:
        return KernelSignature(
            selector=selector,
            status="insufficient_samples",
            sample_count=len(condition_ids),
            minimum_samples=config.min_group_samples,
            concept_slots=(),
            resolved_features=(),
            diagnostic_features=(),
        )
    stats = _group_stats(output, condition_ids, target_map)
    relevance = {
        feature_id: values["relevance_score"]
        for feature_id, values in stats.items()
    }
    selected = _mrmr_select(
        output,
        condition_ids,
        relevance,
        top_k,
        expected_mechanisms,
        EXPLANATION_USAGE,
        {feature_id: values["usage_class"] for feature_id, values in stats.items()},
    )
    diagnostic_selected = _mrmr_select(
        output,
        condition_ids,
        relevance,
        min(top_k, 5),
        expected_mechanisms,
        DIAGNOSTIC_USAGE,
        {feature_id: values["usage_class"] for feature_id, values in stats.items()},
    )
    small_sample = len(condition_ids) < 12
    resolved = tuple(
        _resolved_metric(item, stats[item["feature_id"]], config, small_sample)
        for item in selected
    )
    diagnostics = tuple(
        _resolved_metric(item, stats[item["feature_id"]], config, small_sample)
        for item in diagnostic_selected
    )
    return KernelSignature(
        selector=selector,
        status=(
            "small_n_descriptive_signature"
            if small_sample
            else "exploratory_signature"
        ),
        sample_count=len(condition_ids),
        candidate_feature_count=len(stats),
        concept_slots=tuple(
            dict.fromkeys(item.concept_id for item in resolved)
        ),
        resolved_features=resolved,
        diagnostic_features=diagnostics,
        warning=(
            "signature uses the correlation layer residual target and inherits its evidence limitations; "
            "duration-coupled and sparse metrics are isolated in diagnostic_features"
        ),
    )


def _global_metric(output, selection):
    association = output.correlation_report.associations[selection["feature_id"]]
    return SignatureMetric(
        **selection,
        sample_count=association.sample_count,
        residual_latency_spearman=association.residual_log_latency_spearman,
        approximate_pvalue=association.approximate_pvalue,
        fdr_qvalue=association.fdr_qvalue,
        direction=(
            "higher_feature_with_higher_latency"
            if association.residual_log_latency_spearman > 0
            else "higher_feature_with_lower_latency"
        ),
        evidence=association.statistical_relation_class,
        selection_reason="global constrained mRMR over correlation relevance",
    )


class KernelSignatureLayer(AnalysisLayer):
    name = "kernel_signatures"
    input_type = CorrelationOutput

    def run(self, layer_input):
        self.require_input(layer_input)
        config = layer_input.context.config
        target_map = layer_input.correlation_report.baseline.residuals
        global_relevance = {
            feature_id: association.relevance_score
            for feature_id, association in layer_input.correlation_report.associations.items()
        }
        global_selected = tuple(
            _global_metric(layer_input, item)
            for item in _mrmr_select(
                layer_input,
                layer_input.context.active_condition_ids,
                global_relevance,
                config.global_top_k,
                allowed_usage=EXPLANATION_USAGE,
            )
        )
        global_diagnostic = tuple(
            _global_metric(layer_input, item)
            for item in _mrmr_select(
                layer_input,
                layer_input.context.active_condition_ids,
                global_relevance,
                min(config.global_top_k, 8),
                allowed_usage=DIAGNOSTIC_USAGE,
            )
        )

        selectors = {
            "semantic_family": defaultdict(list),
            "execution_role": defaultdict(list),
            "semantic_role": defaultdict(list),
            "op_type": defaultdict(list),
            "performance_regime": defaultdict(list),
        }
        for condition_id in layer_input.context.active_condition_ids:
            condition = layer_input.context.dataset.conditions[condition_id]
            selectors["semantic_family"][condition.semantic_family].append(
                condition_id
            )
            selectors["execution_role"][condition.execution_role].append(condition_id)
            selectors["semantic_role"][
                "{}::{}".format(
                    condition.semantic_family, condition.execution_role
                )
            ].append(condition_id)
            selectors["op_type"][condition.op_type].append(condition_id)
            if condition.performance_regime != "unknown":
                selectors["performance_regime"][condition.performance_regime].append(
                    condition_id
                )

        signature_index = {}
        for axis, groups in selectors.items():
            signature_index[axis] = {}
            for value, condition_ids in sorted(groups.items()):
                signature_index[axis][value] = _build_signature(
                    layer_input,
                    {axis: value},
                    condition_ids,
                    target_map,
                    config.group_top_k,
                    _expected_mechanisms(layer_input, condition_ids),
                )

        for op_type, signature in list(signature_index["op_type"].items()):
            if signature.status != "insufficient_samples":
                continue
            condition_ids = selectors["op_type"][op_type]
            family = layer_input.context.dataset.conditions[
                condition_ids[0]
            ].semantic_family
            inherited = signature_index["semantic_family"].get(family)
            if inherited is None:
                continue
            signature_index["op_type"][op_type] = signature.model_copy(
                update={
                    "inherited_family_concept_slots": inherited.concept_slots,
                    "inherited_family_features": tuple(
                        item.feature_id for item in inherited.resolved_features
                    ),
                }
            )

        metric_to_classes = defaultdict(list)
        for axis, groups in signature_index.items():
            for value, signature in groups.items():
                for item in signature.resolved_features + signature.diagnostic_features:
                    metric_to_classes[item.feature_id].append(
                        MetricKernelClassLink(
                            axis=axis,
                            value=value,
                            rho=item.residual_latency_spearman or 0.0,
                            evidence=item.evidence,
                            usage_class=item.usage_class,
                        )
                    )
        normalized_reverse = {}
        for feature_id, values in metric_to_classes.items():
            normalized_reverse[feature_id] = tuple(
                sorted(
                    values,
                    key=lambda item: (-abs(item.rho), item.axis, item.value),
                )
            )

        report = KernelSignatureReport(
            descriptive_canonical_features=(
                layer_input.redundancy_report.descriptive_canonical_features
            ),
            latency_association_set=global_selected,
            diagnostic_association_set=global_diagnostic,
            latency_association_set_evidence=(
                layer_input.correlation_report.evidence_level
            ),
            signature_index=signature_index,
            metric_to_kernel_classes=dict(sorted(normalized_reverse.items())),
            selection_method=SignatureSelectionMethod(
                name="engineering_constrained_mrmr",
                relevance="group residual-latency Spearman weighted by activation",
                redundancy="mean absolute Spearman to already selected features",
                constraints=(
                    "max two per canonical concept",
                    "max four per mechanism",
                    "diagnostic-only metrics are isolated from latency explanation slots",
                ),
            ),
            important_distinction=(
                "descriptive canonical features are selected without latency; the latency association set is a "
                "separate supervised exploratory product; diagnostic metrics are not optimization-effect metrics"
            ),
        )
        return KernelSignatureOutput(
            context=layer_input.context,
            active_feature_ids=layer_input.active_feature_ids,
            projection_report=layer_input.projection_report,
            layer_history=append_layer_record(
                layer_input,
                self.name,
                layer_input.active_feature_ids,
                (report.important_distinction,),
            ),
            quality_report=layer_input.quality_report,
            redundancy_report=layer_input.redundancy_report,
            correlation_report=layer_input.correlation_report,
            signature_report=report,
        )
