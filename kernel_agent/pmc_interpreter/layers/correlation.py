"""控制工作量、shape 与 launch 后的 PMC–latency 条件关联层。"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from ..dataset.reports import (
    CorrelationOutput,
    CorrelationReport,
    LatencyBaselineReport,
    MetricAssociation,
    RedundancyOutput,
)
from .base import (
    AnalysisLayer,
    aligned_values,
    append_layer_record,
    descriptor_for_feature,
    feature_condition_medians,
    log_latency_map,
    transformed_feature_map,
)
from .statistics import (
    approximate_correlation_pvalue,
    benjamini_hochberg,
    cross_fitted_ridge,
    distance_correlation,
    r_squared,
    spearman,
)


def _record_numeric_controls(target, prefix, values):
    for key, raw in values.items():
        name = "{}.{}".format(prefix, key)
        if isinstance(raw, bool):
            target[name] = float(raw)
        elif isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            target[name] = float(raw)
        elif (
            isinstance(raw, (list, tuple))
            and raw
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in raw
            )
        ):
            product = 1.0
            for item in raw:
                product *= float(item)
            target[name + "_product"] = product


def _numeric_controls(output, condition_ids):
    per_condition = {condition_id: {} for condition_id in condition_ids}
    for condition_id in condition_ids:
        condition = output.context.dataset.conditions[condition_id]
        for prefix, values in (
            ("workload", condition.workload),
            ("shape", condition.shapes),
            ("param", condition.params),
            ("launch", condition.launch),
        ):
            _record_numeric_controls(per_condition[condition_id], prefix, values)
    environment = defaultdict(lambda: defaultdict(list))
    for run in output.context.dataset.runs.values():
        if run.condition_id not in per_condition:
            continue
        if run.gpu_clock_hz is not None:
            environment[run.condition_id]["environment.gpu_clock_hz"].append(
                float(run.gpu_clock_hz)
            )
        if run.temperature_c is not None:
            environment[run.condition_id]["environment.temperature_c"].append(
                float(run.temperature_c)
            )
    for condition_id, values in environment.items():
        for key, items in values.items():
            per_condition[condition_id][key] = statistics.fmean(items)
    return per_condition


def _build_design(output, condition_ids):
    numeric = _numeric_controls(output, condition_ids)
    minimum_presence = max(3, int(math.ceil(len(condition_ids) * 0.5)))
    all_keys = sorted({key for values in numeric.values() for key in values})
    numeric_keys = []
    numeric_specs = {}
    for key in all_keys:
        observed = [
            numeric[condition_id][key]
            for condition_id in condition_ids
            if key in numeric[condition_id]
        ]
        if len(observed) < minimum_presence or len(set(observed)) <= 1:
            continue
        use_log = all(value >= 0 for value in observed) and any(
            token in key.lower()
            for token in (
                "flops",
                "bytes",
                "elements",
                "grid",
                "block",
                "work",
                "size",
                "shape",
                "param",
                "dimension",
            )
        )
        transformed = [math.log1p(value) for value in observed] if use_log else observed
        mean = statistics.fmean(transformed)
        stddev = statistics.pstdev(transformed) if len(transformed) > 1 else 1.0
        numeric_keys.append(key)
        numeric_specs[key] = (
            mean,
            max(stddev, 1e-12),
            len(observed) < len(condition_ids),
            use_log,
        )

    category_values = {
        "semantic_family": [
            output.context.dataset.conditions[item].semantic_family
            for item in condition_ids
        ],
        "execution_role": [
            output.context.dataset.conditions[item].execution_role
            for item in condition_ids
        ],
        "op_type": [
            output.context.dataset.conditions[item].op_type for item in condition_ids
        ],
        "dtype": [
            output.context.dataset.conditions[item].dtype for item in condition_ids
        ],
    }
    category_levels = {}
    for name, values in category_values.items():
        counts = Counter(values)
        retained = [level for level in sorted(counts) if counts[level] >= 2]
        category_levels[name] = retained[1:]

    feature_names = ["intercept"]
    for key in numeric_keys:
        feature_names.append(key)
        if numeric_specs[key][2]:
            feature_names.append(key + ".missing")
    for category, levels in category_levels.items():
        feature_names.extend(
            "{}.{}".format(category, level) for level in levels
        )

    design = []
    for condition_id in condition_ids:
        row = [1.0]
        for key in numeric_keys:
            mean, stddev, has_missing, use_log = numeric_specs[key]
            if key in numeric[condition_id]:
                value = numeric[condition_id][key]
                value = math.log1p(value) if use_log else value
                row.append((value - mean) / stddev)
                if has_missing:
                    row.append(0.0)
            else:
                row.append(0.0)
                if has_missing:
                    row.append(1.0)
        condition = output.context.dataset.conditions[condition_id]
        category_current = {
            "semantic_family": condition.semantic_family,
            "execution_role": condition.execution_role,
            "op_type": condition.op_type,
            "dtype": condition.dtype,
        }
        for category, levels in category_levels.items():
            row.extend(
                1.0 if category_current[category] == level else 0.0
                for level in levels
            )
        design.append(row)
    return design, feature_names, numeric_keys


def build_latency_baseline(output):
    latency = log_latency_map(output.context.dataset)
    condition_ids = [
        condition_id
        for condition_id in output.context.active_condition_ids
        if condition_id in latency
    ]
    target = [latency[condition_id] for condition_id in condition_ids]
    if not target:
        return LatencyBaselineReport(
            condition_ids=(),
            predictions={},
            residuals={},
            evidence_level="not_estimable",
            cross_validated_r_squared=0.0,
            controls=(),
            missing_high_value_controls=(
                "workload.algorithmic_flops",
                "workload.algorithmic_bytes",
                "workload.output_elements",
                "launch.grid",
                "launch.block",
                "environment.gpu_clock_hz",
                "environment.temperature_c",
            ),
        )
    design, feature_names, numeric_keys = _build_design(output, condition_ids)
    predictions = cross_fitted_ridge(
        design, target, condition_ids, folds=5, penalty=1.0
    )
    score = r_squared(target, predictions)
    global_mean = statistics.fmean(target)
    if score <= 0:
        predictions = [global_mean for _ in target]
        score = 0.0
        evidence = "descriptive_cross_kernel"
        controls = ()
    else:
        strong_workload_keys = {
            "workload.algorithmic_flops",
            "workload.algorithmic_bytes",
            "workload.output_elements",
        }
        if strong_workload_keys & set(numeric_keys):
            evidence = "workload_controlled_association"
        elif any(
            key.startswith(("shape.", "param.", "launch."))
            for key in numeric_keys
        ):
            evidence = "shape_proxy_controlled_association"
        else:
            evidence = "proxy_controlled_association"
        controls = tuple(feature_names[1:])
    prediction_map = dict(zip(condition_ids, predictions))
    residuals = {
        condition_id: latency[condition_id] - prediction_map[condition_id]
        for condition_id in condition_ids
    }
    return LatencyBaselineReport(
        condition_ids=tuple(condition_ids),
        predictions=prediction_map,
        residuals=residuals,
        evidence_level=evidence,
        cross_validated_r_squared=score,
        controls=controls,
        missing_high_value_controls=tuple(
            key
            for key in (
                "workload.algorithmic_flops",
                "workload.algorithmic_bytes",
                "workload.output_elements",
                "launch.grid",
                "launch.block",
                "environment.gpu_clock_hz",
                "environment.temperature_c",
            )
            if key not in feature_names
        ),
    )


def _residualize_feature(output, condition_ids, feature_values, baseline_evidence):
    if baseline_evidence in {"not_estimable", "descriptive_cross_kernel"}:
        return feature_values, False, 0.0
    design, _, _ = _build_design(output, condition_ids)
    predictions = cross_fitted_ridge(
        design, feature_values, condition_ids, folds=5, penalty=1.0
    )
    score = max(0.0, r_squared(feature_values, predictions))
    residuals = [
        value - prediction
        for value, prediction in zip(feature_values, predictions)
    ]
    return residuals, True, score


def _classify(
    descriptor,
    raw_rho,
    residual_rho,
    residual_dcor,
    qvalue,
    sample_count,
    nonzero_count,
    baseline_evidence,
    config,
):
    if sample_count < 8:
        return "insufficient_evidence"
    if descriptor.target_equivalent:
        return "target_equivalent"
    if descriptor.duration_coupled:
        return "target_coupled"
    if nonzero_count < max(5, int(math.ceil(sample_count * 0.1))):
        return "sparse_activation"
    if baseline_evidence == "descriptive_cross_kernel":
        if qvalue <= config.fdr_threshold and abs(raw_rho) >= config.strong_rho:
            return "raw_monotonic_descriptive"
        if (
            residual_dcor >= config.nonlinear_distance_correlation
            and abs(raw_rho) < config.moderate_rho
        ):
            return "raw_nonlinear_descriptive_candidate"
        if abs(raw_rho) >= config.moderate_rho:
            return "raw_context_descriptive_candidate"
        return "weak_or_none"
    if (
        abs(raw_rho) >= config.strong_rho
        and abs(residual_rho) < 0.2
    ):
        return "workload_indicator"
    if qvalue <= config.fdr_threshold and abs(residual_rho) >= config.strong_rho:
        return "residual_monotonic"
    if (
        residual_dcor >= config.nonlinear_distance_correlation
        and abs(residual_rho) < config.moderate_rho
    ):
        return "residual_nonlinear_candidate"
    if abs(residual_rho) >= config.moderate_rho:
        return "moderate_context_candidate"
    return "weak_or_none"


def _usage_class(relation_class):
    if relation_class in {"target_equivalent", "target_coupled", "sparse_activation"}:
        return "diagnostic_only"
    if relation_class in {
        "residual_monotonic",
        "residual_nonlinear_candidate",
        "moderate_context_candidate",
    }:
        return "latency_informative_candidate"
    if relation_class in {
        "raw_monotonic_descriptive",
        "raw_nonlinear_descriptive_candidate",
        "raw_context_descriptive_candidate",
    }:
        return "descriptive_association"
    if relation_class == "workload_indicator":
        return "workload_indicator"
    return "descriptive_only"


def _explain(descriptor, relation_class, rho, evidence):
    if relation_class in {"target_equivalent", "target_coupled"}:
        return "Useful for diagnosis, but elapsed duration is present in the event or formula."
    if relation_class == "workload_indicator":
        return "Raw latency correlation largely disappears after available workload controls."
    if relation_class == "sparse_activation":
        return "The event is sparse and should be treated as a conditional diagnostic trigger."
    if relation_class == "residual_nonlinear_candidate":
        return (
            "Distance correlation indicates a possible nonlinear or threshold relationship while the monotonic "
            "relationship is weak; inspect a scoped effect curve before assigning direction."
        )
    if relation_class == "raw_nonlinear_descriptive_candidate":
        return (
            "An uncontrolled cross-kernel nonlinear pattern is visible; no conditional direction or optimization "
            "meaning is established."
        )
    if relation_class in {
        "raw_monotonic_descriptive",
        "raw_context_descriptive_candidate",
        "residual_monotonic",
        "moderate_context_candidate",
    }:
        direction = "higher" if rho > 0 else "lower"
        return (
            "{} values accompany higher latency residuals under {} evidence; this is association, not an "
            "optimization direction rule."
        ).format(direction, evidence)
    return "No stable monotonic latency relationship is visible at the current evidence level."


class CorrelationLayer(AnalysisLayer):
    name = "correlation"
    input_type = RedundancyOutput

    def run(self, layer_input):
        self.require_input(layer_input)
        config = layer_input.context.config
        baseline = build_latency_baseline(layer_input)
        raw_latency = log_latency_map(layer_input.context.dataset)
        residual_latency = baseline.residuals
        partial = {}
        pvalues = {}
        residual_feature_values = {}

        for feature_id in layer_input.active_feature_ids:
            transformed = transformed_feature_map(layer_input, feature_id)
            condition_ids = sorted(
                set(transformed) & set(raw_latency) & set(residual_latency)
            )
            feature_values = [transformed[item] for item in condition_ids]
            raw_target = [raw_latency[item] for item in condition_ids]
            residual_target = [residual_latency[item] for item in condition_ids]
            if len(condition_ids) < 3:
                raw_rho = residual_rho = 0.0
                residual_dcor = 0.0
                pvalue = 1.0
                feature_residualized = False
                feature_baseline_r2 = 0.0
                residual_feature = feature_values
            else:
                raw_rho = spearman(feature_values, raw_target)
                residual_feature, feature_residualized, feature_baseline_r2 = (
                    _residualize_feature(
                        layer_input,
                        condition_ids,
                        feature_values,
                        baseline.evidence_level,
                    )
                )
                residual_rho = spearman(residual_feature, residual_target)
                residual_dcor = distance_correlation(
                    residual_feature, residual_target
                )
                pvalue = approximate_correlation_pvalue(
                    residual_rho, len(condition_ids)
                )
            residual_feature_values[feature_id] = dict(
                zip(condition_ids, residual_feature)
            )
            raw_values = feature_condition_medians(
                layer_input, feature_id, condition_ids
            )
            presence = [
                1.0 if raw_values[condition_id] != 0 else 0.0
                for condition_id in condition_ids
            ]
            presence_rho = (
                spearman(presence, residual_target)
                if len(set(presence)) > 1 and residual_target
                else 0.0
            )
            quality = layer_input.quality_report.metrics[feature_id].quality
            relevance = abs(residual_rho) * math.sqrt(
                max(0.0, quality.nonzero_rate)
            )
            partial[feature_id] = {
                "feature_id": feature_id,
                "sample_count": len(condition_ids),
                "raw_log_latency_spearman": raw_rho,
                "residual_log_latency_spearman": residual_rho,
                "residual_log_latency_distance_correlation": residual_dcor,
                "presence_residual_spearman": presence_rho,
                "feature_residualized": feature_residualized,
                "feature_baseline_r_squared": feature_baseline_r2,
                "approximate_pvalue": pvalue,
                "relevance_score": relevance,
            }
            pvalues[feature_id] = pvalue

        qvalues = benjamini_hochberg(pvalues)
        associations = {}
        for feature_id, values in partial.items():
            descriptor = descriptor_for_feature(layer_input, feature_id)
            quality = layer_input.quality_report.metrics[feature_id].quality
            relation_class = _classify(
                descriptor,
                values["raw_log_latency_spearman"],
                values["residual_log_latency_spearman"],
                values["residual_log_latency_distance_correlation"],
                qvalues[feature_id],
                values["sample_count"],
                quality.nonzero_conditions,
                baseline.evidence_level,
                config,
            )
            associations[feature_id] = MetricAssociation(
                **values,
                fdr_qvalue=qvalues[feature_id],
                statistical_relation_class=relation_class,
                usage_class=_usage_class(relation_class),
                explanation=_explain(
                    descriptor,
                    relation_class,
                    values["residual_log_latency_spearman"],
                    baseline.evidence_level,
                ),
            )

        ranked_ids = sorted(
            associations,
            key=lambda feature_id: (
                -associations[feature_id].relevance_score,
                feature_id,
            ),
        )
        report = CorrelationReport(
            evidence_level=baseline.evidence_level,
            baseline=baseline,
            residual_feature_values=residual_feature_values,
            associations=associations,
            top_associations=tuple(
                associations[feature_id] for feature_id in ranked_ids[:50]
            ),
            method=(
                "cross-fitted ridge residualization of log latency and each PMC feature; "
                "Spearman association with Fisher-z p approximation and Benjamini-Hochberg FDR; "
                "distance correlation is an exploratory nonlinear-dependence screen"
            ),
            limitations=(
                "integer counters contain many ties",
                "association does not establish optimization causality",
                "missing workload or environment controls lower the evidence level",
                "distance correlation has no permutation p-value in the current implementation",
            ),
        )
        return CorrelationOutput(
            context=layer_input.context,
            active_feature_ids=layer_input.active_feature_ids,
            projection_report=layer_input.projection_report,
            layer_history=append_layer_record(
                layer_input,
                self.name,
                layer_input.active_feature_ids,
                (report.method,),
            ),
            quality_report=layer_input.quality_report,
            redundancy_report=layer_input.redundancy_report,
            correlation_report=report,
        )
