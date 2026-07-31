"""将严格类型的串行层报告编译为最终 PMC 分析报告。"""

from __future__ import annotations

from ..dataset.reports import (
    AnalysisProvenance,
    AnalysisScope,
    CanonicalMetricSet,
    CanonicalMetricSets,
    DeltaRelationOutput,
    EvidenceIndex,
    LayerReportBundle,
    MetricKnowledge,
    PmcAnalysisReport,
)


def compile_analysis_report(layer_output):
    if type(layer_output) is not DeltaRelationOutput:
        raise TypeError(
            "compile_analysis_report expects DeltaRelationOutput, got {}".format(
                type(layer_output).__name__
            )
        )
    context = layer_output.context
    used_device_ids = tuple(
        sorted(
            {
                condition.device_id
                for condition in context.dataset.conditions.values()
            }
        )
    )
    selected_features = {
        item.feature_id for item in layer_output.signature_report.latency_association_set
    }
    metric_knowledge = {}
    for feature_id, spec in sorted(context.features.specs.items()):
        metric_id = spec.source_metric_ids[0]
        metric_knowledge[feature_id] = MetricKnowledge(
            feature_spec=spec,
            descriptor=context.dataset.metric_catalog[metric_id],
            quality=layer_output.quality_report.metrics.get(feature_id),
            redundancy=layer_output.redundancy_report.metrics.get(feature_id),
            association=layer_output.correlation_report.associations.get(feature_id),
            delta_relation=layer_output.delta_report.metric_relations.get(feature_id),
            selected_in_latency_association_set=feature_id in selected_features,
            kernel_class_links=layer_output.signature_report.metric_to_kernel_classes.get(
                feature_id, ()
            ),
        )

    readiness = layer_output.projection_report.data_readiness
    warnings = list(context.issues)
    if readiness.missing_high_value_fields:
        warnings.append(
            "missing high-value controls: {}".format(
                ", ".join(readiness.missing_high_value_fields)
            )
        )
    if layer_output.correlation_report.evidence_level == "descriptive_cross_kernel":
        warnings.append(
            "latency associations are uncontrolled cross-kernel descriptions"
        )
    if layer_output.delta_report.status == "insufficient_paired_variant_evidence":
        warnings.append(
            "no delta effect rule is generated from cross-kernel correlation"
        )
    if not layer_output.delta_report.effect_rules:
        warnings.append(
            "the current dataset does not support a stable delta-PMC/delta-latency rule"
        )
    if any(
        association.usage_class == "diagnostic_only"
        for association in layer_output.correlation_report.associations.values()
    ):
        warnings.append(
            "diagnostic-only metrics are isolated from latency explanation and optimization effect sets"
        )

    layer_reports = LayerReportBundle(
        projection=layer_output.projection_report,
        quality=layer_output.quality_report,
        redundancy=layer_output.redundancy_report,
        correlation=layer_output.correlation_report,
        kernel_signatures=layer_output.signature_report,
        delta_relations=layer_output.delta_report,
    )
    has_signature_evidence = any(
        signature.resolved_features
        or signature.diagnostic_features
        or signature.inherited_family_features
        for groups in layer_output.signature_report.signature_index.values()
        for signature in groups.values()
    )
    return PmcAnalysisReport(
        scope=AnalysisScope(
            dataset_id=context.dataset.manifest.dataset_id,
            platform=context.dataset.manifest.platform,
            backend=context.dataset.manifest.backend,
            device_ids=used_device_ids,
            cross_platform_policy=(
                "native metrics remain device-scoped; cross-platform alignment uses concept_id, "
                "mapping_level, and mapping_confidence"
            ),
        ),
        data_readiness=readiness,
        layer_history=layer_output.layer_history,
        layer_reports=layer_reports,
        metric_knowledge=metric_knowledge,
        canonical_metric_sets=CanonicalMetricSets(
            descriptive_canonical=CanonicalMetricSet(
                features=(
                    layer_output.redundancy_report.descriptive_canonical_features
                ),
                selection_basis=(
                    "semantic scope, coverage, repeatability, rollup, and same-concept redundancy"
                ),
                uses_latency=False,
                evidence_level="measurement_quality_and_semantic_canonicalization",
            ),
            latency_association=CanonicalMetricSet(
                features=tuple(
                    item.feature_id
                    for item in layer_output.signature_report.latency_association_set
                ),
                selection_basis=(
                    "constrained mRMR over workload/shape-controlled correlation evidence"
                ),
                uses_latency=True,
                evidence_level=(
                    layer_output.signature_report.latency_association_set_evidence
                ),
            ),
        ),
        kernel_signatures=layer_output.signature_report,
        delta_rulebook=layer_output.delta_report.effect_rules,
        evidence_index=EvidenceIndex(
            correlation_level=layer_output.correlation_report.evidence_level,
            correlation_method=layer_output.correlation_report.method,
            kernel_signatures_status=(
                "available" if has_signature_evidence else "not_estimable"
            ),
            kernel_signatures_limit=(
                "group signatures inherit correlation-layer controls and sample-size limits"
            ),
            delta_relations_status=layer_output.delta_report.status,
            valid_delta_pair_count=layer_output.delta_report.valid_pair_count,
            independent_delta_pair_group_count=(
                layer_output.delta_report.independent_pair_group_count
            ),
        ),
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=AnalysisProvenance(
            manifest=context.dataset.manifest,
            devices={
                device_id: context.dataset.devices[device_id]
                for device_id in used_device_ids
            },
            dataset_schema_version=context.dataset.schema_version,
            analysis_config=context.config,
        ),
    )
