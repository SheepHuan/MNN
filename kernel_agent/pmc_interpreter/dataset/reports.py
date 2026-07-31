"""PMC Interpreter 特征、分析层契约和最终报告模型。"""

from __future__ import annotations

from pydantic import Field, JsonValue, model_validator

from .model import (
    DatasetManifest,
    DeviceSpec,
    MetricDescriptor,
    PmcDataset,
    PmcModel,
)


ANALYSIS_REPORT_SCHEMA_VERSION = "mnn-pmc-analysis-report/v1"


class AnalysisConfig(PmcModel):
    min_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    min_unique_values: int = Field(default=3, ge=1)
    min_group_samples: int = Field(default=5, ge=2)
    min_delta_pairs: int = Field(default=6, ge=2)
    global_top_k: int = Field(default=15, ge=1)
    group_top_k: int = Field(default=10, ge=1)
    delta_top_k: int = Field(default=30, ge=1)
    fdr_threshold: float = Field(default=0.1, gt=0.0, le=1.0)
    strong_rho: float = Field(default=0.5, ge=0.0, le=1.0)
    moderate_rho: float = Field(default=0.3, ge=0.0, le=1.0)
    nonlinear_distance_correlation: float = Field(default=0.5, ge=0.0, le=1.0)
    redundancy_weight: float = Field(default=0.35, ge=0.0)
    max_repeat_relative_mad: float = Field(default=0.1, ge=0.0)


class FeatureSpec(PmcModel):
    feature_id: str
    source_metric_ids: tuple[str, ...]
    concept_id: str
    expression: str
    transform: str
    transform_scale: float | None
    duration_coupled: bool
    comparability: str
    value_semantics: tuple[str, ...]


class FeatureStore(PmcModel):
    specs: dict[str, FeatureSpec]
    values: dict[str, dict[str, tuple[float, ...]]]

    @model_validator(mode="after")
    def validate_feature_keys(self) -> "FeatureStore":
        if set(self.specs) != set(self.values):
            raise ValueError("feature specs and values must have identical feature IDs")
        for feature_id, spec in self.specs.items():
            if spec.feature_id != feature_id:
                raise ValueError("feature key/id mismatch: {}".format(feature_id))
        return self


class QuestionReadiness(PmcModel):
    correlation_and_classification: str
    kernel_type_metric_sets: str
    delta_pmc_delta_latency: str


class DataReadiness(PmcModel):
    device_count: int
    condition_count: int
    metric_count: int
    metric_observation_count: int
    latency_condition_count: int
    comparison_pair_count: int
    valid_comparison_pair_count: int
    valid_comparison_group_count: int
    op_type_count: int
    op_types_with_multiple_conditions: int
    semantic_family_counts: dict[str, int]
    execution_role_counts: dict[str, int]
    pmc_conditions_with_independent_repeats: int
    latency_conditions_with_independent_repeats: int
    available_workload_fields: tuple[str, ...]
    available_launch_fields: tuple[str, ...]
    workload_field_coverage: dict[str, float]
    launch_field_coverage: dict[str, float]
    gpu_clock_condition_coverage: float
    temperature_condition_coverage: float
    missing_high_value_fields: tuple[str, ...]
    question_readiness: QuestionReadiness


class AnalysisContext(PmcModel):
    dataset: PmcDataset
    config: AnalysisConfig
    features: FeatureStore
    active_condition_ids: tuple[str, ...]
    issues: tuple[str, ...]


class LayerRecord(PmcModel):
    layer: str
    input_feature_count: int
    output_feature_count: int
    notes: tuple[str, ...] = ()


class ProjectionInput(PmcModel):
    dataset: PmcDataset
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)


class ProjectionReport(PmcModel):
    input_metric_count: int
    projected_feature_count: int
    valid_metric_observation_count: int
    ignored_metric_observation_count: int
    transform_counts: dict[str, int]
    data_readiness: DataReadiness
    notes: tuple[str, ...]


class ProjectionOutput(PmcModel):
    context: AnalysisContext
    active_feature_ids: tuple[str, ...]
    projection_report: ProjectionReport
    layer_history: tuple[LayerRecord, ...]


class MetricQuality(PmcModel):
    observed_conditions: int
    coverage: float
    unique_values: int
    nonzero_conditions: int
    nonzero_rate: float
    cross_condition_mean: float
    cross_condition_stddev: float
    cross_condition_cv: float | None
    conditions_with_repeats: int
    repeat_relative_mad: float | None
    repeat_semantics: str


class MetricQualityDecision(PmcModel):
    feature_id: str
    quality: MetricQuality
    observation_status_counts: dict[str, int]
    canonical_rollup_representative: str
    disposition: str
    exclusion_reasons: tuple[str, ...]


class QualityReport(PmcModel):
    input_feature_count: int
    active_feature_count: int
    native_basename_expression_groups: int
    exclusion_counts: dict[str, int]
    descriptive_canonical_candidates: tuple[str, ...]
    metrics: dict[str, MetricQualityDecision]
    notes: tuple[str, ...]


class QualityOutput(ProjectionOutput):
    quality_report: QualityReport


class RedundancyDecision(PmcModel):
    feature_id: str
    observational_cluster_id: str
    same_concept_representative: str
    included_after_same_concept_dedup: bool


class RedundancyCluster(PmcModel):
    cluster_id: str
    features: tuple[str, ...]
    concepts: tuple[str, ...]
    disposition: str


class RedundancyReport(PmcModel):
    input_feature_count: int
    active_feature_count: int
    observational_cluster_count: int
    cross_concept_equal_vector_clusters: tuple[RedundancyCluster, ...]
    metrics: dict[str, RedundancyDecision]
    policy: str
    descriptive_canonical_features: tuple[str, ...]


class RedundancyOutput(QualityOutput):
    redundancy_report: RedundancyReport


class LatencyBaselineReport(PmcModel):
    condition_ids: tuple[str, ...]
    predictions: dict[str, float]
    residuals: dict[str, float]
    evidence_level: str
    cross_validated_r_squared: float
    controls: tuple[str, ...]
    missing_high_value_controls: tuple[str, ...]


class MetricAssociation(PmcModel):
    feature_id: str
    sample_count: int
    raw_log_latency_spearman: float
    residual_log_latency_spearman: float
    residual_log_latency_distance_correlation: float
    presence_residual_spearman: float
    feature_residualized: bool
    feature_baseline_r_squared: float
    approximate_pvalue: float
    fdr_qvalue: float
    relevance_score: float
    statistical_relation_class: str
    usage_class: str
    explanation: str


class CorrelationReport(PmcModel):
    evidence_level: str
    baseline: LatencyBaselineReport
    residual_feature_values: dict[str, dict[str, float]]
    associations: dict[str, MetricAssociation]
    top_associations: tuple[MetricAssociation, ...]
    method: str
    limitations: tuple[str, ...]


class CorrelationOutput(RedundancyOutput):
    correlation_report: CorrelationReport


class SignatureMetric(PmcModel):
    feature_id: str
    metric_id: str
    native_name: str
    concept_id: str
    mechanism: str
    usage_class: str
    relevance: float
    mean_selected_redundancy: float
    mrmr_score: float
    semantic_mechanism_match: bool
    sample_count: int | None = None
    residual_latency_spearman: float | None = None
    approximate_pvalue: float | None = None
    fdr_qvalue: float | None = None
    direction: str = ""
    evidence: str = ""
    selection_reason: str = ""


class KernelSignature(PmcModel):
    selector: dict[str, str]
    status: str
    sample_count: int
    minimum_samples: int | None = None
    candidate_feature_count: int = 0
    concept_slots: tuple[str, ...]
    resolved_features: tuple[SignatureMetric, ...]
    diagnostic_features: tuple[SignatureMetric, ...]
    inherited_family_concept_slots: tuple[str, ...] = ()
    inherited_family_features: tuple[str, ...] = ()
    warning: str = ""


class MetricKernelClassLink(PmcModel):
    axis: str
    value: str
    rho: float
    evidence: str
    usage_class: str


class SignatureSelectionMethod(PmcModel):
    name: str
    relevance: str
    redundancy: str
    constraints: tuple[str, ...]


class KernelSignatureReport(PmcModel):
    descriptive_canonical_features: tuple[str, ...]
    latency_association_set: tuple[SignatureMetric, ...]
    diagnostic_association_set: tuple[SignatureMetric, ...]
    latency_association_set_evidence: str
    signature_index: dict[str, dict[str, KernelSignature]]
    metric_to_kernel_classes: dict[str, tuple[MetricKernelClassLink, ...]]
    selection_method: SignatureSelectionMethod
    important_distinction: str


class KernelSignatureOutput(CorrelationOutput):
    signature_report: KernelSignatureReport


class RejectedComparison(PmcModel):
    pair_id: str
    reason: str


class DeltaRelationStatistic(PmcModel):
    feature_id: str
    pair_count: int
    independent_pair_group_count: int
    contributing_pair_ids: tuple[str, ...]
    zero_delta_pair_count: int
    relation_class: str
    usage_class: str
    approximate_pvalue: float | None = None
    fdr_qvalue: float | None = None
    delta_spearman: float | None = None
    transformed_slope: float | None = None
    transformed_slope_ci95: tuple[float, float] | None = None
    positive_direction_concordance: float | None = None
    negative_direction_concordance: float | None = None
    directional_concordance: float | None = None
    effect_score: float | None = None
    delta_collinear_group: str = ""
    exclusion_reason: str = ""


class ObservedNumericRange(PmcModel):
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def validate_order(self) -> "ObservedNumericRange":
        if self.minimum > self.maximum:
            raise ValueError("observed range minimum must not exceed maximum")
        return self


class DeltaScopeCondition(PmcModel):
    condition_id: str
    comparison_role: str
    device_id: str
    semantic_equivalence_key: str
    semantic_family: str
    op_type: str
    execution_role: str
    performance_regime: str
    dtype: str
    implementation_id: str
    shapes: dict[str, JsonValue]
    params: dict[str, JsonValue]
    workload: dict[str, JsonValue]
    launch: dict[str, JsonValue]


class DeltaScopeSelector(PmcModel):
    device_ids: tuple[str, ...]
    semantic_families: tuple[str, ...]
    conditions: tuple[DeltaScopeCondition, ...]
    intervention_ids: tuple[str, ...]
    controlled_mechanisms: tuple[str, ...]


class DeltaObservedSupport(PmcModel):
    raw_baseline_feature: ObservedNumericRange
    raw_candidate_feature: ObservedNumericRange
    raw_feature_delta: ObservedNumericRange
    transformed_baseline_feature: ObservedNumericRange
    transformed_candidate_feature: ObservedNumericRange
    transformed_feature_delta: ObservedNumericRange
    latency_ratio: ObservedNumericRange
    delta_log_latency: ObservedNumericRange


class DeltaRule(PmcModel):
    rule_id: str
    scope_selector: DeltaScopeSelector
    feature_id: str
    metric_id: str
    native_name: str
    concept_id: str
    mechanism: str
    metric_delta_definition: str
    latency_delta_definition: str
    observed_support: DeltaObservedSupport
    statistics: DeltaRelationStatistic
    evidence_level: str
    causal_claim_allowed: bool
    interpretation: str


class DeltaCollinearGroup(PmcModel):
    group_id: str
    features: tuple[str, ...]
    warning: str


class FamilyDeltaReadiness(PmcModel):
    pair_count: int
    independent_pair_group_count: int
    status: str


class DeltaRelationReport(PmcModel):
    status: str
    valid_pair_count: int
    valid_pair_ids: tuple[str, ...]
    independent_pair_group_count: int
    minimum_pairs: int
    rejected_pairs: tuple[RejectedComparison, ...]
    metric_relations: dict[str, DeltaRelationStatistic]
    candidate_relations: tuple[DeltaRule, ...]
    effect_rules: tuple[DeltaRule, ...]
    global_effect_rules: tuple[DeltaRule, ...]
    semantic_family_effect_rules: dict[str, tuple[DeltaRule, ...]]
    delta_collinear_groups: tuple[DeltaCollinearGroup, ...]
    semantic_family_readiness: dict[str, FamilyDeltaReadiness]
    required_design: tuple[str, ...]
    warning: str
    multiple_testing: str
    limitations: tuple[str, ...]


class DeltaRelationOutput(KernelSignatureOutput):
    delta_report: DeltaRelationReport


class AnalysisScope(PmcModel):
    dataset_id: str
    platform: str
    backend: str
    device_ids: tuple[str, ...]
    cross_platform_policy: str


class CanonicalMetricSet(PmcModel):
    features: tuple[str, ...]
    selection_basis: str
    uses_latency: bool
    evidence_level: str


class CanonicalMetricSets(PmcModel):
    descriptive_canonical: CanonicalMetricSet
    latency_association: CanonicalMetricSet


class MetricKnowledge(PmcModel):
    feature_spec: FeatureSpec
    descriptor: MetricDescriptor
    quality: MetricQualityDecision | None
    redundancy: RedundancyDecision | None
    association: MetricAssociation | None
    delta_relation: DeltaRelationStatistic | None
    selected_in_latency_association_set: bool
    kernel_class_links: tuple[MetricKernelClassLink, ...]


class EvidenceIndex(PmcModel):
    correlation_level: str
    correlation_method: str
    kernel_signatures_status: str
    kernel_signatures_limit: str
    delta_relations_status: str
    valid_delta_pair_count: int
    independent_delta_pair_group_count: int


class LayerReportBundle(PmcModel):
    projection: ProjectionReport
    quality: QualityReport
    redundancy: RedundancyReport
    correlation: CorrelationReport
    kernel_signatures: KernelSignatureReport
    delta_relations: DeltaRelationReport


class AnalysisProvenance(PmcModel):
    manifest: DatasetManifest
    devices: dict[str, DeviceSpec]
    dataset_schema_version: str
    analysis_config: AnalysisConfig


class PmcAnalysisReport(PmcModel):
    schema_version: str = ANALYSIS_REPORT_SCHEMA_VERSION
    scope: AnalysisScope
    data_readiness: DataReadiness
    layer_history: tuple[LayerRecord, ...]
    layer_reports: LayerReportBundle
    metric_knowledge: dict[str, MetricKnowledge]
    canonical_metric_sets: CanonicalMetricSets
    kernel_signatures: KernelSignatureReport
    delta_rulebook: tuple[DeltaRule, ...]
    evidence_index: EvidenceIndex
    warnings: tuple[str, ...]
    provenance: AnalysisProvenance
