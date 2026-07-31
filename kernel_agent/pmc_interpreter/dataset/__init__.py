"""平台无关 PMC 实验事实、分析报告与文档契约。"""

from .bundle import load_normalized_bundle, write_normalized_bundle
from .catalog import KernelTaxonomy, MetricRegistry
from .document import (
    MARKDOWN_DOCUMENT_SCHEMA_VERSION,
    MarkdownDocument,
    MarkdownSection,
)
from .model import (
    DATASET_SCHEMA_VERSION,
    ComparisonPair,
    DatasetManifest,
    DeviceSpec,
    KernelCondition,
    LatencyObservation,
    MeasurementRun,
    MetricDescriptor,
    MetricObservation,
    PmcDataset,
    PmcModel,
)
from .reports import (
    ANALYSIS_REPORT_SCHEMA_VERSION,
    AnalysisConfig,
    CorrelationOutput,
    DeltaRelationOutput,
    KernelSignatureOutput,
    PmcAnalysisReport,
    ProjectionInput,
    ProjectionOutput,
    QualityOutput,
    RedundancyOutput,
)

__all__ = [
    "ANALYSIS_REPORT_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "MARKDOWN_DOCUMENT_SCHEMA_VERSION",
    "AnalysisConfig",
    "ComparisonPair",
    "CorrelationOutput",
    "DatasetManifest",
    "DeltaRelationOutput",
    "DeviceSpec",
    "KernelCondition",
    "KernelSignatureOutput",
    "KernelTaxonomy",
    "LatencyObservation",
    "MarkdownDocument",
    "MarkdownSection",
    "MeasurementRun",
    "MetricDescriptor",
    "MetricObservation",
    "MetricRegistry",
    "PmcAnalysisReport",
    "PmcDataset",
    "PmcModel",
    "ProjectionInput",
    "ProjectionOutput",
    "QualityOutput",
    "RedundancyOutput",
    "load_normalized_bundle",
    "write_normalized_bundle",
]
