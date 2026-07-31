"""平台无关 PMC Interpreter 公共调用边界。"""

from .dataset import (
    ANALYSIS_REPORT_SCHEMA_VERSION,
    AnalysisConfig,
    MarkdownDocument,
    PmcAnalysisReport,
    PmcDataset,
)
from .pipeline import PmcInterpreter, analyze, analyze_with_document
from .semantics import compile_analysis_report, render_markdown_document

__all__ = [
    "ANALYSIS_REPORT_SCHEMA_VERSION",
    "AnalysisConfig",
    "MarkdownDocument",
    "PmcAnalysisReport",
    "PmcDataset",
    "PmcInterpreter",
    "analyze",
    "analyze_with_document",
    "compile_analysis_report",
    "render_markdown_document",
]
