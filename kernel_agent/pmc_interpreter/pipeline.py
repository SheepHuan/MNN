"""平台无关 PMC Interpreter 的固定串行编排。"""

from __future__ import annotations

from .dataset.reports import AnalysisConfig, ProjectionInput
from .layers.base import ProjectionLayer
from .layers.correlation import CorrelationLayer
from .layers.delta_relation import DeltaRelationLayer
from .layers.quality import QualityLayer
from .layers.redundancy import RedundancyLayer
from .layers.signatures import KernelSignatureLayer
from .semantics import compile_analysis_report, render_markdown_document


class PmcInterpreter:
    """固定执行证据升级顺序，不允许跳层或重排。"""

    def __init__(self, config=None):
        self.config = config or AnalysisConfig()
        self.projection = ProjectionLayer()
        self.quality = QualityLayer()
        self.redundancy = RedundancyLayer()
        self.correlation = CorrelationLayer()
        self.signatures = KernelSignatureLayer()
        self.delta_relation = DeltaRelationLayer()

    def run_layers(self, dataset):
        output = self.projection.run(
            ProjectionInput(dataset=dataset, config=self.config)
        )
        output = self.quality.run(output)
        output = self.redundancy.run(output)
        output = self.correlation.run(output)
        output = self.signatures.run(output)
        return self.delta_relation.run(output)

    def analyze(self, dataset):
        return compile_analysis_report(self.run_layers(dataset))

    def analyze_with_document(self, dataset):
        report = self.analyze(dataset)
        return report, render_markdown_document(report)


def analyze(dataset, config=None):
    return PmcInterpreter(config=config).analyze(dataset)


def analyze_with_document(dataset, config=None):
    return PmcInterpreter(config=config).analyze_with_document(dataset)
