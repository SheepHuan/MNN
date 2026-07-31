#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from pathlib import Path

from kernel_agent.pmc_interpreter import AnalysisConfig
from kernel_agent.pmc_interpreter.dataset import ProjectionInput
from kernel_agent.pmc_interpreter.dataset.sources import (
    load_mnn_kernelreplay_dataset,
)
from kernel_agent.pmc_interpreter.layers import (
    CorrelationLayer,
    KernelSignatureLayer,
    ProjectionLayer,
    QualityLayer,
    RedundancyLayer,
)
from kernel_agent.pmc_interpreter.layers.statistics import (
    distance_correlation,
    spearman,
)


class CorrelationLayerTest(unittest.TestCase):
    def test_distance_correlation_detects_non_monotonic_dependence(self):
        feature = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        target = [value * value for value in feature]
        self.assertAlmostEqual(spearman(feature, target), 0.0)
        self.assertGreater(distance_correlation(feature, target), 0.5)

    def test_uncontrolled_relationship_is_labeled_descriptive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            latencies = {}
            rows = root / "rows.csv"
            with rows.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "case", "metric", "status", "value", "pmu_status",
                    "num_passes", "returncode", "error",
                ])
                writer.writeheader()
                for index in range(1, 13):
                    case = "case_{:02d}".format(index)
                    cases.append({
                        "name": case,
                        "backend": "cuda",
                        "op_type": "relu",
                        "variant": "relu_{}".format(index),
                        "tag": "test",
                        "dtype": "float32",
                        "int_params": {"size": 16},
                    })
                    latencies[case] = float(index + 5)
                    for metric, value in (
                        ("dram__bytes.sum", index * 10),
                        ("sm__inst_executed.sum", (index * 7) % 11 + 1),
                    ):
                        writer.writerow({
                            "case": case,
                            "metric": metric,
                            "status": "VALID",
                            "value": value,
                            "pmu_status": "sampled",
                            "num_passes": 1,
                            "returncode": 0,
                            "error": "",
                        })
            operator_cases = root / "operator_cases.json"
            operator_cases.write_text(
                json.dumps({"cases": cases}), encoding="utf-8"
            )
            latency = root / "latency.json"
            latency.write_text(json.dumps(latencies), encoding="utf-8")

            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases
            )
            projection = ProjectionLayer().run(
                ProjectionInput(dataset=dataset, config=AnalysisConfig())
            )
            quality = QualityLayer().run(projection)
            redundancy = RedundancyLayer().run(quality)
            correlation = CorrelationLayer().run(redundancy)

            association = correlation.correlation_report.associations[
                "cupti::dram__bytes.sum"
            ]
            self.assertGreater(association.raw_log_latency_spearman, 0.9)
            self.assertLessEqual(association.fdr_qvalue, 0.1)
            self.assertEqual(
                association.statistical_relation_class,
                "raw_monotonic_descriptive",
            )
            self.assertEqual(
                correlation.correlation_report.evidence_level,
                "descriptive_cross_kernel",
            )
            self.assertFalse(association.feature_residualized)
            residual_values = (
                correlation.correlation_report.residual_feature_values[
                    "cupti::dram__bytes.sum"
                ]
            )
            self.assertEqual(len(residual_values), 12)
            aligned_ids = sorted(residual_values)
            self.assertAlmostEqual(
                spearman(
                    [residual_values[item] for item in aligned_ids],
                    [
                        correlation.correlation_report.baseline.residuals[item]
                        for item in aligned_ids
                    ],
                ),
                association.residual_log_latency_spearman,
            )

            signatures = KernelSignatureLayer().run(correlation)
            signature = signatures.signature_report.signature_index[
                "op_type"
            ]["relu"]
            self.assertEqual(signature.status, "exploratory_signature")
            self.assertIn("memory.external.bytes", signature.concept_slots)
            self.assertIn(
                "cupti::dram__bytes.sum",
                signatures.signature_report.descriptive_canonical_features,
            )
            self.assertIn(
                "cupti::dram__bytes.sum",
                tuple(
                    metric.feature_id
                    for metric in signatures.signature_report.latency_association_set
                ),
            )

    def test_class_specific_signature_survives_global_direction_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            latencies = {}
            rows = root / "rows.csv"
            with rows.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "case", "metric", "status", "value", "pmu_status",
                    "num_passes", "returncode", "error",
                ])
                writer.writeheader()
                for op_type, prefix, reverse in (
                    ("relu", "elementwise", False),
                    ("matmul", "dense", True),
                ):
                    for index in range(1, 6):
                        case = "{}_{}".format(prefix, index)
                        cases.append({
                            "name": case,
                            "backend": "cuda",
                            "op_type": op_type,
                            "variant": case,
                            "tag": "test",
                            "dtype": "float32",
                            "int_params": {"size": 16},
                        })
                        latencies[case] = float(10 + (6 - index if reverse else index))
                        writer.writerow({
                            "case": case,
                            "metric": "dram__bytes.sum",
                            "status": "VALID",
                            "value": index,
                            "pmu_status": "sampled",
                            "num_passes": 1,
                            "returncode": 0,
                            "error": "",
                        })
            operator_cases = root / "operator_cases.json"
            operator_cases.write_text(
                json.dumps({"cases": cases}), encoding="utf-8"
            )
            latency = root / "latency.json"
            latency.write_text(json.dumps(latencies), encoding="utf-8")
            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases
            )
            config = AnalysisConfig(min_group_samples=5)
            projection = ProjectionLayer().run(
                ProjectionInput(dataset=dataset, config=config)
            )
            quality = QualityLayer().run(projection)
            redundancy = RedundancyLayer().run(quality)
            correlation = CorrelationLayer().run(redundancy)
            association = correlation.correlation_report.associations[
                "cupti::dram__bytes.sum"
            ]
            self.assertLess(
                abs(association.raw_log_latency_spearman), 0.1
            )
            signatures = KernelSignatureLayer().run(correlation)
            for family in ("elementwise_quant", "dense_quant"):
                resolved = signatures.signature_report.signature_index[
                    "semantic_family"
                ][family].resolved_features
                self.assertIn(
                    "cupti::dram__bytes.sum",
                    tuple(item.feature_id for item in resolved),
                )


if __name__ == "__main__":
    unittest.main()
