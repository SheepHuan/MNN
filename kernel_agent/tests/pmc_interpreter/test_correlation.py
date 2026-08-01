#!/usr/bin/env python3
import csv
import hashlib
import json
import math
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

    def test_not_applicable_flops_do_not_look_like_missing_workload(self):
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
                for index in range(5):
                    case = "case_{}".format(index)
                    workload = {
                        "output_elements": 16 * (index + 1),
                        "algorithmic_bytes": 128 * (index + 1),
                    }
                    provenance = {}
                    if index < 2:
                        workload["algorithmic_flops"] = 64 * (index + 1)
                    else:
                        provenance["algorithmic_flops"] = {
                            "status": "not_applicable",
                            "source": "operator_contract",
                            "method": "semantic_formula",
                            "formula": "",
                            "confidence": 1.0,
                            "notes": "data movement",
                        }
                    cases.append({
                        "name": case,
                        "backend": "cuda",
                        "op_type": "sample",
                        "variant": case,
                        "tag": "test",
                        "dtype": "float32",
                        "shapes": {"output0": [16 * (index + 1)]},
                        "workload": workload,
                        "workload_provenance": provenance,
                    })
                    latencies[case] = float(10 + index)
                    writer.writerow({
                        "case": case,
                        "metric": "dram__bytes.sum",
                        "status": "VALID",
                        "value": 100 + index,
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
            readiness = projection.projection_report.data_readiness
            self.assertEqual(
                readiness.workload_field_coverage["algorithmic_flops"],
                1.0,
            )
            self.assertNotIn(
                "algorithmic_flops", readiness.missing_high_value_fields
            )

            quality = QualityLayer().run(projection)
            redundancy = RedundancyLayer().run(quality)
            correlation = CorrelationLayer().run(redundancy)
            self.assertNotIn(
                "workload.algorithmic_flops",
                correlation.correlation_report.baseline.missing_high_value_controls,
            )

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

    def test_environment_controls_are_scoped_to_target_and_feature_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = {
                "status": "theoretical_estimate",
                "source": "test_operator_contract",
                "method": "semantic_formula",
                "formula": "constant test workload",
                "confidence": 1.0,
                "notes": "",
            }
            cases = []
            latency_cases = {}
            for index in range(12):
                case = "case_{:02d}".format(index)
                cases.append({
                    "name": case,
                    "backend": "cuda",
                    "op_type": "sample",
                    "variant": "sample_{}".format(index),
                    "tag": "test",
                    "dtype": "float32",
                    "shapes": {"output0": [1]},
                    "workload": {
                        "output_elements": 1,
                        "algorithmic_flops": 1,
                        "algorithmic_bytes": 8,
                    },
                    "shape_provenance": {"output0": provenance},
                    "workload_provenance": {
                        "output_elements": provenance,
                        "algorithmic_flops": provenance,
                        "algorithmic_bytes": provenance,
                    },
                })
                latency_clock = 1_000_000_000 + index * 20_000_000
                latency_cases[case] = {
                    "latency_us": math.exp(1.0 + index * 0.07),
                    "launch_sampling_status": "sampled",
                    "launch_records": [{
                        "kernel_name": "sample_kernel",
                        "grid": [1, 1, 1],
                        "block": [32, 1, 1],
                        "registers_per_thread": 8,
                        "static_shared_memory_bytes": 0,
                        "dynamic_shared_memory_bytes": 0,
                        "local_memory_per_thread_bytes": 0,
                        "local_memory_total_bytes": 0,
                    }],
                    "environment": {
                        "sampling_source": "nvml",
                        "sampling_status": "sampled",
                        "gpu_clock_hz_before": latency_clock,
                        "gpu_clock_hz_after": latency_clock,
                        "temperature_c_before": 50,
                        "temperature_c_after": 50,
                    },
                }

            operator_cases = root / "operator_cases.json"
            operator_cases.write_text(json.dumps({
                "format": "mnn-kernel-operator-cases",
                "version": 1,
                "cases": cases,
            }), encoding="utf-8")
            latency = root / "latency.json"
            latency.write_text(json.dumps({
                "format": "mnn-kernel-latency",
                "version": 1,
                "source_manifest_sha256": hashlib.sha256(
                    operator_cases.read_bytes()
                ).hexdigest(),
                "cases": latency_cases,
            }), encoding="utf-8")

            rows = root / "rows.csv"
            row_fields = [
                "sweep_plan_id", "sweep_config_id", "case", "metric",
                "status", "value", "pmu_status", "num_passes",
                "returncode", "error", "collection_session_id",
                "order_index", "gpu_clock_hz_before", "gpu_clock_hz_after",
                "temperature_c_before", "temperature_c_after",
                "environment_status", "environment_source",
            ]
            with rows.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=row_fields)
                writer.writeheader()
                order_index = 0
                for index in range(12):
                    case = "case_{:02d}".format(index)
                    measurements = (
                        (
                            "dram__bytes.sum",
                            100 + index * 10,
                            "target-feature-{}".format(index),
                            1_300_000_000 + index * 15_000_000,
                        ),
                        (
                            "sm__inst_executed.sum",
                            50 + (index * 7) % 13,
                            "unrelated-feature-{}".format(index),
                            800_000_000 + index * 30_000_000,
                        ),
                    )
                    for metric, value, session_id, clock in measurements:
                        order_index += 1
                        writer.writerow({
                            "sweep_plan_id": "test-plan",
                            "sweep_config_id": "test-config",
                            "case": case,
                            "metric": metric,
                            "status": "VALID",
                            "value": value,
                            "pmu_status": "sampled",
                            "num_passes": 1,
                            "returncode": 0,
                            "error": "",
                            "collection_session_id": session_id,
                            "order_index": order_index,
                            "gpu_clock_hz_before": clock,
                            "gpu_clock_hz_after": clock,
                            "temperature_c_before": 45,
                            "temperature_c_after": 45,
                            "environment_status": "sampled",
                            "environment_source": "nvml",
                        })

            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases
            )
            altered_runs = dict(dataset.runs)
            alternate_order = (9, 0, 7, 2, 10, 3, 11, 1, 8, 4, 6, 5)
            for index, clock_rank in enumerate(alternate_order):
                run_id = "pmc-session::unrelated-feature-{}".format(index)
                run = altered_runs[run_id]
                clock = 700_000_000 + clock_rank * 70_000_000
                altered_runs[run_id] = run.model_copy(update={
                    "gpu_clock_hz": float(clock),
                    "environment_samples": tuple(
                        sample.model_copy(update={"gpu_clock_hz": float(clock)})
                        for sample in run.environment_samples
                    ),
                })
            altered_dataset = dataset.model_copy(update={"runs": altered_runs})

            def correlate(current_dataset):
                config = AnalysisConfig(min_unique_values=2)
                projection = ProjectionLayer().run(
                    ProjectionInput(dataset=current_dataset, config=config)
                )
                quality = QualityLayer().run(projection)
                redundancy = RedundancyLayer().run(quality)
                return CorrelationLayer().run(redundancy)

            original = correlate(dataset).correlation_report
            altered = correlate(altered_dataset).correlation_report
            self.assertEqual(
                original.baseline.evidence_level,
                "proxy_controlled_association",
            )
            self.assertIn("environment.gpu_clock_hz", original.baseline.controls)
            for condition_id, prediction in original.baseline.predictions.items():
                self.assertAlmostEqual(
                    prediction,
                    altered.baseline.predictions[condition_id],
                )

            target_feature = "cupti::dram__bytes.sum"
            self.assertTrue(
                original.associations[target_feature].feature_residualized
            )
            for condition_id, value in original.residual_feature_values[
                target_feature
            ].items():
                self.assertAlmostEqual(
                    value,
                    altered.residual_feature_values[target_feature][condition_id],
                )

            unrelated_feature = "cupti::sm__inst_executed.sum"
            self.assertTrue(any(
                abs(
                    value
                    - altered.residual_feature_values[unrelated_feature][condition_id]
                ) > 1e-6
                for condition_id, value in original.residual_feature_values[
                    unrelated_feature
                ].items()
            ))


if __name__ == "__main__":
    unittest.main()
