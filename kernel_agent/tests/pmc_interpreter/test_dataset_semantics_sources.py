#!/usr/bin/env python3
import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

import kernel_agent.pmc_interpreter as pmc_interpreter
from kernel_agent.pmc_interpreter import AnalysisConfig, analyze
from kernel_agent.pmc_interpreter.dataset import (
    ANALYSIS_REPORT_SCHEMA_VERSION,
    PmcAnalysisReport,
    PmcDataset,
    load_normalized_bundle,
    write_normalized_bundle,
)
from kernel_agent.pmc_interpreter.dataset.catalog import (
    KernelTaxonomy,
    MetricRegistry,
)
from kernel_agent.pmc_interpreter.dataset.sources import (
    append_comparison_pairs_csv,
    append_control_delta_csv,
    load_mnn_kernelreplay_dataset,
)


class DatasetSemanticsSourcesTest(unittest.TestCase):
    def test_public_api_uses_interpreter_name_only(self):
        self.assertTrue(hasattr(pmc_interpreter, "PmcInterpreter"))
        self.assertIs(pmc_interpreter.PmcDataset, PmcDataset)
        self.assertIs(pmc_interpreter.PmcAnalysisReport, PmcAnalysisReport)
        self.assertFalse(hasattr(pmc_interpreter, "PmcSemanticAnalyzer"))

    def test_removed_modules_have_no_compatibility_aliases(self):
        removed_modules = (
            "kernel_agent.pmc_interpreter.adapters",
            "kernel_agent.pmc_interpreter.model",
            "kernel_agent.pmc_interpreter.analysis_model",
            "kernel_agent.pmc_interpreter.api",
            "kernel_agent.pmc_interpreter.statistics",
            "kernel_agent.pmc_interpreter.knowledge",
            "kernel_agent.pmc_interpreter.semantics.registry",
        )
        for module_name in removed_modules:
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

    def test_pydantic_models_are_frozen_and_forbid_extra_fields(self):
        config = AnalysisConfig()
        self.assertTrue(config.model_config["frozen"])
        self.assertEqual(config.model_config["extra"], "forbid")
        with self.assertRaises(ValidationError):
            config.min_coverage = 0.5
        with self.assertRaises(ValidationError):
            AnalysisConfig.model_validate({"unknown_threshold": 1})

    def test_generic_metric_registry_handles_multiple_native_dialects(self):
        registry = MetricRegistry()
        cuda = registry.describe(
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed", "cupti"
        )
        self.assertEqual(cuda.normalizer, "elapsed_peak")
        self.assertTrue(cuda.duration_coupled)
        self.assertEqual(cuda.mechanism, "parallelism")

        adreno = registry.describe("sp_busy_cycles", "kgsl")
        self.assertEqual(adreno.mechanism, "compute_pipeline")
        self.assertEqual(adreno.unit, "cycle")

        mali = registry.describe("l2_ext_read", "mali")
        self.assertEqual(mali.mechanism, "cache_locality")
        self.assertEqual(mali.hardware_scope, "l2")

    def test_kernel_taxonomy_separates_owner_and_execution_role(self):
        taxonomy = KernelTaxonomy.load()
        add_bias = taxonomy.classify("matmul", "cuda_add_bias_fp32")
        self.assertEqual(add_bias["semantic_family"], "dense_quant")
        self.assertEqual(
            add_bias["execution_role"], "postprocess_pointwise"
        )
        reorder = taxonomy.classify(
            "convolution", "cuda_wino_weight_reorder_fp32"
        )
        self.assertEqual(reorder["execution_role"], "preprocess_pack")

    def _write_mnn_fixture(self, root):
        operator_cases = root / "operator_cases.json"
        operator_cases.write_text(json.dumps({
            "cases": [
                {
                    "name": "case_a",
                    "backend": "cuda",
                    "op_type": "relu",
                    "variant": "cuda_relu_fp32",
                    "tag": "test",
                    "dtype": "float32",
                    "validator": "identity_fp32",
                    "int_params": {"size": 16},
                },
                {
                    "name": "case_b",
                    "backend": "cuda",
                    "op_type": "relu",
                    "variant": "cuda_relu_opt_fp32",
                    "tag": "test",
                    "dtype": "float32",
                    "validator": "identity_fp32",
                    "int_params": {"size": 16},
                },
            ]
        }), encoding="utf-8")
        latency = root / "latency.json"
        latency.write_text(
            json.dumps({"case_a": 10.0, "case_b": 8.0}),
            encoding="utf-8",
        )
        rows = root / "rows.csv"
        with rows.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "case", "metric", "status", "value", "pmu_status",
                "num_passes", "returncode", "error",
            ])
            writer.writeheader()
            for case, value in (("case_a", 1.5), ("case_b", 2.5)):
                writer.writerow({
                    "case": case,
                    "metric": "dram__bytes.sum",
                    "status": "VALID",
                    "value": value,
                    "pmu_status": "sampled",
                    "num_passes": 2,
                    "returncode": 0,
                    "error": "",
                })
        return rows, latency, operator_cases

    def test_dataset_source_round_trip_control_delta_and_analysis_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, latency, operator_cases = self._write_mnn_fixture(root)
            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases
            )
            self.assertEqual(len(dataset.conditions), 2)
            self.assertEqual(len(dataset.comparisons), 1)
            self.assertEqual(dataset.metric_observations[0].value, 1.5)
            self.assertNotEqual(
                dataset.metric_observations[0].run_id,
                dataset.latency_observations[0].run_id,
            )

            normalized = root / "normalized.json"
            write_normalized_bundle(dataset, normalized)
            restored = load_normalized_bundle(normalized)
            self.assertEqual(
                restored.model_dump(mode="json"),
                dataset.model_dump(mode="json"),
            )

            delta = root / "delta.csv"
            with delta.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "case_name", "case_args", "case_runs",
                    "pmu_metric_name", "delta_metric",
                ])
                writer.writeheader()
                writer.writerow({
                    "case_name": "case_a",
                    "case_args": "--measurement=1",
                    "case_runs": 5,
                    "pmu_metric_name": "sp_busy_cycles",
                    "delta_metric": -3.5,
                })
            restored = append_control_delta_csv(
                restored, delta, namespace="kgsl"
            )
            observation = restored.metric_observations[-1]
            self.assertEqual(observation.value, -3.5)
            self.assertEqual(
                observation.value_semantics, "workload_minus_control"
            )
            self.assertTrue(restored.runs[observation.run_id].repeat_id)
            second_namespace = append_control_delta_csv(
                restored, delta, namespace="mali"
            )
            self.assertEqual(
                len(second_namespace.metric_observations),
                len(restored.metric_observations) + 1,
            )
            with self.assertRaisesRegex(ValueError, "already been appended"):
                append_control_delta_csv(
                    second_namespace, delta, namespace="mali"
                )

            explicit_pairs = root / "pairs.csv"
            with explicit_pairs.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "pair_id", "baseline_case", "variant_case", "design",
                ])
                writer.writeheader()
                writer.writerow({
                    "pair_id": "explicit_pair",
                    "baseline_case": "case_a",
                    "variant_case": "case_b",
                    "design": "paired_observational",
                })
            second_namespace = append_comparison_pairs_csv(
                second_namespace, explicit_pairs
            )
            self.assertEqual(len(second_namespace.comparisons), 1)
            self.assertEqual(
                second_namespace.comparisons[0].pair_id, "explicit_pair"
            )

            restored = second_namespace.model_copy(update={
                "manifest": second_namespace.manifest.model_copy(update={
                    "platform": "adreno",
                    "backend": "opencl",
                })
            })
            report = analyze(restored, AnalysisConfig(
                min_unique_values=2,
                min_group_samples=2,
                min_delta_pairs=2,
            ))
            self.assertIs(type(report), PmcAnalysisReport)
            payload = report.model_dump(mode="json")
            self.assertEqual(
                payload["schema_version"], ANALYSIS_REPORT_SCHEMA_VERSION
            )
            self.assertEqual(
                payload["schema_version"], "mnn-pmc-analysis-report/v1"
            )
            self.assertIn(
                "descriptive_canonical", payload["canonical_metric_sets"]
            )
            self.assertIn(
                "latency_association", payload["canonical_metric_sets"]
            )
            self.assertIn(
                "metric_to_kernel_classes", payload["kernel_signatures"]
            )
            self.assertEqual(payload["scope"]["platform"], "adreno")
            self.assertEqual(
                payload["metric_knowledge"]["kgsl::sp_busy_cycles"]
                ["feature_spec"]["transform"],
                "signed_asinh",
            )
            self.assertEqual(
                payload["scope"]["cross_platform_policy"].split(";")[0],
                "native metrics remain device-scoped",
            )


if __name__ == "__main__":
    unittest.main()
