#!/usr/bin/env python3
import csv
import hashlib
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

        atomic_traffic = registry.describe(
            "l1tex__t_bytes_pipe_lsu_mem_global_op_atom.sum", "cupti"
        )
        self.assertEqual(atomic_traffic.mechanism, "atomic_operation")
        self.assertEqual(atomic_traffic.phenomenon_role, "work")

        atomic_conflict = registry.describe(
            "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_atom.sum",
            "cupti",
        )
        self.assertEqual(atomic_conflict.mechanism, "synchronization")
        self.assertEqual(atomic_conflict.phenomenon_role, "symptom")

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

        raster_binary = taxonomy.classify(
            "raster_binary", "cuda_raster_binarymid_fp32"
        )
        self.assertEqual(
            raster_binary["execution_role"], "layout_data_movement"
        )

        conv1d_silu = taxonomy.classify(
            "attention", "cuda_conv1d_silu_fp32"
        )
        self.assertEqual(conv1d_silu["execution_role"], "core_compute")

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
                    "shapes": {"input0": [16], "output0": [16]},
                    "workload": {
                        "output_elements": 16,
                        "algorithmic_bytes": 128,
                    },
                    "launch": {
                        "grid": [16, 1, 1],
                        "block": [64, 1, 1],
                    },
                    "workload_provenance": {
                        "algorithmic_bytes": {
                            "status": "estimated",
                            "source": "operator_case",
                            "method": "semantic_formula",
                            "formula": "(input_elements + output_elements) * 4",
                            "confidence": 1.0,
                        }
                    },
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
                    "shapes": {"input0": [16], "output0": [16]},
                    "workload": {
                        "output_elements": 16,
                        "algorithmic_bytes": 128,
                    },
                    "workload_provenance": {
                        "algorithmic_bytes": {
                            "status": "estimated",
                            "source": "operator_case",
                            "method": "semantic_formula",
                            "formula": "(input_elements + output_elements) * 4",
                            "confidence": 1.0,
                        }
                    },
                },
            ]
        }), encoding="utf-8")
        latency = root / "latency.json"
        manifest_sha256 = hashlib.sha256(operator_cases.read_bytes()).hexdigest()
        latency.write_text(json.dumps({
            "format": "mnn-kernel-latency",
            "version": 1,
            "source_manifest_sha256": manifest_sha256,
            "cases": {
                "case_a": {
                    "latency_us": 10.0,
                    "launch_sampling_status": "sampled",
                    "launch_records": [{
                        "stage_index": 0,
                        "kernel_name": "relu_kernel",
                        "grid": [2, 1, 1],
                        "block": [128, 1, 1],
                        "registers_per_thread": 12,
                        "static_shared_memory_bytes": 0,
                        "dynamic_shared_memory_bytes": 0,
                        "local_memory_per_thread_bytes": 0,
                        "local_memory_total_bytes": 0,
                    }],
                    "environment": {
                        "gpu_clock_hz_before": 1_500_000_000,
                        "gpu_clock_hz_after": 1_490_000_000,
                        "temperature_c_before": 50,
                        "temperature_c_after": 51,
                        "sampling_source": "nvml",
                        "sampling_status": "sampled",
                    },
                },
                "case_b": {
                    "latency_us": 8.0,
                    "launch_sampling_status": (
                        "unavailable:no kernel activity records"
                    ),
                    "launch_records": [],
                    "environment": {
                        "gpu_clock_hz_before": 1_500_000_000,
                        "gpu_clock_hz_after": 1_500_000_000,
                        "temperature_c_before": 51,
                        "temperature_c_after": 51,
                        "sampling_source": "nvml",
                        "sampling_status": "sampled",
                    },
                },
            },
        }), encoding="utf-8")
        rows = root / "rows.csv"
        with rows.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "sweep_plan_id", "sweep_config_id", "case", "metric",
                "status", "value", "pmu_status", "num_passes",
                "returncode", "error", "collection_session_id",
                "order_index", "gpu_clock_hz_before", "gpu_clock_hz_after",
                "temperature_c_before", "temperature_c_after",
                "environment_status", "environment_source",
            ])
            writer.writeheader()
            for case, value in (("case_a", 1.5), ("case_b", 2.5)):
                writer.writerow({
                    "sweep_plan_id": "fixture-plan",
                    "sweep_config_id": "fixture-config",
                    "case": case,
                    "metric": "dram__bytes.sum",
                    "status": "VALID",
                    "value": value,
                    "pmu_status": "sampled",
                    "num_passes": 2,
                    "returncode": 0,
                    "error": "",
                    "collection_session_id": "session-{}".format(case),
                    "order_index": 0 if case == "case_a" else 1,
                    "gpu_clock_hz_before": 1_500_000_000,
                    "gpu_clock_hz_after": 1_490_000_000,
                    "temperature_c_before": 50,
                    "temperature_c_after": 51,
                    "environment_status": "sampled",
                    "environment_source": "nvml",
                })
        return rows, latency, operator_cases

    def test_launch_sampling_status_normalization_and_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, latency, operator_cases = self._write_mnn_fixture(root)
            payload = json.loads(latency.read_text(encoding="utf-8"))
            sampled_record = payload["cases"]["case_a"]["launch_records"][0]
            payload["cases"]["case_a"].update({
                "launch_sampling_status": (
                    "unavailable:CUPTI Activity was not built"
                ),
                "launch_records": [],
            })
            payload["cases"]["case_b"].update({
                "launch_sampling_status": "partial:dropped_records=1",
                "launch_records": [sampled_record],
            })
            latency.write_text(json.dumps(payload), encoding="utf-8")

            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases
            )
            self.assertEqual(
                dataset.runs["latency::case_a"].launch_sampling_status,
                "unavailable",
            )
            self.assertEqual(
                dataset.runs["latency::case_b"].launch_sampling_status,
                "error",
            )
            self.assertEqual(
                len(dataset.runs["latency::case_b"].launch_records),
                1,
            )

            payload["cases"]["case_a"].update({
                "launch_sampling_status": "no_launch",
                "launch_records": [],
            })
            latency.write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases
            )
            self.assertEqual(
                dataset.runs["latency::case_a"].launch_sampling_status,
                "no_launch",
            )

            payload["cases"]["case_a"].update({
                "launch_sampling_status": "sampled",
                "launch_records": [],
            })
            latency.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValidationError,
                "sampled launch collection must contain launch records",
            ):
                load_mnn_kernelreplay_dataset(rows, latency, operator_cases)

    def test_latency_environment_does_not_accept_status_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, latency, operator_cases = self._write_mnn_fixture(root)
            payload = json.loads(latency.read_text(encoding="utf-8"))
            environment = payload["cases"]["case_a"]["environment"]
            environment["status"] = environment.pop("sampling_status")
            latency.write_text(json.dumps(payload), encoding="utf-8")

            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases
            )
            samples = dataset.runs["latency::case_a"].environment_samples
            self.assertTrue(samples)
            self.assertEqual({sample.status for sample in samples}, {"unknown"})

    def test_latency_source_envelope_is_strict_but_legacy_is_exploratory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, latency, operator_cases = self._write_mnn_fixture(root)
            payload = json.loads(latency.read_text(encoding="utf-8"))
            payload["collector"] = "unexpected"
            latency.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "fields must be exactly",
            ):
                load_mnn_kernelreplay_dataset(rows, latency, operator_cases)

            payload.pop("collector")
            case_b = payload["cases"].pop("case_b")
            latency.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "latency/manifest mismatch: 1 missing",
            ):
                load_mnn_kernelreplay_dataset(rows, latency, operator_cases)

            payload["cases"]["case_b"] = case_b
            payload["source_manifest_sha256"] = "0" * 64
            latency.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "source manifest SHA256 does not match",
            ):
                load_mnn_kernelreplay_dataset(rows, latency, operator_cases)

            latency.write_text(
                json.dumps({"case_a": 10.0, "case_b": 8.0}),
                encoding="utf-8",
            )
            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases
            )
            self.assertTrue(
                any("legacy flat latency JSON" in issue for issue in dataset.issues)
            )
            self.assertEqual(
                dataset.runs["latency::case_a"].launch_sampling_status,
                "not_collected",
            )

    def test_valid_metric_observation_rejects_nonfinite_source_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, latency, operator_cases = self._write_mnn_fixture(root)
            with rows.open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            records[0]["value"] = "nan"
            with rows.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "sweep_plan_id", "sweep_config_id", "case", "metric",
                    "status", "value", "pmu_status", "num_passes",
                    "returncode", "error", "collection_session_id",
                    "order_index", "gpu_clock_hz_before", "gpu_clock_hz_after",
                    "temperature_c_before", "temperature_c_after",
                    "environment_status", "environment_source",
                ])
                writer.writeheader()
                writer.writerows(records)

            with self.assertRaisesRegex(
                ValidationError,
                "valid metric observation must contain a finite value",
            ):
                load_mnn_kernelreplay_dataset(rows, latency, operator_cases)

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
            pmc_run = dataset.runs[dataset.metric_observations[0].run_id]
            self.assertEqual(pmc_run.collection_id, "session-case_a")
            self.assertEqual(pmc_run.gpu_clock_hz, 1_495_000_000)
            self.assertEqual(pmc_run.environment_samples[0].source, "nvml")
            self.assertNotEqual(
                dataset.metric_observations[0].run_id,
                dataset.latency_observations[0].run_id,
            )
            self.assertEqual(
                dataset.conditions["case_a"].workload["algorithmic_bytes"],
                128,
            )
            self.assertEqual(
                dataset.conditions["case_a"].workload_provenance[
                    "algorithmic_bytes"
                ].method,
                "semantic_formula",
            )
            latency_run = dataset.runs["latency::case_a"]
            self.assertEqual(latency_run.launch_sampling_status, "sampled")
            self.assertEqual(latency_run.launch_records[0].grid, (2, 1, 1))
            self.assertEqual(
                latency_run.launch_records[0].registers_per_thread,
                12,
            )
            self.assertEqual(latency_run.gpu_clock_hz, 1_495_000_000)
            self.assertEqual(latency_run.temperature_c, 50.5)
            self.assertEqual(len(latency_run.environment_samples), 2)
            self.assertEqual(
                dataset.runs["latency::case_b"].launch_sampling_status,
                "unavailable",
            )
            self.assertEqual(
                dataset.conditions["case_a"].launch["grid"],
                [16, 1, 1],
            )
            self.assertEqual(
                dataset.manifest.metadata["source_consistency"]
                ["latency_source_manifest_matches"],
                True,
            )

            normalized = root / "normalized.json"
            write_normalized_bundle(dataset, normalized)
            restored = load_normalized_bundle(normalized)
            self.assertEqual(
                restored.model_dump(mode="json"),
                dataset.model_dump(mode="json"),
            )
            self.assertEqual(
                restored.runs["latency::case_a"].launch_sampling_status,
                "sampled",
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
            self.assertIn(
                "observed_total_blocks",
                payload["data_readiness"]["available_launch_fields"],
            )
            self.assertIn(
                "observed_max_registers_per_thread",
                payload["data_readiness"]["available_launch_fields"],
            )
            self.assertNotIn(
                "resource_usage",
                payload["data_readiness"]["missing_high_value_fields"],
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
            self.assertIn(
                "uncontrolled cross-kernel descriptive associations",
                payload["canonical_metric_sets"]["latency_association"]
                ["selection_basis"],
            )

            _, document = pmc_interpreter.PmcInterpreter(
                config=AnalysisConfig(
                    min_unique_values=2,
                    min_group_samples=2,
                    min_delta_pairs=2,
                )
            ).analyze_with_document(restored)
            self.assertIn("## 三个核心问题结论", document.markdown)
            self.assertIn("### Semantic family 直接面板", document.markdown)
            self.assertIn("### Op type 直接证据与继承关系", document.markdown)
            self.assertIn("要生成第一版 rulebook", document.markdown)
            self.assertIn("## 下一步完成路径", document.markdown)


if __name__ == "__main__":
    unittest.main()
