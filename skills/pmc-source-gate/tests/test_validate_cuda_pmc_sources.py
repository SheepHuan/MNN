#!/usr/bin/env python3
"""测试 CUDA PMC 原始数据发布门禁。"""

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from kernel_agent.pmc_interpreter.dataset import selection as selection_module
from kernel_agent.pmc_interpreter.dataset.selection import (
    build_case_selection_plan,
    load_case_selection_plan,
    write_case_selection_plan,
)


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "validate_cuda_pmc_sources.py"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_cuda_pmc_sources", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidateCudaPmcSourcesTest(unittest.TestCase):
    SWEEP_CONFIG_ID = "a" * 64

    def setUp(self):
        self.module = load_module()

    def _fixture(
        self,
        root,
        *,
        manifest_op_types=None,
        plan_target_op_types=None,
        case_counts=None,
        minimum_conditions_per_op_type=5,
        plan_policy="condition-balanced",
        metrics=("dram__bytes.sum", "sm__inst_executed.sum"),
    ):
        root.mkdir(parents=True, exist_ok=True)
        manifest_op_types = tuple(
            manifest_op_types or self.module.P0_TARGET_OP_TYPES
        )
        plan_target_op_types = tuple(
            plan_target_op_types or manifest_op_types
        )
        case_counts = dict(case_counts or {})
        cases = []
        latency_cases = {}
        sequence = 0
        for op_type in manifest_op_types:
            for index in range(case_counts.get(op_type, 5)):
                sequence += 1
                case = "{}_case_{}".format(op_type, index)
                cases.append({
                    "name": case,
                    "backend": "cuda",
                    "op_type": op_type,
                    "framework": "mnn",
                    "tag": "test",
                    "variant": "{}_variant_{}".format(op_type, index),
                    "dtype": "float32",
                    "int_params": {"size": index + 1},
                    "shapes": {
                        "input0": [index + 1],
                        "output0": [index + 1],
                    },
                    "shape_provenance": {
                        key: {
                            "status": "theoretical_estimate",
                            "source": "adapter_contract",
                            "method": "semantic_formula",
                            "formula": key,
                            "confidence": 1.0,
                            "notes": "",
                        }
                        for key in ("input0", "output0")
                    },
                    "workload": {
                        "output_elements": index + 1,
                        "algorithmic_bytes": 8 * (index + 1),
                        "algorithmic_flops": 2 * (index + 1),
                    },
                    "workload_provenance": {
                        key: {
                            "status": "theoretical_estimate",
                            "source": "adapter_contract",
                            "method": "semantic_formula",
                            "formula": key,
                            "confidence": 1.0,
                            "notes": "",
                        }
                        for key in (
                            "output_elements",
                            "algorithmic_bytes",
                            "algorithmic_flops",
                        )
                    }
                })
                latency_cases[case] = {
                    "latency_us": 10.0 + sequence,
                    "launch_sampling_status": "sampled",
                    "launch_records": [{
                        "kernel_name": "kernel_{}".format(case),
                        "grid": [index + 1, 1, 1],
                        "block": [128, 1, 1],
                        "registers_per_thread": 16,
                        "static_shared_memory_bytes": 0,
                        "dynamic_shared_memory_bytes": 0,
                        "local_memory_per_thread_bytes": 0,
                        "local_memory_total_bytes": 0,
                    }],
                    "environment": {
                        "sampling_source": "nvml",
                        "sampling_status": "sampled",
                        "gpu_clock_hz_before": 1_500_000_000,
                        "gpu_clock_hz_after": 1_500_000_000,
                        "temperature_c_before": 50.0,
                        "temperature_c_after": 51.0,
                    },
                }
        manifest = root / "operator_cases.json"
        manifest.write_text(json.dumps({
            "format": "mnn-kernel-operator-cases",
            "version": 1,
            "cases": cases,
        }), encoding="utf-8")
        manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()

        plan = build_case_selection_plan(
            manifest,
            backend="cuda",
            policy=plan_policy,
            minimum_conditions_per_op_type=minimum_conditions_per_op_type,
            target_op_types=plan_target_op_types,
        )
        plan_path = root / "selection.json"
        write_case_selection_plan(plan, plan_path)

        latency = root / "latency.json"
        latency.write_text(json.dumps({
            "format": "mnn-kernel-latency",
            "version": 1,
            "source_manifest_sha256": manifest_sha256,
            "cases": latency_cases,
        }), encoding="utf-8")

        valid = root / "valid.csv"
        with valid.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["metric", "value"])
            writer.writeheader()
            for index, metric in enumerate(metrics, 1):
                writer.writerow({"metric": metric, "value": str(index)})

        rows = root / "rows.csv"
        with rows.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.module.ROW_COLUMNS)
            writer.writeheader()
            order_index = 0
            for selected in plan.selected_cases:
                order_index += 1
                session_id = "session-{}".format(selected.case_id)
                for metric in metrics:
                    writer.writerow({
                        "sweep_plan_id": plan.plan_id,
                        "sweep_config_id": self.SWEEP_CONFIG_ID,
                        "case": selected.case_id,
                        "metric": metric,
                        "status": "VALID",
                        "value": "7",
                        "pmu_status": "sampled",
                        "num_passes": "1",
                        "returncode": "0",
                        "error": "",
                        "collection_session_id": session_id,
                        "order_index": str(order_index),
                        "gpu_clock_hz_before": "1500000000",
                        "gpu_clock_hz_after": "1500000000",
                        "temperature_c_before": "50",
                        "temperature_c_after": "51",
                        "environment_status": "sampled",
                        "environment_source": "nvml",
                    })
        return manifest, latency, plan_path, valid, rows

    def test_accepts_complete_p0_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["target_op_types"], list(self.module.P0_TARGET_OP_TYPES))
        self.assertEqual(summary["selected_case_count"], 35)
        self.assertEqual(summary["metric_count"], 2)
        self.assertEqual(summary["row_count"], 70)
        self.assertEqual(summary["session_count"], 35)

    def test_rejects_empty_or_incomplete_semantic_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            manifest = json.loads(paths[0].read_text(encoding="utf-8"))
            manifest_cases = {case["name"]: case for case in manifest["cases"]}
            plan = load_case_selection_plan(paths[2])
            selected = manifest_cases[plan.selected_cases[0].case_id]
            selected["shape_provenance"]["input0"] = {}
            selected["workload_provenance"]["algorithmic_bytes"] = {
                "status": "theoretical_estimate",
                "source": "adapter_contract",
                "method": "semantic_formula",
            }
            errors = []
            self.module._validate_selected_semantics(
                plan.selected_cases, manifest_cases, errors
            )

        self.assertTrue(
            any("shape.input0 provenance 为空" in error for error in errors),
            errors,
        )
        self.assertTrue(
            any(
                "workload.algorithmic_bytes provenance 缺少字段" in error
                for error in errors
            ),
            errors,
        )

    def test_rejects_invalid_semantic_provenance_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            manifest = json.loads(paths[0].read_text(encoding="utf-8"))
            manifest_cases = {case["name"]: case for case in manifest["cases"]}
            plan = load_case_selection_plan(paths[2])
            selected = manifest_cases[plan.selected_cases[0].case_id]
            selected["shape_provenance"]["input0"]["status"] = "inferred"
            selected["shape_provenance"]["output0"]["source"] = " "
            selected["shape_provenance"]["output0"]["method"] = ""
            selected["workload_provenance"]["algorithmic_bytes"]["formula"] = " "
            selected["workload_provenance"]["algorithmic_flops"][
                "status"
            ] = "not_applicable"
            selected["workload_provenance"]["output_elements"][
                "confidence"
            ] = float("nan")
            errors = []
            self.module._validate_selected_semantics(
                plan.selected_cases, manifest_cases, errors
            )

            no_value_provenance = {
                "status": "declared",
                "source": "adapter_contract",
                "method": "semantic_formula",
                "formula": "n",
                "confidence": 1.0,
                "notes": "",
            }
            self.module._validate_provenance(
                "standalone_case",
                "workload",
                "algorithmic_flops",
                no_value_provenance,
                errors,
                value_present=False,
            )

        self.assertTrue(any("status 非法：inferred" in error for error in errors), errors)
        self.assertTrue(any("source 必须是非空字符串" in error for error in errors), errors)
        self.assertTrue(any("method 必须是非空字符串" in error for error in errors), errors)
        self.assertTrue(any("适用事实必须提供非空 formula" in error for error in errors), errors)
        self.assertTrue(any("同时有数值和 not_applicable" in error for error in errors), errors)
        self.assertTrue(any("confidence 必须是 [0, 1] 内有限数" in error for error in errors), errors)
        self.assertTrue(any("没有数值但未标为 not_applicable" in error for error in errors), errors)

    def test_rejects_fractional_launch_geometry_and_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            case = next(iter(payload["cases"]))
            launch = payload["cases"][case]["launch_records"][0]
            launch["grid"][0] = 1.5
            launch["registers_per_thread"] = 16.5
            paths[1].write_text(json.dumps(payload), encoding="utf-8")
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("正三维整数向量" in error for error in summary["errors"]),
            summary["errors"],
        )
        self.assertTrue(
            any("非负整数实测值" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_rejects_non_selected_latency_status_inconsistency_and_unknown_status(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(
                Path(directory),
                case_counts={"matmul": 7},
            )
            plan = load_case_selection_plan(paths[2])
            selected_ids = {case.case_id for case in plan.selected_cases}
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            non_selected = sorted(set(payload["cases"]) - selected_ids)
            self.assertEqual(len(non_selected), 2)

            inconsistent = payload["cases"][non_selected[0]]
            inconsistent["launch_sampling_status"] = "unavailable"
            inconsistent["environment"]["sampling_status"] = "unavailable"

            unknown = payload["cases"][non_selected[1]]
            unknown["launch_sampling_status"] = "future_launch_status"
            unknown["environment"]["sampling_status"] = "future_environment_status"
            paths[1].write_text(json.dumps(payload), encoding="utf-8")

            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("不得携带 launch records" in error for error in summary["errors"]),
            summary["errors"],
        )
        self.assertTrue(
            any("与完整实测值矛盾" in error for error in summary["errors"]),
            summary["errors"],
        )
        self.assertTrue(
            any("launch_sampling_status 未知" in error for error in summary["errors"]),
            summary["errors"],
        )
        self.assertTrue(
            any("environment sampling_status 未知" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_rejects_sampled_latency_without_launch_or_complete_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            case = next(iter(payload["cases"]))
            payload["cases"][case]["launch_records"] = []
            environment = payload["cases"][case]["environment"]
            environment["sampling_source"] = ""
            environment.pop("temperature_c_after")
            paths[1].write_text(json.dumps(payload), encoding="utf-8")

            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("标记 sampled 但没有 launch record" in error for error in summary["errors"]),
            summary["errors"],
        )
        self.assertTrue(
            any("environment sampling_source 为空" in error for error in summary["errors"]),
            summary["errors"],
        )
        self.assertTrue(
            any("sampled environment 缺少实测 temperature_c_after" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_rejects_p0_plan_with_missing_or_extra_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_paths = self._fixture(
                root / "missing",
                plan_target_op_types=self.module.P0_TARGET_OP_TYPES[:-1],
            )
            missing = self.module.validate_sources(
                operator_cases=missing_paths[0],
                latency_json=missing_paths[1],
                selection_plan=missing_paths[2],
                valid_csv=missing_paths[3],
                pmc_rows=missing_paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

            extra_op_types = self.module.P0_TARGET_OP_TYPES + ("elementwise",)
            extra_paths = self._fixture(
                root / "extra",
                manifest_op_types=extra_op_types,
                plan_target_op_types=extra_op_types,
            )
            extra = self.module.validate_sources(
                operator_cases=extra_paths[0],
                latency_json=extra_paths[1],
                selection_plan=extra_paths[2],
                valid_csv=extra_paths[3],
                pmc_rows=extra_paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(missing["valid"])
        self.assertTrue(
            any("缺少 target：transpose" in error for error in missing["errors"]),
            missing["errors"],
        )
        self.assertFalse(extra["valid"])
        self.assertTrue(
            any("多余 target：elementwise" in error for error in extra["errors"]),
            extra["errors"],
        )

    def test_rejects_p0_target_that_is_not_satisfied(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(
                Path(directory),
                case_counts={"transpose": 4},
            )
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any(
                "P0 target transpose 未由完整显式 condition 满足" in error
                for error in summary["errors"]
            ),
            summary["errors"],
        )

    def test_rejects_latency_from_another_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            payload["source_manifest_sha256"] = "0" * 64
            paths[1].write_text(json.dumps(payload), encoding="utf-8")
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )
        self.assertFalse(summary["valid"])
        self.assertTrue(any("source_manifest_sha256" in error for error in summary["errors"]))

    def test_accepts_exact_latency_top_level_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                set(self.module.LATENCY_TOP_LEVEL_FIELDS),
            )
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertTrue(summary["valid"], summary["errors"])

    def test_rejects_latency_with_extra_top_level_field(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            payload["collector"] = "unexpected"
            paths[1].write_text(json.dumps(payload), encoding="utf-8")
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("顶层字段必须精确为" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_rejects_valid_metric_csv_with_extra_column(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            with paths[3].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["metric", "value", "unexpected"],
                )
                writer.writeheader()
                writer.writerow({
                    "metric": "dram__bytes.sum",
                    "value": "1",
                    "unexpected": "extra",
                })
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("cuda_pmc_valid.csv 表头必须精确为 metric,value" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_rejects_invalid_valid_metric_csv_rows(self):
        cases = {
            "negative": (
                [("dram__bytes.sum", "-1"), ("sm__inst_executed.sum", "2")],
                "value 不是非负有限数",
            ),
            "nan": (
                [("dram__bytes.sum", "NaN"), ("sm__inst_executed.sum", "2")],
                "value 不是非负有限数",
            ),
            "duplicate": (
                [("dram__bytes.sum", "1"), ("dram__bytes.sum", "2")],
                "有重复有效 metric：dram__bytes.sum",
            ),
            "empty_metric": (
                [("", "1"), ("sm__inst_executed.sum", "2")],
                "metric 为空",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, (valid_rows, expected_error) in cases.items():
                with self.subTest(name=name):
                    paths = self._fixture(root / name)
                    with paths[3].open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(self.module.VALID_METRIC_COLUMNS)
                        writer.writerows(valid_rows)
                    summary = self.module.validate_sources(
                        operator_cases=paths[0],
                        latency_json=paths[1],
                        selection_plan=paths[2],
                        valid_csv=paths[3],
                        pmc_rows=paths[4],
                        require_complete_targets=True,
                        require_measured_environment=True,
                    )
                    self.assertFalse(summary["valid"])
                    self.assertTrue(
                        any(expected_error in error for error in summary["errors"]),
                        summary["errors"],
                    )

    def test_rejects_truncated_cartesian_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            lines = paths[4].read_text(encoding="utf-8").splitlines()
            paths[4].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )
        self.assertFalse(summary["valid"])
        self.assertTrue(any("行数" in error for error in summary["errors"]))

    def test_rejects_self_signed_plan_that_was_not_rebuilt_by_current_algorithm(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            payload = json.loads(paths[2].read_text(encoding="utf-8"))
            payload["selected_cases"][0]["shapes"]["input0"] = [999]
            payload["plan_id"] = selection_module._plan_id(payload)
            paths[2].write_text(json.dumps(payload), encoding="utf-8")
            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("current selection algorithm" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_rejects_p0_plan_with_wrong_policy_or_condition_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            too_small = self._fixture(
                root / "too-small",
                minimum_conditions_per_op_type=1,
            )
            too_small_summary = self.module.validate_sources(
                operator_cases=too_small[0],
                latency_json=too_small[1],
                selection_plan=too_small[2],
                valid_csv=too_small[3],
                pmc_rows=too_small[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )
            all_cases = self._fixture(root / "all", plan_policy="all")
            all_cases_summary = self.module.validate_sources(
                operator_cases=all_cases[0],
                latency_json=all_cases[1],
                selection_plan=all_cases[2],
                valid_csv=all_cases[3],
                pmc_rows=all_cases[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(too_small_summary["valid"])
        self.assertTrue(
            any("每类 condition 目标必须是 5" in error for error in too_small_summary["errors"]),
            too_small_summary["errors"],
        )
        self.assertFalse(all_cases_summary["valid"])
        self.assertTrue(
            any("policy 必须是 condition-balanced" in error for error in all_cases_summary["errors"]),
            all_cases_summary["errors"],
        )

    def test_rejects_zero_measured_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latency_paths = self._fixture(root / "latency")
            latency_payload = json.loads(latency_paths[1].read_text(encoding="utf-8"))
            selected_case = next(iter(latency_payload["cases"]))
            latency_payload["cases"][selected_case]["environment"]["temperature_c_before"] = 0
            latency_paths[1].write_text(json.dumps(latency_payload), encoding="utf-8")
            latency_summary = self.module.validate_sources(
                operator_cases=latency_paths[0],
                latency_json=latency_paths[1],
                selection_plan=latency_paths[2],
                valid_csv=latency_paths[3],
                pmc_rows=latency_paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

            row_paths = self._fixture(root / "rows")
            with row_paths[4].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows[:2]:
                row["temperature_c_before"] = "0"
            with row_paths[4].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.module.ROW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            row_summary = self.module.validate_sources(
                operator_cases=row_paths[0],
                latency_json=row_paths[1],
                selection_plan=row_paths[2],
                valid_csv=row_paths[3],
                pmc_rows=row_paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(latency_summary["valid"])
        self.assertTrue(
            any("temperature_c_before" in error for error in latency_summary["errors"]),
            latency_summary["errors"],
        )
        self.assertFalse(row_summary["valid"])
        self.assertTrue(
            any("temperature_c_before" in error for error in row_summary["errors"]),
            row_summary["errors"],
        )

    def test_rejects_negative_valid_value_and_sampled_environment_without_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            negative_paths = self._fixture(root / "negative")
            with negative_paths[4].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["value"] = "-1"
            with negative_paths[4].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.module.ROW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            negative = self.module.validate_sources(
                operator_cases=negative_paths[0],
                latency_json=negative_paths[1],
                selection_plan=negative_paths[2],
                valid_csv=negative_paths[3],
                pmc_rows=negative_paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

            source_paths = self._fixture(root / "source")
            with source_paths[4].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows[:2]:
                row["environment_source"] = ""
            with source_paths[4].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.module.ROW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            source = self.module.validate_sources(
                operator_cases=source_paths[0],
                latency_json=source_paths[1],
                selection_plan=source_paths[2],
                valid_csv=source_paths[3],
                pmc_rows=source_paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(negative["valid"])
        self.assertTrue(
            any("非负有限数" in error for error in negative["errors"]),
            negative["errors"],
        )
        self.assertFalse(source["valid"])
        self.assertTrue(
            any("environment_source 为空" in error for error in source["errors"]),
            source["errors"],
        )

    def test_rejects_reopened_or_internally_inconsistent_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = (
                "dram__bytes.sum",
                "sm__inst_executed.sum",
                "lts__t_sectors.sum",
            )
            reopened_paths = self._fixture(root / "reopened", metrics=metrics)
            with reopened_paths[4].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[1]["collection_session_id"] = "inserted-session"
            rows[1]["order_index"] = "2"
            for row in rows[3:]:
                row["order_index"] = str(int(row["order_index"]) + 1)
            with reopened_paths[4].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.module.ROW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            reopened = self.module.validate_sources(
                operator_cases=reopened_paths[0],
                latency_json=reopened_paths[1],
                selection_plan=reopened_paths[2],
                valid_csv=reopened_paths[3],
                pmc_rows=reopened_paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

            mismatch_paths = self._fixture(root / "mismatch")
            with mismatch_paths[4].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[1]["num_passes"] = "2"
            with mismatch_paths[4].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.module.ROW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            mismatch = self.module.validate_sources(
                operator_cases=mismatch_paths[0],
                latency_json=mismatch_paths[1],
                selection_plan=mismatch_paths[2],
                valid_csv=mismatch_paths[3],
                pmc_rows=mismatch_paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(reopened["valid"])
        self.assertTrue(
            any("非连续重开" in error for error in reopened["errors"]),
            reopened["errors"],
        )
        self.assertFalse(mismatch["valid"])
        self.assertTrue(
            any("case/order/result/environment 不一致" in error for error in mismatch["errors"]),
            mismatch["errors"],
        )

    def test_accepts_metric_local_error_inside_one_session(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            with paths[4].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[1].update({
                "status": "OVERFLOW",
                "value": "",
                "error": "metric evaluation overflow",
            })
            with paths[4].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.module.ROW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["status_counts"]["OVERFLOW"], 1)

    def test_rejects_unsuccessful_session_facts_for_allowed_metric_statuses(self):
        invalid_facts = (
            ("pmu_status", "unavailable", "pmu_status 非 sampled"),
            ("returncode", "7", "returncode 非 0"),
            ("num_passes", "0", "num_passes 无效"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for status in ("NOT_FOUND", "OVERFLOW"):
                for field, value, expected_error in invalid_facts:
                    with self.subTest(status=status, field=field):
                        paths = self._fixture(root / "{}_{}".format(status, field))
                        with paths[4].open(newline="", encoding="utf-8") as stream:
                            rows = list(csv.DictReader(stream))
                        first_session = rows[0]["collection_session_id"]
                        for row in rows:
                            if row["collection_session_id"] != first_session:
                                continue
                            row.update({
                                "status": status,
                                "value": "",
                                "error": "metric-local diagnostic",
                                field: value,
                            })
                        with paths[4].open("w", newline="", encoding="utf-8") as stream:
                            writer = csv.DictWriter(
                                stream, fieldnames=self.module.ROW_COLUMNS
                            )
                            writer.writeheader()
                            writer.writerows(rows)

                        summary = self.module.validate_sources(
                            operator_cases=paths[0],
                            latency_json=paths[1],
                            selection_plan=paths[2],
                            valid_csv=paths[3],
                            pmc_rows=paths[4],
                            require_complete_targets=True,
                            require_measured_environment=True,
                        )

                        self.assertFalse(summary["valid"])
                        self.assertTrue(
                            any(
                                status in error and expected_error in error
                                for error in summary["errors"]
                            ),
                            summary["errors"],
                        )

    def test_rejects_noncanonical_sweep_config_id(self):
        invalid_ids = (
            "config-1",
            "A" * 64,
            "g" * 64,
            " " + "a" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, invalid_id in enumerate(invalid_ids):
                with self.subTest(sweep_config_id=invalid_id):
                    paths = self._fixture(root / str(index))
                    with paths[4].open(newline="", encoding="utf-8") as stream:
                        rows = list(csv.DictReader(stream))
                    for row in rows:
                        row["sweep_config_id"] = invalid_id
                    with paths[4].open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(
                            stream, fieldnames=self.module.ROW_COLUMNS
                        )
                        writer.writeheader()
                        writer.writerows(rows)

                    summary = self.module.validate_sources(
                        operator_cases=paths[0],
                        latency_json=paths[1],
                        selection_plan=paths[2],
                        valid_csv=paths[3],
                        pmc_rows=paths[4],
                        require_complete_targets=True,
                        require_measured_environment=True,
                    )

                    self.assertFalse(summary["valid"])
                    self.assertTrue(
                        any(
                            "sweep_config_id 不是 64 位小写 SHA256" in error
                            for error in summary["errors"]
                        ),
                        summary["errors"],
                    )

    def test_rejects_rows_with_extra_data_column(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            lines = paths[4].read_text(encoding="utf-8").splitlines()
            lines[1] += ",unexpected"
            paths[4].write_text("\n".join(lines) + "\n", encoding="utf-8")

            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("包含额外列" in error for error in summary["errors"]),
            summary["errors"],
        )

    def test_formal_p0_rejects_infrastructure_failure_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            with paths[4].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0].update({
                "status": "COMMAND_FAILED",
                "value": "",
                "pmu_status": "stop_failed: test failure",
                "error": "test command failure",
            })
            rows[1].update({
                "status": "MALFORMED_OUTPUT",
                "value": "",
                "pmu_status": "stop_failed: test failure",
                "error": "test malformed output",
            })
            with paths[4].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.module.ROW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            summary = self.module.validate_sources(
                operator_cases=paths[0],
                latency_json=paths[1],
                selection_plan=paths[2],
                valid_csv=paths[3],
                pmc_rows=paths[4],
                require_complete_targets=True,
                require_measured_environment=True,
            )

        self.assertFalse(summary["valid"])
        self.assertTrue(
            any("COMMAND_FAILED" in error for error in summary["errors"]),
            summary["errors"],
        )
        self.assertTrue(
            any("MALFORMED_OUTPUT" in error for error in summary["errors"]),
            summary["errors"],
        )


if __name__ == "__main__":
    unittest.main()
