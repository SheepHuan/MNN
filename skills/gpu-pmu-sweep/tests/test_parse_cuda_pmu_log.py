#!/usr/bin/env python3
import csv
import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOG_MODULE = load("parse_cuda_pmu_log", ROOT / "scripts" / "parse_cuda_pmu_log.py")
SWEEP_MODULE = load("sweep_cuda_kernel_pmc", ROOT / "scripts" / "sweep_cuda_kernel_pmc.py")


class CudaPmuSweepTest(unittest.TestCase):
    def test_parse_fixed_log_and_write_valid_only_csv(self):
        fixture = Path(__file__).parent / "fixtures" / "sample_cuda_pmu_sweep.log"
        records = LOG_MODULE.parse_sweep_log(fixture.read_text(encoding="utf-8"))
        self.assertEqual([record["status"] for record in records], [
            "VALID", "NOT_FOUND", "OVERFLOW", "COMMAND_FAILED",
        ])
        with tempfile.TemporaryDirectory() as directory:
            all_csv = Path(directory) / "all.csv"
            valid_csv = Path(directory) / "valid.csv"
            LOG_MODULE.write_csvs(records, all_csv, valid_csv)
            with valid_csv.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [{"metric": "sm__inst_executed.sum", "value": "826"}])
            with all_csv.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)

    @staticmethod
    def _case(name, op_type, *, dtype="float32", shapes=None, workload=None, params=None,
              variant=None, tag="test"):
        case = {
            "name": name,
            "backend": "cuda",
            "op_type": op_type,
            "variant": variant or name,
            "tag": tag,
            "int_params": params or {},
        }
        if dtype is not None:
            case["dtype"] = dtype
        if shapes is not None:
            case["shapes"] = shapes
        if workload is not None:
            case["workload"] = workload
        return case

    @staticmethod
    def _write_manifest(path, cases):
        path.mkdir(parents=True, exist_ok=True)
        manifest = path / "operator_cases.json"
        manifest.write_text(json.dumps({
            "format": "mnn-kernel-operator-cases",
            "version": 1,
            "cases": cases,
        }), encoding="utf-8")
        return manifest

    def test_condition_balanced_prefers_complete_metadata_and_ignores_manifest_order(self):
        explicit = [
            self._case(
                "explicit_{}".format(index), "sample",
                shapes={"input": [1, index + 1]},
                workload={"output_elements": index + 1},
                params={"size": index + 1},
            )
            for index in range(5)
        ]
        proxies = [
            self._case("proxy_{}".format(index), "sample", params={"size": 100 + index})
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write_manifest(root / "first", proxies + explicit)
            second = self._write_manifest(root / "second", list(reversed(proxies + explicit)))
            first_plan = SWEEP_MODULE.build_case_selection_plan(first, backend="cuda")
            second_plan = SWEEP_MODULE.build_case_selection_plan(second, backend="cuda")

        first_ids = [case.case_id for case in first_plan.selected_cases]
        second_ids = [case.case_id for case in second_plan.selected_cases]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), 5)
        self.assertTrue(all(case.metadata_level == "explicit" for case in first_plan.selected_cases))
        self.assertEqual(first_plan.groups["sample"].status, "satisfied")
        self.assertNotEqual(first_plan.plan_id, second_plan.plan_id)

    def test_condition_balanced_uses_proxy_only_for_fill_and_selects_all_when_insufficient(self):
        mixed = [
            self._case(
                "mixed_explicit_{}".format(index), "mixed",
                shapes={"input": [index + 1]},
                workload={"output_elements": index + 1},
                params={"size": index + 1},
            )
            for index in range(3)
        ] + [
            self._case("mixed_proxy_{}".format(index), "mixed", params={"size": 10 + index})
            for index in range(3)
        ]
        insufficient = [
            self._case(
                "small_{}".format(index), "small", params={"size": index % 4},
                variant="variant_{}".format(index), tag="v{}".format(index),
            )
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_manifest(Path(directory), mixed + insufficient)
            plan = SWEEP_MODULE.build_case_selection_plan(manifest, backend="cuda")

        mixed_cases = [case for case in plan.selected_cases if case.op_type == "mixed"]
        self.assertEqual([case.metadata_level for case in mixed_cases[:3]], ["explicit"] * 3)
        self.assertEqual([case.metadata_level for case in mixed_cases[3:]], ["params_proxy"] * 2)
        self.assertEqual(plan.groups["mixed"].status, "satisfied_with_proxy")
        self.assertTrue(any("mixed needed proxy" in issue for issue in plan.issues))

        small_group = plan.groups["small"]
        self.assertEqual(small_group.distinct_condition_count, 4)
        self.assertEqual(small_group.selected_case_count, 6)
        self.assertEqual(small_group.status, "insufficient_unique_conditions")

    def test_classify_fixed_report_preserves_zero_as_valid(self):
        report = {
            "cases": [{
                "case": "cuda_relu_fp32_smoke",
                "pmu_status": "sampled",
                "pmu_metrics": {"sm__sass_thread_inst_executed_op_fadd_pred_on.sum": 0},
                "latency_us": 1.5,
            }],
        }
        row = SWEEP_MODULE.classify_report(
            report, "cuda_relu_fp32_smoke", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
        )
        self.assertEqual(row["status"], "VALID")
        self.assertEqual(row["value"], "0")
        self.assertEqual(row["pmu_status"], "sampled")

    def test_classify_pmu_start_failure_as_command_failed(self):
        report = {
            "cases": [{
                "case": "cuda_relu_fp32_smoke",
                "pmu_status": "cuda_event_fallback(start_failed: CUPTI_ERROR_INSUFFICIENT_PRIVILEGES)",
                "latency_us": 2.0,
            }],
        }
        row = SWEEP_MODULE.classify_report(report, "cuda_relu_fp32_smoke", "dram__bytes.avg")
        self.assertEqual(row["status"], "COMMAND_FAILED")

    def test_build_json_matches_requested_shape(self):
        payload = SWEEP_MODULE.build_result_json(
            ["case_a", "case_b"],
            [
                {"case": "case_a", "metric": "sm__cycles_elapsed.avg", "status": "VALID",
                 "value": "123", "pmu_status": "sampled", "num_passes": "2",
                 "returncode": "0", "error": ""},
                {"case": "case_a", "metric": "dram__bytes.sum", "status": "NOT_FOUND",
                 "value": "", "pmu_status": "sampled", "num_passes": "2",
                 "returncode": "0", "error": ""},
            ],
        )
        self.assertEqual(payload, {
            "case_a": {"pmc": {"sm__cycles_elapsed.avg": 123}},
            "case_b": {"pmc": {}},
        })

    def test_build_command_contains_one_case_and_one_metric(self):
        argv = SWEEP_MODULE.build_benchmark_argv(
            "./replay_benchmark.out", "../replay_benchmark/kernel_corpus",
            "cuda_relu_fp32_smoke", "sm__cycles_elapsed.avg", "/tmp/result.json",
        )
        self.assertEqual(argv[argv.index("--kernel-corpus-case") + 1], "cuda_relu_fp32_smoke")
        self.assertEqual(argv[argv.index("--perf-counter-events") + 1], "sm__cycles_elapsed.avg")
        self.assertNotIn(",", argv[argv.index("--perf-counter-events") + 1])
        self.assertIn("--kernel-corpus-no-latency", argv)

    def test_build_command_batches_metrics_in_one_cupti_session(self):
        argv = SWEEP_MODULE.build_benchmark_argv(
            "bench", "corpus", "case", ["metric_a", "metric_b"], "/tmp/out.json"
        )
        self.assertEqual(argv[argv.index("--perf-counter-events") + 1], "metric_a,metric_b")

    def test_run_batch_uses_absent_output_path_and_cleans_private_directory(self):
        observed = {}

        def fake_runner(argv, **kwargs):
            output_path = Path(argv[argv.index("--perf-counter-output") + 1])
            observed["output_path"] = output_path
            self.assertFalse(output_path.exists())
            output_path.write_text(json.dumps({
                "cases": [{
                    "case": "cuda_relu_fp32_smoke",
                    "pmu_status": "sampled",
                    "num_passes": 1,
                    "pmu_metrics": {"dram__bytes.avg": 123},
                }],
            }), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        result = SWEEP_MODULE.run_batch(
            "bench", "corpus", "cuda_relu_fp32_smoke", ["dram__bytes.avg"],
            ".", use_sudo=True, runner=fake_runner,
        )
        self.assertEqual(result["dram__bytes.avg"]["status"], "VALID")
        self.assertTrue(result["dram__bytes.avg"]["collection_session_id"].startswith("cuda-pmc-"))
        self.assertEqual(result["dram__bytes.avg"]["environment_status"], "not_collected")
        self.assertEqual(result["dram__bytes.avg"]["gpu_clock_hz_before"], "")
        self.assertEqual(result["dram__bytes.avg"]["temperature_c_after"], "")
        self.assertFalse(observed["output_path"].exists())
        self.assertFalse(observed["output_path"].parent.exists())

    def test_sweep_uses_valid_metrics_and_resume_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_root = root / "corpus"
            manifest = self._write_manifest(corpus_root, [
                self._case("case_a", "sample", params={"size": 16}),
            ])
            valid_csv = root / "valid.csv"
            with valid_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["metric", "value"])
                writer.writeheader()
                writer.writerows([
                    {"metric": "sm__cycles_elapsed.avg", "value": "1"},
                    {"metric": "dram__bytes.sum", "value": "2"},
                ])
            args = SimpleNamespace(
                corpus_root=str(corpus_root), valid_csv=str(valid_csv),
                binary="bench", workdir=str(root), lib_dir=None, sudo=False, runs=1, timeout=5,
                output_csv=str(root / "rows.csv"), output_json=str(root / "result.json"),
                resume=False, max_metrics=2, metrics_per_session=32,
                case_selection="condition-balanced", conditions_per_op_type=5,
                target_op_type=None, selection_plan_input=None, selection_plan_output=None,
                device_fingerprint="test-device",
            )
            calls = []

            def fake_executor(_binary, _corpus, case, metrics, _workdir, _lib_dir, _sudo, _runs, _timeout):
                calls.append((case, tuple(metrics)))
                return {metric: {"status": "VALID", "value": "7", "pmu_status": "sampled",
                                 "num_passes": 2, "returncode": 0, "error": "",
                                 "collection_session_id": "session-1",
                                 "order_index": "", "gpu_clock_hz_before": "",
                                 "gpu_clock_hz_after": "", "temperature_c_before": "",
                                 "temperature_c_after": "", "environment_status": "not_collected",
                                 "environment_source": ""}
                        for metric in metrics}

            payload = SWEEP_MODULE.run_sweep(args, executor=fake_executor)
            self.assertEqual(calls, [("case_a", ("sm__cycles_elapsed.avg", "dram__bytes.sum"))])
            self.assertEqual(payload[calls[0][0]]["pmc"], {
                "sm__cycles_elapsed.avg": 7, "dram__bytes.sum": 7,
            })
            selection_path = Path("{}.selection.json".format(args.output_csv))
            self.assertTrue(selection_path.is_file())
            plan = SWEEP_MODULE.load_case_selection_plan(selection_path)
            self.assertEqual(plan.groups["sample"].status, "insufficient_unique_conditions")
            with Path(args.output_csv).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["sweep_plan_id"], plan.plan_id)
            self.assertEqual([row["order_index"] for row in rows], ["1", "2"])
            self.assertEqual({row["collection_session_id"] for row in rows}, {"session-1"})
            self.assertTrue(all(row["gpu_clock_hz_before"] == "" for row in rows))
            self.assertTrue(all(row["temperature_c_after"] == "" for row in rows))

            args.resume = True
            SWEEP_MODULE.run_sweep(args, executor=lambda *unused: self.fail("resume reran a completed pair"))

            args.runs = 2
            with self.assertRaisesRegex(ValueError, "different sweep configuration"):
                SWEEP_MODULE.run_sweep(args, executor=lambda *unused: None)
            args.runs = 1

            original_manifest = manifest.read_text(encoding="utf-8")
            manifest.write_text(original_manifest + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                SWEEP_MODULE.run_sweep(args, executor=lambda *unused: None)
            manifest.write_text(original_manifest, encoding="utf-8")

            with Path(args.output_csv).open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=SWEEP_MODULE.ROW_COLUMNS)
                writer.writerow(rows[-1])
            with self.assertRaisesRegex(ValueError, "outside the current sweep plan"):
                SWEEP_MODULE.run_sweep(args, executor=lambda *unused: None)

    def test_cli_rejects_removed_op_type_alias(self):
        argv = [
            "--valid-csv", "valid.csv",
            "--corpus-root", "corpus",
            "--binary", "bench",
            "--output-csv", "rows.csv",
            "--output-json", "result.json",
            "--case-selection", "op-type",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                SWEEP_MODULE.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
