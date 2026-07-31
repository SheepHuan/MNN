#!/usr/bin/env python3
import csv
import importlib.util
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

    def test_discover_cuda_cases_uses_only_cuda_cases(self):
        corpus_root = Path("replay_benchmark/kernel_corpus")
        cases = SWEEP_MODULE.discover_cuda_cases(corpus_root)
        self.assertEqual(len(cases), len(set(cases)))
        self.assertIn("cuda_conv_dw_fp32_smoke", cases)
        corpus = json.loads((corpus_root / "operator_cases.json").read_text(encoding="utf-8"))
        cuda_entries = [entry for entry in corpus["cases"] if entry.get("backend") == "cuda"]
        self.assertEqual(len(cases), len({entry["op_type"] for entry in cuda_entries}))
        selected = [entry for entry in corpus["cases"] if entry.get("name") in cases]
        self.assertEqual(len({entry["op_type"] for entry in selected}), len(cases))
        self.assertEqual(len({(entry["op_type"], entry["variant"]) for entry in selected}), len(cases))

        all_cases = SWEEP_MODULE.discover_cuda_cases(
            corpus_root, "all"
        )
        self.assertEqual(len(all_cases), len({entry["name"] for entry in cuda_entries}))
        self.assertEqual(len(all_cases), len(set(all_cases)))

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
        self.assertFalse(observed["output_path"].exists())
        self.assertFalse(observed["output_path"].parent.exists())

    def test_sweep_uses_valid_metrics_and_resume_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_csv = root / "valid.csv"
            with valid_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["metric", "value"])
                writer.writeheader()
                writer.writerows([
                    {"metric": "sm__cycles_elapsed.avg", "value": "1"},
                    {"metric": "dram__bytes.sum", "value": "2"},
                ])
            args = SimpleNamespace(
                corpus_root="replay_benchmark/kernel_corpus", valid_csv=str(valid_csv),
                binary="bench", workdir=".", lib_dir=None, sudo=False, runs=1, timeout=5,
                output_csv=str(root / "rows.csv"), output_json=str(root / "result.json"),
                resume=False, max_cases=1, max_metrics=2, metrics_per_session=32, case_filter=None,
                case_selection="op-type",
            )
            calls = []

            def fake_executor(_binary, _corpus, case, metrics, _workdir, _lib_dir, _sudo, _runs, _timeout):
                calls.append((case, tuple(metrics)))
                return {metric: {"status": "VALID", "value": "7", "pmu_status": "sampled",
                                 "num_passes": 2, "returncode": 0, "error": ""}
                        for metric in metrics}

            payload = SWEEP_MODULE.run_sweep(args, executor=fake_executor)
            self.assertEqual(calls, [("cuda_argmax_fp32_smoke", ("sm__cycles_elapsed.avg", "dram__bytes.sum"))])
            self.assertEqual(payload[calls[0][0]]["pmc"], {
                "sm__cycles_elapsed.avg": 7, "dram__bytes.sum": 7,
            })

            args.resume = True
            SWEEP_MODULE.run_sweep(args, executor=lambda *unused: self.fail("resume reran a completed pair"))

            with (root / "rows.csv").open("a", encoding="utf-8") as stream:
                stream.write("old_case,old_metric,VALID,99,sampled,1,0,\n")
            payload = SWEEP_MODULE.build_result_json_from_csv(
                ["cuda_argmax_fp32_smoke"], root / "rows.csv"
            )
            self.assertNotIn("old_case", payload)


if __name__ == "__main__":
    unittest.main()
