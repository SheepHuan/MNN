#!/usr/bin/env python3
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "sweep_opencl_pmu.py"
SPEC = importlib.util.spec_from_file_location("sweep_opencl_pmu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SweepTest(unittest.TestCase):
    def test_build_model_benchmark_argv_uses_one_event_and_session_runs(self):
        args = SimpleNamespace(workload_runs=5, warmup_runs=2, control_runs=1)
        self.assertEqual(
            MODULE.build_model_benchmark_argv("bench", "/models/resnet.mnn", "gpu_active_cycles", args),
            [
                "bench", "--model-pmu-bench", "--model", "/models/resnet.mnn",
                "--model-pmu-workload-runs", "5", "--model-pmu-warmup-runs", "2",
                "--model-pmu-control-runs", "1", "--perf-counter-events", "gpu_active_cycles",
                "--perf-counter-output", "/tmp/mnn-model-pmu.json",
            ],
        )

    def test_build_model_benchmark_argv_selects_vulkan_forward(self):
        args = SimpleNamespace(workload_runs=3, warmup_runs=1, control_runs=1, model_forward=7)
        argv = MODULE.build_model_benchmark_argv("bench", "/models/resnet.mnn", "l2_any_lookup", args)
        self.assertEqual(argv[argv.index("--model-pmu-forward") + 1], "7")

    def test_extract_model_delta_reads_the_session_case(self):
        report = {
            "format": "mnn-model-pmu-benchmark",
            "cases": [{
                "name": "resnet.mnn",
                "counters": [{"name": "gpu_active_cycles", "control_delta": 11, "workload_delta": 211}],
            }],
        }
        self.assertEqual(MODULE.extract_model_delta(report, "resnet.mnn", "gpu_active_cycles"), "200")

    def test_discover_models_returns_sorted_mnn_files_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.mnn").write_bytes(b"")
            (root / "a.txt").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "a.mnn").write_bytes(b"")
            self.assertEqual(
                [path.name for path in MODULE.discover_models([str(root)])],
                ["a.mnn", "b.mnn"],
            )

    def test_model_sweep_writes_one_row_per_model_and_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.mnn"
            second = root / "b.mnn"
            first.write_bytes(b"")
            second.write_bytes(b"")
            args = SimpleNamespace(
                models="{},{}".format(first, second), model_roots="", events="gpu_active_cycles,compute_tasks",
                model_output_csv=str(root / "models.csv"), model_raw_dir=str(root / "raw"), model_summary_csv=None,
                device=None, remote_root=None, remote_model_root=None, binary="bench", lib_dir=None,
                workload_runs=5, warmup_runs=2, control_runs=1, measurements=1, timeout=7,
            )

            def fake_executor(_binary, model_path, event_name, _args):
                return {"cases": [{
                    "name": Path(model_path).name,
                    "status": "ok",
                    "error": "",
                    "counters": [{"name": event_name, "control_delta": 1, "workload_delta": 4}],
                }]}

            rows = MODULE.run_model_sweep(args, model_executor=fake_executor)
            self.assertEqual(len(rows), 4)
            with (root / "models.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, MODULE.MODEL_CSV_COLUMNS)
                self.assertEqual(len(list(reader)), 4)
            with (root / "models-metrics.csv").open(newline="", encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
                self.assertEqual(summary, [
                    {"pmu_metric_name": "gpu_active_cycles", "valid": "true", "valid_cases": "a.mnn;b.mnn"},
                    {"pmu_metric_name": "compute_tasks", "valid": "true", "valid_cases": "a.mnn;b.mnn"},
                ])

    def test_raw_filename_component_escapes_path_separators(self):
        self.assertEqual(
            MODULE.raw_filename_component("Load/store unit bytes written to L2 per access cycle"),
            "Load%2Fstore%20unit%20bytes%20written%20to%20L2%20per%20access%20cycle",
        )

    def test_all_events_are_discovered_per_device_before_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                events="all", device="rhinopi", remote_root="/mnt/nvme/workspace/replay-benchmark",
                binary="bench", lib_dir=None, timeout=7,
            )

            def fake_runner(_argv, timeout=None):
                self.assertEqual(timeout, 7)
                return {"pmu_status": "available", "pmu_events": ["event_a", "event_b", "event_a"]}

            self.assertEqual(MODULE.discover_device_events(args, runner=fake_runner), ["event_a", "event_b"])

    def test_all_event_sweep_uses_only_discovered_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                cases="buffer_fp32", events="all", output_csv=str(root / "result.csv"), raw_dir=str(root / "raw"),
                summary_csv=None, device="rhinopi", remote_root="/mnt/nvme/workspace/replay-benchmark",
                binary="bench", lib_dir=None, size=1, local_size=32, iterations=1,
                workload_runs=1, warmup_runs=0, measurements=1, timeout=7,
            )
            calls = []

            def fake_discovery(_args):
                return ["rhinopi_only_event"]

            def fake_executor(_binary, case_name, event_name, _args):
                calls.append((case_name, event_name))
                return {"cases": [{"name": case_name, "counters": [{"name": event_name, "control_delta": 1, "workload_delta": 2}]}]}

            MODULE.run_sweep(args, executor=fake_executor, discover=fake_discovery)
            self.assertEqual(calls, [("buffer_fp32", "rhinopi_only_event")])

    def test_batch_case_sweep_keeps_one_row_per_case_and_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                cases="buffer_fp32,buffer_stride", events="gpu_active_cycles",
                output_csv=str(root / "result.csv"), raw_dir=str(root / "raw"), summary_csv=None,
                device=None, remote_root=None, binary="bench", lib_dir=None, size=1, local_size=32,
                iterations=1, workload_runs=3, warmup_runs=0, measurements=1, timeout=7, batch_cases=True,
            )
            calls = []

            def fake_batch_executor(_binary, case_list, event_name, _args):
                calls.append((case_list, event_name))
                return {"cases": [
                    {"name": "buffer_fp32", "counters": [{"name": event_name, "control_delta": 1, "workload_delta": 3}]},
                    {"name": "buffer_stride", "counters": [{"name": event_name, "control_delta": 2, "workload_delta": 2}]},
                ]}

            rows = MODULE.run_sweep(args, batch_executor=fake_batch_executor)
            self.assertEqual(calls, [("buffer_fp32,buffer_stride", "gpu_active_cycles")])
            self.assertEqual([row["delta_metric"] for row in rows], ["2", "0"])

    def test_extract_delta_is_signed_workload_minus_control(self):
        fixture = Path(__file__).parent / "fixtures" / "sample_pmu_report.json"
        report = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(MODULE.extract_delta(report, "buffer_fp32", "gpu_active_cycles"), "750")

    def test_negative_delta_is_preserved(self):
        report = {"cases": [{"name": "buffer_fp32", "counters": [{"name": "gpu_active_cycles", "control_delta": 850, "workload_delta": 100}]}]}
        self.assertEqual(MODULE.extract_delta(report, "buffer_fp32", "gpu_active_cycles"), "-750")

    def test_run_sweep_writes_exact_csv_columns_and_one_row_per_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                cases="buffer_fp32,buffer_stride",
                events="gpu_active_cycles,sp_busy_cycles",
                output_csv=str(root / "result.csv"),
                raw_dir=str(root / "raw"),
                device=None,
                remote_root=None,
                binary="replay_benchmark.out",
                lib_dir=None,
                size=262144,
                local_size=128,
                iterations=30,
                workload_runs=3,
                warmup_runs=1,
                measurements=2,
            )

            def fake_executor(_binary, case_name, event_name, _args):
                return {"cases": [{"name": case_name, "counters": [{"name": event_name, "control_delta": 2, "workload_delta": 9}]}]}

            rows = MODULE.run_sweep(args, executor=fake_executor)
            self.assertEqual(len(rows), 8)
            with (root / "result.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, MODULE.CSV_COLUMNS)
                self.assertEqual(len(list(reader)), 8)
            self.assertEqual(len(list((root / "raw").glob("*.json"))), 8)
            with (root / "result-metrics.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                summary = list(reader)
                self.assertEqual(reader.fieldnames, MODULE.SUMMARY_COLUMNS)
                self.assertEqual(summary, [
                    {"pmu_metric_name": "gpu_active_cycles", "valid": "true", "valid_cases": "buffer_fp32;buffer_stride"},
                    {"pmu_metric_name": "sp_busy_cycles", "valid": "true", "valid_cases": "buffer_fp32;buffer_stride"},
                ])

    def test_failed_runner_keeps_row_without_false_numeric_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                cases="buffer_fp32", events="gpu_active_cycles", output_csv=str(root / "result.csv"),
                raw_dir=str(root / "raw"), size=1, local_size=32, iterations=1,
                workload_runs=1, warmup_runs=0, measurements=1, binary="bench",
            )

            def failed_executor(_binary, _case_name, _event_name, _args):
                raise RuntimeError("unavailable")

            rows = MODULE.run_sweep(args, executor=failed_executor)
            self.assertEqual(rows[0]["delta_metric"], "")

    def test_benchmark_argv_contains_one_event(self):
        args = SimpleNamespace(size=1, local_size=32, iterations=1, workload_runs=1, warmup_runs=0)
        argv = MODULE.build_benchmark_argv("bench", "buffer_fp32", "gpu_active_cycles", args)
        self.assertEqual(argv[argv.index("--perf-counter-events") + 1], "gpu_active_cycles")
        with self.assertRaises(ValueError):
            MODULE.build_benchmark_argv("bench", "buffer_fp32", "a,b", args)

    def test_remote_model_roots_map_dnn_and_llm_packages_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dnn = root / "model.mnn"
            dnn.write_bytes(b"")
            package = root / "llama-package"
            package.mkdir()
            (package / "llm_config.json").write_text("{}", encoding="utf-8")
            llm = package / "llama.mnn"
            llm.write_bytes(b"")
            args = SimpleNamespace(
                workload_runs=1, warmup_runs=0, control_runs=1, precision=None,
                device="rhinopi", remote_root="/remote", remote_model_root=None,
                remote_dnn_root="/remote/models", remote_llm_root="/remote/llm-models",
                timeout=7,
            )
            dnn_argv = MODULE.build_model_benchmark_argv(
                "/remote/bin/replay_benchmark.out", "/remote/models/model.mnn", "event", args,
            )
            self.assertEqual(dnn_argv[dnn_argv.index("--model") + 1], "/remote/models/model.mnn")
            # execute_model's path selection is exercised through a fake runner.
            seen = []
            def fake_runner(argv, timeout=None):
                seen.append(argv)
                return {"cases": []}
            original = MODULE.run_process
            MODULE.run_process = fake_runner
            try:
                MODULE.execute_model("bench", str(dnn), "event", args)
                MODULE.execute_model("bench", str(llm), "event", args)
            finally:
                MODULE.run_process = original
            self.assertIn("/remote/models/model.mnn", seen[0][-1])
            self.assertIn("/remote/llm-models/llama-package/llama.mnn", seen[1][-1])


if __name__ == "__main__":
    unittest.main()
