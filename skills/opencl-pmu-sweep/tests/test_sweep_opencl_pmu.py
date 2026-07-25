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


if __name__ == "__main__":
    unittest.main()
