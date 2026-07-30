#!/usr/bin/env python3
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_backend_valid_metrics.py"
SPEC = importlib.util.spec_from_file_location("merge_backend_valid_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BackendMetricMergeTest(unittest.TestCase):
    def test_classify_metric_uses_cache_memory_compute_other_order(self):
        self.assertEqual(MODULE.classify_metric("gpu_active_cycles"), "compute")
        self.assertEqual(MODULE.classify_metric("tp_l1_cacheline_misses"), "cache")
        self.assertEqual(MODULE.classify_metric("l2_ext_write"), "cache")
        self.assertEqual(MODULE.classify_metric("Texture filtering cycles"), "cache")
        self.assertEqual(MODULE.classify_metric("output_external_write_beats"), "memory")
        self.assertEqual(MODULE.classify_metric("cp_sqe_sync_stall"), "other")
        self.assertEqual(MODULE.classify_metric("gpu_interrupts"), "other")

    def test_merge_keeps_union_and_backend_valid_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opencl = root / "opencl.csv"
            vulkan = root / "vulkan.csv"
            output = root / "merged.csv"
            for path, rows in (
                (opencl, [
                    {"pmu_metric_name": "gpu_active_cycles", "valid": "true", "valid_cases": "buffer_fp32"},
                    {"pmu_metric_name": "l2_any_lookup", "valid": "true", "valid_cases": "buffer_stride"},
                ]),
                (vulkan, [
                    {"pmu_metric_name": "l2_any_lookup", "valid": "true", "valid_cases": "model.mnn"},
                    {"pmu_metric_name": "l2_ext_write", "valid": "true", "valid_cases": "model.mnn"},
                ]),
            ):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=MODULE.INPUT_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)

            rows = MODULE.merge_metric_summaries(opencl, vulkan, output)
            self.assertEqual([row["pmu_metric_name"] for row in rows], [
                "gpu_active_cycles", "l2_any_lookup", "l2_ext_write",
            ])
            self.assertEqual(rows[1]["valid_backends"], "opencl;vulkan")
            self.assertEqual(rows[1]["opencl_valid_cases"], "buffer_stride")
            self.assertEqual(rows[1]["vulkan_valid_cases"], "model.mnn")
            self.assertEqual(rows[2]["opencl_valid"], "false")
            self.assertEqual(rows[2]["category"], "cache")
            with output.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.reader(stream)), MODULE.OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
