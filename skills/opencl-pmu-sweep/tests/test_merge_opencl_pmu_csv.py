import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_opencl_pmu_csv.py"
SPEC = importlib.util.spec_from_file_location("merge_opencl_pmu_csv", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeTest(unittest.TestCase):
    def test_merges_synthetic_and_model_rows_and_summarizes_all_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic = root / "synthetic.csv"
            synthetic.write_text(
                "case_name,case_args,case_runs,pmu_metric_name,delta_metric\n"
                "buffer_fp32,args,3,l2_any_lookup,0\n",
                encoding="utf-8",
            )
            model = root / "model.csv"
            model.write_text(
                "case_name,model_path,case_args,case_runs,pmu_metric_name,delta_metric,status,error\n"
                "MobileNetV2@precision0,models/MobileNetV2.mnn,modelargs,3,l2_any_lookup,12,ok,\n"
                "MobileNetV2@precision0,models/MobileNetV2.mnn,modelargs,3,other_event,,error,failed\n",
                encoding="utf-8",
            )
            rows = MODULE.read_rows([synthetic, model])
            output = root / "all.csv"
            summary = root / "all-metrics.csv"
            MODULE.write_report(rows, output, summary)
            with summary.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [
                    {"pmu_metric_name": "l2_any_lookup", "valid": "true",
                     "valid_cases": "MobileNetV2@precision0"},
                    {"pmu_metric_name": "other_event", "valid": "false", "valid_cases": ""},
                ])
            with output.open(newline="", encoding="utf-8") as stream:
                merged = list(csv.DictReader(stream))
            self.assertEqual(merged[1]["case_args"], "modelargs --model-path=models/MobileNetV2.mnn")
            self.assertEqual(merged[2]["case_args"],
                             "modelargs --model-path=models/MobileNetV2.mnn --status=error --error=failed")

    def test_duplicate_rows_are_idempotently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text(
                "case_name,case_args,case_runs,pmu_metric_name,delta_metric\n"
                "case,args,1,event,1\n",
                encoding="utf-8",
            )
            self.assertEqual(len(MODULE.read_rows([source, source])), 1)


if __name__ == "__main__":
    unittest.main()
