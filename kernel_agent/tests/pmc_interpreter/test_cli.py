#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from pathlib import Path

from kernel_agent.pmc_interpreter.cli import main


class PmcInterpreterCliTest(unittest.TestCase):
    def test_cuda_cli_writes_structured_report_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            latency_values = {}
            rows = root / "rows.csv"
            with rows.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "case",
                        "metric",
                        "status",
                        "value",
                        "pmu_status",
                        "num_passes",
                        "returncode",
                        "error",
                    ],
                )
                writer.writeheader()
                for index in range(1, 5):
                    case = "case_{}".format(index)
                    cases.append(
                        {
                            "name": case,
                            "backend": "cuda",
                            "op_type": "relu",
                            "variant": "cuda_relu_{}".format(index),
                            "tag": "test",
                            "dtype": "float32",
                            "int_params": {"size": 16},
                        }
                    )
                    latency_values[case] = float(index + 5)
                    writer.writerow(
                        {
                            "case": case,
                            "metric": "dram__bytes.sum",
                            "status": "VALID",
                            "value": index * 10,
                            "pmu_status": "sampled",
                            "num_passes": 1,
                            "returncode": 0,
                            "error": "",
                        }
                    )
            operator_cases = root / "operator_cases.json"
            operator_cases.write_text(
                json.dumps({"cases": cases}), encoding="utf-8"
            )
            latency = root / "latency.json"
            latency.write_text(json.dumps(latency_values), encoding="utf-8")
            output = root / "analysis-report.json"
            markdown = root / "analysis-report.md"
            result = main(
                [
                    "--pmc-csv",
                    str(rows),
                    "--latency-json",
                    str(latency),
                    "--operator-cases",
                    str(operator_cases),
                    "--output-json",
                    str(output),
                    "--output-md",
                    str(markdown),
                    "--minimum-group-samples",
                    "3",
                ]
            )
            self.assertEqual(result, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"], "mnn-pmc-analysis-report/v1"
            )
            self.assertEqual(payload["scope"]["platform"], "cuda")
            self.assertIn(
                "descriptive_canonical", payload["canonical_metric_sets"]
            )
            self.assertEqual(
                [record["layer"] for record in payload["layer_history"]],
                [
                    "projection",
                    "quality",
                    "redundancy",
                    "correlation",
                    "kernel_signatures",
                    "delta_relations",
                ],
            )
            markdown_text = markdown.read_text(encoding="utf-8")
            self.assertTrue(markdown_text.startswith("# PMC 语义分析报告\n"))
            self.assertIn("## PMC 相关性分类", markdown_text)
            self.assertIn("## Kernel Type × PMC Signature", markdown_text)
            self.assertIn("## ΔPMC 与 ΔLatency 规则", markdown_text)


if __name__ == "__main__":
    unittest.main()
