#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from pathlib import Path

from kernel_agent.pmc_interpreter import AnalysisConfig
from kernel_agent.pmc_interpreter.dataset import (
    CorrelationOutput,
    DeltaRelationOutput,
    KernelSignatureOutput,
    ProjectionInput,
    ProjectionOutput,
    DeviceSpec,
    PmcDataset,
    QualityOutput,
    RedundancyOutput,
)
from kernel_agent.pmc_interpreter.dataset.sources import (
    load_mnn_kernelreplay_dataset,
)
from kernel_agent.pmc_interpreter.layers import (
    CorrelationLayer,
    DeltaRelationLayer,
    KernelSignatureLayer,
    ProjectionLayer,
    QualityLayer,
    RedundancyLayer,
)


class QualityRedundancyTest(unittest.TestCase):
    def _build_dataset(self, root):
        cases = []
        latencies = {}
        for index in range(1, 5):
            case = "case_{}".format(index)
            cases.append({
                "name": case,
                "backend": "cuda",
                "op_type": "relu",
                "variant": "relu_{}".format(index),
                "tag": "test",
                "dtype": "float32",
                "int_params": {"size": index},
            })
            latencies[case] = float(index + 5)
        operator_cases = root / "operator_cases.json"
        operator_cases.write_text(
            json.dumps({"cases": cases}), encoding="utf-8"
        )
        latency = root / "latency.json"
        latency.write_text(json.dumps(latencies), encoding="utf-8")
        rows = root / "rows.csv"
        with rows.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "case", "metric", "status", "value", "pmu_status",
                "num_passes", "returncode", "error",
            ])
            writer.writeheader()
            for index in range(1, 5):
                case = "case_{}".format(index)
                for metric, value in (
                    ("dram__bytes.avg", index),
                    ("dram__bytes.sum", index * 10),
                    ("lts__t_bytes.sum", index * 10),
                    ("sm__cycles_elapsed.avg", index * 3),
                    ("l1tex__inactive.sum", 0),
                ):
                    writer.writerow({
                        "case": case,
                        "metric": metric,
                        "status": "VALID",
                        "value": value,
                        "pmu_status": "sampled",
                        "num_passes": 1,
                        "returncode": 0,
                        "error": "",
                    })
                    if metric == "dram__bytes.sum":
                        # 同一 legacy 账本中的重复行不是独立 repeat。
                        writer.writerow({
                            "case": case,
                            "metric": metric,
                            "status": "VALID",
                            "value": value,
                            "pmu_status": "sampled",
                            "num_passes": 1,
                            "returncode": 0,
                            "error": "",
                        })
        return load_mnn_kernelreplay_dataset(rows, latency, operator_cases)

    def test_quality_and_same_concept_only_redundancy(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._build_dataset(Path(directory))
            projection = ProjectionLayer().run(
                ProjectionInput(dataset=dataset, config=AnalysisConfig())
            )
            self.assertIs(type(projection), ProjectionOutput)

            quality = QualityLayer().run(projection)
            self.assertIs(type(quality), QualityOutput)
            self.assertIn("cupti::dram__bytes.sum", quality.active_feature_ids)
            self.assertNotIn("cupti::dram__bytes.avg", quality.active_feature_ids)
            self.assertNotIn(
                "cupti::sm__cycles_elapsed.avg", quality.active_feature_ids
            )
            self.assertNotIn(
                "cupti::l1tex__inactive.sum", quality.active_feature_ids
            )
            decision = quality.quality_report.metrics[
                "cupti::dram__bytes.sum"
            ]
            self.assertEqual(decision.quality.conditions_with_repeats, 0)
            self.assertIsNone(decision.quality.repeat_relative_mad)

            redundancy = RedundancyLayer().run(quality)
            self.assertIs(type(redundancy), RedundancyOutput)
            self.assertIn(
                "cupti::dram__bytes.sum", redundancy.active_feature_ids
            )
            self.assertIn(
                "cupti::lts__t_bytes.sum", redundancy.active_feature_ids
            )
            clusters = (
                redundancy.redundancy_report.cross_concept_equal_vector_clusters
            )
            self.assertEqual(len(clusters), 1)
            self.assertEqual(
                clusters[0].disposition, "annotated_only_not_forced"
            )
            self.assertEqual(
                tuple(record.layer for record in redundancy.layer_history),
                ("projection", "quality", "redundancy"),
            )

    def test_layers_reject_skipped_or_wrong_input_types(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._build_dataset(Path(directory))
            projection_input = ProjectionInput(
                dataset=dataset, config=AnalysisConfig(min_group_samples=2)
            )
            projection = ProjectionLayer().run(projection_input)

            with self.assertRaisesRegex(TypeError, "expects ProjectionOutput"):
                QualityLayer().run(projection_input)
            with self.assertRaisesRegex(TypeError, "expects QualityOutput"):
                RedundancyLayer().run(projection)

            quality = QualityLayer().run(projection)
            with self.assertRaisesRegex(TypeError, "expects RedundancyOutput"):
                CorrelationLayer().run(quality)

            redundancy = RedundancyLayer().run(quality)
            correlation = CorrelationLayer().run(redundancy)
            self.assertIs(type(correlation), CorrelationOutput)
            with self.assertRaisesRegex(TypeError, "expects CorrelationOutput"):
                KernelSignatureLayer().run(redundancy)

            signature = KernelSignatureLayer().run(correlation)
            self.assertIs(type(signature), KernelSignatureOutput)
            with self.assertRaisesRegex(TypeError, "expects KernelSignatureOutput"):
                DeltaRelationLayer().run(correlation)

            delta = DeltaRelationLayer().run(signature)
            self.assertIs(type(delta), DeltaRelationOutput)
            self.assertEqual(
                tuple(record.layer for record in delta.layer_history),
                (
                    "projection",
                    "quality",
                    "redundancy",
                    "correlation",
                    "kernel_signatures",
                    "delta_relations",
                ),
            )

    def test_quality_falls_back_when_preferred_rollup_has_low_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._build_dataset(Path(directory))
            retained_observations = tuple(
                observation
                for observation in dataset.metric_observations
                if not (
                    observation.metric_id == "cupti::dram__bytes.sum"
                    and dataset.runs[observation.run_id].condition_id
                    in {"case_3", "case_4"}
                )
            )
            payload = dataset.model_dump(mode="python")
            payload["metric_observations"] = retained_observations
            dataset = PmcDataset.model_validate(payload)
            projection = ProjectionLayer().run(
                ProjectionInput(dataset=dataset, config=AnalysisConfig())
            )
            quality = QualityLayer().run(projection)
            self.assertIn("cupti::dram__bytes.avg", quality.active_feature_ids)
            self.assertNotIn("cupti::dram__bytes.sum", quality.active_feature_ids)
            self.assertEqual(
                quality.quality_report.metrics[
                    "cupti::dram__bytes.avg"
                ].canonical_rollup_representative,
                "cupti::dram__bytes.avg",
            )

    def test_projection_rejects_cross_device_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._build_dataset(Path(directory))
            payload = dataset.model_dump(mode="python")
            payload["devices"]["cuda:second"] = DeviceSpec(
                device_id="cuda:second", vendor="nvidia"
            )
            payload["conditions"]["case_4"] = dataset.conditions[
                "case_4"
            ].model_copy(update={"device_id": "cuda:second"})
            dataset = PmcDataset.model_validate(payload)
            with self.assertRaisesRegex(ValueError, "device-scoped"):
                ProjectionLayer().run(
                    ProjectionInput(dataset=dataset, config=AnalysisConfig())
                )


if __name__ == "__main__":
    unittest.main()
