#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from pathlib import Path

from kernel_agent.pmc_interpreter import AnalysisConfig
from kernel_agent.pmc_interpreter.dataset import (
    ComparisonPair,
    DatasetManifest,
    DeviceSpec,
    KernelCondition,
    LatencyObservation,
    MeasurementRun,
    MetricObservation,
    PmcDataset,
    ProjectionInput,
)
from kernel_agent.pmc_interpreter.dataset.catalog import MetricRegistry
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


METRIC_ID = "cupti::dram__bytes.sum"


class DeltaRelationTest(unittest.TestCase):
    def _run_delta(self, dataset, min_delta_pairs=6):
        projection = ProjectionLayer().run(ProjectionInput(
            dataset=dataset,
            config=AnalysisConfig(
                min_delta_pairs=min_delta_pairs,
                min_group_samples=3,
            ),
        ))
        quality = QualityLayer().run(projection)
        redundancy = RedundancyLayer().run(quality)
        correlation = CorrelationLayer().run(redundancy)
        signatures = KernelSignatureLayer().run(correlation)
        return DeltaRelationLayer().run(signatures)

    def _build_repeated_pair_dataset(
        self,
        *,
        shared_paired_groups,
        family_directions=(("elementwise_quant", 1),),
        controlled_intervention=True,
        shared_repeat_id_collision=False,
        extra_unpaired_repeats=False,
    ):
        device_id = "cuda:test"
        descriptor = MetricRegistry().describe(
            "dram__bytes.sum", "cupti", METRIC_ID
        )
        conditions = {}
        runs = {}
        metric_observations = []
        latency_observations = []
        comparisons = []
        op_types = {
            "elementwise_quant": "relu",
            "dense_quant": "matmul",
        }

        def append_measurement(
            *,
            condition_id,
            sequence_id,
            repeat_id,
            paired_group_id,
            metric_value,
            latency_value,
            family_index,
        ):
            pmc_run_id = "pmc::{}::{}".format(condition_id, sequence_id)
            latency_run_id = "latency::{}::{}".format(
                condition_id, sequence_id
            )
            common_run_fields = {
                "condition_id": condition_id,
                "collector": "synthetic-controlled-test",
                "repeat_id": repeat_id,
                "collection_id": "collection::{}::{}".format(
                    condition_id, sequence_id
                ),
                "paired_run_group_id": paired_group_id,
                "warmup_runs": 1,
                "workload_runs": 1,
                "order_index": sequence_id,
                "gpu_clock_hz": 1_500_000_000.0,
                "temperature_c": 50.0 + family_index / 10.0,
                "cache_policy": "warm",
                "source_ref": "synthetic",
            }
            runs[pmc_run_id] = MeasurementRun(
                run_id=pmc_run_id,
                collection_kind="pmc",
                **common_run_fields,
            )
            runs[latency_run_id] = MeasurementRun(
                run_id=latency_run_id,
                collection_kind="latency",
                **common_run_fields,
            )
            metric_observations.append(MetricObservation(
                observation_id="metric::{}::{}".format(
                    condition_id, sequence_id
                ),
                run_id=pmc_run_id,
                metric_id=METRIC_ID,
                value=metric_value,
                value_semantics="absolute_workload",
                control_value=None,
                workload_value=metric_value,
                status="VALID",
                profiler_pass_count=1,
                quality_flags=(),
                raw_status="sampled",
                error="",
            ))
            latency_observations.append(LatencyObservation(
                observation_id="latency::{}::{}".format(
                    condition_id, sequence_id
                ),
                run_id=latency_run_id,
                latency_us=latency_value,
                status="VALID",
                quality_flags=(),
            ))

        for family_index, (family, direction) in enumerate(
            family_directions, 1
        ):
            for pair_index in range(1, 7):
                pair_id = "{}-pair-{:02d}".format(family, pair_index)
                semantic_key = "{}-workload-{:02d}".format(
                    family, pair_index
                )
                baseline_id = "{}-baseline-{:02d}".format(
                    family, pair_index
                )
                candidate_id = "{}-candidate-{:02d}".format(
                    family, pair_index
                )
                for condition_id, implementation_id in (
                    (baseline_id, "baseline"),
                    (candidate_id, "candidate"),
                ):
                    conditions[condition_id] = KernelCondition(
                        condition_id=condition_id,
                        device_id=device_id,
                        case_id=condition_id,
                        semantic_family=family,
                        op_type=op_types[family],
                        execution_role="core_compute",
                        expected_mechanisms=("dram_memory",),
                        performance_regime="unknown",
                        implementation_id="{}:{}".format(
                            implementation_id, pair_index
                        ),
                        semantic_equivalence_key=semantic_key,
                        validator="identity_fp32",
                        dtype="float32",
                        params={"size": pair_index},
                    )

                metric_delta = float(pair_index * 10)
                baseline_metric = 100.0
                candidate_metric = baseline_metric + metric_delta
                baseline_latency = 10.0
                latency_ratio = 1.0 + pair_index / 10.0
                candidate_latency = (
                    baseline_latency * latency_ratio
                    if direction > 0
                    else baseline_latency / latency_ratio
                )

                for condition_id, role, metric_value, latency_value in (
                    (
                        baseline_id,
                        "baseline",
                        baseline_metric,
                        baseline_latency,
                    ),
                    (
                        candidate_id,
                        "candidate",
                        candidate_metric,
                        candidate_latency,
                    ),
                ):
                    for repeat_index in (1, 2):
                        if shared_paired_groups:
                            paired_group_id = "{}-evidence-{}".format(
                                pair_id, repeat_index
                            )
                        else:
                            paired_group_id = "{}-{}-evidence-{}".format(
                                pair_id, role, repeat_index
                            )
                        repeat_id = (
                            "repeat-1"
                            if shared_repeat_id_collision
                            else "repeat-{}".format(repeat_index)
                        )
                        append_measurement(
                            condition_id=condition_id,
                            sequence_id=repeat_index,
                            repeat_id=repeat_id,
                            paired_group_id=paired_group_id,
                            metric_value=metric_value,
                            latency_value=latency_value,
                            family_index=family_index,
                        )
                    if extra_unpaired_repeats:
                        append_measurement(
                            condition_id=condition_id,
                            sequence_id=3,
                            repeat_id="repeat-2",
                            paired_group_id="{}-{}-unpaired".format(
                                pair_id, role
                            ),
                            metric_value=metric_value,
                            latency_value=latency_value,
                            family_index=family_index,
                        )

                comparisons.append(ComparisonPair(
                    pair_id=pair_id,
                    baseline_condition_id=baseline_id,
                    candidate_condition_id=candidate_id,
                    intervention_id=(
                        "intervention::{}".format(pair_id)
                        if controlled_intervention
                        else ""
                    ),
                    controlled_mechanism=(
                        "dram_memory" if controlled_intervention else ""
                    ),
                    design=(
                        "controlled_intervention"
                        if controlled_intervention
                        else "paired_observational"
                    ),
                    matched_fields=(
                        "op_type",
                        "dtype",
                        "validator",
                        "shape",
                        "params",
                    ),
                    environment_matched=(
                        True if controlled_intervention else None
                    ),
                    randomized_order=(
                        True if controlled_intervention else None
                    ),
                    notes="synthetic delta regression",
                ))

        return PmcDataset(
            manifest=DatasetManifest(
                dataset_id="delta-causal-regression",
                platform="cuda",
                backend="cuda",
                collector="synthetic-controlled-test",
            ),
            devices={
                device_id: DeviceSpec(
                    device_id=device_id,
                    vendor="nvidia",
                    architecture="test",
                    model="test",
                    driver="test",
                )
            },
            conditions=conditions,
            runs=runs,
            metric_catalog={METRIC_ID: descriptor},
            metric_observations=tuple(metric_observations),
            latency_observations=tuple(latency_observations),
            comparisons=tuple(comparisons),
        )

    @staticmethod
    def _condition_paired_groups(dataset, condition_id):
        return {
            run.paired_run_group_id
            for run in dataset.runs.values()
            if run.condition_id == condition_id
            and run.paired_run_group_id
        }

    def test_causal_claim_rejects_disjoint_baseline_candidate_pair_groups(self):
        dataset = self._build_repeated_pair_dataset(
            shared_paired_groups=False
        )
        first_pair = dataset.comparisons[0]
        baseline_groups = self._condition_paired_groups(
            dataset, first_pair.baseline_condition_id
        )
        candidate_groups = self._condition_paired_groups(
            dataset, first_pair.candidate_condition_id
        )
        self.assertEqual(len(baseline_groups), 2)
        self.assertEqual(len(candidate_groups), 2)
        self.assertFalse(baseline_groups & candidate_groups)

        report = self._run_delta(dataset).delta_report
        self.assertEqual(len(report.global_effect_rules), 1)
        rule = report.global_effect_rules[0]
        self.assertEqual(rule.feature_id, METRIC_ID)
        self.assertEqual(rule.evidence_level, "paired_observational")
        self.assertFalse(rule.causal_claim_allowed)

    def test_causal_claim_requires_two_shared_paired_evidence_groups(self):
        dataset = self._build_repeated_pair_dataset(
            shared_paired_groups=True
        )
        for pair in dataset.comparisons:
            baseline_groups = self._condition_paired_groups(
                dataset, pair.baseline_condition_id
            )
            candidate_groups = self._condition_paired_groups(
                dataset, pair.candidate_condition_id
            )
            self.assertEqual(len(baseline_groups & candidate_groups), 2)

        report = self._run_delta(dataset).delta_report
        self.assertEqual(len(report.global_effect_rules), 1)
        rule = report.global_effect_rules[0]
        self.assertEqual(rule.feature_id, METRIC_ID)
        self.assertEqual(rule.evidence_level, "controlled_intervention")
        self.assertTrue(rule.causal_claim_allowed)
        self.assertEqual(rule.scope_selector.device_ids, ("cuda:test",))
        self.assertEqual(
            rule.scope_selector.semantic_families,
            ("elementwise_quant",),
        )
        self.assertEqual(len(rule.scope_selector.conditions), 12)
        self.assertEqual(
            rule.scope_selector.controlled_mechanisms,
            ("dram_memory",),
        )
        self.assertEqual(len(rule.scope_selector.intervention_ids), 6)
        self.assertEqual(
            {
                item.op_type
                for item in rule.scope_selector.conditions
            },
            {"relu"},
        )
        self.assertEqual(
            rule.observed_support.raw_baseline_feature.minimum,
            100.0,
        )
        self.assertEqual(
            rule.observed_support.raw_baseline_feature.maximum,
            100.0,
        )
        self.assertEqual(
            rule.observed_support.raw_feature_delta.minimum,
            10.0,
        )
        self.assertEqual(
            rule.observed_support.raw_feature_delta.maximum,
            60.0,
        )
        self.assertAlmostEqual(
            rule.observed_support.latency_ratio.minimum,
            1.1,
        )
        self.assertAlmostEqual(
            rule.observed_support.latency_ratio.maximum,
            1.6,
        )

    def test_causal_claim_rejects_missing_controlled_mechanism(self):
        dataset = self._build_repeated_pair_dataset(
            shared_paired_groups=True
        )
        comparisons = list(dataset.comparisons)
        comparisons[0] = comparisons[0].model_copy(
            update={"controlled_mechanism": ""}
        )
        payload = dataset.model_dump(mode="python")
        payload["comparisons"] = tuple(comparisons)
        dataset = PmcDataset.model_validate(payload)

        rule = self._run_delta(dataset).delta_report.global_effect_rules[0]
        self.assertFalse(rule.causal_claim_allowed)
        self.assertEqual(rule.evidence_level, "paired_observational")

    def test_causal_claim_ignores_unpaired_repeats(self):
        dataset = self._build_repeated_pair_dataset(
            shared_paired_groups=True,
            shared_repeat_id_collision=True,
            extra_unpaired_repeats=True,
        )

        rule = self._run_delta(dataset).delta_report.global_effect_rules[0]
        self.assertFalse(rule.causal_claim_allowed)
        self.assertEqual(rule.evidence_level, "paired_observational")

    def test_family_rules_keep_report_available_when_global_directions_cancel(self):
        dataset = self._build_repeated_pair_dataset(
            shared_paired_groups=True,
            family_directions=(
                ("elementwise_quant", 1),
                ("dense_quant", -1),
            ),
            controlled_intervention=False,
        )
        report = self._run_delta(dataset).delta_report

        self.assertEqual(report.global_effect_rules, ())
        self.assertTrue(
            report.semantic_family_effect_rules["elementwise_quant"]
        )
        self.assertTrue(report.semantic_family_effect_rules["dense_quant"])
        self.assertTrue(report.effect_rules)
        self.assertEqual(report.status, "paired_association_available")

    def test_family_rules_keep_one_representative_per_distinct_collinear_group(self):
        dataset = self._build_repeated_pair_dataset(
            shared_paired_groups=True,
            family_directions=(
                ("elementwise_quant", 1),
                ("dense_quant", 1),
            ),
            controlled_intervention=False,
        )
        extra_metrics = (
            (
                "cupti::sm__inst_executed.sum",
                "sm__inst_executed.sum",
                "linear",
            ),
            (
                "cupti::l1tex__t_sectors.sum",
                "l1tex__t_sectors.sum",
                "permuted",
            ),
            (
                "cupti::l1tex__data_bank_conflicts.sum",
                "l1tex__data_bank_conflicts.sum",
                "permuted",
            ),
        )
        permuted_delta = {
            1: 10.0,
            2: 30.0,
            3: 20.0,
            4: 50.0,
            5: 40.0,
            6: 60.0,
        }
        metric_catalog = dict(dataset.metric_catalog)
        for metric_id, native_name, _ in extra_metrics:
            metric_catalog[metric_id] = MetricRegistry().describe(
                native_name, "cupti", metric_id
            )

        metric_observations = list(dataset.metric_observations)
        for run in dataset.runs.values():
            if run.collection_kind != "pmc":
                continue
            pair_index = int(run.condition_id.rsplit("-", 1)[1])
            is_candidate = "-candidate-" in run.condition_id
            for metric_id, _, delta_profile in extra_metrics:
                delta = (
                    float(pair_index * 10)
                    if delta_profile == "linear"
                    else permuted_delta[pair_index]
                )
                value = 200.0 + (delta if is_candidate else 0.0)
                metric_observations.append(MetricObservation(
                    observation_id="{}::{}::{}".format(
                        metric_id,
                        run.condition_id,
                        run.repeat_id,
                    ),
                    run_id=run.run_id,
                    metric_id=metric_id,
                    value=value,
                    value_semantics="absolute_workload",
                    control_value=None,
                    workload_value=value,
                    status="VALID",
                    profiler_pass_count=1,
                    quality_flags=(),
                    raw_status="sampled",
                    error="",
                ))

        payload = dataset.model_dump(mode="python")
        payload["metric_catalog"] = metric_catalog
        payload["metric_observations"] = tuple(metric_observations)
        report = self._run_delta(
            PmcDataset.model_validate(payload)
        ).delta_report

        self.assertEqual(len(report.global_effect_rules), 2)
        for family in ("elementwise_quant", "dense_quant"):
            rules = report.semantic_family_effect_rules[family]
            self.assertEqual(len(rules), 2)
            self.assertEqual(
                len({rule.statistics.delta_collinear_group for rule in rules}),
                2,
            )

    def test_zero_delta_is_retained_and_collinear_metrics_are_not_both_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            latencies = {}
            pair_rows = []
            rows = root / "rows.csv"
            with rows.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "case", "metric", "status", "value", "pmu_status",
                    "num_passes", "returncode", "error",
                ])
                writer.writeheader()
                for index in range(1, 9):
                    baseline = "base_{:02d}".format(index)
                    candidate = "candidate_{:02d}".format(index)
                    params = {"size": 100 + index}
                    for case, variant in (
                        (baseline, "baseline"),
                        (candidate, "candidate"),
                    ):
                        cases.append({
                            "name": case,
                            "backend": "cuda",
                            "op_type": "relu",
                            "variant": variant,
                            "tag": "test",
                            "dtype": "float32",
                            "validator": "identity_fp32",
                            "int_params": params,
                        })
                    latencies[baseline] = 10.0
                    latencies[candidate] = 10.0 * (1.0 + index / 10.0)
                    pair_rows.append({
                        "pair_id": "pair_{:02d}".format(index),
                        "baseline_case": baseline,
                        "variant_case": candidate,
                        "design": "paired_observational",
                        "controlled_mechanism": "",
                        "notes": "",
                    })
                    delta = max(0, index - 1) * 10
                    values = {
                        baseline: {
                            "l1tex__data_bank_conflicts.sum": 100,
                            "sm__inst_executed.sum": 300,
                        },
                        candidate: {
                            "l1tex__data_bank_conflicts.sum": 100 + delta,
                            "sm__inst_executed.sum": 300 - delta,
                        },
                    }
                    for case, metrics in values.items():
                        for metric, value in metrics.items():
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
            operator_cases = root / "operator_cases.json"
            operator_cases.write_text(
                json.dumps({"cases": cases}), encoding="utf-8"
            )
            latency = root / "latency.json"
            latency.write_text(json.dumps(latencies), encoding="utf-8")
            pairs = root / "pairs.csv"
            with pairs.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "pair_id", "baseline_case", "variant_case", "design",
                    "controlled_mechanism", "notes",
                ])
                writer.writeheader()
                writer.writerows(pair_rows)

            config = AnalysisConfig(min_delta_pairs=6)
            dataset = load_mnn_kernelreplay_dataset(
                rows, latency, operator_cases, pairs_csv=pairs
            )
            projection = ProjectionLayer().run(
                ProjectionInput(dataset=dataset, config=config)
            )
            quality = QualityLayer().run(projection)
            redundancy = RedundancyLayer().run(quality)
            correlation = CorrelationLayer().run(redundancy)
            signatures = KernelSignatureLayer().run(correlation)
            delta_output = DeltaRelationLayer().run(signatures)

            report = delta_output.delta_report
            self.assertEqual(report.status, "paired_association_available")
            candidates = {
                item.feature_id: item for item in report.candidate_relations
            }
            conflicts = candidates[
                "cupti::l1tex__data_bank_conflicts.sum"
            ]
            instructions = candidates["cupti::sm__inst_executed.sum"]
            self.assertEqual(
                conflicts.statistics.relation_class,
                "increase_associated_with_higher_latency",
            )
            self.assertEqual(
                instructions.statistics.relation_class,
                "increase_associated_with_lower_latency",
            )
            self.assertEqual(conflicts.statistics.zero_delta_pair_count, 1)
            self.assertFalse(conflicts.causal_claim_allowed)
            self.assertEqual(len(report.delta_collinear_groups), 1)
            self.assertEqual(len(report.global_effect_rules), 1)
            self.assertEqual(len(report.effect_rules), 1)
            self.assertEqual(report.independent_pair_group_count, 8)
            self.assertEqual(
                conflicts.statistics.independent_pair_group_count, 8
            )
            self.assertEqual(len(conflicts.statistics.contributing_pair_ids), 8)
            self.assertEqual(
                tuple(record.layer for record in delta_output.layer_history),
                (
                    "projection",
                    "quality",
                    "redundancy",
                    "correlation",
                    "kernel_signatures",
                    "delta_relations",
                ),
            )


if __name__ == "__main__":
    unittest.main()
