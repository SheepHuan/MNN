"""从标准化 bundle 或 MNN 三数据源生成 PMC 结构化报告和 Markdown。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import (
    AnalysisConfig,
    DeviceSpec,
    KernelTaxonomy,
    MetricRegistry,
    load_normalized_bundle,
    write_normalized_bundle,
)
from .dataset.sources import (
    append_comparison_pairs_csv,
    append_control_delta_csv,
    load_mnn_kernelreplay_dataset,
)
from .pipeline import PmcInterpreter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-bundle", help="normalized mnn-pmc-dataset/v1 JSON")
    source.add_argument("--pmc-csv", help="MNN kernel-replay long rows CSV")
    parser.add_argument("--latency-json")
    parser.add_argument("--operator-cases")
    parser.add_argument("--pairs-csv")
    parser.add_argument("--platform", default="cuda")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--namespace", default="cupti")
    parser.add_argument("--collector", default="mnn-kernel-replay")
    parser.add_argument("--dataset-id", default="mnn-kernel-replay")
    parser.add_argument("--device-id")
    parser.add_argument("--device-vendor", default="unknown")
    parser.add_argument("--device-architecture", default="unknown")
    parser.add_argument("--device-model", default="unknown")
    parser.add_argument("--device-driver", default="unknown")
    parser.add_argument("--metric-registry-json")
    parser.add_argument("--taxonomy-json")
    parser.add_argument("--control-delta-csv")
    parser.add_argument("--control-delta-namespace", default="generic")
    parser.add_argument("--normalized-output-json")
    parser.add_argument("--pmc-workload-runs", type=int, default=1)
    parser.add_argument("--latency-workload-runs", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--min-unique-values", type=int, default=3)
    parser.add_argument("--minimum-group-samples", type=int, default=5)
    parser.add_argument("--minimum-delta-pairs", type=int, default=6)
    parser.add_argument("--global-top-k", type=int, default=15)
    parser.add_argument("--group-top-k", type=int, default=10)
    parser.add_argument("--delta-top-k", type=int, default=30)
    parser.add_argument("--fdr-threshold", type=float, default=0.1)
    parser.add_argument("--strong-rho", type=float, default=0.5)
    parser.add_argument("--moderate-rho", type=float, default=0.3)
    parser.add_argument("--nonlinear-distance-correlation", type=float, default=0.5)
    parser.add_argument("--redundancy-weight", type=float, default=0.35)
    parser.add_argument("--max-repeat-relative-mad", type=float, default=0.1)
    return parser.parse_args(argv)


def _load_dataset(args):
    registry = (
        MetricRegistry.from_json(args.metric_registry_json)
        if args.metric_registry_json
        else MetricRegistry()
    )
    if args.input_bundle:
        dataset = load_normalized_bundle(args.input_bundle)
        if args.pairs_csv:
            dataset = append_comparison_pairs_csv(dataset, args.pairs_csv)
    else:
        if not args.latency_json or not args.operator_cases:
            raise ValueError("--pmc-csv requires --latency-json and --operator-cases")
        taxonomy = (
            KernelTaxonomy.load(args.taxonomy_json)
            if args.taxonomy_json
            else KernelTaxonomy.load()
        )
        device = DeviceSpec(
            device_id=args.device_id or "{}:default".format(args.platform),
            vendor=args.device_vendor,
            architecture=args.device_architecture,
            model=args.device_model,
            driver=args.device_driver,
        )
        dataset = load_mnn_kernelreplay_dataset(
            args.pmc_csv,
            args.latency_json,
            args.operator_cases,
            platform=args.platform,
            backend=args.backend,
            namespace=args.namespace,
            collector=args.collector,
            dataset_id=args.dataset_id,
            device=device,
            taxonomy=taxonomy,
            registry=registry,
            pairs_csv=args.pairs_csv,
            pmc_workload_runs=args.pmc_workload_runs,
            latency_workload_runs=args.latency_workload_runs,
        )
    if args.control_delta_csv:
        dataset = append_control_delta_csv(
            dataset,
            args.control_delta_csv,
            namespace=args.control_delta_namespace,
            registry=registry,
        )
    return dataset


def _analysis_config(args):
    return AnalysisConfig(
        min_coverage=args.min_coverage,
        min_unique_values=args.min_unique_values,
        min_group_samples=args.minimum_group_samples,
        min_delta_pairs=args.minimum_delta_pairs,
        global_top_k=args.global_top_k,
        group_top_k=args.group_top_k,
        delta_top_k=args.delta_top_k,
        fdr_threshold=args.fdr_threshold,
        strong_rho=args.strong_rho,
        moderate_rho=args.moderate_rho,
        nonlinear_distance_correlation=args.nonlinear_distance_correlation,
        redundancy_weight=args.redundancy_weight,
        max_repeat_relative_mad=args.max_repeat_relative_mad,
    )


def main(argv=None):
    args = parse_args(argv)
    dataset = _load_dataset(args)
    if args.normalized_output_json:
        write_normalized_bundle(dataset, args.normalized_output_json)

    interpreter = PmcInterpreter(config=_analysis_config(args))
    report, document = interpreter.analyze_with_document(dataset)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        markdown_output = Path(args.output_md)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(document.markdown, encoding="utf-8")
    print(
        "analyzed {} conditions, {} metrics, {} delta rules".format(
            len(dataset.conditions),
            len(dataset.metric_catalog),
            len(report.delta_rulebook),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
