#!/usr/bin/env python3
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "enrich_pmc_workload_metadata.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pmc_workload_metadata", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PmcWorkloadMetadataTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_matmul_formula_is_exact_for_declared_semantics(self):
        description = self.module.describe_case({
            "name": "matmul",
            "backend": "cuda",
            "op_type": "matmul",
            "variant": "cuda_general_batch_matmul_fp32",
            "dtype": "float32",
            "int_params": {"batch": 2, "m": 3, "n": 5, "k": 7},
        })
        self.assertEqual(description["shapes"]["output0"], [2, 3, 5])
        self.assertEqual(description["workload"]["output_elements"], 30)
        self.assertEqual(description["workload"]["algorithmic_flops"], 420)
        self.assertEqual(
            description["workload"]["algorithmic_bytes"],
            4 * (2 * 3 * 7 + 2 * 7 * 5 + 2 * 3 * 5),
        )

    def test_comparison_or_movement_work_does_not_fake_zero_flops(self):
        maximum = self.module.describe_case({
            "name": "maximum",
            "backend": "cuda",
            "op_type": "reduction",
            "variant": "cuda_reduction_max_fp32",
            "dtype": "float32",
            "int_params": {"outside": 2, "axis": 8, "inside": 3},
        })
        self.assertNotIn("algorithmic_flops", maximum["workload"])
        self.assertEqual(
            maximum["workload_provenance"]["algorithmic_flops"]["status"],
            "not_applicable",
        )

        transpose = self.module.describe_case({
            "name": "transpose",
            "backend": "cuda",
            "op_type": "transpose",
            "variant": "cuda_transpose_fp32",
            "dtype": "float32",
            "int_params": {"m": 8, "n": 16},
        })
        self.assertNotIn("algorithmic_flops", transpose["workload"])
        self.assertEqual(transpose["workload"]["algorithmic_bytes"], 1024)

    def test_pool_shape_uses_adapter_geometry_formula(self):
        description = self.module.describe_case({
            "name": "pool",
            "backend": "cuda",
            "op_type": "avgpool",
            "variant": "cuda_avgpool_fp32",
            "dtype": "float32",
            "int_params": {
                "batch": 1,
                "channels": 8,
                "h": 8,
                "w": 8,
                "kernel_size": 3,
                "stride": 2,
                "pad": 1,
            },
        })
        self.assertEqual(description["shapes"]["output0"], [1, 4, 4, 8])
        self.assertEqual(description["workload"]["output_elements"], 128)
        self.assertGreater(description["workload"]["algorithmic_flops"], 0)

    def test_pool_120_variants_use_nchw_layout(self):
        for op_type, variant in (
            ("maxpool", "cuda_maxpool_fp32"),
            ("maxpool", "cuda_maxpool_120_fp32"),
            ("avgpool", "cuda_avgpool_fp32"),
            ("avgpool", "cuda_avgpool_120_fp32"),
        ):
            with self.subTest(op_type=op_type, variant=variant):
                description = self.module.describe_case({
                    "name": variant,
                    "backend": "cuda",
                    "op_type": op_type,
                    "variant": variant,
                    "tag": "1.2.0",
                    "dtype": "float32",
                    "int_params": {
                        "batch": 2,
                        "c": 3,
                        "channels": 3,
                        "h": 6,
                        "w": 8,
                        "kernel_size": 2,
                        "stride": 2,
                        "pad": 0,
                    },
                })
                self.assertEqual(description["shapes"]["input0"], [2, 3, 6, 8])
                expected_width = 3 if variant == "cuda_avgpool_fp32" else 4
                self.assertEqual(
                    description["shapes"]["output0"],
                    [2, 3, 3, expected_width],
                )
                self.assertIn(
                    "NCHW",
                    description["shape_provenance"]["input0"]["notes"],
                )

    def test_avgpool_geometry_matches_each_adapter_path(self):
        base = {
            "name": "avgpool",
            "backend": "cuda",
            "op_type": "avgpool",
            "tag": "1.2.0",
            "dtype": "float32",
            "int_params": {
                "batch": 1,
                "channels": 2,
                "c": 2,
                "h": 6,
                "w": 10,
                "kernel_size": 2,
                "stride": 2,
                "pad": 0,
            },
        }
        shared_adapter = self.module.describe_case(
            dict(base, variant="cuda_avgpool_fp32")
        )
        dedicated_adapter = self.module.describe_case(
            dict(base, variant="cuda_avgpool_120_fp32")
        )
        self.assertEqual(shared_adapter["shapes"]["output0"], [1, 2, 3, 3])
        self.assertEqual(dedicated_adapter["shapes"]["output0"], [1, 2, 3, 5])

    def test_softmax_flops_are_explicitly_a_lower_bound(self):
        description = self.module.describe_case({
            "name": "softmax",
            "backend": "cuda",
            "op_type": "softmax",
            "variant": "cuda_softmax_fp32",
            "dtype": "float32",
            "int_params": {"outside": 2, "axis": 8, "inside": 3},
        })
        provenance = description["workload_provenance"]["algorithmic_flops"]
        self.assertEqual(provenance["status"], "theoretical_estimate")
        self.assertIn("lower bound", provenance["notes"])
        self.assertIn("exponential", provenance["notes"])
        self.assertIn("comparison", provenance["notes"])

    def test_conv_dw_uses_tag_specific_weight_and_bias_storage(self):
        base = {
            "name": "conv_dw",
            "backend": "cuda",
            "op_type": "conv_dw",
            "variant": "cuda_conv_dw_fp32",
            "dtype": "float32",
            "int_params": {
                "iw": 4,
                "ih": 4,
                "channels": 8,
                "kw": 3,
                "sw": 1,
                "pw": 1,
            },
        }
        fp32_weight_case = dict(base, tag="1.2.0")
        fp16_weight_case = dict(base, tag="3.6.0")
        old = self.module.describe_case(fp32_weight_case)
        intermediate = self.module.describe_case(dict(base, tag="2.0.4"))
        current = self.module.describe_case(fp16_weight_case)
        self.assertEqual(old["shapes"]["input0"], [1, 8, 4, 4])
        self.assertEqual(old["shapes"]["weight0"], [8, 3, 3])
        self.assertEqual(old["shapes"]["output0"], [1, 8, 4, 4])
        self.assertEqual(intermediate["shapes"]["input0"], [1, 4, 4, 8])
        self.assertEqual(intermediate["shapes"]["weight0"], [8, 3, 3])
        self.assertEqual(current["shapes"]["input0"], [1, 4, 4, 8])
        self.assertEqual(current["shapes"]["weight0"], [3, 3, 8])
        self.assertEqual(old["workload"]["algorithmic_bytes"], 1344)
        self.assertEqual(current["workload"]["algorithmic_bytes"], 1184)
        self.assertIn(
            "FMA is counted as 2 FLOPs",
            current["workload_provenance"]["algorithmic_flops"]["notes"],
        )
        self.assertIn(
            "bias and activation/clamp work are not counted",
            current["workload_provenance"]["algorithmic_flops"]["notes"],
        )
        self.assertIn(
            "fp16",
            current["workload_provenance"]["algorithmic_bytes"]["notes"],
        )

    def test_transpose_output_shape_follows_adapter_permutation(self):
        for variant in ("cuda_transpose_fp32", "cuda_transpose_local_fp32"):
            with self.subTest(variant=variant):
                description = self.module.describe_case({
                    "name": variant,
                    "backend": "cuda",
                    "op_type": "transpose",
                    "variant": variant,
                    "tag": "3.6.0",
                    "dtype": "float32",
                    "int_params": {"m": 9, "n": 17},
                })
                self.assertEqual(description["shapes"]["input0"], [9, 17])
                self.assertEqual(description["shapes"]["output0"], [17, 9])

        bdl = self.module.describe_case({
            "name": "bdl",
            "backend": "cuda",
            "op_type": "transpose",
            "variant": "cuda_transpose_bdl_to_bld_fp32",
            "tag": "3.6.0",
            "dtype": "float32",
            "int_params": {"batch": 2, "d": 33, "l": 35},
        })
        self.assertEqual(bdl["shapes"]["input0"], [2, 33, 35])
        self.assertEqual(bdl["shapes"]["output0"], [2, 35, 33])

    def test_nhwc_nchw_transpose_shapes_follow_declared_direction(self):
        for variant, expected_input, expected_output in (
            ("cuda_nhwc2nchw_fp32", [2, 3, 4], [2, 4, 3]),
            ("cuda_nhwc2nchw_fp16", [2, 3, 4], [2, 4, 3]),
            ("cuda_nchw2nhwc_fp32", [2, 4, 3], [2, 3, 4]),
            ("cuda_nchw2nhwc_fp16", [2, 4, 3], [2, 3, 4]),
        ):
            with self.subTest(variant=variant):
                description = self.module.describe_case({
                    "name": variant,
                    "backend": "cuda",
                    "op_type": "transpose",
                    "variant": variant,
                    "tag": "3.6.0",
                    "dtype": "float16" if "fp16" in variant else "float32",
                    "int_params": {"outside": 2, "axis": 4, "inside": 3},
                })
                self.assertEqual(description["shapes"]["input0"], expected_input)
                self.assertEqual(description["shapes"]["output0"], expected_output)

    def test_format_conversion_variants_preserve_logical_extent_and_direction(self):
        c_first = [1, 5, 7]
        channel_last = [1, 7, 5]
        for variant, layouts in self.module.TRANSPOSE_FORMAT_LAYOUTS.items():
            with self.subTest(variant=variant):
                description = self.module.describe_case({
                    "name": variant,
                    "backend": "cuda",
                    "op_type": "transpose",
                    "variant": variant,
                    "tag": "3.6.0",
                    "dtype": "float32",
                    "int_params": {"c": 5, "area": 7},
                })
                input_layout, output_layout = layouts
                self.assertEqual(
                    description["shapes"]["input0"],
                    c_first if input_layout in {"nchw", "c4nhw4"} else channel_last,
                )
                self.assertEqual(
                    description["shapes"]["output0"],
                    c_first if output_layout in {"nchw", "c4nhw4"} else channel_last,
                )
                self.assertEqual(description["workload"]["output_elements"], 35)

        for variant, layouts in self.module.TRANSPOSE_PACK_LAYOUTS.items():
            with self.subTest(variant=variant):
                description = self.module.describe_case({
                    "name": variant,
                    "backend": "cuda",
                    "op_type": "transpose",
                    "variant": variant,
                    "tag": "3.6.0",
                    "dtype": "float32",
                    "int_params": {"batch": 1, "c": 5, "area": 7},
                })
                input_layout, output_layout = layouts
                self.assertEqual(
                    description["shapes"]["input0"],
                    c_first if input_layout == "nchw" else channel_last,
                )
                self.assertEqual(
                    description["shapes"]["output0"],
                    c_first if output_layout == "nchw" else channel_last,
                )

    def test_variant_matching_is_exact_for_reduction_and_transpose(self):
        self.assertIsNone(self.module.describe_case({
            "name": "unknown_reduction",
            "backend": "cuda",
            "op_type": "reduction",
            "variant": "cuda_reduction_unknown_fp32",
            "dtype": "float32",
            "int_params": {"outside": 2, "axis": 4, "inside": 1},
        }))
        self.assertIsNone(self.module.describe_case({
            "name": "unknown_transpose",
            "backend": "cuda",
            "op_type": "transpose",
            "variant": "cuda_custom_nchw_transform_fp32",
            "dtype": "float32",
            "int_params": {"c": 4, "area": 4},
        }))

    def test_mixed_half_transpose_is_not_described_as_single_dtype(self):
        for variant in self.module.MIXED_HALF_TRANSPOSE_VARIANTS:
            with self.subTest(variant=variant):
                self.assertIsNone(self.module.describe_case({
                    "name": variant,
                    "backend": "cuda",
                    "op_type": "transpose",
                    "variant": variant,
                    "dtype": "float32",
                    "int_params": {"outside": 1, "axis": 8, "inside": 4},
                }))

    def test_refresh_rejects_conflicting_user_metadata_without_mutation(self):
        document = {
            "cases": [{
                "name": "matmul",
                "backend": "cuda",
                "op_type": "matmul",
                "variant": "cuda_general_batch_matmul_fp32",
                "dtype": "float32",
                "int_params": {"batch": 2, "m": 3, "n": 5, "k": 7},
                "custom_user_field": {"keep": True},
                "shapes": {"output0": [999]},
            }],
        }
        original = json.loads(json.dumps(document))
        with self.assertRaisesRegex(ValueError, r"shapes\.output0"):
            self.module.enrich_document(document)
        self.assertEqual(document, original)

    def test_refresh_is_idempotent_for_equal_and_extra_user_metadata(self):
        manual = {
            "status": "declared",
            "source": "manifest_author",
            "method": "manual",
            "formula": "",
            "confidence": 1.0,
            "notes": "keep",
        }
        document = {
            "cases": [{
                "name": "matmul",
                "backend": "cuda",
                "op_type": "matmul",
                "variant": "cuda_general_batch_matmul_fp32",
                "dtype": "float32",
                "int_params": {"batch": 2, "m": 3, "n": 5, "k": 7},
                "custom_user_field": {"keep": True},
                "shapes": {"output0": [2, 3, 5], "user_shape": [7]},
                "shape_provenance": {"output0": manual},
                "workload": {"algorithmic_bytes": 568, "user_work": 7},
                "workload_provenance": {"algorithmic_bytes": manual},
            }],
        }
        self.module.enrich_document(document)
        first = json.loads(json.dumps(document))
        self.module.enrich_document(document)
        self.assertEqual(document, first)
        case = document["cases"][0]
        self.assertEqual(case["custom_user_field"], {"keep": True})
        self.assertEqual(case["shapes"]["output0"], [2, 3, 5])
        self.assertEqual(case["shapes"]["user_shape"], [7])
        self.assertEqual(case["workload"]["algorithmic_bytes"], 568)
        self.assertEqual(case["workload"]["user_work"], 7)
        self.assertEqual(
            case["shape_provenance"]["output0"]["source"],
            "manifest_author",
        )
        self.assertEqual(
            case["workload_provenance"]["algorithmic_bytes"]["source"],
            "manifest_author",
        )
        self.assertIn("input0", case["shapes"])
        self.assertIn("algorithmic_flops", case["workload"])

    def test_refresh_removes_only_obsolete_tool_owned_metadata(self):
        generated = {
            "status": "estimated",
            "source": self.module.PROVENANCE_SOURCE,
            "method": self.module.PROVENANCE_METHOD,
            "formula": "legacy",
            "confidence": 1.0,
            "notes": "",
        }
        manual = {
            "status": "declared",
            "source": "manifest_author",
            "method": "manual",
            "formula": "",
            "confidence": 1.0,
            "notes": "keep",
        }
        document = {
            "cases": [{
                "name": "mixed",
                "backend": "cuda",
                "op_type": "transpose",
                "variant": "cuda_packcommon_half_4_fp32",
                "dtype": "float16",
                "int_params": {"outside": 1, "axis": 8, "inside": 4},
                "shapes": {"input0": [32], "user_shape": [7]},
                "shape_provenance": {
                    "input0": generated,
                    "user_shape": manual,
                },
                "workload": {"output_elements": 32, "user_work": 7},
                "workload_provenance": {
                    "output_elements": generated,
                    "user_work": manual,
                },
            }],
        }
        enriched, counts = self.module.enrich_document(document)
        self.assertEqual(enriched, 0)
        self.assertEqual(counts, {})
        case = document["cases"][0]
        self.assertEqual(case["shapes"], {"user_shape": [7]})
        self.assertEqual(case["workload"], {"user_work": 7})
        self.assertEqual(set(case["shape_provenance"]), {"user_shape"})
        self.assertEqual(set(case["workload_provenance"]), {"user_work"})

    def test_real_manifest_has_explicit_multi_condition_p0_cohort(self):
        document = json.loads((ROOT / "operator_cases.json").read_text(encoding="utf-8"))
        enriched, counts = self.module.enrich_document(document)
        expected_counts = {
            "avgpool": 6,
            "conv_dw": 14,
            "matmul": 11,
            "maxpool": 8,
            "reduction": 45,
            "softmax": 15,
            "transpose": 23,
        }
        self.assertEqual(enriched, 122)
        self.assertEqual(counts, expected_counts)
        not_applicable_flops = 0
        for op_type in self.module.SUPPORTED_OP_TYPES:
            with self.subTest(op_type=op_type):
                descriptions = [
                    case
                    for case in document["cases"]
                    if case.get("backend") == "cuda"
                    and case.get("op_type") == op_type
                    and case.get("shapes")
                    and case.get("workload")
                ]
                fingerprints = {
                    json.dumps(
                        {
                            "dtype": case.get("dtype"),
                            "shapes": case["shapes"],
                            "workload": case["workload"],
                        },
                        sort_keys=True,
                    )
                    for case in descriptions
                }
                self.assertGreaterEqual(counts.get(op_type, 0), 5)
                self.assertGreaterEqual(len(fingerprints), 5)
                for case in descriptions:
                    workload = case["workload"]
                    self.assertGreater(workload["output_elements"], 0)
                    self.assertGreater(workload["algorithmic_bytes"], 0)
                    if "algorithmic_flops" in workload:
                        self.assertTrue(math.isfinite(workload["algorithmic_flops"]))
                        self.assertGreater(workload["algorithmic_flops"], 0)
                    for provenance in case["workload_provenance"].values():
                        if provenance["status"] != "not_applicable":
                            self.assertEqual(
                                provenance["status"], "theoretical_estimate"
                            )
                    flops_provenance = case["workload_provenance"].get(
                        "algorithmic_flops"
                    )
                    if (
                        flops_provenance
                        and flops_provenance["status"] == "not_applicable"
                    ):
                        not_applicable_flops += 1
                    self.assertNotIn(
                        case.get("variant"),
                        self.module.MIXED_HALF_TRANSPOSE_VARIANTS,
                    )
        self.assertEqual(not_applicable_flops, 44)

    def test_cli_preserves_unrelated_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cases.json"
            output = root / "enriched.json"
            document = {
                "format": "mnn-kernel-operator-cases",
                "version": 1,
                "cases": [{
                    "name": "unknown",
                    "backend": "mali",
                    "op_type": "custom",
                    "int_params": {"size": 3},
                }],
            }
            source.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                self.module.main([
                    "--input", str(source),
                    "--output", str(output),
                ]),
                0,
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                document,
            )


if __name__ == "__main__":
    unittest.main()
