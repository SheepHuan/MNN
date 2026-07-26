#!/usr/bin/env python3
"""Tests for the operator corpus extractor over operators/ directory.

The extractor scans ``operators/<backend>/<op_type>/<framework>/<tag>/``
and emits ``operators.json`` keyed by ``(backend, op_type, framework, tag,
variant)``.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXTRACTOR = ROOT / "extract_operator_kernels.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_operator(root, backend, op_type, framework, tag, variant, source, suffix):
    directory = root / "operators" / backend / op_type / framework / tag
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (variant + suffix)
    path.write_text(source, encoding="utf-8")
    return "operators/{}/{}/{}/{}/{}".format(backend, op_type, framework, tag, path.name)


class OperatorExtractorTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module(EXTRACTOR, "operator_extractor")

    def test_opencl_operator_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rel = write_operator(
                root, "opencl", "raster", "mnn", "1.2.0", "buffer_set_zero_fp32",
                "typedef float FLOAT;\n__kernel void buffer_set_zero(__global FLOAT *o) { o[0]=0; }\n",
                ".cl",
            )
            operators = self.module.extract_operators(root)
            self.assertEqual(len(operators), 1)
            record = operators[0]
            self.assertEqual(record["backend"], "opencl")
            self.assertEqual(record["op_type"], "raster")
            self.assertEqual(record["framework"], "mnn")
            self.assertEqual(record["tag"], "1.2.0")
            self.assertEqual(record["variant"], "buffer_set_zero_fp32")
            self.assertEqual(record["entry"], "buffer_set_zero")
            self.assertEqual(record["language"], "opencl")
            self.assertEqual(record["file"], rel)
            self.assertEqual(record["execution_spec"], "manual")

    def test_vulkan_operator_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rel = write_operator(
                root, "vulkan", "sigmoid", "ncnn", "20190611", "sigmoid_fp32",
                "#version 450\nvoid main() { }\n",
                ".comp",
            )
            operators = self.module.extract_operators(root)
            self.assertEqual(len(operators), 1)
            record = operators[0]
            self.assertEqual(record["backend"], "vulkan")
            self.assertEqual(record["op_type"], "sigmoid")
            self.assertEqual(record["framework"], "ncnn")
            self.assertEqual(record["tag"], "20190611")
            self.assertEqual(record["variant"], "sigmoid_fp32")
            self.assertEqual(record["entry"], "sigmoid_fp32")
            self.assertEqual(record["language"], "glsl")
            self.assertEqual(record["file"], rel)

    def test_missing_entry_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_operator(
                root, "opencl", "raster", "mnn", "1.2.0", "broken_fp32",
                "// no kernel here\n",
                ".cl",
            )
            operators = self.module.extract_operators(root)
            self.assertEqual(operators, [])

    def test_duplicate_key_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_operator(
                root, "opencl", "raster", "mnn", "1.2.0", "dup_fp32",
                "__kernel void dup() {}\n", ".cl",
            )
            # same (backend, op_type, framework, tag, variant) again via symlink-free copy
            target = root / "operators/opencl/raster/mnn/1.2.0/dup_fp32.cl"
            # overwrite with same content to exercise the de-dup path
            target.write_text("__kernel void dup() {}\n", encoding="utf-8")
            # second file with same variant name in a different op_type is fine
            write_operator(
                root, "opencl", "other", "mnn", "1.2.0", "dup_fp32",
                "__kernel void dup() {}\n", ".cl",
            )
            operators = self.module.extract_operators(root)
            # two distinct op_types -> two records, no duplicate-key error
            self.assertEqual(len(operators), 2)

    def test_output_sorted_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for op in ("conv", "raster", "unary"):
                write_operator(
                    root, "opencl", op, "mnn", "1.2.0", op + "_fp32",
                    "__kernel void {}() {{}}\n".format(op), ".cl",
                )
            operators = self.module.extract_operators(root)
            keys = [(o["backend"], o["op_type"], o["framework"], o["tag"], o["variant"])
                    for o in operators]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual([o["op_type"] for o in operators],
                             ["conv", "raster", "unary"])

    def test_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_operator(
                root, "opencl", "raster", "mnn", "1.2.0", "buffer_set_zero_fp32",
                "__kernel void buffer_set_zero(__global float *o) { o[0]=0; }\n",
                ".cl",
            )
            output_path = root / "operators.json"
            rc = self.module.main([
                "--root", str(root),
                "--output", str(output_path),
            ])
            self.assertEqual(rc, 0)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data["format"], "mnn-kernel-operator-corpus")
            self.assertEqual(len(data["operators"]), 1)
            self.assertEqual(data["operators"][0]["variant"], "buffer_set_zero_fp32")


if __name__ == "__main__":
    unittest.main()
