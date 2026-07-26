#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config.json"
COLLECTOR = ROOT / "collect_kernel_sources.py"
VALIDATOR = ROOT / "validate_kernel_sources.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelCorpusTest(unittest.TestCase):
    def test_config_declares_tag_matrix_and_backend_status(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(len(config["frameworks"]["mnn"]["tags"]), 73)
        self.assertEqual(config["frameworks"]["mnn"]["tags"][0], "1.2.0")
        self.assertEqual(config["frameworks"]["mnn"]["tags"][-1], "3.6.0")
        self.assertEqual(len(config["frameworks"]["ncnn"]["tags"]), 37)
        self.assertEqual(config["frameworks"]["ncnn"]["tags"][0], "20190611")
        self.assertEqual(config["frameworks"]["ncnn"]["tags"][-1], "20260526")
        self.assertEqual(config["frameworks"]["mnn"]["backends"]["opencl"], "archived")
        self.assertEqual(config["frameworks"]["mnn"]["backends"]["vulkan"], "archived")
        self.assertEqual(config["frameworks"]["ncnn"]["backends"]["opencl"], "unavailable")
        self.assertEqual(config["frameworks"]["ncnn"]["selectors"][1:3], ["src/command.cpp", "src/command.h"])
        serialized = json.dumps(config)
        self.assertNotIn("schema/private", serialized)
        self.assertNotIn("source/internal", serialized)

    def test_copy_snapshot_filters_and_hashes_source_files(self):
        collector = load_module(COLLECTOR, "kernel_corpus_collector")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            (source / "source/backend/vulkan").mkdir(parents=True)
            (source / "schema/private").mkdir(parents=True)
            (source / "source/backend/vulkan/kernel.comp").write_text("shader", encoding="utf-8")
            (source / "LICENSE").write_text("license", encoding="utf-8")
            (source / "schema/private/secret.h").write_text("secret", encoding="utf-8")
            files = collector.copy_snapshot(
                source, destination, ["source/backend/vulkan"], ["LICENSE"], "mnn/2.8.4"
            )
            self.assertEqual([item["path"] for item in files], ["LICENSE", "source/backend/vulkan/kernel.comp"])
            self.assertEqual(len(files[0]["sha256"]), 64)
            self.assertFalse((destination / "schema/private/secret.h").exists())

    def test_manifest_validation_rejects_changed_hash(self):
        validator = load_module(VALIDATOR, "kernel_corpus_validator")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sources/mnn/2.8.4").mkdir(parents=True)
            file_path = root / "sources/mnn/2.8.4/LICENSE"
            file_path.write_text("license", encoding="utf-8")
            manifest = {
                "versions": [{
                    "framework": "mnn", "tag": "2.8.4", "commit": "a" * 40,
                    "backends": {"opencl": "archived", "vulkan": "archived"},
                    "files": [{"path": "sources/mnn/2.8.4/LICENSE", "sha256": "0" * 64}],
                    "licenses": ["sources/mnn/2.8.4/LICENSE"],
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            errors = validator.validate_manifest(manifest_path, root)
            self.assertTrue(any("sha256" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
