#!/usr/bin/env python3
"""Validate the GPU kernel source corpus manifest and file hashes."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


FORBIDDEN_PARTS = ("schema/private", "source/internal")
FORBIDDEN_SUFFIXES = (".so", ".a", ".mnn", ".bin")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest_path, corpus_root):
    manifest_path = Path(manifest_path)
    corpus_root = Path(corpus_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    expected = set()
    versions = manifest.get("versions")
    if not isinstance(versions, list) or not versions:
        return ["manifest.versions must be a non-empty list"]
    for record in versions:
        for key in ("framework", "tag", "commit", "source_url", "backends", "files", "licenses"):
            if key not in record:
                errors.append("missing {} in version record".format(key))
        if not isinstance(record.get("backends"), dict) or not record.get("backends"):
            errors.append("backend status is missing for {}/{}".format(record.get("framework"), record.get("tag")))
        else:
            for backend, status in record["backends"].items():
                if status not in ("archived", "unavailable"):
                    errors.append("invalid backend status {}={}".format(backend, status))
        record_files = record.get("files", [])
        file_paths = set()
        for item in record_files:
            relative = item.get("path", "")
            file_paths.add(relative)
            expected.add(relative)
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                errors.append("unsafe path {}".format(relative))
                continue
            if any(part in "/".join(path.parts) for part in FORBIDDEN_PARTS):
                errors.append("forbidden path {}".format(relative))
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append("binary/artifact path {}".format(relative))
            disk_path = corpus_root / relative
            if not disk_path.is_file():
                errors.append("missing file {}".format(relative))
            elif item.get("sha256") != sha256_file(disk_path):
                errors.append("sha256 mismatch {}".format(relative))
        if len(file_paths) != len(record_files):
            errors.append("duplicate file path in {}/{}".format(record.get("framework"), record.get("tag")))
        for license_path in record.get("licenses", []):
            if license_path not in file_paths:
                errors.append("license is not listed in files {}".format(license_path))
            if not (corpus_root / license_path).is_file():
                errors.append("missing license {}".format(license_path))
    source_root = corpus_root / "sources"
    if source_root.exists():
        for path in source_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(corpus_root).as_posix()
            if relative not in expected:
                errors.append("unmanifested file {}".format(relative))
            if any(part in relative for part in FORBIDDEN_PARTS):
                errors.append("forbidden file {}".format(relative))
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append("binary/artifact file {}".format(relative))
    return errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    errors = validate_manifest(args.manifest, args.root)
    if errors:
        for error in errors:
            print("ERROR: " + error)
        return 1
    print("valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
