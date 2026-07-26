#!/usr/bin/env python3
"""Collect selected GPU source files from tagged framework repositories."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path, PurePosixPath


SOURCE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inc", ".inl",
    ".cl", ".comp", ".frag", ".geom", ".glsl", ".tesc", ".tese", ".vert",
}
FORBIDDEN_PARTS = {"schema/private", "source/internal"}
EXCLUDED_PARTS = {".git", "build", "builds", "cache", "caches", "models", "records"}


def run_git(repository, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("git failed: {}".format(result.stderr.strip()))
    return result.stdout


def is_forbidden(path):
    normalized = PurePosixPath(path.as_posix())
    parts = normalized.parts
    return "schema/private" in "/".join(parts) or "source/internal" in "/".join(parts)


def is_source_file(path):
    if not path.is_file() or is_forbidden(path):
        return False
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    return path.suffix.lower() in SOURCE_EXTENSIONS


def select_source_files(root, selectors):
    selected = set()
    missing = []
    for selector in selectors:
        candidate = root / selector
        if not candidate.exists():
            missing.append(selector)
            continue
        if candidate.is_file():
            if is_source_file(candidate):
                selected.add(candidate.relative_to(root).as_posix())
            continue
        for path in candidate.rglob("*"):
            if is_source_file(path):
                selected.add(path.relative_to(root).as_posix())
    return sorted(selected), missing


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_snapshot(source_root, destination_root, selectors, licenses, version_key=""):
    source_files, missing = select_source_files(source_root, selectors)
    del missing, version_key
    license_files = []
    for name in licenses:
        path = source_root / name
        if path.is_file() and not is_forbidden(path):
            license_files.append(path.relative_to(source_root).as_posix())
    all_files = sorted(set(source_files + license_files))
    records = []
    for relative in all_files:
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        records.append({"path": relative, "sha256": sha256_file(destination)})
    return records


def build_manifest_record(framework, tag, commit, source_url, backends, files, licenses,
                          missing_selectors=None):
    return {
        "framework": framework,
        "tag": tag,
        "commit": commit,
        "source_url": source_url,
        "backends": dict(sorted(backends.items())),
        "files": sorted(files, key=lambda item: item["path"]),
        "licenses": sorted(licenses),
        "missing_selectors": sorted(missing_selectors or []),
    }


def ensure_repository(repository_url, cache_root, framework):
    cache_root.mkdir(parents=True, exist_ok=True)
    repository = cache_root / (framework + ".git")
    if not repository.exists():
        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repository_url, str(repository)],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["git", "clone", "--no-checkout", repository_url, str(repository)],
                check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        if result.returncode != 0:
            raise RuntimeError("clone failed for {}: {}".format(framework, result.stderr.strip()))
    run_git(repository, "fetch", "--tags", "--force", "origin")
    return repository


def materialize_tag(repository, tag, selectors, licenses, temporary_root):
    commit = run_git(repository, "rev-parse", tag + "^{commit}").strip()
    names = run_git(repository, "ls-tree", "-r", "--name-only", commit).splitlines()
    selected_names = []
    for name in names:
        path = PurePosixPath(name)
        if is_forbidden(Path(name)):
            continue
        if any(name == selector or name.startswith(selector.rstrip("/") + "/") for selector in selectors):
            if Path(name).suffix.lower() in SOURCE_EXTENSIONS:
                selected_names.append(name)
    selected_names.extend(name for name in licenses if name in names)
    selected_names = sorted(set(selected_names))
    if not selected_names:
        raise RuntimeError("no selected source files found for {} {}".format(repository.name, tag))
    archive_result = subprocess.run(
        ["git", "-C", str(repository), "archive", commit, "--", *selected_names],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if archive_result.returncode != 0:
        raise RuntimeError("archive failed for {}: {}".format(tag, archive_result.stderr.decode().strip()))
    archive = archive_result.stdout
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as tar:
        tar.extractall(temporary_root, filter="data")
    missing = [selector for selector in selectors
               if not any(name == selector or name.startswith(selector.rstrip("/") + "/") for name in names)]
    return commit, missing


def collect(config_path, output_root, cache_root):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    versions = []
    source_root = Path(output_root) / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    for framework in sorted(config["frameworks"]):
        framework_config = config["frameworks"][framework]
        repository = ensure_repository(framework_config["repository"], Path(cache_root), framework)
        for tag in framework_config["tags"]:
            with tempfile.TemporaryDirectory(prefix="kernel-corpus-") as temporary:
                extracted = Path(temporary) / "source"
                commit, missing = materialize_tag(
                    repository, tag, framework_config["selectors"], framework_config["licenses"], extracted,
                )
                destination = source_root / framework / tag
                files = copy_snapshot(
                    extracted, destination, framework_config["selectors"], framework_config["licenses"],
                    framework + "/" + tag,
                )
                copied_licenses = [item["path"] for item in files if Path(item["path"]).name in framework_config["licenses"]]
                versions.append(build_manifest_record(
                    framework, tag, commit, framework_config["repository"], framework_config["backends"],
                    [{"path": "sources/{}/{}/{}".format(framework, tag, item["path"]), "sha256": item["sha256"]}
                     for item in files],
                    ["sources/{}/{}/{}".format(framework, tag, name) for name in copied_licenses],
                    missing,
                ))
    manifest = {"version": config["version"], "versions": versions}
    manifest_path = Path(output_root) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    collect(args.config, args.output, args.cache)
