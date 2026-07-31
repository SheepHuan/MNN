"""Read and write the platform-neutral PMC dataset bundle."""

from pathlib import Path

from .model import PmcDataset


def load_normalized_bundle(path):
    return PmcDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_normalized_bundle(dataset, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        dataset.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
