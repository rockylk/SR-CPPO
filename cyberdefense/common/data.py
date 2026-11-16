"""Select validated 16-dimensional telemetry arrays, independent of the shell CWD."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "optc": ("optc_ot_sequence_benign2.npy", "optc_ot_sequence_malicious2.npy"),
    "dapt2020": ("dapt_ot_sequence_benign2.npy", "dapt_ot_sequence_malicious2.npy"),
}


@dataclass
class DataSelection:
    dataset: str = "optc"
    directory: Path | None = None


_selection = DataSelection()


def select_dataset(dataset="optc", directory=None):
    """Set the input pair used by the experiment entry points."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose {tuple(DATASETS)}")
    _selection.dataset = dataset
    _selection.directory = Path(directory).expanduser().resolve() if directory else None


def data_paths(dataset=None, directory=None, horizon=200):
    """Return absolute feature paths; optional label arrays are not model inputs."""
    dataset = dataset or _selection.dataset
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    parent = (
        Path(directory).expanduser().resolve()
        if directory
        else (
            _selection.directory
            if dataset == _selection.dataset and _selection.directory
            else PROJECT_ROOT / "data" / dataset
        )
    )
    paths = tuple(parent / name for name in DATASETS[dataset])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {dataset} data: {path}. Set --data-dir to the directory containing both feature arrays."
            )
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 16 or values.shape[0] < horizon + 1:
            raise ValueError(
                f"Expected N x 16 with N >= {horizon + 1}: {path}, shape={values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite telemetry: {path}")
    return tuple(str(path) for path in paths)
