from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator

import h5py
from stable_worldmodel.data import load_dataset
from stable_worldmodel.data.formats.lance import LanceWriter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mwm.data.metadata import write_dataset_metadata
from mwm.data.paths import local_path
from mwm.swm.restore import validate_restore_columns


REACHER_ENV_ID = "swm/ReacherDMControl-v0"
REACHER_LANCE = "reacher.lance"
REACHER_COLUMNS = ("pixels", "action", "qpos", "qvel", "observation")


def _rows(dataset: h5py.Dataset, start: int, stop: int) -> list:
    return [row for row in dataset[start:stop]]


def _episode_rows(handle: h5py.File, *, progress_every: int = 100) -> Iterator[dict]:
    offsets = handle["ep_offset"][:]
    lengths = handle["ep_len"][:]
    total = len(lengths)
    for index, (offset, length) in enumerate(zip(offsets, lengths)):
        if progress_every > 0 and index % progress_every == 0:
            print(f"Converting Reacher episode {index}/{total}", flush=True)
        start = int(offset)
        stop = start + int(length)
        yield {key: _rows(handle[key], start, stop) for key in REACHER_COLUMNS}


def convert_reacher_h5_to_lance(
    source: Path,
    output: Path,
    *,
    overwrite: bool = False,
    progress_every: int = 100,
) -> Path:
    if output.name != REACHER_LANCE:
        raise ValueError(f"Expected output path to end with {REACHER_LANCE!r}, got {output}.")
    if output.exists() or output.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")
        if output.is_symlink() or output.is_file():
            output.unlink()
        else:
            shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        missing = [key for key in (*REACHER_COLUMNS, "ep_offset", "ep_len") if key not in handle]
        if missing:
            raise KeyError(f"Reacher source is missing required dataset(s): {missing}")
        with LanceWriter(output, mode="error") as writer:
            writer.write_episodes(_episode_rows(handle, progress_every=progress_every))

    write_dataset_metadata(
        output,
        {
            "format": "swm_lance",
            "env_id": REACHER_ENV_ID,
            "restore_spec": "reacher_qpos_match_qpos_qvel",
            "image_shape": [224, 224],
            "action_dim": 2,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
            "dataset": {"pixels_key": "pixels", "action_key": "action"},
            "source": {
                "format": "hdf5",
                "path": str(source),
                "artifact": "reacher.h5",
                "standard": "paper_parity",
                "hf_dataset": "quentinll/lewm-reacher",
            },
        },
    )
    dataset = load_dataset(local_path(output), format="lance")
    try:
        validate_restore_columns(
            REACHER_ENV_ID,
            dataset.column_names,
            import_path="mwm.swm.restore.reacher_qpos_match_restore_spec",
        )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert upstream Le-WM Reacher HDF5 data to Lance.")
    parser.add_argument("--source", default="/tmp/mwm_reacher.h5", help="Path to extracted upstream reacher.h5")
    parser.add_argument("--output", default=f"data/upstream/{REACHER_LANCE}", help="Destination Lance dataset")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination dataset")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N episodes; 0 disables")
    args = parser.parse_args()

    output = convert_reacher_h5_to_lance(
        Path(args.source),
        Path(args.output),
        overwrite=bool(args.overwrite),
        progress_every=int(args.progress_every),
    )
    print(output)


if __name__ == "__main__":
    main()
