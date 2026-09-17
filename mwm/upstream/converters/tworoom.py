from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterator

import h5py
from stable_worldmodel.data import load_dataset
from stable_worldmodel.data.formats.lance import LanceWriter

from mwm.data.metadata import write_dataset_metadata
from mwm.data.paths import local_path
from mwm.upstream.paper_parity import paper_parity_dataset_metadata, paper_parity_dataset_spec


TWOROOM_SPEC = paper_parity_dataset_spec("tworoom")
TWOROOM_ENV_ID = TWOROOM_SPEC.env_id
TWOROOM_LANCE = TWOROOM_SPEC.lance_name
TWOROOM_COLUMNS = (
    "action",
    "distance_to_target",
    "ep_idx",
    "id",
    "observation",
    "pixels",
    "pos_agent",
    "pos_target",
    "proprio",
    "render_time",
    "reward",
    "terminated",
    "truncated",
)


def _rows(dataset: h5py.Dataset, start: int, stop: int) -> list:
    return [row for row in dataset[start:stop]]


def _episode_rows(handle: h5py.File, *, progress_every: int = 200) -> Iterator[dict]:
    offsets = handle["ep_offset"][:]
    lengths = handle["ep_len"][:]
    total = len(lengths)
    for index, (offset, length) in enumerate(zip(offsets, lengths)):
        if progress_every > 0 and index % progress_every == 0:
            print(f"Converting TwoRoom episode {index}/{total}", flush=True)
        start = int(offset)
        stop = start + int(length)
        yield {key: _rows(handle[key], start, stop) for key in TWOROOM_COLUMNS}


def convert_tworoom_h5_to_lance(
    source: Path,
    output: Path,
    *,
    overwrite: bool = False,
    progress_every: int = 200,
) -> Path:
    if output.name != TWOROOM_LANCE:
        raise ValueError(f"Expected output path to end with {TWOROOM_LANCE!r}, got {output}.")
    if output.exists() or output.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")
        if output.is_symlink() or output.is_file():
            output.unlink()
        else:
            shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        missing = [key for key in (*TWOROOM_COLUMNS, "ep_offset", "ep_len") if key not in handle]
        if missing:
            raise KeyError(f"TwoRoom source is missing required dataset(s): {missing}")
        with LanceWriter(output, mode="error") as writer:
            writer.write_episodes(_episode_rows(handle, progress_every=progress_every))

    write_dataset_metadata(output, paper_parity_dataset_metadata("tworoom", source_format="lance"))
    dataset = load_dataset(local_path(output), format="lance")
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert upstream Le-WM TwoRoom HDF5 data to Lance.")
    parser.add_argument("--source", required=True, help="Path to extracted tworoom.h5")
    parser.add_argument("--output", default=f"data/upstream/{TWOROOM_LANCE}", help="Destination Lance dataset")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination dataset")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N episodes; 0 disables")
    args = parser.parse_args()

    output = convert_tworoom_h5_to_lance(
        Path(args.source),
        Path(args.output),
        overwrite=bool(args.overwrite),
        progress_every=int(args.progress_every),
    )
    print(output)


if __name__ == "__main__":
    main()
