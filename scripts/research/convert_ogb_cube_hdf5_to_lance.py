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


OGB_CUBE_ENV_ID = "swm/OGBCube-v0"
OGB_CUBE_LANCE = "ogb_cube_single_expert.lance"
OGB_CUBE_RESTORE_SPEC = "ogbench_cube_single_qpos_qvel_target_pose"
OGB_CUBE_RESTORE_IMPORT_PATH = "mwm.ogbench.restore.ogbench_cube_restore_spec"
OGB_CUBE_COLUMNS = (
    "pixels",
    "action",
    "observation",
    "qpos",
    "qvel",
    "privileged/block_0_pos",
    "privileged/block_0_quat",
)
OGB_CUBE_SOURCE_ALIASES = {
    "privileged/block_0_pos": ("privileged/block_0_pos", "privileged_block_0_pos"),
    "privileged/block_0_quat": ("privileged/block_0_quat", "privileged_block_0_quat"),
}


def _source_key(handle: h5py.File, canonical: str) -> str:
    candidates = OGB_CUBE_SOURCE_ALIASES.get(canonical, (canonical,))
    for candidate in candidates:
        if candidate in handle:
            return candidate
    raise KeyError(f"OGBench Cube source is missing required dataset {canonical!r}; tried {candidates}.")


def _rows(dataset: h5py.Dataset, start: int, stop: int) -> list:
    return [row for row in dataset[start:stop]]


def _episode_rows(handle: h5py.File, *, progress_every: int = 100) -> Iterator[dict]:
    offsets = handle["ep_offset"][:]
    lengths = handle["ep_len"][:]
    source_keys = {canonical: _source_key(handle, canonical) for canonical in OGB_CUBE_COLUMNS}
    total = len(lengths)
    for index, (offset, length) in enumerate(zip(offsets, lengths)):
        if progress_every > 0 and index % progress_every == 0:
            print(f"Converting OGBench Cube episode {index}/{total}", flush=True)
        start = int(offset)
        stop = start + int(length)
        yield {
            canonical: _rows(handle[source_key], start, stop)
            for canonical, source_key in source_keys.items()
        }


def _replace_output(output: Path, *, overwrite: bool) -> None:
    if not (output.exists() or output.is_symlink()):
        return
    if not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")
    if output.is_symlink() or output.is_file():
        output.unlink()
    else:
        shutil.rmtree(output)


def _write_metadata(output: Path, source: Path) -> None:
    write_dataset_metadata(
        output,
        {
            "format": "swm_lance",
            "env_id": OGB_CUBE_ENV_ID,
            "restore_spec": OGB_CUBE_RESTORE_SPEC,
            "image_shape": [224, 224],
            "action_dim": 5,
            "action_low": [-1.0] * 5,
            "action_high": [1.0] * 5,
            "dataset": {"pixels_key": "pixels", "action_key": "action"},
            "source": {
                "format": "hdf5",
                "artifact": "ogbench/cube_single_expert.h5",
                "path": str(source),
                "standard": "paper_parity",
                "hf_dataset": "quentinll/lewm-cube",
            },
        },
    )


def convert_ogb_cube_hdf5_to_lance(
    source: Path | str,
    output: Path | str,
    *,
    overwrite: bool = False,
    progress_every: int = 100,
    source_h5: Path | str | None = None,
) -> Path:
    source = Path(source)
    output = Path(output)
    if output.name != OGB_CUBE_LANCE:
        raise ValueError(f"Expected output path to end with {OGB_CUBE_LANCE!r}, got {output}.")
    _replace_output(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        missing = [key for key in ("ep_offset", "ep_len") if key not in handle]
        if missing:
            raise KeyError(f"OGBench Cube source is missing required dataset(s): {missing}")
        for canonical in OGB_CUBE_COLUMNS:
            _source_key(handle, canonical)
        with LanceWriter(output, mode="error") as writer:
            writer.write_episodes(_episode_rows(handle, progress_every=progress_every))

    _write_metadata(output, Path(source_h5 or source))

    dataset = load_dataset(local_path(output), format="lance")
    try:
        validate_restore_columns(
            OGB_CUBE_ENV_ID,
            dataset.column_names,
            import_path=OGB_CUBE_RESTORE_IMPORT_PATH,
        )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert upstream Le-WM OGBench Cube HDF5 data to Lance.")
    parser.add_argument("--source", required=True, help="Path to extracted ogbench/cube_single_expert.h5")
    parser.add_argument("--output", default=f"data/upstream/{OGB_CUBE_LANCE}", help="Destination Lance dataset")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination dataset")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N episodes; 0 disables")
    args = parser.parse_args()

    output = convert_ogb_cube_hdf5_to_lance(
        args.source,
        args.output,
        overwrite=bool(args.overwrite),
        progress_every=int(args.progress_every),
    )
    print(output)


if __name__ == "__main__":
    main()
