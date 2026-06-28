from __future__ import annotations

import argparse
import json
from pathlib import Path

from mwm.data.metadata import write_dataset_metadata
from mwm.upstream.paper_parity import paper_parity_dataset_spec, paper_parity_hdf5_metadata, paper_parity_lance_metadata


PUSHT_LANCE = paper_parity_dataset_spec("pusht").lance_name
REACHER_LANCE = paper_parity_dataset_spec("reacher").lance_name
TWOROOM_LANCE = paper_parity_dataset_spec("tworoom").lance_name
OGB_CUBE_LANCE = paper_parity_dataset_spec("ogb_cube").lance_name
OGB_CUBE_H5 = Path(paper_parity_dataset_spec("ogb_cube").source_artifacts["hdf5"])


def _require_lance_dataset(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing upstream Lance dataset {path}. "
            "Paper-parity data prep is Lance-only; provide the prebuilt Lance artifact."
        )
    if not (path / "_versions").is_dir() or not (path / "data").is_dir():
        raise FileNotFoundError(f"Invalid Lance dataset at {path}; expected _versions/ and data/ directories.")


def _write_pusht_metadata(path: Path) -> None:
    write_dataset_metadata(path, paper_parity_lance_metadata("pusht"))


def _write_tworoom_metadata(path: Path) -> None:
    write_dataset_metadata(path, paper_parity_lance_metadata("tworoom"))


def _write_reacher_metadata(path: Path) -> None:
    write_dataset_metadata(path, paper_parity_lance_metadata("reacher"))


def _write_ogb_cube_metadata(path: Path, *, source_h5: Path | None = None) -> None:
    write_dataset_metadata(path, paper_parity_hdf5_metadata("ogb_cube", source_path=source_h5))


def prepare_pusht(root: Path) -> Path:
    lance_path = root / PUSHT_LANCE
    _require_lance_dataset(lance_path)
    _write_pusht_metadata(lance_path)
    return lance_path


def prepare_reacher(root: Path) -> Path:
    lance_path = root / REACHER_LANCE
    _require_lance_dataset(lance_path)
    _write_reacher_metadata(lance_path)
    return lance_path


def prepare_tworoom(root: Path) -> Path:
    lance_path = root / TWOROOM_LANCE
    _require_lance_dataset(lance_path)
    _write_tworoom_metadata(lance_path)
    return lance_path


def _stable_worldmodel_dataset_path() -> Path | None:
    try:
        from stable_worldmodel.data import get_cache_dir
    except Exception:
        return None
    return Path(get_cache_dir(None, sub_folder="datasets")) / OGB_CUBE_H5


def _ogb_cube_h5_candidates(root: Path, source_h5: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if source_h5 is not None:
        candidates.append(Path(source_h5))
    candidates.extend(
        [
            root / OGB_CUBE_H5,
            root / OGB_CUBE_H5.name,
        ]
    )
    stable_path = _stable_worldmodel_dataset_path()
    if stable_path is not None:
        candidates.append(stable_path)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def prepare_ogb_cube(root: Path, *, source_h5: str | Path | None = None, convert_if_missing: bool = True) -> Path:
    lance_path = root / OGB_CUBE_LANCE
    if lance_path.is_dir():
        _require_lance_dataset(lance_path)
        _write_ogb_cube_metadata(lance_path, source_h5=Path(source_h5) if source_h5 is not None else None)
        return lance_path
    if not convert_if_missing:
        _require_lance_dataset(lance_path)
        return lance_path

    for candidate in _ogb_cube_h5_candidates(root, source_h5):
        if candidate.is_file():
            from mwm.upstream.converters.ogb_cube import convert_ogb_cube_hdf5_to_lance

            return convert_ogb_cube_hdf5_to_lance(candidate, lance_path, source_h5=candidate)
    candidates = "\n  - ".join(str(path) for path in _ogb_cube_h5_candidates(root, source_h5))
    raise FileNotFoundError(
        f"Missing upstream OGBench Cube Lance dataset {lance_path} and no extracted HDF5 source was found. "
        "Download/decompress the Le-WM Cube data, then run:\n"
        f"  python -m mwm.upstream.converters.ogb_cube --source <cube_single_expert.h5> --output {lance_path}\n"
        f"Checked:\n  - {candidates}"
    )


def main(root: str | Path = "data/upstream", *, source_h5: str | Path | None = None, convert_ogb_cube: bool = True) -> None:
    root = Path(root)
    out = {
        "pusht": str(prepare_pusht(root)),
        "reacher": str(prepare_reacher(root)),
        "tworoom": str(prepare_tworoom(root)),
        "ogb_cube": str(prepare_ogb_cube(root, source_h5=source_h5, convert_if_missing=convert_ogb_cube)),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare upstream Le-WM Lance datasets and metadata.")
    parser.add_argument("root", nargs="?", default="data/upstream", help="Directory containing upstream Lance datasets")
    parser.add_argument("--source-h5", default=None, help="Optional OGBench Cube HDF5 source for conversion")
    parser.add_argument(
        "--no-convert-ogb-cube",
        action="store_true",
        help="Require a prebuilt OGBench Cube Lance dataset instead of converting from HDF5",
    )
    args = parser.parse_args()
    main(args.root, source_h5=args.source_h5, convert_ogb_cube=not args.no_convert_ogb_cube)
