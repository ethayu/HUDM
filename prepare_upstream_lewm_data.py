from __future__ import annotations

import json
from pathlib import Path

from mwm.data.metadata import write_dataset_metadata


PUSHT_LANCE = "pusht_expert_train.lance"
REACHER_LANCE = "reacher.lance"
TWOROOM_LANCE = "tworoom.lance"
OGB_CUBE_LANCE = "ogb_cube_single_expert.lance"
OGB_CUBE_H5 = Path("ogbench") / "cube_single_expert.h5"


def _require_lance_dataset(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing upstream Lance dataset {path}. "
            "Paper-parity data prep is Lance-only; provide the prebuilt Lance artifact."
        )
    if not (path / "_versions").is_dir() or not (path / "data").is_dir():
        raise FileNotFoundError(f"Invalid Lance dataset at {path}; expected _versions/ and data/ directories.")


def _write_pusht_metadata(path: Path) -> None:
    write_dataset_metadata(
        path,
        {
            "format": "swm_lance",
            "env_id": "swm/PushT-v1",
            "restore_spec": "pusht_state_goal_state",
            "image_shape": [224, 224],
            "action_dim": 2,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
            "dataset": {"pixels_key": "pixels", "action_key": "action"},
            "source": {
                "format": "lance",
                "artifact": PUSHT_LANCE,
                "standard": "paper_parity",
            },
        },
    )


def _write_tworoom_metadata(path: Path) -> None:
    write_dataset_metadata(
        path,
        {
            "format": "swm_lance",
            "env_id": "swm/TwoRoom-v1",
            "restore_spec": "point_state_goal_state",
            "image_shape": [224, 224],
            "action_dim": 2,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
            "dataset": {"pixels_key": "pixels", "action_key": "action"},
            "source": {
                "format": "lance",
                "artifact": TWOROOM_LANCE,
                "standard": "paper_parity",
            },
        },
    )


def _write_reacher_metadata(path: Path) -> None:
    write_dataset_metadata(
        path,
        {
            "format": "swm_lance",
            "env_id": "swm/ReacherDMControl-v0",
            "restore_spec": "reacher_qpos_match_qpos_qvel",
            "image_shape": [224, 224],
            "action_dim": 2,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
            "dataset": {"pixels_key": "pixels", "action_key": "action"},
            "source": {
                "format": "lance",
                "artifact": REACHER_LANCE,
                "standard": "paper_parity",
                "hf_dataset": "quentinll/lewm-reacher",
            },
        },
    )


def _write_ogb_cube_metadata(path: Path, *, source_h5: Path | None = None) -> None:
    source: dict[str, object] = {
        "format": "hdf5",
        "artifact": str(OGB_CUBE_H5),
        "standard": "paper_parity",
        "hf_dataset": "quentinll/lewm-cube",
    }
    if source_h5 is not None:
        source["path"] = str(source_h5)
    write_dataset_metadata(
        path,
        {
            "format": "swm_lance",
            "env_id": "swm/OGBCube-v0",
            "restore_spec": "ogbench_cube_single_qpos_qvel_target_pose",
            "image_shape": [224, 224],
            "action_dim": 5,
            "action_low": [-1.0] * 5,
            "action_high": [1.0] * 5,
            "dataset": {"pixels_key": "pixels", "action_key": "action"},
            "source": source,
        },
    )


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
            from scripts.research.convert_ogb_cube_hdf5_to_lance import convert_ogb_cube_hdf5_to_lance

            return convert_ogb_cube_hdf5_to_lance(candidate, lance_path, source_h5=candidate)
    candidates = "\n  - ".join(str(path) for path in _ogb_cube_h5_candidates(root, source_h5))
    raise FileNotFoundError(
        f"Missing upstream OGBench Cube Lance dataset {lance_path} and no extracted HDF5 source was found. "
        "Download/decompress the Le-WM Cube data, then run:\n"
        f"  python scripts/research/convert_ogb_cube_hdf5_to_lance.py --source <cube_single_expert.h5> --output {lance_path}\n"
        f"Checked:\n  - {candidates}"
    )


def main() -> None:
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/upstream")
    out = {
        "pusht": str(prepare_pusht(root)),
        "reacher": str(prepare_reacher(root)),
        "tworoom": str(prepare_tworoom(root)),
        "ogb_cube": str(prepare_ogb_cube(root)),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
