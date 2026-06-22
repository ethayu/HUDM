from __future__ import annotations

import json
from pathlib import Path

from mwm.data.metadata import write_dataset_metadata


PUSHT_LANCE = "pusht_expert_train.lance"
TWOROOM_LANCE = "tworoom.lance"


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


def prepare_pusht(root: Path) -> Path:
    lance_path = root / PUSHT_LANCE
    _require_lance_dataset(lance_path)
    _write_pusht_metadata(lance_path)
    return lance_path


def prepare_tworoom(root: Path) -> Path:
    lance_path = root / TWOROOM_LANCE
    _require_lance_dataset(lance_path)
    _write_tworoom_metadata(lance_path)
    return lance_path


def main() -> None:
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/upstream")
    out = {
        "pusht": str(prepare_pusht(root)),
        "tworoom": str(prepare_tworoom(root)),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
