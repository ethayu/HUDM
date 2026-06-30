from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PaperParityDatasetSpec:
    key: str
    lance_name: str
    env_id: str
    restore_spec: str
    action_dim: int
    source_artifacts: Mapping[str, str]
    hf_dataset: str | None = None


PAPER_PARITY_DATASETS: dict[str, PaperParityDatasetSpec] = {
    "pusht": PaperParityDatasetSpec(
        key="pusht",
        lance_name="pusht_expert_train.lance",
        env_id="swm/PushT-v1",
        restore_spec="pusht_state_goal_state",
        action_dim=2,
        source_artifacts={"lance": "pusht_expert_train.lance"},
    ),
    "tworoom": PaperParityDatasetSpec(
        key="tworoom",
        lance_name="tworoom.lance",
        env_id="swm/TwoRoom-v1",
        restore_spec="point_state_goal_state",
        action_dim=2,
        source_artifacts={"lance": "tworoom.lance"},
    ),
    "reacher": PaperParityDatasetSpec(
        key="reacher",
        lance_name="reacher.lance",
        env_id="swm/ReacherDMControl-v0",
        restore_spec="reacher_qpos_match_qpos_qvel",
        action_dim=2,
        source_artifacts={"lance": "reacher.lance", "hdf5": "reacher.h5"},
        hf_dataset="quentinll/lewm-reacher",
    ),
    "ogb_cube": PaperParityDatasetSpec(
        key="ogb_cube",
        lance_name="ogb_cube_single_expert.lance",
        env_id="swm/OGBCube-v0",
        restore_spec="ogbench_cube_single_qpos_qvel_target_pose",
        action_dim=5,
        source_artifacts={"hdf5": "ogbench/cube_single_expert.h5"},
        hf_dataset="quentinll/lewm-cube",
    ),
}


def paper_parity_dataset_spec(name: str) -> PaperParityDatasetSpec:
    try:
        return PAPER_PARITY_DATASETS[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(PAPER_PARITY_DATASETS))
        raise KeyError(f"Unknown paper-parity dataset {name!r}; known datasets: {known}") from exc


def paper_parity_dataset_metadata(
    name: str,
    *,
    source_format: str,
    source_path: str | Path | None = None,
    source_artifact: str | None = None,
) -> dict[str, Any]:
    spec = paper_parity_dataset_spec(name)
    source_format = str(source_format)
    artifact = source_artifact or spec.source_artifacts[source_format]
    source: dict[str, Any] = {
        "format": source_format,
        "artifact": str(artifact),
        "standard": "paper_parity",
    }
    if source_path is not None:
        source["path"] = str(source_path)
    if spec.hf_dataset is not None:
        source["hf_dataset"] = spec.hf_dataset
    return {
        "format": "swm_lance",
        "env_id": spec.env_id,
        "restore_spec": spec.restore_spec,
        "image_shape": [224, 224],
        "action_dim": int(spec.action_dim),
        "action_low": [-1.0] * int(spec.action_dim),
        "action_high": [1.0] * int(spec.action_dim),
        "dataset": {"pixels_key": "pixels", "action_key": "action"},
        "source": source,
    }


def paper_parity_lance_metadata(name: str) -> dict[str, Any]:
    return paper_parity_dataset_metadata(name, source_format="lance")


def paper_parity_hdf5_metadata(name: str, source_path: str | Path | None = None) -> dict[str, Any]:
    return paper_parity_dataset_metadata(name, source_format="hdf5", source_path=source_path)


__all__ = [
    "PAPER_PARITY_DATASETS",
    "PaperParityDatasetSpec",
    "paper_parity_dataset_metadata",
    "paper_parity_hdf5_metadata",
    "paper_parity_lance_metadata",
    "paper_parity_dataset_spec",
]
