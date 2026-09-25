"""Pre-outcome episode covariates for the five-anchor diagnostic.

The schedule diagnostic evaluates every planner cell on the same first fifty
start/goal windows from each canonical release manifest.  This module derives
two environment-specific covariates from those windows without reading any
planner result.  The resulting table can therefore be joined to episode
outcomes by ``(canonical_config, episode_index)`` without post-treatment
leakage.

Only the small state columns required below are read from Lance; pixels and
benchmark output directories are never opened.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import yaml

from mwm.data.manifest import load_manifest
from mwm.io import file_sha256, jsonable


SCHEMA_VERSION = "mwm.release20260728.episode_covariates/v1"
EPISODES_PER_MATRIX = 50
EXPECTED_MATRICES = 8
EXPECTED_ROWS = EXPECTED_MATRICES * EPISODES_PER_MATRIX
WALL_CENTER_PX = 112.0
NUMERIC_EPS = 1e-12

DEFAULT_PLAN_PATH = Path(
    "reports/research/release20260728_schedule_screening/five_anchor_plan.json"
)
DEFAULT_JSON_PATH = Path(
    "reports/research/release20260728_schedule_screening/"
    "five_anchor_episode_covariates.json"
)
DEFAULT_CSV_PATH = Path(
    "reports/research/release20260728_schedule_screening/"
    "five_anchor_episode_covariates.csv"
)

COVARIATE_NAMES: dict[str, tuple[str, str]] = {
    "swm/PushT-v1": ("block_translation_px", "block_rotation_rad"),
    "swm/ReacherDMControl-v0": (
        "max_wrapped_joint_displacement_rad",
        "expert_joint_path_tortuosity",
    ),
    "swm/OGBCube-v0": ("cube_translation_m", "cube_rotation_rad"),
    "swm/TwoRoom-v1": ("cross_room", "expert_path_tortuosity"),
}

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "swm/PushT-v1": ("state",),
    "swm/ReacherDMControl-v0": ("qpos",),
    "swm/OGBCube-v0": (
        "privileged/block_0_pos",
        "privileged/block_0_quat",
    ),
    "swm/TwoRoom-v1": ("proprio", "observation"),
}

CSV_COVARIATE_COLUMNS = tuple(
    sorted({name for names in COVARIATE_NAMES.values() for name in names})
)
CSV_COLUMNS = (
    "record_key",
    "canonical_config",
    "canonical_config_sha256",
    "canonical_manifest_config",
    "canonical_manifest_config_sha256",
    "canonical_manifest_path",
    "canonical_manifest_file_sha256",
    "canonical_manifest_sha256",
    "dataset_path",
    "env_id",
    "goal_offset",
    "effective_goal_offset",
    "episode_index",
    "dataset_episode",
    "start_step",
    "goal_step",
    "start_row",
    "goal_row",
    *CSV_COVARIATE_COLUMNS,
    "max_raw_joint_displacement_rad",
    "reacher_angle_branch_cut",
    "reacher_tortuosity_degenerate",
    "tworoom_wall_axis",
    "tworoom_tortuosity_degenerate",
)


class EpisodeCovariateError(RuntimeError):
    """Raised when source provenance or episode geometry violates the contract."""


class RowReader(Protocol):
    @property
    def column_names(self) -> Sequence[str]: ...

    def take(self, row_ids: Sequence[int], columns: Sequence[str]) -> Mapping[str, Sequence[Any]]: ...


class _LanceRowReader:
    def __init__(self, path: Path) -> None:
        try:
            import lance
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise EpisodeCovariateError(
                "Episode covariate extraction requires the `lance` package."
            ) from exc
        self._dataset = lance.dataset(str(path))
        self._column_names = tuple(str(field.name) for field in self._dataset.schema)

    @property
    def column_names(self) -> Sequence[str]:
        return self._column_names

    def take(self, row_ids: Sequence[int], columns: Sequence[str]) -> Mapping[str, Sequence[Any]]:
        return self._dataset.take(list(row_ids), columns=list(columns)).to_pydict()


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EpisodeCovariateError(f"Expected a YAML mapping in {path}.")
    return dict(payload)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def wrap_angle_delta(delta: Any) -> np.ndarray:
    """Map angular differences to the principal interval ``[-pi, pi]``."""

    values = np.asarray(delta, dtype=np.float64)
    return np.arctan2(np.sin(values), np.cos(values))


def quaternion_geodesic_rad(start: Any, goal: Any) -> float:
    """Return the sign-invariant geodesic angle between two quaternions."""

    q0 = np.asarray(start, dtype=np.float64).reshape(-1)
    q1 = np.asarray(goal, dtype=np.float64).reshape(-1)
    if q0.shape != (4,) or q1.shape != (4,):
        raise EpisodeCovariateError(
            f"Expected two four-component quaternions, got {q0.shape} and {q1.shape}."
        )
    norm0 = float(np.linalg.norm(q0))
    norm1 = float(np.linalg.norm(q1))
    if not math.isfinite(norm0) or not math.isfinite(norm1) or min(norm0, norm1) <= NUMERIC_EPS:
        raise EpisodeCovariateError("Quaternion norm is zero or non-finite.")
    dot = float(np.dot(q0 / norm0, q1 / norm1))
    return float(2.0 * math.acos(float(np.clip(abs(dot), 0.0, 1.0))))


def _path_tortuosity(
    path: Any,
    *,
    periodic: bool,
) -> tuple[float | None, bool]:
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise EpisodeCovariateError(f"Expected a path with at least two points, got {values.shape}.")
    increments = np.diff(values, axis=0)
    endpoint = values[-1] - values[0]
    if periodic:
        increments = wrap_angle_delta(increments)
        endpoint = wrap_angle_delta(endpoint)
    length = float(np.linalg.norm(increments, axis=1).sum())
    chord = float(np.linalg.norm(endpoint))
    if not math.isfinite(length) or not math.isfinite(chord):
        raise EpisodeCovariateError("Path length or endpoint distance is non-finite.")
    if chord <= NUMERIC_EPS:
        return None, True
    ratio = length / chord
    if ratio < 1.0 - 1e-7:
        raise EpisodeCovariateError(
            f"Path tortuosity violates the triangle inequality: length={length}, chord={chord}."
        )
    return float(max(1.0, ratio)), False


def infer_tworoom_wall_axis(observation: Any) -> tuple[int, str]:
    """Infer TwoRoom's wall-normal coordinate from encoded doorway centers.

    Returns coordinate index 0/1 and the human-readable wall orientation.
    A vertical wall has constant door x=112, so room membership is determined
    by the agent's x coordinate.  A horizontal wall analogously uses y.
    """

    values = np.asarray(observation, dtype=np.float64).reshape(-1)
    if values.size < 10 or (values.size - 4) % 2:
        raise EpisodeCovariateError(
            f"Malformed TwoRoom observation shape {values.shape}; expected agent/target plus door pairs."
        )
    doors = values[4:].reshape(-1, 2)
    active = doors[np.any(np.abs(doors) > 1e-6, axis=1)]
    if not len(active):
        raise EpisodeCovariateError("TwoRoom observation has no active doorway coordinates.")
    vertical = bool(np.all(np.isclose(active[:, 0], WALL_CENTER_PX, atol=1e-4, rtol=0.0)))
    horizontal = bool(np.all(np.isclose(active[:, 1], WALL_CENTER_PX, atol=1e-4, rtol=0.0)))
    if vertical == horizontal:
        raise EpisodeCovariateError(
            f"TwoRoom wall axis is ambiguous from doorway coordinates {active.tolist()}."
        )
    return (0, "vertical") if vertical else (1, "horizontal")


def compute_episode_covariates(
    env_id: str,
    columns: Mapping[str, Any],
) -> tuple[dict[str, float | bool | None], dict[str, Any]]:
    """Compute the two pre-specified covariates and environment audit fields."""

    env = str(env_id)
    missing = sorted(set(REQUIRED_COLUMNS.get(env, ())) - set(columns))
    if env not in COVARIATE_NAMES:
        raise EpisodeCovariateError(f"Unsupported release environment {env!r}.")
    if missing:
        raise EpisodeCovariateError(f"{env} window is missing columns {missing}.")

    audit: dict[str, Any] = {}
    if env == "swm/PushT-v1":
        state = np.asarray(columns["state"], dtype=np.float64)
        if state.ndim != 2 or state.shape[1] < 5:
            raise EpisodeCovariateError(f"Malformed PushT state path {state.shape}.")
        delta = state[-1] - state[0]
        covariates = {
            "block_translation_px": float(np.linalg.norm(delta[2:4])),
            "block_rotation_rad": float(abs(wrap_angle_delta(delta[4]).item())),
        }
    elif env == "swm/ReacherDMControl-v0":
        qpos = np.asarray(columns["qpos"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 2:
            raise EpisodeCovariateError(f"Malformed Reacher qpos path {qpos.shape}.")
        raw = qpos[-1] - qpos[0]
        wrapped = wrap_angle_delta(raw)
        tortuosity, degenerate = _path_tortuosity(qpos, periodic=True)
        covariates = {
            "max_wrapped_joint_displacement_rad": float(np.max(np.abs(wrapped))),
            "expert_joint_path_tortuosity": tortuosity,
        }
        audit = {
            "max_raw_joint_displacement_rad": float(np.max(np.abs(raw))),
            "reacher_angle_branch_cut": bool(
                np.any(np.abs(np.abs(raw) - np.abs(wrapped)) > 1e-7)
            ),
            "reacher_tortuosity_degenerate": bool(degenerate),
        }
    elif env == "swm/OGBCube-v0":
        position = np.asarray(columns["privileged/block_0_pos"], dtype=np.float64)
        quaternion = np.asarray(columns["privileged/block_0_quat"], dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3:
            raise EpisodeCovariateError(f"Malformed OGB Cube position path {position.shape}.")
        if quaternion.ndim != 2 or quaternion.shape[1] != 4:
            raise EpisodeCovariateError(f"Malformed OGB Cube quaternion path {quaternion.shape}.")
        covariates = {
            "cube_translation_m": float(np.linalg.norm(position[-1] - position[0])),
            "cube_rotation_rad": quaternion_geodesic_rad(quaternion[0], quaternion[-1]),
        }
    else:
        position = np.asarray(columns["proprio"], dtype=np.float64)
        observation = np.asarray(columns["observation"], dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 2:
            raise EpisodeCovariateError(f"Malformed TwoRoom position path {position.shape}.")
        if observation.ndim != 2:
            raise EpisodeCovariateError(f"Malformed TwoRoom observation path {observation.shape}.")
        wall_axis, wall_orientation = infer_tworoom_wall_axis(observation[0])
        # Geometry is episode-static. Refuse to infer from a stale or changing
        # observation layout rather than silently assigning room membership.
        for row in observation[1:]:
            row_axis, row_orientation = infer_tworoom_wall_axis(row)
            if (row_axis, row_orientation) != (wall_axis, wall_orientation):
                raise EpisodeCovariateError("TwoRoom wall geometry changes inside a manifest window.")
        tortuosity, degenerate = _path_tortuosity(position, periodic=False)
        covariates = {
            "cross_room": bool(
                (position[0, wall_axis] < WALL_CENTER_PX)
                != (position[-1, wall_axis] < WALL_CENTER_PX)
            ),
            "expert_path_tortuosity": tortuosity,
        }
        audit = {
            "tworoom_wall_axis": wall_orientation,
            "tworoom_tortuosity_degenerate": bool(degenerate),
        }

    expected = set(COVARIATE_NAMES[env])
    if set(covariates) != expected:
        raise EpisodeCovariateError(
            f"Internal covariate schema mismatch for {env}: {sorted(covariates)}."
        )
    return covariates, audit


def _reader_rows(
    reader: RowReader,
    row_ids: Sequence[int],
    columns: Sequence[str],
) -> dict[str, np.ndarray]:
    fetched = reader.take(row_ids, columns)
    if set(fetched) != set(columns):
        raise EpisodeCovariateError(
            f"Dataset reader returned columns {sorted(fetched)}, expected {sorted(columns)}."
        )
    result: dict[str, np.ndarray] = {}
    for name in columns:
        values = np.asarray(fetched[name])
        if len(values) != len(row_ids):
            raise EpisodeCovariateError(
                f"Column {name!r} returned {len(values)} rows for {len(row_ids)} requested rows."
            )
        result[name] = values
    return result


def _record_key(canonical_config: str, episode_index: int) -> str:
    return f"{canonical_config}::episode_index={int(episode_index):04d}"


def _extract_matrix_records(
    *,
    root: Path,
    plan_matrix: Mapping[str, Any],
    opener: Callable[[Path], RowReader],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_config = str(plan_matrix.get("canonical_config", ""))
    if not canonical_config:
        raise EpisodeCovariateError("Plan matrix is missing canonical_config.")
    config_path = _resolve(root, canonical_config)
    config_sha = file_sha256(config_path)
    expected_config_sha = plan_matrix.get("canonical_config_sha256")
    if expected_config_sha and str(expected_config_sha) != config_sha:
        raise EpisodeCovariateError(
            f"Canonical config hash changed for {canonical_config}: "
            f"expected {expected_config_sha}, got {config_sha}."
        )
    config = _load_yaml_mapping(config_path)
    env_id = str(config.get("env_id", ""))
    if env_id not in COVARIATE_NAMES:
        raise EpisodeCovariateError(f"Unsupported env_id {env_id!r} in {canonical_config}.")

    manifest_block = config.get("manifest")
    if not isinstance(manifest_block, dict) or not manifest_block.get("config"):
        raise EpisodeCovariateError(f"{canonical_config} is missing manifest.config.")
    manifest_config_ref = str(manifest_block["config"])
    manifest_config_path = _resolve(root, manifest_config_ref)
    manifest_config_sha = file_sha256(manifest_config_path)
    manifest_config = _load_yaml_mapping(manifest_config_path)
    if not manifest_config.get("path"):
        raise EpisodeCovariateError(f"{manifest_config_ref} is missing path.")
    manifest_ref = str(manifest_config["path"])
    manifest_path = _resolve(root, manifest_ref)
    manifest_file_sha = file_sha256(manifest_path)
    manifest = load_manifest(manifest_path)
    manifest_hash = str(manifest.get("manifest_sha256", ""))
    expected_manifest_hash = plan_matrix.get("canonical_manifest_sha256")
    if expected_manifest_hash and str(expected_manifest_hash) != manifest_hash:
        raise EpisodeCovariateError(
            f"Canonical manifest changed for {canonical_config}: "
            f"expected {expected_manifest_hash}, got {manifest_hash}."
        )
    if str(manifest.get("env_id")) != env_id:
        raise EpisodeCovariateError(
            f"Manifest env {manifest.get('env_id')!r} does not match config env {env_id!r}."
        )
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < EPISODES_PER_MATRIX:
        raise EpisodeCovariateError(
            f"Manifest {manifest_ref} has fewer than {EPISODES_PER_MATRIX} pairs."
        )
    pairs = pairs[:EPISODES_PER_MATRIX]
    goal_offset = int(manifest.get("goal_offset"))
    effective_offset = int(manifest.get("effective_goal_offset", goal_offset))
    run_goal_offset = int(
        config.get("run_defaults", {}).get("eval", {}).get("goal_offset", goal_offset)
    )
    if run_goal_offset != goal_offset:
        raise EpisodeCovariateError(
            f"Config goal_offset={run_goal_offset} does not match manifest goal_offset={goal_offset}."
        )

    dataset_ref = str(manifest.get("dataset_path", ""))
    if not dataset_ref:
        raise EpisodeCovariateError(f"Manifest {manifest_ref} is missing dataset_path.")
    dataset_path = _resolve(root, dataset_ref)
    reader = opener(dataset_path)
    required = ("episode_idx", "step_idx", *REQUIRED_COLUMNS[env_id])
    missing_columns = sorted(set(required) - set(str(name) for name in reader.column_names))
    if missing_columns:
        raise EpisodeCovariateError(
            f"Dataset {dataset_ref} is missing required columns {missing_columns}."
        )

    row_ids = sorted(
        {
            row
            for pair in pairs
            for row in range(int(pair["start_row"]), int(pair["goal_row"]) + 1)
        }
    )
    fetched = _reader_rows(reader, row_ids, required)
    row_lookup = {row: position for position, row in enumerate(row_ids)}

    records: list[dict[str, Any]] = []
    for episode_index, pair in enumerate(pairs):
        dataset_episode = int(pair["episode"])
        start_step = int(pair["start_step"])
        goal_step = int(pair["goal_step"])
        start_row = int(pair["start_row"])
        goal_row = int(pair["goal_row"])
        if goal_step - start_step != effective_offset or goal_row - start_row != effective_offset:
            raise EpisodeCovariateError(
                f"{canonical_config} episode_index={episode_index} does not span "
                f"effective_goal_offset={effective_offset}."
            )
        window_rows = list(range(start_row, goal_row + 1))
        indices = [row_lookup[row] for row in window_rows]
        observed_episodes = np.asarray(fetched["episode_idx"])[indices].astype(np.int64)
        observed_steps = np.asarray(fetched["step_idx"])[indices].astype(np.int64)
        expected_steps = np.arange(start_step, goal_step + 1, dtype=np.int64)
        if not np.array_equal(observed_episodes, np.full(len(indices), dataset_episode)):
            raise EpisodeCovariateError(
                f"{canonical_config} episode_index={episode_index} crosses dataset episodes."
            )
        if not np.array_equal(observed_steps, expected_steps):
            raise EpisodeCovariateError(
                f"{canonical_config} episode_index={episode_index} has non-contiguous or mismatched steps."
            )
        window = {
            name: np.asarray(fetched[name])[indices]
            for name in REQUIRED_COLUMNS[env_id]
        }
        covariates, audit = compute_episode_covariates(env_id, window)
        key = _record_key(canonical_config, episode_index)
        records.append(
            {
                "record_key": key,
                "canonical_config": canonical_config,
                "canonical_config_sha256": config_sha,
                "canonical_manifest_config": manifest_config_ref,
                "canonical_manifest_config_sha256": manifest_config_sha,
                "canonical_manifest_path": manifest_ref,
                "canonical_manifest_file_sha256": manifest_file_sha,
                "canonical_manifest_sha256": manifest_hash,
                "dataset_path": dataset_ref,
                "env_id": env_id,
                "goal_offset": goal_offset,
                "effective_goal_offset": effective_offset,
                "episode_index": episode_index,
                "dataset_episode": dataset_episode,
                "start_step": start_step,
                "goal_step": goal_step,
                "start_row": start_row,
                "goal_row": goal_row,
                "covariates": covariates,
                "audit": audit,
            }
        )

    matrix_provenance = {
        "canonical_config": canonical_config,
        "canonical_config_sha256": config_sha,
        "canonical_manifest_config": manifest_config_ref,
        "canonical_manifest_config_sha256": manifest_config_sha,
        "canonical_manifest_path": manifest_ref,
        "canonical_manifest_file_sha256": manifest_file_sha,
        "canonical_manifest_sha256": manifest_hash,
        "dataset_path": dataset_ref,
        "env_id": env_id,
        "goal_offset": goal_offset,
        "effective_goal_offset": effective_offset,
        "episodes": len(records),
        "unique_dataset_episodes": len({record["dataset_episode"] for record in records}),
    }
    return records, matrix_provenance


def extract_episode_covariates(
    *,
    repo_root: str | Path,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    dataset_opener: Callable[[Path], RowReader] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    resolved_plan = _resolve(root, plan_path)
    plan = json.loads(resolved_plan.read_text(encoding="utf-8"))
    matrices = plan.get("matrices")
    if not isinstance(matrices, list) or len(matrices) != EXPECTED_MATRICES:
        raise EpisodeCovariateError(
            f"Expected {EXPECTED_MATRICES} matrices in {resolved_plan}, "
            f"found {len(matrices) if isinstance(matrices, list) else 'malformed'}."
        )
    opener = dataset_opener or _LanceRowReader
    all_records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for matrix in matrices:
        if not isinstance(matrix, dict):
            raise EpisodeCovariateError("Plan matrices must be mappings.")
        records, matrix_provenance = _extract_matrix_records(
            root=root,
            plan_matrix=matrix,
            opener=opener,
        )
        all_records.extend(records)
        provenance.append(matrix_provenance)
    all_records.sort(key=lambda row: (str(row["canonical_config"]), int(row["episode_index"])))
    records_by_key = {str(row["record_key"]): row for row in all_records}
    if len(records_by_key) != len(all_records):
        raise EpisodeCovariateError("Duplicate canonical_config/episode_index record key.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "five_anchor_plan": str(
                resolved_plan.relative_to(root) if resolved_plan.is_relative_to(root) else resolved_plan
            ),
            "five_anchor_plan_file_sha256": file_sha256(resolved_plan),
        },
        "design": {
            "matrices": EXPECTED_MATRICES,
            "episodes_per_matrix": EPISODES_PER_MATRIX,
            "rows": EXPECTED_ROWS,
            "window_semantics": "inclusive_start_and_goal_rows",
            "join_key": ["canonical_config", "episode_index"],
            "covariates_by_env": {
                env: list(names) for env, names in sorted(COVARIATE_NAMES.items())
            },
        },
        "matrix_provenance": provenance,
        "records": records_by_key,
    }
    validate_covariate_payload(payload)
    return payload


def validate_covariate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise EpisodeCovariateError(f"Unsupported covariate schema {payload.get('schema_version')!r}.")
    raw_records = payload.get("records")
    if not isinstance(raw_records, dict) or len(raw_records) != EXPECTED_ROWS:
        raise EpisodeCovariateError(
            f"Expected {EXPECTED_ROWS} keyed covariate records, "
            f"found {len(raw_records) if isinstance(raw_records, dict) else 'malformed'}."
        )
    records = list(raw_records.values())
    by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    env_goals: Counter[tuple[str, int]] = Counter()
    for key, record in raw_records.items():
        if not isinstance(record, dict):
            raise EpisodeCovariateError(f"Covariate record {key!r} is not a mapping.")
        canonical_config = str(record.get("canonical_config", ""))
        episode_index = int(record.get("episode_index", -1))
        if key != _record_key(canonical_config, episode_index) or record.get("record_key") != key:
            raise EpisodeCovariateError(f"Covariate record key mismatch for {key!r}.")
        env_id = str(record.get("env_id", ""))
        if env_id not in COVARIATE_NAMES:
            raise EpisodeCovariateError(f"Unsupported environment {env_id!r} in {key}.")
        covariates = record.get("covariates")
        if not isinstance(covariates, dict) or set(covariates) != set(COVARIATE_NAMES[env_id]):
            raise EpisodeCovariateError(f"Wrong covariate fields for {key}.")
        for name, value in covariates.items():
            if name == "cross_room":
                if not isinstance(value, bool):
                    raise EpisodeCovariateError(f"{key} cross_room must be boolean.")
                continue
            if value is None:
                expected_flag = (
                    record.get("audit", {}).get("tworoom_tortuosity_degenerate")
                    if name == "expert_path_tortuosity"
                    else record.get("audit", {}).get("reacher_tortuosity_degenerate")
                )
                if expected_flag is not True:
                    raise EpisodeCovariateError(f"{key} has an unexplained null {name}.")
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise EpisodeCovariateError(f"{key} has invalid {name}={value!r}.")
            if name.endswith("rotation_rad") or "wrapped_joint" in name:
                if numeric > math.pi + 1e-7:
                    raise EpisodeCovariateError(f"{key} has out-of-range angular covariate {name}={numeric}.")
            if "tortuosity" in name and numeric < 1.0 - 1e-7:
                raise EpisodeCovariateError(f"{key} has tortuosity below one: {numeric}.")
        audit = record.get("audit")
        if not isinstance(audit, dict):
            raise EpisodeCovariateError(f"{key} audit must be a mapping.")
        if env_id == "swm/ReacherDMControl-v0":
            required_audit = {
                "max_raw_joint_displacement_rad",
                "reacher_angle_branch_cut",
                "reacher_tortuosity_degenerate",
            }
            if set(audit) != required_audit:
                raise EpisodeCovariateError(f"Wrong Reacher audit fields for {key}.")
            if not isinstance(audit["reacher_angle_branch_cut"], bool) or not isinstance(
                audit["reacher_tortuosity_degenerate"], bool
            ):
                raise EpisodeCovariateError(f"Malformed Reacher flags for {key}.")
        elif env_id == "swm/TwoRoom-v1":
            if set(audit) != {"tworoom_wall_axis", "tworoom_tortuosity_degenerate"}:
                raise EpisodeCovariateError(f"Wrong TwoRoom audit fields for {key}.")
            if audit["tworoom_wall_axis"] not in {"vertical", "horizontal"} or not isinstance(
                audit["tworoom_tortuosity_degenerate"], bool
            ):
                raise EpisodeCovariateError(f"Malformed TwoRoom audit fields for {key}.")
        elif audit:
            raise EpisodeCovariateError(f"Unexpected audit fields for {key}: {sorted(audit)}.")
        by_config[canonical_config].append(record)
        env_goals[(env_id, int(record["goal_offset"]))] += 1

    if len(by_config) != EXPECTED_MATRICES:
        raise EpisodeCovariateError(f"Expected {EXPECTED_MATRICES} canonical configs, found {len(by_config)}.")
    for canonical_config, matrix_records in by_config.items():
        indices = sorted(int(record["episode_index"]) for record in matrix_records)
        if indices != list(range(EPISODES_PER_MATRIX)):
            raise EpisodeCovariateError(f"Noncanonical episode indices for {canonical_config}.")
        invariants = {
            (
                str(record["env_id"]),
                int(record["goal_offset"]),
                int(record["effective_goal_offset"]),
                str(record["canonical_config_sha256"]),
                str(record["canonical_manifest_sha256"]),
                str(record["dataset_path"]),
            )
            for record in matrix_records
        }
        if len(invariants) != 1:
            raise EpisodeCovariateError(f"Matrix-level provenance changes within {canonical_config}.")
    expected_env_goals = {
        (env_id, goal): EPISODES_PER_MATRIX
        for env_id in COVARIATE_NAMES
        for goal in (25, 50)
    }
    if dict(env_goals) != expected_env_goals:
        raise EpisodeCovariateError(
            f"Expected one 50-episode goal-25 and goal-50 matrix per environment, got {dict(env_goals)}."
        )
    return {
        "status": "passed",
        "rows": len(records),
        "matrices": len(by_config),
        "episodes_per_matrix": EPISODES_PER_MATRIX,
    }


def _csv_text(payload: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    records = sorted(
        payload["records"].values(),
        key=lambda row: (str(row["canonical_config"]), int(row["episode_index"])),
    )
    for record in records:
        flat = {name: record.get(name) for name in CSV_COLUMNS}
        flat.update(record.get("covariates", {}))
        flat.update(record.get("audit", {}))
        writer.writerow(flat)
    return output.getvalue()


def write_episode_covariates(
    payload: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    validate_covariate_payload(payload)
    json_text = json.dumps(jsonable(dict(payload)), indent=2, sort_keys=True) + "\n"
    _atomic_write_text(Path(json_path), json_text)
    _atomic_write_text(Path(csv_path), _csv_text(payload))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract pre-outcome episode covariates for the five-anchor diagnostic."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV_PATH))
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    payload = extract_episode_covariates(repo_root=root, plan_path=args.plan)
    json_path = _resolve(root, args.json_output)
    csv_path = _resolve(root, args.csv_output)
    write_episode_covariates(payload, json_path=json_path, csv_path=csv_path)
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": len(payload["records"]),
                "json": str(json_path),
                "csv": str(csv_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COVARIATE_NAMES",
    "EPISODES_PER_MATRIX",
    "EpisodeCovariateError",
    "SCHEMA_VERSION",
    "compute_episode_covariates",
    "extract_episode_covariates",
    "infer_tworoom_wall_axis",
    "quaternion_geodesic_rad",
    "validate_covariate_payload",
    "wrap_angle_delta",
    "write_episode_covariates",
]
