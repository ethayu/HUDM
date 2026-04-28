from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


@dataclass(frozen=True)
class RestoreSpec:
    spec_id: str
    env_ids: tuple[str, ...] = ()
    env_prefixes: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    needs_restore_recorder: bool = False
    eval_callables: tuple[dict[str, Any], ...] = ()

    def matches(self, env_id: str) -> bool:
        return env_id in self.env_ids or any(env_id.startswith(prefix) for prefix in self.env_prefixes)

    def validate_columns(self, columns: set[str]) -> None:
        missing = sorted(col for col in self.required_columns if col not in columns)
        if missing:
            raise ValueError(
                f"Restore spec {self.spec_id!r} requires dataset columns {missing}. "
                "Collect the dataset with the matching SWM restore wrapper or provide a restore adapter."
            )


RESTORE_SPECS: tuple[RestoreSpec, ...] = (
    RestoreSpec(
        spec_id="pusht_state_goal_state",
        env_ids=("swm/PushT-v1",),
        required_columns=("state", "goal_state"),
        eval_callables=(
            {"method": "_set_state", "args": {"state": {"value": "state", "in_dataset": True}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state", "in_dataset": True}}},
        ),
    ),
    RestoreSpec(
        spec_id="dmcontrol_qpos_qvel",
        env_ids=(
            "swm/HumanoidDMControl-v0",
            "swm/CheetahDMControl-v0",
            "swm/HopperDMControl-v0",
            "swm/ReacherDMControl-v0",
            "swm/WalkerDMControl-v0",
            "swm/AcrobotDMControl-v0",
            "swm/PendulumDMControl-v0",
            "swm/CartpoleDMControl-v0",
            "swm/BallInCupDMControl-v0",
            "swm/FingerDMControl-v0",
            "swm/ManipulatorDMControl-v0",
            "swm/QuadrupedDMControl-v0",
        ),
        required_columns=("qpos", "qvel"),
        eval_callables=(
            {
                "method": "set_state",
                "args": {
                    "qpos": {"value": "qpos", "in_dataset": True},
                    "qvel": {"value": "qvel", "in_dataset": True},
                },
            },
        ),
    ),
    RestoreSpec(
        spec_id="point_state_goal_state",
        env_ids=("swm/TwoRoom-v1", "swm/Piecewise-v0"),
        required_columns=("state", "goal_state"),
        eval_callables=(
            {"method": "_set_state", "args": {"state": {"value": "state", "in_dataset": True}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state", "in_dataset": True}}},
        ),
    ),
    RestoreSpec(
        spec_id="ogbench_restore_state",
        env_ids=("swm/OGBCube-v0", "swm/OGBScene-v0", "swm/OGBPointMaze-v0", "swm/OGBMaze-v0"),
        required_columns=("restore_state",),
        needs_restore_recorder=True,
        eval_callables=(
            {
                "method": "set_restore_state",
                "args": {"restore_state": {"value": "restore_state", "in_dataset": True}},
            },
        ),
    ),
)


def _import_object(path: str) -> Any:
    module_name, sep, attr = str(path).partition(":")
    if not sep:
        module_name, sep, attr = str(path).rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import path must be 'module:attr' or 'module.attr', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _normalize_user_spec(raw: RestoreSpec | dict[str, Any], env_id: str) -> RestoreSpec:
    if isinstance(raw, RestoreSpec):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Restore adapter for {env_id} must return RestoreSpec or dict, got {type(raw).__name__}")
    return RestoreSpec(
        spec_id=str(raw.get("spec_id", f"user:{env_id}")),
        env_ids=tuple(raw.get("env_ids", (env_id,))),
        env_prefixes=tuple(raw.get("env_prefixes", ())),
        required_columns=tuple(raw.get("required_columns", ())),
        needs_restore_recorder=bool(raw.get("needs_restore_recorder", False)),
        eval_callables=tuple(raw.get("eval_callables", raw.get("callables", ()))),
    )


def user_restore_spec(import_path: str, env_id: str, columns: list[str] | set[str]) -> RestoreSpec:
    builder = _import_object(import_path)
    if not callable(builder):
        raise TypeError(f"Restore adapter {import_path!r} is not callable.")
    cols = tuple(sorted(str(c) for c in columns))
    try:
        raw = builder(env_id=env_id, columns=cols)
    except TypeError:
        try:
            raw = builder(env_id, cols)
        except TypeError:
            raw = builder(cols)
    spec = _normalize_user_spec(raw, env_id)
    spec.validate_columns(set(cols))
    return spec


def restore_spec_for_env(
    env_id: str,
    import_path: str | None = None,
    columns: list[str] | set[str] | None = None,
) -> RestoreSpec:
    if import_path:
        if columns is None:
            raise ValueError("User restore adapters require dataset columns for validation.")
        return user_restore_spec(import_path, env_id, columns)
    for spec in RESTORE_SPECS:
        if spec.matches(env_id):
            return spec
    supported = ", ".join(sorted(e for spec in RESTORE_SPECS for e in spec.env_ids))
    raise ValueError(
        f"SWM env {env_id!r} has no built-in HUDM restore adapter. "
        f"Supported env ids: {supported}. Provide restore.import_path for arbitrary restorable SWM envs."
    )


def validate_restore_columns(
    env_id: str,
    columns: list[str] | set[str],
    import_path: str | None = None,
) -> RestoreSpec:
    spec = restore_spec_for_env(env_id, import_path=import_path, columns=columns if import_path else None)
    spec.validate_columns(set(columns))
    return spec


def eval_callables_for_env(
    env_id: str,
    columns: list[str] | set[str],
    import_path: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    spec = validate_restore_columns(env_id, columns, import_path=import_path)
    return spec.spec_id, [dict(callable_spec) for callable_spec in spec.eval_callables]
