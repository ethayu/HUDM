from __future__ import annotations

from typing import Any

from mwm.swm.restore import RestoreSpec


OGBENCH_QPOS_QVEL_COLUMNS = ("qpos", "qvel")
OGBENCH_CUBE_BLOCK0_POSE_COLUMNS = (
    ("privileged/block_0_pos", "privileged/block_0_quat"),
    ("privileged_block_0_pos", "privileged_block_0_quat"),
)


def ogbench_state_restore_callables() -> tuple[dict[str, Any], ...]:
    return (
        {
            "method": "set_state",
            "args": {
                "qpos": {"value": "qpos", "in_dataset": True},
                "qvel": {"value": "qvel", "in_dataset": True},
            },
        },
    )


def ogbench_cube_restore_spec(
    env_id: str = "swm/OGBCube-v0",
    columns: list[str] | set[str] = (),
) -> RestoreSpec:
    """Restore spec for the upstream Le-WM OGBench Cube-single task."""

    available = {str(col) for col in columns}
    block_pos, block_quat = OGBENCH_CUBE_BLOCK0_POSE_COLUMNS[0]
    for candidate_pos, candidate_quat in OGBENCH_CUBE_BLOCK0_POSE_COLUMNS:
        if {candidate_pos, candidate_quat}.issubset(available):
            block_pos, block_quat = candidate_pos, candidate_quat
            break
    return RestoreSpec(
        spec_id="ogbench_cube_single_qpos_qvel_target_pose",
        env_ids=(str(env_id),),
        required_columns=(
            *OGBENCH_QPOS_QVEL_COLUMNS,
            block_pos,
            block_quat,
        ),
        eval_callables=(
            *ogbench_state_restore_callables(),
            {
                "method": "set_target_pos",
                "args": {
                    "cube_id": {"value": 0, "in_dataset": False},
                    "target_pos": {"value": f"goal_{block_pos}", "in_dataset": True},
                    "target_quat": {"value": f"goal_{block_quat}", "in_dataset": True},
                },
            },
        ),
    )


__all__ = [
    "OGBENCH_CUBE_BLOCK0_POSE_COLUMNS",
    "OGBENCH_QPOS_QVEL_COLUMNS",
    "ogbench_cube_restore_spec",
    "ogbench_state_restore_callables",
]
