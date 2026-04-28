from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

try:
    from shapely import affinity
    from shapely.geometry import box
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - optional runtime dependency
    affinity = None
    box = None
    unary_union = None


SUCCESS_POS_THRESHOLD = 1.0
SUCCESS_ANGLE_THRESHOLD = math.pi / 9.0
GT_T_SCALE_PX = 40.0
GT_T_STEM_W_PX = GT_T_SCALE_PX
GT_T_STEM_H_PX = 3.0 * GT_T_SCALE_PX
GT_T_BAR_W_PX = 4.0 * GT_T_SCALE_PX
GT_T_BAR_H_PX = GT_T_SCALE_PX


def angle_delta(a: float, b: float) -> float:
    d = float(a) - float(b)
    return abs(math.atan2(math.sin(d), math.cos(d)))


def pose_metrics(
    goal_state: np.ndarray,
    cur_state: np.ndarray,
    *,
    pos_threshold: float = SUCCESS_POS_THRESHOLD,
    angle_threshold: float = SUCCESS_ANGLE_THRESHOLD,
) -> Dict[str, Any]:
    goal = np.asarray(goal_state, dtype=np.float32)
    cur = np.asarray(cur_state, dtype=np.float32)
    eef_diff = float(np.linalg.norm(goal[:2] - cur[:2]))
    pos_diff = float(np.linalg.norm(goal[2:4] - cur[2:4]))
    ang_diff = float(angle_delta(float(goal[4]), float(cur[4])))
    state_dist = float(np.linalg.norm(goal - cur))
    success = bool(pos_diff < float(pos_threshold) and ang_diff < float(angle_threshold))
    return {
        "success": success,
        "pos_diff": pos_diff,
        "angle_diff": ang_diff,
        "eef_diff": eef_diff,
        "state_dist": state_dist,
    }


def termination_success(term: dict | None) -> bool:
    if term is None:
        return False
    if "success_and_done" in term:
        return bool(term["success_and_done"])
    return bool(term.get("success", False)) and bool(term.get("done", False))


def tee_pose_polygon_px(state: np.ndarray) -> Any | None:
    if affinity is None or box is None or unary_union is None:
        return None
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.shape[0] < 5:
        return None
    tee_local = unary_union(
        [
            box(-0.5 * GT_T_BAR_W_PX, 0.0, 0.5 * GT_T_BAR_W_PX, GT_T_BAR_H_PX),
            box(
                -0.5 * GT_T_STEM_W_PX,
                GT_T_BAR_H_PX,
                0.5 * GT_T_STEM_W_PX,
                GT_T_BAR_H_PX + GT_T_STEM_H_PX,
            ),
        ]
    )
    tee_rot = affinity.rotate(tee_local, float(s[4]), origin=(0.0, 0.0), use_radians=True)
    return affinity.translate(tee_rot, xoff=float(s[2]), yoff=float(s[3]))


def tee_pose_coverage_px(goal_state: np.ndarray, cur_state: np.ndarray) -> float | None:
    goal_poly = tee_pose_polygon_px(goal_state)
    cur_poly = tee_pose_polygon_px(cur_state)
    if goal_poly is None or cur_poly is None:
        return None
    goal_area = float(goal_poly.area)
    if goal_area <= 0.0:
        return None
    return float(goal_poly.intersection(cur_poly).area / goal_area)
