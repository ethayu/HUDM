from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


SUCCESS_POS_THRESHOLD = 10.0
SUCCESS_ANGLE_THRESHOLD = math.pi / 9.0


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

