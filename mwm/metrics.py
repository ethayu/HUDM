from __future__ import annotations

from typing import Any

import numpy as np


def success_rate(successes: list[bool] | tuple[bool, ...]) -> float:
    return float(np.mean([bool(x) for x in successes]) * 100.0) if successes else 0.0


def aggregate_policy_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    total_replans = int(sum(int(x.get("replans", 0)) for x in items))
    total_actions = int(sum(int(x.get("actions_recorded", 0)) for x in items))
    total_time = float(sum(float(x.get("total_plan_time_sec", 0.0)) for x in items))
    total_bits = int(sum(int(x.get("total_bits_used_estimate", 0)) for x in items))
    return {
        "actions_recorded": total_actions,
        "replans": total_replans,
        "total_plan_time_sec": total_time,
        "mean_plan_time_sec": total_time / total_replans if total_replans else 0.0,
        "total_bits_used_estimate": total_bits,
    }


__all__ = ["aggregate_policy_diagnostics", "success_rate"]
