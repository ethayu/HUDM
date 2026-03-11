from __future__ import annotations

from typing import Optional

import plan as single_plan

from hudm.config import resolve_plan_spec
from hudm.specs import PlanSpec


def load_plan_spec(cfg_path: str) -> PlanSpec:
    return resolve_plan_spec(cfg_path)


def run_plan_spec(
    spec_or_path: str | PlanSpec,
    *,
    rollout_selection: Optional[dict] = None,
    schedule_name: Optional[str] = None,
    print_summary: bool = True,
) -> dict:
    spec = load_plan_spec(spec_or_path) if isinstance(spec_or_path, str) else spec_or_path
    return single_plan.run_plan_session(
        spec.runtime_cfg,
        rollout_selection=rollout_selection,
        schedule_name=schedule_name,
        print_summary=print_summary,
    )


def save_plan_result(result: dict, run_dir: str, *, save_media: bool = True) -> dict:
    return single_plan.save_plan_result(result, run_dir, save_media=save_media)
