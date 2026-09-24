"""Shape-only dynamics-FLOP calibration for scheduled CEM runs.

For a checkpoint + horizon, measure audited dynamics FLOPs per candidate for a
rollout level sequence (batch=1, samples=1; FlopCounterMode counts are linear
in samples). Then reconstruct a run's dynamics_flops_total as
  sum_L count(base level L) * pop_size * f(rollout_levels(L)).
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mwm.checkpoint_io import load_world_model_from_checkpoint  # noqa: E402
from mwm.diagnostics.flops import FLOP_ACCOUNTING_DYNAMICS_AUDIT  # noqa: E402
from mwm.fidelity import FidelityScheduler  # noqa: E402


class Calib:
    def __init__(self, ckpt: str, horizon: int, history: int = 1):
        self.model, _, _ = load_world_model_from_checkpoint(ROOT / ckpt, None, torch.device("cpu"))
        self.horizon, self.history = horizon, history
        self.K = [int(k) for k in self.model.K]

    @functools.lru_cache(maxsize=None)
    def per_candidate(self, levels: tuple[int, ...]) -> int:
        m = self.model
        infos = {
            "pixels": torch.zeros(1, 1, self.history, 1),
            "emb": torch.zeros(1, 1, self.history, int(m.D)),
        }
        actions = torch.zeros(1, 1, self.horizon, int(m.action_dim))
        with torch.no_grad():
            out = m.rollout_with_schedule(infos, actions, list(levels), flop_accounting=FLOP_ACCOUNTING_DYNAMICS_AUDIT)
        assert "_mwm_flop_audit_error" not in out, out.get("_mwm_flop_audit_error")
        return int(out["_mwm_dynamics_flops"])

    def scheduler(self, sched_cfg: dict) -> FidelityScheduler:
        return FidelityScheduler.from_config(
            sched_cfg, num_levels=len(self.K), horizon=self.horizon, levels=tuple(self.K),
            min_k=int(getattr(self.model, "min_k", 1)), max_k=int(self.model.D),
            supports_arbitrary_k=bool(getattr(self.model, "supports_arbitrary_k", False)),
        )

    def run_flops(self, run: dict, sched_cfg: dict) -> int:
        sch = self.scheduler(sched_cfg)
        counts = json.loads(run["schedule_level_counts"])
        total = 0
        for lvl, n in counts.items():
            seq = tuple(sch._rollout_levels(int(lvl)))
            total += int(n) * int(run["pop_size"]) * self.per_candidate(seq)
        return total
