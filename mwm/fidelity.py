from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp_progress(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _interp_level(start_idx: int, end_idx: int, progress: float) -> int:
    value = float(start_idx) + (float(end_idx) - float(start_idx)) * _clamp_progress(progress)
    return int(round(value))


@dataclass(frozen=True)
class FidelityDecision:
    base_level_idx: int
    rollout_level_indices: list[int]
    cem_progress: float = 0.0
    mpc_progress: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def level_idx(self) -> int:
        return int(self.base_level_idx)


class FidelityScheduler:
    """Small explicit fidelity scheduler for MWM planning and tests."""

    def __init__(self, policy: str, num_levels: int, horizon: int, **kwargs: Any) -> None:
        self.policy = str(policy).lower()
        self.num_levels = int(num_levels)
        self.horizon = int(horizon)
        self.kwargs = dict(kwargs)
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be > 0, got {self.num_levels}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")
        if self.policy not in {"fixed", "linear_cem", "table"}:
            raise ValueError(f"Unknown MWM fidelity policy {policy!r}")

    @classmethod
    def from_config(
        cls,
        cfg: "FidelityScheduler | dict[str, Any] | None",
        *,
        num_levels: int,
        horizon: int,
    ) -> "FidelityScheduler":
        if isinstance(cfg, FidelityScheduler):
            return cfg
        raw = dict(cfg or {})
        policy = raw.pop("policy", raw.pop("mode", "fixed"))
        return cls(str(policy), num_levels=num_levels, horizon=horizon, **raw)

    def resolve_level(self, value: Any, *, base_level_idx: int | None = None, field_name: str = "level") -> int:
        if value is None:
            raise ValueError(f"{field_name} must be set.")
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be an int or token, got bool.")
        if isinstance(value, int):
            return self._validated_level(value, field_name)
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{field_name} must be integer-valued, got {value}.")
            return self._validated_level(int(value), field_name)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"coarsest", "min", "lowest"}:
                return 0
            if token in {"finest", "max", "highest"}:
                return self.num_levels - 1
            if token in {"base", "auto"}:
                if base_level_idx is None:
                    raise ValueError(f"{field_name}={value!r} requires a base level.")
                return self._validated_level(int(base_level_idx), field_name)
            return self._validated_level(int(token), field_name)
        raise ValueError(f"{field_name} has unsupported type {type(value).__name__}.")

    def _validated_level(self, idx: int, field_name: str) -> int:
        if int(idx) < 0 or int(idx) >= self.num_levels:
            raise ValueError(f"{field_name}={idx} is outside [0, {self.num_levels - 1}]")
        return int(idx)

    def _fixed_decision(self, cem_progress: float, mpc_progress: float) -> FidelityDecision:
        base = self.resolve_level(self.kwargs.get("base_level", self.kwargs.get("level", "finest")), field_name="base_level")
        rollout = self._rollout_levels(base, default="base")
        return self._decision(base, rollout, cem_progress, mpc_progress)

    def _linear_cem_decision(self, cem_progress: float, mpc_progress: float) -> FidelityDecision:
        start = self.resolve_level(self.kwargs.get("start_level", "coarsest"), field_name="start_level")
        end = self.resolve_level(self.kwargs.get("end_level", "finest"), field_name="end_level")
        base = self._validated_level(_interp_level(start, end, cem_progress), "base_level")
        rollout = self._rollout_levels(base, default="base")
        return self._decision(base, rollout, cem_progress, mpc_progress)

    def _table_decision(self, cem_iter: int, cem_progress: float, mpc_progress: float) -> FidelityDecision:
        rows = list(self.kwargs.get("table", self.kwargs.get("steps", [])))
        if not rows:
            raise ValueError("table fidelity policy requires a non-empty table.")
        idx = min(max(0, int(cem_iter)), len(rows) - 1)
        row = dict(rows[idx])
        base = self.resolve_level(row.get("base_level", row.get("level", "finest")), field_name=f"table[{idx}].base_level")
        raw_rollout = row.get("rollout_levels", row.get("rollout_level", self.kwargs.get("rollout_level", "base")))
        rollout = self._coerce_rollout(raw_rollout, base, field_name=f"table[{idx}].rollout")
        return self._decision(base, rollout, cem_progress, mpc_progress, {"table_index": idx})

    def _rollout_levels(self, base_level_idx: int, *, default: Any) -> list[int]:
        if "rollout_levels" in self.kwargs:
            raw = self.kwargs["rollout_levels"]
        else:
            raw = self.kwargs.get("rollout_level", default)
        return self._coerce_rollout(raw, base_level_idx, field_name="rollout")

    def _coerce_rollout(self, raw: Any, base_level_idx: int, *, field_name: str) -> list[int]:
        if isinstance(raw, dict):
            mode = str(raw.get("mode", "fixed")).lower()
            if mode == "linear":
                start = self.resolve_level(raw.get("start_level", "base"), base_level_idx=base_level_idx, field_name=f"{field_name}.start_level")
                end = self.resolve_level(raw.get("end_level", "coarsest"), base_level_idx=base_level_idx, field_name=f"{field_name}.end_level")
                if self.horizon == 1:
                    return self._validate_rollout([start], base_level_idx, field_name)
                levels = [_interp_level(start, end, t / (self.horizon - 1)) for t in range(self.horizon)]
                return self._validate_rollout(levels, base_level_idx, field_name)
            if mode != "fixed":
                raise ValueError(f"{field_name}.mode must be fixed or linear, got {mode!r}")
            raw = raw.get("level", "base")
        if isinstance(raw, (list, tuple)):
            if len(raw) != self.horizon:
                raise ValueError(f"{field_name} must have horizon={self.horizon} entries, got {len(raw)}")
            levels = [self.resolve_level(x, base_level_idx=base_level_idx, field_name=f"{field_name}[{i}]") for i, x in enumerate(raw)]
            return self._validate_rollout(levels, base_level_idx, field_name)
        level = self.resolve_level(raw, base_level_idx=base_level_idx, field_name=field_name)
        return self._validate_rollout([level] * self.horizon, base_level_idx, field_name)

    def _validate_rollout(self, levels: list[int], base_level_idx: int, field_name: str) -> list[int]:
        out = [self._validated_level(int(x), field_name) for x in levels]
        for idx in out:
            if idx > int(base_level_idx):
                raise ValueError(f"{field_name} cannot use level {idx} finer than base level {base_level_idx}.")
        for prev, cur in zip(out, out[1:]):
            if cur > prev:
                raise ValueError(f"{field_name} cannot move from lower to higher fidelity within one rollout: {out}")
        return out

    def _decision(
        self,
        base: int,
        rollout: list[int],
        cem_progress: float,
        mpc_progress: float,
        metadata: dict[str, Any] | None = None,
    ) -> FidelityDecision:
        return FidelityDecision(
            base_level_idx=int(base),
            rollout_level_indices=[int(x) for x in rollout],
            cem_progress=float(cem_progress),
            mpc_progress=float(mpc_progress),
            metadata=dict(metadata or {}),
        )

    def decision(
        self,
        *,
        cem_iter: int,
        n_iter: int,
        mpc_progress: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> FidelityDecision:
        del context
        cem_progress = 1.0 if int(n_iter) <= 1 else int(cem_iter) / max(1, int(n_iter) - 1)
        if self.policy == "fixed":
            return self._fixed_decision(cem_progress, float(mpc_progress))
        if self.policy == "linear_cem":
            return self._linear_cem_decision(cem_progress, float(mpc_progress))
        return self._table_decision(int(cem_iter), cem_progress, float(mpc_progress))


__all__ = ["FidelityDecision", "FidelityScheduler"]
