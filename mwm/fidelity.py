from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


LEGACY_SCHEDULER_KEYS = {
    "policy",
    "level",
    "base_level",
    "start_level",
    "end_level",
    "rollout_level",
    "rollout_levels",
    "table",
    "steps",
}


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
    """Adapter-agnostic scheduler for MPC, CEM, and rollout fidelity."""

    def __init__(
        self,
        *,
        num_levels: int,
        horizon: int,
        enabled: bool = True,
        mpc: Mapping[str, Any] | None = None,
        cem: Mapping[str, Any] | None = None,
        rollout: Mapping[str, Any] | None = None,
    ) -> None:
        self.num_levels = int(num_levels)
        self.horizon = int(horizon)
        self.enabled = bool(enabled)
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be > 0, got {self.num_levels}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")
        self.mpc_cfg = self._stage_cfg(mpc, "planner.scheduler.mpc", default={"mode": "fixed", "level": "finest"})
        self.cem_cfg = self._stage_cfg(cem, "planner.scheduler.cem", default={"mode": "fixed", "level": "base"})
        self.rollout_cfg = self._stage_cfg(
            rollout,
            "planner.scheduler.rollout",
            default={"mode": "fixed", "level": "base"},
        )

    @classmethod
    def from_config(
        cls,
        cfg: "FidelityScheduler | Mapping[str, Any] | None",
        *,
        num_levels: int,
        horizon: int,
    ) -> "FidelityScheduler":
        if isinstance(cfg, FidelityScheduler):
            return cfg
        raw = dict(cfg or {})
        legacy = LEGACY_SCHEDULER_KEYS & set(raw)
        if legacy:
            keys = ", ".join(sorted(legacy))
            raise ValueError(
                "legacy planner.scheduler schema is no longer supported. "
                "Use nested planner.scheduler.{mpc,cem,rollout}; legacy keys: "
                f"{keys}."
            )
        allowed = {"enabled", "mpc", "cem", "rollout"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown planner.scheduler keys: {sorted(unknown)}")
        return cls(
            num_levels=num_levels,
            horizon=horizon,
            enabled=bool(raw.get("enabled", True)),
            mpc=raw.get("mpc"),
            cem=raw.get("cem"),
            rollout=raw.get("rollout"),
        )

    def resolve_level(
        self,
        value: Any,
        *,
        base_level_idx: int | None = None,
        field_name: str = "level",
        allow_base: bool = True,
    ) -> int:
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
                if not allow_base:
                    raise ValueError(f"{field_name} cannot use base; mpc has no prior base level.")
                if base_level_idx is None:
                    raise ValueError(f"{field_name}={value!r} requires a base level.")
                return self._validated_level(int(base_level_idx), field_name)
            try:
                return self._validated_level(int(token), field_name)
            except ValueError as exc:
                raise ValueError(f"{field_name} has unsupported level token {value!r}.") from exc
        raise ValueError(f"{field_name} has unsupported type {type(value).__name__}.")

    def _validated_level(self, idx: int, field_name: str) -> int:
        if int(idx) < 0 or int(idx) >= self.num_levels:
            raise ValueError(f"{field_name}={idx} is outside [0, {self.num_levels - 1}]")
        return int(idx)

    def _stage_cfg(self, raw: Mapping[str, Any] | None, field_name: str, *, default: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(default if raw is None else raw)
        unknown = set(cfg) - {"mode", "level", "start_level", "end_level"}
        if unknown:
            raise ValueError(f"Unknown {field_name} keys: {sorted(unknown)}")
        mode = str(cfg.get("mode", "fixed")).lower()
        if mode not in {"fixed", "linear"}:
            raise ValueError(f"{field_name}.mode must be fixed or linear, got {mode!r}.")
        cfg["mode"] = mode
        return cfg

    def _stage_level_index(
        self,
        cfg: Mapping[str, Any],
        *,
        stage_name: str,
        progress: float,
        base_level_idx: int | None,
    ) -> int:
        mode = str(cfg.get("mode", "fixed")).lower()
        allow_base = base_level_idx is not None
        if mode == "fixed":
            default_level = "finest" if stage_name == "mpc" else "base"
            return self.resolve_level(
                cfg.get("level", default_level),
                base_level_idx=base_level_idx,
                field_name=f"planner.scheduler.{stage_name}.level",
                allow_base=allow_base,
            )
        if stage_name == "mpc":
            start_default, end_default = "coarsest", "finest"
        elif stage_name == "cem":
            start_default, end_default = "base", "finest"
        else:
            start_default, end_default = "base", "coarsest"
        start = self.resolve_level(
            cfg.get("start_level", start_default),
            base_level_idx=base_level_idx,
            field_name=f"planner.scheduler.{stage_name}.start_level",
            allow_base=allow_base,
        )
        end = self.resolve_level(
            cfg.get("end_level", end_default),
            base_level_idx=base_level_idx,
            field_name=f"planner.scheduler.{stage_name}.end_level",
            allow_base=allow_base,
        )
        return self._validated_level(_interp_level(start, end, progress), f"planner.scheduler.{stage_name}.level")

    def _rollout_levels(self, base_level_idx: int) -> list[int]:
        mode = str(self.rollout_cfg.get("mode", "fixed")).lower()
        if mode == "fixed":
            idx = self.resolve_level(
                self.rollout_cfg.get("level", "base"),
                base_level_idx=base_level_idx,
                field_name="planner.scheduler.rollout.level",
            )
            return self._validate_rollout([idx] * self.horizon, base_level_idx)
        start = self.resolve_level(
            self.rollout_cfg.get("start_level", "base"),
            base_level_idx=base_level_idx,
            field_name="planner.scheduler.rollout.start_level",
        )
        end = self.resolve_level(
            self.rollout_cfg.get("end_level", "coarsest"),
            base_level_idx=base_level_idx,
            field_name="planner.scheduler.rollout.end_level",
        )
        if self.horizon == 1:
            levels = [start]
        else:
            levels = [_interp_level(start, end, step / (self.horizon - 1)) for step in range(self.horizon)]
        return self._validate_rollout(levels, base_level_idx)

    def _validate_rollout(self, levels: list[int], base_level_idx: int) -> list[int]:
        out = [self._validated_level(int(level), "planner.scheduler.rollout") for level in levels]
        if len(out) != self.horizon:
            raise ValueError(f"planner.scheduler.rollout must have horizon={self.horizon} entries, got {len(out)}.")
        for idx in out:
            if idx > int(base_level_idx):
                raise ValueError(
                    f"planner.scheduler.rollout cannot use level {idx} finer than base level {base_level_idx}."
                )
        for prev, cur in zip(out, out[1:]):
            if cur > prev:
                raise ValueError(
                    "planner.scheduler.rollout cannot move from lower to higher fidelity within one rollout: "
                    f"{out}"
                )
        return out

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
        mpc_progress = _clamp_progress(float(mpc_progress))
        if self.enabled:
            mpc_idx = self._stage_level_index(
                self.mpc_cfg,
                stage_name="mpc",
                progress=mpc_progress,
                base_level_idx=None,
            )
            base_idx = self._stage_level_index(
                self.cem_cfg,
                stage_name="cem",
                progress=cem_progress,
                base_level_idx=mpc_idx,
            )
            rollout = self._rollout_levels(base_idx)
        else:
            mpc_idx = self.num_levels - 1
            base_idx = self.num_levels - 1
            rollout = [base_idx] * self.horizon
        return FidelityDecision(
            base_level_idx=int(base_idx),
            rollout_level_indices=[int(x) for x in rollout],
            cem_progress=float(cem_progress),
            mpc_progress=float(mpc_progress),
            metadata={
                "enabled": bool(self.enabled),
                "mpc_level_idx": int(mpc_idx),
                "base_level_idx": int(base_idx),
                "terminal_level_idx": int(rollout[-1]),
                "mpc_mode": str(self.mpc_cfg.get("mode", "fixed")),
                "cem_mode": str(self.cem_cfg.get("mode", "fixed")),
                "rollout_mode": str(self.rollout_cfg.get("mode", "fixed")),
            },
        )


__all__ = ["FidelityDecision", "FidelityScheduler", "LEGACY_SCHEDULER_KEYS"]
