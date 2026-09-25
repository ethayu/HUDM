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
    base_level_idx: int | None
    rollout_level_indices: list[int | None]
    cem_progress: float = 0.0
    mpc_progress: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    base_k: int | None = None
    rollout_ks: list[int] | None = None

    @property
    def level_idx(self) -> int:
        if self.base_level_idx is None:
            raise ValueError("This fidelity decision uses a non-anchor K and has no level index.")
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
        fidelity_unit: str = "level",
        levels: list[int] | tuple[int, ...] | None = None,
        min_k: int = 1,
        max_k: int | None = None,
        supports_arbitrary_k: bool = False,
        selectable_ks: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.num_levels = int(num_levels)
        self.horizon = int(horizon)
        self.enabled = bool(enabled)
        self.fidelity_unit = str(fidelity_unit).strip().lower()
        if self.fidelity_unit not in {"level", "k"}:
            raise ValueError(f"planner.scheduler.fidelity_unit must be 'level' or 'k', got {fidelity_unit!r}.")
        self.levels = [int(k) for k in levels] if levels is not None else None
        if self.levels is not None and len(self.levels) != self.num_levels:
            raise ValueError(f"Scheduler received {len(self.levels)} K values for {self.num_levels} levels.")
        self.min_k = int(min_k)
        self.max_k = int(max_k) if max_k is not None else (max(self.levels) if self.levels else None)
        self.supports_arbitrary_k = bool(supports_arbitrary_k)
        self.selectable_ks = [int(k) for k in selectable_ks] if selectable_ks is not None else None
        if self.fidelity_unit == "k":
            if not self.supports_arbitrary_k:
                raise ValueError("planner.scheduler.fidelity_unit=k requires a model with arbitrary-K dynamics.")
            if self.levels is None or self.max_k is None:
                raise ValueError("planner.scheduler.fidelity_unit=k requires model K anchors and a maximum K.")
            if self.min_k <= 0 or self.min_k > self.max_k:
                raise ValueError(f"Invalid scheduler K range [{self.min_k}, {self.max_k}].")
            if self.selectable_ks is not None:
                if not self.selectable_ks:
                    raise ValueError("planner K selection cannot be empty.")
                if self.selectable_ks != sorted(set(self.selectable_ks)):
                    raise ValueError(f"planner K values must be sorted and unique, got {self.selectable_ks}.")
                if self.selectable_ks[0] < self.min_k or self.selectable_ks[-1] > self.max_k:
                    raise ValueError(
                        f"planner K values {self.selectable_ks} are outside the model-supported "
                        f"range [{self.min_k}, {self.max_k}]."
                    )
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be > 0, got {self.num_levels}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")
        selector = "k" if self.fidelity_unit == "k" else "level"
        self.mpc_cfg = self._stage_cfg(
            mpc,
            "planner.scheduler.mpc",
            default={"mode": "fixed", selector: "finest"},
        )
        self.cem_cfg = self._stage_cfg(
            cem,
            "planner.scheduler.cem",
            default={"mode": "fixed", selector: "base"},
        )
        self.rollout_cfg = self._stage_cfg(
            rollout,
            "planner.scheduler.rollout",
            default={"mode": "fixed", selector: "base"},
        )

    @classmethod
    def from_config(
        cls,
        cfg: "FidelityScheduler | Mapping[str, Any] | None",
        *,
        num_levels: int,
        horizon: int,
        levels: list[int] | tuple[int, ...] | None = None,
        min_k: int = 1,
        max_k: int | None = None,
        supports_arbitrary_k: bool = False,
        selectable_ks: list[int] | tuple[int, ...] | None = None,
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
        allowed = {"enabled", "fidelity_unit", "mpc", "cem", "rollout"}
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
            fidelity_unit=str(raw.get("fidelity_unit", "level")),
            levels=levels,
            min_k=int(min_k),
            max_k=max_k,
            supports_arbitrary_k=bool(supports_arbitrary_k),
            selectable_ks=selectable_ks,
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

    def resolve_k(
        self,
        value: Any,
        *,
        base_k: int | None = None,
        field_name: str = "k",
        allow_base: bool = True,
    ) -> int:
        if value is None:
            raise ValueError(f"{field_name} must be set.")
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be an integer or token, got bool.")
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"coarsest", "min", "lowest"}:
                selected_min = self.selectable_ks[0] if self.selectable_ks is not None else self.min_k
                return self._validated_k(selected_min, field_name)
            if token in {"finest", "max", "highest"}:
                selected_max = self.selectable_ks[-1] if self.selectable_ks is not None else self.max_k
                return self._validated_k(selected_max, field_name)
            if token in {"base", "auto"}:
                if not allow_base or base_k is None:
                    raise ValueError(f"{field_name}={value!r} requires a base K.")
                return self._validated_k(base_k, field_name)
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"{field_name} has unsupported K token {value!r}.") from exc
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{field_name} must be integer-valued, got {value}.")
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"{field_name} has unsupported type {type(value).__name__}.")
        return self._validated_k(value, field_name)

    def _validated_k(self, k: int | None, field_name: str) -> int:
        if k is None or self.max_k is None:
            raise ValueError(f"{field_name} cannot be resolved without a model K range.")
        value = int(k)
        if value < self.min_k or value > self.max_k:
            raise ValueError(f"{field_name}={value} is outside [{self.min_k}, {self.max_k}].")
        if self.selectable_ks is not None and value not in self.selectable_ks:
            raise ValueError(
                f"{field_name}={value} is not in configured K values {self.selectable_ks}."
            )
        return value

    def _interp_k(self, start: int, end: int, progress: float) -> int:
        if self.selectable_ks is None:
            return _interp_level(start, end, progress)
        start_idx = self.selectable_ks.index(int(start))
        end_idx = self.selectable_ks.index(int(end))
        return self.selectable_ks[_interp_level(start_idx, end_idx, progress)]

    def _anchor_index(self, k: int) -> int | None:
        if self.levels is None or int(k) not in self.levels:
            return None
        return self.levels.index(int(k))

    def _stage_cfg(self, raw: Mapping[str, Any] | None, field_name: str, *, default: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(default if raw is None else raw)
        level_keys = {"level", "start_level", "end_level"}
        k_keys = {"k", "start_k", "end_k"}
        unknown = set(cfg) - {"mode"} - level_keys - k_keys
        if unknown:
            raise ValueError(f"Unknown {field_name} keys: {sorted(unknown)}")
        mode = str(cfg.get("mode", "fixed")).lower()
        if mode not in {"fixed", "linear"}:
            raise ValueError(f"{field_name}.mode must be fixed or linear, got {mode!r}.")
        cfg["mode"] = mode
        forbidden = k_keys if self.fidelity_unit == "level" else level_keys
        present_forbidden = sorted(set(cfg) & forbidden)
        if present_forbidden:
            raise ValueError(
                f"{field_name} uses fidelity_unit={self.fidelity_unit!r} but defines {present_forbidden}."
            )
        selector = "k" if self.fidelity_unit == "k" else "level"
        start_selector = f"start_{selector}"
        end_selector = f"end_{selector}"
        if mode == "fixed" and ({start_selector, end_selector} & set(cfg)):
            raise ValueError(
                f"{field_name}.mode='fixed' uses {selector}; remove {start_selector}/{end_selector}."
            )
        if mode == "linear" and selector in cfg:
            raise ValueError(
                f"{field_name}.mode='linear' uses {start_selector}/{end_selector}; remove {selector}."
            )
        return cfg

    def _stage_k(
        self,
        cfg: Mapping[str, Any],
        *,
        stage_name: str,
        progress: float,
        base_k: int | None,
    ) -> int:
        mode = str(cfg.get("mode", "fixed")).lower()
        allow_base = base_k is not None
        if mode == "fixed":
            default_k = "finest" if stage_name == "mpc" else "base"
            return self.resolve_k(
                cfg.get("k", default_k),
                base_k=base_k,
                field_name=f"planner.scheduler.{stage_name}.k",
                allow_base=allow_base,
            )
        if stage_name == "mpc":
            start_default, end_default = "coarsest", "finest"
        elif stage_name == "cem":
            start_default, end_default = "base", "finest"
        else:
            start_default, end_default = "base", "coarsest"
        start = self.resolve_k(
            cfg.get("start_k", start_default),
            base_k=base_k,
            field_name=f"planner.scheduler.{stage_name}.start_k",
            allow_base=allow_base,
        )
        end = self.resolve_k(
            cfg.get("end_k", end_default),
            base_k=base_k,
            field_name=f"planner.scheduler.{stage_name}.end_k",
            allow_base=allow_base,
        )
        return self._validated_k(self._interp_k(start, end, progress), f"planner.scheduler.{stage_name}.k")

    def _rollout_ks(self, base_k: int) -> list[int]:
        mode = str(self.rollout_cfg.get("mode", "fixed")).lower()
        if mode == "fixed":
            k = self.resolve_k(
                self.rollout_cfg.get("k", "base"),
                base_k=base_k,
                field_name="planner.scheduler.rollout.k",
            )
            values = [k] * self.horizon
        else:
            start = self.resolve_k(
                self.rollout_cfg.get("start_k", "base"),
                base_k=base_k,
                field_name="planner.scheduler.rollout.start_k",
            )
            end = self.resolve_k(
                self.rollout_cfg.get("end_k", "coarsest"),
                base_k=base_k,
                field_name="planner.scheduler.rollout.end_k",
            )
            values = (
                [start]
                if self.horizon == 1
                else [self._interp_k(start, end, step / (self.horizon - 1)) for step in range(self.horizon)]
            )
        out = [self._validated_k(value, "planner.scheduler.rollout.k") for value in values]
        for previous, current in zip(out, out[1:]):
            if current > previous:
                raise ValueError(f"rollout cannot increase K within one rollout: {out}.")
        return out

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
        if self.fidelity_unit == "k":
            if self.enabled:
                mpc_k = self._stage_k(
                    self.mpc_cfg,
                    stage_name="mpc",
                    progress=mpc_progress,
                    base_k=None,
                )
                base_k = self._stage_k(
                    self.cem_cfg,
                    stage_name="cem",
                    progress=cem_progress,
                    base_k=mpc_k,
                )
                rollout_ks = self._rollout_ks(base_k)
            else:
                mpc_k = self._validated_k(self.max_k, "planner.scheduler.mpc.k")
                base_k = mpc_k
                rollout_ks = [base_k] * self.horizon
            mpc_idx = self._anchor_index(mpc_k)
            base_idx = self._anchor_index(base_k)
            rollout_indices = [self._anchor_index(k) for k in rollout_ks]
            return FidelityDecision(
                base_level_idx=base_idx,
                rollout_level_indices=rollout_indices,
                cem_progress=float(cem_progress),
                mpc_progress=float(mpc_progress),
                metadata={
                    "enabled": bool(self.enabled),
                    "fidelity_unit": "k",
                    "mpc_level_idx": mpc_idx,
                    "base_level_idx": base_idx,
                    "terminal_level_idx": rollout_indices[-1],
                    "mpc_k": int(mpc_k),
                    "base_k": int(base_k),
                    "terminal_k": int(rollout_ks[-1]),
                    "rollout_ks": [int(k) for k in rollout_ks],
                    "mpc_mode": str(self.mpc_cfg.get("mode", "fixed")),
                    "cem_mode": str(self.cem_cfg.get("mode", "fixed")),
                    "rollout_mode": str(self.rollout_cfg.get("mode", "fixed")),
                },
                base_k=int(base_k),
                rollout_ks=[int(k) for k in rollout_ks],
            )
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
        base_k = self.levels[base_idx] if self.levels is not None else None
        rollout_ks = [self.levels[idx] for idx in rollout] if self.levels is not None else None
        metadata = {
            "enabled": bool(self.enabled),
            "fidelity_unit": "level",
            "mpc_level_idx": int(mpc_idx),
            "base_level_idx": int(base_idx),
            "terminal_level_idx": int(rollout[-1]),
            "mpc_mode": str(self.mpc_cfg.get("mode", "fixed")),
            "cem_mode": str(self.cem_cfg.get("mode", "fixed")),
            "rollout_mode": str(self.rollout_cfg.get("mode", "fixed")),
        }
        if self.levels is not None and rollout_ks is not None:
            metadata.update(
                {
                    "mpc_k": int(self.levels[mpc_idx]),
                    "base_k": int(base_k),
                    "terminal_k": int(rollout_ks[-1]),
                    "rollout_ks": [int(k) for k in rollout_ks],
                }
            )
        return FidelityDecision(
            base_level_idx=int(base_idx),
            rollout_level_indices=[int(x) for x in rollout],
            cem_progress=float(cem_progress),
            mpc_progress=float(mpc_progress),
            metadata=metadata,
            base_k=int(base_k) if base_k is not None else None,
            rollout_ks=[int(k) for k in rollout_ks] if rollout_ks is not None else None,
        )


__all__ = ["FidelityDecision", "FidelityScheduler", "LEGACY_SCHEDULER_KEYS"]
