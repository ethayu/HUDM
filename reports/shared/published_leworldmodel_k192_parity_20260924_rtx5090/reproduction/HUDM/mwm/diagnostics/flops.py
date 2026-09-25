from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")

FLOP_ACCOUNTING_NONE = "none"
FLOP_ACCOUNTING_DYNAMICS_AUDIT = "dynamics_audit"


def normalize_flop_accounting(value: object) -> str:
    mode = str(value or FLOP_ACCOUNTING_NONE).strip().lower()
    if mode in {"", "0", "false", "off", "disabled", "disable", "none"}:
        return FLOP_ACCOUNTING_NONE
    if mode in {"dynamics", "dynamics_audit", "profile_dynamics"}:
        return FLOP_ACCOUNTING_DYNAMICS_AUDIT
    raise ValueError(
        f"Unknown flop_accounting mode {value!r}; use 'none' or 'dynamics_audit'."
    )


def decision_flop_accounting(decision: object) -> str:
    metadata = getattr(decision, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return FLOP_ACCOUNTING_NONE
    return normalize_flop_accounting(metadata.get("flop_accounting", FLOP_ACCOUNTING_NONE))


def profile_dynamics_call(fn: Callable[[], T], *, enabled: bool) -> tuple[T, int, str | None]:
    if not enabled:
        return fn(), 0, None
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:  # pragma: no cover - depends on torch build
        return fn(), 0, f"{type(exc).__name__}: {exc}"
    try:
        with FlopCounterMode(display=False) as counter:
            out = fn()
        return out, int(counter.get_total_flops()), None
    except Exception as exc:  # pragma: no cover - profiler fallback guard
        return fn(), 0, f"{type(exc).__name__}: {exc}"


__all__ = [
    "FLOP_ACCOUNTING_DYNAMICS_AUDIT",
    "FLOP_ACCOUNTING_NONE",
    "decision_flop_accounting",
    "normalize_flop_accounting",
    "profile_dynamics_call",
]
