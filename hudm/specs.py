from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping-like config, got {type(value).__name__}")


def _stable_signature(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PlanSpec:
    name: str
    config_path: str | None
    task: dict[str, Any]
    budget: dict[str, Any]
    planner: dict[str, Any]
    backend: dict[str, Any]
    artifacts: dict[str, Any]
    clean_cfg: DictConfig
    runtime_cfg: DictConfig

    def active_backend_kind(self) -> str:
        return str(self.backend["kind"]).lower()

    def active_backend_cfg(self) -> dict[str, Any]:
        return dict(self.backend[self.active_backend_kind()])

    def task_signature(self) -> str:
        return _stable_signature(self.task)

    def rollout_signature(self) -> str:
        init_goal = copy.deepcopy(dict(self.task.get("init_goal", {})))
        return _stable_signature(init_goal)

    def budget_signature(self) -> str:
        return _stable_signature(self.budget)

    def variant_compatibility_signature(self) -> str:
        payload = {
            "task": self.task,
            "budget": self.budget,
            "planner": {
                k: v for k, v in self.planner.items() if k != "fidelity"
            },
            "backend": self.backend,
        }
        kind = self.active_backend_kind()
        payload["backend"][kind] = {
            k: v
            for k, v in self.backend[kind].items()
            if k != "fidelity"
        }
        return _stable_signature(payload)


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    plan: PlanSpec


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config_path: str | None
    shared_plan: PlanSpec
    variants: list[ExperimentVariant]
    baseline: str
    rollouts: dict[str, Any]
    execution: dict[str, Any]
    terminal: dict[str, Any]
    reporting: dict[str, Any]

    def variant_names(self) -> list[str]:
        return [variant.name for variant in self.variants]

    def baseline_variant(self) -> ExperimentVariant:
        for variant in self.variants:
            if variant.name == self.baseline:
                return variant
        raise KeyError(f"Unknown baseline variant: {self.baseline}")


@dataclass(frozen=True)
class BenchmarkEntry:
    name: str
    experiment_config: str


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    config_path: str | None
    entries: list[BenchmarkEntry]
    output_root: str


def make_plan_spec(
    *,
    name: str,
    config_path: str | None,
    clean_cfg: DictConfig,
    runtime_cfg: DictConfig,
) -> PlanSpec:
    return PlanSpec(
        name=name,
        config_path=config_path,
        task=_to_plain_dict(clean_cfg.task),
        budget=_to_plain_dict(clean_cfg.budget),
        planner=_to_plain_dict(clean_cfg.planner),
        backend=_to_plain_dict(clean_cfg.backend),
        artifacts=_to_plain_dict(clean_cfg.artifacts),
        clean_cfg=clean_cfg,
        runtime_cfg=runtime_cfg,
    )
