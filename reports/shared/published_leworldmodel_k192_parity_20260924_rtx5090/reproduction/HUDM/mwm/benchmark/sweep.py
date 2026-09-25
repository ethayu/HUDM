from __future__ import annotations

import itertools
import json
import math
from copy import deepcopy
from typing import Any

from omegaconf import OmegaConf

from mwm.benchmark.config import as_plain_dict, safe_name


_LABELS = {
    "planner.pop_size": "pop",
    "planner.elite_frac": "elite",
    "planner.n_iter": "iter",
}


def _value_token(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Sweep values must be finite, got {value!r}.")
        text = format(value, ".12g")
    else:
        text = str(value)
    return safe_name(text.replace("-", "m").replace(".", "p"))


def _set_path(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid sweep parameter path {path!r}.")
    target = mapping
    for part in parts[:-1]:
        child = target.get(part)
        if child is None:
            child = {}
            target[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot apply sweep parameter {path!r}: {part!r} is not a mapping.")
        target = child
    target[parts[-1]] = deepcopy(value)


def _sweep_axes(cfg: Any) -> list[tuple[str, list[Any]]]:
    raw = as_plain_dict(cfg.get("sweep", {}))
    axes: list[tuple[str, list[Any]]] = []
    for path, values in raw.items():
        if not isinstance(path, str) or "." not in path:
            raise ValueError(
                "Sweep keys must be dotted config paths such as "
                "`planner.pop_size`, `planner.elite_frac`, or `planner.n_iter`."
            )
        if OmegaConf.is_config(values):
            values = OmegaConf.to_container(values, resolve=True)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep parameter {path!r} must be a non-empty list.")
        axes.append((path, list(values)))
    return axes


def _sweep_exclusions(cfg: Any, sweep_paths: set[str]) -> list[dict[str, Any]]:
    raw = cfg.get("sweep_exclude", [])
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("sweep_exclude must be a list of parameter mappings.")

    exclusions: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if OmegaConf.is_config(item):
            item = OmegaConf.to_container(item, resolve=True)
        if not isinstance(item, dict) or not item:
            raise ValueError(f"sweep_exclude[{index}] must be a non-empty parameter mapping.")
        unknown = sorted(set(item) - sweep_paths)
        if unknown:
            raise ValueError(
                f"sweep_exclude[{index}] contains parameters not present in sweep: {unknown}."
            )
        exclusions.append(dict(item))
    return exclusions


def _is_excluded(params: dict[str, Any], exclusions: list[dict[str, Any]]) -> bool:
    return any(all(params[path] == value for path, value in exclusion.items()) for exclusion in exclusions)


def sweep_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def expand_benchmark_runs(cfg: Any) -> list[Any]:
    """Expand each base run over the same Cartesian parameter grid."""

    base_runs = list(cfg.get("runs", []))
    axes = _sweep_axes(cfg)
    if not axes:
        return [OmegaConf.create(OmegaConf.to_container(run, resolve=True)) for run in base_runs]

    paths = [path for path, _ in axes]
    exclusions = _sweep_exclusions(cfg, set(paths))
    combinations = [
        values
        for values in itertools.product(*(axis_values for _, axis_values in axes))
        if not _is_excluded(dict(zip(paths, values)), exclusions)
    ]
    if not combinations:
        raise ValueError("sweep_exclude removes every sweep combination.")
    expanded: list[Any] = []
    matrix_index = 0
    for base_run in base_runs:
        base = OmegaConf.to_container(base_run, resolve=True)
        if not isinstance(base, dict):
            raise ValueError("Each benchmark run must be a mapping.")
        base_name = safe_name(str(base.get("name", base.get("role", "run"))))
        for values in combinations:
            params = {path: deepcopy(value) for path, value in zip(paths, values)}
            run = deepcopy(base)
            for path, value in params.items():
                _set_path(run, path, value)

            # Eval templates commonly pin topk. That would make elite_frac a
            # no-op, so an elite-fraction sweep explicitly selects fractional
            # elite sizing unless topk is itself swept.
            if "planner.elite_frac" in params and "planner.topk" not in params:
                _set_path(run, "planner.topk", None)

            suffix = "__".join(
                f"{_LABELS.get(path, safe_name(path.replace('.', '_')))}{_value_token(value)}"
                for path, value in params.items()
            )
            cell_id = safe_name(f"{base_name}__{suffix}")
            run["base_name"] = base_name
            run["strategy"] = str(base.get("role", base_name))
            run["name"] = cell_id
            run["cell_id"] = cell_id
            run["sweep_params"] = params
            run["sweep_key"] = sweep_key(params)
            run["matrix_index"] = matrix_index
            expanded.append(OmegaConf.create(run))
            matrix_index += 1
    return expanded


__all__ = ["expand_benchmark_runs", "sweep_key"]
