from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULTS = {
    "output_dir": "rollouts/mwm_benchmark",
    "title": "MWM Benchmark",
    "env_id": None,
    "seed": 0,
    "eval_config": None,
    "manifest": {},
    "runs": [],
}


def safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return text.strip("_") or "run"


def require_no_legacy_fields(cfg: Any) -> None:
    if "gate" in cfg:
        raise ValueError("legacy benchmark field `gate` is no longer supported; derive roles from runs.")


def require_no_legacy_run_fields(run: Any) -> None:
    for field in ("manifest_group", "overrides", "config"):
        if field in run:
            raise ValueError(f"legacy benchmark field runs[].{field} is no longer supported.")


def as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping, got {type(value).__name__}.")
    return dict(value)


def load_manifest_config(cfg: Any) -> dict[str, Any]:
    raw = as_plain_dict(cfg.get("manifest", {}))
    if "config" in raw:
        manifest_cfg = OmegaConf.load(str(raw.pop("config")))
        merged = {**as_plain_dict(manifest_cfg), **raw}
    else:
        merged = raw
    group = str(merged.get("group", "")).strip()
    if not group:
        raise ValueError("Benchmark config must define manifest.group or manifest.config with group.")
    if "path" not in merged:
        manifest_dir = Path(str(merged.get("dir", "rollouts/manifests")))
        merged["path"] = str(manifest_dir / f"{safe_name(group)}_seed{int(cfg.seed)}.json")
    merged["group"] = group
    merged["path"] = str(merged["path"])
    return merged


def checkpoint_mapping(run: Any) -> dict[str, Any]:
    if "checkpoint" not in run:
        raise ValueError("Each benchmark run must define runs[].checkpoint.")
    raw = run.checkpoint
    if isinstance(raw, str):
        return {"run_dir": raw, "epoch": None}
    checkpoint = as_plain_dict(raw)
    if "run_dir" not in checkpoint:
        raise ValueError("runs[].checkpoint must be a path string or mapping with run_dir.")
    checkpoint.setdefault("epoch", None)
    return checkpoint


def benchmark_eval_template(cfg: Any) -> Any:
    if not cfg.get("eval_config", None):
        raise ValueError("Benchmark config must define top-level eval_config.")
    if not cfg.get("env_id", None):
        raise ValueError("Benchmark config must define top-level env_id.")
    template = OmegaConf.load(str(cfg.eval_config))
    template_env = str(template.get("env_id", cfg.env_id))
    if template_env != str(cfg.env_id):
        raise ValueError(f"Benchmark env_id={cfg.env_id!r} does not match eval_config env_id={template_env!r}.")
    template.env_id = str(cfg.env_id)
    template.eval.seed = int(cfg.seed)
    return template


def merged_run_config(cfg: Any, run: Any) -> tuple[str, Any]:
    require_no_legacy_run_fields(run)
    name = safe_name(str(run.get("name", run.get("role", "run"))))
    run_cfg = OmegaConf.create(OmegaConf.to_container(benchmark_eval_template(cfg), resolve=True))
    run_cfg.checkpoint = checkpoint_mapping(run)
    planner = run.get("planner", None)
    if planner is not None:
        run_cfg.planner = OmegaConf.merge(run_cfg.get("planner", {}), planner)
    run_env = run.get("env", None)
    if run_env is not None:
        run_cfg.env = OmegaConf.merge(run_cfg.get("env", {}), run_env)
    run_eval = run.get("eval", None)
    if run_eval is not None:
        run_cfg.eval = OmegaConf.merge(run_cfg.get("eval", {}), run_eval)
        run_cfg.eval.seed = int(cfg.seed)
    return name, run_cfg


def write_temp_config(cfg: Any) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="mwm_benchmark_", delete=False)
    tmp.write(OmegaConf.to_yaml(cfg))
    tmp.close()
    return tmp.name


def manifest_path(cfg: Any) -> Path:
    return Path(str(load_manifest_config(cfg)["path"]))


def role(run: Any, cfg: Any) -> str:
    del cfg
    if run.get("role", None):
        return str(run.role)
    return str(run.get("name", "run"))


def normalize_role_filter(roles: Any = None) -> set[str]:
    if roles is None:
        return set()
    if isinstance(roles, str):
        raw_items = [roles]
    else:
        raw_items = list(roles)
    selected: set[str] = set()
    for item in raw_items:
        for part in str(item).split(","):
            item_role = part.strip()
            if item_role:
                selected.add(item_role)
    return selected


def filter_resolved_by_roles(cfg: Any, resolved: list[tuple[Any, Any]], roles: Any = None) -> list[tuple[Any, Any]]:
    del cfg
    selected = normalize_role_filter(roles)
    if not selected:
        return resolved
    filtered = [(run, run_cfg) for run, run_cfg in resolved if role(run, run_cfg) in selected]
    if not filtered:
        raise ValueError(f"No benchmark runs matched requested roles: {sorted(selected)}")
    return filtered


def validate_benchmark_matrix(cfg: Any, resolved: list[tuple[Any, Any]]) -> None:
    require_no_legacy_fields(cfg)
    load_manifest_config(cfg)
    identities = [(str(run_cfg.get("env_id", "")), int(run_cfg.eval.seed), role(run, run_cfg)) for run, run_cfg in resolved]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ValueError(f"Benchmark has duplicate cells: {duplicates}")
    env_ids = {identity[0] for identity in identities}
    seeds = {identity[1] for identity in identities}
    if env_ids != {str(cfg.env_id)}:
        raise ValueError(f"Benchmark runs must all use env_id={cfg.env_id!r}, got {sorted(env_ids)}.")
    if seeds != {int(cfg.seed)}:
        raise ValueError(f"Benchmark runs must all use seed={int(cfg.seed)}, got {sorted(seeds)}.")


__all__ = [
    "DEFAULTS",
    "as_plain_dict",
    "benchmark_eval_template",
    "checkpoint_mapping",
    "filter_resolved_by_roles",
    "load_manifest_config",
    "manifest_path",
    "merged_run_config",
    "normalize_role_filter",
    "require_no_legacy_fields",
    "require_no_legacy_run_fields",
    "role",
    "safe_name",
    "validate_benchmark_matrix",
    "write_temp_config",
]
