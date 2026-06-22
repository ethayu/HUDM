from __future__ import annotations

from typing import Any

import numpy as np


def uses_standardized_action_space(model: Any, metadata: dict[str, Any], cfg: Any) -> bool:
    del model
    raw = cfg.eval.get("action_preprocessing", cfg.data.get("action_preprocessing", "auto"))
    if isinstance(raw, bool):
        return raw
    mode = str(raw).lower()
    if mode in {"none", "identity", "raw", "false", "0"}:
        return False
    if mode in {"standard_scaler", "standard-scaler", "zscore", "normalized", "true", "1"}:
        return True
    if mode != "auto":
        raise ValueError(f"Unknown action_preprocessing mode {raw!r}.")

    preprocessing = metadata.get("action_preprocessing", metadata.get("action_normalization", None))
    if str(preprocessing).lower() in {"standard_scaler", "standard-scaler", "zscore", "normalized"}:
        return True
    dataset_meta = metadata.get("dataset", {})
    normalized = dataset_meta.get("normalized_columns", []) if isinstance(dataset_meta, dict) else []
    if "action" in {str(x) for x in normalized}:
        return True
    return False


def stat_keys_for_action_process(cfg: Any) -> list[str]:
    raw = cfg.data.get("keys_to_cache", cfg.data.get("process_keys", ["action", "proprio", "state"]))
    keys = [str(k) for k in (list(raw) if raw is not None else [])]
    action_key = str(cfg.data.get("action_key", "action"))
    if action_key not in keys:
        keys.insert(0, action_key)
    return [k for k in keys if k != str(cfg.data.get("pixels_key", "pixels"))]


def available_stat_keys_for_action_process(cfg: Any, columns: Any) -> list[str]:
    available = {str(column) for column in columns}
    return [key for key in stat_keys_for_action_process(cfg) if key in available]


def fit_standard_scaler(dataset: Any, key: str) -> Any:
    from sklearn import preprocessing

    values = np.asarray(dataset.get_col_data(key))
    values = values.reshape(values.shape[0], -1)
    if np.issubdtype(values.dtype, np.number):
        values = values[~np.isnan(values).any(axis=1)]
    if values.size == 0:
        raise ValueError(f"Cannot fit action preprocessing scaler for empty dataset column {key!r}.")
    scaler = preprocessing.StandardScaler()
    scaler.fit(values)
    return scaler


def build_eval_process(dataset: Any, model: Any, metadata: dict[str, Any], cfg: Any) -> dict[str, Any]:
    if not uses_standardized_action_space(model, metadata, cfg):
        return {}
    columns = {str(c) for c in getattr(dataset, "column_names", [])}
    action_key = str(cfg.data.get("action_key", "action"))
    if action_key not in columns:
        raise ValueError(
            f"Checkpoint expects standardized actions, but dataset column {action_key!r} is not loaded."
        )
    process: dict[str, Any] = {}
    for key in stat_keys_for_action_process(cfg):
        if key not in columns:
            continue
        scaler = fit_standard_scaler(dataset, key)
        process[key] = scaler
        if key != action_key:
            process[f"goal_{key}"] = scaler
    return process


_available_stat_keys_for_action_process = available_stat_keys_for_action_process
_build_eval_process = build_eval_process
_fit_standard_scaler = fit_standard_scaler
_stat_keys_for_action_process = stat_keys_for_action_process
_uses_standardized_action_space = uses_standardized_action_space


__all__ = [
    "available_stat_keys_for_action_process",
    "build_eval_process",
    "fit_standard_scaler",
    "stat_keys_for_action_process",
    "uses_standardized_action_space",
    "_available_stat_keys_for_action_process",
    "_build_eval_process",
    "_fit_standard_scaler",
    "_stat_keys_for_action_process",
    "_uses_standardized_action_space",
]
