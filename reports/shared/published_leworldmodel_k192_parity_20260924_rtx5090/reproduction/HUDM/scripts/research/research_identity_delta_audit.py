#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path("/vast/projects/dineshj/lab/ethanyu/code/" + "H" + "UDM")
ARTIFACT_ROOT = Path(os.environ.get("MWM_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
OUT_DIR = REPO_ROOT / "reports" / "research" / "identity_delta"

ENVS: dict[str, dict[str, Any]] = {
    "pusht": {
        "label": "PushT",
        "env_id": "swm/PushT-v1",
        "dataset": "data/upstream/pusht_expert_train.lance",
        "train_config": "configs/train/mwm_lewm_pusht_upstream.yaml",
        "eval_config": "configs/eval/paper_pusht.yaml",
        "benchmark_config": "configs/benchmark/paper_parity_pusht.yaml",
        "rollout_dir": "rollouts/mwm_paper_parity_pusht",
        "train_log": "logs/mwm_train_pusht_identity_6192391.out",
        "lightning_ckpt": "logs/mwm_training/retrained_lewm_identity_pusht_upstream/checkpoints/last.ckpt",
        "checkpoints": {
            "upstream_lewm_converted": "checkpoints_mwm/upstream_lewm_pusht",
            "retrained_lewm_identity": "checkpoints_mwm/retrained_lewm_identity_pusht_upstream",
        },
    },
    "tworoom": {
        "label": "TwoRoom",
        "env_id": "swm/TwoRoom-v1",
        "dataset": "data/upstream/tworoom.lance",
        "train_config": "configs/train/mwm_lewm_tworoom_upstream.yaml",
        "eval_config": "configs/eval/paper_tworoom.yaml",
        "benchmark_config": "configs/benchmark/paper_parity_tworoom.yaml",
        "rollout_dir": "rollouts/mwm_paper_parity_tworoom",
        "train_log": "logs/mwm_train_tworoom_identity_6192392.out",
        "lightning_ckpt": "logs/mwm_training/retrained_lewm_identity_tworoom_upstream/checkpoints/last.ckpt",
        "checkpoints": {
            "upstream_lewm_converted": "checkpoints_mwm/upstream_lewm_tworoom",
            "retrained_lewm_identity": "checkpoints_mwm/retrained_lewm_identity_tworoom_upstream",
        },
    },
}


META_FIELDS = [
    "role",
    "fresh_init",
    "adapter_family",
    "architecture_version",
    "training_backend",
    "levels",
    "D",
    "action_dim",
    "action_block",
    "image_shape",
    "restore_spec",
    "source_config_sha256",
    "component_policy",
    "loss_scope",
    "training_recipe",
    "dataset",
    "action_preprocessing",
    "epoch",
    "last_checkpoint",
]


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TABLE_RE = re.compile(r"\|\s*([^|]+?)\s*\|\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\|")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def nested_get(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def load_checkpoint_bundle(run_dir: Path) -> dict[str, Any]:
    return {
        "path": str(run_dir),
        "config_sha256": file_sha256(run_dir / "config.json"),
        "metadata_sha256": file_sha256(run_dir / "world_metadata.json"),
        "config": read_json(run_dir / "config.json"),
        "metadata": read_json(run_dir / "world_metadata.json"),
    }


def compare_checkpoints() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_key, spec in ENVS.items():
        roles = {
            role: load_checkpoint_bundle(ARTIFACT_ROOT / rel)
            for role, rel in spec["checkpoints"].items()
        }
        comparisons = []
        upstream = roles["upstream_lewm_converted"]["metadata"]
        identity = roles["retrained_lewm_identity"]["metadata"]
        for field in META_FIELDS:
            comparisons.append(
                {
                    "field": field,
                    "upstream": nested_get(upstream, field),
                    "identity": nested_get(identity, field),
                    "same": nested_get(upstream, field) == nested_get(identity, field),
                }
            )
        config_comparisons = []
        for field in [
            "target",
            "kwargs.family",
            "kwargs.K",
            "kwargs.expected_D",
            "kwargs.action_dim",
            "kwargs.action_block",
            "kwargs.image_shape",
            "kwargs.normalize_imagenet",
            "kwargs.component_policy",
            "kwargs.source_config_sha256",
            "kwargs.training_recipe",
        ]:
            config_comparisons.append(
                {
                    "field": field,
                    "upstream": nested_get(roles["upstream_lewm_converted"]["config"], field),
                    "identity": nested_get(roles["retrained_lewm_identity"]["config"], field),
                    "same": nested_get(roles["upstream_lewm_converted"]["config"], field)
                    == nested_get(roles["retrained_lewm_identity"]["config"], field),
                }
            )
        out[env_key] = {
            "roles": roles,
            "metadata_comparison": comparisons,
            "config_comparison": config_comparisons,
        }
    return out


def load_rollout_summaries() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_key, spec in ENVS.items():
        summary_path = ARTIFACT_ROOT / spec["rollout_dir"] / "summary.csv"
        rows = []
        with summary_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
        out[env_key] = {"summary_csv": str(summary_path), "rows": rows}
    return out


def arrow_column_to_numpy(col: Any) -> np.ndarray:
    ctype = col.type
    if pa.types.is_fixed_size_list(ctype):
        return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), ctype.list_size)
    if pa.types.is_integer(ctype) or pa.types.is_floating(ctype):
        return col.to_numpy(zero_copy_only=False)
    raise TypeError(f"Unsupported numeric column type: {ctype}")


def load_numeric_column(dataset: Any, column: str) -> np.ndarray:
    chunks = []
    reader = dataset.scanner(columns=[column]).to_reader()
    for batch in reader:
        idx = batch.schema.get_field_index(column)
        if idx < 0:
            raise KeyError(column)
        chunks.append(arrow_column_to_numpy(batch.column(idx)))
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def summarize_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    flat = flat.astype(np.float64, copy=False)
    if flat.size:
        flat = flat[~np.isnan(flat).any(axis=1)]
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    return {
        "shape": list(arr.shape),
        "valid_rows": int(flat.shape[0]),
        "mean": np.mean(flat, axis=0).tolist() if flat.size else [],
        "std_population": np.std(flat, axis=0, ddof=0).tolist() if flat.size else [],
        "std_sample": np.std(flat, axis=0, ddof=1).tolist() if flat.shape[0] > 1 else [],
        "min": np.min(flat, axis=0).tolist() if flat.size else [],
        "max": np.max(flat, axis=0).tolist() if flat.size else [],
        "quantiles": {
            str(q): np.quantile(flat, q, axis=0).tolist() if flat.size else [] for q in quantiles
        },
    }


def summarize_episode_lengths(ep_ids: np.ndarray) -> dict[str, Any]:
    if ep_ids.size == 0:
        lengths = np.array([], dtype=np.int64)
    else:
        changes = np.flatnonzero(np.diff(ep_ids) != 0) + 1
        offsets = np.concatenate([[0], changes])
        lengths = np.diff(np.concatenate([offsets, [len(ep_ids)]])).astype(np.int64)
    span = 20  # frameskip=5, history_size + num_preds = 4
    clip_count = int(np.maximum(lengths - span + 1, 0).sum()) if lengths.size else 0
    return {
        "episode_count": int(lengths.size),
        "length_min": int(lengths.min()) if lengths.size else 0,
        "length_max": int(lengths.max()) if lengths.size else 0,
        "length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "length_std": float(lengths.std(ddof=0)) if lengths.size else 0.0,
        "length_quantiles": {
            str(q): float(np.quantile(lengths, q)) if lengths.size else 0.0
            for q in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
        },
        "train_clip_count_before_split": clip_count,
        "random_split_train_count_0_9": int(math.floor(clip_count * 0.9)),
        "random_split_val_count_0_1": clip_count - int(math.floor(clip_count * 0.9)),
    }


def summarize_pixels(dataset: Any, column: str = "pixels", limit: int = 32) -> dict[str, Any]:
    arrays = []
    sizes = []
    reader = dataset.scanner(columns=[column], limit=limit).to_reader()
    for batch in reader:
        col = batch.column(batch.schema.get_field_index(column))
        for blob in col.to_pylist():
            with Image.open(BytesIO(blob)) as img:
                arr = np.asarray(img.convert("RGB"))
            arrays.append(arr)
            sizes.append(list(arr.shape))
            if len(arrays) >= limit:
                break
        if len(arrays) >= limit:
            break
    if not arrays:
        return {"sample_count": 0}
    stack = np.stack(arrays)
    return {
        "sample_count": len(arrays),
        "shape_first": sizes[0],
        "all_shapes_same": all(s == sizes[0] for s in sizes),
        "dtype": str(stack.dtype),
        "min": int(stack.min()),
        "max": int(stack.max()),
        "mean": float(stack.mean()),
        "std_population": float(stack.std(ddof=0)),
    }


def audit_datasets() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_key, spec in ENVS.items():
        path = ARTIFACT_ROOT / spec["dataset"]
        dataset = lance.dataset(path)
        schema = dataset.schema
        ep_ids = load_numeric_column(dataset, "episode_idx")
        numeric = {}
        for field in schema:
            if field.name in {"episode_idx", "step_idx"}:
                continue
            if pa.types.is_fixed_size_list(field.type) or pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                numeric[field.name] = summarize_array(load_numeric_column(dataset, field.name))
        out[env_key] = {
            "path": str(path),
            "metadata_path": str(path.with_suffix(path.suffix + ".metadata.json")),
            "sidecar_metadata": read_json(path.with_suffix(path.suffix + ".metadata.json")),
            "row_count": int(dataset.count_rows()),
            "columns": [{"name": field.name, "type": str(field.type)} for field in schema],
            "episode_lengths": summarize_episode_lengths(ep_ids),
            "numeric_columns": numeric,
            "pixel_sample": summarize_pixels(dataset, "pixels"),
        }
    return out


def parse_training_log(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"path": str(log_path), "exists": False}
    parts = [log_path.read_text(encoding="utf-8", errors="replace")]
    if log_path.suffix == ".out":
        err_path = log_path.with_suffix(".err")
        if err_path.is_file():
            parts.append(err_path.read_text(encoding="utf-8", errors="replace"))
    text = ANSI_RE.sub("", "\n".join(parts))
    metrics: dict[str, list[float]] = {}
    for key, value in TABLE_RE.findall(text):
        key = " ".join(key.split())
        if "/" not in key:
            continue
        metrics.setdefault(key, []).append(float(value))
    return {
        "path": str(log_path),
        "exists": True,
        "max_epochs_reached": "`Trainer.fit` stopped: `max_epochs=10` reached." in text,
        "exact_training_complete": "Exact Le-WM training complete." in text,
        "metric_last": {key: vals[-1] for key, vals in sorted(metrics.items()) if vals},
        "metric_count": {key: len(vals) for key, vals in sorted(metrics.items())},
    }


def inspect_lightning_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return {"path": str(path), "exists": path.is_file(), "error": repr(exc)}
    if not path.is_file():
        return {"path": str(path), "exists": False}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    schedulers = ckpt.get("lr_schedulers", [])
    return {
        "path": str(path),
        "exists": True,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "pytorch_lightning_version": ckpt.get("pytorch-lightning_version"),
        "state_dict_keys": len(ckpt.get("state_dict", {})),
        "optimizer_states": len(ckpt.get("optimizer_states", [])),
        "lr_schedulers": schedulers,
        "hparams": {
            key: ckpt.get("hyper_parameters", {}).get(key)
            for key in ["seed", "env_id", "schedule", "train", "optim", "loss", "data", "model"]
        },
    }


def audit_training() -> dict[str, Any]:
    out = {}
    for env_key, spec in ENVS.items():
        out[env_key] = {
            "log": parse_training_log(ARTIFACT_ROOT / spec["train_log"]),
            "lightning_checkpoint": inspect_lightning_checkpoint(ARTIFACT_ROOT / spec["lightning_ckpt"]),
        }
    return out


def audit_configs() -> dict[str, Any]:
    out = {}
    for env_key, spec in ENVS.items():
        out[env_key] = {}
        for key in ["train_config", "eval_config", "benchmark_config"]:
            path = REPO_ROOT / spec[key]
            out[env_key][key] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "data": read_yaml(path),
            }
        rollout_dir = ARTIFACT_ROOT / spec["rollout_dir"]
        resolved = sorted(rollout_dir.glob("*/resolved_config.yaml"))
        out[env_key]["resolved_configs"] = [
            {"path": str(path), "sha256": file_sha256(path), "data": read_yaml(path)} for path in resolved
        ]
    return out


def fmt(value: Any) -> str:
    text = json.dumps(jsonable(value), sort_keys=True)
    if len(text) > 160:
        text = text[:157] + "..."
    return text


def display_path(value: Any) -> str:
    text = str(value)
    text = text.replace(str(ARTIFACT_ROOT), "${MWM_ARTIFACT_ROOT}")
    text = text.replace(str(REPO_ROOT), "${WORKTREE_ROOT}")
    return text


def write_dataset_markdown(dataset_audit: dict[str, Any]) -> None:
    lines = [
        "# Identity Delta Dataset Audit",
        "",
        "Artifact root: `${MWM_ARTIFACT_ROOT}`",
        "",
        "| env | rows | episodes | length mean | length min/max | train clips before split | columns | action mean | action std(sample) | pixel sample |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
    ]
    for env_key, data in dataset_audit.items():
        lengths = data["episode_lengths"]
        action = data["numeric_columns"]["action"]
        cols = ", ".join(c["name"] for c in data["columns"])
        lines.append(
            "| {env} | {rows} | {eps} | {mean:.2f} | {mn}/{mx} | {clips} | `{cols}` | {amean} | {astd} | {pix} |".format(
                env=ENVS[env_key]["label"],
                rows=data["row_count"],
                eps=lengths["episode_count"],
                mean=lengths["length_mean"],
                mn=lengths["length_min"],
                mx=lengths["length_max"],
                clips=lengths["train_clip_count_before_split"],
                cols=cols,
                amean=fmt(action["mean"]),
                astd=fmt(action["std_sample"]),
                pix=f"{data['pixel_sample']['sample_count']} samples, mean={data['pixel_sample']['mean']:.2f}, std={data['pixel_sample']['std_population']:.2f}",
            )
        )
    lines.extend(
        [
            "",
            "## Per-Environment Details",
            "",
        ]
    )
    for env_key, data in dataset_audit.items():
        lines.extend(
            [
                f"### {ENVS[env_key]['label']}",
                "",
                f"- Path: `{display_path(data['path'])}`",
                f"- Sidecar: `{display_path(data['metadata_path'])}`",
                f"- Episode length quantiles: `{fmt(data['episode_lengths']['length_quantiles'])}`",
            ]
        )
        for col, stats in data["numeric_columns"].items():
            lines.append(
                f"- `{col}`: shape `{stats['shape']}`, mean `{fmt(stats['mean'])}`, std(sample) `{fmt(stats['std_sample'])}`, min `{fmt(stats['min'])}`, max `{fmt(stats['max'])}`"
            )
        lines.append("")
    (OUT_DIR / "dataset_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_static_markdown(static_audit: dict[str, Any], rollout_audit: dict[str, Any], training_audit: dict[str, Any]) -> None:
    lines = [
        "# Identity Delta Static Audit",
        "",
        "Artifact root: `${MWM_ARTIFACT_ROOT}`",
        "",
    ]
    for env_key in ENVS:
        lines.extend([f"## {ENVS[env_key]['label']}", ""])
        lines.extend(["### Rollout Rows", ""])
        lines.append("| role | success | episodes | seed | manifest | config | wall sec |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- | ---: |")
        for row in rollout_audit[env_key]["rows"]:
            lines.append(
                f"| {row['role']} | {row['success_rate']} | {row['episodes']} | {row['seed']} | `{row['manifest_sha256'][:12]}` | `{row['config_sha256'][:12]}` | {float(row['wall_time_sec']):.2f} |"
            )
        lines.extend(["", "### Metadata Differences", ""])
        lines.append("| field | upstream | identity | same |")
        lines.append("| --- | --- | --- | --- |")
        for row in static_audit[env_key]["metadata_comparison"]:
            if row["same"] and row["field"] not in {"adapter_family", "architecture_version", "levels", "action_dim", "action_block", "image_shape", "restore_spec", "source_config_sha256", "component_policy", "action_preprocessing"}:
                continue
            lines.append(f"| `{row['field']}` | `{fmt(row['upstream'])}` | `{fmt(row['identity'])}` | {row['same']} |")
        lines.extend(["", "### Training Log Last Metrics", ""])
        log = training_audit[env_key]["log"]
        ckpt = training_audit[env_key]["lightning_checkpoint"]
        lines.append(f"- Training log: `{display_path(log['path'])}`")
        lines.append(f"- Max epochs reached: `{log.get('max_epochs_reached')}`")
        lines.append(f"- Lightning checkpoint epoch/global_step: `{ckpt.get('epoch')}` / `{ckpt.get('global_step')}`")
        for key, value in sorted(log.get("metric_last", {}).items()):
            if key.startswith("fit/") or key.startswith("validate/"):
                lines.append(f"- Last `{key}`: `{value}`")
        lines.append("")
    (OUT_DIR / "static_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    static_audit = compare_checkpoints()
    config_audit = audit_configs()
    rollout_audit = load_rollout_summaries()
    dataset_audit = audit_datasets()
    training_audit = audit_training()
    payload = {
        "artifact_root": str(ARTIFACT_ROOT),
        "repo_root": str(REPO_ROOT),
        "static": static_audit,
        "configs": config_audit,
        "rollouts": rollout_audit,
        "datasets": dataset_audit,
        "training": training_audit,
    }
    (OUT_DIR / "audit_raw.json").write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    write_dataset_markdown(dataset_audit)
    write_static_markdown(static_audit, rollout_audit, training_audit)
    print(f"Wrote {OUT_DIR / 'audit_raw.json'}")
    print(f"Wrote {OUT_DIR / 'dataset_audit.md'}")
    print(f"Wrote {OUT_DIR / 'static_audit.md'}")


if __name__ == "__main__":
    main()
