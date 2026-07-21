from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np

from mwm.io import load_json


ProgressCallback = Callable[[str], None]
_MODEL_CACHE: dict[tuple[str, int, str], tuple[Any, dict[str, Any], int]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(str(message))


def _resolve_lance_table(path: str | Path) -> tuple[Path, str]:
    location = Path(path).resolve()
    if location.name.endswith(".lance"):
        return location.parent, location.stem
    if location.is_dir():
        tables = sorted(location.glob("*.lance"))
        if len(tables) == 1:
            return location, tables[0].stem
    raise ValueError(f"review rendering requires one local .lance table, got {location}")


def _source_columns(callables: list[dict[str, Any]], available: set[str], pixels_key: str) -> list[str]:
    selected = {str(pixels_key)}
    if "seed" in available:
        selected.add("seed")
    for spec in callables:
        for arg in spec.get("args", {}).values():
            if not isinstance(arg, dict) or not arg.get("in_dataset", True):
                continue
            value = str(arg.get("value", ""))
            source = value.removeprefix("goal_")
            if source in available:
                selected.add(source)
            elif value in available:
                selected.add(value)
    missing = sorted(selected - available)
    if missing:
        raise ValueError(f"review dataset is missing required columns: {missing}")
    return sorted(selected)


class TargetedLanceReviewDataset:
    """Read only the global rows needed to replay one reviewed episode."""

    def __init__(
        self,
        path: str | Path,
        *,
        start_row: int,
        pixels_key: str,
        action_key: str,
        env_id: str,
        restore_import_path: str | None,
    ) -> None:
        import lancedb

        from mwm.swm.restore import eval_callables_for_env

        database, table_name = _resolve_lance_table(path)
        table = lancedb.connect(str(database)).open_table(table_name)
        self._lance = table.to_lance()
        self._row_count = int(table.count_rows())
        self._available = {
            str(name)
            for name in table.schema.names
            if str(name) not in {"episode_idx", "step_idx"}
        }
        self.restore_spec_id, self.eval_callables = eval_callables_for_env(
            str(env_id),
            sorted(self._available),
            import_path=restore_import_path,
        )
        self.path = str(Path(path))
        self.uri = self.path
        self.start_row = int(start_row)
        self.pixels_key = str(pixels_key)
        self.action_key = str(action_key)
        self.column_names = _source_columns(
            self.eval_callables,
            self._available,
            self.pixels_key,
        )

    def close(self) -> None:
        return None

    @staticmethod
    def _decode_images(values: list[Any]) -> Any:
        import torch
        from PIL import Image

        frames = []
        for value in values:
            with Image.open(io.BytesIO(bytes(value))) as image:
                array = np.asarray(image.convert("RGB")).copy()
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(frames)

    @staticmethod
    def _numeric_tensor(column: Any) -> Any:
        import pyarrow as pa
        import torch

        if pa.types.is_fixed_size_list(column.type):
            values = column.flatten().to_numpy(zero_copy_only=False)
            array = values.reshape(len(column), column.type.list_size)
        elif pa.types.is_list(column.type):
            array = np.asarray(column.to_pylist(), dtype=np.float32)
        else:
            array = column.to_numpy(zero_copy_only=False)
        return torch.as_tensor(np.asarray(array).copy())

    def _take(self, rows: list[int], columns: list[str]) -> dict[str, Any]:
        import pyarrow as pa

        if not rows:
            raise ValueError("review row selection is empty")
        if min(rows) < 0 or max(rows) >= self._row_count:
            raise IndexError(
                f"review rows {min(rows)}..{max(rows)} are outside dataset row count {self._row_count}"
            )
        batch = self._lance.take(rows, columns=columns)
        out: dict[str, Any] = {}
        for name in columns:
            column = batch.column(name).combine_chunks()
            if pa.types.is_binary(column.type) or pa.types.is_large_binary(column.type):
                out[name] = self._decode_images(column.to_pylist())
            else:
                out[name] = self._numeric_tensor(column)
        return out

    def load_chunk(
        self,
        episodes_idx: Any,
        start_steps: Any,
        end_steps: Any,
    ) -> list[dict[str, Any]]:
        if len(episodes_idx) != 1 or len(start_steps) != 1 or len(end_steps) != 1:
            raise ValueError("targeted review datasets support exactly one replay episode")
        length = int(end_steps[0]) - int(start_steps[0])
        rows = list(range(self.start_row, self.start_row + length))
        return [self._take(rows, self.column_names)]

    def get_frame(self, row: int, pixels_key: str | None = None) -> Any:
        key = str(pixels_key or self.pixels_key)
        return self._take([int(row)], [key])[key][0]


@dataclass
class ReviewRuntime:
    cfg: Any
    device: Any
    model: Any
    metadata: dict[str, Any]
    dataset: TargetedLanceReviewDataset
    env_id: str
    image_shape: tuple[int, int]
    restore_spec_id: str
    eval_callables: list[dict[str, Any]]
    model_cache_hit: bool = False

    def close(self) -> None:
        self.dataset.close()


def _load_review_model(cfg: Any, callback: ProgressCallback | None) -> tuple[Any, dict[str, Any], Any, bool]:
    from mwm.eval.runtime import resolve_device

    device = resolve_device(str(cfg.device))
    checkpoint = Path(str(cfg.checkpoint.run_dir)).resolve()
    weights = checkpoint / "weights.pt"
    cache_key = (str(checkpoint), int(weights.stat().st_mtime_ns), str(device))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            model, metadata, epoch = cached
            _progress(callback, "Using cached checkpoint")
            return model, dict(metadata), device, True
        _progress(callback, f"Loading checkpoint on {device}")
        from mwm.checkpoint_io import load_world_model_from_checkpoint

        model, metadata, epoch = load_world_model_from_checkpoint(
            checkpoint,
            None if cfg.checkpoint.epoch is None else int(cfg.checkpoint.epoch),
            device=device,
        )
        _MODEL_CACHE[cache_key] = (model, dict(metadata), int(epoch))
        return model, dict(metadata), device, False


def load_review_runtime(
    cfg_path: str | Path,
    *,
    start_row: int,
    goal_row: int,
    load_model: bool,
    progress: ProgressCallback | None = None,
) -> ReviewRuntime:
    from mwm.config_cli import load_config
    from mwm.data.paths import local_path
    from mwm.eval.runtime import DEFAULTS
    from mwm.eval.validation import validate_dataset_metadata
    from mwm.swm.envs import parse_image_shape

    cfg = load_config(DEFAULTS, str(cfg_path), [])
    checkpoint = Path(str(cfg.checkpoint.run_dir))
    if load_model:
        model, metadata, device, cache_hit = _load_review_model(cfg, progress)
    else:
        _progress(progress, "Reading checkpoint metadata")
        model = None
        device = None
        cache_hit = False
        metadata = load_json(checkpoint / "world_metadata.json")

    goal_offset = int(cfg.eval.goal_offset)
    if int(goal_row) - int(start_row) != goal_offset:
        raise ValueError(
            f"review trace row span {int(goal_row) - int(start_row)} does not match goal_offset={goal_offset}"
        )
    restore_import_path = None if cfg.get("restore", None) is None else cfg.restore.get("import_path", None)
    _progress(progress, f"Reading {goal_offset + 1} targeted dataset rows")
    dataset = TargetedLanceReviewDataset(
        local_path(cfg.data.path),
        start_row=int(start_row),
        pixels_key=str(cfg.data.get("pixels_key", "pixels")),
        action_key=str(cfg.data.get("action_key", "action")),
        env_id=str(metadata["env_id"]),
        restore_import_path=restore_import_path,
    )
    validate_dataset_metadata(dataset, metadata, cfg)
    if str(metadata["restore_spec"]) != dataset.restore_spec_id:
        raise ValueError(
            f"Runtime restore spec {dataset.restore_spec_id!r} does not match "
            f"checkpoint restore_spec={metadata['restore_spec']!r}."
        )
    return ReviewRuntime(
        cfg=cfg,
        device=device,
        model=model,
        metadata=metadata,
        dataset=dataset,
        env_id=str(metadata["env_id"]),
        image_shape=parse_image_shape(metadata["image_shape"]),
        restore_spec_id=dataset.restore_spec_id,
        eval_callables=dataset.eval_callables,
        model_cache_hit=cache_hit,
    )


__all__ = [
    "ReviewRuntime",
    "TargetedLanceReviewDataset",
    "load_review_runtime",
]
