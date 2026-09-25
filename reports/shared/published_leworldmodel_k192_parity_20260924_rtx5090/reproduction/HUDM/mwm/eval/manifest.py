from __future__ import annotations

from pathlib import Path
from typing import Any

from mwm.data.manifest import (
    generate_manifest,
    load_manifest,
    manifest_file_sha256,
    manifest_sha256,
    write_manifest,
)
from mwm.data.sampling import StartGoalPair, sample_start_goal_pairs
from mwm.dependency_refs import dependency_refs
from mwm.eval.validation import dataset_path, dataset_runtime_metadata, validate_manifest
from mwm.eval.runtime import effective_goal_offset, normalize_goal_indexing


def _pair_with_goal_offset(pair: StartGoalPair, offset: int) -> StartGoalPair:
    delta = int(offset) - (int(pair.goal_step) - int(pair.start_step))
    return StartGoalPair(
        episode=int(pair.episode),
        start_step=int(pair.start_step),
        goal_step=int(pair.goal_step) + delta,
        start_row=int(pair.start_row),
        goal_row=int(pair.goal_row) + delta,
    )


def manifest_row_to_pair(row: dict[str, Any]) -> StartGoalPair:
    return StartGoalPair(
        episode=int(row["episode"]),
        start_step=int(row["start_step"]),
        goal_step=int(row["goal_step"]),
        start_row=int(row["start_row"]),
        goal_row=int(row["goal_row"]),
    )


def pairs_for_eval(
    *,
    dataset: Any,
    cfg: Any,
    env_id: str,
    restore_spec_id: str,
) -> tuple[list[StartGoalPair], dict[str, Any] | None]:
    manifest_path = cfg.eval.get("manifest_path", None)
    if manifest_path:
        manifest = load_manifest(str(manifest_path))
        validate_manifest(
            manifest,
            path=str(manifest_path),
            dataset=dataset,
            cfg=cfg,
            env_id=env_id,
            restore_spec_id=restore_spec_id,
        )
        pairs = [manifest_row_to_pair(row) for row in manifest.get("pairs", [])]
        return pairs, {
            "path": str(manifest_path),
            "sha256": manifest_file_sha256(str(manifest_path)),
            "manifest_sha256": manifest.get("manifest_sha256", manifest_sha256(manifest)),
        }
    pairs = sample_start_goal_pairs(
        dataset,
        count=int(cfg.eval.episodes),
        goal_offset_steps=int(cfg.eval.goal_offset),
        seed=int(cfg.eval.seed),
        mode=str(cfg.eval.get("sampling", "mwm")),
    )
    goal_indexing = normalize_goal_indexing(str(cfg.eval.get("goal_indexing", "exact")))
    effective_offset = effective_goal_offset(int(cfg.eval.goal_offset), goal_indexing)
    if effective_offset != int(cfg.eval.goal_offset):
        pairs = [_pair_with_goal_offset(pair, effective_offset) for pair in pairs]
    write_path = cfg.eval.get("write_manifest_path", None)
    if not write_path:
        return pairs, None
    manifest = generate_manifest(
        env_id=env_id,
        dataset_path=dataset_path(dataset, cfg),
        pairs=pairs,
        goal_offset=int(cfg.eval.goal_offset),
        goal_indexing=goal_indexing,
        effective_goal_offset=effective_offset,
        eval_budget=int(cfg.eval.budget),
        seed=int(cfg.eval.seed),
        restore_spec=restore_spec_id,
        dataset_metadata=dataset_runtime_metadata(dataset, cfg),
        dependency_shas=dependency_refs(Path(__file__).resolve().parents[2]),
    )
    write_manifest(str(write_path), manifest)
    return pairs, {
        "path": str(write_path),
        "sha256": manifest_file_sha256(str(write_path)),
        "manifest_sha256": manifest.get("manifest_sha256", manifest_sha256(manifest)),
    }


__all__ = ["manifest_row_to_pair", "pairs_for_eval"]
