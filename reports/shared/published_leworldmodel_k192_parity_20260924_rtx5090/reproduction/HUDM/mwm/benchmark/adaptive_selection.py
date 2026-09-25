"""Deterministic discovery/confirmation selection for dense benchmark sweeps.

The selector deliberately separates two questions:

* The first ``discovery_episodes`` paired outcomes decide which matrix cells are
  promoted.
* Outcomes after that prefix are summarized only for already-promoted cells.

This separation makes it possible to reuse the first 50 episodes of an existing
250-episode result without using its held-out outcomes to choose winners.  All
success rates in this module are fractions in ``[0, 1]`` (unlike the benchmark
CSV, whose historical ``success_rate`` field is a percentage).

The uncertainty rule uses the exact, one-sided McNemar/sign test on discordant
paired episodes.  A cell is called confidence-dominated only when a cell with
no greater per-episode cost has significantly more paired wins after a global
Bonferroni correction over every cost-eligible directed comparison.  This is a
conservative family-wise guard; empirical and near-frontier rules can only add
promotions, never remove them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "mwm.adaptive_selection/v1"
UNCERTAINTY_METHOD = "paired_exact_mcnemar_global_bonferroni"


class AdaptiveSelectionError(ValueError):
    """Raised when screen inputs cannot support an auditable selection."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AdaptiveSelectionError(f"Selection metadata is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _as_outcome(value: Any, *, matrix_index: int, episode_index: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise AdaptiveSelectionError(
        f"matrix_index={matrix_index} episode {episode_index} must be a boolean or 0/1, got {value!r}."
    )


def _outcomes_for_row(
    row: Mapping[str, Any],
    matrix_index: int,
    episode_outcomes: Mapping[Any, Sequence[Any]] | None,
) -> list[bool]:
    raw: Any = None
    if episode_outcomes is not None:
        keys = (matrix_index, str(matrix_index), row.get("cell_id"), row.get("name"))
        for key in keys:
            if key is not None and key in episode_outcomes:
                raw = episode_outcomes[key]
                break
    if raw is None:
        raw = row.get("episode_successes")
    if raw is None and isinstance(row.get("swm_results"), Mapping):
        raw = row["swm_results"].get("episode_successes")
    if raw is None or isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise AdaptiveSelectionError(
            f"matrix_index={matrix_index} has no episode-level outcomes; pass episode_outcomes or "
            "include episode_successes in the row."
        )
    return [_as_outcome(value, matrix_index=matrix_index, episode_index=i) for i, value in enumerate(raw)]


def _discovery_source(row: Mapping[str, Any], *, episodes: int, matrix_index: int) -> dict[str, Any]:
    """Freeze the artifact that supplied the discovery prefix.

    Hybrid selection can mix prefixes from already-complete canonical
    250-episode cells with dedicated 50-episode screen cells.  Persisting this
    identity prevents a finalizer from silently replacing a screen prefix with
    a later canonical completion.
    """

    raw = row.get("discovery_source")
    if isinstance(raw, Mapping):
        source = dict(raw)
        kind = str(source.get("kind", "")).strip()
    elif raw is not None:
        source = {}
        kind = str(raw).strip()
    else:
        source = {}
        kind = "canonical_confirmation" if episodes > 50 else "screen"
    if kind not in {"canonical_confirmation", "screen"}:
        raise AdaptiveSelectionError(
            f"matrix_index={matrix_index} discovery_source.kind must be "
            f"'canonical_confirmation' or 'screen', got {kind!r}."
        )
    frozen = {
        "kind": kind,
        "output_json": str(source.get("output_json", row.get("output_json", ""))),
        "config_sha256": str(source.get("config_sha256", row.get("config_sha256", ""))),
        "manifest_sha256": str(source.get("manifest_sha256", row.get("manifest_sha256", ""))),
        "manifest_file_sha256": str(
            source.get("manifest_file_sha256", row.get("manifest_file_sha256", ""))
        ),
    }
    return frozen


def _wilson(successes: int, episodes: int, confidence: float = 0.95) -> list[float]:
    """Return a two-sided Wilson score interval without a scipy dependency."""

    if episodes <= 0:
        return [0.0, 1.0]
    # 1.959963984540054 is Phi^-1(0.975), kept explicit so output is stable
    # across Python versions and platforms.
    if confidence != 0.95:
        raise AdaptiveSelectionError("Only the versioned 0.95 descriptive interval is supported.")
    z = 1.959963984540054
    rate = successes / episodes
    z2 = z * z
    denominator = 1.0 + z2 / episodes
    center = (rate + z2 / (2.0 * episodes)) / denominator
    half_width = z * math.sqrt(rate * (1.0 - rate) / episodes + z2 / (4.0 * episodes**2)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _outcome_summary(outcomes: Sequence[bool], start: int, stop: int) -> dict[str, Any]:
    selected = outcomes[start:stop]
    successes = sum(selected)
    episodes = len(selected)
    return {
        "episode_slice": [start, stop],
        "episodes": episodes,
        "successes": successes,
        "success_rate_fraction": successes / episodes if episodes else None,
        "wilson_95_interval": _wilson(successes, episodes) if episodes else None,
    }


def _binomial_upper_tail(successes: int, trials: int) -> float:
    """P(Binomial(trials, 0.5) >= successes), exactly summed."""

    if trials <= 0:
        return 1.0
    if successes <= 0:
        return 1.0
    return float(sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials))


def _empirical_frontier(costs: np.ndarray, rates: np.ndarray, indices: np.ndarray) -> set[int]:
    order = sorted(range(len(indices)), key=lambda pos: (float(costs[pos]), -float(rates[pos]), int(indices[pos])))
    frontier: set[int] = set()
    best_rate = float("-inf")
    for pos in order:
        rate = float(rates[pos])
        if rate > best_rate:
            frontier.add(pos)
            best_rate = rate
    return frontier


def _cost_regions(costs: np.ndarray, requested_regions: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    unique_costs = sorted({float(value) for value in costs})
    region_count = min(requested_regions, len(unique_costs))
    cost_to_region = {
        cost: min(region_count - 1, rank * region_count // len(unique_costs))
        for rank, cost in enumerate(unique_costs)
    }
    regions = np.asarray([cost_to_region[float(cost)] for cost in costs], dtype=np.int64)
    bounds = []
    for region in range(region_count):
        values = [cost for cost, assigned in cost_to_region.items() if assigned == region]
        bounds.append(
            {
                "index": region,
                "label": f"q{region + 1}_of_{region_count}",
                "min_cost_per_episode": min(values),
                "max_cost_per_episode": max(values),
            }
        )
    return regions, bounds


def _paired_confidence_dominators(
    discovery: np.ndarray,
    costs: np.ndarray,
    rates: np.ndarray,
    matrix_indices: np.ndarray,
    *,
    familywise_alpha: float,
) -> tuple[list[list[dict[str, Any]]], int]:
    """Return significant no-more-expensive paired dominators for every row."""

    count = len(matrix_indices)
    eligible = costs[:, None] <= costs[None, :]
    np.fill_diagonal(eligible, False)
    comparison_count = int(np.count_nonzero(eligible))
    if comparison_count == 0:
        return [[] for _ in range(count)], 0

    outcomes = discovery.astype(np.int16, copy=False)
    q_wins = outcomes @ (1 - outcomes).T
    tail_cache: dict[tuple[int, int], float] = {}
    result: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for candidate in range(count):
        possible = np.flatnonzero(eligible[:, candidate] & (rates > rates[candidate]))
        for dominator in possible:
            wins = int(q_wins[dominator, candidate])
            losses = int(q_wins[candidate, dominator])
            discordant = wins + losses
            cache_key = (wins, discordant)
            raw_p = tail_cache.get(cache_key)
            if raw_p is None:
                raw_p = _binomial_upper_tail(wins, discordant)
                tail_cache[cache_key] = raw_p
            adjusted_p = min(1.0, raw_p * comparison_count)
            if adjusted_p <= familywise_alpha:
                result[candidate].append(
                    {
                        "matrix_index": int(matrix_indices[dominator]),
                        "cost_per_episode": float(costs[dominator]),
                        "success_rate_fraction": float(rates[dominator]),
                        "paired_wins": wins,
                        "paired_losses": losses,
                        "discordant_episodes": discordant,
                        "raw_one_sided_p": raw_p,
                        "bonferroni_adjusted_p": adjusted_p,
                    }
                )
        result[candidate].sort(
            key=lambda item: (
                float(item["bonferroni_adjusted_p"]),
                float(item["cost_per_episode"]),
                int(item["matrix_index"]),
            )
        )
    return result, comparison_count


def build_selection_manifest(
    screen_rows: Iterable[Mapping[str, Any]],
    *,
    episode_outcomes: Mapping[Any, Sequence[Any]] | None = None,
    discovery_episodes: int = 50,
    cost_key: str = "dynamics_flops_total",
    near_frontier_margin: float = 0.05,
    cost_regions: int = 4,
    familywise_alpha: float = 0.05,
    retain_uncertainty_nondominated: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sparse, uncertainty-aware promotion manifest.

    ``screen_rows`` must identify each original matrix cell with a unique
    ``matrix_index``.  Costs are normalized by the row's full ``episodes``
    count, while selection success is always recomputed from the first
    ``discovery_episodes`` episode outcomes.  The aggregate ``success_rate``
    field is intentionally ignored.
    """

    if isinstance(discovery_episodes, bool) or int(discovery_episodes) <= 0:
        raise AdaptiveSelectionError("discovery_episodes must be a positive integer.")
    discovery_episodes = int(discovery_episodes)
    if not 0.0 <= float(near_frontier_margin) <= 1.0:
        raise AdaptiveSelectionError("near_frontier_margin must be in [0, 1].")
    near_frontier_margin = float(near_frontier_margin)
    if isinstance(cost_regions, bool) or int(cost_regions) <= 0:
        raise AdaptiveSelectionError("cost_regions must be a positive integer.")
    cost_regions = int(cost_regions)
    if not 0.0 < float(familywise_alpha) < 1.0:
        raise AdaptiveSelectionError("familywise_alpha must be in (0, 1).")
    familywise_alpha = float(familywise_alpha)
    if not isinstance(retain_uncertainty_nondominated, bool):
        raise AdaptiveSelectionError("retain_uncertainty_nondominated must be a boolean.")

    prepared: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for source in screen_rows:
        if not isinstance(source, Mapping):
            raise AdaptiveSelectionError(f"Every screen row must be a mapping, got {type(source).__name__}.")
        raw_index = source.get("matrix_index")
        if isinstance(raw_index, bool):
            raise AdaptiveSelectionError("matrix_index must be an integer, not a boolean.")
        try:
            matrix_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise AdaptiveSelectionError(f"Invalid matrix_index {raw_index!r}.") from exc
        if matrix_index < 0 or matrix_index in seen_indices:
            raise AdaptiveSelectionError(f"matrix_index must be unique and nonnegative, got {matrix_index}.")
        seen_indices.add(matrix_index)

        outcomes = _outcomes_for_row(source, matrix_index, episode_outcomes)
        try:
            episodes = int(source.get("episodes"))
        except (TypeError, ValueError) as exc:
            raise AdaptiveSelectionError(f"matrix_index={matrix_index} has invalid episodes.") from exc
        if episodes != len(outcomes):
            raise AdaptiveSelectionError(
                f"matrix_index={matrix_index} declares episodes={episodes}, but has {len(outcomes)} outcomes."
            )
        if episodes < discovery_episodes:
            raise AdaptiveSelectionError(
                f"matrix_index={matrix_index} has {episodes} outcomes; {discovery_episodes} are required."
            )

        try:
            total_cost = float(source.get(cost_key))
        except (TypeError, ValueError) as exc:
            raise AdaptiveSelectionError(f"matrix_index={matrix_index} has invalid {cost_key!r}.") from exc
        if not math.isfinite(total_cost) or total_cost < 0.0:
            raise AdaptiveSelectionError(
                f"matrix_index={matrix_index} requires finite, nonnegative {cost_key}; got {total_cost!r}."
            )
        schedule = str(source.get("schedule") or source.get("strategy") or source.get("role") or "").strip()
        if not schedule:
            raise AdaptiveSelectionError(f"matrix_index={matrix_index} has no schedule/strategy/role label.")
        prepared.append(
            {
                "matrix_index": matrix_index,
                "cell_id": str(source.get("cell_id") or source.get("name") or matrix_index),
                "schedule": schedule,
                "episodes": episodes,
                "outcomes": outcomes,
                "total_cost": total_cost,
                "cost_per_episode": total_cost / episodes,
                "discovery_source": _discovery_source(
                    source,
                    episodes=episodes,
                    matrix_index=matrix_index,
                ),
            }
        )

    if not prepared:
        raise AdaptiveSelectionError("At least one screen row is required.")
    prepared.sort(key=lambda item: int(item["matrix_index"]))

    matrix_indices = np.asarray([item["matrix_index"] for item in prepared], dtype=np.int64)
    costs = np.asarray([item["cost_per_episode"] for item in prepared], dtype=np.float64)
    discovery = np.asarray(
        [item["outcomes"][:discovery_episodes] for item in prepared],
        dtype=np.int16,
    )
    rates = discovery.mean(axis=1)
    empirical_frontier = _empirical_frontier(costs, rates, matrix_indices)
    regions, region_bounds = _cost_regions(costs, cost_regions)
    dominators, comparison_count = _paired_confidence_dominators(
        discovery,
        costs,
        rates,
        matrix_indices,
        familywise_alpha=familywise_alpha,
    )

    best_at_or_below = np.asarray(
        [float(np.max(rates[costs <= cost])) for cost in costs],
        dtype=np.float64,
    )
    gaps = best_at_or_below - rates

    representative_positions: set[int] = set()
    representative_for_group: dict[tuple[str, int], int] = {}
    for schedule in sorted({str(item["schedule"]) for item in prepared}):
        for region in sorted({int(regions[pos]) for pos, item in enumerate(prepared) if item["schedule"] == schedule}):
            members = [
                pos
                for pos, item in enumerate(prepared)
                if item["schedule"] == schedule and int(regions[pos]) == region
            ]
            representative = min(
                members,
                key=lambda pos: (-float(rates[pos]), float(costs[pos]), int(matrix_indices[pos])),
            )
            representative_positions.add(representative)
            representative_for_group[(schedule, region)] = representative

    all_cells: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for pos, item in enumerate(prepared):
        reasons: list[dict[str, Any]] = []
        if pos in empirical_frontier:
            reasons.append({"code": "empirical_pareto"})
        if retain_uncertainty_nondominated and not dominators[pos]:
            reasons.append(
                {
                    "code": "paired_uncertainty_nondominated",
                    "method": UNCERTAINTY_METHOD,
                    "familywise_alpha": familywise_alpha,
                }
            )
        if float(gaps[pos]) <= near_frontier_margin + 1e-15:
            reasons.append(
                {
                    "code": "near_frontier",
                    "margin": near_frontier_margin,
                    "success_gap": float(gaps[pos]),
                    "best_success_at_no_greater_cost": float(best_at_or_below[pos]),
                }
            )
        if pos in representative_positions:
            region = int(regions[pos])
            reasons.append(
                {
                    "code": "schedule_cost_representative",
                    "schedule": str(item["schedule"]),
                    "cost_region": region,
                    "cost_region_label": region_bounds[region]["label"],
                }
            )

        selected = bool(reasons)
        if selected:
            selected_indices.append(int(item["matrix_index"]))
            if int(item["episodes"]) > discovery_episodes:
                confirmation = {
                    "status": "available",
                    **_outcome_summary(
                        item["outcomes"],
                        discovery_episodes,
                        int(item["episodes"]),
                    ),
                }
            else:
                confirmation = {
                    "status": "pending",
                    "episode_slice": [discovery_episodes, int(item["episodes"])],
                    "episodes": 0,
                }
        else:
            # Do not inspect or report held-out statistics for a cell rejected
            # by discovery, even if the caller supplied a legacy 250-episode
            # artifact containing them.
            confirmation = {"status": "withheld_by_design"}

        uncertainty = {
            "status": "confidence_dominated" if dominators[pos] else "nondominated",
            "method": UNCERTAINTY_METHOD,
            "global_comparisons": comparison_count,
            "familywise_alpha": familywise_alpha,
            "dominator_count": len(dominators[pos]),
            "best_dominator": dominators[pos][0] if dominators[pos] else None,
        }
        region = int(regions[pos])
        representative = representative_for_group[(str(item["schedule"]), region)]
        all_cells.append(
            {
                "matrix_index": int(item["matrix_index"]),
                "cell_id": str(item["cell_id"]),
                "schedule": str(item["schedule"]),
                "observed_episodes": int(item["episodes"]),
                "discovery_source": dict(item["discovery_source"]),
                "cost": {
                    "key": cost_key,
                    "total": float(item["total_cost"]),
                    "per_episode": float(item["cost_per_episode"]),
                },
                "cost_region": region,
                "cost_region_label": region_bounds[region]["label"],
                "discovery": _outcome_summary(item["outcomes"], 0, discovery_episodes),
                "frontier_success_gap": float(gaps[pos]),
                "paired_uncertainty": uncertainty,
                "exclusion_interpretation": (
                    None
                    if selected
                    else "screened_out_for_resource_allocation_not_proven_truly_dominated"
                ),
                "schedule_cost_representative_matrix_index": int(matrix_indices[representative]),
                "selected": selected,
                "reasons": reasons,
                "confirmation": confirmation,
            }
        )

    discovery_fingerprint = [
        {
            "matrix_index": int(item["matrix_index"]),
            "cell_id": str(item["cell_id"]),
            "schedule": str(item["schedule"]),
            "episodes": int(item["episodes"]),
            "total_cost": float(item["total_cost"]),
            "discovery_source": dict(item["discovery_source"]),
            "discovery_outcomes": item["outcomes"][:discovery_episodes],
        }
        for item in prepared
    ]
    selected_set = set(selected_indices)
    confirmation_fingerprint = [
        {
            "matrix_index": int(item["matrix_index"]),
            "heldout_outcomes": item["outcomes"][discovery_episodes:],
        }
        for item in prepared
        if int(item["matrix_index"]) in selected_set
    ]
    selected_cells = [cell for cell in all_cells if cell["selected"]]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_config": {
            "selection_basis": "discovery_prefix_only",
            "discovery_episodes": discovery_episodes,
            "cost_key": cost_key,
            "cost_normalization": "total_divided_by_observed_episodes",
            "near_frontier_margin": near_frontier_margin,
            "cost_region_method": "equal_rank_unique_cost_quantiles",
            "requested_cost_regions": cost_regions,
            "realized_cost_regions": len(region_bounds),
            "cost_region_bounds": region_bounds,
            "uncertainty_method": UNCERTAINTY_METHOD,
            "familywise_alpha": familywise_alpha,
            "global_comparisons": comparison_count,
            "retain_uncertainty_nondominated": retain_uncertainty_nondominated,
            "paired_episode_alignment_required": True,
            "descriptive_interval": "wilson_95",
        },
        "screen": {
            "cell_count": len(prepared),
            "input_sha256": _sha256(discovery_fingerprint),
            "minimum_observed_episodes": min(int(item["episodes"]) for item in prepared),
            "maximum_observed_episodes": max(int(item["episodes"]) for item in prepared),
        },
        "confirmation": {
            "selection_leakage_guard": "heldout outcomes do not influence selected_matrix_indices",
            "input_sha256_for_selected_cells": _sha256(confirmation_fingerprint),
            "available_cell_count": sum(
                cell["confirmation"]["status"] == "available" for cell in selected_cells
            ),
            "pending_cell_count": sum(cell["confirmation"]["status"] == "pending" for cell in selected_cells),
        },
        "selected_count": len(selected_indices),
        "inference_caveat": (
            "Selection is a resource-allocation decision from a finite discovery sample. "
            "A screened-out cell is not proven absent from the population Pareto frontier."
        ),
        "selected_matrix_indices": selected_indices,
        "selected_cells": selected_cells,
        "all_cells": all_cells,
        "provenance": dict(provenance or {}),
    }
    # Validate both JSON representability and the public sparse-index contract.
    _canonical_bytes(manifest)
    validate_selection_manifest(manifest)
    return manifest


def validate_selection_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the fields consumed by sparse benchmark execution/finalization."""

    if not isinstance(manifest, Mapping):
        raise AdaptiveSelectionError("Selection manifest must be a mapping.")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AdaptiveSelectionError(
            f"Unsupported selection schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION!r}."
        )
    raw_indices = manifest.get("selected_matrix_indices")
    if not isinstance(raw_indices, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_indices):
        raise AdaptiveSelectionError("selected_matrix_indices must be a list of integers.")
    if raw_indices != sorted(set(raw_indices)) or any(value < 0 for value in raw_indices):
        raise AdaptiveSelectionError("selected_matrix_indices must be sorted, unique, and nonnegative.")
    if manifest.get("selected_count") != len(raw_indices):
        raise AdaptiveSelectionError("selected_count does not match selected_matrix_indices.")
    selected_cells = manifest.get("selected_cells")
    all_cells = manifest.get("all_cells")
    if not isinstance(selected_cells, list) or not isinstance(all_cells, list):
        raise AdaptiveSelectionError("selected_cells and all_cells must be lists.")
    selected_from_cells = [cell.get("matrix_index") for cell in selected_cells if isinstance(cell, Mapping)]
    if selected_from_cells != raw_indices:
        raise AdaptiveSelectionError("selected_cells do not match selected_matrix_indices.")
    selected_from_audit = [
        cell.get("matrix_index")
        for cell in all_cells
        if isinstance(cell, Mapping) and cell.get("selected") is True
    ]
    if selected_from_audit != raw_indices:
        raise AdaptiveSelectionError("all_cells selection flags do not match selected_matrix_indices.")
    _canonical_bytes(manifest)


def write_selection_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    """Atomically write a validated selection manifest and return its path."""

    validate_selection_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return str(output)


def load_selection_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a selection manifest."""

    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_selection_manifest(manifest)
    return manifest


__all__ = [
    "AdaptiveSelectionError",
    "SCHEMA_VERSION",
    "UNCERTAINTY_METHOD",
    "build_selection_manifest",
    "load_selection_manifest",
    "validate_selection_manifest",
    "write_selection_manifest",
]
