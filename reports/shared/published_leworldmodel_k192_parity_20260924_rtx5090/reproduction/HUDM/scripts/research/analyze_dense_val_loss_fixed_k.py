from __future__ import annotations

import argparse
import ast
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


LEVELS = (6, 12, 48, 96, 144, 192)
LEVEL_COLUMNS = {k: f"validate/pred_loss_l{idx}(K={k})" for idx, k in enumerate(LEVELS)}
ENV_NAME_TO_SLUG = {
    "PushT": "pusht",
    "Reacher": "reacher",
    "OGB Cube": "ogb_cube",
    "TwoRoom": "tworoom",
}
ENV_ORDER = ("pusht", "reacher", "ogb_cube", "tworoom")
LR_TOKEN_TO_VALUE = {"1em6": "1e-6", "1em5": "1e-5", "2em5": "2e-5", "5em5": "5e-5"}
LR_ORDER = ("1e-6", "1e-5", "2e-5", "5e-5")
LR_FLOAT_TO_VALUE = {float(value): value for value in LR_ORDER}
RUN_RE = re.compile(r"^lr(?P<lr_token>1em6|1em5|2em5|5em5)_fixed_k(?P<k>\d+)$")
DEFAULT_ROOT = Path("reports/research/dense_val_loss_fixed_k_20260721")
DEFAULT_TRAINING_CSV = Path("reports/dense_lr_early_stop_warm10_20260702_070021/training_val_losses.csv")


def _rank_quality(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=False)


def load_training_losses(path: Path) -> pd.DataFrame:
    training = pd.read_csv(path)
    selected = training.loc[training["selected_for_export"].astype(str).str.lower() == "true"].copy()
    if len(selected) != 16:
        raise ValueError(f"Expected 16 exported validation rows, found {len(selected)} in {path}")
    rows = []
    for record in selected.to_dict(orient="records"):
        env_slug = ENV_NAME_TO_SLUG[str(record["env"])]
        for k, column in LEVEL_COLUMNS.items():
            rows.append(
                {
                    "env": env_slug,
                    "learning_rate": LR_FLOAT_TO_VALUE[float(record["learning_rate"])],
                    "run_name": str(record["run_name"]),
                    "epoch": int(record["epoch"]),
                    "K": int(k),
                    "level_idx": LEVELS.index(k),
                    "val_loss": float(record[column]),
                    "val_rollout_loss": float(record["validate/rollout_loss"]),
                }
            )
    return pd.DataFrame(rows)


def _parse_schedule_counts(raw: object) -> dict[str, int]:
    if isinstance(raw, dict):
        return {str(key): int(value) for key, value in raw.items()}
    text = str(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    return {str(key): int(value) for key, value in parsed.items()}


def load_planning_results(root: Path) -> pd.DataFrame:
    rows = []
    for env_slug in ENV_ORDER:
        run_summary_paths = sorted((root / "rollouts").glob(f"{env_slug}*/[0-9][0-9][0-9]_*/summary.json"))
        raw_records = []
        for summary_path in run_summary_paths:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            record = dict(payload["run"])
            record["_source"] = str(summary_path)
            raw_records.append(record)
        summary = pd.DataFrame(raw_records)
        if summary.empty:
            raise ValueError(f"No per-run summaries found for {env_slug} under {root / 'rollouts'}")
        duplicated = summary.loc[summary.duplicated("name", keep=False)]
        for name, group in duplicated.groupby("name"):
            if group["success_rate"].nunique() != 1 or group["manifest_sha256"].nunique() != 1:
                raise ValueError(f"Conflicting duplicate fixed-K results for {name}: {group.to_dict(orient='records')}")
        summary = summary.drop_duplicates("name", keep="first")
        if len(summary) != 24:
            raise ValueError(
                f"Expected 24 unique fixed-K rows for {env_slug}, found {len(summary)} across {len(run_summary_paths)} files"
            )
        manifest_hashes = set(summary["manifest_sha256"].astype(str))
        if len(manifest_hashes) != 1 or "" in manifest_hashes:
            raise ValueError(f"Expected one immutable manifest hash for {env_slug}, got {manifest_hashes}")
        for record in summary.to_dict(orient="records"):
            match = RUN_RE.match(str(record["name"]))
            if not match:
                raise ValueError(f"Unexpected benchmark name {record['name']!r} in {record['_source']}")
            k = int(match.group("k"))
            expected_idx = LEVELS.index(k)
            counts = _parse_schedule_counts(record["schedule_level_counts"])
            if set(counts) != {str(expected_idx)}:
                raise ValueError(
                    f"Run {record['name']} was not fixed at level {expected_idx}: schedule counts={counts}"
                )
            episodes = int(record["episodes"])
            successes = int(round(float(record["success_rate"]) * episodes / 100.0))
            rows.append(
                {
                    "env": env_slug,
                    "learning_rate": LR_TOKEN_TO_VALUE[match.group("lr_token")],
                    "K": k,
                    "level_idx": expected_idx,
                    "success_rate": float(record["success_rate"]),
                    "successes": successes,
                    "episodes": episodes,
                    "manifest_sha256": str(record["manifest_sha256"]),
                    "bits_used_total": int(record["bits_used_total"]),
                    "plan_time_total_sec": float(record["plan_time_total_sec"]),
                    "wall_time_sec": float(record["wall_time_sec"]),
                    "output_json": str(record["output_json"]),
                    "schedule_level_counts": json.dumps(counts, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def assemble_rows(training: pd.DataFrame, planning: pd.DataFrame) -> pd.DataFrame:
    key = ["env", "learning_rate", "K", "level_idx"]
    merged = training.merge(planning, on=key, how="outer", validate="one_to_one", indicator=True)
    if len(merged) != 96 or set(merged["_merge"]) != {"both"}:
        raise ValueError(f"Expected a complete 96-cell join, got {merged['_merge'].value_counts().to_dict()}")
    merged = merged.drop(columns="_merge")
    merged["log_val_loss"] = np.log(merged["val_loss"])
    group = merged.groupby(["env", "K"], sort=False)
    merged["loss_quality_z"] = group["log_val_loss"].transform(
        lambda values: -(values - values.mean()) / values.std(ddof=0)
    )
    merged["success_centered"] = merged["success_rate"] - group["success_rate"].transform("mean")
    merged["loss_rank_within_env_k"] = group["val_loss"].rank(method="average", ascending=True)
    merged["success_rank_within_env_k"] = group["success_rate"].rank(method="average", ascending=False)
    return merged.sort_values(
        ["env", "learning_rate", "K"],
        key=lambda col: col.map({**{v: i for i, v in enumerate(ENV_ORDER)}, **{v: i for i, v in enumerate(LR_ORDER)}})
        if col.name in {"env", "learning_rate"}
        else col,
    ).reset_index(drop=True)


def pooled_rank_correlation(frame: pd.DataFrame) -> float:
    quality_ranks = frame.groupby(["env", "K", "bootstrap_id"] if "bootstrap_id" in frame else ["env", "K"])[
        "val_loss"
    ].rank(method="average", ascending=True)
    success_ranks = frame.groupby(["env", "K", "bootstrap_id"] if "bootstrap_id" in frame else ["env", "K"])[
        "success_rate"
    ].rank(method="average", ascending=False)
    return float(stats.pearsonr(quality_ranks, success_ranks).statistic)


def quality_slope(frame: pd.DataFrame) -> float:
    quality = frame["loss_quality_z"].to_numpy(float)
    outcome = frame["success_centered"].to_numpy(float)
    return float(np.dot(quality, outcome) / np.dot(quality, quality))


def bootstrap_strata(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    groups = [group for _, group in frame.groupby(["env", "K"], sort=False)]
    sampled = rng.integers(0, len(groups), size=(repeats, len(groups)))
    if statistic is pooled_rank_correlation:
        numerators = []
        quality_squares = []
        success_squares = []
        for group in groups:
            quality = stats.rankdata(group["val_loss"], method="average") - 2.5
            success = stats.rankdata(-group["success_rate"], method="average") - 2.5
            numerators.append(float(np.dot(quality, success)))
            quality_squares.append(float(np.dot(quality, quality)))
            success_squares.append(float(np.dot(success, success)))
        numerator = np.asarray(numerators)[sampled].sum(axis=1)
        quality_square = np.asarray(quality_squares)[sampled].sum(axis=1)
        success_square = np.asarray(success_squares)[sampled].sum(axis=1)
        values = numerator / np.sqrt(quality_square * success_square)
    elif statistic is quality_slope:
        numerators = np.asarray(
            [float(np.dot(group["loss_quality_z"], group["success_centered"])) for group in groups]
        )
        denominators = np.asarray(
            [float(np.dot(group["loss_quality_z"], group["loss_quality_z"])) for group in groups]
        )
        values = numerators[sampled].sum(axis=1) / denominators[sampled].sum(axis=1)
    else:
        raise ValueError(f"No vectorized stratum bootstrap for statistic {statistic.__name__}")
    return tuple(float(value) for value in np.nanquantile(values, [0.025, 0.975]))


def permutation_pvalue(
    frame: pd.DataFrame,
    observed: float,
    statistic: Callable[[pd.DataFrame], float],
    *,
    rng: np.random.Generator,
    repeats: int,
) -> float:
    groups = [group for _, group in frame.groupby(["env", "K"], sort=False)]
    if statistic is pooled_rank_correlation:
        quality = np.stack(
            [stats.rankdata(group["val_loss"], method="average") - 2.5 for group in groups]
        )
        success = np.stack(
            [stats.rankdata(-group["success_rate"], method="average") - 2.5 for group in groups]
        )
        denominator = math.sqrt(float(np.square(quality).sum() * np.square(success).sum()))
    elif statistic is quality_slope:
        quality = np.stack([group["loss_quality_z"].to_numpy(float) for group in groups])
        success = np.stack([group["success_centered"].to_numpy(float) for group in groups])
        denominator = float(np.square(quality).sum())
    else:
        raise ValueError(f"No vectorized permutation test for statistic {statistic.__name__}")
    extreme = 0
    chunk_size = 5_000
    for start in range(0, repeats, chunk_size):
        size = min(chunk_size, repeats - start)
        permutations = np.argsort(rng.random((size, len(groups), quality.shape[1])), axis=2)
        shuffled = np.take_along_axis(quality[None, :, :], permutations, axis=2)
        null = np.multiply(shuffled, success[None, :, :]).sum(axis=(1, 2)) / denominator
        extreme += int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return float((1 + extreme) / (repeats + 1))


def pair_concordance(frame: pd.DataFrame) -> dict[str, float | int]:
    concordant = discordant = ties = 0
    for _, group in frame.groupby(["env", "K"], sort=False):
        for left, right in combinations(group.to_dict(orient="records"), 2):
            loss_delta = float(left["val_loss"]) - float(right["val_loss"])
            success_delta = float(left["success_rate"]) - float(right["success_rate"])
            if success_delta == 0:
                ties += 1
            elif loss_delta * success_delta < 0:
                concordant += 1
            else:
                discordant += 1
    decided = concordant + discordant
    return {
        "concordant": concordant,
        "discordant": discordant,
        "planning_ties": ties,
        "decided_pairs": decided,
        "concordance_rate": float(concordant / decided) if decided else math.nan,
    }


def selection_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows = []
    for (env, k), group in frame.groupby(["env", "K"], sort=False):
        selected = group.loc[group["val_loss"].idxmin()]
        best_success = float(group["success_rate"].max())
        selected_success = float(selected["success_rate"])
        rows.append(
            {
                "env": env,
                "K": int(k),
                "loss_selected_lr": selected["learning_rate"],
                "loss_selected_success_rate": selected_success,
                "best_planning_success_rate": best_success,
                "regret_points": best_success - selected_success,
                "selected_is_planning_best": bool(selected_success == best_success),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "strata": int(len(table)),
        "top1_matches": int(table["selected_is_planning_best"].sum()),
        "top1_match_rate": float(table["selected_is_planning_best"].mean()),
        "mean_regret_points": float(table["regret_points"].mean()),
        "median_regret_points": float(table["regret_points"].median()),
        "max_regret_points": float(table["regret_points"].max()),
    }
    return table, summary


def within_checkpoint_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, lr), group in frame.groupby(["env", "learning_rate"], sort=False):
        raw = stats.spearmanr(-np.log(group["val_loss"]), group["success_rate"]).statistic
        relative = stats.spearmanr(group["loss_quality_z"], group["success_rate"]).statistic
        rows.append(
            {
                "env": env,
                "learning_rate": lr,
                "raw_loss_vs_success_spearman": float(raw),
                "head_relative_quality_vs_success_spearman": float(relative),
            }
        )
    return pd.DataFrame(rows)


def level_selection_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows = []
    for (env, lr), group in frame.groupby(["env", "learning_rate"], sort=False):
        selected = group.loc[group["val_loss"].idxmin()]
        best_success = float(group["success_rate"].max())
        best_ks = sorted(int(value) for value in group.loc[group["success_rate"] == best_success, "K"])
        selected_success = float(selected["success_rate"])
        rows.append(
            {
                "env": env,
                "learning_rate": lr,
                "lowest_raw_loss_K": int(selected["K"]),
                "lowest_loss_K_success_rate": selected_success,
                "best_planning_K": ",".join(str(value) for value in best_ks),
                "best_planning_success_rate": best_success,
                "regret_points": best_success - selected_success,
                "selected_is_planning_best": bool(int(selected["K"]) in best_ks),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "checkpoints": int(len(table)),
        "top1_matches": int(table["selected_is_planning_best"].sum()),
        "top1_match_rate": float(table["selected_is_planning_best"].mean()),
        "mean_regret_points": float(table["regret_points"].mean()),
        "median_regret_points": float(table["regret_points"].median()),
        "max_regret_points": float(table["regret_points"].max()),
    }
    return table, summary


def per_stratum_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, k), group in frame.groupby(["env", "K"], sort=False):
        rho = stats.spearmanr(-group["val_loss"], group["success_rate"]).statistic
        rows.append({"env": env, "K": int(k), "spearman": float(rho)})
    return pd.DataFrame(rows)


def write_plots(frame: pd.DataFrame, strata: pd.DataFrame, output_dir: Path) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    paths = []

    grid = sns.relplot(
        data=frame,
        x="loss_quality_z",
        y="success_rate",
        hue="K",
        style="learning_rate",
        col="env",
        col_order=ENV_ORDER,
        kind="scatter",
        height=4,
        aspect=0.9,
        palette="viridis",
        s=95,
    )
    grid.set_axis_labels("", "")
    grid.set_titles("{col_name}")
    grid.figure.supxlabel("Validation-head quality (within environment and K; higher is better)", y=0.03)
    grid.figure.supylabel("Fixed-K success (%)", x=0.01)
    grid.figure.suptitle("Matched validation loss vs fixed-K planning", y=0.98)
    grid.figure.subplots_adjust(bottom=0.18, top=0.82, wspace=0.12)
    path = plot_dir / "matched_loss_vs_success.png"
    grid.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    paths.append(str(path))

    pivot = strata.pivot(index="env", columns="K", values="spearman").reindex(index=ENV_ORDER, columns=LEVELS)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(pivot, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Within-environment/K rank correlation across learning rates\n(lower loss vs higher fixed-K success)")
    ax.set_xlabel("K")
    ax.set_ylabel("Environment")
    path = plot_dir / "per_env_k_spearman.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(len(ENV_ORDER), 2, figsize=(14, 15), constrained_layout=True)
    for row_idx, env in enumerate(ENV_ORDER):
        env_frame = frame.loc[frame["env"] == env]
        for col_idx, (value, title, fmt, cmap) in enumerate(
            (
                ("val_loss", "validation MSE", ".2g", "mako_r"),
                ("success_rate", "fixed-K success (%)", ".0f", "rocket"),
            )
        ):
            pivot = env_frame.pivot(index="learning_rate", columns="K", values=value).reindex(
                index=LR_ORDER, columns=LEVELS
            )
            sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=axes[row_idx, col_idx], cbar=True)
            axes[row_idx, col_idx].set_title(f"{env}: {title}")
            axes[row_idx, col_idx].set_xlabel("K")
            axes[row_idx, col_idx].set_ylabel("learning rate")
    path = plot_dir / "loss_and_success_heatmaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    return paths


def fmt(value: float, digits: int = 2) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: fmt(float(value), 3))
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in record) + " |" for record in display.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dense validation-head loss vs matched fixed-K planning.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--permutation-repeats", type=int, default=50_000)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    rows = assemble_rows(load_training_losses(args.training_csv), load_planning_results(args.root))
    rng = np.random.default_rng(20260721)
    pooled_rho = pooled_rank_correlation(rows)
    pooled_rho_ci = bootstrap_strata(
        rows, pooled_rank_correlation, rng=rng, repeats=args.bootstrap_repeats
    )
    pooled_rho_p = permutation_pvalue(
        rows,
        pooled_rho,
        pooled_rank_correlation,
        rng=rng,
        repeats=args.permutation_repeats,
    )
    slope = quality_slope(rows)
    slope_ci = bootstrap_strata(rows, quality_slope, rng=rng, repeats=args.bootstrap_repeats)
    slope_p = permutation_pvalue(
        rows,
        slope,
        quality_slope,
        rng=rng,
        repeats=args.permutation_repeats,
    )

    env_rows = []
    for env in ENV_ORDER:
        subset = rows.loc[rows["env"] == env].copy()
        rho = pooled_rank_correlation(subset)
        ci = bootstrap_strata(subset, pooled_rank_correlation, rng=rng, repeats=args.bootstrap_repeats)
        env_rows.append({"env": env, "rank_rho": rho, "ci_low": ci[0], "ci_high": ci[1]})
    env_table = pd.DataFrame(env_rows)

    k_rows = []
    for k in LEVELS:
        subset = rows.loc[rows["K"] == k].copy()
        rho = pooled_rank_correlation(subset)
        ci = bootstrap_strata(subset, pooled_rank_correlation, rng=rng, repeats=args.bootstrap_repeats)
        k_rows.append({"K": k, "rank_rho": rho, "ci_low": ci[0], "ci_high": ci[1]})
    k_table = pd.DataFrame(k_rows)

    concordance = pair_concordance(rows)
    selection_table, selection = selection_summary(rows)
    checkpoint_table = within_checkpoint_correlations(rows)
    level_selection_table, level_selection = level_selection_summary(rows)
    stratum_table = per_stratum_correlations(rows)
    plot_paths = write_plots(rows, stratum_table, args.root)

    rows.to_csv(args.root / "matched_cells.csv", index=False)
    env_table.to_csv(args.root / "per_environment_association.csv", index=False)
    k_table.to_csv(args.root / "per_k_association.csv", index=False)
    stratum_table.to_csv(args.root / "per_environment_k_association.csv", index=False)
    selection_table.to_csv(args.root / "loss_selected_lr_regret.csv", index=False)
    checkpoint_table.to_csv(args.root / "within_checkpoint_association.csv", index=False)
    level_selection_table.to_csv(args.root / "loss_selected_k_regret.csv", index=False)

    raw_checkpoint = checkpoint_table["raw_loss_vs_success_spearman"].replace([np.inf, -np.inf], np.nan).dropna()
    relative_checkpoint = checkpoint_table["head_relative_quality_vs_success_spearman"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    summary = {
        "design": {
            "cells": int(len(rows)),
            "environments": list(ENV_ORDER),
            "learning_rates": list(LR_ORDER),
            "levels": list(LEVELS),
            "episodes_per_cell": sorted(set(int(value) for value in rows["episodes"])),
            "seed": 42,
            "matched_predictor": "exported-checkpoint validate/pred_loss_l{level_idx}(K={K})",
            "matched_outcome": "success rate under fixed mpc/cem/rollout level K",
        },
        "pooled_within_env_k": {
            "rank_correlation": pooled_rho,
            "rank_correlation_stratum_bootstrap_95ci": list(pooled_rho_ci),
            "rank_correlation_stratified_permutation_p": pooled_rho_p,
            "success_point_slope_per_1sd_better_log_loss": slope,
            "slope_stratum_bootstrap_95ci": list(slope_ci),
            "slope_stratified_permutation_p": slope_p,
        },
        "pair_concordance": concordance,
        "per_environment": env_table.to_dict(orient="records"),
        "per_k": k_table.to_dict(orient="records"),
        "loss_selected_lr": selection,
        "loss_selected_k_within_checkpoint": level_selection,
        "within_checkpoint_across_k": {
            "raw_loss_quality_spearman_median": float(raw_checkpoint.median()),
            "raw_loss_quality_spearman_mean": float(raw_checkpoint.mean()),
            "head_relative_quality_spearman_median": float(relative_checkpoint.median()),
            "head_relative_quality_spearman_mean": float(relative_checkpoint.mean()),
        },
        "plots": plot_paths,
    }
    (args.root / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    direction = (
        "does not provide a reliable ordering signal"
        if pooled_rho_ci[0] <= 0 <= pooled_rho_ci[1]
        else "is directionally predictive"
    )
    report = f"""# Dense MWM validation loss vs fixed-K planning

## Bottom line

There are two different answers depending on what is being compared:

1. **Across checkpoints at the same environment and K:** matched per-head validation MSE **{direction}** for fixed-K planning success. The pooled within-environment/K rank correlation is **{fmt(pooled_rho)}** (stratum-bootstrap 95% CI **[{fmt(pooled_rho_ci[0])}, {fmt(pooled_rho_ci[1])}]**, stratified permutation **p={fmt(pooled_rho_p, 4)}**). A one-standard-deviation improvement in log validation loss corresponds to **{fmt(slope)} success points** (95% CI **[{fmt(slope_ci[0])}, {fmt(slope_ci[1])}]**).
2. **Across K levels inside one checkpoint:** raw head loss is a bad level selector. The median correlation between lower loss and higher planning success is **{fmt(float(raw_checkpoint.median()))}**; the lowest-loss K is planning-best in only **{level_selection['top1_matches']}/{level_selection['checkpoints']}** checkpoints and loses **{fmt(level_selection['mean_regret_points'])} success points** on average.

The practical model-selection test tells the same story: choosing the learning rate with the lowest validation loss for each environment/K also chooses a best fixed-K planner in **{selection['top1_matches']}/{selection['strata']}** cases ({100 * selection['top1_match_rate']:.1f}%), with mean regret **{fmt(selection['mean_regret_points'])} points** and worst regret **{fmt(selection['max_regret_points'])} points**.

## Experimental design

- Models: the 4 environments × 4 learning rates from the `hudm-mwm-dense-lr-sweep-20260704` release, using each run's exported best checkpoint.
- Levels: K = {', '.join(str(k) for k in LEVELS)}.
- Predictor: the exported epoch's matched `validate/pred_loss_lN(K=K)` value. This is a per-element latent prediction MSE.
- Outcome: 50-episode success rate with MPC, every CEM iteration, and every rollout transition fixed to the same level K.
- Pairing: all 24 cells within an environment use the same immutable seed-42 episode manifest.
- Primary estimand: within each environment/K, do the four learning-rate checkpoints rank the same way by lower validation loss and higher planning success? This avoids treating environment difficulty or the intrinsic scale of different heads as predictive signal.

## Association by environment

{markdown_table(env_table, ['env', 'rank_rho', 'ci_low', 'ci_high'])}

## Association by K

Each row below pools the four environments while retaining within-environment rank comparisons across the four learning rates.

{markdown_table(k_table, ['K', 'rank_rho', 'ci_low', 'ci_high'])}

## Pairwise and selection diagnostics

Among checkpoint pairs with different planning success, **{concordance['concordant']}/{concordance['decided_pairs']}** ({100 * concordance['concordance_rate']:.1f}%) are concordant: the checkpoint with lower matched validation loss plans better. There are **{concordance['planning_ties']}** additional pairs tied in planning success.

{markdown_table(selection_table, ['env', 'K', 'loss_selected_lr', 'loss_selected_success_rate', 'best_planning_success_rate', 'regret_points', 'selected_is_planning_best'])}

## Comparing levels inside one checkpoint

Raw head losses should be interpreted cautiously across K. Each value is an MSE over a different latent prefix, while the planner uses a terminal **sum** over K coordinates; changing K also changes the transition head and the goal-cost representation. Across the 16 checkpoints, the median within-checkpoint Spearman correlation between lower raw head loss and higher planning success is **{fmt(float(raw_checkpoint.median()))}**. After expressing each head loss relative to the other learning rates for the same environment/K, the median is **{fmt(float(relative_checkpoint.median()))}**.

If one nevertheless selects K by the lowest raw per-head validation MSE inside each checkpoint, it selects a best planning K in only **{level_selection['top1_matches']}/{level_selection['checkpoints']}** cases ({100 * level_selection['top1_match_rate']:.1f}%). The mean planning regret is **{fmt(level_selection['mean_regret_points'])} points**, and the worst is **{fmt(level_selection['max_regret_points'])} points**.

{markdown_table(level_selection_table, ['env', 'learning_rate', 'lowest_raw_loss_K', 'lowest_loss_K_success_rate', 'best_planning_K', 'best_planning_success_rate', 'regret_points', 'selected_is_planning_best'])}

Therefore, a conspicuously low or high loss at one K is useful as a training-health diagnostic, but it is not by itself evidence that this K will be correspondingly strong or weak for planning relative to the other levels. Fixed-K evaluation remains necessary.

## Important limitations

- There are only four independently trained checkpoints per environment (the four learning rates), so uncertainty remains large even with 96 model-level cells.
- Each planning cell has 50 episodes from one seed-42 manifest. Pairing improves comparisons, but this is not a multi-seed generalization study.
- Validation MSE measures one-step latent prediction on the validation distribution. Planning also depends on goal-distance geometry, candidate ranking, compounding rollout error, and CEM optimization.
- The exported checkpoint was selected by mean validation rollout loss across heads, not by the individual matched head.

## Artifacts

- `matched_cells.csv`: all 96 joined validation/planning cells.
- `per_environment_k_association.csv`: 24 small-sample rank correlations.
- `per_k_association.csv`: matched loss/planning association for each K pooled across environments.
- `loss_selected_lr_regret.csv`: whether the lowest-loss learning rate is the best planner.
- `loss_selected_k_regret.csv`: whether the lowest-loss K inside a checkpoint is its best planning K.
- `within_checkpoint_association.csv`: across-K correlations for each checkpoint.
- `analysis_summary.json`: machine-readable summary.
- `plots/`: matched scatter, per-stratum correlations, and loss/success heatmaps.
"""
    (args.root / "report.md").write_text(report, encoding="utf-8")
    print(args.root / "report.md")


if __name__ == "__main__":
    main()
