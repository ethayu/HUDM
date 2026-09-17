from __future__ import annotations

import json
import math
from pathlib import Path
import textwrap
from typing import Any

import numpy as np

from mwm.benchmark.analysis import env_label, float_metric, mean_metric, paired_rows, role_label, sorted_rows
from mwm.benchmark.pareto import _schedule_label, _strategy_color_map, pareto_frontier
from mwm.benchmark.plot_contract import (
    EFFICIENCY_RATIOS_PLOT,
    PAIRED_SUCCESS_DELTA_PLOT,
    SCHEDULE_LEVEL_USAGE_PLOT,
    SCHEDULE_USAGE_BY_ROLE_PLOT,
    SUCCESS_BY_ENV_ROLE_PLOT,
    SUCCESS_VS_COMPUTE_PLOT,
    SUCCESS_VS_WALL_TIME_PLOT,
)


def write_default_plots(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    compact: bool = False,
) -> list[str]:
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = sorted_rows(rows)
    plots: list[str] = []
    schedules = sorted({_schedule_label(row) for row in rows})
    strategy_colors = _strategy_color_map(schedules)
    role_order = {"upstream_lewm_converted": 0, "retrained_lewm_identity": 1, "mwm_scheduled": 2, "mwm_dense": 3}
    colors = {
        "upstream_lewm_converted": "#2f6fbb",
        "retrained_lewm_identity": "#7a5fb4",
        "mwm_scheduled": "#d76f1f",
        "mwm_dense": "#b279a2",
    }

    def _save(fig: Any, name: str) -> None:
        path = root / name
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(path))

    def _roles() -> list[str]:
        roles = sorted({str(row.get("role", "")) for row in rows if str(row.get("role", ""))}, key=lambda role: role_order.get(role, 99))
        return roles or [""]

    def _envs() -> list[str]:
        return sorted({str(row.get("env_id", "")) for row in rows if str(row.get("env_id", ""))})

    def _pareto_scatter(x_key: str, name: str, xlabel: str, title: str) -> None:
        valid = [
            row
            for row in rows
            if np.isfinite(float_metric(row.get(x_key), float("nan")))
            and np.isfinite(float_metric(row.get("success_rate"), float("nan")))
        ]
        frontier = pareto_frontier(valid, cost_key=x_key)
        frontier_objects = {id(row) for row in frontier}
        dominated = [row for row in valid if id(row) not in frontier_objects]
        fig, ax = plt.subplots(figsize=(10, 5.6))
        ax.scatter(
            [float_metric(row.get(x_key)) for row in dominated],
            [float_metric(row.get("success_rate")) for row in dominated],
            s=16,
            color="#64748b",
            alpha=0.07,
            edgecolors="none",
            rasterized=True,
            label=f"Dominated cells ({len(dominated):,})",
            zorder=1,
        )
        for schedule in schedules:
            schedule_rows = [row for row in valid if _schedule_label(row) == schedule]
            schedule_frontier = pareto_frontier(schedule_rows, cost_key=x_key)
            if not schedule_frontier:
                continue
            ax.plot(
                [float_metric(row.get(x_key)) for row in schedule_frontier],
                [float_metric(row.get("success_rate")) for row in schedule_frontier],
                color=strategy_colors[schedule],
                linewidth=1.65,
                alpha=0.78,
                marker="o",
                markersize=2.6,
                markeredgewidth=0,
                zorder=3,
            )
        if frontier:
            frontier_x = [float_metric(row.get(x_key)) for row in frontier]
            frontier_y = [float_metric(row.get("success_rate")) for row in frontier]
            ax.plot(
                frontier_x,
                frontier_y,
                color="#64748b",
                linewidth=1.35,
                linestyle=":",
                marker="o",
                markersize=4.5,
                markerfacecolor="none",
                markeredgecolor="#64748b",
                markeredgewidth=0.8,
                alpha=0.68,
                label=f"Global Pareto frontier · reference ({len(frontier):,})",
                zorder=2,
            )
        positive_x = [float_metric(row.get(x_key)) for row in valid if float_metric(row.get(x_key)) > 0]
        use_log = bool(positive_x and max(positive_x) / min(positive_x) >= 20)
        if use_log:
            ax.set_xscale("log")
        ax.set_xlabel(f"{xlabel} (log scale)" if use_log else xlabel)
        ax.set_ylabel("Success rate (%)")
        ax.set_title(title, loc="left", fontsize=13, fontweight="semibold")
        ax.set_ylim(-2, 102)
        ax.grid(True, which="major", color="#cbd5e1", alpha=0.45, linewidth=0.8)
        ax.grid(False, which="minor")
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        handles.insert(
            1,
            Line2D([0], [0], color="#2563eb", linewidth=1.65, marker="o", markersize=3, alpha=0.78),
        )
        labels.insert(1, "Per-strategy frontiers (see color key)")
        ax.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(1, 1.13),
            frameon=False,
            ncol=3,
            fontsize=9,
        )
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, name)

    _pareto_scatter(
        "dynamics_flops_total",
        SUCCESS_VS_COMPUTE_PLOT,
        "Audited dynamics FLOPs (lower is better)",
        "Success vs audited dynamics compute",
    )
    _pareto_scatter(
        "wall_time_sec",
        SUCCESS_VS_WALL_TIME_PLOT,
        "Wall time in seconds (lower is better)",
        "Success vs wall time",
    )

    def _strategy_legend() -> None:
        rows_per_column = max(1, math.ceil(len(schedules) / 2))
        fig, ax = plt.subplots(figsize=(14, max(4.8, rows_per_column * 0.56)))
        ax.set_axis_off()
        ax.text(
            0.01,
            1.02,
            "Strategy color key",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="semibold",
        )
        for index, schedule in enumerate(schedules):
            column = index // rows_per_column
            row_index = index % rows_per_column
            x = 0.01 + column * 0.5
            y = 0.94 - row_index * (0.88 / max(1, rows_per_column - 1))
            ax.plot(
                [x, x + 0.025],
                [y, y],
                transform=ax.transAxes,
                color=strategy_colors[schedule],
                linewidth=3.2,
                solid_capstyle="round",
            )
            label = "\n".join(textwrap.wrap(schedule, width=62))
            ax.text(
                x + 0.034,
                y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=8.5,
            )
        _save(fig, "strategy_legend.png")

    _strategy_legend()

    if compact:
        return plots

    envs = _envs()
    roles = _roles()
    if envs and any(roles):
        fig, ax = plt.subplots(figsize=(max(6, len(envs) * 2.2), 4.2))
        x = np.arange(len(envs), dtype=float)
        width = min(0.34, 0.76 / max(1, len(roles)))
        offsets = (np.arange(len(roles), dtype=float) - (len(roles) - 1) / 2.0) * width
        for idx, role in enumerate(roles):
            means = [
                mean_metric(row.get("success_rate") for row in rows if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == role)
                for env_id in envs
            ]
            centers = x + offsets[idx]
            ax.bar(centers, means, width=width, label=role_label(role), color=colors.get(role), alpha=0.82)
            for env_idx, env_id in enumerate(envs):
                seed_rows = [
                    row
                    for row in rows
                    if str(row.get("env_id", "")) == env_id and str(row.get("role", "")) == role
                ]
                jitter = np.linspace(-width * 0.25, width * 0.25, max(1, len(seed_rows)))
                for j_idx, row in enumerate(seed_rows):
                    y = float_metric(row.get("success_rate"), float("nan"))
                    if not np.isnan(y):
                        ax.scatter(centers[env_idx] + jitter[j_idx], y, color="#172026", s=22, zorder=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([env_label(env_id) for env_id in envs])
        ax.set_ylabel("success rate (%)")
        ax.set_title("Mean success by environment and role")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, SUCCESS_BY_ENV_ROLE_PLOT)

    pairs = paired_rows(rows)
    if pairs:
        multiple_roles = len({str(pair["comparison_role"]) for pair in pairs}) > 1
        labels = [
            f"{env_label(pair['env_id'])} s{pair['seed']}"
            + (f"\n{role_label(str(pair['comparison_role']))}" if multiple_roles else "")
            for pair in pairs
        ]
        deltas = [float_metric(pair["delta_success"], float("nan")) for pair in pairs]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.72), 4.2))
        x = np.arange(len(labels), dtype=float)
        bar_colors = ["#0f7b3f" if delta > 0 else "#b42318" if delta < 0 else "#627282" for delta in deltas]
        ax.axhline(0, color="#172026", linewidth=1)
        ax.bar(x, deltas, color=bar_colors, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("comparison - upstream success (percentage points)")
        ax.set_title("Paired success delta by seed and role")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, PAIRED_SUCCESS_DELTA_PLOT)

        ratio_labels = list(labels)
        wall = [float_metric(pair["wall_ratio"], float("nan")) for pair in pairs]
        compute = [float_metric(pair["compute_ratio"], float("nan")) for pair in pairs]
        fig, ax = plt.subplots(figsize=(max(7, len(ratio_labels) * 0.72), 4.2))
        x = np.arange(len(ratio_labels), dtype=float)
        width = 0.36
        ax.axhline(1.0, color="#172026", linewidth=1)
        ax.bar(x - width / 2, wall, width=width, label="wall-time ratio", color="#6b7f2a", alpha=0.82)
        ax.bar(x + width / 2, compute, width=width, label="compute ratio", color="#7a4e9f", alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(ratio_labels, rotation=35, ha="right")
        ax.set_ylabel("comparison / upstream")
        ax.set_title("Efficiency ratios by paired seed and role")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, EFFICIENCY_RATIOS_PLOT)

    level_totals: dict[str, int] = {}
    for row in rows:
        try:
            counts = json.loads(str(row.get("schedule_level_counts", "{}")))
        except json.JSONDecodeError:
            counts = {}
        for level, count in counts.items():
            level_totals[str(level)] = level_totals.get(str(level), 0) + int(count)
    if level_totals:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = sorted(
            level_totals,
            key=lambda x: (0, int(str(x))) if str(x).isdigit() else (1, str(x)),
        )
        ax.bar(labels, [level_totals[k] for k in labels])
        ax.set_xlabel("base fidelity level")
        ax.set_ylabel("CEM cost calls")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, SCHEDULE_LEVEL_USAGE_PLOT)

        grouped_counts: dict[tuple[str, str], dict[str, int]] = {}
        for row in rows:
            env_id = str(row.get("env_id", ""))
            role = str(row.get("role", ""))
            try:
                counts = json.loads(str(row.get("schedule_level_counts", "{}")))
            except json.JSONDecodeError:
                counts = {}
            slot = grouped_counts.setdefault((env_id, role), {})
            for level, count in counts.items():
                slot[str(level)] = slot.get(str(level), 0) + int(count)
        if grouped_counts:
            groups = sorted(grouped_counts, key=lambda item: (item[0], role_order.get(item[1], 99), item[1]))
            labels = [f"{env_label(env_id)}\n{role_label(role)}" for env_id, role in groups]
            levels = sorted(
                {level for counts in grouped_counts.values() for level in counts},
                key=lambda x: (0, int(str(x))) if str(x).isdigit() else (1, str(x)),
            )
            fig, ax = plt.subplots(figsize=(max(7, len(groups) * 0.86), 4.4))
            x = np.arange(len(groups), dtype=float)
            bottom = np.zeros(len(groups), dtype=float)
            palette = ["#2f6fbb", "#d76f1f", "#0f7b3f", "#7a4e9f", "#64748b", "#c2410c"]
            for idx, level in enumerate(levels):
                vals = np.array([grouped_counts[group].get(level, 0) for group in groups], dtype=float)
                ax.bar(x, vals, bottom=bottom, label=str(level), color=palette[idx % len(palette)], alpha=0.84)
                bottom += vals
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel("CEM cost calls")
            ax.set_title("Schedule level usage by environment and role")
            ax.legend(title="level", loc="best", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            _save(fig, SCHEDULE_USAGE_BY_ROLE_PLOT)
    return plots


__all__ = ["write_default_plots"]
