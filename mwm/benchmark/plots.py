from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mwm.benchmark.analysis import env_label, float_metric, mean_metric, paired_rows, role_label, sorted_rows


def write_default_plots(output_dir: str | Path, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = sorted_rows(rows)
    plots: list[str] = []
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
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots.append(str(path))

    def _roles() -> list[str]:
        roles = sorted({str(row.get("role", "")) for row in rows if str(row.get("role", ""))}, key=lambda role: role_order.get(role, 99))
        return roles or [""]

    def _envs() -> list[str]:
        return sorted({str(row.get("env_id", "")) for row in rows if str(row.get("env_id", ""))})

    def _scatter(x_key: str, y_key: str, name: str, xlabel: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        roles = _roles()
        for role in roles:
            role_rows = [row for row in rows if str(row.get("role", "")) == role]
            x = [float_metric(row.get(x_key), float("nan")) for row in role_rows]
            y = [float_metric(row.get(y_key), float("nan")) for row in role_rows]
            ax.scatter(x, y, label=role_label(role) if role else "runs", color=colors.get(role), alpha=0.85)
            if len(rows) <= 18:
                for row, x_val, y_val in zip(role_rows, x, y):
                    if not np.isnan(x_val) and not np.isnan(y_val):
                        label = f"{env_label(str(row.get('env_id', '')))} s{int(row.get('seed', 0))}"
                        ax.annotate(label, (x_val, y_val), fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("success rate (%)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, name)

    _scatter("bits_used_total", "success_rate", "success_vs_compute.png", "latent work bits")
    _scatter("wall_time_sec", "success_rate", "success_vs_wall_time.png", "wall time (sec)")

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
        _save(fig, "success_by_env_role.png")

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
        _save(fig, "paired_success_delta.png")

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
        _save(fig, "efficiency_ratios.png")

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
        _save(fig, "schedule_level_usage.png")

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
            _save(fig, "schedule_usage_by_role.png")
    return plots


__all__ = ["write_default_plots"]
