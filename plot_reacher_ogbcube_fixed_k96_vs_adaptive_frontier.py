"""Build rollouts/reacher_ogbcube_goal25_fixed_k96_vs_adaptive_frontier.png:
same question as plot_tworoom_fixed_k96_vs_adaptive_frontier.py, for Reacher
and OGB Cube -- does simply fixing K=96 for the whole plan (no fidelity
transitions at all) beat the pooled adaptive/annealing schedule frontier at
matched compute?

Two Pareto frontiers per env, same env/manifest/horizon/episodes
(goal25, horizon=2, 100 episodes, dense checkpoint
mwm_paper10_<env>_k96_120_144_168_192):

- "Fixed K=96" frontier: release20260728_<env>_goal25_densek96_fixed_cem_grid
  (25 runs, pop_size in {20,50,100,150,200} x n_iter in {5,10,15,20,30}, MPC/
  CEM/Rollout all pinned to level 0, no scheduling at all).
- "Adaptive schedules" frontier: pooled Pareto frontier across all
  changing-fidelity dense schedules x the same 25 pop/n_iter combos, from
  release20260728_dense_<env>_all_fidelity_schedules.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIXED_COLOR = "#111214"
ADAPTIVE_COLOR = "#2a5f9e"

ENVS = [("reacher", "Reacher"), ("ogb_cube", "OGB Cube")]


def load_rows(output_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(output_dir.iterdir()):
        summary = d / "summary.json"
        if not d.is_dir() or not summary.exists():
            continue
        rows.append(json.loads(summary.read_text())["run"])
    return rows


def pareto_step(xs: list[float], ys: list[float]):
    order = np.argsort(xs)
    xs_s, ys_s = np.array(xs)[order], np.array(ys)[order]
    px, py = [xs_s[0]], [ys_s[0]]
    best = ys_s[0]
    for x, y in zip(xs_s[1:], ys_s[1:]):
        if y >= best:
            best = y
            px.append(x)
            py.append(y)
    return px, py


def make_plot(out_path: str, x_max: float = 60) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))

    for (env_key, env_title), ax in zip(ENVS, axes.flat):
        fixed_rows = load_rows(Path(f"reports/research/release20260728_{env_key}_goal25_densek96_fixed_cem_grid"))
        fixed_x = [r["bits_used_total"] / r["episodes"] / 1e6 for r in fixed_rows]
        fixed_y = [r["success_rate"] for r in fixed_rows]

        adaptive_rows = load_rows(Path(f"reports/research/release20260728_dense_{env_key}_all_fidelity_schedules"))
        adaptive_rows = [r for r in adaptive_rows if not str(r.get("schedule", "")).strip().startswith("single K=")]
        adaptive_x = [r["bits_used_total"] / r["episodes"] / 1e6 for r in adaptive_rows]
        adaptive_y = [r["success_rate"] for r in adaptive_rows]

        ax.scatter(adaptive_x, adaptive_y, color=ADAPTIVE_COLOR, s=10, alpha=0.15, linewidths=0, zorder=1)
        afx, afy = pareto_step(adaptive_x, adaptive_y)
        ax.step(afx, afy, where="post", color=ADAPTIVE_COLOR, linewidth=3.0, zorder=3,
                label="Adaptive schedules (pooled Pareto frontier)")
        ax.scatter(afx, afy, color=ADAPTIVE_COLOR, s=40, zorder=4, edgecolors="white", linewidths=0.8)

        ax.scatter(fixed_x, fixed_y, color=FIXED_COLOR, s=24, alpha=0.5, linewidths=0, zorder=2)
        ffx, ffy = pareto_step(fixed_x, fixed_y)
        ax.step(ffx, ffy, where="post", color=FIXED_COLOR, linewidth=3.0, zorder=5,
                label="Fixed K=96, no scheduling (Pareto frontier)")
        ax.scatter(ffx, ffy, color=FIXED_COLOR, s=44, marker="s", zorder=6, edgecolors="white", linewidths=0.8)

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 103)
        ax.set_xlabel("Bits per episode (×10⁶)", fontsize=11)
        ax.set_title(f"{env_title}  goal_offset=25  100 episodes  horizon 2", fontsize=12.5, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", frameon=True, facecolor="white",
                  edgecolor="0.8", framealpha=0.95)

    axes[0].set_ylabel("Success rate (%)", fontsize=11)
    fig.suptitle(
        "Fixed K=96 vs. pooled adaptive schedules (same checkpoint, same CEM grid)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(out_path="rollouts/reacher_ogbcube_goal25_fixed_k96_vs_adaptive_frontier.png")
