"""Build rollouts/tworoom_goal50_fixed_k96_vs_adaptive_frontier.png: goal50
counterpart of plot_tworoom_fixed_k96_vs_adaptive_frontier.py (goal25) --
does fixed K=96 (no scheduling) beat the adaptive-schedule frontier at a
harder task (goal_offset=50, horizon=4/receding_horizon=4, budget=100)?

Two Pareto frontiers, same env/manifest/horizon/episodes (TwoRoom, goal50,
horizon=4, 100 episodes, dense checkpoint mwm_paper10_tworoom_k96_120_144_168_192):

- "Fixed K=96" frontier: release20260728_tworoom_goal50_densek96_fixed_cem_grid_horizon4
  (25 runs, pop_size in {20,50,100,150,200} x n_iter in {5,10,15,20,30}).
- "Representative adaptive schedules" frontier: 6 diverse schedule shapes x
  the same 25 CEM settings (150 runs), from
  release20260728_tworoom_goal50_representative_schedules_horizon4 -- NOT
  the full 26-schedule sweep (not run at this horizon/goal, see that
  config's docstring for which 6 were chosen and why).
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


def make_plot(out_path: str) -> None:
    fixed_rows = load_rows(Path("reports/research/release20260728_tworoom_goal50_densek96_fixed_cem_grid_horizon4"))
    fixed_x = [r["bits_used_total"] / r["episodes"] / 1e6 for r in fixed_rows]
    fixed_y = [r["success_rate"] for r in fixed_rows]

    adaptive_rows = load_rows(Path("reports/research/release20260728_tworoom_goal50_representative_schedules_horizon4"))
    adaptive_x = [r["bits_used_total"] / r["episodes"] / 1e6 for r in adaptive_rows]
    adaptive_y = [r["success_rate"] for r in adaptive_rows]

    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    ax.scatter(adaptive_x, adaptive_y, color=ADAPTIVE_COLOR, s=10, alpha=0.15, linewidths=0, zorder=1)
    afx, afy = pareto_step(adaptive_x, adaptive_y)
    ax.step(afx, afy, where="post", color=ADAPTIVE_COLOR, linewidth=3.0, zorder=3,
            label="Representative adaptive schedules (pooled Pareto frontier, 6 schedules x 25 CEM settings)")
    ax.scatter(afx, afy, color=ADAPTIVE_COLOR, s=40, zorder=4, edgecolors="white", linewidths=0.8)

    ax.scatter(fixed_x, fixed_y, color=FIXED_COLOR, s=24, alpha=0.5, linewidths=0, zorder=2)
    ffx, ffy = pareto_step(fixed_x, fixed_y)
    ax.step(ffx, ffy, where="post", color=FIXED_COLOR, linewidth=3.0, zorder=5,
            label="Fixed K=96, no scheduling (Pareto frontier, 25 CEM settings)")
    ax.scatter(ffx, ffy, color=FIXED_COLOR, s=44, marker="s", zorder=6, edgecolors="white", linewidths=0.8)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Bits per episode (×10⁶)", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title(
        "TwoRoom  goal_offset=50  100 episodes  horizon 4\n"
        "fixed K=96 vs. representative adaptive schedules (same checkpoint, same CEM grid)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", frameon=True, facecolor="white",
              edgecolor="0.8", framealpha=0.95)
    fig.tight_layout()

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(out_path="rollouts/tworoom_goal50_fixed_k96_vs_adaptive_frontier.png")
