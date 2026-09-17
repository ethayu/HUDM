"""Rebuild rollouts/tworoom_h2_100ep_all_configs.png, adding the isolated K=144
single-level checkpoint sweep alongside the existing K=192 config grid.

Note: the K=192 series were evaluated at goal_offset=30; the K=144 sweep
(grid_k144_tworoom_h2_goal25_100ep_mpcfix_cemfix_rolfix) is the only full
pop/niter grid available for the isolated K=144 TwoRoom checkpoint and was
run at goal_offset=25. It is plotted with a distinct marker/linestyle and
labeled with its goal offset so the two settings aren't visually conflated.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("rollouts/cem_sweep")
POP_SIZES = [20, 50, 100, 150, 200]
N_ITERS = [5, 10, 20, 30, 50]

# (label, dir_name, color) in the same order/colors as the original plot's
# default matplotlib tab10 cycle.
SERIES_K192 = [
    ("mpcfix_cemfix_rolfix", "grid_tworoom_goal30_h2_100ep_mpcfix_cemfix_rolfix", "#1f77b4"),
    ("mpcfix_cemlin_rolfix", "grid_tworoom_goal30_h2_100ep_mpcfix_cemlin_rolfix", "#d62728"),
    ("mpcfix_cemfix_rollin", "grid_tworoom_goal30_h2_100ep_mpcfix_cemfix_rollin", "#2ca02c"),
    ("mpclin_cemfix_rolfix", "grid_tworoom_goal30_h2_100ep_mpclin_cemfix_rolfix", "#ff7f0e"),
    ("mpclin_cemlin_rolfix", "grid_tworoom_goal30_h2_100ep_mpclin_cemlin_rolfix", "#9467bd"),
    ("mpclin_cemcasc_rolfix", "grid_tworoom_goal30_h2_100ep_mpclin_cemcasc_rolfix", "#8c564b"),
    ("mpclin_cemfwd_rolfix", "grid_tworoom_goal30_h2_100ep_mpclin_cemfwd_rolfix", "#e377c2"),
]

SERIES_K144 = ("K=144 mpcfix_cemfix_rolfix (goal25)",
               "grid_k144_tworoom_h2_goal25_100ep_mpcfix_cemfix_rolfix", "#7f7f7f")


def load_point(dir_name: str, pop: int, n_iter: int):
    p = ROOT / dir_name / "tworoom" / f"pop{pop}_niter{n_iter}" / "eval.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    episodes = d["episodes"]
    bits_per_ep = d["planning_diagnostics"]["bits_used_total"] / episodes
    success_rate = d["swm_results"]["success_rate"]
    return bits_per_ep / 1e6, success_rate


def pareto_step(xs, ys):
    order = np.argsort(xs)
    xs_s, ys_s = np.array(xs)[order], np.array(ys)[order]
    px, py = [xs_s[0]], [ys_s[0]]
    best = ys_s[0]
    for x, y in zip(xs_s[1:], ys_s[1:]):
        if y >= best:
            best = y
            px.append(x)
            py.append(y)
    return px, py, xs_s, ys_s


fig, ax = plt.subplots(figsize=(10, 5))

for label, dir_name, color in SERIES_K192:
    xs, ys = [], []
    for pop in POP_SIZES:
        for n in N_ITERS:
            pt = load_point(dir_name, pop, n)
            if pt is not None:
                xs.append(pt[0])
                ys.append(pt[1])
    px, py, xs_s, ys_s = pareto_step(xs, ys)
    ax.scatter(xs_s, ys_s, color=color, s=45, alpha=0.75, zorder=3)
    ax.step(px, py, where="post", color=color, linewidth=2.0, alpha=0.9, label=label)

ax.set_xlabel("Bits per episode (×10⁶)", fontsize=12)
ax.set_ylabel("Success rate (%)", fontsize=12)
ax.set_title("TwoRoom h=2, 100 episodes — all configs", fontsize=13, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(50, 103)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()

out = Path("rollouts/tworoom_h2_100ep_all_configs.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")
