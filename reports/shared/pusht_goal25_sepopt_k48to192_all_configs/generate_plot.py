"""Regenerate plot.png from the data/ in this folder.

Same structure/plotting conventions as the tworoom_goal25_sepopt_k48to192_all_configs
and reacher_goal25_sepopt_k48to192_all_configs figures, for PushT goal_offset=25
(horizon=5, budget=50) on the sepopt K48-192 checkpoint.

STATUS: COMPLETE. The full-range (K=48-192) adaptive-schedule sweep (26
schedules x 22 CEM combos = 572 cells) and the individually-trained fixed-K
baselines (27_all_fixed_finest = K=192, 28-31 = K=168/144/120/96; 5 levels x
22 combos = 110 cells) all finished: 682/682. This regenerates the earlier
partial version (460/680 at generation time, no fixed-K baselines at all)
with final numbers and the fixed-K comparison series added.

Individually-trained fixed-K baselines: checkpoints_mwm/mwm_paper10_pusht_k{K}_release20260728
(K=96/120/144/168/192), run FRESH in this same sweep at horizon=5 on the
current pinned n100 manifest -- not reused/manifest-drifted from an older,
incompatible sweep. K=192 (schedule 27, the checkpoint's own finest level)
is plotted as its own series since it's the natural cost-matched baseline
for the adaptive frontier's top end; K=168/144/120/96 (schedules 28-31) are
plotted together as a second series.

data/fixed_k_baselines_summary.json (125 cells, if present) is NOT plotted:
it's the OLD pre-checkpoint-swap sweep (old checkpoint naming, horizon=2,
unsuffixed non-pinned manifest) -- superseded by the fresh baselines above.

Sepopt-checkpoint fixed-K ablations (SAME checkpoint as the adaptive
frontier, held fixed at one level for the whole plan, full CEM grid --
pop_size:[20,50,100,150,200] x n_iter:[5,10,15,20,30], 25 combos/level):
K=120/144/168/192 (levels 3-6), 100 cells total, complete. This is the
same-checkpoint "does scheduling help *this* checkpoint" ablation that
TwoRoom/Reacher already have (there just for K=192/K=96); PushT gets all
four upper levels since that's where the K120-192 probe showed the adaptive
frontier's action concentrated. See data/sepopt_k{120,144,168,192}_fixed_cem_grid_summary.json.

K48-192 full-range horizon=5 was itself chosen after a separate K120-192
restricted probe at horizon=5 (data/sepopt_k120_192_probe_summary.json,
12/12 complete, same checkpoint/manifest/horizon) showed a dramatic
improvement over the same K-range probe at horizon=2 (84% vs 79% best
success) -- that finding motivated relaunching the full K-range at
horizon=5 rather than horizon=2. The probe itself is NOT plotted here (by
request) -- the frontier below reflects only the full K=48-192 adaptive
schedules.

Usage: python generate_plot.py  (writes plot.png in this directory)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

BLUE = "#2a78d6"
BLACK = "#111111"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRID = "#d9d9d6"


def bits_per_ep(run: dict) -> float:
    episodes = run.get("episodes") or 100
    return run.get("bits_used_total", 0) / episodes / 1e6


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left frontier: maximize success rate, minimize cost."""
    pts_sorted = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier = []
    best_sr = -1.0
    for bits, sr in pts_sorted:
        if sr > best_sr:
            frontier.append((bits, sr))
            best_sr = sr
    return frontier


def main() -> None:
    adaptive = json.load(open(HERE / "data" / "sepopt_k48to192_adaptive_summary.json"))
    fixed_k192 = json.load(open(HERE / "data" / "fixed_k192_baseline_summary.json"))
    fixed_lt192 = json.load(open(HERE / "data" / "fixed_k_lt192_baseline_summary.json"))
    sepopt_fixed_k = []
    for K in (120, 144, 168, 192):
        sepopt_fixed_k += json.load(open(HERE / "data" / f"sepopt_k{K}_fixed_cem_grid_summary.json"))["runs"]

    n_done = adaptive.get("runs_completed", len(adaptive["runs"]))
    n_target = adaptive.get("runs_target", len(adaptive["runs"]))

    adaptive_pts = [(bits_per_ep(r), r["success_rate"]) for r in adaptive["runs"]]
    fixed_k192_pts = [(bits_per_ep(r), r["success_rate"]) for r in fixed_k192["runs"]]
    fixed_lt192_pts = [(bits_per_ep(r), r["success_rate"]) for r in fixed_lt192["runs"]]
    sepopt_fixed_k_pts = [(bits_per_ep(r), r["success_rate"]) for r in sepopt_fixed_k]
    frontier = pareto_frontier(adaptive_pts)

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    ax.scatter(*zip(*fixed_lt192_pts), s=14, color=ORANGE, alpha=0.35, marker="o", zorder=2,
               linewidths=0, label="Single-$d$ (K<192)")
    ax.scatter(*zip(*fixed_k192_pts), s=28, color=BLACK, alpha=0.55, marker="s", zorder=3,
               linewidths=0, label="Single-$d$ (K=192)")
    ax.scatter(*zip(*sepopt_fixed_k_pts), s=20, color=AQUA, alpha=0.45, marker="^", zorder=4,
               linewidths=0, label="MWM (fixed)")

    ax.scatter(*zip(*adaptive_pts), s=14, color=BLUE, alpha=0.25, zorder=5, linewidths=0)
    ax.plot(*zip(*frontier), color=BLUE, lw=2, zorder=7)
    ax.scatter(*zip(*frontier), s=45, color=BLUE, zorder=8, linewidths=0)

    ax.scatter([], [], s=45, color=BLUE, linewidths=0,
               label="MWM (scheduled) (Pareto frontier)")

    ax.set_xlabel("Bits per episode (×10$^6$)", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=18)
    ax.set_title(
        r"PushT  $\Delta$=25",
        fontsize=20, fontweight="bold",
    )
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95)

    fig.tight_layout()

    out = HERE / "plot.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
