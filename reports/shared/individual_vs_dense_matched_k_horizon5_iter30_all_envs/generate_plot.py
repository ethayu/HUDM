"""Regenerate plot.png from the data/ in this folder.

Individually-trained-K vs. dense-matched-K success rate, all four
environments (TwoRoom, OGB Cube, Reacher, PushT), at the full
K=[48,72,96,120,144,168,192] range -- for BOTH goal_offset=25 (solid lines,
filled markers) and goal_offset=50 (dashed lines, hollow markers), overlaid
on the same 4 panels. Same fixed eval protocol otherwise: horizon=5,
receding_horizon=5, pop_size=300, elite_frac=0.1 (30 elites), n_iter=30, 100
episodes. See the sibling configs under
configs/research/{env}_goal{25,50}_individual_vs_dense_matched_k_horizon5_iter30.yaml
(K=96-192), configs/research/{env}_goal{25,50}_individual_k48_72_horizon5_iter30.yaml
(individually-trained K=48/72), and
configs/research/{env}_goal{25,50}_sepopt_dense_k48_72_horizon5_iter30.yaml
(dense K=48/72) in the main repo for the exact run definitions.

"Individually-trained":
- K=96-192: checkpoints_mwm/mwm_paper10_{env}_k{K}_release20260728
- K=48/72: checkpoints_single_k48to192_sepopt_retrain_20260922_{env}/k{K}/checkpoints_mwm/mwm_paper10_{env}_k{K}_sepopt_retrain_20260920
  -- a SEPARATE from-scratch single-K training run (2026-09-22 release);
  release20260728 never trained standalone K=48/72 models, so this fills
  that gap with a real (not placeholder) individually-trained checkpoint.

"Dense (matched K)":
- K=96-192: checkpoints_mwm/mwm_paper10_{env}_k96_120_144_168_192_release20260728
  (K=[96..192] matryoshka checkpoint), held fixed at that K, no scheduling.
- K=48/72: the SEPARATE sepopt K=[48,72,96,120,144,168,192] matryoshka
  checkpoint used elsewhere in this project, held fixed at that K, no
  scheduling. Different training run than the K=96-192 dense checkpoint
  above, so its own K=96 value doesn't need to (and doesn't exactly) match
  the K=96-192 series at the K=96 seam -- both are plotted as one connected
  line per goal for readability, but treat the two checkpoint families as
  what they are.

Color always encodes series type (individual / dense); linestyle +
filled-vs-hollow markers always encode goal_offset (solid+filled = 25,
dashed+hollow = 50) -- consistent across both series types, so the 25 vs.
50 comparison reads the same way everywhere in the figure.

Usage: python generate_plot.py  (writes plot.png in this directory)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

ORANGE = "#eb6834"
BLUE = "#2a78d6"
GRID = "#d9d9d6"

LEVELS = [96, 120, 144, 168, 192]
EXT_LEVELS = [48, 72]
ALL_LEVELS = EXT_LEVELS + LEVELS

ENVS = [
    ("tworoom", "TwoRoom"),
    ("ogb_cube", "OGB Cube"),
    ("reacher", "Reacher"),
    ("pusht", "PushT"),
]

GOALS = [
    (25, "", "-", True),   # (goal_offset, data filename suffix, linestyle, filled)
    (50, "_goal50", "--", False),
]


def load_success_by_k(env: str, suffix: str) -> tuple[dict[int, float], dict[int, float]]:
    """K=96-192 paired series: individually-trained vs. dense-matched."""
    data = json.load(open(HERE / "data" / f"{env}{suffix}_summary.json"))
    individual: dict[int, float] = {}
    dense: dict[int, float] = {}
    for run in data["runs"]:
        name = run["name"]
        if name.startswith("individual_k"):
            k = int(name.removeprefix("individual_k"))
            individual[k] = run["success_rate"]
        elif name.startswith("dense_matched_k"):
            k = int(name.removeprefix("dense_matched_k"))
            dense[k] = run["success_rate"]
    return individual, dense


def load_individual_k48_72(env: str, suffix: str) -> dict[int, float]:
    """Real individually-trained K=48/72, from the SEPARATE from-scratch
    single-K retrain release (2026-09-22) -- release20260728 never trained
    standalone models at these levels."""
    data = json.load(open(HERE / "data" / f"{env}{suffix}_individual_k48_72_summary.json"))
    out: dict[int, float] = {}
    for run in data["runs"]:
        name = run["name"]
        if name.startswith("individual_k"):
            k = int(name.removeprefix("individual_k"))
            out[k] = run["success_rate"]
    return out


def load_dense_k48_72(env: str, suffix: str) -> dict[int, float]:
    """Dense K=48/72, from the SEPARATE sepopt K=[48..192] checkpoint (not
    the release20260728 K=[96..192] dense checkpoint used for K=96-192)."""
    fname = f"{env}_sepopt_k48_72_summary.json" if not suffix else f"{env}{suffix}_sepopt_k48_72_summary.json"
    data = json.load(open(HERE / "data" / fname))
    out: dict[int, float] = {}
    for run in data["runs"]:
        name = run["name"]
        if name.startswith("sepopt_dense_k"):
            k = int(name.removeprefix("sepopt_dense_k"))
            out[k] = run["success_rate"]
    return out


def main() -> None:
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 12), dpi=150, sharex=True)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, (env, label) in zip(axes, ENVS):
        ax.set_facecolor("#fcfcfb")

        for goal, suffix, ls, filled in GOALS:
            individual, dense = load_success_by_k(env, suffix)
            individual.update(load_individual_k48_72(env, suffix))
            dense.update(load_dense_k48_72(env, suffix))

            mfc_o = ORANGE if filled else "white"
            mfc_b = BLUE if filled else "white"

            ax.plot(ALL_LEVELS, [individual[k] for k in ALL_LEVELS], color=ORANGE, marker="o",
                     lw=2, ms=6, ls=ls, markerfacecolor=mfc_o,
                     label=f"Individually-trained K (goal={goal})")
            ax.plot(ALL_LEVELS, [dense[k] for k in ALL_LEVELS], color=BLUE, marker="s",
                     lw=2, ms=6, ls=ls, markerfacecolor=mfc_b,
                     label=f"Dense (matched K, goal={goal})")

        ax.set_ylim(-5, 105)
        ax.set_ylabel("Success rate (%)", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left")
        ax.grid(True, color=GRID, lw=0.8)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=9, framealpha=0.95,
               ncol=2, bbox_to_anchor=(0.5, -0.01))
    axes[-1].set_xlabel("K (fidelity level)", fontsize=11)
    axes[-1].set_xticks(ALL_LEVELS)

    fig.suptitle(
        "Individually-trained K vs. dense-matched K (no scheduling)\n"
        "goal=25 (solid/filled) vs. goal=50 (dashed/hollow)\n"
        "100 episodes, horizon=5, pop=300/elite=30/iter=30",
        fontsize=11.5, fontweight="bold", y=0.998,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    out = HERE / "plot.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
