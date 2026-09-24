"""Paper version of plot.png: 1x4 panels, paper terminology, no baked-in title.

Same data and loaders as generate_plot.py. Writes a vector PDF sized for the
ICLR text width (5.5in).

Usage: python generate_paper_plot.py [OUTPUT_PDF]
  (default: fixed_level_planning.pdf in this directory)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from generate_plot import (
    ALL_LEVELS, BLUE, GOALS, GRID, HERE, ORANGE,
    load_dense_k48_72, load_individual_k48_72, load_success_by_k,
)

ENVS = [
    ("tworoom", "TwoRoom"),
    ("pusht", "PushT"),
    ("ogb_cube", "OGBench-Cube"),
    ("reacher", "Reacher"),
]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fixed_level_planning.pdf"
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.75), sharey=True)
    for ax, (env, label) in zip(axes, ENVS):
        for goal, suffix, ls, filled in GOALS:
            individual, dense = load_success_by_k(env, suffix)
            individual.update(load_individual_k48_72(env, suffix))
            dense.update(load_dense_k48_72(env, suffix))
            for series, color, marker in ((individual, ORANGE, "o"), (dense, BLUE, "s")):
                ax.plot(ALL_LEVELS, [series[k] for k in ALL_LEVELS], color=color, marker=marker,
                        lw=1.1, ms=3, ls=ls, markerfacecolor=color if filled else "white",
                        markeredgewidth=0.8)
        ax.set_title(label, fontsize=8)
        ax.set_ylim(-5, 105)
        ax.set_xticks(ALL_LEVELS[::2])
        ax.set_xlabel(r"Level $d$")
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(length=2, pad=1.5)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[0].set_ylabel("Success rate (%)")

    handles = [
        Line2D([], [], color=ORANGE, marker="o", ms=3, lw=1.1, label=r"Single-$d$"),
        Line2D([], [], color=BLUE, marker="s", ms=3, lw=1.1, label="MWM (fixed)"),
        Line2D([], [], color="0.35", ls="-", marker="o", ms=3, lw=1.1, label=r"$\Delta=25$"),
        Line2D([], [], color="0.35", ls="--", marker="o", ms=3, lw=1.1,
               markerfacecolor="white", label=r"$\Delta=50$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
