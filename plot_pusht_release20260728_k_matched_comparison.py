"""Build rollouts/pusht_release20260728_goal{25,50}_k_matched_comparison.png:
for each K in the paper K sweep (96/120/144/168/192), compare the
individually-trained single-K PushT checkpoint against the same K level
sliced out of the jointly-trained dense multi-K checkpoint
(mwm_paper10_pusht_k96_120_144_168_192_release20260728), at the fixed
paper-default CEM setting used by
configs/research/release20260728_pusht_goal{25,50}_k_matched_comparison.yaml.

Grouped bar chart: one bar pair per K, "individually trained" vs
"dense-matched (multi-K checkpoint)".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INDIVIDUAL_COLOR = "#c9a06a"
DENSE_COLOR = "#2a5f9e"

K_VALUES = [96, 120, 144, 168, 192]


def load_rows(output_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for d in sorted(output_dir.iterdir()):
        summary = d / "summary.json"
        if not d.is_dir() or not summary.exists():
            continue
        run = json.loads(summary.read_text())["run"]
        rows[run["base_name"]] = run
    return rows


def make_plot(output_dir: str, goal_offset: int, episodes: int, out_path: str) -> None:
    rows = load_rows(Path(output_dir))

    individual_success = []
    dense_success = []
    for k in K_VALUES:
        ind = rows.get(f"individual_k{k}")
        dense = rows.get(f"dense_matched_k{k}")
        if ind is None or dense is None:
            raise SystemExit(f"missing run(s) for K={k} in {output_dir}")
        individual_success.append(ind["success_rate"])
        dense_success.append(dense["success_rate"])

    x = range(len(K_VALUES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    bars_ind = ax.bar([i - width / 2 for i in x], individual_success, width,
                       color=INDIVIDUAL_COLOR, label="Individually trained (single K)", zorder=3)
    bars_dense = ax.bar([i + width / 2 for i in x], dense_success, width,
                         color=DENSE_COLOR, label="Dense-matched (K=[96,120,144,168,192] checkpoint)", zorder=3)

    for bars in (bars_ind, bars_dense):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"K={k}" for k in K_VALUES])
    ax.set_xlabel("Fidelity level K", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_ylim(0, max(individual_success + dense_success) * 1.2 + 5)
    ax.set_title(
        f"PushT  goal_offset={goal_offset}  {episodes} episodes\n"
        f"individually-trained vs. dense-matched-K (paper-default CEM: pop=300, iter=30)",
        fontsize=12.5, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9.5, loc="upper left", frameon=True, facecolor="white",
              edgecolor="0.8", framealpha=0.95)
    fig.tight_layout()

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(
        output_dir="reports/research/release20260728_pusht_goal25_k_matched_comparison",
        goal_offset=25,
        episodes=100,
        out_path="rollouts/pusht_release20260728_goal25_k_matched_comparison.png",
    )
    make_plot(
        output_dir="reports/research/release20260728_pusht_goal50_k_matched_comparison",
        goal_offset=50,
        episodes=100,
        out_path="rollouts/pusht_release20260728_goal50_k_matched_comparison.png",
    )
