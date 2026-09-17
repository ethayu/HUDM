"""Build rollouts/release20260728_goal25_k_matched_comparison_horizon5_n200_all_envs.png:
same style as plot_release20260728_k_matched_comparison_all_envs.py, but for
the horizon=5/receding_horizon=5, n=200-episode reruns
(reports/research/release20260728_<env>_goal25_k_matched_comparison_horizon5_n200),
run specifically for tighter statistical power on the individually-trained
vs. dense-matched-K comparison after the original n=100/horizon=2 result
showed a borderline-significant TwoRoom K=96-vs-K=192 gap.

For TwoRoom and OGB Cube, the dense-matched-K line is extended down to K=6
using checkpoints_mwm/mwm_dense_tworoom / mwm_dense_ogb_cube (K=[6,12,48,96,144,192])
-- a DIFFERENT dense checkpoint from the paper10 one used for K=96/144/192
(which has no K<96 levels), sourced from
release20260728_<env>_goal25_mwm_dense_<env>_low_k_horizon5_n200. Same eval
protocol (horizon=5, pop=300/n_iter=10, goal_offset=25, 200 episodes, same
manifest) so the two checkpoints' points are directly comparable.

Each subplot: solid+circle = individually trained, dashed+square =
dense-matched (same multi-K checkpoint sliced at that K level).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_VALUES = [96, 120, 144, 168, 192]
LOW_K_VALUES = [6, 12, 48]

ENVS = [
    ("pusht", "PushT"),
    ("reacher", "Reacher"),
    ("tworoom", "TwoRoom"),
    ("ogb_cube", "OGB Cube"),
]

# Envs with a low-K (6/12/48) extension of the dense-matched-K line, and the
# checkpoint-specific run-name prefix used in their low-K report dirs.
LOW_K_SOURCE = {
    "tworoom": "mwm_dense_tworoom",
    "ogb_cube": "mwm_dense_ogb_cube",
}

ENV_COLOR = {
    "pusht": "#2a5f9e",
    "reacher": "#c9622a",
    "tworoom": "#2a9e6e",
    "ogb_cube": "#8a4fbf",
}


def load_rows(output_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for d in sorted(output_dir.iterdir()):
        summary = d / "summary.json"
        if not d.is_dir() or not summary.exists():
            continue
        run = json.loads(summary.read_text())["run"]
        rows[run["base_name"]] = run
    return rows


def make_plot(out_path: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8), sharex=False, sharey=True)

    for (env_key, env_title), ax in zip(ENVS, axes.flat):
        output_dir = Path(f"reports/research/release20260728_{env_key}_goal25_k_matched_comparison_horizon5_n200")
        rows = load_rows(output_dir)
        color = ENV_COLOR[env_key]

        individual = []
        dense = []
        for k in K_VALUES:
            ind = rows.get(f"individual_k{k}")
            den = rows.get(f"dense_matched_k{k}")
            if ind is None or den is None:
                raise SystemExit(f"missing run(s) for K={k} in {output_dir}")
            individual.append(ind["success_rate"])
            dense.append(den["success_rate"])

        k_axis = list(K_VALUES)
        dense_axis = list(dense)
        low_k_prefix = LOW_K_SOURCE.get(env_key)
        if low_k_prefix is not None:
            low_k_dir = Path(f"reports/research/release20260728_{env_key}_goal25_{low_k_prefix}_low_k_horizon5_n200")
            low_rows = load_rows(low_k_dir)
            low_dense = []
            for k in LOW_K_VALUES:
                r = low_rows.get(f"{low_k_prefix}_k{k}")
                if r is None:
                    raise SystemExit(f"missing {low_k_prefix}_k{k} in {low_k_dir}")
                low_dense.append(r["success_rate"])
            k_axis = LOW_K_VALUES + k_axis
            dense_axis = low_dense + dense_axis

        ax.plot(K_VALUES, individual, color=color, linestyle="-", marker="o",
                markersize=6, linewidth=2.2, label="Individually trained")
        ax.plot(k_axis, dense_axis, color=color, linestyle="--", marker="s",
                markersize=6, linewidth=2.2, alpha=0.75, label="Dense-matched")

        ax.set_xscale("log")
        ax.set_xticks(k_axis)
        ax.set_xticklabels([str(k) for k in k_axis])
        ax.minorticks_off()
        ax.set_ylim(0, 105)
        ax.set_title(env_title, fontsize=12.5, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", frameon=True, facecolor="white",
                  edgecolor="0.8", framealpha=0.95)

    for ax in axes:
        ax.set_xlabel("Fidelity level K", fontsize=11)
    axes[0].set_ylabel("Success rate (%)", fontsize=11)

    fig.suptitle(
        "Individually-trained vs. dense-matched-K  (goal_offset=25, horizon=5, 200 episodes)\n"
        "solid+circle = single-K training, dashed+square = multi-K (dense) checkpoint sliced at K",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(out_path="rollouts/release20260728_goal25_k_matched_comparison_horizon5_n200_all_envs.png")
