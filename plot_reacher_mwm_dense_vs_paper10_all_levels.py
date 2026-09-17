"""Build rollouts/reacher_mwm_dense_vs_paper10_all_levels_horizon5_n200.png:
for Reacher, compares TWO independently-trained dense multi-K checkpoints'
full per-level success rate curves -- same recipe as
plot_tworoom_ogbcube_mwm_dense_vs_paper10_all_levels.py:

- checkpoints_mwm/mwm_dense_reacher (K=[6,12,48,96,144,192], predates the
  July 28 release) -- ALL SIX of its own levels, from
  release20260728_reacher_goal25_mwm_dense_reacher_{low,high}_k_horizon5_n200.
- checkpoints_mwm/mwm_paper10_reacher_k96_120_144_168_192_release20260728
  (K=[96,120,144,168,192], July 28 release) -- all five of its own levels,
  from release20260728_reacher_goal25_k_matched_comparison_horizon5_n200
  (dense_matched_k* rows).

Same eval protocol for every point: horizon=5/receding_horizon=5,
pop=300/elite_frac=0.1/n_iter=10, goal_offset=25, 200 episodes, same n=200
manifest -- so the two checkpoints' curves are directly comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MWM_DENSE_K = [6, 12, 48, 96, 144, 192]
PAPER10_K = [96, 120, 144, 168, 192]

ENV_KEY, ENV_TITLE = "reacher", "Reacher"
ENV_COLOR = "#c1523a"
MWM_DENSE_COLOR = "#111214"


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
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))

    low_dir = Path(f"reports/research/release20260728_{ENV_KEY}_goal25_mwm_dense_{ENV_KEY}_low_k_horizon5_n200")
    high_dir = Path(f"reports/research/release20260728_{ENV_KEY}_goal25_mwm_dense_{ENV_KEY}_high_k_horizon5_n200")
    low_rows = load_rows(low_dir)
    high_rows = load_rows(high_dir)
    mwm_dense_success = []
    for k in MWM_DENSE_K:
        r = low_rows.get(f"mwm_dense_{ENV_KEY}_k{k}") or high_rows.get(f"mwm_dense_{ENV_KEY}_k{k}")
        if r is None:
            raise SystemExit(f"missing mwm_dense_{ENV_KEY}_k{k}")
        mwm_dense_success.append(r["success_rate"])

    paper10_dir = Path(f"reports/research/release20260728_{ENV_KEY}_goal25_k_matched_comparison_horizon5_n200")
    paper10_rows = load_rows(paper10_dir)
    paper10_success = []
    for k in PAPER10_K:
        r = paper10_rows.get(f"dense_matched_k{k}")
        if r is None:
            raise SystemExit(f"missing dense_matched_k{k} in {paper10_dir}")
        paper10_success.append(r["success_rate"])

    ax.plot(MWM_DENSE_K, mwm_dense_success, color=MWM_DENSE_COLOR, linestyle="-", marker="o",
            markersize=7, linewidth=2.4,
            label="mwm_dense_" + ENV_KEY + "\n(K=[6,12,48,96,144,192])")
    ax.plot(PAPER10_K, paper10_success, color=ENV_COLOR, linestyle="--", marker="s",
            markersize=7, linewidth=2.4,
            label="mwm_paper10_" + ENV_KEY + "\n(K=[96,120,144,168,192], Jul 28)")

    ax.set_xscale("log")
    all_k = sorted(set(MWM_DENSE_K) | set(PAPER10_K))
    ax.set_xticks(all_k)
    ax.set_xticklabels([str(k) for k in all_k])
    ax.minorticks_off()
    ax.set_ylim(0, 105)
    ax.set_xlabel("Fidelity level K", fontsize=11)
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_title(
        f"{ENV_TITLE}\nTwo independently-trained dense multi-K checkpoints, full per-level curves\n"
        "(goal_offset=25, horizon=5, 200 episodes, same manifest/CEM budget)",
        fontsize=11.5, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True, facecolor="white",
              edgecolor="0.8", framealpha=0.95)

    fig.tight_layout()

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(out_path="rollouts/reacher_mwm_dense_vs_paper10_all_levels_horizon5_n200.png")
