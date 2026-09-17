"""TwoRoom-only companion to plot_tworoom_ogbcube_mwm_dense_vs_paper10_all_levels.py:
adds the new dense K=[12,24,36,48,60,72,84,96] separate-optimizer checkpoint
(mwm_paper10_tworoom_k12to96_sepopt_20260915, release
hudm-mwm-dense-k12to96-sepopt-20260916) as a third curve alongside the two
checkpoints already on that plot -- mwm_dense_tworoom (K=[6,12,48,96,144,192])
and mwm_paper10_tworoom_k96_120_144_168_192_release20260728
(K=[96,120,144,168,192]).

Same eval protocol as the reference plot: horizon=5/receding_horizon=5,
pop=300/elite_frac=0.1/n_iter=10, goal_offset=25, 200 episodes, same n=200
manifest -- so all three curves are directly comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MWM_DENSE_K = [6, 12, 48, 96, 144, 192]
PAPER10_K = [96, 120, 144, 168, 192]
SEPOPT_K = [12, 24, 36, 48, 60, 72, 84, 96]


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
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    low_dir = Path("reports/research/release20260728_tworoom_goal25_mwm_dense_tworoom_low_k_horizon5_n200")
    high_dir = Path("reports/research/release20260728_tworoom_goal25_mwm_dense_tworoom_high_k_horizon5_n200")
    low_rows = load_rows(low_dir)
    high_rows = load_rows(high_dir)
    mwm_dense_success = []
    for k in MWM_DENSE_K:
        r = low_rows.get(f"mwm_dense_tworoom_k{k}") or high_rows.get(f"mwm_dense_tworoom_k{k}")
        if r is None:
            raise SystemExit(f"missing mwm_dense_tworoom_k{k}")
        mwm_dense_success.append(r["success_rate"])

    paper10_dir = Path("reports/research/release20260728_tworoom_goal25_k_matched_comparison_horizon5_n200")
    paper10_rows = load_rows(paper10_dir)
    paper10_success = []
    for k in PAPER10_K:
        r = paper10_rows.get(f"dense_matched_k{k}")
        if r is None:
            raise SystemExit(f"missing dense_matched_k{k}")
        paper10_success.append(r["success_rate"])

    sepopt_dir = Path("reports/research/tworoom_goal25_k12to96_sepopt_20260916_all_levels_horizon5_n200")
    sepopt_rows = load_rows(sepopt_dir)
    sepopt_success = []
    for k in SEPOPT_K:
        r = sepopt_rows.get(f"mwm_k12to96_sepopt_k{k}")
        if r is None:
            raise SystemExit(f"missing mwm_k12to96_sepopt_k{k}")
        sepopt_success.append(r["success_rate"])

    ax.plot(MWM_DENSE_K, mwm_dense_success, color="#111214", linestyle="-", marker="o",
            markersize=7, linewidth=2.4,
            label="mwm_dense_tworoom\n(K=[6,12,48,96,144,192])")
    ax.plot(PAPER10_K, paper10_success, color="#2a9e6e", linestyle="--", marker="s",
            markersize=7, linewidth=2.4,
            label="mwm_paper10_tworoom\n(K=[96,120,144,168,192], Jul 28)")
    ax.plot(SEPOPT_K, sepopt_success, color="#c1523a", linestyle="-.", marker="^",
            markersize=8, linewidth=2.4,
            label="mwm_k12to96_sepopt\n(K=[12..96], Sep 16, sep. optimizers)")

    ax.set_xscale("log")
    all_k = sorted(set(MWM_DENSE_K) | set(PAPER10_K) | set(SEPOPT_K))
    ax.set_xticks(all_k)
    ax.set_xticklabels([str(k) for k in all_k], rotation=45)
    ax.minorticks_off()
    ax.set_ylim(0, 105)
    ax.set_xlabel("Fidelity level K", fontsize=11)
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_title("TwoRoom", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="center left", frameon=True, facecolor="white",
              edgecolor="0.8", framealpha=0.95)

    fig.suptitle(
        "TwoRoom: new dense K=12..96 separate-optimizer checkpoint vs. existing dense multi-K checkpoints\n"
        "(goal_offset=25, horizon=5, 200 episodes, same manifest/CEM budget)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(out_path="rollouts/tworoom_k12to96_sepopt_vs_existing_horizon5_n200.png")
