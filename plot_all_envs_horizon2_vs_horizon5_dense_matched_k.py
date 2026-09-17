"""Build rollouts/release20260728_goal25_dense_matched_k_horizon2_vs_horizon5_all_envs.png:
one row of four subplots (PushT, Reacher, TwoRoom, OGB Cube), each comparing
the dense multi-K checkpoint's per-K success rate at horizon=2/receding_horizon=2
(baseline, from each env's release20260728_<env>_goal25_k_matched_comparison.yaml)
vs. horizon=5/receding_horizon=5 (from release20260728_<env>_goal25_dense_matched_k_horizon5.yaml),
holding CEM budget fixed per env (paper-default pop=300, elite_frac=0.1;
n_iter=30 for PushT, 10 for the other three).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_VALUES = [96, 120, 144, 168, 192]

ENVS = [
    ("pusht", "PushT"),
    ("reacher", "Reacher"),
    ("tworoom", "TwoRoom"),
    ("ogb_cube", "OGB Cube"),
]

HORIZON2_COLOR = "#c9622a"
HORIZON5_COLOR = "#2a5f9e"


def load_rows(output_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for d in sorted(output_dir.iterdir()):
        summary = d / "summary.json"
        if not d.is_dir() or not summary.exists():
            continue
        run = json.loads(summary.read_text())["run"]
        rows[run["base_name"]] = run
    return rows


def success_by_k(output_dir: str) -> list[float]:
    rows = load_rows(Path(output_dir))
    return [rows[f"dense_matched_k{k}"]["success_rate"] for k in K_VALUES]


def make_plot(out_path: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8), sharex=True, sharey=True)

    for (env_key, env_title), ax in zip(ENVS, axes.flat):
        horizon2 = success_by_k(f"reports/research/release20260728_{env_key}_goal25_k_matched_comparison")
        horizon5 = success_by_k(f"reports/research/release20260728_{env_key}_goal25_dense_matched_k_horizon5")

        ax.plot(K_VALUES, horizon2, color=HORIZON2_COLOR, linestyle="-", marker="o",
                markersize=6, linewidth=2.2, label="horizon=2")
        ax.plot(K_VALUES, horizon5, color=HORIZON5_COLOR, linestyle="-", marker="s",
                markersize=6, linewidth=2.2, label="horizon=5")

        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 105)
        ax.set_title(env_title, fontsize=12.5, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", frameon=True, facecolor="white",
                  edgecolor="0.8", framealpha=0.95)

    for ax in axes:
        ax.set_xlabel("Fidelity level K", fontsize=11)
    axes[0].set_ylabel("Success rate (%)", fontsize=11)

    fig.suptitle(
        "Dense multi-K checkpoint, same CEM budget per env: horizon=2 vs. horizon=5\n"
        "(goal_offset=25, 100 episodes; receding_horizon = horizon in both cases)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(out_path="rollouts/release20260728_goal25_dense_matched_k_horizon2_vs_horizon5_all_envs.png")
