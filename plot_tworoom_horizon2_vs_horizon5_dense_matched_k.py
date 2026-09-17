"""Build rollouts/tworoom_goal25_dense_matched_k_horizon2_vs_horizon5.png:
compares the dense multi-K checkpoint's per-K success rate at the paper-default
CEM setting (pop_size=300, elite_frac=0.1, n_iter=10) under two horizons:

- horizon=2 / receding_horizon=2 (baseline, from
  release20260728_tworoom_goal25_k_matched_comparison.yaml -- 10 replans of
  10 actions each over the 100-step episode)
- horizon=5 / receding_horizon=5 (from
  release20260728_tworoom_goal25_dense_matched_k_horizon5.yaml -- 4 replans
  of 25 actions each)

Same CEM budget both times -- only horizon/receding_horizon changed -- to
isolate whether fewer, longer replanning commitments close the gap between
low-K and high-K success rates on the same jointly-trained checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_VALUES = [96, 120, 144, 168, 192]

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
    out = []
    for k in K_VALUES:
        r = rows.get(f"dense_matched_k{k}")
        if r is None:
            raise SystemExit(f"missing dense_matched_k{k} in {output_dir}")
        out.append(r["success_rate"])
    return out


def make_plot(out_path: str) -> None:
    horizon2 = success_by_k("reports/research/release20260728_tworoom_goal25_k_matched_comparison")
    horizon5 = success_by_k("reports/research/release20260728_tworoom_goal25_dense_matched_k_horizon5")

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ax.plot(K_VALUES, horizon2, color=HORIZON2_COLOR, linestyle="-", marker="o",
            markersize=7, linewidth=2.4, label="horizon=2, receding_horizon=2 (10 replans x 10 actions)")
    ax.plot(K_VALUES, horizon5, color=HORIZON5_COLOR, linestyle="-", marker="s",
            markersize=7, linewidth=2.4, label="horizon=5, receding_horizon=5 (4 replans x 25 actions)")

    for xs, ys, color in [(K_VALUES, horizon2, HORIZON2_COLOR), (K_VALUES, horizon5, HORIZON5_COLOR)]:
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, color=color)

    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Fidelity level K", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title(
        "TwoRoom dense multi-K checkpoint: same CEM budget (pop=300, n_iter=10),\n"
        "horizon=2 vs. horizon=5 -- goal_offset=25, 100 episodes",
        fontsize=12.5, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9.5, loc="lower left", frameon=True, facecolor="white",
              edgecolor="0.8", framealpha=0.95)
    fig.tight_layout()

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(out_path="rollouts/tworoom_goal25_dense_matched_k_horizon2_vs_horizon5.png")
