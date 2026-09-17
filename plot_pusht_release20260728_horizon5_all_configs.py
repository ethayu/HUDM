"""Build rollouts/pusht_release20260728_goal25_horizon5_100ep_all_configs.png
from the release20260728 PushT horizon5 all-fidelity schedule sweep report.

Unlike the plain "one color per schedule" style (still available via
`highlight=False`), the default here foregrounds the two things that matter
for the paper's headline claim:

- every individual changing-fidelity (adaptive) schedule is still plotted,
  but muted to a thin gray background cloud, EXCEPT one bold blue step line:
  the pooled Pareto frontier across all of them -- "the best any adaptive
  schedule achieves at this bits budget" -- which is the actual winning
  method at every point on the compute axis (no single fixed schedule wins
  everywhere, so no single schedule should be bolded instead).
- the four low-K fixed baselines (K=96/120/144/168) are muted to a single
  thin dashed gray-tan style (they all collapse near the bottom and are not
  the comparison of interest here); the K=192 baseline -- full base
  dimensionality, our reference point -- is drawn bold, black, and on top of
  everything else.

- stray leftover report directories whose "schedule" text doesn't match any
  run currently declared in the sweep config (renamed/removed runs) are
  dropped,
- dense schedules whose MPC/CEM/Rollout text never contains a "->" (i.e. the
  scheduler never actually transitions fidelity) are dropped as uninformative
  duplicates of a single-K baseline.

This is the horizon=5 (paper Appendix D setting) counterpart of
plot_pusht_release20260728_all_configs.py's goal25 default (horizon=2) run.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def declared_schedules(config_path: str) -> list[str]:
    """Ordered list of `schedule:` strings actually declared as active runs in
    the sweep config, used to drop stray leftover report directories from
    earlier/renamed run definitions that don't match any current run."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    return [str(run["schedule"]) for run in cfg["runs"]]


def load_rows(output_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(output_dir.iterdir()):
        summary = d / "summary.json"
        if not d.is_dir() or not summary.exists():
            continue
        rows.append(json.loads(summary.read_text())["run"])
    return rows


def pareto_step(xs: list[float], ys: list[float]):
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


def make_plot(config_path: str, output_dir: str, env_title: str, goal_offset: int, episodes: int, out_path: str, x_clip: float | None = None) -> None:
    rows = load_rows(Path(output_dir))
    if not rows:
        raise SystemExit(f"no summary.json files found under {output_dir}")

    active = set(declared_schedules(config_path))

    by_schedule: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        schedule = str(row.get("schedule", row.get("base_name", "")))
        if schedule not in active:
            continue  # stray leftover directory from an earlier/renamed run, not a currently declared run
        if schedule not in by_schedule:
            by_schedule[schedule] = []
            order.append(schedule)
        by_schedule[schedule].append(row)

    # A dense (multi-fidelity checkpoint) schedule only earns a spot if at
    # least one of its MPC/CEM/Rollout axes actually transitions between
    # levels ("->"); a schedule with no transition anywhere never leaves its
    # default fidelity and is a redundant duplicate of a single-K baseline.
    dense_labels = [s for s in order if not s.strip().startswith("single K=") and "->" in s]
    baseline_labels = [s for s in order if s.strip().startswith("single K=")]
    # baselines sorted by descending K so the legend/coloring matches the
    # tworoom/ogb_cube reference (finest single-K first, coarsest last).
    def baseline_k(label: str) -> int:
        try:
            return int(label.split("K=")[1].split()[0].rstrip("|"))
        except (IndexError, ValueError):
            return 0

    baseline_labels.sort(key=baseline_k, reverse=True)

    dropped = [s for s in order if s not in dense_labels and s not in baseline_labels]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    CLOUD_GRAY = "#b7bcc2"
    FRONTIER_BLUE = "#2a5f9e"
    LOWK_TAN = "#c9a06a"
    BASELINE_BLACK = "#111214"

    max_x = 0.0

    # 1. every individual adaptive (changing-fidelity) schedule: thin gray
    # background cloud + step line, unlabeled -- context, not the headline.
    for label in dense_labels:
        pts = by_schedule[label]
        xs = [r["bits_used_total"] / r["episodes"] / 1e6 for r in pts]
        ys = [r["success_rate"] for r in pts]
        px, py, xs_s, ys_s = pareto_step(xs, ys)
        ax.scatter(xs_s, ys_s, color=CLOUD_GRAY, s=16, alpha=0.35, linewidths=0, zorder=1)
        ax.step(px, py, where="post", color=CLOUD_GRAY, linewidth=0.9, alpha=0.55, zorder=1)
        max_x = max(max_x, max(xs))

    # 2. the winning method at any given budget: the pooled Pareto frontier
    # across ALL changing-fidelity schedules combined -- no single fixed
    # schedule wins everywhere, so the frontier itself is "the win."
    pooled_x = [r["bits_used_total"] / r["episodes"] / 1e6 for label in dense_labels for r in by_schedule[label]]
    pooled_y = [r["success_rate"] for label in dense_labels for r in by_schedule[label]]
    fx, fy = pareto_step(pooled_x, pooled_y)[:2]
    ax.step(fx, fy, where="post", color=FRONTIER_BLUE, linewidth=3.2, zorder=4,
            label="Winning adaptive schedules (Pareto frontier, changing fidelity)")
    ax.scatter(fx, fy, color=FRONTIER_BLUE, s=46, zorder=5, edgecolors="white", linewidths=1.0)

    # 3. low-K fixed baselines: muted, one shared style, not the comparison
    # of interest here.
    for i, label in enumerate([l for l in baseline_labels if baseline_k(l) != 192]):
        pts = by_schedule[label]
        xs = [r["bits_used_total"] / r["episodes"] / 1e6 for r in pts]
        ys = [r["success_rate"] for r in pts]
        px, py, xs_s, ys_s = pareto_step(xs, ys)
        ax.scatter(xs_s, ys_s, color=LOWK_TAN, s=16, alpha=0.45, linewidths=0, zorder=1)
        ax.step(px, py, where="post", color=LOWK_TAN, linewidth=1.1, linestyle="--", alpha=0.7, zorder=1,
                label="Fixed K < 192 (96/120/144/168)" if i == 0 else None)
        max_x = max(max_x, max(xs))

    # 4. our baseline: single K=192 (full base dimensionality), bold and on top.
    baseline_192 = next(l for l in baseline_labels if baseline_k(l) == 192)
    pts = by_schedule[baseline_192]
    xs = [r["bits_used_total"] / r["episodes"] / 1e6 for r in pts]
    ys = [r["success_rate"] for r in pts]
    px, py, xs_s, ys_s = pareto_step(xs, ys)
    ax.step(px, py, where="post", color=BASELINE_BLACK, linewidth=2.6, zorder=6,
            label="Fixed K=192 (baseline, full width)")
    ax.scatter(xs_s, ys_s, color=BASELINE_BLACK, s=44, marker="s", zorder=7,
               edgecolors="white", linewidths=1.0)
    max_x = max(max_x, max(xs))

    if x_clip is not None:
        ax.set_xlim(0, x_clip)
        clip_note = f"; x clipped at {x_clip:.0f}M bits/ep"
    else:
        ax.set_xlim(0, max_x * 1.03)
        clip_note = ""

    ax.set_xlabel("Bits per episode (×10⁶)", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title(
        f"{env_title}  goal_offset={goal_offset}  {episodes} episodes  horizon 5\n"
        f"adaptive fidelity scheduling vs. fixed-K{clip_note}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9.5, loc="lower right", frameon=True, facecolor="white",
              edgecolor="0.8", framealpha=0.95)
    fig.tight_layout()

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  (dense schedules: {len(dense_labels)}, baselines: {len(baseline_labels)}, dropped fixed-fidelity: {len(dropped)})")
    if dropped:
        print("  dropped:", dropped)


if __name__ == "__main__":
    make_plot(
        config_path="configs/research/release20260728_dense_pusht_horizon5_all_fidelity_schedules.yaml",
        output_dir="reports/research/release20260728_dense_pusht_horizon5_all_fidelity_schedules",
        env_title="PushT",
        goal_offset=25,
        episodes=100,
        out_path="rollouts/pusht_release20260728_goal25_horizon5_100ep_all_configs.png",
        x_clip=200,
    )
