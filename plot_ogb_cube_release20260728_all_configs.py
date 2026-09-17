"""Build rollouts/ogb_cube_release20260728_goal{25,50}_100ep_all_configs.png from the
release20260728 all-fidelity schedule sweep reports, in the same style as
rollouts/tworoom_release20260728_goal{25,50}_100ep_all_configs.png:

- one scatter + pareto step line per dense-checkpoint schedule (bits/ep vs
  success rate, swept over pop_size x n_iter),
- stray leftover report directories whose "schedule" text doesn't match any
  run currently declared in the sweep config (renamed/removed runs) are
  dropped,
- dense schedules whose MPC/CEM/Rollout text never contains a "->" (i.e. the
  scheduler never actually transitions fidelity) are dropped as uninformative
  duplicates of a single-K baseline,
- the five individually-trained single-K baselines ("single K=..." schedules)
  are always kept even though they are single-level by construction.
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
    # tworoom reference (finest single-K first, coarsest last).
    def baseline_k(label: str) -> int:
        try:
            return int(label.split("K=")[1].split()[0].rstrip("|"))
        except (IndexError, ValueError):
            return 0

    baseline_labels.sort(key=baseline_k, reverse=True)

    dropped = [s for s in order if s not in dense_labels and s not in baseline_labels]

    dense_colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(1, len(dense_labels))))
    baseline_colors = plt.cm.cool(np.linspace(0.15, 0.95, max(1, len(baseline_labels))))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    def plot_group(label: str, color) -> float:
        pts = by_schedule[label]
        xs = [r["bits_used_total"] / r["episodes"] / 1e6 for r in pts]
        ys = [r["success_rate"] for r in pts]
        px, py, xs_s, ys_s = pareto_step(xs, ys)
        ax.scatter(xs_s, ys_s, color=color, s=40, alpha=0.7, zorder=3)
        ax.step(px, py, where="post", color=color, linewidth=2.0, alpha=0.9, label=label)
        return max(xs)

    max_x = 0.0
    for label, color in zip(dense_labels, dense_colors):
        max_x = max(max_x, plot_group(label, color))
    for label, color in zip(baseline_labels, baseline_colors):
        max_x = max(max_x, plot_group(label, color))

    if x_clip is not None:
        ax.set_xlim(0, x_clip)
        clip_note = f"; x clipped at {x_clip:.0f}M bits/ep"
    else:
        ax.set_xlim(0, max_x * 1.03)
        clip_note = ""

    ax.set_xlabel("Bits per episode (×10⁶)", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title(
        f"{env_title}  goal_offset={goal_offset}  {episodes} episodes  all configs\n"
        f"(fixed-fidelity schedules removed; single-K baselines always shown{clip_note})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  (dense schedules: {len(dense_labels)}, baselines: {len(baseline_labels)}, dropped fixed-fidelity: {len(dropped)})")
    if dropped:
        print("  dropped:", dropped)


if __name__ == "__main__":
    make_plot(
        config_path="configs/research/release20260728_dense_ogb_cube_all_fidelity_schedules.yaml",
        output_dir="reports/research/release20260728_dense_ogb_cube_all_fidelity_schedules",
        env_title="OGB Cube",
        goal_offset=25,
        episodes=100,
        out_path="rollouts/ogb_cube_release20260728_goal25_100ep_all_configs.png",
        x_clip=250,
    )
    make_plot(
        config_path="configs/research/release20260728_dense_ogb_cube_goal50_plan100_execute20_all_fidelity_schedules.yaml",
        output_dir="reports/research/release20260728_dense_ogb_cube_goal50_plan100_execute20_all_fidelity_schedules",
        env_title="OGB Cube",
        goal_offset=50,
        episodes=100,
        out_path="rollouts/ogb_cube_release20260728_goal50_100ep_all_configs.png",
        x_clip=500,
    )
