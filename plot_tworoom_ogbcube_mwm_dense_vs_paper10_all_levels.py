"""Build the "two independently-trained dense multi-K checkpoints, full
per-level curves" comparison plot for TwoRoom, OGB Cube, and Reacher:

- checkpoints_mwm/mwm_dense_tworoom / mwm_dense_ogb_cube / mwm_dense_reacher
  (K=[6,12,48,96,144,192], predates the July 28 release) -- ALL SIX of its
  own levels, from release20260728_<env>_goal<N>_mwm_dense_<env>_{low,high}_k_horizon5[_n200].
- checkpoints_mwm/mwm_paper10_<env>_k96_120_144_168_192_release20260728
  (K=[96,120,144,168,192], July 28 release) -- all five of its own levels,
  from release20260728_<env>_goal<N>_k_matched_comparison_horizon5[_n200]
  (dense_matched_k* rows).

Same eval protocol for every point within a given goal_offset: horizon=5/
receding_horizon=5, pop=300/elite_frac=0.1/n_iter=10, same manifest -- so the
two checkpoints' curves are directly comparable. goal_offset=25 uses a
freshly-sampled n=200 manifest (report dirs suffixed _n200); goal_offset=50
reuses the existing 100-episode goal50 manifest (no _n200 suffix) for a
faster turnaround -- episode count differs between the two goal offsets as a
result.

Run directly for the goal_offset=25/n=200 version (the original); call
make_plot(..., goal_offset=50, dir_suffix="", episodes=100) for goal_offset=50.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MWM_DENSE_K = [6, 12, 48, 96, 144, 192]
PAPER10_K = [96, 120, 144, 168, 192]

ENVS = [("tworoom", "TwoRoom"), ("ogb_cube", "OGB Cube"), ("reacher", "Reacher"), ("pusht", "PushT")]

ENV_COLOR = {"tworoom": "#2a9e6e", "ogb_cube": "#8a4fbf", "reacher": "#c1523a", "pusht": "#2f6db3"}
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


def make_plot(out_path: str, goal_offset: int = 25, dir_suffix: str = "_n200", episodes: int = 200,
              overrides: dict[str, tuple[str, int]] | None = None) -> None:
    """overrides: per-env (dir_suffix, episodes) pairs that take precedence over the
    global dir_suffix/episodes -- e.g. PushT reuses its existing 100-episode manifest
    at goal_offset=25 while the other envs use a freshly-sampled n=200 one."""
    overrides = overrides or {}
    fig, axes = plt.subplots(1, len(ENVS), figsize=(5.6 * len(ENVS), 5.2), sharey=True)

    for (env_key, env_title), ax in zip(ENVS, axes.flat):
        color = ENV_COLOR[env_key]
        env_dir_suffix, env_episodes = overrides.get(env_key, (dir_suffix, episodes))

        low_dir = Path(f"reports/research/release20260728_{env_key}_goal{goal_offset}_mwm_dense_{env_key}_low_k_horizon5{env_dir_suffix}")
        high_dir = Path(f"reports/research/release20260728_{env_key}_goal{goal_offset}_mwm_dense_{env_key}_high_k_horizon5{env_dir_suffix}")
        low_rows = load_rows(low_dir)
        high_rows = load_rows(high_dir)
        mwm_dense_success = []
        for k in MWM_DENSE_K:
            r = low_rows.get(f"mwm_dense_{env_key}_k{k}") or high_rows.get(f"mwm_dense_{env_key}_k{k}")
            if r is None:
                raise SystemExit(f"missing mwm_dense_{env_key}_k{k}")
            mwm_dense_success.append(r["success_rate"])

        paper10_dir = Path(f"reports/research/release20260728_{env_key}_goal{goal_offset}_k_matched_comparison_horizon5{env_dir_suffix}")
        paper10_rows = load_rows(paper10_dir)
        paper10_success = []
        for k in PAPER10_K:
            r = paper10_rows.get(f"dense_matched_k{k}")
            if r is None:
                raise SystemExit(f"missing dense_matched_k{k} in {paper10_dir}")
            paper10_success.append(r["success_rate"])

        ax.plot(MWM_DENSE_K, mwm_dense_success, color=MWM_DENSE_COLOR, linestyle="-", marker="o",
                markersize=7, linewidth=2.4,
                label="mwm_dense_" + env_key + "\n(K=[6,12,48,96,144,192])")
        ax.plot(PAPER10_K, paper10_success, color=color, linestyle="--", marker="s",
                markersize=7, linewidth=2.4,
                label="mwm_paper10_" + env_key + "\n(K=[96,120,144,168,192], Jul 28)")

        ax.set_xscale("log")
        all_k = sorted(set(MWM_DENSE_K) | set(PAPER10_K))
        ax.set_xticks(all_k)
        ax.set_xticklabels([str(k) for k in all_k])
        ax.minorticks_off()
        ax.set_ylim(0, 105)
        ax.set_xlabel("Fidelity level K", fontsize=11)
        title = env_title if env_episodes == episodes else f"{env_title}\n({env_episodes} episodes)"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower left", frameon=True, facecolor="white",
                  edgecolor="0.8", framealpha=0.95)

    axes[0].set_ylabel("Success rate (%)", fontsize=11)

    fig.suptitle(
        "Two independently-trained dense multi-K checkpoints, full per-level curves\n"
        f"(goal_offset={goal_offset}, horizon=5, {episodes} episodes, same manifest/CEM budget)",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_plot(out_path="rollouts/tworoom_ogbcube_mwm_dense_vs_paper10_all_levels_horizon5_n200.png",
              goal_offset=25, dir_suffix="_n200", episodes=200)
