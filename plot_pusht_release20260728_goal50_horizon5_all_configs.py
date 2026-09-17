"""Build rollouts/pusht_release20260728_goal50_horizon5_100ep_all_configs.png
from the release20260728 PushT goal50 horizon5 all-fidelity schedule sweep report.

goal50 counterpart of plot_pusht_release20260728_horizon5_all_configs.py's
goal25 default -- same horizon=5 (paper Appendix D) setting, same plotting
convention (muted adaptive-schedule cloud + pooled Pareto frontier, muted
low-K fixed baselines, bold K=192 baseline). See that module's docstring for
the full rationale.
"""
from __future__ import annotations

from plot_pusht_release20260728_horizon5_all_configs import make_plot

if __name__ == "__main__":
    make_plot(
        config_path="configs/research/release20260728_dense_pusht_goal50_horizon5_all_fidelity_schedules.yaml",
        output_dir="reports/research/release20260728_dense_pusht_goal50_horizon5_all_fidelity_schedules",
        env_title="PushT",
        goal_offset=50,
        episodes=100,
        out_path="rollouts/pusht_release20260728_goal50_horizon5_100ep_all_configs.png",
        x_clip=200,
    )
