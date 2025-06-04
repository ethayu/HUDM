"""plan.py — Closed-loop CEM planning via a YAML configuration file.

Example usage
-------------
python plan.py configs/plan.yaml

The configuration structure intentionally mirrors *configs/sim.yaml* for
consistency, but only the relevant fields are required:

plan:
  env:                     # kwargs forwarded to PushTWrapper
    add_noise: 2
    noise_std: [1.0,1.0,0.5,0.5,0.02,0,0]
    with_velocity: true
    with_target:  true

  horizon:        10       # planning horizon
  steps:          50       # max environment steps
  pop_size:      128
  elite_frac:    0.1
  n_iter:         5
  n_env_samples:  4        # roll-outs per candidate in the env-sampling path
  var_threshold:  0.5      # per-dimension variance cutoff
  render:        true

Other fields are accepted but ignored.  All entries have sensible defaults so
you can start with an empty YAML and override as needed.
"""

# -----------------------------------------------------------------------------
# Add the project root to PYTHONPATH so the script works when executed from
# outside the repository directory (same approach as *simulate.py*).
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from typing import Tuple

# Ensure that `pusht`, `planning`, etc. can be imported even when this script is
# launched via an absolute path from another working directory.
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

import gym
from gym.envs.registration import register
from pusht.pusht_wrapper import PushTWrapper
from planning.cem import CEMPlanner


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def make_cost_fn(goal_state: np.ndarray):
    """L2 distance (xy) + small weighted angle error."""

    def _cost(states: torch.Tensor) -> torch.Tensor:
        goal = torch.as_tensor(goal_state, dtype=states.dtype, device=states.device)
        pos_err = torch.norm(states[:, 2:4] - goal[2:4], dim=1)
        ang_err = torch.abs(torch.atan2(torch.sin(states[:, 4] - goal[4]),
                                        torch.cos(states[:, 4] - goal[4])))
        return pos_err + 0.1 * ang_err

    return _cost


def closed_loop_cem(
    env: PushTWrapper,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    cfg: DictConfig,
) -> Tuple[bool, list[np.ndarray]]:
    """Run closed-loop CEM with parameters from *cfg* and return (success, traj)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    planner = CEMPlanner(
        dynamics_ensemble=None,
        cost_fn=make_cost_fn(goal_state),
        action_dim=env.action_dim,
        horizon=cfg.horizon,
        pop_size=cfg.pop_size,
        elite_frac=cfg.elite_frac,
        n_iter=cfg.n_iter,
        var_threshold=cfg.var_threshold,
        gt_env=env,
        n_env_samples=cfg.n_env_samples,
        device=device,
    )

    # Buffers: only previous step is needed for planning horizon>0 but we keep
    # them general in case you want to experiment later.
    H_hist = 1
    state_hist = torch.as_tensor(init_state, dtype=torch.float32, device=device).view(1, H_hist, -1)
    action_hist = torch.zeros(1, H_hist, env.action_dim, device=device)
    mask_hist = torch.ones(1, H_hist, env.state_dim, dtype=torch.bool, device=device)

    trajectory = [init_state.copy()]

    cur_state = init_state.copy()
    for t in range(cfg.steps):
        best_seq = planner.plan(state_hist, action_hist, mask_hist,
                                agg_mode="average", n_impute=1)

        first_action = best_seq[0].cpu().numpy()

        _, reward, done, info = env.step(first_action)
        cur_state = info["state"]
        trajectory.append(cur_state.copy())

        # update histories (keep last H_hist entries)
        state_t = torch.as_tensor(cur_state, dtype=torch.float32, device=device).view(1, 1, -1)
        action_t = torch.as_tensor(first_action, dtype=torch.float32, device=device).view(1, 1, -1)
        mask_t = torch.ones_like(state_t, dtype=torch.bool)

        state_hist = torch.cat([state_hist[:, -H_hist + 1 :], state_t], dim=1)
        action_hist = torch.cat([action_hist[:, -H_hist + 1 :], action_t], dim=1)
        mask_hist = torch.cat([mask_hist[:, -H_hist + 1 :], mask_t], dim=1)

        if env.eval_state(goal_state, cur_state)["success"]:
            return True, trajectory

        if cfg.render:
            import time
            time.sleep(0.05)

    return False, trajectory


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(cfg_path: str):
    cfg_root = OmegaConf.load(cfg_path)

    # Provide defaults so small configs work --------------------------------
    defaults = {
        "env_id": "pusht",
        "env": {
            "with_velocity": True,
            "with_target": True,
            "add_noise": 0,
            "noise_std": 0.0,
        },
        "horizon": 10,
        "steps": 50,
        "pop_size": 128,
        "elite_frac": 0.1,
        "n_iter": 5,
        "n_env_samples": 4,
        "var_threshold": 1.0,
        "render": False,
    }

    cfg = OmegaConf.merge(defaults, cfg_root.get("plan", cfg_root))

    # ------------- Environment --------------------------------------------
    # ---------------- Environment loading (same as simulate.py) ----------
    env_id = cfg.get("env_id", "pusht")

    # Register our wrapper under the requested id (no-op if called twice)
    try:
        register(
            id=env_id,
            entry_point="pusht.pusht_wrapper:PushTWrapper",
            max_episode_steps=300,
            reward_threshold=1.0,
        )
    except gym.error.Error:
        # Already registered
        pass

    env_kwargs = cfg.env
    env: PushTWrapper = gym.make(env_id, **env_kwargs)

    # sample initial / goal states (seed=0 for determinism)
    init_state, goal_state = env.sample_random_init_goal_states(seed=0)
    env.prepare(seed=0, init_state=init_state)

    success, traj = closed_loop_cem(env, init_state, goal_state, cfg)

    print("Reached goal:" if success else "Failed:", success)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plan.py <path/to/config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
