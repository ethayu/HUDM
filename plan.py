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
  replan_every:  1        # execute this many actions before the next plan

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


# ---------------------------------------------------------------------------
# Closed-loop CEM planner
#   env_step     : environment instance we actually step in the real world
#   env_rollout  : *separate* instance used only for candidate-sequence roll-outs
# ---------------------------------------------------------------------------

def closed_loop_cem(
    env_step: PushTWrapper,
    env_rollout: PushTWrapper,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    cfg: DictConfig,
) -> Tuple[bool, list[np.ndarray], list[np.ndarray]]:
    """Run closed-loop CEM with parameters from *cfg* and return
    (success flag, trajectory of states, RGB frames).  *frames* is an empty
    list when cfg.save is False.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    planner = CEMPlanner(
        dynamics_ensemble=None,
        cost_fn=make_cost_fn(goal_state),
        action_dim=env_step.action_dim,
        horizon=cfg.horizon,
        pop_size=cfg.pop_size,
        elite_frac=cfg.elite_frac,
        n_iter=cfg.n_iter,
        var_threshold=cfg.var_threshold,
        gt_env=env_rollout,
        n_env_samples=cfg.n_env_samples,
        device=device,
    )

    # Buffers: only previous step is needed for planning horizon>0 but we keep
    # them general in case we want to experiment later.
    H_hist = 1
    state_hist = torch.as_tensor(init_state, dtype=torch.float32, device=device).view(1, H_hist, -1)
    action_hist = torch.zeros(1, H_hist, env_step.action_dim, device=device)
    mask_hist = torch.ones(1, H_hist, env_step.state_dim, dtype=torch.bool, device=device)

    trajectory: list[np.ndarray] = [init_state.copy()]

    # Frames for optional saving ------------------------------------------------
    frames: list[np.ndarray] = []
    if cfg.save:
        # Capture the initial state frame before any action is taken
        frames.append(env_step.render("rgb_array"))

    cur_state = init_state.copy()
    # ------------------------------------------------------------------
    #  Determine how many actions to execute before re-planning.  The new
    #  *replan_every* parameter lives under the same `plan:` section in the
    #  YAML config and defaults to 1 which recovers the previous closed-loop
    #  behaviour (plan → execute one action → re-plan).
    # ------------------------------------------------------------------
    replan_every: int = int(getattr(cfg, "replan_every", 1))
    replan_every = max(1, replan_every)

    t = 0  # environment step counter
    while t < cfg.steps:
        # ----------------------------------------------------------
        # Disable expensive pygame rendering during the many internal
        # rollouts performed by CEM.  We toggle a `headless` flag understood
        # by PushTEnv._render_frame so that rgb_array frames become a cheap
        # 1×1 placeholder.
        # ----------------------------------------------------------
        env_rollout.headless = True
        best_seq = planner.plan(
            state_hist,
            action_hist,
            mask_hist,
            agg_mode="average",
            n_impute=1,
        )
        env_rollout.headless = False

        # Execute the first *replan_every* actions from the planned sequence
        n_exec = min(replan_every, cfg.horizon)

        for i in range(n_exec):
            if t >= cfg.steps:
                break  # safety guard

            action_i = best_seq[i].cpu().numpy()
            print(f"Executing action {action_i}")

            # Step the environment with the chosen action
            _, reward, done, info = env_step.step(action_i)
            cur_state = info["state"]
            trajectory.append(cur_state.copy())

            # ---- Logging ------------------------------------------------
            dist = env_step.eval_state(goal_state, cur_state)["state_dist"]
            print(f"step {t:03d}  dist {dist:6.1f}")

            # ---- Update histories --------------------------------------
            state_t = torch.as_tensor(cur_state, dtype=torch.float32, device=device).view(1, 1, -1)
            action_t = torch.as_tensor(action_i, dtype=torch.float32, device=device).view(1, 1, -1)
            mask_t = torch.ones_like(state_t, dtype=torch.bool)

            state_hist = torch.cat([state_hist[:, -H_hist + 1 :], state_t], dim=1)
            action_hist = torch.cat([action_hist[:, -H_hist + 1 :], action_t], dim=1)
            mask_hist = torch.cat([mask_hist[:, -H_hist + 1 :], mask_t], dim=1)

            # ---- Rendering --------------------------------------------
            if cfg.render:
                try:
                    env_step.render("human")
                except Exception as e:
                    print(f"[render disabled] {e}")
                    cfg.render = False

            if cfg.save:
                frames.append(env_step.render("rgb_array"))

            # ---- Success check -----------------------------------------
            if env_step.eval_state(goal_state, cur_state)["success"]:
                return True, trajectory, frames

            t += 1  # increment global step counter

    return False, trajectory, frames


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
        "save": False,
        "replan_every": 1,
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
    # Gym ≥0.26 includes an automatic environment checker that assumes the
    # reset() method returns (obs, info) where *info* is a dict. The PushT
    # environment instead returns (obs, state) with *state* being an ndarray.
    # Disable the checker so that the legacy signature is accepted.
    # Visualisation / real‐step environment (full rendering capability)
    env_vis_wrapped = gym.make(
        env_id,
        disable_env_checker=True,   # skip strict checks (reset output etc.)
        apply_api_compatibility=False,  # keep original (obs, reward, done, info) API
        **env_kwargs,
    )

    # Separate **headless** environment for CEM roll-outs ------------------
    env_rollout_wrapped = gym.make(
        env_id,
        disable_env_checker=True,
        apply_api_compatibility=False,
        **env_kwargs,
    )

    # Strip wrappers for both
    def _unwrap(e):
        while hasattr(e, "env"):
            e = e.env
        return e

    env_vis: PushTWrapper = _unwrap(env_vis_wrapped)
    env_rollout: PushTWrapper = _unwrap(env_rollout_wrapped)

    # sample initial / goal states (seed=0 for determinism)
    init_state, goal_state = env_vis.sample_random_init_goal_states(seed=0)
    env_vis.prepare(seed=0, init_state=init_state)
    env_rollout.prepare(seed=0, init_state=init_state)  # sync initial state

    # ------------------------------------------------------------------
    # Run planner -------------------------------------------------------
    # ------------------------------------------------------------------

    save_frames = bool(cfg.save)
    success, traj, frames = closed_loop_cem(
        env_vis, env_rollout, init_state, goal_state, cfg
    )

    # ------------------------------------------------------------------
    # Persist rollout video if requested --------------------------------
    # ------------------------------------------------------------------
    if save_frames:
        try:
            import imageio.v2 as imageio  # imageio>=2.9
        except ModuleNotFoundError:  # pragma: no cover
            import imageio

        rollout_root = "rollouts"
        os.makedirs(rollout_root, exist_ok=True)

        from datetime import datetime

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(rollout_root, f"plan_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)

        # Determine filename index so that multiple rollouts in the same run
        # do not overwrite each other (even though the current script only
        # generates one rollout).  Counting any existing .mp4 files.
        existing = [f for f in os.listdir(run_dir) if f.endswith(".gif")]
        vid_fname = f"rollout_{len(existing):03d}.gif"
        vid_path = os.path.join(run_dir, vid_fname)

        # Write video to disk ------------------------------------------------
        fps = 15  # rough real-time rate; PushT runs at 30 but many headless
        print(f"[save] Writing rollout GIF to {vid_path} …")
        imageio.mimwrite(vid_path, frames, fps=fps)
        print(f"[save] Video saved ({len(frames)} frames)")

    print("Reached goal:" if success else "Failed:", success)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plan.py <path/to/config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
