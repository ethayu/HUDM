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
from models.world.model import HierWorldModel
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
from datasets.zarr_episodes import ZarrPushTEpisodes

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def make_cost_fn(goal_state: np.ndarray):
    """L2 distance (xy) + small weighted angle error."""

    def _cost(states: torch.Tensor) -> torch.Tensor:
        goal = torch.as_tensor(goal_state, dtype=states.dtype, device=states.device)
        eef_err = torch.norm(states[:, :2] - goal[:2], dim=1)
        pos_err = torch.norm(states[:, 2:4] - goal[2:4], dim=1)
        ang_err = torch.abs(torch.atan2(torch.sin(states[:, 4] - goal[4]),
                                        torch.cos(states[:, 4] - goal[4])))
        return pos_err + 0.1 * ang_err + eef_err

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
    init_obs: np.ndarray,
    cfg: DictConfig,
    dynamics_ensemble: HierWorldModel,
    gt_actions: np.ndarray
) -> Tuple[bool, list[np.ndarray], list[np.ndarray]]:
    """Run closed-loop CEM with parameters from *cfg* and return
    (success flag, trajectory of states, RGB frames).  *frames* is an empty
    list when cfg.save is False.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    planner = CEMPlanner(
        dynamics_ensemble=dynamics_ensemble,
        cost_fn=make_cost_fn(goal_state),
        action_dim=env_step.action_dim,
        horizon=cfg.planning_horizon,
        pop_size=cfg.pop_size,
        elite_frac=cfg.elite_frac,
        n_iter=cfg.n_iter,
        var_threshold=cfg.var_threshold,
        gt_env=None,
        gt_actions=gt_actions,
        n_env_samples=cfg.n_env_samples,
        device=device,
    )

    # Buffers: only previous step is needed for planning horizon>0 but we keep
    # them general in case we want to experiment later.
    H_hist = 1
    state_hist = torch.as_tensor(init_state, dtype=torch.float32, device=device).view(1, H_hist, -1)
    action_hist = torch.zeros(1, H_hist, env_step.action_dim, device=device)
    obs_hist = torch.as_tensor(init_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
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
            obs_hist,
            mask_hist,
            agg_mode="average",
            n_impute=1,
            gt_goal=goal_state,
        )
        env_rollout.headless = False

        # Execute the first *replan_every* actions from the planned sequence
        n_exec = min(replan_every, cfg.planning_horizon)
        pos_diffs = []
        angle_diffs = []
        eef_diffs = []
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
            dist = env_step.eval_state(goal_state, cur_state)
            pd = dist["pos_diff"]
            ad = dist["angle_diff"] 
            ed = dist["eef_diff"]
            print(f"step {t:03d}  pos_diff {pd:6.1f} angle_diff {ad:6.1f} eef_diff {ed:6.1f}")
            pos_diffs.append(pd)
            angle_diffs.append(ad)
            eef_diffs.append(ed)
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
                return True, trajectory, frames, pos_diffs, angle_diffs, eef_diffs

            t += 1  # increment global step counter

    return False, trajectory, frames, pos_diffs, angle_diffs, eef_diffs


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
        "planning_horizon": 5,
        "sample_horizon": 5,
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
        #disable_env_checker=True,   # skip strict checks (reset output etc.)
        #apply_api_compatibility=False,  # keep original (obs, reward, done, info) API
        **env_kwargs,
    )

    # Separate **headless** environment for CEM roll-outs ------------------
    env_rollout_wrapped = gym.make(
        env_id,
        #disable_env_checker=True,
        #apply_api_compatibility=False,
        **env_kwargs,
    )

    # Strip wrappers for both
    def _unwrap(e):
        while hasattr(e, "env"):
            e = e.env
        return e

    env_vis: PushTWrapper = PushTWrapper(with_velocity=True, with_target=True, add_noise=0)#_unwrap(env_vis_wrapped)
    env_rollout: PushTWrapper = PushTWrapper(with_velocity=True, with_target=True, add_noise=0)#_unwrap(env_rollout_wrapped)

    # sample initial / goal states (seed=0 for determinism)
    #init_state, goal_state = env_vis.sample_random_init_goal_states(seed=0,random_goal=True)
    # sample from synthetic dataset
    va_ds = ZarrPushTEpisodes(cfg.zarr_path, split='valid', split_ratio=cfg.split_ratio)
    observations, states, actions = va_ds.sample_traj_segment_from_dset(cfg.sample_horizon)

    env_vis.prepare(seed=0, init_state=states[0])
    dataset_gt_frames = []
    dataset_gt_frames.append(env_vis.render("rgb_array"))
    for i in range(len(states)):
        env_vis.prepare(seed=0, init_state=states[i])
        dataset_gt_frames.append(env_vis.render("rgb_array"))

    _, states = env_vis.rollout(seed=0, init_state=states[0], actions=actions)
    gt_frames = []
    for i in range(len(states)):
        env_vis.prepare(seed=0, init_state=states[i])
        gt_frames.append(env_vis.render("rgb_array"))

    init_state = states[0]
    goal_state = states[-1]
    init_obs = observations[0]
    env_vis.prepare(seed=0, init_state=init_state)
    gt_start = env_vis.render("rgb_array")
    env_rollout.prepare(seed=0, init_state=init_state)  # sync initial state

    # ------------------------------------------------------------------
    # Run planner -------------------------------------------------------
    # ------------------------------------------------------------------

     # Attempt to locate latest run dir
    ckpt_root = "/home/aurora/HUDM-mwm/checkpoints_world"
    run_dirs = [os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))]
    if not run_dirs:
        print('No checkpoint run directories found; using random weights')
        dynamics_ensemble = None
    else:
        latest = max(run_dirs, key=os.path.getmtime)
        print("loaded :", latest)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_cfg = OmegaConf.load(os.path.join(latest, 'world.yaml'))
        #load_model
        
        dynamics_ensemble = HierWorldModel(
            K=list(model_cfg.model.K),
            D=int(model_cfg.model.D),
            action_dim=model_cfg.data.action_dim,
            decoder_mode=str(getattr(model_cfg.model, "decoder_mode", "per_level")),
            dynamics_mode=str(getattr(model_cfg.model, "dynamics_mode", "per_level")),
        ).to(device)

        enc_p = os.path.join(latest, 'encoder.pt')
        if os.path.isfile(enc_p):
            dynamics_ensemble.encoder.load_state_dict(torch.load(enc_p, map_location=device))
        decoder_mode = str(getattr(model_cfg.model, "decoder_mode", "per_level"))
        if decoder_mode == "shared":
            dp = os.path.join(latest, "decoder.pt")
            if os.path.isfile(dp):
                dynamics_ensemble.decoder.load_state_dict(torch.load(dp, map_location=device))
        else:
            for li in range(len(model_cfg.model.K)):
                dp = os.path.join(latest, f'decoder_l{li}.pt')
                if os.path.isfile(dp):
                    dynamics_ensemble.decoders[li].load_state_dict(torch.load(dp, map_location=device))

        dynamics_ensemble.eval()
        

    save_frames = bool(cfg.save)
    success, traj, frames, pos_diffs, angle_diffs, eef_diffs = closed_loop_cem(
        env_vis, env_rollout, init_state, goal_state, init_obs,cfg, dynamics_ensemble=dynamics_ensemble, gt_actions=None#actions
    )

    env_vis.prepare(seed=0, init_state=goal_state)
    gt_goal = env_vis.render("rgb_array")

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
        existing = [f for f in os.listdir(run_dir) if f.endswith(".mp4")]
        vid_fname = f"rollout_{len(existing):03d}.mp4"
        vid_path = os.path.join(run_dir, vid_fname)

        # Write video to disk ------------------------------------------------
        fps = 15  # rough real-time rate; PushT runs at 30 but many headless
        print(f"[save] Writing rollout GIF to {vid_path} …")
        imageio.mimwrite(vid_path, frames, fps=fps)
        print(f"[save] Video saved ({len(frames)} frames)")

        #Save gt frames
        gt_vid_path = os.path.join(run_dir, "gt_frames.mp4")
        imageio.mimwrite(gt_vid_path, gt_frames, fps=fps)
        print(f"[save] GT frames saved ({len(gt_frames)} frames)")

        #Save dataset gt frames
        dataset_gt_vid_path = os.path.join(run_dir, "dataset_gt_frames.mp4")
        imageio.mimwrite(dataset_gt_vid_path, dataset_gt_frames, fps=fps)
        print(f"[save] Dataset GT frames saved ({len(dataset_gt_frames)} frames)")

        #Save gt goal and start overlaid on the same image
        # Overlay gt_start and gt_goal with alpha blending
        # Ensure both images are the same shape
        if gt_start.shape == gt_goal.shape:
            # Use start as base, overlay goal with 60% opacity for better visibility
            alpha = 0.6  # transparency for the goal overlay
            overlay = (gt_start.astype(np.float32) * (1 - alpha) + 
                      gt_goal.astype(np.float32) * alpha).astype(np.uint8)
        else:
            # If shapes don't match, just use goal (fallback)
            overlay = gt_goal
        
        gt_path = os.path.join(run_dir, "gt_start_goal_overlay.png")
        imageio.imwrite(gt_path, overlay)

        #plot pos_diffs and angle_diffs
        import matplotlib.pyplot as plt
        #xaxis is the step number
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.title('Positional, Angular and End-effector Distance vs Step')
        plt.plot(range(len(pos_diffs)), pos_diffs, label='pos_diffs')
        plt.plot(range(len(angle_diffs)), angle_diffs, label='angle_diffs')
        plt.plot(range(len(eef_diffs)), eef_diffs, label='eef_diffs')
        plt.legend()
        plt.savefig(os.path.join(run_dir, "pos_diffs_angle_diffs_eef_diffs.png"))
        plt.close() 

    print("Reached goal:" if success else "Failed:", success)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plan.py <path/to/config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
