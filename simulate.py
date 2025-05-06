# sr.py
"""
Simulation script for MaskedDynamicsEnsemble: side-by-side model vs ground truth rollout,
with variance-based state dropout, interactive replay, and CEM-based planning.
Supports merging a training config (train.yaml) with a simulation config (sim.yaml).
"""
import os
import sys
import threading
import queue
import time
import argparse

import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig

# ensure project src directory is on path
sys.path.append(os.path.dirname(__file__))

from models.ensemble import MaskedDynamicsEnsemble
from planning.cem import CEMPlanner
import gym
from gym.envs.registration import register


def start_input_thread(input_queue: queue.Queue):
    """Spawn a thread that pushes user keypresses into the queue."""
    def _reader():
        while True:
            inp = input()
            input_queue.put(inp)
    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def load_model_and_envs(cfg: DictConfig, device: torch.device):
    model = MaskedDynamicsEnsemble(cfg).to(device)

    # point to the *run*-specific folder, not the top‐level checkpoint_dir
    ckpt_root = cfg.train.checkpoint_dir
    # pick the latest run subdirectory by modification time
    run_dirs = [
        os.path.join(ckpt_root, d)
        for d in os.listdir(ckpt_root)
        if os.path.isdir(os.path.join(ckpt_root, d))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {ckpt_root}")
    latest_run = max(run_dirs, key=os.path.getmtime)

    # load the *model* checkpoint
    ckpt_files = [
        os.path.join(latest_run, f)
        for f in os.listdir(latest_run)
        if f.endswith('.pt')
    ]
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt files found in {latest_run}")
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    print(f"Loading model checkpoint: {latest_ckpt}")
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    model.eval()

    # register and instantiate envs
    register(
        id=cfg.sim.env_id,
        entry_point="pusht.pusht_wrapper:PushTWrapper",
        max_episode_steps=300,
        reward_threshold=1.0,
    )
    env_kwargs = cfg.sim.get('env_kwargs', {})
    env_model = gym.make(cfg.sim.env_id, **env_kwargs)
    env_gt    = gym.make(cfg.sim.env_id, **env_kwargs)
    env_model.reset()
    env_gt.reset()
    return model, env_model, env_gt

def process_states(raw_states: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """
    De-normalize and (optionally) convert sin/cos → θ.
    Accepts tensors whose *last dim* is the state dim.
    """
    orig_shape   = raw_states.shape            # (..., D)
    flat_states  = raw_states.reshape(-1, orig_shape[-1])   # (N, D)

    # ------------------- de-normalise --------------------
    if hasattr(cfg.data, "state_mean") and hasattr(cfg.data, "state_std"):
        mean = torch.tensor(cfg.data.state_mean, device=flat_states.device)
        std  = torch.tensor(cfg.data.state_std,  device=flat_states.device)
        flat_states = flat_states * std + mean

    # ------------------- sin/cos → θ ---------------------
    if cfg.sim.get("has_sincos", False):
        sin_idx, cos_idx = cfg.sim.sin_idx, cfg.sim.cos_idx
        theta = torch.atan2(flat_states[:, sin_idx],
                            flat_states[:, cos_idx]) % (2 * torch.pi)
        theta = theta.unsqueeze(1)

        keep_cols = [i for i in range(flat_states.size(1))
                     if i not in (sin_idx, cos_idx)]
        parts = [flat_states[:, i:i+1] for i in keep_cols]
        insert_pos = min(sin_idx, cos_idx)
        parts.insert(insert_pos, theta)
        flat_states = torch.cat(parts, dim=1)

    # --------------- restore original shape --------------
    return flat_states.reshape(*orig_shape[:-1], -1)


def simulate_rollouts(model, init_states, init_actions, init_mask, action_seqs, var_threshold, device, gt_states=None):
    with torch.no_grad():
        states_traj, masks_traj = model.rollout_with_dropout(
            init_states.to(device),
            init_actions.to(device),
            init_mask.to(device),
            action_seqs.to(device),
            var_threshold,
            gt_states=gt_states.to(device) if gt_states is not None else None
        )
    return states_traj.cpu(), masks_traj.cpu()


class SimulatorVisualizer:
    def __init__(self, env_model, env_gt):
        self.env_model = env_model
        self.env_gt    = env_gt
        self.init = False

    def init_plot(self, first_pred_vis, first_gt_vis):
        self.init = True
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 5))
        self.im_model = self.axes[0].imshow(first_pred_vis)
        self.axes[0].set_title('Model Prediction')
        self.axes[0].axis('off')
        self.im_gt = self.axes[1].imshow(first_gt_vis)
        self.axes[1].set_title('Ground Truth')
        self.axes[1].axis('off')
        self.text_var = self.axes[0].text(
            0.05, 0.95, '', transform=self.axes[0].transAxes,
            color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5)
        )
        plt.ion()
        plt.show()

    def update(self, pred_vis, gt_vis, var_vec):
        self.im_model.set_data(pred_vis)
        self.im_gt.set_data(gt_vis)
        var_text = ','.join([f"{v:.2f}" for v in var_vec])
        self.text_var.set_text(f"Var: {var_text}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(sim_cfg_path: str):
    # load your sim.yaml first
    sim_cfg = OmegaConf.load(sim_cfg_path)

    # where all runs are stored
    ckpt_root = 'checkpoints'
    run_dirs = [
        os.path.join(ckpt_root, d)
        for d in os.listdir(ckpt_root)
        if os.path.isdir(os.path.join(ckpt_root, d))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories under {ckpt_root}")

    # resolve candidate run directory
    cand = sim_cfg.sim.checkpoint_dir
    if os.path.isabs(cand):
        candidate_run = cand
    else:
        candidate_run = os.path.join(ckpt_root, cand)

    train_cfg_path = os.path.join(candidate_run, "config.yaml")

    # if that isn't a valid run folder, fall back to the latest under ckpt_root
    if not (os.path.isdir(candidate_run) and os.path.isfile(train_cfg_path)):
        latest_run = max(run_dirs, key=os.path.getmtime)
        print(f"  → '{candidate_run}' not found; using latest: {latest_run}")
        train_cfg_path = os.path.join(latest_run, "config.yaml")
    else:
        print(f"  → Using specified run folder: {candidate_run}")

    print(f"Loading training config: {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)

    # merge: train_cfg provides defaults, sim_cfg overrides
    cfg = OmegaConf.merge(train_cfg, sim_cfg)
    H = cfg.data.num_hist

    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.train.no_cuda else 'cpu')
    model, env_model, env_gt = load_model_and_envs(cfg, device)

    # load full trajectories
    from datasets.pusht_dset import load_pusht_slice_train_val
    datasets, traj_dset = load_pusht_slice_train_val(
        n_rollout=cfg.data.n_rollout,
        data_path=cfg.data.path,
        normalize_action=cfg.data.normalize_action,
        split_ratio=cfg.data.split_ratio,
        num_hist=cfg.data.num_hist,
        frameskip=1,
        with_velocity=cfg.data.with_velocity,
    )
    valid_trajs = traj_dset['valid']

    # simulate first batch_size trajectories
    for idx, (_, act_seq, state_seq, _) in enumerate(valid_trajs):
        act_t   = torch.tensor(act_seq, dtype=torch.float32, device=device)
        state_t = torch.tensor(state_seq, dtype=torch.float32, device=device)

        init_state = state_t[:1, :]  # first state
        init_states  = init_state.repeat(1, H, 1)        # (1,H,D)  replicate first state
        init_actions = torch.zeros(1, H, cfg.model.action_dim, device=device)
        init_mask   = torch.ones_like(init_states, dtype=torch.bool)  # start fully observed

        # future actions
        future_actions = act_t[1:, :].unsqueeze(0)  

        # planning
        if cfg.sim.get('use_planner', False):
            def cost_fn(final_states):
                if cfg.sim.planner_kwargs.use_gt:
                    goal_state = process_states(torch.zeros(final_states.shape[0], 8), cfg)
                    return torch.norm(final_states[:, 2:4] - goal_state[:, 2:4], dim=1)
                else:
                    return torch.norm(final_states[:, 2:4], dim=1)
            planner = CEMPlanner(
                dynamics_ensemble=model,
                cost_fn=cost_fn,
                action_dim=cfg.model.action_dim,
                horizon=future_actions.size(1),
                pop_size=cfg.sim.planner_kwargs.pop_size,
                elite_frac=cfg.sim.planner_kwargs.elite_frac,
                n_iter=cfg.sim.planner_kwargs.n_iter,
                var_threshold=cfg.sim.var_threshold,
                gt_env=env_gt if cfg.sim.planner_kwargs.use_gt else None,
                device=device,
            )
            if cfg.sim.planner_kwargs.get('closed_loop', False):
                viz = SimulatorVisualizer(env_model, env_gt)
                while True:
                    best_seq = planner.plan(process_states(init_state, cfg)[0].cpu().numpy() if cfg.sim.planner_kwargs.use_gt else init_states, init_actions, init_mask, agg_mode=cfg.sim.planner_kwargs.agg_mode, n_impute=cfg.sim.planner_kwargs.n_impute)
                    ret = env_gt.step(best_seq[0])[0]
                    gt_vis = ret['visual']
                    init_state = ret['state']
                    if not viz.init:
                        viz.init_plot(gt_vis, gt_vis)
                    else:
                        viz.update(gt_vis, gt_vis, [0, 0, 0, 0, 0, 0, 0])
                    time.sleep(cfg.sim.render_interval)
            else:
                best_seq = planner.plan(process_states(init_state, cfg)[0].cpu().numpy() if cfg.sim.planner_kwargs.use_gt else init_states, init_actions, init_mask, agg_mode=cfg.sim.planner_kwargs.agg_mode, n_impute=cfg.sim.planner_kwargs.n_impute)
                action_seqs = best_seq.unsqueeze(0)
        else:
            action_seqs = future_actions
        # rollout
        states_pred, masks_pred = simulate_rollouts(
            model, init_states, init_actions, init_mask, action_seqs,
            cfg.sim.var_threshold, device, state_t if cfg.sim.get('reset_state', False) else None
        )

        # de-normalize predictions
        pred_flat = states_pred.squeeze(0)
        pred_denorm = process_states(pred_flat, cfg)

        # de-normalize ground truth up to same length
        T1 = pred_denorm.size(0)
        gt_flat = state_t[:T1, :]
        gt_denorm = process_states(gt_flat, cfg)

        # interactive input
        input_queue = queue.Queue()
        start_input_thread(input_queue)

        # initialize visualizer
        first_pred_vis = env_model.prepare(0, pred_denorm[0].cpu().numpy())[0]['visual']
        first_gt_vis   = env_gt.prepare(0, gt_denorm[0].cpu().numpy())[0]['visual']
        viz = SimulatorVisualizer(env_model, env_gt)
        viz.init_plot(first_pred_vis, first_gt_vis)

        # playback
        while True:
            for t in range(T1 - 1):
                pred_vis = env_model.prepare(0, pred_denorm[t].cpu().numpy())[0]['visual']
                if cfg.sim.get('use_planner', False):
                    pred_action = action_seqs[0, t].cpu().numpy()
                    gt_vis = env_gt.step(pred_action)[0]['visual']
                else:
                    gt_vis = env_gt.prepare(0, gt_denorm[t].cpu().numpy())[0]['visual']
                var_vec  = masks_pred[0, t].tolist()
                viz.update(pred_vis, gt_vis, var_vec)
                time.sleep(cfg.sim.render_interval)
            if not input_queue.empty() and input_queue.get() == '':
                break

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python simulate.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
