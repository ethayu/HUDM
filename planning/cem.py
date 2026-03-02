# dynamics_model/new_model/planning/cem.py
"""
Cross-Entropy Method (CEM) planner for MaskedDynamicsEnsemble.
Generates candidate action sequences, rolls them out through the ensemble,
applies variance-dropout, imputes missing dimensions, and selects the best
sequence according to a provided cost function.
"""
import torch
import numpy as np
from typing import Callable, Optional
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid, save_image

ACTION_MEAN = torch.tensor([-0.0087, 0.0068]).cuda()
ACTION_STD = torch.tensor([0.2019, 0.2002]).cuda()
STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
class CEMPlanner:
    def __init__(
        self,
        dynamics_ensemble,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        action_dim: int,
        horizon: int,
        pop_size: int = 256,
        elite_frac: float = 0.1,
        n_iter: int = 5,
        var_threshold: float = 0.1,
        gt_env: Optional[object] = None,      # optional ground-truth env for sampling
        gt_actions: Optional[np.ndarray] = None,
        n_env_samples: int = 4,               # number of env rollouts per candidate when no learned model
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dynamics_ensemble: MaskedDynamicsEnsemble instance
            cost_fn: function(states: (pop, state_dim)) -> costs: (pop,)
            action_dim: dimensionality of each action vector
            horizon: number of timesteps to plan over
            pop_size: number of candidate sequences per iteration
            elite_frac: fraction of candidates to consider elite
            n_iter: number of CEM iterations
            var_threshold: variance threshold for dropout
            device: torch device
        """
        self.model = dynamics_ensemble  # can be None when using pure env sampling
        self.cost_fn = cost_fn
        self.action_dim = action_dim
        self.horizon = horizon
        self.pop_size = pop_size
        self.n_elite = max(1, int(pop_size * elite_frac))
        self.n_iter = n_iter
        self.var_threshold = var_threshold
        self.device = device or torch.device('cpu')
        # If no model is provided but a ground-truth env is, we fall back to
        # empirical sampling of the env to obtain mean / variance predictions.
        self.gt_env = gt_env
        # We treat *use_gt* as "use the ground-truth env for rollouts", which is
        # possible both in the *model-free* case (dynamics_ensemble is None) and
        # when the user explicitly wants to benchmark performance against the
        # GT environment despite having a model.  In the latter case we simply
        # ignore the ensemble and go down the env-sampling branch.
        self.use_env_sampling = gt_env is not None  # env available for rollouts
        if dynamics_ensemble is not None and not self.use_env_sampling:
            # standard model-based planning
            self.use_env_sampling = False
        self.use_env_sampling = False

        self.use_gt = self.use_env_sampling  # backward compat alias
        self.gt_actions = gt_actions
        # Rollout sampling parameters
        self.n_env_samples = max(1, n_env_samples)

        # Initialize sampling distribution parameters
        # Mean and std for each timestep and action dimension
        self.mu = torch.zeros(horizon, action_dim, device=self.device)
        self.std = torch.ones(horizon, action_dim, device=self.device)
    
    def preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        img_size = 96
        im = Image.fromarray(obs.astype(np.uint8)) if obs.dtype != np.uint8 else Image.fromarray(obs)
        im = im.resize((img_size, img_size), Image.BILINEAR)
        return (np.asarray(im).astype(np.float32) / 255.0) * 2.0 - 1.0
    def denorm(self,x):
        return (x * 0.5 + 0.5).clamp(0,1)

    def plan(
        self,
        state_hist:  torch.Tensor,   # (1,H,D)
        action_hist: torch.Tensor,   # (1,H,A)
        obs_hist: torch.Tensor,   # (1,H,)
        mask_hist:   torch.Tensor,   # (1,H,D)
        agg_mode: str = "average",   # {"max","min","average"}
        n_impute: int = 4,            # ≥1
        gt_goal: np.ndarray = None
    ) -> torch.Tensor:
        """
        Run CEM and return the best action sequence.

        The cost of each candidate trajectory is computed by
        1. Sampling `n_impute` completions of the missing dims
        2. Evaluating `cost_fn` for each completion
        3. Aggregating across imputations using agg_mode.
        """
        assert agg_mode in {"max", "min", "average"}, "agg_mode must be max/min/average"
        self.mu.zero_()
        self.std.fill_(1.0)
        P = self.pop_size

        for nn in range(self.n_iter):

            # -------- sample candidate action sequences -------------------
            actions = torch.normal(
                self.mu.unsqueeze(0).expand(P, -1, -1),
                self.std.unsqueeze(0).expand(P, -1, -1)
            ).to(self.device)  # (P, horizon, A)

            #insert gt actions
            if self.gt_actions is not None:
                actions[0] = self.gt_actions

            if self.use_env_sampling:
                # ----------------------------------------------------------
                #  Evaluate each candidate by empirical env rollouts
                # ----------------------------------------------------------
                final_states_list = []
                final_obses_list = []
                final_masks_list  = []
                var_last_list     = []

                # Flatten tensors for easier numpy conversion once
                state_hist_np = state_hist[0, -1].detach().cpu().numpy()  # (D,)

                for p in range(P):
                    act_seq_np = actions[p].detach().cpu().numpy()

                    # Collect multiple rollouts
                    sample_states = []  # (S, horizon, D)
                    for s_idx in range(self.n_env_samples):
                        seed = np.random.randint(0, 2**31 - 1)
                        self.gt_env.prepare(seed=seed, init_state=state_hist_np)
                        obses, states = self.gt_env.rollout(seed, state_hist_np, act_seq_np)
                        # states returned shape (horizon+1, D); discard first
                        sample_states.append(states[1:])
                        """
                        if(p==0):
                            a = [torch.tensor(self.preprocess_obs(obs), device=self.device).permute(2,0,1) for obs in obses["visual"]]
                            a = torch.stack(a, dim=0)
                            a = self.denorm(a)
                            save_image(a, f"gt_rollout_ep0.png")
                        """
                    sample_states = np.stack(sample_states, axis=0)  # (S,K,D)

                    mu_pred_np  = sample_states.mean(axis=0)          # (K,D)
                    var_pred_np = sample_states.var(axis=0)           # (K,D)

                    # Build dropout masks over time
                    masks = []
                    prev_mask = np.ones(mu_pred_np.shape[1], dtype=bool)
                    for k in range(mu_pred_np.shape[0]):
                        new_mask = np.logical_and(prev_mask, var_pred_np[k] <= self.var_threshold)
                        masks.append(new_mask)
                        prev_mask = new_mask
                    masks = np.stack(masks, axis=0)  # (K,D)

                    final_states_list.append(torch.tensor(mu_pred_np[-1], device=self.device))
                    final_obses_list.append(torch.tensor(self.preprocess_obs(obses["visual"][-1]), device=self.device).permute(2,0,1))
                    final_masks_list.append(torch.tensor(masks[-1],  device=self.device, dtype=torch.bool))
                    var_last_list.append(torch.tensor(var_pred_np[-1], device=self.device))

                final_states = torch.stack(final_states_list, dim=0)  # (P,D)
                final_masks  = torch.stack(final_masks_list,  dim=0)  # (P,D)
                final_obses  = torch.stack(final_obses_list,  dim=0)  # (P,H,)
                var_pred     = torch.stack(var_last_list,     dim=0)  # (P,D)
                #import pdb; pdb.set_trace()

                # Imputation & cost aggregation ---------------------------
                # normalised in [-1,1]; save_image expects [0,1], but make_grid handles; we will clamp after denorm
                
                gt_goal_embedding = self.model.encoder(gt_goal.unsqueeze(0).cuda())
                final_states_embedding = self.model.encoder(final_obses.float().cuda())
                
                recon = self.model.decode(2, final_states_embedding).squeeze(0)
                gt_recon = self.model.decode(2, gt_goal_embedding).squeeze(0)
                """
                B, L, A = actions.shape
                actions = (actions - ACTION_MEAN) / ACTION_STD
                a_null = torch.zeros((B, 1, A), device=self.device, dtype=actions.dtype)
                a_full = torch.cat([actions,a_null], dim=1)  # (B,L,A)  
                a_full_flat = a_full.view(B * (L+1), -1)

                obs_hist = obs_hist.expand(P, -1, -1, -1, -1).clone()[:,-1,:]   # (P,H,)
                z = self.model.encoder(obs_hist)

                for li, k in enumerate(self.model.K):
                    z_prev = z[:, :k]
                    preds = []
                    for t in range(L):
                        z_prev = self.model.dynamics[li].step(z_prev, a_full[:, t, :])
                        preds.append(z_prev)
                    pred_roll = torch.stack(preds, dim=1)  # (B,L,k)

                    recon_s = []
                    for t in range(L):
                        recon_s.append(self.model.decode(li, pred_roll[:,t]))
                    recon_s = torch.stack(recon_s, dim=1)  # (B,L,k)
                    save_image(self.denorm(recon_s[0]), f"recon_rollout_level{li}_ep0.png")
                    import pdb; pdb.set_trace()
                """
                #save_image(denorm(recon), f"recon_level{level}.png")
                #save_image(denorm((final_obses)), "gt.png")
                
            
                #diff = (gt_goal.unsqueeze(0).cuda() - final_obses)
                #costs = torch.linalg.norm(diff.flatten(start_dim=1), ord=2, dim=1)
                #costs = self.cost_fn(final_states_embedding, gt_goal_embedding)  # (P,)
                #diff = (denorm(recon.cuda()) - denorm(gt_recon.cuda()))
                diff = (gt_goal_embedding - final_states_embedding)
                costs = torch.linalg.norm(diff.flatten(start_dim=1), ord=2, dim=1)
                print("costs", costs)

            else:
                # ---- Use learned ensemble (original behaviour) ----------
                s_hist = state_hist.expand(P, -1, -1).clone()[:,-1,:]   # (P,H,D)

                obs_hist = obs_hist.expand(P, -1, -1, -1, -1).clone()[:,-1,:]   # (P,H,)

                z = self.model.encoder(obs_hist)


                pred_roll_z_k = []
                pred_roll_s_k = []

                gt_goal_embedding = self.model.encoder(gt_goal.unsqueeze(0).cuda())

                actions = (actions - ACTION_MEAN) / ACTION_STD
                costs_samples = []
                for li, k in enumerate(self.model.K):
                    preds_z = []
                    preds_s = []
                    z_prev = z[:, :k]
                    for t in range(actions.shape[1]):
                        z_prev = self.model.dynamics[li].step(z_prev, actions[:, t, :])
                        preds_z.append(z_prev)
                        pred_s = self.model.decoders[li](z_prev)
                        preds_s.append(pred_s)
                    pred_roll_z = torch.stack(preds_z, dim=1)
                    pred_roll_s = torch.stack(preds_s, dim=1)
                    pred_roll_z_k.append(pred_roll_z)
                    pred_roll_s_k.append(pred_roll_s) #largest dimension, #candidate, time, D ; 4,10,50,7
                    final_states_embedding = pred_roll_z[:, -1, :]
                    diff = (gt_goal_embedding[:,:k] - final_states_embedding)
                    costs_samples.append(torch.linalg.norm(diff.flatten(start_dim=1), ord=2, dim=1))

                costs = torch.mean(torch.stack(costs_samples, dim=1), dim=1)

            # -------------------------------------------------- CEM update
            elite_idxs   = costs.topk(self.n_elite, largest=False).indices
            print("elite costs", costs[elite_idxs[0]])
            actions = actions*ACTION_STD+ACTION_MEAN
            #import pdb; pdb.set_trace()
            elite_actions = actions[elite_idxs]           # (E,horizon,A)
            if self.use_gt:
                elite_actions = elite_actions.to(self.device)
            self.mu  = elite_actions.mean(dim=0)
            self.std = elite_actions.std(dim=0) + 1e-6
            
            # Compute average cost over samples (dim=1)
            avg_cost = costs.mean(dim=0).data.cpu().numpy()  # shape: (4,)
            """
            # Plot
            k = torch.arange(len(avg_cost))

            plt.figure()
            plt.plot(k, avg_cost)
            plt.xlabel("k")
            plt.ylabel(f"Average cost per sample")
            plt.title(f"Average Cost vs. k at iter {nn}")
            # Save figure
            plt.savefig(f"avg_cost_vs_k_iter{nn}.png", dpi=200, bbox_inches="tight")
            plt.close()
            """

            #plot the costs at each iter


        return self.mu.cpu().detach()#, costs_stack