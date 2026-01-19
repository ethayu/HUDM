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
        self.use_gt = self.use_env_sampling  # backward compat alias

        # Rollout sampling parameters
        self.n_env_samples = max(1, n_env_samples)

        # Initialize sampling distribution parameters
        # Mean and std for each timestep and action dimension
        self.mu = torch.zeros(horizon, action_dim, device=self.device)
        self.std = torch.ones(horizon, action_dim, device=self.device)*5

    def plan(
        self,
        state_hist:  torch.Tensor,   # (1,H,D)
        action_hist: torch.Tensor,   # (1,H,A)
        mask_hist:   torch.Tensor,   # (1,H,D)
        agg_mode: str = "average",   # {"max","min","average"}
        n_impute: int = 4            # ≥1
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

            if self.use_env_sampling:
                # ----------------------------------------------------------
                #  Evaluate each candidate by empirical env rollouts
                # ----------------------------------------------------------
                final_states_list = []
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
                        _, states = self.gt_env.rollout(seed, state_hist_np, act_seq_np)
                        # states returned shape (horizon+1, D); discard first
                        sample_states.append(states[1:])

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
                    final_masks_list.append(torch.tensor(masks[-1],  device=self.device, dtype=torch.bool))
                    var_last_list.append(torch.tensor(var_pred_np[-1], device=self.device))

                final_states = torch.stack(final_states_list, dim=0)  # (P,D)
                final_masks  = torch.stack(final_masks_list,  dim=0)  # (P,D)
                var_pred     = torch.stack(var_last_list,     dim=0)  # (P,D)

                # Imputation & cost aggregation ---------------------------
                costs_samples = []
                for _ in range(n_impute):
                    noise   = torch.randn_like(final_states)
                    imputes = final_states + torch.sqrt(var_pred) * noise
                    comp    = torch.where(final_masks, final_states, imputes)
                    costs_samples.append(self.cost_fn(comp))
                costs_stack = torch.stack(costs_samples, dim=0)  # (n_impute,P)
                if   agg_mode == "max":
                    costs = costs_stack.max(dim=0).values
                elif agg_mode == "min":
                    costs = costs_stack.min(dim=0).values
                else:
                    costs = costs_stack.mean(dim=0)

            else:
                # ---- Use learned ensemble (original behaviour) ----------
                s_hist = state_hist.expand(P, -1, -1).clone()[:,-1,:]   # (P,H,D)

                s_flat = s_hist.view(P, -1)

                z = self.model.encoder(s_flat)

                pred_roll_z_k = []
                pred_roll_s_k = []
                for li, k in enumerate(self.model.K):
                    preds_z = []
                    preds_s = []
                    z_prev = z[:, :k]
                    for t in range(self.horizon):
                        z_prev = self.model.dynamics[li].step(z_prev, actions[:, t, :])
                        preds_z.append(z_prev)
                        pred_s = self.model.decoders[li](z_prev)
                        preds_s.append(pred_s)
                    pred_roll_z = torch.stack(preds_z, dim=1)
                    pred_roll_s = torch.stack(preds_s, dim=1)
                    pred_roll_z_k.append(pred_roll_z)
                    pred_roll_s_k.append(pred_roll_s) #largest dimension, #candidate, time, D ; 4,10,50,7

                # multiple imputations
                costs_samples = []
                for k in range(len(self.model.K)):
                    comp = pred_roll_s_k[k][ :, -1, :]
                    costs_samples.append(self.cost_fn(comp))

                costs_stack = torch.stack(costs_samples, dim=0)
                if agg_mode == "max":
                    costs = costs_stack.max(dim=0).values
                elif agg_mode == "min":
                    costs = costs_stack.min(dim=0).values
                else:
                    #costs = costs_stack[-1]
                    costs = costs_stack.mean(dim=0)

            # -------------------------------------------------- CEM update
            elite_idxs    = costs.topk(self.n_elite, largest=False).indices
            elite_actions = actions[elite_idxs,0]           # (E,horizon,A)
            if self.use_gt:
                elite_actions = elite_actions.to(self.device)
            import pdb; pdb.set_trace()
            self.mu  = elite_actions.mean(dim=0)
            self.std = elite_actions.std(dim=0) + 1e-6
            # Compute average cost over samples (dim=1)
            avg_cost = costs_stack.mean(dim=1).data.cpu().numpy()  # shape: (4,)

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

            #plot the costs at each iter


        return self.mu.cpu().detach(), costs_stack