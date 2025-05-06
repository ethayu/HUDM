# dynamics_model/new_model/planning/cem.py
"""
Cross-Entropy Method (CEM) planner for MaskedDynamicsEnsemble.
Generates candidate action sequences, rolls them out through the ensemble,
applies variance-dropout, imputes missing dimensions, and selects the best
sequence according to a provided cost function.
"""
import torch
from typing import Callable, Optional

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
        gt_env: Optional[object] = None,
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
        self.model = dynamics_ensemble
        self.cost_fn = cost_fn
        self.action_dim = action_dim
        self.horizon = horizon
        self.pop_size = pop_size
        self.n_elite = max(1, int(pop_size * elite_frac))
        self.n_iter = n_iter
        self.var_threshold = var_threshold
        self.device = device or torch.device('cpu')
        self.use_gt = gt_env is not None
        self.gt_env = gt_env

        # Initialize sampling distribution parameters
        # Mean and std for each timestep and action dimension
        self.mu = torch.zeros(horizon, action_dim, device=self.device)
        self.std = torch.ones(horizon, action_dim, device=self.device)

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
        P = self.pop_size
        H = state_hist.size(1)

        for _ in range(self.n_iter):

            actions = torch.normal(
                self.mu.unsqueeze(0).expand(P, -1, -1),
                self.std.unsqueeze(0).expand(P, -1, -1)
            ).to(self.device)

            if self.use_gt:
                actions_np = actions.detach().cpu().numpy()
                self.gt_env.prepare(0, state_hist[0, -1])
                for a in actions_np[0]:
                    state = self.gt_env.step(a)[0]["state"]
                final_states = torch.tensor(state, device=self.device).expand(P, -1)
                costs = self.cost_fn(final_states)

            else:
                s_hist = state_hist.expand(P, -1, -1).clone()   # (P,H,D)
                a_hist = action_hist.expand(P, -1, -1).clone()  # (P,H,A)
                m_hist = mask_hist.expand(P, -1, -1).clone()    # (P,H,D)

                states_traj, masks_traj, pred_vars = self.model.rollout_with_dropout(
                    s_hist, a_hist, m_hist, actions, self.var_threshold, return_vars=True
                )
                final_states = states_traj[:, -1, :]   # (P,D)
                final_masks  = masks_traj[:, -1, :]    # True = kept

                # ------ get μ,σ² so we can impute dropped dims ----------------
                mu_pred, var_pred = final_states, pred_vars # (P,D)

                # ------ multiple imputations & cost aggregation --------------
                costs_samples = []
                for _ in range(n_impute):
                    noise   = torch.randn_like(mu_pred)
                    imputes = mu_pred + torch.sqrt(var_pred) * noise
                    comp    = torch.where(final_masks, final_states, imputes)  # (P,D)
                    costs_samples.append(self.cost_fn(comp))                   # (P,)

                costs_stack = torch.stack(costs_samples, dim=0)  # (n_impute,P)
                if   agg_mode == "max":
                    costs = costs_stack.max(dim=0).values
                elif agg_mode == "min":
                    costs = costs_stack.min(dim=0).values
                else:  # "average"
                    costs = costs_stack.mean(dim=0)

            # -------------------------------------------------- CEM update
            elite_idxs    = costs.topk(self.n_elite, largest=False).indices
            elite_actions = actions[elite_idxs]           # (E,horizon,A)
            if self.use_gt:
                elite_actions = elite_actions.to(self.device)

            self.mu  = elite_actions.mean(dim=0)
            self.std = elite_actions.std(dim=0) + 1e-6

        return self.mu.cpu().detach()