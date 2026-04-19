from __future__ import annotations

import unittest

import torch

from planning.latent_cem import LatentCEMPlanner
from planning.latent_cem_batch import BatchedLatentCEMPlanner


class TinyWorldModel:
    def __init__(self):
        self.K = [1, 2, 4]
        self.D = 4
        self.num_members = 1

    def eval(self):
        return self

    def predict_next(self, level: int, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        k = self.K[level]
        action_term = a.sum(dim=1, keepdim=True).expand(-1, k)
        return z[:, :k] + 0.05 * action_term + 0.01 * float(level + 1)

    def predict_next_stats(self, level: int, z: torch.Tensor, a: torch.Tensor):
        mu = self.predict_next(level, z, a)
        var = torch.zeros_like(mu)
        return mu, var


class BatchedLatentPlannerTests(unittest.TestCase):
    def setUp(self):
        self.world_model = TinyWorldModel()
        self.objective = {
            "latent_metric": "l2",
            "terminal_weight": 1.0,
            "running_weight": 0.2,
            "action_l2_weight": 0.1,
        }
        self.common_kwargs = {
            "world_model": self.world_model,
            "horizon": 5,
            "action_dim": 2,
            "pop_size": 32,
            "elite_frac": 0.25,
            "n_iter": 4,
            "init_std": 0.4,
            "objective_cfg": self.objective,
            "warm_start": True,
            "device": torch.device("cpu"),
        }
        self.fidelity_cfgs = [
            {
                "enabled": True,
                "mpc": {"mode": "fixed", "level": "finest"},
                "cem": {"mode": "fixed", "level": "base"},
                "rollout": {"mode": "fixed", "level": "base", "uncertainty": {"criterion": "mean", "threshold": 0.1, "percentile": 0.8, "min_level": "coarsest", "max_downshifts_per_step": 1}},
            },
            {
                "enabled": True,
                "mpc": {"mode": "linear", "start_level": "coarsest", "end_level": "finest"},
                "cem": {"mode": "linear", "start_level": "base", "end_level": "finest"},
                "rollout": {"mode": "linear", "start_level": "base", "end_level": "coarsest", "uncertainty": {"criterion": "mean", "threshold": 0.1, "percentile": 0.8, "min_level": "coarsest", "max_downshifts_per_step": 1}},
            },
        ]

    def test_batched_matches_serial(self):
        z0 = torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float32)
        z_goal = torch.tensor([[0.5, 0.1, -0.3, 0.2]], dtype=torch.float32)
        seeds = [123, 987]

        serial_results = []
        for fidelity_cfg, seed in zip(self.fidelity_cfgs, seeds):
            planner = LatentCEMPlanner(fidelity_cfg=fidelity_cfg, **self.common_kwargs)
            action_seq, info = planner.plan(z0, z_goal, mpc_progress=0.3, warm_start_steps=0, seed=seed)
            serial_results.append((action_seq, info))

        batched = BatchedLatentCEMPlanner(fidelity_cfgs=self.fidelity_cfgs, **self.common_kwargs)
        batched_results = batched.plan_batch(
            z0=z0.expand(len(self.fidelity_cfgs), -1),
            z_goal=z_goal.expand(len(self.fidelity_cfgs), -1),
            mpc_progress=0.3,
            warm_start_steps=0,
            seeds=seeds,
        )

        for (serial_actions, serial_info), batched_result in zip(serial_results, batched_results):
            self.assertTrue(torch.allclose(serial_actions, batched_result.action_seq, atol=1e-6, rtol=1e-6))
            self.assertEqual(serial_info.base_level_idx, batched_result.info.base_level_idx)
            self.assertEqual(serial_info.start_level_idx, batched_result.info.start_level_idx)
            self.assertEqual(serial_info.rollout_level_indices, batched_result.info.rollout_level_indices)


if __name__ == "__main__":
    unittest.main()
