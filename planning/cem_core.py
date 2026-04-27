from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


def _clamp_progress(progress: float) -> float:
    return max(0.0, min(1.0, float(progress)))


def _interp_level(start_idx: int, end_idx: int, progress: float) -> int:
    p = _clamp_progress(progress)
    value = float(start_idx) + (float(end_idx) - float(start_idx)) * p
    return int(round(value))


EvaluatePopulationFn = Callable[
    [torch.Tensor, int, List[int], int],
    Tuple[torch.Tensor, List[int], int],
]


class SharedCEMCore:
    """
    Shared CEM optimizer + fidelity scheduling used by both WM and GT-env backends.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        pop_size: int,
        elite_frac: float,
        n_iter: int,
        init_std: float,
        action_low: Optional[float],
        action_high: Optional[float],
        fidelity_cfg: Optional[Dict[str, Any]],
        num_levels: int,
        rollout_modes: Sequence[str],
        device: Optional[torch.device] = None,
        min_std: float = 0.01,
    ):
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.pop_size = int(pop_size)
        self.n_elite = max(1, int(round(self.pop_size * float(elite_frac))))
        self.n_iter = int(n_iter)
        self.init_std = float(init_std)
        self.min_std = float(min_std)
        self.action_low = action_low
        self.action_high = action_high
        self.device = device or torch.device("cpu")

        self.mu = torch.zeros(self.horizon, self.action_dim, device=self.device)
        self.std = torch.full((self.horizon, self.action_dim), self.init_std, device=self.device)
        self._has_distribution = False

        self.fidelity_cfg = fidelity_cfg or {}
        self.fidelity_enabled = bool(self.fidelity_cfg.get("enabled", True))
        self.num_levels = int(num_levels)
        self.mpc_cfg = self.as_cfg_dict(self.fidelity_cfg.get("mpc", {}), "fidelity.mpc")
        self.cem_cfg = self.as_cfg_dict(self.fidelity_cfg.get("cem", {}), "fidelity.cem")
        self.rollout_cfg = self.as_cfg_dict(self.fidelity_cfg.get("rollout", {}), "fidelity.rollout")
        self.mpc_mode = str(self.mpc_cfg.get("mode", "linear")).lower()
        self.cem_mode = str(self.cem_cfg.get("mode", "linear")).lower()
        self.rollout_mode = str(self.rollout_cfg.get("mode", "fixed")).lower()
        self.allowed_rollout_modes = {str(m).lower() for m in rollout_modes}

        if self.horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {self.horizon}")
        if self.pop_size <= 0:
            raise ValueError(f"pop_size must be > 0, got {self.pop_size}")
        if self.n_iter <= 0:
            raise ValueError(f"n_iter must be > 0, got {self.n_iter}")
        if self.init_std <= 0.0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")
        if self.min_std <= 0.0:
            raise ValueError(f"min_std must be > 0, got {self.min_std}")
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be > 0, got {self.num_levels}")
        if self.mpc_mode not in {"fixed", "linear"}:
            raise ValueError(
                f"Unknown fidelity.mpc.mode '{self.mpc_mode}'. Use 'fixed' or 'linear'."
            )
        if self.cem_mode not in {"fixed", "linear"}:
            raise ValueError(
                f"Unknown fidelity.cem.mode '{self.cem_mode}'. Use 'fixed' or 'linear'."
            )
        if self.rollout_mode not in self.allowed_rollout_modes:
            modes = ", ".join(sorted(self.allowed_rollout_modes))
            raise ValueError(
                f"Unknown fidelity.rollout.mode '{self.rollout_mode}'. "
                f"Allowed modes: {modes}."
            )

    def reset_distribution(self) -> None:
        self.mu.zero_()
        self.std.fill_(self.init_std)
        self._has_distribution = False

    def initialize_distribution(self, warm_start: bool = False, shift_steps: int = 0) -> None:
        if int(shift_steps) < 0:
            raise ValueError(f"shift_steps must be >= 0, got {shift_steps}")
        if bool(warm_start) and self._has_distribution:
            s = int(shift_steps)
            mu_prev = self.mu.clone()
            std_prev = self.std.clone()
            self.mu[:-s] = mu_prev[s:]
            self.mu[-s:] = 0.0
            self.std[:-s] = std_prev[s:]
            self.std[-s:] = self.init_std
        else:
            self.mu.zero_()
            self.std.fill_(self.init_std)

    def make_generator(self, rng_seed: Optional[int] = None) -> Optional[torch.Generator]:
        if rng_seed is None:
            return None
        device_name = "cuda" if self.device.type == "cuda" else "cpu"
        gen = torch.Generator(device=device_name)
        gen.manual_seed(int(rng_seed))
        return gen

    def sample_population(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        noise = torch.randn(
            self.pop_size,
            self.horizon,
            self.action_dim,
            generator=generator,
            device=self.device,
        )
        actions = self.mu.unsqueeze(0) + noise * self.std.unsqueeze(0)
        actions = self._clamp_action_tensor(actions)
        return actions

    def update_distribution(self, actions: torch.Tensor, costs: torch.Tensor) -> None:
        if not torch.is_tensor(costs):
            costs = torch.as_tensor(costs, dtype=torch.float32, device=self.device)
        else:
            costs = costs.to(self.device)
        elite_idxs = costs.topk(self.n_elite, largest=False).indices
        elite = actions[elite_idxs]
        #import pdb; pdb.set_trace()
        new_mu = elite.mean(dim=0)
        # unbiased=False avoids NaNs when n_elite==1 (Bessel ddof would divide by zero).
        std = elite.std(dim=0, unbiased=False)
        new_std = std.clamp(min=self.min_std)
        self._has_distribution = True
        alpha = 0#0.6
        self.mu = alpha * self.mu + (1 - alpha) * new_mu
        self.std = alpha * self.std + (1 - alpha) * new_std

    @staticmethod
    def as_cfg_dict(raw: Any, field_name: str) -> Dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"{field_name} must be a mapping, got {type(raw).__name__}.")
        return dict(raw)

    def _validated_level_index(self, idx: int) -> int:
        if idx < 0 or idx >= self.num_levels:
            raise ValueError(f"Level index {idx} out of range [0, {self.num_levels - 1}]")
        return idx

    def resolve_level_spec(
        self,
        raw_value: Any,
        base_level_idx: Optional[int],
        field_name: str,
    ) -> int:
        if raw_value is None:
            raise ValueError(
                f"{field_name} must be set to an int or token "
                "('coarsest', 'finest', or 'base')."
            )
        if isinstance(raw_value, bool):
            raise ValueError(f"{field_name} must be an int or token, got bool.")
        if isinstance(raw_value, int):
            return self._validated_level_index(raw_value)
        if isinstance(raw_value, float):
            if raw_value.is_integer():
                return self._validated_level_index(int(raw_value))
            raise ValueError(f"{field_name} must be integer-valued, got {raw_value}.")
        if isinstance(raw_value, str):
            token = raw_value.strip().lower()
            if token in {"coarsest", "min"}:
                return 0
            if token in {"finest", "max"}:
                return self.num_levels - 1
            if token in {"base", "auto"}:
                if base_level_idx is None:
                    raise ValueError(f"{field_name}='{raw_value}' requires a base level context.")
                return self._validated_level_index(int(base_level_idx))
            try:
                return self._validated_level_index(int(token))
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} has unknown token '{raw_value}'. "
                    "Use int, 'coarsest', 'finest', or 'base'."
                ) from exc
        raise ValueError(f"{field_name} has unsupported type {type(raw_value).__name__}.")

    def _stage_level_index(
        self,
        stage_cfg: Dict[str, Any],
        stage_mode: str,
        progress: float,
        field_prefix: str,
        base_level_idx: Optional[int] = None,
    ) -> int:
        if stage_mode == "fixed":
            level = stage_cfg.get("level", "finest")
            return self.resolve_level_spec(level, base_level_idx, f"{field_prefix}.level")
        start = stage_cfg.get("start_level", "coarsest")
        end = stage_cfg.get("end_level", "finest")
        start_idx = self.resolve_level_spec(start, base_level_idx, f"{field_prefix}.start_level")
        end_idx = self.resolve_level_spec(end, base_level_idx, f"{field_prefix}.end_level")
        idx = _interp_level(start_idx, end_idx, progress)
        return self._validated_level_index(idx)

    def base_level_index(self, mpc_progress: float, cem_progress: float) -> int:
        if not self.fidelity_enabled:
            return self.num_levels - 1
        mpc_idx = self._stage_level_index(
            self.mpc_cfg,
            self.mpc_mode,
            mpc_progress,
            "fidelity.mpc",
            base_level_idx=None,
        )
        return self._stage_level_index(
            self.cem_cfg,
            self.cem_mode,
            cem_progress,
            "fidelity.cem",
            base_level_idx=mpc_idx,
        )

    def rollout_level_indices(self, base_level_idx: int) -> List[int]:
        
        if self.rollout_mode == "fixed":
            level = self.rollout_cfg.get("level", "base")
            idx = self.resolve_level_spec(level, base_level_idx, "fidelity.rollout.level")
            return [idx] * self.horizon
        
        if self.rollout_mode == "linear":
            start = self.rollout_cfg.get("start_level", "base")
            end = self.rollout_cfg.get("end_level", "coarsest")
            start_idx = self.resolve_level_spec(
                start, base_level_idx, "fidelity.rollout.start_level"
            )
            end_idx = self.resolve_level_spec(
                end, base_level_idx, "fidelity.rollout.end_level"
            )
            if self.horizon == 1:
                return [start_idx]
            levels: List[int] = []
            for t in range(self.horizon):
                p = t / (self.horizon - 1)
                idx = _interp_level(start_idx, end_idx, p)
                levels.append(self._validated_level_index(idx))
            return levels

        if self.rollout_mode == "uncertainty_downshift":
            return []

        raise ValueError(f"Unknown rollout fidelity mode: {self.rollout_mode}")

    def _clamp_action_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_low is None and self.action_high is None:
            return actions
        low = -float("inf") if self.action_low is None else float(self.action_low)
        high = float("inf") if self.action_high is None else float(self.action_high)
        return actions.clamp(min=low, max=high)

    @torch.no_grad()
    def optimize(
        self,
        mpc_progress: float,
        evaluate_population: EvaluatePopulationFn,
        warm_start: bool = False,
        shift_steps: int = 0,
        rng_seed: Optional[int] = None,
        inject_actions: Optional[torch.Tensor] = None,
        inject_count: int = 1,
    ) -> tuple[torch.Tensor, int, List[int], int]:
        self.initialize_distribution(warm_start=warm_start, shift_steps=int(shift_steps))
        generator = self.make_generator(rng_seed=rng_seed)
        total_bits = 0
        eff_rollout_levels: List[int] = []

        inj: Optional[torch.Tensor] = None
        n_inj = 0
        if inject_actions is not None:
            inj = inject_actions.to(device=self.device, dtype=torch.float32).reshape(
                self.horizon, self.action_dim
            )
            inj = self._clamp_action_tensor(inj)
            n_inj = min(max(0, int(inject_count)), self.pop_size)

        for it in range(self.n_iter):
            cem_progress = 1.0 if self.n_iter == 1 else it / (self.n_iter - 1)
            base_level_idx = self.base_level_index(mpc_progress, cem_progress)
            rollout_levels = self.rollout_level_indices(base_level_idx)

            actions = self.sample_population(generator=generator)
            if inj is not None and n_inj > 0:
                actions[:n_inj] = inj.unsqueeze(0).expand(n_inj, -1, -1)
            costs, levels_used, bits_used = evaluate_population(
                actions,
                int(base_level_idx),
                list(rollout_levels),
                it,
            )
            #print(f"costs: {costs}")
            if not torch.is_tensor(costs):
                costs = torch.as_tensor(costs, dtype=torch.float32, device=self.device)
            else:
                costs = costs.to(self.device)
            if costs.ndim != 1 or int(costs.shape[0]) != self.pop_size:
                raise ValueError(
                    f"evaluate_population must return costs with shape ({self.pop_size},), "
                    f"got {tuple(costs.shape)}"
                )

            total_bits += int(bits_used)
            eff_rollout_levels = [int(x) for x in levels_used]
            self.update_distribution(actions, costs)

        final_level_idx = self.base_level_index(mpc_progress, 1.0)
        if self.rollout_mode == "uncertainty_downshift":
            final_rollout_levels = eff_rollout_levels
            if len(final_rollout_levels) > 0:
                final_level_idx = int(final_rollout_levels[-1])
        else:
            final_rollout_levels = self.rollout_level_indices(final_level_idx)

        self._has_distribution = True
        return (
            self.mu.detach().cpu(),
            int(final_level_idx),
            [int(x) for x in final_rollout_levels],
            int(total_bits),
        )
