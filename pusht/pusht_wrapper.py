import numpy as np
import cv2
import torch
from pusht.pusht_env import PushTEnv
from pusht.utils import aggregate_dct
from planning.fidelity import apply_fidelity

class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
            render_size=96,
            relative=True,
            action_scale=100,
            add_noise: int = 0,
            noise_std = 0.1,
            **kwargs,
        ):
        """
        Args
        -----
        with_velocity / with_target : see PushTEnv

        add_noise :
            0 → no noise (default)
            1 → additive Gaussian noise is applied to every action dimension
            2 → additive Gaussian noise is applied to every *state* dimension

        noise_std : float | Sequence[float]
            Standard deviation(s) of the Gaussian noise.  Can be
            • scalar → same σ for all dimensions, or
            • sequence / list / np.ndarray of length = action_dim (mode 1) or
              state_dim  (mode 2).
        """
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target,
            render_size=render_size,
            relative=relative,
            action_scale=action_scale,
            **kwargs,
        )
        self.action_dim = self.action_space.shape[0]
        self.add_noise = add_noise  # 0: none, 1: action noise, 2: state noise

        # --------------------------------------------------------------
        # Noise standard deviation can be:
        #   • scalar        → same σ across all dims
        #   • sequence/list → per-dim σ (length must match action_dim or
        #                      state_dim depending on noise mode)
        # We keep the original user-supplied object for reference, actual
        # vector σ is materialised lazily in the step() call so that we can
        # raise useful dimension-mismatch errors only when the information
        # (action_dim / state_dim) is available.
        # --------------------------------------------------------------
        self._noise_std_raw = noise_std
        
        # Set state dimension based on velocity option
        if self.with_velocity:
            self.state_dim = 7  # [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        else:
            self.state_dim = 5  # [agent_x, agent_y, T_x, T_y, angle]

        # Planning-time fidelity controls (used by gt_env planning backend).
        self._planning_fidelity_enabled = False
        self._planning_fidelity_num_levels = 1
        self._planning_fidelity_level_idx = 0
        self._planning_fidelity_cfg = {
            "mode": "blur_avgpool",
            "blur_sigma_max": 2.0,
            "pool_scale_max": 4,
            "quantize_levels_min": 8,
            "quantize_levels_max": 256,
            "action_noise_std_max": 0.0,
            "downsample_output": False,
            "min_downsample_size": 12,
        }

    def configure_planning_fidelity(
        self,
        enabled: bool,
        num_levels: int,
        cfg: dict | None = None,
    ) -> None:
        self._planning_fidelity_enabled = bool(enabled)
        self._planning_fidelity_num_levels = max(1, int(num_levels))
        self._planning_fidelity_level_idx = self._planning_fidelity_num_levels - 1
        if cfg is not None:
            merged = dict(self._planning_fidelity_cfg)
            merged.update(cfg)
            self._planning_fidelity_cfg = merged

    def set_planning_fidelity_level(self, level_idx: int) -> None:
        li = int(level_idx)
        li = max(0, min(li, self._planning_fidelity_num_levels - 1))
        self._planning_fidelity_level_idx = li

    def _planning_fidelity_level(self) -> float:
        if self._planning_fidelity_num_levels <= 1:
            return 1.0
        return float(self._planning_fidelity_level_idx) / float(self._planning_fidelity_num_levels - 1)

    def _apply_planning_fidelity_visual(self, visual: np.ndarray) -> np.ndarray:
        if (not self._planning_fidelity_enabled) or visual is None:
            return visual

        cfg = self._planning_fidelity_cfg
        level = self._planning_fidelity_level()
        x = torch.as_tensor(visual.astype(np.float32) / 255.0)
        x = apply_fidelity(
            x,
            level=level,
            mode=str(cfg.get("mode", "blur_avgpool")),
            blur_sigma_max=float(cfg.get("blur_sigma_max", 2.0)),
            pool_scale_max=int(cfg.get("pool_scale_max", 4)),
            quantize_levels_min=int(cfg.get("quantize_levels_min", 8)),
            quantize_levels_max=int(cfg.get("quantize_levels_max", 256)),
        )
        img = x.detach().cpu().numpy()

        if bool(cfg.get("downsample_output", False)):
            h, w = img.shape[:2]
            min_side = min(h, w)
            min_target = max(4, int(cfg.get("min_downsample_size", 12)))
            target = int(round(min_target + level * (min_side - min_target)))
            target = max(4, min(target, min_side))
            if target < min_side:
                if h <= w:
                    h2 = target
                    w2 = max(1, int(round(w * (target / h))))
                else:
                    w2 = target
                    h2 = max(1, int(round(h * (target / w))))
                img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)

        return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)

    def set_goal_from_state(self, goal_state) -> None:
        goal_state = np.asarray(goal_state, dtype=np.float32)
        if goal_state.shape[0] < 5:
            raise ValueError(f"goal_state must have at least 5 dims, got {goal_state.shape}")
        self.set_task_goal(goal_state[2:5])

    def set_start_from_state(self, start_state) -> None:
        start_state = np.asarray(start_state, dtype=np.float32)
        if start_state.shape[0] < 5:
            raise ValueError(f"start_state must have at least 5 dims, got {start_state.shape}")
        self.set_task_start(start_state[2:5])

    def sample_dataset_init_goal_states(
        self,
        dataset,
        trajectory_len: int,
        split: str = "valid",
        split_ratio: float = 0.8,
        seed: int = 0,
    ):
        if trajectory_len <= 0:
            raise ValueError(f"trajectory_len must be > 0, got {trajectory_len}")

        # Accept either a zarr path or a dataset-like object with state/starts/ends arrays.
        if isinstance(dataset, str):
            try:
                import zarr
            except Exception as exc:
                raise ImportError("zarr not installed. pip install zarr") from exc
            root = zarr.open_group(dataset, mode="r")
            state_arr = root["data"]["state"]    
            action_arr = root["data"]["action"]        
            ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
            starts = np.zeros_like(ends)
            starts[0] = 0
            for i in range(1, len(ends)):
                starts[i] = ends[i - 1] + 1
        else:
            state_arr = dataset.state
            ends = np.asarray(dataset.ends, dtype=np.int64)
            starts = np.asarray(dataset.starts, dtype=np.int64)

        n_ep = len(ends)
        if n_ep == 0:
            raise ValueError("Dataset has no episodes.")
        n_train = int(float(split_ratio) * n_ep)
        split_l = str(split).lower()
        if split_l == "train":
            ep_idx = np.arange(0, n_train)
        else:
            ep_idx = np.arange(n_train, n_ep)
        if len(ep_idx) == 0:
            raise ValueError(f"No episodes in split '{split}'.")

        candidates = []
        for ei in ep_idx:
            s = int(starts[ei])
            e = int(ends[ei])
            # states are indexed per transition step; choosing i and i+T gives a T-step trajectory.
            if e - s >= trajectory_len:
                candidates.append((ei, s, e))
        if not candidates:
            raise ValueError(
                f"No episodes in split '{split}' have length >= {trajectory_len + 1} states."
            )

        rng = np.random.default_rng(seed=seed)
        ei, s, e = candidates[int(rng.integers(0, len(candidates)))]
        start_idx = int(rng.integers(s, e - trajectory_len + 1))
        goal_idx = start_idx + trajectory_len

        init_state = np.asarray(state_arr[start_idx], dtype=np.float32)
        _, states = self.rollout(seed=0, init_state=init_state, actions=action_arr[start_idx:goal_idx])
        goal_state = np.asarray(states[-1], dtype=np.float32)
        if init_state.shape[0] == 5:
            init_state = np.concatenate([init_state, np.zeros(2, dtype=init_state.dtype)], axis=0)
        if goal_state.shape[0] == 5:
            goal_state = np.concatenate([goal_state, np.zeros(2, dtype=goal_state.dtype)], axis=0)
        if init_state.shape[0] < self.state_dim or goal_state.shape[0] < self.state_dim:
            raise ValueError(
                f"Dataset state dim ({init_state.shape[0]}) is smaller than env.state_dim ({self.state_dim})."
            )
        init_state = init_state[: self.state_dim]
        goal_state = goal_state[: self.state_dim]
        
        gt_frames = []
        for i in range(len(states)):
            self.prepare(seed=0, init_state=states[i])
            gt_frames.append(self.render("rgb_array"))
        meta = {
            "episode_index": int(ei),
            "start_index": int(start_idx),
            "goal_index": int(goal_idx),
            "trajectory_len": int(trajectory_len),
            "split": split_l,
            "actions":action_arr[start_idx:goal_idx],
        }
        return init_state, goal_state, meta
    
    def step(self, action):
        """
        Override step method to add Gaussian noise to state if enabled
        """
        """Execute one environment step with optional additive Gaussian noise.

        add_noise == 0  →  deterministic (original PushTEnv step)
        add_noise == 1  →  noise added to the *action* before stepping
        add_noise == 2  →  noise added to the *observed next state* afterwards

        When `noise_std` is a sequence its length must match the relevant
        dimension (action_dim or state_dim).  A scalar standard deviation is
        broadcast to all dimensions.
        """

        # ------------------------------------------------------------------
        def _vector_std(std_raw, dim_required):
            """Return per-dim σ vector as a np.ndarray of length *dim_required*."""
            if np.isscalar(std_raw):
                return float(std_raw)  # scalar ok → broadcast later
            std_arr = np.asarray(std_raw, dtype=float)
            if std_arr.shape[0] != dim_required:
                raise ValueError(
                    f"noise_std length {std_arr.shape[0]} does not match required "
                    f"dimension {dim_required}"
                )
            return std_arr
        # Planning-fidelity action noise (coarser level -> more noise).
        if self._planning_fidelity_enabled:
            std_max = float(self._planning_fidelity_cfg.get("action_noise_std_max", 0.0))
            if std_max > 0.0:
                level = self._planning_fidelity_level()
                std = (1.0 - level) * std_max
                if std > 0.0:
                    action = np.asarray(action, dtype=np.float32) + np.random.normal(
                        loc=0.0, scale=std, size=self.action_dim
                    )

        # ----------------------------- no noise ----------------------------
        if self.add_noise == 0 or self._planning_fidelity_enabled:
            obs, reward, done, info = super().step(action)
            obs["visual"] = self._apply_planning_fidelity_visual(obs.get("visual"))
            return obs, reward, done, info

        # ------------------ noise on ACTION --------------------------------
        elif self.add_noise == 1:
            scale = _vector_std(self._noise_std_raw, self.action_dim)
            noise = np.random.normal(loc=0.0, scale=scale, size=self.action_dim)
            action = action + noise
            obs, reward, done, info = super().step(action)
            obs["visual"] = self._apply_planning_fidelity_visual(obs.get("visual"))
            return obs, reward, done, info

        # ------------------ noise on STATE ---------------------------------
        elif self.add_noise == 2:
            obs, reward, done, info = super().step(action)
            scale = _vector_std(self._noise_std_raw, self.state_dim)
            noise = np.random.normal(loc=0.0, scale=scale, size=self.state_dim)
            info['state'] = info['state'] + noise
            obs["visual"] = self._apply_planning_fidelity_visual(obs.get("visual"))
            return obs, reward, done, info

        # Invalid add_noise option
        raise ValueError(f"Unknown add_noise mode {self.add_noise}; expected 0,1,2")

    def sample_random_init_goal_states(self, seed, random_goal=False):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state(return_goal=False):
            if self.with_velocity:
                if return_goal:
                    return np.array(
                        [
                            50,
                            50,
                            256,
                            256,
                            np.pi / 4,
                            0,
                            0,  # agent velocities default 0
                        ]
                    )
                else:
                    return np.array(
                        [
                            rs.randint(50, 450),
                            rs.randint(50, 450),
                            rs.randint(100, 400),
                            rs.randint(100, 400),
                            rs.randn() * 2 * np.pi - np.pi,
                            0,
                            0,  # agent velocities default 0
                        ]
                    )
            else:
                if return_goal:
                    return np.array(
                        [
                            50,
                            50,
                            256,
                            256,
                            np.pi / 4,
                        ]
                    )
                else:
                    return np.array(
                        [
                            rs.randint(50, 450),
                            rs.randint(50, 450),
                            rs.randint(100, 400),
                            rs.randint(100, 400),
                            rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        
        init_state = generate_state()
        goal_state = generate_state(return_goal=True)
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # if position difference is < 20, and angle difference < np.pi/9, then success
        eef_diff = np.linalg.norm(goal_state[:2] - cur_state[:2])
        pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9 #and eef_diff < 20
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'pos_diff': pos_diff,
            'angle_diff': angle_diff,
            'eef_diff': eef_diff,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state, goal_state=None):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset_to_state = init_state
        if goal_state is not None:
            goal_pose = goal_state[2:5]
        else:
            goal_pose = None
        self._setup(goal_pose=goal_pose)
        obs, state = self.reset()
        obs["visual"] = self._apply_planning_fidelity_visual(obs.get("visual"))
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states
