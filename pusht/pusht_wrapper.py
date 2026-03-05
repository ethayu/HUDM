import numpy as np
import cv2
import torch
from pusht.pusht_env import PushTEnv, pymunk_to_shapely
from pusht.utils import aggregate_dct
from planning.fidelity import apply_fidelity

class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
            render_size=512,
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
        reconstruct_goal_state: int = 0,
        min_block_pos_delta: float = 10.0,
        min_block_angle_delta: float = np.pi / 9.0,
        max_resample_tries: int = 512,
        require_agent_in_frame: bool = True,
    ):
        if trajectory_len <= 0:
            raise ValueError(f"trajectory_len must be > 0, got {trajectory_len}")

        # Accept either a zarr path or a dataset-like object with state/starts/ends arrays.
        action_format = "unknown"
        action_abs_format = "unknown"
        action_scale_from_data = float(self.action_scale)
        action_relative_from_data = bool(self.relative)
        action_arr = None
        action_abs_arr = None
        if isinstance(dataset, str):
            try:
                import zarr
            except Exception as exc:
                raise ImportError("zarr not installed. pip install zarr") from exc
            root = zarr.open_group(dataset, mode="r")
            state_arr = root["data"]["state"]
            action_arr = root["data"].get("action", None)
            action_abs_arr = root["data"].get("action_abs", None)
            action_format = str(root.attrs.get("action_format", "unknown")).strip().lower()
            action_abs_format = str(root.attrs.get("action_abs_format", "unknown")).strip().lower()
            if "env_action_scale" in root.attrs:
                try:
                    action_scale_from_data = float(root.attrs["env_action_scale"])
                except Exception:
                    action_scale_from_data = float(self.action_scale)
            if "env_relative" in root.attrs:
                try:
                    action_relative_from_data = bool(root.attrs["env_relative"])
                except Exception:
                    action_relative_from_data = bool(self.relative)
            ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
            starts = np.zeros_like(ends)
            starts[0] = 0
            for i in range(1, len(ends)):
                starts[i] = ends[i - 1] + 1
        else:
            state_arr = dataset.state
            action_arr = getattr(dataset, "action", None)
            if action_arr is None:
                action_arr = getattr(dataset, "actions", None)
            action_abs_arr = getattr(dataset, "action_abs", None)
            if action_abs_arr is None:
                action_abs_arr = getattr(dataset, "actions_abs", None)
            action_format = str(getattr(dataset, "action_format", "unknown")).strip().lower()
            action_abs_format = str(getattr(dataset, "action_abs_format", "unknown")).strip().lower()
            if hasattr(dataset, "env_action_scale"):
                try:
                    action_scale_from_data = float(getattr(dataset, "env_action_scale"))
                except Exception:
                    action_scale_from_data = float(self.action_scale)
            if hasattr(dataset, "env_relative"):
                try:
                    action_relative_from_data = bool(getattr(dataset, "env_relative"))
                except Exception:
                    action_relative_from_data = bool(self.relative)
            ends = np.asarray(dataset.ends, dtype=np.int64)
            starts = np.asarray(dataset.starts, dtype=np.int64)

        def _normalize_action_format(fmt: str) -> str:
            fl = str(fmt).strip().lower()
            if fl in {"env_input", "relative_input", "relative"}:
                return "env_input"
            if fl in {"absolute_target", "absolute", "abs_target"}:
                return "absolute_target"
            return "unknown"

        action_format = _normalize_action_format(action_format)
        action_abs_format = _normalize_action_format(action_abs_format)
        if action_abs_arr is not None and action_abs_format == "unknown":
            # Backward-compatible default: data/action_abs denotes absolute targets.
            action_abs_format = "absolute_target"

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
        pos_thresh = float(min_block_pos_delta)
        ang_thresh = float(min_block_angle_delta)
        if pos_thresh < 0.0:
            raise ValueError(f"min_block_pos_delta must be >= 0, got {min_block_pos_delta}")
        if ang_thresh < 0.0:
            raise ValueError(f"min_block_angle_delta must be >= 0, got {min_block_angle_delta}")
        n_tries = max(1, int(max_resample_tries))
        reconstruct_mode = int(reconstruct_goal_state)
        if reconstruct_mode not in {0, 1, 2, 3}:
            raise ValueError(
                "reconstruct_goal_state must be one of {0,1,2,3}, "
                f"got {reconstruct_goal_state}"
            )
        frame_min = 0.0
        frame_max = float(getattr(self, "window_size", 512))
        require_agent_visible = bool(require_agent_in_frame)

        def _pad_and_trim_state(x: np.ndarray) -> np.ndarray:
            s = np.asarray(x, dtype=np.float32)
            if s.shape[0] == 5:
                s = np.concatenate([s, np.zeros(2, dtype=s.dtype)], axis=0)
            if s.shape[0] < self.state_dim:
                raise ValueError(
                    f"Dataset state dim ({s.shape[0]}) is smaller than env.state_dim ({self.state_dim})."
                )
            return s[: self.state_dim]

        def _agent_in_frame(state: np.ndarray) -> bool:
            s = np.asarray(state, dtype=np.float32)
            if s.shape[0] < 2:
                return False
            xy = s[:2]
            return bool(np.all(xy >= frame_min) and np.all(xy <= frame_max))

        def _convert_rollout_actions(
            action_seg_raw: np.ndarray,
            state_seg_pre: np.ndarray,
            mode: str,
        ) -> tuple[np.ndarray, str]:
            a_raw = np.asarray(action_seg_raw, dtype=np.float32)
            s_pre = np.asarray(state_seg_pre, dtype=np.float32)
            if a_raw.shape[0] != s_pre.shape[0]:
                raise ValueError(
                    "state/action segment length mismatch for reconstruction, "
                    f"state_len={int(s_pre.shape[0])}, action_len={int(a_raw.shape[0])}."
                )

            if mode == "env_input":
                return a_raw.copy(), "env_input"
            if mode == "absolute_target":
                denom = max(1e-8, float(action_scale_from_data))
                if bool(action_relative_from_data):
                    return (a_raw - s_pre[:, :2]) / denom, "absolute_target->env_input"
                return a_raw / denom, "absolute_target->env_input_nonrelative"
            raise ValueError(f"Unsupported action conversion mode: {mode}")

        def _rollout_goal_from_segment(
            init_state: np.ndarray,
            action_segment_raw: np.ndarray,
            state_segment_pre: np.ndarray,
            conversion_mode: str,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
            rollout_actions, conversion_label = _convert_rollout_actions(
                action_seg_raw=action_segment_raw,
                state_seg_pre=state_segment_pre,
                mode=conversion_mode,
            )
            _, states = self.rollout(seed=0, init_state=init_state.copy(), actions=rollout_actions)
            states_arr = np.asarray(states, dtype=np.float32)
            return _pad_and_trim_state(states_arr[-1]), rollout_actions, states_arr, conversion_label

        rollout_action_mode = "none"
        rollout_action_source = "none"
        rollout_action_arr = None
        if reconstruct_mode in {1, 2, 3}:
            if action_format in {"env_input", "absolute_target"} and action_arr is not None:
                rollout_action_mode = action_format
                rollout_action_source = "action"
                rollout_action_arr = action_arr
            elif action_abs_arr is not None:
                if action_abs_format != "absolute_target":
                    raise ValueError(
                        "data/action_abs exists but action_abs_format is unsupported; "
                        f"expected absolute_target, got {action_abs_format!r}."
                    )
                rollout_action_mode = "absolute_target"
                rollout_action_source = "action_abs"
                rollout_action_arr = action_abs_arr
            elif action_arr is not None:
                # Legacy datasets may not provide attrs. Infer once, using a small probe set.
                probe_count = min(8, len(candidates))
                if probe_count <= 0:
                    raise ValueError("Unable to infer action format: dataset has no valid probe candidates.")
                infer_rng = np.random.default_rng(seed=seed + 13579)
                probe_idx = infer_rng.integers(0, len(candidates), size=probe_count)
                fmt_errors = {"env_input": [], "absolute_target": []}
                for ci in np.asarray(probe_idx, dtype=np.int64):
                    _, ps, pe = candidates[int(ci)]
                    p_start = int(infer_rng.integers(ps, pe - trajectory_len + 1))
                    p_goal = p_start + trajectory_len
                    p_init = _pad_and_trim_state(state_arr[p_start])
                    p_goal_state = _pad_and_trim_state(state_arr[p_goal])
                    p_actions = np.asarray(action_arr[p_start:p_goal], dtype=np.float32)
                    p_states = np.asarray(state_arr[p_start:p_goal], dtype=np.float32)
                    if int(p_actions.shape[0]) != int(trajectory_len):
                        continue
                    for fmt in ("env_input", "absolute_target"):
                        try:
                            p_recon, _, _, _ = _rollout_goal_from_segment(
                                init_state=p_init,
                                action_segment_raw=p_actions,
                                state_segment_pre=p_states,
                                conversion_mode=fmt,
                            )
                            err = float(np.max(np.abs(p_recon - p_goal_state)))
                        except Exception:
                            err = float("inf")
                        fmt_errors[fmt].append(err)
                env_err = float(np.median(fmt_errors["env_input"])) if fmt_errors["env_input"] else float("inf")
                abs_err = (
                    float(np.median(fmt_errors["absolute_target"]))
                    if fmt_errors["absolute_target"]
                    else float("inf")
                )
                if (not np.isfinite(env_err)) and (not np.isfinite(abs_err)):
                    raise ValueError(
                        "Unable to infer action format for reconstruction from legacy dataset. "
                        "Provide zarr attrs (`action_format`) or an `action_abs` array."
                    )
                rollout_action_mode = "absolute_target" if abs_err < env_err else "env_input"
                rollout_action_source = "action"
                rollout_action_arr = action_arr
            else:
                raise ValueError(
                    "reconstruct_goal_state requires dataset actions, but neither `action` nor "
                    "`action_abs` arrays were found."
                )

        fallback = None
        fallback_score = -np.inf
        for attempt_idx in range(n_tries):
            ei, s, e = candidates[int(rng.integers(0, len(candidates)))]
            start_idx = int(rng.integers(s, e - trajectory_len + 1))
            goal_idx = start_idx + trajectory_len

            init_state = _pad_and_trim_state(state_arr[start_idx])
            goal_state_stored = _pad_and_trim_state(state_arr[goal_idx])
            goal_state = goal_state_stored.copy()
            action_seg = None
            conversion_label = None
            state_seq_rollout = None
            used_action_rollout = False
            if reconstruct_mode in {1, 2, 3}:
                action_seg_raw = np.asarray(rollout_action_arr[start_idx:goal_idx], dtype=np.float32)
                if int(action_seg_raw.shape[0]) != int(trajectory_len):
                    raise ValueError(
                        "reconstruct_goal_state requires action segment length equal to trajectory_len, "
                        f"got len={int(action_seg_raw.shape[0])}, trajectory_len={int(trajectory_len)}"
                    )
                state_seg_pre = np.asarray(state_arr[start_idx:goal_idx], dtype=np.float32)
                goal_state_recon, action_seg, state_seq_rollout, conversion_label = _rollout_goal_from_segment(
                    init_state=init_state,
                    action_segment_raw=action_seg_raw,
                    state_segment_pre=state_seg_pre,
                    conversion_mode=rollout_action_mode,
                )
                goal_state_recon = _pad_and_trim_state(state_seq_rollout[-1])
                max_abs_diff = float(np.max(np.abs(goal_state_recon - goal_state_stored)))
                if reconstruct_mode == 1:
                    if not np.allclose(goal_state_recon, goal_state_stored, atol=1e-4, rtol=1e-4):
                        raise ValueError(
                            "reconstruct_goal_state=1 mismatch: reconstructed goal state differs from stored "
                            f"(episode={int(ei)}, start={int(start_idx)}, goal={int(goal_idx)}, "
                            f"max_abs_diff={max_abs_diff:.6g}, action_mode={rollout_action_mode}, "
                            f"action_source={rollout_action_source})."
                        )
                    goal_state = goal_state_stored
                else:
                    goal_state = goal_state_recon
                    used_action_rollout = True

            pos_diff = float(np.linalg.norm(goal_state[2:4] - init_state[2:4]))
            ang_diff = float(np.abs(goal_state[4] - init_state[4]))
            ang_diff = float(np.minimum(ang_diff, 2.0 * np.pi - ang_diff))
            init_agent_in_frame = _agent_in_frame(init_state)
            goal_agent_in_frame = _agent_in_frame(goal_state)
            agent_visible = (not require_agent_visible) or (
                init_agent_in_frame and goal_agent_in_frame
            )

            # Avoid trivially solved tasks at reset: require meaningful block-pose change.
            if agent_visible and ((pos_diff >= pos_thresh) or (ang_diff >= ang_thresh)):
                meta = {
                    "episode_index": int(ei),
                    "start_index": int(start_idx),
                    "goal_index": int(goal_idx),
                    "trajectory_len": int(trajectory_len),
                    "split": split_l,
                    "pos_diff": pos_diff,
                    "angle_diff": ang_diff,
                    "resample_tries": int(attempt_idx + 1),
                    "used_action_rollout": bool(used_action_rollout),
                    "reconstruct_goal_state": reconstruct_mode,
                    "action_format": action_format,
                    "action_abs_format": action_abs_format,
                    "action_source": rollout_action_source,
                    "action_mode": rollout_action_mode,
                    "force_gt_action_replay": bool(reconstruct_mode == 3),
                    "init_agent_in_frame": bool(init_agent_in_frame),
                    "goal_agent_in_frame": bool(goal_agent_in_frame),
                    "require_agent_in_frame": bool(require_agent_visible),
                }
                if action_seg is not None and used_action_rollout:
                    meta["actions"] = np.asarray(action_seg, dtype=np.float32).tolist()
                    if conversion_label is not None:
                        meta["action_conversion"] = str(conversion_label)
                    if (reconstruct_mode == 3) and (state_seq_rollout is not None):
                        meta["gt_state_trajectory"] = np.asarray(state_seq_rollout, dtype=np.float32).tolist()
                        meta["gt_state_trajectory_source"] = "action_rollout"
                return init_state, goal_state, meta

            score = max(
                pos_diff / (pos_thresh + 1e-8),
                ang_diff / (ang_thresh + 1e-8),
            )
            if not agent_visible:
                score -= 1e6
            if score > fallback_score:
                fallback_score = score
                fallback = (
                    init_state,
                    goal_state,
                    ei,
                    start_idx,
                    goal_idx,
                    pos_diff,
                    ang_diff,
                    attempt_idx + 1,
                    used_action_rollout,
                    reconstruct_mode,
                    rollout_action_source,
                    rollout_action_mode,
                    None if (action_seg is None or (not used_action_rollout)) else np.asarray(action_seg, dtype=np.float32).tolist(),
                    None
                    if (state_seq_rollout is None or (not used_action_rollout) or reconstruct_mode != 3)
                    else np.asarray(state_seq_rollout, dtype=np.float32).tolist(),
                    bool(init_agent_in_frame),
                    bool(goal_agent_in_frame),
                )

        if fallback is not None:
            (
                init_state,
                goal_state,
                ei,
                start_idx,
                goal_idx,
                pos_diff,
                ang_diff,
                n_used,
                used_action_rollout,
                fallback_reconstruct_mode,
                fallback_action_source,
                fallback_action_mode,
                action_seg_list,
                state_seq_rollout_list,
                fallback_init_agent_in_frame,
                fallback_goal_agent_in_frame,
            ) = fallback
            meta = {
                "episode_index": int(ei),
                "start_index": int(start_idx),
                "goal_index": int(goal_idx),
                "trajectory_len": int(trajectory_len),
                "split": split_l,
                "pos_diff": float(pos_diff),
                "angle_diff": float(ang_diff),
                "resample_tries": int(n_used),
                "fallback": True,
                "used_action_rollout": bool(used_action_rollout),
                "reconstruct_goal_state": int(fallback_reconstruct_mode),
                "action_format": action_format,
                "action_abs_format": action_abs_format,
                "action_source": str(fallback_action_source),
                "action_mode": str(fallback_action_mode),
                "force_gt_action_replay": bool(int(fallback_reconstruct_mode) == 3),
                "init_agent_in_frame": bool(fallback_init_agent_in_frame),
                "goal_agent_in_frame": bool(fallback_goal_agent_in_frame),
                "require_agent_in_frame": bool(require_agent_visible),
            }
            if action_seg_list is not None:
                meta["actions"] = action_seg_list
            if state_seq_rollout_list is not None:
                meta["gt_state_trajectory"] = state_seq_rollout_list
                meta["gt_state_trajectory_source"] = "action_rollout"
            return init_state, goal_state, meta

        raise RuntimeError(
            "Unable to sample init/goal states from dataset; no candidate states were available."
        )
    
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

    def eval_termination(self, goal_state, cur_state, done=None, info=None):
        """
        Unified termination eval used by planners and debug logging.

        Returns metric success (wrapper thresholds), env-done status, coverage,
        and the strict success gate requiring both metric success and done.
        """
        metrics = self.eval_state(goal_state, cur_state)

        coverage = None
        if isinstance(info, dict) and ("final_coverage" in info):
            cov_raw = info.get("final_coverage", None)
            if cov_raw is not None:
                coverage = float(cov_raw)
        if coverage is None:
            try:
                goal_body = self._get_goal_pose_body(self.goal_pose)
                goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
                block_geom = pymunk_to_shapely(self.block, self.block.shapes)
                goal_area = float(goal_geom.area)
                if goal_area > 0.0:
                    coverage = float(goal_geom.intersection(block_geom).area / goal_area)
            except Exception:
                coverage = None

        if done is None:
            done_flag = (
                coverage is not None
                and coverage > float(getattr(self, "success_threshold", np.inf))
            )
        else:
            done_flag = bool(done)

        out = dict(metrics)
        out["done"] = bool(done_flag)
        out["coverage"] = coverage
        out["success_and_done"] = bool(metrics["success"]) and bool(done_flag)
        return out

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
