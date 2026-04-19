from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from hudm.metrics import pose_metrics
from pusht.pusht_particle_warp import (
    GT_PUSHER_RADIUS,
    GT_T_BAR_H,
    GT_T_BAR_W,
    GT_T_STEM_H,
    GT_T_STEM_W,
    PushTWarpBatchEnv,
    PushTWarpEnv,
    PushTWarpParams,
    build_t_particle_hierarchy,
    wp,
)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return float(value)


@dataclass
class ParticleDeviceBatchContext:
    level_idx: int
    batch_env: PushTWarpBatchEnv
    start_states: np.ndarray
    goal_states: np.ndarray
    current_states: np.ndarray


class PushTParticleBackend:
    """
    Adapter that exposes PushT-like state/observation semantics on top of Warp particle sims.

    State convention (PushT-compatible):
      [agent_x, agent_y, block_x, block_y, theta, agent_vx, agent_vy]
    where position/velocity units are in PushT pixel-space (0..512 domain).
    """

    def __init__(
        self,
        with_velocity: bool = True,
        with_target: bool = True,
        render_size: int = 512,
        relative: bool = True,
        action_scale: float = 100.0,
        device: str = "auto",
        particle_counts: Optional[Sequence[int]] = None,
        warp_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 0,
    ):
        if wp is None:
            raise ImportError(
                "warp-lang is not installed. Install with `pip install warp-lang` to use particle_sim backend."
            )

        self.with_velocity = bool(with_velocity)
        self.with_target = bool(with_target)
        self.render_size = int(render_size)
        self.relative = bool(relative)
        self.action_scale = float(action_scale)

        self.state_dim = 7
        self.action_dim = 2

        warp_cfg = dict(warp_cfg or {})
        self._warp_cfg = warp_cfg
        self._seed = int(seed)

        self._device = self._resolve_device(str(device))
        particle_radius = _optional_float(warp_cfg.get("particle_radius", None))

        if particle_counts is None:
            particle_counts = warp_cfg.get("particle_counts", [1, 4, 16, 64, 128, 256])
        particle_counts = [int(c) for c in list(particle_counts)]
        self._levels = build_t_particle_hierarchy(
            particle_counts=particle_counts,
            stem_w=float(warp_cfg.get("stem_w", GT_T_STEM_W)),
            stem_h=float(warp_cfg.get("stem_h", GT_T_STEM_H)),
            bar_w=float(warp_cfg.get("bar_w", GT_T_BAR_W)),
            bar_h=float(warp_cfg.get("bar_h", GT_T_BAR_H)),
            min_particles=int(warp_cfg.get("min_particles", 1)),
        )
        self.spacings = [float(level.spacing) for level in self._levels]
        self.target_particle_counts = [int(level.target_particle_count) for level in self._levels]
        self.resolved_particle_counts = [int(level.rest_offsets.shape[0]) for level in self._levels]
        self._planning_fidelity_enabled = True
        self._planning_fidelity_num_levels = len(self._levels)
        self._planning_fidelity_level_idx = self._planning_fidelity_num_levels - 1

        self._sim_params: List[PushTWarpParams] = []
        self._sims: List[PushTWarpEnv] = []
        for li, level in enumerate(self._levels):
            params = self._make_sim_params(level_idx=li, particle_radius=particle_radius)
            self._sim_params.append(params)
            sim = self._new_sim_for_level(li, clone_idx=0)
            self._sims.append(sim)
        self._batch_sim_pools: dict[int, list[PushTWarpEnv]] = {int(li): [] for li in range(len(self._levels))}
        self._device_batch_envs: dict[tuple[int, int], PushTWarpBatchEnv] = {}

        sim0 = self._sims[0]
        self.xmin, self.xmax = float(sim0.xmin), float(sim0.xmax)
        self.ymin, self.ymax = float(sim0.ymin), float(sim0.ymax)
        self._xrange = max(1e-8, self.xmax - self.xmin)
        self._yrange = max(1e-8, self.ymax - self.ymin)

        self._current_state = np.zeros((self.state_dim,), dtype=np.float32)
        self._goal_state = np.zeros((self.state_dim,), dtype=np.float32)
        self._start_state = np.zeros((self.state_dim,), dtype=np.float32)
        self._current_snapshot: Optional[dict[str, np.ndarray]] = None

    def _make_sim_params(self, level_idx: int, particle_radius: Optional[float]) -> PushTWarpParams:
        li = max(0, min(int(level_idx), len(self._levels) - 1))
        level = self._levels[li]
        return PushTWarpParams(
            xmin=float(self._warp_cfg.get("xmin", -0.25)),
            xmax=float(self._warp_cfg.get("xmax", 0.25)),
            ymin=float(self._warp_cfg.get("ymin", -0.25)),
            ymax=float(self._warp_cfg.get("ymax", 0.25)),
            spacing=float(level.spacing),
            min_particles=int(self._warp_cfg.get("min_particles", 1)),
            force_single_particle=bool(level.is_single_particle),
            rest_offsets=np.asarray(level.rest_offsets, dtype=np.float32).copy(),
            pose_offset_local=np.asarray(level.pose_offset_local, dtype=np.float32).copy(),
            particle_radius=particle_radius,
            radius_scale=float(self._warp_cfg.get("radius_scale", 1.0)),
            radius_clip_spacing=bool(self._warp_cfg.get("radius_clip_spacing", False)),
            stem_w=float(self._warp_cfg.get("stem_w", GT_T_STEM_W)),
            stem_h=float(self._warp_cfg.get("stem_h", GT_T_STEM_H)),
            bar_w=float(self._warp_cfg.get("bar_w", GT_T_BAR_W)),
            bar_h=float(self._warp_cfg.get("bar_h", GT_T_BAR_H)),
            pusher_radius=float(self._warp_cfg.get("pusher_radius", GT_PUSHER_RADIUS)),
            sim_hz=int(self._warp_cfg.get("sim_hz", 100)),
            control_hz=int(self._warp_cfg.get("control_hz", 10)),
            pusher_k_p=float(self._warp_cfg.get("pusher_k_p", 100.0)),
            pusher_k_v=float(self._warp_cfg.get("pusher_k_v", 20.0)),
            substeps=int(self._warp_cfg.get("substeps", 16)),
            iters=int(self._warp_cfg.get("iters", 8)),
            mu=float(self._warp_cfg.get("mu", 0.6)),
            contact_alpha=float(self._warp_cfg.get("contact_alpha", 0.35)),
            ground_friction_accel=float(self._warp_cfg.get("ground_friction_accel", 2.0)),
            rest_speed_eps=float(self._warp_cfg.get("rest_speed_eps", 0.01)),
            lin_damp=float(self._warp_cfg.get("lin_damp", 0.995)),
            vel_damp=float(self._warp_cfg.get("vel_damp", 0.999)),
            alpha_rigid=float(self._warp_cfg.get("alpha_rigid", 1.0)),
        )

    def _new_sim_for_level(self, level_idx: int, clone_idx: int) -> PushTWarpEnv:
        li = max(0, min(int(level_idx), len(self._sim_params) - 1))
        return PushTWarpEnv(
            device=self._device,
            params=self._sim_params[li],
            seed=int(self._seed + 17 * li + 1000003 * max(0, int(clone_idx))),
        )

    def _batch_sims_for_level(self, level_idx: int, count: int) -> list[PushTWarpEnv]:
        li = max(0, min(int(level_idx), len(self._sims) - 1))
        pool = self._batch_sim_pools.setdefault(li, [])
        while len(pool) < int(count):
            pool.append(self._new_sim_for_level(li, clone_idx=len(pool) + 1))
        return pool[: int(count)]

    def _device_batch_env_for_level(self, level_idx: int, batch_size: int) -> PushTWarpBatchEnv:
        li = max(0, min(int(level_idx), len(self._sim_params) - 1))
        key = (li, int(batch_size))
        env = self._device_batch_envs.get(key, None)
        if env is None:
            env = PushTWarpBatchEnv(
                device=self._device,
                params=self._sim_params[li],
                batch_size=int(batch_size),
                seed=int(self._seed + 17 * li),
            )
            self._device_batch_envs[key] = env
        return env

    def supports_cuda_native_batch(self) -> bool:
        return str(self._device).lower().startswith("cuda")

    def _sim_for_level(self, level_idx: Optional[int] = None) -> PushTWarpEnv:
        li = self._planning_fidelity_level_idx if level_idx is None else int(level_idx)
        li = max(0, min(li, len(self._sims) - 1))
        return self._sims[li]

    def _resolve_device(self, device: str) -> str:
        d = (device or "auto").strip().lower()
        if d in {"auto", ""}:
            try:
                return "cuda:0" if bool(wp.is_cuda_available()) else "cpu"
            except Exception:
                return "cpu"
        if d == "cuda":
            return "cuda:0"
        return d

    def _active_sim(self) -> PushTWarpEnv:
        return self._sims[int(self._planning_fidelity_level_idx)]

    @property
    def num_levels(self) -> int:
        return int(self._planning_fidelity_num_levels)

    def configure_planning_fidelity(
        self,
        enabled: bool,
        num_levels: int,
        cfg: Optional[dict] = None,
    ) -> None:
        del cfg
        self._planning_fidelity_enabled = bool(enabled)
        if int(num_levels) != len(self._sims):
            raise ValueError(
                f"particle fidelity level mismatch: requested {num_levels}, but hierarchy defines {len(self._sims)} levels"
            )
        self._planning_fidelity_num_levels = int(num_levels)
        self._planning_fidelity_level_idx = self._planning_fidelity_num_levels - 1

    def set_planning_fidelity_level(self, level_idx: int) -> None:
        li = int(level_idx)
        li = max(0, min(li, self._planning_fidelity_num_levels - 1))
        prev = int(self._planning_fidelity_level_idx)
        if li == prev:
            return

        source_snapshot = self._capture_sim_snapshot(self._sims[prev])
        self._planning_fidelity_level_idx = li

        # Keep the newly selected fidelity simulator synchronized to the
        # current physical state before switching resolutions.
        if source_snapshot is not None:
            self._set_sim_from_snapshot(self._active_sim(), source_snapshot)
            self._current_state = self._sim_state(self._active_sim())
            self._current_snapshot = self._capture_sim_snapshot(self._active_sim())
        elif self._current_state is not None:
            self._set_sim_from_state(self._active_sim(), self._current_state, self._goal_state)
            self._current_state = self._sim_state(self._active_sim())
            self._current_snapshot = self._capture_sim_snapshot(self._active_sim())

    def num_particles(self, level_idx: Optional[int] = None) -> int:
        li = self._planning_fidelity_level_idx if level_idx is None else int(level_idx)
        li = max(0, min(li, len(self._sims) - 1))
        return int(self._sims[li].num_particles)

    def spacing(self, level_idx: Optional[int] = None) -> float:
        li = self._planning_fidelity_level_idx if level_idx is None else int(level_idx)
        li = max(0, min(li, len(self._sims) - 1))
        return float(self.spacings[li])

    def particle_radius(self, level_idx: Optional[int] = None) -> float:
        li = self._planning_fidelity_level_idx if level_idx is None else int(level_idx)
        li = max(0, min(li, len(self._sims) - 1))
        return float(self._sims[li].pr)

    def _level(self, level_idx: Optional[int] = None):
        if not hasattr(self, "_levels") or len(self._levels) <= 0:
            return None
        li = self._planning_fidelity_level_idx if level_idx is None else int(level_idx)
        li = max(0, min(li, len(self._levels) - 1))
        return self._levels[li]

    def _level_offset_world(self, theta: float, level_idx: Optional[int] = None) -> np.ndarray:
        level = self._level(level_idx)
        if level is None:
            return np.zeros((2,), dtype=np.float32)
        offset = np.asarray(level.pose_offset_local[:2], dtype=np.float32)
        c = math.cos(float(theta))
        s = math.sin(float(theta))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (offset.reshape(1, 2) @ rot.T).reshape(2).astype(np.float32)

    def _gt_pose_to_internal_pose(self, pose_world: Sequence[float], level_idx: Optional[int] = None) -> np.ndarray:
        pose = np.asarray(pose_world, dtype=np.float32).reshape(3)
        internal = pose.copy()
        internal[:2] = pose[:2] + self._level_offset_world(float(pose[2]), level_idx=level_idx)
        return internal

    def _internal_pose_to_gt_pose(self, pose_world: Sequence[float], level_idx: Optional[int] = None) -> np.ndarray:
        pose = np.asarray(pose_world, dtype=np.float32).reshape(3)
        gt_pose = pose.copy()
        gt_pose[:2] = pose[:2] - self._level_offset_world(float(pose[2]), level_idx=level_idx)
        return gt_pose

    def _world_xy_to_pix(self, xy_world: Sequence[float] | np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_world, dtype=np.float32)
        out = np.empty(xy.shape, dtype=np.float32)
        out[..., 0] = (xy[..., 0] - self.xmin) / self._xrange * 512.0
        out[..., 1] = (xy[..., 1] - self.ymin) / self._yrange * 512.0
        return out.astype(np.float32)

    def current_pusher_position(self, *, level_idx: Optional[int] = None, pixel: bool = True) -> np.ndarray:
        sim = self._sim_for_level(level_idx)
        xy_world = self._pusher_position_from_sim(sim, pixel=False)
        if not bool(pixel):
            return xy_world.copy()
        return self._world_xy_to_pix(xy_world)

    def current_particle_positions(self, *, level_idx: Optional[int] = None, pixel: bool = True) -> np.ndarray:
        sim = self._sim_for_level(level_idx)
        pts_world = self._particle_positions_from_sim(sim, pixel=False)
        if not bool(pixel):
            return pts_world.copy()
        return self._world_xy_to_pix(pts_world)

    def current_particle_cloud_state(
        self,
        *,
        level_idx: Optional[int] = None,
        pixel: bool = True,
    ) -> dict[str, np.ndarray]:
        return {
            "pusher_xy": self.current_pusher_position(level_idx=level_idx, pixel=pixel),
            "particle_xy": self.current_particle_positions(level_idx=level_idx, pixel=pixel),
        }

    def _ensure_state_dim(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.shape[0] == 5:
            s = np.concatenate([s, np.zeros((2,), dtype=np.float32)], axis=0)
        if s.shape[0] < 7:
            pad = 7 - s.shape[0]
            s = np.concatenate([s, np.zeros((pad,), dtype=np.float32)], axis=0)
        return s[:7].astype(np.float32)

    def _pix_to_world_xy(self, xy: Sequence[float]) -> np.ndarray:
        arr = np.asarray(xy, dtype=np.float32)
        out = np.empty(arr.shape, dtype=np.float32)
        out[..., 0] = self.xmin + (arr[..., 0] / 512.0) * self._xrange
        out[..., 1] = self.ymin + (arr[..., 1] / 512.0) * self._yrange
        return out.astype(np.float32)

    def _world_to_pix_xy(self, xy: Sequence[float]) -> np.ndarray:
        x, y = float(xy[0]), float(xy[1])
        px = (x - self.xmin) / self._xrange * 512.0
        py = (y - self.ymin) / self._yrange * 512.0
        return np.array([px, py], dtype=np.float32)

    def _world_vel_to_pix(self, vel: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vel, dtype=np.float32)
        out = np.empty(arr.shape, dtype=np.float32)
        out[..., 0] = arr[..., 0] * (512.0 / self._xrange)
        out[..., 1] = arr[..., 1] * (512.0 / self._yrange)
        return out.astype(np.float32)

    def _pix_vel_to_world(self, vel: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vel, dtype=np.float32)
        out = np.empty(arr.shape, dtype=np.float32)
        out[..., 0] = arr[..., 0] * (self._xrange / 512.0)
        out[..., 1] = arr[..., 1] * (self._yrange / 512.0)
        return out.astype(np.float32)

    def _pix_delta_to_world(self, dxy: Sequence[float]) -> np.ndarray:
        arr = np.asarray(dxy, dtype=np.float32)
        out = np.empty(arr.shape, dtype=np.float32)
        out[..., 0] = arr[..., 0] * (self._xrange / 512.0)
        out[..., 1] = arr[..., 1] * (self._yrange / 512.0)
        return out.astype(np.float32)

    @staticmethod
    def _pusher_position_from_sim(sim: PushTWarpEnv, *, pixel: bool = False) -> np.ndarray:
        del pixel
        return np.asarray(sim.pusher_pos[:2], dtype=np.float32).copy()

    @staticmethod
    def _particle_positions_from_sim(sim: PushTWarpEnv, *, pixel: bool = False) -> np.ndarray:
        del pixel
        return np.asarray(sim.get_particle_positions(), dtype=np.float32)[:, :2].copy()

    def _sim_state(self, sim: PushTWarpEnv) -> np.ndarray:
        obj_pose_w = self._internal_pose_to_gt_pose(sim.get_object_pose().astype(np.float32))
        pusher_xy_w = np.asarray(sim.pusher_pos[:2], dtype=np.float32)

        agent_xy = self._world_to_pix_xy(pusher_xy_w)
        block_xy = self._world_to_pix_xy(obj_pose_w[:2])
        theta = float(_wrap_pi(float(obj_pose_w[2])))

        v_agent_pix = self._world_vel_to_pix(sim.get_pusher_velocity())

        out = np.zeros((7,), dtype=np.float32)
        out[0:2] = agent_xy
        out[2:4] = block_xy
        out[4] = theta
        out[5:7] = v_agent_pix
        return out

    def _goal_pose_world(self, goal_state: np.ndarray) -> np.ndarray:
        g = self._ensure_state_dim(goal_state)
        goal_w = self._pix_to_world_xy(g[2:4])
        goal_pose = np.array([goal_w[0], goal_w[1], g[4]], dtype=np.float32)
        return self._gt_pose_to_internal_pose(goal_pose)

    def _goal_pose_world_for_level(self, goal_state: np.ndarray, level_idx: int) -> np.ndarray:
        g = self._ensure_state_dim(goal_state)
        goal_w = self._pix_to_world_xy(g[2:4])
        goal_pose = np.array([goal_w[0], goal_w[1], g[4]], dtype=np.float32)
        return self._gt_pose_to_internal_pose(goal_pose, level_idx=level_idx)

    def _state_to_world_components_for_level(
        self,
        state: np.ndarray,
        goal_state: np.ndarray,
        level_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = self._ensure_state_dim(state)
        pusher_w = self._pix_to_world_xy(s[:2])
        block_pose_w = self._gt_pose_to_internal_pose(
            np.array([*self._pix_to_world_xy(s[2:4]), float(s[4])], dtype=np.float32),
            level_idx=level_idx,
        )
        pusher_vel_w = self._pix_vel_to_world(s[5:7])
        goal_pose_w = self._goal_pose_world_for_level(goal_state, level_idx=level_idx)
        return pusher_w, block_pose_w, pusher_vel_w, goal_pose_w

    def _state_batch_from_components(
        self,
        *,
        pusher_xy_world: np.ndarray,
        pusher_velocity_world: np.ndarray,
        obj_pose_world: np.ndarray,
        level_idx: int,
    ) -> np.ndarray:
        pusher_xy = np.asarray(pusher_xy_world, dtype=np.float32).reshape(-1, 2)
        pusher_velocity = np.asarray(pusher_velocity_world, dtype=np.float32).reshape(-1, 2)
        obj_pose = np.asarray(obj_pose_world, dtype=np.float32).reshape(-1, 3)
        gt_obj_pose = np.asarray(
            [self._internal_pose_to_gt_pose(pose, level_idx=level_idx) for pose in obj_pose],
            dtype=np.float32,
        )
        agent_xy = self._world_xy_to_pix(pusher_xy)
        block_xy = self._world_xy_to_pix(gt_obj_pose[:, :2])
        agent_vel = self._world_vel_to_pix(pusher_velocity)

        out = np.zeros((pusher_xy.shape[0], self.state_dim), dtype=np.float32)
        out[:, 0:2] = agent_xy
        out[:, 2:4] = block_xy
        out[:, 4] = np.asarray([_wrap_pi(float(v)) for v in gt_obj_pose[:, 2]], dtype=np.float32)
        out[:, 5:7] = agent_vel
        return out

    def _set_sim_from_state(self, sim: PushTWarpEnv, state: np.ndarray, goal_state: np.ndarray) -> dict:
        s = self._ensure_state_dim(state)

        pusher_w = self._pix_to_world_xy(s[:2])
        block_pose_w = self._gt_pose_to_internal_pose(
            np.array([*self._pix_to_world_xy(s[2:4]), float(s[4])], dtype=np.float32)
        )
        pusher_vel_w = self._pix_vel_to_world(s[5:7])

        sim.set_state(
            pusher_xy=pusher_w,
            obj_xy=block_pose_w[:2],
            obj_theta=float(block_pose_w[2]),
            pusher_velocity=pusher_vel_w,
            goal_pose=self._goal_pose_world(goal_state),
        )
        return sim._make_obs()

    def _capture_sim_snapshot(self, sim: PushTWarpEnv) -> Optional[dict[str, np.ndarray]]:
        if sim is None:
            return None
        if hasattr(sim, "capture_state"):
            snap = sim.capture_state()
            if snap is None:
                return None
            return {
                k: np.asarray(v, dtype=np.float32).copy()
                for k, v in dict(snap).items()
            }

        obj_pose = np.asarray(sim.get_object_pose(), dtype=np.float32).reshape(3)
        obj_twist = np.asarray(sim.get_object_twist(), dtype=np.float32).reshape(3)
        return {
            "pusher_xy": np.asarray(sim.pusher_pos[:2], dtype=np.float32).copy(),
            "pusher_velocity": np.asarray(sim.get_pusher_velocity(), dtype=np.float32).reshape(2),
            "obj_pose": obj_pose.copy(),
            "obj_twist": obj_twist.copy(),
            "goal_pose": np.asarray(getattr(sim, "goal_pose", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3),
        }

    def _set_sim_from_snapshot(self, sim: PushTWarpEnv, snapshot: dict[str, np.ndarray]) -> dict:
        snap = dict(snapshot or {})
        if len(snap) <= 0:
            return self._set_sim_from_state(sim, self._current_state, self._goal_state)
        if hasattr(sim, "restore_state"):
            return sim.restore_state(snap)

        return sim.set_state(
            pusher_xy=np.asarray(snap["pusher_xy"], dtype=np.float32).reshape(2),
            pusher_velocity=np.asarray(
                snap.get("pusher_velocity", np.zeros((2,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(2),
            obj_pose=np.asarray(snap["obj_pose"], dtype=np.float32).reshape(3),
            obj_twist=np.asarray(
                snap.get("obj_twist", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3),
            goal_pose=np.asarray(
                snap.get("goal_pose", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3),
        )

    def _state_matches_current(self, state: np.ndarray, atol: float = 1e-3) -> bool:
        if self._current_snapshot is None:
            return False
        s = self._ensure_state_dim(state)
        return bool(np.allclose(s, self._current_state, atol=float(atol), rtol=1e-4))

    def _proprio_from_state(self, state: np.ndarray) -> np.ndarray:
        if self.with_velocity:
            return np.concatenate([state[:2], state[5:7]], axis=0).astype(np.float32)
        return state[:2].astype(np.float32)

    def _world_to_img_xy(self, xy_world: np.ndarray) -> tuple[int, int]:
        pxy = self._world_to_pix_xy(xy_world)
        u = int(round((pxy[0] / 512.0) * (self.render_size - 1)))
        # Match PushT/Pygame image convention: origin at top-left, +y downward.
        v = int(round((pxy[1] / 512.0) * (self.render_size - 1)))
        return u, v

    def _world_len_to_img_px(self, l_world: float) -> int:
        sx = (self.render_size - 1) / max(1e-8, self._xrange)
        sy = (self.render_size - 1) / max(1e-8, self._yrange)
        s = 0.5 * (sx + sy)
        return max(1, int(round(float(l_world) * s)))

    def _transform_points(self, r0: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
        theta = float(pose_world[2])
        c = math.cos(theta)
        s = math.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = (r0[:, :2] @ R.T) + pose_world[:2][None, :]
        return pts.astype(np.float32)

    def _draw_particles(
        self,
        img: np.ndarray,
        pts_world: np.ndarray,
        color: tuple[int, int, int],
        radius_px: int,
        thickness: int = -1,
    ) -> None:
        for p in pts_world:
            u, v = self._world_to_img_xy(p)
            cv2.circle(img, (u, v), radius_px, color, thickness=thickness, lineType=cv2.LINE_AA)

    def _render_frame(
        self,
        *,
        rest_offsets: np.ndarray,
        particle_positions: np.ndarray,
        pusher_xy_world: np.ndarray,
        goal_pose_world: np.ndarray,
        particle_radius_world: float,
        pusher_radius_world: float,
        start_state: Optional[np.ndarray] = None,
        include_start_pose: bool = False,
        level_idx: Optional[int] = None,
    ) -> np.ndarray:
        img = np.full((self.render_size, self.render_size, 3), 255, dtype=np.uint8)
        pr_px = self._world_len_to_img_px(float(particle_radius_world))
        pusher_r_px = self._world_len_to_img_px(float(pusher_radius_world))

        goal_particles_w = self._transform_points(np.asarray(rest_offsets, dtype=np.float32), goal_pose_world)
        self._draw_particles(img, goal_particles_w, color=(180, 235, 180), radius_px=pr_px, thickness=-1)

        if include_start_pose and start_state is not None:
            start_arr = self._ensure_state_dim(np.asarray(start_state, dtype=np.float32))
            start_w = self._gt_pose_to_internal_pose(
                np.array([*self._pix_to_world_xy(start_arr[2:4]), float(start_arr[4])], dtype=np.float32),
                level_idx=level_idx,
            )
            start_particles_w = self._transform_points(np.asarray(rest_offsets, dtype=np.float32), start_w)
            self._draw_particles(img, start_particles_w, color=(220, 90, 40), radius_px=pr_px, thickness=-1)

        self._draw_particles(
            img,
            np.asarray(particle_positions, dtype=np.float32),
            color=(112, 128, 144),
            radius_px=pr_px,
            thickness=-1,
        )

        p_u, p_v = self._world_to_img_xy(np.asarray(pusher_xy_world, dtype=np.float32).reshape(2))
        cv2.circle(img, (p_u, p_v), pusher_r_px, (65, 105, 225), thickness=-1, lineType=cv2.LINE_AA)
        return img

    def _render_sim(
        self,
        sim: PushTWarpEnv,
        *,
        start_state: Optional[np.ndarray] = None,
        include_start_pose: bool = False,
    ) -> np.ndarray:
        return self._render_frame(
            rest_offsets=sim.rest_offsets(),
            particle_positions=sim.get_particle_positions().astype(np.float32),
            pusher_xy_world=np.asarray(sim.pusher_pos[:2], dtype=np.float32),
            goal_pose_world=np.asarray(sim.goal_pose, dtype=np.float32).reshape(3),
            particle_radius_world=float(sim.pr),
            pusher_radius_world=float(sim.pusher_r),
            start_state=start_state,
            include_start_pose=include_start_pose,
            level_idx=int(getattr(self, "_planning_fidelity_level_idx", 0)),
        )

    def _obs_from_sim(
        self,
        sim: PushTWarpEnv,
        *,
        cur_state: np.ndarray,
        with_visual: bool,
        start_state: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray | None]:
        return {
            "visual": self._render_sim(sim, start_state=start_state, include_start_pose=False) if bool(with_visual) else None,
            "proprio": self._proprio_from_state(cur_state),
            "state": cur_state.copy(),
        }

    def render(self, mode: str = "rgb_array", include_start_pose: bool = False) -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"PushTParticleBackend supports mode='rgb_array' only, got {mode}")
        return self._render_sim(self._active_sim(), start_state=self._start_state, include_start_pose=include_start_pose)

    def _hydrate_batch_sim(
        self,
        sim: PushTWarpEnv,
        *,
        level_idx: int,
        snapshot: Optional[dict[str, np.ndarray]] = None,
        state: Optional[np.ndarray] = None,
        goal_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if state is not None:
            if goal_state is None:
                raise ValueError("goal_state is required when hydrating a batch sim from state.")
            self._set_sim_from_state(sim, state, goal_state)
        elif snapshot is not None:
            snapshot_payload = {
                k: np.asarray(v, dtype=np.float32).copy()
                for k, v in dict(snapshot).items()
            }
            if goal_state is not None:
                snapshot_payload["goal_pose"] = self._goal_pose_world(goal_state)
            self._set_sim_from_snapshot(sim, snapshot_payload)
        else:
            raise ValueError("Either state or snapshot must be provided to hydrate a batch sim.")
        return self._sim_state(sim)

    def prepare_batch(
        self,
        *,
        level_idx: int,
        seeds: Optional[Sequence[int]],
        init_states: np.ndarray,
        goal_states: np.ndarray,
        with_visual: bool = True,
    ) -> tuple[dict[str, np.ndarray | None], np.ndarray, list[dict[str, np.ndarray]]]:
        del seeds
        init_arr = np.asarray(init_states, dtype=np.float32)
        goal_arr = np.asarray(goal_states, dtype=np.float32)
        if init_arr.ndim != 2 or goal_arr.ndim != 2:
            raise ValueError(
                "prepare_batch expects init_states and goal_states with shape (B, state_dim)."
            )
        if init_arr.shape[0] != goal_arr.shape[0]:
            raise ValueError(
                f"prepare_batch batch mismatch: init_states={init_arr.shape}, goal_states={goal_arr.shape}"
            )
        sims = self._batch_sims_for_level(level_idx, init_arr.shape[0])
        states_batch = np.zeros((init_arr.shape[0], self.state_dim), dtype=np.float32)
        proprio_batch = np.zeros((init_arr.shape[0], 4 if self.with_velocity else 2), dtype=np.float32)
        visuals: list[np.ndarray] = []
        snapshots: list[dict[str, np.ndarray]] = []
        for idx, sim in enumerate(sims):
            cur_state = self._hydrate_batch_sim(
                sim,
                level_idx=level_idx,
                state=init_arr[idx],
                goal_state=goal_arr[idx],
            )
            states_batch[idx] = cur_state.copy()
            proprio_batch[idx] = self._proprio_from_state(cur_state)
            if bool(with_visual):
                visuals.append(self._render_sim(sim))
            snapshot_payload = self._capture_sim_snapshot(sim) or {}
            snapshot_payload["start_state"] = self._ensure_state_dim(init_arr[idx]).copy()
            snapshots.append(snapshot_payload)
        obs = {
            "visual": np.stack(visuals, axis=0) if bool(with_visual) else None,
            "proprio": proprio_batch,
            "state": states_batch.copy(),
        }
        return obs, states_batch.copy(), snapshots

    def step_batch(
        self,
        *,
        level_idx: int,
        snapshots: Sequence[dict[str, np.ndarray]],
        goal_states: np.ndarray,
        actions: np.ndarray,
        with_visual: bool = True,
    ) -> tuple[dict[str, np.ndarray | None], np.ndarray, np.ndarray, list[dict], list[dict[str, np.ndarray]]]:
        goal_arr = np.asarray(goal_states, dtype=np.float32)
        action_arr = np.asarray(actions, dtype=np.float32)
        if goal_arr.ndim != 2 or action_arr.ndim != 2:
            raise ValueError("step_batch expects goal_states and actions with shape (B, ...).")
        batch_size = int(action_arr.shape[0])
        if len(snapshots) != batch_size or goal_arr.shape[0] != batch_size:
            raise ValueError(
                f"step_batch batch mismatch: snapshots={len(snapshots)}, goal_states={goal_arr.shape}, actions={action_arr.shape}"
            )
        sims = self._batch_sims_for_level(level_idx, batch_size)
        states_batch = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        proprio_batch = np.zeros((batch_size, 4 if self.with_velocity else 2), dtype=np.float32)
        reward_batch = np.zeros((batch_size,), dtype=np.float32)
        done_batch = np.zeros((batch_size,), dtype=bool)
        visuals: list[np.ndarray] = []
        info_batch: list[dict] = []
        next_snapshots: list[dict[str, np.ndarray]] = []
        for idx, sim in enumerate(sims):
            prior_snapshot = {
                k: np.asarray(v, dtype=np.float32).copy()
                for k, v in dict(snapshots[idx]).items()
            }
            self._hydrate_batch_sim(
                sim,
                level_idx=level_idx,
                snapshot=prior_snapshot,
                goal_state=goal_arr[idx],
            )
            obs_i, reward_i, done_i, info_i = self._step_sim(
                sim,
                action=action_arr[idx],
                with_visual=bool(with_visual),
                goal_state=goal_arr[idx],
            )
            cur_state = np.asarray(info_i["state"], dtype=np.float32).copy()
            states_batch[idx] = cur_state
            proprio_batch[idx] = self._proprio_from_state(cur_state)
            reward_batch[idx] = float(reward_i)
            done_batch[idx] = bool(done_i)
            if bool(with_visual):
                visual = obs_i.get("visual", None)
                if visual is None:
                    raise ValueError("step_batch expected visual observations when with_visual=True.")
                visuals.append(np.asarray(visual, dtype=np.uint8).copy())
            info_batch.append(dict(info_i))
            next_snapshot = self._capture_sim_snapshot(sim) or {}
            if "start_state" in prior_snapshot:
                next_snapshot["start_state"] = prior_snapshot["start_state"].copy()
            next_snapshots.append(next_snapshot)
        obs = {
            "visual": np.stack(visuals, axis=0) if bool(with_visual) else None,
            "proprio": proprio_batch,
            "state": states_batch.copy(),
        }
        return obs, reward_batch, done_batch, info_batch, next_snapshots

    def current_particle_cloud_state_batch(
        self,
        *,
        level_idx: int,
        snapshots: Sequence[dict[str, np.ndarray]],
        pixel: bool = True,
    ) -> dict[str, np.ndarray]:
        sims = self._batch_sims_for_level(level_idx, len(snapshots))
        pusher_xy = []
        particle_xy = []
        for idx, sim in enumerate(sims):
            self._hydrate_batch_sim(sim, level_idx=level_idx, snapshot=snapshots[idx])
            pusher_world = self._pusher_position_from_sim(sim, pixel=False)
            particle_world = self._particle_positions_from_sim(sim, pixel=False)
            pusher_xy.append(self._world_xy_to_pix(pusher_world) if bool(pixel) else pusher_world)
            particle_xy.append(self._world_xy_to_pix(particle_world) if bool(pixel) else particle_world)
        return {
            "pusher_xy": np.stack(pusher_xy, axis=0).astype(np.float32),
            "particle_xy": np.stack(particle_xy, axis=0).astype(np.float32),
        }

    def render_batch(
        self,
        *,
        level_idx: int,
        snapshots: Sequence[dict[str, np.ndarray]],
        start_states: Optional[np.ndarray] = None,
        include_start_pose: bool = False,
    ) -> np.ndarray:
        start_arr = None if start_states is None else np.asarray(start_states, dtype=np.float32)
        if start_arr is not None and (start_arr.ndim != 2 or start_arr.shape[0] != len(snapshots)):
            raise ValueError(
                "render_batch start_states must have shape (B, state_dim) matching snapshots."
            )
        sims = self._batch_sims_for_level(level_idx, len(snapshots))
        frames: list[np.ndarray] = []
        for idx, sim in enumerate(sims):
            self._hydrate_batch_sim(sim, level_idx=level_idx, snapshot=snapshots[idx])
            start_state = None if start_arr is None else start_arr[idx]
            if start_state is None and include_start_pose:
                start_state = dict(snapshots[idx]).get("start_state", None)
            frames.append(self._render_sim(sim, start_state=start_state, include_start_pose=include_start_pose))
        return np.stack(frames, axis=0).astype(np.uint8)

    def prepare_batch_device(
        self,
        *,
        level_idx: int,
        init_states: np.ndarray,
        goal_states: np.ndarray,
        with_visual: bool = True,
    ) -> tuple[dict[str, np.ndarray | None], np.ndarray, ParticleDeviceBatchContext]:
        if not self.supports_cuda_native_batch():
            raise RuntimeError("CUDA-native batch prepare requires a CUDA particle backend device.")

        init_arr = np.asarray(init_states, dtype=np.float32)
        goal_arr = np.asarray(goal_states, dtype=np.float32)
        if init_arr.ndim != 2 or goal_arr.ndim != 2:
            raise ValueError("prepare_batch_device expects init_states and goal_states with shape (B, state_dim).")
        if init_arr.shape[0] != goal_arr.shape[0]:
            raise ValueError(
                f"prepare_batch_device batch mismatch: init_states={init_arr.shape}, goal_states={goal_arr.shape}"
            )

        batch_env = self._device_batch_env_for_level(level_idx, init_arr.shape[0])
        pusher_xy = np.zeros((init_arr.shape[0], 2), dtype=np.float32)
        pusher_velocity = np.zeros((init_arr.shape[0], 2), dtype=np.float32)
        obj_pose = np.zeros((init_arr.shape[0], 3), dtype=np.float32)
        goal_pose = np.zeros((init_arr.shape[0], 3), dtype=np.float32)
        for idx in range(init_arr.shape[0]):
            pusher_xy[idx], obj_pose[idx], pusher_velocity[idx], goal_pose[idx] = self._state_to_world_components_for_level(
                init_arr[idx],
                goal_arr[idx],
                level_idx=int(level_idx),
            )
        batch_env.set_state_batch(
            pusher_xy=pusher_xy,
            obj_pose=obj_pose,
            goal_pose=goal_pose,
            pusher_velocity=pusher_velocity,
            active_mask=np.ones((init_arr.shape[0],), dtype=np.int32),
        )
        states_batch = self._state_batch_from_components(
            pusher_xy_world=batch_env.pusher_pos[:, :2],
            pusher_velocity_world=batch_env.get_pusher_velocity_batch(),
            obj_pose_world=batch_env.get_object_pose_batch(),
            level_idx=int(level_idx),
        )
        ctx = ParticleDeviceBatchContext(
            level_idx=int(level_idx),
            batch_env=batch_env,
            start_states=init_arr.copy(),
            goal_states=goal_arr.copy(),
            current_states=states_batch.copy(),
        )
        obs = {
            "visual": self.render_batch_device(context=ctx, include_start_pose=False) if bool(with_visual) else None,
            "proprio": np.asarray([self._proprio_from_state(state) for state in states_batch], dtype=np.float32),
            "state": states_batch.copy(),
        }
        return obs, states_batch.copy(), ctx

    def _relevel_batch_device_context(
        self,
        context: ParticleDeviceBatchContext,
        level_idx: int,
    ) -> ParticleDeviceBatchContext:
        li = int(level_idx)
        if li == int(context.level_idx):
            return context
        source_env = context.batch_env
        target_env = self._device_batch_env_for_level(li, context.goal_states.shape[0])
        obj_pose = source_env.get_object_pose_batch()
        obj_twist = source_env.get_object_twist_batch()
        pusher_xy = source_env.pusher_pos[:, :2].copy()
        pusher_velocity = source_env.get_pusher_velocity_batch()
        goal_pose = np.asarray(
            [self._goal_pose_world_for_level(goal_state, level_idx=li) for goal_state in context.goal_states],
            dtype=np.float32,
        )
        target_env.set_state_batch(
            pusher_xy=pusher_xy,
            obj_pose=obj_pose,
            goal_pose=goal_pose,
            obj_twist=obj_twist,
            pusher_velocity=pusher_velocity,
            active_mask=source_env._active_host.copy(),
        )
        context.level_idx = li
        context.batch_env = target_env
        context.current_states = self._state_batch_from_components(
            pusher_xy_world=target_env.pusher_pos[:, :2],
            pusher_velocity_world=target_env.get_pusher_velocity_batch(),
            obj_pose_world=target_env.get_object_pose_batch(),
            level_idx=li,
        )
        return context

    def step_batch_device(
        self,
        *,
        context: ParticleDeviceBatchContext,
        actions: np.ndarray,
        active_mask: np.ndarray,
        with_visual: bool = True,
    ) -> tuple[dict[str, np.ndarray | None], np.ndarray, np.ndarray, list[dict]]:
        active_arr = np.asarray(active_mask, dtype=bool).reshape(context.goal_states.shape[0])
        batch_env = context.batch_env
        action_arr = np.asarray(actions, dtype=np.float32).reshape(context.goal_states.shape[0], 2)
        if self.relative:
            action_world = self._pix_delta_to_world(action_arr * float(self.action_scale))
        else:
            target_world = self._pix_to_world_xy(action_arr * float(self.action_scale))
            action_world = target_world - np.asarray(batch_env.pusher_pos[:, :2], dtype=np.float32)

        obj_pose_batch, reward_batch, done_batch = batch_env.step_batch(action_world, active_mask=active_arr)
        states_batch = self._state_batch_from_components(
            pusher_xy_world=batch_env.pusher_pos[:, :2],
            pusher_velocity_world=batch_env.get_pusher_velocity_batch(),
            obj_pose_world=obj_pose_batch,
            level_idx=int(context.level_idx),
        )
        context.current_states = states_batch.copy()

        obs = {
            "visual": self.render_batch_device(context=context, include_start_pose=False) if bool(with_visual) else None,
            "proprio": np.asarray([self._proprio_from_state(state) for state in states_batch], dtype=np.float32),
            "state": states_batch.copy(),
        }
        info_batch: list[dict] = []
        for idx in range(states_batch.shape[0]):
            info_batch.append(
                {
                    "state": states_batch[idx].copy(),
                    "metrics": self.eval_state(context.goal_states[idx], states_batch[idx]),
                    "num_particles": int(batch_env.num_particles),
                    "spacing": float(batch_env.spacing),
                    "batch_impl": "cuda_native",
                }
            )
        return obs, reward_batch.astype(np.float32), done_batch.astype(bool), info_batch

    def current_particle_cloud_state_batch_device(
        self,
        *,
        context: ParticleDeviceBatchContext,
        pixel: bool = True,
        lane_indices: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        batch_env = context.batch_env
        particle_world = batch_env.get_particle_positions_batch()
        pusher_world = np.asarray(batch_env.pusher_pos[:, :2], dtype=np.float32)
        if lane_indices is not None:
            idx = np.asarray(lane_indices, dtype=np.int64).reshape(-1)
            particle_world = particle_world[idx]
            pusher_world = pusher_world[idx]
        if not bool(pixel):
            return {
                "pusher_xy": pusher_world.astype(np.float32),
                "particle_xy": particle_world.astype(np.float32),
            }
        return {
            "pusher_xy": self._world_xy_to_pix(pusher_world).astype(np.float32),
            "particle_xy": self._world_xy_to_pix(particle_world).astype(np.float32),
        }

    def render_batch_device(
        self,
        *,
        context: ParticleDeviceBatchContext,
        start_states: Optional[np.ndarray] = None,
        include_start_pose: bool = False,
        lane_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        batch_env = context.batch_env
        particle_world = batch_env.get_particle_positions_batch()
        pusher_world = np.asarray(batch_env.pusher_pos[:, :2], dtype=np.float32)
        goal_pose = np.asarray(batch_env.goal_pose, dtype=np.float32)
        start_arr = context.start_states if start_states is None else np.asarray(start_states, dtype=np.float32)
        if lane_indices is not None:
            idx = np.asarray(lane_indices, dtype=np.int64).reshape(-1)
            particle_world = particle_world[idx]
            pusher_world = pusher_world[idx]
            goal_pose = goal_pose[idx]
            start_arr = None if start_arr is None else start_arr[idx]

        frames: list[np.ndarray] = []
        rest_offsets = batch_env.rest_offsets()
        for idx in range(particle_world.shape[0]):
            start_state = None if start_arr is None else start_arr[idx]
            frames.append(
                self._render_frame(
                    rest_offsets=rest_offsets,
                    particle_positions=particle_world[idx],
                    pusher_xy_world=pusher_world[idx],
                    goal_pose_world=goal_pose[idx],
                    particle_radius_world=float(batch_env.pr),
                    pusher_radius_world=float(batch_env.pusher_r),
                    start_state=start_state,
                    include_start_pose=include_start_pose,
                    level_idx=int(context.level_idx),
                )
            )
        return np.stack(frames, axis=0).astype(np.uint8)

    def _goal_state_for_sim(self, sim: PushTWarpEnv, goal_state: Optional[np.ndarray] = None) -> np.ndarray:
        if goal_state is not None:
            return self._ensure_state_dim(goal_state)
        goal_pose_gt = self._internal_pose_to_gt_pose(np.asarray(sim.goal_pose, dtype=np.float32).reshape(3))
        goal = np.zeros((7,), dtype=np.float32)
        goal[2:4] = self._world_to_pix_xy(goal_pose_gt[:2])
        goal[4] = float(goal_pose_gt[2])
        return goal

    def _step_sim(
        self,
        sim: PushTWarpEnv,
        *,
        action: np.ndarray,
        with_visual: bool,
        goal_state: Optional[np.ndarray] = None,
    ) -> tuple[dict, float, bool, dict]:
        a = np.asarray(action, dtype=np.float32).reshape(2)
        if self.relative:
            delta_px = a * float(self.action_scale)
            a_world = self._pix_delta_to_world(delta_px)
        else:
            target_w = self._pix_to_world_xy(a * float(self.action_scale))
            cur_w = np.asarray(sim.pusher_pos[:2], dtype=np.float32)
            a_world = target_w - cur_w

        _, reward, done_sim, info_sim = sim.step(a_world)
        cur_state = self._sim_state(sim)
        obs = self._obs_from_sim(sim, cur_state=cur_state, with_visual=with_visual)
        metrics = self.eval_state(self._goal_state_for_sim(sim, goal_state=goal_state), cur_state)
        info = {
            "state": cur_state.copy(),
            "metrics": metrics,
            "num_particles": int(sim.num_particles),
            "spacing": float(sim.spacing),
            "sim_info": info_sim,
        }
        return obs, float(reward), bool(done_sim), info

    def prepare(
        self,
        seed: int,
        init_state: np.ndarray,
        goal_state: Optional[np.ndarray] = None,
        with_visual: bool = True,
    ) -> tuple[dict, np.ndarray]:
        del seed
        init_state = self._ensure_state_dim(init_state)
        if goal_state is None:
            goal_state = self._goal_state.copy()
        goal_state = self._ensure_state_dim(goal_state)

        self._start_state = init_state.copy()
        self._goal_state = goal_state.copy()

        sim = self._active_sim()
        if self._state_matches_current(init_state):
            sim.set_goal_pose(self._goal_pose_world(goal_state))
        else:
            obs_sim = self._set_sim_from_state(sim, init_state, goal_state)
            del obs_sim

        cur_state = self._sim_state(sim)
        self._current_state = cur_state.copy()
        self._current_snapshot = self._capture_sim_snapshot(sim)
        obs = self._obs_from_sim(sim, cur_state=cur_state, with_visual=with_visual)
        return obs, cur_state.copy()

    def step(self, action: np.ndarray, with_visual: bool = True):
        sim = self._active_sim()
        obs, reward, done, info = self._step_sim(
            sim,
            action=np.asarray(action, dtype=np.float32),
            with_visual=bool(with_visual),
            goal_state=self._goal_state,
        )
        self._current_state = np.asarray(info["state"], dtype=np.float32).copy()
        self._current_snapshot = self._capture_sim_snapshot(sim)
        return obs, float(reward), bool(done), info

    def eval_state(self, goal_state: np.ndarray, cur_state: np.ndarray) -> dict:
        goal_state = self._ensure_state_dim(goal_state)
        cur_state = self._ensure_state_dim(cur_state)
        return pose_metrics(goal_state, cur_state)
