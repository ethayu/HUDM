from __future__ import annotations

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

        self._sims: List[PushTWarpEnv] = []
        for li, level in enumerate(self._levels):
            params = PushTWarpParams(
                xmin=float(warp_cfg.get("xmin", -0.25)),
                xmax=float(warp_cfg.get("xmax", 0.25)),
                ymin=float(warp_cfg.get("ymin", -0.25)),
                ymax=float(warp_cfg.get("ymax", 0.25)),
                spacing=float(level.spacing),
                min_particles=int(warp_cfg.get("min_particles", 1)),
                force_single_particle=bool(level.is_single_particle),
                rest_offsets=np.asarray(level.rest_offsets, dtype=np.float32).copy(),
                particle_radius=particle_radius,
                radius_scale=float(warp_cfg.get("radius_scale", 1.0)),
                radius_clip_spacing=bool(warp_cfg.get("radius_clip_spacing", False)),
                stem_w=float(warp_cfg.get("stem_w", GT_T_STEM_W)),
                stem_h=float(warp_cfg.get("stem_h", GT_T_STEM_H)),
                bar_w=float(warp_cfg.get("bar_w", GT_T_BAR_W)),
                bar_h=float(warp_cfg.get("bar_h", GT_T_BAR_H)),
                pusher_radius=float(warp_cfg.get("pusher_radius", GT_PUSHER_RADIUS)),
                sim_hz=int(warp_cfg.get("sim_hz", 100)),
                control_hz=int(warp_cfg.get("control_hz", 10)),
                pusher_k_p=float(warp_cfg.get("pusher_k_p", 100.0)),
                pusher_k_v=float(warp_cfg.get("pusher_k_v", 20.0)),
                substeps=int(warp_cfg.get("substeps", 16)),
                iters=int(warp_cfg.get("iters", 8)),
                mu=float(warp_cfg.get("mu", 0.6)),
                contact_alpha=float(warp_cfg.get("contact_alpha", 0.35)),
                ground_friction_accel=float(warp_cfg.get("ground_friction_accel", 2.0)),
                rest_speed_eps=float(warp_cfg.get("rest_speed_eps", 0.01)),
                lin_damp=float(warp_cfg.get("lin_damp", 0.995)),
                vel_damp=float(warp_cfg.get("vel_damp", 0.999)),
                alpha_rigid=float(warp_cfg.get("alpha_rigid", 1.0)),
            )
            sim = PushTWarpEnv(device=self._device, params=params, seed=self._seed + 17 * li)
            self._sims.append(sim)

        sim0 = self._sims[0]
        self.xmin, self.xmax = float(sim0.xmin), float(sim0.xmax)
        self.ymin, self.ymax = float(sim0.ymin), float(sim0.ymax)
        self._xrange = max(1e-8, self.xmax - self.xmin)
        self._yrange = max(1e-8, self.ymax - self.ymin)

        self._current_state = np.zeros((self.state_dim,), dtype=np.float32)
        self._goal_state = np.zeros((self.state_dim,), dtype=np.float32)
        self._start_state = np.zeros((self.state_dim,), dtype=np.float32)
        self._current_snapshot: Optional[dict[str, np.ndarray]] = None

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
        xy_world = np.asarray(sim.pusher_pos[:2], dtype=np.float32)
        if not bool(pixel):
            return xy_world.copy()
        return self._world_xy_to_pix(xy_world)

    def current_particle_positions(self, *, level_idx: Optional[int] = None, pixel: bool = True) -> np.ndarray:
        sim = self._sim_for_level(level_idx)
        pts_world = np.asarray(sim.get_particle_positions(), dtype=np.float32)[:, :2]
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
        x, y = float(xy[0]), float(xy[1])
        wx = self.xmin + (x / 512.0) * self._xrange
        wy = self.ymin + (y / 512.0) * self._yrange
        return np.array([wx, wy], dtype=np.float32)

    def _world_to_pix_xy(self, xy: Sequence[float]) -> np.ndarray:
        x, y = float(xy[0]), float(xy[1])
        px = (x - self.xmin) / self._xrange * 512.0
        py = (y - self.ymin) / self._yrange * 512.0
        return np.array([px, py], dtype=np.float32)

    def _world_vel_to_pix(self, vel: Sequence[float]) -> np.ndarray:
        vx, vy = float(vel[0]), float(vel[1])
        return np.array(
            [
                vx * (512.0 / self._xrange),
                vy * (512.0 / self._yrange),
            ],
            dtype=np.float32,
        )

    def _pix_vel_to_world(self, vel: Sequence[float]) -> np.ndarray:
        vx, vy = float(vel[0]), float(vel[1])
        return np.array(
            [
                vx * (self._xrange / 512.0),
                vy * (self._yrange / 512.0),
            ],
            dtype=np.float32,
        )

    def _pix_delta_to_world(self, dxy: Sequence[float]) -> np.ndarray:
        dx, dy = float(dxy[0]), float(dxy[1])
        return np.array(
            [
                dx * (self._xrange / 512.0),
                dy * (self._yrange / 512.0),
            ],
            dtype=np.float32,
        )

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

    def render(self, mode: str = "rgb_array", include_start_pose: bool = False) -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"PushTParticleBackend supports mode='rgb_array' only, got {mode}")

        sim = self._active_sim()
        img = np.full((self.render_size, self.render_size, 3), 255, dtype=np.uint8)
        pr_px = self._world_len_to_img_px(float(sim.pr))
        pusher_r_px = self._world_len_to_img_px(float(sim.pusher_r))

        # Goal overlay (light green) uses current fidelity particle layout.
        r0 = sim.rest_offsets()
        goal_w = self._goal_pose_world(self._goal_state)
        goal_pts = self._transform_points(r0, goal_w)
        self._draw_particles(img, goal_pts, color=(180, 235, 180), radius_px=pr_px, thickness=-1)

        # Optional start overlay (outlined blue-ish markers).
        if include_start_pose:
            start_w = self._gt_pose_to_internal_pose(
                np.array([*self._pix_to_world_xy(self._start_state[2:4]), float(self._start_state[4])], dtype=np.float32)
            )
            start_pts = self._transform_points(r0, start_w)
            self._draw_particles(img, start_pts, color=(220, 90, 40), radius_px=pr_px, thickness=1)

        obj_pts = sim.get_particle_positions().astype(np.float32)
        self._draw_particles(img, obj_pts, color=(112, 128, 144), radius_px=pr_px, thickness=-1)

        pusher_xy_w = np.asarray(sim.pusher_pos[:2], dtype=np.float32)
        p_u, p_v = self._world_to_img_xy(pusher_xy_w)
        cv2.circle(img, (p_u, p_v), pusher_r_px, (65, 105, 225), thickness=-1, lineType=cv2.LINE_AA)

        return img

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
        obs = {
            "visual": self.render("rgb_array", include_start_pose=False) if bool(with_visual) else None,
            "proprio": self._proprio_from_state(cur_state),
            "state": cur_state.copy(),
        }
        return obs, cur_state.copy()

    def step(self, action: np.ndarray, with_visual: bool = True):
        a = np.asarray(action, dtype=np.float32).reshape(2)
        sim = self._active_sim()

        if self.relative:
            delta_px = a * float(self.action_scale)
            a_world = self._pix_delta_to_world(delta_px)
        else:
            # Absolute target in PushT pixel-space.
            target_w = self._pix_to_world_xy(a * float(self.action_scale))
            cur_w = np.asarray(sim.pusher_pos[:2], dtype=np.float32)
            a_world = target_w - cur_w

        _, reward, done_sim, info_sim = sim.step(a_world)

        cur_state = self._sim_state(sim)
        self._current_state = cur_state.copy()
        self._current_snapshot = self._capture_sim_snapshot(sim)
        obs = {
            "visual": self.render("rgb_array", include_start_pose=False) if bool(with_visual) else None,
            "proprio": self._proprio_from_state(cur_state),
            "state": cur_state.copy(),
        }
        metrics = self.eval_state(self._goal_state, cur_state)
        # Keep planner metrics separate from env termination. The particle
        # backend should terminate only when the simulated env reports success,
        # which is coverage-based like the real PushT env.
        done = bool(done_sim)

        info = {
            "state": cur_state.copy(),
            "metrics": metrics,
            "num_particles": self.num_particles(),
            "spacing": self.spacing(),
            "sim_info": info_sim,
        }
        return obs, float(reward), done, info

    def eval_state(self, goal_state: np.ndarray, cur_state: np.ndarray) -> dict:
        goal_state = self._ensure_state_dim(goal_state)
        cur_state = self._ensure_state_dim(cur_state)
        return pose_metrics(goal_state, cur_state)
