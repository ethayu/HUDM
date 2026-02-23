from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - optional dependency path
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


def _points_in_t_grid(
    stem_w: float,
    stem_h: float,
    bar_w: float,
    bar_h: float,
    spacing: float,
) -> np.ndarray:
    xs = np.arange(-bar_w * 0.5, bar_w * 0.5 + 1e-9, spacing, dtype=np.float32)
    ys = np.arange(-(stem_h + bar_h) * 0.5, (stem_h + bar_h) * 0.5 + 1e-9, spacing, dtype=np.float32)

    pts = []
    for y in ys:
        for x in xs:
            in_stem = (abs(x) <= stem_w * 0.5) and (y <= bar_h * 0.5) and (y >= -stem_h + bar_h * 0.5)
            in_bar = (abs(x) <= bar_w * 0.5) and (abs(y - bar_h * 0.5) <= bar_h * 0.5)
            if in_stem or in_bar:
                pts.append((x, y, 0.0))
    arr = np.array(pts, dtype=np.float32)

    com = arr[:, :2].mean(axis=0)
    arr[:, 0] -= com[0]
    arr[:, 1] -= com[1]
    return arr


def _rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PushTWarpParams:
    xmin: float = -0.25
    xmax: float = 0.25
    ymin: float = -0.25
    ymax: float = 0.25
    particle_radius: float = 0.006
    spacing: float = 0.012
    stem_w: float = 0.05
    stem_h: float = 0.10
    bar_w: float = 0.12
    bar_h: float = 0.04
    pusher_radius: float = 0.015
    pusher_speed: float = 0.6
    frame_dt: float = 1.0 / 10.0
    substeps: int = 16
    iters: int = 8
    mu: float = 0.6
    lin_damp: float = 0.995
    vel_damp: float = 0.999
    alpha_rigid: float = 1.0


if wp is not None:

    @wp.kernel
    def _zero_scalar(out: wp.array(dtype=float)):
        out[0] = 0.0


    @wp.kernel
    def _predict_positions(
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        x_pred: wp.array(dtype=wp.vec3),
        dt: float,
        lin_damp: float,
    ):
        i = wp.tid()
        if inv_m[i] == 0.0:
            x_pred[i] = wp.vec3(x[i][0], x[i][1], 0.0)
            v[i] = wp.vec3(0.0, 0.0, 0.0)
            return

        vi = v[i] * lin_damp
        xi = x[i] + dt * vi
        x_pred[i] = wp.vec3(xi[0], xi[1], 0.0)
        v[i] = wp.vec3(vi[0], vi[1], 0.0)


    @wp.kernel
    def _project_walls_aabb(
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        pr: float,
    ):
        i = wp.tid()
        if inv_m[i] == 0.0:
            return
        xi = x_pred[i]
        xw = wp.clamp(xi[0], xmin + pr, xmax - pr)
        yw = wp.clamp(xi[1], ymin + pr, ymax - pr)
        x_pred[i] = wp.vec3(xw, yw, 0.0)


    @wp.kernel
    def _pusher_contact(
        x_prev: wp.array(dtype=wp.vec3),
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        pusher_pos: wp.vec3,
        pusher_r: float,
        pr: float,
        mu: float,
    ):
        i = wp.tid()
        if inv_m[i] == 0.0:
            return

        xi = x_pred[i]
        d = xi - pusher_pos
        dist = wp.length(d)
        min_dist = pusher_r + pr

        if dist < min_dist and dist > 1e-8:
            n = d / dist
            pen = min_dist - dist

            xi = xi + n * pen

            dp = xi - x_prev[i]
            tang = dp - n * wp.dot(dp, n)
            tang_len = wp.length(tang)
            max_tang = mu * pen

            if tang_len > max_tang and tang_len > 1e-8:
                tang = tang * (max_tang / tang_len)
                xi = x_prev[i] + (n * wp.dot(dp, n) + tang)

            x_pred[i] = wp.vec3(xi[0], xi[1], 0.0)


    @wp.kernel
    def _accum_com(
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        sum_m: wp.array(dtype=float),
        sum_x: wp.array(dtype=float),
        sum_y: wp.array(dtype=float),
    ):
        i = wp.tid()
        invmi = inv_m[i]
        if invmi == 0.0:
            return
        mi = 1.0 / invmi
        xi = x_pred[i]
        wp.atomic_add(sum_m, 0, mi)
        wp.atomic_add(sum_x, 0, mi * xi[0])
        wp.atomic_add(sum_y, 0, mi * xi[1])


    @wp.kernel
    def _accum_ab(
        x_pred: wp.array(dtype=wp.vec3),
        r0: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        c: wp.vec3,
        sum_a: wp.array(dtype=float),
        sum_b: wp.array(dtype=float),
    ):
        i = wp.tid()
        invmi = inv_m[i]
        if invmi == 0.0:
            return
        mi = 1.0 / invmi

        ri0 = r0[i]
        ri = x_pred[i] - c

        a = ri0[0] * ri[1] - ri0[1] * ri[0]
        b = ri0[0] * ri[0] + ri0[1] * ri[1]

        wp.atomic_add(sum_a, 0, mi * a)
        wp.atomic_add(sum_b, 0, mi * b)


    @wp.kernel
    def _shape_match_project(
        x_pred: wp.array(dtype=wp.vec3),
        r0: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        c: wp.vec3,
        theta: float,
        alpha: float,
    ):
        i = wp.tid()
        if inv_m[i] == 0.0:
            return

        ct = wp.cos(theta)
        st = wp.sin(theta)
        ri0 = r0[i]

        rx = ct * ri0[0] - st * ri0[1]
        ry = st * ri0[0] + ct * ri0[1]
        target = wp.vec3(c[0] + rx, c[1] + ry, 0.0)

        xi = x_pred[i]
        x_pred[i] = xi + alpha * (target - xi)


    @wp.kernel
    def _finalize(
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        x_pred: wp.array(dtype=wp.vec3),
        dt: float,
        vel_damp: float,
    ):
        i = wp.tid()
        if inv_m[i] == 0.0:
            return
        vi = (x_pred[i] - x[i]) / dt
        v[i] = wp.vec3(vi[0], vi[1], 0.0) * vel_damp
        x[i] = wp.vec3(x_pred[i][0], x_pred[i][1], 0.0)


class PushTWarpEnv:
    """
    Particle-based PushT simulator in Warp.

    This class keeps simulation in continuous world coordinates and leaves pixel-space
    conversions to the adapter layer.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        params: Optional[PushTWarpParams] = None,
        seed: int = 0,
    ):
        if wp is None:  # pragma: no cover - depends on optional package
            raise ImportError(
                "warp-lang is required for particle_sim backend. Install with `pip install warp-lang`."
            ) from _WARP_IMPORT_ERROR

        wp.init()
        self.params = params or PushTWarpParams()
        self.device_name = str(device)
        self.device = wp.get_device(self.device_name)

        p = self.params
        self.xmin, self.xmax, self.ymin, self.ymax = p.xmin, p.xmax, p.ymin, p.ymax
        self.pr = float(p.particle_radius)
        self.spacing = float(p.spacing)
        self.pusher_r = float(p.pusher_radius)
        self.pusher_speed = float(p.pusher_speed)

        self.frame_dt = float(p.frame_dt)
        self.substeps = int(p.substeps)
        self.iters = int(p.iters)
        self.mu = float(p.mu)
        self.lin_damp = float(p.lin_damp)
        self.vel_damp = float(p.vel_damp)
        self.alpha_rigid = float(p.alpha_rigid)

        self.rng = np.random.default_rng(seed)

        x0 = _points_in_t_grid(
            stem_w=float(p.stem_w),
            stem_h=float(p.stem_h),
            bar_w=float(p.bar_w),
            bar_h=float(p.bar_h),
            spacing=float(p.spacing),
        )
        self.N = int(x0.shape[0])

        c0 = x0[:, :2].mean(axis=0, keepdims=True).astype(np.float32)
        r0 = x0.copy()
        r0[:, 0] -= c0[0, 0]
        r0[:, 1] -= c0[0, 1]

        self.x = wp.array(np.zeros((self.N, 3), dtype=np.float32), dtype=wp.vec3, device=self.device)
        self.v = wp.array(np.zeros((self.N, 3), dtype=np.float32), dtype=wp.vec3, device=self.device)
        self.x_pred = wp.array(np.zeros((self.N, 3), dtype=np.float32), dtype=wp.vec3, device=self.device)

        inv_m = np.full((self.N,), 1.0, dtype=np.float32)
        self.inv_m = wp.array(inv_m, dtype=float, device=self.device)
        self.r0 = wp.array(r0, dtype=wp.vec3, device=self.device)

        self.sum_m = wp.array(np.zeros((1,), dtype=np.float32), dtype=float, device=self.device)
        self.sum_x = wp.array(np.zeros((1,), dtype=np.float32), dtype=float, device=self.device)
        self.sum_y = wp.array(np.zeros((1,), dtype=np.float32), dtype=float, device=self.device)
        self.sum_a = wp.array(np.zeros((1,), dtype=np.float32), dtype=float, device=self.device)
        self.sum_b = wp.array(np.zeros((1,), dtype=np.float32), dtype=float, device=self.device)

        self.pusher_pos = np.array([0.0, -0.12, 0.0], dtype=np.float32)
        self.goal_pose = np.array([0.10, 0.10, 0.0], dtype=np.float32)

        self._last_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._last_pusher_pos = self.pusher_pos.copy()

        self.reset()

    @property
    def num_particles(self) -> int:
        return self.N

    def rest_offsets(self) -> np.ndarray:
        return self.r0.numpy().astype(np.float32)

    def reset(
        self,
        obj_xy: Optional[Tuple[float, float]] = None,
        obj_theta: Optional[float] = None,
        pusher_xy: Optional[Tuple[float, float]] = None,
        goal_pose: Optional[Tuple[float, float, float]] = None,
    ) -> dict:
        if obj_xy is None:
            obj_xy = (-0.08, -0.02)
        if obj_theta is None:
            obj_theta = 0.6
        if pusher_xy is None:
            pusher_xy = (-0.12, -0.18)
        if goal_pose is not None:
            self.goal_pose = np.array(goal_pose, dtype=np.float32)

        r0_host = self.r0.numpy()
        R = _rot2(float(obj_theta))
        xy = np.array(obj_xy, dtype=np.float32)

        x_init = np.zeros((self.N, 3), dtype=np.float32)
        x_init[:, :2] = (r0_host[:, :2] @ R.T) + xy[None, :]
        self.x.assign(x_init)
        self.v.zero_()
        self.x_pred.assign(x_init)

        self.pusher_pos = np.array([pusher_xy[0], pusher_xy[1], 0.0], dtype=np.float32)
        self._last_pusher_pos = self.pusher_pos.copy()
        self._last_pose = self.get_object_pose()
        return self._make_obs()

    def set_goal_pose(self, goal_pose: Sequence[float]) -> None:
        g = np.asarray(goal_pose, dtype=np.float32).reshape(3)
        self.goal_pose = g.copy()

    def set_state(
        self,
        pusher_xy: Sequence[float],
        obj_xy: Sequence[float],
        obj_theta: float,
        goal_pose: Optional[Sequence[float]] = None,
    ) -> dict:
        gx = None if goal_pose is None else tuple(np.asarray(goal_pose, dtype=np.float32).reshape(3).tolist())
        return self.reset(
            obj_xy=tuple(np.asarray(obj_xy, dtype=np.float32).reshape(2).tolist()),
            obj_theta=float(obj_theta),
            pusher_xy=tuple(np.asarray(pusher_xy, dtype=np.float32).reshape(2).tolist()),
            goal_pose=gx,
        )

    def _make_obs(self) -> dict:
        return {
            "obj_pose": self._last_pose.copy(),
            "pusher_pos": self.pusher_pos[:2].copy(),
            "goal_pose": self.goal_pose.copy(),
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(2)

        target = self.pusher_pos[:2] + action
        delta = target - self.pusher_pos[:2]
        dist = float(np.linalg.norm(delta))
        max_dist = self.pusher_speed * self.frame_dt
        if dist > max_dist and dist > 1e-8:
            delta *= (max_dist / dist)

        self._last_pusher_pos = self.pusher_pos.copy()
        self.pusher_pos[:2] += delta

        self._simulate_frame()

        pose = self.get_object_pose()
        self._last_pose = pose

        dx = pose[0] - self.goal_pose[0]
        dy = pose[1] - self.goal_pose[1]
        dtheta = _wrap_pi(float(pose[2] - self.goal_pose[2]))
        reward = float(-(dx * dx + dy * dy + 0.1 * dtheta * dtheta))
        done = (dx * dx + dy * dy) < (0.01 ** 2) and (abs(dtheta) < 0.15)

        info = {
            "object_pose": pose.copy(),
            "pusher_velocity": self.get_pusher_velocity().copy(),
            "object_velocity": self.get_object_velocity().copy(),
        }
        return self._make_obs(), reward, bool(done), info

    def _simulate_frame(self) -> None:
        dt = self.frame_dt / max(1, self.substeps)
        ppos = wp.vec3(float(self.pusher_pos[0]), float(self.pusher_pos[1]), 0.0)

        for _ in range(self.substeps):
            wp.launch(
                kernel=_predict_positions,
                dim=self.N,
                inputs=[self.x, self.v, self.inv_m, self.x_pred, dt, self.lin_damp],
                device=self.device,
            )

            for _ in range(self.iters):
                wp.launch(
                    kernel=_project_walls_aabb,
                    dim=self.N,
                    inputs=[self.x_pred, self.inv_m, self.xmin, self.xmax, self.ymin, self.ymax, self.pr],
                    device=self.device,
                )
                wp.launch(
                    kernel=_pusher_contact,
                    dim=self.N,
                    inputs=[self.x, self.x_pred, self.inv_m, ppos, self.pusher_r, self.pr, self.mu],
                    device=self.device,
                )
                c, theta = self._compute_com_and_theta()
                wp.launch(
                    kernel=_shape_match_project,
                    dim=self.N,
                    inputs=[self.x_pred, self.r0, self.inv_m, c, theta, self.alpha_rigid],
                    device=self.device,
                )

            wp.launch(
                kernel=_finalize,
                dim=self.N,
                inputs=[self.x, self.v, self.inv_m, self.x_pred, dt, self.vel_damp],
                device=self.device,
            )

    def _compute_com_and_theta(self) -> tuple[wp.vec3, float]:
        wp.launch(_zero_scalar, dim=1, inputs=[self.sum_m], device=self.device)
        wp.launch(_zero_scalar, dim=1, inputs=[self.sum_x], device=self.device)
        wp.launch(_zero_scalar, dim=1, inputs=[self.sum_y], device=self.device)

        wp.launch(
            kernel=_accum_com,
            dim=self.N,
            inputs=[self.x_pred, self.inv_m, self.sum_m, self.sum_x, self.sum_y],
            device=self.device,
        )
        sum_m = float(self.sum_m.numpy()[0])
        if sum_m <= 1e-8:
            return wp.vec3(0.0, 0.0, 0.0), 0.0

        cx = float(self.sum_x.numpy()[0]) / sum_m
        cy = float(self.sum_y.numpy()[0]) / sum_m
        c = wp.vec3(cx, cy, 0.0)

        wp.launch(_zero_scalar, dim=1, inputs=[self.sum_a], device=self.device)
        wp.launch(_zero_scalar, dim=1, inputs=[self.sum_b], device=self.device)

        wp.launch(
            kernel=_accum_ab,
            dim=self.N,
            inputs=[self.x_pred, self.r0, self.inv_m, c, self.sum_a, self.sum_b],
            device=self.device,
        )
        a = float(self.sum_a.numpy()[0])
        b = float(self.sum_b.numpy()[0])
        theta = math.atan2(a, b) if (abs(a) + abs(b)) > 1e-12 else 0.0
        return c, float(theta)

    def get_object_pose(self) -> np.ndarray:
        c, theta = self._compute_com_and_theta()
        return np.array([float(c[0]), float(c[1]), float(theta)], dtype=np.float32)

    def get_particle_positions(self) -> np.ndarray:
        x = self.x.numpy().astype(np.float32)
        return x[:, :2].copy()

    def get_particle_velocities(self) -> np.ndarray:
        v = self.v.numpy().astype(np.float32)
        return v[:, :2].copy()

    def get_pusher_velocity(self) -> np.ndarray:
        dt = max(1e-8, float(self.frame_dt))
        return ((self.pusher_pos - self._last_pusher_pos)[:2] / dt).astype(np.float32)

    def get_object_velocity(self) -> np.ndarray:
        vel = self.get_particle_velocities()
        if vel.shape[0] == 0:
            return np.zeros(2, dtype=np.float32)
        return vel.mean(axis=0).astype(np.float32)
