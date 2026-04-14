from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - optional dependency path
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


PUSHT_RENDER_PIXELS = 512.0
PUSHT_WORLD_SIZE = 0.5
PUSHT_WORLD_PER_PIXEL = PUSHT_WORLD_SIZE / PUSHT_RENDER_PIXELS

GT_T_SCALE_PX = 40.0
GT_T_STEM_W = GT_T_SCALE_PX * PUSHT_WORLD_PER_PIXEL
GT_T_STEM_H = 3.0 * GT_T_SCALE_PX * PUSHT_WORLD_PER_PIXEL
GT_T_BAR_W = 4.0 * GT_T_SCALE_PX * PUSHT_WORLD_PER_PIXEL
GT_T_BAR_H = GT_T_SCALE_PX * PUSHT_WORLD_PER_PIXEL
GT_PUSHER_RADIUS = 15.0 * PUSHT_WORLD_PER_PIXEL


# -----------------------------
# Geometry helpers (CPU)
# -----------------------------

def _points_in_t_grid(
    stem_w: float,
    stem_h: float,
    bar_w: float,
    bar_h: float,
    spacing: float,
    min_particles: int = 1,
) -> np.ndarray:
    """
    Build a PushT T in the same local-coordinate convention as the GT env.

    The GT tee is defined as:
    - a horizontal bar spanning x in [-bar_w/2, bar_w/2], y in [0, bar_h]
    - a vertical stem spanning x in [-stem_w/2, stem_w/2], y in [bar_h, bar_h + stem_h]

    Sampling intentionally uses one uniform occupancy grid over the whole tee.
    That means coarse spacings can under-resolve thin features, but at a truly
    fine canonical spacing the resulting cloud converges to the intended GT
    geometry without shape-specific sampling hacks.

    Important: PushT state/rendering uses the body's local origin directly.
    Although the Pymunk body stores a non-zero center of gravity, that does not
    shift the local vertex coordinates used by render/eval/state. So the local
    origin here must stay at the bar/stem junction frame, not at any centroid.

    If the grid sampling yields fewer than `min_particles`, collapse to one point.
    """
    xs = np.arange(-bar_w * 0.5, bar_w * 0.5 + 1e-9, spacing, dtype=np.float32)
    ys = np.arange(0.0, bar_h + stem_h + 1e-9, spacing, dtype=np.float32)

    pts = []
    for y in ys:
        for x in xs:
            in_stem = (abs(float(x)) <= stem_w * 0.5) and (y >= bar_h) and (y <= bar_h + stem_h)
            in_bar = (abs(float(x)) <= bar_w * 0.5) and (y >= 0.0) and (y <= bar_h)
            if in_stem or in_bar:
                pts.append((float(x), float(y), 0.0))

    if len(pts) < max(1, int(min_particles)):
        pts = [(0.0, 0.0, 0.0)]

    return np.array(pts, dtype=np.float32)


def _rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PushTParticleLevel:
    rest_offsets: np.ndarray
    spacing: float
    achieved_cover_radius: float
    target_particle_count: int
    pose_offset_local: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    is_canonical: bool = False
    is_single_particle: bool = False


def _greedy_farthest_point_order(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    n = int(pts.shape[0])
    if n <= 0:
        raise ValueError("points_xy must contain at least one point.")
    if n == 1:
        return np.asarray([0], dtype=np.int64), np.asarray([0.0], dtype=np.float32)

    order = np.empty((n,), dtype=np.int64)
    coverage = np.empty((n,), dtype=np.float32)

    center = pts.mean(axis=0, keepdims=True)
    d2_center = np.sum((pts - center) ** 2, axis=1)
    first = int(np.argmin(d2_center))
    order[0] = first

    min_d2 = np.sum((pts - pts[first:first + 1]) ** 2, axis=1)
    min_d2[first] = 0.0
    coverage[0] = float(np.sqrt(np.max(min_d2)))

    selected = np.zeros((n,), dtype=bool)
    selected[first] = True
    for i in range(1, n):
        next_idx = int(np.argmax(min_d2))
        order[i] = next_idx
        selected[next_idx] = True
        d2_next = np.sum((pts - pts[next_idx:next_idx + 1]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_next)
        min_d2[selected] = 0.0
        coverage[i] = float(np.sqrt(np.max(min_d2)))

    return order, coverage.astype(np.float32)


def _cluster_points_by_prefix(points_xy: np.ndarray, center_order: np.ndarray, prefix_size: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    n = int(pts.shape[0])
    k = int(prefix_size)
    if n <= 0:
        raise ValueError("points_xy must contain at least one point.")
    if k <= 0 or k > n:
        raise ValueError(f"prefix_size must be in [1, {n}], got {prefix_size}")

    if k == 1:
        return np.zeros((1, 3), dtype=np.float32)

    selected = np.asarray(center_order[:k], dtype=np.int64).reshape(-1)
    centers = pts[selected]
    d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assign = np.argmin(d2, axis=1)

    coarse = np.zeros((k, 3), dtype=np.float32)
    for j in range(k):
        members = pts[assign == j]
        if members.shape[0] <= 0:
            coarse[j, :2] = centers[j]
        else:
            coarse[j, :2] = members.mean(axis=0)

    return coarse.astype(np.float32)


def _effective_spacing_from_count(area_t: float, count: int) -> float:
    return float(math.sqrt(max(float(area_t), 1e-12) / max(int(count), 1)))


def _choose_canonical_spacing_for_target_count(
    *,
    target_particle_count: int,
    stem_w: float,
    stem_h: float,
    bar_w: float,
    bar_h: float,
    min_particles: int,
) -> tuple[float, np.ndarray]:
    target = max(int(min_particles), int(target_particle_count))
    area_t = float(stem_w) * float(stem_h) + float(bar_w) * float(bar_h)
    nominal = max(1e-5, math.sqrt(max(area_t, 1e-12) / max(target, 1)))
    cache: dict[float, np.ndarray] = {}

    def points_for_spacing(spacing: float) -> np.ndarray:
        sp = max(1e-6, float(spacing))
        key = float(f"{sp:.12f}")
        cached = cache.get(key, None)
        if cached is None:
            cached = _points_in_t_grid(
                stem_w=float(stem_w),
                stem_h=float(stem_h),
                bar_w=float(bar_w),
                bar_h=float(bar_h),
                spacing=sp,
                min_particles=int(min_particles),
            ).astype(np.float32)
            cache[key] = cached
        return cached

    lo = nominal * 0.125
    hi = nominal * 8.0
    while points_for_spacing(lo).shape[0] < target and lo > 1e-6:
        lo *= 0.5
    while points_for_spacing(hi).shape[0] > target:
        hi *= 2.0

    best_spacing = nominal
    best_points = points_for_spacing(best_spacing)

    def score(spacing: float, count: int) -> tuple[float, int, float]:
        return (
            abs(int(count) - target),
            0 if int(count) >= target else 1,
            float(spacing),
        )

    best_score = score(best_spacing, best_points.shape[0])
    for spacing in (lo, hi):
        pts = points_for_spacing(spacing)
        s = score(spacing, pts.shape[0])
        if s < best_score:
            best_spacing = float(spacing)
            best_points = pts
            best_score = s

    for _ in range(36):
        mid = math.sqrt(lo * hi)
        pts = points_for_spacing(mid)
        mid_count = int(pts.shape[0])
        s = score(mid, mid_count)
        if s < best_score:
            best_spacing = float(mid)
            best_points = pts
            best_score = s
        if mid_count >= target:
            lo = mid
        else:
            hi = mid

    return float(best_spacing), best_points.astype(np.float32)


def _tee_structure_metrics(
    points_xy: np.ndarray,
    *,
    canonical_bbox: np.ndarray,
    stem_w: float,
    stem_h: float,
    bar_w: float,
    bar_h: float,
) -> tuple[int, float]:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] <= 0:
        return 4, float("inf")
    if pts.shape[0] <= 4:
        return 0, 0.0

    # The hierarchy builder evaluates both GT-local clouds and centered
    # rest-offset clouds. Normalize away translation before checking whether
    # the coarse points still span the expected T structure.
    pts = pts.copy()
    pts[:, 0] -= 0.5 * float(pts[:, 0].min() + pts[:, 0].max())
    pts[:, 1] -= float(pts[:, 1].min())

    bar_center_y = 0.5 * float(bar_h)
    junction_y = float(bar_h)
    top_stem_y = float(bar_h) + 0.55 * float(stem_h)
    stem_x_tol = max(float(stem_w), 0.15 * float(bar_w))
    bar_y_tol = max(0.9 * float(bar_h), 0.15 * float(stem_h))

    left_arm = np.any((pts[:, 0] <= -0.25 * float(bar_w)) & (np.abs(pts[:, 1] - bar_center_y) <= bar_y_tol))
    right_arm = np.any((pts[:, 0] >= 0.25 * float(bar_w)) & (np.abs(pts[:, 1] - bar_center_y) <= bar_y_tol))
    stem_top = np.any((np.abs(pts[:, 0]) <= stem_x_tol) & (pts[:, 1] >= top_stem_y))
    junction = np.any((np.abs(pts[:, 0]) <= stem_x_tol) & (np.abs(pts[:, 1] - junction_y) <= bar_y_tol))

    bbox = np.ptp(pts, axis=0)
    canonical_bbox = np.asarray(canonical_bbox, dtype=np.float32).reshape(2)
    bbox_penalty = float(
        abs(float(bbox[0]) / max(float(canonical_bbox[0]), 1e-6) - 1.0)
        + abs(float(bbox[1]) / max(float(canonical_bbox[1]), 1e-6) - 1.0)
    )
    missing = int(not left_arm) + int(not right_arm) + int(not stem_top) + int(not junction)
    return missing, bbox_penalty


def _select_soft_particle_count(
    *,
    target_particle_count: int,
    total_particles: int,
    center_order: np.ndarray,
    coverage: np.ndarray,
    canonical_gt_xy: np.ndarray,
    stem_w: float,
    stem_h: float,
    bar_w: float,
    bar_h: float,
) -> int:
    target = max(1, min(int(target_particle_count), int(total_particles)))
    if target <= 4 or target >= int(total_particles):
        return target

    low_slack = max(2, int(math.ceil(0.20 * target)))
    high_slack = max(4, int(math.ceil(0.35 * target)))
    lo = max(5, target - low_slack)
    hi = min(int(total_particles) - 1, target + high_slack)
    canonical_bbox = np.ptp(np.asarray(canonical_gt_xy, dtype=np.float32), axis=0)

    best_k = target
    best_score: Optional[tuple[int, int, float, float]] = None
    for k in range(lo, hi + 1):
        coarse_gt = _cluster_points_by_prefix(canonical_gt_xy, center_order, k)
        missing, bbox_penalty = _tee_structure_metrics(
            coarse_gt[:, :2],
            canonical_bbox=canonical_bbox,
            stem_w=stem_w,
            stem_h=stem_h,
            bar_w=bar_w,
            bar_h=bar_h,
        )
        score = (
            int(missing),
            abs(int(k) - target),
            float(bbox_penalty),
            float(coverage[k - 1]),
        )
        if best_score is None or score < best_score:
            best_k = int(k)
            best_score = score

    return int(best_k)


def build_t_particle_hierarchy(
    *,
    particle_counts: Sequence[int],
    stem_w: float,
    stem_h: float,
    bar_w: float,
    bar_h: float,
    min_particles: int = 1,
) -> list[PushTParticleLevel]:
    counts = [int(c) for c in list(particle_counts)]
    if len(counts) <= 0:
        raise ValueError("particle_counts must contain at least one value.")
    if any(c <= 0 for c in counts):
        raise ValueError("All particle_counts values must be >= 1.")
    if any(b <= a for a, b in zip(counts[:-1], counts[1:])):
        raise ValueError("particle_counts must be strictly increasing.")

    finest_target = int(counts[-1])
    canonical_spacing, canonical_gt = _choose_canonical_spacing_for_target_count(
        target_particle_count=finest_target,
        stem_w=float(stem_w),
        stem_h=float(stem_h),
        bar_w=float(bar_w),
        bar_h=float(bar_h),
        min_particles=int(min_particles),
    )
    area_t = float(stem_w) * float(stem_h) + float(bar_w) * float(bar_h)
    canonical_pose_offset = canonical_gt.mean(axis=0, keepdims=True).astype(np.float32)
    canonical = canonical_gt.copy()
    canonical[:, :2] -= canonical_pose_offset[0, :2]

    levels_fine_to_coarse: list[PushTParticleLevel] = [
        PushTParticleLevel(
            rest_offsets=canonical.copy(),
            pose_offset_local=canonical_pose_offset.reshape(3).copy(),
            spacing=float(canonical_spacing),
            achieved_cover_radius=0.0,
            target_particle_count=int(finest_target),
            is_canonical=True,
            is_single_particle=bool(canonical.shape[0] == 1),
        )
    ]
    if canonical.shape[0] <= 1:
        return levels_fine_to_coarse[::-1]

    center_order, coverage = _greedy_farthest_point_order(canonical_gt[:, :2])
    prev_k = int(canonical.shape[0])
    for target_count in reversed(counts[:-1]):
        k = _select_soft_particle_count(
            target_particle_count=int(target_count),
            total_particles=int(canonical.shape[0]),
            center_order=center_order,
            coverage=coverage,
            canonical_gt_xy=canonical_gt[:, :2],
            stem_w=float(stem_w),
            stem_h=float(stem_h),
            bar_w=float(bar_w),
            bar_h=float(bar_h),
        )
        if k >= int(canonical.shape[0]) or k == prev_k:
            continue
        coarse_gt = _cluster_points_by_prefix(canonical_gt[:, :2], center_order, k)
        coarse_pose_offset = coarse_gt.mean(axis=0, keepdims=True).astype(np.float32)
        rest_offsets = coarse_gt.copy()
        rest_offsets[:, :2] -= coarse_pose_offset[0, :2]
        levels_fine_to_coarse.append(
            PushTParticleLevel(
                rest_offsets=rest_offsets,
                pose_offset_local=coarse_pose_offset.reshape(3).copy(),
                spacing=_effective_spacing_from_count(area_t, k),
                achieved_cover_radius=float(coverage[k - 1]),
                target_particle_count=int(target_count),
                is_single_particle=bool(k == 1),
            )
        )
        prev_k = k

    return levels_fine_to_coarse[::-1]


@dataclass(frozen=True)
class PushTWarpParams:
    xmin: float = -0.25
    xmax: float = 0.25
    ymin: float = -0.25
    ymax: float = 0.25

    spacing: float = 0.012
    stem_w: float = GT_T_STEM_W
    stem_h: float = GT_T_STEM_H
    bar_w: float = GT_T_BAR_W
    bar_h: float = GT_T_BAR_H
    min_particles: int = 1
    force_single_particle: bool = False
    rest_offsets: Optional[np.ndarray] = None

    particle_radius: Optional[float] = None  # None -> auto-scale with N
    radius_scale: float = 1.0
    radius_clip_spacing: bool = False

    pusher_radius: float = GT_PUSHER_RADIUS
    sim_hz: int = 100
    control_hz: int = 10
    pusher_k_p: float = 100.0
    pusher_k_v: float = 20.0

    substeps: int = 16
    iters: int = 8
    mu: float = 0.6
    contact_alpha: float = 0.35
    ground_friction_accel: float = 2.0
    rest_speed_eps: float = 0.01
    lin_damp: float = 0.995
    vel_damp: float = 0.999
    alpha_rigid: float = 1.0


# -----------------------------
# Warp kernels
# -----------------------------

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
        contact_alpha: float,
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
            raw_pen = min_dist - dist
            pen = raw_pen * contact_alpha

            xi = xi + n * pen

            dp = xi - x_prev[i]
            tang = dp - n * wp.dot(dp, n)
            tang_len = wp.length(tang)
            max_tang = mu * raw_pen

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


    @wp.kernel
    def _apply_ground_friction(
        v: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        dt: float,
        friction_accel: float,
        rest_speed_eps: float,
    ):
        i = wp.tid()
        if inv_m[i] == 0.0:
            return

        vi = v[i]
        speed = wp.sqrt(vi[0] * vi[0] + vi[1] * vi[1])
        if speed <= rest_speed_eps:
            v[i] = wp.vec3(0.0, 0.0, 0.0)
            return

        new_speed = speed - friction_accel * dt
        if new_speed <= rest_speed_eps:
            v[i] = wp.vec3(0.0, 0.0, 0.0)
            return

        scale = new_speed / speed
        v[i] = wp.vec3(vi[0] * scale, vi[1] * scale, 0.0)


# -----------------------------
# Environment
# -----------------------------

class PushTWarpEnv:
    def __init__(
        self,
        device: str = "cuda:0",
        params: Optional[PushTWarpParams] = None,
        seed: int = 0,
        **kwargs,
    ):
        if wp is None:  # pragma: no cover - depends on optional package
            raise ImportError(
                "warp-lang is required for particle_sim backend. Install with `pip install warp-lang`."
            ) from _WARP_IMPORT_ERROR

        wp.init()
        if params is None:
            self.params = PushTWarpParams(**kwargs) if kwargs else PushTWarpParams()
        else:
            if len(kwargs) > 0:
                raise ValueError("Provide either `params=PushTWarpParams(...)` or direct kwargs, not both.")
            self.params = params
        self.device_name = str(device)
        self.device = wp.get_device(self.device_name)

        p = self.params
        self.xmin, self.xmax, self.ymin, self.ymax = float(p.xmin), float(p.xmax), float(p.ymin), float(p.ymax)
        self.spacing = float(p.spacing)

        self.stem_w = float(p.stem_w)
        self.stem_h = float(p.stem_h)
        self.bar_w = float(p.bar_w)
        self.bar_h = float(p.bar_h)

        self.pusher_r = float(p.pusher_radius)
        self.sim_hz = int(p.sim_hz)
        self.control_hz = int(p.control_hz)
        if self.sim_hz <= 0:
            raise ValueError(f"sim_hz must be > 0, got {self.sim_hz}")
        if self.control_hz <= 0:
            raise ValueError(f"control_hz must be > 0, got {self.control_hz}")
        if self.sim_hz % self.control_hz != 0:
            raise ValueError(
                f"sim_hz must be divisible by control_hz, got sim_hz={self.sim_hz}, control_hz={self.control_hz}"
            )
        self.pusher_k_p = float(p.pusher_k_p)
        self.pusher_k_v = float(p.pusher_k_v)
        self.control_dt = 1.0 / float(self.control_hz)
        self.sim_dt = 1.0 / float(self.sim_hz)
        self.controller_steps = self.sim_hz // self.control_hz

        self.substeps = int(p.substeps)
        self.iters = int(p.iters)
        self.mu = float(p.mu)
        self.contact_alpha = float(p.contact_alpha)
        self.ground_friction_accel = float(p.ground_friction_accel)
        self.rest_speed_eps = float(p.rest_speed_eps)
        self.lin_damp = float(p.lin_damp)
        self.vel_damp = float(p.vel_damp)
        self.alpha_rigid = float(p.alpha_rigid)

        self.rng = np.random.default_rng(seed)

        self.force_single_particle = bool(p.force_single_particle)
        if p.rest_offsets is not None:
            x0 = np.asarray(p.rest_offsets, dtype=np.float32).reshape(-1, 3).copy()
            if x0.shape[0] <= 0:
                raise ValueError("rest_offsets must contain at least one particle.")
            x0[:, :2] -= x0[:, :2].mean(axis=0, keepdims=True)
        elif self.force_single_particle:
            x0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        else:
            x0 = _points_in_t_grid(
                stem_w=self.stem_w,
                stem_h=self.stem_h,
                bar_w=self.bar_w,
                bar_h=self.bar_h,
                spacing=self.spacing,
                min_particles=int(p.min_particles),
            )
        self.N = int(x0.shape[0])

        area_t = self.stem_w * self.stem_h + self.bar_w * self.bar_h
        if p.particle_radius is None:
            r = math.sqrt(area_t / (self.N * math.pi)) * float(p.radius_scale)
            if bool(p.radius_clip_spacing) and self.N > 1:
                r = float(np.clip(r, 0.35 * self.spacing, 1.50 * self.spacing))
            self.pr = float(r)
        else:
            self.pr = float(p.particle_radius)

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
        self.pusher_velocity = np.zeros((2,), dtype=np.float32)
        self.goal_pose = np.array([0.10, 0.10, 0.0], dtype=np.float32)

        self._theta_state = 0.0
        self._last_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.reset()

    @property
    def num_particles(self) -> int:
        return int(self.N)

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
        self.pusher_velocity = np.zeros((2,), dtype=np.float32)
        self._theta_state = float(_wrap_pi(float(obj_theta)))

        self._last_pose = self.get_object_pose()
        return self._make_obs()

    def set_goal_pose(self, goal_pose: Sequence[float]) -> None:
        g = np.asarray(goal_pose, dtype=np.float32).reshape(3)
        self.goal_pose = g.copy()

    def set_state(
        self,
        pusher_xy: Optional[Sequence[float]] = None,
        obj_xy: Optional[Sequence[float]] = None,
        obj_theta: Optional[float] = None,
        goal_pose: Optional[Sequence[float]] = None,
        obj_pose: Optional[Sequence[float]] = None,
        obj_twist: Optional[Sequence[float]] = None,
        pusher_velocity: Optional[Sequence[float]] = None,
    ) -> dict:
        """
        Compatibility shim supporting both:
        - set_state(pusher_xy=..., obj_xy=..., obj_theta=..., goal_pose=...)
        - set_state(obj_pose=(x,y,theta), obj_twist=(vx,vy,omega), pusher_xy=...)
        """
        if obj_pose is not None:
            op = np.asarray(obj_pose, dtype=np.float32).reshape(3)
            obj_xy = op[:2]
            obj_theta = float(op[2])

        if pusher_xy is None:
            pusher_xy = np.asarray(self.pusher_pos[:2], dtype=np.float32)
        if obj_xy is None:
            obj_xy = np.asarray(self.get_object_pose()[:2], dtype=np.float32)
        if obj_theta is None:
            obj_theta = float(self.get_object_pose()[2])

        gx = None if goal_pose is None else tuple(np.asarray(goal_pose, dtype=np.float32).reshape(3).tolist())
        self.reset(
            obj_xy=tuple(np.asarray(obj_xy, dtype=np.float32).reshape(2).tolist()),
            obj_theta=float(obj_theta),
            pusher_xy=tuple(np.asarray(pusher_xy, dtype=np.float32).reshape(2).tolist()),
            goal_pose=gx,
        )
        if pusher_velocity is not None:
            pv = np.asarray(pusher_velocity, dtype=np.float32).reshape(-1)
            if pv.shape[0] < 2:
                raise ValueError(
                    f"pusher_velocity must have at least 2 dims, got {tuple(pv.shape)}"
                )
            self.pusher_velocity = pv[:2].astype(np.float32).copy()

        # Optional twist initialization after positional reset.
        if obj_twist is not None:
            twist = np.asarray(obj_twist, dtype=np.float32).reshape(3)
            vx, vy, omega = float(twist[0]), float(twist[1]), float(twist[2])

            x_host = self.x.numpy().astype(np.float32)
            center = x_host[:, :2].mean(axis=0, keepdims=True)
            r = x_host[:, :2] - center
            perp = np.stack([-r[:, 1], r[:, 0]], axis=1)
            v_host = np.zeros((self.N, 3), dtype=np.float32)
            v_host[:, :2] = np.array([vx, vy], dtype=np.float32)[None, :] + omega * perp
            self.v.assign(v_host)
            self.x_pred.assign(self.x.numpy().astype(np.float32))

        self._last_pose = self.get_object_pose()
        return self._make_obs()

    def capture_state(self) -> dict[str, np.ndarray]:
        return {
            "pusher_xy": np.asarray(self.pusher_pos[:2], dtype=np.float32).copy(),
            "pusher_velocity": self.get_pusher_velocity().astype(np.float32),
            "obj_pose": self.get_object_pose().astype(np.float32),
            "obj_twist": self.get_object_twist().astype(np.float32),
            "goal_pose": np.asarray(self.goal_pose, dtype=np.float32).copy(),
        }

    def restore_state(self, state: dict[str, Sequence[float]]) -> dict:
        if not isinstance(state, dict):
            raise ValueError(f"restore_state expects a dict snapshot, got {type(state).__name__}")
        return self.set_state(
            pusher_xy=state.get("pusher_xy", None),
            pusher_velocity=state.get("pusher_velocity", None),
            obj_pose=state.get("obj_pose", None),
            obj_twist=state.get("obj_twist", None),
            goal_pose=state.get("goal_pose", None),
        )

    def _make_obs(self) -> dict:
        pose = self._last_pose.copy()
        return {
            "obj_pose": pose,
            "obj_twist": self.get_object_twist(),
            "pusher_pos": self.pusher_pos[:2].copy(),
            "goal_pose": self.goal_pose.copy(),
            "meta": {
                "num_particles": int(self.N),
                "particle_radius": float(self.pr),
                "force_single_particle": bool(self.force_single_particle),
            },
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(2)

        target = self.pusher_pos[:2] + action
        pusher_path, final_velocity = self._build_pusher_path(target)
        self._simulate_frame(pusher_path=pusher_path)
        self.pusher_pos = pusher_path[-1].copy()
        self.pusher_velocity = final_velocity.copy()

        pose = self.get_object_pose()
        self._last_pose = pose

        dx = pose[0] - self.goal_pose[0]
        dy = pose[1] - self.goal_pose[1]
        dtheta = _wrap_pi(float(pose[2] - self.goal_pose[2]))
        reward = float(-(dx * dx + dy * dy + 0.1 * dtheta * dtheta))
        done = (dx * dx + dy * dy) < (0.01 ** 2) and (abs(dtheta) < 0.15)

        obs = self._make_obs()
        info = {
            "obj_pose": obs["obj_pose"].copy(),
            "obj_twist": obs["obj_twist"].copy(),
            "meta": dict(obs["meta"]),
        }
        return obs, float(reward), bool(done), info

    def _simulate_frame(
        self,
        pusher_path: Optional[np.ndarray] = None,
    ) -> None:
        if pusher_path is None:
            pusher_path = np.asarray([self.pusher_pos.copy(), self.pusher_pos.copy()], dtype=np.float32)
        else:
            pusher_path = np.asarray(pusher_path, dtype=np.float32).reshape(-1, 3)
        if pusher_path.shape[0] < 2:
            raise ValueError("pusher_path must contain at least two points.")
        num_segments = int(pusher_path.shape[0] - 1)
        dt = self.control_dt / float(num_segments)

        # Replay the PD controller path at its native controller cadence instead
        # of resampling it at an unrelated substep count. This keeps particle
        # contact timing aligned with GT PushT's 100 Hz integration.
        for seg_idx in range(num_segments):
            pxy = pusher_path[seg_idx + 1, :2]
            ppos = wp.vec3(float(pxy[0]), float(pxy[1]), 0.0)

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
                    inputs=[
                        self.x,
                        self.x_pred,
                        self.inv_m,
                        ppos,
                        self.pusher_r,
                        self.pr,
                        self.mu,
                        self.contact_alpha,
                    ],
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

            wp.launch(
                kernel=_apply_ground_friction,
                dim=self.N,
                inputs=[
                    self.v,
                    self.inv_m,
                    dt,
                    self.ground_friction_accel,
                    self.rest_speed_eps,
                ],
                device=self.device,
            )

    def _compute_com_and_theta(self) -> tuple[object, float]:
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
        if self.N <= 1 or (abs(a) + abs(b)) <= 1e-12:
            theta = float(self._theta_state)
        else:
            theta = float(_wrap_pi(math.atan2(a, b)))
            self._theta_state = theta
        return c, theta

    def get_object_pose(self) -> np.ndarray:
        c, theta = self._compute_com_and_theta()
        return np.array([float(c[0]), float(c[1]), float(theta)], dtype=np.float32)

    def get_object_twist(self) -> np.ndarray:
        v_host = self.v.numpy()[:, :2].astype(np.float32)
        x_host = self.x.numpy()[:, :2].astype(np.float32)
        if v_host.shape[0] <= 0:
            return np.zeros((3,), dtype=np.float32)

        v_com = v_host.mean(axis=0)
        r = x_host - x_host.mean(axis=0, keepdims=True)
        perp = np.stack([-r[:, 1], r[:, 0]], axis=1)
        num = float((perp * (v_host - v_com)).sum())
        den = float((perp * perp).sum()) + 1e-9
        omega = num / den
        return np.array([v_com[0], v_com[1], omega], dtype=np.float32)

    def get_particle_positions(self) -> np.ndarray:
        return self.x.numpy().astype(np.float32)[:, :2].copy()

    def get_particle_velocities(self) -> np.ndarray:
        return self.v.numpy().astype(np.float32)[:, :2].copy()

    def get_pusher_velocity(self) -> np.ndarray:
        return self.pusher_velocity.astype(np.float32).copy()

    def get_object_velocity(self) -> np.ndarray:
        # Backward-compatible alias used by existing adapters.
        return self.get_object_twist()[:2].copy()

    def _build_pusher_path(self, target_xy: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target_xy, dtype=np.float64).reshape(2)
        pos = np.asarray(self.pusher_pos[:2], dtype=np.float64).copy()
        vel = np.asarray(self.pusher_velocity, dtype=np.float64).copy()

        path = np.zeros((self.controller_steps + 1, 3), dtype=np.float32)
        path[0, :2] = pos.astype(np.float32)

        for step_idx in range(self.controller_steps):
            acc = self.pusher_k_p * (target - pos) - self.pusher_k_v * vel
            vel = vel + acc * self.sim_dt
            pos = pos + vel * self.sim_dt
            path[step_idx + 1, :2] = pos.astype(np.float32)

        return path, vel.astype(np.float32)


if __name__ == "__main__":
    if wp is None:
        raise SystemExit("warp-lang is not installed.")
    levels = build_t_particle_hierarchy(
        particle_counts=[1, 4, 16, 64, 128, 256],
        stem_w=GT_T_STEM_W,
        stem_h=GT_T_STEM_H,
        bar_w=GT_T_BAR_W,
        bar_h=GT_T_BAR_H,
        min_particles=1,
    )
    for li, level in enumerate(levels):
        env = PushTWarpEnv(
            device="cuda:0",
            params=PushTWarpParams(
                spacing=level.spacing,
                rest_offsets=level.rest_offsets,
                force_single_particle=bool(level.is_single_particle),
                particle_radius=None,
                min_particles=1,
            ),
        )
        meta = env.reset()["meta"]
        print(
            f"level={li} spacing={level.spacing:.4f} "
            f"N={meta['num_particles']:4d} pr={meta['particle_radius']:.4f}"
        )
