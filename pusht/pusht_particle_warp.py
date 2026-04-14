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


def _gt_tee_body_com_local(
    stem_w: float,
    stem_h: float,
    bar_w: float,
    bar_h: float,
) -> np.ndarray:
    del stem_w, bar_w
    bar_cog_y = 0.5 * float(bar_h)
    stem_cog_y = float(bar_h) + 0.5 * float(stem_h)
    return np.asarray([0.0, 0.5 * (bar_cog_y + stem_cog_y), 0.0], dtype=np.float32)


def _rect_moment_about_origin(
    mass: float,
    width: float,
    height: float,
    centroid_xy: Sequence[float],
) -> float:
    cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    return float(mass) * (
        ((float(width) * float(width)) + (float(height) * float(height))) / 12.0
        + cx * cx
        + cy * cy
    )


def _gt_tee_body_inertia(
    stem_w: float,
    stem_h: float,
    bar_w: float,
    bar_h: float,
) -> float:
    del stem_w, stem_h
    # Match PushTEnv.add_tee() exactly, including its current duplicated
    # inertia term for the second rectangle.
    bar_centroid = np.asarray([0.0, 0.5 * float(bar_h)], dtype=np.float32)
    bar_inertia = _rect_moment_about_origin(
        mass=1.0,
        width=float(bar_w),
        height=float(bar_h),
        centroid_xy=bar_centroid,
    )
    return float(bar_inertia + bar_inertia)


def _closest_point_on_segment(
    point_xy: Sequence[float],
    seg_start_xy: Sequence[float],
    seg_end_xy: Sequence[float],
) -> tuple[np.ndarray, float]:
    point = np.asarray(point_xy, dtype=np.float32).reshape(2)
    start = np.asarray(seg_start_xy, dtype=np.float32).reshape(2)
    end = np.asarray(seg_end_xy, dtype=np.float32).reshape(2)
    seg = end - start
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 <= 1e-12:
        return start.astype(np.float32), 0.0
    t = float(np.clip(np.dot(point - start, seg) / seg_len2, 0.0, 1.0))
    return (start + seg * t).astype(np.float32), t


def _swept_particle_contact(
    pusher_start: Sequence[float],
    pusher_end: Sequence[float],
    particle_xy: Sequence[float],
    *,
    pusher_r: float,
    pr: float,
) -> dict[str, object]:
    particle = np.asarray(particle_xy, dtype=np.float32).reshape(2)
    closest_xy, t = _closest_point_on_segment(particle, pusher_start, pusher_end)
    delta = particle - closest_xy
    dist = float(np.linalg.norm(delta))
    min_dist = float(pusher_r) + float(pr)
    hit = dist < min_dist
    normal = np.zeros((2,), dtype=np.float32)
    if hit:
        if dist > 1e-8:
            normal = (delta / dist).astype(np.float32)
        else:
            seg = np.asarray(pusher_end, dtype=np.float32).reshape(2) - np.asarray(pusher_start, dtype=np.float32).reshape(2)
            seg_norm = float(np.linalg.norm(seg))
            if seg_norm > 1e-8:
                normal = np.asarray([-seg[1] / seg_norm, seg[0] / seg_norm], dtype=np.float32)
            else:
                normal = np.asarray([1.0, 0.0], dtype=np.float32)
    return {
        "hit": bool(hit),
        "toi": float(t),
        "closest_xy": closest_xy.astype(np.float32),
        "penetration": max(0.0, min_dist - dist),
        "min_dist": float(min_dist),
        "normal_xy": normal.astype(np.float32),
    }


def _normalize_particle_radii(particle_xy: np.ndarray, pr: float | Sequence[float] | np.ndarray) -> np.ndarray:
    pts = np.asarray(particle_xy, dtype=np.float32).reshape(-1, 2)
    radii = np.asarray(pr, dtype=np.float32).reshape(-1)
    if radii.size == 1:
        return np.full((pts.shape[0],), float(radii[0]), dtype=np.float32)
    if radii.shape[0] != pts.shape[0]:
        raise ValueError(
            f"particle radii shape mismatch: got {radii.shape[0]} radii for {pts.shape[0]} particles"
        )
    return radii.astype(np.float32)


def _resolve_particle_union_endpoint(
    pusher_start: Sequence[float],
    pusher_end: Sequence[float],
    particle_xy: np.ndarray,
    *,
    pusher_r: float,
    pr: float | Sequence[float] | np.ndarray,
    eps: float = 1e-5,
    max_iters: int = 16,
) -> np.ndarray:
    start = np.asarray(pusher_start, dtype=np.float32).reshape(2)
    end = np.asarray(pusher_end, dtype=np.float32).reshape(2)
    pts = np.asarray(particle_xy, dtype=np.float32).reshape(-1, 2)
    radii = _normalize_particle_radii(pts, pr)
    seg = end - start
    seg_len2 = float(np.dot(seg, seg))

    best_t = 1.0
    if pts.shape[0] > 0:
        start_d = np.linalg.norm(start[None, :] - pts, axis=1)
        if np.any(start_d < (float(pusher_r) + radii)):
            best_t = 0.0
    if seg_len2 > 1e-12:
        for center, radius in zip(pts, radii):
            min_dist = float(pusher_r) + float(radius)
            rel = start - center
            if float(np.dot(rel, rel)) <= (min_dist * min_dist):
                continue
            a = seg_len2
            b = 2.0 * float(np.dot(rel, seg))
            c = float(np.dot(rel, rel) - (min_dist * min_dist))
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                continue
            sqrt_disc = math.sqrt(max(disc, 0.0))
            t0 = (-b - sqrt_disc) / (2.0 * a)
            if 0.0 <= t0 <= best_t:
                best_t = float(t0)

    point = (start + seg * best_t).astype(np.float32)
    fallback_dir = seg.astype(np.float32)
    if float(np.linalg.norm(fallback_dir)) <= 1e-8:
        fallback_dir = np.asarray([1.0, 0.0], dtype=np.float32)

    for _ in range(max_iters):
        max_pen = 0.0
        best_push = None
        for center, radius in zip(pts, radii):
            min_dist = float(pusher_r) + float(radius)
            delta = point - center
            dist = float(np.linalg.norm(delta))
            pen = min_dist - dist
            if pen <= 0.0:
                continue
            if dist > 1e-8:
                normal = (delta / dist).astype(np.float32)
            else:
                dir_norm = float(np.linalg.norm(fallback_dir))
                if dir_norm > 1e-8:
                    normal = (fallback_dir / dir_norm).astype(np.float32)
                else:
                    normal = np.asarray([1.0, 0.0], dtype=np.float32)
            if pen > max_pen:
                max_pen = float(pen)
                best_push = normal * float(pen + eps)
        if best_push is None:
            break
        point = (point + best_push).astype(np.float32)
        fallback_dir = best_push.astype(np.float32)

    return point.astype(np.float32)


def _resolve_particle_contact(
    pusher_start: Sequence[float],
    pusher_end: Sequence[float],
    pusher_r: float,
    particle_xy: np.ndarray,
    particle_r: float | Sequence[float] | np.ndarray,
) -> dict[str, object]:
    pts = np.asarray(particle_xy, dtype=np.float32).reshape(-1, 2)
    radii = _normalize_particle_radii(pts, particle_r)

    best_hit: Optional[dict[str, object]] = None
    for center, radius in zip(pts, radii):
        hit = _swept_particle_contact(
            pusher_start,
            pusher_end,
            center,
            pusher_r=float(pusher_r),
            pr=float(radius),
        )
        if not bool(hit["hit"]):
            continue
        if best_hit is None or float(hit["toi"]) < float(best_hit["toi"]):
            best_hit = hit

    projected_xy = _resolve_particle_union_endpoint(
        pusher_start,
        pusher_end,
        pts,
        pusher_r=float(pusher_r),
        pr=radii,
    )

    result: dict[str, object] = {
        "hit": bool(best_hit is not None),
        "collided": bool(best_hit is not None),
        "toi": float(best_hit["toi"]) if best_hit is not None else 1.0,
        "projected_xy": projected_xy.astype(np.float32),
        "resolved_xy": projected_xy.astype(np.float32),
        "pusher_end": projected_xy.astype(np.float32),
    }
    if best_hit is not None:
        result["closest_xy"] = np.asarray(best_hit["closest_xy"], dtype=np.float32)
        result["normal_xy"] = np.asarray(best_hit["normal_xy"], dtype=np.float32)
        result["penetration"] = float(best_hit["penetration"])
        result["min_dist"] = float(best_hit["min_dist"])
    return result


def _fit_contact_rigid_delta(
    points_xy: np.ndarray,
    delta_xy: np.ndarray,
    weights: np.ndarray,
    center_xy: Sequence[float],
    inertia: float = 1.0,
    ridge: float = 1e-8,
) -> dict[str, float]:
    del ridge
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    delta = np.asarray(delta_xy, dtype=np.float32).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    center = np.asarray(center_xy, dtype=np.float32).reshape(2)
    if pts.shape[0] != delta.shape[0] or pts.shape[0] != w.shape[0]:
        raise ValueError("points_xy, delta_xy, and weights must have matching leading dimensions.")

    mask = np.isfinite(w) & (w > 1e-8)
    mask &= np.all(np.isfinite(pts), axis=1)
    mask &= np.all(np.isfinite(delta), axis=1)
    if not np.any(mask):
        return {
            "tx": 0.0,
            "ty": 0.0,
            "dtheta": 0.0,
            "raw_dtheta": 0.0,
            "sum_w": 0.0,
            "num_contacts": 0.0,
            "max_disp": 0.0,
            "r_max": 0.0,
            "contact_x": 0.0,
            "contact_y": 0.0,
            "delta_x": 0.0,
            "delta_y": 0.0,
            "lever_x": 0.0,
            "lever_y": 0.0,
            "inertia": float(inertia),
        }

    pts = pts[mask]
    delta = delta[mask]
    w = w[mask]

    r = pts - center[None, :]
    w64 = w.astype(np.float64)

    sum_w = float(np.sum(w64))
    if sum_w <= 1e-8:
        return {
            "tx": 0.0,
            "ty": 0.0,
            "dtheta": 0.0,
            "raw_dtheta": 0.0,
            "sum_w": 0.0,
            "num_contacts": 0.0,
            "max_disp": 0.0,
            "r_max": 0.0,
            "contact_x": 0.0,
            "contact_y": 0.0,
            "delta_x": 0.0,
            "delta_y": 0.0,
            "lever_x": 0.0,
            "lever_y": 0.0,
            "inertia": float(inertia),
        }

    max_disp = float(np.linalg.norm(delta, axis=1).max(initial=0.0))
    r_max = float(np.linalg.norm(r, axis=1).max(initial=0.0))

    contact_xy = np.sum(pts.astype(np.float64) * w64[:, None], axis=0) / max(sum_w, 1e-8)
    delta_mean = np.sum(delta.astype(np.float64) * w64[:, None], axis=0) / max(sum_w, 1e-8)
    lever = contact_xy - center.astype(np.float64)
    tx = float(delta_mean[0])
    ty = float(delta_mean[1])
    inertia = float(inertia)
    if np.isfinite(inertia) and inertia > 1e-8:
        raw_dtheta = float((lever[0] * delta_mean[1] - lever[1] * delta_mean[0]) / inertia)
    else:
        raw_dtheta = 0.0
    dtheta = raw_dtheta

    tmag = float(np.hypot(tx, ty))
    if max_disp > 0.0 and np.isfinite(tmag) and tmag > max_disp:
        scale = max_disp / max(tmag, 1e-8)
        tx *= scale
        ty *= scale
    elif not np.isfinite(tmag):
        tx = 0.0
        ty = 0.0

    if not np.isfinite(dtheta):
        dtheta = 0.0
    if max_disp > 0.0:
        theta_cap = max_disp / max(r_max, 1e-6)
        dtheta = float(np.clip(dtheta, -theta_cap, theta_cap))
    else:
        dtheta = 0.0

    return {
        "tx": float(tx),
        "ty": float(ty),
        "dtheta": float(dtheta),
        "raw_dtheta": float(raw_dtheta),
        "sum_w": float(sum_w),
        "num_contacts": float(pts.shape[0]),
        "max_disp": float(max_disp),
        "r_max": float(r_max),
        "contact_x": float(contact_xy[0]),
        "contact_y": float(contact_xy[1]),
        "delta_x": float(delta_mean[0]),
        "delta_y": float(delta_mean[1]),
        "lever_x": float(lever[0]),
        "lever_y": float(lever[1]),
        "inertia": float(inertia),
    }


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
    pose_offset_local: Optional[np.ndarray] = None

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
    def _pusher_swept_contact(
        x_prev: wp.array(dtype=wp.vec3),
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        pusher_start: wp.vec3,
        pusher_end: wp.vec3,
        pusher_r: float,
        pr: float,
        mu: float,
        contact_alpha: float,
        contact_delta: wp.array(dtype=wp.vec3),
        contact_weight: wp.array(dtype=float),
    ):
        i = wp.tid()
        contact_delta[i] = wp.vec3(0.0, 0.0, 0.0)
        contact_weight[i] = 0.0
        if inv_m[i] == 0.0:
            return

        xi = x_pred[i]
        seg = pusher_end - pusher_start
        seg_len2 = wp.dot(seg, seg)
        closest = pusher_start
        if seg_len2 > 1e-8:
            t = wp.clamp(wp.dot(xi - pusher_start, seg) / seg_len2, 0.0, 1.0)
            closest = pusher_start + seg * t

        d = xi - closest
        dist = wp.length(d)
        min_dist = pusher_r + pr

        if dist < min_dist:
            n = wp.vec3(1.0, 0.0, 0.0)
            if dist > 1e-8:
                n = d / dist
            elif seg_len2 > 1e-8:
                seg_len = wp.sqrt(seg_len2)
                n = wp.vec3(-seg[1] / seg_len, seg[0] / seg_len, 0.0)
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

            contact_delta[i] = wp.vec3(xi[0] - x_pred[i][0], xi[1] - x_pred[i][1], 0.0)
            contact_weight[i] = raw_pen


    @wp.kernel
    def _apply_rigid_transform(
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        center: wp.vec3,
        tx: float,
        ty: float,
        dtheta: float,
    ):
        i = wp.tid()
        if inv_m[i] == 0.0:
            return

        xi = x_pred[i]
        rel = xi - center
        ct = wp.cos(dtheta)
        st = wp.sin(dtheta)
        rx = ct * rel[0] - st * rel[1]
        ry = st * rel[0] + ct * rel[1]
        x_pred[i] = wp.vec3(center[0] + tx + rx, center[1] + ty + ry, 0.0)


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


    @wp.kernel
    def _batch_predict_positions(
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        x_pred: wp.array(dtype=wp.vec3),
        dt: float,
        lin_damp: float,
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            x_pred[i] = wp.vec3(x[i][0], x[i][1], 0.0)
            return

        local = i - lane * num_particles
        if inv_m[local] == 0.0:
            x_pred[i] = wp.vec3(x[i][0], x[i][1], 0.0)
            v[i] = wp.vec3(0.0, 0.0, 0.0)
            return

        vi = v[i] * lin_damp
        xi = x[i] + dt * vi
        x_pred[i] = wp.vec3(xi[0], xi[1], 0.0)
        v[i] = wp.vec3(vi[0], vi[1], 0.0)


    @wp.kernel
    def _batch_project_walls_aabb(
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        pr: float,
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            return

        local = i - lane * num_particles
        if inv_m[local] == 0.0:
            return
        xi = x_pred[i]
        xw = wp.clamp(xi[0], xmin + pr, xmax - pr)
        yw = wp.clamp(xi[1], ymin + pr, ymax - pr)
        x_pred[i] = wp.vec3(xw, yw, 0.0)


    @wp.kernel
    def _batch_pusher_swept_contact(
        x_prev: wp.array(dtype=wp.vec3),
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        pusher_start: wp.array(dtype=wp.vec3),
        pusher_end: wp.array(dtype=wp.vec3),
        pusher_r: float,
        pr: float,
        mu: float,
        contact_alpha: float,
        contact_delta: wp.array(dtype=wp.vec3),
        contact_weight: wp.array(dtype=float),
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        contact_delta[i] = wp.vec3(0.0, 0.0, 0.0)
        contact_weight[i] = 0.0
        if active[lane] == 0:
            return

        local = i - lane * num_particles
        if inv_m[local] == 0.0:
            return

        xi = x_pred[i]
        seg = pusher_end[lane] - pusher_start[lane]
        seg_len2 = wp.dot(seg, seg)
        closest = pusher_start[lane]
        if seg_len2 > 1e-8:
            t = wp.clamp(wp.dot(xi - pusher_start[lane], seg) / seg_len2, 0.0, 1.0)
            closest = pusher_start[lane] + seg * t

        d = xi - closest
        dist = wp.length(d)
        min_dist = pusher_r + pr

        if dist < min_dist:
            n = wp.vec3(1.0, 0.0, 0.0)
            if dist > 1e-8:
                n = d / dist
            elif seg_len2 > 1e-8:
                seg_len = wp.sqrt(seg_len2)
                n = wp.vec3(-seg[1] / seg_len, seg[0] / seg_len, 0.0)
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

            contact_delta[i] = wp.vec3(xi[0] - x_pred[i][0], xi[1] - x_pred[i][1], 0.0)
            contact_weight[i] = raw_pen


    @wp.kernel
    def _batch_apply_rigid_transform(
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        center: wp.array(dtype=wp.vec3),
        tx: wp.array(dtype=float),
        ty: wp.array(dtype=float),
        dtheta: wp.array(dtype=float),
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            return

        local = i - lane * num_particles
        if inv_m[local] == 0.0:
            return

        xi = x_pred[i]
        rel = xi - center[lane]
        ct = wp.cos(dtheta[lane])
        st = wp.sin(dtheta[lane])
        rx = ct * rel[0] - st * rel[1]
        ry = st * rel[0] + ct * rel[1]
        x_pred[i] = wp.vec3(center[lane][0] + tx[lane] + rx, center[lane][1] + ty[lane] + ry, 0.0)


    @wp.kernel
    def _batch_accum_com(
        x_pred: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        sum_m: wp.array(dtype=float),
        sum_x: wp.array(dtype=float),
        sum_y: wp.array(dtype=float),
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            return

        local = i - lane * num_particles
        invmi = inv_m[local]
        if invmi == 0.0:
            return
        mi = 1.0 / invmi
        xi = x_pred[i]
        wp.atomic_add(sum_m, lane, mi)
        wp.atomic_add(sum_x, lane, mi * xi[0])
        wp.atomic_add(sum_y, lane, mi * xi[1])


    @wp.kernel
    def _batch_accum_ab(
        x_pred: wp.array(dtype=wp.vec3),
        r0: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        center: wp.array(dtype=wp.vec3),
        sum_a: wp.array(dtype=float),
        sum_b: wp.array(dtype=float),
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            return

        local = i - lane * num_particles
        invmi = inv_m[local]
        if invmi == 0.0:
            return
        mi = 1.0 / invmi

        ri0 = r0[local]
        ri = x_pred[i] - center[lane]

        a = ri0[0] * ri[1] - ri0[1] * ri[0]
        b = ri0[0] * ri[0] + ri0[1] * ri[1]

        wp.atomic_add(sum_a, lane, mi * a)
        wp.atomic_add(sum_b, lane, mi * b)


    @wp.kernel
    def _batch_shape_match_project(
        x_pred: wp.array(dtype=wp.vec3),
        r0: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        center: wp.array(dtype=wp.vec3),
        theta: wp.array(dtype=float),
        alpha: float,
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            return

        local = i - lane * num_particles
        if inv_m[local] == 0.0:
            return

        ct = wp.cos(theta[lane])
        st = wp.sin(theta[lane])
        ri0 = r0[local]

        rx = ct * ri0[0] - st * ri0[1]
        ry = st * ri0[0] + ct * ri0[1]
        target = wp.vec3(center[lane][0] + rx, center[lane][1] + ry, 0.0)

        xi = x_pred[i]
        x_pred[i] = xi + alpha * (target - xi)


    @wp.kernel
    def _batch_finalize(
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        x_pred: wp.array(dtype=wp.vec3),
        dt: float,
        vel_damp: float,
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            return

        local = i - lane * num_particles
        if inv_m[local] == 0.0:
            return
        vi = (x_pred[i] - x[i]) / dt
        v[i] = wp.vec3(vi[0], vi[1], 0.0) * vel_damp
        x[i] = wp.vec3(x_pred[i][0], x_pred[i][1], 0.0)


    @wp.kernel
    def _batch_apply_ground_friction(
        v: wp.array(dtype=wp.vec3),
        inv_m: wp.array(dtype=float),
        dt: float,
        friction_accel: float,
        rest_speed_eps: float,
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            return

        local = i - lane * num_particles
        if inv_m[local] == 0.0:
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


    @wp.kernel
    def _batch_accum_contact_fit(
        x_pred: wp.array(dtype=wp.vec3),
        contact_delta: wp.array(dtype=wp.vec3),
        contact_weight: wp.array(dtype=float),
        center: wp.array(dtype=wp.vec3),
        sum_w: wp.array(dtype=float),
        weighted_px: wp.array(dtype=float),
        weighted_py: wp.array(dtype=float),
        weighted_dx: wp.array(dtype=float),
        weighted_dy: wp.array(dtype=float),
        max_disp_sq: wp.array(dtype=float),
        r_max_sq: wp.array(dtype=float),
        num_particles: int,
        active: wp.array(dtype=int),
    ):
        i = wp.tid()
        lane = i // num_particles
        if active[lane] == 0:
            return

        wi = contact_weight[i]
        if wi <= 1e-8:
            return

        xi = x_pred[i]
        di = contact_delta[i]
        rel = xi - center[lane]
        disp_sq = di[0] * di[0] + di[1] * di[1]
        r_sq = rel[0] * rel[0] + rel[1] * rel[1]

        wp.atomic_add(sum_w, lane, wi)
        wp.atomic_add(weighted_px, lane, wi * xi[0])
        wp.atomic_add(weighted_py, lane, wi * xi[1])
        wp.atomic_add(weighted_dx, lane, wi * di[0])
        wp.atomic_add(weighted_dy, lane, wi * di[1])
        wp.atomic_max(max_disp_sq, lane, disp_sq)
        wp.atomic_max(r_max_sq, lane, r_sq)


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
        if p.pose_offset_local is None:
            self.pose_offset_local = np.zeros((3,), dtype=np.float32)
        else:
            self.pose_offset_local = np.asarray(p.pose_offset_local, dtype=np.float32).reshape(3).copy()
        self.gt_body_com_local = _gt_tee_body_com_local(
            stem_w=self.stem_w,
            stem_h=self.stem_h,
            bar_w=self.bar_w,
            bar_h=self.bar_h,
        )
        self.body_com_from_cloud_local = (
            self.gt_body_com_local - self.pose_offset_local
        ).astype(np.float32)
        self.gt_body_inertia = float(
            _gt_tee_body_inertia(
                stem_w=self.stem_w,
                stem_h=self.stem_h,
                bar_w=self.bar_w,
                bar_h=self.bar_h,
            )
        )

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
        self.contact_delta = wp.array(np.zeros((self.N, 3), dtype=np.float32), dtype=wp.vec3, device=self.device)
        self.contact_weight = wp.array(np.zeros((self.N,), dtype=np.float32), dtype=float, device=self.device)

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
        resolved_path = self._simulate_frame(pusher_path=pusher_path)
        self.pusher_pos = resolved_path[-1].copy()
        if resolved_path.shape[0] >= 2:
            realized_velocity = (resolved_path[-1, :2] - resolved_path[-2, :2]) / max(self.sim_dt, 1e-8)
            if np.all(np.isfinite(realized_velocity)):
                self.pusher_velocity = realized_velocity.astype(np.float32)
            else:
                self.pusher_velocity = final_velocity.copy()
        else:
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

    def _resolve_endpoint_against_particles(
        self,
        prev_world: Sequence[float],
        target_world: Sequence[float],
        particle_xy: np.ndarray,
    ) -> np.ndarray:
        return _resolve_particle_union_endpoint(
            prev_world,
            target_world,
            particle_xy,
            pusher_r=float(self.pusher_r),
            pr=float(self.pr),
        )

    def _rotate_local_xy(self, local_xy: Sequence[float], theta: float) -> np.ndarray:
        local = np.asarray(local_xy, dtype=np.float32).reshape(2)
        c = math.cos(float(theta))
        s = math.sin(float(theta))
        return np.asarray(
            [
                c * float(local[0]) - s * float(local[1]),
                s * float(local[0]) + c * float(local[1]),
            ],
            dtype=np.float32,
        )

    def _body_com_from_cloud_local_xy(self) -> np.ndarray:
        if hasattr(self, "body_com_from_cloud_local"):
            return np.asarray(self.body_com_from_cloud_local[:2], dtype=np.float32).copy()
        pose_offset = np.asarray(
            getattr(self, "pose_offset_local", np.zeros((3,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3)
        gt_body_com = _gt_tee_body_com_local(
            stem_w=float(getattr(self, "stem_w", GT_T_STEM_W)),
            stem_h=float(getattr(self, "stem_h", GT_T_STEM_H)),
            bar_w=float(getattr(self, "bar_w", GT_T_BAR_W)),
            bar_h=float(getattr(self, "bar_h", GT_T_BAR_H)),
        )
        return (gt_body_com[:2] - pose_offset[:2]).astype(np.float32)

    def _gt_body_inertia_value(self) -> float:
        if hasattr(self, "gt_body_inertia"):
            return float(self.gt_body_inertia)
        return float(
            _gt_tee_body_inertia(
                stem_w=float(getattr(self, "stem_w", GT_T_STEM_W)),
                stem_h=float(getattr(self, "stem_h", GT_T_STEM_H)),
                bar_w=float(getattr(self, "bar_w", GT_T_BAR_W)),
                bar_h=float(getattr(self, "bar_h", GT_T_BAR_H)),
            )
        )

    def _body_com_world_from_cloud_pose(self, cloud_center: object, theta: float) -> np.ndarray:
        cloud_xy = np.asarray([float(cloud_center[0]), float(cloud_center[1])], dtype=np.float32)
        return (cloud_xy + self._rotate_local_xy(self._body_com_from_cloud_local_xy(), theta)).astype(np.float32)

    def _cloud_center_world_from_body_com(self, body_com_world: Sequence[float], theta: float) -> np.ndarray:
        body_com_xy = np.asarray(body_com_world, dtype=np.float32).reshape(2)
        return (body_com_xy - self._rotate_local_xy(self._body_com_from_cloud_local_xy(), theta)).astype(np.float32)

    def _solve_contact_rigid_delta(self, body_com_world: Sequence[float]) -> tuple[float, float, float]:
        if not (
            hasattr(getattr(self, "x_pred", None), "numpy")
            and hasattr(getattr(self, "contact_delta", None), "numpy")
            and hasattr(getattr(self, "contact_weight", None), "numpy")
        ):
            return 0.0, 0.0, 0.0

        center_xy = np.asarray(body_com_world, dtype=np.float32).reshape(2)
        points_xy = self.x_pred.numpy().astype(np.float32)[:, :2]
        delta_xy = self.contact_delta.numpy().astype(np.float32)[:, :2]
        weights = self.contact_weight.numpy().astype(np.float32).reshape(-1)
        fit = _fit_contact_rigid_delta(
            points_xy=points_xy,
            delta_xy=delta_xy,
            weights=weights,
            center_xy=center_xy,
            inertia=self._gt_body_inertia_value(),
        )
        return float(fit["tx"]), float(fit["ty"]), float(fit["dtheta"])

    def _simulate_frame(
        self,
        pusher_path: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if pusher_path is None:
            pusher_path = np.asarray([self.pusher_pos.copy(), self.pusher_pos.copy()], dtype=np.float32)
        else:
            pusher_path = np.asarray(pusher_path, dtype=np.float32).reshape(-1, 3)
        if pusher_path.shape[0] < 2:
            raise ValueError("pusher_path must contain at least two points.")
        num_segments = int(pusher_path.shape[0] - 1)
        dt = self.control_dt / float(num_segments)
        resolved_path = pusher_path.copy()
        if hasattr(self, "pusher_pos"):
            resolved_path[0, :2] = np.asarray(self.pusher_pos[:2], dtype=np.float32)

        # Replay the PD controller path at its native controller cadence instead
        # of resampling it at an unrelated substep count. This keeps particle
        # contact timing aligned with GT PushT's 100 Hz integration.
        for seg_idx in range(num_segments):
            prev_pxy = resolved_path[seg_idx, :2].astype(np.float32, copy=False)
            commanded_pxy = pusher_path[seg_idx + 1, :2].astype(np.float32, copy=False)
            resolved_pxy = commanded_pxy.copy()
            pstart = wp.vec3(float(prev_pxy[0]), float(prev_pxy[1]), 0.0)
            pend = wp.vec3(float(commanded_pxy[0]), float(commanded_pxy[1]), 0.0)
            contact_delta = getattr(self, "contact_delta", None)
            contact_weight = getattr(self, "contact_weight", None)

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
                    kernel=_pusher_swept_contact,
                    dim=self.N,
                    inputs=[
                        self.x,
                        self.x_pred,
                        self.inv_m,
                        pstart,
                        pend,
                        self.pusher_r,
                        self.pr,
                        self.mu,
                        self.contact_alpha,
                        contact_delta,
                        contact_weight,
                    ],
                    device=self.device,
                )

                c, theta = self._compute_com_and_theta()
                body_com_world = self._body_com_world_from_cloud_pose(c, theta)
                tx, ty, dtheta = self._solve_contact_rigid_delta(body_com_world)
                body_com_wp = wp.vec3(float(body_com_world[0]), float(body_com_world[1]), 0.0)

                wp.launch(
                    kernel=_apply_rigid_transform,
                    dim=self.N,
                    inputs=[self.x_pred, self.inv_m, body_com_wp, tx, ty, dtheta],
                    device=self.device,
                )

                theta_target = float(_wrap_pi(float(theta) + float(dtheta)))
                body_com_after = np.asarray(
                    [float(body_com_world[0]) + tx, float(body_com_world[1]) + ty],
                    dtype=np.float32,
                )
                c_target_xy = self._cloud_center_world_from_body_com(body_com_after, theta_target)
                c_target = wp.vec3(float(c_target_xy[0]), float(c_target_xy[1]), 0.0)
                wp.launch(
                    kernel=_shape_match_project,
                    dim=self.N,
                    inputs=[self.x_pred, self.r0, self.inv_m, c_target, theta_target, self.alpha_rigid],
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

            if hasattr(getattr(self, "x", None), "numpy"):
                particle_xy = self.get_particle_positions().astype(np.float32)
                resolved_pxy = self._resolve_endpoint_against_particles(
                    prev_world=prev_pxy,
                    target_world=commanded_pxy,
                    particle_xy=particle_xy,
                ).astype(np.float32)

            resolved_path[seg_idx + 1, :2] = resolved_pxy.astype(np.float32)

        return resolved_path.astype(np.float32)

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


class PushTWarpBatchEnv:
    def __init__(
        self,
        device: str = "cuda:0",
        params: Optional[PushTWarpParams] = None,
        batch_size: int = 1,
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
        if p.pose_offset_local is None:
            self.pose_offset_local = np.zeros((3,), dtype=np.float32)
        else:
            self.pose_offset_local = np.asarray(p.pose_offset_local, dtype=np.float32).reshape(3).copy()
        self.gt_body_com_local = _gt_tee_body_com_local(
            stem_w=self.stem_w,
            stem_h=self.stem_h,
            bar_w=self.bar_w,
            bar_h=self.bar_h,
        )
        self.body_com_from_cloud_local = (
            self.gt_body_com_local - self.pose_offset_local
        ).astype(np.float32)
        self.gt_body_inertia = float(
            _gt_tee_body_inertia(
                stem_w=self.stem_w,
                stem_h=self.stem_h,
                bar_w=self.bar_w,
                bar_h=self.bar_h,
            )
        )

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
        self.batch_size = max(1, int(batch_size))
        self.total_particles = int(self.batch_size * self.N)

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
        self._r0_host = r0.copy()
        self._inv_m_host = np.full((self.N,), 1.0, dtype=np.float32)

        self.x = wp.zeros(shape=self.total_particles, dtype=wp.vec3, device=self.device)
        self.v = wp.zeros(shape=self.total_particles, dtype=wp.vec3, device=self.device)
        self.x_pred = wp.zeros(shape=self.total_particles, dtype=wp.vec3, device=self.device)
        self.contact_delta = wp.zeros(shape=self.total_particles, dtype=wp.vec3, device=self.device)
        self.contact_weight = wp.zeros(shape=self.total_particles, dtype=float, device=self.device)

        self.inv_m = wp.array(self._inv_m_host, dtype=float, device=self.device)
        self.r0 = wp.array(self._r0_host, dtype=wp.vec3, device=self.device)

        self.active = wp.zeros(shape=self.batch_size, dtype=int, device=self.device)
        self.all_active = wp.full(shape=self.batch_size, value=1, dtype=int, device=self.device)
        self.sum_m = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.sum_x = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.sum_y = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.sum_a = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.sum_b = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.sum_w = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.weighted_px = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.weighted_py = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.weighted_dx = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.weighted_dy = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.max_disp_sq = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.r_max_sq = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)

        self.pusher_start = wp.zeros(shape=self.batch_size, dtype=wp.vec3, device=self.device)
        self.pusher_end = wp.zeros(shape=self.batch_size, dtype=wp.vec3, device=self.device)
        self.cloud_center = wp.zeros(shape=self.batch_size, dtype=wp.vec3, device=self.device)
        self.body_com_center = wp.zeros(shape=self.batch_size, dtype=wp.vec3, device=self.device)
        self.shape_match_center = wp.zeros(shape=self.batch_size, dtype=wp.vec3, device=self.device)
        self.tx = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.ty = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.dtheta = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)
        self.theta_target = wp.zeros(shape=self.batch_size, dtype=float, device=self.device)

        self.pusher_pos = np.zeros((self.batch_size, 3), dtype=np.float32)
        self.pusher_velocity = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.goal_pose = np.zeros((self.batch_size, 3), dtype=np.float32)
        self._theta_state = np.zeros((self.batch_size,), dtype=np.float32)
        self._last_pose = np.zeros((self.batch_size, 3), dtype=np.float32)
        self._active_host = np.ones((self.batch_size,), dtype=np.int32)

        self._update_active(np.ones((self.batch_size,), dtype=np.int32))
        default_obj_xy = np.tile(np.asarray([[-0.08, -0.02]], dtype=np.float32), (self.batch_size, 1))
        default_obj_pose = np.concatenate(
            [default_obj_xy, np.full((self.batch_size, 1), 0.6, dtype=np.float32)],
            axis=1,
        )
        default_pusher_xy = np.tile(np.asarray([[-0.12, -0.18]], dtype=np.float32), (self.batch_size, 1))
        default_goal_pose = np.tile(np.asarray([[0.10, 0.10, 0.0]], dtype=np.float32), (self.batch_size, 1))
        self.set_state_batch(
            pusher_xy=default_pusher_xy,
            obj_pose=default_obj_pose,
            goal_pose=default_goal_pose,
            active_mask=np.ones((self.batch_size,), dtype=np.int32),
        )

    @property
    def num_particles(self) -> int:
        return int(self.N)

    def rest_offsets(self) -> np.ndarray:
        return self._r0_host.copy()

    def _update_active(self, active_mask: np.ndarray) -> None:
        active_arr = np.asarray(active_mask, dtype=np.int32).reshape(-1)
        if active_arr.shape[0] != self.batch_size:
            raise ValueError(
                f"active_mask must have length {self.batch_size}, got {active_arr.shape[0]}"
            )
        self._active_host = active_arr.copy()
        self.active.assign(self._active_host)

    def _body_com_world_from_cloud_pose_batch(
        self,
        cloud_center_xy: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        centers = np.asarray(cloud_center_xy, dtype=np.float32).reshape(self.batch_size, 2)
        theta_arr = np.asarray(theta, dtype=np.float32).reshape(self.batch_size)
        offset = np.asarray(self.body_com_from_cloud_local[:2], dtype=np.float32).reshape(1, 2)
        ct = np.cos(theta_arr).reshape(-1, 1).astype(np.float32)
        st = np.sin(theta_arr).reshape(-1, 1).astype(np.float32)
        rot = np.concatenate(
            [
                ct * offset[:, :1] - st * offset[:, 1:2],
                st * offset[:, :1] + ct * offset[:, 1:2],
            ],
            axis=1,
        ).astype(np.float32)
        return (centers + rot).astype(np.float32)

    def _cloud_center_world_from_body_com_batch(
        self,
        body_com_xy: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        centers = np.asarray(body_com_xy, dtype=np.float32).reshape(self.batch_size, 2)
        theta_arr = np.asarray(theta, dtype=np.float32).reshape(self.batch_size)
        offset = np.asarray(self.body_com_from_cloud_local[:2], dtype=np.float32).reshape(1, 2)
        ct = np.cos(theta_arr).reshape(-1, 1).astype(np.float32)
        st = np.sin(theta_arr).reshape(-1, 1).astype(np.float32)
        rot = np.concatenate(
            [
                ct * offset[:, :1] - st * offset[:, 1:2],
                st * offset[:, :1] + ct * offset[:, 1:2],
            ],
            axis=1,
        ).astype(np.float32)
        return (centers - rot).astype(np.float32)

    def _build_particle_state_batch(
        self,
        obj_pose: np.ndarray,
        obj_twist: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(obj_pose, dtype=np.float32).reshape(self.batch_size, 3)
        theta = pose[:, 2].astype(np.float32)
        ct = np.cos(theta).reshape(-1, 1).astype(np.float32)
        st = np.sin(theta).reshape(-1, 1).astype(np.float32)

        r = self._r0_host[None, :, :2]
        rx = ct * r[:, :, 0] - st * r[:, :, 1]
        ry = st * r[:, :, 0] + ct * r[:, :, 1]

        x_host = np.zeros((self.batch_size, self.N, 3), dtype=np.float32)
        x_host[:, :, 0] = rx + pose[:, None, 0]
        x_host[:, :, 1] = ry + pose[:, None, 1]

        v_host = np.zeros_like(x_host)
        if obj_twist is not None:
            twist = np.asarray(obj_twist, dtype=np.float32).reshape(self.batch_size, 3)
            center = x_host[:, :, :2].mean(axis=1, keepdims=True)
            rel = x_host[:, :, :2] - center
            perp = np.stack([-rel[:, :, 1], rel[:, :, 0]], axis=2).astype(np.float32)
            v_host[:, :, :2] = twist[:, None, :2] + twist[:, None, 2:3] * perp
        return x_host.reshape(self.total_particles, 3), v_host.reshape(self.total_particles, 3)

    def set_state_batch(
        self,
        *,
        pusher_xy: np.ndarray,
        obj_pose: np.ndarray,
        goal_pose: Optional[np.ndarray] = None,
        obj_twist: Optional[np.ndarray] = None,
        pusher_velocity: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
    ) -> None:
        pusher_xy_arr = np.asarray(pusher_xy, dtype=np.float32).reshape(self.batch_size, 2)
        obj_pose_arr = np.asarray(obj_pose, dtype=np.float32).reshape(self.batch_size, 3)
        if goal_pose is None:
            goal_pose_arr = self.goal_pose.copy()
        else:
            goal_pose_arr = np.asarray(goal_pose, dtype=np.float32).reshape(self.batch_size, 3)
        if pusher_velocity is None:
            pusher_velocity_arr = np.zeros((self.batch_size, 2), dtype=np.float32)
        else:
            pusher_velocity_arr = np.asarray(pusher_velocity, dtype=np.float32).reshape(self.batch_size, 2)

        x_host, v_host = self._build_particle_state_batch(obj_pose_arr, obj_twist=obj_twist)
        self.x.assign(x_host)
        self.v.assign(v_host)
        self.x_pred.assign(x_host)
        self.contact_delta.zero_()
        self.contact_weight.zero_()

        self.pusher_pos[:, :2] = pusher_xy_arr
        self.pusher_pos[:, 2] = 0.0
        self.pusher_velocity[:, :] = pusher_velocity_arr
        self.goal_pose[:, :] = goal_pose_arr
        self._theta_state[:] = obj_pose_arr[:, 2]
        self._last_pose[:, :] = obj_pose_arr
        if active_mask is None:
            self._update_active(np.ones((self.batch_size,), dtype=np.int32))
        else:
            self._update_active(np.asarray(active_mask, dtype=np.int32).reshape(self.batch_size))

    def _compute_com_and_theta_batch(self) -> np.ndarray:
        self.sum_m.zero_()
        self.sum_x.zero_()
        self.sum_y.zero_()
        wp.launch(
            kernel=_batch_accum_com,
            dim=self.total_particles,
            inputs=[self.x_pred, self.inv_m, self.sum_m, self.sum_x, self.sum_y, self.N, self.all_active],
            device=self.device,
        )

        sum_m = self.sum_m.numpy().astype(np.float32)
        sum_x = self.sum_x.numpy().astype(np.float32)
        sum_y = self.sum_y.numpy().astype(np.float32)

        center = np.zeros((self.batch_size, 3), dtype=np.float32)
        valid = sum_m > 1e-8
        center[valid, 0] = sum_x[valid] / np.maximum(sum_m[valid], 1e-8)
        center[valid, 1] = sum_y[valid] / np.maximum(sum_m[valid], 1e-8)
        self.cloud_center.assign(center)

        self.sum_a.zero_()
        self.sum_b.zero_()
        wp.launch(
            kernel=_batch_accum_ab,
            dim=self.total_particles,
            inputs=[self.x_pred, self.r0, self.inv_m, self.cloud_center, self.sum_a, self.sum_b, self.N, self.all_active],
            device=self.device,
        )

        sum_a = self.sum_a.numpy().astype(np.float32)
        sum_b = self.sum_b.numpy().astype(np.float32)
        theta = self._theta_state.astype(np.float32).copy()
        if self.N > 1:
            informative = valid & ((np.abs(sum_a) + np.abs(sum_b)) > 1e-12)
            theta[informative] = np.arctan2(sum_a[informative], sum_b[informative]).astype(np.float32)
            theta[informative] = np.asarray([_wrap_pi(float(v)) for v in theta[informative]], dtype=np.float32)
            self._theta_state[:] = theta

        pose = np.zeros((self.batch_size, 3), dtype=np.float32)
        pose[:, :2] = center[:, :2]
        pose[:, 2] = theta
        return pose

    def get_object_pose_batch(self) -> np.ndarray:
        return self._compute_com_and_theta_batch().astype(np.float32)

    def get_object_twist_batch(self) -> np.ndarray:
        x_host = self.x.numpy().astype(np.float32).reshape(self.batch_size, self.N, 3)[:, :, :2]
        v_host = self.v.numpy().astype(np.float32).reshape(self.batch_size, self.N, 3)[:, :, :2]
        if self.N <= 0:
            return np.zeros((self.batch_size, 3), dtype=np.float32)

        v_com = v_host.mean(axis=1)
        center = x_host.mean(axis=1, keepdims=True)
        rel = x_host - center
        perp = np.stack([-rel[:, :, 1], rel[:, :, 0]], axis=2).astype(np.float32)
        num = np.sum(perp * (v_host - v_com[:, None, :]), axis=(1, 2), dtype=np.float32)
        den = np.sum(perp * perp, axis=(1, 2), dtype=np.float32) + 1e-9
        omega = (num / den).astype(np.float32)
        return np.concatenate([v_com.astype(np.float32), omega[:, None]], axis=1)

    def get_particle_positions_batch(self) -> np.ndarray:
        return self.x.numpy().astype(np.float32).reshape(self.batch_size, self.N, 3)[:, :, :2].copy()

    def get_particle_velocities_batch(self) -> np.ndarray:
        return self.v.numpy().astype(np.float32).reshape(self.batch_size, self.N, 3)[:, :, :2].copy()

    def get_pusher_velocity_batch(self) -> np.ndarray:
        return self.pusher_velocity.astype(np.float32).copy()

    def _compute_contact_fit_batch(self, body_com_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        body_com = np.zeros((self.batch_size, 3), dtype=np.float32)
        body_com[:, :2] = np.asarray(body_com_world, dtype=np.float32).reshape(self.batch_size, 2)
        self.body_com_center.assign(body_com)

        self.sum_w.zero_()
        self.weighted_px.zero_()
        self.weighted_py.zero_()
        self.weighted_dx.zero_()
        self.weighted_dy.zero_()
        self.max_disp_sq.zero_()
        self.r_max_sq.zero_()
        wp.launch(
            kernel=_batch_accum_contact_fit,
            dim=self.total_particles,
            inputs=[
                self.x_pred,
                self.contact_delta,
                self.contact_weight,
                self.body_com_center,
                self.sum_w,
                self.weighted_px,
                self.weighted_py,
                self.weighted_dx,
                self.weighted_dy,
                self.max_disp_sq,
                self.r_max_sq,
                self.N,
                self.active,
            ],
            device=self.device,
        )

        sum_w = self.sum_w.numpy().astype(np.float32)
        weighted_px = self.weighted_px.numpy().astype(np.float32)
        weighted_py = self.weighted_py.numpy().astype(np.float32)
        weighted_dx = self.weighted_dx.numpy().astype(np.float32)
        weighted_dy = self.weighted_dy.numpy().astype(np.float32)
        max_disp = np.sqrt(np.maximum(self.max_disp_sq.numpy().astype(np.float32), 0.0))
        r_max = np.sqrt(np.maximum(self.r_max_sq.numpy().astype(np.float32), 0.0))

        tx = np.zeros((self.batch_size,), dtype=np.float32)
        ty = np.zeros((self.batch_size,), dtype=np.float32)
        dtheta = np.zeros((self.batch_size,), dtype=np.float32)

        valid = sum_w > 1e-8
        if np.any(valid):
            delta_x = np.zeros((self.batch_size,), dtype=np.float32)
            delta_y = np.zeros((self.batch_size,), dtype=np.float32)
            contact_x = np.zeros((self.batch_size,), dtype=np.float32)
            contact_y = np.zeros((self.batch_size,), dtype=np.float32)
            delta_x[valid] = weighted_dx[valid] / sum_w[valid]
            delta_y[valid] = weighted_dy[valid] / sum_w[valid]
            contact_x[valid] = weighted_px[valid] / sum_w[valid]
            contact_y[valid] = weighted_py[valid] / sum_w[valid]

            tx = delta_x.astype(np.float32)
            ty = delta_y.astype(np.float32)
            lever_x = contact_x - np.asarray(body_com_world, dtype=np.float32)[:, 0]
            lever_y = contact_y - np.asarray(body_com_world, dtype=np.float32)[:, 1]
            raw_dtheta = (lever_x * delta_y - lever_y * delta_x) / max(float(self.gt_body_inertia), 1e-8)
            dtheta = raw_dtheta.astype(np.float32)

            trans_mag = np.hypot(tx, ty).astype(np.float32)
            scale_mask = (max_disp > 0.0) & np.isfinite(trans_mag) & (trans_mag > max_disp)
            tx[scale_mask] = tx[scale_mask] * (max_disp[scale_mask] / np.maximum(trans_mag[scale_mask], 1e-8))
            ty[scale_mask] = ty[scale_mask] * (max_disp[scale_mask] / np.maximum(trans_mag[scale_mask], 1e-8))
            bad_mask = ~np.isfinite(trans_mag)
            tx[bad_mask] = 0.0
            ty[bad_mask] = 0.0

            bad_theta = ~np.isfinite(dtheta)
            dtheta[bad_theta] = 0.0
            theta_cap = np.where(max_disp > 0.0, max_disp / np.maximum(r_max, 1e-6), 0.0).astype(np.float32)
            clip_mask = max_disp > 0.0
            dtheta[clip_mask] = np.clip(dtheta[clip_mask], -theta_cap[clip_mask], theta_cap[clip_mask]).astype(np.float32)
            dtheta[~clip_mask] = 0.0

        return tx.astype(np.float32), ty.astype(np.float32), dtheta.astype(np.float32)

    def _build_pusher_path_batch(
        self,
        target_xy: np.ndarray,
        active_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target_xy, dtype=np.float64).reshape(self.batch_size, 2)
        pos = np.asarray(self.pusher_pos[:, :2], dtype=np.float64).copy()
        vel = np.asarray(self.pusher_velocity, dtype=np.float64).copy()
        active = np.asarray(active_mask, dtype=bool).reshape(self.batch_size)

        path = np.zeros((self.controller_steps + 1, self.batch_size, 3), dtype=np.float32)
        path[0, :, :2] = pos.astype(np.float32)

        for step_idx in range(self.controller_steps):
            acc = self.pusher_k_p * (target - pos) - self.pusher_k_v * vel
            vel_next = vel + acc * self.sim_dt
            pos_next = pos + vel_next * self.sim_dt
            vel[active] = vel_next[active]
            pos[active] = pos_next[active]
            path[step_idx + 1, :, :2] = pos.astype(np.float32)

        return path, vel.astype(np.float32)

    def step_batch(
        self,
        action_world: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action_arr = np.asarray(action_world, dtype=np.float32).reshape(self.batch_size, 2)
        step_active = (
            np.asarray(active_mask, dtype=bool).reshape(self.batch_size)
            if active_mask is not None
            else self._active_host.astype(bool)
        )
        self._update_active(step_active.astype(np.int32))

        target_xy = self.pusher_pos[:, :2].copy()
        target_xy[step_active] = target_xy[step_active] + action_arr[step_active]
        pusher_path, final_velocity = self._build_pusher_path_batch(target_xy, step_active)
        resolved_path = pusher_path.copy()
        resolved_path[0, :, :2] = self.pusher_pos[:, :2].copy()

        for seg_idx in range(self.controller_steps):
            prev_pxy = resolved_path[seg_idx, :, :2].astype(np.float32, copy=False)
            commanded_pxy = pusher_path[seg_idx + 1, :, :2].astype(np.float32, copy=False)

            start_host = np.zeros((self.batch_size, 3), dtype=np.float32)
            end_host = np.zeros((self.batch_size, 3), dtype=np.float32)
            start_host[:, :2] = prev_pxy
            end_host[:, :2] = commanded_pxy
            self.pusher_start.assign(start_host)
            self.pusher_end.assign(end_host)

            wp.launch(
                kernel=_batch_predict_positions,
                dim=self.total_particles,
                inputs=[self.x, self.v, self.inv_m, self.x_pred, self.sim_dt, self.lin_damp, self.N, self.active],
                device=self.device,
            )

            for _ in range(self.iters):
                wp.launch(
                    kernel=_batch_project_walls_aabb,
                    dim=self.total_particles,
                    inputs=[self.x_pred, self.inv_m, self.xmin, self.xmax, self.ymin, self.ymax, self.pr, self.N, self.active],
                    device=self.device,
                )
                wp.launch(
                    kernel=_batch_pusher_swept_contact,
                    dim=self.total_particles,
                    inputs=[
                        self.x,
                        self.x_pred,
                        self.inv_m,
                        self.pusher_start,
                        self.pusher_end,
                        self.pusher_r,
                        self.pr,
                        self.mu,
                        self.contact_alpha,
                        self.contact_delta,
                        self.contact_weight,
                        self.N,
                        self.active,
                    ],
                    device=self.device,
                )

                pose = self._compute_com_and_theta_batch()
                body_com_world_xy = self._body_com_world_from_cloud_pose_batch(pose[:, :2], pose[:, 2])
                tx, ty, dtheta = self._compute_contact_fit_batch(body_com_world_xy)

                body_com_host = np.zeros((self.batch_size, 3), dtype=np.float32)
                body_com_host[:, :2] = body_com_world_xy
                self.body_com_center.assign(body_com_host)
                self.tx.assign(tx)
                self.ty.assign(ty)
                self.dtheta.assign(dtheta)

                wp.launch(
                    kernel=_batch_apply_rigid_transform,
                    dim=self.total_particles,
                    inputs=[
                        self.x_pred,
                        self.inv_m,
                        self.body_com_center,
                        self.tx,
                        self.ty,
                        self.dtheta,
                        self.N,
                        self.active,
                    ],
                    device=self.device,
                )

                theta_target = np.asarray([_wrap_pi(float(v)) for v in (pose[:, 2] + dtheta)], dtype=np.float32)
                body_com_after_xy = body_com_world_xy + np.stack([tx, ty], axis=1).astype(np.float32)
                shape_match_xy = self._cloud_center_world_from_body_com_batch(body_com_after_xy, theta_target)
                shape_match_host = np.zeros((self.batch_size, 3), dtype=np.float32)
                shape_match_host[:, :2] = shape_match_xy
                self.shape_match_center.assign(shape_match_host)
                self.theta_target.assign(theta_target)

                wp.launch(
                    kernel=_batch_shape_match_project,
                    dim=self.total_particles,
                    inputs=[
                        self.x_pred,
                        self.r0,
                        self.inv_m,
                        self.shape_match_center,
                        self.theta_target,
                        self.alpha_rigid,
                        self.N,
                        self.active,
                    ],
                    device=self.device,
                )

            wp.launch(
                kernel=_batch_finalize,
                dim=self.total_particles,
                inputs=[self.x, self.v, self.inv_m, self.x_pred, self.sim_dt, self.vel_damp, self.N, self.active],
                device=self.device,
            )
            wp.launch(
                kernel=_batch_apply_ground_friction,
                dim=self.total_particles,
                inputs=[
                    self.v,
                    self.inv_m,
                    self.sim_dt,
                    self.ground_friction_accel,
                    self.rest_speed_eps,
                    self.N,
                    self.active,
                ],
                device=self.device,
            )

            if np.any(step_active):
                particle_xy = self.get_particle_positions_batch()
                for lane_idx in np.flatnonzero(step_active):
                    resolved_pxy = _resolve_particle_union_endpoint(
                        prev_pxy[lane_idx],
                        commanded_pxy[lane_idx],
                        particle_xy[lane_idx],
                        pusher_r=float(self.pusher_r),
                        pr=float(self.pr),
                    )
                    resolved_path[seg_idx + 1, lane_idx, :2] = np.asarray(resolved_pxy, dtype=np.float32)

        if np.any(step_active):
            self.pusher_pos[step_active, :2] = resolved_path[-1, step_active, :2]
            self.pusher_pos[step_active, 2] = 0.0
            if resolved_path.shape[0] >= 2:
                realized_velocity = (
                    resolved_path[-1, step_active, :2] - resolved_path[-2, step_active, :2]
                ) / max(self.sim_dt, 1e-8)
                finite_mask = np.all(np.isfinite(realized_velocity), axis=1)
                self.pusher_velocity[step_active, :] = final_velocity[step_active]
                if np.any(finite_mask):
                    step_indices = np.flatnonzero(step_active)
                    self.pusher_velocity[step_indices[finite_mask], :] = realized_velocity[finite_mask].astype(np.float32)
            else:
                self.pusher_velocity[step_active, :] = final_velocity[step_active]

        pose = self.get_object_pose_batch()
        self._last_pose[:, :] = pose
        dx = pose[:, 0] - self.goal_pose[:, 0]
        dy = pose[:, 1] - self.goal_pose[:, 1]
        dtheta = np.asarray(
            [_wrap_pi(float(pose[idx, 2] - self.goal_pose[idx, 2])) for idx in range(self.batch_size)],
            dtype=np.float32,
        )
        reward = -(dx * dx + dy * dy + 0.1 * dtheta * dtheta).astype(np.float32)
        done = ((dx * dx + dy * dy) < (0.01 ** 2)) & (np.abs(dtheta) < 0.15)
        return pose.astype(np.float32), reward.astype(np.float32), done.astype(bool)


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
