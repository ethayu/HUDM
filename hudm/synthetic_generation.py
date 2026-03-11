from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from hudm.metrics import angle_delta


SCENE_CENTER = np.asarray([256.0, 256.0], dtype=np.float32)
DEFAULT_BOUNDS = (15.0, 491.0)
T_SCALE = 40.0
T_LOCAL_POLYGONS: tuple[np.ndarray, ...] = (
    np.asarray(
        [
            (-2.0 * T_SCALE, T_SCALE),
            (2.0 * T_SCALE, T_SCALE),
            (2.0 * T_SCALE, 0.0),
            (-2.0 * T_SCALE, 0.0),
        ],
        dtype=np.float32,
    ),
    np.asarray(
        [
            (-0.5 * T_SCALE, T_SCALE),
            (-0.5 * T_SCALE, 4.0 * T_SCALE),
            (0.5 * T_SCALE, 4.0 * T_SCALE),
            (0.5 * T_SCALE, T_SCALE),
        ],
        dtype=np.float32,
    ),
)


@dataclass(frozen=True)
class QualityProfile:
    name: str
    bounds_low: float
    bounds_high: float
    spawn_clearance: float
    agent_wall_clearance: float
    agent_block_clearance: float
    wall_adjacent_clearance: float
    max_wall_adjacent_fraction: float | None
    meaningful_motion_threshold: float
    glitch_motion_threshold: float
    glitch_agent_block_distance: float
    min_contact_fraction_floor: float


@dataclass(frozen=True)
class AcceptanceSpec:
    mode_profile: str
    quality_profile: str
    min_net_translation: float
    min_total_angle: float
    min_contact_fraction: float
    min_motion_fraction: float
    max_total_angle: float | None
    strong_translation_threshold: float | None
    require_translation_or_rotation: bool


@dataclass(frozen=True)
class ManeuverCandidate:
    family: str
    name: str
    action_abs_seq: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RolloutDiagnostics:
    total_block_translation_path: float
    net_block_translation: float
    total_absolute_angle_change: float
    motion_fraction: float
    contact_fraction: float
    first_contact_step: int
    first_meaningful_motion_step: int
    wall_adjacent_fraction: float
    suspicious_no_contact_motion_count: int
    out_of_bounds_steps: int
    min_wall_clearance: float
    start_wall_clearance: float
    end_wall_clearance: float
    max_block_step_displacement: float
    num_steps: int


@dataclass(frozen=True)
class EpisodeAttempt:
    init_state: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    actions_abs: np.ndarray
    diagnostics: RolloutDiagnostics
    rejection_reasons: tuple[str, ...]
    selected_families: tuple[str, ...]
    selected_candidates: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return len(self.rejection_reasons) == 0


QUALITY_PROFILES: dict[str, QualityProfile] = {
    "strict": QualityProfile(
        name="strict",
        bounds_low=DEFAULT_BOUNDS[0],
        bounds_high=DEFAULT_BOUNDS[1],
        spawn_clearance=10.0,
        agent_wall_clearance=20.0,
        agent_block_clearance=24.0,
        wall_adjacent_clearance=10.0,
        max_wall_adjacent_fraction=0.10,
        meaningful_motion_threshold=1.5,
        glitch_motion_threshold=4.0,
        glitch_agent_block_distance=70.0,
        min_contact_fraction_floor=0.10,
    ),
    "balanced": QualityProfile(
        name="balanced",
        bounds_low=DEFAULT_BOUNDS[0],
        bounds_high=DEFAULT_BOUNDS[1],
        spawn_clearance=8.0,
        agent_wall_clearance=16.0,
        agent_block_clearance=20.0,
        wall_adjacent_clearance=8.0,
        max_wall_adjacent_fraction=0.15,
        meaningful_motion_threshold=1.25,
        glitch_motion_threshold=4.5,
        glitch_agent_block_distance=72.0,
        min_contact_fraction_floor=0.08,
    ),
    "loose": QualityProfile(
        name="loose",
        bounds_low=DEFAULT_BOUNDS[0],
        bounds_high=DEFAULT_BOUNDS[1],
        spawn_clearance=6.0,
        agent_wall_clearance=12.0,
        agent_block_clearance=18.0,
        wall_adjacent_clearance=6.0,
        max_wall_adjacent_fraction=None,
        meaningful_motion_threshold=1.0,
        glitch_motion_threshold=5.0,
        glitch_agent_block_distance=75.0,
        min_contact_fraction_floor=0.05,
    ),
}

MODE_PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "train": {"translate": 0.45, "rotate": 0.20, "sweep": 0.25, "recover": 0.10},
    "planning": {"translate": 0.70, "rotate": 0.10, "sweep": 0.10, "recover": 0.10},
    "balanced": {"translate": 0.55, "rotate": 0.15, "sweep": 0.20, "recover": 0.10},
}

MODE_ACCEPTANCE: dict[str, dict[str, Any]] = {
    "train": {
        "min_net_translation": 20.0,
        "min_total_angle": 0.35,
        "min_contact_fraction": 0.10,
        "min_motion_fraction": 0.00,
        "max_total_angle": None,
        "strong_translation_threshold": None,
        "require_translation_or_rotation": True,
    },
    "planning": {
        "min_net_translation": 30.0,
        "min_total_angle": 0.0,
        "min_contact_fraction": 0.12,
        "min_motion_fraction": 0.12,
        "max_total_angle": 1.20,
        "strong_translation_threshold": 45.0,
        "require_translation_or_rotation": False,
    },
    "balanced": {
        "min_net_translation": 24.0,
        "min_total_angle": 0.45,
        "min_contact_fraction": 0.10,
        "min_motion_fraction": 0.00,
        "max_total_angle": None,
        "strong_translation_threshold": None,
        "require_translation_or_rotation": True,
    },
}


def normalize(v: Sequence[float], *, fallback: Sequence[float] = (1.0, 0.0)) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return arr / norm


def rotate_vec(v: Sequence[float], radians: float) -> np.ndarray:
    c = math.cos(float(radians))
    s = math.sin(float(radians))
    arr = np.asarray(v, dtype=np.float32)
    return np.asarray([c * arr[0] - s * arr[1], s * arr[0] + c * arr[1]], dtype=np.float32)


def perp(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    return np.asarray([-arr[1], arr[0]], dtype=np.float32)


def clip_point(point: Sequence[float], low: float, high: float) -> np.ndarray:
    arr = np.asarray(point, dtype=np.float32)
    return np.clip(arr, low, high).astype(np.float32)


def angle_limited(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def state_block_pose(state: Sequence[float]) -> tuple[np.ndarray, float]:
    arr = np.asarray(state, dtype=np.float32)
    return arr[2:4].copy(), float(arr[4])


def state_agent_pos(state: Sequence[float]) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    return arr[:2].copy()


def transform_local_points(points: np.ndarray, block_pos: Sequence[float], angle: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    pos = np.asarray(block_pos, dtype=np.float32).reshape(1, 2)
    return pts @ rot.T + pos


def block_world_polygons(
    state: Sequence[float],
    *,
    local_polygons: Sequence[np.ndarray] = T_LOCAL_POLYGONS,
) -> list[np.ndarray]:
    block_pos, angle = state_block_pose(state)
    return [transform_local_points(poly, block_pos, angle) for poly in local_polygons]


def block_world_vertices(
    state: Sequence[float],
    *,
    local_polygons: Sequence[np.ndarray] = T_LOCAL_POLYGONS,
) -> np.ndarray:
    polys = block_world_polygons(state, local_polygons=local_polygons)
    return np.concatenate(polys, axis=0).astype(np.float32)


def block_shape_multipolygon(
    state: Sequence[float],
    *,
    local_polygons: Sequence[np.ndarray] = T_LOCAL_POLYGONS,
) -> MultiPolygon:
    polys = [Polygon(poly) for poly in block_world_polygons(state, local_polygons=local_polygons)]
    return MultiPolygon(polys)


def wall_clearance(
    vertices: np.ndarray,
    *,
    low: float = DEFAULT_BOUNDS[0],
    high: float = DEFAULT_BOUNDS[1],
) -> float:
    verts = np.asarray(vertices, dtype=np.float32)
    x = verts[:, 0]
    y = verts[:, 1]
    clearances = np.asarray(
        [
            float(np.min(x - low)),
            float(np.min(high - x)),
            float(np.min(y - low)),
            float(np.min(high - y)),
        ],
        dtype=np.float32,
    )
    return float(np.min(clearances))


def is_out_of_bounds(
    vertices: np.ndarray,
    *,
    low: float = DEFAULT_BOUNDS[0],
    high: float = DEFAULT_BOUNDS[1],
) -> bool:
    verts = np.asarray(vertices, dtype=np.float32)
    return bool(np.any(verts < low) or np.any(verts > high))


def agent_block_distance(
    state: Sequence[float],
    *,
    local_polygons: Sequence[np.ndarray] = T_LOCAL_POLYGONS,
) -> float:
    poly = block_shape_multipolygon(state, local_polygons=local_polygons)
    point = Point(*state_agent_pos(state))
    return float(poly.distance(point))


def validate_spawn_state(
    state: Sequence[float],
    quality_profile: str,
    *,
    local_polygons: Sequence[np.ndarray] = T_LOCAL_POLYGONS,
) -> tuple[bool, tuple[str, ...], dict[str, Any]]:
    quality = QUALITY_PROFILES[str(quality_profile)]
    state_arr = np.asarray(state, dtype=np.float32)
    vertices = block_world_vertices(state_arr, local_polygons=local_polygons)
    clearance = wall_clearance(vertices, low=quality.bounds_low, high=quality.bounds_high)
    out_of_bounds = is_out_of_bounds(vertices, low=quality.bounds_low, high=quality.bounds_high)
    agent = state_agent_pos(state_arr)
    agent_clearance = float(
        min(
            float(agent[0] - quality.bounds_low),
            float(quality.bounds_high - agent[0]),
            float(agent[1] - quality.bounds_low),
            float(quality.bounds_high - agent[1]),
        )
    )
    agent_dist = agent_block_distance(state_arr, local_polygons=local_polygons)
    reasons: list[str] = []
    if out_of_bounds:
        reasons.append("spawn_block_oob")
    if clearance < quality.spawn_clearance:
        reasons.append("spawn_block_near_wall")
    if agent_clearance < quality.agent_wall_clearance:
        reasons.append("spawn_agent_near_wall")
    if agent_dist < quality.agent_block_clearance:
        reasons.append("spawn_agent_overlap_block")
    meta = {
        "block_clearance": clearance,
        "agent_wall_clearance": agent_clearance,
        "agent_block_distance": agent_dist,
        "out_of_bounds": out_of_bounds,
    }
    return len(reasons) == 0, tuple(reasons), meta


def resolve_mode_weights(
    mode_profile: str,
    overrides: Mapping[str, float | None] | None = None,
) -> dict[str, float]:
    if mode_profile not in MODE_PROFILE_WEIGHTS:
        raise ValueError(f"Unknown mode profile: {mode_profile}")
    weights = dict(MODE_PROFILE_WEIGHTS[mode_profile])
    if overrides:
        for key in ("translate", "rotate", "sweep", "recover"):
            value = overrides.get(key, None)
            if value is not None:
                weights[key] = float(value)
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0.0:
        raise ValueError("At least one maneuver weight must be > 0.")
    return {key: max(0.0, float(value)) / total for key, value in weights.items()}


def resolve_acceptance_spec(mode_profile: str, quality_profile: str) -> AcceptanceSpec:
    if mode_profile not in MODE_ACCEPTANCE:
        raise ValueError(f"Unknown mode profile: {mode_profile}")
    quality = QUALITY_PROFILES[str(quality_profile)]
    base = dict(MODE_ACCEPTANCE[mode_profile])
    relax = 1.0
    if quality.name == "balanced":
        relax = 0.9
    elif quality.name == "loose":
        relax = 0.7
    min_contact_fraction = max(
        quality.min_contact_fraction_floor,
        float(base["min_contact_fraction"]) * relax,
    )
    min_motion_fraction = float(base["min_motion_fraction"]) * relax
    min_net_translation = float(base["min_net_translation"]) * relax
    min_total_angle = float(base["min_total_angle"]) * relax
    max_total_angle = base["max_total_angle"]
    if max_total_angle is not None and quality.name == "balanced":
        max_total_angle = float(max_total_angle) * 1.15
    elif max_total_angle is not None and quality.name == "loose":
        max_total_angle = None
    strong_translation = base["strong_translation_threshold"]
    if strong_translation is not None and quality.name == "balanced":
        strong_translation = float(strong_translation) * 0.9
    elif strong_translation is not None and quality.name == "loose":
        strong_translation = float(strong_translation) * 0.75
    return AcceptanceSpec(
        mode_profile=str(mode_profile),
        quality_profile=str(quality_profile),
        min_net_translation=min_net_translation,
        min_total_angle=min_total_angle,
        min_contact_fraction=min_contact_fraction,
        min_motion_fraction=min_motion_fraction,
        max_total_angle=max_total_angle,
        strong_translation_threshold=strong_translation,
        require_translation_or_rotation=bool(base["require_translation_or_rotation"]),
    )


def _support_point(vertices: np.ndarray, direction: Sequence[float], *, slack: float = 12.0) -> np.ndarray:
    d = normalize(direction)
    verts = np.asarray(vertices, dtype=np.float32)
    projections = verts @ d
    min_proj = float(np.min(projections))
    mask = projections <= (min_proj + float(slack))
    selected = verts[mask]
    if selected.shape[0] <= 0:
        selected = verts[np.argmin(projections)][None, :]
    return np.mean(selected, axis=0).astype(np.float32)


def _interpolate_targets(
    agent_pos: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    *,
    horizon: int,
    low: float = DEFAULT_BOUNDS[0],
    high: float = DEFAULT_BOUNDS[1],
) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if len(waypoints) <= 0:
        raise ValueError("waypoints must be non-empty")
    points = [np.asarray(agent_pos, dtype=np.float32)] + [clip_point(wp, low, high) for wp in waypoints]
    n_segments = len(points) - 1
    base = horizon // n_segments
    rem = horizon % n_segments
    seq: list[np.ndarray] = []
    for seg_idx in range(n_segments):
        steps = base + (1 if seg_idx < rem else 0)
        start = points[seg_idx]
        end = points[seg_idx + 1]
        for step_idx in range(steps):
            alpha = float(step_idx + 1) / float(steps)
            seq.append(((1.0 - alpha) * start + alpha * end).astype(np.float32))
    return np.stack(seq, axis=0).astype(np.float32)


def build_contact_candidates(
    state: Sequence[float],
    family: str,
    rng: np.random.Generator,
    *,
    mode_profile: str | None = None,
    horizon: int = 6,
    low: float = DEFAULT_BOUNDS[0],
    high: float = DEFAULT_BOUNDS[1],
    local_polygons: Sequence[np.ndarray] = T_LOCAL_POLYGONS,
) -> list[ManeuverCandidate]:
    state_arr = np.asarray(state, dtype=np.float32)
    agent = state_agent_pos(state_arr)
    block_pos, theta = state_block_pose(state_arr)
    center_dir = normalize(SCENE_CENTER - block_pos, fallback=(1.0, 0.0))
    wall_risk = wall_clearance(block_world_vertices(state_arr, local_polygons=local_polygons), low=low, high=high)
    candidates: list[ManeuverCandidate] = []
    vertices = block_world_vertices(state_arr, local_polygons=local_polygons)

    def add_candidate(name: str, waypoints: Sequence[Sequence[float]], **metadata: Any) -> None:
        seq = _interpolate_targets(agent, waypoints, horizon=horizon, low=low, high=high)
        candidates.append(
            ManeuverCandidate(
                family=str(family),
                name=name,
                action_abs_seq=seq,
                metadata=dict(metadata),
            )
        )

    if family == "translate":
        dir_offset_deg = 10.0 if str(mode_profile) == "planning" else 22.5
        tangent_span = 8.0 if str(mode_profile) == "planning" else 18.0
        directions = [
            center_dir,
            rotate_vec(center_dir, math.radians(dir_offset_deg)),
            rotate_vec(center_dir, math.radians(-dir_offset_deg)),
        ]
        for idx, move_dir in enumerate(directions):
            contact = _support_point(vertices, move_dir)
            tangent = normalize(perp(move_dir))
            tangent_offset = float(rng.uniform(-tangent_span, tangent_span))
            pre = contact - move_dir * 85.0 + tangent * tangent_offset
            approach = contact - move_dir * 55.0 + tangent * tangent_offset
            push_end = contact + move_dir * 100.0
            add_candidate(
                f"translate_{idx}",
                [pre, approach, push_end],
                move_dir=move_dir.tolist(),
                contact_point=contact.tolist(),
            )
    elif family == "rotate":
        if wall_risk < 24.0:
            return []
        stem_points = [np.asarray([-28.0, 95.0], dtype=np.float32), np.asarray([28.0, 95.0], dtype=np.float32)]
        for side_idx, local_pt in enumerate(stem_points):
            contact = transform_local_points(local_pt[None, :], block_pos, theta)[0]
            radial = normalize(contact - block_pos, fallback=(1.0, 0.0))
            for rot_sign in (-1.0, 1.0):
                tangent = normalize(rot_sign * perp(radial) + 0.25 * center_dir)
                pre = contact - radial * 48.0 - tangent * 12.0
                push_end = contact + tangent * 72.0
                add_candidate(
                    f"rotate_{side_idx}_{'cw' if rot_sign < 0 else 'ccw'}",
                    [pre, contact, push_end],
                    radial=radial.tolist(),
                    tangent=tangent.tolist(),
                )
    elif family == "sweep":
        for sweep_sign in (-1.0, 1.0):
            move_dir = normalize(rotate_vec(center_dir, math.radians(12.0 * sweep_sign)))
            orbit_dir = normalize(rotate_vec(move_dir, math.radians(90.0 * sweep_sign)))
            contact = _support_point(vertices, move_dir)
            orbit = block_pos + orbit_dir * float(rng.uniform(60.0, 80.0))
            pre = orbit - move_dir * 18.0
            push_end = contact + move_dir * 68.0
            add_candidate(
                f"sweep_{'left' if sweep_sign < 0 else 'right'}",
                [pre, orbit, contact, push_end],
                move_dir=move_dir.tolist(),
            )
    elif family == "recover":
        move_dir = center_dir
        contact = _support_point(vertices, move_dir)
        stage = contact - move_dir * 100.0
        safe = clip_point((stage + SCENE_CENTER) * 0.5, low + 12.0, high - 12.0)
        add_candidate(
            "recover_center",
            [safe, stage],
            move_dir=move_dir.tolist(),
            contact_point=contact.tolist(),
        )
        tangent = normalize(perp(move_dir))
        for side_idx, sign in enumerate((-1.0, 1.0)):
            flank = contact - move_dir * 95.0 + tangent * (sign * 28.0)
            add_candidate(
                f"recover_flank_{side_idx}",
                [safe, flank],
                move_dir=move_dir.tolist(),
            )
    else:
        raise ValueError(f"Unknown maneuver family: {family}")
    return candidates


def env_input_from_abs_target(
    agent_pos: Sequence[float],
    target_abs: Sequence[float],
    *,
    env_action_scale: float,
    relative: bool,
) -> np.ndarray:
    target = np.asarray(target_abs, dtype=np.float32)
    if relative:
        return (target - np.asarray(agent_pos, dtype=np.float32)) / float(env_action_scale)
    return target / float(env_action_scale)


def weighted_choice(
    weights: Mapping[str, float],
    rng: np.random.Generator,
) -> str:
    keys = list(weights.keys())
    vals = np.asarray([max(0.0, float(weights[key])) for key in keys], dtype=np.float64)
    total = float(np.sum(vals))
    if total <= 0.0:
        raise ValueError("All weights are zero.")
    probs = vals / total
    idx = int(rng.choice(len(keys), p=probs))
    return str(keys[idx])


def choose_maneuver_family(
    state: Sequence[float],
    mode_weights: Mapping[str, float],
    rng: np.random.Generator,
    *,
    mode_profile: str,
    quality_profile: str,
    recent_motion_fraction: float | None = None,
    recent_contact_fraction: float | None = None,
    local_polygons: Sequence[np.ndarray] = T_LOCAL_POLYGONS,
) -> str:
    quality = QUALITY_PROFILES[str(quality_profile)]
    weights = dict(mode_weights)
    clearance = wall_clearance(
        block_world_vertices(state, local_polygons=local_polygons),
        low=quality.bounds_low,
        high=quality.bounds_high,
    )
    if clearance < max(quality.spawn_clearance + 6.0, 18.0):
        return "recover"
    if recent_motion_fraction is not None and recent_contact_fraction is not None:
        if recent_motion_fraction < 0.05 and recent_contact_fraction < 0.05:
            weights["sweep"] = max(weights.get("sweep", 0.0), 0.25)
            weights["recover"] = max(weights.get("recover", 0.0), 0.20)
    if str(mode_profile) == "planning":
        weights["rotate"] = 0.0
        if not (
            recent_motion_fraction is not None
            and recent_contact_fraction is not None
            and recent_motion_fraction < 0.05
            and recent_contact_fraction < 0.05
        ):
            weights["sweep"] = 0.0
    if clearance < 28.0:
        weights["rotate"] = 0.0
        weights["recover"] = max(weights.get("recover", 0.0), 0.25)
    return weighted_choice(weights, rng)


def analyze_rollout(
    states_before: Sequence[np.ndarray],
    states_after: Sequence[np.ndarray],
    contacts: Sequence[int],
    *,
    quality_profile: str,
    local_polygons: Sequence[np.ndarray] = T_LOCAL_POLYGONS,
) -> RolloutDiagnostics:
    quality = QUALITY_PROFILES[str(quality_profile)]
    if len(states_before) != len(states_after) or len(states_before) != len(contacts):
        raise ValueError("rollout record lengths must match")
    if len(states_after) <= 0:
        return RolloutDiagnostics(
            total_block_translation_path=0.0,
            net_block_translation=0.0,
            total_absolute_angle_change=0.0,
            motion_fraction=0.0,
            contact_fraction=0.0,
            first_contact_step=-1,
            first_meaningful_motion_step=-1,
            wall_adjacent_fraction=0.0,
            suspicious_no_contact_motion_count=0,
            out_of_bounds_steps=0,
            min_wall_clearance=0.0,
            start_wall_clearance=0.0,
            end_wall_clearance=0.0,
            max_block_step_displacement=0.0,
            num_steps=0,
        )

    block_before = np.stack([np.asarray(s, dtype=np.float32)[2:4] for s in states_before], axis=0)
    block_after = np.stack([np.asarray(s, dtype=np.float32)[2:4] for s in states_after], axis=0)
    block_step = np.linalg.norm(block_after - block_before, axis=1)
    angles_before = [float(np.asarray(s, dtype=np.float32)[4]) for s in states_before]
    angles_after = [float(np.asarray(s, dtype=np.float32)[4]) for s in states_after]
    angle_steps = np.asarray(
        [angle_delta(a_after, a_before) for a_before, a_after in zip(angles_before, angles_after)],
        dtype=np.float32,
    )
    clearances = np.asarray(
        [
            wall_clearance(
                block_world_vertices(s_after, local_polygons=local_polygons),
                low=quality.bounds_low,
                high=quality.bounds_high,
            )
            for s_after in states_after
        ],
        dtype=np.float32,
    )
    oob = np.asarray(
        [
            is_out_of_bounds(
                block_world_vertices(s_after, local_polygons=local_polygons),
                low=quality.bounds_low,
                high=quality.bounds_high,
            )
            for s_after in states_after
        ],
        dtype=bool,
    )
    contact_arr = np.asarray([int(c) for c in contacts], dtype=np.int32)
    meaningful_motion = block_step > float(quality.meaningful_motion_threshold)
    wall_adjacent = clearances < float(quality.wall_adjacent_clearance)
    agent_block_dists = np.asarray(
        [np.linalg.norm(np.asarray(s, dtype=np.float32)[:2] - np.asarray(s, dtype=np.float32)[2:4]) for s in states_before],
        dtype=np.float32,
    )
    suspicious = (block_step > float(quality.glitch_motion_threshold)) & (contact_arr <= 0) & (
        agent_block_dists > float(quality.glitch_agent_block_distance)
    )
    contact_steps = np.flatnonzero(contact_arr > 0)
    motion_steps = np.flatnonzero(meaningful_motion)

    return RolloutDiagnostics(
        total_block_translation_path=float(np.sum(block_step)),
        net_block_translation=float(np.linalg.norm(block_after[-1] - block_before[0])),
        total_absolute_angle_change=float(np.sum(angle_steps)),
        motion_fraction=float(np.mean(meaningful_motion.astype(np.float32))),
        contact_fraction=float(np.mean((contact_arr > 0).astype(np.float32))),
        first_contact_step=int(contact_steps[0]) if contact_steps.size > 0 else -1,
        first_meaningful_motion_step=int(motion_steps[0]) if motion_steps.size > 0 else -1,
        wall_adjacent_fraction=float(np.mean(wall_adjacent.astype(np.float32))),
        suspicious_no_contact_motion_count=int(np.sum(suspicious)),
        out_of_bounds_steps=int(np.sum(oob)),
        min_wall_clearance=float(np.min(clearances)),
        start_wall_clearance=float(clearances[0]),
        end_wall_clearance=float(clearances[-1]),
        max_block_step_displacement=float(np.max(block_step)) if block_step.size > 0 else 0.0,
        num_steps=int(len(states_after)),
    )


def reject_rollout(
    diagnostics: RolloutDiagnostics,
    mode_profile: str,
    quality_profile: str,
) -> tuple[bool, tuple[str, ...]]:
    quality = QUALITY_PROFILES[str(quality_profile)]
    spec = resolve_acceptance_spec(mode_profile, quality_profile)
    reasons: list[str] = []

    if diagnostics.out_of_bounds_steps > 0:
        reasons.append("episode_block_oob")
    if diagnostics.suspicious_no_contact_motion_count > 0:
        reasons.append("episode_glitch_motion")
    if quality.max_wall_adjacent_fraction is not None and diagnostics.wall_adjacent_fraction > float(
        quality.max_wall_adjacent_fraction
    ):
        reasons.append("episode_wall_adjacent")

    if quality.name == "loose":
        if diagnostics.contact_fraction < quality.min_contact_fraction_floor:
            reasons.append("episode_no_contact")
        if diagnostics.net_block_translation < 10.0 and diagnostics.total_absolute_angle_change < 0.20:
            reasons.append("episode_dead_motion")
        return len(reasons) > 0, tuple(reasons)

    if diagnostics.contact_fraction < spec.min_contact_fraction:
        reasons.append("episode_low_contact")
    if diagnostics.motion_fraction < spec.min_motion_fraction:
        reasons.append("episode_low_motion_fraction")
    if diagnostics.first_meaningful_motion_step > 20:
        reasons.append("episode_late_motion")

    if spec.require_translation_or_rotation:
        if diagnostics.net_block_translation < spec.min_net_translation and diagnostics.total_absolute_angle_change < spec.min_total_angle:
            reasons.append("episode_low_translation_or_rotation")
    else:
        if diagnostics.net_block_translation < spec.min_net_translation:
            reasons.append("episode_low_translation")

    if spec.max_total_angle is not None and diagnostics.total_absolute_angle_change > spec.max_total_angle:
        strong_threshold = spec.strong_translation_threshold
        if strong_threshold is None or diagnostics.net_block_translation < strong_threshold:
            reasons.append("episode_excess_rotation")

    return len(reasons) > 0, tuple(reasons)


def score_shadow_rollout(
    family: str,
    diagnostics: RolloutDiagnostics,
    mode_profile: str,
    quality_profile: str,
) -> float:
    quality = QUALITY_PROFILES[str(quality_profile)]
    score = 0.0
    score += 15.0 * diagnostics.contact_fraction
    score += 0.2 * diagnostics.min_wall_clearance
    score += 0.15 * diagnostics.end_wall_clearance
    score -= 100.0 * float(diagnostics.out_of_bounds_steps)
    score -= 40.0 * float(diagnostics.suspicious_no_contact_motion_count)
    if quality.max_wall_adjacent_fraction is not None:
        score -= 25.0 * diagnostics.wall_adjacent_fraction

    if family == "translate":
        score += 2.7 * diagnostics.net_block_translation
        score += 0.8 * diagnostics.total_block_translation_path
        score -= 0.7 * diagnostics.total_absolute_angle_change
    elif family == "rotate":
        score += 15.0 * diagnostics.total_absolute_angle_change
        score += 3.0 * diagnostics.contact_fraction
        score -= 0.3 * diagnostics.net_block_translation
    elif family == "sweep":
        score += 1.5 * diagnostics.total_block_translation_path
        score += 6.0 * diagnostics.contact_fraction
        score += 2.0 * diagnostics.total_absolute_angle_change
    elif family == "recover":
        score += 2.5 * (diagnostics.end_wall_clearance - diagnostics.start_wall_clearance)
        score += 4.0 * diagnostics.contact_fraction
        score += 0.5 * diagnostics.net_block_translation
    else:
        raise ValueError(f"Unknown family: {family}")

    if mode_profile == "planning":
        if family == "translate":
            score += 0.8 * diagnostics.net_block_translation
            score -= 4.0 * diagnostics.total_absolute_angle_change
        elif family == "sweep":
            score -= 3.0 * diagnostics.total_absolute_angle_change
        elif family == "rotate":
            score -= 10.0 * diagnostics.total_absolute_angle_change
    elif mode_profile == "train":
        if family == "rotate":
            score += 2.0 * diagnostics.total_absolute_angle_change
        elif family == "sweep":
            score += 1.0 * diagnostics.contact_fraction

    return float(score)


def summarize_rejections(reasons: Sequence[Sequence[str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in reasons:
        counter.update(item)
    return dict(sorted(counter.items()))
