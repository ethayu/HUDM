#!/usr/bin/env python3
"""
Interactive/scripted backend debugger for PushT planning backends.

Examples:
  python scripts/debug_planning_backend.py --config configs/plan.yaml --backend gt_env --keyboard
  python scripts/debug_planning_backend.py --config configs/plan.yaml --backend particle_sim --actions "0.2,0;0.2,0;0,0.2"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import cv2
import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hudm.session import load_plan_cfg
from pusht.pusht_particle_backend import PushTParticleBackend
from pusht.pusht_wrapper import PushTWrapper


def _max_steps_reached(step_idx: int, max_steps: int) -> bool:
    ms = int(max_steps)
    return ms > 0 and step_idx >= ms


def _parse_actions(text: Optional[str]) -> List[np.ndarray]:
    if text is None or str(text).strip() == "":
        return []
    out: List[np.ndarray] = []
    chunks = [c.strip() for c in str(text).split(";") if c.strip()]
    for c in chunks:
        vals = [float(x.strip()) for x in c.split(",") if x.strip()]
        if len(vals) != 2:
            raise ValueError(f"Each action must have 2 values, got '{c}'")
        out.append(np.asarray(vals, dtype=np.float32))
    return out


def _compose_ui(
    image: np.ndarray,
    lines: List[str],
    display_size: int,
    panel_width: int,
    font_scale: float,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.uint8)
    vis = cv2.resize(img, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((display_size, display_size + panel_width, 3), 235, dtype=np.uint8)
    canvas[:, :display_size] = vis
    canvas[:, display_size:] = np.array([28, 28, 30], dtype=np.uint8)

    x0 = display_size + 12
    y = 24
    line_h = max(18, int(round(22 * font_scale / 0.48)))

    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += line_h
        if y > display_size - 8:
            break

    # Divider
    cv2.line(canvas, (display_size, 0), (display_size, display_size - 1), (85, 85, 90), 1)
    return canvas


class GTEnvDebugAdapter:
    def __init__(self, cfg, render_size: int):
        self.cfg = cfg
        env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
        env_kwargs["render_size"] = int(render_size)
        self.env = PushTWrapper(**env_kwargs)
        self.num_levels = int(cfg.fidelity.num_levels)
        self.env.configure_planning_fidelity(
            enabled=True,
            num_levels=self.num_levels,
            cfg=OmegaConf.to_container(cfg.gt_env.fidelity_env, resolve=True),
        )
        self.level_idx = self.num_levels - 1
        self.env.set_planning_fidelity_level(self.level_idx)
        self.goal_state = None
        self.cur_state = None

    def sample_states(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return self.env.sample_random_init_goal_states(seed=int(seed))

    def reset(self, init_state: np.ndarray, goal_state: np.ndarray, seed: int = 0):
        self.goal_state = np.asarray(goal_state, dtype=np.float32)
        self.env.set_task_start(np.asarray(init_state, dtype=np.float32)[2:5])
        self.env.set_task_goal(self.goal_state[2:5])
        obs, state = self.env.prepare(seed=int(seed), init_state=init_state, goal_state=goal_state)
        self.cur_state = np.asarray(state, dtype=np.float32)
        return obs, self.cur_state.copy()

    def set_level(self, level_idx: int) -> None:
        li = int(max(0, min(level_idx, self.num_levels - 1)))
        self.level_idx = li
        self.env.set_planning_fidelity_level(li)

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.cur_state = np.asarray(info["state"], dtype=np.float32)
        return obs, float(reward), bool(done), info

    def render(self) -> np.ndarray:
        return self.env.render("rgb_array", include_start_pose=True)

    def eval_state(self) -> dict:
        return self.env.eval_state(self.goal_state, self.cur_state)

    def fidelity_label(self) -> str:
        return f"L{self.level_idx}/{self.num_levels - 1}"


class ParticleDebugAdapter:
    def __init__(self, cfg, render_size: int):
        self.cfg = cfg
        spacings = [float(s) for s in list(cfg.particle_env.fidelity_env.spacings)]
        self.backend = PushTParticleBackend(
            with_velocity=bool(cfg.env.with_velocity),
            with_target=bool(cfg.env.with_target),
            render_size=int(render_size),
            relative=True,
            action_scale=100.0,
            device=str(cfg.particle_env.fidelity_env.device),
            fidelity_spacings=spacings,
            warp_cfg=OmegaConf.to_container(cfg.particle_env.fidelity_env, resolve=True),
            seed=int(cfg.init_goal.dataset.seed),
        )
        self.num_levels = len(spacings)
        self.level_idx = self.num_levels - 1
        self.backend.set_planning_fidelity_level(self.level_idx)
        self.goal_state = None
        self.cur_state = None

        self._sampler_env = PushTWrapper(**cfg.env)

    def sample_states(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return self._sampler_env.sample_random_init_goal_states(seed=int(seed))

    def reset(self, init_state: np.ndarray, goal_state: np.ndarray, seed: int = 0):
        obs, state = self.backend.prepare(
            seed=int(seed),
            init_state=np.asarray(init_state, dtype=np.float32),
            goal_state=np.asarray(goal_state, dtype=np.float32),
        )
        self.goal_state = np.asarray(goal_state, dtype=np.float32)
        self.cur_state = np.asarray(state, dtype=np.float32)
        return obs, self.cur_state.copy()

    def set_level(self, level_idx: int) -> None:
        li = int(max(0, min(level_idx, self.num_levels - 1)))
        self.level_idx = li
        self.backend.set_planning_fidelity_level(li)

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.backend.step(np.asarray(action, dtype=np.float32))
        self.cur_state = np.asarray(info["state"], dtype=np.float32)
        return obs, float(reward), bool(done), info

    def render(self) -> np.ndarray:
        return self.backend.render("rgb_array", include_start_pose=True)

    def eval_state(self) -> dict:
        return self.backend.eval_state(self.goal_state, self.cur_state)

    def fidelity_label(self) -> str:
        sp = float(self.backend.spacing(self.level_idx))
        nparts = int(self.backend.num_particles(self.level_idx))
        pr = float(self.backend.particle_radius(self.level_idx))
        return f"L{self.level_idx}/{self.num_levels - 1} -> spacing={sp:.4f}, N={nparts}, r={pr:.4f}"


def _handle_control_key(
    key: int,
    adapter,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    seed: int,
):
    if key in (27, ord("q"), ord("Q")):
        return "quit"

    if key in (ord("r"), ord("R")):
        adapter.reset(init_state=init_state, goal_state=goal_state, seed=seed)
        return "redraw"

    if key in (ord("["), ord("-")):
        adapter.set_level(adapter.level_idx - 1)
        return "redraw"

    if key in (ord("]"), ord("=")):
        adapter.set_level(adapter.level_idx + 1)
        return "redraw"

    if ord("0") <= key <= ord("9"):
        level = int(key - ord("0"))
        adapter.set_level(level)
        return "redraw"

    return "none"


def _key_to_action(key: int, key_action_mag: float) -> np.ndarray:
    a = np.zeros((2,), dtype=np.float32)
    if key in (ord("w"), ord("W")):
        a[1] = key_action_mag
    elif key in (ord("s"), ord("S")):
        a[1] = -key_action_mag
    elif key in (ord("a"), ord("A")):
        a[0] = -key_action_mag
    elif key in (ord("d"), ord("D")):
        a[0] = key_action_mag
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/plan.yaml", help="Path to plan config")
    ap.add_argument("--backend", choices=["gt_env", "particle_sim"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fidelity-level", type=int, default=-1, help="Initial level index; -1 means finest")
    ap.add_argument("--actions", type=str, default="", help="Semicolon list of actions: 'ax,ay;ax,ay;...'")
    ap.add_argument("--keyboard", action="store_true", help="Enable realtime key control")
    ap.add_argument("--max-steps", type=int, default=0, help="Max env steps; <=0 means no limit")
    ap.add_argument("--key-action-mag", type=float, default=0.25, help="Action magnitude for WASD keys")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--render-size", type=int, default=224, help="Backend render resolution before display upsampling")
    ap.add_argument("--display-size", type=int, default=560, help="Displayed image side length in pixels")
    ap.add_argument("--panel-width", type=int, default=430, help="HUD side panel width in pixels")
    ap.add_argument("--font-scale", type=float, default=0.58, help="HUD font scale")
    ap.add_argument("--save", type=str, default="", help="Optional GIF path")
    ap.add_argument("--no-window", action="store_true", help="Disable cv2 window display")
    ap.add_argument("--stop-on-done", action="store_true", help="Stop automatically when env returns done=true")
    args = ap.parse_args()

    cfg = load_plan_cfg(args.config)

    backend = str(args.backend or cfg.backend).lower()
    if backend not in {"gt_env", "particle_sim"}:
        raise ValueError(
            f"debug backend must be 'gt_env' or 'particle_sim', got '{backend}'. "
            "Pass --backend explicitly when plan.backend is wm."
        )
    if backend == "gt_env":
        adapter = GTEnvDebugAdapter(cfg, render_size=int(args.render_size))
    else:
        adapter = ParticleDebugAdapter(cfg, render_size=int(args.render_size))

    init_state, goal_state = adapter.sample_states(seed=args.seed)
    adapter.reset(init_state=init_state, goal_state=goal_state, seed=args.seed)

    init_level = adapter.num_levels - 1 if int(args.fidelity_level) < 0 else int(args.fidelity_level)
    adapter.set_level(init_level)

    scripted_actions = _parse_actions(args.actions)
    save_frames = bool(str(args.save).strip())
    frames: List[np.ndarray] = []
    delay_ms = max(1, int(round(1000.0 / max(1e-6, args.fps))))

    def _emit_frame(step_idx: int) -> None:
        frame = adapter.render()
        metrics = adapter.eval_state()
        success = bool(metrics.get("success", False))
        lines = [
            f"backend: {backend}",
            f"step: {step_idx}",
            f"fidelity: {adapter.fidelity_label()}",
            f"success: {success}",
            f"dist: {metrics['state_dist']:.2f}  pos: {metrics['pos_diff']:.2f}  ang: {metrics['angle_diff']:.3f}",
            "",
            "Controls",
            "W/A/S/D: move (realtime)",
            "no key: no-op",
            "[ ] or -/=: fidelity down/up",
            "0-9: set fidelity level",
            "R: reset",
            "Q / Esc: quit",
        ]
        composed = _compose_ui(
            frame,
            lines=lines,
            display_size=int(args.display_size),
            panel_width=int(args.panel_width),
            font_scale=float(args.font_scale),
        )
        if save_frames:
            frames.append(composed)
        if not args.no_window:
            cv2.imshow("backend_debug", cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

    step_idx = 0
    _emit_frame(step_idx)

    exit_reason = "completed"

    # Scripted phase.
    for a in scripted_actions:
        _, _, done, _ = adapter.step(a)
        step_idx += 1
        _emit_frame(step_idx)
        if not args.no_window:
            key = cv2.waitKey(max(1, delay_ms // 3)) & 0xFF
            mode = _handle_control_key(
                key,
                adapter,
                init_state=init_state,
                goal_state=goal_state,
                seed=args.seed,
            )
            if mode == "quit":
                exit_reason = "quit"
                break
            if mode == "redraw":
                _emit_frame(step_idx)
        if bool(args.stop_on_done) and done:
            exit_reason = "done"
            break
        if _max_steps_reached(step_idx, int(args.max_steps)):
            exit_reason = "max_steps"
            break

    # Keyboard/manual phase (continuous realtime stepping; no key => no-op action).
    if args.keyboard and not _max_steps_reached(step_idx, int(args.max_steps)):
        if args.no_window:
            print("[warn] --keyboard requested with --no-window; skipping keyboard loop.")
        else:
            while not _max_steps_reached(step_idx, int(args.max_steps)):
                key = cv2.waitKey(delay_ms) & 0xFF
                mode = _handle_control_key(
                    key,
                    adapter,
                    init_state=init_state,
                    goal_state=goal_state,
                    seed=args.seed,
                )
                if mode == "quit":
                    exit_reason = "quit"
                    break
                action = _key_to_action(key, float(args.key_action_mag))
                _, _, done, _ = adapter.step(action)
                step_idx += 1
                _emit_frame(step_idx)
                if bool(args.stop_on_done) and done:
                    exit_reason = "done"
                    break
            if _max_steps_reached(step_idx, int(args.max_steps)) and exit_reason == "completed":
                exit_reason = "max_steps"

    if not args.no_window:
        cv2.destroyAllWindows()

    if save_frames:
        try:
            import imageio.v2 as imageio
        except Exception:
            import imageio

        if len(frames) > 0:
            imageio.mimwrite(args.save, frames, fps=max(1, int(round(args.fps))))
            print(f"[save] wrote {len(frames)} frames -> {args.save}")

    print(f"[exit] reason={exit_reason} steps={step_idx}")


if __name__ == "__main__":
    main()
