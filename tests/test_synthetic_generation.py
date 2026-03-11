from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

import numpy as np
import zarr

from hudm.synthetic_generation import (
    DEFAULT_BOUNDS,
    RolloutDiagnostics,
    build_contact_candidates,
    block_world_vertices,
    reject_rollout,
    resolve_mode_weights,
    score_shadow_rollout,
    validate_spawn_state,
)


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT_PATH = os.path.join(ROOT, "scripts", "generate_synth.py")
SPEC = importlib.util.spec_from_file_location("generate_synth", SCRIPT_PATH)
GENERATE_SYNTH = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(GENERATE_SYNTH)


class SyntheticGenerationTests(unittest.TestCase):
    def test_failed_episode_batch_limit_resolver_supports_unlimited(self):
        self.assertEqual(GENERATE_SYNTH._resolve_max_failed_episode_batches(10, -1), -1)
        self.assertEqual(GENERATE_SYNTH._resolve_max_failed_episode_batches(10, 0), 80)
        self.assertEqual(GENERATE_SYNTH._resolve_max_failed_episode_batches(10, 7), 7)

    def test_contact_aware_ou_offset_is_deterministic(self):
        target = np.asarray([256.0, 256.0], dtype=np.float32)
        ou_state = np.zeros(2, dtype=np.float32)
        perturbed_a, state_a = GENERATE_SYNTH._apply_contact_aware_ou_offset(
            target,
            rng=np.random.default_rng(123),
            ou_state=ou_state,
            theta=0.35,
            sigma=4.0,
            dt=1.0,
            low=DEFAULT_BOUNDS[0],
            high=DEFAULT_BOUNDS[1],
        )
        perturbed_b, state_b = GENERATE_SYNTH._apply_contact_aware_ou_offset(
            target,
            rng=np.random.default_rng(123),
            ou_state=ou_state,
            theta=0.35,
            sigma=4.0,
            dt=1.0,
            low=DEFAULT_BOUNDS[0],
            high=DEFAULT_BOUNDS[1],
        )
        self.assertTrue(np.allclose(perturbed_a, perturbed_b))
        self.assertTrue(np.allclose(state_a, state_b))
        self.assertFalse(np.allclose(perturbed_a, target))
        self.assertTrue(np.all(perturbed_a >= DEFAULT_BOUNDS[0]))
        self.assertTrue(np.all(perturbed_a <= DEFAULT_BOUNDS[1]))

    def test_validate_spawn_state_rejects_wall_overlap(self):
        state = np.asarray([256.0, 256.0, 90.0, 256.0, 0.0, 0.0, 0.0], dtype=np.float32)
        valid, reasons, meta = validate_spawn_state(state, "strict")
        self.assertFalse(valid)
        self.assertTrue("spawn_block_oob" in reasons or "spawn_block_near_wall" in reasons)
        self.assertLess(meta["block_clearance"], 10.0)

    def test_candidate_generation_includes_translate_and_rotate(self):
        rng = np.random.default_rng(0)
        state = np.asarray([160.0, 360.0, 256.0, 256.0, 0.0, 0.0, 0.0], dtype=np.float32)
        translate = build_contact_candidates(state, "translate", rng)
        rotate = build_contact_candidates(state, "rotate", rng)
        self.assertGreaterEqual(len(translate), 3)
        self.assertGreaterEqual(len(rotate), 4)
        self.assertEqual(translate[0].action_abs_seq.shape, (6, 2))
        self.assertEqual(rotate[0].action_abs_seq.shape, (6, 2))

    def test_planning_profile_scores_translate_above_rotate(self):
        state = np.asarray([160.0, 360.0, 256.0, 256.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rng_t = np.random.default_rng(1)
        rng_r = np.random.default_rng(1)
        shadow_env = GENERATE_SYNTH.build_generator_env(with_velocity=True, headless=True)
        translate_candidates = build_contact_candidates(state, "translate", rng_t)
        rotate_candidates = build_contact_candidates(state, "rotate", rng_r)

        best_translate = max(
            score_shadow_rollout(
                "translate",
                GENERATE_SYNTH._simulate_candidate_rollout(
                    shadow_env,
                    current_state=state,
                    candidate=cand,
                    quality_profile="strict",
                    contact_aware_ou_state=None,
                    contact_aware_ou_seed=None,
                    contact_aware_ou_theta=0.35,
                    contact_aware_ou_sigma=0.0,
                    contact_aware_ou_dt=1.0,
                ),
                mode_profile="planning",
                quality_profile="strict",
            )
            for cand in translate_candidates
        )
        best_rotate = max(
            score_shadow_rollout(
                "rotate",
                GENERATE_SYNTH._simulate_candidate_rollout(
                    shadow_env,
                    current_state=state,
                    candidate=cand,
                    quality_profile="strict",
                    contact_aware_ou_state=None,
                    contact_aware_ou_seed=None,
                    contact_aware_ou_theta=0.35,
                    contact_aware_ou_sigma=0.0,
                    contact_aware_ou_dt=1.0,
                ),
                mode_profile="planning",
                quality_profile="strict",
            )
            for cand in rotate_candidates
        )
        self.assertGreater(best_translate, best_rotate)

    def test_reject_rollout_detects_suspicious_no_contact_motion(self):
        diagnostics = RolloutDiagnostics(
            total_block_translation_path=12.0,
            net_block_translation=8.0,
            total_absolute_angle_change=0.1,
            motion_fraction=0.2,
            contact_fraction=0.0,
            first_contact_step=-1,
            first_meaningful_motion_step=0,
            wall_adjacent_fraction=0.0,
            suspicious_no_contact_motion_count=1,
            out_of_bounds_steps=0,
            min_wall_clearance=20.0,
            start_wall_clearance=20.0,
            end_wall_clearance=20.0,
            max_block_step_displacement=5.0,
            num_steps=5,
        )
        rejected, reasons = reject_rollout(diagnostics, mode_profile="balanced", quality_profile="strict")
        self.assertTrue(rejected)
        self.assertIn("episode_glitch_motion", reasons)

    def test_mode_profile_acceptance_differs_between_train_and_planning(self):
        diagnostics = RolloutDiagnostics(
            total_block_translation_path=18.0,
            net_block_translation=18.0,
            total_absolute_angle_change=0.5,
            motion_fraction=0.2,
            contact_fraction=0.2,
            first_contact_step=1,
            first_meaningful_motion_step=1,
            wall_adjacent_fraction=0.0,
            suspicious_no_contact_motion_count=0,
            out_of_bounds_steps=0,
            min_wall_clearance=40.0,
            start_wall_clearance=40.0,
            end_wall_clearance=38.0,
            max_block_step_displacement=3.0,
            num_steps=8,
        )
        train_rejected, _ = reject_rollout(diagnostics, mode_profile="train", quality_profile="strict")
        planning_rejected, planning_reasons = reject_rollout(diagnostics, mode_profile="planning", quality_profile="strict")
        self.assertFalse(train_rejected)
        self.assertTrue(planning_rejected)
        self.assertIn("episode_low_translation", planning_reasons)

    def test_generate_split_retries_after_failed_episode_batch(self):
        original_rollout = GENERATE_SYNTH.rollout_contact_aware_episode
        original_render = GENERATE_SYNTH.render_episode_frames
        calls = {"count": 0}

        def fake_rollout(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] <= 2:
                return GENERATE_SYNTH.EpisodeAttempt(
                    init_state=np.zeros(7, dtype=np.float32),
                    states=np.zeros((1, 7), dtype=np.float32),
                    actions=np.zeros((1, 2), dtype=np.float32),
                    actions_abs=np.zeros((1, 2), dtype=np.float32),
                    diagnostics=RolloutDiagnostics(
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
                        min_wall_clearance=20.0,
                        start_wall_clearance=20.0,
                        end_wall_clearance=20.0,
                        max_block_step_displacement=0.0,
                        num_steps=1,
                    ),
                    rejection_reasons=("episode_low_translation",),
                    selected_families=("translate",),
                    selected_candidates=("translate_0",),
                )
            return GENERATE_SYNTH.EpisodeAttempt(
                init_state=np.zeros(7, dtype=np.float32),
                states=np.zeros((1, 7), dtype=np.float32),
                actions=np.zeros((1, 2), dtype=np.float32),
                actions_abs=np.zeros((1, 2), dtype=np.float32),
                diagnostics=RolloutDiagnostics(
                    total_block_translation_path=40.0,
                    net_block_translation=40.0,
                    total_absolute_angle_change=0.1,
                    motion_fraction=0.5,
                    contact_fraction=0.5,
                    first_contact_step=0,
                    first_meaningful_motion_step=0,
                    wall_adjacent_fraction=0.0,
                    suspicious_no_contact_motion_count=0,
                    out_of_bounds_steps=0,
                    min_wall_clearance=20.0,
                    start_wall_clearance=20.0,
                    end_wall_clearance=20.0,
                    max_block_step_displacement=2.0,
                    num_steps=1,
                ),
                rejection_reasons=(),
                selected_families=("translate",),
                selected_candidates=("translate_0",),
            )

        def fake_render(*args, **kwargs):
            return np.zeros((1, 8, 8, 3), dtype=np.float32)

        try:
            GENERATE_SYNTH.rollout_contact_aware_episode = fake_rollout
            GENERATE_SYNTH.render_episode_frames = fake_render
            _, _, frames, _, stats = GENERATE_SYNTH.generate_split(
                n_eps=1,
                split_name="train",
                len_min=8,
                len_max=8,
                policy="contact_aware",
                action_scale=1.0,
                rng=np.random.default_rng(0),
                decision_env=None,
                shadow_env=None,
                render_env=None,
                with_velocity=True,
                img_size=8,
                mode_profile="planning",
                quality_profile="strict",
                weights=resolve_mode_weights("planning"),
                max_attempts_per_episode=2,
                max_failed_episode_batches=4,
                workers=1,
                contact_aware_ou_theta=0.35,
                contact_aware_ou_sigma=0.0,
                contact_aware_ou_dt=1.0,
                ou_theta=0.15,
                ou_sigma=0.2,
                ou_dt=1.0,
                ou_mu=None,
            )
        finally:
            GENERATE_SYNTH.rollout_contact_aware_episode = original_rollout
            GENERATE_SYNTH.render_episode_frames = original_render

        self.assertEqual(len(frames), 1)
        self.assertEqual(int(stats["accepted_episodes"]), 1)
        self.assertEqual(int(stats["failed_episode_batches"]), 1)
        self.assertEqual(int(stats["rejected_attempts"]), 2)

    def test_generate_tiny_dataset_keeps_schema_and_counts(self):
        rng = np.random.default_rng(5)
        decision_env = GENERATE_SYNTH.build_generator_env(with_velocity=True, headless=True)
        shadow_env = GENERATE_SYNTH.build_generator_env(with_velocity=True, headless=True)
        render_env = GENERATE_SYNTH.build_generator_env(with_velocity=True, headless=False)
        weights = resolve_mode_weights("planning")

        train = GENERATE_SYNTH.generate_split(
            n_eps=1,
            split_name="train",
            len_min=8,
            len_max=10,
            policy="contact_aware",
            action_scale=1.0,
            rng=rng,
            decision_env=decision_env,
            shadow_env=shadow_env,
            render_env=render_env,
            with_velocity=True,
            img_size=32,
            mode_profile="planning",
            quality_profile="strict",
            weights=weights,
            max_attempts_per_episode=32,
            max_failed_episode_batches=0,
            workers=2,
            contact_aware_ou_theta=0.35,
            contact_aware_ou_sigma=0.0,
            contact_aware_ou_dt=1.0,
            ou_theta=0.15,
            ou_sigma=0.2,
            ou_dt=1.0,
            ou_mu=None,
        )
        valid = GENERATE_SYNTH.generate_split(
            n_eps=1,
            split_name="valid",
            len_min=8,
            len_max=10,
            policy="contact_aware",
            action_scale=1.0,
            rng=rng,
            decision_env=decision_env,
            shadow_env=shadow_env,
            render_env=render_env,
            with_velocity=True,
            img_size=32,
            mode_profile="planning",
            quality_profile="strict",
            weights=weights,
            max_attempts_per_episode=32,
            max_failed_episode_batches=0,
            workers=2,
            contact_aware_ou_theta=0.35,
            contact_aware_ou_sigma=0.0,
            contact_aware_ou_dt=1.0,
            ou_theta=0.15,
            ou_sigma=0.2,
            ou_dt=1.0,
            ou_mu=None,
        )

        train_actions, train_actions_abs, train_frames, train_states, train_stats = train
        valid_actions, valid_actions_abs, valid_frames, valid_states, valid_stats = valid

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_out = os.path.join(tmpdir, "synth_test.zarr")
            GENERATE_SYNTH.save_zarr(
                zarr_out,
                train_frames + valid_frames,
                train_actions + valid_actions,
                train_actions_abs + valid_actions_abs,
                train_states + valid_states,
                env_action_scale=float(decision_env.action_scale),
                env_relative=bool(decision_env.relative),
                extra_attrs={
                    "generator_policy": "contact_aware",
                    "mode_profile": "planning",
                    "quality_profile": "strict",
                    "weight_translate": float(weights["translate"]),
                    "weight_rotate": float(weights["rotate"]),
                    "weight_sweep": float(weights["sweep"]),
                    "weight_recover": float(weights["recover"]),
                    "accepted_episodes": 2,
                    "rejected_attempts": int(train_stats["rejected_attempts"]) + int(valid_stats["rejected_attempts"]),
                },
            )
            root = zarr.open_group(zarr_out, mode="r")
            self.assertIn("img", root["data"])
            self.assertIn("action", root["data"])
            self.assertIn("action_abs", root["data"])
            self.assertIn("state", root["data"])
            self.assertIn("episode_ends", root["meta"])
            self.assertEqual(int(root.attrs["accepted_episodes"]), 2)
            self.assertEqual(int(root["meta"]["episode_ends"].shape[0]), 2)
            states = np.asarray(root["data"]["state"][:], dtype=np.float32)
            for state in states:
                vertices = block_world_vertices(state)
                self.assertFalse(np.any(vertices < DEFAULT_BOUNDS[0]) or np.any(vertices > DEFAULT_BOUNDS[1]))

    def test_planning_profile_has_more_translation_than_train(self):
        train_rng = np.random.default_rng(11)
        planning_rng = np.random.default_rng(11)
        train_env = GENERATE_SYNTH.build_generator_env(with_velocity=True, headless=True)
        train_shadow = GENERATE_SYNTH.build_generator_env(with_velocity=True, headless=True)
        planning_env = GENERATE_SYNTH.build_generator_env(with_velocity=True, headless=True)
        planning_shadow = GENERATE_SYNTH.build_generator_env(with_velocity=True, headless=True)

        def collect(mode_profile: str, rng, env, shadow, count: int):
            diags = []
            weights = resolve_mode_weights(mode_profile)
            while len(diags) < count:
                attempt = GENERATE_SYNTH.rollout_contact_aware_episode(
                    env,
                    shadow,
                    T=18,
                    rng=rng,
                    with_velocity=True,
                    mode_profile=mode_profile,
                    quality_profile="strict",
                    weights=weights,
                    spawn_attempts=32,
                    contact_aware_ou_theta=0.35,
                    contact_aware_ou_sigma=0.0,
                    contact_aware_ou_dt=1.0,
                )
                if attempt.accepted:
                    diags.append(attempt.diagnostics)
            return diags

        train_diags = collect("train", train_rng, train_env, train_shadow, 6)
        planning_diags = collect("planning", planning_rng, planning_env, planning_shadow, 6)
        train_translation = np.median([diag.net_block_translation for diag in train_diags])
        planning_translation = np.median([diag.net_block_translation for diag in planning_diags])
        train_angle_median = np.median([diag.total_absolute_angle_change for diag in train_diags])
        planning_angle_median = np.median([diag.total_absolute_angle_change for diag in planning_diags])

        self.assertGreater(planning_translation, train_translation)
        self.assertGreater(train_angle_median, planning_angle_median)


if __name__ == "__main__":
    unittest.main()
