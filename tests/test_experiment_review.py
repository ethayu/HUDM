from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from hudm.experiment_bundle import EXPERIMENT_JSON, PAIRED_VS_BASELINE_CSV, RUNS_CSV, SELECTED_ROLLOUTS_JSON, VARIANTS_CSV
from hudm.experiment_review import (
    _compute_vs_outcome_figure,
    _figure_html,
    _media_description,
    _paired_summary_figure,
    _replan_rows,
    _single_run_trace_figure,
    _variant_relationship_figure,
    _variant_paired_summary_figure,
    _variant_stepwise_figure,
    _variant_success_figure,
    _variant_histogram_figure,
    ExperimentReviewApp,
    MediaRenderTask,
    build_run_media_section,
    build_run_page,
    build_summary_page,
    build_variant_page,
    make_review_handler,
    load_experiment_review_data,
    render_media_for_run,
    resolve_row,
)


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class ExperimentReviewTests(unittest.TestCase):
    def _make_experiment_dir(self, tmpdir: str) -> tuple[str, str]:
        run_dir = os.path.join(tmpdir, "experiment_demo")
        trace_dir = os.path.join(run_dir, "traces", "variant_a", "rollout_0")
        os.makedirs(trace_dir, exist_ok=True)

        with open(os.path.join(run_dir, EXPERIMENT_JSON), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "reviewer_version": 1,
                    "experiment_name": "demo",
                    "baseline_variant": "variant_a",
                    "variant_order": ["variant_a"],
                },
                f,
            )
        with open(os.path.join(run_dir, SELECTED_ROLLOUTS_JSON), "w", encoding="utf-8") as f:
            json.dump([{"rollout_id": "rollout_0"}], f)
        _write_csv(
            os.path.join(run_dir, VARIANTS_CSV),
            [
                {
                    "variant_name": "variant_a",
                    "n_rollouts": 1,
                    "success_rate": 1.0,
                    "mean_final_pos_diff": 1.25,
                    "mean_final_angle_diff": 0.1,
                    "mean_final_eef_diff": 0.2,
                    "mean_final_coverage": 0.96,
                    "mean_bits_used_total": 100.0,
                    "mean_flops_used_total": 200.0,
                    "mean_plan_time_total_sec": 0.50,
                    "paired_success_wins_vs_baseline": 0,
                    "paired_success_losses_vs_baseline": 0,
                    "paired_success_ties_vs_baseline": 1,
                    "paired_final_pos_better_vs_baseline": 0,
                    "paired_final_pos_worse_vs_baseline": 0,
                    "paired_final_pos_ties_vs_baseline": 1,
                }
            ],
        )
        _write_csv(
            os.path.join(run_dir, RUNS_CSV),
            [
                {
                    "variant_name": "variant_a",
                    "rollout_id": "rollout_0",
                    "rollout_index": 0,
                    "success": 1,
                    "termination_reason": "env_done",
                    "final_pos_diff": 1.25,
                    "final_angle_diff": 0.1,
                    "final_eef_diff": 0.2,
                    "final_coverage": 0.96,
                    "bits_used_total": 100.0,
                    "flops_used_total": 200.0,
                    "plan_time_total_sec": 0.50,
                    "executed_steps": 3,
                    "plans": 1,
                }
            ],
        )
        _write_csv(
            os.path.join(run_dir, PAIRED_VS_BASELINE_CSV),
            [
                {
                    "variant_name": "variant_a",
                    "rollout_id": "rollout_0",
                    "success_delta": 0,
                    "final_pos_diff_delta": 0.0,
                    "final_coverage_delta": 0.0,
                    "bits_used_total_delta": 0.0,
                    "plan_time_total_sec_delta": 0.0,
                }
            ],
        )
        with open(os.path.join(trace_dir, "trace.json"), "w", encoding="utf-8") as f:
            json.dump({"replans": [{"replan_idx": 0, "step_start": 0, "mpc_progress": 0.0, "base_level_idx": 0, "bits_used_estimate": 0, "plan_time_sec": 0.1, "action_seq": [[0.0, 0.0]]}]}, f)
        with open(os.path.join(trace_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"ok": True}, f)
        with open(os.path.join(trace_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write("step log\n")
        npz_data = {
            "pos_diffs": [1.25, 1.0, 0.75],
            "angle_diffs": [0.1, 0.09, 0.08],
            "eef_diffs": [0.2, 0.2, 0.2],
            "coverages": [0.8, 0.9, 0.96],
            "executed_actions": [[0.5, 0.0]],
            "trajectory": [[256.0, 256.0, 0, 0, 0], [306.0, 256.0, 0, 0, 0]],
        }
        import numpy as np

        np.savez_compressed(os.path.join(trace_dir, "trace.npz"), **{k: np.asarray(v, dtype=np.float32) for k, v in npz_data.items()})
        return run_dir, trace_dir

    def _write_partial_trace(
        self,
        run_dir: str,
        *,
        variant_name: str,
        rollout_id: str,
        success: bool,
        final_pos_diff: float,
        final_angle_diff: float,
        final_eef_diff: float,
        final_coverage: float,
        bits_used_total: int,
        flops_used_total: int,
        plan_time_total_sec: float,
        termination_reason: str,
        episode_index: int,
        start_index: int,
        goal_index: int,
    ) -> str:
        import numpy as np

        trace_dir = os.path.join(run_dir, "traces", variant_name, rollout_id)
        os.makedirs(trace_dir, exist_ok=True)
        trace_payload = {
            "backend": "particle_sim",
            "success": bool(success),
            "schedule_name": variant_name,
            "plan_config": {"backend": "particle_sim"},
            "sample": {
                "episode_index": episode_index,
                "start_index": start_index,
                "goal_index": goal_index,
            },
            "run_stats": {
                "plans": 2,
                "bits_used_total": int(bits_used_total),
                "flops_used_total": int(flops_used_total),
                "plan_time_total_sec": float(plan_time_total_sec),
                "termination_reason": termination_reason,
                "termination_step": 2,
                "termination_metric_success": bool(success),
                "termination_done": bool(success),
                "termination_pos_diff": float(final_pos_diff),
                "termination_angle_diff": float(final_angle_diff),
                "termination_eef_diff": float(final_eef_diff),
                "termination_coverage": float(final_coverage),
            },
            "replans": [
                {
                    "replan_idx": 0,
                    "step_start": 0,
                    "mpc_progress": 0.0,
                    "base_level_idx": 0,
                    "bits_used_estimate": bits_used_total,
                    "plan_time_sec": float(plan_time_total_sec) / 2.0,
                    "action_seq": [[0.0, 0.0]],
                }
            ],
        }
        metadata_payload = {
            "created_at": "20260331_120000",
            "backend": "particle_sim",
            "source": "dataset",
            "success": bool(success),
            "planned_steps": 3,
            "plans": 2,
            "bits_used_total": int(bits_used_total),
            "flops_used_total": int(flops_used_total),
            "plan_time_total_sec": float(plan_time_total_sec),
            "shared_plan_time_total_sec": float(plan_time_total_sec) * 0.75,
            "termination_reason": termination_reason,
            "termination_step": 2,
            "termination_metric_success": bool(success),
            "termination_done": bool(success),
            "termination_pos_diff": float(final_pos_diff),
            "termination_angle_diff": float(final_angle_diff),
            "termination_eef_diff": float(final_eef_diff),
            "termination_coverage": float(final_coverage),
            "sample": {
                "episode_index": episode_index,
                "start_index": start_index,
                "goal_index": goal_index,
            },
            "trace_json": "trace.json",
            "trace_npz": "trace.npz",
        }
        npz_data = {
            "pos_diffs": np.asarray([final_pos_diff + 0.5, final_pos_diff + 0.2, final_pos_diff], dtype=np.float32),
            "angle_diffs": np.asarray([final_angle_diff + 0.05, final_angle_diff + 0.02, final_angle_diff], dtype=np.float32),
            "eef_diffs": np.asarray([final_eef_diff + 0.1, final_eef_diff + 0.04, final_eef_diff], dtype=np.float32),
            "coverages": np.asarray([max(0.0, final_coverage - 0.1), max(0.0, final_coverage - 0.03), final_coverage], dtype=np.float32),
            "executed_actions": np.asarray([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=np.float32),
            "trajectory": np.asarray(
                [
                    [256.0, 256.0, 0.0, 0.0, 0.0],
                    [266.0, 256.0, 0.0, 0.0, 0.0],
                    [276.0, 256.0, 0.0, 0.0, 0.0],
                    [286.0, 256.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        }
        with open(os.path.join(trace_dir, "trace.json"), "w", encoding="utf-8") as f:
            json.dump(trace_payload, f)
        with open(os.path.join(trace_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f)
        with open(os.path.join(trace_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write("partial run log\n")
        np.savez_compressed(os.path.join(trace_dir, "trace.npz"), **npz_data)
        return trace_dir

    def _make_partial_experiment_dir(self, tmpdir: str, *, include_second_variant: bool = False) -> str:
        run_dir = os.path.join(tmpdir, "experiment_partial")
        self._write_partial_trace(
            run_dir,
            variant_name="variant_a",
            rollout_id="rollout_0",
            success=True,
            final_pos_diff=1.0,
            final_angle_diff=0.1,
            final_eef_diff=0.2,
            final_coverage=0.95,
            bits_used_total=100,
            flops_used_total=200,
            plan_time_total_sec=0.5,
            termination_reason="env_done",
            episode_index=7,
            start_index=11,
            goal_index=19,
        )
        if include_second_variant:
            self._write_partial_trace(
                run_dir,
                variant_name="variant_b",
                rollout_id="rollout_0",
                success=False,
                final_pos_diff=3.5,
                final_angle_diff=0.3,
                final_eef_diff=0.4,
                final_coverage=0.55,
                bits_used_total=250,
                flops_used_total=400,
                plan_time_total_sec=1.25,
                termination_reason="max_steps",
                episode_index=7,
                start_index=11,
                goal_index=19,
            )
        os.makedirs(os.path.join(run_dir, "traces", "variant_a", "rollout_pending"), exist_ok=True)
        return run_dir

    def test_build_summary_page_has_variant_links_and_downloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            data = load_experiment_review_data(run_dir)
            html = build_summary_page(data)
            self.assertIn("demo", html)
            self.assertIn("/variant?name=variant_a", html)
            self.assertIn("/files/experiment.json", html)
            self.assertNotIn("experiment_report.html", html)
            self.assertIn("plot-shell", html)
            self.assertIn("minmax(min(420px, 100%), 1fr)", html)
            self.assertIn("width: min(100%, 320px)", html)
            self.assertIn("resizePlotlyFigures", html)
            self.assertIn("100.00 b", html)
            self.assertIn("Mean FLOPs", html)
            self.assertIn("200.00 FLOPs", html)
            self.assertNotIn("Cross-Variant Compute Distributions", html)
            self.assertNotIn("Final Pos", html)
            self.assertNotIn("kpi-label'>Baseline", html)

    def test_load_experiment_review_data_falls_back_to_completed_partial_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_partial_experiment_dir(tmpdir, include_second_variant=True)
            data = load_experiment_review_data(run_dir)

            self.assertEqual(data.experiment_name, "experiment_partial")
            self.assertTrue(bool(data.meta.get("partial_bundle")))
            self.assertEqual(data.variant_order, ["variant_a", "variant_b"])
            self.assertEqual(len(data.run_rows), 2)
            self.assertEqual(len(data.variant_rows), 2)
            self.assertEqual(sorted(row["rollout_id"] for row in data.run_rows), ["rollout_0", "rollout_0"])
            self.assertEqual(resolve_row(data, "variant_a", "rollout_0")["rollout_index"], 0)
            self.assertAlmostEqual(data.variant_by_name["variant_a"]["mean_final_coverage"], 0.95)
            self.assertAlmostEqual(data.variant_by_name["variant_b"]["mean_final_coverage"], 0.55)
            self.assertEqual(len(data.paired_rows), 1)

            html = build_summary_page(data)
            self.assertIn("Reference Variant", html)
            self.assertIn("/variant?name=variant_a", html)
            self.assertIn("/variant?name=variant_b", html)
            self.assertNotIn("rollout_pending", html)

    def test_experiment_review_app_refreshes_partial_trace_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_partial_experiment_dir(tmpdir, include_second_variant=False)
            app = ExperimentReviewApp(run_dir)
            self.assertEqual(len(app.data.run_rows), 1)

            self._write_partial_trace(
                run_dir,
                variant_name="variant_a",
                rollout_id="rollout_1",
                success=True,
                final_pos_diff=0.8,
                final_angle_diff=0.08,
                final_eef_diff=0.18,
                final_coverage=0.97,
                bits_used_total=90,
                flops_used_total=180,
                plan_time_total_sec=0.45,
                termination_reason="env_done",
                episode_index=8,
                start_index=12,
                goal_index=20,
            )

            html = app.summary_page()
            self.assertEqual(len(app.data.run_rows), 2)
            self.assertIn("rollout_1", html)

    def test_replan_rows_use_human_bits_and_saved_action_horizon(self):
        rows = _replan_rows(
            {
                "replans": [
                    {
                        "replan_idx": 2,
                        "step_start": 5,
                        "mpc_progress": 0.5,
                        "start_level_idx": 1,
                        "base_level_idx": 3,
                        "bits_used_estimate": 94264885248,
                        "plan_time_sec": 1.25,
                        "action_horizon": 15,
                    }
                ]
            }
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action_horizon"], 15)
        self.assertEqual(rows[0]["base_level_idx"], 1)
        self.assertEqual(rows[0]["bits_used_estimate__display"], "94.26 Gb")

    def test_replan_rows_fallback_to_rollout_level_length_when_action_horizon_missing(self):
        rows = _replan_rows(
            {
                "replans": [
                    {
                        "replan_idx": 0,
                        "step_start": 0,
                        "mpc_progress": 0.0,
                        "base_level_idx": 1,
                        "bits_used_estimate": 1000,
                        "plan_time_sec": 0.1,
                        "rollout_level_indices": [3, 3, 3, 3],
                    }
                ]
            }
        )
        self.assertEqual(rows[0]["action_horizon"], 4)
        self.assertEqual(rows[0]["bits_used_estimate__display"], "1.00 Kb")

    def test_replan_rows_derive_start_level_from_plan_config_when_missing(self):
        rows = _replan_rows(
            {
                "plan_config": {
                    "mpc": {"horizon": 10},
                    "cem": {"pop_size": 16, "elite_frac": 0.25, "n_iter": 4, "init_std": 1.0},
                    "fidelity": {
                        "enabled": True,
                        "num_levels": 4,
                        "mpc": {"mode": "linear", "start_level": "coarsest", "end_level": "finest"},
                        "cem": {"mode": "linear", "start_level": "base", "end_level": "finest"},
                        "rollout": {"mode": "fixed", "level": "base"},
                    },
                },
                "replans": [
                    {
                        "replan_idx": 1,
                        "step_start": 5,
                        "mpc_progress": 2.0 / 3.0,
                        "base_level_idx": 3,
                        "bits_used_estimate": 1000,
                        "plan_time_sec": 0.1,
                        "action_seq": [[0.0, 0.0]],
                    }
                ],
            }
        )
        self.assertEqual(rows[0]["base_level_idx"], 2)

    def test_build_variant_page_contains_success_and_stepwise_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            data = load_experiment_review_data(run_dir)
            html = build_variant_page(data, "variant_a")
            self.assertIn("Success Analysis", html)
            self.assertIn("Stepwise Trace Summary", html)
            self.assertIn("variantRunsTable", html)
            self.assertIn("Mean FLOPs", html)
            self.assertIn("200.00 FLOPs", html)
            self.assertNotIn("Final Pos", html)
            self.assertNotIn("Reference Comparison", html)

    def test_build_summary_page_uses_shared_success_analysis_controls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            _write_csv(
                os.path.join(run_dir, VARIANTS_CSV),
                [
                    {
                        "variant_name": "variant_a",
                        "n_rollouts": 1,
                        "success_rate": 1.0,
                        "mean_final_pos_diff": 1.25,
                        "mean_final_angle_diff": 0.1,
                        "mean_final_eef_diff": 0.2,
                        "mean_final_coverage": 0.96,
                        "mean_bits_used_total": 100.0,
                        "mean_flops_used_total": 200.0,
                        "mean_plan_time_total_sec": 0.50,
                        "paired_success_wins_vs_baseline": 0,
                        "paired_success_losses_vs_baseline": 0,
                        "paired_success_ties_vs_baseline": 1,
                    },
                    {
                        "variant_name": "variant_b",
                        "n_rollouts": 1,
                        "success_rate": 0.0,
                        "mean_final_pos_diff": 3.0,
                        "mean_final_angle_diff": 0.2,
                        "mean_final_eef_diff": 0.4,
                        "mean_final_coverage": 0.50,
                        "mean_bits_used_total": 200.0,
                        "mean_flops_used_total": 400.0,
                        "mean_plan_time_total_sec": 1.00,
                        "paired_success_wins_vs_baseline": 2,
                        "paired_success_losses_vs_baseline": 3,
                        "paired_success_ties_vs_baseline": 4,
                    },
                ],
            )
            _write_csv(
                os.path.join(run_dir, RUNS_CSV),
                [
                    {
                        "variant_name": "variant_a",
                        "rollout_id": "rollout_0",
                        "rollout_index": 0,
                        "success": 1,
                        "termination_reason": "env_done",
                        "final_pos_diff": 1.25,
                        "final_angle_diff": 0.1,
                        "final_eef_diff": 0.2,
                        "final_coverage": 0.96,
                        "bits_used_total": 100.0,
                        "flops_used_total": 200.0,
                        "plan_time_total_sec": 0.50,
                        "executed_steps": 3,
                        "plans": 1,
                    },
                    {
                        "variant_name": "variant_b",
                        "rollout_id": "rollout_1",
                        "rollout_index": 1,
                        "success": 0,
                        "termination_reason": "max_steps",
                        "final_pos_diff": 5.0,
                        "final_angle_diff": 0.5,
                        "final_eef_diff": 0.7,
                        "final_coverage": 0.40,
                        "bits_used_total": 200.0,
                        "flops_used_total": 400.0,
                        "plan_time_total_sec": 2.0,
                        "executed_steps": 3,
                        "plans": 2,
                    },
                ],
            )
            experiment_path = os.path.join(run_dir, EXPERIMENT_JSON)
            with open(experiment_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": 1,
                        "reviewer_version": 1,
                        "experiment_name": "demo",
                        "baseline_variant": "variant_a",
                        "variant_order": ["variant_a", "variant_b"],
                    },
                    f,
                )

            data = load_experiment_review_data(run_dir)
            html = build_summary_page(data)
            self.assertIn("overview-success-display", html)
            self.assertIn("data-barmode-group='overview-success-analysis'", html)
            self.assertIn("overview-reference-select", html)
            self.assertIn("Reference Variant", html)
            success_analysis_section = html.split("Variant Success Analysis", 1)[1].split("Compute vs Outcome", 1)[0]
            self.assertNotIn("data-barmode-target=", success_analysis_section)
            self.assertNotIn("kpi-label'>Baseline", html)

    def test_render_media_for_run_writes_into_review_cache_and_run_page_links_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)

            def fake_render(experiment_root, *, schedule, rollout_id, media, output_dir):
                self.assertEqual(os.path.abspath(experiment_root), os.path.abspath(run_dir))
                self.assertEqual(schedule, "variant_a")
                self.assertEqual(rollout_id, "rollout_0")
                self.assertEqual(media, ["closed_loop_replay"])
                self.assertIn(os.path.join("review_cache", "media", "variant_a", "rollout_0"), output_dir)
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, "closed_loop_replay.mp4")
                with open(out_path, "wb") as f:
                    f.write(b"fake_video")
                return [out_path]

            with mock.patch("hudm.experiment_review.planning_media.render_media", side_effect=fake_render):
                outputs, errors = render_media_for_run(
                    run_dir,
                    variant_name="variant_a",
                    rollout_id="rollout_0",
                    media=["closed_loop_replay"],
                )
            self.assertEqual(len(outputs), 1)
            self.assertEqual(errors, [])

            data = load_experiment_review_data(run_dir)
            row = resolve_row(data, "variant_a", "rollout_0")
            detail_html = build_run_page(data, row, notice="Rendered 1 media artifact(s).")
            self.assertIn("Rendered 1 media artifact(s).", detail_html)
            self.assertIn("/files/review_cache/media/variant_a/rollout_0/closed_loop_replay.mp4", detail_html)
            self.assertIn("Single-Run Trace Curves", detail_html)
            self.assertIn("media-grid", detail_html)
            self.assertIn("media-card", detail_html)
            self.assertIn("data-async-render='true'", detail_html)
            self.assertIn(
                "data-tooltip='s_{t+1} = f(s_t, a_t), where f is the GT environment transition. States are rendered in the GT environment. Here T is bounded by plan.budget.max_env_steps, and is smaller if the rollout terminates early.'",
                detail_html,
            )
            self.assertIn(
                "s_{t+1} = f(s_t, a_t), where f is the GT environment transition. States are rendered in the GT environment. Here T is bounded by plan.budget.max_env_steps, and is smaller if the rollout terminates early.",
                detail_html,
            )

    def test_variant_histogram_legend_groups_cover_every_subplot(self):
        rows = [
            {"success": 1, "final_pos_diff": 1.0, "final_coverage": 0.95, "executed_steps": 2, "plans": 1},
            {"success": 1, "final_pos_diff": 1.5, "final_coverage": 0.97, "executed_steps": 3, "plans": 1},
            {"success": 0, "final_pos_diff": 6.0, "final_coverage": 0.65, "executed_steps": 9, "plans": 4},
            {"success": 0, "final_pos_diff": 8.0, "final_coverage": 0.55, "executed_steps": 10, "plans": 5},
        ]
        fig = _variant_histogram_figure(
            rows,
            metric_specs=[
                ("final_pos_diff", "Final Pos Diff"),
                ("final_coverage", "Final Coverage"),
                ("executed_steps", "Executed Steps"),
                ("plans", "Plans"),
            ],
            title="demo",
        )

        success_traces = [trace for trace in fig.data if trace.name == "success"]
        failure_traces = [trace for trace in fig.data if trace.name == "failure"]

        self.assertEqual(len(success_traces), 4)
        self.assertEqual(len(failure_traces), 4)
        self.assertEqual({trace.legendgroup for trace in success_traces}, {"success"})
        self.assertEqual({trace.legendgroup for trace in failure_traces}, {"failure"})
        self.assertTrue(success_traces[0].showlegend)
        self.assertFalse(success_traces[1].showlegend)
        self.assertTrue(failure_traces[0].showlegend)
        self.assertFalse(failure_traces[1].showlegend)
        self.assertGreaterEqual(fig.layout.height, 700)

    def test_final_coverage_histograms_use_explicit_point_zero_two_bins(self):
        rows = [
            {"success": 1, "final_coverage": 0.95},
            {"success": 0, "final_coverage": 0.55},
        ]
        success_fig = _variant_success_figure(rows, "variant_a")
        coverage_histograms = [trace for trace in success_fig.data if getattr(trace, "type", "") == "histogram"]
        self.assertTrue(coverage_histograms)
        for trace in coverage_histograms:
            self.assertEqual(trace.xbins.start, 0.0)
            self.assertEqual(trace.xbins.end, 1.0)
            self.assertEqual(trace.xbins.size, 0.02)

        metric_fig = _variant_histogram_figure(
            rows,
            metric_specs=[("final_coverage", "Final Coverage")],
            title="demo",
        )
        metric_histograms = [trace for trace in metric_fig.data if getattr(trace, "type", "") == "histogram"]
        self.assertTrue(metric_histograms)
        for trace in metric_histograms:
            self.assertEqual(trace.xbins.start, 0.0)
            self.assertEqual(trace.xbins.end, 1.0)
            self.assertEqual(trace.xbins.size, 0.02)

    def test_media_descriptions_use_backend_label_from_trace_meta(self):
        trace_meta = {
            "backend": "wm",
            "plan_config": {
                "backend": "wm",
                "world_model": {"run_dir": "/tmp/checkpoints/world_model_demo"},
            },
        }
        planner_desc = _media_description("planner_view_replay", trace_meta=trace_meta)
        predicted_desc = _media_description("predicted_backend_replay", trace_meta=trace_meta)
        self.assertEqual(
            planner_desc,
            "s_{t+1} = f(s_t, a_t), where f is the GT environment transition. "
            "States are rendered in the planner backend (wm (world_model_demo)). "
            "Here T is bounded by plan.budget.max_env_steps, and is smaller if the rollout terminates early.",
        )
        self.assertEqual(
            predicted_desc,
            "s_{t+1} = f(s_t, a_t), where f is the planner backend (wm (world_model_demo)). "
            "States are rendered in the planner backend (wm (world_model_demo)). "
            "Here T is plan.planner.horizon * num_replans.",
        )

    def test_figure_html_adds_barmode_toggle_only_for_stackable_figures(self):
        histogram_fig = _variant_histogram_figure(
            [
                {"success": 1, "final_coverage": 0.95},
                {"success": 0, "final_coverage": 0.55},
            ],
            metric_specs=[("final_coverage", "Final Coverage")],
            title="demo",
        )
        histogram_html = _figure_html(histogram_fig)
        self.assertIn("data-barmode-target=", histogram_html)
        self.assertIn(">Stacked</option>", histogram_html)
        self.assertIn(">Overlay</option>", histogram_html)
        histogram_html_without_toggle = _figure_html(histogram_fig, show_barmode_toggle=False)
        self.assertNotIn("data-barmode-target=", histogram_html_without_toggle)

        relationship_html = _figure_html(
            _variant_relationship_figure(
                [
                    {"rollout_id": "success_row", "success": 1, "final_coverage": 0.95, "bits_used_total": 100.0, "plan_time_total_sec": 1.0},
                    {"rollout_id": "failure_row", "success": 0, "final_coverage": 0.40, "bits_used_total": 200.0, "plan_time_total_sec": 2.0},
                ],
                "variant_a",
            )
        )
        self.assertNotIn("data-barmode-target=", relationship_html)

    def test_overview_and_success_analysis_hide_default_trace_legends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            data = load_experiment_review_data(run_dir)

            overview_fig = _compute_vs_outcome_figure(data)
            self.assertTrue(all(not trace.showlegend for trace in overview_fig.data))
            self.assertTrue(all("<extra></extra>" in (trace.hovertemplate or "") for trace in overview_fig.data))
            self.assertEqual(len(overview_fig.data), 4)
            self.assertEqual(overview_fig.layout.annotations[0].text, "Bits vs Success")
            self.assertEqual(overview_fig.layout.annotations[1].text, "Plan Time vs Success")
            self.assertEqual(overview_fig.layout.annotations[2].text, "Bits vs Coverage")
            self.assertEqual(overview_fig.layout.annotations[3].text, "Plan Time vs Coverage")
            hovertemplates = [trace.hovertemplate for trace in overview_fig.data]
            self.assertIn("Success Rate: %{y:.1%}<extra></extra>", hovertemplates[0])
            self.assertIn("Mean Coverage: %{y:.4f}<extra></extra>", hovertemplates[2])

            variant_fig = _variant_success_figure(data.run_rows, "variant_a")
            self.assertTrue(all(not trace.showlegend for trace in variant_fig.data))
            hovertemplates = [trace.hovertemplate for trace in variant_fig.data]
            self.assertEqual(hovertemplates[0], "Outcome: %{x}<br>Runs: %{y}<extra></extra>")
            self.assertEqual(hovertemplates[1], "Termination: %{x}<br>Runs: %{y}<extra></extra>")
            self.assertIn("Final Coverage: %{x:.4f}<br>Runs: %{y}<extra></extra>", hovertemplates)
            self.assertEqual(
                hovertemplates[-1],
                "Success Probability: %{x:.1%}<br>Posterior Density: %{y:.4f}<extra></extra>",
            )

    def test_variant_paired_summary_hides_plotly_default_trace_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            variants_path = os.path.join(run_dir, VARIANTS_CSV)
            _write_csv(
                variants_path,
                [
                    {
                        "variant_name": "variant_a",
                        "n_rollouts": 1,
                        "success_rate": 1.0,
                        "mean_final_pos_diff": 1.25,
                        "mean_final_angle_diff": 0.1,
                        "mean_final_eef_diff": 0.2,
                        "mean_final_coverage": 0.96,
                        "mean_bits_used_total": 100.0,
                        "mean_flops_used_total": 200.0,
                        "mean_plan_time_total_sec": 0.50,
                        "paired_success_wins_vs_baseline": 0,
                        "paired_success_losses_vs_baseline": 0,
                        "paired_success_ties_vs_baseline": 1,
                        "paired_final_pos_better_vs_baseline": 0,
                        "paired_final_pos_worse_vs_baseline": 0,
                        "paired_final_pos_ties_vs_baseline": 1,
                    },
                    {
                        "variant_name": "variant_b",
                        "n_rollouts": 1,
                        "success_rate": 0.0,
                        "mean_final_pos_diff": 3.0,
                        "mean_final_angle_diff": 0.2,
                        "mean_final_eef_diff": 0.4,
                        "mean_final_coverage": 0.50,
                        "mean_bits_used_total": 200.0,
                        "mean_flops_used_total": 400.0,
                        "mean_plan_time_total_sec": 1.00,
                        "paired_success_wins_vs_baseline": 2,
                        "paired_success_losses_vs_baseline": 3,
                        "paired_success_ties_vs_baseline": 4,
                        "paired_final_pos_better_vs_baseline": 5,
                        "paired_final_pos_worse_vs_baseline": 6,
                        "paired_final_pos_ties_vs_baseline": 7,
                    },
                ],
            )
            experiment_path = os.path.join(run_dir, EXPERIMENT_JSON)
            with open(experiment_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": 1,
                        "reviewer_version": 1,
                        "experiment_name": "demo",
                        "baseline_variant": "variant_a",
                        "variant_order": ["variant_a", "variant_b"],
                    },
                    f,
                )

            data = load_experiment_review_data(run_dir)
            _write_csv(
                os.path.join(run_dir, RUNS_CSV),
                [
                    {
                        "variant_name": "variant_a",
                        "rollout_id": "rollout_0",
                        "rollout_index": 0,
                        "success": 1,
                        "termination_reason": "env_done",
                        "final_pos_diff": 1.25,
                        "final_angle_diff": 0.1,
                        "final_eef_diff": 0.2,
                        "final_coverage": 0.96,
                        "bits_used_total": 100.0,
                        "flops_used_total": 200.0,
                        "plan_time_total_sec": 0.50,
                        "executed_steps": 3,
                        "plans": 1,
                    },
                    {
                        "variant_name": "variant_b",
                        "rollout_id": "rollout_0",
                        "rollout_index": 0,
                        "success": 0,
                        "termination_reason": "max_steps",
                        "final_pos_diff": 3.0,
                        "final_angle_diff": 0.2,
                        "final_eef_diff": 0.4,
                        "final_coverage": 0.50,
                        "bits_used_total": 200.0,
                        "flops_used_total": 400.0,
                        "plan_time_total_sec": 1.00,
                        "executed_steps": 3,
                        "plans": 2,
                    },
                ],
            )
            data = load_experiment_review_data(run_dir)
            fig = _variant_paired_summary_figure(data, "variant_b", "variant_a")
            self.assertIsNotNone(fig)
            assert fig is not None
            self.assertFalse(fig.layout.showlegend)
            self.assertEqual(len(fig.data), 2)
            self.assertEqual(fig.layout.annotations[0].text, "Success vs variant_a")
            self.assertEqual(fig.layout.annotations[1].text, "Coverage vs variant_a")
            self.assertTrue(all(not trace.showlegend for trace in fig.data))
            self.assertTrue(all((trace.hovertemplate or "").endswith("<extra></extra>") for trace in fig.data))

    def test_variant_relationship_and_stepwise_expose_success_failure_legend_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            failure_trace_dir = os.path.join(run_dir, "traces", "variant_a", "rollout_1")
            os.makedirs(failure_trace_dir, exist_ok=True)
            _write_csv(
                os.path.join(run_dir, RUNS_CSV),
                [
                    {
                        "variant_name": "variant_a",
                        "rollout_id": "rollout_0",
                        "rollout_index": 0,
                        "success": 1,
                        "termination_reason": "env_done",
                        "final_pos_diff": 1.25,
                        "final_angle_diff": 0.1,
                        "final_eef_diff": 0.2,
                        "final_coverage": 0.96,
                        "bits_used_total": 100.0,
                        "flops_used_total": 200.0,
                        "plan_time_total_sec": 0.50,
                        "executed_steps": 3,
                        "plans": 1,
                    },
                    {
                        "variant_name": "variant_a",
                        "rollout_id": "rollout_1",
                        "rollout_index": 1,
                        "success": 0,
                        "termination_reason": "max_steps",
                        "final_pos_diff": 5.0,
                        "final_angle_diff": 0.5,
                        "final_eef_diff": 0.7,
                        "final_coverage": 0.4,
                        "bits_used_total": 200.0,
                        "flops_used_total": 400.0,
                        "plan_time_total_sec": 2.0,
                        "executed_steps": 3,
                        "plans": 2,
                    },
                ],
            )
            with open(os.path.join(failure_trace_dir, "trace.json"), "w", encoding="utf-8") as f:
                json.dump({"replans": []}, f)
            with open(os.path.join(failure_trace_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump({"ok": True}, f)
            with open(os.path.join(failure_trace_dir, "run.log"), "w", encoding="utf-8") as f:
                f.write("failure log\n")
            import numpy as np

            np.savez_compressed(
                os.path.join(failure_trace_dir, "trace.npz"),
                pos_diffs=np.asarray([5.0, 4.0, 3.0], dtype=np.float32),
                angle_diffs=np.asarray([0.5, 0.45, 0.4], dtype=np.float32),
                eef_diffs=np.asarray([0.7, 0.65, 0.6], dtype=np.float32),
                coverages=np.asarray([0.2, 0.3, 0.4], dtype=np.float32),
            )
            data = load_experiment_review_data(run_dir)
            rows = [
                {
                    "rollout_id": "success_row",
                    "success": 1,
                    "final_pos_diff": 1.0,
                    "final_coverage": 0.95,
                    "bits_used_total": 100.0,
                    "plan_time_total_sec": 1.0,
                    "executed_steps": 3,
                },
                {
                    "rollout_id": "failure_row",
                    "success": 0,
                    "final_pos_diff": 5.0,
                    "final_coverage": 0.40,
                    "bits_used_total": 200.0,
                    "plan_time_total_sec": 2.0,
                    "executed_steps": 3,
                },
            ]

            relationship_fig = _variant_relationship_figure(rows, "variant_a")
            shown_relationship_legends = [trace for trace in relationship_fig.data if trace.showlegend]
            self.assertEqual([trace.name for trace in shown_relationship_legends], ["success", "failure"])
            self.assertEqual({trace.legendgroup for trace in relationship_fig.data}, {"success", "failure"})
            self.assertEqual(relationship_fig.layout.annotations[0].text, "Bits vs Coverage")
            self.assertEqual(relationship_fig.layout.annotations[1].text, "Plan Time vs Coverage")
            self.assertEqual(relationship_fig.layout.annotations[2].text, "Bits vs Plan Time")

            stepwise_fig = _variant_stepwise_figure(data, "variant_a")
            shown_stepwise_legends = [trace for trace in stepwise_fig.data if trace.showlegend]
            self.assertEqual([trace.name for trace in shown_stepwise_legends], ["success", "failure"])
            self.assertEqual(
                {trace.legendgroup for trace in stepwise_fig.data if trace.legendgroup is not None},
                {"success", "failure"},
            )

    def test_paired_summary_figure_is_hidden_when_no_matched_reference_runs_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            variants_path = os.path.join(run_dir, VARIANTS_CSV)
            _write_csv(
                variants_path,
                [
                    {
                        "variant_name": "variant_a",
                        "n_rollouts": 1,
                        "success_rate": 1.0,
                        "mean_final_pos_diff": 1.25,
                        "mean_final_angle_diff": 0.1,
                        "mean_final_eef_diff": 0.2,
                        "mean_final_coverage": 0.96,
                        "mean_bits_used_total": 100.0,
                        "mean_flops_used_total": 200.0,
                        "mean_plan_time_total_sec": 0.50,
                        "paired_success_wins_vs_baseline": 0,
                        "paired_success_losses_vs_baseline": 0,
                        "paired_success_ties_vs_baseline": 0,
                        "paired_final_pos_better_vs_baseline": 0,
                        "paired_final_pos_worse_vs_baseline": 0,
                        "paired_final_pos_ties_vs_baseline": 0,
                    },
                    {
                        "variant_name": "variant_b",
                        "n_rollouts": 1,
                        "success_rate": 0.0,
                        "mean_final_pos_diff": 3.0,
                        "mean_final_angle_diff": 0.2,
                        "mean_final_eef_diff": 0.4,
                        "mean_final_coverage": 0.50,
                        "mean_bits_used_total": 200.0,
                        "mean_flops_used_total": 400.0,
                        "mean_plan_time_total_sec": 1.00,
                        "paired_success_wins_vs_baseline": 0,
                        "paired_success_losses_vs_baseline": 0,
                        "paired_success_ties_vs_baseline": 0,
                        "paired_final_pos_better_vs_baseline": 0,
                        "paired_final_pos_worse_vs_baseline": 0,
                        "paired_final_pos_ties_vs_baseline": 0,
                    },
                ],
            )
            experiment_path = os.path.join(run_dir, EXPERIMENT_JSON)
            with open(experiment_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": 1,
                        "reviewer_version": 1,
                        "experiment_name": "demo",
                        "baseline_variant": "variant_a",
                        "variant_order": ["variant_a", "variant_b"],
                    },
                    f,
                )

            data = load_experiment_review_data(run_dir)
            self.assertIsNone(_paired_summary_figure(data, "variant_a"))

    def test_single_run_trace_figure_labels_single_step_rollouts_explicitly(self):
        fig = _single_run_trace_figure(
            {
                "pos_diffs": [1.0],
                "angle_diffs": [0.1],
                "eef_diffs": [0.2],
                "coverages": [0.95],
            },
            "variant_a",
            "rollout_0",
        )

        self.assertIn("1 recorded step", fig.layout.title.text)
        self.assertEqual(fig.data[0].mode, "markers")
        self.assertEqual(list(fig.data[0].x), [1])
        self.assertTrue(any("Only 1 recorded step" in annotation.text for annotation in fig.layout.annotations))

    def test_single_run_trace_figure_adds_replan_markers(self):
        fig = _single_run_trace_figure(
            {
                "pos_diffs": [3.0, 2.0, 1.0],
                "angle_diffs": [0.3, 0.2, 0.1],
                "eef_diffs": [6.0, 5.0, 4.0],
                "coverages": [0.6, 0.8, 0.95],
            },
            "variant_a",
            "rollout_0",
            trace_meta={
                "replans": [
                    {
                        "replan_idx": 0,
                        "step_start": 1,
                        "base_level_idx": 3,
                        "bits_used_estimate": 1234567890,
                        "plan_time_sec": 1.25,
                    }
                ]
            },
        )

        replan_traces = [trace for trace in fig.data if trace.name == "Replan"]
        self.assertEqual(len(replan_traces), 4)
        self.assertEqual(list(replan_traces[0].x), [2])
        self.assertEqual(replan_traces[0].customdata[0][0], 0)

    def test_media_render_queue_updates_run_page_status_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            app = ExperimentReviewApp(run_dir)
            started = threading.Event()
            finish = threading.Event()

            def fake_render(experiment_root, *, schedule, rollout_id, media, output_dir):
                started.set()
                self.assertEqual(media, ["closed_loop_replay"])
                self.assertTrue(finish.wait(timeout=2))
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, "closed_loop_replay.mp4")
                with open(out_path, "wb") as f:
                    f.write(b"fake_video")
                return [out_path]

            with mock.patch("hudm.experiment_review.planning_media.render_media", side_effect=fake_render):
                queued, skipped = app.queue_media_render(
                    variant_name="variant_a",
                    rollout_id="rollout_0",
                    media=["closed_loop_replay"],
                )
                self.assertEqual((queued, skipped), (1, 0))
                self.assertTrue(started.wait(timeout=1))

                in_progress_html = app.run_page(variant_name="variant_a", rollout_id="rollout_0")
                self.assertIn("status-running", in_progress_html)
                self.assertNotIn("Rendering in background", in_progress_html)
                self.assertNotIn("refreshes every 3 seconds", in_progress_html)

                finish.set()
                deadline = time.time() + 2
                while time.time() < deadline:
                    task = app.media_tasks_for_run("variant_a", "rollout_0").get("closed_loop_replay")
                    if task is not None and task.status == "succeeded":
                        break
                    time.sleep(0.05)

                done_html = app.run_page(variant_name="variant_a", rollout_id="rollout_0")
                self.assertIn("/files/review_cache/media/variant_a/rollout_0/closed_loop_replay.mp4", done_html)

    def test_particle_media_render_task_uses_serialized_lock(self):
        class RecorderLock:
            def __init__(self):
                self.enter_count = 0
                self.exit_count = 0

            def __enter__(self):
                self.enter_count += 1
                return self

            def __exit__(self, exc_type, exc, tb):
                self.exit_count += 1
                return False

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            app = ExperimentReviewApp(run_dir)
            key = ("variant_a", "rollout_0", "closed_loop_replay")
            app._media_tasks[key] = MediaRenderTask(
                variant_name="variant_a",
                rollout_id="rollout_0",
                media_name="closed_loop_replay",
                status="pending",
            )
            recorder_lock = RecorderLock()

            with mock.patch("hudm.experiment_review._trace_backend_for_run", return_value="particle_sim"):
                with mock.patch("hudm.experiment_review._PARTICLE_MEDIA_RENDER_LOCK", recorder_lock):
                    with mock.patch(
                        "hudm.experiment_review.render_media_for_run",
                        return_value=(["/tmp/closed_loop_replay.mp4"], []),
                    ):
                        app._run_media_render_task(
                            variant_name="variant_a",
                            rollout_id="rollout_0",
                            media_name="closed_loop_replay",
                        )

            task = app.media_tasks_for_run("variant_a", "rollout_0")["closed_loop_replay"]
            self.assertEqual(recorder_lock.enter_count, 1)
            self.assertEqual(recorder_lock.exit_count, 1)
            self.assertEqual(task.status, "succeeded")
            self.assertEqual(task.outputs, ["/tmp/closed_loop_replay.mp4"])

    def test_build_run_media_section_marks_active_media(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            data = load_experiment_review_data(run_dir)
            row = resolve_row(data, "variant_a", "rollout_0")
            media_html, active_media = build_run_media_section(
                data,
                row,
                media_tasks={
                    "closed_loop_replay": mock.Mock(status="running"),
                },
            )
            self.assertTrue(active_media)
            self.assertIn("id='mediaSection'", media_html)
            self.assertIn("data-active-media='true'", media_html)

    def test_render_endpoint_can_return_json_fragment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            app = ExperimentReviewApp(run_dir)
            app.queue_media_render = mock.Mock(return_value=(1, 0))
            app.media_tasks_for_run = mock.Mock(return_value={"closed_loop_replay": mock.Mock(status="running")})
            handler_cls = make_review_handler(app)
            handler = handler_cls.__new__(handler_cls)
            handler.path = "/render?variant=variant_a&rollout_id=rollout_0&media=closed_loop_replay&format=json"
            status = {}
            headers = []
            payload = bytearray()
            handler.send_response = lambda code: status.setdefault("code", code)
            handler.send_header = lambda key, value: headers.append((key, value))
            handler.end_headers = lambda: None
            handler.wfile = type("W", (), {"write": lambda self, data: payload.extend(data)})()
            handler.review_app = app

            handler.do_GET()

            self.assertEqual(status["code"], 200)
            body = json.loads(payload.decode("utf-8"))
            self.assertIn("media_html", body)
            self.assertTrue(body["active_media"])
            self.assertEqual(body["notice_html"], "")

    def test_handler_ignores_client_disconnect_without_sending_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            app = ExperimentReviewApp(run_dir)
            handler_cls = make_review_handler(app)
            handler = handler_cls.__new__(handler_cls)
            handler.path = "/run?variant=variant_a&rollout_id=rollout_0"
            handler.review_app = app
            handler._send_html = mock.Mock(side_effect=BrokenPipeError())
            handler.send_error = mock.Mock(side_effect=AssertionError("send_error should not be called"))

            handler.do_GET()

            handler._send_html.assert_called_once()
            handler.send_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
