from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from unittest import mock

from hudm.experiment_review import (
    build_run_page,
    build_summary_page,
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

        _write_csv(
            os.path.join(run_dir, "summary.csv"),
            [
                {
                    "variant_name": "variant_a",
                    "n_rollouts": 1,
                    "success_rate": 1.0,
                    "mean_final_pos_diff": 1.25,
                    "mean_final_coverage": 0.96,
                    "mean_bits_used_total": 100.0,
                    "mean_plan_time_total_sec": 0.50,
                }
            ],
        )
        _write_csv(
            os.path.join(run_dir, "per_rollout.csv"),
            [
                {
                    "variant_name": "variant_a",
                    "rollout_id": "rollout_0",
                    "success": 1,
                    "termination_reason": "env_done",
                    "final_pos_diff": 1.25,
                    "final_coverage": 0.96,
                }
            ],
        )
        with open(os.path.join(run_dir, "experiment_resolved.json"), "w", encoding="utf-8") as f:
            json.dump({"name": "demo", "baseline": "variant_a"}, f)
        with open(os.path.join(run_dir, "experiment_report.html"), "w", encoding="utf-8") as f:
            f.write("<html></html>")
        with open(os.path.join(trace_dir, "trace.json"), "w", encoding="utf-8") as f:
            json.dump({"ok": True}, f)
        with open(os.path.join(trace_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write("step log\n")
        return run_dir, trace_dir

    def test_build_summary_page_has_review_links(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._make_experiment_dir(tmpdir)
            data = load_experiment_review_data(run_dir)
            html = build_summary_page(data)
            self.assertIn("demo", html)
            self.assertIn("/run?variant=variant_a&rollout_id=rollout_0", html)
            self.assertIn("/files/experiment_report.html", html)

    def test_render_media_for_run_updates_detail_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, trace_dir = self._make_experiment_dir(tmpdir)

            def fake_render(experiment_root, *, schedule, rollout_id, media):
                self.assertEqual(os.path.abspath(experiment_root), os.path.abspath(run_dir))
                self.assertEqual(schedule, "variant_a")
                self.assertEqual(rollout_id, "rollout_0")
                self.assertEqual(media, ["closed_loop_replay"])
                out_path = os.path.join(trace_dir, "closed_loop_replay.mp4")
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
            self.assertIn("closed_loop_replay", detail_html)
            self.assertIn("/files/traces/variant_a/rollout_0/closed_loop_replay.mp4", detail_html)


if __name__ == "__main__":
    unittest.main()
