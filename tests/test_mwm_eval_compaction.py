from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mwm.benchmark.eval_compaction import COMPLETION_FILES, EvalCompactionError, compact_completed_eval
from scripts.research.compact_release20260728_policy_diagnostics import (
    BENCHMARK_TARGETS,
    compact_release_outputs,
    release_output_roots,
)


class EvalCompactionTests(unittest.TestCase):
    def _completed_cell(self, root: Path, *, equal: bool = True) -> Path:
        run_dir = root / "000_cell"
        run_dir.mkdir()
        diagnostics = {"summary": {"replans": 2}, "trace": [{"base_k": 192}]}
        payload = {
            "planning_diagnostics": diagnostics,
            "policy_diagnostics": diagnostics if equal else {"summary": {"replans": 3}},
            "batches": [{"planning_diagnostics": diagnostics}],
            "review_rollouts": [{"episode_index": 0}],
            "swm_results": {"success_rate": 100.0},
        }
        (run_dir / "eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        for name in COMPLETION_FILES:
            if name != "eval.json":
                (run_dir / name).write_text("{}\n", encoding="utf-8")
        return run_dir

    def test_compacts_exact_duplicate_atomically_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._completed_cell(Path(tmp))
            original = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
            result = compact_completed_eval(run_dir)
            compacted = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "compacted")
            self.assertGreater(result["reclaimed_bytes"], 0)
            self.assertNotIn("policy_diagnostics", compacted)
            self.assertEqual(compacted["planning_diagnostics"], original["planning_diagnostics"])
            self.assertEqual(compacted["batches"], original["batches"])
            self.assertEqual(compacted["review_rollouts"], original["review_rollouts"])
            self.assertFalse(list(run_dir.glob(".eval.json.*.tmp")))
            self.assertEqual(compact_completed_eval(run_dir)["status"], "already_compacted")

    def test_retries_transient_nonblocking_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._completed_cell(Path(tmp))
            real_replace = os.replace
            calls = 0

            def flaky_replace(source: str | Path, destination: str | Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking")
                real_replace(source, destination)

            with (
                patch("mwm.benchmark.eval_compaction.os.replace", side_effect=flaky_replace),
                patch("mwm.benchmark.eval_compaction.time.sleep"),
            ):
                result = compact_completed_eval(run_dir)

            self.assertEqual(result["status"], "compacted")
            self.assertEqual(calls, 2)
            self.assertNotIn(
                "policy_diagnostics", json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
            )

    def test_skips_partial_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._completed_cell(Path(tmp))
            (run_dir / "episode_traces.jsonl").unlink()
            before = (run_dir / "eval.json").read_bytes()

            result = compact_completed_eval(run_dir)

            self.assertEqual(result["status"], "partial")
            self.assertIn("episode_traces.jsonl", result["missing"])
            self.assertEqual((run_dir / "eval.json").read_bytes(), before)

    def test_refuses_nonidentical_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._completed_cell(Path(tmp), equal=False)
            before = (run_dir / "eval.json").read_bytes()

            with self.assertRaisesRegex(EvalCompactionError, "differs"):
                compact_completed_eval(run_dir)

            self.assertEqual((run_dir / "eval.json").read_bytes(), before)

    def test_dry_run_validates_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._completed_cell(Path(tmp))
            before = (run_dir / "eval.json").read_bytes()

            result = compact_completed_eval(run_dir, dry_run=True)

            self.assertEqual(result["status"], "would_compact")
            self.assertEqual((run_dir / "eval.json").read_bytes(), before)

    def test_eval_runner_does_not_emit_policy_diagnostics(self) -> None:
        runner = (Path(__file__).resolve().parents[1] / "mwm/eval/runner.py").read_text(encoding="utf-8")
        self.assertNotIn('"policy_diagnostics"', runner)

    def test_release_compactor_is_scoped_to_eight_exact_roots(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        roots = release_output_roots(repo_root)
        expected = tuple(repo_root / output for _, output in BENCHMARK_TARGETS)
        self.assertEqual(len(roots), 8)
        self.assertEqual(roots, expected)

    def test_release_compactor_rejects_non_allowlisted_target_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "target indices"):
            compact_release_outputs(target_indices=(8,))


if __name__ == "__main__":
    unittest.main()
