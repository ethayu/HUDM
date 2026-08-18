from __future__ import annotations

import errno
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from omegaconf import OmegaConf

from mwm.benchmark.eval_artifacts import (
    COMPLETION_FILES,
    EvalArtifactError,
    compress_completed_eval,
    inspect_eval_artifact,
    load_eval_artifact,
    load_eval_capsule,
    load_planning_diagnostics,
    planning_sidecar_matches_capsule,
    validate_eval_storage_reference,
)
from mwm.benchmark import eval_artifacts as artifact_module
from mwm.benchmark.review_media import record_review_media
from mwm.benchmark.review_server import rollout_page_html
from mwm.benchmark.render_review import render_benchmark_review
from mwm.benchmark.matrix import _completed_run, _configure_run_paths
from mwm.io import load_json, write_json
from scripts.research.compact_release20260728_policy_diagnostics import BENCHMARK_TARGETS
from scripts.research.compress_release20260728_eval_artifacts import compress_release_outputs


def canonical(value: object) -> str:
    return json.dumps(value, allow_nan=True, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def decompress_zstd(path: Path) -> bytes:
    import zstandard

    with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
        return reader.read()


class EvalArtifactTests(unittest.TestCase):
    @mock.patch.object(artifact_module.time, "sleep")
    @mock.patch.object(
        artifact_module.os,
        "fsync",
        side_effect=(OSError(errno.EAGAIN, "temporarily unavailable"), None),
    )
    def test_directory_fsync_retries_transient_ceph_eagain(
        self,
        fsync: mock.Mock,
        sleep: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_module._fsync_directory(Path(tmp))

        self.assertEqual(fsync.call_count, 2)
        sleep.assert_called_once_with(0.05)

    def test_release_migrator_reuses_the_exact_eight_root_allowlist(self) -> None:
        self.assertEqual(len(BENCHMARK_TARGETS), 8)
        with self.assertRaisesRegex(ValueError, "target indices"):
            compress_release_outputs(target_indices=(8,))

    def _payload(self, *, with_policy: bool = False) -> dict[str, object]:
        traces = [
            {"batch_start": 0, "batch_end": 1, "cem_iter": 0, "base_k": 96},
            {"batch_start": 1, "batch_end": 2, "cem_iter": 0, "base_k": 192},
        ]
        diagnostics = {
            "summary": {"cem_cost_calls": 2, "candidate_action_values": 20},
            "trace": traces,
            "plans": 2,
            "steps": 2,
            "cem_cost_calls": 2,
            "candidate_action_values": 20,
        }
        payload: dict[str, object] = {
            "benchmark_name": "synthetic",
            "role": "mwm_scheduled",
            "env_id": "swm/Test-v1",
            "episodes": 2,
            "goal_offset": 25,
            "checkpoint_run_dir": "checkpoints_mwm/example",
            "dependencies": {"local_repo": {"commit_id": "abc"}},
            "manifest": {"path": "manifest.json", "manifest_sha256": "semantic"},
            "swm_results": {"success_rate": 50.0, "episode_successes": [True, False]},
            "planning_diagnostics": diagnostics,
            "batches": [
                {
                    "pairs": [{"episode": 0, "start_step": 0, "goal_step": 25}],
                    "planning_diagnostics": {"summary": {"cem_cost_calls": 1}, "trace": traces[:1]},
                    "review_trace": {"action_trace": [[[1.0], [float("nan")]]], "model_action_trace": [[[2.0]]]},
                    "swm_results": {"episode_successes": [True]},
                },
                {
                    "pairs": [{"episode": 1, "start_step": 1, "goal_step": 26}],
                    "planning_diagnostics": {"summary": {"cem_cost_calls": 1}, "trace": traces[1:]},
                    "review_trace": {"action_trace": [[[3.0]]], "model_action_trace": [[[4.0]]]},
                    "swm_results": {"episode_successes": [False]},
                },
            ],
            "review_rollouts": [
                {"episode_index": 0, "success": True, "action_trace": [[1.0]], "fidelity_trace": [{"K": 96}]},
                {"episode_index": 1, "success": False, "action_trace": [[3.0]], "fidelity_trace": [{"K": 192}]},
            ],
            "review_media": {"rollouts": {}},
            "videos": [],
        }
        if with_policy:
            payload["policy_diagnostics"] = diagnostics
        return payload

    def _completed_cell(self, root: Path, *, with_policy: bool = False) -> tuple[Path, dict[str, object]]:
        run_dir = root / "000_cell"
        run_dir.mkdir()
        payload = self._payload(with_policy=with_policy)
        write_json(run_dir / "eval.json", payload)
        for name in COMPLETION_FILES:
            if name == "eval.json":
                continue
            if name == "planning_diagnostics.json":
                write_json(run_dir / name, dict(payload["planning_diagnostics"]))
            else:
                (run_dir / name).write_text("{}\n", encoding="utf-8")
        return run_dir, payload

    def test_legacy_and_policy_pruned_plain_json_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            full = self._payload(with_policy=True)
            write_json(path, full)
            self.assertEqual(canonical(load_eval_artifact(path)), canonical(full))
            full.pop("policy_diagnostics")
            write_json(path, full)
            self.assertEqual(canonical(load_eval_artifact(path)), canonical(full))

    def test_compression_round_trips_science_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, original = self._completed_cell(Path(tmp), with_policy=True)
            expected = dict(original)
            expected.pop("policy_diagnostics")

            result = compress_completed_eval(run_dir)

            self.assertEqual(result["status"], "compressed")
            capsule = load_eval_capsule(run_dir / "eval.json", verify="compressed_hash")
            self.assertEqual(capsule["_artifact"]["version"], 2)
            self.assertIsInstance(capsule["planning_diagnostics"]["trace"], dict)
            expanded = load_eval_artifact(run_dir / "eval.json")
            self.assertEqual(canonical(expanded), canonical(expected))
            self.assertEqual(
                canonical(load_planning_diagnostics(run_dir)),
                canonical(expected["planning_diagnostics"]),
            )
            self.assertTrue(planning_sidecar_matches_capsule(run_dir, capsule))
            self.assertEqual(compress_completed_eval(run_dir)["status"], "already_compressed")
            self.assertEqual(
                inspect_eval_artifact(run_dir / "eval.json", verify="full")["representation"],
                "capsule+archive",
            )
            self.assertTrue(validate_eval_storage_reference(run_dir / "eval.json"))

    def test_archive_uses_canonical_batch_trace_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            compress_completed_eval(run_dir)
            capsule = load_json(run_dir / "eval.json")
            archive_path = run_dir / capsule["_artifact"]["archive"]["path"]
            envelope = json.loads(decompress_zstd(archive_path))
            batches = envelope["sections"]["batches"]
            self.assertEqual(
                [batch["_planning_trace_ref"] for batch in batches],
                [{"start": 0, "stop": 1}, {"start": 1, "stop": 2}],
            )
            self.assertNotIn("trace", batches[0]["planning_diagnostics"])

    def test_corruption_and_unsafe_archive_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            compress_completed_eval(run_dir)
            capsule = load_json(run_dir / "eval.json")
            archive_path = run_dir / capsule["_artifact"]["archive"]["path"]
            raw = bytearray(archive_path.read_bytes())
            raw[len(raw) // 2] ^= 1
            archive_path.write_bytes(raw)
            with self.assertRaisesRegex(EvalArtifactError, "SHA-256"):
                load_eval_artifact(run_dir / "eval.json")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            compress_completed_eval(run_dir)
            capsule = load_json(run_dir / "eval.json")
            capsule["_artifact"]["archive"]["path"] = "../escape.json.zst"
            write_json(run_dir / "eval.json", capsule)
            with self.assertRaisesRegex(EvalArtifactError, "direct|escapes"):
                inspect_eval_artifact(run_dir / "eval.json")
            self.assertFalse(validate_eval_storage_reference(run_dir / "eval.json"))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            write_json(path, {"_artifact": {"schema": "unknown", "version": 99}})
            with self.assertRaisesRegex(EvalArtifactError, "unsupported eval artifact schema"):
                load_eval_artifact(path)

    def test_metadata_validation_rejects_size_mismatched_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            compress_completed_eval(run_dir)
            capsule = load_json(run_dir / "eval.json")
            archive_path = run_dir / capsule["_artifact"]["archive"]["path"]
            self.assertTrue(validate_eval_storage_reference(run_dir / "eval.json", verify="metadata"))

            archive_path.write_bytes(archive_path.read_bytes()[:-1])

            self.assertFalse(validate_eval_storage_reference(run_dir / "eval.json", verify="metadata"))

    def test_gzip_archives_are_read_for_rolling_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, expected = self._completed_cell(Path(tmp))
            compress_completed_eval(run_dir)
            capsule = load_json(run_dir / "eval.json")
            old_path = run_dir / capsule["_artifact"]["archive"]["path"]
            uncompressed = decompress_zstd(old_path)
            gzip_path = run_dir / "eval.details.test.json.gz"
            with gzip.open(gzip_path, "wb", compresslevel=6) as handle:
                handle.write(uncompressed)
            compressed = gzip_path.read_bytes()
            metadata = capsule["_artifact"]["archive"]
            metadata.update(
                path=gzip_path.name,
                codec="gzip",
                compressed_bytes=len(compressed),
                compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            )
            write_json(run_dir / "eval.json", capsule)
            self.assertEqual(canonical(load_eval_artifact(run_dir / "eval.json")), canonical(expected))

    def test_partial_and_mismatched_sidecars_are_never_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            (run_dir / "episode_traces.jsonl").unlink()
            before = (run_dir / "eval.json").read_bytes()
            self.assertEqual(compress_completed_eval(run_dir)["status"], "partial")
            self.assertEqual((run_dir / "eval.json").read_bytes(), before)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            write_json(run_dir / "planning_diagnostics.json", {"different": True})
            before = (run_dir / "eval.json").read_bytes()
            with self.assertRaisesRegex(EvalArtifactError, "sidecar differs"):
                compress_completed_eval(run_dir)
            self.assertEqual((run_dir / "eval.json").read_bytes(), before)

    def test_scientifically_distinct_batch_trace_is_preserved_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, payload = self._completed_cell(Path(tmp))
            payload["batches"][1]["planning_diagnostics"]["trace"] = [{"unique": True}]
            write_json(run_dir / "eval.json", payload)
            compress_completed_eval(run_dir)
            self.assertEqual(canonical(load_eval_artifact(run_dir / "eval.json")), canonical(payload))

    def test_review_media_updates_only_capsule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            compress_completed_eval(run_dir)
            capsule = load_json(run_dir / "eval.json")
            archive_path = run_dir / capsule["_artifact"]["archive"]["path"]
            archive_before = archive_path.read_bytes()

            record_review_media(
                run_dir / "eval.json",
                episode_index=0,
                kind="env",
                path=run_dir / "review_media" / "episode_0000" / "env.mp4",
                source_trace_type="action_trace",
            )

            self.assertEqual(archive_path.read_bytes(), archive_before)
            expanded = load_eval_artifact(run_dir / "eval.json")
            self.assertIn("env", expanded["review_media"]["rollouts"]["episode_0000"])
            page = rollout_page_html(run_dir.parent, run_dir / "eval.json", 0)
            self.assertIn("synthetic", page)
            self.assertIn("episode 0", page)

    def test_static_review_materializes_compressed_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, expected = self._completed_cell(root)
            compress_completed_eval(run_dir)
            write_json(
                root / "summary.json",
                {
                    "title": "Compressed review",
                    "runs": [
                        {
                            "name": "synthetic",
                            "output_json": str(run_dir / "eval.json"),
                        }
                    ],
                },
            )

            with (
                mock.patch("mwm.benchmark.html.write_review_html") as write_review,
                mock.patch("mwm.benchmark.pareto.write_pareto_html", return_value=str(root / "plots" / "pareto.html")),
                mock.patch("mwm.benchmark.plots.write_default_plots", return_value=[]),
                mock.patch("mwm.benchmark.summary.write_per_env_table", return_value=[]),
                mock.patch("mwm.benchmark.summary.write_summary_csv"),
                mock.patch("mwm.io.write_metrics_jsonl"),
            ):
                render_benchmark_review(root)

            outputs = write_review.call_args.args[3]
            self.assertEqual(canonical(outputs[0]["review_rollouts"]), canonical(expected["review_rollouts"]))
            self.assertEqual(outputs[0]["review_rollouts"][0]["action_trace"], [[1.0]])

    def test_matrix_resume_recognizes_valid_archive_and_rejects_missing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, _ = self._completed_cell(root)
            manifest_path = root / "manifest.json"
            run_cfg = OmegaConf.create(
                {
                    "env_id": "swm/Test-v1",
                    "eval": {"seed": 42, "episodes": 2},
                    "planner": {"pop_size": 20},
                }
            )
            _configure_run_paths(run_cfg, run_dir, manifest_path)
            (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(run_cfg), encoding="utf-8")
            write_json(run_dir / "summary.json", {"run": {"name": "synthetic"}})
            compress_completed_eval(run_dir)

            completed = _completed_run(run_dir, run_cfg, manifest_path)
            self.assertIsNotNone(completed)
            self.assertEqual(completed[1]["_artifact"]["version"], 2)
            materialized = _completed_run(run_dir, run_cfg, manifest_path, materialize=True)
            self.assertIsNotNone(materialized)
            self.assertEqual(materialized[1]["review_rollouts"][0]["action_trace"], [[1.0]])

            capsule = load_json(run_dir / "eval.json")
            (run_dir / capsule["_artifact"]["archive"]["path"]).unlink()
            self.assertIsNone(_completed_run(run_dir, run_cfg, manifest_path))

    def test_archive_first_commit_is_recoverable_at_both_json_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            with mock.patch.object(artifact_module, "_atomic_write_json", side_effect=OSError("capsule write")):
                with self.assertRaisesRegex(OSError, "capsule write"):
                    compress_completed_eval(run_dir)
            self.assertNotIn("_artifact", load_json(run_dir / "eval.json"))
            self.assertEqual(len(list(run_dir.glob("eval.details.*.json.zst"))), 1)
            self.assertEqual(compress_completed_eval(run_dir)["status"], "compressed")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = self._completed_cell(Path(tmp))
            real_atomic = artifact_module._atomic_write_json
            calls = 0

            def fail_sidecar(path: Path, payload: dict[str, object]) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("sidecar write")
                real_atomic(path, payload)

            with mock.patch.object(artifact_module, "_atomic_write_json", side_effect=fail_sidecar):
                with self.assertRaisesRegex(OSError, "sidecar write"):
                    compress_completed_eval(run_dir)
            self.assertIn("_artifact", load_json(run_dir / "eval.json"))
            self.assertNotIn("_artifact", load_json(run_dir / "planning_diagnostics.json"))
            self.assertEqual(compress_completed_eval(run_dir)["status"], "repaired")
            self.assertEqual(compress_completed_eval(run_dir)["status"], "already_compressed")


if __name__ == "__main__":
    unittest.main()
