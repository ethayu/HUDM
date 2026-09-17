from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.research.packed_b200_release20260728_profile import (
    CELL_SPECS,
    NUM_SHARDS,
    ProfileError,
    SELECTED_INDICES,
    build_cell_command,
    first_difference,
    normalize_scientific_payload,
    split_cpu_affinity,
    validate_benchmark_sources,
    validate_output_root,
)


class PackedB200ReleaseProfileTests(unittest.TestCase):
    def test_exact_four_cells_and_two_approved_indices(self) -> None:
        self.assertEqual(len(CELL_SPECS), 4)
        self.assertEqual(tuple(sorted({spec.matrix_index for spec in CELL_SPECS})), SELECTED_INDICES)
        self.assertEqual({spec.goal for spec in CELL_SPECS}, {25, 50})
        self.assertEqual({spec.population for spec in CELL_SPECS}, {20, 200})
        self.assertTrue(all(spec.matrix_index in {0, 105} for spec in CELL_SPECS))

    def test_live_benchmark_sources_resolve_to_the_pinned_cells(self) -> None:
        records = validate_benchmark_sources()
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["cells"] == NUM_SHARDS for record in records.values()))

    def test_output_scope_must_be_new_direct_child_for_the_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "profile"
            allowed = base / "job_123_attempt0"
            self.assertEqual(
                validate_output_root(allowed, profile_base=base, job_id="123"),
                allowed.resolve(strict=False),
            )
            with self.assertRaisesRegex(ProfileError, "directly below"):
                validate_output_root(allowed / "nested", profile_base=base, job_id="123")
            with self.assertRaisesRegex(ProfileError, "start with"):
                validate_output_root(base / "job_999_attempt0", profile_base=base, job_id="123")
            allowed.mkdir(parents=True)
            with self.assertRaisesRegex(ProfileError, "reuse existing"):
                validate_output_root(allowed, profile_base=base, job_id="123")

    def test_scientific_normalization_removes_only_declared_variance(self) -> None:
        planning = {
            "plan_time_total_sec": 1.0,
            "trace": [
                {
                    "cost_time_sec": 0.1,
                    "base_k": 192,
                    "candidate_action_values": 1000,
                }
            ],
            "summary": {
                "mean_plan_time_sec": 0.5,
                "total_plan_time_sec": 1.0,
                "dynamics_flops_total": 123,
            },
        }
        payload = {
            "wall_time_sec": 4.0,
            "dependencies": {"local_repo": {"commit_id": "old"}},
            "config": {"resolved_path": "/old/eval.yaml", "sha256": "old"},
            "videos": [],
            "planning_diagnostics": planning,
            "policy_diagnostics": copy.deepcopy(planning),
            "batches": [
                {
                    "videos": [],
                    "planning_diagnostics": copy.deepcopy(planning),
                    "review_trace": {"action_trace": [[[0.125]]]},
                }
            ],
            "swm_results": {"episode_successes": [True]},
        }
        changed = copy.deepcopy(payload)
        changed["wall_time_sec"] = 8.0
        changed["dependencies"] = {"local_repo": {"commit_id": "new"}}
        changed["config"] = {"resolved_path": "/new/eval.yaml", "sha256": "new"}
        changed["planning_diagnostics"]["trace"][0]["cost_time_sec"] = 0.9
        changed["policy_diagnostics"] = copy.deepcopy(changed["planning_diagnostics"])
        changed["batches"][0]["planning_diagnostics"]["trace"][0]["cost_time_sec"] = 0.8
        self.assertEqual(normalize_scientific_payload(payload), normalize_scientific_payload(changed))

        changed["batches"][0]["review_trace"]["action_trace"] = [[[0.126]]]
        self.assertNotEqual(normalize_scientific_payload(payload), normalize_scientific_payload(changed))

    def test_unequal_legacy_policy_diagnostics_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProfileError, "differs"):
            normalize_scientific_payload(
                {"planning_diagnostics": {"plans": 1}, "policy_diagnostics": {"plans": 2}, "batches": []}
            )

    def test_unexpected_video_payload_is_not_hidden_by_normalization(self) -> None:
        with self.assertRaisesRegex(ProfileError, "top-level videos"):
            normalize_scientific_payload({"videos": ["unexpected.mp4"], "batches": []})

    def test_first_difference_reports_precise_nested_path(self) -> None:
        difference = first_difference(
            {"review_rollouts": [{"action_trace": [1.0, 2.0]}]},
            {"review_rollouts": [{"action_trace": [1.0, 3.0]}]},
        )
        self.assertEqual(difference["path"], "$.review_rollouts[0].action_trace[1]")

    def test_cell_command_is_single_exact_shard_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = CELL_SPECS[1]
            command = build_cell_command(
                python=Path("/python"),
                config_path=Path("/config.yaml"),
                run_root=root / "new-output",
                spec=spec,
                cpu_affinity=(1, 2),
                time_path=root / "time.txt",
            )
            self.assertIn(str(NUM_SHARDS), command)
            self.assertEqual(command[-1], "105")
            self.assertNotIn("--resume", command)
            self.assertIn(f"output_dir={root / 'new-output'}", command)

    def test_cpu_split_is_disjoint_and_complete(self) -> None:
        first, second = split_cpu_affinity(range(7))
        self.assertFalse(set(first) & set(second))
        self.assertEqual(set(first) | set(second), set(range(7)))
        self.assertLessEqual(abs(len(first) - len(second)), 1)

    def test_slurm_script_is_one_nice_full_b200_allocation(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/slurm/slurm_mwm_release20260728_packed_b200_profile.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=dgx-b200", script)
        self.assertIn("#SBATCH --qos=dgx", script)
        self.assertIn("#SBATCH --gpus=1", script)
        self.assertIn("#SBATCH --cpus-per-task=28", script)
        self.assertIn("#SBATCH --mem=224G", script)
        self.assertIn("#SBATCH --nice=10000", script)
        self.assertNotIn("#SBATCH --array", script)
        self.assertNotIn("scancel", script)
        self.assertNotIn("scontrol", script)
        self.assertNotIn("nvidia-cuda-mps-control", script)


if __name__ == "__main__":
    unittest.main()
