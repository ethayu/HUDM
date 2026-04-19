from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from hudm import experiment
from hudm.specs import ExperimentSpec, ExperimentVariant, make_plan_spec
from omegaconf import OmegaConf
from scripts.create_dummy_multivariant_bundle import FAKE_VARIANTS, build_dummy_bundle
from hudm.experiment_bundle import (
    EXPERIMENT_JSON,
    PAIRED_VS_BASELINE_CSV,
    RUNS_CSV,
    SELECTED_ROLLOUTS_JSON,
    VARIANTS_CSV,
    migrate_legacy_experiment_dir,
    write_experiment_bundle,
)
from hudm.experiment_review import load_experiment_review_data


class ExperimentReportingTests(unittest.TestCase):
    def _make_plan_spec(self, name: str, backend_kind: str, *, backend_cfg: dict | None = None):
        clean_cfg = OmegaConf.create(
            {
                "task": {
                    "env_id": "pusht",
                    "env": {},
                    "init_goal": {},
                },
                "budget": {"max_env_steps": 1},
                "planner": {
                    "horizon": 1,
                    "replan_every": 1,
                    "cem": {"pop_size": 4, "elite_frac": 0.5},
                },
                "backend": {
                    "kind": backend_kind,
                    backend_kind: backend_cfg or {},
                },
                "artifacts": {"save": False},
            }
        )
        runtime_cfg = OmegaConf.create({"save": False})
        return make_plan_spec(name=name, config_path=None, clean_cfg=clean_cfg, runtime_cfg=runtime_cfg)

    def _make_experiment_spec(self, tmpdir: str, variants: list[ExperimentVariant]) -> ExperimentSpec:
        return ExperimentSpec(
            name="scheduler_demo",
            config_path=None,
            shared_plan=self._make_plan_spec("shared", "wm", backend_cfg={"shared": True}),
            baseline=variants[0].name if variants else None,
            variants=variants,
            rollouts={"seed": 0, "num_rollouts": 1, "sample_without_replacement": True},
            execution={"mode": "process", "max_workers": 4},
            terminal={"mode": "quiet"},
            reporting={"output_root": tmpdir},
        )

    def _fake_row(self, variant_name: str, rollout_id: str = "r0", rollout_index: int = 0) -> dict:
        return {
            "variant_name": variant_name,
            "rollout_id": rollout_id,
            "rollout_index": rollout_index,
            "success": 1,
        }

    def test_group_wm_variants_groups_backend_lanes_by_compatibility_signature(self):
        shared_backend = {"world_model": {"device": "cpu"}}
        wm_a = ExperimentVariant(name="wm_a", plan=self._make_plan_spec("wm_a", "wm", backend_cfg=shared_backend))
        wm_b = ExperimentVariant(name="wm_b", plan=self._make_plan_spec("wm_b", "wm", backend_cfg=shared_backend))
        gt = ExperimentVariant(name="gt_a", plan=self._make_plan_spec("gt_a", "gt_env", backend_cfg={"rollout_samples": 1}))

        batched, singles = experiment._group_wm_variants([wm_a, wm_b, gt])

        self.assertEqual([[variant.name for variant in group] for group in batched], [["wm_a", "wm_b"]])
        self.assertEqual([variant.name for variant in singles], ["gt_a"])

    def test_run_experiment_dispatches_backend_lanes_without_real_multiprocessing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_backend = {"world_model": {"device": "cpu"}}
            wm_a = ExperimentVariant(name="wm_a", plan=self._make_plan_spec("wm_a", "wm", backend_cfg=shared_backend))
            wm_b = ExperimentVariant(name="wm_b", plan=self._make_plan_spec("wm_b", "wm", backend_cfg=shared_backend))
            gt = ExperimentVariant(name="gt_a", plan=self._make_plan_spec("gt_a", "gt_env", backend_cfg={"rollout_samples": 1}))
            spec = self._make_experiment_spec(tmpdir, [wm_a, wm_b, gt])

            submitted_tasks: list[dict] = []
            captured_bundle: dict[str, object] = {}

            class _FakeFuture:
                def __init__(self, fn, task):
                    self._fn = fn
                    self._task = task

                def result(self):
                    return self._fn(self._task)

            class _FakeExecutor:
                def __init__(self, *args, **kwargs):
                    del args, kwargs

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

                def submit(self, fn, task):
                    submitted_tasks.append(task)
                    return _FakeFuture(fn, task)

            with mock.patch.object(experiment, "enumerate_rollout_candidates", return_value=[{"rollout_id": "r0"}]):
                with mock.patch.object(
                    experiment,
                    "select_rollouts",
                    return_value=[{"rollout_id": "r0", "rollout_index": 0}],
                ):
                    with mock.patch.object(
                        experiment,
                        "_run_wm_batched_variants",
                        return_value=[self._fake_row("wm_a"), self._fake_row("wm_b")],
                    ) as wm_lane:
                        with mock.patch.object(
                            experiment,
                            "_run_variant_task",
                            side_effect=lambda task: self._fake_row(str(task["variant_name"])),
                        ) as single_lane:
                            with mock.patch.object(
                                experiment,
                                "ProcessPoolExecutor",
                                _FakeExecutor,
                            ):
                                with mock.patch.object(
                                    experiment,
                                    "as_completed",
                                    side_effect=lambda futures: list(futures),
                                ):
                                    with mock.patch.object(
                                        experiment,
                                        "aggregate_summary",
                                        return_value=([], []),
                                    ):
                                        with mock.patch.object(
                                            experiment,
                                            "write_experiment_bundle",
                                            side_effect=lambda run_dir, **kwargs: captured_bundle.update(kwargs),
                                        ):
                                            run_dir = experiment.run_experiment(spec)

            self.assertTrue(run_dir.startswith(tmpdir))
            self.assertEqual(wm_lane.call_count, 1)
            self.assertEqual([variant.name for variant in wm_lane.call_args.kwargs["variants"]], ["wm_a", "wm_b"])
            self.assertEqual(single_lane.call_count, 1)
            self.assertEqual(len(submitted_tasks), 1)
            self.assertEqual(submitted_tasks[0]["variant_name"], "gt_a")
            self.assertEqual(
                {row["variant_name"] for row in captured_bundle["run_rows"]},
                {"wm_a", "wm_b", "gt_a"},
            )

    def test_run_experiment_falls_back_to_serial_when_process_lane_executor_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gt = ExperimentVariant(name="gt_a", plan=self._make_plan_spec("gt_a", "gt_env", backend_cfg={"rollout_samples": 1}))
            spec = self._make_experiment_spec(tmpdir, [gt])

            submitted_tasks: list[dict] = []
            captured_bundle: dict[str, object] = {}

            class _FailingFuture:
                def result(self):
                    raise RuntimeError("pickling failed")

            class _FailingExecutor:
                def __init__(self, *args, **kwargs):
                    del args, kwargs

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

                def submit(self, fn, task):
                    submitted_tasks.append(task)
                    del fn
                    return _FailingFuture()

            with mock.patch.object(experiment, "enumerate_rollout_candidates", return_value=[{"rollout_id": "r0"}]):
                with mock.patch.object(
                    experiment,
                    "select_rollouts",
                    return_value=[{"rollout_id": "r0", "rollout_index": 0}],
                ):
                    with mock.patch.object(
                        experiment,
                        "_run_wm_batched_variants",
                        return_value=[],
                    ):
                        with mock.patch.object(
                            experiment,
                            "_run_variant_task",
                            side_effect=lambda task: self._fake_row(str(task["variant_name"])),
                        ) as single_lane:
                            with mock.patch.object(
                                experiment,
                                "ProcessPoolExecutor",
                                _FailingExecutor,
                            ):
                                with mock.patch.object(
                                    experiment,
                                    "as_completed",
                                    side_effect=lambda futures: list(futures),
                                ):
                                    with mock.patch.object(
                                        experiment,
                                        "aggregate_summary",
                                        return_value=([], []),
                                    ):
                                        with mock.patch.object(
                                            experiment,
                                            "write_experiment_bundle",
                                            side_effect=lambda run_dir, **kwargs: captured_bundle.update(kwargs),
                                        ):
                                            with mock.patch("builtins.print") as print_mock:
                                                run_dir = experiment.run_experiment(spec)

            self.assertTrue(run_dir.startswith(tmpdir))
            self.assertEqual(len(submitted_tasks), 1)
            self.assertEqual(submitted_tasks[0]["variant_name"], "gt_a")
            self.assertEqual(single_lane.call_count, 1)
            self.assertEqual(
                [row["variant_name"] for row in captured_bundle["run_rows"]],
                ["gt_a"],
            )
            self.assertTrue(
                any(
                    args
                    and isinstance(args[0], str)
                    and args[0].startswith("[experiment][warn] process parallelism unavailable (pickling failed); falling back to serial.")
                    for args, _ in ((call.args, call.kwargs) for call in print_mock.call_args_list)
                )
            )

    def test_run_task_lane_particle_device_slots_limit_workers_and_patch_task_devices(self):
        tasks = [
            {
                "cfg": {"particle_env": {"fidelity_env": {"device": "auto"}}},
                "variant_name": f"particle_{idx}",
            }
            for idx in range(3)
        ]
        submitted_devices: list[str] = []
        worker_counts: list[int] = []

        class _FakeFuture:
            def __init__(self, fn, task):
                self._fn = fn
                self._task = task

            def result(self):
                return self._fn(self._task)

        class _FakeExecutor:
            def __init__(self, *args, **kwargs):
                del args
                worker_counts.append(int(kwargs["max_workers"]))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, task):
                submitted_devices.append(str(task["cfg"]["particle_env"]["fidelity_env"]["device"]))
                return _FakeFuture(fn, task)

        with mock.patch.object(experiment, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.object(experiment, "as_completed", side_effect=lambda futures: list(futures)):
                with mock.patch.object(
                    experiment,
                    "_run_variant_task",
                    side_effect=lambda task: self._fake_row(str(task["variant_name"])),
                ):
                    rows = experiment._run_task_lane(
                        tasks,
                        exec_mode="process",
                        max_workers=6,
                        lane_key="task_pool:particle_sim",
                        backend_kind="particle_sim",
                        device_slots=["cuda:0", "cuda:1"],
                    )

        self.assertEqual(worker_counts, [2])
        self.assertEqual(submitted_devices, ["cuda:0", "cuda:1", "cuda:0"])
        self.assertEqual([row["variant_name"] for row in rows], ["particle_0", "particle_1", "particle_2"])

    def test_run_task_lane_particle_auto_device_uses_cpu_parallelism_when_cuda_is_unavailable(self):
        tasks = [
            {
                "cfg": {"particle_env": {"fidelity_env": {"device": "auto"}}},
                "variant_name": f"particle_{idx}",
            }
            for idx in range(2)
        ]
        worker_counts: list[int] = []

        class _FakeFuture:
            def __init__(self, fn, task):
                self._fn = fn
                self._task = task

            def result(self):
                return self._fn(self._task)

        class _FakeExecutor:
            def __init__(self, *args, **kwargs):
                del args
                worker_counts.append(int(kwargs["max_workers"]))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, task):
                return _FakeFuture(fn, task)

        with mock.patch.object(experiment, "ProcessPoolExecutor", _FakeExecutor):
            with mock.patch.object(experiment, "as_completed", side_effect=lambda futures: list(futures)):
                with mock.patch.object(experiment.torch.cuda, "is_available", return_value=False):
                    with mock.patch.object(
                        experiment,
                        "_run_variant_task",
                        side_effect=lambda task: self._fake_row(str(task["variant_name"])),
                    ):
                        experiment._run_task_lane(
                            tasks,
                            exec_mode="process",
                            max_workers=5,
                            lane_key="task_pool:particle_sim",
                            backend_kind="particle_sim",
                        )

        self.assertEqual(worker_counts, [2])

    def test_wm_batched_variants_allocate_shared_plan_time(self):
        class FakeEnv:
            action_dim = 2

            def __init__(self):
                self._step_calls = 0

            def prepare(self, seed=0, init_state=None, goal_state=None):
                del seed, goal_state
                state = np.asarray(init_state, dtype=np.float32).copy()
                obs = {"visual": np.zeros((4, 4, 3), dtype=np.uint8)}
                return obs, state

            def render(self, mode, include_start_pose=False):
                del mode, include_start_pose
                return np.zeros((4, 4, 3), dtype=np.uint8)

            def eval_termination(self, goal_state, cur_state, done=None, info=None):
                del goal_state, cur_state, info
                reached = bool(done)
                return {
                    "done": reached,
                    "success": reached,
                    "pos_diff": 0.0,
                    "angle_diff": 0.0,
                    "eef_diff": 0.0,
                    "coverage": 1.0 if reached else 0.0,
                }

            def step(self, action):
                del action
                self._step_calls += 1
                next_state = np.full((7,), float(self._step_calls), dtype=np.float32)
                obs = {"visual": np.zeros((4, 4, 3), dtype=np.uint8)}
                return obs, 0.0, True, {"state": next_state}

        class FakeBatchPlanner:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def plan_batch(self, z0, z_goal, mpc_progress=0.0, warm_start_steps=0, seeds=None):
                del z0, z_goal, mpc_progress, warm_start_steps, seeds
                results = []
                for fill_value in (0.0, 1.0):
                    info = SimpleNamespace(
                        base_level_idx=1,
                        base_k=2,
                        rollout_level_indices=[1],
                        rollout_latent_losses=[0.5],
                        iter_best_rollout_latent_losses=[],
                        bits_used_estimate=32,
                        plan_time_sec=8.0,
                    )
                    results.append(
                        SimpleNamespace(
                            action_seq=torch.full((1, 2), fill_value, dtype=torch.float32),
                            info=info,
                        )
                    )
                return results

        runtime_cfg = OmegaConf.create(
            {
                "env_id": "pusht",
                "env": {},
                "backend": "wm",
                "mpc": {"horizon": 1, "steps": 1, "replan_every": 1},
                "cem": {
                    "pop_size": 4,
                    "elite_frac": 0.5,
                    "n_iter": 1,
                    "init_std": 1.0,
                    "action_low": None,
                    "action_high": None,
                    "warm_start": True,
                },
                "objective": {},
                "fidelity": {},
                "save": False,
            }
        )
        variants = [
            SimpleNamespace(name="variant_a", plan=SimpleNamespace(runtime_cfg=runtime_cfg)),
            SimpleNamespace(name="variant_b", plan=SimpleNamespace(runtime_cfg=runtime_cfg)),
        ]
        init_state = np.zeros((7,), dtype=np.float32)
        goal_state = np.ones((7,), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                experiment,
                "build_plan_runtime",
                return_value={"env": FakeEnv(), "wm": object(), "device": torch.device("cpu"), "wm_cfg": None},
            ):
                with mock.patch.object(experiment, "_make_exec_env", side_effect=lambda cfg: FakeEnv()):
                    with mock.patch.object(
                        experiment,
                        "load_selected_rollout",
                        return_value=(init_state, goal_state, {"rollout_id": "r0", "rollout_index": 0}),
                    ):
                        with mock.patch.object(experiment, "set_goal_pose", side_effect=lambda env, state: None):
                            with mock.patch.object(experiment, "set_start_pose", side_effect=lambda env, state: None):
                                with mock.patch.object(
                                    experiment,
                                    "set_execution_fidelity_finest",
                                    side_effect=lambda env: None,
                                ):
                                    with mock.patch.object(
                                        experiment,
                                        "encode_visual",
                                        side_effect=lambda wm, visual, device: torch.zeros((1, 4), dtype=torch.float32),
                                    ):
                                        with mock.patch.object(
                                            experiment,
                                            "_wm_termination_latent_loss",
                                            return_value=0.0,
                                        ):
                                            with mock.patch.object(
                                                experiment,
                                                "BatchedLatentCEMPlanner",
                                                FakeBatchPlanner,
                                            ):
                                                with mock.patch.object(
                                                    experiment,
                                                    "save_plan_result",
                                                    side_effect=lambda *args, **kwargs: None,
                                                ):
                                                    with mock.patch.object(
                                                        experiment,
                                                        "result_row",
                                                        side_effect=lambda result, run_dir: {
                                                            "variant_name": result["variant_name"],
                                                            "plan_time_total_sec": result["run_stats"]["plan_time_total_sec"],
                                                            "shared_plan_time_total_sec": result["run_stats"][
                                                                "shared_plan_time_total_sec"
                                                            ],
                                                            "replan_plan_time_sec": result["trace"]["replans"][0][
                                                                "plan_time_sec"
                                                            ],
                                                            "replan_shared_plan_time_sec": result["trace"]["replans"][0][
                                                                "shared_plan_time_sec"
                                                            ],
                                                            "plan_time_allocation": result["trace"]["replans"][0][
                                                                "plan_time_allocation"
                                                            ],
                                                        },
                                                    ):
                                                        rows = experiment._run_wm_batched_variants(
                                                            variants=variants,
                                                            selection={"rollout_id": "r0", "rollout_index": 0},
                                                            run_root=tmpdir,
                                                            seed_base=0,
                                                            terminal_mode="quiet",
                                                        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["plan_time_total_sec"], 4.0)
            self.assertEqual(row["shared_plan_time_total_sec"], 8.0)
            self.assertEqual(row["replan_plan_time_sec"], 4.0)
            self.assertEqual(row["replan_shared_plan_time_sec"], 8.0)
            self.assertEqual(row["plan_time_allocation"], "equal_split")

    def test_dummy_multivariant_bundle_reuses_rollout_ids_across_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = os.path.join(tmpdir, "source_bundle")
            trace_root = os.path.join(source_dir, "traces", "variant_a")
            os.makedirs(os.path.join(trace_root, "rollout_0"), exist_ok=True)
            os.makedirs(os.path.join(trace_root, "rollout_1"), exist_ok=True)
            for rollout_id in ("rollout_0", "rollout_1"):
                rollout_dir = os.path.join(trace_root, rollout_id)
                with open(os.path.join(rollout_dir, "trace.json"), "w", encoding="utf-8") as f:
                    json.dump({"ok": True}, f)
                with open(os.path.join(rollout_dir, "trace.npz"), "wb") as f:
                    f.write(b"npz")
                with open(os.path.join(rollout_dir, "run.log"), "w", encoding="utf-8") as f:
                    f.write("log\n")

            write_experiment_bundle(
                source_dir,
                experiment_payload={
                    "schema_version": 1,
                    "reviewer_version": 1,
                    "experiment_name": "source_demo",
                    "variant_order": ["variant_a"],
                    "num_rollouts": 2,
                },
                selected_rollouts=[
                    {"rollout_id": "rollout_0", "rollout_index": 0},
                    {"rollout_id": "rollout_1", "rollout_index": 1},
                ],
                run_rows=[
                    {
                        "variant_name": "variant_a",
                        "rollout_id": "rollout_0",
                        "rollout_index": 0,
                        "success": 1,
                        "success_and_done": 1,
                        "termination_reason": "env_done",
                        "executed_steps": 1,
                        "plans": 1,
                        "final_pos_diff": 1.0,
                        "final_angle_diff": 0.1,
                        "final_eef_diff": 0.2,
                        "best_pos_diff": 1.0,
                        "best_angle_diff": 0.1,
                        "best_eef_diff": 0.2,
                        "final_coverage": 0.9,
                        "auc_pos_diff": 1.0,
                        "auc_angle_diff": 0.1,
                        "auc_eef_diff": 0.2,
                        "bits_used_total": 100.0,
                        "bits_used_per_step": 100.0,
                        "flops_used_total": 200.0,
                        "flops_used_per_step": 200.0,
                        "plan_time_total_sec": 0.5,
                        "plan_time_per_replan_sec": 0.5,
                    },
                    {
                        "variant_name": "variant_a",
                        "rollout_id": "rollout_1",
                        "rollout_index": 1,
                        "success": 0,
                        "success_and_done": 0,
                        "termination_reason": "max_steps",
                        "executed_steps": 2,
                        "plans": 1,
                        "final_pos_diff": 2.0,
                        "final_angle_diff": 0.2,
                        "final_eef_diff": 0.3,
                        "best_pos_diff": 2.0,
                        "best_angle_diff": 0.2,
                        "best_eef_diff": 0.3,
                        "final_coverage": 0.4,
                        "auc_pos_diff": 2.0,
                        "auc_angle_diff": 0.2,
                        "auc_eef_diff": 0.3,
                        "bits_used_total": 150.0,
                        "bits_used_per_step": 75.0,
                        "flops_used_total": 300.0,
                        "flops_used_per_step": 150.0,
                        "plan_time_total_sec": 0.7,
                        "plan_time_per_replan_sec": 0.7,
                    },
                ],
                variant_rows=[{"variant_name": "variant_a", "n_rollouts": 2, "success_rate": 0.5}],
                paired_rows=[],
            )

            output_dir = os.path.join(tmpdir, "dummy_bundle")
            build_dummy_bundle(source_dir, output_dir, overwrite=False)

            with open(os.path.join(output_dir, RUNS_CSV), "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            by_variant: dict[str, set[str]] = {}
            for row in rows:
                by_variant.setdefault(str(row["variant_name"]), set()).add(str(row["rollout_id"]))

            self.assertEqual(set(by_variant.keys()), set(FAKE_VARIANTS))
            self.assertTrue(all(rollout_ids == {"rollout_0", "rollout_1"} for rollout_ids in by_variant.values()))

            with open(os.path.join(output_dir, EXPERIMENT_JSON), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["num_rollouts"], 2)

    def test_run_experiment_accepts_plain_dict_runtime_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_plan = mock.Mock(clean_cfg=OmegaConf.create({"task": {"name": "demo"}}))
            variant_plan = mock.Mock(
                runtime_cfg=OmegaConf.create({"artifacts": {"save": False}}),
                clean_cfg=OmegaConf.create({"backend": {"kind": "gt_env"}}),
            )
            spec = ExperimentSpec(
                name="demo",
                config_path=None,
                shared_plan=shared_plan,
                baseline="variant_a",
                variants=[ExperimentVariant(name="variant_a", plan=variant_plan)],
                rollouts={"seed": 0, "num_rollouts": 1, "sample_without_replacement": True},
                execution={"mode": "serial", "max_workers": 1},
                terminal={"mode": "quiet"},
                reporting={"output_root": tmpdir},
            )
            fake_row = {
                "variant_name": "variant_a",
                "rollout_id": "r0",
                "rollout_index": 0,
                "success": 1,
                "success_and_done": 1,
                "termination_reason": "env_done",
                "executed_steps": 1,
                "plans": 1,
                "final_pos_diff": 1.0,
                "final_angle_diff": 0.1,
                "final_eef_diff": 0.2,
                "best_pos_diff": 1.0,
                "best_angle_diff": 0.1,
                "best_eef_diff": 0.2,
                "final_coverage": 0.9,
                "auc_pos_diff": 1.0,
                "auc_angle_diff": 0.1,
                "auc_eef_diff": 0.2,
                "bits_used_total": 100.0,
                "bits_used_per_step": 100.0,
                "flops_used_total": 200.0,
                "flops_used_per_step": 200.0,
                "plan_time_total_sec": 0.5,
                "plan_time_per_replan_sec": 0.5,
            }
            with mock.patch.object(experiment, "enumerate_rollout_candidates", return_value=[{"rollout_id": "r0"}]):
                with mock.patch.object(experiment, "select_rollouts", return_value=[{"rollout_id": "r0", "rollout_index": 0}]):
                    with mock.patch.object(
                        experiment,
                        "_build_execution_lanes",
                        return_value=[
                            {
                                "lane_type": "task_pool",
                                "backend_kind": "gt_env",
                                "lane_key": "task_pool:gt_env",
                                "tasks": [
                                    {
                                        "cfg": {"save": False},
                                        "selection": {"rollout_id": "r0", "rollout_index": 0},
                                        "variant_name": "variant_a",
                                        "run_dir": os.path.join(tmpdir, "traces", "variant_a", "r0"),
                                        "terminal_mode": "quiet",
                                        "backend_kind": "gt_env",
                                        "lane_key": "task_pool:gt_env",
                                    }
                                ],
                            }
                        ],
                    ):
                        with mock.patch.object(experiment, "_run_variant_task", return_value=fake_row):
                            run_dir = experiment.run_experiment(spec)

            with open(os.path.join(run_dir, EXPERIMENT_JSON), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["rollouts"]["num_rollouts"], 1)
            self.assertEqual(payload["execution"]["mode"], "serial")
            self.assertEqual(payload["terminal"]["mode"], "quiet")
            self.assertEqual(payload["reporting"]["output_root"], tmpdir)
            self.assertEqual(payload["baseline_variant"], "variant_a")

    def test_run_variant_task_writes_run_log(self):
        fake_result = {
            "cfg": mock.Mock(save=False),
            "runtime": {"backend": "gt_env"},
            "success": True,
            "trajectory": [],
            "frames": [],
            "planner_frames": [],
            "run_stats": {},
            "trace": {},
            "init_state": [],
            "goal_state": [],
            "sample_meta": {},
        }
        task = {
            "cfg": {"save": False},
            "selection": {"rollout_id": "r0", "rollout_index": 0},
            "variant_name": "variant_a",
            "run_dir": "",
            "terminal_mode": "compact",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            task["run_dir"] = os.path.join(tmpdir, "traces", "variant_a", "r0")
            with mock.patch.object(experiment, "run_plan_session", return_value=fake_result):
                with mock.patch.object(experiment, "save_plan_result", side_effect=lambda *args, **kwargs: print("saved")):
                    with mock.patch.object(experiment, "result_row", return_value={"variant_name": "variant_a", "rollout_id": "r0"}):
                        row = experiment._run_variant_task(task)

            self.assertEqual(row["variant_name"], "variant_a")
            log_path = os.path.join(task["run_dir"], "run.log")
            self.assertTrue(os.path.isfile(log_path))
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("saved", content)

    def test_write_experiment_bundle_emits_canonical_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "experiment_demo")
            write_experiment_bundle(
                run_dir,
                experiment_payload={
                    "schema_version": 1,
                    "reviewer_version": 1,
                    "experiment_name": "demo",
                    "variant_order": ["variant_a"],
                },
                selected_rollouts=[{"rollout_id": "rollout_0"}],
                run_rows=[{"variant_name": "variant_a", "rollout_id": "rollout_0", "success": 1}],
                variant_rows=[{"variant_name": "variant_a", "n_rollouts": 1, "success_rate": 1.0}],
                paired_rows=[{"variant_name": "variant_a", "rollout_id": "rollout_0", "success_delta": 0}],
            )

            for filename in (
                EXPERIMENT_JSON,
                RUNS_CSV,
                VARIANTS_CSV,
                PAIRED_VS_BASELINE_CSV,
                SELECTED_ROLLOUTS_JSON,
            ):
                self.assertTrue(os.path.isfile(os.path.join(run_dir, filename)), msg=filename)

            self.assertFalse(os.path.exists(os.path.join(run_dir, "experiment_report.html")))
            self.assertFalse(os.path.exists(os.path.join(run_dir, "summary.csv")))
            with open(os.path.join(run_dir, EXPERIMENT_JSON), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["experiment_name"], "demo")

    def test_migrate_legacy_experiment_dir_writes_canonical_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "experiment_legacy")
            os.makedirs(os.path.join(run_dir, "traces", "variant_a", "rollout_0"), exist_ok=True)

            with open(os.path.join(run_dir, "experiment_resolved.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": "legacy_demo",
                        "baseline": "variant_a",
                        "rollouts": {"num_rollouts": 1},
                        "execution": {"mode": "serial"},
                        "terminal": {"mode": "compact"},
                        "reporting": {"output_root": "rollouts"},
                        "shared_plan": {"imports": ["task/demo.yaml"]},
                        "variants": [{"name": "variant_a", "imports": ["backend/demo.yaml"]}],
                    },
                    f,
                    indent=2,
                )
            with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "experiment_name": "legacy_demo",
                        "created_at": "2026-03-16T00:00:00",
                        "num_rollouts": 1,
                        "baseline_variant": "variant_a",
                        "summary": [],
                    },
                    f,
                    indent=2,
                )
            with open(os.path.join(run_dir, SELECTED_ROLLOUTS_JSON), "w", encoding="utf-8") as f:
                json.dump([{"rollout_id": "rollout_0", "rollout_index": 0}], f, indent=2)
            with open(os.path.join(run_dir, "per_rollout.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("variant_name,rollout_id,success\nvariant_a,rollout_0,1\n")
            with open(os.path.join(run_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("variant_name,n_rollouts,success_rate\nvariant_a,1,1.0\n")

            migrate_legacy_experiment_dir(run_dir)

            for filename in (
                EXPERIMENT_JSON,
                RUNS_CSV,
                VARIANTS_CSV,
                PAIRED_VS_BASELINE_CSV,
                SELECTED_ROLLOUTS_JSON,
            ):
                self.assertTrue(os.path.isfile(os.path.join(run_dir, filename)), msg=filename)

            data = load_experiment_review_data(run_dir)
            self.assertEqual(data.experiment_name, "legacy_demo")
            self.assertEqual(data.baseline_variant, "variant_a")
            self.assertEqual(len(data.run_rows), 1)
            self.assertEqual(len(data.variant_rows), 1)


if __name__ == "__main__":
    unittest.main()
