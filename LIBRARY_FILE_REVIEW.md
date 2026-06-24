# MWM Library File Review

Scope: current working tree source, configs, docs, tests, and scripts. Excluded from this file-by-file review are generated or bulky runtime artifacts: `data/`, `rollouts/`, `logs/`, `checkpoints_mwm/`, `.worktrees/`, `.pytest_cache/`, `__pycache__/`, and `*.pyc`.

## System Purpose

This repository is a Stable-WM-compatible Matryoshka World Models benchmark and evaluation library. The active contract is narrow: build MWM checkpoints from Stable-WM Le-WM configs, train/evaluate canonical checkpoints on Lance datasets, run scheduled/fixed fidelity CEM planning, and verify benchmark artifacts with reproducible manifests.

## Root Files

- `.gitignore`: Keeps generated model/data/rollout/cache artifacts out of git.
- `README.md`: User-facing overview, quick start, architecture, benchmark roles, and local/Slurm workflow notes.
- `REVIEW_GUIDE.md`: Reviewer contract: current expected runtime surface, removed legacy APIs, and validation commands.
- `LIBRARY_FILE_REVIEW.md`: This file; a file-by-file orientation map for reviewing the current library surface.
- `requirements.txt`: Python dependency list, including pinned `stable-worldmodel[env]==0.1.0`.
- `collect_mwm_data.py`: CLI for collecting Stable-WM world rollouts into Lance datasets and writing MWM dataset metadata sidecars.
- `prepare_upstream_lewm.py`: Converts trusted upstream Le-WM checkpoints into canonical MWM identity-parity checkpoints by building an MWM shell and copying upstream weights.
- `prepare_upstream_lewm_data.py`: Verifies prebuilt upstream Lance datasets exist and writes paper-parity metadata sidecars for PushT and TwoRoom.
- `train_mwm.py`: Thin root CLI for Le-WM base-adapter training or exporting a Lightning checkpoint to canonical MWM format.
- `eval_mwm.py`: Thin root CLI that delegates checkpoint evaluation to `mwm.eval.runner`.
- `benchmark_mwm.py`: Thin root CLI that delegates benchmark matrix execution to `mwm.benchmark.matrix`.
- `verify_mwm_data.py`: Thin root CLI that delegates Lance dataset/config verification to `mwm.data.verify`.
- `verify_mwm_benchmark.py`: Thin root CLI that delegates static or output benchmark verification to `mwm.benchmark.verify`.
- `render_benchmark_review.py`: Re-renders plots, CSV/JSONL summaries, and HTML review pages from an existing benchmark output directory.

## `mwm` Package

- `mwm/__init__.py`: Canonical package marker with a lazy failure for retired root symbols such as `MWMWorldModel`.
- `mwm/imports.py`: Import-path resolver for `module.attr` or `module:attr` strings.
- `mwm/io.py`: JSON, JSONL metrics, numpy/tensor-to-JSON conversion, and file SHA utilities.
- `mwm/config_cli.py`: Shared OmegaConf loader with dotlist override support.
- `mwm/dependency_refs.py`: Captures package versions, package metadata hashes, and local git commit/dirty diff hash for provenance.
- `mwm/fidelity.py`: Fidelity scheduler and decision objects for fixed, linear-CEM, and table-driven MWM planning.
- `mwm/checkpoint_contract.py`: Semantic validator for canonical MWM checkpoint configs/metadata, including levels, action specs, component policy, and adapter family.
- `mwm/checkpoint_io.py`: Reads, writes, validates, instantiates, and loads canonical checkpoint directories containing `config.json`, `weights.pt`, and `world_metadata.json`.

### Adapters

- `mwm/adapters/__init__.py`: Public adapter API barrel; imports the Le-WM adapter for registration.
- `mwm/adapters/base.py`: Adapter protocol plus `ComponentGroup`, `ComponentPolicy`, `StableWMBaseSpec`, and policy validation.
- `mwm/adapters/builder.py`: Generic public builder `build_mwm_from_stable_config`; detects family, resolves adapter spec, builds model, and records importable config.
- `mwm/adapters/constants.py`: Shared adapter architecture version constant for Le-WM base-adaptive checkpoints.
- `mwm/adapters/lewm.py`: Only concrete adapter; parses Stable-WM Le-WM config, shares encoder/projector, creates per-K transition tails, scales head dimensions, validates action dims, and registers itself.
- `mwm/adapters/registry.py`: Adapter registry and target-to-family detection for Le-WM, PreJEPA, and PLDM family names.
- `mwm/adapters/stable_config.py`: Stable-WM `config.json` loading, root target extraction, and config file SHA hashing.

### Models

- `mwm/models/__init__.py`: Public model/loss/preprocessing exports.
- `mwm/models/core.py`: Generic `MWMWorldModel` runtime base with encode, per-level dynamics rollout, scheduled rollout, decode, and cost-with-fidelity methods.
- `mwm/models/base_adaptive.py`: Active Le-WM-shaped `MatryoshkaWorldModel`; shared image encoder/projector, per-level transition packages, training loss, fixed-level rollout, and planner cost.
- `mwm/models/transitions.py`: `TransitionPackage(action_encoder, predictor, pred_proj)` wrapper for per-level latent prediction.
- `mwm/models/losses.py`: Level-weighted aggregation, latent regularizer routing, and matryoshka base-loss composition.
- `mwm/models/objectives.py`: Le-WM-style MWM training objective over encoded latents and per-level prefix predictions.
- `mwm/models/planning_costs.py`: Helpers that enforce fixed-level rollout decisions for the current base-adaptive evaluator.
- `mwm/models/world_model.py`: Compatibility facade for older import paths; re-exports current model/loss/preprocess symbols.

### Data

- `mwm/data/__init__.py`: Empty data package marker.
- `mwm/data/loading.py`: Stable-WM dataset loader wrapper that installs the MWM training sample transform.
- `mwm/data/manifest.py`: Immutable eval manifest creation/loading, logical manifest hash, and file hash support.
- `mwm/data/metadata.py`: Dataset metadata sidecar path, read, and write helpers.
- `mwm/data/module.py`: Minimal Lightning `DataModule` around prebuilt train/validation loaders.
- `mwm/data/paths.py`: Local path resolver that returns an absolute path only when the target exists.
- `mwm/data/sampling.py`: Deterministic start-goal pair sampling in MWM or Stable-WM-compatible modes.
- `mwm/data/transforms.py`: Training sample transform, z-score scaler, action/pixel normalization, and stable-pretraining image transform assembly.
- `mwm/data/verify.py`: Lance-only data config verifier and CLI modes for default, paper-parity, and all config sets.

### Evaluation

- `mwm/eval/__init__.py`: Empty eval package marker.
- `mwm/eval/action_preprocessing.py`: Metadata/config-driven action standardization detection and scaler fitting for eval-time action/proprio/state columns.
- `mwm/eval/execution.py`: Runs batched SWM evaluation worlds, builds policies, restores starts/goals, gathers videos, and combines SWM/MWM diagnostics.
- `mwm/eval/manifest.py`: Converts manifest rows to `StartGoalPair`s or samples/writes manifests for eval runs.
- `mwm/eval/policy.py`: Stable-WM policy wrapper that tracks action calls, plan time, latent work, and solver diagnostics.
- `mwm/eval/policy_builder.py`: Constructs `MWMScheduledCEMSolver`, `PlanConfig`, image transforms, and `MWMWorldModelPolicy`.
- `mwm/eval/runner.py`: Main evaluation orchestrator: config load, checkpoint load, dataset load, metadata/restore validation, manifest selection, batch execution, and output JSON writing.
- `mwm/eval/validation.py`: Dataset metadata validation, manifest validation, keys-to-load resolution, dataset path/runtime metadata helpers, and dataset close helper.

### Benchmarking

- `mwm/benchmark/__init__.py`: Empty benchmark package marker.
- `mwm/benchmark/analysis.py`: Common row sorting, role/env labels, paired baseline-vs-comparison rows, and aggregate outcome rows.
- `mwm/benchmark/config.py`: Benchmark defaults and config expansion/validation, including manifest config, role filtering, run config merging, and duplicate-cell rejection.
- `mwm/benchmark/html.py`: Static HTML review renderer with status cards, warnings, outcome tables, plots, drilldown links, notes, and media links.
- `mwm/benchmark/io.py`: Per-run sidecar writer plus public re-exports for common IO helpers.
- `mwm/benchmark/matrix.py`: Executes benchmark run matrices, manages shared manifests, logs failures, writes summaries/sidecars/traces/plots/review HTML.
- `mwm/benchmark/plots.py`: Matplotlib plot generation for success vs compute/time, by-env success, paired deltas, efficiency ratios, and scheduler usage.
- `mwm/benchmark/summary.py`: Converts eval payloads into summary rows and writes aggregate/per-env CSV tables.
- `mwm/benchmark/verify.py`: Static and output verifier for benchmark completeness, shared manifests, dependency refs, checkpoint role contracts, paper targets, plots, HTML links, and sidecars.

### Planning, Preprocessing, SWM

- `mwm/planning/__init__.py`: Empty planning package marker.
- `mwm/planning/scheduled_cem.py`: Stable-WM-compatible CEM solver with explicit MWM fidelity decisions, diagnostics, batching, warm starts, top-k updates, and optional action clamping.
- `mwm/preprocessing/__init__.py`: Public preprocessing exports.
- `mwm/preprocessing/images.py`: ImageNet normalization, MWM policy image transform, BCHW layout helper, idempotent preprocessing, and stable-pretraining transform setup.
- `mwm/swm/__init__.py`: Empty Stable-WM integration package marker.
- `mwm/swm/envs.py`: Stable-WM world factory, image-shape/env-kwargs parsing, continuous Box action validation, and action-space inference.
- `mwm/swm/restore.py`: Built-in and user-provided restore specs for PushT, TwoRoom/Piecewise, and DMControl-style datasets; returns eval callables for SWM starts/goals.

### Training

- `mwm/training/__init__.py`: Training package marker.
- `mwm/training/lewm.py`: Training CLI orchestration: load config, seed, build run dir, prepare data/model, run Lightning, save canonical checkpoint, or dispatch export mode.
- `mwm/training/lewm_config.py`: Training defaults, run-directory creation, and OmegaConf-to-container conversion.
- `mwm/training/lewm_data.py`: Lance dataset loading/splitting, transform installation, restore validation, model config resolution, dataset metadata, and checkpoint metadata preparation.
- `mwm/training/lewm_model.py`: Resolves model dimensions from dataset/base config, locates Stable-WM cached config, builds trainable MWM from base, and merges model metadata.
- `mwm/training/lewm_lightning.py`: Stable-pretraining/Lightning training loop, DataLoaders, optimizer/scheduler config, callbacks, module forward, selected checkpoint reload, and train-info output.
- `mwm/training/lewm_callbacks.py`: ModelCheckpoint builder, all-level plateau early stopping, callback assembly, and export checkpoint selection policy.
- `mwm/training/lewm_export.py`: Loads Lightning `model.*` state into MWM and exports a canonical checkpoint without retraining.
- `mwm/training/lewm_runtime.py`: Device/strategy resolution, trainer-root cleanup, and total LR scheduler step calculation.

## Config Files

- `configs/collect/mwm_pusht.yaml`: Collects 100 PushT Lance episodes at 224px with default random policy and built-in restore metadata.
- `configs/collect/mwm_tworoom.yaml`: Collects 100 TwoRoom Lance episodes at 224px with longer max steps and more envs.
- `configs/eval/mwm_lewm_pusht.yaml`: PushT eval on local collected dataset with linear CEM fidelity schedule and auto action preprocessing.
- `configs/eval/mwm_lewm_tworoom.yaml`: TwoRoom eval on local collected dataset with TwoRoom columns and linear CEM fidelity schedule.
- `configs/eval/paper_pusht.yaml`: Paper-parity PushT eval against upstream Lance data with fixed finest-level planning and Stable-WM sampling.
- `configs/eval/paper_tworoom.yaml`: Paper-parity TwoRoom eval against upstream Lance data with fixed finest-level planning and Stable-WM sampling.
- `configs/local/collect_pusht_smoke.yaml`: Tiny local PushT collection smoke config.
- `configs/local/eval_pusht_smoke.yaml`: CPU-safe two-episode PushT eval smoke config.
- `configs/local/benchmark_pusht_smoke.yaml`: One-role local benchmark wrapper around the PushT smoke eval config.
- `configs/local/train_pusht_cpu_smoke.yaml`: Opt-in one-epoch CPU training smoke for a single K=192 PushT identity model.
- `configs/manifest/pusht_paper_seed42.yaml`: Named manifest location for PushT paper seed 42.
- `configs/manifest/tworoom_paper_seed42.yaml`: Named manifest location for TwoRoom paper seed 42.
- `configs/train/mwm_lewm_pusht.yaml`: PushT identity-style retraining on locally collected data with K=[192].
- `configs/train/mwm_lewm_tworoom.yaml`: TwoRoom identity-style retraining on locally collected data with K=[192].
- `configs/train/mwm_lewm_pusht_upstream.yaml`: PushT identity retraining on upstream paper-parity data.
- `configs/train/mwm_lewm_tworoom_upstream.yaml`: TwoRoom identity retraining on upstream paper-parity data.
- `configs/train/mwm_scheduled_pusht.yaml`: PushT scheduled MWM training with K=[48,96,144].
- `configs/train/mwm_scheduled_tworoom.yaml`: TwoRoom scheduled MWM training with K=[48,96,144].
- `configs/train/mwm_dense_pusht.yaml`: PushT dense-level MWM training with K=[6,12,48,96,144,192].
- `configs/train/mwm_dense_tworoom.yaml`: TwoRoom dense-level MWM training with K=[6,12,48,96,144,192].
- `configs/benchmark/paper_parity_pusht.yaml`: PushT paper target benchmark comparing converted upstream and retrained identity checkpoints.
- `configs/benchmark/paper_parity_tworoom.yaml`: TwoRoom paper target benchmark comparing converted upstream and retrained identity checkpoints.
- `configs/benchmark/scheduled_pusht.yaml`: PushT benchmark comparing converted upstream to scheduled MWM.
- `configs/benchmark/scheduled_tworoom.yaml`: TwoRoom benchmark comparing converted upstream to scheduled MWM.
- `configs/benchmark/dense_pusht.yaml`: PushT benchmark comparing converted upstream to dense-level MWM.
- `configs/benchmark/dense_tworoom.yaml`: TwoRoom benchmark comparing converted upstream to dense-level MWM.
- `configs/research/identity_delta_pusht_eval.yaml`: Artifact-root-parametric PushT eval config for identity-vs-upstream seed sweep.
- `configs/research/identity_delta_tworoom_eval.yaml`: Artifact-root-parametric TwoRoom eval config for identity-vs-upstream seed sweep.
- `configs/research/identity_delta_pusht_benchmark.yaml`: PushT seed-sweep benchmark comparing upstream and retrained identity roles.
- `configs/research/identity_delta_tworoom_benchmark.yaml`: TwoRoom seed-sweep benchmark comparing upstream and retrained identity roles.
- `configs/research/train_mwm_dense_pusht_highk_weighted.yaml`: Research PushT dense training with high-K-weighted level losses into dense debug outputs.
- `configs/research/train_mwm_dense_tworoom_highk_weighted.yaml`: Research TwoRoom dense training with high-K-weighted level losses into dense debug outputs.
- `configs/research/train_mwm_dense_pusht_highk_weighted_converge.yaml`: PushT high-K weighted dense training with convergence early stopping and best-checkpoint export.
- `configs/research/train_mwm_dense_tworoom_highk_weighted_converge.yaml`: TwoRoom high-K weighted dense training with convergence early stopping and best-checkpoint export.

## Scripts

- `scripts/README.md`: Script boundary rules and subdirectory roles.
- `scripts/local/local_verify.sh`: Desktop verification: py_compile, pytest, and local static benchmark check.
- `scripts/local/local_benchmark_smoke.sh`: Local PushT smoke benchmark after checking required data/checkpoint artifacts.
- `scripts/local/local_train_smoke.sh`: Opt-in CPU training smoke wrapper.
- `scripts/research/research_identity_delta_audit.py`: Deep audit script comparing identity/upstream checkpoints, configs, datasets, logs, rollouts, and writing markdown/json research reports.
- `scripts/research/research_identity_delta_collect.py`: Aggregates seed-sweep benchmark summaries, failure overlaps, and identity-minus-upstream deltas.
- `scripts/research/run_cem_sweep.py`: Research helper that runs a CEM scheduler/population sweep across selected envs and writes aggregate sweep results.
- `scripts/research/research_identity_seed_sweep.sh`: Slurm-only multi-seed identity-delta benchmark driver for PushT and TwoRoom.
- `scripts/research/research_train_dense_highk_converge.sh`: Runs a selected high-K weighted dense convergence training config.
- `scripts/research/research_train_dense_highk_converge.sbatch`: Slurm wrapper for high-K dense convergence training with environment diagnostics.
- `scripts/research/slurm_research_identity_seed_sweep.sbatch`: Slurm wrapper for the identity seed sweep.
- `scripts/slurm/run_mwm_train_identity_env.sh`: Slurm-allocation runner for identity training by env.
- `scripts/slurm/run_mwm_train_scheduled_env.sh`: Slurm-allocation runner for scheduled MWM training by env.
- `scripts/slurm/run_mwm_train_dense_env.sh`: Slurm-allocation runner for dense MWM training by env.
- `scripts/slurm/run_mwm_paper_parity.sh`: End-to-end paper-parity allocation workflow: prepare upstream checkpoints/data, verify, benchmark upstream, train identity, benchmark both.
- `scripts/slurm/run_mwm_identity_parity.sh`: Identity-parity benchmark runner that continues across PushT/TwoRoom failures and exits nonzero if any failed.
- `scripts/slurm/run_mwm_scheduled_comparison.sh`: Scheduled comparison benchmark runner that verifies data and continues across env failures.
- `scripts/slurm/run_mwm_dense_comparison.sh`: Dense comparison benchmark runner with the same continue-and-report pattern.
- `scripts/slurm/slurm_mwm_train_pusht_identity.sbatch`: One-GPU PushT identity training batch job.
- `scripts/slurm/slurm_mwm_train_tworoom_identity.sbatch`: One-GPU TwoRoom identity training batch job.
- `scripts/slurm/slurm_mwm_train_pusht_scheduled.sbatch`: One-GPU PushT scheduled MWM training batch job.
- `scripts/slurm/slurm_mwm_train_tworoom_scheduled.sbatch`: One-GPU TwoRoom scheduled MWM training batch job.
- `scripts/slurm/slurm_mwm_train_pusht_dense.sbatch`: One-GPU PushT dense MWM training batch job.
- `scripts/slurm/slurm_mwm_train_tworoom_dense.sbatch`: One-GPU TwoRoom dense MWM training batch job.
- `scripts/slurm/slurm_mwm_paper_parity.sbatch`: Batch job for the full paper-parity workflow.
- `scripts/slurm/slurm_mwm_identity_parity.sbatch`: Batch job for identity-parity benchmark comparison.
- `scripts/slurm/slurm_mwm_scheduled_comparison.sbatch`: Batch job for scheduled benchmark comparison.
- `scripts/slurm/slurm_mwm_dense_comparison.sbatch`: Batch job for dense benchmark comparison.
- `scripts/slurm/submit_mwm_identity_split.sh`: Submits PushT and TwoRoom identity training jobs plus dependent identity benchmark.
- `scripts/slurm/submit_mwm_scheduled_split.sh`: Submits PushT and TwoRoom scheduled training jobs plus dependent scheduled benchmark and monitor commands.
- `scripts/slurm/submit_mwm_dense_split.sh`: Submits PushT and TwoRoom dense training jobs plus dependent dense benchmark and monitor commands.
- `scripts/slurm/poll_mwm_identity_jobs.sh`: Polls/report status for identity workflow Slurm jobs.
- `scripts/slurm/poll_mwm_scheduled_jobs.sh`: Polls/report status for scheduled workflow Slurm jobs.
- `scripts/slurm/poll_mwm_dense_jobs.sh`: Polls/report status for dense workflow Slurm jobs.

## Docs And Reports

- `docs/mwm_adapter_contract.md`: Contract for implementing completed Stable-WM base adapters and keeping generic MWM runtime semantics centralized.
- `docs/superpowers/specs/2026-05-28-base-adaptive-mwm-design.md`: Design spec for the base-adaptive MWM framework.
- `docs/superpowers/specs/2026-05-30-dense-mwm-performance-debug.md`: Research spec for dense MWM performance investigation.
- `docs/superpowers/specs/2026-05-30-identity-upstream-delta-research.md`: Research spec for identity-vs-upstream delta investigation.
- `docs/superpowers/plans/2026-05-30-local-desktop-workflow.md`: Implementation plan for local desktop smoke workflow support.
- `docs/superpowers/plans/2026-05-30-scheduled-mwm-comparison.md`: Implementation plan for scheduled MWM comparison workflow.
- `reports/research/README.md`: Short index for research report folders.
- `reports/research/dense_debug/report.md`: Dense MWM performance debugging narrative report.
- `reports/research/dense_debug/summary.json`: Machine-readable dense debug summary.
- `reports/research/dense_debug/next_training_plan.md`: Dense debug follow-up training plan in prose.
- `reports/research/dense_debug/next_training_plan.json`: Machine-readable dense debug follow-up plan.
- `reports/research/dense_debug/equal_long_failure_investigation.md`: Investigation notes for equal-long dense failure behavior.
- `reports/research/dense_debug/equal_long_failure_investigation.json`: Machine-readable equal-long failure investigation facts.
- `reports/research/identity_delta/report.md`: Identity-vs-upstream performance delta report.
- `reports/research/identity_delta/summary.json`: Machine-readable identity delta summary.
- `reports/research/identity_delta/static_audit.md`: Static checkpoint/config/training audit writeup.
- `reports/research/identity_delta/dataset_audit.md`: Dataset distribution audit writeup.
- `reports/research/identity_delta/audit_raw.json`: Large raw audit payload backing the identity delta reports.
- `reports/research/identity_delta/seed_sweep_summary.csv`: Tabular seed-sweep aggregate summary.
- `reports/research/identity_delta/seed_sweep_summary.json`: Machine-readable seed-sweep aggregate summary.
- `reports/research/identity_delta/seed_sweep/.gitignore`: Keeps generated per-seed sweep outputs out of git.

## Tests

- `tests/test_mwm_base_adapter.py`: Adapter policy, Stable-WM config loading/family detection, Le-WM adapter shape scaling, generic builder target, and unsupported-family behavior.
- `tests/test_mwm_core.py`: Core model, loss, checkpoint, Le-WM parity, data transform, policy/action preprocessing, scheduled CEM, Lightning runtime/callback/export behavior, and scheduler monotonicity tests.
- `tests/test_mwm_artifacts.py`: Manifest/checkpoint/benchmark verifier/data verifier contracts, paper target checks, failure logging, role filtering, and artifact sidecar behavior.
- `tests/test_mwm_config_cli.py`: Shared config loader and root CLI export override rejection tests.
- `tests/test_mwm_data_boundaries.py`: Ensures data helpers, transforms, and preprocessing live in canonical modules after migration.
- `tests/test_mwm_local_workflow.py`: Ensures local smoke configs/scripts stay small, CPU-safe, and not Slurm/PARCC gated.
- `tests/test_mwm_migration_hygiene.py`: Enforces module split, absence of retired facades, public script imports, and canonical ownership boundaries.
- `tests/test_mwm_repo_hygiene.py`: Broad repo/config/script/doc hygiene tests for Lance-only paths, scheduler branch use, Slurm gating, removed legacy surfaces, and grouped workflows.

## Quick Review Notes

- The library's current public runtime path is `build_mwm_from_stable_config -> LeWMStableWMAdapter -> MatryoshkaWorldModel -> MWMWorldModelPolicy -> MWMScheduledCEMSolver`.
- The single active adapter is Le-WM. PreJEPA/PLDM are recognized as families by name, but no runtime adapter is implemented for them.
- Canonical checkpoints are deliberately strict: exactly `config.json`, `weights.pt`, and `world_metadata.json`.
- Eval and training are Lance-only; HDF5 and legacy source-object checkpoint paths are intentionally absent.
- Current scheduled planning can choose different base levels across CEM iterations, but `MatryoshkaWorldModel` currently enforces fixed-level rollouts within each plan.
- The worktree is dirty and mid-refactor: several old modules are deleted while replacement modules are untracked/new. This review describes the current filesystem state, not pristine `HEAD`.
