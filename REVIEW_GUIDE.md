# MWM V1 Review Guide

This guide is current for the on-disk repository state audited on 2026-05-27. The
previous guide was directionally correct but not complete enough for the review
directive because it only summarized the main files. This version explains the
functionality of every source/config/test/review artifact and groups bulky
generated outputs by directory and repeated file type.

## Review Mission

The repository implements Matryoshka World Models (MWM) as a Stable-WM-compatible
benchmark stack. The narrow review story is:

1. Collect Stable-WM datasets as Lance datasets with sidecar metadata.
2. Convert trusted upstream Le-WM objects into canonical MWM checkpoints.
3. Train local exact single-fidelity Le-WM and multi-fidelity MWM checkpoints
   through the same Le-WM base adapter.
4. Evaluate every role through the same `MWMWorldModel` and scheduled CEM path.
5. Generate a fixed benchmark matrix and verify that its artifacts are complete,
   reproducible, and structurally comparable.

The code intentionally removes the previous SWM-era runtime layout. Current
runtime entrypoints are the `*_mwm.py` scripts and the `mwm/` package.

## High-Level Pipeline

```text
configs/collect_mwm_*.yaml
  -> collect_mwm_data.py
  -> data/*.lance + data/*.metadata.json

prepare_upstream_lewm.py
  -> checkpoints_mwm/upstream_sources/*.pt
  -> checkpoints_mwm/upstream_lewm_*/{config.json,weights.pt,world_metadata.json}

configs/train_mwm_*.yaml
  -> train_mwm.py
  -> logs/mwm_training/*/checkpoints/*.ckpt
  -> checkpoints_mwm/{retrained_lewm_single_*,mwm_scheduled_*}/canonical files

configs/eval_mwm_*.yaml
  -> eval_mwm.py
  -> rollouts/.../eval.json and diagnostics

configs/benchmark_mwm.yaml
  -> benchmark_mwm.py
  -> rollouts/mwm_benchmark/*
  -> verify_mwm_benchmark.py
```

## Critical Review Invariants

- All evaluated checkpoints must load through `mwm.checkpoints.load_world_model_from_checkpoint`.
- Trainable Le-WM MWM checkpoints must declare
  `architecture_version: lewm_base_adapter_v1`; old generic trainable
  checkpoints are intentionally rejected and must be retrained.
- Trainable Le-WM MWM must use one shared Le-WM trunk and fresh per-`K`
  transition packages (`action_encoder`, `predictor`, `pred_proj`). It must not
  instantiate generic per-level dynamics or default image decoders.
- Fair MWM training uses Stable-WM `config.json` as an architecture source only.
  It must fresh-initialize the encoder and all duplicated tails, and checkpoint
  metadata must record `adapter_family`, `source_config_sha256`,
  `component_policy`, `fresh_init`, `loss_scope`, and `training_recipe`.
- Base-adaptive configs must declare the top-level component policy. Le-WM is
  implemented; PreJEPA/DINO-WM and PLDM only declare groups and must raise until
  explicit Stable-WM training recipe artifacts are available.
- `K=[D]` is constructor/loss/gradient/optimizer-step exact to direct base
  Le-WM. Multi-`K` training computes only the requested prefix losses; `K` may
  omit `D`.
- A canonical checkpoint directory must contain only `config.json`, `weights.pt`,
  and `world_metadata.json`.
- Checkpoint metadata format must be `mwm_world_v1`.
- Runtime datasets must be Lance (`format: lance`) with sidecar format
  `swm_lance`.
- The checkpoint, dataset sidecar, and runtime restore adapter must agree on
  `env_id`, `restore_spec`, action bounds, action dimensions, and image shape.
- `action_dim` in checkpoint metadata is the base environment action dimension,
  while `action_spec.dim` is the stacked/planner action dimension
  (`base_dim * action_block`).
- Evaluation manifests are immutable logical references for start/goal pairs.
  Roles for the same `(env_id, seed)` should share the same manifest hash.
- `FidelityScheduler` must never permit a rollout that moves from lower to higher
  fidelity within one rollout.
- The benchmark gate must contain exactly 18 cells:
  2 envs x 3 seeds x 3 roles.

## Commands To Reproduce The Review Path

```bash
python collect_mwm_data.py configs/collect_mwm_pusht.yaml
python collect_mwm_data.py configs/collect_mwm_tworoom.yaml
python verify_mwm_data.py
python prepare_upstream_lewm.py
python train_mwm.py configs/train_mwm_lewm_pusht.yaml
python train_mwm.py configs/train_mwm_lewm_tworoom.yaml
python train_mwm.py configs/train_mwm_scheduled_pusht.yaml
python train_mwm.py configs/train_mwm_scheduled_tworoom.yaml
python benchmark_mwm.py configs/benchmark_mwm.yaml
python verify_mwm_benchmark.py configs/benchmark_mwm.yaml
```

Benchmark-only gate:

```bash
scripts/run_mwm_benchmark_gate.sh
```

Paper-parity evaluator sanity gate:

```bash
python prepare_upstream_lewm_data.py
python verify_mwm_data.py \
  configs/train_mwm_lewm_pusht_upstream.yaml \
  configs/train_mwm_lewm_tworoom_upstream.yaml \
  configs/eval_mwm_paper_pusht.yaml \
  configs/eval_mwm_paper_tworoom.yaml
python train_mwm.py configs/train_mwm_lewm_pusht_upstream.yaml
python train_mwm.py configs/train_mwm_lewm_tworoom_upstream.yaml
python benchmark_mwm.py configs/benchmark_mwm_paper_parity.yaml
python verify_mwm_benchmark.py configs/benchmark_mwm_paper_parity.yaml
```

In this path, `planner.batch_size: 1` is locked to the upstream CEM solver
chunking default so RNG grouping matches the official Le-WM evaluation path as
closely as possible. The parity-relevant Le-WM evaluation settings are the
paper protocol and official eval profile:
`horizon: 5`, `receding_horizon: 5`, `action_block: 5`,
`pop_size: 300`, `topk: 30`, `init_std: 1.0`, `n_iter: 30`,
`goal_offset: 25`, `budget: 50`, `episodes: 50`, ImageNet image size 224, and
standardized non-pixel columns. These values match the upstream Le-WM eval
configs and CEM solver config
(`config/eval/solver/cem.yaml`) in https://github.com/lucas-maes/le-wm.

The retrained Le-WM paper-parity config intentionally follows the paper
protocol first, using upstream code only to confirm implementation semantics:
10 training epochs, Stable-WM Lance loading,
`num_steps = history_size + num_preds`, `frameskip: 5`, ImageNet image
preprocessing, z-score normalization for non-pixel columns, a
Stable-Pretraining random clip split, AdamW with warmup-cosine scheduling,
`bf16` precision, and the upstream Le-WM prediction/SIGReg loss. The released
training YAML currently says `max_epochs: 100`; that is treated as a repo
default, not the paper-parity target.
The config uses upstream's `num_workers: 6`; worker compatibility is checked
with a file-backed loader probe because stdin-launched multiprocessing probes can
fail for reasons unrelated to the training script. It sets `pin_memory: false`
in this environment because pinned memory is a throughput-only setting and can
put non-picklable locks on Lightning's spawned-worker path. Exact training uses
a minimal prebuilt-loader LightningDataModule so Stable-Pretraining does not
attach trainer state to Lance-backed datasets before worker spawn.
The paper-parity eval config uses `eval.sampling: stable_worldmodel`, matching
upstream row-based start/goal sampling and sorted eval order for PushT. Converted
or exact Le-WM checkpoints use `action_preprocessing: standard_scaler`: CEM plans
in the normalized action space the world model was trained on, then the policy
inverse-transforms only the selected action before stepping the environment.

Full train-plus-benchmark gate:

```bash
scripts/run_mwm_v1_gate.sh
```

## Top-Level Files

- `.gitignore`: Ignores generated training/eval artifacts (`data/`, `logs/`,
  `rollouts/`, `checkpoints_mwm/`, `*.pt`, caches, HDF5 scratch files). The
  current worktree still contains ignored generated artifacts on disk for review.
- `README.md`: User-facing project overview, quick-start commands, architecture
  bullets, benchmark roles, and pointer to this review guide.
- `REVIEW_GUIDE.md`: This file. It is the reviewer map and acceptance checklist.
- `requirements.txt`: Runtime dependency list: PyTorch, Stable-WM,
  Stable-Pretraining, simulation/image libraries, plotting, and config/logging
  dependencies.

## Entrypoint Scripts

- `collect_mwm_data.py`: Reads a collection config, builds a Stable-WM world,
  installs either a user policy or `RandomPolicy`, records episodes to a new
  Lance dataset, validates restore columns, and writes dataset sidecar metadata.
  Review refusal-to-append behavior in `_record_dataset_to_path`; it prevents
  accidental mixed datasets.
- `prepare_upstream_lewm.py`: Downloads or loads trusted upstream Le-WM objects,
  saves source objects under `checkpoints_mwm/upstream_sources/`, imports them
  through `LeWMObjectImporter`, and exports canonical single-fidelity MWM
  checkpoints with dependency metadata.
- `prepare_upstream_lewm_data.py`: Prepares the public upstream PushT Le-WM
  dataset for paper-parity debugging by downloading the compressed HDF5 artifact,
  decompressing it, delegating HDF5-to-Lance conversion to Stable-WM, and writing
  MWM sidecar metadata. This keeps HDF5 reading out of the MWM runtime.
- `train_mwm.py`: Loads Lance training data, performs the Stable-Pretraining
  random split, infers image/action dimensions, builds the Le-WM base adapter
  for both `K=[D]` and scheduled `K`, trains with the Le-WM AdamW/warmup-cosine
  recipe, and saves a canonical checkpoint. Review exact-backend selection,
  `_prepare_exact_lewm_context`, `_load_exact_lewm_train_valid_datasets`, and
  the `action_block`/frameskip handling carefully.
- `eval_mwm.py`: Loads a canonical checkpoint and Lance dataset, validates
  metadata compatibility, samples or loads immutable start/goal manifests,
  fits evaluation-only action/stat scalers when required by the checkpoint,
  builds `MWMWorldModelPolicy` with `MWMScheduledCEMSolver`, runs Stable-WM
  evaluation batches, combines policy diagnostics, and writes `eval.json`.
- `benchmark_mwm.py`: Expands `configs/benchmark_mwm.yaml` into 18 eval runs,
  enforces gate matrix uniqueness/completeness, shares manifests across roles,
  captures each eval log, writes per-run sidecars, aggregate CSV/JSONL/JSON, plots,
  and `review.html`.
- `verify_mwm_data.py`: Verifies that configured train/eval datasets exist, are
  Lance datasets, and have required metadata keys before expensive training/eval.
- `verify_mwm_benchmark.py`: Re-validates completed benchmark artifacts: required
  files, exact summary row regeneration, matrix cells, shared manifests, dependency
  refs, checkpoint schemas, diagnostics, plots, review HTML links, and optional
  paper-target parity checks.
- `render_benchmark_review.py`: Regenerates benchmark summary CSV/JSONL, per-env
  table, plots, and static HTML from an existing `rollouts/mwm_benchmark` output
  directory without rerunning evaluation.

## Package Files

- `mwm/__init__.py`: Public package marker; exports `MWMWorldModel`.
- `mwm/adapters/__init__.py`: Re-exports adapter base classes and component
  dataclass.
- `mwm/adapters/lewm.py`: Adapter/importer implementation for Le-WM. It defines
  common adapter contracts, image preprocessing, CNN/ViT backbones,
  `LeWMMatryoshkaWorldModel`, per-`K` transition packages, proportional head
  scaling, trusted-object importing, and builder functions used by checkpoint
  config targets. Review source-class validation, expected components,
  single-fidelity import restriction, `K=[D]` exactness, and `action_spec`
  construction.
- `mwm/benchmark/__init__.py`: Empty namespace marker for benchmark helpers.
- `mwm/benchmark/artifacts.py`: Artifact writer and renderer. It writes JSON,
  CSV, JSONL, per-run sidecars, per-environment tables, seven default plots, and
  static review HTML with gate cards, warnings, paired comparisons, drilldowns,
  and media links. Review that summary rows contain the fields verified later.
- `mwm/checkpoints.py`: Canonical checkpoint API. It writes/loads config,
  weights, metadata, artifact hashes, and rejects non-canonical extra files.
  It also instantiates models from import targets. This is a high-risk file
  because it defines the serialization boundary.
- `mwm/data/__init__.py`: Empty namespace marker for data helpers.
- `mwm/data/manifest.py`: Immutable evaluation manifest creation and validation.
  It canonicalizes JSON, stores a logical `manifest_sha256`, computes file hashes,
  and rejects tampered manifests on load.
- `mwm/data/stable_wm.py`: Stable-WM dataset glue. It transforms samples into
  `x`/`a` tensors, handles channel layout and action rows, loads Stable-WM
  datasets, reads/writes sidecar metadata, and samples deterministic start/goal
  pairs from dataset episode offsets.
- `mwm/dependency_refs.py`: Captures package versions/hashes, optional VCS commit
  IDs, and local git commit/dirty diff hash. Benchmark and checkpoint artifacts
  use this to make dependency state reviewable.
- `mwm/eval/__init__.py`: Empty namespace marker for eval helpers.
- `mwm/eval/policy.py`: Stable-WM policy integration. It standardizes images to
  224x224 CHW tensors, counts model parameters, wraps Stable-WM
  `WorldModelPolicy`, preserves upstream action inverse-transform semantics,
  times action calls, resets solver traces, and emits planning diagnostics.
- `mwm/fidelity.py`: Fidelity scheduler and decision objects. Supports `fixed`,
  `linear_cem`, and `table` policies; resolves tokens like `coarsest`, `finest`,
  and `base`; validates horizon length and forbids low-to-high rollout schedules.
- `mwm/metrics.py`: Small helpers for success rate and aggregate policy
  diagnostics. Currently not central to the benchmark path but useful for summary
  calculations.
- `mwm/models/__init__.py`: Re-exports `MWMWorldModel`.
- `mwm/models/world_model.py`: Legacy generic model utilities and the public
  `MWMWorldModel` contract. The trainable Le-WM path no longer uses its default
  dynamics/decoder stack; reviewers should treat those generic helpers as
  non-production compatibility code until removed.
- `mwm/planning/__init__.py`: Empty namespace marker for planners.
- `mwm/planning/scheduled_cem.py`: Stable-WM-compatible CEM planner that asks a
  `FidelityScheduler` which level to use at each CEM iteration, samples/clamps
  candidate action sequences, calls `get_cost_with_fidelity`, updates elite mean
  and variance, and records diagnostics.
- `mwm/swm/__init__.py`: Empty namespace marker for Stable-WM integration.
- `mwm/swm/envs.py`: Stable-WM environment helpers. Parses image shapes and env
  kwargs, imports user objects, adds restore wrappers when needed, constructs
  `stable_worldmodel.World`, validates continuous finite Box action spaces, and
  can infer action bounds.
- `mwm/swm/restore.py`: Restore adapter registry. Defines built-in restore specs
  for PushT, TwoRoom/Piecewise, DMControl, and OGBench; supports user restore
  specs; validates required dataset columns; returns Stable-WM eval callables.
- `mwm/swm/wrappers.py`: OGBench restore wrapper that records/restores
  concatenated MuJoCo `qpos`/`qvel` state and exposes `set_restore_state` to
  Stable-WM dataset evaluation.
- `mwm/training.py`: Training helper layer around `mwm_prediction_loss`. Provides
  a Stable-Pretraining-compatible module/fallback module and an optional
  Stable-WM SIGReg builder.

## Config Files

- `configs/collect_mwm_pusht.yaml`: Collects 100 PushT episodes at 224x224 with
  4 envs into `data/pusht_swm.lance`.
- `configs/collect_mwm_tworoom.yaml`: Collects 100 TwoRoom episodes at 224x224
  with 8 envs into `data/tworoom_swm.lance`.
- `configs/train_mwm_lewm_pusht.yaml`: Trains PushT single-fidelity retrained
  Le-WM checkpoint with `K: [192]` through the exact Stable-WM Le-WM backend:
  history/context prediction, projector heads, standardized non-pixel columns,
  SIGReg, AdamW, and a canonical MWM checkpoint export.
- `configs/train_mwm_lewm_tworoom.yaml`: Same exact single-fidelity backend for
  TwoRoom with run name `retrained_lewm_single_tworoom`.
- `configs/train_mwm_lewm_pusht_upstream.yaml`: PushT paper-parity retrain on
  `data/upstream/pusht_expert_train.lance`; this is the strict upstream-data
  debug path used to compare converted upstream Le-WM and a from-scratch
  single-level MWM/Le-WM checkpoint.
- `configs/train_mwm_lewm_tworoom_upstream.yaml`: TwoRoom paper-parity retrain
  on `data/upstream/tworoom.lance`.
- `configs/train_mwm_scheduled_pusht.yaml`: Trains PushT multi-fidelity MWM with
  `K: [48, 96, 144, 192]`, same action block/training backend, run name
  `mwm_scheduled_pusht`.
- `configs/train_mwm_scheduled_tworoom.yaml`: Same as above for TwoRoom with run
  name `mwm_scheduled_tworoom`.
- `configs/eval_mwm_lewm_pusht.yaml`: Base PushT eval config. Uses upstream
  checkpoint by default, 6 episodes, goal offset 25, budget 50, action block 5,
  linear CEM schedule, and writes PushT manifest/output paths unless overridden.
- `configs/eval_mwm_lewm_tworoom.yaml`: Base TwoRoom eval config. Uses upstream
  checkpoint by default, 6 episodes, goal offset 100, budget 150, action block 5,
  paper CEM iteration count 10, linear CEM schedule, and writes TwoRoom
  manifest/output paths unless overridden.
- `configs/eval_mwm_paper_pusht.yaml`: PushT paper-parity eval config using the
  official upstream Lance dataset, 50 episodes, Stable-WM start/goal sampling,
  standardized action planning, and fixed finest-level CEM.
- `configs/eval_mwm_paper_tworoom.yaml`: TwoRoom paper-parity eval config using
  the official upstream Lance dataset, 50 episodes, goal offset 25, budget 50,
  CEM `batch_size: 1`, CEM `n_iter: 30`, standardized action planning, and
  fixed finest-level CEM.
- `configs/benchmark_mwm.yaml`: The required benchmark matrix. Defines output
  directory, gate envs/seeds/roles, and 18 runs with checkpoint overrides for
  upstream converted, retrained single, and scheduled roles.
- `configs/benchmark_mwm_paper_parity.yaml`: Paper-parity sanity matrix for
  PushT and TwoRoom, seed 42, comparing converted upstream Le-WM against exact
  from-scratch single-level MWM/Le-WM checkpoints. Its verifier targets come
  from arXiv v1 Fig. 6: PushT 96.0% and Two-Room 87.0%, with a 5 percentage
  point tolerance and the same tolerance for retrained-single vs upstream.

## Scripts

- `scripts/run_mwm_benchmark_gate.sh`: Runs benchmark and benchmark verification
  from the repo root using `$MWM_PYTHON` or the hardcoded MWM conda Python.
- `scripts/run_mwm_paper_parity.sh`: Prepares upstream converted checkpoints and
  upstream Lance datasets, verifies paper-parity data configs, trains exact
  single-level PushT/TwoRoom baselines, then runs and verifies the paper-parity
  benchmark.
- `scripts/run_mwm_v1_gate.sh`: Runs data verification, upstream conversion,
  four training jobs, benchmark, and benchmark verification.
- `scripts/slurm_mwm_benchmark_gate.sbatch`: SLURM wrapper for benchmark-only gate;
  requests one 90GB GPU, prepares env vars/directories, then executes the
  benchmark gate script.
- `scripts/slurm_mwm_paper_parity.sbatch`: SLURM wrapper for the paper-parity
  sanity gate; requests one B200 GPU and a four-day walltime.
- `scripts/slurm_mwm_v1_gate.sbatch`: SLURM wrapper for the full gate; requests
  one 90GB GPU for up to seven days, prints CUDA sanity info, then executes the
  full gate script.

## Tests

- `tests/test_mwm_core.py`: Unit tests for trusted Le-WM import, canonical
  checkpoint roundtrip and extra-file rejection, component validation, decoder
  presence, action block semantics, trainer-root cleanup, and fidelity scheduler
  monotonicity.
- `tests/test_mwm_artifacts.py`: Unit tests for manifest logical vs file hashes,
  duplicate benchmark-cell rejection, benchmark failure log capture, checkpoint
  metadata verification, Lance metadata verification, plot generation, and review
  HTML drilldowns.
- `tests/test_mwm_repo_hygiene.py`: Repo hygiene tests that scan source/config/doc
  files for forbidden legacy symbols and assert configs are Lance-only and use the
  scheduler branch rather than old fidelity/baseline keys.

## Generated Data Files

These files are ignored by git but currently present on disk for review.

- `data/pusht_swm.lance.metadata.json`: Sidecar for the PushT Lance dataset:
  `swm_lance`, 100 episodes, 224x224 images, action bounds `[-1, 1]`, action dim
  2, restore spec `pusht_state_goal_state`, keys `pixels` and `action`.
- `data/tworoom_swm.lance.metadata.json`: Sidecar for the TwoRoom Lance dataset:
  `swm_lance`, 100 episodes, 224x224 images, action bounds `[-1, 1]`, action dim
  2, restore spec `point_state_goal_state`, keys `pixels` and `action`.
- `data/pusht_swm.lance/_transactions/*.txn`,
  `data/pusht_swm.lance/_versions/*.manifest`, and
  `data/pusht_swm.lance/data/*.lance`: Lance internals for the PushT dataset.
  Review through Lance tooling rather than hand-editing.
- `data/tworoom_swm.lance/_transactions/*.txn`,
  `data/tworoom_swm.lance/_versions/*.manifest`, and
  `data/tworoom_swm.lance/data/*.lance`: Lance internals for the TwoRoom dataset.
- `data/upstream/pusht_expert_train.lance.metadata.json`: Sidecar for the
  official PushT Le-WM dataset converted to Lance by Stable-WM tooling from the
  public compressed HDF5 artifact.
- `data/upstream/tworoom.lance.metadata.json`: Sidecar for the official TwoRoom
  Le-WM dataset extracted from the public compressed Lance archive.

## Generated Checkpoint Files

Each canonical checkpoint directory should contain exactly:

- `config.json`: Import target plus constructor kwargs used to rebuild the model.
- `weights.pt`: PyTorch state dict for the canonical MWM model.
- `world_metadata.json`: Review metadata: format, env, restore spec, action spec,
  levels, dataset refs, dependency refs, artifact hashes, and training/upstream
  provenance.

Generated checkpoint directories. These are ignored runtime artifacts; if their
metadata predates the current configs, regenerate them with the gate scripts
before making performance claims.

- `checkpoints_mwm/upstream_lewm_pusht/`: Converted upstream PushT Le-WM object,
  single level `K=[192]`, target `mwm.adapters.lewm.build_mwm_lewm_from_object`,
  role `upstream_lewm_converted`.
- `checkpoints_mwm/upstream_lewm_tworoom/`: Converted upstream TwoRoom Le-WM
  object, single level `K=[192]`, same importer target, role
  `upstream_lewm_converted`.
- `checkpoints_mwm/retrained_lewm_single_pusht/`: Locally trained PushT
  single-fidelity checkpoint, expected after refresh to use
  `mwm.adapters.lewm.build_mwm_lewm` with `architecture_version:
  lewm_base_adapter_v1`, `K=[192]`, and action block 5.
- `checkpoints_mwm/retrained_lewm_single_tworoom/`: Locally trained TwoRoom
  single-fidelity checkpoint, expected after refresh to use
  `mwm.adapters.lewm.build_mwm_lewm` with `architecture_version:
  lewm_base_adapter_v1`, `K=[192]`, and action block 5.
- `checkpoints_mwm/retrained_lewm_single_pusht_upstream/`: Paper-parity PushT
  exact Le-WM retrain on the official upstream dataset, exported through the
  Le-WM base adapter target `mwm.adapters.lewm.build_mwm_lewm`.
- `checkpoints_mwm/retrained_lewm_single_tworoom_upstream/`: Paper-parity
  TwoRoom exact Le-WM retrain on the official upstream dataset, exported through
  the Le-WM base adapter target `mwm.adapters.lewm.build_mwm_lewm`.
- `checkpoints_mwm/mwm_scheduled_pusht/`: Locally trained PushT multi-fidelity
  MWM, `K=[48,96,144,192]`, action block 5.
- `checkpoints_mwm/mwm_scheduled_tworoom/`: Locally trained TwoRoom multi-fidelity
  MWM, `K=[48,96,144,192]`, action block 5.
- `checkpoints_mwm/upstream_sources/upstream_lewm_pusht_object.pt`: Trusted raw
  upstream object cached before conversion.
- `checkpoints_mwm/upstream_sources/upstream_lewm_tworoom_object.pt`: Trusted raw
  upstream object cached before conversion.

## Generated Training Logs

- `logs/mwm_training/*/checkpoints/epoch=0-step=*.ckpt`: Stable-Pretraining
  Lightning checkpoint selected by training. These are not the canonical runtime
  checkpoints; `train_mwm.py` exports canonical files into `checkpoints_mwm/`.
- `logs/mwm_training/*/checkpoints/last.ckpt`: Last Lightning checkpoint for the
  corresponding training run.
- `logs/mwm_training/*/environment.json`: Environment snapshot from
  Stable-Pretraining callbacks.
- `logs/mwm_training/*/requirements_frozen.txt`: Frozen package list captured
  during each training run.
- `logs/mwm_cell001_debug_6154603.out` and `.err`: Debug run logs showing an
  earlier action dimension mismatch (`Expected action_dim=2, got 10`) plus
  environment warnings. Useful provenance, not the final passing gate.
- `logs/mwm_v1_gate_6154337.out` and `.err`: Failed SLURM setup attempt due to
  permission errors creating output directories.
- `logs/mwm_v1_gate_6154361.out` and `.err`: Partial full-gate training/benchmark
  attempt; useful for training callback output and earlier benchmark progress.
- `logs/mwm_v1_gate_6154642.out` and `.err`: Full-gate attempt that reached
  benchmark verification output with 18 runs and required plots.
- `logs/mwm_v1_gate_6156159.out` and `.err`: Later full-gate attempt showing all
  18 benchmark runs starting/completing progress in stdout and training warnings
  in stderr.

## Generated Benchmark Output

Top-level benchmark files:

- `rollouts/mwm_benchmark/summary.json`: Aggregate benchmark report containing
  title, output dir, all run summary rows, manifest groups, per-env table path,
  and plot refs.
- `rollouts/mwm_benchmark/summary.csv`: CSV view of the aggregate run rows.
- `rollouts/mwm_benchmark/metrics.jsonl`: JSONL copy of aggregate run rows; the
  verifier expects it to exactly match `summary.json["runs"]`.
- `rollouts/mwm_benchmark/per_env_summary.csv`: Mean success by environment and
  role.
- `rollouts/mwm_benchmark/review.html`: Static visual review report with gate
  cards, warnings, plots, paired comparisons, run drilldowns, review notes, and
  media links.
- `rollouts/mwm_benchmark/plots/efficiency_ratios.png`: Paired MWM/upstream wall
  and compute ratios.
- `rollouts/mwm_benchmark/plots/paired_success_delta.png`: Seed-paired success
  delta between scheduled MWM and upstream Le-WM.
- `rollouts/mwm_benchmark/plots/schedule_level_usage.png`: Aggregate CEM cost
  calls by base fidelity level.
- `rollouts/mwm_benchmark/plots/schedule_usage_by_role.png`: Schedule usage split
  by environment and role.
- `rollouts/mwm_benchmark/plots/success_by_env_role.png`: Mean success by env and
  role with seed points.
- `rollouts/mwm_benchmark/plots/success_vs_compute.png`: Success vs latent work.
- `rollouts/mwm_benchmark/plots/success_vs_wall_time.png`: Success vs wall time.

Manifest files:

- `rollouts/mwm_benchmark/manifests/pusht_seed0.json`
- `rollouts/mwm_benchmark/manifests/pusht_seed1.json`
- `rollouts/mwm_benchmark/manifests/pusht_seed2.json`
- `rollouts/mwm_benchmark/manifests/tworoom_seed0.json`
- `rollouts/mwm_benchmark/manifests/tworoom_seed1.json`
- `rollouts/mwm_benchmark/manifests/tworoom_seed2.json`

Each manifest records dataset metadata, dependency refs, restore spec, eval
budget, goal offset, seed, and deterministic start/goal pairs. The three roles
for a given env/seed should reference the same manifest.

Per-run benchmark directories:

- `rollouts/mwm_benchmark/000_pusht_seed0_upstream_lewm_converted/`
- `rollouts/mwm_benchmark/001_pusht_seed0_retrained_lewm_single/`
- `rollouts/mwm_benchmark/002_pusht_seed0_mwm_scheduled/`
- `rollouts/mwm_benchmark/003_pusht_seed1_upstream_lewm_converted/`
- `rollouts/mwm_benchmark/004_pusht_seed1_retrained_lewm_single/`
- `rollouts/mwm_benchmark/005_pusht_seed1_mwm_scheduled/`
- `rollouts/mwm_benchmark/006_pusht_seed2_upstream_lewm_converted/`
- `rollouts/mwm_benchmark/007_pusht_seed2_retrained_lewm_single/`
- `rollouts/mwm_benchmark/008_pusht_seed2_mwm_scheduled/`
- `rollouts/mwm_benchmark/009_tworoom_seed0_upstream_lewm_converted/`
- `rollouts/mwm_benchmark/010_tworoom_seed0_retrained_lewm_single/`
- `rollouts/mwm_benchmark/011_tworoom_seed0_mwm_scheduled/`
- `rollouts/mwm_benchmark/012_tworoom_seed1_upstream_lewm_converted/`
- `rollouts/mwm_benchmark/013_tworoom_seed1_retrained_lewm_single/`
- `rollouts/mwm_benchmark/014_tworoom_seed1_mwm_scheduled/`
- `rollouts/mwm_benchmark/015_tworoom_seed2_upstream_lewm_converted/`
- `rollouts/mwm_benchmark/016_tworoom_seed2_retrained_lewm_single/`
- `rollouts/mwm_benchmark/017_tworoom_seed2_mwm_scheduled/`

Each per-run directory contains:

- `resolved_config.yaml`: Fully resolved eval config after benchmark overrides.
- `eval.json`: Full evaluation payload including SWM results, batches, manifest
  info, dependencies, model accounting, planning diagnostics, and videos.
- `metrics.jsonl`: One-line summary row for that run.
- `episode_traces.jsonl`: Per-evaluation-episode start/goal and success trace.
- `summary.json`: Sidecar containing the same run summary row.
- `dependencies.json`: Dependency refs copied from `eval.json`.
- `planning_diagnostics.json`: Planning diagnostics copied from `eval.json`.
- `run.log`: Captured stdout/stderr from the eval run.

## Paper-Parity Benchmark Output

- `rollouts/mwm_paper_parity/summary.json`, `.csv`, `metrics.jsonl`,
  `per_env_summary.csv`, `review.html`, and `plots/*.png`: Same artifact schema
  as the full benchmark, but for the paper-parity sanity matrix.
- Current paper-parity investigation evidence is recorded in
  `docs/superpowers/paper-parity-investigation-2026-05-28.md`. The strict
  PushT target gate is intentionally still incomplete: the converted evaluator
  and Stable-WM reference path reached 98.0%, while the raw upstream Le-WM
  evaluator reached 92.0% on the same seed-42 protocol.
- After rerunning `scripts/run_mwm_paper_parity.sh`, this directory should
  contain four cells: PushT/TwoRoom x upstream converted/retrained single.

## Legacy Deletions To Notice In Review

The current worktree deletes the previous SWM-era entrypoints, package tree,
world-model modules, planning modules, dataset glue, configs, tests, and docs.
That is consistent with the repo moving to the MWM V1 review surface, but the
reviewer should confirm no required functionality was accidentally lost.

## Suggested Review Order

1. Start with `README.md`, this guide, and `configs/benchmark_mwm.yaml`.
2. Review the data/checkpoint boundaries: `mwm/data/stable_wm.py`,
   `mwm/data/manifest.py`, `mwm/checkpoints.py`, and `verify_mwm_benchmark.py`.
3. Review the model/planner path: `mwm/models/world_model.py`,
   `mwm/fidelity.py`, `mwm/planning/scheduled_cem.py`, and `mwm/eval/policy.py`.
4. Review the Le-WM adapter/import path: `mwm/adapters/lewm.py` and
   `prepare_upstream_lewm.py`.
5. Review orchestration: `collect_mwm_data.py`, `train_mwm.py`, `eval_mwm.py`,
   `benchmark_mwm.py`, and the shell/SLURM scripts.
6. Review tests and generated artifacts last, checking whether the tests cover
   the invariants above and whether the generated output actually matches the
   claimed matrix.

## Acceptance Checklist

- `python -m py_compile` passes for entrypoints and `mwm/**/*.py`.
- Unit tests pass for core model/checkpoints/scheduler/artifacts/hygiene.
- `python verify_mwm_data.py` passes before training/eval.
- `python verify_mwm_benchmark.py configs/benchmark_mwm.yaml` passes after the
  benchmark.
- `python verify_mwm_benchmark.py configs/benchmark_mwm_paper_parity.yaml`
  passes after the paper-parity sanity gate, including upstream-vs-paper and
  retrained-single-vs-upstream success checks.
- The benchmark has exactly 18 cells and no duplicate `(env_id, seed, role)` rows.
- Every run has nonzero CEM work diagnostics.
- Every checkpoint has a valid `action_spec`, matching config kwargs and metadata.
- Benchmark roles satisfy checkpoint contracts: upstream is converted through
  the trusted Le-WM object importer, retrained single is exact Le-WM `K=[192]`,
  and scheduled MWM is `K=[48,96,144,192]`.
- Each env/seed shares one manifest across all three roles.
- `review.html` links all run drilldowns and embeds all seven required plots.
- Generated logs are interpreted as provenance; final claims should rely on the
  verifier and current aggregate artifacts, not on earlier failed debug logs.
