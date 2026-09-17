# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Matryoshka World Models (MWM): a Stable-WM-compatible benchmark/evaluation library for multi-fidelity world models. The contract is intentionally narrow — every evaluated checkpoint is built through the Stable-WM adapter builder, datasets are Lance-only, and all benchmark roles run through the same scheduled CEM evaluator. Read `README.md` for the quick-start commands and `REVIEW_GUIDE.md` for the current contract vs. removed legacy surface before making structural changes.

## Commands

Local desktop (syntax checks, tests, static benchmark validation, smoke runs only — not paper-scale):

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
scripts/local/local_verify.sh          # py_compile all files + pytest -q + static benchmark verify
scripts/local/local_benchmark_smoke.sh # tiny local benchmark (needs a prepared Lance dataset + checkpoint)
RUN_CPU_TRAIN_SMOKE=1 scripts/local/local_train_smoke.sh  # opt-in, slow
```

Use `MWM_PYTHON=/path/to/python` if your interpreter isn't named `python`. On the cluster, `REVIEW_GUIDE.md` pins an explicit conda `mwm` env path for the same commands (`py_compile`, `pytest -q`, `mwm.data.verify --paper-parity`, `mwm.benchmark.verify ... --static-only`).

Single test: `python -m pytest tests/test_mwm_core.py::test_name -q`

Full pipeline (data → train → eval → benchmark):

```bash
python -m mwm.data.collection configs/collect/mwm_pusht.yaml
python -m mwm.data.verify
python -m mwm.upstream.lewm_checkpoints
python -m mwm.training.stable_wm configs/train/mwm_lewm_scheduled_pusht.yaml
python -m mwm.eval.runner configs/eval/mwm_lewm_pusht.yaml
python -m mwm.benchmark.matrix configs/benchmark/scheduled_pusht.yaml
python -m mwm.benchmark.verify configs/benchmark/scheduled_pusht.yaml
python -m mwm.benchmark.render_review rollouts/mwm_benchmark
```

Real GPU training/benchmarking runs go through the Slurm scripts in `scripts/slurm/` on PARCC/Betty (one B200 GPU per job; `scripts/slurm/submit_mwm_scheduled_split.sh` submits paired train jobs plus a dependent comparison benchmark). Don't try to reproduce paper-scale numbers from local CPU smoke runs.

Generated datasets, checkpoints, rollouts, logs, and caches are gitignored — don't try to check them in.

## Architecture

Construction path: `build_mwm_from_stable_config` (`mwm/adapters/builder.py`) → family adapter (`LeWMStableWMAdapter` / `PreJEPAStableWMAdapter` in `mwm/adapters/`) → concrete runtime (`LeWMMatryoshkaWorldModel` / `PreJEPAMatryoshkaWorldModel` in `mwm/models/`) → `MWMWorldModelPolicy` → `MWMScheduledCEMSolver` (`mwm/planning/scheduled_cem.py`).

**Adapters vs. runtime models — the key boundary.** Adapters (`mwm/adapters/*.py`) are construction code only: they parse a Stable-WM `config.json`, declare component groups (which modules form the shared "latent producer" vs. per-level "tail"), derive the authoritative latent dim `D` from the base config (never from `max(K)`), and instantiate per-`K` modules. They must not reimplement matryoshka aggregation, invent losses, or delegate to an upstream source object at runtime. Family runtime classes (`mwm/models/lewm.py`, `mwm/models/prejepa.py`) own everything else: shared-latent reuse, per-level loss aggregation, rollout, and planner cost. See `docs/mwm_adapter_contract.md` before touching or adding an adapter — it's the checklist for what belongs where, including how to add a new base family.

**Matryoshka training**: `K=[D]` is an identity-parity check (must match the base training path exactly), not a special code path. Multi-`K` training encodes once and aggregates requested prefix losses (`mwm.models.losses`, `mwm.models.objectives`). Adapters declare which top-level component groups are shared vs. duplicated per level (e.g. Le-WM shares `encoder + projector` and duplicates the transition tail; PreJEPA/DINO-WM shares the image patch backbone with per-level patch predictors and fixed extra encoders).

**Checkpoints**: canonical checkpoint = a directory with exactly `config.json`, `weights.pt`, `world_metadata.json`. `mwm/checkpoint_io.py` handles read/write/instantiate; `mwm/checkpoint_contract.py` owns semantic validation (levels, action spec, component policy, adapter family). No other checkpoint shape is supported — no upstream-object targets, no source-object delegation at runtime.

**Data**: Lance-only. `mwm/data/metadata.py`, `sampling.py`, `transforms.py`, `manifest.py` own dataset metadata sidecars, deterministic start/goal sampling, training transforms, and immutable eval manifests, respectively. HDF5 paths are legacy and should stay removed; `mwm/upstream/converters/` is where HDF5→Lance one-time conversion lives for paper-parity sources (Reacher, OGBench Cube).

**Benchmark roles**: the active scheduled-MWM comparison is PushT and TwoRoom at shared seed `42`: `upstream_lewm_converted` (upstream Le-WM imported into an identity-parity `K=[192]` checkpoint, no retraining) vs. `mwm_scheduled` (this repo's multi-fidelity training with `K=[48,96,144]`). `retrained_lewm_identity` is reserved for the separate paper-parity `K=[D]` sanity check, not the scheduled comparison.

**Devices**: paper-parity validation runs single-GPU (`train.devices: 1`) so comparisons stay on the same path being validated. Multi-GPU/DDP is opt-in via train config (`devices`, `strategy`, `num_nodes`, etc.) but no current parity job uses it — don't turn it on for parity runs without a reason.

## Conventions specific to this repo

- When adding a new Stable-WM base family, do not write a placeholder/guessed adapter. Inspect the actual Stable-WM config/model first, write down the component map (latent producer, authoritative `D`, per-level tail modules, exact-K vs. scaled-internal fields, objective contract, inference contract), then implement per `docs/mwm_adapter_contract.md`.
- Adapters should not export family-named builder facades — `build_mwm_from_stable_config` is the only public construction API; checkpoint `config.json` targets stay generic and carry `family` in kwargs so new adapters don't churn existing checkpoints/configs.
- Inference must stay base-aligned: preserve the base's action preprocessing, frame skip/action block, image preprocessing, and rollout horizon semantics — don't invent a new inference contract per family.
- `LIBRARY_FILE_REVIEW.md` is a maintained file-by-file map of the whole `mwm/` surface — check it for orientation before a broad refactor instead of re-deriving the package layout from scratch.
