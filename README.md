# MWM

Matryoshka World Models (MWM) is a Stable-WM-compatible benchmark and evaluation repo for multi-fidelity world models.

The review story is intentionally narrow: every evaluated checkpoint is loaded as an `MWMWorldModel`, datasets are Lance, and all benchmark roles run through the same scheduled CEM evaluator. Trainable Le-WM MWM keeps a shared Le-WM encoder/projector trunk and adds fresh per-`K` transition heads (`action_encoder`, `predictor`, `pred_proj`) trained with Le-WM loss semantics.

## Quick Start

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

The full gate is:

```bash
scripts/run_mwm_v1_gate.sh
```

To rerun only the benchmark matrix from existing canonical checkpoints:

```bash
scripts/run_mwm_benchmark_gate.sh
```

To rerun the paper-parity evaluator sanity check:

```bash
scripts/run_mwm_paper_parity.sh
```

On Betty/PARCC, submit the paper-parity gate first and the full MWM gate after it:

```bash
scripts/submit_mwm_gates.sh
```

## Architecture

- `mwm.models.world_model.MWMWorldModel` is the runtime model contract, and
  `MatryoshkaWorldModel` owns the shared multi-level shell used by base adapters.
- `mwm.adapters.lewm` is the stable public facade for Le-WM builders.
- `mwm.adapters.lewm_stable` derives Le-WM components from Stable-WM configs or
  trusted upstream objects, then returns the normal `MatryoshkaWorldModel`.
  `K=[192]` is constructor/loss/optimizer exact to the base Le-WM path;
  multi-`K` training encodes once and aggregates requested prefix losses only.
- `mwm.checkpoints` reads and writes strict canonical checkpoints containing `config.json`, `weights.pt`, and `world_metadata.json`.
- `mwm.planning.scheduled_cem` is the active evaluator/planner path.
- `mwm.data.stable_wm` is Lance-only data glue for Stable-WM datasets and immutable eval manifests.

## Base-Adaptive MWM

MWM reads Stable-WM `config.json` files for architecture and never copies source
weights for fair training. The Stable-WM config is an architecture oracle; the
training recipe comes from the MWM YAML and is applied across matryoshka levels.
Adapters declare top-level component groups, then configs choose which groups are
shared or duplicated. Le-WM is implemented first: `encoder + projector` are the
shared latent producer, while `action_encoder + predictor + pred_proj` are
fresh per-`K` transition tails.

PreJEPA/DINO-WM and PLDM currently expose component-group declarations only.
They fail explicitly until an explicit Stable-WM training recipe artifact is
available, so unknown bases cannot silently fall through to generic MWM dynamics.

## Benchmark Roles

The benchmark matrix is PushT and TwoRoom, seeds `0,1,2`, with:

- `upstream_lewm_converted`: upstream Le-WM imported into a canonical single-fidelity MWM checkpoint.
- `retrained_lewm_single`: exact Le-WM single-level training with `K=[192]`, exported as a canonical MWM checkpoint.
- `mwm_scheduled`: this repo's multi-fidelity training with `K=[48,96,144,192]`.

Generated datasets, checkpoints, rollouts, logs, and caches are intentionally ignored by git. See `REVIEW_GUIDE.md` for the code-review map and acceptance checks.
