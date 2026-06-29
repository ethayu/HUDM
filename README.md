# MWM

Matryoshka World Models (MWM) is a Stable-WM-compatible benchmark and evaluation repo for multi-fidelity world models.

The review story is intentionally narrow: every evaluated checkpoint is built through the Stable-WM adapter builder, datasets are Lance, and all benchmark roles run through the same scheduled CEM evaluator. Le-WM and PreJEPA/DINO-WM now have explicit MWM runtime classes instead of a single generic shell with family-specific strategy hooks.

## Quick Start

```bash
python -m mwm.data.collection configs/collect/mwm_pusht.yaml
python -m mwm.data.collection configs/collect/mwm_tworoom.yaml
python -m mwm.data.verify
python -m mwm.upstream.lewm_checkpoints
python -m mwm.upstream.lewm_data
python -m mwm.training.stable_wm configs/train/mwm_lewm_pusht.yaml
python -m mwm.training.stable_wm configs/train/mwm_lewm_tworoom.yaml
python -m mwm.training.stable_wm configs/train/mwm_lewm_scheduled_pusht.yaml
python -m mwm.training.stable_wm configs/train/mwm_lewm_scheduled_tworoom.yaml
python -m mwm.eval.runner configs/eval/mwm_lewm_pusht.yaml
python -m mwm.benchmark.matrix configs/benchmark/scheduled_pusht.yaml
python -m mwm.benchmark.verify configs/benchmark/scheduled_pusht.yaml
python -m mwm.benchmark.render_review rollouts/mwm_benchmark
```

The upstream paper-parity sanity check is:

```bash
scripts/slurm/run_mwm_paper_parity.sh
```

To rerun only the scheduled-MWM comparisons from existing canonical checkpoints:

```bash
scripts/slurm/run_mwm_scheduled_comparison.sh
```

To rerun the paper-parity evaluator sanity check:

```bash
scripts/slurm/run_mwm_paper_parity.sh
```

On Betty/PARCC, submit true scheduled-MWM training as split one-GPU jobs with a dependent comparison benchmark:

```bash
scripts/slurm/submit_mwm_scheduled_split.sh
```

## Local Desktop Workflow

Local machines are supported for syntax checks, tests, static benchmark
validation, and tiny smoke runs. Full paper-scale training and benchmark runs
remain GPU-oriented and should use the Slurm scripts on PARCC/Betty.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
scripts/local/local_verify.sh
```

For a tiny local benchmark after preparing or copying
`data/upstream/pusht_expert_train.lance` and
`checkpoints_mwm/upstream_lewm_pusht`:

```bash
scripts/local/local_benchmark_smoke.sh
```

Optional CPU training smoke is deliberately opt-in because it can be slow:

```bash
RUN_CPU_TRAIN_SMOKE=1 scripts/local/local_train_smoke.sh
```

Use `MWM_PYTHON=/path/to/python` if your Python is not named `python`.

## Architecture

- `mwm.models.common.MatryoshkaRuntimeModel` is the shared runtime marker; concrete family behavior lives in `mwm.models.lewm.LeWMMatryoshkaWorldModel` and `mwm.models.prejepa.PreJEPAMatryoshkaWorldModel`.
- `mwm.adapters.lewm` derives Le-WM components from Stable-WM configs,
  registers the Le-WM adapter, then returns `LeWMMatryoshkaWorldModel`.
  `K=[192]` is constructor/loss/optimizer exact to the base Le-WM path;
  multi-`K` training encodes once and aggregates requested prefix losses only.
- `mwm.adapters.prejepa` derives transformer patch-backbone and extra-encoder components, then returns `PreJEPAMatryoshkaWorldModel`.
- `mwm.checkpoint_io` reads and writes canonical checkpoints containing `config.json`, `weights.pt`, and `world_metadata.json`;
  `mwm.checkpoint_contract` owns semantic config/metadata validation.
- `mwm.planning.scheduled_cem` is the active evaluator/planner path.
- `mwm.data.metadata`, `mwm.data.sampling`, `mwm.data.transforms`, and `mwm.data.manifest` own Lance dataset metadata,
  start/goal sampling, training transforms, and immutable eval manifests.
- `docs/mwm_adapter_contract.md` is the checklist for implementing or extending
  Stable-WM base adapters such as Le-WM, PreJEPA/DINO-WM, or PLDM.

## Base-Adaptive MWM

MWM reads Stable-WM `config.json` files for architecture and never copies source
weights for fair training. The Stable-WM config is the architecture source; the
training recipe comes from the MWM YAML and is applied across matryoshka levels.
Adapters declare top-level component groups, then configs choose which groups are
shared or duplicated. Le-WM uses `encoder + projector` as the shared latent
producer and fresh per-`K` transition tails; PreJEPA/DINO-WM uses a shared image
patch backbone with per-`K` patch predictors and fixed extra encoders.

Additional bases should be added as real adapters and concrete runtime classes
after inspecting their Stable-WM config/model. There are no placeholder runtime
adapters.

## Benchmark Roles

The active scheduled-MWM comparison is PushT and TwoRoom on the shared paper-parity seed `42`, with:

- `upstream_lewm_converted`: upstream Le-WM imported into a canonical identity-parity `K=[192]` MWM checkpoint.
- `mwm_scheduled`: this repo's multi-fidelity training with `K=[48,96,144]`.

The separate paper-parity check still uses `retrained_lewm_identity` for the `K=[D]` sanity check.

## Training Resources

Current validation uses single-GPU Slurm jobs. The checked-in Slurm scripts
request one B200 GPU per training or benchmark job, and the training defaults
keep Lightning at `train.devices: 1` so paper-parity comparisons stay on the
same path being validated here. Multi-GPU/DDP is opt-in through train config
keys such as `devices`, `strategy`, `num_nodes`, `sync_batchnorm`, and
`use_distributed_sampler`; no current parity job uses those knobs.

Generated datasets, checkpoints, rollouts, logs, and caches are intentionally ignored by git. See `REVIEW_GUIDE.md` for the code-review map and acceptance checks.
