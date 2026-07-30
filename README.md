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

## Benchmark Sweeps

A benchmark config can define a Cartesian parameter sweep with dotted keys.
Every combination is applied to every entry under `runs`, including an
upstream checkpoint row. For example:

```yaml
run_defaults:
  planner:
    flop_accounting: dynamics_audit
sweep:
  planner.pop_size: [100, 300]
  planner.elite_frac: [0.05, 0.1]
  planner.n_iter: [5, 10]
```

Each expanded cell has its own run directory and saved sweep parameters, so an
interrupted local run can reuse completed cells:

```bash
python -m mwm.benchmark.matrix CONFIG.yaml --resume
```

Large matrices can be split deterministically across workers. Run one command
per zero-based shard, then finalize after all shards have completed:

```bash
python -m mwm.benchmark.matrix CONFIG.yaml --resume --shard-index 0 --num-shards 4
python -m mwm.benchmark.matrix CONFIG.yaml --finalize-only
```

The aggregate `review.html` embeds an interactive success-versus-cost Pareto
plot. Its default cost axis is audited dynamics FLOPs; legend names use the
human-readable `schedule` field, and hovering a point shows its sweep parameters.

The generated `review.html` is a static aggregate report. To inspect aligned
successes and failures episode by episode, play existing videos, or render
missing environment and latent-reconstruction media on demand, start the local
review server:

```bash
python -m mwm.benchmark.render_review rollouts/mwm_benchmark --serve
```

Open the printed localhost URL, use **Rollout Review** to compare the same
episode number across benchmark roles, and start with failed episodes before
sampling successes. Environment videos replay stored actions in the simulator;
latent videos compare dataset observations with decoded representations and are
not planner-prediction videos.

**Latent predictive rollout** is a separate media mode from reconstruction. It
replays the stored policy actions to capture the actual online observations,
anchors the model from that exact history at every MPC replan, and calls
`model.rollout_with_schedule(...)` autoregressively for the complete action
blocks executed before the next replan. Its left panel is actual replay; its
right panel shows decoded block-endpoint predictions. Blue frames mark the
actual-history reset boundary. Incomplete terminal action blocks are reported
and omitted rather than extrapolated.

The `eval.budget` setting is the benchmark's maximum number of primitive
environment actions (50 in the paper PushT config). It is distinct from
`env.max_episode_steps`, which is the simulator safety cap. Successful
vectorized evaluations may terminate earlier; new review artifacts keep only
the actions that were actually executed instead of the later masked slots.

On-demand environment replay does not load model weights or fit a dataset-wide
action scaler. It selects only `goal_offset + 1` rows (26 for PushT) needed for
the initial state, goal state, and comparison frames. Latent reconstruction
does require the checkpoint, but a running review server caches it across
requests. Render requests are serialized and duplicate clicks share the same
job, while already-rendered videos are served directly.

New evaluations save both environment-space `action_trace` and the exact
pre-inverse-transform `model_action_trace`. For older artifacts, predictive
review scans only the Lance action column to recover the standardization mean
and scale, caching those statistics by Lance dataset version.

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

Research configs may opt Le-WM into `slimmable_transformer_v1`. That variant
uses one nested action-conditioned causal transition, trains configured anchor
prefixes plus a sampled non-anchor prefix, and permits literal `K` selection at
inference while executing prefix-sliced attention and MLP widths.

Evaluation configs may use the root-level `K` shortcut. An explicit list keeps
the existing discrete behavior, while one hyphenated item denotes an inclusive
integer range:

```yaml
K: [96-192]
```

For a shared slimmable Le-WM checkpoint this makes every integer width from 96
through 192 selectable by the existing planner schedule. `K: [96, 144, 192]`
remains valid and discrete. Range syntax is inference-only and fails during
planner configuration when the loaded checkpoint is not
`lewm_shared_slimmable_transformer_v1`; legacy checkpoints continue to use
their exact explicit anchor list.

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
