# Identity-Upstream Delta Research Report

Branch: `codex/identity-upstream-delta`
Base commit: `98fb81042ad7fcfce4be963f3d5a5c97e1174d04`
Worktree: `${WORKTREE_ROOT}`
Artifact root audited: `${MWM_ARTIFACT_ROOT}`

## Executive Answer

The observed PushT result, upstream `98.0` versus retrained identity `92.0` at seed `42`, is primarily an evaluator/manifest sample effect, not a stable identity-checkpoint regression. The required five-seed paired sweep reverses the starting impression:

| env | seeds | upstream mean | identity mean | mean identity-upstream | min/max delta |
| --- | --- | ---: | ---: | ---: | --- |
| PushT | `0,1,2,42,100` | 86.0 | 92.0 | +6.0 | -6.0 / +12.0 |
| TwoRoom | `0,1,2,42,100` | 84.4 | 85.6 | +1.2 | -4.0 / +8.0 |

PushT seed `42` reproduces the original `98.0`/`92.0` rates, but it is the only swept PushT seed where identity is below upstream. Identity seed `42` is exactly at the PushT identity five-seed mean, while upstream seed `42` is an unusually favorable upstream manifest. PushT upstream ranges from `74.0` to `98.0` across the five 50-episode manifests.

Classification:

| candidate cause | verdict | evidence |
| --- | --- | --- |
| Evaluator/manifest variance | primary cause of the observed seed-42 delta | Five-seed paired sweep shows PushT deltas `+8,+6,+10,-6,+12`; the negative delta is not stable. |
| Training recipe/convergence | not supported as the main cause | Both identity trainings reached `max_epochs=10`; final validation pred losses are present; identity is competitive or better in aggregate. |
| Training data/dataset version | contributes to paper-target mismatch, not to the paired identity delta | Identity and upstream are evaluated on the same Lance data/manifests; official reference shows PushT HDF5 `92.0` versus Lance `98.0` for the same converted upstream policy. |
| Evaluator/config mismatch | not supported for paired runs | Within each benchmark cell, roles share the same manifest, eval seed, planner settings, action preprocessing, restore spec, and data path. |
| Checkpoint selection | possible follow-up, not current best explanation | Canonical identity exports are final epoch exports; Lightning only kept `last.ckpt`, no best checkpoint. Aggregate eval does not show a negative identity gap. |
| Code mismatch | unlikely | Targeted repo tests pass, including the K=[D] Le-WM direct-backend init/forward/grad/AdamW parity test. Static metadata matches architecture-critical fields. |

## Seed Sweep

Command:

```bash
sbatch scripts/slurm_research_identity_seed_sweep.sbatch
```

Current replay command: `sbatch scripts/research/slurm_research_identity_seed_sweep.sbatch`.

Completed job: `6201181`, state `COMPLETED`, exit `0:0`, elapsed `00:23:10`, allocation `gres/gpu:90gb=1`, partition `b200-mig90`.

The first submission, job `6201097`, failed in `00:00:19` before any eval episodes because `OmegaConf.from_dotlist()` cannot merge `runs.0...` overrides into a YAML list. I fixed the research wrapper to generate per-seed configs with OmegaConf and resubmitted as job `6201181`.

Paired results:

| env | seed | upstream | identity | identity-upstream | shared failures | upstream-only failures | identity-only failures |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PushT | 0 | 84.0 | 92.0 | +8.0 | `4,30,45` | `6,12,28,31,38` | `49` |
| PushT | 1 | 88.0 | 94.0 | +6.0 | `36,39,48` | `6,17,22` | none |
| PushT | 2 | 86.0 | 96.0 | +10.0 | `32,41` | `23,24,39,46,48` | none |
| PushT | 42 | 98.0 | 92.0 | -6.0 | none | `23` | `15,17,32,42` |
| PushT | 100 | 74.0 | 86.0 | +12.0 | `0,2,9,26,36,40` | `3,13,22,24,27,38,41` | `32` |
| TwoRoom | 0 | 84.0 | 80.0 | -4.0 | `1,3,16,17,18,28` | `8,14` | `24,27,39,48` |
| TwoRoom | 1 | 92.0 | 94.0 | +2.0 | `8,45,47` | `49` | none |
| TwoRoom | 2 | 80.0 | 88.0 | +8.0 | `1,3,11,19` | `10,14,18,34,42,48` | `22,29` |
| TwoRoom | 42 | 86.0 | 90.0 | +4.0 | `3,17,31` | `1,7,8,14` | `28,38` |
| TwoRoom | 100 | 80.0 | 76.0 | -4.0 | `1,10,28,29,30,35,38,41,49` | `3` | `16,32,33` |

Aggregate files:

- `reports/research/identity_delta/seed_sweep_summary.csv`
- `reports/research/identity_delta/seed_sweep_summary.json`
- Raw per-seed benchmark outputs were generated under `reports/research/identity_delta/seed_sweep/` during job `6201181`; the aggregate CSV/JSON above preserve the result table. The large raw run directories were not kept in git.

## Static And Training Audit

Static audit output:

- `reports/research/identity_delta/static_audit.md`
- `reports/research/identity_delta/audit_raw.json`

Architecture-critical metadata matches between upstream converted and retrained identity checkpoints:

- `adapter_family`: `lewm`
- `architecture_version`: `lewm_base_adapter_v1`
- `levels`: `[192]`
- `action_dim`: `2`
- `action_block`: `5`
- `image_shape`: `[224,224]`
- `source_config_sha256`: `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`
- `component_policy`: shared `latent_producer`, per-level `transition`, no reconstructor
- `action_preprocessing`: `standard_scaler`

Expected differences are that upstream checkpoints are converted pretrained Le-WM artifacts, while identity checkpoints are fresh K=[D] trainings with `training_backend=stable_worldmodel_lewm`, `fresh_init=true`, dataset path metadata, training recipe metadata, and `epoch=10`.

Training logs:

| env | Slurm log job | reached max epochs | Lightning last epoch/global step | last val loss | last val pred loss |
| --- | --- | --- | --- | ---: | ---: |
| PushT | `6192391` | true | `9 / 139000` | 0.122972 | 0.003396 |
| TwoRoom | `6192392` | true | `9 / 51000` | 0.158983 | 0.007544 |

There is no evidence that either identity run stopped early. The remaining checkpoint-selection caveat is that the training used `save_top_k=0`/`save_last=true`, so there is no best-validation checkpoint to compare against the final export.

## Dataset Audit

Dataset audit output: `reports/research/identity_delta/dataset_audit.md`.

| env | rows | episodes | mean episode length | action std | action range |
| --- | ---: | ---: | ---: | --- | --- |
| PushT | 2,336,736 | 18,685 | 125.06 | `[0.2085,0.2067]` | min `[-1.4947,-1.4655]`, max `[2.0427,1.7921]` |
| TwoRoom | 920,809 | 10,000 | 92.08 | `[0.8675,0.8686]` | min `[-1,-1]`, max `[1,1]` |

PushT has a much narrower action distribution with out-of-bounds tails relative to env action bounds, while TwoRoom actions saturate the full `[-1,1]` range. This likely contributes to PushT's higher manifest/planner sensitivity. It does not by itself show that identity was trained on the wrong data.

The official PushT upstream reference matrix at `${MWM_ARTIFACT_ROOT}/rollouts/lewm_official_reference/reference_matrix_6169579.json` is important: upstream converted Le-WM scores `92.0` on HDF5 and `98.0` on Lance for the same policy/reference check. That is direct evidence that the paper-target mismatch is sensitive to dataset representation/manifest details.

## Answers To The Required Questions

1. The observed delta is best classified as evaluator/manifest variance. It is not training-data, training-recipe/convergence, checkpoint-selection, or code mismatch as a primary cause. Dataset representation does explain the upstream-paper mismatch and remains a provenance caveat for comparing a converted upstream model to a fresh retraining, but it does not explain a paired same-manifest identity deficit because the deficit does not persist across seeds.

2. PushT identity does not generally underperform upstream. On the original seed `42`, upstream is unusually high (`98.0`) while identity is at its five-seed mean (`92.0`). TwoRoom does not show a negative identity result because its seed-42 paired delta is `+4.0`, and its five-seed mean delta is also near neutral (`+1.2`). PushT is more manifest-sensitive: upstream varies by 24 points across five seeds, compared with 12 points for TwoRoom.

3. Split the tolerance. The current `1%` paper target tolerance should not be used as a single gate for both upstream-paper and identity-upstream checks. With 50 episodes, one episode is 2 percentage points, so a 1 point tolerance is below the evaluator resolution. Recommended gates:
   - Upstream-paper: treat as an evaluator calibration/range check tied to dataset representation and manifest family, not a hard 96 +/- 1 gate.
   - Identity-upstream: use a paired same-manifest multi-seed aggregate check. A practical starting gate is mean identity-upstream delta >= -5 points over the five fixed seeds, with the full per-seed table reported rather than failing on a single seed.

4. The exact next experiment should be a higher-episode PushT-only paired evaluator run before any retraining:

```bash
env MWM_SWEEP_ENVS=pusht \
  MWM_SWEEP_EPISODES=200 \
  MWM_SWEEP_SEEDS="0 1 2 42 100" \
  sbatch scripts/research/slurm_research_identity_seed_sweep.sbatch
```

This directly tests whether the PushT identity advantage/neutrality survives lower sampling noise. Only if this 200-episode paired run shows a stable negative identity delta should we spend GPU time on training variants such as a PushT identity run with best-checkpoint selection and epoch-10/15/20 exports.

## Slurm Inspection And Exact Script

Before launching the GPU sweep, I inspected current PARCC/Slurm availability and partition details with:

```bash
sinfo -s
parcc_sfree.py
scontrol show partition b200-mig90
scontrol show node dgx029
```

The local Slurm state showed `b200-mig90` up, node `dgx029`, `gres/gpu:90gb=16`, and enough free MIG slices. I also checked public PARCC/Slurm references: PARCC lists `betty.b200.mig90` as a 90 GB B200 slice limited to one per job, and Slurm docs describe `#SBATCH` scripts plus `--gres=gpu:<type>:1` GPU requests.

Submitted script:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mwm_identity_seed_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:90gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
```

## Commands Run

```bash
git worktree add -b codex/identity-upstream-delta "${WORKTREE_ROOT}" origin/multienv-support
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python --version
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_repo_hygiene.py tests/test_mwm_local_workflow.py -q
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python scripts/research_identity_delta_audit.py
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/research/identity_delta_pusht_benchmark.yaml --static-only --roles upstream_lewm_converted retrained_lewm_identity
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/research/identity_delta_tworoom_benchmark.yaml --static-only --roles upstream_lewm_converted retrained_lewm_identity
bash -n scripts/research_identity_seed_sweep.sh scripts/slurm_research_identity_seed_sweep.sbatch
sbatch scripts/slurm_research_identity_seed_sweep.sbatch
sacct -j 6201181 --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocTRES%80 -P
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python scripts/research_identity_delta_collect.py
```

Current replay equivalents use `scripts/research/research_identity_delta_audit.py`, `scripts/research/research_identity_delta_collect.py`, `scripts/research/slurm_research_identity_seed_sweep.sbatch`, and `python -m mwm.benchmark.verify`.

## Blockers

No unresolved blocker. One failed Slurm submission (`6201097`) exposed a research-wrapper override bug; it was fixed and replaced by completed job `6201181`.
