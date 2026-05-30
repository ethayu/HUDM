# Identity vs Upstream Performance Delta Research Spec

**Owner:** Research agent 1  
**Branch/worktree:** create a fresh worktree from `origin/multienv-support`, branch `codex/identity-upstream-delta`  
**Runtime:** conda env `mwm` or `MWM_PYTHON=/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python` on PARCC  
**Cluster rule:** do not run GPU or long-running jobs on login nodes. Before launching GPU jobs, inspect the PARCC Slurm docs and record the exact `sbatch` command/script in your report.

## Goal

Explain why retrained identity MWM (`K=[D]`) does not exactly match upstream converted Le-WM under the current evaluator, especially on PushT.

Current observations from the main validation run:

```text
PushT:
  upstream_lewm_converted: 98.0
  retrained_lewm_identity: 92.0

TwoRoom:
  upstream_lewm_converted: 86.0
  retrained_lewm_identity: 90.0
```

Strict PushT verifier failed because upstream is `98.0` versus configured paper target `96.0 +/- 1.0`, and identity is `92.0`, six percentage points below upstream.

## Research Questions

1. Is the PushT identity delta caused by training data distribution, dataset version, or train split differences?
2. Is it caused by training recipe differences: epochs, optimizer, LR, weight decay, precision, regularization, checkpoint selection, early stopping, or loss terms?
3. Is it caused by evaluation protocol: manifest, seed, CEM parameters, action preprocessing, frameskip/action block, restore state, goal offset, or environment kwargs?
4. Is the apparent upstream-versus-paper mismatch (`98` vs `96`) just evaluator variance, a manifest/sample delta, or a paper target/config mismatch?
5. Does identity parity hold at the architecture/objective level but fail empirically only because the retrained checkpoint is undertrained or trained on a shifted distribution?

## Required Starting Artifacts

Use these current configs and outputs when present:

```text
configs/benchmark/paper_parity_pusht.yaml
configs/benchmark/paper_parity_tworoom.yaml
configs/eval/paper_pusht.yaml
configs/eval/paper_tworoom.yaml
configs/train/mwm_lewm_pusht_upstream.yaml
configs/train/mwm_lewm_tworoom_upstream.yaml
checkpoints_mwm/upstream_lewm_pusht
checkpoints_mwm/upstream_lewm_tworoom
checkpoints_mwm/retrained_lewm_identity_pusht_upstream
checkpoints_mwm/retrained_lewm_identity_tworoom_upstream
rollouts/mwm_paper_parity_pusht
rollouts/mwm_paper_parity_tworoom
```

If rollouts are missing in your worktree, regenerate them through Slurm or copy from the main artifact location. Do not run the benchmark on a login node.

## Investigation Plan

### 1. Static Checkpoint and Config Audit

Write a short script or notebook-style Python file under `reports/research/identity_delta/` that dumps and compares:

- `world_metadata.json` for upstream converted and retrained identity checkpoints.
- `config.json` for both checkpoints.
- `configs/train/mwm_lewm_*_upstream.yaml`.
- `configs/eval/paper_*.yaml`.
- Benchmark resolved configs from `rollouts/mwm_paper_parity_*/*/resolved_config.yaml` if present.

Minimum fields to compare:

```text
adapter_family
architecture_version
training_backend
levels
D
action_dim
action_block
image_shape
restore_spec
source_config_sha256
component_policy
loss_scope
training_recipe
dataset path/format/metadata
dependency refs
optimizer/lr/weight_decay
max_epochs
precision
seed
checkpoint epoch
action preprocessing
planner horizon/receding_horizon/action_block/pop_size/topk/n_iter
eval goal_offset/episodes/num_envs/sampling/manifest path
```

Deliverable: table of exact differences and an interpretation of which differences are expected versus suspicious.

### 2. Dataset Distribution Audit

Compare the training/eval data used by identity and upstream parity:

```text
data/upstream/pusht_expert_train.lance
data/upstream/tworoom.lance
```

Compute:

- Row count and episode count.
- Episode length distribution.
- Available columns.
- Metadata sidecar fields.
- Action mean/std/min/max and per-dimension quantiles.
- Proprio/state mean/std/min/max when available.
- Pixel shape, dtype, range, and a small sample mean/std.
- Whether action normalization statistics in eval match what checkpoint metadata expects.

Deliverable: `reports/research/identity_delta/dataset_audit.md` plus a machine-readable JSON or CSV summary.

### 3. Evaluator and Manifest Sensitivity

Run small but controlled benchmark sweeps to estimate whether the PushT delta is stable or within sample variance.

Use benchmark configs copied into `configs/research/` that have `paper_targets.enabled: false` or no `paper_targets`, so exploratory verification does not stop analysis. Use the same roles:

```text
upstream_lewm_converted
retrained_lewm_identity
```

Required sweep:

- PushT seeds: `0, 1, 2, 42, 100`.
- TwoRoom seeds: `0, 1, 2, 42, 100`.
- Keep `episodes: 50` for paper-like runs when resources allow. If resource-limited, first run `episodes: 10` smoke sweep and state clearly that it is only directional.

Recommended exact Slurm pattern after inspecting PARCC docs:

```bash
sbatch --job-name=mwm_identity_seed_sweep \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --partition=b200-mig90 \
  --gres=gpu:90gb:1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=08:00:00 \
  --wrap='cd "$SLURM_SUBMIT_DIR" && export MWM_PYTHON=/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python && scripts/research_identity_seed_sweep.sh'
```

The agent may instead create `scripts/research_identity_seed_sweep.sh` and a matching `scripts/slurm_research_identity_seed_sweep.sbatch` in its branch. Keep those scripts research-scoped and do not modify production benchmark configs unless necessary.

Deliverable: success-rate table by env/seed/role, manifest hash, and diagnostics.

### 4. Training Recipe and Convergence Audit

Inspect available training logs and checkpoint metadata for:

- Training loss trend.
- Validation loss trend.
- Whether the selected canonical checkpoint is final, best, or a salvaged checkpoint.
- Whether identity training ran to the intended `max_epochs`.
- Whether regularization/loss terms match base training semantics.
- Whether precision and device settings differ from upstream training.

If logs are missing, state that explicitly and use checkpoint metadata plus Slurm logs.

Optional follow-up if evidence suggests undertraining:

- Relaunch PushT identity training for longer with a new run name, not overwriting existing checkpoint.
- Compare final and best checkpoint if Lightning checkpoints exist.

Do not launch this optional training unless the static audit points to undertraining or checkpoint-selection ambiguity.

### 5. Final Report

Create:

```text
reports/research/identity_delta/report.md
reports/research/identity_delta/summary.json
```

The report must include:

- Executive answer: most likely cause(s), confidence level, and why.
- Exact commands run.
- Job IDs for any Slurm runs.
- Tables for checkpoint/config differences, dataset stats, and eval seed sweep.
- Whether evaluator parity is trustworthy.
- Whether the 1 percent paper tolerance should be changed, kept, or split into separate upstream-paper and identity-upstream tolerances.
- Recommended next experiment with the smallest expected cost.

## Acceptance Criteria

The research is complete when:

- Every research question above has an evidence-backed answer or a clearly named unknown.
- PushT identity delta is localized to one or more of: training data, training recipe/convergence, evaluator/manifest variance, checkpoint selection, or code mismatch.
- TwoRoom behavior is explained, especially why identity is not below upstream there.
- The final report includes enough commands and artifact paths for another agent to reproduce the conclusion.
