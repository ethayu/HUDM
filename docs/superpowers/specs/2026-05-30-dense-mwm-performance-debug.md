# Dense MWM Performance Debug Research Spec

**Owner:** Research agent 2  
**Branch/worktree:** create a fresh worktree from `origin/multienv-support`, branch `codex/dense-mwm-performance-debug`  
**Runtime:** conda env `mwm` or `MWM_PYTHON=/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python` on PARCC  
**Cluster rule:** do not run GPU or long-running jobs on login nodes. Before launching GPU jobs, inspect the PARCC Slurm docs and record the exact `sbatch` command/script in your report.

## Goal

Explain why dense MWM underperforms upstream in the current benchmark, despite local experiments suggesting dense models can be strong.

Current observations:

```text
Dense PushT:
  upstream_lewm_converted: 98.0
  mwm_dense: 60.0

Dense TwoRoom:
  upstream_lewm_converted: 86.0
  mwm_dense: 64.0

Scheduled model, expected bad baseline:
  PushT scheduled: 8.0
  TwoRoom scheduled: 30.0
```

Dense levels should be:

```text
K = [6, 12, 48, 96, 144, 192]
```

The key debug expectation is: evaluating the dense checkpoint with planning fixed at the highest-fidelity level (`D`, finest) should be comparable to upstream or at least much better than the scheduled dense benchmark. If fixed-finest is strong, the problem is likely planner/schedule. If fixed-finest is weak, the problem is likely training convergence, checkpoint selection, or model construction/objective.

## Research Questions

1. Does dense fixed-finest planning recover near-upstream performance?
2. If fixed-finest is strong, which scheduler or fidelity-switching policy causes the benchmark drop?
3. If fixed-finest is weak, is dense training under-converged or using the wrong checkpoint?
4. Are all dense levels present, correctly ordered, and being used by the planner as intended?
5. Do training diagnostics show dense needs longer training than the current 10 epochs?
6. Is the dense benchmark accidentally evaluating stale checkpoints, wrong configs, wrong manifests, or a mismatched inference recipe?

## Required Starting Artifacts

Use these configs and outputs when present:

```text
configs/benchmark/dense_pusht.yaml
configs/benchmark/dense_tworoom.yaml
configs/eval/paper_pusht.yaml
configs/eval/paper_tworoom.yaml
configs/train/mwm_dense_pusht.yaml
configs/train/mwm_dense_tworoom.yaml
checkpoints_mwm/mwm_dense_pusht
checkpoints_mwm/mwm_dense_tworoom
checkpoints_mwm/upstream_lewm_pusht
checkpoints_mwm/upstream_lewm_tworoom
rollouts/mwm_dense_pusht
rollouts/mwm_dense_tworoom
```

If rollouts are missing in your worktree, regenerate through Slurm or copy from the main artifact location. Do not run full benchmarks on the login node.

## Investigation Plan

### 1. Dense Checkpoint and Config Audit

Create `reports/research/dense_debug/checkpoint_audit.md` comparing:

- `checkpoints_mwm/mwm_dense_pusht/world_metadata.json`
- `checkpoints_mwm/mwm_dense_tworoom/world_metadata.json`
- `checkpoints_mwm/*/config.json`
- `configs/train/mwm_dense_*.yaml`
- `configs/benchmark/dense_*.yaml`
- Existing dense rollout resolved configs.

Minimum fields:

```text
levels
D
adapter_family
architecture_version
training_backend
source_config_sha256
component_policy
loss_scope
training_recipe
action_dim/action_block
image_shape
restore_spec
max_epochs
precision
checkpoint_every_n_train_steps
optimizer/lr/weight_decay
train seed
eval seed
planner scheduler policy/start_level/end_level/rollout_level
manifest group/path/hash
```

Flag anything that suggests stale checkpoint, wrong levels, wrong run name, or wrong evaluator path.

### 2. Scheduler Diagnostics Audit

Parse existing dense rollout payloads:

```text
rollouts/mwm_dense_pusht/*/eval.json
rollouts/mwm_dense_tworoom/*/eval.json
rollouts/mwm_dense_pusht/*/planning_diagnostics.json
rollouts/mwm_dense_tworoom/*/planning_diagnostics.json
```

Summarize:

- `schedule_level_counts`
- total/mean plan time
- total CEM cost calls
- candidate action values
- bits used estimate
- success by episode
- whether dense used the expected levels

Deliverable: `reports/research/dense_debug/scheduler_diagnostics.md`.

### 3. Fixed-Level Benchmark Ladder

Create research benchmark configs under `configs/research/` to evaluate dense checkpoints with fixed planner levels.

Minimum ladder:

```text
PushT dense fixed finest
TwoRoom dense fixed finest
PushT dense fixed coarsest
TwoRoom dense fixed coarsest
PushT dense fixed middle, e.g. 96
TwoRoom dense fixed middle, e.g. 96
```

Use the same manifest group and eval config as paper parity. For fixed finest, the scheduler should be:

```yaml
planner:
  scheduler:
    policy: fixed
    level: finest
    rollout_level:
      mode: fixed
      level: base
```

For explicit numeric levels, inspect `mwm/planning/scheduled_cem.py` and existing scheduler parsing before choosing syntax. If numeric-level fixed scheduling is not supported, record that and test supported named levels only.

Recommended exact Slurm pattern after inspecting PARCC docs:

```bash
sbatch --job-name=mwm_dense_fixed_ladder \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --partition=b200-mig90 \
  --gres=gpu:90gb:1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=08:00:00 \
  --wrap='cd "$SLURM_SUBMIT_DIR" && export MWM_PYTHON=/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python && scripts/research_dense_fixed_ladder.sh'
```

Deliverable: table of success and compute metrics for each env/level/role.

### 4. Schedule Policy Ablations

If fixed-finest is strong, evaluate scheduler variants:

- Existing `linear_cem`.
- Fixed finest.
- Fixed middle if supported.
- Fixed coarsest.
- A conservative policy that starts at middle or finest earlier.
- Same schedule but different `rollout_level` if the planner supports it.

For each policy, record success, compute, schedule counts, and wall time.

Deliverable: answer whether dense model quality is good and the scheduler is the bottleneck.

### 5. Convergence and Longer Training Probe

If fixed-finest is weak, audit convergence first:

- Parse Slurm training logs for dense PushT/TwoRoom.
- Find last epoch and loss trend.
- Compare dense loss trend to identity and scheduled training if logs exist.
- Check whether dense checkpoint was exported from final weights.

If evidence supports undertraining, launch a longer dense training probe with a new run name, preserving the current checkpoint:

```bash
sbatch --job-name=mwm_dense_pusht_long \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --partition=dgx-b200 \
  --gres=gpu:B200:1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=1-00:00:00 \
  --wrap='cd "$SLURM_SUBMIT_DIR" && export MWM_PYTHON=/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python && $MWM_PYTHON train_mwm.py configs/train/mwm_dense_pusht.yaml --set train.run_name=mwm_dense_pusht_long --set schedule.max_epochs=30'
```

Repeat TwoRoom only if PushT suggests convergence is the cause or resources allow.

After longer training completes, evaluate fixed-finest first, then scheduler. Do not overwrite existing canonical dense checkpoints.

### 6. Final Report

Create:

```text
reports/research/dense_debug/report.md
reports/research/dense_debug/summary.json
```

The report must include:

- Executive answer: scheduler problem, convergence problem, stale artifact problem, or mixed.
- Exact commands and Slurm job IDs.
- Fixed-finest result as the primary debug signal.
- Per-level or per-scheduler performance table.
- Training convergence evidence.
- Recommended next training/eval configuration.

## Acceptance Criteria

The research is complete when:

- Dense fixed-finest performance is measured for PushT and TwoRoom, or a blocking reason is documented.
- Existing dense scheduler diagnostics are parsed and interpreted.
- The report identifies whether to focus next on longer training or scheduler design.
- If longer training is launched, the new checkpoint/run name is separate from canonical checkpoints.
- Another agent can reproduce the key result from the report alone.
