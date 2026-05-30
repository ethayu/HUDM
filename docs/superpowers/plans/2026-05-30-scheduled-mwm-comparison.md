# Scheduled MWM Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train fresh multi-level Le-WM-derived MWM checkpoints with `K=[48,96,144]`, then compare scheduled-fidelity planning against upstream Le-WM performance and efficiency.

**Architecture:** The train configs keep the base-adaptive MWM contract: shared `encoder+projector`, per-level Le-WM transition tails, shared latent SIGReg, and no reconstructor contribution. The benchmark uses shared paper-parity manifests and compares only `upstream_lewm_converted` against `mwm_scheduled`, so efficiency plots and schedule diagnostics are focused on the fidelity-switching planner.

**Tech Stack:** Python, PyTorch Lightning through `stable_pretraining`, Stable-WM Lance data, Slurm `sbatch`, `benchmark_mwm.py`, `verify_mwm_benchmark.py`.

---

### Task 1: Align Scheduled Configs To Paper-Parity Inputs

**Files:**
- Modify: `configs/train/mwm_scheduled_pusht.yaml`
- Modify: `configs/train/mwm_scheduled_tworoom.yaml`
- Modify: `configs/benchmark/scheduled_pusht.yaml`

- [ ] **Step 1: Update scheduled training data and seed**

Set both scheduled train configs to use the same paper-parity Lance data as the upstream/identity checks:

```yaml
seed: 3072
data:
  path: data/upstream/pusht_expert_train.lance
```

and:

```yaml
seed: 3072
data:
  path: data/upstream/tworoom.lance
```

Keep `model.K: [48, 96, 144]`, `model.D: 192`, `action_block: 5`, `schedule.max_epochs: 10`, and `train.run_name: mwm_scheduled_{env}`.

- [ ] **Step 2: Replace the benchmark matrix with scheduled comparison**

Write one single-environment benchmark config per environment, for example
`configs/benchmark/scheduled_pusht.yaml`:

```yaml
env_id: swm/PushT-v1
seed: 42
eval_config: configs/eval/paper_pusht.yaml
manifest:
  config: configs/manifest/pusht_paper_seed42.yaml
runs:
  - role: upstream_lewm_converted
    checkpoint: checkpoints_mwm/upstream_lewm_pusht
  - role: mwm_scheduled
    checkpoint: checkpoints_mwm/mwm_scheduled_pusht
```

Keep upstream on fixed finest/base planning, and override only the scheduled MWM runs to use:

```yaml
planner:
  scheduler:
    policy: linear_cem
    start_level: coarsest
    end_level: finest
    rollout_level:
      mode: fixed
      level: base
```

### Task 2: Add True MWM Slurm Launch Path

**Files:**
- Create: `scripts/run_mwm_train_scheduled_env.sh`
- Create: `scripts/slurm_mwm_train_pusht_scheduled.sbatch`
- Create: `scripts/slurm_mwm_train_tworoom_scheduled.sbatch`
- Create: `scripts/run_mwm_scheduled_comparison.sh`
- Create: `scripts/slurm_mwm_scheduled_comparison.sbatch`
- Create: `scripts/submit_mwm_scheduled_split.sh`
- Create: `scripts/poll_mwm_scheduled_jobs.sh`

- [ ] **Step 1: Create the Slurm-only train wrapper**

`scripts/run_mwm_train_scheduled_env.sh` must reject non-Slurm execution, run `verify_mwm_data.py --paper-parity`, and then dispatch:

```bash
configs/train/mwm_scheduled_pusht.yaml
configs/train/mwm_scheduled_tworoom.yaml
```

- [ ] **Step 2: Create two one-GPU train sbatch files**

Use `dgx-b200`, `--gres=gpu:B200:1`, `--cpus-per-task=16`, `--mem=128G`, `--time=4-00:00:00`, and call `scripts/run_mwm_train_scheduled_env.sh pusht` or `tworoom`.

- [ ] **Step 3: Create the comparison benchmark wrapper**

`scripts/run_mwm_scheduled_comparison.sh` must reject non-Slurm execution, run:

```bash
$PY verify_mwm_data.py --paper-parity
$PY benchmark_mwm.py configs/benchmark/scheduled_pusht.yaml
$PY verify_mwm_benchmark.py configs/benchmark/scheduled_pusht.yaml
```

- [ ] **Step 4: Create benchmark sbatch**

Use `b200-mig90`, `--gres=gpu:90gb:1`, `--cpus-per-task=16`, `--mem=128G`, `--time=2-00:00:00`, and call `scripts/run_mwm_scheduled_comparison.sh`.

- [ ] **Step 5: Create the submitter**

`scripts/submit_mwm_scheduled_split.sh` must submit the two train jobs separately, then submit the benchmark with:

```bash
sbatch --parsable --dependency="afterok:${PUSHT_JOB}:${TWOROOM_JOB}" scripts/slurm_mwm_scheduled_comparison.sbatch
```

Print the exact `squeue` and `sacct` monitor commands.

### Task 3: Add Verification Coverage

**Files:**
- Modify: `tests/test_mwm_artifacts.py`
- Modify: `tests/test_mwm_repo_hygiene.py`

- [ ] **Step 1: Assert benchmark contract**

Add a static-only benchmark test that calls:

```python
report = verify_benchmark_static("configs/benchmark/scheduled_pusht.yaml", check_checkpoints=False)
```

and asserts `runs == 4`, roles are upstream and scheduled only, and `output_dir == "rollouts/mwm_scheduled_comparison"`.

- [ ] **Step 2: Assert scheduled configs use paper-parity data**

In repo hygiene tests, assert scheduled train configs point under `data/upstream/`.

### Task 4: Verify And Launch

**Files:**
- No code files beyond Tasks 1-3.

- [ ] **Step 1: Run focused tests**

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_artifacts.py tests/test_mwm_repo_hygiene.py
```

- [ ] **Step 2: Run static benchmark verification**

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/benchmark/scheduled_pusht.yaml --static-only --roles upstream_lewm_converted mwm_scheduled
```

- [ ] **Step 3: Submit jobs**

Per PARCC Slurm docs, GPU work is submitted by `sbatch`, not run directly on the login node:

```bash
scripts/submit_mwm_scheduled_split.sh
```

Expected job chain:

```text
mwm_train_pusht_scheduled
mwm_train_tworoom_scheduled
mwm_scheduled_comparison, dependent on both training jobs afterok
```
