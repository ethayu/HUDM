# Reacher Identity-Upstream Delta Research Report

Branch: `codex/reacher-identity-upstream-delta`
Audit head: `2f54b271d7eeb3c77cb0a03b7299d9f9adb8948b`
Base from `origin/multienv-support`: `2f54b271d7eeb3c77cb0a03b7299d9f9adb8948b`
Worktree: `${WORKTREE_ROOT}`
Artifact root audited: `${MWM_ARTIFACT_ROOT}`

## Executive Answer

The observed Reacher result is upstream `80.0` versus retrained identity `86.0` on a single 50-episode manifest. That is a three-episode difference, because each episode is worth 2 percentage points. The paired discordance is identity-better on 8 episodes and upstream-better on 5 episodes, with a two-sided sign-test p-value of `0.581`. This is not strong evidence that the retrained identity checkpoint is truly better; it is best classified as evaluator/manifest sampling variance plus ordinary planner noise at low episode count.

There is no evidence in the audited artifacts for a training-data mismatch, restore/qpos-goal bug, architecture mismatch, or failed convergence. Both checkpoints evaluate on the same manifest, same dataset, same Reacher qpos-match restore spec, same `K=[192]`, same flattened action spec `dim=10`, and same eval pipeline.

## Required Answers

1. Cause classification: primary cause is evaluator/manifest variance at 50 episodes. Training data, training recipe/convergence, checkpoint selection, restore/qpos-goal handling, and code mismatch are not supported as primary causes by the current evidence. Checkpoint selection remains a small unresolved caveat because the identity export uses the final/last Lightning checkpoint, not a best-validation checkpoint.

2. Why identity appears to outperform upstream: it wins by only 3 episodes on one fixed manifest. The identity model is also trained directly against the current Lance data and preprocessing recipe, while upstream is a converted Le-WM artifact. That could plausibly move a few borderline CEM decisions, but this run is too small to call it a real performance advantage.

3. Meaningfulness of +6pp: weak. The run has 2pp granularity, independent-binomial diff standard error is `7.49` pp, and the paired sign test is not significant. The failure sets are mostly different rather than a fixed subset of broken Reacher goals: shared failures `[27, 48]`, upstream-only failures `[1, 6, 12, 16, 22, 26, 41, 46]`, identity-only failures `[7, 18, 25, 31, 33]`.

4. Tolerance recommendation: split upstream-paper and identity-upstream checks, and make them episode-count aware. A single 1 percent gate is below the resolution of a 50-episode run. Use `max(1pp, 100 / episodes)` as a minimum reporting resolution, and use paired count/seed-sweep checks for identity-upstream rather than a one-manifest percent delta.

5. Exact next experiment: run a five-seed paired Reacher sweep at 200 episodes per seed for the same two checkpoints before retraining anything:

```bash
MWM_REACHER_SWEEP_EPISODES=200 MWM_REACHER_SWEEP_SEEDS="0 1 2 42 100" sbatch scripts/research/slurm_research_reacher_identity_seed_sweep.sbatch
```

No new GPU jobs were submitted for this investigation. Current Slurm state was inspected with `sinfo -s` and `scontrol show partition dgx-b200`; the `dgx-b200` partition is up, has B200 GPUs, and the exact research-only sbatch script is recorded in `scripts/research/slurm_research_reacher_identity_seed_sweep.sbatch`.

## Observed Rollout

| role | success | episodes | seed | manifest | output |
| --- | ---: | ---: | ---: | --- | --- |
| `upstream_lewm_converted` | 80.0 | 50 | 42 | `04caca1fe5c2` | `rollouts/mwm_paper_parity_reacher/000_upstream/eval.json` |
| `retrained_lewm_identity` | 86.0 | 50 | 42 | `04caca1fe5c2` | `rollouts/mwm_paper_parity_reacher/001_retrained_identity/eval.json` |

- Manifest path: `rollouts/manifests/reacher_paper_seed42.json`
- Manifest sha256: `7069e71542035990340226345861c3d007b0d85c6e0d6617680dbc5b4199ffa6`
- Immutable manifest hash: `04caca1fe5c2387a102291e617b7f1aaac0a2d31aed4008947010596eee59cb5`
- Upstream failures: `[1, 6, 12, 16, 22, 26, 27, 41, 46, 48]`
- Identity failures: `[7, 18, 25, 27, 31, 33, 48]`
- Shared successes/failures: `35` / `2`
- Identity-better/upstream-better discordant counts: `8` / `5`

## Restore And Dataset

- Restore import path: `mwm.swm.restore.reacher_qpos_match_restore_spec`
- Restore spec id: `reacher_qpos_match_qpos_qvel`
- Required columns: `['qpos', 'qvel']`
- Eval callable 1: `set_state(qpos=<start qpos>, qvel=<start qvel>)`
- Eval callable 2: `set_target_qpos(target_qpos=<goal qpos>)`
- Eval callables raw: `[{"args": {"qpos": {"in_dataset": true, "value": "qpos"}, "qvel": {"in_dataset": true, "value": "qvel"}}, "method": "set_state"}, {"args"...`
- Missing-column checks: `{"qpos": "Restore spec 'reacher_qpos_match_qpos_qvel' requires dataset columns ['qpos']. Collect the dataset with the matching SWM restor...`
- Dataset rows: `2010000`
- Dataset episodes: `10000`
- Episode length mean/min/max: `201.00` / `201` / `201`
- Dataset sidecar restore spec: `reacher_qpos_match_qpos_qvel`
- Dataset raw action dim: `2`
- Dataset source: `{"artifact": "reacher.lance", "format": "lance", "hf_dataset": "quentinll/lewm-reacher", "standard": "paper_parity"}`

## Checkpoints

| field | upstream | identity | same |
| --- | --- | --- | --- |
| `env_id` | `"swm/ReacherDMControl-v0"` | `"swm/ReacherDMControl-v0"` | True |
| `restore_spec` | `"reacher_qpos_match_qpos_qvel"` | `"reacher_qpos_match_qpos_qvel"` | True |
| `adapter_family` | `"lewm"` | `"lewm"` | True |
| `architecture_version` | `"lewm_base_adapter_v1"` | `"lewm_base_adapter_v1"` | True |
| `levels` | `[192]` | `[192]` | True |
| `D` | `null` | `192` | False |
| `action_dim` | `2` | `2` | True |
| `action_block` | `5` | `5` | True |
| `action_spec` | `{"base_dim": 2, "block": 5, "dim": 10}` | `{"base_dim": 2, "block": 5, "dim": 10}` | True |
| `image_shape` | `[224, 224]` | `[224, 224]` | True |
| `source_config_sha256` | `"2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"` | `"2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"` | True |
| `fresh_init` | `false` | `true` | False |
| `epoch` | `null` | `9` | False |
| `best_checkpoint` | `null` | `null` | True |
| `last_checkpoint` | `null` | `"logs/mwm_training/retrained_lewm_identity_reacher_upstream/checkpoints/last.ckpt"` | False |

- Eval model accounting upstream: `{"D": 192, "K": [192], "num_levels": 1, "parameters": 18034478}`
- Eval model accounting identity: `{"D": 192, "K": [192], "num_levels": 1, "parameters": 18034478}`
- Upstream flattened action dim from config: `10`
- Identity flattened action dim from config: `10`

## Training Recipe And Convergence

- Identity train config: `configs/train/mwm_lewm_reacher_upstream.yaml`
- Data path in train config: `data/upstream/reacher.lance`
- Train seed: `3072`
- Max epochs configured: `10`
- Train job log: `${MWM_ARTIFACT_ROOT}/logs/mwm_train_reacher_identity_6782935.out`
- Historical train job id: `6782935`
- Max epochs reached: `True`
- Exact training complete marker: `True`
- Last fit/pred loss: `0.021538827568292618`
- Last validate/pred loss: `0.024026503786444664`

The identity run reached epoch 9 of 9 and exported `checkpoints_mwm/retrained_lewm_identity_reacher_upstream`. This does not prove optimal convergence, but it rules out an obvious early-stop or missing-export explanation for the 50-episode result.

## Config Audit

- Benchmark config: `configs/benchmark/paper_parity_reacher.yaml` sha `b7e4add9f406`
- Eval config: `configs/eval/paper_reacher.yaml` sha `359049c5c7be`
- Train config: `configs/train/mwm_lewm_reacher_upstream.yaml` sha `9883a0d4b8d0`
- Eval env kwargs: `{"task": "qpos_match"}`
- Eval restore import path: `mwm.swm.restore.reacher_qpos_match_restore_spec`
- Eval keys_to_load: `['pixels', 'action', 'qpos', 'qvel', 'observation']`
- Train `K`: `[192]`
- Train frameskip/action_block: `5` / `5`

## Commands And Jobs

Commands used in this investigation:

```bash
git worktree add .worktrees/codex-reacher-identity-upstream-delta -b codex/reacher-identity-upstream-delta origin/multienv-support
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m py_compile $(rg --files -g '*.py')
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_local_workflow.py tests/test_mwm_repo_hygiene.py tests/test_mwm_core.py tests/test_mwm_artifacts.py
sinfo -s
scontrol show partition dgx-b200
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python scripts/research/research_reacher_identity_delta_audit.py
bash -n scripts/research/research_reacher_identity_seed_sweep.sh scripts/research/slurm_research_reacher_identity_seed_sweep.sbatch
```

Historical jobs referenced from existing artifacts:

- Identity Reacher training: `6782935`
- Identity parity benchmark including Reacher: `6784362`
- New GPU jobs submitted by this investigation: none

## Blockers

No blocker for the static conclusion. The only remaining uncertainty is statistical: the current Reacher comparison has one 50-episode manifest, so it cannot establish whether the +6pp identity advantage persists across seeds or higher episode counts.
