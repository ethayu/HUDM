# Prompt: Dense MWM Performance Debug Agent

You are Research Agent 2. Work in a fresh git worktree from `origin/multienv-support` on branch `codex/dense-mwm-performance-debug`. Do not reuse another agent's worktree.

Your mission is to debug why dense MWM underperforms in the benchmark even though local experiments suggest dense models can be strong. Use this spec:

```text
docs/superpowers/specs/2026-05-30-dense-mwm-performance-debug.md
```

Current observed results:

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

Primary debug hypothesis:

Evaluating the dense checkpoint with planning fixed at the highest-fidelity level (`D`, finest) should get comparable performance if the model itself is good. If fixed-finest is strong, investigate scheduler policy. If fixed-finest is weak, investigate convergence/checkpoint/training.

Constraints:

- Use conda env `mwm`: `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.
- Do not run GPU or long-running jobs on login nodes.
- Before any GPU job, inspect PARCC Slurm docs and record the exact `sbatch` command/script you will run.
- Do not overwrite canonical checkpoints or benchmark configs.
- Put exploratory configs/scripts under `configs/research/` and `scripts/research_*`.
- Put final artifacts under `reports/research/dense_debug/`.
- Commit your scripts/config/report changes in your branch.

Suggested first commands:

```bash
git fetch origin
git worktree add ../mwm-dense-debug -b codex/dense-mwm-performance-debug origin/multienv-support
cd ../mwm-dense-debug
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/benchmark/dense_pusht.yaml --static-only --roles upstream_lewm_converted mwm_dense
```

Minimum deliverables:

```text
reports/research/dense_debug/report.md
reports/research/dense_debug/summary.json
```

Your report must answer:

1. What is dense fixed-finest performance on PushT and TwoRoom?
2. If fixed-finest is good, what scheduler behavior causes the benchmark drop?
3. If fixed-finest is bad, is dense undertrained, stale, or configured wrong?
4. Does dense need longer train time? If so, what concrete training config should be run next?
5. What exact next experiment should be run?

Return a concise final message with branch name, commit SHA(s), key findings, commands/job IDs, and any blockers.
