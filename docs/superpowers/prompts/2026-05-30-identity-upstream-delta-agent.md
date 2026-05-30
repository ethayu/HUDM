# Prompt: Identity vs Upstream Delta Research Agent

You are Research Agent 1. Work in a fresh git worktree from `origin/multienv-support` on branch `codex/identity-upstream-delta`. Do not reuse another agent's worktree.

Your mission is to investigate exactly why retrained identity MWM (`K=[D]`) differs from upstream converted Le-WM. Use this spec:

```text
docs/superpowers/specs/2026-05-30-identity-upstream-delta-research.md
```

Current observed results:

```text
PushT:
  upstream_lewm_converted: 98.0
  retrained_lewm_identity: 92.0

TwoRoom:
  upstream_lewm_converted: 86.0
  retrained_lewm_identity: 90.0
```

Constraints:

- Use conda env `mwm`: `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.
- Do not run GPU or long-running jobs on login nodes.
- Before any GPU job, inspect PARCC Slurm docs and record the exact `sbatch` command/script you will run.
- Do not overwrite canonical checkpoints or benchmark configs.
- Put exploratory configs/scripts under `configs/research/` and `scripts/research_*`.
- Put final artifacts under `reports/research/identity_delta/`.
- Commit your scripts/config/report changes in your branch.

Suggested first commands:

```bash
git fetch origin
git worktree add ../mwm-identity-delta -b codex/identity-upstream-delta origin/multienv-support
cd ../mwm-identity-delta
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_artifacts.py::MWMArtifactTests::test_benchmark_verifier_static_only_accepts_paper_parity_config
```

Minimum deliverables:

```text
reports/research/identity_delta/report.md
reports/research/identity_delta/summary.json
```

Your report must answer:

1. Is the delta primarily training data, training recipe/convergence, evaluator/manifest variance, checkpoint selection, or code mismatch?
2. Why does PushT identity underperform upstream while TwoRoom identity does not?
3. Is the current 1 percent paper target tolerance the right check, or should upstream-paper and identity-upstream tolerances be separated?
4. What exact next experiment should be run?

Return a concise final message with branch name, commit SHA(s), key findings, commands/job IDs, and any blockers.
