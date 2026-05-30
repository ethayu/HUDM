# Local Desktop Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make HUDM/MWM usable on a regular desktop for install checks, static validation, tiny smoke data/eval/benchmark runs, and optional CPU training smoke tests without weakening the existing Slurm-safe cluster workflow.

**Architecture:** Keep cluster scripts Slurm-guarded and add separate local entrypoints/configs. Add a small shared config-loading helper so local scripts can override config values without proliferating near-duplicate YAMLs. Treat full paper-scale training/benchmarking as GPU/cluster work; local workflow is for smoke, debugging, and contributor confidence.

**Tech Stack:** Python, OmegaConf, pytest, Bash, Stable-WM Lance datasets, existing `train_mwm.py`/`eval_mwm.py`/`benchmark_mwm.py`/`verify_mwm_benchmark.py` entrypoints.

---

## File Structure

- Create `mwm/config_cli.py`: shared helper for loading a YAML config, merging defaults, and applying `--set key=value` dotlist overrides.
- Modify `collect_mwm_data.py`, `train_mwm.py`, `eval_mwm.py`, `benchmark_mwm.py`: support local CLI overrides through `--set`.
- Modify `verify_mwm_benchmark.py`: add `--no-checkpoints` for static config checks on machines without copied checkpoints.
- Create `configs/local/collect_pusht_smoke.yaml`: tiny desktop PushT Lance collection config.
- Create `configs/local/eval_pusht_smoke.yaml`: tiny eval config against `checkpoints_mwm/upstream_lewm_pusht`.
- Create `configs/local/benchmark_pusht_smoke.yaml`: one-env smoke benchmark using the local eval config and manifest group.
- Create `configs/local/train_pusht_cpu_smoke.yaml`: optional one-batch CPU train smoke config.
- Create `scripts/local_verify.sh`: runs syntax/tests/static checks with normal `python`, no Slurm.
- Create `scripts/local_benchmark_smoke.sh`: preflights local data/checkpoint and runs the smoke benchmark.
- Create `scripts/local_train_smoke.sh`: explicit opt-in CPU smoke training wrapper.
- Modify `README.md` and `REVIEW_GUIDE.md`: document local vs Slurm workflows and limitations.
- Modify `tests/test_mwm_repo_hygiene.py`: enforce local scripts are not Slurm-gated or PARCC-path hardcoded, while cluster scripts remain Slurm-gated.
- Create `tests/test_mwm_config_cli.py`: unit tests for CLI override parsing/merge behavior.

---

### Task 1: Shared Config Override Helper

**Files:**
- Create: `mwm/config_cli.py`
- Create: `tests/test_mwm_config_cli.py`

- [ ] **Step 1: Write failing tests for config override loading**

Create `tests/test_mwm_config_cli.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from mwm.config_cli import load_config


class ConfigCLITests(unittest.TestCase):
    def test_load_config_applies_dotlist_overrides_after_yaml(self) -> None:
        defaults = {"train": {"no_cuda": False, "batch_size": 8}, "eval": {"episodes": 4}}
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text(
                "train:\n  batch_size: 16\neval:\n  episodes: 10\n",
                encoding="utf-8",
            )

            cfg = load_config(defaults, cfg_path, overrides=["train.no_cuda=true", "eval.episodes=2"])

            self.assertEqual(cfg.train.batch_size, 16)
            self.assertEqual(cfg.eval.episodes, 2)
            self.assertTrue(cfg.train.no_cuda)

    def test_load_config_accepts_empty_overrides(self) -> None:
        defaults = {"device": "auto"}
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.yaml"
            cfg_path.write_text("device: cpu\n", encoding="utf-8")

            cfg = load_config(defaults, cfg_path)

            self.assertEqual(cfg.device, "cpu")
            self.assertTrue(OmegaConf.is_config(cfg))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_config_cli.py
```

Expected: fail with `ModuleNotFoundError: No module named 'mwm.config_cli'`.

- [ ] **Step 3: Implement minimal helper**

Create `mwm/config_cli.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


def load_config(defaults: dict[str, Any], cfg_path: str | Path, overrides: Iterable[str] = ()) -> Any:
    cfg = OmegaConf.merge(defaults, OmegaConf.load(str(cfg_path)))
    override_items = [str(item) for item in overrides]
    if override_items:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_items))
    return cfg
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_config_cli.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add mwm/config_cli.py tests/test_mwm_config_cli.py
git commit -m "feat: add config override loader"
```

---

### Task 2: Add `--set` Overrides to Local-Safe Python Entrypoints

**Files:**
- Modify: `collect_mwm_data.py`
- Modify: `train_mwm.py`
- Modify: `eval_mwm.py`
- Modify: `benchmark_mwm.py`
- Test: `tests/test_mwm_repo_hygiene.py`

- [ ] **Step 1: Write failing hygiene test for entrypoint override support**

Add this test to `tests/test_mwm_repo_hygiene.py`:

```python
    def test_desktop_entrypoints_accept_set_overrides(self) -> None:
        entrypoints = [
            ROOT / "collect_mwm_data.py",
            ROOT / "train_mwm.py",
            ROOT / "eval_mwm.py",
            ROOT / "benchmark_mwm.py",
        ]
        for path in entrypoints:
            text = path.read_text(encoding="utf-8")
            self.assertIn("load_config", text, path)
            self.assertIn("--set", text, path)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_desktop_entrypoints_accept_set_overrides
```

Expected: fail because the entrypoints still call `OmegaConf.merge(DEFAULTS, OmegaConf.load(...))` directly.

- [ ] **Step 3: Patch `collect_mwm_data.py`**

Import the helper:

```python
from mwm.config_cli import load_config
```

Change `main`:

```python
def main(cfg_path: str, *, overrides: list[str] | None = None) -> None:
    cfg = load_config(DEFAULTS, cfg_path, overrides or [])
```

Replace the `__main__` block with:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect a Stable-WM Lance dataset for MWM.")
    parser.add_argument("config", help="Collection YAML config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist override")
    args = parser.parse_args()
    main(args.config, overrides=args.set)
```

- [ ] **Step 4: Patch `eval_mwm.py`**

Import the helper:

```python
from mwm.config_cli import load_config
```

Change `main`:

```python
def main(cfg_path: str, *, overrides: list[str] | None = None) -> None:
    cfg = load_config(DEFAULTS, cfg_path, overrides or [])
```

Replace the `__main__` block with:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an MWM checkpoint.")
    parser.add_argument("config", help="Evaluation YAML config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist override")
    args = parser.parse_args()
    main(args.config, overrides=args.set)
```

- [ ] **Step 5: Patch `benchmark_mwm.py`**

Import the helper:

```python
from mwm.config_cli import load_config
```

Change `main`:

```python
def main(cfg_path: str, *, roles: Any = None, overrides: list[str] | None = None) -> None:
    cfg = load_config(DEFAULTS, cfg_path, overrides or [])
```

Add the CLI option:

```python
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist override")
```

Call:

```python
main(args.config, roles=args.roles, overrides=args.set)
```

- [ ] **Step 6: Patch `train_mwm.py`**

Import the helper:

```python
from mwm.config_cli import load_config
```

Change both config loads:

```python
cfg = load_config(DEFAULTS, cfg_path, overrides or [])
```

Change `main`:

```python
def main(cfg_path: str, *, overrides: list[str] | None = None) -> None:
```

Use `argparse` for the normal train path and preserve export behavior:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or export an MWM checkpoint.")
    parser.add_argument("config", help="Training YAML config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist override")
    parser.add_argument("--export-from-lightning", metavar="CHECKPOINT")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.export_from_lightning:
        export_lewm_base_adapter_lightning_checkpoint(args.config, args.export_from_lightning, output_dir=args.output_dir)
    else:
        main(args.config, overrides=args.set)
```

- [ ] **Step 7: Run targeted tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_config_cli.py tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_desktop_entrypoints_accept_set_overrides
```

Expected: pass.

- [ ] **Step 8: Run syntax checks**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m py_compile collect_mwm_data.py train_mwm.py eval_mwm.py benchmark_mwm.py mwm/config_cli.py
```

Expected: exit 0.

- [ ] **Step 9: Commit**

```bash
git add collect_mwm_data.py train_mwm.py eval_mwm.py benchmark_mwm.py tests/test_mwm_repo_hygiene.py
git commit -m "feat: support local config overrides"
```

---

### Task 3: Local Smoke Configs

**Files:**
- Create: `configs/local/collect_pusht_smoke.yaml`
- Create: `configs/local/eval_pusht_smoke.yaml`
- Create: `configs/local/benchmark_pusht_smoke.yaml`
- Create: `configs/local/train_pusht_cpu_smoke.yaml`
- Test: `tests/test_mwm_repo_hygiene.py`

- [ ] **Step 1: Write failing test for local config contract**

Add this test to `tests/test_mwm_repo_hygiene.py`:

```python
    def test_local_desktop_configs_are_small_and_cpu_safe(self) -> None:
        local_dir = ROOT / "configs" / "local"
        expected = {
            "collect_pusht_smoke.yaml",
            "eval_pusht_smoke.yaml",
            "benchmark_pusht_smoke.yaml",
            "train_pusht_cpu_smoke.yaml",
        }
        self.assertEqual({path.name for path in local_dir.glob("*.yaml")}, expected)

        train_cfg = yaml.safe_load((local_dir / "train_pusht_cpu_smoke.yaml").read_text(encoding="utf-8"))
        self.assertTrue(train_cfg["train"]["no_cuda"])
        self.assertEqual(train_cfg["train"]["cpu_devices"], 1)
        self.assertLessEqual(train_cfg["train"]["batch_size"], 2)
        self.assertLessEqual(train_cfg["schedule"]["max_epochs"], 1)
        self.assertLessEqual(float(train_cfg["train"]["limit_train_batches"]), 1.0)

        eval_cfg = yaml.safe_load((local_dir / "eval_pusht_smoke.yaml").read_text(encoding="utf-8"))
        self.assertEqual(eval_cfg["device"], "cpu")
        self.assertLessEqual(eval_cfg["eval"]["episodes"], 2)
        self.assertEqual(eval_cfg["eval"]["num_envs"], 1)
        self.assertFalse(eval_cfg["eval"]["save_video"])
        self.assertLessEqual(eval_cfg["planner"]["pop_size"], 16)
        self.assertLessEqual(eval_cfg["planner"]["n_iter"], 2)

        bench_cfg = yaml.safe_load((local_dir / "benchmark_pusht_smoke.yaml").read_text(encoding="utf-8"))
        self.assertEqual(bench_cfg["env_id"], "swm/PushT-v1")
        self.assertEqual(bench_cfg["eval_config"], "configs/local/eval_pusht_smoke.yaml")
        self.assertEqual(bench_cfg["manifest"]["group"], "local_pusht_smoke")
        self.assertEqual(len(bench_cfg["runs"]), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_local_desktop_configs_are_small_and_cpu_safe
```

Expected: fail because `configs/local` does not exist.

- [ ] **Step 3: Create local collection config**

Create `configs/local/collect_pusht_smoke.yaml`:

```yaml
env_id: swm/PushT-v1
image_shape: 96
max_episode_steps: 50
num_envs: 1
episodes: 4
seed: 7
output_path: data/local/pusht_smoke.lance
format: lance
goal_conditioned: true
env_kwargs: {}
policy:
  import_path: null
restore:
  import_path: null
```

- [ ] **Step 4: Create local eval config**

Create `configs/local/eval_pusht_smoke.yaml`:

```yaml
checkpoint:
  run_dir: checkpoints_mwm/upstream_lewm_pusht
  epoch: null
env_id: swm/PushT-v1
data:
  path: data/upstream/pusht_expert_train.lance
  format: lance
  split_ratio: 0.9
  pixels_key: pixels
  action_key: action
  keys_to_cache: [action, proprio, state]
  action_preprocessing: standard_scaler
eval:
  episodes: 2
  goal_offset: 25
  seed: 42
  budget: 8
  num_envs: 1
  output_path: rollouts/local_pusht_smoke/eval.json
  manifest_path: null
  sampling: stable_worldmodel
  write_manifest_path: rollouts/manifests/local_pusht_smoke.json
  save_video: false
  video_path: rollouts/local_pusht_smoke/videos
env:
  max_episode_steps: 100
  goal_conditioned: true
  kwargs: {}
restore:
  import_path: null
planner:
  horizon: 5
  receding_horizon: 5
  action_block: 5
  batch_size: 1
  pop_size: 16
  topk: 4
  elite_frac: 0.25
  n_iter: 2
  init_std: 1.0
  seed: 42
  warm_start: true
  clamp_actions: false
  std_unbiased: true
  scheduler:
    policy: fixed
    level: finest
    rollout_level:
      mode: fixed
      level: base
device: cpu
```

- [ ] **Step 5: Create local benchmark config**

Create `configs/local/benchmark_pusht_smoke.yaml`:

```yaml
output_dir: rollouts/local_benchmark_pusht_smoke
title: Local PushT Smoke Benchmark
env_id: swm/PushT-v1
seed: 42
eval_config: configs/local/eval_pusht_smoke.yaml
manifest:
  group: local_pusht_smoke
  path: rollouts/manifests/local_pusht_smoke.json
runs:
  - name: upstream_smoke
    role: upstream_lewm_converted
    checkpoint: checkpoints_mwm/upstream_lewm_pusht
```

- [ ] **Step 6: Create optional local train config**

Create `configs/local/train_pusht_cpu_smoke.yaml`:

```yaml
seed: 7
env_id: swm/PushT-v1
base:
  family: lewm
  checkpoint: models--quentinll--lewm-pusht
mwm:
  component_policy:
    shared: [latent_producer]
    per_level: [transition]
    reconstructor: []
  loss_terms:
    regularizers: shared_latent
    reconstructor_detach_encoder: true
    reconstructor_contributes_to_encoder_loss: false
data:
  path: data/upstream/pusht_expert_train.lance
  format: lance
  split_ratio: 0.9
  pixels_key: pixels
  action_key: action
  frameskip: 5
  keys_to_load: [pixels, action, proprio, state]
  keys_to_cache: [action, proprio, state]
model:
  D: 192
  K: [192]
  action_dim: auto
  action_block: 5
  image_shape: auto
  history_size: 3
  num_preds: 1
train:
  batch_size: 1
  horizon: 4
  num_workers: 0
  drop_last: true
  prefetch_factor: null
  pin_memory: false
  no_cuda: true
  cpu_devices: 1
  checkpoint_dir: checkpoints_mwm
  run_name: local_pusht_cpu_smoke
  backend: stable_worldmodel_lewm
  timestamp_run_dir: false
  clean_trainer_root: true
  limit_train_batches: 1
  limit_val_batches: 1
  checkpoint_every_n_train_steps: 0
  gradient_clip_val: 1.0
  matmul_precision: high
  slurm_auto_requeue: false
optim:
  lr: 0.00005
  weight_decay: 0.001
loss:
  rollout_weight: 1.0
  recon_weight: 0.0
  sigreg_weight: 0.0
  history_size: 3
  num_preds: 1
schedule:
  max_epochs: 1
```

- [ ] **Step 7: Run tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_local_desktop_configs_are_small_and_cpu_safe
```

Expected: pass.

- [ ] **Step 8: Run static benchmark config check without checkpoints**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python - <<'PY'
from verify_mwm_benchmark import verify_benchmark_static
print(verify_benchmark_static("configs/local/benchmark_pusht_smoke.yaml", check_checkpoints=False))
PY
```

Expected: report with `runs: 1`, `env_id: swm/PushT-v1`, and manifest group `local_pusht_smoke`.

- [ ] **Step 9: Commit**

```bash
git add configs/local tests/test_mwm_repo_hygiene.py
git commit -m "feat: add local smoke configs"
```

---

### Task 4: Local Scripts With Preflight Checks

**Files:**
- Create: `scripts/local_verify.sh`
- Create: `scripts/local_benchmark_smoke.sh`
- Create: `scripts/local_train_smoke.sh`
- Test: `tests/test_mwm_repo_hygiene.py`

- [ ] **Step 1: Write failing hygiene test for local scripts**

Add this test to `tests/test_mwm_repo_hygiene.py`:

```python
    def test_local_scripts_are_not_slurm_gated_or_parcc_path_bound(self) -> None:
        scripts = [
            ROOT / "scripts" / "local_verify.sh",
            ROOT / "scripts" / "local_benchmark_smoke.sh",
            ROOT / "scripts" / "local_train_smoke.sh",
        ]
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            self.assertNotIn("SLURM_JOB_ID", text, script)
            self.assertNotIn("/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python", text, script)
            self.assertIn('${MWM_PYTHON:-python}', text, script)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_local_scripts_are_not_slurm_gated_or_parcc_path_bound
```

Expected: fail because local scripts do not exist.

- [ ] **Step 3: Create local verify script**

Create `scripts/local_verify.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-python}"

cd "$ROOT"

"$PY" -m py_compile $(rg --files -g '*.py')
"$PY" -m pytest -q
"$PY" verify_mwm_benchmark.py configs/local/benchmark_pusht_smoke.yaml --static-only --no-checkpoints
```

- [ ] **Step 4: Create local benchmark smoke script**

Create `scripts/local_benchmark_smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-python}"

cd "$ROOT"

if [[ ! -d data/upstream/pusht_expert_train.lance ]]; then
  echo "Missing data/upstream/pusht_expert_train.lance. Run prepare_upstream_lewm_data.py or copy the prepared Lance dataset." >&2
  exit 2
fi
if [[ ! -f checkpoints_mwm/upstream_lewm_pusht/world_metadata.json ]]; then
  echo "Missing checkpoints_mwm/upstream_lewm_pusht. Run prepare_upstream_lewm.py or copy the prepared checkpoint." >&2
  exit 2
fi

"$PY" benchmark_mwm.py configs/local/benchmark_pusht_smoke.yaml
"$PY" verify_mwm_benchmark.py configs/local/benchmark_pusht_smoke.yaml
```

- [ ] **Step 5: Create local train smoke script**

Create `scripts/local_train_smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MWM_PYTHON:-python}"

cd "$ROOT"

if [[ "${RUN_CPU_TRAIN_SMOKE:-0}" != "1" ]]; then
  echo "CPU train smoke can be slow. Re-run with RUN_CPU_TRAIN_SMOKE=1 to opt in." >&2
  exit 2
fi

"$PY" train_mwm.py configs/local/train_pusht_cpu_smoke.yaml
```

- [ ] **Step 6: Make scripts executable**

Run:

```bash
chmod +x scripts/local_verify.sh scripts/local_benchmark_smoke.sh scripts/local_train_smoke.sh
```

- [ ] **Step 7: Run script syntax and hygiene tests**

Run:

```bash
bash -n scripts/local_verify.sh scripts/local_benchmark_smoke.sh scripts/local_train_smoke.sh
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_local_scripts_are_not_slurm_gated_or_parcc_path_bound
```

Expected: both pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/local_verify.sh scripts/local_benchmark_smoke.sh scripts/local_train_smoke.sh tests/test_mwm_repo_hygiene.py
git commit -m "feat: add local workflow scripts"
```

---

### Task 5: Static Verifier `--no-checkpoints`

**Files:**
- Modify: `verify_mwm_benchmark.py`
- Test: `tests/test_mwm_artifacts.py`

- [ ] **Step 1: Write failing test for CLI-accessible no-checkpoints static verification**

Add this test to `tests/test_mwm_artifacts.py`:

```python
    def test_benchmark_static_cli_can_skip_checkpoint_contracts(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "verify_mwm_benchmark.py",
                "configs/local/benchmark_pusht_smoke.yaml",
                "--static-only",
                "--no-checkpoints",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"check_checkpoints": false', result.stdout.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_artifacts.py::MWMArtifactTests::test_benchmark_static_cli_can_skip_checkpoint_contracts
```

Expected: fail because `--no-checkpoints` is not accepted.

- [ ] **Step 3: Patch verifier CLI**

Change `main` signature:

```python
def main(cfg_path: str, *, static_only: bool = False, roles: Any = None, check_checkpoints: bool = True) -> None:
    report = (
        verify_benchmark_static(cfg_path, roles=roles, check_checkpoints=check_checkpoints)
        if static_only
        else verify_benchmark_output(cfg_path, roles=roles)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
```

Add parser option:

```python
parser.add_argument("--no-checkpoints", action="store_true", help="Skip checkpoint contract checks in --static-only mode")
```

Call:

```python
main(
    args.config,
    static_only=args.static_only,
    roles=args.roles,
    check_checkpoints=not args.no_checkpoints,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_artifacts.py::MWMArtifactTests::test_benchmark_static_cli_can_skip_checkpoint_contracts
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add verify_mwm_benchmark.py tests/test_mwm_artifacts.py
git commit -m "feat: allow checkpoint-free static benchmark checks"
```

---

### Task 6: Desktop Documentation

**Files:**
- Modify: `README.md`
- Modify: `REVIEW_GUIDE.md`

- [ ] **Step 1: Write failing doc hygiene test**

Add this test to `tests/test_mwm_repo_hygiene.py`:

```python
    def test_docs_describe_local_and_slurm_workflows_separately(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        review = (ROOT / "REVIEW_GUIDE.md").read_text(encoding="utf-8")
        for text in (readme, review):
            self.assertIn("Local Desktop Workflow", text)
            self.assertIn("Slurm", text)
            self.assertIn("scripts/local_verify.sh", text)
            self.assertIn("scripts/local_benchmark_smoke.sh", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_docs_describe_local_and_slurm_workflows_separately
```

Expected: fail because docs do not contain a local workflow section.

- [ ] **Step 3: Patch `README.md`**

Add after Quick Start:

```markdown
## Local Desktop Workflow

Local machines are supported for syntax checks, tests, static benchmark validation, and tiny smoke runs. Full paper-scale training and benchmark runs remain GPU-oriented and should use the Slurm scripts on PARCC/Betty.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
scripts/local_verify.sh
```

For a tiny local benchmark after preparing or copying `data/upstream/pusht_expert_train.lance` and `checkpoints_mwm/upstream_lewm_pusht`:

```bash
scripts/local_benchmark_smoke.sh
```

Optional CPU training smoke is deliberately opt-in because it can be slow:

```bash
RUN_CPU_TRAIN_SMOKE=1 scripts/local_train_smoke.sh
```

Use `MWM_PYTHON=/path/to/python` if your Python is not named `python`.
```

- [ ] **Step 4: Patch `REVIEW_GUIDE.md`**

Add under Validation Commands:

```markdown
### Local Desktop Workflow

Use local scripts when reviewing on a desktop or laptop without Slurm:

```bash
scripts/local_verify.sh
scripts/local_benchmark_smoke.sh
```

These scripts do not require `SLURM_JOB_ID` and default to `${MWM_PYTHON:-python}`. They are smoke workflows only; do not treat CPU smoke numbers as paper-scale benchmark evidence.
```

- [ ] **Step 5: Run doc test**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_docs_describe_local_and_slurm_workflows_separately
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add README.md REVIEW_GUIDE.md tests/test_mwm_repo_hygiene.py
git commit -m "docs: document local desktop workflow"
```

---

### Task 7: Final Verification

**Files:**
- No new source files.

- [ ] **Step 1: Run full tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run Python compile**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m py_compile $(rg --files -g '*.py')
```

Expected: exit 0.

- [ ] **Step 3: Run shell syntax**

Run:

```bash
bash -n scripts/*.sh scripts/*.sbatch
```

Expected: exit 0.

- [ ] **Step 4: Run static benchmark checks**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/local/benchmark_pusht_smoke.yaml --static-only --no-checkpoints
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/benchmark/scheduled_pusht.yaml --static-only --roles upstream_lewm_converted mwm_scheduled
```

Expected: both commands print JSON reports and exit 0.

- [ ] **Step 5: Run local verify script**

Run:

```bash
MWM_PYTHON=/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python scripts/local_verify.sh
```

Expected: exits 0.

- [ ] **Step 6: Confirm git state**

Run:

```bash
git status -sb
```

Expected: no unstaged or uncommitted tracked changes.

---

## Self-Review

- Spec coverage: This plan adds local config overrides, local smoke configs, local scripts, documentation, and tests while preserving Slurm guards for cluster scripts.
- Placeholder scan: No task uses unresolved placeholder wording or undefined function names. Code snippets define the helper and exact script contents.
- Type consistency: `load_config(defaults, cfg_path, overrides)` is used consistently across tasks. CLI overrides are `list[str] | None`. Local static verification relies on `--no-checkpoints`, implemented before `scripts/local_verify.sh` is expected to pass.
