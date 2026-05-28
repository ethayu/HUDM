# Paper-Parity Investigation, 2026-05-28

This note records the current state of the Le-WM paper-parity gate for the
base-adaptive MWM implementation. It is intentionally not a pass artifact: the
strict PushT paper target remains unresolved.

## Target Provenance

- Le-WM PushT paper target: 96.0% success. In Appendix Table 5 this is reported
  as 96.0 +/- 2.83 over three training seeds on the 50-trajectory PushT setup.
- Le-WM TwoRoom paper target: 87.0% success, from the Figure 6 reported value.
- Repository verifier tolerance: 1.0 percentage point for upstream paper parity.
- Fresh single-level `K=[D]` tolerance: within 5.0 percentage points of upstream
  Le-WM on the same evaluator/manifest.

## Runs

- Slurm job `6169551` ran `scripts/slurm_mwm_paper_parity.sbatch` after locking
  CEM `planner.batch_size: 1` to match the upstream CEM chunking default.
- Slurm job `6169560` ran the official Le-WM `eval.py` path through
  `scripts/slurm_lewm_official_pusht_eval.sbatch`.

## Current Results

- `rollouts/mwm_paper_reference/summary.json`:
  - PushT converted upstream Le-WM: 98.0%.
  - PushT Stable-WM reference path: 98.0%.
  - TwoRoom converted upstream Le-WM: 86.0%.
  - TwoRoom Stable-WM reference path: 86.0%.
- `rollouts/lewm_official_reference/pusht_eval_6169560.txt`:
  - PushT official Le-WM `eval.py`: 92.0%.

TwoRoom is within the 87.0 +/- 1.0 pp gate. PushT is not: both the MWM
converted path and Stable-WM reference path are 2.0 pp above the paper mean, and
the raw official evaluator is 4.0 pp below the paper mean.

## Inputs Checked

- Official Le-WM repo used for the raw evaluator:
  `/vast/projects/dineshj/lab/ethanyu/tmp/le-wm-official`, commit
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Raw Hugging Face PushT weights:
  `/vast/projects/dineshj/lab/ethanyu/tmp/lewm-hf-pusht/weights.pt`,
  SHA256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`.
- Raw Hugging Face PushT config:
  `/vast/projects/dineshj/lab/ethanyu/cache/huggingface/hub/models--quentinll--lewm-pusht/blobs/2e534ee5c6286721c0ef967e24bddbfde482abf9`.
- Raw HDF5 dataset:
  `data/upstream/pusht_expert_train.h5`.

The saved upstream object in
`checkpoints_mwm/upstream_sources/upstream_lewm_pusht_object.pt` has the same
303 tensor keys and bit-identical tensors as the raw Hugging Face weights
(maximum absolute tensor difference 0.0).

PushT numeric dataset columns match between raw HDF5 and the Stable-WM Lance
dataset: `action`, `proprio`, and `state` have maximum absolute difference 0.0
on full-column and sampled-row checks. Seed-42 sampling matches the official
50-row manifest.

Decoded Lance image columns differ from raw HDF5 pixels because the Stable-WM
Lance writer stores JPEG-encoded image columns. On sampled PushT chunks, decoded
Lance-vs-HDF5 pixel differences had maximum uint8 differences around 20-23 and
mean differences around 0.10-0.12.

## Interpretation

The current evidence does not prove a bug in the MWM evaluator alone. The
Stable-WM reference path agrees with the converted MWM evaluator at 98.0%, while
the raw official Le-WM evaluator gives 92.0% on the same seed-42 protocol. The
paper's 96.0% PushT target is a three-training-seed mean, not a guaranteed score
for the released checkpoint on a single seed-42 evaluation.

Do not mark the full implementation complete until the PushT target mismatch is
resolved by a corrected checkpoint/data/protocol choice, an accepted target
interpretation, or an explicit user decision to change the gate semantics.
