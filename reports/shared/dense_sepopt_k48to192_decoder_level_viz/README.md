# Dense sepopt K48-192 checkpoints -- per-level decoder visualizations

What each matryoshka level of the dense sepopt checkpoints actually "sees",
decoded back to pixels. Every column is one level, K=[48,72,96,120,144,168,192],
decoded with that level's own decoder; the first column is the ground-truth frame.

Checkpoints (all epoch 9, K=[48,72,96,120,144,168,192]):

| env | checkpoint | history_size |
|---|---|---|
| OGB-Cube | `checkpoints_dense_k48to192_sepopt_20260917/checkpoints_mwm/mwm_paper10_ogb_cube_k48_72_96_120_144_168_192_sepopt_actckpt_20260917` | 3 |
| PushT | `checkpoints_mwm/mwm_paper10_pusht_k48_72_96_120_144_168_192_sepopt_actckpt_20260917` | 3 |
| Reacher | `checkpoints_mwm/mwm_paper10_reacher_k48_72_96_120_144_168_192_sepopt_actckpt_20260917` | 3 |
| TwoRoom | `checkpoints_mwm/mwm_paper10_tworoom_k48_72_96_120_144_168_192_sepopt_20260917` | 1 |

Frames come from the goal25 exact-seed42 n200 manifests
(`configs/manifest/data/release20260728/<env>_goal25_exact_seed42_n200.json`).

## Files (`<env>` in `ogb_cube`, `pusht`, `reacher`, `tworoom`)

**Encoder -> decoder reconstructions** (`scripts/research/visualize_decoder_levels.py`):
each real frame is encoded once by the shared encoder, then decoded at every level.

| file | rows |
|---|---|
| `<env>_k48to192_sepopt_start.png` | start frames of manifest pairs 0-3 |
| `<env>_k48to192_sepopt_start_p2.png` | start frames of pairs 4-9 |
| `<env>_k48to192_sepopt_start_p3.png` | start frames of pairs 10-15 |
| `<env>_k48to192_sepopt_goal_p1.png` | goal frames of pairs 0-5 |
| `<env>_k48to192_sepopt_rollout_pair{0,1}.png` | one episode, every 5 env steps from the pair's start (t=+0) to its goal (t=+25); every frame encoded from the real image |

**Dynamics rollouts, decoded** (`scripts/research/visualize_decoder_rollout.py`):
each level's own transition model is stepped with the recorded dataset actions
(z-scored per base dim, 5 raw actions flattened per model step exactly as in
training), and each predicted embedding is decoded with that level's decoder.
One row = one model step = 5 env steps, t=+0 (start) to t=+25 (goal).

| file | mode |
|---|---|
| `<env>_k48to192_sepopt_dynrollout_ar_pair{0,1}.png` | autoregressive: only t=0 is encoded, predictions are fed back (open-loop, errors compound) |
| `<env>_k48to192_sepopt_dynrollout_tf_pair{0,1}.png` | teacher-forced: each t+1 predicted from the encoded real frame(s) up to t, within the checkpoint's history window (one-step error only) |

The per-panel "latent mse" label is the MSE between the predicted and the encoded
K-prefix. It is **not comparable across K**: the low-K prefixes vary very little
across frames (TwoRoom K=48 reads ~0.000 while its decoded image is wrong).

## Observations

- **Decoder quality (encoded real frames).** Low K loses the small,
  control-relevant object first; the background is always right. The
  reconstructions become sharp at roughly K~72 (TwoRoom), K~96 (Reacher) and
  K~144+ (PushT). In OGB-Cube the purple gripper stays blurry even at K=192.
  One PushT frame (`pusht_..._start_p2.png`, row 2, ep 6650) is reconstructed
  wrong at every K, which points at the shared latent rather than the decoders.
- **TwoRoom K=48 doorway ghost.** The K=48 decoder often paints a faint agent
  blob at the doorway in addition to (or instead of) the true agent. In the
  dynamics rollouts, both K=48 and K=72 collapse completely onto that doorway
  blob from the first predicted step, **in both autoregressive and
  teacher-forced mode**, i.e. those levels' dynamics predict "agent at the door"
  even when given a real encoded frame. K>=96 track the agent in both modes.
- **Reacher dynamics.** Autoregressive K=48-96 smear into ghost arms after the
  first step; K>=120 track the ground truth for all 25 steps. The dynamics
  cut-off (~K=120) is above the decoder cut-off (~K=96).
- **PushT dynamics.** K=48-96 dissolve into a gray smear by t=+10-20; K=120/144
  follow the motion but drift (displaced agent, distorted T by t=+25); only
  K=168/192 stay sharp through the goal.
- **OGB-Cube dynamics.** All levels keep the arm pose and cube roughly right
  over 25 steps; differences between levels are small. The gripper blurs
  further with time at every K.

## Regenerating

Run from the repo root. The shared `mwm` conda env can fail on `from PIL import
Image` for some users; any env with torch, lance, PIL and matplotlib works with
`PYTHONPATH=.`. CPU is fine (~1.5 min per image).

```bash
CK=checkpoints_mwm/mwm_paper10_pusht_k48_72_96_120_144_168_192_sepopt_actckpt_20260917
M=configs/manifest/data/release20260728/pusht_goal25_exact_seed42_n200.json
OUT=reports/shared/dense_sepopt_k48to192_decoder_level_viz

# encoder -> decoder, one frame per manifest pair
PYTHONPATH=. python scripts/research/visualize_decoder_levels.py --checkpoint $CK --manifest $M \
  --num-samples 6 --start-index 4 --row-field start_row --out $OUT/pusht_k48to192_sepopt_start_p2.png
# encoder -> decoder, successive timesteps of one episode
PYTHONPATH=. python scripts/research/visualize_decoder_levels.py --checkpoint $CK --manifest $M \
  --rollout-pair 0 --num-timesteps 6 --stride 5 --out $OUT/pusht_k48to192_sepopt_rollout_pair0.png
# dynamics rollout, decoded
PYTHONPATH=. python scripts/research/visualize_decoder_rollout.py --checkpoint $CK --manifest $M \
  --pair 0 --num-steps 5 --mode autoregressive --out $OUT/pusht_k48to192_sepopt_dynrollout_ar_pair0.png
```
