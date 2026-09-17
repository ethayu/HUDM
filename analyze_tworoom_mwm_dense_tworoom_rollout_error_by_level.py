"""Read-only diagnostic (no training, no checkpoint/data mutation).

Question: for the jointly-trained ("dense") multi-K TwoRoom checkpoint, does the
K=192 tail's rollout-prediction error compound faster over multiple future
steps than K=96's does, on real held-out episodes (real pixels + real actions,
no CEM, no synthetic actions)?

Data path used: the raw stable_worldmodel LanceDataset reader
(`stable_worldmodel.data.load_dataset`, format="lance") pointed at
data/upstream/tworoom.lance, with the SAME frameskip=5 used at training
(configs/train/mwm_lewm_dense_tworoom.yaml: data.frameskip=5, model.action_block=5).
Actions are z-score normalized with mwm.data.transforms.column_normalizer
(the exact class training uses, ZScoreScaler fit via torch mean/std), applied
as a per-sample dataset `transform` so it runs BEFORE the library's internal
`action.reshape(num_steps, -1)` block-concat step -- this ordering matters:
normalization is fit/applied per raw (2-dim) action step, and only afterwards
are `action_block=5` consecutive raw (already-normalized) steps concatenated
into the model's 10-dim block action. This mirrors
mwm.training.stable_wm_transforms.build_stable_wm_adapter_dataset_transform
exactly, minus the pixel ToImage/Resize stage (pixels are already 224x224 and
the task asks for raw pixels into model.encode(..., already_preprocessed=False)).
No saved action-normalization stats exist in the checkpoint's world_metadata.json
(only action_low/high, which are raw env bounds, not z-score stats) or config.json,
so refitting ZScoreScaler on the full dataset is exactly what training itself
does at transform-construction time (not a deviation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/aurora/HUDM-mwm-ethan")
# Different dense checkpoint from the original analysis: K=[6,12,48,96,144,192]
# (May 30, predates the July 28 paper10 release), vs the paper10 checkpoint's
# K=[96,120,144,168,192]. K=96 is level_idx=3 and K=192 is level_idx=5 here.
CHECKPOINT_DIR = REPO_ROOT / "checkpoints_mwm/mwm_dense_tworoom"
LANCE_PATH = REPO_ROOT / "data/upstream/tworoom.lance"

N_FUTURE_STEPS = 15  # rollout steps t=1..15
NUM_EPISODES = 30
FRAMESKIP = 5  # == action_block, from configs/train/mwm_lewm_dense_tworoom.yaml
LEVEL_IDX_K96 = 3
LEVEL_IDX_K192 = 5
SEED = 0

import sys

sys.path.insert(0, str(REPO_ROOT))

from mwm.checkpoint_io import load_world_model_from_checkpoint  # noqa: E402
from mwm.data.transforms import column_normalizer  # noqa: E402


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # nvidia-smi showed GPU0 lightly used by desktop compositor only (no other
    # compute jobs) and GPU1 fully idle; either is fine, use GPU1 to be safe.
    try:
        torch.zeros(1, device="cuda:1")
        return torch.device("cuda:1")
    except Exception:
        return torch.device("cuda:0")


def main() -> None:
    device = pick_device()
    print(f"Using device: {device}")

    model, metadata, _epoch = load_world_model_from_checkpoint(CHECKPOINT_DIR, epoch=None, device=device)
    model.eval()
    print(f"Loaded model K={model.K} D={model.D} supports_arbitrary_k={model.supports_arbitrary_k}")
    assert not model.supports_arbitrary_k, "expected legacy per-level architecture for this checkpoint"
    assert model.K[LEVEL_IDX_K96] == 96 and model.K[LEVEL_IDX_K192] == 192

    from stable_worldmodel.data import load_dataset

    num_steps = N_FUTURE_STEPS + 1  # frame 0 (current) + N future frames
    raw_ds = load_dataset(
        str(LANCE_PATH),
        format="lance",
        frameskip=FRAMESKIP,
        num_steps=num_steps,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    print(f"Dataset rows/clips available: {len(raw_ds)}; span={raw_ds.span} raw steps per sample")

    # Fit the action z-score scaler on the FULL dataset's raw action column,
    # exactly like mwm.training.stable_wm_transforms.build_stable_wm_adapter_dataset_transform
    # does at training time (there are no saved stats to reuse -- see module docstring).
    raw_ds.transform = column_normalizer(raw_ds, "action", "action")

    rng = np.random.default_rng(SEED)
    # Episodes are contiguous blocks of `offsets`/`lengths`; sample starts near
    # the beginning of episodes long enough to hold a full span, for reproducibility.
    valid_indices = [i for i, (ep, start) in enumerate(raw_ds.clip_indices) if start == 0]
    print(f"Episodes with a valid full-length clip starting at step 0: {len(valid_indices)}")
    chosen = rng.choice(valid_indices, size=min(NUM_EPISODES, len(valid_indices)), replace=False)

    samples = [raw_ds[int(i)] for i in chosen]
    print(f"Loaded {len(samples)} episode segments, each {num_steps} pixel frames / {num_steps} action blocks")

    pixels = torch.stack([s["pixels"] for s in samples]).to(device)  # (B, num_steps, C, H, W) uint8
    actions = torch.stack([s["action"] for s in samples]).to(device)  # (B, num_steps, 10) normalized block actions
    B = pixels.shape[0]
    print(f"pixels shape={tuple(pixels.shape)} dtype={pixels.dtype}; actions shape={tuple(actions.shape)}")

    # action_sequence for rollout_at_level must have horizon = history(=1) + n_steps.
    # We want N_FUTURE_STEPS predicted frames -> n_steps = N_FUTURE_STEPS - 1 ->
    # horizon = N_FUTURE_STEPS, i.e. use action blocks 0..N_FUTURE_STEPS-1 (the
    # last loaded action block, index num_steps-1, corresponds to a transition
    # beyond our last real frame and is unused).
    action_sequence = actions[:, :N_FUTURE_STEPS, :].unsqueeze(1)  # (B, samples=1, horizon, action_dim)

    results = {}
    with torch.no_grad():
        for name, level_idx in [("K96", LEVEL_IDX_K96), ("K192", LEVEL_IDX_K192)]:
            k = model.K[level_idx]
            infos = {"pixels": pixels[:, :1].unsqueeze(1)}  # (B, samples=1, history=1, C,H,W)
            out = model.rollout_at_level(infos, action_sequence, level_idx)
            pred_emb = out["predicted_emb"]  # (B, samples=1, history+n_steps+1, k) == (B,1,N_FUTURE_STEPS+1,k)
            pred_emb = pred_emb[:, 0]  # (B, N_FUTURE_STEPS+1, k); index 0 == history frame (real, re-encoded)
            assert pred_emb.shape[1] == N_FUTURE_STEPS + 1, pred_emb.shape

            # Encode REAL future frames independently (ground truth), sliced to k dims.
            future_pixels = pixels[:, 1 : N_FUTURE_STEPS + 1]  # (B, N_FUTURE_STEPS, C,H,W)
            real_emb_full = model.encode(future_pixels, already_preprocessed=False)  # (B, N_FUTURE_STEPS, D)
            real_emb = real_emb_full[..., :k]

            pred_future = pred_emb[:, 1:]  # (B, N_FUTURE_STEPS, k) -- drop the t=0 history slot
            mse_per_t = ((pred_future - real_emb) ** 2).mean(dim=(0, 2))  # (N_FUTURE_STEPS,)
            results[name] = mse_per_t.cpu().numpy()
            print(f"{name} (k={k}) MSE(t): {np.array2string(results[name], precision=6)}")

    print("\n=== Summary table: t, MSE_K96, MSE_K192, MSE_K192/MSE_K96, relK96(t)=MSE(t)/MSE(1), relK192(t) ===")
    mse96 = results["K96"]
    mse192 = results["K192"]
    rel96 = mse96 / mse96[0]
    rel192 = mse192 / mse192[0]
    rows = []
    for t in range(N_FUTURE_STEPS):
        row = {
            "t": t + 1,
            "mse_k96": float(mse96[t]),
            "mse_k192": float(mse192[t]),
            "ratio_192_over_96": float(mse192[t] / mse96[t]),
            "rel_growth_k96": float(rel96[t]),
            "rel_growth_k192": float(rel192[t]),
        }
        rows.append(row)
        print(
            f"t={row['t']:2d}  mse_k96={row['mse_k96']:.6f}  mse_k192={row['mse_k192']:.6f}  "
            f"ratio={row['ratio_192_over_96']:.2f}  rel96={row['rel_growth_k96']:.3f}  rel192={row['rel_growth_k192']:.3f}"
        )

    out_path = REPO_ROOT / "analysis_output_tworoom_mwm_dense_tworoom_rollout_error_by_level.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_episodes": int(B),
                "n_future_steps": N_FUTURE_STEPS,
                "frameskip": FRAMESKIP,
                "seed": SEED,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
