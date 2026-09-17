"""Read-only diagnostic (no training, no checkpoint/data mutation).

Generalizes analyze_tworoom_mwm_dense_tworoom_rollout_error_by_level.py from
just K=96 vs K=192 to ALL declared levels of checkpoints_mwm/mwm_dense_tworoom
(K=[6,12,48,96,144,192]): does rollout-prediction error compound faster over
multiple future steps at higher K, across the whole fidelity ladder?

Same methodology: real pixels + real actions (no CEM, no synthetic actions),
autoregressive model.rollout_at_level(...) per level, ground truth = an
independent model.encode(...) of the real future frame sliced to k dims.
See the paired K96-vs-K192-only script's docstring for data-loading and
normalization details (identical here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/aurora/HUDM-mwm-ethan")
CHECKPOINT_DIR = REPO_ROOT / "checkpoints_mwm/mwm_dense_tworoom"
LANCE_PATH = REPO_ROOT / "data/upstream/tworoom.lance"

N_FUTURE_STEPS = 15  # rollout steps t=1..15
NUM_EPISODES = 30
FRAMESKIP = 5  # == action_block, from configs/train/mwm_lewm_dense_tworoom.yaml
SEED = 0

sys.path.insert(0, str(REPO_ROOT))

from mwm.checkpoint_io import load_world_model_from_checkpoint  # noqa: E402
from mwm.data.transforms import column_normalizer  # noqa: E402


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
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

    from stable_worldmodel.data import load_dataset

    num_steps = N_FUTURE_STEPS + 1
    raw_ds = load_dataset(
        str(LANCE_PATH),
        format="lance",
        frameskip=FRAMESKIP,
        num_steps=num_steps,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    print(f"Dataset rows/clips available: {len(raw_ds)}; span={raw_ds.span} raw steps per sample")

    raw_ds.transform = column_normalizer(raw_ds, "action", "action")

    rng = np.random.default_rng(SEED)
    valid_indices = [i for i, (ep, start) in enumerate(raw_ds.clip_indices) if start == 0]
    print(f"Episodes with a valid full-length clip starting at step 0: {len(valid_indices)}")
    chosen = rng.choice(valid_indices, size=min(NUM_EPISODES, len(valid_indices)), replace=False)

    samples = [raw_ds[int(i)] for i in chosen]
    print(f"Loaded {len(samples)} episode segments, each {num_steps} pixel frames / {num_steps} action blocks")

    pixels = torch.stack([s["pixels"] for s in samples]).to(device)
    actions = torch.stack([s["action"] for s in samples]).to(device)
    B = pixels.shape[0]
    print(f"pixels shape={tuple(pixels.shape)} dtype={pixels.dtype}; actions shape={tuple(actions.shape)}")

    action_sequence = actions[:, :N_FUTURE_STEPS, :].unsqueeze(1)

    results = {}
    with torch.no_grad():
        for level_idx, k in enumerate(model.K):
            infos = {"pixels": pixels[:, :1].unsqueeze(1)}
            out = model.rollout_at_level(infos, action_sequence, level_idx)
            pred_emb = out["predicted_emb"]
            pred_emb = pred_emb[:, 0]
            assert pred_emb.shape[1] == N_FUTURE_STEPS + 1, pred_emb.shape

            future_pixels = pixels[:, 1 : N_FUTURE_STEPS + 1]
            real_emb_full = model.encode(future_pixels, already_preprocessed=False)
            real_emb = real_emb_full[..., :k]

            pred_future = pred_emb[:, 1:]
            mse_per_t = ((pred_future - real_emb) ** 2).mean(dim=(0, 2))
            results[k] = mse_per_t.cpu().numpy()
            print(f"K={k:3d} (level_idx={level_idx}) MSE(t): {np.array2string(results[k], precision=6)}")

    print("\n=== Summary: t, then MSE per K, then relative growth (MSE(t)/MSE(1)) per K ===")
    k_values = list(model.K)
    rel = {k: results[k] / results[k][0] for k in k_values}
    rows = []
    header = "t   " + "".join(f"K={k:<9d}" for k in k_values)
    print(header)
    for t in range(N_FUTURE_STEPS):
        row = {"t": t + 1}
        for k in k_values:
            row[f"mse_k{k}"] = float(results[k][t])
            row[f"rel_growth_k{k}"] = float(rel[k][t])
        rows.append(row)
        line = f"t={t+1:2d} " + "".join(f"{results[k][t]:<11.6f}" for k in k_values)
        print(line)

    print("\n=== relative growth (MSE(t)/MSE(1)) per K ===")
    print(header)
    for t in range(N_FUTURE_STEPS):
        line = f"t={t+1:2d} " + "".join(f"{rel[k][t]:<11.3f}" for k in k_values)
        print(line)

    out_path = REPO_ROOT / "analysis_output_tworoom_mwm_dense_tworoom_rollout_error_all_levels.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_episodes": int(B),
                "n_future_steps": N_FUTURE_STEPS,
                "frameskip": FRAMESKIP,
                "seed": SEED,
                "k_values": k_values,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
