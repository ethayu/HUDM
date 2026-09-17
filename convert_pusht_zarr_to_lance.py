"""Convert pusht_cchi_v7_replay.zarr expert dataset to SWM Lance format at 224x224."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import lance
import zarr
from PIL import Image


ZARR_PATH = "/home/aurora/pusht/pusht_cchi_v7_replay.zarr"
OUT_PATH = "data/local/pusht_expert_224.lance"
IMAGE_SIZE = 224
GOAL_OFFSET = 25
# Fixed PushT goal pose: center of arena (256,256), 45-degree rotation
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4], dtype=np.float32)


def encode_jpeg(arr_hwc_uint8: np.ndarray, size: int) -> bytes:
    img = Image.fromarray(arr_hwc_uint8).resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def float_to_zarr_img(arr: np.ndarray) -> np.ndarray:
    """zarr stores images as float32 in [0,1] — convert to uint8."""
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def main():
    out = Path(OUT_PATH)
    if out.exists():
        raise FileExistsError(f"{out} already exists — delete it first.")
    out.parent.mkdir(parents=True, exist_ok=True)

    z = zarr.open(ZARR_PATH, "r")
    imgs = z["data/img"][:]        # (N, 96, 96, 3) float32 [0,1]
    states = z["data/state"][:]    # (N, 5) float32: agent_pos, block_pos, block_angle
    actions = z["data/action"][:]  # (N, 2)
    ep_ends = z["meta/episode_ends"][:]  # (E,) cumulative end indices

    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_episodes = len(ep_ends)
    print(f"Converting {n_episodes} episodes, {len(imgs)} total steps …")

    rows = []
    total_steps = 0

    for ep_idx in range(n_episodes):
        start = ep_starts[ep_idx]
        end = ep_ends[ep_idx]          # exclusive
        ep_len = end - start

        for step in range(ep_len):
            abs_step = start + step
            goal_abs = min(abs_step + GOAL_OFFSET, end - 1)

            # Images
            pix_bytes = encode_jpeg(float_to_zarr_img(imgs[abs_step]), IMAGE_SIZE)
            goal_bytes = encode_jpeg(float_to_zarr_img(imgs[goal_abs]), IMAGE_SIZE)

            # State: 5D zarr → 7D lance (append two zero velocity dims)
            raw = states[abs_step].astype(np.float32)      # [ax, ay, bx, by, ba]
            state7 = np.concatenate([raw, [0.0, 0.0]])

            raw_goal = states[goal_abs].astype(np.float32)
            goal_state7 = np.concatenate([raw_goal, [0.0, 0.0]])

            rows.append({
                "episode_idx": ep_idx,
                "step_idx": step,
                "pos_agent":     raw[:2].tolist(),
                "vel_agent":     [0.0, 0.0],
                "block_pose":    raw[2:5].tolist(),
                "goal_pose":     GOAL_POSE.tolist(),
                "goal_state":    goal_state7.tolist(),
                "goal_proprio":  goal_state7[[0,1,5,6]].tolist(),
                "n_contacts":    [0.0],
                "goal":          goal_bytes,
                "render_time":   [0.0],
                "pixels":        pix_bytes,
                "proprio":       state7[[0,1,5,6]].tolist(),
                "state":         state7.tolist(),
                "reward":        [0.0],
                "terminated":    [0.0],
                "truncated":     [float(step == ep_len - 1)],
                "action":        actions[abs_step].tolist(),
                "id":            [float(total_steps)],
            })
            total_steps += 1

        if (ep_idx + 1) % 20 == 0:
            print(f"  {ep_idx + 1}/{n_episodes} episodes …")

    print(f"Building schema and writing {total_steps} rows to Lance …")

    def _fsl(n):
        return pa.list_(pa.float32(), n)

    schema = pa.schema([
        pa.field("episode_idx",  pa.int32()),
        pa.field("step_idx",     pa.int32()),
        pa.field("pos_agent",    _fsl(2)),
        pa.field("vel_agent",    _fsl(2)),
        pa.field("block_pose",   _fsl(3)),
        pa.field("goal_pose",    _fsl(3)),
        pa.field("goal_state",   _fsl(7)),
        pa.field("goal_proprio", _fsl(4)),
        pa.field("n_contacts",   _fsl(1)),
        pa.field("goal",         pa.binary()),
        pa.field("render_time",  _fsl(1)),
        pa.field("pixels",       pa.binary()),
        pa.field("proprio",      _fsl(4)),
        pa.field("state",        _fsl(7)),
        pa.field("reward",       _fsl(1)),
        pa.field("terminated",   _fsl(1)),
        pa.field("truncated",    _fsl(1)),
        pa.field("action",       _fsl(2)),
        pa.field("id",           _fsl(1)),
    ])

    # Build pyarrow table
    cols = {k: [] for k in schema.names}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)

    arrays = []
    for field in schema:
        name = field.name
        if field.type == pa.int32():
            arrays.append(pa.array(cols[name], type=pa.int32()))
        elif field.type == pa.binary():
            arrays.append(pa.array(cols[name], type=pa.binary()))
        else:
            n = field.type.list_size
            arrays.append(pa.array(
                [pa.array(v, type=pa.float32()) for v in cols[name]],
                type=_fsl(n),
            ))

    table = pa.Table.from_arrays(arrays, schema=schema)
    lance.write_dataset(table, str(out))

    # Write metadata sidecar
    metadata = {
        "format": "swm_lance",
        "env_id": "swm/PushT-v1",
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "max_episode_steps": int(max(ep_ends - ep_starts)),
        "num_envs": 1,
        "episodes": int(n_episodes),
        "seed": 0,
        "restore_spec": "pusht_state",
        "action_dim": 2,
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "source": "pusht_cchi_v7_replay",
        "dataset": {"pixels_key": "pixels", "action_key": "action"},
    }
    out.with_suffix(".lance.metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Done. Wrote {total_steps} steps ({n_episodes} episodes) to {out}")


if __name__ == "__main__":
    main()
