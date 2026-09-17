"""Re-render zarr expert states through SWM PushT to get model-compatible images."""
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
OUT_PATH = "data/local/pusht_expert_swm2_224.lance"
IMAGE_SIZE = 224
GOAL_OFFSET = 25  # used only for episode pairing (goal image lookahead)


def encode_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def main():
    out = Path(OUT_PATH)
    if out.exists():
        raise FileExistsError(f"{out} already exists — delete it first.")
    out.parent.mkdir(parents=True, exist_ok=True)

    z = zarr.open(ZARR_PATH, "r")
    states_zarr = z["data/state"][:]    # (N, 5): agent_pos(2), block_pos(2), block_angle(1)
    actions = z["data/action"][:]       # (N, 2)
    ep_ends = z["meta/episode_ends"][:]

    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_episodes = len(ep_ends)

    # Build SWM world (renders at 224x224)
    from mwm.swm.envs import make_swm_world
    world = make_swm_world(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        max_episode_steps=200,
        goal_conditioned=True,
    )
    env = world.envs.envs[0].unwrapped

    print(f"Re-rendering {n_episodes} episodes, {len(states_zarr)} total steps …")

    rows = []
    total_steps = 0

    for ep_idx in range(n_episodes):
        start = ep_starts[ep_idx]
        end = ep_ends[ep_idx]
        ep_len = end - start

        # Episode goal = final state of the expert episode (block at ~256,256,pi/4)
        raw_final = states_zarr[end - 1].astype(np.float64)
        goal_state7 = np.concatenate([raw_final, [0.0, 0.0]])

        # Render goal image once per episode (SWM render of final state with goal overlay)
        obs_g, info_g = env.reset(options={"state": goal_state7, "goal_state": goal_state7})
        goal_arr = env.render()   # block at final position, goal overlay = same
        goal_bytes_ep = encode_jpeg(Image.fromarray(goal_arr).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

        for step in range(ep_len):
            abs_step = start + step
            raw = states_zarr[abs_step].astype(np.float64)    # 5D
            # 7D state (add zero velocity)
            state7 = np.concatenate([raw, [0.0, 0.0]])

            # Render current obs with goal overlay = episode's final state
            obs, info = env.reset(options={"state": state7, "goal_state": goal_state7})
            # render() gives the current-state image (with goal overlay showing final state)
            pix_arr = env.render()   # H x W x 3 uint8
            goal_arr = goal_bytes_ep  # reuse episode goal bytes (already encoded)

            pix_img = Image.fromarray(pix_arr).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            pix_bytes = encode_jpeg(pix_img)

            # goal_arr is already encoded bytes (episode-level goal image)
            goal_bytes = goal_arr

            rows.append({
                "episode_idx": ep_idx,
                "step_idx": step,
                "pos_agent":     state7[:2].astype(np.float32).tolist(),
                "vel_agent":     [0.0, 0.0],
                "block_pose":    state7[2:5].astype(np.float32).tolist(),
                "goal_pose":     goal_state7[2:5].astype(np.float32).tolist(),
                "goal_state":    goal_state7.astype(np.float32).tolist(),
                "goal_proprio":  goal_state7[[0,1,5,6]].astype(np.float32).tolist(),
                "n_contacts":    [0.0],
                "goal":          goal_bytes,
                "render_time":   [0.0],
                "pixels":        pix_bytes,
                "proprio":       state7[[0,1,5,6]].astype(np.float32).tolist(),
                "state":         state7.astype(np.float32).tolist(),
                "reward":        [0.0],
                "terminated":    [0.0],
                "truncated":     [float(step == ep_len - 1)],
                "action":        actions[abs_step].astype(np.float32).tolist(),
                "id":            [float(total_steps)],
            })
            total_steps += 1

        if (ep_idx + 1) % 20 == 0:
            print(f"  {ep_idx + 1}/{n_episodes} episodes ({total_steps} steps) …")

    world.close()

    print(f"Writing {total_steps} rows to Lance …")

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

    metadata = {
        "format": "swm_lance",
        "env_id": "swm/PushT-v1",
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE],
        "max_episode_steps": int(max(ep_ends - ep_starts)),
        "num_envs": 1,
        "episodes": int(n_episodes),
        "seed": 0,
        "restore_spec": "pusht_state_goal_state",
        "action_dim": 2,
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "source": "pusht_cchi_v7_replay_swm_rendered_v2",
        "dataset": {"pixels_key": "pixels", "action_key": "action"},
    }
    out.with_suffix(".lance.metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Done. {total_steps} steps ({n_episodes} episodes) → {out}")


if __name__ == "__main__":
    main()
