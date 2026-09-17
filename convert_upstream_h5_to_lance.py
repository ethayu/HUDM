"""Decompress pusht_expert_train.h5.zst and convert to Lance format."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ZST_PATH = Path("data/upstream/pusht_expert_train.h5.zst")
H5_PATH  = Path("data/upstream/pusht_expert_train.h5")
LANCE_PATH = Path("data/upstream/pusht_expert_train.lance")


def main() -> None:
    if not ZST_PATH.exists():
        print(f"ERROR: {ZST_PATH} not found. Download it first.", file=sys.stderr)
        sys.exit(1)

    if LANCE_PATH.exists():
        print(f"Lance dataset already exists at {LANCE_PATH}. Delete it first to re-convert.")
        sys.exit(0)

    # Step 1: decompress .h5.zst → .h5
    if not H5_PATH.exists():
        print(f"Decompressing {ZST_PATH} → {H5_PATH} …")
        result = subprocess.run(
            ["zstd", "-d", str(ZST_PATH), "-o", str(H5_PATH)],
            check=True,
        )
        print(f"Decompressed: {H5_PATH} ({H5_PATH.stat().st_size / 1e9:.1f} GB)")
    else:
        print(f"H5 already decompressed at {H5_PATH}")

    # Step 2: convert H5 → Lance
    print(f"Converting {H5_PATH} → {LANCE_PATH} …")
    from stable_worldmodel.data import convert
    convert(str(H5_PATH), str(LANCE_PATH), source_format="hdf5", dest_format="lance")
    print(f"Converted to {LANCE_PATH}")

    # Step 3: delete the large H5 file to free space
    print(f"Deleting {H5_PATH} …")
    H5_PATH.unlink()
    print("Done. Space freed.")

    # Step 4: write metadata sidecar
    print("Writing metadata …")
    from mwm.data.stable_wm import write_dataset_metadata
    write_dataset_metadata(
        str(LANCE_PATH),
        {
            "format": "swm_lance",
            "env_id": "swm/PushT-v1",
            "restore_spec": "pusht_state_goal_state",
            "image_shape": [224, 224],
            "action_dim": 2,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
            "dataset": {"pixels_key": "pixels", "action_key": "action"},
            "source": {
                "format": "lance",
                "artifact": "pusht_expert_train.lance",
                "standard": "paper_parity",
            },
        },
    )
    print("Metadata written. Upstream dataset ready at", LANCE_PATH)


if __name__ == "__main__":
    main()
