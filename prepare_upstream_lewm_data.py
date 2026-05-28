from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from mwm.data.stable_wm import write_dataset_metadata


PUSHT_REPO = "quentinll/lewm-pusht"
PUSHT_FILENAME = "pusht_expert_train.h5.zst"
PUSHT_URL = "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst"
TWOROOM_REPO = "quentinll/lewm-tworooms"
TWOROOM_FILENAME = "tworoom.tar.zst"
TWOROOM_URL = "https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    tmp.replace(dest)


def _download_hf_dataset(repo_id: str, filename: str, dest: Path, fallback_url: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        _download(fallback_url, dest)
        return

    try:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(dest.parent),
            )
        )
    except Exception as exc:
        print(f"Hugging Face download failed ({exc}); falling back to direct URL.")
        _download(fallback_url, dest)
        return

    if path.resolve() != dest.resolve():
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copyfile(path, tmp)
        tmp.replace(dest)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}")


def _write_pusht_metadata(path: Path) -> None:
    write_dataset_metadata(
        path,
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
                "url": PUSHT_URL,
                "format": "hdf5.zst",
                "converted_by": "stable_worldmodel.data.convert",
            },
        },
    )


def _write_tworoom_metadata(path: Path) -> None:
    write_dataset_metadata(
        path,
        {
            "format": "swm_lance",
            "env_id": "swm/TwoRoom-v1",
            "restore_spec": "point_state_goal_state",
            "image_shape": [224, 224],
            "action_dim": 2,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
            "dataset": {"pixels_key": "pixels", "action_key": "action"},
            "source": {
                "url": TWOROOM_URL,
                "format": "hdf5.tar.zst",
                "converted_by": "stable_worldmodel.data.convert",
            },
        },
    )


def _convert_hdf5_to_lance(h5_path: Path, lance_path: Path) -> None:
    from stable_worldmodel.data import convert

    convert(
        str(h5_path.resolve()),
        str(lance_path.resolve()),
        source_format="hdf5",
        dest_format="lance",
        mode="overwrite",
    )


def _extract_hdf5_from_archive(root: Path, archive_path: Path, target_h5: Path) -> Path:
    if target_h5.exists():
        return target_h5
    staging = root / ".extract_tworoom"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    print(f"Extracting {archive_path} -> {staging}")
    _run(["tar", "--use-compress-program=zstd", "-xf", str(archive_path), "-C", str(staging)])
    candidates = sorted(staging.rglob("*.h5"), key=lambda p: (p.name != target_h5.name, len(p.parts), str(p)))
    if not candidates:
        shutil.rmtree(staging)
        raise FileNotFoundError(f"Could not find an HDF5 dataset after extracting {archive_path}")
    source = candidates[0]
    print(f"Moving extracted HDF5 dataset {source} -> {target_h5}")
    shutil.move(str(source), str(target_h5))
    shutil.rmtree(staging)
    return target_h5


def prepare_pusht(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    zst_path = root / "pusht_expert_train.h5.zst"
    h5_path = root / "pusht_expert_train.h5"
    lance_path = root / "pusht_expert_train.lance"
    if not lance_path.exists():
        if not h5_path.exists():
            if not zst_path.exists():
                print(f"Downloading {PUSHT_URL} -> {zst_path}")
                _download_hf_dataset(PUSHT_REPO, PUSHT_FILENAME, zst_path, PUSHT_URL)
            print(f"Decompressing {zst_path} -> {h5_path}")
            _run(["zstd", "-d", "-f", "-o", str(h5_path), str(zst_path)])
        print(f"Converting {h5_path} -> {lance_path}")
        _convert_hdf5_to_lance(h5_path, lance_path)
    _write_pusht_metadata(lance_path)
    return lance_path


def prepare_tworoom(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "tworoom.tar.zst"
    h5_path = root / "tworoom.h5"
    lance_path = root / "tworoom.lance"
    metadata_path = lance_path.with_suffix(lance_path.suffix + ".metadata.json")
    if lance_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = metadata.get("source", {})
        if source.get("converted_by") != "stable_worldmodel.data.convert":
            print(f"Removing legacy TwoRoom Lance dataset {lance_path}")
            shutil.rmtree(lance_path)
    if not lance_path.exists():
        if not archive_path.exists():
            print(f"Downloading {TWOROOM_URL} -> {archive_path}")
            _download_hf_dataset(TWOROOM_REPO, TWOROOM_FILENAME, archive_path, TWOROOM_URL)
        _extract_hdf5_from_archive(root, archive_path, h5_path)
        print(f"Converting {h5_path} -> {lance_path}")
        _convert_hdf5_to_lance(h5_path, lance_path)
    _write_tworoom_metadata(lance_path)
    return lance_path


def main() -> None:
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/upstream")
    out = {
        "pusht": str(prepare_pusht(root)),
        "tworoom": str(prepare_tworoom(root)),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
