from __future__ import annotations

from pathlib import Path
import re


_SWM_VIDEO_RE = re.compile(r"^(env|episode|episode_remaining)_(\d+)\.mp4$")
_VIDEO_KIND_ORDER = {"env": 0, "episode": 1, "episode_remaining": 2}


def _video_sort_key(path: Path) -> tuple[int, int, str]:
    match = _SWM_VIDEO_RE.match(path.name)
    if match is None:
        return (99, 0, path.name)
    kind, index = match.groups()
    return (_VIDEO_KIND_ORDER[kind], int(index), path.name)


def collect_video_paths(video_dir: str | Path) -> list[Path]:
    """Return canonical SWM-rendered videos in stable env/episode order."""

    directory = Path(video_dir)
    if not directory.is_dir():
        return []
    paths = [path for path in directory.glob("*.mp4") if path.is_file() and _SWM_VIDEO_RE.match(path.name)]
    return sorted(paths, key=_video_sort_key)


__all__ = ["collect_video_paths"]
