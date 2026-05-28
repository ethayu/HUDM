from __future__ import annotations

import hashlib
import importlib.metadata as package_metadata
import json
import subprocess
from pathlib import Path
from typing import Any


def _metadata_sha256(dist: package_metadata.Distribution) -> str:
    h = hashlib.sha256()
    for name in ("METADATA", "RECORD", "direct_url.json"):
        text = dist.read_text(name)
        if text:
            h.update(name.encode("utf-8"))
            h.update(b"\0")
            h.update(text.encode("utf-8"))
            h.update(b"\0")
    return h.hexdigest()


def _package_ref(name: str) -> dict[str, Any] | None:
    try:
        dist = package_metadata.distribution(name)
    except package_metadata.PackageNotFoundError:
        return None
    ref: dict[str, Any] = {"version": dist.version, "sha256": _metadata_sha256(dist)}
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return ref
    try:
        info = json.loads(direct_url)
    except json.JSONDecodeError:
        ref["direct_url"] = direct_url
        return ref
    vcs = info.get("vcs_info", {})
    if "commit_id" in vcs:
        ref["commit_id"] = vcs["commit_id"]
    if "url" in info:
        ref["url"] = info["url"]
    return ref


def _local_repo_ref(root: Path) -> dict[str, Any] | None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines()
    except Exception:
        return None
    ref: dict[str, Any] = {
        "commit_id": commit,
        "dirty": bool(diff or status),
    }
    if diff or status:
        h = hashlib.sha256()
        h.update(diff)
        h.update("\n".join(status).encode("utf-8"))
        ref["diff_sha256"] = h.hexdigest()
    return ref


def dependency_refs(root: str | Path | None = None) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for name in ("stable-worldmodel", "stable-pretraining", "torch"):
        ref = _package_ref(name)
        if ref is not None:
            refs[name] = ref
    local = _local_repo_ref(Path(root or Path(__file__).resolve().parents[1]))
    if local is not None:
        refs["local_repo"] = local
    return refs


__all__ = ["dependency_refs"]
