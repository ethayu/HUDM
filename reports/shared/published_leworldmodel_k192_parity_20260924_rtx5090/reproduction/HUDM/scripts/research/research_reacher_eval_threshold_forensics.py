from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = (
    ROOT
    / "reports"
    / "research"
    / "single_level_reacher_parity_20260806"
    / "reacher_eval_threshold_forensics.json"
)
LEWM_INITIAL = "83f97d72ad067855bc89a1b74b4aff11d4dfdf0c"
SWM_QPOS_INTRO = "2096f6a17498f6283881e141d1c0579536815485"
SWM_THRESHOLD_CHANGE = "7259d6f0282e746c1d6b1cbddf34ae6ca2e172f8"
SWM_005_WHEEL_SHA256 = "c344003d77670e80411d534a1926219efa259c19b6b7da1bbd4f9386c205777b"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mwm-parity-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _threshold(source: str) -> float:
    match = re.search(r"_DEFAULT_QPOS_THRESHOLD\s*=\s*([0-9.]+)", source)
    if match is None:
        raise RuntimeError("Could not find _DEFAULT_QPOS_THRESHOLD")
    return float(match.group(1))


def _github_commit_dates(repo: str, commit: str) -> dict[str, str]:
    payload = json.loads(_download(f"https://api.github.com/repos/{repo}/commits/{commit}"))
    return {
        "author_date": str(payload["commit"]["author"]["date"]),
        "committer_date": str(payload["commit"]["committer"]["date"]),
    }


def main() -> None:
    lewm_config_url = (
        f"https://raw.githubusercontent.com/lucas-maes/le-wm/{LEWM_INITIAL}/config/eval/reacher.yaml"
    )
    intro_url = (
        "https://raw.githubusercontent.com/galilai-group/stable-worldmodel/"
        f"{SWM_QPOS_INTRO}/stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py"
    )
    changed_url = (
        "https://raw.githubusercontent.com/galilai-group/stable-worldmodel/"
        f"{SWM_THRESHOLD_CHANGE}/stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py"
    )
    lewm_config_bytes = _download(lewm_config_url)
    intro_bytes = _download(intro_url)
    changed_bytes = _download(changed_url)
    lewm_config = lewm_config_bytes.decode("utf-8")
    intro_source = intro_bytes.decode("utf-8")
    changed_source = changed_bytes.decode("utf-8")
    if "task: qpos_match" not in lewm_config or "method: set_target_qpos" not in lewm_config:
        raise RuntimeError("Initial LeWM Reacher config does not contain the expected qpos-match contract")
    if _threshold(intro_source) != 0.1 or _threshold(changed_source) != 0.05:
        raise RuntimeError("Stable WorldModel threshold timeline did not match 0.1 -> 0.05")

    pypi = json.loads(_download("https://pypi.org/pypi/stable-worldmodel/0.0.5/json"))
    wheel_record = next(
        item for item in pypi["urls"] if str(item["filename"]).endswith("py3-none-any.whl")
    )
    wheel_bytes = _download(str(wheel_record["url"]))
    if _sha256(wheel_bytes) != SWM_005_WHEEL_SHA256:
        raise RuntimeError("Stable WorldModel 0.0.5 wheel hash mismatch")
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
        names = set(archive.namelist())
        wheel_reacher_path = "stable_worldmodel/envs/dmcontrol/reacher.py"
        wheel_reacher = archive.read(wheel_reacher_path).decode("utf-8")
    custom_task_path = "stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py"
    if "qpos_match" in wheel_reacher or custom_task_path in names:
        raise RuntimeError("Stable WorldModel 0.0.5 unexpectedly contains the qpos-match task")

    import stable_worldmodel.envs.dmcontrol.custom_tasks.reacher as installed_task

    installed_path = Path(installed_task.__file__).resolve()
    installed_source_bytes = installed_path.read_bytes()
    installed_threshold = _threshold(installed_source_bytes.decode("utf-8"))
    if installed_threshold != 0.05:
        raise RuntimeError(f"Installed qpos threshold changed unexpectedly: {installed_threshold}")

    lewm_dates = _github_commit_dates("lucas-maes/le-wm", LEWM_INITIAL)
    intro_dates = _github_commit_dates("galilai-group/stable-worldmodel", SWM_QPOS_INTRO)
    change_dates = _github_commit_dates(
        "galilai-group/stable-worldmodel", SWM_THRESHOLD_CHANGE
    )

    payload = {
        "status": "pass",
        "initial_lewm": {
            "commit": LEWM_INITIAL,
            **lewm_dates,
            "reacher_config_url": lewm_config_url,
            "reacher_config_sha256": _sha256(lewm_config_bytes),
            "task": "qpos_match",
            "goal_callable": "set_target_qpos(goal_qpos)",
        },
        "stable_worldmodel_qpos_intro": {
            "commit": SWM_QPOS_INTRO,
            **intro_dates,
            "source_url": intro_url,
            "source_sha256": _sha256(intro_bytes),
            "qpos_threshold": _threshold(intro_source),
        },
        "stable_worldmodel_threshold_change": {
            "commit": SWM_THRESHOLD_CHANGE,
            **change_dates,
            "source_url": changed_url,
            "source_sha256": _sha256(changed_bytes),
            "qpos_threshold": _threshold(changed_source),
        },
        "stable_worldmodel_0_0_5": {
            "release_upload_time": str(wheel_record["upload_time_iso_8601"]),
            "wheel_url": str(wheel_record["url"]),
            "wheel_sha256": _sha256(wheel_bytes),
            "qpos_match_present": False,
        },
        "installed_stable_worldmodel": {
            "version": metadata.version("stable-worldmodel"),
            "source_path": str(installed_path),
            "source_sha256": _sha256(installed_source_bytes),
            "qpos_threshold": installed_threshold,
        },
        "paper_figure_6": {
            "arxiv_v1_submission_time": "2026-03-13T19:48:14Z",
            "arxiv_url": "https://arxiv.org/abs/2603.19312",
            "reacher_panel_url": "https://arxiv.org/html/2603.19312v1/x9.png",
            "visible_lewm_success_rate_percent": 86,
        },
        "conclusion": (
            "Reacher evaluation semantics are a versioned recipe input. The qpos-match task "
            "landed March 11 with a 0.1-radian per-joint success threshold, one day before "
            "LeWM's initial code author date and two days before the arXiv v1 submission. An "
            "unrelated Stable WorldModel commit changed it to 0.05 on March 16, three days "
            "after v1; the initial LeWM commit was not published on GitHub until March 23. "
            "PyPI 0.0.5 contains neither task, so the initial unpinned installation instructions "
            "were insufficient to reproduce Reacher. MWM must set and record the threshold "
            "explicitly."
        ),
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
