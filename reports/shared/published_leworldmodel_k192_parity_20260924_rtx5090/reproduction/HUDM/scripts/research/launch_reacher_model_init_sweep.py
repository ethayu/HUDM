from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SELECTION_BASIS = (
    "minimum absolute mean candidate-minus-upstream delta under the paper's "
    "10-iteration Reacher CEM recipe"
)
SELECTION_INTERPRETATION = (
    "Operational branch choice for the follow-on construction-seed sensitivity sweep; "
    "it is not proof of the released checkpoint's historical LR horizon because the "
    "released construction RNG and training runtime are unrecoverable."
)
SEEDS = (0, 1, 2, 42, 100)
EXECUTION_FILES = (
    "scripts/research/slurm_launch_reacher_model_init_sweep_20260806.sbatch",
    "scripts/research/launch_reacher_model_init_sweep.py",
    "scripts/research/slurm_research_train_reacher_model_init_sweep_20260806.sbatch",
    "scripts/research/slurm_research_eval_reacher_model_init_sweep_20260806.sbatch",
    "scripts/research/research_reacher_checkpoint_n500.sh",
    "scripts/research/slurm_collect_reacher_model_init_n500.sbatch",
    "scripts/research/collect_reacher_model_init_n500.py",
    "scripts/research/slurm_finalize_single_level_reacher_parity_20260806.sbatch",
    "scripts/research/audit_single_level_reacher_parity.py",
)
COMPARISON = (
    ROOT
    / "reports"
    / "research"
    / "single_level_reacher_parity_20260806"
    / "scheduler_branch_comparison.json"
)
RECEIPT = (
    ROOT
    / "reports"
    / "research"
    / "single_level_reacher_parity_20260806"
    / "model_init_sweep_submission.json"
)
BRANCHES: dict[str, dict[str, str]] = {
    "paper10": {
        "train_config": (
            "configs/research/"
            "train_mwm_lewm_reacher_k192_paper10_nativeh5_nodecoder_init3072_fit0_20260806.yaml"
        ),
        "lr_max_epochs": "10",
        "sweep_tag": "paper10",
        "report_prefix": "single_level_reacher_model_init_paper10_parity_20260806",
        "eval_config": "configs/eval/paper_reacher.yaml",
        "baseline_report_tag": "single_level_reacher_paper10_nativeh5_parity_20260806",
        "baseline_candidate_role": "paper10_nativeh5_nodecoder_init3072_fit0",
    },
    "epoch10_horizon100": {
        "train_config": (
            "configs/research/"
            "train_mwm_lewm_reacher_k192_epoch10_horizon100_nativeh5_nodecoder_init3072_fit0_20260806.yaml"
        ),
        "lr_max_epochs": "100",
        "sweep_tag": "epoch10_horizon100",
        "report_prefix": (
            "single_level_reacher_model_init_epoch10_horizon100_parity_20260806"
        ),
        "eval_config": "configs/eval/paper_reacher.yaml",
        "baseline_report_tag": (
            "single_level_reacher_epoch10_horizon100_nativeh5_parity_20260806"
        ),
        "baseline_candidate_role": (
            "epoch10_horizon100_nativeh5_nodecoder_init3072_fit0"
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recipe(path: Path) -> tuple[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"Scheduler comparison did not pass: {path}")
    branch = payload.get("selected_scheduler_branch")
    if branch not in BRANCHES:
        raise RuntimeError(f"Unsupported selected scheduler branch: {branch!r}")
    if payload.get("selection_basis") != SELECTION_BASIS:
        raise RuntimeError("Scheduler comparison has an unexpected selection basis")
    if payload.get("selection_interpretation") != SELECTION_INTERPRETATION:
        raise RuntimeError("Scheduler comparison is missing the provenance limitation")
    if payload.get("historical_scheduler_identification") != (
        "unrecoverable_from_released_artifact"
    ):
        raise RuntimeError("Scheduler comparison overstates historical identification")
    return str(branch), dict(BRANCHES[str(branch)])


def _assert_fresh(root: Path, recipe: dict[str, str], receipt: Path) -> None:
    if receipt.exists():
        raise RuntimeError(f"Refusing to overwrite submission receipt: {receipt}")
    report_root = root / "reports" / "research" / recipe["report_prefix"]
    if report_root.exists():
        raise RuntimeError(f"Refusing to mix with an existing sweep report: {report_root}")
    for seed in SEEDS:
        run_name = (
            f"mwm_reacher_k192_{recipe['sweep_tag']}_nativeh5_nodecoder_"
            f"init{seed}_fit0_20260806"
        )
        for path in (
            root / "checkpoints_mwm" / run_name,
            root / "logs" / "mwm_training" / run_name,
        ):
            if path.exists():
                raise RuntimeError(f"Refusing to mix with an existing sweep artifact: {path}")
        for eval_seed in SEEDS:
            output = (
                root
                / "rollouts"
                / f"{recipe['report_prefix']}_init{seed}_seed{eval_seed}_n500"
            )
            if output.exists():
                raise RuntimeError(
                    f"Refusing to mix with an existing sweep artifact: {output}"
                )


def _submit(root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["sbatch", "--parsable", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {completed.stdout!r}")
    return job_id


def submission_commands(recipe: dict[str, str]) -> list[list[str]]:
    train_export = (
        "ALL,"
        f"MWM_TRAIN_CONFIG={recipe['train_config']},"
        f"MWM_EXPECTED_LR_MAX_EPOCHS={recipe['lr_max_epochs']},"
        f"MWM_INIT_SWEEP_TAG={recipe['sweep_tag']}"
    )
    eval_export = (
        "ALL,"
        f"MWM_INIT_SWEEP_TAG={recipe['sweep_tag']},"
        f"MWM_INIT_SWEEP_REPORT_PREFIX={recipe['report_prefix']},"
        f"MWM_EVAL_CONFIG={recipe['eval_config']}"
    )
    collect_export = (
        "ALL,"
        f"MWM_INIT_SWEEP_REPORT_PREFIX={recipe['report_prefix']},"
        f"MWM_BASELINE_REPORT_TAG={recipe['baseline_report_tag']},"
        f"MWM_BASELINE_CANDIDATE_ROLE={recipe['baseline_candidate_role']}"
    )
    return [
        [
            "--export",
            train_export,
            "scripts/research/slurm_research_train_reacher_model_init_sweep_20260806.sbatch",
        ],
        [
            "--export",
            eval_export,
            "scripts/research/slurm_research_eval_reacher_model_init_sweep_20260806.sbatch",
        ],
        [
            "--export",
            collect_export,
            "scripts/research/slurm_collect_reacher_model_init_n500.sbatch",
        ],
        ["scripts/research/slurm_finalize_single_level_reacher_parity_20260806.sbatch"],
    ]


def launch(root: Path, comparison: Path, receipt: Path) -> dict[str, Any]:
    branch, recipe = load_recipe(comparison)
    _assert_fresh(root, recipe, receipt)
    commands = submission_commands(recipe)
    payload: dict[str, Any] = {
        "status": "launching",
        "selected_scheduler_branch": branch,
        "scheduler_comparison": str(comparison.relative_to(root)),
        "scheduler_comparison_sha256": _sha256(comparison),
        "recipe": recipe,
        "recipe_file_sha256s": {
            "train_config": _sha256(root / recipe["train_config"]),
            "eval_config": _sha256(root / recipe["eval_config"]),
        },
        "execution_file_sha256s": {
            path: _sha256(root / path) for path in EXECUTION_FILES
        },
        "model_init_seeds": list(SEEDS),
        "evaluation_seeds": list(SEEDS),
        "episodes_per_evaluation_seed": 500,
        "jobs": {},
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_job = _submit(root, commands[0])
    payload["jobs"]["training_array"] = train_job
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    eval_job = _submit(root, [f"--dependency=afterok:{train_job}", *commands[1]])
    payload["jobs"]["evaluation_array"] = eval_job
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    collect_job = _submit(root, [f"--dependency=afterok:{eval_job}", *commands[2]])
    payload["jobs"]["collector"] = collect_job
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_audit_job = _submit(
        root, [f"--dependency=afterok:{collect_job}", *commands[3]]
    )
    payload["jobs"]["final_audit"] = final_audit_job
    payload["status"] = "submitted"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the required model-construction seed sweep for the selected LR branch."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    comparison = args.comparison.resolve()
    receipt = args.receipt.resolve()
    branch, recipe = load_recipe(comparison)
    _assert_fresh(root, recipe, receipt)
    if not args.submit:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "selected_scheduler_branch": branch,
                    "recipe": recipe,
                    "commands": submission_commands(recipe),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(launch(root, comparison, receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
