from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import torch


ROOT = Path(__file__).resolve().parents[2]
ENVS = ("pusht", "ogb_cube", "tworoom")
SEEDS = (0, 1, 2, 42, 100)
ROLES = {
    "upstream": "upstream_lewm_converted",
    "current_aug2": "release_single_k192_isolateddec",
    "h10_rerun": "scheduler_h10_rerun_v2",
    "h100": "scheduler_h100_v2",
}
T95_DF4 = 2.7764451051977987


def _row(path: Path, role: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == role]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {role!r} row in {path}, found {len(rows)}")
    return rows[0]


def _summary(values: list[float]) -> dict[str, object]:
    spread = stdev(values)
    half_width = T95_DF4 * spread / math.sqrt(len(values))
    center = mean(values)
    return {
        "mean": center,
        "sample_stdev": spread,
        "ci95": [center - half_width, center + half_width],
        "by_seed": dict(zip((str(seed) for seed in SEEDS), values, strict=True)),
    }


def _resolved_training_config(env_name: str, horizon: int) -> dict[str, object]:
    run_name = f"mwm_scheduler_ablation_{env_name}_k192_h{horizon}_seed3072_20260807"
    path = (
        ROOT
        / "logs"
        / "mwm_training"
        / run_name
        / "csv_logs"
        / "version_0"
        / "checkpoints"
        / "last.ckpt"
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("hyper_parameters")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Missing resolved training config in {path}")
    if (
        int(cfg["schedule"]["max_epochs"]) != 10
        or int(cfg["schedule"]["lr_max_epochs"]) != horizon
        or int(cfg["seed"]) != 3072
        or list(cfg["model"]["K"]) != [192]
        or float(cfg["loss"]["sigreg_weight"]) != 0.10
        or cfg["decoder_training"]["enabled"] is not True
        or str(cfg["train"]["run_name"]) != run_name
    ):
        raise RuntimeError(f"Resolved scheduler-ablation contract mismatch in {path}")
    return cfg


def _without_allowed_differences(cfg: dict[str, object]) -> dict[str, object]:
    normalized = deepcopy(cfg)
    normalized["schedule"]["lr_max_epochs"] = "<scheduler-horizon>"
    normalized["train"]["run_name"] = "<run-name>"
    # Slurm injects per-array-task provenance after config resolution. It is
    # execution metadata, not part of the training recipe.
    normalized.pop("slurm", None)
    for key in tuple(normalized):
        if key.startswith("slurm."):
            normalized.pop(key)
    return normalized


def main() -> None:
    payload: dict[str, object] = {
        "status": "pass",
        "objective": "Change only the LR scheduler horizon from 10 to 100 while stopping at epoch 10.",
        "episodes_per_seed": 500,
        "seeds": list(SEEDS),
        "environments": {},
    }
    for env_name in ENVS:
        h10_cfg = _resolved_training_config(env_name, 10)
        h100_cfg = _resolved_training_config(env_name, 100)
        if _without_allowed_differences(h10_cfg) != _without_allowed_differences(h100_cfg):
            raise RuntimeError(f"Training branches differ outside scheduler horizon/run name for {env_name}")

        rates = {name: [] for name in ROLES}
        paired_rows = []
        for seed in SEEDS:
            summary_path = (
                ROOT
                / "rollouts"
                / f"mwm_release20260728_dense_k192_eval_size_{env_name}_seed{seed}_n500"
                / "summary.csv"
            )
            rows = {name: _row(summary_path, role) for name, role in ROLES.items()}
            manifest_hashes = {row["manifest_sha256"] for row in rows.values()}
            manifest_file_hashes = {row["manifest_file_sha256"] for row in rows.values()}
            if len(manifest_hashes) != 1 or len(manifest_file_hashes) != 1:
                raise RuntimeError(f"Manifest mismatch for {env_name} seed {seed}")
            if any(int(row["episodes"]) != 500 or int(row["seed"]) != seed for row in rows.values()):
                raise RuntimeError(f"Seed/episode mismatch for {env_name} seed {seed}")
            values = {name: float(row["success_rate"]) for name, row in rows.items()}
            for name, value in values.items():
                rates[name].append(value)
            paired_rows.append(
                {
                    "seed": seed,
                    **values,
                    "h100_minus_h10": values["h100"] - values["h10_rerun"],
                    "h10_rerun_minus_current_aug2": values["h10_rerun"] - values["current_aug2"],
                    "h100_minus_upstream": values["h100"] - values["upstream"],
                    "manifest_sha256": next(iter(manifest_hashes)),
                    "manifest_file_sha256": next(iter(manifest_file_hashes)),
                }
            )

        comparisons = {
            "current_aug2_minus_upstream": [
                current - upstream for current, upstream in zip(rates["current_aug2"], rates["upstream"], strict=True)
            ],
            "h10_rerun_minus_current_aug2": [
                rerun - current for rerun, current in zip(rates["h10_rerun"], rates["current_aug2"], strict=True)
            ],
            "h100_minus_h10": [
                h100 - h10 for h100, h10 in zip(rates["h100"], rates["h10_rerun"], strict=True)
            ],
            "h100_minus_upstream": [
                h100 - upstream for h100, upstream in zip(rates["h100"], rates["upstream"], strict=True)
            ],
        }
        payload["environments"][env_name] = {
            "paired_results": paired_rows,
            "success_rates": {name: _summary(values) for name, values in rates.items()},
            "paired_differences": {name: _summary(values) for name, values in comparisons.items()},
            "training_contract": {
                "epochs": 10,
                "scheduler_horizons": [10, 100],
                "K": [192],
                "seed": 3072,
                "sigreg_weight": 0.10,
                "decoder_training": "enabled_isolated_optimizer",
                "only_resolved_config_differences": ["schedule.lr_max_epochs", "train.run_name"],
            },
        }

    destination = ROOT / "reports" / "research" / "crossenv_scheduler_ablation_20260807" / "n500_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
