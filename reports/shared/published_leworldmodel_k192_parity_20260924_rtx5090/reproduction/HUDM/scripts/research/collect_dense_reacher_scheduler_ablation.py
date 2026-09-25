from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import torch


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2, 42, 100)
ROLES = {
    "upstream": "upstream_lewm_converted",
    "july28_dense": "release_dense_fixed_finest",
    "spt_h10": "dense_spt_isolateddec_h10",
    "spt_h100": "dense_spt_isolateddec_h100",
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


def _resolved_training_config(horizon: int) -> dict[str, object]:
    run_name = f"mwm_dense_reacher_spt_isolateddec_h{horizon}_seed3072_20260809"
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
    actual = {
        "max_epochs": int(cfg["schedule"]["max_epochs"]),
        "lr_max_epochs": int(cfg["schedule"]["lr_max_epochs"]),
        "seed": int(cfg["seed"]),
        "K": list(cfg["model"]["K"]),
        "D": int(cfg["model"]["D"]),
        "sigreg": float(cfg["loss"]["sigreg_weight"]),
        "decoder": cfg["decoder_training"]["enabled"],
        "decoder_mode": str(cfg["decoder_training"]["mode"]),
        "detach_decoder": cfg["mwm"]["loss_terms"]["reconstructor_detach_encoder"],
        "decoder_encoder_loss": cfg["mwm"]["loss_terms"]["reconstructor_contributes_to_encoder_loss"],
        "run_name": str(cfg["train"]["run_name"]),
        "optimizer_states": len(checkpoint.get("optimizer_states", [])),
        "lr_schedulers": len(checkpoint.get("lr_schedulers", [])),
    }
    expected = {
        "max_epochs": 10,
        "lr_max_epochs": horizon,
        "seed": 3072,
        "K": [96, 120, 144, 168, 192],
        "D": 192,
        "sigreg": 0.10,
        "decoder": True,
        "decoder_mode": "separate_optimizer",
        "detach_decoder": True,
        "decoder_encoder_loss": False,
        "run_name": run_name,
        "optimizer_states": 2,
        "lr_schedulers": 2,
    }
    if actual != expected:
        raise RuntimeError(f"Resolved scheduler-ablation contract mismatch in {path}: {actual} != {expected}")
    return cfg


def _without_allowed_differences(cfg: dict[str, object]) -> dict[str, object]:
    normalized = deepcopy(cfg)
    normalized["schedule"]["lr_max_epochs"] = "<scheduler-horizon>"
    normalized["train"]["run_name"] = "<run-name>"
    normalized.pop("slurm", None)
    for key in tuple(normalized):
        if key.startswith("slurm."):
            normalized.pop(key)
    return normalized


def _markdown(payload: dict[str, object]) -> str:
    rates = payload["success_rates"]
    lines = [
        "# Dense Reacher scheduler ablation",
        "",
        "All entries are N=500 across seeds 0, 1, 2, 42, and 100, using the archived July 28 evaluator and identical manifests.",
        "",
        "| Model | Success rate (%) | Difference vs upstream (pp) |",
        "|---|---:|---:|",
    ]
    upstream = rates["upstream"]["mean"]
    labels = {
        "upstream": "Upstream LeWM",
        "july28_dense": "July 28 dense release",
        "spt_h10": "Current SPT dense, 10-epoch LR horizon",
        "spt_h100": "Current SPT dense, 100-epoch LR horizon",
    }
    for name in ROLES:
        center = rates[name]["mean"]
        low, high = rates[name]["ci95"]
        lines.append(
            f"| {labels[name]} | {center:.2f} (95% CI {low:.2f}, {high:.2f}) | {center - upstream:+.2f} |"
        )
    lines.extend(
        [
            "",
            f"The 100-horizon minus 10-horizon paired mean difference is {payload['paired_differences']['spt_h100_minus_h10']['mean']:+.2f} percentage points.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    h10_cfg = _resolved_training_config(10)
    h100_cfg = _resolved_training_config(100)
    if _without_allowed_differences(h10_cfg) != _without_allowed_differences(h100_cfg):
        raise RuntimeError("Training branches differ outside scheduler horizon/run name")

    rates = {name: [] for name in ROLES}
    paired_rows = []
    for seed in SEEDS:
        summary_path = (
            ROOT
            / "rollouts"
            / f"mwm_release20260728_dense_k192_eval_size_reacher_seed{seed}_n500"
            / "summary.csv"
        )
        rows = {name: _row(summary_path, role) for name, role in ROLES.items()}
        manifest_hashes = {row["manifest_sha256"] for row in rows.values()}
        manifest_file_hashes = {row["manifest_file_sha256"] for row in rows.values()}
        if len(manifest_hashes) != 1 or len(manifest_file_hashes) != 1:
            raise RuntimeError(f"Manifest mismatch for Reacher seed {seed}")
        if any(int(row["episodes"]) != 500 or int(row["seed"]) != seed for row in rows.values()):
            raise RuntimeError(f"Seed/episode mismatch for Reacher seed {seed}")
        values = {name: float(row["success_rate"]) for name, row in rows.items()}
        for name, value in values.items():
            rates[name].append(value)
        paired_rows.append(
            {
                "seed": seed,
                **values,
                "july28_dense_minus_upstream": values["july28_dense"] - values["upstream"],
                "spt_h10_minus_upstream": values["spt_h10"] - values["upstream"],
                "spt_h100_minus_upstream": values["spt_h100"] - values["upstream"],
                "spt_h100_minus_h10": values["spt_h100"] - values["spt_h10"],
                "manifest_sha256": next(iter(manifest_hashes)),
                "manifest_file_sha256": next(iter(manifest_file_hashes)),
            }
        )

    comparisons = {
        "july28_dense_minus_upstream": [
            dense - upstream for dense, upstream in zip(rates["july28_dense"], rates["upstream"], strict=True)
        ],
        "spt_h10_minus_upstream": [
            h10 - upstream for h10, upstream in zip(rates["spt_h10"], rates["upstream"], strict=True)
        ],
        "spt_h100_minus_upstream": [
            h100 - upstream for h100, upstream in zip(rates["spt_h100"], rates["upstream"], strict=True)
        ],
        "spt_h100_minus_h10": [
            h100 - h10 for h100, h10 in zip(rates["spt_h100"], rates["spt_h10"], strict=True)
        ],
    }
    payload: dict[str, object] = {
        "status": "pass",
        "objective": "Change only the LR scheduler horizon from 10 to 100 while stopping dense Reacher training at epoch 10.",
        "episodes_per_seed": 500,
        "seeds": list(SEEDS),
        "paired_results": paired_rows,
        "success_rates": {name: _summary(values) for name, values in rates.items()},
        "paired_differences": {name: _summary(values) for name, values in comparisons.items()},
        "training_contract": {
            "epochs": 10,
            "scheduler_horizons": [10, 100],
            "K": [96, 120, 144, 168, 192],
            "seed": 3072,
            "sigreg_weight": 0.10,
            "decoder_training": "enabled_isolated_optimizer",
            "only_resolved_config_differences": ["schedule.lr_max_epochs", "train.run_name"],
        },
    }
    destination = ROOT / "reports" / "research" / "dense_reacher_scheduler_ablation_20260809"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "n500_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
