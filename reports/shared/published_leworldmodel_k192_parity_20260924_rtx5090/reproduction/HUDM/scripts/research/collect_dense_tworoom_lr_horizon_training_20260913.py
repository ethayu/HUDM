from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/research/dense_tworoom_lr_horizon_epochs_20260913"
HORIZONS = (10, 100)
EPOCHS = tuple(range(10))
RUN_NAMES = {
    10: "mwm_dense_tworoom_lr_horizon_h10_epochs_20260913",
    100: "mwm_dense_tworoom_lr_horizon_h100_epochs_20260913",
}
METRICS = (
    "validate/loss_epoch",
    "validate/pred_loss_epoch",
    "validate/recon_loss_epoch",
    "validate/sigreg_loss_epoch",
    *(f"validate/pred_loss_l{level}_epoch" for level in range(5)),
    *(f"validate/recon_loss_l{level}_epoch" for level in range(5)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_branch(horizon: int) -> tuple[Path, list[dict[str, object]]]:
    path = ROOT / "logs/mwm_training" / RUN_NAMES[horizon] / "csv_logs/version_0/metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = [row for row in csv.DictReader(handle) if row.get("validate/loss_epoch")]
    if len(raw_rows) != len(EPOCHS):
        raise RuntimeError(f"h{horizon}: expected 10 validation epoch rows, found {len(raw_rows)}")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        epoch = int(raw["epoch"])
        if any(not raw.get(metric) for metric in METRICS):
            missing = [metric for metric in METRICS if not raw.get(metric)]
            raise RuntimeError(f"h{horizon}/epoch{epoch}: missing validation metrics: {missing}")
        rows.append(
            {
                "epoch": epoch,
                "step": int(raw["step"]),
                "metrics": {metric: float(raw[metric]) for metric in METRICS},
            }
        )
    rows.sort(key=lambda row: int(row["epoch"]))
    if [int(row["epoch"]) for row in rows] != list(EPOCHS):
        raise RuntimeError(f"h{horizon}: validation epoch set is incomplete or duplicated")
    return path, rows


def main() -> None:
    branches: dict[str, object] = {}
    branch_rows: dict[int, list[dict[str, object]]] = {}
    for horizon in HORIZONS:
        path, rows = _read_branch(horizon)
        branch_rows[horizon] = rows
        branches[f"h{horizon}"] = {
            "run_name": RUN_NAMES[horizon],
            "metrics_csv": str(path.relative_to(ROOT)),
            "metrics_csv_sha256": _sha256(path),
            "rows": rows,
            "best_epochs_by_metric": {
                metric: [
                    int(row["epoch"])
                    for row in rows
                    if float(row["metrics"][metric])
                    == min(float(candidate["metrics"][metric]) for candidate in rows)
                ]
                for metric in METRICS
            },
        }

    h10 = branch_rows[10]
    h100 = branch_rows[100]
    deltas = {
        metric: {
            str(epoch): float(h100[epoch]["metrics"][metric]) - float(h10[epoch]["metrics"][metric])
            for epoch in EPOCHS
        }
        for metric in METRICS
    }
    terminal_ratios = {
        metric: float(h100[-1]["metrics"][metric]) / float(h10[-1]["metrics"][metric])
        for metric in METRICS
    }
    payload = {
        "status": "pass",
        "horizons": list(HORIZONS),
        "epochs": list(EPOCHS),
        "metrics": list(METRICS),
        "branches": branches,
        "h100_minus_h10_by_epoch": deltas,
        "terminal_h100_over_h10": terminal_ratios,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = REPORT_ROOT / "training_trajectory.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
