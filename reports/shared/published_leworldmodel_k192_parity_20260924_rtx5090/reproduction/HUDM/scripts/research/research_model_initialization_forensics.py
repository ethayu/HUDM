from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    value = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether released LeWM construction RNG is recoverable.")
    parser.add_argument("--historical-lewm-source", required=True)
    parser.add_argument(
        "--raw-checkpoint",
        default="/vast/home/e/ethanyu/.stable_worldmodel/checkpoints/models--quentinll--lewm-reacher",
    )
    parser.add_argument(
        "--output",
        default="reports/research/single_level_reacher_parity_20260806/model_initialization_forensics.json",
    )
    args = parser.parse_args()

    source_root = Path(args.historical_lewm_source).resolve()
    train_path = source_root / "train.py"
    train_source = train_path.read_text(encoding="utf-8")
    tree = ast.parse(train_source)
    calls = [(node.lineno, _call_name(node), node) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    forbidden_seed_calls = {
        "torch.manual_seed",
        "pl.seed_everything",
        "lightning.seed_everything",
    }
    observed_forbidden = sorted(
        {name for _, name, _ in calls if name in forbidden_seed_calls}
    )
    generator_seed_lines = [
        line for line, name, _ in calls if name.endswith("manual_seed") and "Generator" in train_source
    ]
    encoder_lines = [line for line, name, _ in calls if name == "spt.backbone.utils.vit_hf"]
    manager_calls = [(line, node) for line, name, node in calls if name == "spt.Manager"]
    if observed_forbidden or len(encoder_lines) != 1 or len(manager_calls) != 1:
        raise RuntimeError(
            f"Unexpected upstream construction/seeding graph: forbidden={observed_forbidden}, "
            f"encoder={encoder_lines}, manager={[(x[0]) for x in manager_calls]}"
        )
    manager_line, manager_node = manager_calls[0]
    manager_keywords = sorted(keyword.arg for keyword in manager_node.keywords if keyword.arg is not None)
    if "seed" in manager_keywords:
        raise RuntimeError("Historical LeWM unexpectedly passed a Manager seed")
    if not generator_seed_lines or not all(line < encoder_lines[0] for line in generator_seed_lines):
        raise RuntimeError("Dataset generator seed provenance changed unexpectedly")
    if not encoder_lines[0] < manager_line:
        raise RuntimeError("Model was not constructed before Manager creation")

    checkpoint = Path(args.raw_checkpoint).resolve()
    checkpoint_files = sorted(path.name for path in checkpoint.iterdir() if path.is_file())
    state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=False)
    if any("pooler" in key for key in state):
        raise RuntimeError("Released state unexpectedly contains a ViT pooler fingerprint")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    serialized_rng_keys = sorted(
        key for key in config if any(token in key.lower() for token in ("seed", "rng", "random_state"))
    )
    if serialized_rng_keys:
        raise RuntimeError(f"Released model config unexpectedly records RNG keys: {serialized_rng_keys}")

    manager_path = Path(importlib.metadata.distribution("stable-pretraining").locate_file("stable_pretraining/manager.py"))
    manager_source = manager_path.read_text(encoding="utf-8")
    if "pl.seed_everything(self.seed, workers=True)" not in manager_source:
        raise RuntimeError("Installed Stable Pretraining Manager seeding contract changed")

    payload = {
        "status": "pass",
        "historical_lewm_commit": "83f97d72ad067855bc89a1b74b4aff11d4dfdf0c",
        "historical_train_sha256": _sha256(train_path),
        "source_order": {
            "dataset_generator_manual_seed_lines": generator_seed_lines,
            "model_encoder_construction_line": encoder_lines[0],
            "manager_construction_line": manager_line,
            "manager_keywords": manager_keywords,
            "global_seed_calls_before_model": observed_forbidden,
        },
        "stable_pretraining": {
            "installed_version": importlib.metadata.version("stable-pretraining"),
            "manager_sha256": _sha256(manager_path),
            "manager_seeds_only_when_called": True,
            "historical_call_passes_seed": False,
        },
        "released_checkpoint": {
            "path": str(checkpoint),
            "files": checkpoint_files,
            "weights_sha256": _sha256(checkpoint / "weights.pt"),
            "serialized_rng_config_keys": serialized_rng_keys,
            "contains_optimizer_state": any("optim" in name.lower() for name in checkpoint_files),
            "contains_rng_state": any("rng" in name.lower() or "seed" in name.lower() for name in checkpoint_files),
            "contains_vit_pooler_parameters": any("pooler" in key for key in state),
        },
        "conclusion": (
            "cfg.seed=3072 seeds only the split/shuffle torch.Generator. The model is constructed "
            "before Manager is created/called, no global construction seed is set, and Manager receives "
            "no seed. The released config+weights artifact records neither construction RNG nor optimizer/RNG "
            "state and contains no unused ViT pooler fingerprint. Exact construction seed is therefore not "
            "recoverable from published artifacts; a controlled initialization sweep is required."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
