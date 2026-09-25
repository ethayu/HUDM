from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


JUNE_COMMIT = "c09bff78e51798b01620a252ff073b1c3efc732f"
REACHER_CONFIG_SHA256 = "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"
MODEL_SEED = 3072
FORWARD_SEED = 91_3072


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_source_config() -> Path:
    stablewm_root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    return stablewm_root / "checkpoints" / "models--quentinll--lewm-reacher" / "config.json"


def _tensor_record(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    finite = value.isfinite() if value.is_floating_point() else None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "all_finite": True if finite is None else bool(finite.all().item()),
    }


def _named_records(values: Any, *, excluded_prefixes: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    records = []
    for name, tensor in values:
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        records.append({"name": name, **_tensor_record(tensor)})
    return records


def _output_records(output: dict[str, Any], *, ignored: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        name: _tensor_record(value)
        for name, value in output.items()
        if name not in ignored and hasattr(value, "detach")
    }


def _build_snapshot(variant: str, source_config_path: Path, output: Path) -> None:
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))

    from mwm.adapters.base import ComponentPolicy
    from mwm.adapters.builder import build_mwm_from_stable_config

    torch.manual_seed(MODEL_SEED)
    common = {
        "family": "lewm",
        "source_config": source_config,
        "source_config_sha256": REACHER_CONFIG_SHA256,
        "training_recipe": {
            "history_size": 3,
            "num_preds": 1,
            "action_preprocessing": "standard_scaler",
            "loss_scope": {
                "regularizers": "shared_latent",
                "reconstructor_detach_encoder": True,
                "reconstructor_contributes_to_encoder_loss": False,
            },
        },
        "K": (192,),
        "action_dim": 10,
        "expected_D": 192,
        "action_block": 5,
        "image_shape": (224, 224),
        "normalize_imagenet": True,
    }
    if variant == "historical":
        policy = ComponentPolicy(
            shared=("latent_producer",),
            per_level=("transition",),
            reconstructor=(),
        )
    elif variant == "current":
        policy = ComponentPolicy(
            shared=("latent_producer",),
            per_level=("transition",),
            reconstructor=("decoder",),
        )
    else:
        raise ValueError(f"Unknown snapshot variant {variant!r}.")
    model = build_mwm_from_stable_config(**common, component_policy=policy)

    excluded = ("decoders.",)
    parameters = _named_records(model.named_parameters(), excluded_prefixes=excluded)
    buffers = _named_records(model.named_buffers(), excluded_prefixes=excluded)
    decoder_parameters = _named_records(
        ((name, value) for name, value in model.named_parameters() if name.startswith("decoders."))
    )

    input_generator = torch.Generator(device="cpu").manual_seed(FORWARD_SEED)
    pixels = torch.randn(2, 4, 3, 224, 224, generator=input_generator)
    action = torch.randn(2, 4, 10, generator=input_generator)
    inputs = {
        "pixels": _tensor_record(pixels),
        "action": _tensor_record(action),
    }

    forwards: dict[str, Any] = {}
    for mode_index, mode in enumerate(("eval", "train")):
        model.eval() if mode == "eval" else model.train()

        torch.manual_seed(FORWARD_SEED + 100 * mode_index + 1)
        encoded = model._encode_pixels(pixels.clone(), already_preprocessed=True)

        torch.manual_seed(FORWARD_SEED + 100 * mode_index + 2)
        action_encoded = model.transitions[0].action_encoder(action[:, :3].clone())

        torch.manual_seed(FORWARD_SEED + 100 * mode_index + 3)
        predicted = model._predict_prefix(0, encoded[:, :3].clone(), action[:, :3].clone())

        torch.manual_seed(FORWARD_SEED + 100 * mode_index + 4)
        loss_kwargs = {
            "level_weights": None,
            "rollout_weight": 1.0,
            "sigreg": None,
            "sigreg_weight": 0.0,
            "sigreg_scope": "shared_latent",
        }
        if variant == "current":
            loss_kwargs["decoder_training_enabled"] = False
        losses = model.training_loss(
            {"pixels": pixels.clone(), "action": action.clone()},
            **loss_kwargs,
        )
        forwards[mode] = {
            "encoded": _tensor_record(encoded),
            "action_encoded": _tensor_record(action_encoded),
            "predicted": _tensor_record(predicted),
            "losses": _output_records(
                losses,
                ignored=("world_loss",) if variant == "current" else (),
            ),
            "world_loss_alias": (
                _tensor_record(losses["world_loss"])
                if variant == "current" and "world_loss" in losses
                else None
            ),
        }

    import mwm

    snapshot = {
        "variant": variant,
        "mwm_source": str(Path(mwm.__file__).resolve()),
        "source_config": str(source_config_path.resolve()),
        "source_config_sha256": _sha256_file(source_config_path),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "architecture_version": str(model.architecture_version),
        "K": [int(value) for value in model.K],
        "D": int(model.D),
        "action_dim": int(model.action_dim),
        "action_block": int(model.action_block),
        "history_size": int(model.history_size),
        "num_preds": int(model.num_preds),
        "parameters": parameters,
        "buffers": buffers,
        "decoder_parameters": decoder_parameters,
        "world_parameter_tensors": len(parameters),
        "world_parameter_count": sum(record["numel"] for record in parameters),
        "inputs": inputs,
        "forwards": forwards,
    }
    output.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def _safe_extract_archive(archive: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        destination_resolved = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe path in git archive: {member.name!r}")
        handle.extractall(destination)


def _archive_historical_tree(repo: Path, commit: str, destination: Path) -> str:
    result = subprocess.run(
        ["git", "archive", "--format=tar", commit, "--", "mwm"],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"git archive failed with exit code {result.returncode}:\n{stderr}")
    _safe_extract_archive(result.stdout, destination)
    return stderr


def _run_snapshot(
    *,
    python: Path,
    script: Path,
    source_root: Path,
    variant: str,
    source_config: Path,
    output: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(source_root),
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [
        str(python),
        str(script),
        "--snapshot",
        variant,
        "--source-config",
        str(source_config),
        "--snapshot-output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=source_root,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{variant} snapshot failed with exit code {result.returncode}.\n"
            f"command: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    imported_source = Path(snapshot["mwm_source"]).resolve()
    expected_root = source_root.resolve()
    if expected_root not in imported_source.parents:
        raise RuntimeError(
            f"{variant} snapshot imported mwm from {imported_source}, outside isolated source root {expected_root}."
        )
    return snapshot


def _compare_exact(expected: Any, actual: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append(f"{path}: expected mapping, got {type(actual).__name__}")
            return
        if set(expected) != set(actual):
            mismatches.append(f"{path}: keys differ: {sorted(set(expected) ^ set(actual))}")
            return
        for key in expected:
            _compare_exact(expected[key], actual[key], f"{path}.{key}", mismatches)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            actual_length = len(actual) if isinstance(actual, list) else "not-a-list"
            mismatches.append(f"{path}: list lengths differ: {len(expected)} != {actual_length}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_exact(left, right, f"{path}[{index}]", mismatches)
        return
    if expected != actual:
        mismatches.append(f"{path}: {expected!r} != {actual!r}")


def _compare_snapshots(historical: dict[str, Any], current: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key in (
        "source_config_sha256",
        "K",
        "D",
        "action_dim",
        "action_block",
        "history_size",
        "num_preds",
        "world_parameter_tensors",
        "world_parameter_count",
        "parameters",
        "buffers",
        "inputs",
    ):
        _compare_exact(historical[key], current[key], key, mismatches)
    for mode in ("eval", "train"):
        historical_forward = historical["forwards"][mode]
        current_forward = current["forwards"][mode]
        for key in ("encoded", "action_encoded", "predicted", "losses"):
            _compare_exact(
                historical_forward[key],
                current_forward[key],
                f"forwards.{mode}.{key}",
                mismatches,
            )
        _compare_exact(
            current_forward["losses"]["loss"],
            current_forward["world_loss_alias"],
            f"forwards.{mode}.world_loss_alias",
            mismatches,
        )
    if historical["decoder_parameters"]:
        mismatches.append("historical snapshot unexpectedly contains decoder parameters")
    if not current["decoder_parameters"]:
        mismatches.append("current snapshot contains no decoder parameters")
    return mismatches


def _run_comparison(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[2]
    script = Path(__file__).resolve()
    python = Path(args.python).expanduser().resolve()
    source_config = Path(args.source_config).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pass_marker = output_dir / "PASS"
    pass_marker.unlink(missing_ok=True)

    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter does not exist: {python}")
    if not source_config.is_file():
        raise FileNotFoundError(f"Reacher source config does not exist: {source_config}")
    actual_sha256 = _sha256_file(source_config)
    if actual_sha256 != REACHER_CONFIG_SHA256:
        raise RuntimeError(
            f"Reacher source config SHA256 differs from the audited config: "
            f"{actual_sha256} != {REACHER_CONFIG_SHA256}"
        )

    report: dict[str, Any] = {
        "status": "fail",
        "comparison": "exact_cpu_float32",
        "historical_commit": args.commit,
        "model_seed": MODEL_SEED,
        "forward_seed": FORWARD_SEED,
        "source_config": str(source_config),
        "source_config_sha256": actual_sha256,
    }
    report_path = output_dir / "identity_report.json"
    try:
        with tempfile.TemporaryDirectory(prefix="mwm-june-identity-") as temp_name:
            temp_root = Path(temp_name)
            historical_root = temp_root / "historical"
            historical_root.mkdir()
            archive_stderr = _archive_historical_tree(repo, args.commit, historical_root)
            historical = _run_snapshot(
                python=python,
                script=script,
                source_root=historical_root,
                variant="historical",
                source_config=source_config,
                output=temp_root / "historical.json",
            )
            current = _run_snapshot(
                python=python,
                script=script,
                source_root=repo,
                variant="current",
                source_config=source_config,
                output=temp_root / "current.json",
            )
            mismatches = _compare_snapshots(historical, current)
            report.update(
                {
                    "status": "pass" if not mismatches else "fail",
                    "git_archive_stderr": archive_stderr,
                    "mismatches": mismatches,
                    "historical": historical,
                    "current": current,
                }
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        raise

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if report["status"] != "pass":
        raise AssertionError(
            f"June/current Reacher K=192 identity failed with {len(report['mismatches'])} mismatch(es); "
            f"see {report_path}"
        )
    pass_marker.write_text(
        "exact June c09bff78/current Reacher K=192 world identity passed\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "historical_commit": args.commit,
                "world_parameter_tensors": report["current"]["world_parameter_tensors"],
                "world_parameter_count": report["current"]["world_parameter_count"],
                "forward_modes": ["eval", "train"],
                "report": str(report_path),
                "pass_marker": str(pass_marker),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the June c09bff78 and current Reacher K=192 LeWM world implementations."
    )
    parser.add_argument("--commit", default=JUNE_COMMIT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-config", default=str(_default_source_config()))
    parser.add_argument("--output", default="reports/research/reacher_k192_june_identity")
    parser.add_argument("--snapshot", choices=("historical", "current"), help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.snapshot:
        if not args.snapshot_output:
            parser.error("--snapshot-output is required with --snapshot")
        _build_snapshot(
            args.snapshot,
            Path(args.source_config).expanduser().resolve(),
            Path(args.snapshot_output).expanduser().resolve(),
        )
        return
    _run_comparison(args)


if __name__ == "__main__":
    main()
