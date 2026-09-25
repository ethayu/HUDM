from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from omegaconf import OmegaConf

from mwm.benchmark.config import DEFAULTS, load_manifest_config
from mwm.config_cli import load_config
from mwm.data.manifest import load_manifest, manifest_sha256
from mwm.io import file_sha256


SCREENING_SCHEMA_VERSION = "mwm_release20260728_screening_v1"
SCREEN_EPISODES = 50
NUM_MATRIX_CELLS = 3120
DEFAULT_GENERATED_DIR = Path(
    "reports/research/release20260728_schedule_screening/generated"
)

RELEASE_CONFIGS = (
    "configs/research/release20260728_dense_ogb_cube_all_fidelity_schedules.yaml",
    "configs/research/release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules.yaml",
    "configs/research/release20260728_dense_pusht_all_fidelity_schedules.yaml",
    "configs/research/release20260728_dense_pusht_goal50_plan50_execute20_all_fidelity_schedules.yaml",
    "configs/research/release20260728_dense_reacher_all_fidelity_schedules.yaml",
    "configs/research/release20260728_dense_reacher_goal50_plan50_execute20_all_fidelity_schedules.yaml",
    "configs/research/release20260728_dense_tworoom_all_fidelity_schedules.yaml",
    "configs/research/release20260728_dense_tworoom_goal50_plan50_execute20_all_fidelity_schedules.yaml",
)


class ScreeningPreparationError(RuntimeError):
    """Raised when a generated screening asset would violate the prefix contract."""


def screen_storage_preflight(
    canonical_output_dir: str | Path,
    screen_output_dir: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Inspect output isolation without creating directories or symlinks.

    On the lab filesystem the canonical roots are Ceph symlinks. In that case a
    missing or ordinary-directory screen root is unsafe because matrix startup
    would silently put large artifacts on VAST. On a copied/cloud deployment,
    where the canonical root is absent or an ordinary directory, a regular
    isolated screen root is valid.
    """

    root = Path(repo_root).resolve()
    canonical = _resolve_from_root(root, canonical_output_dir)
    screen = _resolve_from_root(root, screen_output_dir)
    canonical_is_symlink = canonical.is_symlink()
    screen_is_symlink = screen.is_symlink()
    issues: list[str] = []
    if canonical_is_symlink and not screen_is_symlink:
        issues.append(
            f"canonical output is a storage symlink but screen output is not: {screen}"
        )
    if screen_is_symlink and not screen.resolve().is_dir():
        issues.append(f"screen output symlink target is missing or not a directory: {screen}")
    if screen.exists() and not screen.is_dir():
        issues.append(f"screen output exists but is not a directory: {screen}")
    if canonical.exists() and screen.exists() and canonical.resolve() == screen.resolve():
        issues.append("canonical and screen outputs resolve to the same directory")
    return {
        "ready": not issues,
        "canonical_output": str(canonical),
        "screen_output": str(screen),
        "canonical_is_symlink": canonical_is_symlink,
        "screen_is_symlink": screen_is_symlink,
        "issues": issues,
    }


@dataclass(frozen=True)
class PreparedScreen:
    canonical_config: str
    canonical_config_sha256: str
    canonical_manifest: str
    canonical_manifest_sha256: str
    canonical_output_dir: str
    screen_config: str
    screen_manifest: str
    screen_manifest_sha256: str
    screen_output_dir: str
    episodes: int
    matrix_cells: int


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_from_root(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_idempotent(path: Path, text: str, *, force: bool) -> None:
    if path.is_file():
        previous = path.read_text(encoding="utf-8")
        if previous == text:
            return
        if not force:
            raise ScreeningPreparationError(
                f"Refusing to replace stale generated asset {path}; rerun with --force "
                "after reviewing the canonical config or manifest change."
            )
    _atomic_write_text(path, text)


def prefix_manifest(
    canonical: Mapping[str, Any],
    *,
    canonical_path: str,
    episodes: int = SCREEN_EPISODES,
) -> dict[str, Any]:
    """Return a hashed manifest whose pairs are exactly ``canonical[:episodes]``.

    All dataset, restoration, seed, goal and budget provenance is inherited from
    the canonical manifest. The explicit parent hash makes the nesting
    independently auditable without relying on file names.
    """

    pairs = canonical.get("pairs", [])
    if not isinstance(pairs, list):
        raise ScreeningPreparationError("Canonical manifest pairs must be a list.")
    if len(pairs) < episodes:
        raise ScreeningPreparationError(
            f"Canonical manifest has {len(pairs)} pairs; cannot take a {episodes}-pair prefix."
        )
    parent_hash = str(canonical.get("manifest_sha256", ""))
    if not parent_hash:
        raise ScreeningPreparationError("Canonical manifest is missing manifest_sha256.")

    payload = dict(canonical)
    payload.pop("manifest_sha256", None)
    # JSON round-tripping ensures the generated structure does not share nested
    # mutable objects with the loaded canonical payload.
    payload["pairs"] = json.loads(json.dumps(pairs[:episodes]))
    payload["screening_provenance"] = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "canonical_manifest": str(canonical_path),
        "canonical_manifest_sha256": parent_hash,
        "prefix_start": 0,
        "prefix_stop": int(episodes),
    }
    payload["manifest_sha256"] = manifest_sha256(payload)
    return payload


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _matrix_cell_count(cfg: Any) -> int:
    """Count expanded cells without constructing thousands of OmegaConf nodes."""

    sweep = _plain(cfg.get("sweep", {}))
    exclusions = _plain(cfg.get("sweep_exclude", []))
    runs = _plain(cfg.get("runs", []))
    if not isinstance(sweep, dict) or not isinstance(exclusions, list) or not isinstance(runs, list):
        raise ScreeningPreparationError("Malformed runs, sweep, or sweep_exclude section.")
    axes = list(sweep)
    combinations = itertools.product(*(sweep[path] for path in axes))
    valid = 0
    for values in combinations:
        params = dict(zip(axes, values))
        if any(
            isinstance(exclusion, dict)
            and all(params.get(path) == value for path, value in exclusion.items())
            for exclusion in exclusions
        ):
            continue
        valid += 1
    return len(runs) * valid


def _validate_screen_config(canonical_cfg: Any, screen_cfg: Any) -> None:
    # Matrix indices and cell IDs are pure functions of these three sections.
    # Exact structural equality therefore proves identity without materializing
    # 6,240 heavyweight OmegaConf run nodes for every source config.
    for key in ("runs", "sweep", "sweep_exclude"):
        if _plain(canonical_cfg.get(key)) != _plain(screen_cfg.get(key)):
            raise ScreeningPreparationError(
                f"Screening generation changed the canonical {key} section."
            )
    screen_cells = _matrix_cell_count(screen_cfg)
    if screen_cells != NUM_MATRIX_CELLS:
        raise ScreeningPreparationError(
            f"Expected {NUM_MATRIX_CELLS} matrix cells, got {screen_cells}."
        )
    episodes = int(screen_cfg.run_defaults.eval.episodes)
    num_envs = int(screen_cfg.run_defaults.eval.num_envs)
    if episodes != SCREEN_EPISODES:
        raise ScreeningPreparationError(
            f"Screen config has eval.episodes={episodes}, expected {SCREEN_EPISODES}."
        )
    if num_envs != SCREEN_EPISODES:
        raise ScreeningPreparationError(
            "The release screen expects one unchanged 50-environment solver batch; "
            f"got eval.num_envs={num_envs}."
        )


def prepare_screen(
    canonical_config: str | Path,
    *,
    repo_root: str | Path,
    generated_dir: str | Path = DEFAULT_GENERATED_DIR,
    force: bool = False,
) -> PreparedScreen:
    root = Path(repo_root).resolve()
    canonical_config_path = _resolve_from_root(root, canonical_config).resolve()
    if not canonical_config_path.is_file():
        raise ScreeningPreparationError(
            f"Canonical benchmark config does not exist: {canonical_config_path}"
        )
    canonical_cfg = load_config(DEFAULTS, canonical_config_path)
    canonical_manifest_info = load_manifest_config(canonical_cfg)
    canonical_manifest_path = _resolve_from_root(
        root, canonical_manifest_info["path"]
    ).resolve()
    canonical_manifest = load_manifest(canonical_manifest_path)
    if len(canonical_manifest.get("pairs", [])) != 250:
        raise ScreeningPreparationError(
            f"Expected a 250-pair canonical manifest at {canonical_manifest_path}."
        )
    if int(canonical_cfg.run_defaults.eval.episodes) != 250:
        raise ScreeningPreparationError(
            f"Expected eval.episodes=250 in {canonical_config_path}."
        )

    target_dir = _resolve_from_root(root, generated_dir).resolve()
    screen_stem = f"{canonical_config_path.stem}_screen50"
    screen_config_path = target_dir / f"{screen_stem}.yaml"
    screen_manifest_path = target_dir / f"{screen_stem}_manifest.json"
    canonical_output_dir = str(canonical_cfg.output_dir)
    screen_output_dir = f"{canonical_output_dir}_screen50"

    canonical_config_ref = _relative_or_absolute(canonical_config_path, root)
    canonical_manifest_ref = _relative_or_absolute(canonical_manifest_path, root)
    screen_config_ref = _relative_or_absolute(screen_config_path, root)
    screen_manifest_ref = _relative_or_absolute(screen_manifest_path, root)

    screen_manifest = prefix_manifest(
        canonical_manifest,
        canonical_path=canonical_manifest_ref,
    )
    screen_manifest_text = json.dumps(screen_manifest, indent=2, sort_keys=True) + "\n"
    _write_idempotent(screen_manifest_path, screen_manifest_text, force=force)

    screen_payload = OmegaConf.to_container(canonical_cfg, resolve=True)
    if not isinstance(screen_payload, dict):
        raise ScreeningPreparationError("Canonical benchmark config must be a mapping.")
    screen_payload["output_dir"] = screen_output_dir
    screen_payload["title"] = f"{canonical_cfg.title} — 50-Episode Discovery Screen"
    screen_payload["manifest"] = {
        "group": f"{canonical_manifest_info['group']}_screen50",
        "path": screen_manifest_ref,
    }
    screen_payload["run_defaults"]["eval"]["episodes"] = SCREEN_EPISODES
    screen_payload["screening"] = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "phase": "discovery",
        "episodes": SCREEN_EPISODES,
        "held_out_episode_start": SCREEN_EPISODES,
        "canonical_episodes": 250,
        "canonical_config": canonical_config_ref,
        "canonical_config_sha256": file_sha256(canonical_config_path),
        "canonical_manifest": canonical_manifest_ref,
        "canonical_manifest_sha256": str(canonical_manifest["manifest_sha256"]),
        "selection_unit": "matrix_index",
    }
    screen_config_text = OmegaConf.to_yaml(OmegaConf.create(screen_payload))
    _write_idempotent(screen_config_path, screen_config_text, force=force)

    # Reload written assets, including hash validation, rather than validating
    # only the in-memory objects that produced them.
    loaded_screen_manifest = load_manifest(screen_manifest_path)
    if loaded_screen_manifest["pairs"] != canonical_manifest["pairs"][:SCREEN_EPISODES]:
        raise ScreeningPreparationError(
            f"Generated manifest is not the exact canonical prefix: {screen_manifest_path}"
        )
    screen_cfg = load_config(DEFAULTS, screen_config_path)
    _validate_screen_config(canonical_cfg, screen_cfg)
    loaded_screen_manifest_info = load_manifest_config(screen_cfg)
    if _resolve_from_root(root, loaded_screen_manifest_info["path"]).resolve() != screen_manifest_path:
        raise ScreeningPreparationError("Generated config does not reference its screen manifest.")

    return PreparedScreen(
        canonical_config=canonical_config_ref,
        canonical_config_sha256=file_sha256(canonical_config_path),
        canonical_manifest=canonical_manifest_ref,
        canonical_manifest_sha256=str(canonical_manifest["manifest_sha256"]),
        canonical_output_dir=canonical_output_dir,
        screen_config=screen_config_ref,
        screen_manifest=screen_manifest_ref,
        screen_manifest_sha256=str(loaded_screen_manifest["manifest_sha256"]),
        screen_output_dir=screen_output_dir,
        episodes=SCREEN_EPISODES,
        matrix_cells=NUM_MATRIX_CELLS,
    )


def prepare_all(
    *,
    repo_root: str | Path,
    generated_dir: str | Path = DEFAULT_GENERATED_DIR,
    canonical_configs: Iterable[str | Path] = RELEASE_CONFIGS,
    force: bool = False,
) -> list[PreparedScreen]:
    root = Path(repo_root).resolve()
    prepared = [
        prepare_screen(
            config,
            repo_root=root,
            generated_dir=generated_dir,
            force=force,
        )
        for config in canonical_configs
    ]
    if len(prepared) != 8:
        raise ScreeningPreparationError(
            f"The release workflow requires exactly eight matrices, got {len(prepared)}."
        )
    target_dir = _resolve_from_root(root, generated_dir).resolve()
    plan_path = target_dir / "screening_plan.json"
    plan = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "phase": "discovery",
        "episodes": SCREEN_EPISODES,
        "canonical_episodes": 250,
        "held_out_episode_range": [SCREEN_EPISODES, 250],
        "confirmation_contract": (
            "Selected matrix indices run in the unchanged canonical 250-episode "
            "outputs; primary confirmation statistics use episodes [50, 250)."
        ),
        "matrices": [asdict(item) for item in prepared],
    }
    _write_idempotent(
        plan_path,
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        force=force,
    )
    return prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate nested 50-episode discovery configs for the eight "
            "release20260728 schedule matrices. This does not submit jobs."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (default: inferred from this module).",
    )
    parser.add_argument(
        "--generated-dir",
        default=str(DEFAULT_GENERATED_DIR),
        help="Directory for generated configs/manifests and screening_plan.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace stale generated assets after canonical inputs change.",
    )
    parser.add_argument(
        "--collect-todo",
        action="store_true",
        help=(
            "Also scan completed canonical/screen cells, pin immutable discovery "
            "sources, and write deterministic TODO index ledgers."
        ),
    )
    args = parser.parse_args(argv)
    prepared = prepare_all(
        repo_root=args.repo_root,
        generated_dir=args.generated_dir,
        force=args.force,
    )
    for item in prepared:
        storage = screen_storage_preflight(
            item.canonical_output_dir,
            item.screen_output_dir,
            repo_root=args.repo_root,
        )
        storage_label = "ready" if storage["ready"] else "NOT READY"
        print(
            f"{item.screen_config}: {item.matrix_cells} cells -> "
            f"{item.screen_output_dir} [storage: {storage_label}]"
        )
        for issue in storage["issues"]:
            print(f"  storage preflight: {issue}")
    if args.collect_todo:
        from mwm.benchmark.screening_sources import collect_all_sources

        summary = collect_all_sources(
            prepared,
            repo_root=args.repo_root,
            generated_dir=args.generated_dir,
        )
        print(json.dumps(summary["totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GENERATED_DIR",
    "NUM_MATRIX_CELLS",
    "PreparedScreen",
    "RELEASE_CONFIGS",
    "SCREENING_SCHEMA_VERSION",
    "SCREEN_EPISODES",
    "ScreeningPreparationError",
    "prefix_manifest",
    "prepare_all",
    "prepare_screen",
    "screen_storage_preflight",
]
