from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterator, Literal

from mwm.io import load_json


EVAL_ARTIFACT_SCHEMA = "mwm.benchmark.eval_artifact"
EVAL_ARTIFACT_VERSION = 2
ARCHIVE_SCHEMA = "mwm.benchmark.eval_archive"
ARCHIVE_VERSION = 1
PLANNING_REF_SCHEMA = "mwm.benchmark.planning_diagnostics_ref"
PLANNING_REF_VERSION = 1
MAX_CAPSULE_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
COMPLETION_FILES = (
    "eval.json",
    "resolved_config.yaml",
    "metrics.jsonl",
    "summary.json",
    "planning_diagnostics.json",
    "episode_traces.jsonl",
)
VerifyLevel = Literal["metadata", "compressed_hash", "full"]


class EvalArtifactError(RuntimeError):
    """Raised when a benchmark evaluation artifact is malformed or unsafe."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=True,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # Review-media indexes are intentionally mutable and are not scientific
    # evaluation output. Everything else is bound by the scientific hash.
    return {key: value for key, value in payload.items() if key != "review_media"}


def _scientific_sha256(payload: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(_scientific_payload(payload)))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), mode)
            json.dump(payload, handle, allow_nan=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        retryable = {errno.EAGAIN, errno.EWOULDBLOCK}
        for attempt, delay in enumerate((0.0, 0.05, 0.25, 1.0), start=1):
            if delay:
                time.sleep(delay)
            try:
                os.fsync(fd)
                return
            except OSError as exc:
                if exc.errno not in retryable or attempt == 4:
                    raise
    finally:
        os.close(fd)


def _atomic_replace(source: Path, destination: Path) -> None:
    retryable = {errno.EAGAIN, errno.EWOULDBLOCK}
    for attempt, delay in enumerate((0.0, 0.05, 0.25, 1.0), start=1):
        if delay:
            time.sleep(delay)
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if exc.errno not in retryable or attempt == 4:
                raise


@contextmanager
def eval_artifact_lock(run_dir: str | Path) -> Iterator[None]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".eval_artifact.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _artifact_marker(payload: dict[str, Any]) -> dict[str, Any] | None:
    marker = payload.get("_artifact")
    if marker is None:
        return None
    if not isinstance(marker, dict):
        raise EvalArtifactError("eval artifact marker is not a mapping")
    if marker.get("schema") != EVAL_ARTIFACT_SCHEMA:
        raise EvalArtifactError(f"unsupported eval artifact schema {marker.get('schema')!r}")
    return dict(marker)


def is_compressed_eval_payload(payload: dict[str, Any]) -> bool:
    return _artifact_marker(payload) is not None


def _validate_marker(eval_path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    marker = _artifact_marker(payload)
    if marker is None:
        raise EvalArtifactError(f"{eval_path}: not a compressed benchmark eval capsule")
    if int(marker.get("version", -1)) != EVAL_ARTIFACT_VERSION:
        raise EvalArtifactError(f"{eval_path}: unsupported eval artifact version {marker.get('version')!r}")
    if marker.get("representation") != "capsule+archive":
        raise EvalArtifactError(f"{eval_path}: unsupported representation {marker.get('representation')!r}")
    archive = marker.get("archive")
    if not isinstance(archive, dict):
        raise EvalArtifactError(f"{eval_path}: artifact archive metadata is not a mapping")
    raw_path = str(archive.get("path", ""))
    if not raw_path or Path(raw_path).is_absolute():
        raise EvalArtifactError(f"{eval_path}: archive path must be a relative filename")
    archive_candidate = eval_path.parent / raw_path
    if Path(raw_path).name != raw_path or archive_candidate.is_symlink():
        raise EvalArtifactError(f"{eval_path}: archive path must be a direct, non-symlink filename")
    archive_path = archive_candidate.resolve()
    try:
        archive_path.relative_to(eval_path.parent.resolve())
    except ValueError as exc:
        raise EvalArtifactError(f"{eval_path}: archive path escapes the completed cell") from exc
    if archive_path.parent != eval_path.parent.resolve():
        raise EvalArtifactError(f"{eval_path}: archive must live directly beside eval.json")
    if not archive_path.is_file():
        raise EvalArtifactError(f"{eval_path}: missing or unsafe archive {archive_path}")
    expected_bytes = int(archive.get("compressed_bytes", -1))
    if expected_bytes < 0 or archive_path.stat().st_size != expected_bytes:
        raise EvalArtifactError(f"{eval_path}: compressed archive size mismatch")
    return marker, archive_path


def _decompress_archive(path: Path, codec: str, *, max_output_bytes: int) -> bytes:
    if max_output_bytes < 0 or max_output_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise EvalArtifactError(f"unsafe declared archive size {max_output_bytes}")
    if codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
            raise EvalArtifactError("zstandard is required to read this benchmark artifact") from exc
        with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
            raw = reader.read(max_output_bytes + 1)
            if len(raw) > max_output_bytes:
                raise EvalArtifactError(f"{path}: decompressed archive exceeds its declared size")
            return raw
    if codec == "gzip":
        with gzip.open(path, "rb") as handle:
            raw = handle.read(max_output_bytes + 1)
            if len(raw) > max_output_bytes:
                raise EvalArtifactError(f"{path}: decompressed archive exceeds its declared size")
            return raw
    raise EvalArtifactError(f"unsupported benchmark archive codec {codec!r}")


def _load_archive(eval_path: Path, marker: dict[str, Any], archive_path: Path) -> dict[str, Any]:
    archive_meta = dict(marker["archive"])
    compressed_sha = _sha256_file(archive_path)
    if compressed_sha != str(archive_meta.get("compressed_sha256", "")):
        raise EvalArtifactError(f"{eval_path}: compressed archive SHA-256 mismatch")
    uncompressed_bytes = int(archive_meta.get("uncompressed_bytes", -1))
    raw = _decompress_archive(
        archive_path,
        str(archive_meta.get("codec", "")),
        max_output_bytes=uncompressed_bytes,
    )
    if len(raw) != uncompressed_bytes:
        raise EvalArtifactError(f"{eval_path}: uncompressed archive size mismatch")
    if _sha256_bytes(raw) != str(archive_meta.get("uncompressed_sha256", "")):
        raise EvalArtifactError(f"{eval_path}: uncompressed archive SHA-256 mismatch")
    try:
        envelope = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvalArtifactError(f"{eval_path}: archive does not contain valid JSON") from exc
    if not isinstance(envelope, dict):
        raise EvalArtifactError(f"{eval_path}: archive JSON is not an object")
    if envelope.get("schema") != ARCHIVE_SCHEMA or int(envelope.get("version", -1)) != ARCHIVE_VERSION:
        raise EvalArtifactError(f"{eval_path}: unsupported archive schema/version")
    sections = envelope.get("sections")
    if not isinstance(sections, dict):
        raise EvalArtifactError(f"{eval_path}: archive sections are not a mapping")
    return dict(sections)


def _materialize_batches(
    batches: Any,
    planning_diagnostics: dict[str, Any],
    *,
    eval_path: Path,
) -> list[dict[str, Any]]:
    if not isinstance(batches, list):
        raise EvalArtifactError(f"{eval_path}: archived batches are not a list")
    top_trace = planning_diagnostics.get("trace", [])
    if not isinstance(top_trace, list):
        raise EvalArtifactError(f"{eval_path}: canonical planning trace is not a list")
    materialized: list[dict[str, Any]] = []
    for raw_batch in batches:
        if not isinstance(raw_batch, dict):
            raise EvalArtifactError(f"{eval_path}: archived batch is not a mapping")
        batch = dict(raw_batch)
        trace_ref = batch.pop("_planning_trace_ref", None)
        if trace_ref is not None:
            if not isinstance(trace_ref, dict):
                raise EvalArtifactError(f"{eval_path}: malformed batch planning trace reference")
            start = int(trace_ref.get("start", -1))
            stop = int(trace_ref.get("stop", -1))
            if start < 0 or stop < start or stop > len(top_trace):
                raise EvalArtifactError(f"{eval_path}: batch planning trace slice is out of range")
            diagnostics = batch.get("planning_diagnostics")
            if not isinstance(diagnostics, dict) or "trace" in diagnostics:
                raise EvalArtifactError(f"{eval_path}: referenced batch diagnostics are malformed")
            diagnostics = dict(diagnostics)
            diagnostics["trace"] = top_trace[start:stop]
            batch["planning_diagnostics"] = diagnostics
        materialized.append(batch)
    return materialized


def load_eval_capsule(eval_path: str | Path, *, verify: VerifyLevel = "metadata") -> dict[str, Any]:
    path = Path(eval_path)
    payload = load_json(path)
    marker = _artifact_marker(payload)
    if marker is None:
        return payload
    marker, archive_path = _validate_marker(path, payload)
    if verify in {"compressed_hash", "full"}:
        expected = str(marker["archive"].get("compressed_sha256", ""))
        if not expected or _sha256_file(archive_path) != expected:
            raise EvalArtifactError(f"{path}: compressed archive SHA-256 mismatch")
    if verify == "full":
        _load_archive(path, marker, archive_path)
    return payload


def load_eval_artifact(eval_path: str | Path, *, verify: VerifyLevel = "full") -> dict[str, Any]:
    path = Path(eval_path)
    capsule = load_json(path)
    marker = _artifact_marker(capsule)
    if marker is None:
        return capsule
    marker, archive_path = _validate_marker(path, capsule)
    if verify == "metadata":
        raise EvalArtifactError("full eval materialization requires archive verification")
    if verify == "compressed_hash":
        expected = str(marker["archive"].get("compressed_sha256", ""))
        if not expected or _sha256_file(archive_path) != expected:
            raise EvalArtifactError(f"{path}: compressed archive SHA-256 mismatch")
    sections = _load_archive(path, marker, archive_path)
    diagnostics = sections.get("planning_diagnostics")
    if not isinstance(diagnostics, dict):
        raise EvalArtifactError(f"{path}: archive lacks planning_diagnostics")
    payload = {key: value for key, value in capsule.items() if key not in {"_artifact", "planning_diagnostics"}}
    payload["planning_diagnostics"] = diagnostics
    payload["batches"] = _materialize_batches(sections.get("batches"), diagnostics, eval_path=path)
    review_rollouts = sections.get("review_rollouts")
    if not isinstance(review_rollouts, list):
        raise EvalArtifactError(f"{path}: archive review_rollouts are not a list")
    payload["review_rollouts"] = review_rollouts
    if _scientific_sha256(payload) != str(marker.get("scientific_payload_sha256", "")):
        raise EvalArtifactError(f"{path}: reconstructed scientific payload SHA-256 mismatch")
    expected_sections = marker.get("section_sha256", {})
    if not isinstance(expected_sections, dict):
        raise EvalArtifactError(f"{path}: section hashes are not a mapping")
    for key in ("planning_diagnostics", "batches", "review_rollouts"):
        if _sha256_bytes(_canonical_bytes(payload[key])) != str(expected_sections.get(key, "")):
            raise EvalArtifactError(f"{path}: reconstructed section hash mismatch for {key}")
    return payload


def inspect_eval_artifact(eval_path: str | Path, *, verify: VerifyLevel = "metadata") -> dict[str, Any]:
    path = Path(eval_path)
    capsule = load_json(path)
    marker = _artifact_marker(capsule)
    if marker is None:
        return {"representation": "legacy", "eval_path": str(path), "eval_bytes": path.stat().st_size}
    marker, archive_path = _validate_marker(path, capsule)
    if verify in {"compressed_hash", "full"}:
        expected = str(marker["archive"].get("compressed_sha256", ""))
        if not expected or _sha256_file(archive_path) != expected:
            raise EvalArtifactError(f"{path}: compressed archive SHA-256 mismatch")
    if verify == "full":
        load_eval_artifact(path, verify="full")
    return {
        "representation": "capsule+archive",
        "eval_path": str(path),
        "eval_bytes": path.stat().st_size,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": str(marker["archive"]["compressed_sha256"]),
    }


def eval_artifact_signature(eval_path: str | Path) -> tuple[Any, ...]:
    path = Path(eval_path)
    try:
        stat = path.stat()
    except OSError:
        return (None,)
    signature: list[Any] = [(stat.st_mtime_ns, stat.st_size)]
    if stat.st_size > MAX_CAPSULE_BYTES:
        return tuple(signature)
    try:
        payload = load_json(path)
        marker = _artifact_marker(payload)
        if marker is None:
            return tuple(signature)
        _, archive_path = _validate_marker(path, payload)
        archive_stat = archive_path.stat()
        signature.append((archive_path.name, archive_stat.st_mtime_ns, archive_stat.st_size))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, EvalArtifactError):
        signature.append(("invalid",))
    return tuple(signature)


def validate_eval_storage_reference(eval_path: str | Path, *, verify: VerifyLevel = "metadata") -> bool:
    path = Path(eval_path)
    try:
        stat = path.stat()
        if stat.st_size > MAX_CAPSULE_BYTES:
            return True
        payload = load_json(path)
        if _artifact_marker(payload) is None:
            return True
        load_eval_capsule(path, verify=verify)
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError, EvalArtifactError):
        return False


def _planning_ref_payload(eval_path: Path, diagnostics: dict[str, Any], section_sha256: str) -> dict[str, Any]:
    aggregate = {key: value for key, value in diagnostics.items() if key != "trace"}
    return {
        "_artifact": {
            "schema": PLANNING_REF_SCHEMA,
            "version": PLANNING_REF_VERSION,
            "eval_path": eval_path.name,
            "json_pointer": "/planning_diagnostics",
            "section_sha256": section_sha256,
        },
        **aggregate,
        "trace": {
            "$artifact_ref": f"{eval_path.name}#/planning_diagnostics/trace",
            "length": len(diagnostics.get("trace", [])),
        },
    }


def _planning_ref_marker(payload: dict[str, Any]) -> dict[str, Any] | None:
    marker = payload.get("_artifact")
    if not isinstance(marker, dict) or marker.get("schema") != PLANNING_REF_SCHEMA:
        return None
    if int(marker.get("version", -1)) != PLANNING_REF_VERSION:
        raise EvalArtifactError(f"unsupported planning diagnostics ref version {marker.get('version')!r}")
    return dict(marker)


def load_planning_diagnostics(run_dir: str | Path, *, verify: VerifyLevel = "full") -> dict[str, Any]:
    root = Path(run_dir)
    sidecar_path = root / "planning_diagnostics.json"
    sidecar = load_json(sidecar_path)
    marker = _planning_ref_marker(sidecar)
    if marker is None:
        return sidecar
    eval_name = str(marker.get("eval_path", ""))
    if Path(eval_name).name != eval_name or eval_name != "eval.json":
        raise EvalArtifactError(f"{sidecar_path}: unsafe eval reference {eval_name!r}")
    diagnostics = load_eval_artifact(root / eval_name, verify=verify).get("planning_diagnostics")
    if not isinstance(diagnostics, dict):
        raise EvalArtifactError(f"{sidecar_path}: referenced planning diagnostics are missing")
    if _sha256_bytes(_canonical_bytes(diagnostics)) != str(marker.get("section_sha256", "")):
        raise EvalArtifactError(f"{sidecar_path}: planning diagnostics section hash mismatch")
    return dict(diagnostics)


def planning_sidecar_matches_capsule(run_dir: str | Path, capsule: dict[str, Any]) -> bool:
    root = Path(run_dir)
    sidecar = load_json(root / "planning_diagnostics.json")
    marker = _artifact_marker(capsule)
    if marker is None:
        return sidecar == capsule.get("planning_diagnostics", {})
    expected = str(dict(marker.get("section_sha256", {})).get("planning_diagnostics", ""))
    ref = _planning_ref_marker(sidecar)
    if ref is not None:
        return (
            str(ref.get("eval_path", "")) == "eval.json"
            and str(ref.get("json_pointer", "")) == "/planning_diagnostics"
            and str(ref.get("section_sha256", "")) == expected
        )
    return _sha256_bytes(_canonical_bytes(sidecar)) == expected


def update_eval_capsule(
    eval_path: str | Path,
    updater: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    path = Path(eval_path)
    with eval_artifact_lock(path.parent):
        payload = load_json(path)
        immutable_before = {key: value for key, value in payload.items() if key != "review_media"}
        updater(payload)
        immutable_after = {key: value for key, value in payload.items() if key != "review_media"}
        if _canonical_bytes(immutable_before) != _canonical_bytes(immutable_after):
            raise EvalArtifactError("eval capsule updates may change only review_media")
        if _artifact_marker(payload) is not None:
            if len(_canonical_bytes(payload)) > MAX_CAPSULE_BYTES:
                raise EvalArtifactError(f"{path}: compressed eval capsule is unexpectedly large")
            _validate_marker(path, payload)
        _atomic_write_json(path, payload)
        return payload


def _canonicalize_batches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    top_trace = payload.get("planning_diagnostics", {}).get("trace", [])
    if not isinstance(top_trace, list):
        raise EvalArtifactError("top-level planning diagnostics trace is not a list")
    raw_batches = payload.get("batches")
    if not isinstance(raw_batches, list):
        raise EvalArtifactError("eval payload batches are not a list")
    batches: list[dict[str, Any]] = []
    offset = 0
    for raw_batch in raw_batches:
        if not isinstance(raw_batch, dict):
            raise EvalArtifactError("eval payload contains a non-mapping batch")
        batch = dict(raw_batch)
        diagnostics = batch.get("planning_diagnostics")
        if not isinstance(diagnostics, dict):
            batches.append(batch)
            continue
        trace = diagnostics.get("trace")
        if not isinstance(trace, list):
            batches.append(batch)
            continue
        stop = offset + len(trace)
        if top_trace[offset:stop] == trace:
            batch["planning_diagnostics"] = {key: value for key, value in diagnostics.items() if key != "trace"}
            batch["_planning_trace_ref"] = {"start": offset, "stop": stop}
            offset = stop
        else:
            # Preserve a scientifically distinct batch trace verbatim.
            batches.append(batch)
            continue
        batches.append(batch)
    return batches


def _capsule_and_envelope(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    diagnostics = payload.get("planning_diagnostics")
    batches = payload.get("batches")
    review_rollouts = payload.get("review_rollouts")
    if not isinstance(diagnostics, dict) or not isinstance(batches, list) or not isinstance(review_rollouts, list):
        raise EvalArtifactError("eval payload lacks benchmark planning/batch/review sections")
    canonical_batches = _canonicalize_batches(payload)
    sections = {
        "planning_diagnostics": diagnostics,
        "batches": canonical_batches,
        "review_rollouts": review_rollouts,
    }
    envelope = {"schema": ARCHIVE_SCHEMA, "version": ARCHIVE_VERSION, "sections": sections}
    archived_keys = {"planning_diagnostics", "batches", "review_rollouts"}
    capsule = {key: value for key, value in payload.items() if key not in archived_keys}
    aggregate = {key: value for key, value in diagnostics.items() if key != "trace"}
    capsule["planning_diagnostics"] = {
        **aggregate,
        "trace": {
            "$artifact_ref": "archive#/sections/planning_diagnostics/trace",
            "length": len(diagnostics.get("trace", [])),
        },
    }
    # Section hashes describe the fully reconstructed legacy-shaped sections,
    # not the canonical archive's internal trace references.
    section_hashes = {
        "planning_diagnostics": _sha256_bytes(_canonical_bytes(diagnostics)),
        "batches": _sha256_bytes(_canonical_bytes(batches)),
        "review_rollouts": _sha256_bytes(_canonical_bytes(review_rollouts)),
    }
    return capsule, envelope, section_hashes


def _write_zstd_temp(path: Path, raw: bytes, *, mode: int) -> tuple[Path, int, str]:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise EvalArtifactError("zstandard is required to create compressed benchmark artifacts") from exc
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            with zstandard.ZstdCompressor(level=6).stream_writer(handle, closefd=False) as writer:
                writer.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path, temp_path.stat().st_size, _sha256_file(temp_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def compress_completed_eval(run_dir: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    root = Path(run_dir)
    missing = [name for name in COMPLETION_FILES if not (root / name).is_file()]
    if missing:
        return {"status": "partial", "run_dir": str(root), "missing": missing, "reclaimed_bytes": 0}
    with eval_artifact_lock(root):
        missing = [name for name in COMPLETION_FILES if not (root / name).is_file()]
        if missing:
            return {"status": "partial", "run_dir": str(root), "missing": missing, "reclaimed_bytes": 0}
        eval_path = root / "eval.json"
        before_bytes = eval_path.stat().st_size + (root / "planning_diagnostics.json").stat().st_size
        raw_payload = load_json(eval_path)
        marker = _artifact_marker(raw_payload)
        if marker is not None:
            expanded = load_eval_artifact(eval_path, verify="full")
            section_sha = str(dict(marker.get("section_sha256", {})).get("planning_diagnostics", ""))
            sidecar = load_json(root / "planning_diagnostics.json")
            if _planning_ref_marker(sidecar) is None:
                if _sha256_bytes(_canonical_bytes(sidecar)) != section_sha:
                    raise EvalArtifactError(f"{root}: legacy planning sidecar differs from compressed eval")
                if not dry_run:
                    _atomic_write_json(
                        root / "planning_diagnostics.json",
                        _planning_ref_payload(eval_path, expanded["planning_diagnostics"], section_sha),
                    )
                return {
                    "status": "would_repair" if dry_run else "repaired",
                    "run_dir": str(root),
                    "reclaimed_bytes": (
                        0
                        if dry_run
                        else before_bytes
                        - eval_path.stat().st_size
                        - (root / "planning_diagnostics.json").stat().st_size
                    ),
                }
            load_planning_diagnostics(root, verify="full")
            return {"status": "already_compressed", "run_dir": str(root), "reclaimed_bytes": 0}

        payload = dict(raw_payload)
        if "policy_diagnostics" in payload:
            if payload.get("policy_diagnostics") != payload.get("planning_diagnostics"):
                raise EvalArtifactError(f"{eval_path}: policy diagnostics differ from planning diagnostics")
            payload.pop("policy_diagnostics")
        diagnostics_sidecar = load_json(root / "planning_diagnostics.json")
        if diagnostics_sidecar != payload.get("planning_diagnostics"):
            raise EvalArtifactError(f"{root}: planning diagnostics sidecar differs from eval payload")
        capsule, envelope, section_hashes = _capsule_and_envelope(payload)
        if dry_run:
            return {"status": "would_compress", "run_dir": str(root), "reclaimed_bytes": 0}

        scientific_sha = _scientific_sha256(payload)
        archive_raw = _canonical_bytes(envelope)
        archive_name = f"eval.details.{scientific_sha[:16]}.json.zst"
        archive_path = root / archive_name
        archive_existed = archive_path.exists()
        temp_path: Path | None = None
        if archive_existed:
            if archive_path.is_symlink():
                raise EvalArtifactError(f"{archive_path}: refusing an existing symlink archive")
            compressed_bytes = archive_path.stat().st_size
            compressed_sha = _sha256_file(archive_path)
            raw_check = _decompress_archive(archive_path, "zstd", max_output_bytes=len(archive_raw))
            if raw_check != archive_raw:
                raise EvalArtifactError(f"{archive_path}: existing crash-recovery archive has different content")
        else:
            temp_path, compressed_bytes, compressed_sha = _write_zstd_temp(
                archive_path,
                archive_raw,
                mode=eval_path.stat().st_mode & 0o777,
            )
            if _decompress_archive(temp_path, "zstd", max_output_bytes=len(archive_raw)) != archive_raw:
                temp_path.unlink(missing_ok=True)
                raise EvalArtifactError(f"{root}: zstd archive verification failed")
            _atomic_replace(temp_path, archive_path)
            temp_path = None
            _fsync_directory(root)
        capsule["_artifact"] = {
            "schema": EVAL_ARTIFACT_SCHEMA,
            "version": EVAL_ARTIFACT_VERSION,
            "representation": "capsule+archive",
            "canonicalization_version": 1,
            "scientific_payload_sha256": scientific_sha,
            "section_sha256": section_hashes,
            "archive": {
                "path": archive_name,
                "codec": "zstd",
                "compressed_bytes": compressed_bytes,
                "compressed_sha256": compressed_sha,
                "uncompressed_bytes": len(archive_raw),
                "uncompressed_sha256": _sha256_bytes(archive_raw),
            },
        }
        try:
            if len(_canonical_bytes(capsule)) > MAX_CAPSULE_BYTES:
                raise EvalArtifactError(f"{root}: eval capsule exceeds the {MAX_CAPSULE_BYTES}-byte safety limit")
            # The archive is durable first; eval.json is the atomic commit marker.
            _atomic_write_json(eval_path, capsule)
            loaded = load_eval_artifact(eval_path, verify="full")
            if _scientific_sha256(loaded) != scientific_sha:
                raise EvalArtifactError(f"{root}: post-commit eval validation failed")
            # The old full sidecar remains valid during the preceding window.
            _atomic_write_json(
                root / "planning_diagnostics.json",
                _planning_ref_payload(
                    eval_path,
                    loaded["planning_diagnostics"],
                    section_hashes["planning_diagnostics"],
                ),
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        before_total = before_bytes + (compressed_bytes if archive_existed else 0)
        after_bytes = (
            eval_path.stat().st_size
            + archive_path.stat().st_size
            + (root / "planning_diagnostics.json").stat().st_size
        )
        return {
            "status": "compressed",
            "run_dir": str(root),
            "archive_path": str(archive_path),
            "reclaimed_bytes": before_total - after_bytes,
            "before_bytes": before_total,
            "after_bytes": after_bytes,
        }


__all__ = [
    "ARCHIVE_SCHEMA",
    "COMPLETION_FILES",
    "EVAL_ARTIFACT_SCHEMA",
    "EVAL_ARTIFACT_VERSION",
    "EvalArtifactError",
    "compress_completed_eval",
    "eval_artifact_lock",
    "eval_artifact_signature",
    "inspect_eval_artifact",
    "is_compressed_eval_payload",
    "load_eval_artifact",
    "load_eval_capsule",
    "load_planning_diagnostics",
    "planning_sidecar_matches_capsule",
    "update_eval_capsule",
    "validate_eval_storage_reference",
]
