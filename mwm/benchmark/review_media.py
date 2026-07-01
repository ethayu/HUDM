from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Iterable

from mwm.eval.videos import collect_video_paths
from mwm.eval.review_trace import fidelity_trace_from_planning_trace
from mwm.io import jsonable, load_json, write_json


RENDERER_VERSION = "review_media_v1"
MEDIA_DIRNAME = "review_media"


class ReviewMediaUnsupported(RuntimeError):
    """Raised when a rollout cannot be rendered from available artifacts."""


@dataclass(frozen=True)
class RenderedMedia:
    kind: str
    path: str
    source_trace_type: str
    warnings: list[str]


def rollout_key(episode_index: int) -> str:
    return f"episode_{int(episode_index):04d}"


def rollout_media_dir(eval_path: str | Path, episode_index: int) -> Path:
    return Path(eval_path).parent / MEDIA_DIRNAME / rollout_key(episode_index)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write_json(tmp, payload)
    tmp.replace(path)


def rollout_by_index(payload: dict[str, Any], episode_index: int) -> dict[str, Any]:
    for rollout in payload.get("review_rollouts", []):
        if int(rollout.get("episode_index", -1)) == int(episode_index):
            return dict(rollout)
    raise KeyError(f"eval payload has no review_rollouts entry for episode_index={int(episode_index)}")


def record_review_media(
    eval_path: str | Path,
    *,
    episode_index: int,
    kind: str,
    path: str | Path,
    source_trace_type: str,
    warnings: Iterable[str] | None = None,
) -> dict[str, Any]:
    eval_file = Path(eval_path)
    payload = load_json(eval_file)
    media = payload.setdefault("review_media", {})
    media.setdefault("renderer_version", RENDERER_VERSION)
    media["updated_at"] = datetime.now(timezone.utc).isoformat()
    rollouts = media.setdefault("rollouts", {})
    entry = rollouts.setdefault(rollout_key(episode_index), {})
    entry[str(kind)] = {
        "kind": str(kind),
        "path": str(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "renderer_version": RENDERER_VERSION,
        "source_trace_type": str(source_trace_type),
        "warnings": [str(w) for w in (warnings or [])],
    }
    _atomic_write_json(eval_file, payload)
    return payload


def review_media_entry(payload: dict[str, Any], episode_index: int, kind: str) -> dict[str, Any] | None:
    return (
        payload.get("review_media", {})
        .get("rollouts", {})
        .get(rollout_key(episode_index), {})
        .get(str(kind))
    )


def existing_media_path(payload: dict[str, Any], episode_index: int, kind: str) -> Path | None:
    entry = review_media_entry(payload, episode_index, kind)
    if not isinstance(entry, dict) or not entry.get("path"):
        return None
    path = Path(str(entry["path"]))
    return path if path.is_file() else None


class FixedActionPolicy:
    def __init__(self, action_trace: list[Any]) -> None:
        self.action_trace = list(action_trace)
        self._idx = 0

    def set_env(self, env: Any) -> None:
        self.env = env

    def reset_trace(self) -> None:
        self._idx = 0

    def get_action(self, info_dict: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        del info_dict, kwargs
        if self._idx >= len(self.action_trace):
            raise RuntimeError(
                f"fixed action replay exhausted action_trace after {self._idx} primitive action(s)"
            )
        action = self.action_trace[self._idx]
        self._idx += 1
        try:
            import numpy as np

            arr = np.asarray(action, dtype=np.float32)
            return arr[None, ...] if arr.ndim == 1 else arr
        except Exception:
            return [action]


def _resolved_config_for_eval(eval_path: Path) -> Path:
    cfg_path = eval_path.parent / "resolved_config.yaml"
    if not cfg_path.is_file():
        raise ReviewMediaUnsupported(f"missing resolved config for review rendering: {cfg_path}")
    return cfg_path


def render_environment_video(
    eval_path: str | Path,
    *,
    episode_index: int,
    force: bool = False,
) -> RenderedMedia:
    eval_file = Path(eval_path)
    payload = load_json(eval_file)
    existing = existing_media_path(payload, episode_index, "env")
    if existing is not None and not force:
        return RenderedMedia("env", str(existing), "action_trace", [])
    rollout = rollout_by_index(payload, episode_index)
    action_trace = rollout.get("action_trace")
    if not isinstance(action_trace, list) or not action_trace:
        raise ReviewMediaUnsupported("exact environment rendering requires review_rollouts[].action_trace")

    from mwm.eval.runtime import load_eval_runtime
    from mwm.swm.envs import make_swm_world, parse_env_kwargs
    from omegaconf import OmegaConf

    runtime = load_eval_runtime(str(_resolved_config_for_eval(eval_file)))
    out_dir = rollout_media_dir(eval_file, episode_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "env.mp4"
    tmp_video_dir = out_dir / "env_tmp"
    shutil.rmtree(tmp_video_dir, ignore_errors=True)
    try:
        env_kwargs = parse_env_kwargs(OmegaConf.to_container(runtime.cfg.env.get("kwargs", {}), resolve=True))
        world = make_swm_world(
            runtime.env_id,
            num_envs=1,
            image_shape=runtime.image_shape,
            max_episode_steps=int(runtime.cfg.env.max_episode_steps),
            goal_conditioned=bool(runtime.cfg.env.goal_conditioned),
            env_kwargs=env_kwargs,
        )
        try:
            policy = FixedActionPolicy(action_trace)
            world.set_policy(policy)
            world.evaluate(
                dataset=runtime.dataset,
                episodes_idx=[int(rollout.get("dataset_episode", rollout.get("episode", 0)))],
                start_steps=[int(rollout.get("start_step", 0))],
                eval_budget=min(int(runtime.cfg.eval.budget), len(action_trace)),
                callables=runtime.eval_callables,
                goal_offset=int(runtime.cfg.eval.goal_offset),
                video=str(tmp_video_dir),
            )
        finally:
            world.close()
        videos = collect_video_paths(tmp_video_dir)
        if not videos:
            raise ReviewMediaUnsupported("SWM environment did not produce a supported video")
        shutil.move(str(videos[0]), target)
    finally:
        runtime.close()
        shutil.rmtree(tmp_video_dir, ignore_errors=True)

    record_review_media(
        eval_file,
        episode_index=episode_index,
        kind="env",
        path=target,
        source_trace_type="action_trace",
        warnings=[],
    )
    return RenderedMedia("env", str(target), "action_trace", [])


def _dataset_frame(dataset: Any, row: int, pixels_key: str) -> Any:
    def first_frame(value: Any) -> Any:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 4:
            return value[0]
        if isinstance(value, list) and value:
            return value[0]
        return value

    try:
        item = dataset[int(row)]
        if isinstance(item, dict) and str(pixels_key) in item:
            return first_frame(item[str(pixels_key)])
        if not isinstance(item, dict):
            return first_frame(item)
    except Exception:
        pass
    if hasattr(dataset, "get_col_data"):
        values = dataset.get_col_data(str(pixels_key))
        return first_frame(values[int(row)])
    raise KeyError(f"dataset row {int(row)} did not contain pixels key {pixels_key!r}")


def _to_bchw_float(image: Any, device: Any) -> Any:
    import torch
    import torch.nn.functional as F

    from mwm.preprocessing.images import MWM_IMAGE_SIZE, image_tensor_to_bchw

    tensor = torch.as_tensor(image, device=device)
    if not tensor.is_floating_point():
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
        if tensor.numel() and float(tensor.detach().max().cpu().item()) > 2.0:
            tensor = tensor / 255.0
    tensor = image_tensor_to_bchw(tensor)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape[-2:]) != MWM_IMAGE_SIZE:
        tensor = F.interpolate(tensor, size=MWM_IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True)
    return tensor.clamp(0.0, 1.0)


def _to_hwc_uint8(image: Any) -> Any:
    import numpy as np
    import torch

    tensor = torch.as_tensor(image).detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
        tensor = tensor[:3].permute(1, 2, 0)
    arr = tensor.float().clamp(0.0, 1.0).numpy()
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr[..., :3] * 255.0).round().astype(np.uint8)


def _label_frame(frame: Any, label: str) -> Any:
    try:
        import cv2

        cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    except Exception:
        pass
    return frame


def _usable_decoder(model: Any, metadata: dict[str, Any]) -> bool:
    if not callable(getattr(model, "decode", None)):
        return False
    if str(metadata.get("adapter_family", "")).lower() == "prejepa":
        return False
    policy = metadata.get("component_policy", {})
    if isinstance(policy, dict) and policy.get("reconstructor") == []:
        return False
    decoders = getattr(model, "decoders", None)
    return decoders is None or len(decoders) > 0


def render_latent_reconstruction_video(
    eval_path: str | Path,
    *,
    episode_index: int,
    force: bool = False,
) -> RenderedMedia:
    eval_file = Path(eval_path)
    payload = load_json(eval_file)
    existing = existing_media_path(payload, episode_index, "latent_reconstruction")
    if existing is not None and not force:
        return RenderedMedia("latent_reconstruction", str(existing), "fidelity_trace", [])
    rollout = rollout_by_index(payload, episode_index)
    fidelity_trace = rollout.get("fidelity_trace")
    if not isinstance(fidelity_trace, list) or not fidelity_trace:
        raise ReviewMediaUnsupported("latent reconstruction requires review_rollouts[].fidelity_trace")

    from mwm.eval.runtime import load_eval_runtime

    runtime = load_eval_runtime(str(_resolved_config_for_eval(eval_file)))
    if not _usable_decoder(runtime.model, runtime.metadata):
        runtime.close()
        raise ReviewMediaUnsupported("checkpoint does not expose a usable latent decoder")

    out_dir = rollout_media_dir(eval_file, episode_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "latent_reconstruction.mp4"
    frames = []
    try:
        import imageio.v2 as imageio
        import numpy as np
        import torch

        pixels_key = str(runtime.cfg.data.get("pixels_key", "pixels"))
        start_row = int(rollout.get("start_row", 0))
        with torch.inference_mode():
            for item in fidelity_trace:
                t = int(item.get("t", 0))
                level_idx = int(item.get("level_idx", 0))
                actual = _to_bchw_float(_dataset_frame(runtime.dataset, start_row + t, pixels_key), runtime.device)
                encoded = runtime.model.encode(actual, already_preprocessed=False)
                latent = encoded.get("emb") if isinstance(encoded, dict) else encoded
                recon = runtime.model.decode(level_idx, latent).clamp(0.0, 1.0)
                left = _label_frame(_to_hwc_uint8(actual), f"actual t={t}")
                right = _label_frame(
                    _to_hwc_uint8(recon),
                    f"recon L={level_idx} K={int(item.get('K', level_idx))}",
                )
                if left.shape[:2] != right.shape[:2]:
                    try:
                        import cv2

                        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass
                frames.append(np.concatenate([left, right], axis=1))
        if not frames:
            raise ReviewMediaUnsupported("latent reconstruction produced no frames")
        imageio.mimsave(target, frames, fps=10)
    finally:
        runtime.close()

    record_review_media(
        eval_file,
        episode_index=episode_index,
        kind="latent_reconstruction",
        path=target,
        source_trace_type="fidelity_trace",
        warnings=[],
    )
    return RenderedMedia("latent_reconstruction", str(target), "fidelity_trace", [])


def render_rollout_media(
    eval_path: str | Path,
    *,
    episode_index: int,
    sources: Iterable[str],
    force: bool = False,
) -> dict[str, Any]:
    rendered: list[dict[str, Any]] = []
    warnings: list[str] = []
    normalized_sources = [str(source) for source in sources]
    if "both" in normalized_sources:
        normalized_sources = ["env", "latent"]
    for source in normalized_sources:
        try:
            if source == "env":
                media = render_environment_video(eval_path, episode_index=episode_index, force=force)
            elif source in {"latent", "latent_reconstruction"}:
                media = render_latent_reconstruction_video(eval_path, episode_index=episode_index, force=force)
            else:
                warnings.append(f"unknown media source {source!r}")
                continue
            rendered.append(
                {
                    "kind": media.kind,
                    "path": media.path,
                    "source_trace_type": media.source_trace_type,
                    "warnings": media.warnings,
                }
            )
        except ReviewMediaUnsupported as exc:
            warnings.append(f"{source}: {exc}")
    return {"episode_index": int(episode_index), "rendered": rendered, "warnings": warnings}


__all__ = [
    "ReviewMediaUnsupported",
    "RenderedMedia",
    "existing_media_path",
    "fidelity_trace_from_planning_trace",
    "record_review_media",
    "render_environment_video",
    "render_latent_reconstruction_video",
    "render_rollout_media",
    "review_media_entry",
    "rollout_by_index",
    "rollout_key",
    "rollout_media_dir",
]
