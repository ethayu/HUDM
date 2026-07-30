from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
import shutil
import math
from typing import Any, Callable, Iterable

from mwm.eval.videos import collect_video_paths
from mwm.eval.review_trace import fidelity_trace_from_planning_trace
from mwm.io import jsonable, load_json, write_json


RENDERER_VERSION = "review_media_v1"
MEDIA_DIRNAME = "review_media"
ProgressCallback = Callable[[str], None]


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


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(str(message))


def _finite_action(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_finite_action(item) for item in value)
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def valid_action_prefix(action_trace: Iterable[Any]) -> list[Any]:
    actions: list[Any] = []
    invalid_seen = False
    for action in action_trace:
        if _finite_action(action):
            if invalid_seen:
                raise ReviewMediaUnsupported("action_trace contains valid actions after an invalid/masked step")
            actions.append(action)
        else:
            invalid_seen = True
    return actions


@contextmanager
def _review_render_lock(eval_path: str | Path):
    import fcntl

    lock_dir = Path(eval_path).parent / MEDIA_DIRNAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / ".render.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


class CapturingFixedActionPolicy(FixedActionPolicy):
    """Replay stored actions while retaining the exact observations seen by policy."""

    def __init__(self, action_trace: list[Any]) -> None:
        super().__init__(action_trace)
        self.observations: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _snapshot(info_dict: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        snapshot: dict[str, Any] = {}
        for key in ("pixels", "action", "step_idx"):
            value = info_dict.get(key)
            if value is not None:
                snapshot[key] = np.asarray(value).copy()
        return snapshot

    def capture(self, step: int, info_dict: dict[str, Any]) -> None:
        self.observations[int(step)] = self._snapshot(info_dict)

    def get_action(self, info_dict: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if isinstance(info_dict, dict):
            self.capture(self._idx, info_dict)
        return super().get_action(info_dict, **kwargs)


def predictive_replan_segments(
    fidelity_trace: list[dict[str, Any]],
    *,
    action_count: int,
    action_block: int,
    receding_horizon: int,
) -> list[dict[str, Any]]:
    """Map executed primitive actions to complete scheduled rollout blocks."""

    block = max(1, int(action_block))
    replan_every = block * max(1, int(receding_horizon))
    trace_by_block: dict[tuple[int, int], dict[str, Any]] = {}
    for item in fidelity_trace:
        if not isinstance(item, dict):
            continue
        key = (int(item.get("replan_idx", -1)), int(item.get("block_idx", -1)))
        trace_by_block.setdefault(key, item)

    segments: list[dict[str, Any]] = []
    for anchor in range(0, max(0, int(action_count)), replan_every):
        replan_idx = anchor // replan_every
        available = min(replan_every, int(action_count) - anchor)
        complete_blocks = available // block
        if complete_blocks <= 0:
            continue
        blocks: list[dict[str, Any]] = []
        for block_idx in range(complete_blocks):
            trace = trace_by_block.get((replan_idx, block_idx))
            if trace is None:
                raise ReviewMediaUnsupported(
                    f"fidelity_trace is missing replan {replan_idx} block {block_idx}"
                )
            if trace.get("level_idx") is None:
                raise ReviewMediaUnsupported(
                    f"fidelity_trace has no resolved level for replan {replan_idx} block {block_idx}"
                )
            primitive_start = anchor + block_idx * block
            primitive_end = primitive_start + block
            blocks.append(
                {
                    "block_idx": int(block_idx),
                    "level_idx": int(trace["level_idx"]),
                    "K": int(trace.get("K", trace["level_idx"])),
                    "primitive_start": int(primitive_start),
                    "primitive_end": int(primitive_end),
                    "distance_since_anchor": int(primitive_end - anchor),
                }
            )
        segments.append(
            {
                "replan_idx": int(replan_idx),
                "anchor_step": int(anchor),
                "replan_every": int(replan_every),
                "blocks": blocks,
            }
        )
    return segments


def _captured_pixels(observations: dict[int, dict[str, Any]], step: int) -> Any:
    item = observations.get(int(step))
    if not isinstance(item, dict) or item.get("pixels") is None:
        raise ReviewMediaUnsupported(f"actual replay did not capture pixels at primitive step {int(step)}")
    return item["pixels"]


def _resolved_config_for_eval(eval_path: Path) -> Path:
    cfg_path = eval_path.parent / "resolved_config.yaml"
    if not cfg_path.is_file():
        raise ReviewMediaUnsupported(f"missing resolved config for review rendering: {cfg_path}")
    return cfg_path


def _replay_actual_observations(
    runtime: Any,
    rollout: dict[str, Any],
    actions: list[Any],
    *,
    progress: ProgressCallback | None = None,
) -> tuple[dict[int, dict[str, Any]], int]:
    from omegaconf import OmegaConf

    from mwm.swm.envs import make_swm_world, parse_env_kwargs

    env_kwargs = parse_env_kwargs(OmegaConf.to_container(runtime.cfg.env.get("kwargs", {}), resolve=True))
    world = make_swm_world(
        runtime.env_id,
        num_envs=1,
        image_shape=runtime.image_shape,
        max_episode_steps=int(runtime.cfg.env.max_episode_steps),
        goal_conditioned=bool(runtime.cfg.env.goal_conditioned),
        env_kwargs=env_kwargs,
    )
    policy = CapturingFixedActionPolicy(actions)
    try:
        world.set_policy(policy)
        _progress(progress, f"Replaying {len(actions)} stored action(s) to capture actual anchors")
        world.evaluate(
            dataset=runtime.dataset,
            episodes_idx=[int(rollout.get("dataset_episode", rollout.get("episode", 0)))],
            start_steps=[int(rollout.get("start_step", 0))],
            eval_budget=min(int(runtime.cfg.eval.budget), len(actions)),
            callables=runtime.eval_callables,
            goal_offset=int(runtime.cfg.eval.goal_offset),
        )
        policy.capture(policy._idx, world.infos)
        return policy.observations, int(policy._idx)
    finally:
        world.close()


def model_actions_for_rollout(
    rollout: dict[str, Any],
    runtime: Any,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[Any, str]:
    """Return primitive model-space actions, using a narrow legacy fallback."""

    import numpy as np

    env_actions = valid_action_prefix(rollout.get("action_trace") or [])
    saved = rollout.get("model_action_trace")
    if isinstance(saved, list) and saved:
        model_actions = valid_action_prefix(saved)
        if len(model_actions) < len(env_actions):
            raise ReviewMediaUnsupported(
                f"model_action_trace has {len(model_actions)} actions but replay executed {len(env_actions)}"
            )
        return np.asarray(model_actions[: len(env_actions)], dtype=np.float32), "model_action_trace"

    from mwm.benchmark.replay_runtime import load_lance_action_stats
    from mwm.eval.action_preprocessing import uses_standardized_action_space

    if not uses_standardized_action_space(runtime.model, runtime.metadata, runtime.cfg):
        return np.asarray(env_actions, dtype=np.float32), "identity action space"
    stats, cache_hit = load_lance_action_stats(
        runtime.dataset.path,
        str(runtime.cfg.data.get("action_key", "action")),
        progress=progress,
    )
    source = f"legacy Lance action statistics v{stats.dataset_version}"
    if cache_hit:
        source += " (cached)"
    return stats.transform(env_actions), source


def piecewise_predictive_latents(
    model: Any,
    observations: dict[int, dict[str, Any]],
    model_actions: Any,
    segments: list[dict[str, Any]],
    *,
    device: Any,
) -> list[dict[str, Any]]:
    """Call the model's scheduled rollout once per actual MPC anchor."""

    import numpy as np
    import torch

    values = np.asarray(model_actions, dtype=np.float32)
    outputs: list[dict[str, Any]] = []
    for segment in segments:
        anchor_step = int(segment["anchor_step"])
        anchor_pixels = _to_bchw_float(_captured_pixels(observations, anchor_step), device)
        if anchor_pixels.ndim != 5:
            raise ReviewMediaUnsupported(
                f"captured policy pixels must have (B,T,C,H,W), got {tuple(anchor_pixels.shape)}"
            )
        history = int(anchor_pixels.shape[1])
        if history != 1:
            raise ReviewMediaUnsupported(
                "predictive rendering currently requires the one-frame online history used by planning; "
                f"captured history={history}"
            )
        blocks = list(segment["blocks"])
        model_k = [int(value) for value in getattr(model, "K", [])]
        for block in blocks:
            level_idx = int(block["level_idx"])
            if model_k:
                if level_idx < 0 or level_idx >= len(model_k):
                    raise ReviewMediaUnsupported(
                        f"scheduled level {level_idx} is outside checkpoint levels 0..{len(model_k) - 1}"
                    )
                if int(block["K"]) != model_k[level_idx]:
                    raise ReviewMediaUnsupported(
                        f"fidelity trace K={int(block['K'])} does not match checkpoint "
                        f"K={model_k[level_idx]} at level {level_idx}"
                    )
        primitive = np.concatenate(
            [values[int(item["primitive_start"]) : int(item["primitive_end"])] for item in blocks],
            axis=0,
        )
        action_blocks = primitive.reshape(len(blocks), -1)
        expected_dim = int(getattr(model, "action_dim", action_blocks.shape[-1]))
        if int(action_blocks.shape[-1]) != expected_dim:
            raise ReviewMediaUnsupported(
                f"grouped model actions have dim {int(action_blocks.shape[-1])}, expected {expected_dim}"
            )
        levels = [int(item["level_idx"]) for item in blocks]
        action_sequence = torch.as_tensor(action_blocks, device=device).unsqueeze(0).unsqueeze(0)
        infos = {"pixels": anchor_pixels.unsqueeze(1)}
        prediction = model.rollout_with_schedule(infos, action_sequence, levels)
        predicted = prediction.get("predicted_emb")
        if predicted is None or int(predicted.shape[2]) < history + len(blocks):
            raise ReviewMediaUnsupported("rollout_with_schedule returned too few predicted endpoints")
        for block_index, block in enumerate(blocks):
            outputs.append(
                {
                    **block,
                    "replan_idx": int(segment["replan_idx"]),
                    "anchor_step": anchor_step,
                    "latent": predicted[0, 0, history + block_index].detach(),
                }
            )
    return outputs


def render_environment_video(
    eval_path: str | Path,
    *,
    episode_index: int,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> RenderedMedia:
    eval_file = Path(eval_path)
    payload = load_json(eval_file)
    existing = existing_media_path(payload, episode_index, "env")
    if existing is not None and not force:
        _progress(progress, "Using existing environment video")
        return RenderedMedia("env", str(existing), "action_trace", [])
    rollout = rollout_by_index(payload, episode_index)
    action_trace = rollout.get("action_trace")
    if not isinstance(action_trace, list) or not action_trace:
        raise ReviewMediaUnsupported("exact environment rendering requires review_rollouts[].action_trace")
    actions = valid_action_prefix(action_trace)
    if not actions:
        raise ReviewMediaUnsupported("action_trace has no executable actions")
    if rollout.get("start_row") is None or rollout.get("goal_row") is None:
        raise ReviewMediaUnsupported("targeted environment rendering requires start_row and goal_row")

    from mwm.benchmark.replay_runtime import load_review_runtime
    from mwm.swm.envs import make_swm_world, parse_env_kwargs
    from omegaconf import OmegaConf

    out_dir = rollout_media_dir(eval_file, episode_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "env.mp4"
    if target.is_file() and not force:
        record_review_media(
            eval_file,
            episode_index=episode_index,
            kind="env",
            path=target,
            source_trace_type="action_trace",
            warnings=["Recovered an existing unindexed environment video."],
        )
        _progress(progress, "Recovered existing environment video")
        return RenderedMedia(
            "env",
            str(target),
            "action_trace",
            ["Recovered an existing unindexed environment video."],
        )
    runtime = load_review_runtime(
        _resolved_config_for_eval(eval_file),
        start_row=int(rollout["start_row"]),
        goal_row=int(rollout["goal_row"]),
        load_model=False,
        progress=progress,
    )
    tmp_video_dir = out_dir / "env_tmp"
    shutil.rmtree(tmp_video_dir, ignore_errors=True)
    try:
        _progress(progress, "Starting simulator")
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
            policy = FixedActionPolicy(actions)
            world.set_policy(policy)
            replay_budget = min(int(runtime.cfg.eval.budget), len(actions))
            _progress(
                progress,
                f"Replaying {replay_budget} stored action(s) and encoding video",
            )
            world.evaluate(
                dataset=runtime.dataset,
                episodes_idx=[int(rollout.get("dataset_episode", rollout.get("episode", 0)))],
                start_steps=[int(rollout.get("start_step", 0))],
                eval_budget=replay_budget,
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

    _progress(progress, "Recording environment media")
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

    if hasattr(dataset, "get_frame"):
        return first_frame(dataset.get_frame(int(row), str(pixels_key)))
    if hasattr(dataset, "get_row_data"):
        item = dataset.get_row_data(int(row))
        if isinstance(item, dict) and str(pixels_key) in item:
            return first_frame(item[str(pixels_key)])
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
        leading = tuple(tensor.shape[:-3])
        channels = int(tensor.shape[-3])
        tensor = F.interpolate(
            tensor.reshape(-1, channels, *tensor.shape[-2:]),
            size=MWM_IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(*leading, channels, *MWM_IMAGE_SIZE)
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


def _captured_actual_frame(observations: dict[int, dict[str, Any]], step: int, device: Any) -> Any:
    tensor = _to_bchw_float(_captured_pixels(observations, int(step)), device)
    if tensor.ndim == 5:
        tensor = tensor[0, -1].unsqueeze(0)
    return _to_hwc_uint8(tensor)


def _label_frame(frame: Any, label: str) -> Any:
    try:
        import cv2

        cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    except Exception:
        pass
    return frame


def _label_frame_lines(frame: Any, lines: list[str], *, boundary: bool = False) -> Any:
    try:
        import cv2
        import numpy as np

        out = np.asarray(frame).copy()
        line_height = 19
        panel_height = 8 + line_height * len(lines)
        overlay = out.copy()
        color = (27, 82, 143) if boundary else (16, 24, 40)
        cv2.rectangle(overlay, (0, 0), (out.shape[1], panel_height), color, thickness=-1)
        cv2.addWeighted(overlay, 0.82, out, 0.18, 0.0, dst=out)
        for index, line in enumerate(lines):
            cv2.putText(
                out,
                str(line),
                (8, 18 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return out
    except Exception:
        return frame


def _match_frame_size(reference: Any, candidate: Any) -> Any:
    if reference.shape[:2] == candidate.shape[:2]:
        return candidate


def _predictive_display_frame(frame: Any) -> Any:
    """Upscale compact env frames so detailed predictive labels remain legible."""

    target_width = max(448, int(frame.shape[1]))
    if target_width == int(frame.shape[1]):
        return frame
    target_height = int(round(frame.shape[0] * target_width / frame.shape[1]))
    try:
        import cv2

        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    except Exception:
        import numpy as np

        factor = max(1, target_width // int(frame.shape[1]))
        return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)
    try:
        import cv2

        return cv2.resize(candidate, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    except Exception:
        return candidate


def _usable_decoder(model: Any, metadata: dict[str, Any]) -> bool:
    if not latent_decoder_metadata_supported(metadata):
        return False
    if not callable(getattr(model, "decode", None)):
        return False
    decoders = getattr(model, "decoders", None)
    return decoders is None or len(decoders) > 0


def latent_decoder_metadata_supported(metadata: dict[str, Any]) -> bool:
    if str(metadata.get("adapter_family", "")).lower() == "prejepa":
        return False
    policy = metadata.get("component_policy", {})
    if isinstance(policy, dict) and policy.get("reconstructor") == []:
        return False
    return True


def rollout_checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = payload.get("checkpoint_run_dir")
    if not checkpoint:
        return None
    path = Path(str(checkpoint)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    metadata_path = path / "world_metadata.json"
    return load_json(metadata_path) if metadata_path.is_file() else None


def render_latent_reconstruction_video(
    eval_path: str | Path,
    *,
    episode_index: int,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> RenderedMedia:
    eval_file = Path(eval_path)
    payload = load_json(eval_file)
    existing = existing_media_path(payload, episode_index, "latent_reconstruction")
    if existing is not None and not force:
        _progress(progress, "Using existing latent reconstruction")
        return RenderedMedia("latent_reconstruction", str(existing), "fidelity_trace", [])
    rollout = rollout_by_index(payload, episode_index)
    fidelity_trace = rollout.get("fidelity_trace")
    if not isinstance(fidelity_trace, list) or not fidelity_trace:
        raise ReviewMediaUnsupported("latent reconstruction requires review_rollouts[].fidelity_trace")
    action_trace = rollout.get("action_trace")
    if isinstance(action_trace, list) and action_trace:
        actions = valid_action_prefix(action_trace)
        if actions and len(fidelity_trace) > len(actions):
            # Older vectorized eval artifacts retained masked diagnostics after
            # an env had terminated. Those slots were never executed.
            fidelity_trace = fidelity_trace[: len(actions)]
    if rollout.get("start_row") is None or rollout.get("goal_row") is None:
        raise ReviewMediaUnsupported("targeted latent rendering requires start_row and goal_row")

    checkpoint_metadata = rollout_checkpoint_metadata(payload)
    if checkpoint_metadata is not None and not latent_decoder_metadata_supported(checkpoint_metadata):
        raise ReviewMediaUnsupported("checkpoint does not include a latent reconstruction decoder")

    from mwm.benchmark.replay_runtime import load_review_runtime

    out_dir = rollout_media_dir(eval_file, episode_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "latent_reconstruction.mp4"
    if target.is_file() and not force:
        record_review_media(
            eval_file,
            episode_index=episode_index,
            kind="latent_reconstruction",
            path=target,
            source_trace_type="fidelity_trace",
            warnings=["Recovered an existing unindexed latent reconstruction."],
        )
        _progress(progress, "Recovered existing latent reconstruction")
        return RenderedMedia(
            "latent_reconstruction",
            str(target),
            "fidelity_trace",
            ["Recovered an existing unindexed latent reconstruction."],
        )
    runtime = load_review_runtime(
        _resolved_config_for_eval(eval_file),
        start_row=int(rollout["start_row"]),
        goal_row=int(rollout["goal_row"]),
        load_model=True,
        progress=progress,
    )
    if not _usable_decoder(runtime.model, runtime.metadata):
        runtime.close()
        raise ReviewMediaUnsupported("checkpoint does not expose a usable latent decoder")

    frames = []
    try:
        import imageio.v2 as imageio
        import numpy as np
        import torch

        pixels_key = str(runtime.cfg.data.get("pixels_key", "pixels"))
        start_row = int(rollout.get("start_row", 0))
        with torch.inference_mode():
            total_frames = len(fidelity_trace)
            for frame_index, item in enumerate(fidelity_trace):
                if frame_index == 0 or frame_index % 5 == 0:
                    _progress(progress, f"Decoding latent frame {frame_index + 1}/{total_frames}")
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
        _progress(progress, "Encoding latent reconstruction video")
        imageio.mimsave(target, frames, fps=10)
    finally:
        runtime.close()

    _progress(progress, "Recording latent media")
    record_review_media(
        eval_file,
        episode_index=episode_index,
        kind="latent_reconstruction",
        path=target,
        source_trace_type="fidelity_trace",
        warnings=[],
    )
    return RenderedMedia("latent_reconstruction", str(target), "fidelity_trace", [])


def render_latent_predictive_rollout_video(
    eval_path: str | Path,
    *,
    episode_index: int,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> RenderedMedia:
    """Render piecewise open-loop model predictions between actual MPC replans."""

    eval_file = Path(eval_path)
    payload = load_json(eval_file)
    kind = "latent_predictive_rollout"
    existing = existing_media_path(payload, episode_index, kind)
    if existing is not None and not force:
        _progress(progress, "Using existing latent predictive rollout")
        return RenderedMedia(kind, str(existing), "model_action_trace+fidelity_trace+actual_replay", [])

    rollout = rollout_by_index(payload, episode_index)
    actions = valid_action_prefix(rollout.get("action_trace") or [])
    fidelity_trace = rollout.get("fidelity_trace")
    if not actions:
        raise ReviewMediaUnsupported("latent predictive rollout requires action_trace")
    if not isinstance(fidelity_trace, list) or not fidelity_trace:
        raise ReviewMediaUnsupported("latent predictive rollout requires fidelity_trace")
    if rollout.get("start_row") is None or rollout.get("goal_row") is None:
        raise ReviewMediaUnsupported("latent predictive rollout requires start_row and goal_row")
    checkpoint_metadata = rollout_checkpoint_metadata(payload)
    if checkpoint_metadata is not None and not latent_decoder_metadata_supported(checkpoint_metadata):
        raise ReviewMediaUnsupported("checkpoint does not include a latent reconstruction decoder")

    from mwm.benchmark.replay_runtime import load_review_runtime

    out_dir = rollout_media_dir(eval_file, episode_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{kind}.mp4"
    if target.is_file() and not force:
        warning = "Recovered an existing unindexed latent predictive rollout."
        record_review_media(
            eval_file,
            episode_index=episode_index,
            kind=kind,
            path=target,
            source_trace_type="model_action_trace+fidelity_trace+actual_replay",
            warnings=[warning],
        )
        _progress(progress, "Recovered existing latent predictive rollout")
        return RenderedMedia(kind, str(target), "model_action_trace+fidelity_trace+actual_replay", [warning])

    runtime = load_review_runtime(
        _resolved_config_for_eval(eval_file),
        start_row=int(rollout["start_row"]),
        goal_row=int(rollout["goal_row"]),
        load_model=True,
        progress=progress,
    )
    warnings: list[str] = []
    try:
        if not _usable_decoder(runtime.model, runtime.metadata):
            raise ReviewMediaUnsupported("checkpoint does not expose a usable latent decoder")
        if not callable(getattr(runtime.model, "rollout_with_schedule", None)):
            raise ReviewMediaUnsupported("checkpoint does not expose rollout_with_schedule")

        observations, executed_actions = _replay_actual_observations(
            runtime,
            rollout,
            actions,
            progress=progress,
        )
        executed_actions = min(executed_actions, len(actions))
        model_actions, action_source = model_actions_for_rollout(rollout, runtime, progress=progress)
        model_actions = model_actions[:executed_actions]
        action_block = int(runtime.cfg.planner.action_block)
        receding_horizon = int(runtime.cfg.planner.receding_horizon)
        segments = predictive_replan_segments(
            fidelity_trace,
            action_count=executed_actions,
            action_block=action_block,
            receding_horizon=receding_horizon,
        )
        if not segments:
            raise ReviewMediaUnsupported(
                f"episode executed {executed_actions} primitive action(s), fewer than one complete "
                f"action block of {action_block}"
            )
        predicted_actions = sum(len(segment["blocks"]) * action_block for segment in segments)
        incomplete_actions = executed_actions - predicted_actions
        if incomplete_actions:
            warnings.append(
                f"Skipped {incomplete_actions} executed primitive action(s) that did not complete an action block."
            )

        import imageio.v2 as imageio
        import numpy as np
        import torch

        with torch.inference_mode():
            _progress(progress, f"Predicting {sum(len(x['blocks']) for x in segments)} scheduled block endpoint(s)")
            predictions = piecewise_predictive_latents(
                runtime.model,
                observations,
                model_actions,
                segments,
                device=runtime.device,
            )
            by_replan: dict[int, list[dict[str, Any]]] = {}
            for item in predictions:
                by_replan.setdefault(int(item["replan_idx"]), []).append(item)

            frames: list[Any] = []
            total_predictions = len(predictions)
            rendered_predictions = 0
            for segment in segments:
                replan_idx = int(segment["replan_idx"])
                anchor_step = int(segment["anchor_step"])
                actual_anchor = _predictive_display_frame(
                    _captured_actual_frame(observations, anchor_step, runtime.device)
                )
                left_anchor = _label_frame_lines(
                    actual_anchor,
                    [
                        f"ACTUAL REPLAY | replan {replan_idx} anchor t={anchor_step}",
                        "history passed to MPC planning",
                    ],
                    boundary=True,
                )
                right_anchor = _label_frame_lines(
                    actual_anchor.copy(),
                    [
                        f"PREDICTION RESET | replan {replan_idx}",
                        "open-loop state re-anchored from actual history",
                    ],
                    boundary=True,
                )
                frames.append(np.concatenate([left_anchor, right_anchor], axis=1))

                for item in by_replan.get(replan_idx, []):
                    rendered_predictions += 1
                    _progress(
                        progress,
                        f"Decoding predictive endpoint {rendered_predictions}/{total_predictions}",
                    )
                    endpoint = int(item["primitive_end"])
                    actual = _predictive_display_frame(
                        _captured_actual_frame(observations, endpoint, runtime.device)
                    )
                    predicted = runtime.model.decode(
                        int(item["level_idx"]), item["latent"].unsqueeze(0)
                    ).clamp(0.0, 1.0)
                    predicted_frame = _predictive_display_frame(_to_hwc_uint8(predicted))
                    predicted_frame = _match_frame_size(actual, predicted_frame)
                    action_range = f"actions [{int(item['primitive_start'])}:{endpoint})"
                    left = _label_frame_lines(
                        actual,
                        [
                            f"ACTUAL REPLAY | replan {replan_idx} block {int(item['block_idx'])}",
                            f"{action_range} | endpoint t={endpoint}",
                        ],
                    )
                    right = _label_frame_lines(
                        predicted_frame,
                        [
                            f"PREDICTED | replan {replan_idx} block {int(item['block_idx'])} "
                            f"L={int(item['level_idx'])} K={int(item['K'])}",
                            f"{action_range} | +{int(item['distance_since_anchor'])} from actual anchor",
                        ],
                    )
                    frames.append(np.concatenate([left, right], axis=1))

        if not frames:
            raise ReviewMediaUnsupported("latent predictive rollout produced no frames")
        _progress(progress, "Encoding latent predictive rollout video")
        imageio.mimsave(target, frames, fps=2)
    finally:
        runtime.close()

    _progress(progress, "Recording latent predictive rollout media")
    record_review_media(
        eval_file,
        episode_index=episode_index,
        kind=kind,
        path=target,
        source_trace_type=f"{action_source}+fidelity_trace+actual_replay",
        warnings=warnings,
    )
    return RenderedMedia(kind, str(target), f"{action_source}+fidelity_trace+actual_replay", warnings)


def _render_rollout_media_unlocked(
    eval_path: str | Path,
    *,
    episode_index: int,
    sources: Iterable[str],
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    rendered: list[dict[str, Any]] = []
    warnings: list[str] = []
    normalized_sources = [str(source) for source in sources]
    if "all" in normalized_sources:
        normalized_sources = ["env", "latent", "predictive"]
    elif "both" in normalized_sources:
        # Backward compatibility for older review pages and API clients.
        normalized_sources = ["env", "latent"]
    for source in normalized_sources:
        try:
            if source == "env":
                media = render_environment_video(
                    eval_path,
                    episode_index=episode_index,
                    force=force,
                    progress=progress,
                )
            elif source in {"latent", "latent_reconstruction"}:
                media = render_latent_reconstruction_video(
                    eval_path,
                    episode_index=episode_index,
                    force=force,
                    progress=progress,
                )
            elif source in {"predictive", "latent_predictive_rollout"}:
                media = render_latent_predictive_rollout_video(
                    eval_path,
                    episode_index=episode_index,
                    force=force,
                    progress=progress,
                )
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
            warnings.extend(f"{source}: {warning}" for warning in media.warnings)
        except ReviewMediaUnsupported as exc:
            warnings.append(f"{source}: {exc}")
    return {"episode_index": int(episode_index), "rendered": rendered, "warnings": warnings}


def render_rollout_media(
    eval_path: str | Path,
    *,
    episode_index: int,
    sources: Iterable[str],
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    _progress(progress, "Waiting for exclusive render access")
    with _review_render_lock(eval_path):
        return _render_rollout_media_unlocked(
            eval_path,
            episode_index=episode_index,
            sources=sources,
            force=force,
            progress=progress,
        )


__all__ = [
    "ReviewMediaUnsupported",
    "RenderedMedia",
    "existing_media_path",
    "fidelity_trace_from_planning_trace",
    "latent_decoder_metadata_supported",
    "model_actions_for_rollout",
    "piecewise_predictive_latents",
    "predictive_replan_segments",
    "record_review_media",
    "render_environment_video",
    "render_latent_reconstruction_video",
    "render_latent_predictive_rollout_video",
    "render_rollout_media",
    "review_media_entry",
    "rollout_by_index",
    "rollout_checkpoint_metadata",
    "rollout_key",
    "rollout_media_dir",
    "valid_action_prefix",
]
