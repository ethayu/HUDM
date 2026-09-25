from __future__ import annotations

from typing import Any, Sequence

import torch.nn as nn

from mwm.preprocessing.images import ImageNetPreprocess


class MatryoshkaRuntimeModel(nn.Module):
    """Shared marker base for planner-facing MWM runtime implementations."""

    architecture_version = "matryoshka_runtime_v1"

    def _init_runtime_state(
        self,
        *,
        encoder: nn.Module,
        projector: nn.Module,
        transitions: Sequence[nn.Module],
        decoders: Sequence[nn.Module],
        K: Sequence[int],
        D: int,
        action_dim: int,
        action_block: int,
        image_shape: Sequence[int],
        normalize_imagenet: bool,
        history_size: int,
        num_preds: int,
        head_architectures: Sequence[dict[str, Any]],
        decoder_architectures: Sequence[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        architecture_version: str | None = None,
    ) -> None:
        self.encoder = encoder
        self.projector = projector
        self.transitions = nn.ModuleList(list(transitions))
        self.decoders = nn.ModuleList(list(decoders))
        self.K = [int(k) for k in K]
        self.D = int(D)
        self.action_dim = int(action_dim)
        self.action_block = int(action_block)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.normalize_imagenet = bool(normalize_imagenet)
        self.preprocess = ImageNetPreprocess() if self.normalize_imagenet else None
        self.history_size = int(history_size)
        self.num_preds = int(num_preds)
        self.head_architectures = [dict(x) for x in head_architectures]
        self.decoder_architectures = [dict(x) for x in decoder_architectures or []]
        if len(self.K) != len(self.transitions):
            raise ValueError(f"K has {len(self.K)} entries but transitions has {len(self.transitions)}.")
        if len(self.K) != len(self.decoders):
            raise ValueError(f"K has {len(self.K)} entries but decoders has {len(self.decoders)}.")
        if not self.K:
            raise ValueError("K must contain at least one fidelity level.")
        if any(k <= 0 or k > self.D for k in self.K):
            raise ValueError(f"All K values must be in [1, D={self.D}], got {self.K}.")
        if len(set(self.K)) != len(self.K):
            raise ValueError(f"K values must be unique, got {self.K}.")
        arch = str(architecture_version or self.architecture_version)
        meta = dict(metadata or {})
        meta.setdefault("architecture_version", arch)
        meta.setdefault("image_shape", [int(x) for x in self.image_shape])
        meta.setdefault("action_block", int(self.action_block))
        meta.setdefault("action_dim", int(self.action_dim) // max(1, int(self.action_block)))
        meta.setdefault("normalize_imagenet", bool(self.normalize_imagenet))
        meta.setdefault(
            "preprocessing_spec",
            {
                "image": "imagenet" if self.normalize_imagenet else "identity",
                "layout": "BCHW",
                "image_shape": [int(x) for x in self.image_shape],
            },
        )
        meta.setdefault(
            "action_spec",
            {
                "dim": int(self.action_dim),
                "base_dim": int(self.action_dim) // max(1, int(self.action_block)),
                "block": int(self.action_block),
            },
        )
        meta.setdefault("head_architectures", self.head_architectures)
        meta.setdefault("decoder_architectures", self.decoder_architectures)
        meta.setdefault("levels", list(self.K))
        self.metadata = meta
        self._last_cost_diagnostics: dict[str, Any] = {}

    @property
    def num_levels(self) -> int:
        return len(getattr(self, "K"))


__all__ = ["MatryoshkaRuntimeModel"]
