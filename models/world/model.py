import torch
import torch.nn as nn
from typing import List

from .encoder import CNNEncoder
from .decoder import UpconvDecoder
from .dynamics import TinyTransformerDynamics


class HierWorldModel(nn.Module):
    def __init__(
        self,
        K: List[int],
        D: int,
        action_dim: int = 2,
        input: str = "images",
        decoder_mode: str = "per_level",
        dynamics_mode: str = "per_level",
        image_shape: int | tuple[int, int] = 96,
    ):
        super().__init__()
        assert max(K) == D, "Largest K must equal D"
        if decoder_mode not in {"per_level", "shared"}:
            raise ValueError(f"decoder_mode must be 'per_level' or 'shared', got {decoder_mode}")
        if dynamics_mode not in {"per_level", "shared"}:
            raise ValueError(f"dynamics_mode must be 'per_level' or 'shared', got {dynamics_mode}")
        if input != "images":
            raise ValueError(f"HUDM v1 is RGB-only and requires input='images', got {input!r}")
        self.K = K
        self.D = D
        if isinstance(image_shape, int):
            image_shape = (image_shape, image_shape)
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.decoder_mode = decoder_mode
        self.dynamics_mode = dynamics_mode
        self.encoder = CNNEncoder(out_dim=D, image_shape=self.image_shape)
        if decoder_mode == "per_level":
            self.decoders = nn.ModuleList([UpconvDecoder(in_dim=k, image_shape=self.image_shape) for k in K])
        else:
            self.decoder = UpconvDecoder(in_dim=D, image_shape=self.image_shape)
        if dynamics_mode == "per_level":
            self.dynamics = nn.ModuleList([TinyTransformerDynamics(k_dim=k, action_dim=action_dim) for k in K])
        else:
            self.dynamics = TinyTransformerDynamics(k_dim=D, action_dim=action_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _pad_prefix(self, z: torch.Tensor, k: int) -> torch.Tensor:
        if k == self.D:
            return z
        z_pad = torch.zeros_like(z)
        z_pad[..., :k] = z[..., :k]
        return z_pad

    def decode(self, level: int, z: torch.Tensor) -> torch.Tensor:
        k = self.K[level]
        if self.decoder_mode == "per_level":
            return self.decoders[level](z[..., :k])
        z_pad = self._pad_prefix(z, k)
        return self.decoder(z_pad)

    def predict_next(self, level: int, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        k = self.K[level]
        if self.dynamics_mode == "per_level":
            return self.dynamics[level].step(z[..., :k], a)
        z_pad = self._pad_prefix(z, k)
        z_next = self.dynamics.step(z_pad, a)
        return z_next[..., :k]

    @torch.no_grad()
    def rollout(self, level: int, z: torch.Tensor, a_seq: torch.Tensor, detach_each_step: bool = True) -> torch.Tensor:
        k = self.K[level]
        if self.dynamics_mode == "per_level":
            return self.dynamics[level].rollout(z[..., :k], a_seq, detach_each_step=detach_each_step)
        z_pad = self._pad_prefix(z, k)
        for i in range(a_seq.size(1)):
            if detach_each_step:
                z_pad = z_pad.detach()
            z_pad = self.dynamics.step(z_pad, a_seq[:, i, :])
            if k < self.D:
                z_pad[..., k:] = 0.0
        return z_pad[..., :k]
