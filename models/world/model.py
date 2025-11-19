import torch
import torch.nn as nn
from typing import List

from .encoder import CNNEncoder
from .decoder import UpconvDecoder
from .dynamics import TinyTransformerDynamics


class HierWorldModel(nn.Module):
    def __init__(self, K: List[int], D: int, action_dim: int = 2):
        super().__init__()
        assert max(K) == D, "Largest K must equal D"
        self.K = K
        self.D = D
        self.encoder = CNNEncoder(out_dim=D)
        # One decoder and dynamics per level (input dim varies)
        self.decoders = nn.ModuleList([UpconvDecoder(in_dim=k) for k in K])
        self.dynamics = nn.ModuleList([TinyTransformerDynamics(k_dim=k, action_dim=action_dim) for k in K])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, level: int, z: torch.Tensor) -> torch.Tensor:
        k = self.K[level]
        return self.decoders[level](z[:, :k])

    def predict_next(self, level: int, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        k = self.K[level]
        return self.dynamics[level].step(z[:, :k], a)

    @torch.no_grad()
    def rollout(self, level: int, z: torch.Tensor, a_seq: torch.Tensor, detach_each_step: bool = True) -> torch.Tensor:
        k = self.K[level]
        return self.dynamics[level].rollout(z[:, :k], a_seq, detach_each_step=detach_each_step)

