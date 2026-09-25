from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_width(value: int, k: int, D: int) -> int:
    """Use the same linear width rule as the existing per-level Le-WM adapter."""

    if int(value) <= 0 or int(k) <= 0 or int(D) <= 0:
        raise ValueError(f"scaled_width requires positive values, got value={value}, k={k}, D={D}.")
    return max(1, int(round(float(value) * float(k) / float(D))))


class SlimmableLinear(nn.Module):
    """A maximum-size linear layer evaluated through prefix parameter slices."""

    def __init__(self, max_in_features: int, max_out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        self.max_in_features = int(max_in_features)
        self.max_out_features = int(max_out_features)
        self.weight = nn.Parameter(torch.empty(self.max_out_features, self.max_in_features))
        self.bias = nn.Parameter(torch.empty(self.max_out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.max_in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, *, out_features: int) -> torch.Tensor:
        in_features = int(x.shape[-1])
        out_features = int(out_features)
        if in_features <= 0 or in_features > self.max_in_features:
            raise ValueError(
                f"SlimmableLinear input width must be in [1, {self.max_in_features}], got {in_features}."
            )
        if out_features <= 0 or out_features > self.max_out_features:
            raise ValueError(
                f"SlimmableLinear output width must be in [1, {self.max_out_features}], got {out_features}."
            )
        bias = self.bias[:out_features] if self.bias is not None else None
        return F.linear(x, self.weight[:out_features, :in_features], bias)


class SlimmableLayerNorm(nn.Module):
    def __init__(self, max_features: int, *, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.max_features = int(max_features)
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.max_features))
            self.bias = nn.Parameter(torch.zeros(self.max_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = int(x.shape[-1])
        if k <= 0 or k > self.max_features:
            raise ValueError(f"LayerNorm width must be in [1, {self.max_features}], got {k}.")
        weight = self.weight[:k] if self.weight is not None else None
        bias = self.bias[:k] if self.bias is not None else None
        return F.layer_norm(x, (k,), weight, bias, self.eps)


class SlimmableBatchNorm1d(nn.Module):
    """BatchNorm over a prefix of one shared maximum-width statistics vector."""

    def __init__(
        self,
        max_features: int,
        *,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.max_features = int(max_features)
        self.eps = float(eps)
        self.momentum = momentum
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.max_features))
            self.bias = nn.Parameter(torch.zeros(self.max_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.max_features))
            self.register_buffer("running_var", torch.ones(self.max_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = int(x.shape[-1])
        if k <= 0 or k > self.max_features:
            raise ValueError(f"BatchNorm width must be in [1, {self.max_features}], got {k}.")
        flat = x.reshape(-1, k)
        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = float(self.momentum)
        else:
            exponential_average_factor = 0.0 if self.momentum is None else float(self.momentum)
        running_mean = self.running_mean[:k] if self.running_mean is not None else None
        running_var = self.running_var[:k] if self.running_var is not None else None
        weight = self.weight[:k] if self.weight is not None else None
        bias = self.bias[:k] if self.bias is not None else None
        out = F.batch_norm(
            flat,
            running_mean,
            running_var,
            weight,
            bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
        return out.reshape_as(x)


class SlimmableMLP(nn.Module):
    def __init__(
        self,
        D: int,
        max_hidden_dim: int,
        *,
        norm: str = "layer_norm",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.max_hidden_dim = int(max_hidden_dim)
        self.fc1 = SlimmableLinear(self.D, self.max_hidden_dim)
        norm_key = str(norm).strip().lower()
        if norm_key in {"batch_norm", "batchnorm", "batchnorm1d"}:
            self.norm: nn.Module = SlimmableBatchNorm1d(self.max_hidden_dim)
        elif norm_key in {"layer_norm", "layernorm"}:
            self.norm = SlimmableLayerNorm(self.max_hidden_dim)
        elif norm_key in {"identity", "none", ""}:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported slimmable MLP normalization {norm!r}.")
        activation_key = str(activation).strip().lower()
        if activation_key != "gelu":
            raise ValueError(f"Unsupported slimmable MLP activation {activation!r}; expected gelu.")
        self.fc2 = SlimmableLinear(self.max_hidden_dim, self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = int(x.shape[-1])
        hidden = scaled_width(self.max_hidden_dim, k, self.D)
        x = self.fc1(x, out_features=hidden)
        x = self.norm(x)
        x = F.gelu(x)
        return self.fc2(x, out_features=k)


class SlimmableActionEncoder(nn.Module):
    """Le-WM Embedder with a shared, prefix-sliced embedding MLP."""

    def __init__(
        self,
        *,
        action_dim: int,
        D: int,
        smoothed_dim: int = 10,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.D = int(D)
        self.smoothed_dim = int(smoothed_dim)
        self.mlp_scale = int(mlp_scale)
        self.patch_embed = nn.Conv1d(self.action_dim, self.smoothed_dim, kernel_size=1, stride=1)
        self.fc1 = SlimmableLinear(self.smoothed_dim, self.mlp_scale * self.D)
        self.fc2 = SlimmableLinear(self.mlp_scale * self.D, self.D)

    def forward(self, action: torch.Tensor, *, k: int) -> torch.Tensor:
        k = int(k)
        if int(action.shape[-1]) != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {int(action.shape[-1])}.")
        x = action.to(dtype=self.patch_embed.weight.dtype).transpose(1, 2)
        x = self.patch_embed(x).transpose(1, 2)
        hidden = self.mlp_scale * k
        x = self.fc1(x, out_features=hidden)
        x = F.silu(x)
        return self.fc2(x, out_features=k)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class SlimmableAttention(nn.Module):
    def __init__(
        self,
        *,
        D: int,
        max_heads: int,
        max_dim_head: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.max_heads = int(max_heads)
        self.max_dim_head = int(max_dim_head)
        self.max_inner_dim = self.max_heads * self.max_dim_head
        self.dropout = float(dropout)
        self.project_out_at_max = not (self.max_heads == 1 and self.max_dim_head == self.D)
        self.norm = SlimmableLayerNorm(self.D)
        self.to_qkv = nn.Linear(self.D, 3 * self.max_inner_dim, bias=False)
        self.to_out: nn.Module = (
            SlimmableLinear(self.max_inner_dim, self.D)
            if self.project_out_at_max
            else nn.Identity()
        )

    def active_shape(self, k: int) -> tuple[int, int, int]:
        heads = scaled_width(self.max_heads, int(k), self.D)
        dim_head = scaled_width(self.max_dim_head, int(k), self.D)
        return heads, dim_head, heads * dim_head

    def forward(self, x: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        k = int(x.shape[-1])
        heads, dim_head, inner = self.active_shape(k)
        x = self.norm(x)
        weight = self.to_qkv.weight.reshape(3, self.max_inner_dim, self.D)[:, :inner, :k]
        qkv = F.linear(x, weight.reshape(3 * inner, k)).reshape(*x.shape[:-1], 3, inner)
        q, key, value = qkv.unbind(dim=-2)
        batch, time = int(x.shape[0]), int(x.shape[1])
        q = q.reshape(batch, time, heads, dim_head).transpose(1, 2)
        key = key.reshape(batch, time, heads, dim_head).transpose(1, 2)
        value = value.reshape(batch, time, heads, dim_head).transpose(1, 2)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, key, value, dropout_p=drop, is_causal=bool(causal))
        out = out.transpose(1, 2).reshape(batch, time, inner)
        if heads == 1 and dim_head == k:
            return out
        if not isinstance(self.to_out, SlimmableLinear):
            raise RuntimeError("Active attention requires an output projection not present at maximum width.")
        return self.to_out(out, out_features=k)


class SlimmableFeedForward(nn.Module):
    def __init__(self, *, D: int, max_hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.D = int(D)
        self.max_hidden_dim = int(max_hidden_dim)
        self.dropout = float(dropout)
        self.norm = SlimmableLayerNorm(self.D)
        self.fc1 = SlimmableLinear(self.D, self.max_hidden_dim)
        self.fc2 = SlimmableLinear(self.max_hidden_dim, self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = int(x.shape[-1])
        hidden = scaled_width(self.max_hidden_dim, k, self.D)
        x = self.norm(x)
        x = self.fc1(x, out_features=hidden)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x, out_features=k)
        return F.dropout(x, p=self.dropout, training=self.training)


class SlimmableConditionalBlock(nn.Module):
    def __init__(
        self,
        *,
        D: int,
        max_heads: int,
        max_dim_head: int,
        max_mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.attn = SlimmableAttention(
            D=self.D,
            max_heads=int(max_heads),
            max_dim_head=int(max_dim_head),
            dropout=float(dropout),
        )
        self.mlp = SlimmableFeedForward(D=self.D, max_hidden_dim=int(max_mlp_dim), dropout=float(dropout))
        self.modulation = nn.Linear(self.D, 6 * self.D)
        nn.init.constant_(self.modulation.weight, 0)
        nn.init.constant_(self.modulation.bias, 0)

    def _condition(self, c: torch.Tensor, k: int) -> tuple[torch.Tensor, ...]:
        weight = self.modulation.weight.reshape(6, self.D, self.D)[:, :k, :k]
        bias = self.modulation.bias.reshape(6, self.D)[:, :k]
        mod = F.linear(F.silu(c), weight.reshape(6 * k, k), bias.reshape(6 * k))
        return tuple(mod.reshape(*c.shape[:-1], 6, k).unbind(dim=-2))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        k = int(x.shape[-1])
        if tuple(c.shape) != tuple(x.shape):
            raise ValueError(f"Condition shape {tuple(c.shape)} must match latent shape {tuple(x.shape)}.")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._condition(c, k)
        norm_x = F.layer_norm(x, (k,), eps=1e-6)
        x = x + gate_msa * self.attn(_modulate(norm_x, shift_msa, scale_msa))
        norm_x = F.layer_norm(x, (k,), eps=1e-6)
        x = x + gate_mlp * self.mlp(_modulate(norm_x, shift_mlp, scale_mlp))
        return x


class SlimmablePredictor(nn.Module):
    """A causal Le-WM predictor whose active feature width is selected at call time."""

    def __init__(
        self,
        *,
        D: int,
        num_frames: int,
        depth: int,
        max_heads: int,
        max_dim_head: int,
        max_mlp_dim: int,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.num_frames = int(num_frames)
        self.depth = int(depth)
        self.max_heads = int(max_heads)
        self.max_dim_head = int(max_dim_head)
        self.max_mlp_dim = int(max_mlp_dim)
        self.dropout = float(dropout)
        self.emb_dropout = float(emb_dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, self.D))
        self.layers = nn.ModuleList(
            [
                SlimmableConditionalBlock(
                    D=self.D,
                    max_heads=self.max_heads,
                    max_dim_head=self.max_dim_head,
                    max_mlp_dim=self.max_mlp_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = SlimmableLayerNorm(self.D)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SlimmablePredictor expects (B,T,K), got {tuple(x.shape)}.")
        if tuple(c.shape) != tuple(x.shape):
            raise ValueError(f"Condition shape {tuple(c.shape)} must match latent shape {tuple(x.shape)}.")
        k = int(x.shape[-1])
        time = int(x.shape[1])
        if k <= 0 or k > self.D:
            raise ValueError(f"Predictor K must be in [1, {self.D}], got {k}.")
        if time > self.num_frames:
            raise ValueError(f"Predictor time length {time} exceeds num_frames={self.num_frames}.")
        x = x + self.pos_embedding[:, :time, :k]
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        for block in self.layers:
            x = block(x, c)
        return self.norm(x)


class SharedSlimmableTransition(nn.Module):
    """One nested Le-WM transition used by every latent prefix width."""

    def __init__(
        self,
        *,
        D: int,
        action_dim: int,
        num_frames: int,
        depth: int,
        max_heads: int,
        max_dim_head: int,
        max_mlp_dim: int,
        predictor_dropout: float = 0.0,
        predictor_emb_dropout: float = 0.0,
        action_smoothed_dim: int = 10,
        action_mlp_scale: int = 4,
        pred_proj_hidden_dim: int,
        pred_proj_norm: str = "batch_norm",
    ) -> None:
        super().__init__()
        self.D = int(D)
        self.action_encoder = SlimmableActionEncoder(
            action_dim=int(action_dim),
            D=self.D,
            smoothed_dim=action_smoothed_dim,
            mlp_scale=int(action_mlp_scale),
        )
        self.predictor = SlimmablePredictor(
            D=self.D,
            num_frames=int(num_frames),
            depth=int(depth),
            max_heads=int(max_heads),
            max_dim_head=int(max_dim_head),
            max_mlp_dim=int(max_mlp_dim),
            dropout=float(predictor_dropout),
            emb_dropout=float(predictor_emb_dropout),
        )
        self.pred_proj = SlimmableMLP(
            self.D,
            int(pred_proj_hidden_dim),
            norm=str(pred_proj_norm),
        )

    def predict(self, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        k = int(emb.shape[-1])
        if k <= 0 or k > self.D:
            raise ValueError(f"Shared dynamics K must be in [1, {self.D}], got {k}.")
        action_emb = self.action_encoder(action, k=k)
        pred = self.predictor(emb, action_emb)
        flat = pred.reshape(-1, k)
        return self.pred_proj(flat).reshape(*pred.shape[:-1], k)

    def architecture(self) -> dict[str, Any]:
        return {
            "D": self.D,
            "num_frames": self.predictor.num_frames,
            "depth": self.predictor.depth,
            "max_heads": self.predictor.max_heads,
            "max_dim_head": self.predictor.max_dim_head,
            "max_mlp_dim": self.predictor.max_mlp_dim,
            "attention_project_out": self.predictor.layers[0].attn.project_out_at_max
            if self.predictor.layers
            else None,
            "predictor_dropout": self.predictor.dropout,
            "predictor_emb_dropout": self.predictor.emb_dropout,
            "action_smoothed_dim": self.action_encoder.smoothed_dim,
            "action_mlp_scale": self.action_encoder.mlp_scale,
            "pred_proj_hidden_dim": self.pred_proj.max_hidden_dim,
            "pred_proj_norm": type(self.pred_proj.norm).__name__,
        }


__all__ = [
    "SharedSlimmableTransition",
    "SlimmableActionEncoder",
    "SlimmableAttention",
    "SlimmableBatchNorm1d",
    "SlimmableConditionalBlock",
    "SlimmableLayerNorm",
    "SlimmableLinear",
    "SlimmableMLP",
    "SlimmablePredictor",
    "scaled_width",
]
