import torch
import torch.nn as nn


class TinyTransformerDynamics(nn.Module):
    """
    Predict next latent prefix z_{t+1}[:k] from (z_t[:k], a_t).
    A tiny transformer over two tokens (z, action) + a CLS token.
    """

    def __init__(self, k_dim: int, action_dim: int = 2, d_model: int = 128, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.k = k_dim
        self.action_dim = action_dim
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        self.z_proj = nn.Linear(k_dim, d_model)
        self.a_proj = nn.Linear(action_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 2*d_model), nn.ReLU(inplace=True), nn.Linear(2*d_model, k_dim)
        )

    def step(self, z_k: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        B = z_k.size(0)
        tokens = torch.cat([
            self.cls.expand(B, 1, -1),
            self.z_proj(z_k).unsqueeze(1),
            self.a_proj(a_t).unsqueeze(1),
        ], dim=1)
        h = self.enc(tokens)
        cls = h[:, 0]
        delta = self.head(cls)
        return z_k + delta  # residual prediction

    @torch.no_grad()
    def rollout(self, z0_k: torch.Tensor, a_seq: torch.Tensor, detach_each_step: bool = True) -> torch.Tensor:
        """
        Predict z at t+T given initial z_t and action sequence a_{t:t+T-1}.
        If detach_each_step is True, breaks gradients between steps (no BPTT >1).
        """
        assert a_seq.dim() == 3, "a_seq must be (B,T,A)"
        B, T, A = a_seq.shape
        z = z0_k
        for i in range(T):
            z_next = self.step(z, a_seq[:, i, :])
            if detach_each_step:
                z = z_next.detach()
            else:
                z = z_next
        return z
