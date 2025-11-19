import torch
import torch.nn as nn


class UpconvDecoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        C = 32
        self.fc = nn.Linear(in_dim, C*8*6*6)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(C*8, C*4, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 6->12
            nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 12->24
            nn.ConvTranspose2d(C*2, C,   4, stride=2, padding=1), nn.ReLU(inplace=True),  # 24->48
            nn.ConvTranspose2d(C,   3,   4, stride=2, padding=1), nn.Tanh(),              # 48->96
        )

    def forward(self, z_prefix: torch.Tensor) -> torch.Tensor:
        h = self.fc(z_prefix)
        h = h.view(h.size(0), -1, 6, 6)
        x = self.up(h)
        return x

