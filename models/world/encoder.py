import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        C = 32
        self.backbone = nn.Sequential(
            nn.Conv2d(3, C, 5, stride=2, padding=2), nn.ReLU(inplace=True),  # 96->48
            nn.Conv2d(C, C*2, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 48->24
            nn.Conv2d(C*2, C*4, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 24->12
            nn.Conv2d(C*4, C*8, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 12->6
            nn.Conv2d(C*8, C*8, 3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C*8*6*6, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.proj(h)
        return z

class StateEncoder(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, out_dim: int = 512):
        super().__init__()
        self.fc = self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x)
        return z
