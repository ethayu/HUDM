import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, image_shape: int | tuple[int, int] = 96):
        super().__init__()
        if isinstance(image_shape, int):
            image_shape = (image_shape, image_shape)
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        C = 32
        self.backbone = nn.Sequential(
            nn.Conv2d(3, C, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(C, C*2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C*2, C*4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C*4, C*8, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C*8, C*8, 3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C*8*6*6, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = F.adaptive_avg_pool2d(h, (6, 6))
        z = self.proj(h)
        return z
