import torch
import torch.nn as nn
import torch.nn.functional as F


class UpconvDecoder(nn.Module):
    def __init__(self, in_dim: int, image_shape: int | tuple[int, int] = 96):
        super().__init__()
        if isinstance(image_shape, int):
            image_shape = (image_shape, image_shape)
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        C = 8
        self.fc = nn.Linear(in_dim, C*8*6*6)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(C*8, C*4, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 6->12
            nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 12->24
            nn.ConvTranspose2d(C*2, C,   4, stride=2, padding=1), nn.ReLU(inplace=True),  # 24->48
            nn.ConvTranspose2d(C,   3,   4, stride=2, padding=1),              # 48->96
        )

    def forward(self, z_prefix: torch.Tensor) -> torch.Tensor:
        h = self.fc(z_prefix)
        h = h.view(h.size(0), -1, 6, 6)
        x = self.up(h)
        if tuple(x.shape[-2:]) != self.image_shape:
            x = F.interpolate(x, size=self.image_shape, mode="bilinear", align_corners=False)
        return x
