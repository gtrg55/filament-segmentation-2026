"""ResNet34 U-Net producing a per-pixel "annotated as filament" probability at full resolution.

The network is trained on the union mask of one randomly drawn annotator copy per image, so across epochs
the target of a pixel is the fraction of annotators that outlined it: the sigmoid output estimates the
consensus probability. The decoder ends with a full-resolution block that sees the raw image again, so that
thin filaments are not lost to the stride-2 stem of the encoder.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int) -> None:
        super().__init__()
        self.block = conv_block(cin + skip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class FilamentUNet(nn.Module):
    def __init__(self, encoder_weights: Path | None = None) -> None:
        super().__init__()
        backbone = torchvision.models.resnet34(weights=None)
        if encoder_weights is not None:
            backbone.load_state_dict(torch.load(encoder_weights, map_location="cpu", weights_only=True))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # stride 2, 64 ch
        self.pool = backbone.maxpool  # stride 4
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.up4 = Up(512, 256, 256)  # stride 16
        self.up3 = Up(256, 128, 128)  # stride 8
        self.up2 = Up(128, 64, 64)  # stride 4
        self.up1 = Up(64, 64, 32)  # stride 2
        self.full = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True))  # stride 1 image features
        self.up0 = Up(32, 16, 16)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        """`gray`: B x 1 x H x W in [0, 1]; returns logits B x 1 x H x W (H, W multiples of 32)."""
        x = (gray.expand(-1, 3, -1, -1) - self.mean) / self.std
        s0 = self.full(x)
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        y = self.up4(s5, s4)
        y = self.up3(y, s3)
        y = self.up2(y, s2)
        y = self.up1(y, s1)
        y = self.up0(y, s0)
        return self.head(y)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (total + eps)).mean()
