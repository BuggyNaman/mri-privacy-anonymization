"""Models and losses for ROI-preserving MRI anonymization.

* ``BaselineAutoencoder`` – plain 3-stage convolutional autoencoder (128 -> 16
  -> 128) trained with uniform MSE.
* ``ProposedSharpCAE``    – same bottleneck layout *plus* encoder/decoder skip
  connections, so fine tumour texture inside the ROI survives compression.
* ``SharpROILoss``        – asymmetric L2 + 0.5 * L1, weighted 25.0 inside the
  ROI and 2.0 in the periphery (a 12.5x ratio).
* ``MedicalResNet18``     – ImageNet-pretrained ResNet-18 adapted to 1-channel
  grayscale MRI for downstream tumour classification.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------------------------------------------------------------- #
# Autoencoders
# --------------------------------------------------------------------------- #
class BaselineAutoencoder(nn.Module):
    """Standard 3-stage convolutional autoencoder: 128 -> 64 -> 32 -> 16 -> ..."""

    def __init__(self, base_channels: int = 32, bottleneck_channels: int = 16) -> None:
        super().__init__()
        c1, c2 = base_channels, base_channels * 2

        self.encoder = nn.Sequential(
            self._block(1, c1),          # 128 -> 64
            self._block(c1, c2),         # 64  -> 32
            self._block(c2, c2),         # 32  -> 16
            nn.Conv2d(c2, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            self._up_block(bottleneck_channels, c2),  # 16 -> 32
            self._up_block(c2, c1),                   # 32 -> 64
            self._up_block(c1, c1),                   # 64 -> 128
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _SkipEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.body(x)
        return features, self.pool(features)


class _SkipDecoderBlock(nn.Module):
    """Upsample, concatenate the matching encoder skip, then refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.body = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(torch.cat([x, skip], dim=1))


class ProposedSharpCAE(nn.Module):
    """ROI-preserving autoencoder with full-resolution encoder/decoder skips.

    The skips carry un-pooled high-frequency detail (tumour margins, tissue
    texture) straight into the decoder, so the sharp central ROI survives
    compression that the baseline bottleneck destroys.
    """

    def __init__(self, base_channels: int = 32, bottleneck_channels: int = 16) -> None:
        super().__init__()
        c1, c2 = base_channels, base_channels * 2

        self.enc1 = _SkipEncoderBlock(1, c1)           # 128 -> 64
        self.enc2 = _SkipEncoderBlock(c1, c2)          # 64  -> 32
        self.enc3 = _SkipEncoderBlock(c2, c2)          # 32  -> 16

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c2, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        self.dec3 = _SkipDecoderBlock(bottleneck_channels, c2, c2)  # 16 -> 32
        self.dec2 = _SkipDecoderBlock(c2, c2, c1)                   # 32 -> 64
        self.dec1 = _SkipDecoderBlock(c1, c1, c1)                   # 64 -> 128
        self.head = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


# --------------------------------------------------------------------------- #
# ROI-weighted loss
# --------------------------------------------------------------------------- #
class SharpROILoss(nn.Module):
    """Asymmetric ROI-weighted L2 + 0.5 * L1.

    ``weight = roi_weight * mask + periphery_weight * (1 - mask)`` makes a
    pixel inside the diagnostic ROI 12.5x more expensive than a periphery pixel
    with the defaults (25.0 vs 2.0). The L1 term sharpens edges so the brain
    texture does not drift toward the blurry MSE mean.
    """

    def __init__(self, roi_weight: float = 25.0, periphery_weight: float = 2.0) -> None:
        super().__init__()
        self.roi_weight = float(roi_weight)
        self.periphery_weight = float(periphery_weight)
        self.l1_scale = 0.5

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        roi_mask: torch.Tensor,
    ) -> torch.Tensor:
        if roi_mask.shape != prediction.shape:
            roi_mask = roi_mask.expand_as(prediction)

        weight = self.roi_weight * roi_mask + self.periphery_weight * (1.0 - roi_mask)
        l2 = weight * (prediction - target) ** 2
        l1 = weight * torch.abs(prediction - target)
        return (l2 + self.l1_scale * l1).mean()


# --------------------------------------------------------------------------- #
# Downstream classifier
# --------------------------------------------------------------------------- #
class MedicalResNet18(nn.Module):
    """ImageNet-pretrained ResNet-18 adapted for 1-channel grayscale MRI."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        original_conv = backbone.conv1
        feature_dim = backbone.fc.in_features
        self.backbone = backbone
        # Sum RGB-matched weights into a single grayscale filter; this keeps the
        # pretrained edge detectors instead of discarding them for random noise.
        self.backbone.conv1 = nn.Conv2d(
            1, original_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(
                    original_conv.weight.sum(dim=1, keepdim=True)
                )
        self.backbone.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
