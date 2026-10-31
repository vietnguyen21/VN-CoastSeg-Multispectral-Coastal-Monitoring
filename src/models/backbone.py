"""Multispectral SSL4EO-L ResNet-50 Feature Extractor for VN-CoastSeg-Mask2Former.

Customized 7-channel ResNet-50 backbone pre-trained on Landsat 8 surface reflectance
imagery (SSL4EO-L Collection 2, MOCO v2). Extracts hierarchical multi-scale feature
maps {C2, C3, C4, C5} at strides {4, 8, 16, 32} with channels {256, 512, 1024, 2048}.

Physical Channel Mapping:
  - Band 1: Coastal Aerosol (0.43 - 0.45 um)
  - Band 2: Blue (0.45 - 0.51 um)
  - Band 3: Green (0.53 - 0.59 um)
  - Band 4: Red (0.64 - 0.67 um)
  - Band 5: Near Infrared NIR (0.85 - 0.88 um) - Primary Mangrove Discriminator
  - Band 6: Shortwave Infrared SWIR-1 (1.57 - 1.65 um) - Aquaculture/Moisture
  - Band 7: Shortwave Infrared SWIR-2 (2.11 - 2.29 um) - Salinity/Soil Substrate
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    from src.models.torch_compat import torch, nn, F
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class BottleneckBlock(nn.Module):
    """ResNet Bottleneck block with 1x1, 3x3, and 1x1 convolutions."""
    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Any) -> Any:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class SSL4EOLandsatResNet50(nn.Module):
    """SSL4EO-L ResNet-50 feature extractor accepting 7-band Landsat 8 inputs.

    Output multi-scale feature maps:
      - C2: stride 4,  256 channels,  (B, 256, 64, 64)
      - C3: stride 8,  512 channels,  (B, 512, 32, 32)
      - C4: stride 16, 1024 channels, (B, 1024, 16, 16)
      - C5: stride 32, 2048 channels, (B, 2048, 8, 8)
    """

    def __init__(
        self,
        in_channels: int = 7,
        pretrained_path: Optional[str] = None,
        freeze_at: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = 64

        # conv1 modified for 7 spectral channels
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # ResNet-50 stages: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(planes=64, num_blocks=3, stride=1)   # C2: 256
        self.layer2 = self._make_layer(planes=128, num_blocks=4, stride=2)  # C3: 512
        self.layer3 = self._make_layer(planes=256, num_blocks=6, stride=2)  # C4: 1024
        self.layer4 = self._make_layer(planes=512, num_blocks=3, stride=2)  # C5: 2048

        self.out_channels = [256, 512, 1024, 2048]
        self.out_strides = [4, 8, 16, 32]

        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
        else:
            logger.info("SSL4EO-L ResNet-50 initialized with default weights (no checkpoint path specified).")

        if freeze_at > 0:
            self.freeze_stages(freeze_at)

    def _make_layer(self, planes: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * BottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * BottleneckBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BottleneckBlock.expansion),
            )

        layers = []
        layers.append(
            BottleneckBlock(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * BottleneckBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(in_planes=self.in_planes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x: Any) -> Dict[str, Any]:
        """Forward feature extraction.

        Args:
            x: Input tensor of shape (B, 7, H, W), typically (B, 7, 256, 256).

        Returns:
            Dictionary containing {'c2': C2, 'c3': C3, 'c4': C4, 'c5': C5}.
        """
        # Standard forward pass through ResNet-50 convolutions
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # MaxPool with stride 2
        if HAS_TORCH:
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        else:
            out = F.interpolate(out, scale_factor=0.5, mode="nearest")

        c2 = self.layer1(out)  # stride 4
        c3 = self.layer2(c2)   # stride 8
        c4 = self.layer3(c3)   # stride 16
        c5 = self.layer4(c4)   # stride 32

        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}

    def _project_channels(self, feat: Any, target_dim: int) -> Any:
        """Expands/projects channels to target dimension in fallback mode."""
        b, c_in, h, w = feat.shape
        if c_in == target_dim:
            return feat
        out = [0.0] * (b * target_dim * h * w)
        flat = feat._flat
        stride_hw = h * w
        for bi in range(b):
            for ci in range(target_dim):
                src_c = ci % c_in
                weight = 1.0 + 0.05 * (ci // c_in)
                src_base = bi * (c_in * stride_hw) + src_c * stride_hw
                dst_base = bi * (target_dim * stride_hw) + ci * stride_hw
                for p in range(stride_hw):
                    out[dst_base + p] = flat[src_base + p] * weight
        from src.models.torch_compat import Tensor
        return Tensor(out, shape=(b, target_dim, h, w))

    def load_pretrained_weights(self, checkpoint_path: str):
        """Loads SSL4EO-L pre-trained weights from local path or cache."""
        if not os.path.exists(checkpoint_path):
            logger.warning(
                "SSL4EO-L pre-trained checkpoint not found at %s. Using initialized weights.",
                checkpoint_path,
            )
            return

        try:
            if HAS_TORCH:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # Adapt conv1 if checkpoint has 3 channels instead of 7
                if "conv1.weight" in state_dict and state_dict["conv1.weight"].shape[1] == 3:
                    w3 = state_dict["conv1.weight"]
                    w7 = torch.zeros(64, 7, 7, 7, dtype=w3.dtype)
                    # Map standard RGB to bands 4, 3, 2, NIR to 5, SWIR to 6, 7, Coastal to 1
                    w7[:, 3, :, :] = w3[:, 0, :, :]
                    w7[:, 2, :, :] = w3[:, 1, :, :]
                    w7[:, 1, :, :] = w3[:, 2, :, :]
                    w7[:, 4, :, :] = w3[:, 0, :, :]  # NIR
                    w7[:, 5, :, :] = w3[:, 1, :, :]  # SWIR1
                    w7[:, 6, :, :] = w3[:, 2, :, :]  # SWIR2
                    w7[:, 0, :, :] = w3[:, 2, :, :]  # Coastal
                    state_dict["conv1.weight"] = w7
                self.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded SSL4EO-L pre-trained weights from %s", checkpoint_path)
        except Exception as e:
            logger.warning("Failed to load weights from %s: %s. Using initialized weights.", checkpoint_path, e)

    def freeze_stages(self, stage: int = 1):
        """Freezes backbone stages up to specified stage (1..4)."""
        if stage >= 1:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.bn1.parameters():
                p.requires_grad = False
            for p in self.layer1.parameters():
                p.requires_grad = False
        if stage >= 2:
            for p in self.layer2.parameters():
                p.requires_grad = False
        if stage >= 3:
            for p in self.layer3.parameters():
                p.requires_grad = False
        if stage >= 4:
            for p in self.layer4.parameters():
                p.requires_grad = False


def build_ssl4eo_resnet50(
    in_chans: int = 7,
    pretrained_path: Optional[str] = None,
    freeze_at: int = 0,
) -> SSL4EOLandsatResNet50:
    """Factory function for SSL4EO-L ResNet-50 backbone."""
    return SSL4EOLandsatResNet50(
        in_channels=in_chans,
        pretrained_path=pretrained_path,
        freeze_at=freeze_at,
    )
