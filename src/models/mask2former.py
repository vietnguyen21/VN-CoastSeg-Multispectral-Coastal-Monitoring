"""Unified Multispectral Mask2Former for Vietnam Coastal Segmentation (VN-CoastSeg).

Integrates:
  1. SSL4EO-L ResNet-50 7-band feature extractor -> {C2, C3, C4, C5}.
  2. Pure PyTorch MSDeformAttn Pixel Decoder -> per-pixel embeddings (B, 256, 64, 64).
  3. Masked-Attention Transformer Decoder -> pred_logits (B, Q, 7) & pred_masks (B, Q, 64, 64).
  4. End-to-end dense semantic segmentation inference method predict_semantic_mask.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    from src.models.torch_compat import torch, nn, F
    HAS_TORCH = False

from src.models.backbone import SSL4EOLandsatResNet50, build_ssl4eo_resnet50
from src.models.pixel_decoder import MSDeformAttnPixelDecoder
from src.models.transformer_decoder import MaskedAttentionDecoder


class MultispectralMask2Former(nn.Module):
    """Multispectral Mask2Former for Coastal Mangrove & Aquaculture Segmentation.

    Attributes:
        backbone: SSL4EO-L ResNet-50 7-band feature extractor.
        pixel_decoder: MSDeformAttn pixel decoder fusing multi-scale features.
        transformer_decoder: Masked-attention query decoder.
        num_classes: Number of foreground coastal target classes (default 6).
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_classes: int = 6,
        num_queries: int = 50,
        hidden_dim: int = 256,
        num_decoder_layers: int = 9,
        pretrained_backbone_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 1. 7-band SSL4EO-L ResNet-50 Backbone
        self.backbone = build_ssl4eo_resnet50(
            in_chans=in_channels,
            pretrained_path=pretrained_backbone_path,
        )

        # 2. MSDeformAttn Pixel Decoder
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_channels=[256, 512, 1024, 2048],
            hidden_dim=hidden_dim,
        )

        # 3. Masked Attention Transformer Decoder
        self.transformer_decoder = MaskedAttentionDecoder(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
        )

    def forward(self, x: Any) -> Dict[str, Any]:
        """Forward pass.

        Args:
            x: Input Landsat 8 7-band tensor (B, 7, H, W), e.g. (B, 7, 256, 256).

        Returns:
            Dictionary containing:
              - 'pred_logits': (B, num_queries, num_classes + 1)
              - 'pred_masks':  (B, num_queries, 64, 64)
              - 'aux_outputs': List of intermediate decoder layer outputs.
        """
        features = self.backbone(x)
        mask_features, multi_scale_features = self.pixel_decoder(features)
        outputs = self.transformer_decoder(multi_scale_features, mask_features)
        return outputs

    def predict_semantic_mask(self, x: Any) -> Any:
        """Inference method returning dense semantic segmentation map.

        Args:
            x: Input Landsat 8 7-band tensor (B, 7, H, W).

        Returns:
            Tensor of shape (B, H, W), with class predictions in {0..num_classes - 1}.
        """
        b, c, h, w = x.shape
        outputs = self.forward(x)

        # Class probabilities excluding no-object class: (B, Q, num_classes)
        pred_logits = outputs["pred_logits"]
        if HAS_TORCH:
            class_probs = F.softmax(pred_logits, dim=-1)[..., :-1]
        else:
            sm = pred_logits.softmax(dim=-1)
            class_probs = sm[:, :, : self.num_classes]

        # Mask probabilities: (B, Q, 64, 64)
        pred_masks = outputs["pred_masks"]

        if HAS_TORCH:
            mask_probs = torch.sigmoid(pred_masks)
            # Combine queries into 6 class probabilities at (B, num_classes, 64, 64)
            sem_probs_low = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
            # Upsample the 6 class probability maps to input resolution (B, num_classes, H, W)
            sem_probs = F.interpolate(sem_probs_low, size=(h, w), mode="bilinear", align_corners=False)
            return torch.argmax(sem_probs, dim=1)
        else:
            mask_probs = pred_masks.sigmoid()
            from src.models.torch_compat import torch as _t
            sem_probs_low = _t.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)
            sem_probs = F.interpolate(sem_probs_low, size=(h, w), mode="bilinear")
            return _t.argmax(sem_probs, dim=1)


# Project alias as specified in ADRs and original request
VNCoastSegMask2Former = MultispectralMask2Former
