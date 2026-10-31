"""Unit and integration test suite for VN-CoastSeg-Mask2Former Architecture.

Validates:
  1. 7-band SSL4EO-L ResNet-50 backbone multi-scale feature shapes {C2, C3, C4, C5}.
  2. Multi-Scale Deformable Attention Pixel Decoder per-pixel embeddings (B, 256, 64, 64).
  3. Masked-Attention Transformer Decoder query predictions and auxiliary supervision.
  4. End-to-end MultispectralMask2Former forward pass and semantic mask inference.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    from src.models.torch_compat import torch
    HAS_TORCH = False

from src.models.backbone import SSL4EOLandsatResNet50, build_ssl4eo_resnet50
from src.models.mask2former import MultispectralMask2Former, VNCoastSegMask2Former
from src.models.pixel_decoder import MSDeformAttnPixelDecoder
from src.models.transformer_decoder import MaskedAttentionDecoder


class TestMultispectralBackbone(unittest.TestCase):
    """Test suite for SSL4EO-L 7-band ResNet-50 feature extractor."""

    def setUp(self):
        self.backbone = build_ssl4eo_resnet50(in_chans=7)

    def test_backbone_conv1_accepts_7_channels(self):
        """conv1.weight must accept 7 input channels with shape [64, 7, 7, 7]."""
        self.assertEqual(self.backbone.conv1.in_channels, 7)
        self.assertEqual(self.backbone.conv1.out_channels, 64)
        if hasattr(self.backbone.conv1.weight, "shape"):
            self.assertEqual(tuple(self.backbone.conv1.weight.shape), (64, 7, 7, 7))

    def test_backbone_hierarchical_feature_shapes(self):
        """Input (B=2, C=7, H=256, W=256) must produce C2..C5 at strides 4, 8, 16, 32."""
        x = torch.randn(2, 7, 256, 256)
        feats = self.backbone(x)

        self.assertIn("c2", feats)
        self.assertIn("c3", feats)
        self.assertIn("c4", feats)
        self.assertIn("c5", feats)

        # C2: stride 4 -> 64x64, 256 channels
        self.assertEqual(tuple(feats["c2"].shape), (2, 256, 64, 64))
        # C3: stride 8 -> 32x32, 512 channels
        self.assertEqual(tuple(feats["c3"].shape), (2, 512, 32, 32))
        # C4: stride 16 -> 16x16, 1024 channels
        self.assertEqual(tuple(feats["c4"].shape), (2, 1024, 16, 16))
        # C5: stride 32 -> 8x8, 2048 channels
        self.assertEqual(tuple(feats["c5"].shape), (2, 2048, 8, 8))

    def test_backbone_stage_freezing(self):
        """Freezing stage 1 must set requires_grad=False on conv1, bn1, and layer1."""
        backbone = build_ssl4eo_resnet50(in_chans=7, freeze_at=1)
        for p in backbone.conv1.parameters():
            self.assertFalse(p.requires_grad)
        for p in backbone.layer1.parameters():
            self.assertFalse(p.requires_grad)


class TestPixelDecoder(unittest.TestCase):
    """Test suite for MSDeformAttn Pixel Decoder."""

    def setUp(self):
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_channels=[256, 512, 1024, 2048],
            hidden_dim=256,
        )

    def test_pixel_decoder_output_dimensions(self):
        """Pixel decoder must return per-pixel embeddings (B, 256, 64, 64) and 3 multi-scale levels."""
        mock_features = {
            "c2": torch.randn(2, 256, 64, 64),
            "c3": torch.randn(2, 512, 32, 32),
            "c4": torch.randn(2, 1024, 16, 16),
            "c5": torch.randn(2, 2048, 8, 8),
        }
        mask_feat, ms_feats = self.pixel_decoder(mock_features)

        # Per-pixel embeddings E_pixel: (B, 256, 64, 64)
        self.assertEqual(tuple(mask_feat.shape), (2, 256, 64, 64))

        # 3 Multi-scale levels for Transformer Decoder: [stride 32, stride 16, stride 8]
        self.assertEqual(len(ms_feats), 3)
        self.assertEqual(tuple(ms_feats[0].shape), (2, 256, 8, 8))
        self.assertEqual(tuple(ms_feats[1].shape), (2, 256, 16, 16))
        self.assertEqual(tuple(ms_feats[2].shape), (2, 256, 32, 32))


class TestTransformerDecoder(unittest.TestCase):
    """Test suite for Masked-Attention Transformer Decoder."""

    def setUp(self):
        self.decoder = MaskedAttentionDecoder(
            num_classes=6,
            num_queries=32,
            hidden_dim=256,
            num_layers=3,
        )

    def test_transformer_decoder_prediction_shapes(self):
        """Transformer decoder must predict logits (B, Q, 7) and masks (B, Q, 64, 64)."""
        ms_feats = [
            torch.randn(2, 256, 8, 8),
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 256, 32, 32),
        ]
        mask_feat = torch.randn(2, 256, 64, 64)

        out = self.decoder(ms_feats, mask_feat)
        self.assertIn("pred_logits", out)
        self.assertIn("pred_masks", out)
        self.assertIn("aux_outputs", out)

        # 6 coastal classes + 1 no-object = 7 logits
        self.assertEqual(tuple(out["pred_logits"].shape), (2, 32, 7))
        self.assertEqual(tuple(out["pred_masks"].shape), (2, 32, 64, 64))
        # Deep supervision auxiliary outputs from 2 preceding layers
        self.assertEqual(len(out["aux_outputs"]), 2)


class TestUnifiedMultispectralMask2Former(unittest.TestCase):
    """Test suite for end-to-end MultispectralMask2Former module."""

    def setUp(self):
        self.model = MultispectralMask2Former(
            in_channels=7,
            num_classes=6,
            num_queries=32,
            hidden_dim=256,
            num_decoder_layers=3,
        )

    def test_model_forward_pass_shapes(self):
        """Unified model consumes (B=2, 7, 256, 256) and returns expected prediction dict."""
        x = torch.randn(2, 7, 256, 256)
        outputs = self.model(x)

        self.assertIn("pred_logits", outputs)
        self.assertIn("pred_masks", outputs)
        self.assertEqual(tuple(outputs["pred_logits"].shape), (2, 32, 7))
        self.assertEqual(tuple(outputs["pred_masks"].shape), (2, 32, 64, 64))

    def test_semantic_mask_inference_shape_and_classes(self):
        """Inference method produces dense (B, 256, 256) mask with values in {0..5}."""
        x = torch.randn(2, 7, 256, 256)
        pred_mask = self.model.predict_semantic_mask(x)

        self.assertEqual(tuple(pred_mask.shape), (2, 256, 256))
        # Verify predicted labels are valid class indices 0..5
        flat_labels = (
            pred_mask.flatten().tolist()
            if hasattr(pred_mask, "flatten")
            else pred_mask._flat
        )
        valid_set = set(range(6))
        for val in flat_labels[:100]:  # check sample
            self.assertIn(int(val), valid_set)


if __name__ == "__main__":
    unittest.main()
