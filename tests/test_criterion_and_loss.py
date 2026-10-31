"""Unit and integration test suite for Hungarian Matcher and Set Prediction Loss."""

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

from src.models.criterion import HungarianMatcher, SetPredictionLoss


class TestCriterionAndLoss(unittest.TestCase):
    """Test suite for Hungarian matching and Set Prediction Loss."""

    def setUp(self):
        self.criterion = SetPredictionLoss(
            num_classes=6,
            cost_weights={"loss_cls": 2.0, "loss_bce": 5.0, "loss_dice": 5.0},
            num_points=1024,
        )

    def test_hungarian_bipartite_matching(self):
        """Matcher must produce optimal bipartite pairing between queries and ground-truth masks."""
        matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0, num_classes=6)

        outputs = {
            "pred_logits": torch.randn(2, 8, 7),
            "pred_masks": torch.randn(2, 8, 32, 32),
        }
        targets = [
            {
                "labels": [0, 1],  # Mangrove, Aquaculture
                "masks": torch.randn(2, 32, 32),
            },
            {
                "labels": [2],     # Open water
                "masks": torch.randn(1, 32, 32),
            },
        ]

        indices = matcher(outputs, targets)
        self.assertEqual(len(indices), 2)

        # Batch item 0: 2 ground truth targets
        src_0, tgt_0 = indices[0]
        self.assertEqual(len(src_0), 2)
        self.assertEqual(len(tgt_0), 2)
        # All matched query indices must be unique (bipartite)
        self.assertEqual(len(set(src_0)), 2)

        # Batch item 1: 1 ground truth target
        src_1, tgt_1 = indices[1]
        self.assertEqual(len(src_1), 1)
        self.assertEqual(len(tgt_1), 1)

    def test_loss_components_and_weighting(self):
        """SetPredictionLoss must compute positive finite loss_cls, loss_bce, loss_dice and weighted sum."""
        outputs = {
            "pred_logits": torch.randn(2, 10, 7),
            "pred_masks": torch.randn(2, 10, 32, 32),
        }
        # Dense targets with classes 0, 1, 2 and ignore_index 255
        targets = torch.tensor([
            [[0, 1, 2], [1, 0, 255]],
            [[2, 1, 0], [0, 255, 255]],
        ])

        loss_dict = self.criterion(outputs, targets)

        self.assertIn("loss", loss_dict)
        self.assertIn("loss_cls", loss_dict)
        self.assertIn("loss_bce", loss_dict)
        self.assertIn("loss_dice", loss_dict)

        l_total = float(loss_dict["loss"].item() if hasattr(loss_dict["loss"], "item") else loss_dict["loss"])
        l_cls = float(loss_dict["loss_cls"].item() if hasattr(loss_dict["loss_cls"], "item") else loss_dict["loss_cls"])
        l_bce = float(loss_dict["loss_bce"].item() if hasattr(loss_dict["loss_bce"], "item") else loss_dict["loss_bce"])
        l_dice = float(loss_dict["loss_dice"].item() if hasattr(loss_dict["loss_dice"], "item") else loss_dict["loss_dice"])

        self.assertGreater(l_total, 0.0)
        self.assertGreater(l_cls, 0.0)
        self.assertGreater(l_bce, 0.0)
        self.assertGreater(l_dice, 0.0)

        # Weighted combination: 2.0 * cls + 5.0 * bce + 5.0 * dice
        expected_total = 2.0 * l_cls + 5.0 * l_bce + 5.0 * l_dice
        self.assertAlmostEqual(l_total, expected_total, places=4)

    def test_deep_supervision_auxiliary_losses(self):
        """Auxiliary decoder layer outputs must contribute to total loss when present."""
        outputs = {
            "pred_logits": torch.randn(2, 8, 7),
            "pred_masks": torch.randn(2, 8, 32, 32),
            "aux_outputs": [
                {
                    "pred_logits": torch.randn(2, 8, 7),
                    "pred_masks": torch.randn(2, 8, 32, 32),
                }
            ],
        }
        targets = torch.tensor([
            [[0, 1], [2, 255]],
            [[1, 0], [255, 255]],
        ])

        loss_dict = self.criterion(outputs, targets)
        self.assertIn("aux_0_loss", loss_dict)
        self.assertGreater(loss_dict["aux_0_loss"].item(), 0.0)

    def test_backward_gradient_computation(self):
        """Loss backpropagation must execute cleanly without NaN or infinite gradients."""
        pred_logits = torch.randn(1, 4, 7, requires_grad=True)
        pred_masks = torch.randn(1, 4, 16, 16, requires_grad=True)
        outputs = {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
        }
        targets = torch.tensor([[[0, 1], [1, 255]]])
        loss_dict = self.criterion(outputs, targets)
        loss = loss_dict["loss"]

        # Backward step
        loss.backward()
        self.assertIsNotNone(pred_logits.grad)
        self.assertIsNotNone(pred_masks.grad)
        self.assertFalse(torch.isnan(pred_logits.grad).any())
        self.assertFalse(torch.isinf(pred_logits.grad).any())


if __name__ == "__main__":
    unittest.main()
