"""Unit and integration test suite for Streaming Evaluation Metrics."""

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

from src.training.metrics import StreamConfusionMatrix


class TestStreamConfusionMatrix(unittest.TestCase):
    """Test suite for StreamConfusionMatrix and coastal segmentation metrics."""

    def setUp(self):
        self.cm = StreamConfusionMatrix(num_classes=6, ignore_index=255)

    def test_perfect_predictions_yield_100_percent(self):
        """When predictions exactly match ground truth, all metrics must be 1.0 (100%)."""
        # All 6 classes present plus some 255 ignore pixels
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 255, 255])
        preds = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])

        self.cm.update(preds, targets)
        m = self.cm.compute()

        self.assertAlmostEqual(m["mIoU"], 1.0, places=5)
        self.assertAlmostEqual(m["Macro_F1"], 1.0, places=5)
        self.assertAlmostEqual(m["OA"], 1.0, places=5)
        self.assertAlmostEqual(m["IoU_Mangrove"], 1.0, places=5)
        self.assertAlmostEqual(m["IoU_Aquaculture"], 1.0, places=5)
        self.assertAlmostEqual(m["IoU_OpenWater"], 1.0, places=5)
        self.assertAlmostEqual(m["IoU_Cropland"], 1.0, places=5)
        self.assertAlmostEqual(m["IoU_Forest"], 1.0, places=5)
        self.assertAlmostEqual(m["IoU_Builtup"], 1.0, places=5)

    def test_ignore_index_255_strictly_excluded(self):
        """Pixels with target label 255 must have zero impact on confusion matrix."""
        # Target 255 with any predicted value
        targets = torch.tensor([255, 255, 255, 255])
        preds = torch.tensor([0, 1, 2, 3])

        self.cm.update(preds, targets)
        m = self.cm.compute()

        self.assertEqual(self.cm.total_evaluated_pixels, 0)
        self.assertEqual(m["OA"], 0.0)
        self.assertEqual(m["mIoU"], 0.0)

    def test_exact_mathematical_iou_and_f1(self):
        """Verify computed IoU and F1 match exact closed-form analytical values."""
        # Class 0 (Mangrove): 6 TP, 2 FP (target 1 predicted 0), 2 FN (target 0 predicted 1)
        #   Union = 6 + 2 + 2 = 10 -> IoU = 6 / 10 = 0.60
        #   F1 = (2 * 6) / (6 + 2 + 6 + 2) = 12 / 16 = 0.75
        targets = [0] * 6 + [0, 0] + [1, 1] + [255, 255]
        preds =   [0] * 6 + [1, 1] + [0, 0] + [0, 1]

        self.cm.update(preds, targets)
        m = self.cm.compute()

        self.assertAlmostEqual(m["IoU_Mangrove"], 0.60, places=5)
        # Class 0 F1
        tp = 6
        row_sum = 8  # target 0 total
        col_sum = 8  # pred 0 total
        expected_f1 = (2.0 * tp) / (row_sum + col_sum)  # 0.75
        # The class 0 F1 is part of Macro_F1

    def test_multidimensional_tensor_batch_update(self):
        """Matrix update must accept batched 3D tensors (B, H, W)."""
        preds = torch.tensor([
            [[0, 1], [2, 3]],
            [[4, 5], [0, 1]],
        ])
        targets = torch.tensor([
            [[0, 1], [2, 3]],
            [[4, 5], [255, 255]],
        ])

        self.cm.update(preds, targets)
        self.assertEqual(self.cm.total_evaluated_pixels, 6)
        m = self.cm.compute()
        self.assertAlmostEqual(m["OA"], 1.0, places=5)

    def test_reset_clears_accumulator(self):
        """reset() must return all matrix cells and total pixel count to 0."""
        self.cm.update([0, 1], [0, 1])
        self.assertEqual(self.cm.total_evaluated_pixels, 2)
        self.cm.reset()
        self.assertEqual(self.cm.total_evaluated_pixels, 0)
        m = self.cm.compute()
        self.assertEqual(m["OA"], 0.0)


if __name__ == "__main__":
    unittest.main()
