"""Streaming Evaluation Metrics for VN-CoastSeg-Mask2Former.

Maintains an accumulated 2D confusion matrix for the 6 standardized coastal classes:
  0: Mangrove & Coastal Wetland
  1: Aquaculture (Fish/Shrimp Ponds)
  2: Open Water (Rivers, Estuaries, Ocean)
  3: Cropland & Paddy Rice
  4: Terrestrial Forest & Shrubland
  5: Built-up, Urban & Bareland

Strictly masks out ignore_index = 255 (background / non-coastal land cover).
Computes mIoU, class-wise IoU (specifically IoU_Mangrove and IoU_Aquaculture),
Macro F1 score, and Overall Accuracy (OA).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

CLASS_NAMES = [
    "Mangrove & Coastal Wetland",
    "Aquaculture (Fish/Shrimp)",
    "Open Water",
    "Cropland & Paddy Rice",
    "Terrestrial Forest",
    "Built-up / Bareland",
]


class StreamConfusionMatrix:
    """Streaming confusion matrix accumulator for coastal semantic segmentation."""

    def __init__(self, num_classes: int = 6, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix: List[List[int]] = [
            [0] * num_classes for _ in range(num_classes)
        ]
        self.total_evaluated_pixels = 0

    def reset(self):
        """Resets the accumulated confusion matrix to zero."""
        self.matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        self.total_evaluated_pixels = 0

    def update(self, preds: Any, targets: Any):
        """Updates the confusion matrix with predictions and ground-truth masks.

        Args:
            preds: Predicted class labels tensor/array of shape (..., ) or (B, H, W).
            targets: Ground-truth class labels tensor/array of shape (..., ) or (B, H, W).
        """
        # Flatten predictions and targets to lists of ints
        p_flat = self._to_flat_list(preds)
        t_flat = self._to_flat_list(targets)

        if len(p_flat) != len(t_flat):
            raise ValueError(
                f"Prediction count ({len(p_flat)}) and target count ({len(t_flat)}) must match"
            )

        for p, t in zip(p_flat, t_flat):
            # Strictly ignore 255 and out-of-range targets
            if t == self.ignore_index:
                continue
            if not (0 <= t < self.num_classes):
                continue
            # Clip predictions to valid class range
            p_clamped = max(0, min(self.num_classes - 1, int(p)))
            self.matrix[int(t)][p_clamped] += 1
            self.total_evaluated_pixels += 1

    @staticmethod
    def _to_flat_list(arr: Any) -> List[int]:
        if hasattr(arr, "_flat"):
            return [int(x) for x in arr._flat]
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        if hasattr(arr, "tolist"):
            val = arr.tolist()
            return StreamConfusionMatrix._to_flat_list(val)
        if hasattr(arr, "flatten"):
            val = arr.flatten()
            if hasattr(val, "_flat"):
                return [int(x) for x in val._flat]
            return [int(x) for x in val]
        if isinstance(arr, (list, tuple)):
            res = []
            for item in arr:
                if isinstance(item, (list, tuple)):
                    res.extend(StreamConfusionMatrix._to_flat_list(item))
                else:
                    res.append(int(item))
            return res
        return [int(arr)]

    def compute(self) -> Dict[str, float]:
        """Computes mIoU, class-wise IoU, Macro F1, and Overall Accuracy.

        Returns:
            Dictionary with:
              - 'mIoU': Mean Intersection over Union across 6 classes
              - 'IoU_Mangrove': IoU for Class 0
              - 'IoU_Aquaculture': IoU for Class 1
              - 'IoU_OpenWater': IoU for Class 2
              - 'IoU_Cropland': IoU for Class 3
              - 'IoU_Forest': IoU for Class 4
              - 'IoU_Builtup': IoU for Class 5
              - 'Macro_F1': Macro-averaged F1 score
              - 'OA': Overall Accuracy
        """
        ious: List[float] = []
        f1_scores: List[float] = []
        total_correct = 0

        for c in range(self.num_classes):
            tp = self.matrix[c][c]
            row_sum = sum(self.matrix[c][j] for j in range(self.num_classes))
            col_sum = sum(self.matrix[i][c] for i in range(self.num_classes))
            total_correct += tp

            union = row_sum + col_sum - tp
            iou_c = float(tp) / union if union > 0 else 0.0
            ious.append(iou_c)

            f1_den = row_sum + col_sum
            f1_c = (2.0 * float(tp)) / f1_den if f1_den > 0 else 0.0
            f1_scores.append(f1_c)

        miou = sum(ious) / self.num_classes
        macro_f1 = sum(f1_scores) / self.num_classes
        oa = float(total_correct) / self.total_evaluated_pixels if self.total_evaluated_pixels > 0 else 0.0

        metrics = {
            "mIoU": float(miou),
            "Macro_F1": float(macro_f1),
            "OA": float(oa),
            "IoU_Mangrove": float(ious[0]),
            "IoU_Aquaculture": float(ious[1]),
            "IoU_OpenWater": float(ious[2]),
            "IoU_Cropland": float(ious[3]),
            "IoU_Forest": float(ious[4]),
            "IoU_Builtup": float(ious[5]),
        }
        return metrics

    def summary_table(self) -> str:
        """Returns a formatted ASCII summary table of evaluation metrics."""
        metrics = self.compute()
        lines = [
            "=" * 64,
            f"{'VN-CoastSeg Evaluation Metric Summary':^64}",
            "=" * 64,
            f"{'Metric':<36} | {'Value':>12}",
            "-" * 64,
            f"{'Mean IoU (mIoU)':<36} | {metrics['mIoU'] * 100:>11.2f}%",
            f"{'Macro F1 Score':<36} | {metrics['Macro_F1'] * 100:>11.2f}%",
            f"{'Overall Accuracy (OA)':<36} | {metrics['OA'] * 100:>11.2f}%",
            "-" * 64,
            f"{'Class 0: Mangrove & Wetland':<36} | {metrics['IoU_Mangrove'] * 100:>11.2f}%",
            f"{'Class 1: Aquaculture (Shrimp/Fish)':<36} | {metrics['IoU_Aquaculture'] * 100:>11.2f}%",
            f"{'Class 2: Open Water':<36} | {metrics['IoU_OpenWater'] * 100:>11.2f}%",
            f"{'Class 3: Cropland & Paddy':<36} | {metrics['IoU_Cropland'] * 100:>11.2f}%",
            f"{'Class 4: Terrestrial Forest':<36} | {metrics['IoU_Forest'] * 100:>11.2f}%",
            f"{'Class 5: Built-up / Bareland':<36} | {metrics['IoU_Builtup'] * 100:>11.2f}%",
            "=" * 64,
        ]
        return "\n".join(lines)
