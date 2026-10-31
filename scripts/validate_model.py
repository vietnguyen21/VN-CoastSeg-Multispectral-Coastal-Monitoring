#!/usr/bin/env python3
"""Comprehensive Quantitative Model Validation & Benchmark Evaluation Script.

Evaluates VN-CoastSeg-Mask2Former against authentic VLUCD 30m ground-truth labels
using genuine Landsat 8 Level-2 Surface Reflectance coastal imagery.

Outputs:
  - Confusion Matrix (6x6 discrete coastal taxonomy)
  - Mean Intersection over Union (mIoU)
  - Class-wise IoU (specifically IoU_Mangrove and IoU_Aquaculture)
  - Macro F1-Score & Overall Accuracy (OA)
  - JSON benchmark report saved to reports/model_validation_report.json
"""

import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

from src.data.remapper import VLUCDRemapper, TARGET_CLASSES, CLASS_NAMES
from src.training.metrics import StreamConfusionMatrix
from src.app import load_model, run_model_inference, classify_by_spectral_signatures


def resize_mask_nearest(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Zero-dependency nearest-neighbor 2D array resizing."""
    orig_h, orig_w = mask.shape
    row_idx = (np.arange(target_h) * (orig_h / target_h)).astype(int)
    col_idx = (np.arange(target_w) * (orig_w / target_w)).astype(int)
    row_idx = np.clip(row_idx, 0, orig_h - 1)
    col_idx = np.clip(col_idx, 0, orig_w - 1)
    return mask[row_idx[:, None], col_idx]


def run_validation(
    vlucd_tif: str = "1990-2020VNM_v21.09_L2_30m/1990-2020VNM_v21.09_L2_30m/VLUCD_L2_30m_2020.tif",
    checkpoint_path: str = "checkpoints/smoke_checkpoint.pt",
    output_report: str = "reports/model_validation_report.json",
):
    print("=" * 76)
    print("VN-CoastSeg-Mask2Former: MODEL VALIDATION & QUANTITATIVE BENCHMARK")
    print("=" * 76)

    vlucd_file = Path(vlucd_tif)
    if not vlucd_file.exists():
        print(f"[ERROR] VLUCD dataset not found at {vlucd_file.resolve()}")
        sys.exit(1)

    # Test patches to evaluate
    patches = [
        {
            "name": "Red River Delta (Xuan Thuy National Park)",
            "file": "data/samples/red_river_delta_landsat8_7band.tif",
            "region": "Northern Coastal Vietnam (Nam Dinh)",
            "type": "Estuarine Mangrove & Mudflat",
        },
        {
            "name": "Thai Thuy Coastal Zone",
            "file": "data/samples/ca_mau_landsat8_7band.tif",
            "region": "Northern Coastal Vietnam (Thai Binh)",
            "type": "Aquaculture Ponds & Coastal Dykes",
        },
        {
            "name": "Tien Lang Estuary",
            "file": "data/samples/can_gio_landsat8_7band.tif",
            "region": "Northern Coastal Vietnam (Hai Phong)",
            "type": "Mangrove Wetland & Alluvial Rice",
        },
    ]

    remapper = VLUCDRemapper()
    metrics_tracker = StreamConfusionMatrix(num_classes=6, ignore_index=255)
    model = load_model(checkpoint_path=checkpoint_path)

    evaluated_patches_info = []

    print(f"\n[*] Evaluating against VLUCD 2020 Ground-Truth: {vlucd_file.name}")
    print(f"[*] Checkpoint under test: {checkpoint_path}\n")

    with rasterio.open(str(vlucd_file)) as v_src:
        for p in patches:
            p_path = Path(p["file"])
            if not p_path.exists():
                print(f"[!] Warning: Patch file {p_path} not found. Skipping.")
                continue

            with rasterio.open(str(p_path)) as p_src:
                h, w = p_src.height, p_src.width
                crs = p_src.crs
                bounds = p_src.bounds
                wgs_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
                bands = p_src.read().astype(np.float32)

            # Read matching VLUCD ground truth window
            win = from_bounds(*wgs_bounds, v_src.transform)
            raw_target = v_src.read(1, window=win)

            # Resize to match patch dimensions (256, 256) exactly
            if raw_target.shape != (h, w):
                raw_target = resize_mask_nearest(raw_target, h, w)

            # Remap VLUCD classes to 6 target classes + 255 ignore
            target_mask = remapper.remap(raw_target)

            # Run model prediction
            bands_list = [bands[i].tolist() for i in range(7)]
            pred_mask = np.array(run_model_inference(model, bands_list), dtype=np.int64)

            # Update metrics
            metrics_tracker.update(pred_mask, target_mask)

            # Single patch stats
            valid_px = int(np.sum(target_mask != 255))
            correct_px = int(np.sum((target_mask != 255) & (pred_mask == target_mask)))
            patch_acc = (correct_px / valid_px * 100.0) if valid_px > 0 else 0.0

            patch_summary = {
                "name": p["name"],
                "region": p["region"],
                "type": p["type"],
                "dimensions": f"{w}x{h}",
                "valid_pixels": valid_px,
                "accuracy": round(patch_acc, 2),
            }
            evaluated_patches_info.append(patch_summary)

            print(f"  -> Evaluated: {p['name']} ({p['type']})")
            print(f"     Valid GT Pixels: {valid_px:,} | Patch Accuracy: {patch_acc:.2f}%\n")

    # Compute overall benchmark metrics
    metrics = metrics_tracker.compute()
    cm = metrics_tracker.matrix

    # Compute class-wise precision, recall, and F1
    per_class_results = []
    for c in range(6):
        c_name = CLASS_NAMES[c]
        tp = cm[c][c]
        gt_total = sum(cm[c])
        pred_total = sum(cm[r][c] for r in range(6))
        union = gt_total + pred_total - tp

        iou = (tp / union * 100.0) if union > 0 else 0.0
        prec = (tp / pred_total * 100.0) if pred_total > 0 else 0.0
        rec = (tp / gt_total * 100.0) if gt_total > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_results.append({
            "class_id": c,
            "class_name": c_name,
            "tp": tp,
            "gt_pixels": gt_total,
            "pred_pixels": pred_total,
            "iou": round(iou, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2),
        })

    print("=" * 76)
    print("QUANTITATIVE VALIDATION BENCHMARK RESULTS")
    print("=" * 76)
    oa_pct = metrics.get("OA", 0.0) * 100.0
    miou_pct = metrics.get("mIoU", 0.0) * 100.0
    f1_pct = metrics.get("Macro_F1", 0.0) * 100.0

    print(f"Evaluated Pixels      : {metrics_tracker.total_evaluated_pixels:,} pixels")
    print(f"Overall Accuracy (OA) : {oa_pct:.2f}%")
    print(f"Mean IoU (mIoU)       : {miou_pct:.2f}%")
    print(f"Macro F1-Score        : {f1_pct:.2f}%\n")

    print("Class-wise Segmentation Performance:")
    print("-" * 76)
    print(f"{'Class ID':<9}{'Target Class Name':<32}{'IoU (%)':<11}{'Prec (%)':<11}{'Recall (%)':<11}{'F1 (%)':<9}")
    print("-" * 76)
    for cr in per_class_results:
        print(f"{cr['class_id']:<9}{cr['class_name']:<32}{cr['iou']:<11.2f}{cr['precision']:<11.2f}{cr['recall']:<11.2f}{cr['f1']:<9.2f}")
    print("-" * 76)

    # Export JSON report
    report_data = {
        "benchmark_metadata": {
            "model_architecture": "MultispectralMask2Former (SSL4EO-L ResNet-50 + MSDeformAttn)",
            "checkpoint": checkpoint_path,
            "ground_truth_dataset": "VLUCD 30m Level-2 (Nationwide Vietnam)",
            "sensor": "USGS Landsat 8 OLI Level-2 Surface Reflectance (7 optical channels)",
            "total_pixels_evaluated": metrics_tracker.total_evaluated_pixels,
        },
        "overall_metrics": {
            "overall_accuracy_pct": round(oa_pct, 2),
            "mIoU_pct": round(miou_pct, 2),
            "macro_f1_pct": round(f1_pct, 2),
            "iou_mangrove_pct": round(metrics.get("IoU_Mangrove", 0.0) * 100.0, 2),
            "iou_aquaculture_pct": round(metrics.get("IoU_Aquaculture", 0.0) * 100.0, 2),
        },
        "class_wise_metrics": per_class_results,
        "confusion_matrix": cm,
        "evaluated_patches": evaluated_patches_info,
    }

    out_file = Path(output_report)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Validation report exported to: {out_file.resolve()}")
    print("=" * 76)
    return report_data


if __name__ == "__main__":
    run_validation()
