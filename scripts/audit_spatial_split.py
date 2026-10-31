"""Audit script verifying spatial disjointness and zero data leakage.

Asserts:
    1. Zero bounding box intersection between any blocks assigned to different splits.
    2. Zero shared patch IDs across Train, Val, and Test splits.
    3. Minimum spatial separation buffer >= 1.0 km between splits.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spatial_split import (
    COASTAL_BLOCKS,
    DisjointGeographicSplitter,
    GeographicBlock,
)


def run_spatial_audit() -> bool:
    print("=" * 70)
    print("VN-CoastSeg-Mask2Former: Disjoint Geographic Block Splitting Audit")
    print("=" * 70)

    splitter = DisjointGeographicSplitter(COASTAL_BLOCKS)

    # 1. Block Disjointness Verification
    is_disjoint, violations = splitter.verify_disjointness()
    if not is_disjoint:
        print(f"[FAIL] Spatial overlaps detected between blocks: {violations}")
        return False
    print("[PASS] Pairwise bounding box check: 100% disjoint, 0 overlaps detected across all 12 blocks.")

    # 2. Buffer Margin Auditing
    splits = ["train", "val", "test"]
    all_margins_valid = True
    print("\nSpatial Separation Margins:")
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s_a, s_b = splits[i], splits[j]
            dist_km = splitter.get_min_buffer_margin_km(s_a, s_b)
            status = "PASS" if dist_km >= 1.0 else "FAIL"
            if dist_km < 1.0:
                all_margins_valid = False
            print(f"  - [{status}] Split '{s_a}' <-> Split '{s_b}': min buffer = {dist_km:.2f} km (>= 1.0 km required)")

    if not all_margins_valid:
        print("[FAIL] Buffer separation requirement violated.")
        return False

    # 3. Block distribution across regions
    regions = {}
    split_counts = {}
    for b in COASTAL_BLOCKS:
        regions[b.region] = regions.get(b.region, 0) + 1
        split_counts[b.split] = split_counts.get(b.split, 0) + 1

    print("\nRegional Distribution:")
    for reg, count in regions.items():
        print(f"  - {reg}: {count} blocks")

    print("\nSplit Allocation:")
    for split_name, count in split_counts.items():
        pct = (count / len(COASTAL_BLOCKS)) * 100
        print(f"  - {split_name}: {count} blocks ({pct:.1f}%)")

    # 4. Simulation of Patch Manifest Disjointness
    mock_patches = []
    for b in COASTAL_BLOCKS:
        # Generate 5 test patches inside each block
        for k in range(5):
            lon_c = b.lon_min + (b.lon_max - b.lon_min) * (k + 1) / 7.0
            lat_c = b.lat_min + (b.lat_max - b.lat_min) * (k + 1) / 7.0
            mock_patches.append({
                "patch_id": f"P_{b.block_id}_{k}",
                "lon_min": lon_c - 0.01,
                "lat_min": lat_c - 0.01,
                "lon_max": lon_c + 0.01,
                "lat_max": lat_c + 0.01,
            })

    partitioned = splitter.split_patches(mock_patches)
    train_ids = {p["patch_id"] for p in partitioned["train"]}
    val_ids = {p["patch_id"] for p in partitioned["val"]}
    test_ids = {p["patch_id"] for p in partitioned["test"]}

    inter_train_val = train_ids.intersection(val_ids)
    inter_train_test = train_ids.intersection(test_ids)
    inter_val_test = val_ids.intersection(test_ids)

    assert len(inter_train_val) == 0, f"Leakage: Train & Val share {inter_train_val}"
    assert len(inter_train_test) == 0, f"Leakage: Train & Test share {inter_train_test}"
    assert len(inter_val_test) == 0, f"Leakage: Val & Test share {inter_val_test}"

    print("\nManifest Leakage Audit:")
    print(f"  - [PASS] Train ({len(train_ids)}) intersection Val ({len(val_ids)}) = 0")
    print(f"  - [PASS] Train ({len(train_ids)}) intersection Test ({len(test_ids)}) = 0")
    print(f"  - [PASS] Val ({len(val_ids)}) intersection Test ({len(test_ids)}) = 0")

    print("\n" + "=" * 70)
    print("AUDIT RESULT: ALL SPATIAL SPLIT ASSERTIONS PASSED SUCCESSFULLY")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_spatial_audit()
    sys.exit(0 if success else 1)
