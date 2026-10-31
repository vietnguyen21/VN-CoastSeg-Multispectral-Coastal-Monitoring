"""Comprehensive test suite for Disjoint Geographic Block Splitting.

Verifies:
    1. Zero spatial overlap across all 12 coastal blocks.
    2. Buffer separation distance (>= 1.0 km / 0.05 deg) between disjoint splits.
    3. Accurate point-in-block spatial containment and regional assignment.
    4. Mathematical disjointness of Train, Val, and Test patch manifests.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.spatial_split import (
    COASTAL_BLOCKS,
    DisjointGeographicSplitter,
    GeographicBlock,
)


class TestDisjointGeographicSplitting(unittest.TestCase):
    """Test suite for spatial block splitting and zero data leakage verification."""

    def setUp(self):
        self.splitter = DisjointGeographicSplitter(COASTAL_BLOCKS)

    def test_block_count_and_split_proportions(self):
        """Verify 12 coastal blocks distributed across North, South, and Mekong Delta."""
        self.assertEqual(len(COASTAL_BLOCKS), 12)

        regions = {"Northern": 0, "Southern": 0, "Mekong Delta": 0}
        splits = {"train": 0, "val": 0, "test": 0}

        for block in COASTAL_BLOCKS:
            regions[block.region] += 1
            splits[block.split] += 1

        # Exactly 4 blocks per region (2 Train, 1 Val, 1 Test)
        for reg, count in regions.items():
            self.assertEqual(count, 4, f"Region {reg} should have 4 blocks, found {count}")

        # Overall 6 Train (50% blocks, ~70% area), 3 Val (25%), 3 Test (25%)
        self.assertEqual(splits["train"], 6)
        self.assertEqual(splits["val"], 3)
        self.assertEqual(splits["test"], 3)

    def test_absolute_pairwise_block_disjointness(self):
        """Mathematical proof: Every pair of blocks has strictly zero bounding box overlap."""
        is_disjoint, violations = self.splitter.verify_disjointness()
        self.assertTrue(
            is_disjoint,
            f"Spatial disjointness violated! Overlapping blocks found: {violations}",
        )
        self.assertEqual(len(violations), 0)

    def test_minimum_spatial_separation_buffer_margin(self):
        """Verify that blocks of different splits are separated by at least 1.0 km."""
        splits = ["train", "val", "test"]
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s_a, s_b = splits[i], splits[j]
                min_dist_km = self.splitter.get_min_buffer_margin_km(s_a, s_b)
                self.assertGreaterEqual(
                    min_dist_km,
                    1.0,
                    f"Buffer between {s_a} and {s_b} is {min_dist_km:.2f} km, which is less than 1.0 km!",
                )

    def test_ground_truth_site_block_assignments(self):
        """Verify coordinate assignment for known coastal monitoring sites."""
        test_cases = [
            # Xuan Thuy National Park (Nam Dinh) -> N3 (Val)
            (106.55, 20.25, "val", "N3", "Northern"),
            # Cat Ba Island & Ha Long (Quang Ninh) -> N4 (Test)
            (107.05, 20.80, "test", "N4", "Northern"),
            # Thai Binh coastal farm -> N2 (Train)
            (106.55, 20.45, "train", "N2", "Northern"),
            # Can Gio Biosphere Reserve Core -> S1 (Train)
            (106.90, 10.45, "train", "S1", "Southern"),
            # Can Gio North estuary -> S3 (Val)
            (106.90, 10.70, "val", "S3", "Southern"),
            # Vung Tau / Thi Vai coast -> S4 (Test)
            (107.20, 10.45, "test", "S4", "Southern"),
            # Nam Can / Ca Mau aquaculture -> M1 (Train)
            (105.00, 8.95, "train", "M1", "Mekong Delta"),
            # Cape Ca Mau / Dat Mui National Park -> M4 (Test)
            (104.70, 8.65, "test", "M4", "Mekong Delta"),
            # Soc Trang coastal pioneer mangroves -> M3 (Val)
            (106.00, 9.50, "val", "M3", "Mekong Delta"),
            # Points outside coastal blocks (e.g., Hanoi inland)
            (105.85, 21.02, None, None, None),
        ]

        for lon, lat, exp_split, exp_block, exp_region in test_cases:
            assignment = self.splitter.assign_point(lon, lat)
            if exp_split is None:
                self.assertIsNone(
                    assignment,
                    f"Point ({lon}, {lat}) should be outside coastal blocks, but was assigned to {assignment}",
                )
            else:
                self.assertIsNotNone(
                    assignment,
                    f"Point ({lon}, {lat}) should be assigned to {exp_block}, but got None",
                )
                split_name, block_id, region = assignment
                self.assertEqual(split_name, exp_split)
                self.assertEqual(block_id, exp_block)
                self.assertEqual(region, exp_region)

    def test_split_manifest_disjoint_ids(self):
        """Ensure partitioned manifests contain zero shared patch IDs across Train, Val, and Test."""
        mock_patches = [
            {"patch_id": f"P_N1_{i}", "lon_min": 106.70, "lat_min": 20.75, "lon_max": 106.75, "lat_max": 20.80}
            for i in range(10)
        ] + [
            {"patch_id": f"P_N3_{i}", "lon_min": 106.45, "lat_min": 20.15, "lon_max": 106.50, "lat_max": 20.20}
            for i in range(5)
        ] + [
            {"patch_id": f"P_N4_{i}", "lon_min": 107.05, "lat_min": 20.80, "lon_max": 107.10, "lat_max": 20.85}
            for i in range(5)
        ]

        split_dict = self.splitter.split_patches(mock_patches)
        train_ids = {p["patch_id"] for p in split_dict["train"]}
        val_ids = {p["patch_id"] for p in split_dict["val"]}
        test_ids = {p["patch_id"] for p in split_dict["test"]}

        self.assertEqual(len(train_ids), 10)
        self.assertEqual(len(val_ids), 5)
        self.assertEqual(len(test_ids), 5)

        # Pairwise set intersections must be strictly empty
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

    def test_manifest_csv_export(self):
        """Verify export_manifests creates valid CSV files and summary JSON."""
        mock_patches = [
            {
                "patch_id": "P1",
                "lon_min": 106.70,
                "lat_min": 20.75,
                "lon_max": 106.75,
                "lat_max": 20.80,
                "image_path": "data/patches/P1_img.npy",
                "mask_path": "data/patches/P1_mask.npy",
            }
        ]
        split_dict = self.splitter.split_patches(mock_patches)

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.splitter.export_manifests(split_dict, tmp_dir)
            for split_name in ("train", "val", "test"):
                self.assertIn(split_name, paths)
                self.assertTrue(paths[split_name].exists())

            summary_file = Path(tmp_dir) / "split_summary.json"
            self.assertTrue(summary_file.exists())


if __name__ == "__main__":
    unittest.main()
