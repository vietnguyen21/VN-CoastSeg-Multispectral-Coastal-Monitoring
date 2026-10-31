"""Comprehensive unit and integration test suite for VN-CoastSeg Data Pipeline.

Tests:
    1. VLUCD 18->6 class remapping and ignore_index=255.
    2. Zero-dependency GeoTIFF windowed reader and real VLUCD 2020 raster metadata.
    3. Spatial 256x256 patch extraction, coastal filtering, and spectral simulation.
    4. PyTorch Dataset and DataLoader loading (7, 256, 256) images and (256, 256) masks.
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.dataset import (
    SSL4EO_L_LANDSAT8_MEAN,
    SSL4EO_L_LANDSAT8_STD,
    CoastalSegmentationDataset,
    SimpleDataLoader,
    create_dataloader,
)
from src.data.patch_extractor import (
    GeoTIFFWindowReader,
    PatchExtractor,
    PatchMetadata,
)
from src.data.remapper import (
    CLASS_NAMES,
    CLASS_PALETTE,
    IGNORE_INDEX,
    NUM_CLASSES,
    RAW_VLUCD_TO_TARGET,
    REMAP_LUT,
    TARGET_CLASSES,
    VLUCDRemapper,
    remap_vlucd,
)

RAW_TIFF_PATH = Path(
    "/mnt/c/Workspace/earth_observations/1990-2020VNM_v21.09_L2_30m/1990-2020VNM_v21.09_L2_30m/VLUCD_L2_30m_2020.tif"
)


class TestVLUCDRemapper(unittest.TestCase):
    """Test suite for VLUCD 18->6 class remapper."""

    def setUp(self):
        self.remapper = VLUCDRemapper()

    def test_all_18_raw_classes_mapped_correctly(self):
        """Verify each of the 18 VLUCD Level-2 discrete classes maps to the exact target class."""
        expected_mappings = {
            # Mangrove & Coastal Wetland (Class 0)
            15: 0,
            16: 0,
            # Aquaculture (Class 1)
            19: 1,
            # Open Water (Class 2)
            18: 2,
            # Cropland & Paddy Rice (Class 3)
            3: 3,
            4: 3,
            5: 3,
            6: 3,
            # Terrestrial Forest & Shrubland (Class 4)
            7: 4,
            9: 4,
            10: 4,
            11: 4,
            12: 4,
            14: 4,
            20: 4,
            # Built-up, Urban & Bareland (Class 5)
            1: 5,
            2: 5,
            8: 5,
            # Background / NoData (ignore_index 255)
            0: 255,
            13: 255,
            17: 255,
            255: 255,
        }

        for raw_val, expected_target in expected_mappings.items():
            mapped = self.remapper.remap(raw_val)
            self.assertEqual(
                mapped,
                expected_target,
                f"Raw class {raw_val} expected {expected_target}, got {mapped}",
            )

    def test_lut_completeness_and_bounds(self):
        """Ensure LUT contains 256 values and all mapped indices are in {0..5, 255}."""
        self.assertEqual(len(REMAP_LUT), 256)
        valid_indices = set(range(6)) | {IGNORE_INDEX}
        for i, val in enumerate(REMAP_LUT):
            self.assertIn(
                val,
                valid_indices,
                f"LUT entry {i} has invalid target index {val}",
            )

    def test_nested_list_remapping(self):
        """Test remapping on 2D nested list simulating a 4x4 image patch."""
        patch = [
            [15, 16, 19, 18],
            [3, 7, 1, 0],
            [11, 20, 8, 255],
            [13, 17, 15, 19],
        ]
        expected = [
            [0, 0, 1, 2],
            [3, 4, 5, 255],
            [4, 4, 5, 255],
            [255, 255, 0, 1],
        ]
        remapped = self.remapper.remap(patch)
        self.assertEqual(remapped, expected)

    def test_bytes_buffer_remapping(self):
        """Test high-speed remapping on raw byte buffers."""
        raw_bytes = bytes([15, 19, 18, 3, 7, 1, 0, 255])
        expected_bytes = bytes([0, 1, 2, 3, 4, 5, 255, 255])
        remapped = self.remapper.remap(raw_bytes)
        self.assertEqual(remapped, expected_bytes)

    def test_gis_indexing_conversion(self):
        """Test conversion of tensor indices (0..5, 255) to GIS 1-based indexing (1..6, 0)."""
        tensor_indices = [0, 1, 2, 3, 4, 5, 255]
        expected_gis = [1, 2, 3, 4, 5, 6, 0]
        gis_mapped = self.remapper.to_gis_classes(tensor_indices)
        self.assertEqual(gis_mapped, expected_gis)

    def test_palette_and_metadata_consistency(self):
        """Ensure palettes, class names, and target dictionaries have consistent counts."""
        self.assertEqual(len(CLASS_NAMES), NUM_CLASSES)
        self.assertEqual(len(TARGET_CLASSES), NUM_CLASSES + 1)
        for c in range(NUM_CLASSES):
            self.assertIn(c, CLASS_PALETTE)
            r, g, b = CLASS_PALETTE[c]
            self.assertTrue(0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255)
        self.assertIn(IGNORE_INDEX, CLASS_PALETTE)


class TestGeoTIFFWindowReader(unittest.TestCase):
    """Test suite for GeoTIFF header parsing and windowed reading."""

    @unittest.skipUnless(RAW_TIFF_PATH.exists(), "Raw VLUCD GeoTIFF file not present")
    def test_raw_geotiff_header_and_dimensions(self):
        """Verify binary IFD tag parsing on actual 1.5 GB VLUCD 2020 GeoTIFF."""
        reader = GeoTIFFWindowReader(RAW_TIFF_PATH)
        self.assertEqual(reader.width, 27916)
        self.assertEqual(reader.length, 55526)
        self.assertEqual(reader.bits_per_sample, 8)
        self.assertEqual(reader.compression, 1)  # Uncompressed
        self.assertAlmostEqual(reader.origin_x, 102.0414297536, places=4)
        self.assertAlmostEqual(reader.origin_y, 23.4231218758, places=4)
        self.assertAlmostEqual(reader.scale_x, 0.0002694945, places=6)
        self.assertAlmostEqual(reader.scale_y, 0.0002694945, places=6)

    @unittest.skipUnless(RAW_TIFF_PATH.exists(), "Raw VLUCD GeoTIFF file not present")
    def test_coordinate_transformation_roundtrip(self):
        """Ensure pixel <-> geographic coordinate conversion is accurate and reversible."""
        reader = GeoTIFFWindowReader(RAW_TIFF_PATH)
        test_col, test_row = 18000, 48000
        lon, lat = reader.pixel_to_geo(test_col, test_row)
        rec_col, rec_row = reader.geo_to_pixel(lon, lat)
        self.assertEqual(test_col, rec_col)
        self.assertEqual(test_row, rec_row)

    @unittest.skipUnless(RAW_TIFF_PATH.exists(), "Raw VLUCD GeoTIFF file not present")
    def test_windowed_read_can_gio_mangrove_patch(self):
        """Verify windowed read extracts a valid 256x256 patch without high memory usage."""
        reader = GeoTIFFWindowReader(RAW_TIFF_PATH)
        col = 18028
        row = 48138
        window = reader.read_window(col, row, width=256, height=256)
        self.assertEqual(len(window), 256)
        self.assertEqual(len(window[0]), 256)

        # Count raw values
        counts = {}
        for r in window:
            for v in r:
                counts[v] = counts.get(v, 0) + 1

        # Can Gio coordinate must contain Mangrove Forest (15) and Open Water (18)
        self.assertIn(15, counts)
        self.assertIn(18, counts)
        self.assertGreater(counts[15], 1000)


class TestPatchExtractor(unittest.TestCase):
    """Test suite for coastal patch extraction and quality filtering."""

    @unittest.skipUnless(RAW_TIFF_PATH.exists(), "Raw VLUCD GeoTIFF file not present")
    def test_extract_and_filter_can_gio_patch(self):
        """Verify patch extraction, remapping, coastal target counting, and block assignment."""
        extractor = PatchExtractor(
            vlucd_tiff_path=RAW_TIFF_PATH,
            patch_size=256,
            target_min_ratio=0.05,
            nodata_max_ratio=0.50,
        )

        col = 18028
        row = 48138
        mask, meta = extractor.extract_patch(col, row)

        self.assertEqual(len(mask), 256)
        self.assertEqual(len(mask[0]), 256)
        self.assertIsInstance(meta, PatchMetadata)

        # Can Gio is in Southern Region
        self.assertEqual(meta.region, "Southern")
        self.assertTrue(meta.is_valid)
        self.assertGreaterEqual(meta.target_pixel_count, 3277)
        self.assertLessEqual(meta.nodata_count, 32768)

    def test_synthetic_7band_patch_properties(self):
        """Verify synthetic Landsat 8 7-band reflectance generation."""
        # Create a synthetic 256x256 mask with mangrove (0), water (2), and bareland (5)
        mask = [[0 if c < 128 else 2 for c in range(256)] for _ in range(256)]
        bands = PatchExtractor.generate_synthetic_7band_patch(None, mask)

        self.assertEqual(len(bands), 7)
        self.assertEqual(len(bands[0]), 256)
        self.assertEqual(len(bands[0][0]), 256)

        # Mangrove region (c=10) has high NIR (Band 5, index 4)
        mangrove_nir = bands[4][10][10]
        mangrove_red = bands[3][10][10]
        self.assertGreater(mangrove_nir, mangrove_red)
        self.assertAlmostEqual(mangrove_nir, 0.35, places=2)

        # Water region (c=200) has near-zero NIR (Band 5, index 4)
        water_nir = bands[4][10][200]
        self.assertLess(water_nir, 0.05)


class TestCoastalSegmentationDataset(unittest.TestCase):
    """Test suite for PyTorch Dataset, augmentations, and DataLoader."""

    def setUp(self):
        # Create synthetic samples for unit testing
        h, w = 256, 256
        self.mock_samples = []
        for i in range(10):
            mask = [[(r * 6 // h) for c in range(w)] for r in range(h)]
            # Ensure mangrove & aquaculture presence
            for r in range(50):
                for c in range(100):
                    mask[r][c] = 0 if c < 50 else 1
            image = PatchExtractor.generate_synthetic_7band_patch(None, mask)
            self.mock_samples.append({
                "patch_id": f"mock_patch_{i}",
                "split": "train" if i < 7 else ("val" if i < 9 else "test"),
                "image": image,
                "mask": mask,
            })

    def test_dataset_item_shape_and_types(self):
        """Dataset must yield image of shape (7, 256, 256) and mask (256, 256)."""
        dataset = CoastalSegmentationDataset(
            samples=self.mock_samples,
            is_train=False,
            normalize=True,
            norm_type="minmax",
        )
        self.assertEqual(len(dataset), 10)

        image, mask = dataset[0]

        # Shape checks
        if hasattr(image, "shape"):
            self.assertEqual(tuple(image.shape), (7, 256, 256))
            self.assertEqual(tuple(mask.shape), (256, 256))
        else:
            self.assertEqual(len(image), 7)
            self.assertEqual(len(image[0]), 256)
            self.assertEqual(len(mask), 256)

    def test_zscore_normalization(self):
        """Dataset must support z-score normalization with SSL4EO-L statistics."""
        dataset = CoastalSegmentationDataset(
            samples=self.mock_samples,
            is_train=False,
            normalize=True,
            norm_type="zscore",
        )
        image, _ = dataset[0]
        # Should execute without error and produce normalized values
        if hasattr(image, "shape"):
            self.assertEqual(tuple(image.shape), (7, 256, 256))
        else:
            self.assertEqual(len(image), 7)

    def test_training_augmentations_preserve_shapes_and_classes(self):
        """Augmentations must retain spatial dimensions and valid class label set."""
        dataset = CoastalSegmentationDataset(
            samples=self.mock_samples,
            is_train=True,  # Enables random flips and 90-deg rotations
            normalize=True,
        )

        valid_classes = set(range(6)) | {IGNORE_INDEX}
        for _ in range(5):
            image, mask = dataset[0]
            if hasattr(mask, "tolist"):
                mask_list = mask.tolist()
            else:
                mask_list = mask

            # Dimension checks
            self.assertEqual(len(mask_list), 256)
            self.assertEqual(len(mask_list[0]), 256)

            # Class value domain check
            for r in range(256):
                for c in range(256):
                    self.assertIn(mask_list[r][c], valid_classes)

    def test_simple_dataloader_batching(self):
        """DataLoader must yield batched tensors (B, 7, 256, 256) and (B, 256, 256)."""
        dataset = CoastalSegmentationDataset(samples=self.mock_samples, is_train=False)
        dataloader = create_dataloader(dataset, batch_size=3, shuffle=False)

        batch_count = 0
        total_samples = 0
        for batch_img, batch_msk in dataloader:
            batch_count += 1
            if hasattr(batch_img, "shape"):
                b_size = batch_img.shape[0]
                self.assertEqual(batch_img.shape[1], 7)
                self.assertEqual(batch_img.shape[2], 256)
                self.assertEqual(batch_img.shape[3], 256)
                self.assertEqual(batch_msk.shape[1], 256)
                self.assertEqual(batch_msk.shape[2], 256)
            else:
                b_size = len(batch_img)
            total_samples += b_size

        self.assertEqual(batch_count, 4)  # 10 samples with batch_size 3 = 4 batches
        self.assertEqual(total_samples, 10)


if __name__ == "__main__":
    unittest.main()
