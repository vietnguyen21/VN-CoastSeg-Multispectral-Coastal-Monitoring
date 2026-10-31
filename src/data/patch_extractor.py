"""Patch Extractor & Windowed GeoTIFF Reader for VN-CoastSeg-Mask2Former.

Extracts 256x256 spatial patches from large-scale VLUCD and Landsat 8 GeoTIFFs
using windowed reads, applying coastal target pixel filtering (>= 5% Mangrove + Aquaculture)
and NoData filtering (<= 50% NoData).
"""

import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from src.data.remapper import (
    CLASS_PALETTE,
    IGNORE_INDEX,
    NUM_CLASSES,
    REMAP_LUT,
    TARGET_CLASSES,
    VLUCDRemapper,
)
from src.data.spatial_split import (
    COASTAL_BLOCKS,
    DisjointGeographicSplitter,
    GeographicBlock,
)


@dataclass
class PatchMetadata:
    """Metadata describing a single extracted spatial patch."""

    patch_id: str
    region: str
    block_id: str
    split: str
    col_offset: int
    row_offset: int
    width: int
    height: int
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    num_mangrove: int
    num_aquaculture: int
    target_pixel_count: int
    target_ratio: float
    nodata_count: int
    nodata_ratio: float
    is_valid: bool
    image_path: Optional[str] = None
    mask_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GeoTIFFWindowReader:
    """Zero-dependency windowed reader for uncompressed classic GeoTIFF rasters.

    Seamlessly reads arbitrary sub-windows without loading the entire raster
    into memory. Falls back to rasterio if available.
    """

    def __init__(self, tiff_path: Union[str, Path]):
        self.path = Path(tiff_path)
        if not self.path.exists():
            raise FileNotFoundError(f"GeoTIFF file not found: {self.path}")

        self.has_rasterio = False
        try:
            import rasterio
            self.has_rasterio = True
        except ImportError:
            pass

        self._parse_tiff_header()

    def _parse_tiff_header(self):
        """Parses basic Classic TIFF tags to extract dimensions, offsets, and georeferencing."""
        with open(self.path, "rb") as f:
            byte_order = f.read(2)
            if byte_order == b"II":
                endian = "<"
            elif byte_order == b"MM":
                endian = ">"
            else:
                raise ValueError(f"Not a valid TIFF file: {self.path}")
            self.endian = endian

            magic = struct.unpack(endian + "H", f.read(2))[0]
            if magic != 42:
                raise ValueError(f"Unsupported TIFF version/magic: {magic}")

            first_ifd_offset = struct.unpack(endian + "I", f.read(4))[0]
            f.seek(first_ifd_offset)

            num_entries = struct.unpack(endian + "H", f.read(2))[0]
            tags = {}
            for _ in range(num_entries):
                tag, tag_type, count, val_or_offset = struct.unpack(endian + "HHI I", f.read(12))
                tags[tag] = (tag_type, count, val_or_offset)

            # Essential tags
            self.width = tags.get(256, (None, None, 0))[2]   # ImageWidth
            self.length = tags.get(257, (None, None, 0))[2]  # ImageLength
            self.bits_per_sample = tags.get(258, (None, None, 8))[2]
            self.compression = tags.get(259, (None, None, 1))[2]
            self.rows_per_strip = tags.get(278, (None, None, 1))[2]

            # Strip offsets
            strip_tag = tags.get(273)
            if strip_tag:
                tag_type, count, val_or_offset = strip_tag
                if count == 1:
                    self.first_strip_offset = val_or_offset
                    self.bytes_per_row = self.width * (self.bits_per_sample // 8)
                else:
                    # Array of offsets
                    f.seek(val_or_offset)
                    self.first_strip_offset = struct.unpack(endian + "I", f.read(4))[0]
                    self.bytes_per_row = self.width * (self.bits_per_sample // 8)
            else:
                self.first_strip_offset = 0
                self.bytes_per_row = self.width

            # Georeferencing
            # ModelPixelScaleTag (33550)
            if 33550 in tags:
                _, _, scale_offset = tags[33550]
                f.seek(scale_offset)
                scale = struct.unpack(endian + "3d", f.read(24))
                self.scale_x, self.scale_y, _ = scale
            else:
                self.scale_x, self.scale_y = 1.0, 1.0

            # ModelTiepointTag (33922)
            if 33922 in tags:
                _, _, tie_offset = tags[33922]
                f.seek(tie_offset)
                tie = struct.unpack(endian + "6d", f.read(48))
                self.origin_x, self.origin_y = tie[3], tie[4]
            else:
                self.origin_x, self.origin_y = 0.0, 0.0

    def pixel_to_geo(self, col: int, row: int) -> Tuple[float, float]:
        """Converts (col, row) pixel indices to (longitude, latitude) coordinates."""
        lon = self.origin_x + col * self.scale_x
        lat = self.origin_y - row * self.scale_y
        return lon, lat

    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """Converts (longitude, latitude) coordinates to (col, row) pixel indices."""
        col = int(round((lon - self.origin_x) / self.scale_x))
        row = int(round((self.origin_y - lat) / self.scale_y))
        return col, row

    def read_window(
        self, col_off: int, row_off: int, width: int = 256, height: int = 256
    ) -> List[List[int]]:
        """Reads a 2D window of pixel values from the GeoTIFF.

        Returns:
            2D list of shape (height, width) with raw uint8 integer pixel values.
        """
        if self.has_rasterio:
            import rasterio
            from rasterio.windows import Window
            with rasterio.open(self.path) as src:
                window = Window(col_off, row_off, width, height)
                arr = src.read(1, window=window)
                return arr.tolist()

        # Direct binary seek implementation
        window = []
        with open(self.path, "rb") as f:
            for r in range(height):
                current_row = row_off + r
                if 0 <= current_row < self.length:
                    row_start_offset = (
                        self.first_strip_offset + current_row * self.bytes_per_row + col_off
                    )
                    f.seek(row_start_offset)
                    row_bytes = f.read(width)
                    window.append(list(row_bytes))
                else:
                    window.append([0] * width)
        return window

    def read_window_bytes(
        self, col_off: int, row_off: int, width: int = 256, height: int = 256
    ) -> bytes:
        """Reads a window as a contiguous bytes object of length height * width."""
        chunks = []
        with open(self.path, "rb") as f:
            for r in range(height):
                current_row = row_off + r
                if 0 <= current_row < self.length:
                    row_start_offset = (
                        self.first_strip_offset + current_row * self.bytes_per_row + col_off
                    )
                    f.seek(row_start_offset)
                    chunks.append(f.read(width))
                else:
                    chunks.append(b"\x00" * width)
        return b"".join(chunks)


class PatchExtractor:
    """Automated patch extraction pipeline with coastal filtering and spatial splitting."""

    def __init__(
        self,
        vlucd_tiff_path: Union[str, Path],
        patch_size: int = 256,
        target_min_ratio: float = 0.05,  # >= 5% coastal pixels (3,277 / 65,536)
        nodata_max_ratio: float = 0.50,  # <= 50% NoData (32,768 / 65,536)
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.vlucd_reader = GeoTIFFWindowReader(vlucd_tiff_path)
        self.patch_size = patch_size
        self.target_min_ratio = target_min_ratio
        self.nodata_max_ratio = nodata_max_ratio
        self.remapper = VLUCDRemapper()
        self.splitter = DisjointGeographicSplitter(COASTAL_BLOCKS)
        self.output_dir = Path(output_dir) if output_dir else Path("data/patches")
        self.total_pixels = patch_size * patch_size
        self.min_target_pixels = math.ceil(self.total_pixels * self.target_min_ratio)
        self.max_nodata_pixels = math.floor(self.total_pixels * self.nodata_max_ratio)

    def extract_patch(
        self, col_off: int, row_off: int
    ) -> Tuple[List[List[int]], PatchMetadata]:
        """Extracts and filters a single 256x256 patch at the given coordinate.

        Returns:
            Tuple of (remapped_mask: List[List[int]], metadata: PatchMetadata)
        """
        raw_window = self.vlucd_reader.read_window(
            col_off, row_off, self.patch_size, self.patch_size
        )
        remapped_mask = self.remapper.remap(raw_window)

        # Geographic bounding box
        lon_min, lat_max = self.vlucd_reader.pixel_to_geo(col_off, row_off)
        lon_max, lat_min = self.vlucd_reader.pixel_to_geo(
            col_off + self.patch_size, row_off + self.patch_size
        )

        # Count classes
        num_mangrove = 0
        num_aquaculture = 0
        nodata_count = 0

        for row in remapped_mask:
            for val in row:
                if val == 0:
                    num_mangrove += 1
                elif val == 1:
                    num_aquaculture += 1
                elif val == IGNORE_INDEX:
                    nodata_count += 1

        target_count = num_mangrove + num_aquaculture
        target_ratio = target_count / self.total_pixels
        nodata_ratio = nodata_count / self.total_pixels

        is_valid = (
            target_count >= self.min_target_pixels
            and nodata_count <= self.max_nodata_pixels
        )

        # Spatial block and split assignment
        center_lon = (lon_min + lon_max) / 2.0
        center_lat = (lat_min + lat_max) / 2.0
        assignment = self.splitter.assign_point(center_lon, center_lat)

        if assignment:
            split_name, block_id, region = assignment
        else:
            split_name, block_id, region = "unassigned", "NONE", "Unknown"
            is_valid = False  # Reject patches outside defined geographic blocks

        patch_id = f"VN_{region[:3].upper()}_{block_id}_r{row_off}_c{col_off}"

        metadata = PatchMetadata(
            patch_id=patch_id,
            region=region,
            block_id=block_id,
            split=split_name,
            col_offset=col_off,
            row_offset=row_off,
            width=self.patch_size,
            height=self.patch_size,
            lon_min=lon_min,
            lat_min=lat_min,
            lon_max=lon_max,
            lat_max=lat_max,
            num_mangrove=num_mangrove,
            num_aquaculture=num_aquaculture,
            target_pixel_count=target_count,
            target_ratio=target_ratio,
            nodata_count=nodata_count,
            nodata_ratio=nodata_ratio,
            is_valid=is_valid,
        )

        return remapped_mask, metadata

    def generate_synthetic_7band_patch(
        self, mask: List[List[int]]
    ) -> List[List[List[float]]]:
        """Generates realistic 7-band surface reflectance [0.0, 1.0] conditioned on land cover.

        Used for offline testing and verification when Landsat 8 scenes are being preprocessed.
        Spectral profiles:
            Bands: [SR_B1 (Aerosol), SR_B2 (Blue), SR_B3 (Green), SR_B4 (Red),
                    SR_B5 (NIR), SR_B6 (SWIR1), SR_B7 (SWIR2)]
        """
        # Physical mean reflectance signatures per target class
        class_spectra = {
            0: [0.03, 0.04, 0.07, 0.04, 0.35, 0.14, 0.06],  # Mangrove (High NIR, low Red, low SWIR)
            1: [0.08, 0.09, 0.08, 0.07, 0.03, 0.01, 0.01],  # Aquaculture (Turbid water, low NIR)
            2: [0.06, 0.05, 0.04, 0.03, 0.01, 0.005, 0.005], # Open Water (Clear/Estuary)
            3: [0.04, 0.05, 0.08, 0.06, 0.28, 0.16, 0.09],  # Cropland
            4: [0.03, 0.04, 0.06, 0.03, 0.40, 0.18, 0.08],  # Forest
            5: [0.12, 0.14, 0.16, 0.18, 0.22, 0.25, 0.22],  # Built-up / Bareland
            IGNORE_INDEX: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Background/NoData
        }

        h, w = len(mask), len(mask[0])
        # 7 channels of (h, w)
        bands = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(7)]

        for r in range(h):
            for c in range(w):
                cls = mask[r][c]
                spectrum = class_spectra.get(cls, class_spectra[IGNORE_INDEX])
                for b in range(7):
                    bands[b][r][c] = spectrum[b]

        return bands

    def extract_block_patches(
        self, block: GeographicBlock, stride: int = 256, max_patches: Optional[int] = None
    ) -> List[Tuple[List[List[int]], PatchMetadata]]:
        """Extracts valid patches across the bounding box of a GeographicBlock."""
        col_min, row_min = self.vlucd_reader.geo_to_pixel(block.lon_min, block.lat_max)
        col_max, row_max = self.vlucd_reader.geo_to_pixel(block.lon_max, block.lat_min)

        col_start = max(0, col_min)
        row_start = max(0, row_min)
        col_end = min(self.vlucd_reader.width - self.patch_size, col_max)
        row_end = min(self.vlucd_reader.length - self.patch_size, row_max)

        valid_patches = []
        for r in range(row_start, row_end, stride):
            for c in range(col_start, col_end, stride):
                mask, meta = self.extract_patch(c, r)
                if meta.is_valid:
                    valid_patches.append((mask, meta))
                    if max_patches and len(valid_patches) >= max_patches:
                        return valid_patches
        return valid_patches
