"""Data processing, remapping, patching, splitting, and dataset package."""

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
    CLASS_HEX_COLORS,
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
from src.data.spatial_split import (
    COASTAL_BLOCKS,
    DisjointGeographicSplitter,
    GeographicBlock,
)

__all__ = [
    "TARGET_CLASSES",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "IGNORE_INDEX",
    "CLASS_PALETTE",
    "CLASS_HEX_COLORS",
    "RAW_VLUCD_TO_TARGET",
    "REMAP_LUT",
    "VLUCDRemapper",
    "remap_vlucd",
    "PatchMetadata",
    "GeoTIFFWindowReader",
    "PatchExtractor",
    "GeographicBlock",
    "COASTAL_BLOCKS",
    "DisjointGeographicSplitter",
    "SSL4EO_L_LANDSAT8_MEAN",
    "SSL4EO_L_LANDSAT8_STD",
    "CoastalSegmentationDataset",
    "SimpleDataLoader",
    "create_dataloader",
]
