"""VLUCD Class Remapper for VN-CoastSeg-Mask2Former.

Standardizes raw VLUCD Level-2 30m discrete land cover classes into the
6 coastal monitoring target classes + ignore_index=255.

Raw VLUCD Level-2 classes:
    0: Background / Outside Vietnam -> 255 (NoData / ignore_index)
    1: Residence 1 (High Developed Urban) -> 5 (Built-up, Urban & Bareland)
    2: Residence 2 (Low Developed Rural)  -> 5 (Built-up, Urban & Bareland)
    3: Rice Paddies                      -> 3 (Cropland & Paddy Rice)
    4: Woody Crops                       -> 3 (Cropland & Paddy Rice)
    5: Other Crops                       -> 3 (Cropland & Paddy Rice)
    6: In-house Crops                    -> 3 (Cropland & Paddy Rice)
    7: Grassland / Herbaceous            -> 4 (Terrestrial Forest & Shrubland)
    8: Barren Land                       -> 5 (Built-up, Urban & Bareland)
    9: Scrubland / Shrub                 -> 4 (Terrestrial Forest & Shrubland)
   10: Deciduous Broadleaf Forest        -> 4 (Terrestrial Forest & Shrubland)
   11: Evergreen Broadleaf Forest        -> 4 (Terrestrial Forest & Shrubland)
   12: Evergreen Needleleaf Forest       -> 4 (Terrestrial Forest & Shrubland)
   13: Unassigned                        -> 255 (NoData / ignore_index)
   14: Plantation Land                   -> 4 (Terrestrial Forest & Shrubland)
   15: Mangrove Forest                   -> 0 (Mangrove & Coastal Wetland)
   16: Inland / Coastal Wetland          -> 0 (Mangrove & Coastal Wetland)
   17: Unassigned                        -> 255 (NoData / ignore_index)
   18: Open Water                        -> 2 (Open Water)
   19: Aquaculture (Fish/Shrimp Ponds)   -> 1 (Aquaculture)
   20: Bamboo Area                       -> 4 (Terrestrial Forest & Shrubland)
  255: NoData / Ignored                  -> 255 (NoData / ignore_index)

Target Classes:
    0: Mangrove & Coastal Wetland
    1: Aquaculture
    2: Open Water
    3: Cropland & Paddy Rice
    4: Terrestrial Forest & Shrubland
    5: Built-up, Urban & Bareland
  255: Background / NoData (ignore_index for CrossEntropy / Set Prediction Loss)
"""

from typing import Any, Dict, List, Tuple, Union

# Target Class Metadata
TARGET_CLASSES: Dict[int, str] = {
    0: "Mangrove & Coastal Wetland",
    1: "Aquaculture",
    2: "Open Water",
    3: "Cropland & Paddy Rice",
    4: "Terrestrial Forest & Shrubland",
    5: "Built-up, Urban & Bareland",
    255: "Background / NoData",
}

CLASS_NAMES: List[str] = [
    "Mangrove & Coastal Wetland",
    "Aquaculture",
    "Open Water",
    "Cropland & Paddy Rice",
    "Terrestrial Forest & Shrubland",
    "Built-up, Urban & Bareland",
]

NUM_CLASSES = 6
IGNORE_INDEX = 255

# Color Palette for Visualization and GIS exports (RGB uint8)
CLASS_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (44, 160, 44),     # Mangrove: Forest Green (#2ca02c)
    1: (31, 119, 180),    # Aquaculture: Steel Blue (#1f77b4)
    2: (0, 0, 128),       # Open Water: Navy Blue (#000080)
    3: (227, 119, 194),   # Cropland: Pink/Magenta (#e377c2)
    4: (188, 189, 34),    # Forest: Olive Green (#bcbd22)
    5: (214, 39, 40),     # Built-up: Brick Red (#d62728)
    255: (255, 255, 255), # Background/NoData: White (#ffffff)
}

# Hex Color Codes
CLASS_HEX_COLORS: Dict[int, str] = {
    0: "#2ca02c",
    1: "#1f77b4",
    2: "#000080",
    3: "#e377c2",
    4: "#bcbd22",
    5: "#d62728",
    255: "#ffffff",
}

# 18 Raw VLUCD Classes mapped to 6 Target Classes
RAW_VLUCD_TO_TARGET: Dict[int, int] = {
    0: IGNORE_INDEX,   # Background / Outside Vietnam
    1: 5,              # Residence 1 (High Developed)
    2: 5,              # Residence 2 (Low Developed)
    3: 3,              # Rice Paddies
    4: 3,              # Woody Crops
    5: 3,              # Other Crops
    6: 3,              # In-house Crops
    7: 4,              # Grassland
    8: 5,              # Barren Land
    9: 4,              # Scrubland
    10: 4,             # Deciduous Broadleaf Forest
    11: 4,             # Evergreen Broadleaf Forest
    12: 4,             # Evergreen Needleleaf Forest
    13: IGNORE_INDEX,  # Unassigned
    14: 4,             # Plantation Land
    15: 0,             # Mangrove Forest
    16: 0,             # Inland / Coastal Wetland
    17: IGNORE_INDEX,  # Unassigned
    18: 2,             # Open Water
    19: 1,             # Aquaculture (Fish/Shrimp Ponds)
    20: 4,             # Bamboo Area
    255: IGNORE_INDEX, # NoData
}

# 256-element Lookup Table (LUT) for O(1) indexed remapping
REMAP_LUT: List[int] = [IGNORE_INDEX] * 256
for raw_val, target_val in RAW_VLUCD_TO_TARGET.items():
    if 0 <= raw_val < 256:
        REMAP_LUT[raw_val] = target_val


class VLUCDRemapper:
    """High-performance remapper converting VLUCD 30m labels to 6 target classes.

    Supports:
        - NumPy arrays (2D, 3D batch)
        - PyTorch tensors (2D, 3D, 4D batch)
        - Python nested lists / tuples
        - Raw byte buffers (from GeoTIFF windowed reader)
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        self.ignore_index = ignore_index
        self.lut = list(REMAP_LUT)
        if ignore_index != IGNORE_INDEX:
            for i in range(256):
                if self.lut[i] == IGNORE_INDEX:
                    self.lut[i] = ignore_index

        self._np_lut = None
        self._torch_lut = None

    def _get_np_lut(self):
        if self._np_lut is None:
            try:
                import numpy as np
                self._np_lut = np.array(self.lut, dtype=np.int64)
            except ImportError:
                self._np_lut = None
        return self._np_lut

    def _get_torch_lut(self, device=None):
        try:
            import torch
            if self._torch_lut is None or self._torch_lut.device != device:
                self._torch_lut = torch.tensor(self.lut, dtype=torch.long, device=device)
            return self._torch_lut
        except ImportError:
            return None

    def remap(self, mask: Any) -> Any:
        """Remap raw VLUCD mask values to target classes [0..5, ignore_index].

        Args:
            mask: Input mask as numpy array, torch tensor, bytearray/bytes,
                  or nested list.

        Returns:
            Remapped mask preserving the input container type.
        """
        # 1. PyTorch Tensor support
        if hasattr(mask, "__class__") and "Tensor" in mask.__class__.__name__:
            import torch
            lut = self._get_torch_lut(device=mask.device)
            if lut is not None:
                # Clamp or modulo values to [0, 255]
                clamped = mask.to(torch.long)
                clamped = torch.clamp(clamped, 0, 255)
                return lut[clamped]

        # 2. NumPy ndarray support
        if hasattr(mask, "__class__") and "ndarray" in mask.__class__.__name__:
            import numpy as np
            np_lut = self._get_np_lut()
            if np_lut is not None:
                clamped = np.clip(mask.astype(np.int64), 0, 255)
                return np_lut[clamped]

        # 3. Byte / Bytearray buffer (from windowed file reader)
        if isinstance(mask, (bytes, bytearray)):
            lut = self.lut
            return bytes(lut[b] if b < 256 else self.ignore_index for b in mask)

        # 4. Pure Python nested list / int fallback
        if isinstance(mask, list):
            return self._remap_nested_list(mask)

        if isinstance(mask, int):
            return self.lut[mask] if 0 <= mask < 256 else self.ignore_index

        raise TypeError(f"Unsupported mask type for remapping: {type(mask)}")

    def _remap_nested_list(self, lst: list) -> list:
        lut = self.lut
        if not lst:
            return []
        if isinstance(lst[0], list):
            return [self._remap_nested_list(sub) for sub in lst]
        return [lut[val] if 0 <= val < 256 else self.ignore_index for val in lst]

    def to_gis_classes(self, mask: Any) -> Any:
        """Convert target class indices (0..5, 255) to GIS 1-based indices (1..6, 0).

        GIS Schema:
            0: Background / NoData
            1: Mangrove & Coastal Wetland
            2: Aquaculture
            3: Open Water
            4: Cropland & Paddy Rice
            5: Terrestrial Forest & Shrubland
            6: Built-up, Urban & Bareland
        """
        gis_lut = [0] * 256
        for c in range(6):
            gis_lut[c] = c + 1
        gis_lut[self.ignore_index] = 0

        if hasattr(mask, "__class__") and "Tensor" in mask.__class__.__name__:
            import torch
            tensor_lut = torch.tensor(gis_lut, dtype=torch.long, device=mask.device)
            return tensor_lut[torch.clamp(mask.to(torch.long), 0, 255)]

        if hasattr(mask, "__class__") and "ndarray" in mask.__class__.__name__:
            import numpy as np
            np_lut = np.array(gis_lut, dtype=np.int64)
            return np_lut[np.clip(mask.astype(np.int64), 0, 255)]

        if isinstance(mask, list):
            def _map_gis(lst):
                if not lst:
                    return []
                if isinstance(lst[0], list):
                    return [_map_gis(s) for s in lst]
                return [gis_lut[v] if 0 <= v < 256 else 0 for v in lst]
            return _map_gis(mask)

        if isinstance(mask, int):
            return gis_lut[mask] if 0 <= mask < 256 else 0

        raise TypeError(f"Unsupported mask type: {type(mask)}")


# Singleton module-level function for direct use
_DEFAULT_REMAPPER = VLUCDRemapper()


def remap_vlucd(mask: Any) -> Any:
    """Remap a VLUCD Level-2 mask into 6 target classes + ignore_index=255."""
    return _DEFAULT_REMAPPER.remap(mask)
