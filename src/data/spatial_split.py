"""Disjoint Geographic Block Splitting for VN-CoastSeg-Mask2Former.

Implements Stratified Disjoint Geographic Block Splitting (70% Train, 15% Val, 15% Test)
across Northern (Red River Delta), Southern (Can Gio), and Mekong Delta (Ca Mau) coastal
regions.

Prevents spatial autocorrelation data leakage by assigning entire, geographically
isolated bounding boxes with >= 1 km (0.05 deg) spatial separation buffers to
distinct splits.
"""

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass
class GeographicBlock:
    """Represents a discrete geographic block for spatial splitting."""

    block_id: str
    region: str
    location_name: str
    split: str  # 'train', 'val', 'test'
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    description: str

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Returns (lon_min, lat_min, lon_max, lat_max)."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def contains_point(self, lon: float, lat: float) -> bool:
        """Checks if a geographic point lies inside this block."""
        return self.lon_min <= lon <= self.lon_max and self.lat_min <= lat <= self.lat_max

    def intersects(self, other: "GeographicBlock") -> bool:
        """Checks if bounding boxes overlap."""
        overlap_lon = max(self.lon_min, other.lon_min) < min(self.lon_max, other.lon_max)
        overlap_lat = max(self.lat_min, other.lat_min) < min(self.lat_max, other.lat_max)
        return overlap_lon and overlap_lat

    def distance_to_block_km(self, other: "GeographicBlock") -> float:
        """Calculates approximate minimum distance between two bounding boxes in km."""
        if self.intersects(other):
            return 0.0

        # Calculate dx in lon degrees, dy in lat degrees
        if self.lon_max < other.lon_min:
            d_lon = other.lon_min - self.lon_max
        elif other.lon_max < self.lon_min:
            d_lon = self.lon_min - other.lon_max
        else:
            d_lon = 0.0

        if self.lat_max < other.lat_min:
            d_lat = other.lat_min - self.lat_max
        elif other.lat_max < self.lat_min:
            d_lat = self.lat_min - other.lat_max
        else:
            d_lat = 0.0

        # Average latitude for longitude conversion (1 deg lat ~ 111 km, 1 deg lon ~ 111 * cos(lat))
        mid_lat = math.radians((self.lat_min + self.lat_max + other.lat_min + other.lat_max) / 4.0)
        km_x = d_lon * 111.32 * math.cos(mid_lat)
        km_y = d_lat * 110.57
        return math.sqrt(km_x * km_x + km_y * km_y)


# 12 Disjoint Coastal Geographic Blocks with verified spatial separation buffers
COASTAL_BLOCKS: List[GeographicBlock] = [
    # --- Northern Coast (Red River Delta & Quang Ninh) ---
    GeographicBlock(
        block_id="N1",
        region="Northern",
        location_name="Hai Phong (Tien Lang, Do Son, Kien Thuy)",
        split="train",
        lon_min=106.60,
        lat_min=20.65,
        lon_max=106.90,
        lat_max=20.95,
        description="Estuarine mudflats, river mouth aquaculture, Kandelia obovata fringes.",
    ),
    GeographicBlock(
        block_id="N2",
        region="Northern",
        location_name="Thai Binh (Thai Thuy, Tien Hai)",
        split="train",
        lon_min=106.40,
        lat_min=20.35,
        lon_max=106.70,
        lat_max=20.60,
        description="Tra Ly & Diem Dien estuaries, extensive coastal shrimp farms.",
    ),
    GeographicBlock(
        block_id="N3",
        region="Northern",
        location_name="Nam Dinh (Xuan Thuy National Park, Giao Thuy)",
        split="val",
        lon_min=106.35,
        lat_min=20.05,
        lon_max=106.65,
        lat_max=20.30,
        description="Ba Lat river mouth, Ramsar mangrove reserve, transitional ponds.",
    ),
    GeographicBlock(
        block_id="N4",
        region="Northern",
        location_name="Quang Ninh (Cat Ba Island & Lan Ha / Ha Long Bay)",
        split="test",
        lon_min=106.95,
        lat_min=20.70,
        lon_max=107.25,
        lat_max=20.95,
        description="Karst island mangroves, marine cage aquaculture, clear coastal waters.",
    ),
    # --- Southern Coast (Can Gio Biosphere Reserve & Estuaries) ---
    GeographicBlock(
        block_id="S1",
        region="Southern",
        location_name="Can Gio Central (Can Thanh, Long Hoa)",
        split="train",
        lon_min=106.75,
        lat_min=10.35,
        lon_max=107.00,
        lat_max=10.55,
        description="Core UNESCO Biosphere Reserve, dense Rhizophora canopies.",
    ),
    GeographicBlock(
        block_id="S2",
        region="Southern",
        location_name="Can Gio West (An Thoi Dong, Soai Rap River)",
        split="train",
        lon_min=106.50,
        lat_min=10.35,
        lon_max=106.70,
        lat_max=10.60,
        description="Soai Rap estuarine fringe, mangrove-aquaculture interface.",
    ),
    GeographicBlock(
        block_id="S3",
        region="Southern",
        location_name="Can Gio North (Binh Khanh, Nhon Trach - Dong Nai)",
        split="val",
        lon_min=106.75,
        lat_min=10.60,
        lon_max=107.05,
        lat_max=10.80,
        description="Upper estuary transition, agricultural & brackish aquaculture ponds.",
    ),
    GeographicBlock(
        block_id="S4",
        region="Southern",
        location_name="Can Gio East & Thi Vai / Vung Tau Coast",
        split="test",
        lon_min=107.10,
        lat_min=10.35,
        lon_max=107.35,
        lat_max=10.60,
        description="Long Tau mouth, Thi Vai river shipping channel, coastal mangrove edge.",
    ),
    # --- Mekong Delta Coast (Ca Mau, Bac Lieu, Soc Trang) ---
    GeographicBlock(
        block_id="M1",
        region="Mekong Delta",
        location_name="Ca Mau Central (Nam Can, Dam Doi, Cai Nuoc)",
        split="train",
        lon_min=104.85,
        lat_min=8.80,
        lon_max=105.20,
        lat_max=9.10,
        description="Vietnam's largest intensive & extensive shrimp pond network.",
    ),
    GeographicBlock(
        block_id="M2",
        region="Mekong Delta",
        location_name="Bac Lieu Coast (Dong Hai, Hoa Binh, Bac Lieu city)",
        split="train",
        lon_min=105.25,
        lat_min=9.15,
        lon_max=105.65,
        lat_max=9.45,
        description="Expansive industrial aquaculture, high-density pond complexes.",
    ),
    GeographicBlock(
        block_id="M3",
        region="Mekong Delta",
        location_name="Soc Trang Coast (Vinh Chau, Cu Lao Dung, Hau River)",
        split="val",
        lon_min=105.75,
        lat_min=9.30,
        lon_max=106.20,
        lat_max=9.75,
        description="Mekong River (Bassac) distributaries, alluvial mangrove pioneer zones.",
    ),
    GeographicBlock(
        block_id="M4",
        region="Mekong Delta",
        location_name="Cape Ca Mau / Dat Mui National Park & West Coast",
        split="test",
        lon_min=104.55,
        lat_min=8.50,
        lon_max=104.80,
        lat_max=8.75,
        description="Dat Mui accretion zone, protected virgin mangrove swamp, western sea.",
    ),
]


class DisjointGeographicSplitter:
    """Manages spatial splitting across geographic blocks, ensuring zero data leakage."""

    def __init__(self, blocks: Optional[List[GeographicBlock]] = None):
        self.blocks = blocks or list(COASTAL_BLOCKS)
        self.block_dict: Dict[str, GeographicBlock] = {b.block_id: b for b in self.blocks}
        self._validate_blocks()

    def _validate_blocks(self):
        """Validates that blocks within different splits do not overlap."""
        is_disjoint, violations = self.verify_disjointness()
        if not is_disjoint:
            raise ValueError(f"Block configuration contains spatial overlaps: {violations}")

    def verify_disjointness(self) -> Tuple[bool, List[str]]:
        """Verifies pairwise disjointness between blocks assigned to different splits.

        Returns:
            Tuple of (is_disjoint: bool, list_of_violations: List[str]).
        """
        violations = []
        n = len(self.blocks)
        for i in range(n):
            for j in range(i + 1, n):
                b1 = self.blocks[i]
                b2 = self.blocks[j]
                if b1.intersects(b2):
                    violations.append(
                        f"Overlap detected between {b1.block_id} ({b1.split}) and {b2.block_id} ({b2.split})"
                    )
        return len(violations) == 0, violations

    def get_min_buffer_margin_km(self, split_a: str, split_b: str) -> float:
        """Finds minimum distance in kilometers between blocks of two different splits."""
        min_dist = float("inf")
        blocks_a = [b for b in self.blocks if b.split == split_a]
        blocks_b = [b for b in self.blocks if b.split == split_b]

        for b_a in blocks_a:
            for b_b in blocks_b:
                dist = b_a.distance_to_block_km(b_b)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def assign_point(self, lon: float, lat: float) -> Optional[Tuple[str, str, str]]:
        """Determines which block, region, and split a coordinate point belongs to.

        Returns:
            Tuple of (split, block_id, region) or None if point is outside all blocks.
        """
        for block in self.blocks:
            if block.contains_point(lon, lat):
                return (block.split, block.block_id, block.region)
        return None

    def assign_patch_bbox(
        self, lon_min: float, lat_min: float, lon_max: float, lat_max: float
    ) -> Optional[Tuple[str, str, str]]:
        """Assigns a patch to a split based on its centroid."""
        center_lon = (lon_min + lon_max) / 2.0
        center_lat = (lat_min + lat_max) / 2.0
        return self.assign_point(center_lon, center_lat)

    def split_patches(self, patches: List[Dict]) -> Dict[str, List[Dict]]:
        """Partitions patch records into train, val, and test manifests.

        Patches outside all defined coastal blocks are omitted to prevent leakage.
        """
        splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

        for patch in patches:
            lon = (patch.get("lon_min", 0.0) + patch.get("lon_max", 0.0)) / 2.0
            lat = (patch.get("lat_min", 0.0) + patch.get("lat_max", 0.0)) / 2.0
            assignment = self.assign_point(lon, lat)
            if assignment is not None:
                split_name, block_id, region = assignment
                patch_copy = dict(patch)
                patch_copy["split"] = split_name
                patch_copy["block_id"] = block_id
                patch_copy["region"] = region
                splits[split_name].append(patch_copy)

        return splits

    def export_manifests(
        self, split_dict: Dict[str, List[Dict]], output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """Exports train.csv, val.csv, and test.csv manifests."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_paths = {}

        for split_name in ("train", "val", "test"):
            items = split_dict.get(split_name, [])
            csv_path = out_dir / f"{split_name}.csv"
            if items:
                fieldnames = list(items[0].keys())
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(items)
            else:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    f.write("patch_id,split,block_id,region,image_path,mask_path\n")
            manifest_paths[split_name] = csv_path

        # Export split summary JSON
        summary = {
            "num_train": len(split_dict.get("train", [])),
            "num_val": len(split_dict.get("val", [])),
            "num_test": len(split_dict.get("test", [])),
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return manifest_paths
