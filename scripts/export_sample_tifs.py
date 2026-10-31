#!/usr/bin/env python3
"""Exports authentic 7-band Landsat 8 Surface Reflectance GeoTIFF sample patches.

Prioritizes authentic USGS EarthExplorer Level-2 Landsat 8 scenes in sample_data/,
falling back to organic coastal simulation if raw scenes are unavailable.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.extract_real_landsat_samples import extract_real_patches


def main():
    sample_data_dir = Path("sample_data")
    if sample_data_dir.exists() and list(sample_data_dir.glob("*SR_B1*")):
        print("[*] Found authentic Landsat 8 Level-2 scene in sample_data/. Extracting authentic patches...")
        extract_real_patches(scene_dir="sample_data", output_dir="data/samples", patch_size=256)
    else:
        print("[!] sample_data/ not found or missing SR bands. Exporting organic coastal synthetic patches...")
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from src.app import generate_sample_coastal_patch

        out_path = Path("data/samples")
        out_path.mkdir(parents=True, exist_ok=True)
        presets = [
            ("Can Gio Mangrove Biosphere Reserve", "can_gio_landsat8_7band.tif", 106.85, 10.35),
            ("Ca Mau Peninsula & Southwest Coast", "ca_mau_landsat8_7band.tif", 104.95, 8.65),
            ("Red River Delta (Xuan Thuy National Park)", "red_river_delta_landsat8_7band.tif", 106.50, 20.25),
        ]
        res_deg = 0.00027
        for preset_name, filename, lon, lat in presets:
            bands_list, _, meta = generate_sample_coastal_patch(preset_name, 256)
            bands_arr = np.array(bands_list, dtype=np.float32)
            transform = from_origin(lon, lat + 256 * res_deg, res_deg, res_deg)
            dst_path = out_path / filename
            with rasterio.open(
                str(dst_path),
                "w",
                driver="GTiff",
                height=256,
                width=256,
                count=7,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(bands_arr)
            print(f"[OK] Exported {filename}")


if __name__ == "__main__":
    main()
