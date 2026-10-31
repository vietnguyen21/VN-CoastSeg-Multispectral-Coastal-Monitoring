#!/usr/bin/env python3
"""Utility script to prepare and stack Landsat 8 Level-2 GeoTIFF scenes from USGS EarthExplorer.

When downloading Landsat 8 Collection 2 Level-2 from USGS EarthExplorer, you receive
individual single-band GeoTIFF files:
  - *_SR_B1.TIF (Coastal Aerosol, 0.43 - 0.45 um)
  - *_SR_B2.TIF (Blue, 0.45 - 0.51 um)
  - *_SR_B3.TIF (Green, 0.53 - 0.59 um)
  - *_SR_B4.TIF (Red, 0.64 - 0.67 um)
  - *_SR_B5.TIF (NIR, 0.85 - 0.88 um) - Critical for Mangrove canopy
  - *_SR_B6.TIF (SWIR-1, 1.57 - 1.65 um) - Critical for moisture & aquaculture ponds
  - *_SR_B7.TIF (SWIR-2, 2.11 - 2.29 um)

Note: T_ST_EMIS.TIF or ST_B10.TIF are single-band thermal products, not optical reflectance.

This script stacks the 7 optical bands (SR_B1..SR_B7) into a unified GeoTIFF, scales the DN
values to surface reflectance [0.0, 1.0], and extracts a 256x256 patch for real-time inference.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None


def find_band_file(directory: Path, band_suffix: str) -> Path:
    """Finds band file matching suffix case-insensitively or exact name."""
    patterns = [
        f"*{band_suffix}.TIF",
        f"*{band_suffix}.tif",
        f"*{band_suffix}.TIFF",
        f"{band_suffix}.TIF",
        f"{band_suffix}.tif",
        f"{band_suffix}.TIFF",
        f"*{band_suffix.lower()}*.tif",
        f"*{band_suffix.upper()}*.tif",
    ]
    for pat in patterns:
        matches = list(directory.glob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find band file matching '{band_suffix}' in {directory}")


def stack_and_crop_landsat_scene(
    scene_dir_or_file: str,
    output_path: str = "data/samples/custom_landsat8_7band.tif",
    crop_size: int = 256,
    center_lon_lat: tuple = None,
):
    """Stacks 7 Landsat 8 SR bands and extracts a 256x256 patch."""
    if rasterio is None:
        print("[ERROR] rasterio is required for GeoTIFF stacking. Run inside venv.")
        sys.exit(1)

    # Resolve input path: if user provided e.g. /sample_data, check relative to current dir
    path = Path(scene_dir_or_file)
    if not path.exists() and scene_dir_or_file.startswith("/"):
        relative_candidate = Path(scene_dir_or_file.lstrip("/"))
        if relative_candidate.exists():
            print(f"[*] Note: '{scene_dir_or_file}' resolved to local workspace directory: '{relative_candidate}'")
            path = relative_candidate

    out_file = Path(output_path)
    if not out_file.parent.exists() and output_path.startswith("/"):
        relative_out = Path(output_path.lstrip("/"))
        if relative_out.parent.exists() or Path(".").resolve() in relative_out.resolve().parents:
            out_file = relative_out

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fallback to local data/samples directory if root / is not writable
        fallback_out = Path("data/samples") / out_file.name
        print(f"[!] Warning: Cannot write to '{out_file.parent}'. Saving to '{fallback_out}' instead.")
        fallback_out.parent.mkdir(parents=True, exist_ok=True)
        out_file = fallback_out

    # Check if a single multi-band file was provided
    if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
        print(f"[*] Processing single GeoTIFF file: {path.name}")
        with rasterio.open(str(path)) as src:
            w, h = src.width, src.height
            count = src.count
            print(f"    Dimensions: {w}x{h} px | Bands: {count}")

            # Calculate crop window
            col_off = max(0, (w - crop_size) // 2)
            row_off = max(0, (h - crop_size) // 2)
            win_w = min(crop_size, w)
            win_h = min(crop_size, h)
            window = Window(col_off, row_off, win_w, win_h)
            arr = src.read(window=window).astype(np.float32)
            transform = rasterio.windows.transform(window, src.transform)

            # Replicate to 7 bands if needed
            if count < 7:
                print(f"    [!] Note: File has {count} bands. Replicating to 7 channels.")
                arr = np.tile(arr, (7 // count + 1, 1, 1))[:7]
            else:
                arr = arr[:7]

            # Scale if DN
            if np.nanmax(arr) > 255.0:
                arr = np.clip(arr * 0.0000275 - 0.2, 0.0, 1.0)
            elif np.nanmax(arr) > 1.0:
                arr = np.clip(arr / 255.0, 0.0, 1.0)

            with rasterio.open(
                str(out_file),
                "w",
                driver="GTiff",
                height=win_h,
                width=win_w,
                count=7,
                dtype=np.float32,
                crs=src.crs,
                transform=transform,
            ) as dst:
                dst.write(arr)

        print(f"[OK] Saved cropped 7-band GeoTIFF ({win_w}x{win_h}) -> {out_file}")
        return

    # Check if a directory of individual band files was provided
    scene_dir = path if path.is_dir() else path.parent
    band_suffixes = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

    print(f"[*] Searching for 7 optical Landsat 8 SR bands in: {scene_dir}")
    band_paths = []
    for sfx in band_suffixes:
        try:
            b_path = find_band_file(scene_dir, sfx)
            band_paths.append(b_path)
            print(f"    Found {sfx}: {b_path.name}")
        except FileNotFoundError as e:
            print(f"    [!] Missing {sfx}: {e}")

    if len(band_paths) < 7:
        print("\n[!] EarthExplorer note: Landsat 8 Level-2 provides individual bands:")
        print("    *_SR_B1.TIF to *_SR_B7.TIF (Surface Reflectance).")
        print("    If you only have 1 band (e.g. ST_EMIS), it is a thermal layer.")
        sys.exit(1)

    # Read reference dimensions from Band 1
    with rasterio.open(str(band_paths[0])) as ref:
        w, h = ref.width, ref.height
        crs = ref.crs
        col_off = max(0, (w - crop_size) // 2)
        row_off = max(0, (h - crop_size) // 2)
        window = Window(col_off, row_off, crop_size, crop_size)
        transform = rasterio.windows.transform(window, ref.transform)

    # Stack all 7 bands
    stacked = np.zeros((7, crop_size, crop_size), dtype=np.float32)
    for i, b_path in enumerate(band_paths):
        with rasterio.open(str(b_path)) as b_src:
            band_data = b_src.read(1, window=window).astype(np.float32)
            # Apply Landsat 8 Collection 2 Level-2 SR scale factor
            scaled_data = band_data * 0.0000275 - 0.2
            stacked[i] = np.clip(scaled_data, 0.0, 1.0)

    # Write output 7-band GeoTIFF
    with rasterio.open(
        str(out_file),
        "w",
        driver="GTiff",
        height=crop_size,
        width=crop_size,
        count=7,
        dtype=np.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(stacked)
        band_names = ["SR_B1 (Coastal)", "SR_B2 (Blue)", "SR_B3 (Green)", "SR_B4 (Red)", "SR_B5 (NIR)", "SR_B6 (SWIR-1)", "SR_B7 (SWIR-2)"]
        for idx, name in enumerate(band_names, start=1):
            dst.set_band_description(idx, name)

    print(f"\n[SUCCESS] Stacked 7 bands and saved {crop_size}x{crop_size} patch to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stack Landsat 8 Level-2 bands from EarthExplorer into a 7-band GeoTIFF"
    )
    # Support positional argument or --scene-dir / --crop-single / -i / --input
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Directory containing Landsat 8 SR_B1..B7 files, or a single GeoTIFF file",
    )
    parser.add_argument(
        "--scene-dir",
        "--input",
        "-i",
        dest="input_opt",
        default=None,
        help="Directory containing Landsat 8 SR_B1..B7 files, or a single GeoTIFF file",
    )
    parser.add_argument(
        "--crop-single",
        dest="crop_single",
        default=None,
        help="Path to a single GeoTIFF file to crop and pad to 7 bands",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/samples/custom_landsat8_7band.tif",
        help="Output GeoTIFF path",
    )
    parser.add_argument(
        "-s",
        "--size",
        "--crop-size",
        dest="size",
        type=int,
        default=256,
        help="Crop patch size (default 256)",
    )
    args = parser.parse_args()

    target_input = args.crop_single or args.input_opt or args.input
    if not target_input:
        # Check if sample_data directory exists in current working directory
        if Path("sample_data").is_dir():
            print("[*] No input directory provided. Auto-detecting 'sample_data' in current directory.")
            target_input = "sample_data"
        else:
            parser.print_help()
            print("\n[ERROR] Vui lòng cung cấp đường dẫn thư mục hoặc file đầu vào.")
            sys.exit(1)

    stack_and_crop_landsat_scene(target_input, args.output, args.size)
