"""
label_boulders.py
-----------------
Convert a binary boulder mask GeoTIFF into a labelled instance GeoTIFF
where every boulder gets a unique integer ID (1, 2, 3, ...).

Usage:
  python label_boulders.py --input  <mask.tif or masks_folder/>
                           --output <output_folder>
                           [--min_px 5]

  --min_px : remove components smaller than this many pixels (noise filter)

Output per input file:
  <name>_labeled.tif   uint32 GeoTIFF, 0=background, 1..N=individual boulders
  <name>_stats.csv     one row per boulder: id, area_px, area_m2, centroid_x, centroid_y

A summary CSV with total boulder counts is also written to the output folder.

Run with a Python environment matching requirements.txt:
  python label_boulders.py ...
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import label as ndlabel
from scipy.ndimage import sum as ndsum


def process_mask(mask_path: Path, output_dir: Path, min_px: int = 5):
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)
        profile = src.profile.copy()
        transform = src.transform
        pixel_size = abs(transform.a)  # metres per pixel

    binary = (mask == 1)

    # Connected component labelling (8-connectivity)
    labeled, n_raw = ndlabel(binary, structure=np.ones((3, 3), dtype=bool))

    # Remove small components (noise)
    if min_px > 1:
        sizes = ndsum(binary, labeled, range(1, n_raw + 1))
        remove = np.array(sizes) < min_px
        remove_ids = np.where(remove)[0] + 1
        for rid in remove_ids:
            labeled[labeled == rid] = 0
        # Re-label consecutively
        _, labeled = np.unique(labeled, return_inverse=True)
        labeled = labeled.reshape(mask.shape).astype(np.int32)
        n_boulders = int(labeled.max())
    else:
        n_boulders = n_raw

    print(f"  {mask_path.name}: {n_boulders} boulders (removed {n_raw - n_boulders} tiny components < {min_px} px)")

    # Output naming: strip a trailing "_mask" from the stem, then append "_labeled".
    base_stem = mask_path.stem
    if base_stem.endswith("_mask"):
        base_stem = base_stem[:-len("_mask")]

    # Save labelled GeoTIFF
    out_profile = profile.copy()
    out_profile.update(dtype="uint32", nodata=0, compress="lzw", count=1)
    labeled_path = output_dir / f"{base_stem}_labeled.tif"

    with rasterio.open(labeled_path, "w", **out_profile) as dst:
        dst.write(labeled.astype(np.uint32), 1)

    # Per-boulder stats
    pixel_area_m2 = pixel_size ** 2
    stats_path = output_dir / f"{base_stem}_stats.csv"
    rows = []
    for bid in range(1, n_boulders + 1):
        yx = np.argwhere(labeled == bid)
        area_px = len(yx)
        area_m2 = area_px * pixel_area_m2
        cy, cx = yx.mean(axis=0)
        # Convert pixel centroid to map coordinates
        map_x, map_y = xy(transform, cy, cx)
        rows.append({
            "boulder_id":  bid,
            "area_px":     area_px,
            "area_m2":     round(area_m2, 4),
            "centroid_x":  round(map_x, 4),
            "centroid_y":  round(map_y, 4),
        })

    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["boulder_id", "area_px", "area_m2", "centroid_x", "centroid_y"])
        writer.writeheader()
        writer.writerows(rows)

    return base_stem, n_boulders, stats_path


def main():
    parser = argparse.ArgumentParser(description="Label boulder mask instances with unique IDs")
    parser.add_argument("--input",  required=True, help="Binary mask GeoTIFF or folder of masks")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--min_px", type=int, default=5, help="Min component size in pixels (default: 5)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        mask_files = sorted(input_path.glob("*_mask.tif")) + sorted(input_path.glob("*.tif"))
        # Deduplicate
        seen = set()
        tifs = []
        for p in mask_files:
            if p.name not in seen:
                seen.add(p.name)
                tifs.append(p)
    else:
        tifs = [input_path]

    if not tifs:
        print(f"No TIF files found in {input_path}")
        return

    summary_rows = []
    for tif in tifs:
        stem, n, stats_path = process_mask(tif, output_dir, min_px=args.min_px)
        summary_rows.append({"file": stem, "n_boulders": n, "stats_csv": stats_path.name})
        print(f"    -> {stats_path.parent / (stem + '_labeled.tif')}")
        print(f"    -> {stats_path.name}")

    summary_path = output_dir / "boulder_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "n_boulders", "stats_csv"])
        writer.writeheader()
        writer.writerows(summary_rows)

    total = sum(r["n_boulders"] for r in summary_rows)
    print(f"\nSummary: {len(tifs)} file(s), {total} boulders total")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
