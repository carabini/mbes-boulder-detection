"""
upscale.py
----------
2x linear upscaling of bathymetry and annotation GeoTIFFs. Shared by both the
APOC and nnU-Net pipelines (single copy at the repository root).
- Bathymetry : bilinear interpolation (order=1)
- Annotations: nearest-neighbour interpolation (order=0) — preserves integer class values

Run with a Python environment matching the relevant requirements.txt
(apoc/requirements.txt or nnunet/requirements.txt):
  python upscale.py [options]

Options (all optional):
  --anno-dir    Folder with annotation GeoTIFFs to upscale (default: data/annotations)
  --bathy-dirs  One or more folders to search for matching bathymetry files
                (default: data/bathy data/bathymetry)
  --out-bathy   Output folder for upscaled bathymetry (default: data/bathy_upscaled)
  --out-anno    Output folder for upscaled annotations (default: data/annotations_upscaled)
  --scale       Linear upscaling factor (default: 2)
"""

import argparse
import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import zoom

parser = argparse.ArgumentParser(description="2x linear upscaling of bathymetry and annotation GeoTIFFs")
parser.add_argument("--anno-dir",   default="data/annotations",
                    help="Folder with annotation GeoTIFFs")
parser.add_argument("--bathy-dirs", nargs="+",
                    default=["data/bathy", "data/bathymetry"],
                    help="Folders to search for matching bathymetry files")
parser.add_argument("--out-bathy",  default="data/bathy_upscaled",
                    help="Output folder for upscaled bathymetry")
parser.add_argument("--out-anno",   default="data/annotations_upscaled",
                    help="Output folder for upscaled annotations")
parser.add_argument("--scale",      default=2, type=int, help="Linear upscaling factor (default: 2)")
args = parser.parse_args()

SCALE      = args.scale
BATHY_DIRS = args.bathy_dirs
ANNO_DIR   = args.anno_dir
OUT_BATHY  = args.out_bathy
OUT_ANNO   = args.out_anno

os.makedirs(OUT_BATHY, exist_ok=True)
os.makedirs(OUT_ANNO,  exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def upscaled_transform(src_transform, src_width, src_height, scale):
    """Return new affine transform for upscaled raster (same bounds, smaller pixels)."""
    bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
    # bounds = (west, south, east, north)
    new_w = src_width  * scale
    new_h = src_height * scale
    new_transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], new_w, new_h)
    return new_transform, new_w, new_h


def upscale_bathy(src_path, dst_path, scale):
    with rasterio.open(src_path) as src:
        data    = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata  = src.nodata
        transform, width, height = src.transform, src.width, src.height

    # Mark nodata as NaN
    if nodata is not None:
        data[data == nodata] = np.nan

    # Replace NaN with local mean for interpolation, then restore
    nan_mask = np.isnan(data)
    fill_val = np.nanmean(data) if not np.all(nan_mask) else 0.0
    data_filled = np.where(nan_mask, fill_val, data)

    # Bilinear upscaling
    upscaled = zoom(data_filled, scale, order=1).astype(np.float32)

    # Upscale nan_mask and restore NaN
    upscaled_nan = zoom(nan_mask.astype(np.float32), scale, order=0).astype(bool)
    upscaled[upscaled_nan] = np.nan

    new_transform, new_w, new_h = upscaled_transform(transform, width, height, scale)

    # NaN nodata is not valid for all drivers — keep nodata=None and NaN in data.
    profile.update(
        width=new_w, height=new_h,
        transform=new_transform,
        dtype='float32',
        nodata=None,
        count=1,
    )

    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(upscaled, 1)


def upscale_anno(src_path, dst_path, scale):
    with rasterio.open(src_path) as src:
        data      = src.read(1)
        profile   = src.profile.copy()
        transform, width, height = src.transform, src.width, src.height

    # Nearest-neighbour — preserves class values {0, 1, 2}
    upscaled = zoom(data.astype(np.float32), scale, order=0).astype(np.float32)

    new_transform, new_w, new_h = upscaled_transform(transform, width, height, scale)

    profile.update(
        width=new_w, height=new_h,
        transform=new_transform,
        dtype='float32',
        nodata=None,
        count=1,
    )

    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(upscaled, 1)


# ── process annotations ───────────────────────────────────────────────────────

anno_files = sorted(f for f in os.listdir(ANNO_DIR) if f.endswith('.tif'))
print(f'Upscaling {len(anno_files)} annotation files (nearest-neighbour, {SCALE}x)...')
for fname in anno_files:
    src = os.path.join(ANNO_DIR, fname)
    dst = os.path.join(OUT_ANNO, fname)
    upscale_anno(src, dst, SCALE)
    # Quick check: unique values should still be within {0, 1, 2}
    with rasterio.open(dst) as r:
        vals = np.unique(r.read(1))
    print(f'  {fname} -> unique values: {vals}')

# ── process bathymetry ────────────────────────────────────────────────────────

# Collect unique bathy files needed (one per anno file, deduplicated)
needed = set(anno_files)  # same filenames
print(f'\nUpscaling bathymetry ({SCALE}x, bilinear)...')
found, missing = [], []
for fname in sorted(needed):
    bpath = None
    for bd in BATHY_DIRS:
        p = os.path.join(bd, fname)
        if os.path.exists(p):
            bpath = p
            break
    if bpath is None:
        missing.append(fname)
        continue
    dst = os.path.join(OUT_BATHY, fname)
    upscale_bathy(bpath, dst, SCALE)
    found.append(fname)
    print(f'  {fname}')

print(f'\nDone. Bathy: {len(found)} upscaled, {len(missing)} missing.')
if missing:
    print(f'Missing bathy: {missing}')
print(f'Output:')
print(f'  Annotations: {OUT_ANNO}')
print(f'  Bathymetry:  {OUT_BATHY}')
