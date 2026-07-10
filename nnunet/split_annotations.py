"""
split_annotations.py
--------------------
Splits all non-tiled annotation files (and matching bathy) into 4-6 tiles.
- Splits into 6 tiles (2x3) for larger files, 4 (2x2) for smaller, 2 (1x2) for tiny
- Saves annotation tiles to the --anno-dir folder (alongside the originals)
- Saves bathy tiles to the --bathy-dst folder
- Filters out tile pairs with zero boulder pixels (class=1)
- Originals are removed only with --delete-originals; otherwise the script just
  reports what it would remove

Run with a Python environment matching requirements.txt.

Options:
  --anno-dir          Folder with annotation GeoTIFFs to split (tiles written here too)
  --bathy-src         Folder with matching original (non-tiled) bathymetry
  --bathy-dst         Folder to write split bathymetry tiles to
  --delete-originals  Remove the original (non-tiled) files after splitting
"""

import os, sys, argparse
import numpy as np
import rasterio
from rasterio.transform import from_bounds

sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser(description="Split non-tiled annotation + bathy files into 4-6 tiles")
parser.add_argument("--anno-dir",  default=r"data/annotations_complete",
                    help="Folder with annotation GeoTIFFs (tiles written here too)")
parser.add_argument("--bathy-src", default=r"data/bathymetry",
                    help="Folder with matching original (non-tiled) bathymetry")
parser.add_argument("--bathy-dst", default=r"data/bathy",
                    help="Folder to write split bathymetry tiles to")
parser.add_argument("--delete-originals", action="store_true",
                    help="Remove the original (non-tiled) files after splitting (default: off)")
args = parser.parse_args()

ANNO_DIR         = args.anno_dir
BATHY_SRC        = args.bathy_src
BATHY_DST        = args.bathy_dst
DELETE_ORIGINALS = args.delete_originals

# Output folder for split bathy tiles may not exist on a fresh checkout.
os.makedirs(BATHY_DST, exist_ok=True)

def choose_grid(W, H):
    if W >= 150 and H >= 100:
        return 2, 3   # 6 tiles
    elif W >= 80 and H >= 50:
        return 2, 2   # 4 tiles
    else:
        return 1, 2   # 2 tiles

def split_raster(data, profile, rows, cols):
    """Split (H,W) array into rows*cols tiles. Returns list of (tile_data, tile_profile)."""
    H, W = data.shape
    bounds = rasterio.transform.array_bounds(H, W, profile['transform'])

    tiles = []
    for r in range(rows):
        row_s = (H * r) // rows
        row_e = (H * (r+1)) // rows
        for c in range(cols):
            col_s = (W * c) // cols
            col_e = (W * (c+1)) // cols
            tile  = data[row_s:row_e, col_s:col_e]
            th, tw = tile.shape

            x_min = bounds[0] + (col_s / W) * (bounds[2] - bounds[0])
            x_max = bounds[0] + (col_e / W) * (bounds[2] - bounds[0])
            y_max = bounds[3] - (row_s / H) * (bounds[3] - bounds[1])
            y_min = bounds[3] - (row_e / H) * (bounds[3] - bounds[1])

            t_profile = profile.copy()
            t_profile.update(
                width=tw, height=th,
                transform=from_bounds(x_min, y_min, x_max, y_max, tw, th)
            )
            tiles.append((tile, t_profile))
    return tiles

# Identify non-tiled files (everything not starting with dl_)
non_tiled = sorted(f for f in os.listdir(ANNO_DIR)
                   if f.endswith('.tif') and not f.startswith('dl_'))

print(f'Files to split: {len(non_tiled)}')
print()

removed_anno, removed_bathy = [], []
skipped_no_boulder = 0
created = 0

for fname in non_tiled:
    stem = fname[:-4]  # strip .tif
    anno_path  = os.path.join(ANNO_DIR, fname)
    bathy_path = os.path.join(BATHY_SRC, fname)

    # Load annotation
    with rasterio.open(anno_path) as src:
        anno    = src.read(1)
        a_prof  = src.profile.copy()
        W, H    = src.width, src.height

    # Load bathy if available
    bathy, b_prof = None, None
    if os.path.exists(bathy_path):
        with rasterio.open(bathy_path) as src:
            bathy  = src.read(1).astype(np.float32)
            b_prof = src.profile.copy()
        bathy[(bathy < -1e38)] = np.nan
        b_prof['nodata'] = None

    rows, cols = choose_grid(W, H)
    n_tiles = rows * cols
    print(f'{fname} ({W}x{H}) -> {n_tiles} tiles ({rows}x{cols})')

    anno_tiles  = split_raster(anno,  a_prof, rows, cols)
    bathy_tiles = split_raster(bathy, b_prof, rows, cols) if bathy is not None else [None]*n_tiles

    tile_idx = 1
    for (a_tile, a_tp), bt in zip(anno_tiles, bathy_tiles):
        n_boulder = int(np.sum(a_tile == 1))
        out_name  = f'{stem}_{tile_idx:02d}.tif'

        if n_boulder == 0:
            print(f'  {out_name}: 0 boulders -> SKIPPED')
            skipped_no_boulder += 1
            tile_idx += 1
            continue

        # Write annotation tile
        a_tp['dtype'] = 'float32'
        a_tp['nodata'] = None
        with rasterio.open(os.path.join(ANNO_DIR, out_name), 'w', **a_tp) as dst:
            dst.write(a_tile.astype(np.float32), 1)

        # Write bathy tile
        if bt is not None:
            b_tile, b_tp = bt
            b_tp['dtype']  = 'float32'
            b_tp['nodata'] = None
            with rasterio.open(os.path.join(BATHY_DST, out_name), 'w', **b_tp) as dst:
                dst.write(b_tile.astype(np.float32), 1)

        print(f'  {out_name}: {n_boulder} boulder px')
        created += 1
        tile_idx += 1

    # Original files: remove only when explicitly requested.
    bathy_dst_orig = os.path.join(BATHY_DST, fname)
    if DELETE_ORIGINALS:
        os.remove(anno_path)
        removed_anno.append(fname)
        if os.path.exists(bathy_dst_orig):
            os.remove(bathy_dst_orig)
            removed_bathy.append(fname)
    else:
        print(f'  would remove original: {anno_path}')
        if os.path.exists(bathy_dst_orig):
            print(f'  would remove original: {bathy_dst_orig}')

    print()

print('=' * 50)
print(f'Created:           {created} tile pairs')
print(f'Skipped (0 bould): {skipped_no_boulder} tiles')
if DELETE_ORIGINALS:
    print(f'Originals removed: {len(removed_anno)} annotations, {len(removed_bathy)} bathy')
else:
    print('Originals kept (pass --delete-originals to remove them)')
