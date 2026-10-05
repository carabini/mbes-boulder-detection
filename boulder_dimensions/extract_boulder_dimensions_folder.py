#!/usr/bin/env python3
"""
extract_boulder_dimensions_folder.py
------------------------------------
Extract boulder height and footprint area for a whole folder of predictions
from ONE model (nnU-Net or APOC), using the method reported in the paper:

  - Predictions are downsampled (nearest-neighbour) to the original-resolution
    bathymetry grid.
  - Footprint area = number of pixels per segment x pixel area.
  - Boulder top = shallowest depth pixel within the segment.
  - Seafloor level = 10th percentile of valid pixels in a 0.75 m annular ring
    around the segment (excluding pixels belonging to other boulders).
  - Height = boulder top - seafloor level.

This is the large-area counterpart to extract_boulder_dimensions.py (which
compares both models on the two validation sites of the paper). It takes the
output folder of a prediction script directly and scales to rasters with
>100 million pixels: predictions are streamed onto the bathymetry grid instead
of being loaded at the upscaled resolution, and per-segment bounding boxes come
from a single pass over the label image.

Run with a Python environment matching apoc/requirements.txt:
  python extract_boulder_dimensions_folder.py --pred-dir PRED --bathy-dir BATHY
         [--model nnunet|apoc] [--out-dir OUT] [--ring-width-m 0.75] [--max-pixels 0]

--model selects the prediction format (default: nnunet):
  nnunet  <name>_mask.tif from nnunet/predict_folder.py — binary mask
          (1 = boulder, 0 = background, 255 = nodata). Only pixels equal to 1
          count as boulder; segments are the 8-connected components on the
          bathymetry grid, so two boulders that touch after the downsample
          become one segment.
  apoc    <name>_pred.tif from apoc/predict_apoc25.py — instance labels
          (0 = background, >0 = boulder ID). The IDs are kept.

Each prediction is paired with the original-resolution bathymetry
<name>.tif in --bathy-dir (the file the prediction was made from).

Outputs (in --out-dir):
  boulder_dimensions_<name>.csv   one per input file
  boulder_dimensions_all.csv      all files concatenated
  boulder_dimensions_summary.txt
"""

import sys
import io
import os
import glob
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation, find_objects, label as nd_label

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# model -> (prediction file suffix, label written to the 'model' column)
MODELS = {
    'nnunet': ('_mask.tif', 'nnU-Net'),
    'apoc':   ('_pred.tif', 'APOC-25'),
}

parser = argparse.ArgumentParser(description="Extract boulder dimensions for a folder of predictions")
parser.add_argument("--pred-dir",     required=True,
                    help="Folder with the predictions (<name>_mask.tif or <name>_pred.tif)")
parser.add_argument("--bathy-dir",    required=True,
                    help="Folder with the original-resolution bathymetry (<name>.tif)")
parser.add_argument("--model",        default="nnunet", choices=sorted(MODELS),
                    help="Which model produced the predictions (default: nnunet)")
parser.add_argument("--out-dir",      default=".", help="Output folder for CSVs and summary")
parser.add_argument("--ring-width-m", default=0.75, type=float,
                    help="Annular ring width (m) for seafloor estimation")
parser.add_argument("--max-pixels",   default=0, type=int,
                    help="Skip segments larger than this many pixels on the bathymetry grid "
                         "(e.g. reef or outcrop areas segmented as one object); "
                         "0 = keep all (default, paper method)")
args = parser.parse_args()

# Configuration
OUT_DIR      = args.out_dir
RING_WIDTH_M = args.ring_width_m
MAX_PIXELS   = args.max_pixels
PRED_SUFFIX, MODEL_LABEL = MODELS[args.model]

COLUMNS = ['segment_id', 'site', 'model', 'n_pixels', 'area_m2', 'area_cm2',
           'pixel_size_m', 'boulder_top_m', 'seafloor_level_m', 'height_m',
           'ring_n_pixels', 'centroid_x', 'centroid_y']

# site -> number of segments skipped by --max-pixels
SKIPPED = {}


def load_bathy(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        tf  = src.transform
        ps  = abs(float(src.transform.a))
        crs = src.crs
    arr[arr < -1e38] = np.nan
    return arr, tf, ps, crs


def labels_on_bathy_grid(pred_path, dst_tf, dst_crs, dst_shape):
    """Nearest-neighbour downsample of the prediction onto the bathymetry grid,
    returned as int32 instance labels (0 = background).

    The source is passed as a rasterio band so GDAL streams it from disk; the
    upscaled-resolution array is never held in memory.
    """
    dest = np.zeros(dst_shape, dtype=np.int32)
    with rasterio.open(pred_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            dst_transform=dst_tf,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    if args.model == 'nnunet':
        # Test == 1, not > 0: the masks carry nodata = 255, and a "> 0" test
        # would turn the whole nodata area into one enormous segment.
        labels, _ = nd_label(dest == 1, structure=np.ones((3, 3), dtype=bool))  # 8-connectivity
        return labels.astype(np.int32)
    dest[dest < 0] = 0
    return dest


def extract_dimensions(bathy_arr, labels, transform, pixel_size, site, model):
    """
    For every boulder segment compute footprint area, height, and the
    10th-percentile ring-based seafloor level estimate (paper method).

    bathy_arr: float32 with NaN; depth in metres (negative, e.g. -20.5 m)
    labels   : int32 instance labels (0=background, >0=boulder ID)
    """
    ring_width_px = max(1, round(RING_WIDTH_M / pixel_size))
    struct        = np.ones((2 * ring_width_px + 1, 2 * ring_width_px + 1), dtype=bool)
    H, W          = labels.shape
    boulder_mask  = labels > 0

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    n_seg = int((sizes > 0).sum())
    print(f"    {n_seg} segments | "
          f"ring={ring_width_px} px ({ring_width_px * pixel_size:.3f} m actual)")

    n_skipped = int((sizes > MAX_PIXELS).sum()) if MAX_PIXELS > 0 else 0
    if MAX_PIXELS > 0:
        print(f"    skipped {n_skipped} segment(s) > {MAX_PIXELS} px "
              f"(> {MAX_PIXELS * pixel_size ** 2:.1f} m2)")
    SKIPPED[site] = n_skipped

    rows = []
    for seg_id, sl in enumerate(find_objects(labels), start=1):
        if sl is None:
            continue
        if MAX_PIXELS > 0 and sizes[seg_id] > MAX_PIXELS:
            continue

        # Bounding-box crop padded by ring_width_px
        r0 = max(0, sl[0].start - ring_width_px)
        r1 = min(H, sl[0].stop + ring_width_px)
        c0 = max(0, sl[1].start - ring_width_px)
        c1 = min(W, sl[1].stop + ring_width_px)

        seg_crop     = labels[r0:r1, c0:c1] == seg_id
        boulder_crop = boulder_mask[r0:r1, c0:c1]
        bathy_crop   = bathy_arr[r0:r1, c0:c1]

        # Ring = dilation minus all boulder pixels
        dilated   = binary_dilation(seg_crop, structure=struct)
        ring_crop = dilated & ~boulder_crop

        # Boulder top: shallowest depth pixel within the segment
        seg_bathy  = bathy_crop[seg_crop]
        seg_valid  = seg_bathy[~np.isnan(seg_bathy)]
        boulder_top = float(seg_valid.max()) if len(seg_valid) > 0 else np.nan

        # Ring values excluding NaN
        ring_vals  = bathy_crop[ring_crop]
        valid_ring = ring_vals[~np.isnan(ring_vals)]
        ring_n     = int(len(valid_ring))

        if ring_n == 0:
            seafloor_level = np.nan
        else:
            seafloor_level = float(np.percentile(valid_ring, 10))

        height = (np.nan if (np.isnan(boulder_top) or np.isnan(seafloor_level))
                  else float(boulder_top - seafloor_level))

        n_pixels = int(sizes[seg_id])
        area_m2  = float(n_pixels * pixel_size ** 2)

        # Centroid in map coordinates
        rr, cc = np.nonzero(seg_crop)
        x, y = rasterio.transform.xy(transform, rr.mean() + r0, cc.mean() + c0)

        rows.append({
            'segment_id':       seg_id,
            'site':             site,
            'model':            model,
            'n_pixels':         n_pixels,
            'area_m2':          area_m2,
            'area_cm2':         area_m2 * 10_000,
            'pixel_size_m':     pixel_size,
            'boulder_top_m':    boulder_top,
            'seafloor_level_m': seafloor_level,
            'height_m':         height,
            'ring_n_pixels':    ring_n,
            'centroid_x':       float(x),
            'centroid_y':       float(y),
        })

    return pd.DataFrame(rows, columns=COLUMNS)


def process_site(site, pred_path, bathy_path):
    print(f"\n{'='*60}\nSite: {site}\n{'='*60}")
    print(f"  Pred       : {os.path.basename(pred_path)}")

    bathy, tf, ps, crs = load_bathy(bathy_path)
    print(f"  Orig bathy : {bathy.shape}, {ps:.4f} m/px")

    labels = labels_on_bathy_grid(pred_path, tf, crs, bathy.shape)
    return extract_dimensions(bathy, labels, tf, ps, site, MODEL_LABEL)


def compute_summary(df_all):
    lines = ['Boulder Dimension Extraction - Summary Statistics',
             '=' * 65,
             f'Model : {MODEL_LABEL}',
             f'Method: height = max(depth[boulder]) - P10(depth[{RING_WIDTH_M} m ring])',
             'Grid  : original-resolution bathymetry, predictions nearest-neighbour downsampled']
    if MAX_PIXELS > 0:
        lines.append(f'Filter: segments > {MAX_PIXELS} px skipped')
    lines.append('')

    for site in SKIPPED:
        sub = df_all[df_all.site == site]
        n_total = len(sub)
        lines.append(f'Site={site}  Model={MODEL_LABEL}')
        lines.append(f'  n_segments : {n_total}')
        if MAX_PIXELS > 0:
            lines.append(f'  n_skipped  : {SKIPPED[site]}  (> {MAX_PIXELS} px)')
        if n_total == 0:
            lines.append('')
            continue

        n_lt5 = int((sub.ring_n_pixels < 5).sum())
        lines.append(f'  n_ring<5   : {n_lt5}  ({100*n_lt5/n_total:.1f}%)')

        for col, label in [
            ('height_m',      'height      [m]'),
            ('area_m2',       'area        [m2]'),
            ('ring_n_pixels', 'ring_n_px'),
        ]:
            v = sub[col].dropna()
            if len(v) == 0:
                lines.append(f'  {label:20s}: no data')
            else:
                lines.append(
                    f'  {label:20s}: '
                    f'median={np.median(v):.4f}  '
                    f'mean={v.mean():.4f}  '
                    f'std={v.std():.4f}  '
                    f'min={v.min():.4f}  '
                    f'max={v.max():.4f}'
                )
        lines.append('')

    lines += ['-' * 65, f'TOTAL segments: {len(df_all)}']
    return '\n'.join(lines)


def find_jobs():
    """Pair every <name><suffix> in --pred-dir with <name>.tif in --bathy-dir."""
    preds = sorted(glob.glob(os.path.join(args.pred_dir, '*' + PRED_SUFFIX)))
    if not preds:
        sys.exit(f"No *{PRED_SUFFIX} files found in {args.pred_dir}")

    jobs, missing = [], []
    for pred_path in preds:
        site = os.path.basename(pred_path)[:-len(PRED_SUFFIX)]
        bathy_path = os.path.join(args.bathy_dir, site + '.tif')
        if os.path.isfile(bathy_path):
            jobs.append((site, pred_path, bathy_path))
        else:
            missing.append(f"  {os.path.basename(pred_path)}  ->  expected {site}.tif")
    if missing:
        sys.exit(f"No matching bathymetry in {args.bathy_dir} for:\n" + '\n'.join(missing))
    return jobs


def main():
    jobs = find_jobs()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{len(jobs)} file(s) to process ({MODEL_LABEL}):")
    for site, pred_path, _ in jobs:
        print(f"  {os.path.basename(pred_path)}")

    results = []
    for site, pred_path, bathy_path in jobs:
        df_site = process_site(site, pred_path, bathy_path)
        results.append(df_site)
        csv_path = os.path.join(OUT_DIR, f'boulder_dimensions_{site}.csv')
        df_site.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nSaved CSV: {csv_path}  ({len(df_site)} rows)")

    df_all = pd.concat(results, ignore_index=True)
    all_path = os.path.join(OUT_DIR, 'boulder_dimensions_all.csv')
    df_all.to_csv(all_path, index=False, encoding='utf-8')
    print(f"\nSaved combined CSV: {all_path}  ({len(df_all)} rows)")

    summary = compute_summary(df_all)
    print('\n' + summary)
    summary_path = os.path.join(OUT_DIR, 'boulder_dimensions_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'Saved summary: {summary_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
