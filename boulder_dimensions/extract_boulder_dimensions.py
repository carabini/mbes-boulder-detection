#!/usr/bin/env python3
"""
extract_boulder_dimensions.py
------------------------------
Extract boulder height and footprint area from segmentation predictions
(APOC-25 and nnU-Net), matching the method reported in the paper:

  - Predictions are downsampled (nearest-neighbour) to the original-resolution
    bathymetry grid.
  - Footprint area = number of pixels per segment x pixel area.
  - Boulder top = shallowest depth pixel within the segment.
  - Seafloor level = 10th percentile of valid pixels in a 0.75 m annular ring
    around the segment (excluding pixels belonging to other boulders) —
    robust against single outliers.
  - Height = boulder top - seafloor level.

Run with a Python environment matching requirements.txt:
  python extract_boulder_dimensions.py [--out-dir OUT] [--sites-config sites.json] [--ring-width-m 0.75]

--sites-config: JSON file mapping site name -> {bathy_orig, apoc_pred, nnunet_pred}
  file paths (see sites.example.json next to this script for the expected
  structure). If omitted, the built-in generic example paths are used — these
  are placeholders, so supply --sites-config to point at your own data.

Outputs (in --out-dir):
  boulder_dimensions_<site>.csv (one per site)
  boulder_dimensions_summary.txt
"""

import sys
import io
import os
import json
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation, label as nd_label

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

parser = argparse.ArgumentParser(description="Extract boulder dimensions from predicted masks")
parser.add_argument("--out-dir",      default=".", help="Output folder for CSVs and summary")
parser.add_argument("--sites-config", default=None,
                    help="JSON file with site paths; if omitted, uses built-in generic example paths")
parser.add_argument("--ring-width-m", default=0.75, type=float, help="Annular ring width (m) for seafloor estimation")
args = parser.parse_args()

# Configuration
OUT_DIR      = args.out_dir
RING_WIDTH_M = args.ring_width_m

# Built-in generic example paths (placeholders — see sites.example.json).
# Pass --sites-config with a JSON file in the same structure to use your own data.
SITES = {
    'NH': {
        'bathy_orig':   'data/NH_bathy.tif',
        'apoc_pred':    'data/NH_apoc_pred.tif',
        'nnunet_pred':  'data/NH_nnunet_labeled.tif',
    },
    'FMB': {
        'bathy_orig':   'data/FMB_bathy.tif',
        'apoc_pred':    'data/FMB_apoc_pred.tif',
        'nnunet_pred':  'data/FMB_nnunet_labeled.tif',
    },
}

if args.sites_config:
    with open(args.sites_config, 'r', encoding='utf-8') as f:
        SITES = json.load(f)


def load_bathy(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        tf  = src.transform
        ps  = abs(float(src.transform.a))
        crs = src.crs
    arr[arr < -1e38] = np.nan
    return arr, tf, ps, crs


def load_pred(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        tf  = src.transform
        ps  = abs(float(src.transform.a))
        crs = src.crs
    return arr, tf, ps, crs


def ensure_instance_labels(arr):
    if arr.max() <= 1:
        labeled, _ = nd_label(arr > 0, structure=np.ones((3, 3), dtype=bool))  # 8-connectivity
        return labeled.astype(np.int32)
    return arr.astype(np.int32)


def downsample_pred_nn(pred, src_tf, src_crs, dst_tf, dst_crs, dst_shape):
    dest = np.zeros(dst_shape, dtype=np.int32)
    reproject(
        source=pred,
        destination=dest,
        src_transform=src_tf,
        src_crs=src_crs,
        dst_transform=dst_tf,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return dest


def extract_dimensions(bathy_arr, pred_arr, pixel_size, site, model):
    """
    For every boulder segment compute footprint area, height, and the
    10th-percentile ring-based seafloor level estimate (paper method).

    bathy_arr: float32 with NaN; depth in metres (negative, e.g. -20.5 m)
    pred_arr : int32 instance labels (0=background, >0=boulder ID)
    """
    ring_width_px = max(1, round(RING_WIDTH_M / pixel_size))
    struct        = np.ones((2 * ring_width_px + 1, 2 * ring_width_px + 1), dtype=bool)
    H, W          = pred_arr.shape
    boulder_mask  = pred_arr > 0

    seg_ids = np.unique(pred_arr)
    seg_ids = seg_ids[seg_ids > 0]

    print(f"    {len(seg_ids)} segments | "
          f"ring={ring_width_px} px ({ring_width_px * pixel_size:.3f} m actual)")

    rows = []
    for seg_id in seg_ids:
        seg_mask = pred_arr == seg_id

        # Bounding-box crop padded by ring_width_px
        ridx, cidx = np.where(seg_mask)
        r0 = max(0, ridx.min() - ring_width_px)
        r1 = min(H, ridx.max() + ring_width_px + 1)
        c0 = max(0, cidx.min() - ring_width_px)
        c1 = min(W, cidx.max() + ring_width_px + 1)

        seg_crop     = seg_mask[r0:r1, c0:c1]
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

        n_pixels = int(seg_mask.sum())
        area_m2  = float(n_pixels * pixel_size ** 2)

        rows.append({
            'segment_id':      int(seg_id),
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
        })

    return pd.DataFrame(rows)


def process_site(site, cfg):
    print(f"\n{'='*60}\nSite: {site}\n{'='*60}")

    bathy_orig, tf_orig, ps_orig, crs_orig = load_bathy(cfg['bathy_orig'])
    print(f"  Orig bathy : {bathy_orig.shape}, {ps_orig:.4f} m/px")

    dfs = []
    for model, pred_path in [('APOC-25', cfg['apoc_pred']),
                              ('nnU-Net', cfg['nnunet_pred'])]:
        print(f"\n  Model: {model}")
        pred_raw, pred_tf, pred_ps, pred_crs = load_pred(pred_path)
        pred_inst = ensure_instance_labels(pred_raw)
        n_seg = len(np.unique(pred_inst[pred_inst > 0]))
        print(f"    Pred: {pred_raw.shape}, {pred_ps:.4f} m/px, n_boulders={n_seg}")

        pred_orig = downsample_pred_nn(
            pred_inst, pred_tf, pred_crs,
            tf_orig, crs_orig, bathy_orig.shape)

        print(f"\n  [Original @ {ps_orig:.4f} m]")
        df_orig = extract_dimensions(bathy_orig, pred_orig, ps_orig, site, model)
        dfs.append(df_orig)

    return pd.concat(dfs, ignore_index=True)


def compute_summary(df_all):
    lines = ['Boulder Dimension Extraction - Summary Statistics',
             '=' * 65, '']

    for site in sorted(df_all.site.unique()):
        for model in sorted(df_all.model.unique()):
            sub = df_all[(df_all.site == site) & (df_all.model == model)]
            if sub.empty:
                continue

            n_total = len(sub)
            n_lt5   = int((sub.ring_n_pixels < 5).sum())

            lines += [
                f'Site={site}  Model={model}',
                f'  n_segments : {n_total}',
                f'  n_ring<5   : {n_lt5}  ({100*n_lt5/n_total:.1f}%)',
            ]

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

    return '\n'.join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    for site, cfg in SITES.items():
        df_site = process_site(site, cfg)
        results[site] = df_site
        csv_path = os.path.join(OUT_DIR, f'boulder_dimensions_{site}.csv')
        df_site.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nSaved CSV: {csv_path}  ({len(df_site)} rows)")

    df_all = pd.concat(results.values(), ignore_index=True)

    summary = compute_summary(df_all)
    print('\n' + summary)
    summary_path = os.path.join(OUT_DIR, 'boulder_dimensions_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'Saved summary: {summary_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
