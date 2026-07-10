"""
predict_apoc25.py
-----------------
Predict boulders on a GeoTIFF using the APOC-25 classifier (25% sparse annotations,
10:1 bg:boulder ratio, sigma=2,3,4,6) — or any compatible classifier passed via --classifier.

Matches training pipeline:
  - Upscale 2x bilinear before prediction (matches training data preparation)
  - NaN-fill with Gaussian sigma=5
  - Predict with tiled inference (TILE_SIZE + OVERLAP px border)
  - Compute boulder statistics (area, height, L/W, centroid)

Height method: the per-boulder CSV uses the same height definition as the paper
(seafloor reference = the Nth percentile of a ring around the boulder, default
0.75 m ring and 10th percentile; height = shallowest segment pixel minus that
reference). The authoritative dimension extraction for the paper is
boulder_dimensions/extract_boulder_dimensions.py — use that for reported figures.

Run with a Python environment matching requirements.txt:
  python predict_apoc25.py INPUT.tif [OUTPUT_DIR] [options]

  If OUTPUT_DIR is omitted, results are saved next to the input file.

Options (all optional, defaults shown match the paper method):
  --classifier          Path to trained APOC .cl classifier file
                        (default: /path/to/apoc_classifier_upscaled_sparse_bg10to1_25pct_sigma2346_v1.cl)
  --scale               Linear upscaling factor (default: 2)
  --ring-width-m        Annular ring width (m) for height estimation (default: 0.75)
  --seafloor-percentile Percentile of ring depths used as seafloor reference (default: 10)
  --tile-size           Inner tile size (px) for tiled prediction (default: 512)
  --overlap             Overlap border (px) (default: 64)

Outputs (stem = INPUT filename without extension):
  <OUTPUT_DIR>/<stem>_upscaled.tif         — 2x bilinear upscaled bathymetry
  <OUTPUT_DIR>/<stem>_pred.tif             — APOC-25 instance segmentation (raw model output)
  <OUTPUT_DIR>/<stem>_boulder_stats.csv    — per-boulder statistics
"""

import sys, io, os, csv, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import zoom, gaussian_filter, binary_dilation
from skimage.measure import regionprops
import apoc

# ── Config ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Predict boulders on a GeoTIFF using the APOC-25 classifier")
parser.add_argument("input", help="Input bathymetry GeoTIFF")
parser.add_argument("output_dir", nargs="?", default=None,
                    help="Output folder (default: next to input file)")
parser.add_argument("--classifier", default=r"/path/to/apoc_classifier_upscaled_sparse_bg10to1_25pct_sigma2346_v1.cl",
                    help="Path to trained APOC .cl classifier file")
parser.add_argument("--scale",        default=2,    type=int,   help="Linear upscaling factor (default: 2)")
parser.add_argument("--ring-width-m", default=0.75, type=float,
                    help="Annular ring width (m) for height estimation (paper method: 0.75)")
parser.add_argument("--seafloor-percentile", default=10, type=float,
                    help="Percentile of ring depths used as seafloor reference (paper method: 10)")
parser.add_argument("--tile-size",    default=512,  type=int,   help="Inner tile size (px) for tiled prediction")
parser.add_argument("--overlap",      default=64,   type=int,   help="Overlap border (px)")
args = parser.parse_args()

CL_PATH             = args.classifier
SCALE               = args.scale
RING_WIDTH_M        = args.ring_width_m
SEAFLOOR_PERCENTILE = args.seafloor_percentile
TILE_SIZE           = args.tile_size
OVERLAP             = args.overlap

INPUT_FILE = args.input
OUT_DIR = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(INPUT_FILE))

os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]

upscaled_path  = os.path.join(OUT_DIR, f"{stem}_upscaled.tif")
pred_path      = os.path.join(OUT_DIR, f"{stem}_pred.tif")
stats_csv_path = os.path.join(OUT_DIR, f"{stem}_boulder_stats.csv")

print("=" * 70)
print("APOC-25 PREDICTION PIPELINE")
print("=" * 70)
print(f"Classifier : {os.path.basename(CL_PATH)}")
print(f"Input      : {INPUT_FILE}")
print(f"Output dir : {OUT_DIR}")
print(f"Upscale    : {SCALE}x bilinear")
print(f"Ring width : {RING_WIDTH_M} m")
print(f"Seafloor   : P{SEAFLOOR_PERCENTILE:g} of ring depths (paper height method)")


# ── 1. Load input ─────────────────────────────────────────────────────────────
print("\n[1/4] Loading input ...")
with rasterio.open(INPUT_FILE) as src:
    raw = src.read(1).astype(np.float32)
    profile_orig = src.profile.copy()
    src_transform = src.transform
    src_width, src_height = src.width, src.height
    nodata_val = src.nodata
    pixel_sx = abs(src_transform.a)
    pixel_sy = abs(src_transform.e)

# Mark nodata as NaN
if nodata_val is not None and not np.isnan(nodata_val):
    raw[raw == nodata_val] = np.nan
raw[raw < -1e38] = np.nan

nan_orig = ~np.isfinite(raw)
H0, W0 = raw.shape
print(f"  Shape    : {H0} x {W0}  ({H0*W0/1e6:.3f} Mpx)")
print(f"  Pixel    : {pixel_sx:.4f} m x {pixel_sy:.4f} m")
print(f"  NaN      : {nan_orig.sum():,} px ({nan_orig.mean()*100:.1f}%)")
valid = raw[~nan_orig]
print(f"  Depth    : min={valid.min():+.3f}  max={valid.max():+.3f}  std={valid.std():.4f} m")

# The ring radius is derived from the x pixel size only; warn on non-square pixels.
if pixel_sx > 0 and abs(pixel_sx - pixel_sy) / pixel_sx > 0.01:
    print(f"  WARNING  : non-square pixels ({pixel_sx:.4f} x {pixel_sy:.4f} m) — "
          f"ring width is derived from the x pixel size only.")


# ── 2. Upscale 2x bilinear ────────────────────────────────────────────────────
print(f"\n[2/4] Upscaling {SCALE}x (bilinear) ...")
t0 = time.time()

# Fill NaN with local mean before interpolation, then restore
fill_val = float(np.nanmean(raw))
raw_filled_for_zoom = np.where(nan_orig, fill_val, raw)

upscaled = zoom(raw_filled_for_zoom, SCALE, order=1).astype(np.float32)

# Upscale NaN mask with nearest-neighbour and restore
nan_up = zoom(nan_orig.astype(np.float32), SCALE, order=0).astype(bool)
upscaled[nan_up] = np.nan

H, W = upscaled.shape
pixel_sx_up = pixel_sx / SCALE
pixel_sy_up = pixel_sy / SCALE

# Compute new transform (same spatial bounds, smaller pixels)
bounds = rasterio.transform.array_bounds(src_height, src_width, src_transform)
new_transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], W, H)

upscaled_profile = profile_orig.copy()
upscaled_profile.update(
    width=W, height=H,
    transform=new_transform,
    dtype='float32',
    nodata=None,   # NaN nodata invalid for some drivers; keep NaN in data
    count=1,
)
with rasterio.open(upscaled_path, 'w', **upscaled_profile) as dst:
    dst.write(upscaled, 1)

print(f"  Upscaled shape : {H} x {W}  ({H*W/1e6:.3f} Mpx)")
print(f"  Pixel size     : {pixel_sx_up:.4f} m x {pixel_sy_up:.4f} m")
print(f"  Saved          : {upscaled_path}  [{time.time()-t0:.1f}s]")


# ── 3. NaN fill (sigma=5) for APOC input ─────────────────────────────────────
print("\n[3/4] NaN fill (sigma=5) ...")
nan_mask = ~np.isfinite(upscaled)
filled = upscaled.copy()
if nan_mask.any():
    tmp = np.where(nan_mask, 0.0, filled.astype(np.float64))
    smooth = gaussian_filter(tmp, sigma=5)
    filled = np.where(nan_mask, smooth, filled).astype(np.float32)
print(f"  Filled {nan_mask.sum():,} NaN pixels")


# ── 4. Tiled APOC-25 prediction ───────────────────────────────────────────────
print(f"\n[4/4] Predicting (APOC-25) ...")
segmenter = apoc.ObjectSegmenter(opencl_filename=CL_PATH, positive_class_identifier=1)

ring_px  = max(1, round(RING_WIDTH_M / pixel_sx_up))
pad_max  = TILE_SIZE + 2 * OVERLAP
n_th     = int(np.ceil(H / TILE_SIZE))
n_tw     = int(np.ceil(W / TILE_SIZE))
n_tot    = n_th * n_tw
print(f"  Tiles      : {n_th} x {n_tw} = {n_tot} (inner={TILE_SIZE}, overlap={OVERLAP})")
print(f"  Ring       : {ring_px} px (= {RING_WIDTH_M} m)")

out_raw    = np.zeros((H, W), dtype=np.uint32)
offset_raw = np.uint32(0)
n_done = 0;  n_empty = 0;  t_pred = time.time()

for ti in range(n_th):
    for tj in range(n_tw):
        cr0 = ti * TILE_SIZE;   cr1 = min(H, cr0 + TILE_SIZE)
        cc0 = tj * TILE_SIZE;   cc1 = min(W, cc0 + TILE_SIZE)
        pr0 = max(0, cr0 - OVERLAP);  pr1 = min(H, cr1 + OVERLAP)
        pc0 = max(0, cc0 - OVERLAP);  pc1 = min(W, cc1 + OVERLAP)
        or0 = cr0 - pr0;  or1 = or0 + (cr1 - cr0)
        oc0 = cc0 - pc0;  oc1 = oc0 + (cc1 - cc0)

        tile = filled[pr0:pr1, pc0:pc1]
        tile_nan = nan_mask[pr0:pr1, pc0:pc1]
        actual_h, actual_w = tile.shape

        if tile[~tile_nan].size == 0:
            n_empty += 1;  n_done += 1;  continue

        ph = pad_max - actual_h;  pw = pad_max - actual_w
        tile_pad = np.pad(tile, ((0, ph), (0, pw)), mode='symmetric')

        try:
            pred = np.array(segmenter.predict(image=tile_pad),
                            dtype=np.uint32)[:actual_h, :actual_w]
        except Exception as e:
            print(f"  !! predict failed ({ti},{tj}): {e}", flush=True)
            pred = np.zeros((actual_h, actual_w), dtype=np.uint32)

        pred[tile_nan] = 0
        crop = pred[or0:or1, oc0:oc1]
        max_val = int(crop.max())
        if max_val > 0:
            crop[crop > 0] += offset_raw
            offset_raw += np.uint32(max_val)
        out_raw[cr0:cr1, cc0:cc1] = crop
        n_done += 1

        if n_done % 10 == 0 or n_done == n_tot:
            elapsed = time.time() - t_pred
            rate    = n_done / elapsed if elapsed > 0 else 1
            print(f"  [{n_done:3d}/{n_tot}] {n_done/n_tot*100:5.1f}%  "
                  f"elapsed={elapsed:.0f}s  segs={int(offset_raw):,}", flush=True)

print(f"  Done: {int(offset_raw):,} raw segments  empty_tiles={n_empty}  "
      f"[{time.time()-t_pred:.1f}s]")

# Save prediction (raw model output — no post-processing filter)
pred_profile = upscaled_profile.copy()
pred_profile.update(dtype='uint32', nodata=None)
with rasterio.open(pred_path, 'w', **pred_profile) as dst:
    dst.write(out_raw, 1)
print(f"  Saved prediction: {pred_path}")


# ── Boulder statistics ────────────────────────────────────────────────────────
print("\nComputing boulder statistics ...")
t_s       = time.time()
pixel_area = pixel_sx_up * pixel_sy_up
valid_b    = np.isfinite(upscaled)
struct     = np.ones((2 * ring_px + 1, 2 * ring_px + 1), dtype=bool)
labels     = out_raw.astype(np.int32)
props2     = regionprops(labels, intensity_image=upscaled)

# Spatial origin from new_transform (top-left corner)
origin_x = new_transform.c      # easting of left edge
origin_y = new_transform.f      # northing of top edge

n_no_ring = 0;  rows_out = []
for p in props2:
    area_m2    = p.area * pixel_area
    equiv_diam = np.sqrt(4.0 * area_m2 / np.pi)
    maj = p.major_axis_length;  mn = p.minor_axis_length
    lw  = (maj / mn) if mn > 0 else np.nan
    easting  = origin_x + p.centroid[1] * pixel_sx_up
    northing = origin_y - p.centroid[0] * pixel_sy_up
    row_c = int(round(p.centroid[0]))
    col_c = int(round(p.centroid[1]))

    r0, c0, r1, c1 = p.bbox
    pad = ring_px + 1
    pr0 = max(0, r0 - pad);  pr1 = min(H, r1 + pad)
    pc0 = max(0, c0 - pad);  pc1 = min(W, c1 + pad)

    lbl_patch   = labels[pr0:pr1, pc0:pc1]
    bathy_patch = upscaled[pr0:pr1, pc0:pc1]
    valid_patch = valid_b[pr0:pr1, pc0:pc1]

    boulder_mask = (lbl_patch == p.label)
    dilated      = binary_dilation(boulder_mask, structure=struct)
    ring_mask    = (dilated & ~boulder_mask & (lbl_patch == 0) & valid_patch)

    b_depths    = bathy_patch[boulder_mask & valid_patch]
    boulder_top = float(b_depths.max()) if len(b_depths) > 0 else np.nan

    ring_depths = bathy_patch[ring_mask]
    if len(ring_depths) == 0:
        height_m = np.nan;  n_no_ring += 1
    else:
        seafloor_ref = float(np.percentile(ring_depths, SEAFLOOR_PERCENTILE))
        height_m = (boulder_top - seafloor_ref) if not np.isnan(boulder_top) else np.nan

    rows_out.append((p.label, p.area, area_m2, equiv_diam, height_m,
                     len(ring_depths), lw, easting, northing, row_c, col_c))

print(f"  Done: {len(rows_out):,} boulders  no_ring={n_no_ring:,}  [{time.time()-t_s:.1f}s]")

with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'area_px', 'area_m2', 'equiv_diam_m', 'height_m',
                     'ring_px_count', 'lw_ratio', 'easting_m', 'northing_m',
                     'pixel_row', 'pixel_col'])
    for r in rows_out:
        lbl, apx, am2, ed, hm, rpc, lw, ex, ny, rc, cc = r
        writer.writerow([
            lbl, apx,
            f"{am2:.4f}", f"{ed:.4f}",
            f"{hm:.4f}" if not (hm is None or np.isnan(hm)) else "",
            rpc,
            f"{lw:.4f}" if not (lw is None or np.isnan(lw)) else "",
            f"{ex:.3f}", f"{ny:.3f}",
            rc, cc,
        ])
print(f"  CSV saved: {stats_csv_path}")


# ── Summary ───────────────────────────────────────────────────────────────────
heights = np.array([r[4] for r in rows_out if r[4] is not None and not np.isnan(r[4])])
areas   = np.array([r[2] for r in rows_out])
diams   = np.array([r[3] for r in rows_out])

print(f"\n{'='*70}")
print(f"DONE — {stem}")
print(f"{'='*70}")
print(f"  Boulders   : {len(rows_out):,}")
if len(areas) > 0:
    print(f"  Area (m²)  : min={areas.min():.4f}  median={np.median(areas):.4f}  "
          f"max={areas.max():.4f}")
    print(f"  Diam (m)   : min={diams.min():.3f}  median={np.median(diams):.3f}  "
          f"max={diams.max():.3f}")
if len(heights) > 0:
    print(f"  Height (m) : min={heights.min():.3f}  median={np.median(heights):.3f}  "
          f"max={heights.max():.3f}")
    print(f"               p5={np.percentile(heights,5):.3f}  "
          f"p25={np.percentile(heights,25):.3f}  "
          f"p75={np.percentile(heights,75):.3f}  "
          f"p95={np.percentile(heights,95):.3f}")
    neg = (heights < 0).sum()
    print(f"               negative: {neg:,} ({neg/len(heights)*100:.1f}%)")
print(f"\nOutputs:")
print(f"  {upscaled_path}")
print(f"  {pred_path}")
print(f"  {stats_csv_path}")
