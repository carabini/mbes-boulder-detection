"""
predict_folder.py
-----------------
Full inference pipeline: GeoTIFF bathymetry -> georeferenced boulder mask.

Usage:
  python predict_folder.py --input <folder> --output <folder> --results <nnunet_results_folder>

  Activate the nnunet conda environment first, then run:

  python predict_folder.py \
      --input  ./bathy \
      --output ./output \
      --results ./nnunet_weights/results

  --fold      Fold indices to use (default: 0 1 2 3 4, full ensemble)
  --results   Path to nnU-Net results folder (downloaded from the model deposit)
  --raw       Path to nnU-Net raw folder (optional, default: <results>/../raw)
  --preprocessed  Path to nnU-Net preprocessed folder (optional, default: <results>/../preprocessed)
  --tile-size Core tile size (px, upscaled) before splitting large images (default: 4000)
  --overlap   Overlap border (px, upscaled) read around each core tile (default: 128,
              must be < half the tile size). Only the core region is written back,
              so boulders straddling tile borders are not cut.

NoData handling:
  NaN/NoData pixels are filled at the original resolution with a normalised
  Gaussian convolution (sigma=5 px): the blurred data are divided by the blurred
  validity mask, so every filled pixel is a local weighted mean of the valid
  depths around it. Pixels with no valid data in reach get the mean of the valid
  data. The filled raster is then upscaled. This keeps the model input within
  the real depth range, which matters because nnU-Net z-score-normalises each
  image over all pixels, including the filled ones. After stitching, the mask is
  forced to 0 in NoData areas, so no boulders are reported there. (The training
  tiles contained no NoData, where the fill is a no-op.)

Steps per file:
  1. Read GeoTIFF, save georef metadata
  2. Fill NoData locally (original resolution)
  3. 2x bilinear upscale (matches training data format)
  4. Write upscaled bathy GeoTIFF with NoData restored  (for QGIS overlay)
  5. Tile with overlap, convert to temp NIfTI
  6. Run nnUNetv2_predict (sliding window, all folds by default)
  7. Read prediction NIfTI, write only each tile's core region
  8. Mask out NoData areas, write binary mask GeoTIFF  {0=background, 1=boulder}
  9. Write summary.csv, clean up temp files

Output structure:
  output/
    masks/            <name>_mask.tif        uint8, 0=background, 1=boulder, nodata tag 255
    bathy_upscaled/   <name>_upscaled.tif
    summary.csv       boulder pixels per file (count and % of the valid area)
"""

import os
import sys
import csv
import shutil
import argparse
import tempfile
import subprocess
import json
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import zoom, gaussian_filter
import nibabel as nib
from pathlib import Path

# ── nnU-Net settings ──────────────────────────────────────────────────────────
DATASET_ID = "001"   # overridden by --dataset argument
CONFIG     = "2d"

# ── helpers ───────────────────────────────────────────────────────────────────

def read_geotiff(path):
    with rasterio.open(path) as src:
        arr     = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata  = src.nodata
        transform = src.transform
        crs     = src.crs
        width, height = src.width, src.height
        pixel_size = abs(transform.a)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, profile, pixel_size, transform, crs, width, height


def fill_nodata(arr, sigma=5):
    """Fill NaN/NoData by normalised Gaussian convolution.

    The blurred data (NoData set to 0) are divided by the blurred validity mask,
    so the fill value is a local weighted mean of the valid neighbours and is
    not pulled towards 0 as the local NoData fraction grows. Pixels with no
    valid data in reach get the mean of the valid data. Valid pixels are
    returned unchanged."""
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return arr.astype(np.float32)
    if nan_mask.all():
        return np.zeros_like(arr, dtype=np.float32)
    num = gaussian_filter(np.where(nan_mask, 0.0, arr).astype(np.float64), sigma=sigma)
    den = gaussian_filter((~nan_mask).astype(np.float64), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = np.where(den > 1e-8, num / den, np.nan)
    filled = np.where(np.isfinite(filled), filled, np.nanmean(arr))
    return np.where(nan_mask, filled, arr).astype(np.float32)


def upscale_bathy(arr_filled, scale=2):
    """2x bilinear upscale of a gap-free array (fill NoData first)."""
    return zoom(arr_filled, scale, order=1).astype(np.float32)


def upscale_nodata_mask(nan_mask, scale=2):
    """Nearest-neighbour upscale of the NoData mask."""
    return zoom(nan_mask.astype(np.float32), scale, order=0).astype(bool)


def upscaled_transform(src_transform, src_w, src_h, scale):
    bounds = rasterio.transform.array_bounds(src_h, src_w, src_transform)
    new_w, new_h = src_w * scale, src_h * scale
    return from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], new_w, new_h), new_w, new_h


def save_bathy_geotiff(arr, transform, crs, out_path):
    """Save upscaled bathymetry as float32 GeoTIFF."""
    profile = {
        "driver": "GTiff",
        "dtype":  "float32",
        "width":  arr.shape[1],
        "height": arr.shape[0],
        "count":  1,
        "crs":    crs,
        "transform": transform,
        "nodata": None,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


def save_mask_geotiff(arr, transform, crs, out_path):
    """Save binary mask as uint8 GeoTIFF {0=background, 1=boulder}."""
    profile = {
        "driver": "GTiff",
        "dtype":  "uint8",
        "width":  arr.shape[1],
        "height": arr.shape[0],
        "count":  1,
        "crs":    crs,
        "transform": transform,
        "nodata": 255,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


def write_nifti(arr, pixel_size, out_path):
    """Write float32 array as NIfTI for nnU-Net."""
    arr_3d = arr[:, :, np.newaxis]
    affine = np.diag([pixel_size, pixel_size, 1.0, 1.0])
    img = nib.Nifti1Image(arr_3d, affine)
    img.header.set_zooms((pixel_size, pixel_size, 1.0))
    nib.save(img, str(out_path))


def read_nifti(path):
    """Read NIfTI prediction, return 2D uint8 array."""
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def run_nnunet_predict(input_dir, output_dir, folds, dataset_id=None,
                       nnunet_results=None, nnunet_raw=None, nnunet_preprocessed=None):
    """Call nnUNetv2_predict as subprocess.

    folds: list of ints, e.g. [0, 1, 2, 3, 4] for full ensemble.
    Requires nnUNetv2_predict to be available on PATH (activate the conda environment first).
    """
    predict_exe = shutil.which("nnUNetv2_predict")
    if predict_exe is None:
        raise RuntimeError(
            "nnUNetv2_predict not found on PATH.\n"
            "Make sure the nnunet conda environment is activated:\n"
            "  conda activate nnunet-boulder"
        )
    env = os.environ.copy()
    if nnunet_results:
        env["nnUNet_results"] = str(nnunet_results)
    if nnunet_raw:
        env["nnUNet_raw"] = str(nnunet_raw)
    if nnunet_preprocessed:
        env["nnUNet_preprocessed"] = str(nnunet_preprocessed)

    cmd = [
        predict_exe,
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", dataset_id or DATASET_ID,
        "-c", CONFIG,
        "-f", *[str(f) for f in folds],
        "-nps", "1",   # single export process — avoids Windows pipe overflow on large images
    ]
    fold_str = " ".join(str(f) for f in folds)
    print(f"\nRunning nnUNetv2_predict (folds: {fold_str})...")
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"nnUNetv2_predict failed with exit code {result.returncode}")


# ── tiling helpers ────────────────────────────────────────────────────────────

DEFAULT_TILE_PX = 4000  # default core tile dimension (upscaled pixels) before splitting
DEFAULT_OVERLAP = 128   # default overlap border (upscaled pixels)

def compute_tiles(h, w, tile_size=DEFAULT_TILE_PX):
    """Return list of (r0, r1, c0, c1) non-overlapping *core* tile coordinates
    that together cover the whole image."""
    tiles = []
    r = 0
    while r < h:
        r1 = min(r + tile_size, h)
        c = 0
        while c < w:
            c1 = min(c + tile_size, w)
            tiles.append((r, r1, c, c1))
            c = c1
        r = r1
    return tiles


def expand_tile(r0, r1, c0, c1, h, w, overlap):
    """Expand a core tile by `overlap` px on all sides, clamped to the image."""
    er0 = max(0, r0 - overlap)
    er1 = min(h, r1 + overlap)
    ec0 = max(0, c0 - overlap)
    ec1 = min(w, c1 + overlap)
    return er0, er1, ec0, ec1


def tile_transform(full_transform, r0, c0, pixel_size):
    """Compute affine transform for a tile whose top-left pixel is (r0, c0)."""
    x0 = full_transform.c + c0 * pixel_size
    y0 = full_transform.f - r0 * pixel_size
    return rasterio.transform.from_origin(x0, y0, pixel_size, pixel_size)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(input_dir, output_dir, folds=None, scale=2, dataset_id=None,
                 nnunet_results=None, nnunet_raw=None, nnunet_preprocessed=None,
                 tile_size=DEFAULT_TILE_PX, overlap=DEFAULT_OVERLAP):
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    if overlap >= tile_size / 2:
        raise ValueError(f"--overlap ({overlap}) must be < half the tile size ({tile_size}).")
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    masks_dir  = output_dir / "masks"
    bathy_dir  = output_dir / "bathy_upscaled"
    masks_dir.mkdir(parents=True, exist_ok=True)
    bathy_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    tif_files = []
    for p in sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.TIF")):
        if p.name.lower() not in seen:
            seen.add(p.name.lower())
            tif_files.append(p)
    if not tif_files:
        print(f"No .tif files found in {input_dir}")
        return

    print(f"Found {len(tif_files)} file(s) in {input_dir}")

    tmp = Path(tempfile.mkdtemp(prefix="nnunet_predict_"))
    tmp_images = tmp / "images"
    tmp_preds  = tmp / "predictions"
    tmp_images.mkdir()
    tmp_preds.mkdir()

    ds_json = {"channel_names": {"0": "bathymetry"}, "labels": {"background": 0, "boulder": 1}, "file_ending": ".nii.gz"}
    with open(tmp / "dataset.json", "w") as f:
        json.dump(ds_json, f)

    # case_meta maps case_id -> tile info needed for stitching
    case_meta  = {}
    # file_meta maps stem -> (full_shape, full_transform, crs, nan_mask) for stitching
    file_meta  = {}
    case_counter = 0

    print("\n--- Step 1-5: Fill + upscale + write bathy GeoTIFFs + convert to NIfTI ---")
    for tif_path in tif_files:
        stem = tif_path.stem
        print(f"  {tif_path.name}")

        arr, profile, pixel_size, transform, crs, w, h = read_geotiff(tif_path)

        # Fill NoData at the original resolution first, then upscale. Upscaling
        # first would interpolate across every data/NoData edge and mix the
        # valid border pixels with the fill value.
        nan_native    = ~np.isfinite(arr)
        up_arr_filled = upscale_bathy(fill_nodata(arr), scale)   # model input
        up_nan        = upscale_nodata_mask(nan_native, scale)
        up_transform, uw, uh = upscaled_transform(transform, w, h, scale)
        up_pixel_size        = pixel_size / scale

        # Save upscaled bathy GeoTIFF (keeps NaN in the NoData areas)
        up_arr = up_arr_filled.copy()
        up_arr[up_nan] = np.nan
        bathy_out = bathy_dir / f"{stem}_upscaled.tif"
        save_bathy_geotiff(up_arr, up_transform, crs, bathy_out)
        print(f"    Bathy saved: {bathy_out.name}  ({uw}x{uh} px, {up_pixel_size:.4f} m/px, "
              f"NoData {100 * nan_native.mean():.1f}%)")
        del up_arr

        tiles = compute_tiles(uh, uw, tile_size=tile_size)
        n_tiles = len(tiles)
        if n_tiles > 1:
            print(f"    Image too large — splitting into {n_tiles} core tiles "
                  f"(tile {tile_size} px, overlap {overlap} px)")

        file_meta[stem] = {
            "shape":     (uh, uw),
            "transform": up_transform,
            "crs":       crs,
            "nan_mask":  up_nan,
        }

        for (r0, r1, c0, c1) in tiles:
            case_counter += 1
            case_id = f"CASE_{case_counter:03d}"

            # Read an expanded window so border boulders are seen with context.
            er0, er1, ec0, ec1 = expand_tile(r0, r1, c0, c1, uh, uw, overlap)
            tile_arr = up_arr_filled[er0:er1, ec0:ec1]
            # NIfTI transform anchored at the expanded window origin.
            t_transform = tile_transform(up_transform, er0, ec0, up_pixel_size)
            write_nifti(tile_arr, up_pixel_size, tmp_images / f"{case_id}_0000.nii.gz")

            case_meta[case_id] = {
                "stem": stem,
                "r0": r0, "r1": r1, "c0": c0, "c1": c1,     # core region (destination)
                "er0": er0, "ec0": ec0,                     # expanded window origin
            }
            if n_tiles > 1:
                print(f"      [{case_id}] core rows {r0}-{r1}, cols {c0}-{c1}  "
                      f"(read {er0}-{er1}, {ec0}-{ec1})")

    # Run nnU-Net predict on all tiles at once
    run_nnunet_predict(tmp_images, tmp_preds, folds, dataset_id=dataset_id,
                       nnunet_results=nnunet_results, nnunet_raw=nnunet_raw,
                       nnunet_preprocessed=nnunet_preprocessed)

    print("\n--- Step 4: Stitch tiles + convert to GeoTIFF masks ---")

    # Collect tile predictions per file and stitch (core regions only)
    summary = []
    for stem, fmeta in file_meta.items():
        full_h, full_w = fmeta["shape"]
        full_mask = np.zeros((full_h, full_w), dtype=np.uint8)

        for case_id, cmeta in case_meta.items():
            if cmeta["stem"] != stem:
                continue
            pred_nifti = tmp_preds / f"{case_id}.nii.gz"
            if not pred_nifti.exists():
                print(f"  WARNING: prediction not found for {case_id}, skipping")
                continue
            pred = read_nifti(pred_nifti)   # prediction over the expanded window
            # Extract the core region from the expanded prediction and write it back.
            r0, r1, c0, c1 = cmeta["r0"], cmeta["r1"], cmeta["c0"], cmeta["c1"]
            lr0 = r0 - cmeta["er0"];  lr1 = lr0 + (r1 - r0)
            lc0 = c0 - cmeta["ec0"];  lc1 = lc0 + (c1 - c0)
            full_mask[r0:r1, c0:c1] = pred[lr0:lr1, lc0:lc1]

        # No boulders in NoData areas.
        full_mask[fmeta["nan_mask"]] = 0

        mask_out = masks_dir / f"{stem}_mask.tif"
        save_mask_geotiff(full_mask, fmeta["transform"], fmeta["crs"], mask_out)

        n_boulders_px = int((full_mask == 1).sum())
        n_valid_px    = int((~fmeta["nan_mask"]).sum())
        pct = 100 * n_boulders_px / max(n_valid_px, 1)
        summary.append((stem, n_boulders_px, pct))
        print(f"  {mask_out.name}  —  boulder px: {n_boulders_px} ({pct:.2f}% of the valid area)")

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "boulder_px", "pct_of_valid_area"])
        for stem, n_px, pct in summary:
            writer.writerow([stem, n_px, f"{pct:.4f}"])

    # Cleanup
    shutil.rmtree(tmp)
    print(f"\nTemp files cleaned up.")
    print(f"\nDone! Outputs in: {output_dir}")
    print(f"  Masks:          {masks_dir}")
    print(f"  Bathy upscaled: {bathy_dir}")
    print(f"  Summary:        {summary_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boulder detection inference pipeline")
    parser.add_argument("--input",   required=True,  help="Input folder with GeoTIFF files")
    parser.add_argument("--output",  required=True,  help="Output folder")
    parser.add_argument("--results", required=True,
                        help="Path to nnU-Net results folder (downloaded from the model deposit)")
    parser.add_argument("--raw",          default=None,
                        help="Path to nnU-Net raw folder (default: <results>/../raw)")
    parser.add_argument("--preprocessed", default=None,
                        help="Path to nnU-Net preprocessed folder (default: <results>/../preprocessed)")
    parser.add_argument("--fold",    default=[0, 1, 2, 3, 4], type=int, nargs="+",
                        help="nnU-Net fold(s) to use for ensemble (default: 0 1 2 3 4)")
    parser.add_argument("--scale",   default=2, type=int,   help="Upscaling factor (default: 2)")
    parser.add_argument("--dataset", default=None,
                        help="nnU-Net dataset ID, e.g. 001 (default: 001)")
    parser.add_argument("--tile-size", default=DEFAULT_TILE_PX, type=int,
                        help=f"Core tile size (px, upscaled) before splitting (default: {DEFAULT_TILE_PX})")
    parser.add_argument("--overlap",   default=DEFAULT_OVERLAP, type=int,
                        help=f"Overlap border (px, upscaled) read around each core tile, "
                             f"must be < half the tile size (default: {DEFAULT_OVERLAP})")
    args = parser.parse_args()

    results_path = Path(args.results)
    raw_path          = Path(args.raw)          if args.raw          else results_path.parent / "raw"
    preprocessed_path = Path(args.preprocessed) if args.preprocessed else results_path.parent / "preprocessed"

    run_pipeline(args.input, args.output, folds=args.fold, scale=args.scale, dataset_id=args.dataset,
                 nnunet_results=results_path, nnunet_raw=raw_path, nnunet_preprocessed=preprocessed_path,
                 tile_size=args.tile_size, overlap=args.overlap)
