"""
make_sparse_annotations_bg10to1_25pct.py
-----------------------------------------
Sparse annotation: keeps a fraction of boulder segments and adds a background
ring, to simulate sparse (partial) annotation for APOC training.

Run with a Python environment matching requirements.txt:
  python make_sparse_annotations_bg10to1_25pct.py [options]

Options (all optional):
  --input-dir     Folder with fully-annotated GeoTIFFs (default: data/upscaled/train/annotations)
  --output-dir    Folder to write sparse annotation GeoTIFFs to
                  (default: data/sparse/annotations)
  --keep-fraction Fraction of boulder segments to keep (default: 0.25)
  --bg-ratio      Target bg:boulder pixel ratio (default: 10.0)
  --ring-pixels   Radius (px) of background ring around kept boulders (default: 6)
  --random-seed   Random seed for segment selection and background sampling (default: 42)
"""

import argparse
import rasterio
import numpy as np
from pathlib import Path
from scipy import ndimage
import random

parser = argparse.ArgumentParser(description="Create sparse annotations (25% segments, 10:1 bg:boulder)")
parser.add_argument("--input-dir",     default="data/upscaled/train/annotations",
                    help="Folder with fully-annotated GeoTIFFs")
parser.add_argument("--output-dir",    default="data/sparse/annotations",
                    help="Folder to write sparse annotation GeoTIFFs to")
parser.add_argument("--keep-fraction", default=0.25, type=float, help="Fraction of boulder segments to keep")
parser.add_argument("--bg-ratio",      default=10.0, type=float, help="Target bg:boulder pixel ratio")
parser.add_argument("--ring-pixels",   default=6, type=int, help="Radius (px) of background ring")
parser.add_argument("--random-seed",   default=42, type=int)
args = parser.parse_args()

RANDOM_SEED   = args.random_seed
KEEP_FRACTION = args.keep_fraction
BG_RATIO      = args.bg_ratio
RING_PIXELS   = args.ring_pixels
INPUT_DIR     = Path(args.input_dir)
OUTPUT_DIR    = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng_sel = random.Random(RANDOM_SEED)
rng_bg  = np.random.default_rng(RANDOM_SEED)

struct8 = np.ones((3, 3), dtype=bool)
y, x = np.ogrid[-RING_PIXELS:RING_PIXELS+1, -RING_PIXELS:RING_PIXELS+1]
disk  = (x**2 + y**2 <= RING_PIXELS**2)

anno_files = sorted(INPUT_DIR.glob("*.tif"))
print(f"Processing {len(anno_files)} files  |  keep={KEEP_FRACTION*100:.0f}%  bg:boulder={BG_RATIO:.1f}:1\n")

stats = {"files": 0, "segs_total": 0, "segs_kept": 0, "px_boulder": 0, "px_bg": 0}

for f in anno_files:
    with rasterio.open(f) as src:
        data    = src.read(1).astype(np.float32)
        profile = src.profile

    boulder_mask = (data == 1)
    bg_mask      = (data == 2)
    labeled, n_segs = ndimage.label(boulder_mask, structure=struct8)

    if n_segs == 0:
        out = np.zeros_like(data, dtype=np.uint8)
    else:
        n_keep   = max(1, round(n_segs * KEEP_FRACTION))
        kept_ids = set(rng_sel.sample(range(1, n_segs + 1), n_keep))
        kept_boulder = np.zeros_like(boulder_mask)
        for sid in kept_ids:
            kept_boulder |= (labeled == sid)

        boulder_px = int(kept_boulder.sum())
        target_bg  = int(round(boulder_px * BG_RATIO))

        dilated   = ndimage.binary_dilation(kept_boulder, structure=disk)
        ring_mask = dilated & bg_mask & ~kept_boulder
        ring_idx  = np.argwhere(ring_mask)
        extra_idx = np.argwhere(bg_mask & ~dilated)

        if len(ring_idx) >= target_bg:
            chosen      = rng_bg.choice(len(ring_idx), size=target_bg, replace=False)
            bg_selected = ring_idx[chosen]
        else:
            n_extra = target_bg - len(ring_idx)
            if len(extra_idx) >= n_extra:
                chosen_extra = rng_bg.choice(len(extra_idx), size=n_extra, replace=False)
                bg_selected  = np.vstack([ring_idx, extra_idx[chosen_extra]])
            else:
                bg_selected = np.vstack([ring_idx, extra_idx]) if len(extra_idx) else ring_idx

        out = np.zeros(data.shape, dtype=np.uint8)
        out[kept_boulder] = 1
        out[bg_selected[:, 0], bg_selected[:, 1]] = 2

        actual_bg    = int((out == 2).sum())
        actual_ratio = actual_bg / boulder_px if boulder_px > 0 else 0
        stats["segs_total"] += n_segs
        stats["segs_kept"]  += n_keep
        stats["px_boulder"] += boulder_px
        stats["px_bg"]      += actual_bg
        print(f"  {f.name}: {n_segs}->{n_keep} segs | boulder={boulder_px}px  bg={actual_bg}px  ratio={actual_ratio:.1f}:1")

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1, nodata=None)
    with rasterio.open(OUTPUT_DIR / f.name, "w", **out_profile) as dst:
        dst.write(out[np.newaxis, :, :])
    stats["files"] += 1

total_b  = stats["px_boulder"]
total_bg = stats["px_bg"]
print(f"\n=== DONE ===")
print(f"Files: {stats['files']}  |  Segs kept: {stats['segs_kept']}/{stats['segs_total']} ({100*stats['segs_kept']/max(1,stats['segs_total']):.1f}%)")
print(f"Boulder px: {total_b:,}  |  BG px: {total_bg:,}  |  Overall ratio: {total_bg/max(1,total_b):.2f}:1")
print(f"Output: {OUTPUT_DIR}")
