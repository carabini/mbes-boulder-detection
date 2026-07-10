"""
eval_apoc.py
------------
Standalone object-level evaluation of saved APOC predictions — the counterpart to
nnunet/eval_objects.py, so both methods are scored by the *same* published module
(../eval_common.py) and the paper numbers can be reproduced without retraining.

The APOC training script (train_upscaled_sparse_..._v1.py) already computes these
same metrics on the fly while training; this script recomputes them from the saved
prediction rasters, which is handy for reproducing the paper table from the
archived predictions.

Method (identical to the nnU-Net evaluation): 8-connectivity, greedy 1:1 IoU
matching, a detection is a true positive at IoU >= 0.1, Precision/Recall/F1 pooled
over all test tiles. See ../eval_common.py.

Run with a Python environment matching requirements.txt:
  python eval_apoc.py --pred-dir <predictions> --anno-dir <test annotations>

  --pred-dir  Folder of APOC prediction GeoTIFFs (instance labels; 0 = background)
  --anno-dir  Folder of test annotation GeoTIFFs (float; 1 = boulder, 2 = background)
  --iou-threshold  IoU threshold for a true positive (default: 0.1)
  --out-csv   Optional path to write the per-tile table
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage.io import imread

# Shared object-level evaluation (repository root) — the same module the APOC
# training script and the nnU-Net evaluation use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_common


def read_pred(path):
    """Read an APOC prediction raster as int32 instance labels (nodata -> 0)."""
    with rasterio.open(str(path)) as src:
        arr = src.read(1)
    arr = np.asarray(arr).astype(np.int32)
    arr[arr < 0] = 0
    return arr


def main(pred_dir, anno_dir, iou_threshold, out_csv):
    pred_dir = Path(pred_dir)
    anno_dir = Path(anno_dir)

    print(f"\nAPOC object-level evaluation  |  IoU threshold: {iou_threshold}")
    print(f"  {eval_common.method_note(iou_threshold)}")
    print(f"Predictions:  {pred_dir}")
    print(f"Annotations:  {anno_dir}\n")
    print(f"{'File':<45} {'GT':>4} {'pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}  "
          f"{'Recall':>7} {'Prec':>7} {'F1':>7} {'objJ':>7}")
    print("-" * 100)

    tiles, records = [], []
    for apath in sorted(anno_dir.glob("*.tif")):
        ppath = pred_dir / apath.name
        if not ppath.exists():
            print(f"  WARNING: no prediction for {apath.name}, skipping")
            continue

        annotation = imread(str(apath))
        pred = read_pred(ppath)

        m = eval_common.evaluate_tile(annotation == 1, pred, threshold=iou_threshold)
        tiles.append(m)

        tp, fp, fn = m['tp'], m['fp'], m['fn']
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        f1 = (2 * precision * recall / (precision + recall)
              if (precision == precision and recall == recall and (precision + recall) > 0)
              else float('nan'))
        records.append({'filename': apath.name, 'jaccard': m['jaccard'],
                        'precision': precision, 'recall': recall, 'f1': f1,
                        'n_gt': m['n_gt'], 'n_pred': m['n_pred'],
                        'tp': tp, 'fp': fp, 'fn': fn})

        print(f"{apath.name:<45} {m['n_gt']:>4} {m['n_pred']:>5} {tp:>4} {fp:>4} {fn:>4}  "
              f"{recall:>7.3f} {precision:>7.3f} {f1:>7.3f} {m['jaccard']:>7.3f}")

    if not tiles:
        print("No results.")
        return

    pooled = eval_common.pool_metrics(tiles)
    df = pd.DataFrame(records)

    print("-" * 100)
    print(f"{'TOTAL/POOLED':<45} {pooled['n_gt']:>4} {pooled['n_pred']:>5} "
          f"{pooled['tp']:>4} {pooled['fp']:>4} {pooled['fn']:>4}  "
          f"{pooled['recall']:>7.3f} {pooled['precision']:>7.3f} {pooled['f1']:>7.3f} {pooled['jaccard']:>7.3f}")

    print(f"\nSummary (pooled over {len(tiles)} tiles):")
    print(f"  Recall={pooled['recall']:.3f}  Precision={pooled['precision']:.3f}  "
          f"F1={pooled['f1']:.3f}  IoU/object-Jaccard={pooled['jaccard']:.3f}")
    print(f"  (IoU column = mean best-IoU of each GT boulder with its matched detection, "
          f"pooled over all boulders)")

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"\nPer-tile CSV: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object-level evaluation of saved APOC predictions")
    parser.add_argument("--pred-dir", required=True, help="Folder of APOC prediction GeoTIFFs")
    parser.add_argument("--anno-dir", required=True, help="Folder of test annotation GeoTIFFs")
    parser.add_argument("--iou-threshold", default=0.1, type=float, help="IoU threshold (default: 0.1)")
    parser.add_argument("--out-csv", default=None, help="Optional per-tile CSV output path")
    args = parser.parse_args()
    main(args.pred_dir, args.anno_dir, args.iou_threshold, args.out_csv)
