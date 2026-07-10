"""
upscaled_sparse_bg10to1_25pct_sigma2346_v1 — APOC on upscaled tiles, 25% sparse annotation.

Training data (118 tiles, default paths — override via --train-bathy-dir/--train-anno-dir):
  Bathy:   data/upscaled/train/bathy/*.tif
  Masks:   data/sparse/annotations/*.tif
           (25% of boulder segments kept per tile, 6px BG ring, bg:boulder=10:1;
            produced by make_sparse_annotations_bg10to1_25pct.py)

Evaluation (37 tiles, fully annotated, default paths — override via --test-bathy-dir/--test-anno-dir):
  Bathy:   data/upscaled/test/bathy/*.tif
  Masks:   data/upscaled/test/annotations/*.tif  (float32, {1.0=boulder, 2.0=bg})
  Predictions saved to (default, override via --pred-dir): data/predictions_sparse_bg10to1_25pct_sigma2346_v1/

Metrics (shared eval_common.py — same method as the nnU-Net evaluation):
  object-level IoU, greedy 1:1 matching at IoU >= 0.1, 8-connectivity,
  Precision/Recall/F1 pooled over all test tiles.

Run with a Python environment matching requirements.txt:
  python train_upscaled_sparse_bg10to1_25pct_sigma2346_v1.py [options]

Options (all optional):
  --train-bathy-dir  (default: data/upscaled/train/bathy)
  --train-anno-dir   (default: data/sparse/annotations)
  --test-bathy-dir   (default: data/upscaled/test/bathy)
  --test-anno-dir    (default: data/upscaled/test/annotations)
  --classifier       Output path for the trained .cl classifier
                      (default: data/apoc_classifier_upscaled_sparse_bg10to1_25pct_sigma2346_v1.cl)
  --reports-dir      (default: data/reports)
  --pred-dir         (default: data/predictions_sparse_bg10to1_25pct_sigma2346_v1)
  --max-depth        RandomForest max depth (default: 4)
  --n-estimators     RandomForest number of trees (default: 150)
  --iou-threshold    IoU threshold for Precision/Recall/F1 (default: 0.1)
"""
import sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import apoc
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier

# Shared object-level evaluation (repository root) — keeps APOC and nnU-Net
# numbers directly comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_common

# -- Paths ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train APOC on upscaled sparse annotations (25%, bg:boulder=10:1)")
parser.add_argument("--train-bathy-dir", default="data/upscaled/train/bathy")
parser.add_argument("--train-anno-dir",  default="data/sparse/annotations")
parser.add_argument("--test-bathy-dir",  default="data/upscaled/test/bathy")
parser.add_argument("--test-anno-dir",   default="data/upscaled/test/annotations")
parser.add_argument("--classifier",      default="data/apoc_classifier_upscaled_sparse_bg10to1_25pct_sigma2346_v1.cl",
                    help="Output path for the trained .cl classifier")
parser.add_argument("--reports-dir",     default="data/reports")
parser.add_argument("--pred-dir",        default="data/predictions_sparse_bg10to1_25pct_sigma2346_v1")
parser.add_argument("--max-depth",       default=4,   type=int)
parser.add_argument("--n-estimators",    default=150, type=int)
parser.add_argument("--iou-threshold",   default=0.1, type=float, help="IoU threshold for Precision/Recall/F1")
args = parser.parse_args()

train_bathy_folder = args.train_bathy_dir
train_anno_folder  = args.train_anno_dir
test_bathy_folder  = args.test_bathy_dir
test_anno_folder   = args.test_anno_dir

cl_filename  = args.classifier
run_name     = "upscaled_sparse_bg10to1_25pct_sigma2346_v1"
reports_dir  = Path(args.reports_dir)
pred_dir     = Path(args.pred_dir)
reports_dir.mkdir(parents=True, exist_ok=True)
pred_dir.mkdir(parents=True, exist_ok=True)

MAX_DEPTH     = args.max_depth
N_ESTIMATORS  = args.n_estimators
IOU_THRESHOLD = args.iou_threshold   # for Precision/Recall/F1

# -- Helpers -------------------------------------------------------------------
def read_bathy(path):
    arr = imread(str(path)).astype(np.float32)
    arr[arr < -1e38] = np.nan
    return arr

def fill_nodata(image):
    nodata_mask = ~np.isfinite(image)
    if not nodata_mask.any():
        return image
    filled = np.where(nodata_mask, 0.0, image)
    smooth = gaussian_filter(filled.astype(np.float64), sigma=5)
    return np.where(nodata_mask, smooth, image).astype(np.float32)

# Object-level evaluation lives in eval_common.py (8-connectivity labelling,
# greedy 1:1 IoU matching, pooled Precision/Recall/F1) — see per-tile loop below.

# -- Setup APOC ----------------------------------------------------------------
features = ('original '
            'gaussian_blur=2 gaussian_blur=3 gaussian_blur=4 gaussian_blur=6 '
            'difference_of_gaussian=2 difference_of_gaussian=3 difference_of_gaussian=4 difference_of_gaussian=6 '
            'laplace_box_of_gaussian_blur=2 laplace_box_of_gaussian_blur=3 '
            'laplace_box_of_gaussian_blur=4 laplace_box_of_gaussian_blur=6')

apoc.erase_classifier(cl_filename)
segmenter = apoc.ObjectSegmenter(
    opencl_filename=cl_filename,
    positive_class_identifier=1,
    num_ensembles=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
)

# -- Build training pairs ------------------------------------------------------
train_bathy_files = sorted(Path(train_bathy_folder).glob("*.tif"))
all_pairs = []
for bpath in train_bathy_files:
    mpath = Path(train_anno_folder) / bpath.name
    if mpath.exists():
        all_pairs.append((bpath, mpath))

print(f"Training pairs: {len(all_pairs)}  |  depth={MAX_DEPTH}  trees={N_ESTIMATORS}", flush=True)

# -- Collect annotated pixels --------------------------------------------------
all_X, all_y = [], []
segmenter.num_ground_truth_dimensions = 2

for i, (bpath, mpath) in enumerate(all_pairs):
    image = fill_nodata(read_bathy(bpath))
    mask  = np.round(imread(str(mpath))).astype(np.int32)

    feat_stack = segmenter._make_features_potentially_multichannel(features, image)
    segmenter.num_features = len(feat_stack)
    X_i, y_i = segmenter._to_np(feat_stack, mask)
    if len(y_i) > 0:
        all_X.append(X_i)
        all_y.append(y_i)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(all_pairs)}  {sum(len(y) for y in all_y):,} px", flush=True)

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y)
print(f"\nTotal annotated pixels: {len(y):,}  |  features: {X.shape[1]}", flush=True)

unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  class {int(cls)}: {cnt:,} px ({100*cnt/len(y):.1f}%)", flush=True)
bo_px = counts[unique == 1][0] if 1 in unique else 1
bg_px = counts[unique == 2][0] if 2 in unique else 0
print(f"  bg:boulder ratio = {bg_px/bo_px:.1f}:1", flush=True)

# -- Train RandomForest --------------------------------------------------------
print("\nFitting RandomForest ...", flush=True)
clf = RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
                              random_state=0, n_jobs=-1)
clf.fit(X, y)
print(f"  importances: {[round(f, 3) for f in clf.feature_importances_]}", flush=True)

segmenter.classifier = clf
segmenter._feature_importances = clf.feature_importances_
segmenter._X = X
segmenter._y = y
segmenter.positive_class_identifier_from_file = segmenter.positive_class_identifier
segmenter.to_opencl_file(cl_filename)
print(f"Saved classifier: {cl_filename}\n", flush=True)

# -- Evaluate on test files ----------------------------------------------------
print(f"Evaluating on test set (IoU threshold={IOU_THRESHOLD}) ...", flush=True)
print(f"  {eval_common.method_note(IOU_THRESHOLD)}", flush=True)
print(f"Saving predictions to: {pred_dir}\n", flush=True)

test_bathy_files = sorted(Path(test_bathy_folder).glob("*.tif"))
tiles   = []   # per-tile count dicts for pooled metrics
records = []   # per-tile rows (incl. per-tile P/R/F1) for the detailed CSV

for bpath in test_bathy_files:
    anno_path = Path(test_anno_folder) / bpath.name
    if not anno_path.exists():
        print(f"  SKIP {bpath.name}: no annotation", flush=True)
        continue

    bathy      = fill_nodata(read_bathy(bpath))
    pred       = np.array(segmenter.predict(image=bathy))
    annotation = imread(str(anno_path))

    m = eval_common.evaluate_tile(annotation == 1, pred, threshold=IOU_THRESHOLD)
    tiles.append(m)

    # Per-tile Precision/Recall/F1 (detailed table only — headline is pooled)
    tp, fp, fn = m['tp'], m['fp'], m['fn']
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and (precision + recall) > 0)
          else float('nan'))
    records.append({'filename': bpath.name, 'jaccard': m['jaccard'],
                    'precision': precision, 'recall': recall, 'f1': f1,
                    'n_gt': m['n_gt'], 'n_pred': m['n_pred'],
                    'tp': tp, 'fp': fp, 'fn': fn})

    print(f"  {bpath.name:<45s}  GT={m['n_gt']:3d}  pred={m['n_pred']:4d}  "
          f"Jacc={m['jaccard']:.3f}  P={precision:.3f}  "
          f"R={recall:.3f}  F1={f1:.3f}", flush=True)

    with rasterio.open(str(bpath)) as src:
        profile = src.profile.copy()
    profile.update(dtype=rasterio.int32, count=1, nodata=-1)
    with rasterio.open(str(pred_dir / bpath.name), 'w', **profile) as dst:
        dst.write(pred.astype(np.int32)[np.newaxis, :, :])

# -- Summary (pooled headline numbers) -----------------------------------------
df     = pd.DataFrame(records)
pooled = eval_common.pool_metrics(tiles)

print(f"\n=== Summary: {run_name} ===")
print(f"  {eval_common.method_note(IOU_THRESHOLD)}")
print(f"  tiles={len(tiles)}  GT={pooled['n_gt']}  pred={pooled['n_pred']}  "
      f"TP={pooled['tp']}  FP={pooled['fp']}  FN={pooled['fn']}")
print(f"  Precision (pooled): {pooled['precision']:.4f}")
print(f"  Recall    (pooled): {pooled['recall']:.4f}")
print(f"  F1        (pooled): {pooled['f1']:.4f}")
# Headline object Jaccard is the GT-weighted pooled value — identical to what
# eval_apoc.py prints (pooled['jaccard']). The per-tile mean/median below is a
# different (macro) aggregation, kept only as supplementary detail.
print(f"  object Jaccard (pooled): {pooled['jaccard']:.4f}")
if not df.empty and df['jaccard'].notna().any():
    print(f"  object Jaccard (per-tile mean/median): "
          f"mean={float(df['jaccard'].dropna().mean()):.4f}  "
          f"median={float(df['jaccard'].dropna().median()):.4f}", flush=True)

out_csv = reports_dir / f"metrics_{run_name}.csv"
df.to_csv(out_csv, index=False)
print(f"\n  CSV: {out_csv}")
print(f"  Predictions: {pred_dir}", flush=True)
