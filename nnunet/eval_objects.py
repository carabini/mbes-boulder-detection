"""
eval_objects.py
---------------
Object-level evaluation of nnU-Net boulder detection (Dataset001 — the model
reported in the paper).

Complements the pixel-level Dice score with instance-level metrics:
  - Recall    = what fraction of annotated boulders were found
  - Precision = what fraction of detections are real boulders
  - F1        = harmonic mean of precision and recall

Matching uses the shared eval_common.py method so the numbers are directly
comparable with the APOC evaluation: 8-connectivity, greedy 1:1 matching by IoU,
a pair counts as a True Positive at IoU >= 0.1, and Precision/Recall/F1 are
pooled over all tiles (a partial detection still counts as "found").

The test-set case IDs are read from case_mapping.csv (written by
prepare_nnunet.py). If that file is absent, a hardcoded fallback list is used.

Run with a Python environment matching requirements.txt:
  python eval_objects.py                          # uses --code-dir default
  python eval_objects.py --code-dir <path>        # base folder for pred/label paths
  python eval_objects.py --pred-dir <path>        # override prediction folder
  python eval_objects.py --lbl-dir <path>         # override label folder
  python eval_objects.py --case-mapping <csv>     # override case_mapping.csv path
"""

import argparse
import csv
import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# Shared object-level evaluation (repository root) — keeps APOC and nnU-Net
# numbers directly comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_common

# ── case mapping fallback (used only if case_mapping.csv is absent) ────────────
# Historical test set: 118 training cases -> test cases start at CASE_119.
FALLBACK_CASE_OFFSET = 119
FALLBACK_TEST_FILES = [
    "bh_03_01", "bh_03_02", "bh_03_03", "bh_03_04", "bh_03_05", "bh_03_06",
    "bw_02",
    "co_01_01", "co_01_02", "co_01_03", "co_01_04",
    "dl_bw_2_01", "dl_bw_2_02", "dl_bw_2_03", "dl_bw_2_04", "dl_bw_2_05", "dl_bw_2_06",
    "dl_tg_01_04", "dl_tg_01_05", "dl_tg_01_06",
    "jb_05_01", "jb_05_02", "jb_05_03", "jb_05_04", "jb_05_05", "jb_05_06",
    "nh_07_01", "nh_07_02", "nh_07_03", "nh_07_04", "nh_07_05", "nh_07_06",
    "so_07_01", "so_07_02", "so_07_03", "so_07_05", "so_07_06",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_nii(path):
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def area_of(fname):
    parts = fname.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else fname


def load_case_mapping(case_mapping_path):
    """Read (case_id, source_name) pairs for the test split from case_mapping.csv.

    Returns a list of (case_id, source_name) tuples, or None if the CSV is absent.
    """
    if case_mapping_path is None or not case_mapping_path.exists():
        return None
    cases = []
    with open(case_mapping_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split", "").strip().lower() != "test":
                continue
            case_id = row["case_id"].strip()
            src     = Path(row.get("source_filename", "")).stem or case_id
            cases.append((case_id, src))
    return cases


def resolve_cases(case_mapping_path):
    """Return list of (case_id, display_name) for the test set + a source note."""
    mapped = load_case_mapping(case_mapping_path)
    if mapped:
        return mapped, f"case_mapping.csv ({case_mapping_path})"
    cases = [(f"CASE_{FALLBACK_CASE_OFFSET + i:03d}", name)
             for i, name in enumerate(FALLBACK_TEST_FILES)]
    return cases, "hardcoded fallback list (case_mapping.csv not found)"


# ── main ─────────────────────────────────────────────────────────────────────

def main(pred_dir, lbl_dir, iou_threshold, case_mapping_path):

    cases, case_source = resolve_cases(case_mapping_path)

    rows  = []   # per-tile display rows
    tiles = []   # per-tile count dicts for pooled metrics
    print(f"\nObject-level evaluation  |  IoU threshold: {iou_threshold}")
    print(f"  {eval_common.method_note(iou_threshold)}")
    print(f"Case IDs:    {case_source}")
    print(f"Predictions: {pred_dir}")
    print(f"Labels:      {lbl_dir}\n")
    print(f"{'File':<22} {'GT':>5} {'TP':>5} {'FP':>5} {'FN':>5}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'Dice':>7}")
    print("-" * 80)

    for case_id, name in cases:
        pred_path = pred_dir / f"{case_id}.nii.gz"
        gt_path   = lbl_dir  / f"{case_id}.nii.gz"

        if not pred_path.exists():
            print(f"  WARNING: {case_id} prediction not found, skipping")
            continue
        if not gt_path.exists():
            print(f"  WARNING: {case_id} label not found, skipping")
            continue

        pred = load_nii(pred_path)
        gt   = load_nii(gt_path)

        # Resize pred to gt shape if needed (different upscale levels)
        if pred.shape != gt.shape:
            from scipy.ndimage import zoom as sz
            factor = (gt.shape[0] / pred.shape[0], gt.shape[1] / pred.shape[1])
            pred = sz(pred.astype(np.float32), factor, order=0).astype(np.uint8)

        # Object-level counts via the shared method (tiles with zero GT are NOT
        # skipped — their predictions correctly count as false positives).
        m = eval_common.evaluate_tile(gt == 1, pred == 1, threshold=iou_threshold)
        tiles.append(m)

        tp, fp, fn, n_gt = m['tp'], m['fp'], m['fn'], m['n_gt']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

        # Pixel Dice for reference
        inter = int(((gt == 1) & (pred == 1)).sum())
        dice  = 2 * inter / (int((gt == 1).sum()) + int((pred == 1).sum()) + 1e-6)

        print(f"{name:<22} {n_gt:>5} {tp:>5} {fp:>5} {fn:>5}  {recall:>7.3f}  {prec:>7.3f}  {f1:>7.3f}  {dice:>7.3f}")
        rows.append((name, n_gt, tp, fp, fn, recall, prec, f1, dice))

    if not rows:
        print("No results.")
        return

    # Summary (pooled over tiles)
    print("-" * 80)
    pooled = eval_common.pool_metrics(tiles)
    mean_dice = sum(r[8] for r in rows) / len(rows)
    print(f"{'TOTAL/POOLED':<22} {pooled['n_gt']:>5} {pooled['tp']:>5} {pooled['fp']:>5} {pooled['fn']:>5}  "
          f"{pooled['recall']:>7.3f}  {pooled['precision']:>7.3f}  {pooled['f1']:>7.3f}  {mean_dice:>7.3f}")

    # Per area summary (pooled within each area)
    print(f"\n--- Per survey area ---")
    areas = {}
    for r in rows:
        a = area_of(r[0])
        areas.setdefault(a, []).append(r)

    print(f"{'Area':<18} {'GT':>5} {'TP':>5} {'FP':>5} {'FN':>5}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'Dice':>7}")
    print("-" * 72)
    for area, area_rows in sorted(areas.items()):
        a_gt = sum(r[1] for r in area_rows)
        a_tp = sum(r[2] for r in area_rows)
        a_fp = sum(r[3] for r in area_rows)
        a_fn = sum(r[4] for r in area_rows)
        a_rec = a_tp / (a_tp + a_fn) if (a_tp + a_fn) > 0 else 0
        a_pre = a_tp / (a_tp + a_fp) if (a_tp + a_fp) > 0 else 0
        a_f1  = 2 * a_pre * a_rec / (a_pre + a_rec) if (a_pre + a_rec) > 0 else 0
        a_dice = sum(r[8] for r in area_rows) / len(area_rows)
        print(f"{area:<18} {a_gt:>5} {a_tp:>5} {a_fp:>5} {a_fn:>5}  {a_rec:>7.3f}  {a_pre:>7.3f}  {a_f1:>7.3f}  {a_dice:>7.3f}")

    print(f"\nSummary: {pooled['tp']}/{pooled['n_gt']} boulders found  "
          f"(Recall={pooled['recall']:.3f}, Precision={pooled['precision']:.3f}, F1={pooled['f1']:.3f})")
    print(f"IoU/object-Jaccard (mean best-IoU per GT, pooled): {pooled['jaccard']:.3f}")
    print(f"Mean pixel Dice: {mean_dice:.3f}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object-level boulder detection evaluation (Dataset001)")
    parser.add_argument("--code-dir", default="./nnunet",
                        help="Base folder containing pred/label subfolders (used to build default paths)")
    # Preferred dashed spellings, with the historical underscore spellings kept
    # as hidden aliases.
    parser.add_argument("--pred-dir", "--pred_dir", dest="pred_dir", default=None,
                        help="Override prediction folder path")
    parser.add_argument("--lbl-dir",  "--lbl_dir",  dest="lbl_dir",  default=None,
                        help="Override label folder path")
    parser.add_argument("--case-mapping", default=None,
                        help="Path to case_mapping.csv (default: <lbl-dir>/../case_mapping.csv)")
    parser.add_argument("--iou-threshold", default=0.1, type=float,
                        help="IoU threshold for a true positive (default: 0.1)")
    args = parser.parse_args()
    CODE = Path(args.code_dir)

    pred_dir = Path(args.pred_dir) if args.pred_dir else CODE / "predictions_ensemble"
    lbl_dir  = Path(args.lbl_dir)  if args.lbl_dir  else CODE / "raw" / "Dataset001_Boulders" / "labelsTs"

    if args.case_mapping:
        case_mapping_path = Path(args.case_mapping)
    else:
        # dataset root is the parent of labelsTs
        case_mapping_path = lbl_dir.parent / "case_mapping.csv"

    main(pred_dir, lbl_dir, iou_threshold=args.iou_threshold, case_mapping_path=case_mapping_path)
