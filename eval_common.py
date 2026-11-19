"""
eval_common.py
--------------
Shared object-level evaluation used by both boulder-detection methods (APOC and
nnU-Net) so their numbers are directly comparable. This implements the single
evaluation method agreed for the paper:

  - Connectivity: 8-connectivity everywhere
    (scipy.ndimage.label with structure=np.ones((3, 3), dtype=bool)).
  - Matching: greedy one-to-one matching by IoU. The IoU matrix between all GT
    objects and all predicted objects is computed efficiently from an overlap
    (confusion-matrix) histogram; all (gt, pred) pairs with IoU >= threshold are
    sorted by IoU descending and assigned greedily, each GT and each prediction
    matched at most once. A pair counts as a TP if IoU >= threshold (default 0.1).
  - Aggregation: micro / pooled. Each tile returns raw counts
    (tp, fp, fn, n_gt, n_pred); tiles with zero GT objects are NOT skipped, so
    their predictions count as false positives. Counts are summed over all tiles
    and Precision / Recall / F1 are computed once from the pooled totals.
  - "object Jaccard": the mean best-IoU per GT object, computed from the same IoU
    matrix, kept available as a secondary metric.

Depends only on numpy and scipy so it can be imported from either method's
environment.
"""

import numpy as np
from scipy.ndimage import label as _nd_label

# Self-documenting one-liner both scripts print to their logs.
def method_note(threshold=0.1):
    return (f"Evaluation: 8-connectivity, greedy 1:1 matching, IoU>={threshold:g}, "
            "pooled over tiles")

# Backwards-compatible constant (default threshold).
METHOD_NOTE = method_note()

# 8-connectivity structuring element.
_STRUCT8 = np.ones((3, 3), dtype=bool)


def label_objects(binary_mask):
    """Label connected components with 8-connectivity.

    Returns (labeled_int32, n_objects).
    """
    labeled, n = _nd_label(np.asarray(binary_mask) > 0, structure=_STRUCT8)
    return labeled.astype(np.int32), int(n)


def iou_matrix(gt_labeled, pred_labeled):
    """IoU matrix between GT objects (rows) and predicted objects (cols).

    Computed from an overlap histogram (the confusion-matrix overlap approach),
    which is O(n_pixels) rather than O(n_objects * n_pixels).

    Returns an (n_gt, n_pred) float64 array; entry [i, j] is the IoU between
    GT object i+1 and predicted object j+1.
    """
    gt_labeled   = np.asarray(gt_labeled)
    pred_labeled = np.asarray(pred_labeled)
    n_gt   = int(gt_labeled.max()) if gt_labeled.size else 0
    n_pred = int(pred_labeled.max()) if pred_labeled.size else 0
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64)

    gt_flat   = gt_labeled.ravel().astype(np.int64)
    pred_flat = pred_labeled.ravel().astype(np.int64)

    # 2D histogram of (gt_label, pred_label) including the 0/background row+col.
    idx     = gt_flat * (n_pred + 1) + pred_flat
    counts  = np.bincount(idx, minlength=(n_gt + 1) * (n_pred + 1))
    overlap = counts.reshape(n_gt + 1, n_pred + 1)

    inter      = overlap[1:, 1:].astype(np.float64)
    gt_sizes   = overlap[1:, :].sum(axis=1, keepdims=True).astype(np.float64)
    pred_sizes = overlap[:, 1:].sum(axis=0, keepdims=True).astype(np.float64)
    union = gt_sizes + pred_sizes - inter
    union[union == 0] = 1.0
    return inter / union


def match_greedy(iou, threshold=0.1):
    """Greedy one-to-one matching of GT to predictions by IoU.

    All (gt, pred) pairs with IoU >= threshold are sorted by IoU descending and
    assigned greedily; each GT and each prediction is matched at most once.

    Returns (tp, fp, fn, matched_pairs) where matched_pairs is a list of
    (gt_index, pred_index, iou) tuples.
    """
    iou = np.asarray(iou, dtype=np.float64)
    n_gt, n_pred = iou.shape
    matched_pairs = []
    if n_gt and n_pred:
        gis, pis = np.where(iou >= threshold)
        order = np.argsort(iou[gis, pis])[::-1]
        gt_used   = np.zeros(n_gt,   dtype=bool)
        pred_used = np.zeros(n_pred, dtype=bool)
        for k in order:
            gi = int(gis[k]); pi = int(pis[k])
            if gt_used[gi] or pred_used[pi]:
                continue
            gt_used[gi]   = True
            pred_used[pi] = True
            matched_pairs.append((gi, pi, float(iou[gi, pi])))
    tp = len(matched_pairs)
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn, matched_pairs


def _as_labeled(mask_or_labeled):
    """Return an instance-labeled int array (8-connectivity).

    Every input -- binary mask or instance-labelled array -- is binarised
    (value > 0) and relabelled with 8-connectivity, so that all methods are
    evaluated with exactly the same object definition.

    Pre-existing instance labels are deliberately NOT reused: APOC's
    ObjectSegmenter labels with 4-connectivity, and keeping those labels made
    APOC objects split along diagonals (more, smaller objects -> extra false
    positives) while nnU-Net masks were labelled with 8-connectivity.
    Relabelling also removes gaps in label IDs (e.g. tile-cropped
    predict_apoc25.py output) and re-joins objects cut at tile borders.
    """
    arr = np.asarray(mask_or_labeled)
    return label_objects(arr > 0)


def evaluate_tile(gt_binary, pred_binary_or_labeled, threshold=0.1):
    """Evaluate one tile.

    gt_binary               : binary GT mask (boulder vs. not).
    pred_binary_or_labeled  : binary prediction OR an instance-labelled array
                              (always binarised and relabelled, 8-connectivity).

    Returns a dict with raw counts (tp, fp, fn, n_gt, n_pred) and the object
    Jaccard (mean best-IoU per GT object). Tiles with zero GT objects are handled
    here (their predictions count as FP) — callers must NOT skip them.
    """
    gt_labeled, n_gt     = label_objects(gt_binary)
    pred_labeled, n_pred = _as_labeled(pred_binary_or_labeled)

    iou = iou_matrix(gt_labeled, pred_labeled)
    tp, fp, fn, _ = match_greedy(iou, threshold)

    if n_gt == 0:
        jaccard = float('nan')
    elif n_pred == 0:
        jaccard = 0.0
    else:
        jaccard = float(iou.max(axis=1).mean())

    return dict(tp=tp, fp=fp, fn=fn, n_gt=n_gt, n_pred=n_pred, jaccard=jaccard)


def pool_metrics(tile_dicts):
    """Pool per-tile counts into micro-averaged Precision / Recall / F1.

    Returns a dict with pooled totals (tp, fp, fn, n_gt, n_pred),
    precision / recall / f1 computed once from those totals, and the pooled
    object Jaccard (mean best-IoU over ALL ground-truth objects, i.e. the per-tile
    jaccard weighted by the tile's GT count — same pooled aggregation as P/R/F1).
    """
    tp     = sum(int(d['tp'])     for d in tile_dicts)
    fp     = sum(int(d['fp'])     for d in tile_dicts)
    fn     = sum(int(d['fn'])     for d in tile_dicts)
    n_gt   = sum(int(d['n_gt'])   for d in tile_dicts)
    n_pred = sum(int(d['n_pred']) for d in tile_dicts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    if precision == precision and recall == recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float('nan')

    # Pooled object Jaccard: average best-IoU over every GT object across all
    # tiles. per-tile jaccard is the mean best-IoU of that tile's GT objects, so
    # weighting by n_gt and dividing by the total GT count recovers the pooled mean.
    jac_num, jac_den = 0.0, 0
    for d in tile_dicts:
        ng = int(d['n_gt']); j = d.get('jaccard', float('nan'))
        if ng > 0 and j == j:          # skip empty-GT tiles (jaccard is nan there)
            jac_num += j * ng
            jac_den += ng
    jaccard = jac_num / jac_den if jac_den > 0 else float('nan')

    return dict(tp=tp, fp=fp, fn=fn, n_gt=n_gt, n_pred=n_pred,
                precision=precision, recall=recall, f1=f1, jaccard=jaccard)
