# APOC — Sparse Pixel Classification for Boulder Detection

## What is APOC?

APOC (Accelerated Pixel and Object Classification; Haase et al. 2022) is a pixel classifier
based on a **Random Forest** (scikit-learn). What makes it suitable for bathymetric data is its
automatic computation of multi-scale image features: for each pixel, it computes not just the
raw depth value but a set of filtered versions of the image (Gaussian blur, Difference of
Gaussians, Laplacian of Gaussian) at multiple spatial scales (σ = 2, 3, 4, 6 pixels). The
Random Forest then learns which combination of these features distinguishes boulders from
background.

This makes APOC well-suited for detecting small, relief-based objects like boulders in
single-channel bathymetry — the local texture features capture the characteristic 
signature.

## Sparse Annotations

A key advantage of APOC is that **not every pixel needs to be labelled**. Only a subset of
boulder and background pixels are annotated (here: 25% of boulder segments per tile, with a
background ring of 6 px around each annotated boulder). The classifier learns from these
sparse labels and generalises to the full image.

Class schema: `0` = unannotated (ignored during training), `1` = boulder, `2` = background.

## Scripts — Run in This Order

### 1. `../upscale.py` — Bilinear 2× upscaling
Upscales all bathymetry GeoTIFFs and annotation masks by factor 2 (bilinear for bathymetry,
nearest-neighbour for masks). Upscaling brings the median boulder diameter from ~2 px to
~8 px, which improves feature computation and classification. This is the single shared
`upscale.py` at the repository root (the nnU-Net pipeline uses the same file).

### 2. `make_sparse_annotations_bg10to1_25pct.py` — Generate sparse training masks
From the fully annotated upscaled tiles, this script randomly retains 25% of boulder segments
per tile and adds a 6 px background ring around each. Background pixels are sampled to achieve
a background:boulder ratio of 10:1. All other pixels are set to 0 (unannotated).

### 3. `train_upscaled_sparse_bg10to1_25pct_sigma2346_v1.py` — Train and evaluate
Collects image features from all 118 training tiles, trains one Random Forest on all annotated
pixels combined, saves the classifier as a `.cl` file, and evaluates on 37 held-out test tiles
using the shared object-level evaluation (see **Evaluation method** below).

**Run with:**
```
python train_upscaled_sparse_bg10to1_25pct_sigma2346_v1.py
```

### 4. `predict_apoc25.py` — Predict on new GeoTIFFs
Full inference pipeline for new bathymetry files:
1. 2× bilinear upscaling
2. Tiled APOC prediction (512 px tiles, 64 px overlap)
3. Per-boulder statistics (area, height, diameter) → CSV output

The prediction is the raw model output — no post-processing filter is applied.

The per-boulder CSV uses the **same height method as the paper** (seafloor
reference = the Nth percentile of a ring around the boulder; defaults: 0.75 m
ring, 10th percentile). The authoritative dimension extraction for the paper is
`../boulder_dimensions/extract_boulder_dimensions.py` — use that for reported figures.

**Run with:**
```
python predict_apoc25.py INPUT.tif [OUTPUT_DIR]
```

> **Note on default paths:** the scripts' default `--input`/`--output` folders do
> not chain automatically. Pass matching folders between steps so each step reads
> the previous step's output (the relative defaults follow the flow
> `data/upscaled/train/annotations` → `data/sparse/annotations` → training).

## Evaluation method

Both methods share `../eval_common.py` so the numbers are directly comparable:
**8-connectivity** component labelling, **greedy one-to-one matching** by IoU, a
detection counts as a true positive at **IoU ≥ 0.1**, and Precision/Recall/F1 are
**pooled** over all test tiles (raw TP/FP/FN summed, then the metrics computed once).

The training script (step 3) prints these metrics while training. To reproduce
them from saved prediction rasters — without retraining — run the standalone
evaluator (the APOC counterpart to `../nnunet/eval_objects.py`):
```
python eval_apoc.py --pred-dir <predictions> --anno-dir <test annotations>
```
Both entry points call the same `eval_common` functions, so they yield identical
numbers; that is the figure to report in the paper.

## Reproducibility

The Random Forest is trained with `random_state=0`, which fixes the internal random number
generator. Training is therefore **deterministic given an identical software environment and
hardware** — note that APOC's OpenCL/GPU feature computation may vary slightly across devices.

## Requirements

- Python 3.12.4
- Key packages: see `requirements.txt` (in this folder)

Install (from the `apoc/` folder):
```
pip install -r requirements.txt
```

APOC alone can be installed with:
```
pip install apoc==0.12.0
```

> The training script uses APOC internals (`ObjectSegmenter._make_features_*`,
> `_to_np`, `to_opencl_file`), so it requires exactly the pinned version
> `apoc==0.12.0`.


