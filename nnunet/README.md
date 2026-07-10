# nnU-Net — Deep Learning Boulder Detection

## Model overview

| Property | Value |
|---|---|
| Framework | nnU-Net v2, 2D configuration |
| Training areas | 6 survey areas |
| Training tiles | 118 tiles, 2,444 annotated boulders |
| Test tiles | 37 tiles, 604 ground-truth boulders |
| Upscaling | 2× bilinear (applied before training and inference) |
| Cross-validation | 5-fold |
| Ensemble | Softmax average over all 5 folds |
| Test performance | Recall 0.767 · Precision 0.843 · F1 0.803 (IoU ≥ 0.1, unified eval) |

---

## Scripts

All example commands below are run **from the repository root** (e.g.
`python nnunet/eval_objects.py`); the scripts' relative defaults (such as
`eval_objects.py`'s `--code-dir ./nnunet`) assume that working directory.

```
predict_folder.py             Main inference script — GeoTIFF in, boulder mask out
prepare_nnunet.py             Convert GeoTIFF pairs to nnU-Net NIfTI format (+ case_mapping.csv)
../upscale.py                 2× bilinear upscaling utility (shared, at the repository root)
split_annotations.py          Train / test split helper
eval_objects.py               Object-level evaluation (precision, recall, F1, Dice)
label_boulders.py             Convert binary mask to instance labels (unique ID per boulder)
```

---

## Installation

```bash
conda create -n nnunet-boulder python=3.10
conda activate nnunet-boulder
pip install -r requirements.txt
```

---

## Quick Start: Predict on New Data

### 1. Download model weights

Download `boulder_detection_M01_weights.zip` from the IOW DOI repository:  
**https://doi.iow.de/10.12754/data-2026-0008**

### 2. Install the pretrained model

```bash
conda activate nnunet-boulder

# Set the folder where model weights will live
export nnUNet_results=/path/to/your/nnunet/results   # Linux/Mac
set nnUNet_results=C:\path\to\your\nnunet\results     # Windows

nnUNetv2_install_pretrained_model_from_zip boulder_detection_M01_weights.zip
```

### 3. Run predictions

```bash
python nnunet/predict_folder.py \
    --input  /path/to/your/bathymetry_tiles \
    --output /path/to/output_folder \
    --results $nnUNet_results
```

**Input:** folder containing one or more single-band GeoTIFF bathymetry rasters (any resolution, any CRS).  
**Output:**
```
output/
  masks/           <name>_mask.tif        binary mask  (0 = background, 1 = boulder)
  bathy_upscaled/  <name>_upscaled.tif    2× upscaled bathymetry for overlay in QGIS
```

Large tiles are split automatically and stitched back together. Tiles are read
with an overlap border and only their core region is written back, so boulders
straddling tile borders are not cut.

**NoData handling:** the training tiles contained no NoData. During inference,
NaN/NoData pixels are filled locally with a Gaussian blur (the same fill as the
APOC pipeline) before prediction — not a flat 0.0, which would be a large outlier
for depths around −20 m — and the final mask is forced to 0 in NoData areas, so no
boulders are reported there.

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--fold` | `0 1 2 3 4` | Which folds to use (subset = faster, less accurate) |
| `--scale` | `2` | Upscaling factor (must match training: 2) |
| `--dataset` | `001` | nnU-Net dataset ID |
| `--tile-size` | `4000` | Core tile size (px, upscaled) before splitting large images |
| `--overlap` | `128` | Overlap border (px, upscaled) read around each core tile (< half tile size) |

---

## Training from Scratch

### 1. Prepare data

Place bathymetry GeoTIFFs and annotation masks in (paths relative to the
repository root):
```
nnunet/train/bathy/          training tiles
nnunet/train/annotations/    corresponding annotation masks
nnunet/test/bathy/           test tiles
nnunet/test/annotations/     corresponding annotation masks
```

Then convert to nnU-Net format:
```bash
python nnunet/prepare_nnunet.py
```

### 2. Plan and preprocess

```bash
export nnUNet_raw=/path/to/nnunet/raw
export nnUNet_preprocessed=/path/to/nnunet/preprocessed
export nnUNet_results=/path/to/nnunet/results

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

### 3. Train (5-fold cross-validation)

```bash
for fold in 0 1 2 3 4; do
    nnUNetv2_train 1 2d $fold
done
```

Training time: approximately 2 hours per fold on a modern GPU.

---

## Data & Model Weights

Trained model weights and training/test data are archived separately:

| Item | Content | Repository |
|---|---|---|
| `boulder_detection_M01_weights.zip` | Trained model, all 5 folds | https://doi.iow.de/10.12754/data-2026-0008 |
| `raw_geotiffs/` | Original GeoTIFF bathymetry and annotations | https://doi.iow.de/10.12754/data-2026-0007 |

**Model weights DOI:** https://doi.iow.de/10.12754/data-2026-0008
**Training data DOI:** https://doi.iow.de/10.12754/data-2026-0007

---

## Evaluation method

`eval_objects.py` uses the shared `../eval_common.py` so its numbers are directly
comparable with the APOC evaluation: **8-connectivity** component labelling,
**greedy one-to-one matching** by IoU, a detection counts as a true positive at
**IoU ≥ 0.1**, and Precision/Recall/F1 are **pooled** over all test tiles (raw
TP/FP/FN summed, then the metrics computed once). Test-set case IDs are read from
`case_mapping.csv` (written by `prepare_nnunet.py`).

---

## Requirements

- Python 3.10, nnU-Net v2
- Key packages: `nnunetv2`, `torch`, `nibabel`, `numpy`, `rasterio`, `scipy`

Install via the pinned requirements file (from the `nnunet/` folder):
```bash
pip install -r requirements.txt
```
(Install `torch`/`torchvision` matching your CUDA version first — see
https://pytorch.org/get-started/locally/.)
