# Boulder Dimension Extraction

## What it does

`extract_boulder_dimensions.py` computes physical dimensions for each detected boulder
segment, matching the method reported in the paper:

- **Footprint area** — in m², from pixel count × pixel size²
- **Height above seafloor** — boulder top (shallowest depth pixel in the segment) minus
  the 10th-percentile depth of a 0.75 m annular ring around the boulder (robust against
  single outliers)

Predictions are downsampled (nearest-neighbour) to the original-resolution bathymetry
grid before extraction. Results are written to one CSV file per site, plus a summary
text file.

## Input

Two validation sites (NH, FMB), each needing:
- A bathymetry GeoTIFF (float32, depth in metres, nodata = NaN or < -1e38)
- An APOC-25 prediction GeoTIFF and an nnU-Net prediction GeoTIFF
  (instance labels: 0 = background, >0 = individual boulder IDs)

## How to run

```
python extract_boulder_dimensions.py [--out-dir OUT] [--sites-config sites.json] [--ring-width-m 0.75]
```

CSVs and the summary are written to `--out-dir` (default: the current folder `.`).

Without `--sites-config`, the script uses the built-in **generic placeholder**
paths for the NH/FMB sites (the `SITES` dict at the top of the script). Copy
`sites.example.json` (shipped next to the script), edit the paths to point at your
own data, and pass it with `--sites-config sites.example.json`.

## Whole folders / large areas: `extract_boulder_dimensions_folder.py`

Same method, but for a folder of predictions from **one** model, as written by the
prediction scripts. Use this one for new survey areas; it scales to rasters with more
than 100 million pixels.

```
python extract_boulder_dimensions_folder.py --pred-dir PRED --bathy-dir BATHY \
       [--model nnunet|apoc] [--out-dir OUT] [--ring-width-m 0.75] [--max-pixels 0]
```

| `--model` | Prediction files in `--pred-dir` | Boulder pixels |
|---|---|---|
| `nnunet` (default) | `<name>_mask.tif` from `nnunet/predict_folder.py` (pass its `masks/` folder) | value `== 1` (0 = background, 255 = nodata); segments = 8-connected components on the bathymetry grid |
| `apoc` | `<name>_pred.tif` from `apoc/predict_apoc25.py` | instance labels `> 0`, IDs are kept |

Each prediction is paired with `<name>.tif` in `--bathy-dir`, i.e. the
original-resolution bathymetry the prediction was made from. The script stops with a
list of the unmatched files if a bathymetry raster is missing.

Outputs: `boulder_dimensions_<name>.csv` per file, `boulder_dimensions_all.csv`, and
`boulder_dimensions_summary.txt`. In addition to the columns of `extract_boulder_dimensions.py`, each
row carries the segment centroid in map coordinates (`centroid_x`, `centroid_y`).

`--max-pixels N` skips segments larger than N pixels on the bathymetry grid (reef or
outcrop areas segmented as one object). It is off by default and not part of the
paper method.

**Limitation:** the height method assumes the boulder sits *on* the surrounding
seafloor. Boulders lying inside a scour depression can get a height near zero or
slightly negative, because the ring reference lies above the boulder top. These are
not false detections; treat their height as unreliable rather than filtering them out.

## Important notes

- **Height method**: `max(depth[boulder_pixels]) − P10(depth[ring_pixels])`.
  P10 (not median or minimum) is used for the seafloor reference — robust against
  single outliers.
- **Prediction labels** (`extract_boulder_dimensions.py`): instance labels where
  `pred > 0` means boulder (not `pred == 1`). Each connected boulder has a unique
  integer ID. Do not feed it the raw `<name>_mask.tif` of `predict_folder.py`: those
  carry nodata = 255, which `> 0` would count as boulder. Run `nnunet/label_boulders.py`
  first, or use `extract_boulder_dimensions_folder.py --model nnunet`.
- The ring width in pixels is auto-scaled from the pixel size of the GeoTIFF.

## Requirements

- Key packages: `numpy`, `pandas`, `rasterio`, `scipy`
