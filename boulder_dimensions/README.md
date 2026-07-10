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

## Important notes

- **Height method**: `max(depth[boulder_pixels]) − P10(depth[ring_pixels])`.
  P10 (not median or minimum) is used for the seafloor reference — robust against
  single outliers.
- **Prediction labels**: instance labels where `pred > 0` means boulder (not `pred == 1`).
  Each connected boulder has a unique integer ID.
- The ring width in pixels is auto-scaled from the pixel size of the GeoTIFF.

## Requirements

- Key packages: `numpy`, `pandas`, `rasterio`, `scipy`
