# mbes-boulder-detection

Automated detection of boulders on the seafloor from multibeam echo sounder (MBES) bathymetry data (raster). This repository includes two independent methods evaluated on the same dataset from the southwestern Baltic Sea:

- **APOC** (`apoc/`) — sparse pixel classification using a Random Forest with multi-scale image features; requires only a small fraction of pixels to be labelled

- **nnU-Net** (`nnunet/`) — self-configuring deep learning segmentation trained on fully annotated tiles

Boulder physical dimensions (height above seafloor, footprint area) are extracted from predicted masks using the scripts in `boulder_dimensions/`.

---

## Repository structure

```
apoc/                   APOC sparse pixel classifier — scripts and documentation
nnunet/                 nnU-Net deep learning model — scripts and documentation
boulder_dimensions/     Boulder dimension extraction from predicted masks
eval_common.py          Shared object-level evaluation (used by both methods)
upscale.py              Shared 2× bilinear upscaling utility
```

Each method has its own pinned dependencies: `apoc/requirements.txt` and
`nnunet/requirements.txt` (the two run in separate Python environments).

---

## Study area & data

Survey area:  southwestern Baltic Sea  
Input: single-band GeoTIFF bathymetry rasters derived from MBES surveys (0.1, 0.15, and 0.25 m resolution).

---

## Data & model weights

The trained nnU-Net model weights and raw bathymetry/annotation GeoTIFFs are archived separately:

| Item | Content | Repository |
|---|---|---|
| `boulder_detection_M01_weights.zip` | Trained nnU-Net model, all 5 folds | https://doi.iow.de/10.12754/data-2026-0008 |
| `raw_geotiffs/` | Bathymetry and annotation GeoTIFFs (train + test) | https://doi.iow.de/10.12754/data-2026-0007 |

---

## Authors

Aïcha Naumann¹, Robert Haase², Peter Feldens¹, Svenja Papenmeier¹

¹ Leibniz Institute for Baltic Sea Research Warnemünde (IOW), Rostock, Germany  
² Center for Scalable Data Analytics and Artificial Intelligence Dresden/Leipzig (ScaDS.AI), Leipzig University, Leipzig, Germany

Correspondence: aicha.naumann@iow.de

---

## Citation

If you use this code or the trained model in your work, please cite:

> Naumann, A., Haase, R., Feldens, P., Papenmeier, S. (in preparation). Automated seafloor boulder detection from MBES bathymetry using sparse pixel classification and deep learning.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
