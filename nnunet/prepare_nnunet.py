"""
prepare_nnunet.py
-----------------
Converts upscaled GeoTIFF pairs (bathy + annotation) into nnU-Net v2 format:
  - Images  → NIfTI (.nii.gz), float32, single channel (_0000 suffix)
  - Labels  → NIfTI (.nii.gz), uint8, label remapped {1->1 boulder, 2->0 background}
  - Writes  dataset.json
  - Writes  case_mapping.csv  (case_id, split, source_filename)

Input folders:
  train/bathy/        + train/annotations/   → imagesTr/ + labelsTr/
  test/bathy/         + test/annotations/    → imagesTs/ + labelsTs/

A missing annotation for any bathy tile is a hard error: the train/test case
IDs are consecutive and offset-based, so a silently skipped case would break the
mapping that eval_objects.py relies on.

Run with a Python environment matching requirements.txt:
  python prepare_nnunet.py
"""

import csv
import json
import numpy as np
import rasterio
import nibabel as nib
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

CODE_DIR  = Path(__file__).parent
TRAIN_B   = CODE_DIR / "train"  / "bathy"
TRAIN_A   = CODE_DIR / "train"  / "annotations"
TEST_B    = CODE_DIR / "test"   / "bathy"
TEST_A    = CODE_DIR / "test"   / "annotations"

DATASET   = CODE_DIR / "raw" / "Dataset001_Boulders"
IMG_TR    = DATASET / "imagesTr"
LBL_TR    = DATASET / "labelsTr"
IMG_TS    = DATASET / "imagesTs"
LBL_TS    = DATASET / "labelsTs"

# ── helpers ───────────────────────────────────────────────────────────────────

def read_geotiff(path):
    """Read single-band GeoTIFF. Returns (array float32, pixel_size_metres)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        pixel_size = abs(src.transform.a)   # metres per pixel (x-direction)
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, pixel_size


def save_nifti_image(arr, pixel_size, out_path):
    """Save float32 bathymetry array as NIfTI with correct spacing."""
    # nnU-Net 2D: array must be (H, W) → nibabel expects (X, Y, Z) so we add a Z dim
    arr_3d = arr[:, :, np.newaxis]          # (H, W, 1)
    affine = np.diag([pixel_size, pixel_size, 1.0, 1.0])
    img = nib.Nifti1Image(arr_3d, affine)
    img.header.set_zooms((pixel_size, pixel_size, 1.0))
    nib.save(img, str(out_path))


def save_nifti_label(arr, pixel_size, out_path):
    """Remap {1->1 boulder, 2->0 background} and save as uint8 NIfTI."""
    label = np.where(arr == 1, 1, 0).astype(np.uint8)
    label_3d = label[:, :, np.newaxis]
    affine = np.diag([pixel_size, pixel_size, 1.0, 1.0])
    img = nib.Nifti1Image(label_3d, affine)
    img.header.set_zooms((pixel_size, pixel_size, 1.0))
    nib.save(img, str(out_path))


def convert_split(bathy_dir, anno_dir, img_out, lbl_out, start_case, split_name, mapping):
    """Convert one split (train or test).

    start_case : 1-based case number for the first written case.
    mapping    : list appended with (case_id, split, source_filename) rows.

    The case counter increments only when a case is actually written, so the
    train/test offsets stay consistent. A missing annotation raises an error.
    Returns the number of cases written.
    """
    bathy_files = sorted(bathy_dir.glob("*.tif"))
    n = 0
    for bpath in bathy_files:
        apath = anno_dir / bpath.name
        if not apath.exists():
            raise FileNotFoundError(
                f"No annotation for {bpath.name} in {anno_dir}. "
                "Every bathy tile needs a matching annotation — a skipped case "
                "would break the train/test case-offset contract.")

        case_name = f"CASE_{start_case + n:03d}"

        bathy_arr, px = read_geotiff(bpath)
        anno_arr,  _  = read_geotiff(apath)

        save_nifti_image(bathy_arr, px, img_out / f"{case_name}_0000.nii.gz")
        save_nifti_label(anno_arr,  px, lbl_out / f"{case_name}.nii.gz")

        mapping.append((case_name, split_name, bpath.name))

        # Quick label check
        unique = np.unique(np.where(anno_arr == 1, 1, 0))
        print(f"  [{split_name}] {case_name} <- {bpath.name}  px={px:.4f}m  labels={unique}")
        n += 1
    return n


# ── main ──────────────────────────────────────────────────────────────────────

print("=== prepare_nnunet.py ===\n")

# Create all output directories before writing.
for d in (IMG_TR, LBL_TR, IMG_TS, LBL_TS, DATASET):
    d.mkdir(parents=True, exist_ok=True)

mapping = []

print(f"Converting TRAIN set ({TRAIN_B}) ...")
n_train = convert_split(TRAIN_B, TRAIN_A, IMG_TR, LBL_TR,
                        start_case=1, split_name="train", mapping=mapping)

print(f"\nConverting TEST set ({TEST_B}) ...")
n_test = convert_split(TEST_B, TEST_A, IMG_TS, LBL_TS,
                       start_case=n_train + 1, split_name="test", mapping=mapping)

# ── case_mapping.csv ──────────────────────────────────────────────────────────

mapping_csv = DATASET / "case_mapping.csv"
with open(mapping_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "split", "source_filename"])
    writer.writerows(mapping)
print(f"\ncase_mapping.csv written -> {mapping_csv}")

# ── dataset.json ──────────────────────────────────────────────────────────────

dataset_json = {
    "channel_names": {
        "0": "bathymetry"
    },
    "labels": {
        "background": 0,
        "boulder": 1
    },
    "numTraining": n_train,
    "file_ending": ".nii.gz"
}

out_json = DATASET / "dataset.json"
with open(out_json, "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"\ndataset.json written -> {out_json}")
print(f"\nDone.  Train: {n_train} cases   Test: {n_test} cases")
print(f"Output: {DATASET}")
