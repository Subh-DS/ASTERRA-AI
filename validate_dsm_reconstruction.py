from pathlib import Path

import numpy as np
import rasterio


PRED_NDSM = Path(
    r"D:\Asterra AI\outputs\asterra_tiled_inference"
    r"\JAX_Tile_004_nDSM_tiled.npy"
)

DTM_PATH = Path(
    r"D:\Asterra AI\datasets\Urban3D\train\Inputs"
    r"\JAX_Tile_004_DTM.tif"
)

DSM_PATH = Path(
    r"D:\Asterra AI\datasets\Urban3D\train\Inputs"
    r"\JAX_Tile_004_DSM.tif"
)


print("=" * 78)
print("ASTERRA DSM RECONSTRUCTION — CORRECTED VALIDATION")
print("=" * 78)


# ---------------------------------------------------------
# LOAD ASTERRA nDSM
# ---------------------------------------------------------

pred_ndsm = np.load(PRED_NDSM).astype(np.float32)


# ---------------------------------------------------------
# LOAD DTM + MASK
# ---------------------------------------------------------

with rasterio.open(DTM_PATH) as src:
    dtm = src.read(1).astype(np.float32)
    dtm_mask = src.read_masks(1) > 0

    dtm_nodata = src.nodata
    dtm_crs = src.crs

    print("\nDTM")
    print("Shape       :", dtm.shape)
    print("CRS         :", dtm_crs)
    print("Nodata      :", dtm_nodata)
    print("Valid %     :", dtm_mask.mean() * 100)


# ---------------------------------------------------------
# LOAD DSM + MASK
# ---------------------------------------------------------

with rasterio.open(DSM_PATH) as src:
    dsm_gt = src.read(1).astype(np.float32)
    dsm_mask = src.read_masks(1) > 0

    dsm_nodata = src.nodata

    print("\nDSM")
    print("Shape       :", dsm_gt.shape)
    print("CRS         :", src.crs)
    print("Nodata      :", dsm_nodata)
    print("Valid %     :", dsm_mask.mean() * 100)


# ---------------------------------------------------------
# CHECK SHAPES
# ---------------------------------------------------------

if pred_ndsm.shape != dtm.shape:
    raise ValueError(
        f"Prediction shape {pred_ndsm.shape} != DTM shape {dtm.shape}"
    )

if dtm.shape != dsm_gt.shape:
    raise ValueError(
        f"DTM shape {dtm.shape} != DSM shape {dsm_gt.shape}"
    )


# ---------------------------------------------------------
# EXPLICIT NODATA FILTER
# ---------------------------------------------------------

valid = (
    np.isfinite(pred_ndsm)
    & np.isfinite(dtm)
    & np.isfinite(dsm_gt)
    & dtm_mask
    & dsm_mask
)

# Extra safety against nodata sentinel
if dtm_nodata is not None:
    valid &= dtm != dtm_nodata

if dsm_nodata is not None:
    valid &= dsm_gt != dsm_nodata


# ---------------------------------------------------------
# RECONSTRUCT DSM
# ---------------------------------------------------------

pred_dsm = pred_ndsm + dtm


# ---------------------------------------------------------
# EXTRACT VALID PIXELS
# ---------------------------------------------------------

pred = pred_dsm[valid]
gt = dsm_gt[valid]


print("\n" + "=" * 78)
print("VALIDATION MASK")
print("=" * 78)

print("Total pixels :", dsm_gt.size)
print("Valid pixels :", valid.sum())
print("Valid %      :", valid.mean() * 100)
print("Excluded %   :", (1.0 - valid.mean()) * 100)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

error = pred - gt
abs_error = np.abs(error)

mae = np.mean(abs_error)
rmse = np.sqrt(np.mean(error ** 2))
bias = np.mean(error)

corr = np.corrcoef(pred, gt)[0, 1]

ss_res = np.sum((gt - pred) ** 2)
ss_tot = np.sum((gt - np.mean(gt)) ** 2)

r2 = 1.0 - ss_res / ss_tot


print("\n" + "=" * 78)
print("ASTERRA RECONSTRUCTED DSM RESULTS")
print("=" * 78)

print(f"MAE             : {mae:.6f} m")
print(f"RMSE            : {rmse:.6f} m")
print(f"Bias            : {bias:.6f} m")
print(f"Median AE       : {np.median(abs_error):.6f} m")
print(f"P90 AE          : {np.percentile(abs_error, 90):.6f} m")
print(f"Correlation     : {corr:.6f}")
print(f"R²              : {r2:.6f}")


# ---------------------------------------------------------
# RANGE
# ---------------------------------------------------------

print("\n" + "=" * 78)
print("DSM RANGE")
print("=" * 78)

print(f"GT DSM min      : {gt.min():.6f} m")
print(f"GT DSM max      : {gt.max():.6f} m")

print(f"Pred DSM min    : {pred.min():.6f} m")
print(f"Pred DSM max    : {pred.max():.6f} m")

print(f"GT DSM mean     : {gt.mean():.6f} m")
print(f"Pred DSM mean   : {pred.mean():.6f} m")

print(f"GT DSM std      : {gt.std():.6f} m")
print(f"Pred DSM std    : {pred.std():.6f} m")


# ---------------------------------------------------------
# ERROR PERCENTILES
# ---------------------------------------------------------

print("\n" + "=" * 78)
print("ERROR DISTRIBUTION")
print("=" * 78)

print(f"P50 AE          : {np.percentile(abs_error, 50):.6f} m")
print(f"P75 AE          : {np.percentile(abs_error, 75):.6f} m")
print(f"P90 AE          : {np.percentile(abs_error, 90):.6f} m")
print(f"P95 AE          : {np.percentile(abs_error, 95):.6f} m")
print(f"P99 AE          : {np.percentile(abs_error, 99):.6f} m")


print("\n" + "=" * 78)
print("CORRECTED DSM VALIDATION COMPLETE")
print("=" * 78)