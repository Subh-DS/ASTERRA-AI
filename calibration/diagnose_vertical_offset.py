import numpy as np
import rasterio
from pathlib import Path


# ============================================================
# ASTERRA VERTICAL OFFSET DIAGNOSTIC
# ============================================================

PRED_PATH = Path(
    r"D:\Asterra AI\calibration\outputs\metric_dsm_srtm.tif"
)

GT_PATH = Path(
    r"D:\Asterra AI\datasets\Urban3D\train\Inputs\JAX_Tile_004_DSM.tif"
)


# ============================================================
# FILE CHECK
# ============================================================

if not PRED_PATH.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PRED_PATH}"
    )

if not GT_PATH.exists():
    raise FileNotFoundError(
        f"Ground truth file not found:\n{GT_PATH}"
    )


print("=" * 78)
print("ASTERRA — VERTICAL OFFSET DIAGNOSTIC")
print("=" * 78)

print("\nPrediction:")
print(PRED_PATH)

print("\nGround Truth:")
print(GT_PATH)


# ============================================================
# LOAD RASTERS
# ============================================================

with rasterio.open(PRED_PATH) as pred_src:

    pred = pred_src.read(1).astype(np.float64)

    pred_mask = pred_src.read_masks(1) > 0

    pred_crs = pred_src.crs
    pred_transform = pred_src.transform
    pred_nodata = pred_src.nodata


with rasterio.open(GT_PATH) as gt_src:

    gt = gt_src.read(1).astype(np.float64)

    gt_mask = gt_src.read_masks(1) > 0

    gt_crs = gt_src.crs
    gt_transform = gt_src.transform
    gt_nodata = gt_src.nodata


# ============================================================
# GRID CHECK
# ============================================================

if pred.shape != gt.shape:
    raise ValueError(
        f"Shape mismatch: {pred.shape} vs {gt.shape}"
    )

if pred_crs != gt_crs:
    raise ValueError(
        f"CRS mismatch: {pred_crs} vs {gt_crs}"
    )

if pred_transform != gt_transform:
    raise ValueError(
        "Transform mismatch between prediction and GT."
    )


# ============================================================
# VALID MASK
# ============================================================

valid = (
    np.isfinite(pred)
    & np.isfinite(gt)
    & pred_mask
    & gt_mask
)

if pred_nodata is not None:
    valid &= pred != pred_nodata

if gt_nodata is not None:
    valid &= gt != gt_nodata


pred_valid = pred[valid]
gt_valid = gt[valid]


if len(pred_valid) == 0:
    raise ValueError("No valid overlapping pixels.")


# ============================================================
# ORIGINAL METRICS
# ============================================================

error = pred_valid - gt_valid

abs_error = np.abs(error)

mae_before = np.mean(abs_error)

rmse_before = np.sqrt(
    np.mean(error ** 2)
)

bias_before = np.mean(error)

median_bias = np.median(error)

corr_before = np.corrcoef(
    pred_valid,
    gt_valid
)[0, 1]


ss_res_before = np.sum(
    (gt_valid - pred_valid) ** 2
)

ss_tot = np.sum(
    (gt_valid - np.mean(gt_valid)) ** 2
)

r2_before = (
    1.0 - ss_res_before / ss_tot
)


# ============================================================
# OFFSET ESTIMATES
# ============================================================

# Mean error
mean_offset = np.mean(
    pred_valid - gt_valid
)

# Median error
median_offset = np.median(
    pred_valid - gt_valid
)


# ============================================================
# APPLY MEAN OFFSET
# ============================================================

pred_mean_aligned = (
    pred_valid - mean_offset
)

error_mean = (
    pred_mean_aligned - gt_valid
)

abs_error_mean = np.abs(error_mean)

mae_mean = np.mean(
    abs_error_mean
)

rmse_mean = np.sqrt(
    np.mean(error_mean ** 2)
)

bias_mean = np.mean(
    error_mean
)

corr_mean = np.corrcoef(
    pred_mean_aligned,
    gt_valid
)[0, 1]

ss_res_mean = np.sum(
    (gt_valid - pred_mean_aligned) ** 2
)

r2_mean = (
    1.0 - ss_res_mean / ss_tot
)


# ============================================================
# APPLY MEDIAN OFFSET
# ============================================================

pred_median_aligned = (
    pred_valid - median_offset
)

error_median = (
    pred_median_aligned - gt_valid
)

abs_error_median = np.abs(
    error_median
)

mae_median = np.mean(
    abs_error_median
)

rmse_median = np.sqrt(
    np.mean(error_median ** 2)
)

bias_median = np.mean(
    error_median
)

corr_median = np.corrcoef(
    pred_median_aligned,
    gt_valid
)[0, 1]

ss_res_median = np.sum(
    (gt_valid - pred_median_aligned) ** 2
)

r2_median = (
    1.0 - ss_res_median / ss_tot
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 78)
print("VALIDATION")
print("=" * 78)

print("Valid pixels :", len(pred_valid))
print(f"GT mean      : {gt_valid.mean():.6f} m")
print(f"Pred mean    : {pred_valid.mean():.6f} m")


print("\n" + "=" * 78)
print("BEFORE OFFSET CORRECTION")
print("=" * 78)

print(f"MAE         : {mae_before:.6f} m")
print(f"RMSE        : {rmse_before:.6f} m")
print(f"Bias        : {bias_before:.6f} m")
print(f"Median bias : {median_bias:.6f} m")
print(f"Correlation : {corr_before:.6f}")
print(f"R²          : {r2_before:.6f}")


print("\n" + "=" * 78)
print("OFFSET ESTIMATES")
print("=" * 78)

print(
    f"Mean offset   : {mean_offset:.6f} m"
)

print(
    f"Median offset : {median_offset:.6f} m"
)


print("\n" + "=" * 78)
print("AFTER MEAN OFFSET CORRECTION")
print("=" * 78)

print(
    f"Applied offset : {-mean_offset:.6f} m"
)

print(
    f"MAE            : {mae_mean:.6f} m"
)

print(
    f"RMSE           : {rmse_mean:.6f} m"
)

print(
    f"Bias           : {bias_mean:.6f} m"
)

print(
    f"Correlation    : {corr_mean:.6f}"
)

print(
    f"R²             : {r2_mean:.6f}"
)


print("\n" + "=" * 78)
print("AFTER MEDIAN OFFSET CORRECTION")
print("=" * 78)

print(
    f"Applied offset : {-median_offset:.6f} m"
)

print(
    f"MAE            : {mae_median:.6f} m"
)

print(
    f"RMSE           : {rmse_median:.6f} m"
)

print(
    f"Bias           : {bias_median:.6f} m"
)

print(
    f"Correlation    : {corr_median:.6f}"
)

print(
    f"R²             : {r2_median:.6f}"
)


# ============================================================
# DIAGNOSTIC INTERPRETATION
# ============================================================

print("\n" + "=" * 78)
print("DIAGNOSTIC")
print("=" * 78)

if rmse_mean < rmse_before * 0.5:
    print(
        "Large RMSE reduction after constant-offset correction."
    )
    print(
        "This indicates that a substantial portion of the"
    )
    print(
        "error is consistent with a vertical reference/offset"
    )
    print(
        "mismatch."
    )
else:
    print(
        "Constant-offset correction does not explain most"
    )
    print(
        "of the error."
    )
    print(
        "Additional scale, surface-definition, or spatial"
    )
    print(
        "alignment differences require investigation."
    )


print("\n" + "=" * 78)
print("IMPORTANT")
print("=" * 78)

print(
    "The offset-corrected values are diagnostic only."
)

print(
    "No production calibration file has been modified."
)

print(
    "The ASTERRA final height model remains unchanged."
)

print("=" * 78)