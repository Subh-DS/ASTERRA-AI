import numpy as np
import rasterio
from pathlib import Path


# ============================================================
# ASTERRA — OFFLINE AFFINE CALIBRATION DIAGNOSTIC
#
# IMPORTANT:
# This is an evaluation-only experiment.
# It uses Urban3D ground truth to estimate:
#
#       GT ≈ a * Prediction + b
#
# It does NOT modify the production calibration engine.
# It does NOT modify the final ASTERRA model.
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
print("ASTERRA — OFFLINE AFFINE CALIBRATION DIAGNOSTIC")
print("=" * 78)

print("\nPrediction:")
print(PRED_PATH)

print("\nGround Truth:")
print(GT_PATH)


# ============================================================
# LOAD PREDICTION
# ============================================================

with rasterio.open(PRED_PATH) as pred_src:

    pred = pred_src.read(1).astype(np.float64)

    pred_mask = pred_src.read_masks(1) > 0

    pred_crs = pred_src.crs
    pred_transform = pred_src.transform
    pred_nodata = pred_src.nodata


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

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


x = pred[valid]
y = gt[valid]


if len(x) == 0:
    raise ValueError(
        "No valid overlapping pixels found."
    )


# ============================================================
# BASELINE METRICS
# ============================================================

error_before = x - y

mae_before = np.mean(
    np.abs(error_before)
)

rmse_before = np.sqrt(
    np.mean(error_before ** 2)
)

bias_before = np.mean(
    error_before
)

corr_before = np.corrcoef(
    x,
    y
)[0, 1]

ss_res_before = np.sum(
    (y - x) ** 2
)

ss_tot = np.sum(
    (y - np.mean(y)) ** 2
)

r2_before = (
    1.0 -
    ss_res_before / ss_tot
)


# ============================================================
# LEAST-SQUARES AFFINE FIT
#
# y ≈ a*x + b
# ============================================================

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum(
    (x - x_mean) * (y - y_mean)
)

denominator = np.sum(
    (x - x_mean) ** 2
)

if denominator == 0:
    raise ValueError(
        "Prediction has zero variance; affine fit impossible."
    )

a = numerator / denominator

b = y_mean - a * x_mean


# ============================================================
# APPLY AFFINE CALIBRATION
# ============================================================

y_pred_affine = (
    a * x + b
)

error_affine = (
    y_pred_affine - y
)

abs_error_affine = np.abs(
    error_affine
)

mae_affine = np.mean(
    abs_error_affine
)

rmse_affine = np.sqrt(
    np.mean(error_affine ** 2)
)

bias_affine = np.mean(
    error_affine
)

median_ae_affine = np.median(
    abs_error_affine
)

p90_affine = np.percentile(
    abs_error_affine,
    90
)

p95_affine = np.percentile(
    abs_error_affine,
    95
)

p99_affine = np.percentile(
    abs_error_affine,
    99
)

corr_affine = np.corrcoef(
    y_pred_affine,
    y
)[0, 1]

ss_res_affine = np.sum(
    (y - y_pred_affine) ** 2
)

r2_affine = (
    1.0 -
    ss_res_affine / ss_tot
)


# ============================================================
# SCALE / OFFSET INTERPRETATION
# ============================================================

# For the fitted relationship:
#
# GT ≈ a * Prediction + b
#
# a = vertical scale correction
# b = vertical offset correction


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 78)
print("INPUT STATISTICS")
print("=" * 78)

print(
    f"Valid pixels : {len(x)}"
)

print(
    f"Prediction mean : {x.mean():.6f} m"
)

print(
    f"Ground truth mean: {y.mean():.6f} m"
)

print(
    f"Prediction std  : {x.std():.6f} m"
)

print(
    f"Ground truth std: {y.std():.6f} m"
)


print("\n" + "=" * 78)
print("BEFORE AFFINE CALIBRATION")
print("=" * 78)

print(
    f"MAE         : {mae_before:.6f} m"
)

print(
    f"RMSE        : {rmse_before:.6f} m"
)

print(
    f"Bias        : {bias_before:.6f} m"
)

print(
    f"Correlation : {corr_before:.6f}"
)

print(
    f"R²          : {r2_before:.6f}"
)


print("\n" + "=" * 78)
print("AFFINE CALIBRATION MODEL")
print("=" * 78)

print(
    "Relationship:"
)

print(
    f"GT ≈ {a:.10f} × Prediction + {b:.10f}"
)

print(
    f"\nScale factor : {a:.10f}"
)

print(
    f"Offset       : {b:.10f} m"
)


print("\n" + "=" * 78)
print("AFTER AFFINE CALIBRATION")
print("=" * 78)

print(
    f"MAE         : {mae_affine:.6f} m"
)

print(
    f"RMSE        : {rmse_affine:.6f} m"
)

print(
    f"Bias        : {bias_affine:.6f} m"
)

print(
    f"Median AE   : {median_ae_affine:.6f} m"
)

print(
    f"P90 AE      : {p90_affine:.6f} m"
)

print(
    f"P95 AE      : {p95_affine:.6f} m"
)

print(
    f"P99 AE      : {p99_affine:.6f} m"
)

print(
    f"Correlation : {corr_affine:.6f}"
)

print(
    f"R²          : {r2_affine:.6f}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

mae_improvement = (
    (mae_before - mae_affine)
    / mae_before
    * 100
)

rmse_improvement = (
    (rmse_before - rmse_affine)
    / rmse_before
    * 100
)


print("\n" + "=" * 78)
print("IMPROVEMENT")
print("=" * 78)

print(
    f"MAE improvement  : {mae_improvement:.2f}%"
)

print(
    f"RMSE improvement : {rmse_improvement:.2f}%"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 78)
print("DIAGNOSTIC INTERPRETATION")
print("=" * 78)

if mae_affine < mae_before:
    print(
        "Affine calibration reduces the error."
    )

    print(
        "This indicates that the external-DEM result"
    )

    print(
        "contains systematic scale and/or offset differences."
    )

else:
    print(
        "Affine calibration does not improve the result."
    )

    print(
        "The remaining error is likely dominated by"
    )

    print(
        "spatial/terrain/surface-definition differences."
    )


print("\nIMPORTANT:")
print(
    "This affine fit is diagnostic only."
)

print(
    "The fitted parameters were derived using Urban3D"
)

print(
    "ground truth and therefore must NOT be treated as"
)

print(
    "a production calibration constant."
)

print(
    "No ASTERRA model or production calibration file"
)

print(
    "has been modified."
)

print("=" * 78)