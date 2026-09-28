import numpy as np
import rasterio
from pathlib import Path
import argparse


# ============================================================
# ASTERRA CALIBRATED DSM VALIDATOR
# ============================================================
# Validates a calibrated metric DSM against a ground-truth DSM.
#
# Current ASTERRA calibration source:
#   AW3D30
#
# Prediction:
#   metric_dsm_aw3d30.tif
#
# Ground Truth:
#   Urban3D / JAX Tile 004 DSM
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Validate ASTERRA calibrated DSM against ground truth."
)

parser.add_argument(
    "--pred",
    required=True,
    help="Path to predicted DSM GeoTIFF"
)

parser.add_argument(
    "--gt",
    required=True,
    help="Path to ground-truth DSM GeoTIFF"
)

args = parser.parse_args()

PRED_PATH = Path(args.pred)
GT_PATH = Path(args.gt)


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

# Small tolerance for floating-point raster metadata comparison.
TRANSFORM_TOLERANCE = 1e-6

# Minimum number of valid pixels required for validation.
MIN_VALID_PIXELS = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def transforms_match(transform_a, transform_b, tolerance=1e-6):
    """
    Compare raster affine transforms using floating-point tolerance.
    """
    return np.allclose(
        np.array(transform_a),
        np.array(transform_b),
        atol=tolerance,
        rtol=0.0
    )


def safe_correlation(a, b):
    """
    Calculate Pearson correlation safely.

    Returns NaN if either array has zero variance.
    """
    if len(a) < 2:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def safe_percentage(numerator, denominator):
    """
    Safely calculate percentage.
    """
    if denominator == 0:
        return np.nan

    return (numerator / denominator) * 100.0


# ============================================================
# FILE CHECK
# ============================================================

print_section("ASTERRA CALIBRATED DSM — GEO-TIFF VALIDATION")

print("\nChecking input files...")

if not PRED_PATH.exists():
    raise FileNotFoundError(
        f"\nPrediction file not found:\n{PRED_PATH}"
    )

if not GT_PATH.exists():
    raise FileNotFoundError(
        f"\nGround-truth file not found:\n{GT_PATH}"
    )

print("Prediction file : FOUND")
print("Ground truth    : FOUND")

print("\nPrediction:")
print(PRED_PATH)

print("\nGround Truth:")
print(GT_PATH)


# ============================================================
# LOAD PREDICTION
# ============================================================

with rasterio.open(PRED_PATH) as pred_src:

    pred = pred_src.read(1).astype(np.float32)

    pred_mask = pred_src.read_masks(1) > 0

    pred_crs = pred_src.crs
    pred_transform = pred_src.transform
    pred_res = pred_src.res
    pred_bounds = pred_src.bounds

    pred_nodata = pred_src.nodata

    pred_width = pred_src.width
    pred_height = pred_src.height

    pred_dtype = pred_src.dtypes[0]


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

with rasterio.open(GT_PATH) as gt_src:

    gt = gt_src.read(1).astype(np.float32)

    gt_mask = gt_src.read_masks(1) > 0

    gt_crs = gt_src.crs
    gt_transform = gt_src.transform
    gt_res = gt_src.res
    gt_bounds = gt_src.bounds

    gt_nodata = gt_src.nodata

    gt_width = gt_src.width
    gt_height = gt_src.height

    gt_dtype = gt_src.dtypes[0]


# ============================================================
# RASTER INFORMATION
# ============================================================

print_section("RASTER INFORMATION")

print("\nPrediction")
print("-" * 40)

print("Shape          :", pred.shape)
print("Width          :", pred_width)
print("Height         :", pred_height)
print("CRS            :", pred_crs)
print("Resolution     :", pred_res)
print("NoData         :", pred_nodata)
print("Data type      :", pred_dtype)
print("Bounds         :", pred_bounds)
print("Transform      :", pred_transform)

print("\nGround Truth")
print("-" * 40)

print("Shape          :", gt.shape)
print("Width          :", gt_width)
print("Height         :", gt_height)
print("CRS            :", gt_crs)
print("Resolution     :", gt_res)
print("NoData         :", gt_nodata)
print("Data type      :", gt_dtype)
print("Bounds         :", gt_bounds)
print("Transform      :", gt_transform)


# ============================================================
# GRID CONSISTENCY CHECK
# ============================================================

print_section("GRID CONSISTENCY CHECK")


# ------------------------------------------------------------
# Shape
# ------------------------------------------------------------

shape_match = pred.shape == gt.shape

print(
    f"Shape          : {'MATCH' if shape_match else 'MISMATCH'}"
)

if not shape_match:

    raise ValueError(
        "\nShape mismatch.\n"
        f"Prediction shape : {pred.shape}\n"
        f"Ground truth     : {gt.shape}\n\n"
        "Prediction and ground truth must be on the same pixel grid "
        "before calculating pixel-wise DSM metrics."
    )


# ------------------------------------------------------------
# CRS
# ------------------------------------------------------------

crs_match = pred_crs == gt_crs

print(
    f"CRS            : {'MATCH' if crs_match else 'MISMATCH'}"
)

if not crs_match:

    raise ValueError(
        "\nCRS mismatch.\n"
        f"Prediction CRS : {pred_crs}\n"
        f"Ground truth   : {gt_crs}\n\n"
        "Reproject one raster before validation."
    )


# ------------------------------------------------------------
# Resolution
# ------------------------------------------------------------

resolution_match = np.allclose(
    pred_res,
    gt_res,
    atol=TRANSFORM_TOLERANCE,
    rtol=0.0
)

print(
    f"Resolution     : "
    f"{'MATCH' if resolution_match else 'MISMATCH'}"
)

if not resolution_match:

    raise ValueError(
        "\nResolution mismatch.\n"
        f"Prediction resolution : {pred_res}\n"
        f"Ground truth          : {gt_res}"
    )


# ------------------------------------------------------------
# Transform
# ------------------------------------------------------------

transform_match = transforms_match(
    pred_transform,
    gt_transform,
    TRANSFORM_TOLERANCE
)

print(
    f"Transform      : "
    f"{'MATCH' if transform_match else 'MISMATCH'}"
)

if not transform_match:

    raise ValueError(
        "\nTransform mismatch.\n"
        f"Prediction transform : {pred_transform}\n"
        f"Ground truth         : {gt_transform}\n\n"
        "The rasters are not aligned on the same pixel grid."
    )


# ------------------------------------------------------------
# Bounds
# ------------------------------------------------------------

bounds_match = np.allclose(
    np.array(pred_bounds),
    np.array(gt_bounds),
    atol=TRANSFORM_TOLERANCE,
    rtol=0.0
)

print(
    f"Bounds         : "
    f"{'MATCH' if bounds_match else 'MISMATCH'}"
)

if not bounds_match:

    raise ValueError(
        "\nSpatial bounds mismatch.\n"
        f"Prediction bounds : {pred_bounds}\n"
        f"Ground truth      : {gt_bounds}"
    )


print("\nGrid status    : PASS")


# ============================================================
# VALID PIXEL MASK
# ============================================================

print_section("VALID PIXEL MASK")


valid = (
    np.isfinite(pred)
    & np.isfinite(gt)
    & pred_mask
    & gt_mask
)


# ------------------------------------------------------------
# Prediction NoData
# ------------------------------------------------------------

if pred_nodata is not None:

    valid &= ~np.isclose(
        pred,
        pred_nodata,
        equal_nan=False
    )


# ------------------------------------------------------------
# Ground Truth NoData
# ------------------------------------------------------------

if gt_nodata is not None:

    valid &= ~np.isclose(
        gt,
        gt_nodata,
        equal_nan=False
    )


# ============================================================
# EXTRACT VALID PIXELS
# ============================================================

pred_valid = pred[valid]
gt_valid = gt[valid]


total_pixels = gt.size
valid_pixels = int(valid.sum())

invalid_pixels = total_pixels - valid_pixels

valid_percent = safe_percentage(
    valid_pixels,
    total_pixels
)

invalid_percent = safe_percentage(
    invalid_pixels,
    total_pixels
)


print("Total pixels   :", total_pixels)
print("Valid pixels   :", valid_pixels)
print("Invalid pixels :", invalid_pixels)

print(
    f"Valid %        : {valid_percent:.6f}%"
)

print(
    f"Invalid %      : {invalid_percent:.6f}%"
)


if valid_pixels < MIN_VALID_PIXELS:

    raise ValueError(
        "\nNo valid overlapping pixels found between "
        "prediction and ground truth."
    )


# ============================================================
# BASIC VALID DATA STATISTICS
# ============================================================

print_section("VALID DATA STATISTICS")

print("\nGround Truth")
print("-" * 40)

print(f"Min            : {gt_valid.min():.6f} m")
print(f"Max            : {gt_valid.max():.6f} m")
print(f"Mean           : {gt_valid.mean():.6f} m")
print(f"Median         : {np.median(gt_valid):.6f} m")
print(f"Std            : {gt_valid.std():.6f} m")

print("\nPrediction")
print("-" * 40)

print(f"Min            : {pred_valid.min():.6f} m")
print(f"Max            : {pred_valid.max():.6f} m")
print(f"Mean           : {pred_valid.mean():.6f} m")
print(f"Median         : {np.median(pred_valid):.6f} m")
print(f"Std            : {pred_valid.std():.6f} m")


# ============================================================
# ERROR CALCULATION
# ============================================================

error = pred_valid - gt_valid

abs_error = np.abs(error)

squared_error = error ** 2


# ============================================================
# STANDARD DSM METRICS
# ============================================================

mae = float(
    np.mean(abs_error)
)

rmse = float(
    np.sqrt(np.mean(squared_error))
)

bias = float(
    np.mean(error)
)

median_ae = float(
    np.median(abs_error)
)

p90 = float(
    np.percentile(abs_error, 90)
)

p95 = float(
    np.percentile(abs_error, 95)
)

p99 = float(
    np.percentile(abs_error, 99)
)


# ============================================================
# CORRELATION
# ============================================================

corr = safe_correlation(
    pred_valid,
    gt_valid
)


# ============================================================
# R²
# ============================================================

ss_res = float(
    np.sum(
        (gt_valid - pred_valid) ** 2
    )
)

ss_tot = float(
    np.sum(
        (gt_valid - np.mean(gt_valid)) ** 2
    )
)

if ss_tot == 0:

    r2 = np.nan

else:

    r2 = 1.0 - (
        ss_res / ss_tot
    )


# ============================================================
# ERROR STANDARD DEVIATION
# ============================================================

error_std = float(
    np.std(error)
)


# ============================================================
# BASELINE
# ============================================================
# Zero-height prediction baseline.
#
# This asks:
#
# "How much error would there be if every pixel were predicted
# as elevation = 0?"
#
# It is NOT a terrain-model baseline.
# It is simply a sanity/reference baseline.
# ============================================================

zero_error = gt_valid

zero_abs_error = np.abs(
    zero_error
)

zero_mae = float(
    np.mean(zero_abs_error)
)

zero_rmse = float(
    np.sqrt(
        np.mean(zero_error ** 2)
    )
)


# ============================================================
# IMPROVEMENT OVER ZERO BASELINE
# ============================================================

if zero_mae != 0:

    mae_improvement = (
        (zero_mae - mae)
        / zero_mae
    ) * 100.0

else:

    mae_improvement = np.nan


if zero_rmse != 0:

    rmse_improvement = (
        (zero_rmse - rmse)
        / zero_rmse
    ) * 100.0

else:

    rmse_improvement = np.nan


# ============================================================
# FINAL METRICS
# ============================================================

print_section("CALIBRATED DSM RESULTS")

print(f"MAE            : {mae:.6f} m")
print(f"RMSE           : {rmse:.6f} m")
print(f"Bias           : {bias:.6f} m")
print(f"Error Std      : {error_std:.6f} m")
print(f"Median AE      : {median_ae:.6f} m")
print(f"P90 AE         : {p90:.6f} m")
print(f"P95 AE         : {p95:.6f} m")
print(f"P99 AE         : {p99:.6f} m")
print(f"Correlation    : {corr:.6f}")
print(f"R²             : {r2:.6f}")


# ============================================================
# BASELINE RESULTS
# ============================================================

print_section("ZERO-HEIGHT BASELINE")

print(
    f"Zero baseline MAE  : "
    f"{zero_mae:.6f} m"
)

print(
    f"Zero baseline RMSE : "
    f"{zero_rmse:.6f} m"
)


# ============================================================
# IMPROVEMENT
# ============================================================

print_section("IMPROVEMENT OVER ZERO-HEIGHT BASELINE")

if np.isfinite(mae_improvement):

    print(
        f"MAE improvement  : "
        f"{mae_improvement:.2f}%"
    )

else:

    print(
        "MAE improvement  : N/A"
    )


if np.isfinite(rmse_improvement):

    print(
        f"RMSE improvement : "
        f"{rmse_improvement:.2f}%"
    )

else:

    print(
        "RMSE improvement : N/A"
    )


# ============================================================
# RANGE COMPARISON
# ============================================================

print_section("RANGE COMPARISON")

print(
    f"GT min          : "
    f"{gt_valid.min():.6f} m"
)

print(
    f"GT max          : "
    f"{gt_valid.max():.6f} m"
)

print(
    f"Prediction min  : "
    f"{pred_valid.min():.6f} m"
)

print(
    f"Prediction max  : "
    f"{pred_valid.max():.6f} m"
)

print(
    f"GT mean         : "
    f"{gt_valid.mean():.6f} m"
)

print(
    f"Prediction mean : "
    f"{pred_valid.mean():.6f} m"
)

print(
    f"GT std          : "
    f"{gt_valid.std():.6f} m"
)

print(
    f"Prediction std  : "
    f"{pred_valid.std():.6f} m"
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

print_section("ERROR DISTRIBUTION")

print(
    f"Mean signed error : "
    f"{error.mean():.6f} m"
)

print(
    f"Mean absolute error : "
    f"{abs_error.mean():.6f} m"
)

print(
    f"Maximum absolute error : "
    f"{abs_error.max():.6f} m"
)


# ============================================================
# FINAL STATUS
# ============================================================

print_section("VALIDATION STATUS")

print("Raster alignment : PASS")
print("Valid pixels     : PASS")
print("Metric computation: PASS")

print("\nASTERRA CALIBRATED DSM VALIDATION COMPLETE")

print("=" * 78)