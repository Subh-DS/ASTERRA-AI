from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# ASTERRA AI
# FINAL FROZEN HEIGHT MODEL VALIDATION
# ============================================================

ROOT = Path(r"D:\Asterra AI")


# ------------------------------------------------------------
# Input / output paths
# ------------------------------------------------------------

PREDICTION = (
    ROOT
    / "outputs"
    / "asterra_final_inference"
    / "JAX_Tile_004_nDSM.npy"
)

# IMPORTANT:
# Urban3D stores RGB, DSM and DTM together under Inputs.
RGB_PATH = (
    ROOT
    / "datasets"
    / "Urban3D"
    / "train"
    / "Inputs"
    / "JAX_Tile_004_RGB.tif"
)

DSM_PATH = (
    ROOT
    / "datasets"
    / "Urban3D"
    / "train"
    / "Inputs"
    / "JAX_Tile_004_DSM.tif"
)

DTM_PATH = (
    ROOT
    / "datasets"
    / "Urban3D"
    / "train"
    / "Inputs"
    / "JAX_Tile_004_DTM.tif"
)


# ============================================================
# Raster loader
# ============================================================

def load_raster(path):

    if not path.exists():
        raise FileNotFoundError(
            f"\nRaster file not found:\n{path}"
        )

    with rasterio.open(path) as src:

        data = src.read(1).astype(
            np.float32
        )

        profile = src.profile

        shape = src.shape

        transform = src.transform

        crs = src.crs

    return (
        data,
        profile,
        shape,
        transform,
        crs,
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)
    print("ASTERRA FINAL HEIGHT MODEL — FROZEN CHECK")
    print("=" * 78)


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print()
    print("Prediction:")
    print(PREDICTION)

    print()
    print("RGB:")
    print(RGB_PATH)

    print()
    print("DSM:")
    print(DSM_PATH)

    print()
    print("DTM:")
    print(DTM_PATH)


    # --------------------------------------------------------
    # File existence check
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FILE CHECK")
    print("=" * 78)

    for name, path in [
        ("Prediction", PREDICTION),
        ("RGB", RGB_PATH),
        ("DSM", DSM_PATH),
        ("DTM", DTM_PATH),
    ]:

        exists = path.exists()

        print(
            f"{name:<12}: "
            f"{'OK' if exists else 'MISSING'}"
        )

        if not exists:

            raise FileNotFoundError(
                f"\nRequired file does not exist:\n{path}"
            )


    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    pred = np.load(
        PREDICTION
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Load RGB metadata
    # --------------------------------------------------------

    with rasterio.open(RGB_PATH) as src:

        rgb_shape = src.shape

        rgb_transform = src.transform

        rgb_crs = src.crs

        rgb_width = src.width

        rgb_height = src.height


    # --------------------------------------------------------
    # Load DSM / DTM
    # --------------------------------------------------------

    (
        dsm,
        dsm_profile,
        dsm_shape,
        dsm_transform,
        dsm_crs,
    ) = load_raster(
        DSM_PATH
    )

    (
        dtm,
        dtm_profile,
        dtm_shape,
        dtm_transform,
        dtm_crs,
    ) = load_raster(
        DTM_PATH
    )


    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("RASTER SHAPES")
    print("=" * 78)

    print(
        "Prediction:",
        pred.shape
    )

    print(
        "RGB       :",
        rgb_shape
    )

    print(
        "DSM       :",
        dsm.shape
    )

    print(
        "DTM       :",
        dtm.shape
    )


    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if pred.shape != dsm.shape:

        raise RuntimeError(
            "\nPrediction / DSM shape mismatch:\n"
            f"Prediction: {pred.shape}\n"
            f"DSM       : {dsm.shape}"
        )

    if dsm.shape != dtm.shape:

        raise RuntimeError(
            "\nDSM / DTM shape mismatch:\n"
            f"DSM: {dsm.shape}\n"
            f"DTM: {dtm.shape}"
        )

    if rgb_shape != dsm.shape:

        print()
        print(
            "WARNING:"
        )

        print(
            "RGB dimensions differ from DSM/DTM."
        )

        print(
            "RGB:",
            rgb_shape
        )

        print(
            "DSM:",
            dsm.shape
        )

        print(
            "This is not automatically an error,"
            " but should be checked."
        )


    # --------------------------------------------------------
    # Geospatial metadata check
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("GEOSPATIAL METADATA")
    print("=" * 78)

    print(
        "RGB CRS:",
        rgb_crs
    )

    print(
        "DSM CRS:",
        dsm_crs
    )

    print(
        "DTM CRS:",
        dtm_crs
    )


    # --------------------------------------------------------
    # Ground truth nDSM
    # --------------------------------------------------------
    #
    # Urban3D target:
    #
    #       nDSM = DSM - DTM
    #
    # Negative heights are clipped to zero.
    #
    # --------------------------------------------------------

    gt = (
        dsm
        -
        dtm
    )

    gt = np.maximum(
        gt,
        0.0,
    )


    # --------------------------------------------------------
    # Valid pixels
    # --------------------------------------------------------

    valid = (
        np.isfinite(pred)
        &
        np.isfinite(gt)
        &
        np.isfinite(dsm)
        &
        np.isfinite(dtm)
    )

    valid_count = int(
        np.count_nonzero(valid)
    )

    total_pixels = int(
        pred.size
    )

    valid_fraction = (
        valid_count
        /
        total_pixels
        if total_pixels > 0
        else 0.0
    )

    print()
    print("=" * 78)
    print("VALID PIXELS")
    print("=" * 78)

    print(
        f"Total pixels   : {total_pixels}"
    )

    print(
        f"Valid pixels   : {valid_count}"
    )

    print(
        f"Valid fraction : {valid_fraction:.6f}"
    )

    print(
        f"Valid percent  : {valid_fraction * 100:.4f}%"
    )


    if valid_count == 0:

        raise RuntimeError(
            "No valid pixels available for evaluation."
        )


    # --------------------------------------------------------
    # Flatten valid pixels
    # --------------------------------------------------------

    p = pred[valid].astype(
        np.float64
    )

    y = gt[valid].astype(
        np.float64
    )


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error = (
        p
        -
        y
    )

    abs_error = np.abs(
        error
    )


    # MAE

    mae = np.mean(
        abs_error
    )


    # RMSE

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )


    # Bias

    bias = np.mean(
        error
    )


    # Median Absolute Error

    median_ae = np.median(
        abs_error
    )


    # P90 Absolute Error

    p90_ae = np.percentile(
        abs_error,
        90,
    )


    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    if (
        np.std(p) > 0
        and
        np.std(y) > 0
    ):

        corr = np.corrcoef(
            p,
            y,
        )[0, 1]

    else:

        corr = np.nan


    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    ss_res = np.sum(
        (y - p) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:

        r2 = (
            1.0
            -
            (
                ss_res
                /
                ss_tot
            )
        )

    else:

        r2 = np.nan


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    pred_mean = np.mean(
        p
    )

    pred_std = np.std(
        p
    )

    gt_mean = np.mean(
        y
    )

    gt_std = np.std(
        y
    )

    if gt_std > 0:

        std_ratio = (
            pred_std
            /
            gt_std
        )

    else:

        std_ratio = np.nan


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL FROZEN MODEL RESULTS")
    print("=" * 78)

    print(
        f"MAE             : {mae:.6f} m"
    )

    print(
        f"RMSE            : {rmse:.6f} m"
    )

    print(
        f"Bias            : {bias:.6f} m"
    )

    print(
        f"Median AE       : {median_ae:.6f} m"
    )

    print(
        f"P90 AE          : {p90_ae:.6f} m"
    )

    print(
        f"Correlation     : {corr:.6f}"
    )

    print(
        f"R²              : {r2:.6f}"
    )


    # --------------------------------------------------------
    # Distribution statistics
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("DISTRIBUTION STATISTICS")
    print("=" * 78)

    print(
        f"Prediction mean : {pred_mean:.6f} m"
    )

    print(
        f"Prediction std  : {pred_std:.6f} m"
    )

    print(
        f"GT mean         : {gt_mean:.6f} m"
    )

    print(
        f"GT std          : {gt_std:.6f} m"
    )

    print(
        f"Std ratio       : {std_ratio:.6f}"
    )


    # --------------------------------------------------------
    # Ground truth range
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("GROUND-TRUTH RANGE")
    print("=" * 78)

    print(
        f"GT min  : {y.min():.6f} m"
    )

    print(
        f"GT max  : {y.max():.6f} m"
    )

    print(
        f"GT mean : {y.mean():.6f} m"
    )

    print(
        f"GT P95  : {np.percentile(y, 95):.6f} m"
    )

    print(
        f"GT P99  : {np.percentile(y, 99):.6f} m"
    )


    # --------------------------------------------------------
    # Prediction range
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("PREDICTION RANGE")
    print("=" * 78)

    print(
        f"Pred min  : {p.min():.6f} m"
    )

    print(
        f"Pred max  : {p.max():.6f} m"
    )

    print(
        f"Pred mean : {p.mean():.6f} m"
    )

    print(
        f"Pred P95  : {np.percentile(p, 95):.6f} m"
    )

    print(
        f"Pred P99  : {np.percentile(p, 99):.6f} m"
    )


    # --------------------------------------------------------
    # Zero baseline
    # --------------------------------------------------------
    #
    # Useful reference:
    # predicting zero height everywhere.
    #
    # --------------------------------------------------------

    zero_prediction = np.zeros_like(
        y
    )

    zero_error = (
        zero_prediction
        -
        y
    )

    zero_mae = np.mean(
        np.abs(zero_error)
    )

    zero_rmse = np.sqrt(
        np.mean(
            zero_error ** 2
        )
    )


    print()
    print("=" * 78)
    print("ZERO-HEIGHT BASELINE")
    print("=" * 78)

    print(
        f"Zero baseline MAE  : {zero_mae:.6f} m"
    )

    print(
        f"Zero baseline RMSE : {zero_rmse:.6f} m"
    )


    # --------------------------------------------------------
    # Improvement over zero baseline
    # --------------------------------------------------------

    if zero_mae > 0:

        mae_improvement = (
            1.0
            -
            (
                mae
                /
                zero_mae
            )
        ) * 100.0

    else:

        mae_improvement = np.nan


    if zero_rmse > 0:

        rmse_improvement = (
            1.0
            -
            (
                rmse
                /
                zero_rmse
            )
        ) * 100.0

    else:

        rmse_improvement = np.nan


    print()
    print("=" * 78)
    print("IMPROVEMENT OVER ZERO BASELINE")
    print("=" * 78)

    print(
        f"MAE improvement  : {mae_improvement:.2f}%"
    )

    print(
        f"RMSE improvement : {rmse_improvement:.2f}%"
    )


    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FROZEN MODEL CHECK COMPLETE")
    print("=" * 78)

    print()
    print(
        "Checkpoint evaluated:"
    )

    print(
        ROOT
        / "models"
        / "asterra_stage5"
        / "urban3d_v3_1"
        / "ASTERRA_FINAL_HEIGHT_MODEL.pth"
    )

    print()
    print(
        "Target:"
    )

    print(
        "nDSM = max(DSM - DTM, 0)"
    )

    print()
    print(
        "No model weights were modified during this validation."
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()