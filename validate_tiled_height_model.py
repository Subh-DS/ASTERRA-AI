from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(r"D:\Asterra AI")

PREDICTION = (
    ROOT
    / "outputs"
    / "asterra_tiled_inference"
    / "JAX_Tile_004_nDSM_tiled.npy"
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


def load_raster(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Raster not found:\n{path}"
        )

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        shape = src.shape
        crs = src.crs

    return data, shape, crs


def main():

    print()
    print("=" * 78)
    print("ASTERRA TILED HEIGHT MODEL — FROZEN VALIDATION")
    print("=" * 78)

    print()
    print("Prediction:")
    print(PREDICTION)

    print()
    print("DSM:")
    print(DSM_PATH)

    print()
    print("DTM:")
    print(DTM_PATH)

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    if not PREDICTION.exists():
        raise FileNotFoundError(
            f"Prediction not found:\n{PREDICTION}"
        )

    pred = np.load(
        PREDICTION
    ).astype(np.float32)

    # --------------------------------------------------------
    # Load DSM / DTM
    # --------------------------------------------------------

    dsm, dsm_shape, dsm_crs = load_raster(
        DSM_PATH
    )

    dtm, dtm_shape, dtm_crs = load_raster(
        DTM_PATH
    )

    print()
    print("=" * 78)
    print("SHAPES")
    print("=" * 78)

    print("Prediction:", pred.shape)
    print("DSM       :", dsm_shape)
    print("DTM       :", dtm_shape)

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if pred.shape != dsm.shape:
        raise RuntimeError(
            f"Prediction/DSM shape mismatch:\n"
            f"Prediction: {pred.shape}\n"
            f"DSM: {dsm.shape}"
        )

    if dsm.shape != dtm.shape:
        raise RuntimeError(
            f"DSM/DTM shape mismatch:\n"
            f"DSM: {dsm.shape}\n"
            f"DTM: {dtm.shape}"
        )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CRS")
    print("=" * 78)

    print("DSM CRS:", dsm_crs)
    print("DTM CRS:", dtm_crs)

    # --------------------------------------------------------
    # Ground truth
    #
    # Urban3D target:
    # nDSM = max(DSM - DTM, 0)
    # --------------------------------------------------------

    gt = dsm - dtm

    gt = np.maximum(
        gt,
        0.0
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

    p = pred[valid].astype(
        np.float64
    )

    y = gt[valid].astype(
        np.float64
    )

    print()
    print("=" * 78)
    print("VALID PIXELS")
    print("=" * 78)

    print(
        "Total:",
        pred.size
    )

    print(
        "Valid:",
        len(p)
    )

    print(
        "Valid %:",
        f"{100.0 * len(p) / pred.size:.4f}%"
    )

    if len(p) == 0:
        raise RuntimeError(
            "No valid pixels."
        )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    error = p - y

    abs_error = np.abs(
        error
    )

    mae = np.mean(
        abs_error
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    bias = np.mean(
        error
    )

    median_ae = np.median(
        abs_error
    )

    p90_ae = np.percentile(
        abs_error,
        90
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
            y
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
            ss_res / ss_tot
        )

    else:

        r2 = np.nan

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    pred_mean = np.mean(p)
    pred_std = np.std(p)

    gt_mean = np.mean(y)
    gt_std = np.std(y)

    std_ratio = (
        pred_std / gt_std
        if gt_std > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Zero baseline
    # --------------------------------------------------------

    zero_mae = np.mean(
        np.abs(y)
    )

    zero_rmse = np.sqrt(
        np.mean(
            y ** 2
        )
    )

    mae_improvement = (
        1.0 - mae / zero_mae
    ) * 100.0 if zero_mae > 0 else np.nan

    rmse_improvement = (
        1.0 - rmse / zero_rmse
    ) * 100.0 if zero_rmse > 0 else np.nan

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("TILED FROZEN MODEL RESULTS")
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

    print()
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

    print()
    print("=" * 78)
    print("RANGE COMPARISON")
    print("=" * 78)

    print(
        f"GT max    : {y.max():.6f} m"
    )

    print(
        f"Pred max  : {p.max():.6f} m"
    )

    print(
        f"GT P95    : {np.percentile(y, 95):.6f} m"
    )

    print(
        f"Pred P95  : {np.percentile(p, 95):.6f} m"
    )

    print(
        f"GT P99    : {np.percentile(y, 99):.6f} m"
    )

    print(
        f"Pred P99  : {np.percentile(p, 99):.6f} m"
    )

    print()
    print("=" * 78)
    print("ZERO-HEIGHT BASELINE")
    print("=" * 78)

    print(
        f"Baseline MAE  : {zero_mae:.6f} m"
    )

    print(
        f"Baseline RMSE : {zero_rmse:.6f} m"
    )

    print()
    print(
        f"MAE improvement  : {mae_improvement:.2f}%"
    )

    print(
        f"RMSE improvement : {rmse_improvement:.2f}%"
    )

    print()
    print("=" * 78)
    print("TILED VALIDATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()