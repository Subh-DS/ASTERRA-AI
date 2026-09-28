import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image


# ============================================================
# ASTERRA
# ============================================================

ROOT = Path(r"D:\Asterra AI")

sys.path.insert(
    0,
    str(ROOT / "external" / "Depth-Anything-V2")
)

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# PATHS
# ============================================================

DATASET = (
    ROOT
    / "datasets"
    / "S_EO_Stage2_Diverse"
)

CHECKPOINT = (
    ROOT
    / "models"
    / "asterra_seo_v3_1_transfer"
    / "seo_best.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "seo_validation_maps"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_SIZE = 518


# ============================================================
# MODEL
# ============================================================

def build_model():

    print()
    print("Building ASTERRA Stage-2 model...")

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[
            256,
            512,
            1024,
            1024,
        ],
    )

    # S-EO Stage-2 uses unrestricted linear output.
    model.depth_head.scratch.output_conv2[3] = nn.Identity()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state = checkpoint["model"]

    else:
        state = checkpoint

    incompatible = model.load_state_dict(
        state,
        strict=False,
    )

    if incompatible.missing_keys:
        raise RuntimeError(
            "Missing checkpoint keys:\n"
            + "\n".join(
                incompatible.missing_keys[:20]
            )
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys:\n"
            + "\n".join(
                incompatible.unexpected_keys[:20]
            )
        )

    print("Checkpoint loaded successfully.")
    print("Missing keys    : 0")
    print("Unexpected keys : 0")

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# RGB LOADER
# ============================================================

def load_rgb(path):

    img = np.asarray(
        Image.open(path).convert("RGB"),
        dtype=np.float32,
    )

    original_h = img.shape[0]
    original_w = img.shape[1]

    if img.max() > 1.5:
        img /= 255.0

    img = np.clip(
        img,
        0.0,
        1.0,
    )

    pil_img = Image.fromarray(
        (img * 255.0).astype(
            np.uint8
        )
    )

    # 518 = 37 * 14
    pil_img = pil_img.resize(
        (
            MODEL_SIZE,
            MODEL_SIZE,
        ),
        Image.Resampling.BICUBIC,
    )

    img = np.asarray(
        pil_img,
        dtype=np.float32,
    ) / 255.0

    img = np.clip(
        img,
        0.0,
        1.0,
    )

    tensor = torch.from_numpy(
        img
    ).permute(
        2,
        0,
        1,
    ).unsqueeze(0)

    return (
        tensor.to(DEVICE),
        original_h,
        original_w,
    )


# ============================================================
# LOAD TARGET
# ============================================================

def load_target(path):

    return np.load(
        path
    ).astype(
        np.float32
    )


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(path):

    return np.load(
        path
    ).astype(bool)


# ============================================================
# PERCENTILE STRETCH
# ============================================================

def normalize_for_png(
    array,
    mask=None,
    low=2,
    high=98,
):

    arr = np.asarray(
        array,
        dtype=np.float32,
    )

    if mask is None:
        valid = np.isfinite(arr)
    else:
        valid = (
            mask
            & np.isfinite(arr)
        )

    if not np.any(valid):
        return np.zeros(
            arr.shape,
            dtype=np.uint8,
        )

    values = arr[valid]

    lo = np.percentile(
        values,
        low,
    )

    hi = np.percentile(
        values,
        high,
    )

    if hi <= lo:

        result = np.zeros(
            arr.shape,
            dtype=np.uint8,
        )

        result[valid] = 128

        return result

    normalized = (
        (arr - lo)
        / (hi - lo)
        * 255.0
    )

    normalized = np.clip(
        normalized,
        0,
        255,
    )

    normalized[~valid] = 0

    return normalized.astype(
        np.uint8
    )


# ============================================================
# SAVE FLOAT ARRAY
# ============================================================

def save_npy(
    path,
    array,
):

    np.save(
        path,
        array.astype(
            np.float32
        ),
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    pred,
    gt,
    mask,
):

    valid = (
        mask
        & np.isfinite(pred)
        & np.isfinite(gt)
    )

    p = pred[valid].astype(
        np.float64
    )

    y = gt[valid].astype(
        np.float64
    )

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

    pred_std = np.std(p)
    gt_std = np.std(y)

    if (
        pred_std > 0
        and gt_std > 0
    ):

        corr = np.corrcoef(
            p,
            y,
        )[0, 1]

    else:

        corr = np.nan

    if gt_std > 0:

        std_ratio = (
            pred_std
            / gt_std
        )

    else:

        std_ratio = np.nan

    # --------------------------------------------------------
    # Within-crop R²
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
            - ss_res
            / ss_tot
        )

    else:

        r2 = np.nan

    return {
        "valid_pixels": int(
            len(p)
        ),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "corr": float(corr),
        "r2": float(r2),
        "pred_mean": float(
            np.mean(p)
        ),
        "pred_std": float(
            pred_std
        ),
        "gt_mean": float(
            np.mean(y)
        ),
        "gt_std": float(
            gt_std
        ),
        "std_ratio": float(
            std_ratio
        ),
        "median_ae": float(
            np.median(
                abs_error
            )
        ),
        "p90_ae": float(
            np.percentile(
                abs_error,
                90,
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "ASTERRA — S-EO VALIDATION "
        "SPATIAL DIAGNOSTIC"
    )
    print("=" * 78)

    print()
    print("Device    :", DEVICE)
    print("Checkpoint:", CHECKPOINT)
    print("Dataset   :", DATASET)
    print("Output    :", OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        DATASET / "manifest.csv"
    )

    val = manifest[
        manifest["split"]
        .astype(str)
        .str.lower()
        == "validation"
    ].copy()

    print()
    print(
        "Validation samples:",
        len(val),
    )

    print(
        "Validation AOIs:",
        sorted(
            val["aoi"]
            .astype(str)
            .unique()
        ),
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = build_model()

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # INFERENCE
    # ========================================================

    print()
    print("-" * 78)
    print(
        "GENERATING PREDICTION / GT / ERROR MAPS"
    )
    print("-" * 78)

    with torch.no_grad():

        for i, row in (
            val
            .reset_index(drop=True)
            .iterrows()
        ):

            sample_id = str(
                row["id"]
            )

            aoi = str(
                row["aoi"]
            )

            print()
            print(
                f"[{i + 1:02d}/{len(val)}] "
                f"{sample_id}"
            )

            image_path = (
                DATASET
                / row["image"]
            )

            target_path = (
                DATASET
                / row["target"]
            )

            mask_path = (
                DATASET
                / row["mask"]
            )

            # ------------------------------------------------
            # RGB
            # ------------------------------------------------

            (
                rgb,
                original_h,
                original_w,
            ) = load_rgb(
                image_path
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            pred = model(
                rgb
            )

            if isinstance(
                pred,
                (tuple, list),
            ):

                pred = pred[0]

            pred = pred.squeeze()

            if pred.ndim != 2:

                raise RuntimeError(
                    f"Unexpected prediction "
                    f"shape for {sample_id}: "
                    f"{tuple(pred.shape)}"
                )

            # ------------------------------------------------
            # 518 -> original crop dimensions
            # ------------------------------------------------

            if (
                pred.shape[0]
                != original_h
                or
                pred.shape[1]
                != original_w
            ):

                pred = (
                    F.interpolate(
                        pred
                        .unsqueeze(0)
                        .unsqueeze(0),
                        size=(
                            original_h,
                            original_w,
                        ),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(
                        0,
                        1,
                    )
                )

            pred = (
                pred
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            # ------------------------------------------------
            # GT / MASK
            # ------------------------------------------------

            gt = load_target(
                target_path
            )

            mask = load_mask(
                mask_path
            )

            # ------------------------------------------------
            # SHAPE VALIDATION
            # ------------------------------------------------

            if pred.shape != gt.shape:

                raise RuntimeError(
                    f"Prediction/GT shape mismatch "
                    f"for {sample_id}: "
                    f"pred={pred.shape}, "
                    f"gt={gt.shape}"
                )

            if mask.shape != gt.shape:

                raise RuntimeError(
                    f"Mask/GT shape mismatch "
                    f"for {sample_id}: "
                    f"mask={mask.shape}, "
                    f"gt={gt.shape}"
                )

            valid = (
                mask
                & np.isfinite(pred)
                & np.isfinite(gt)
            )

            if not np.any(valid):

                raise RuntimeError(
                    f"No valid pixels for "
                    f"{sample_id}"
                )

            # ------------------------------------------------
            # ERROR MAP
            # ------------------------------------------------

            error = (
                pred - gt
            )

            abs_error = np.abs(
                error
            )

            # ------------------------------------------------
            # MASK INVALID PIXELS
            # ------------------------------------------------

            pred_masked = pred.copy()
            gt_masked = gt.copy()
            error_masked = error.copy()
            abs_error_masked = (
                abs_error.copy()
            )

            pred_masked[
                ~valid
            ] = np.nan

            gt_masked[
                ~valid
            ] = np.nan

            error_masked[
                ~valid
            ] = np.nan

            abs_error_masked[
                ~valid
            ] = np.nan

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            m = compute_metrics(
                pred,
                gt,
                valid,
            )

            results.append(
                {
                    "id":
                        sample_id,

                    "aoi":
                        aoi,

                    **m,
                }
            )

            # =================================================
            # SAMPLE OUTPUT DIRECTORY
            # =================================================

            sample_dir = (
                OUTPUT_DIR
                / sample_id
            )

            sample_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ------------------------------------------------
            # Save numerical arrays
            # ------------------------------------------------

            save_npy(
                sample_dir
                / "prediction.npy",
                pred_masked,
            )

            save_npy(
                sample_dir
                / "ground_truth.npy",
                gt_masked,
            )

            save_npy(
                sample_dir
                / "error.npy",
                error_masked,
            )

            save_npy(
                sample_dir
                / "absolute_error.npy",
                abs_error_masked,
            )

            save_npy(
                sample_dir
                / "mask.npy",
                valid.astype(
                    np.uint8
                ),
            )

            # ------------------------------------------------
            # Save visualization PNGs
            # ------------------------------------------------

            pred_png = normalize_for_png(
                pred,
                valid,
            )

            gt_png = normalize_for_png(
                gt,
                valid,
            )

            abs_error_png = normalize_for_png(
                abs_error,
                valid,
            )

            # Signed error visualization:
            #
            # negative -> 0
            # zero     -> 127
            # positive -> 255
            #
            # percentile range prevents extreme pixels
            # from dominating visualization.
            # ------------------------------------------------

            error_values = error[
                valid
            ]

            err_lo = np.percentile(
                error_values,
                2,
            )

            err_hi = np.percentile(
                error_values,
                98,
            )

            if err_hi > err_lo:

                error_vis = (
                    (
                        error
                        - err_lo
                    )
                    / (
                        err_hi
                        - err_lo
                    )
                    * 255.0
                )

            else:

                error_vis = (
                    np.ones_like(
                        error
                    )
                    * 127.0
                )

            error_vis = np.clip(
                error_vis,
                0,
                255,
            )

            error_vis[
                ~valid
            ] = 0

            error_png = (
                error_vis
                .astype(
                    np.uint8
                )
            )

            Image.fromarray(
                pred_png,
                mode="L",
            ).save(
                sample_dir
                / "prediction.png"
            )

            Image.fromarray(
                gt_png,
                mode="L",
            ).save(
                sample_dir
                / "ground_truth.png"
            )

            Image.fromarray(
                error_png,
                mode="L",
            ).save(
                sample_dir
                / "error_signed.png"
            )

            Image.fromarray(
                abs_error_png,
                mode="L",
            ).save(
                sample_dir
                / "absolute_error.png"
            )

            # ------------------------------------------------
            # RGB copy
            # ------------------------------------------------

            Image.open(
                image_path
            ).convert(
                "RGB"
            ).save(
                sample_dir
                / "rgb.png"
            )

            # ------------------------------------------------
            # Save sample metrics
            # ------------------------------------------------

            with open(
                sample_dir / "metrics.txt",
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    f"ID: {sample_id}\n"
                )

                f.write(
                    f"AOI: {aoi}\n\n"
                )

                for key, value in m.items():

                    f.write(
                        f"{key}: {value}\n"
                    )

            print(
                f"    MAE       : "
                f"{m['mae']:.4f}"
            )

            print(
                f"    RMSE      : "
                f"{m['rmse']:.4f}"
            )

            print(
                f"    Bias      : "
                f"{m['bias']:.4f}"
            )

            print(
                f"    Corr      : "
                f"{m['corr']:.4f}"
            )

            print(
                f"    R2        : "
                f"{m['r2']:.4f}"
            )

            print(
                f"    Std ratio : "
                f"{m['std_ratio']:.4f}"
            )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("=" * 78)

    print(
        "GLOBAL VALIDATION SUMMARY"
    )

    print("=" * 78)

    print(
        df[
            [
                "mae",
                "rmse",
                "bias",
                "corr",
                "r2",
                "pred_mean",
                "gt_mean",
                "pred_std",
                "gt_std",
                "std_ratio",
            ]
        ].mean(
            numeric_only=True
        ).to_string()
    )

    # ========================================================
    # AOI SUMMARY
    # ========================================================

    aoi_summary = (
        df
        .groupby("aoi")
        .agg(
            samples=(
                "id",
                "count",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_bias=(
                "bias",
                "mean",
            ),
            mean_corr=(
                "corr",
                "mean",
            ),
            mean_r2=(
                "r2",
                "mean",
            ),
            mean_pred=(
                "pred_mean",
                "mean",
            ),
            mean_gt=(
                "gt_mean",
                "mean",
            ),
            mean_std_ratio=(
                "std_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 78)

    print(
        "PER-AOI VALIDATION SUMMARY"
    )

    print("=" * 78)

    print(
        aoi_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    sample_csv = (
        OUTPUT_DIR
        / "validation_sample_metrics.csv"
    )

    aoi_csv = (
        OUTPUT_DIR
        / "validation_aoi_metrics.csv"
    )

    df.to_csv(
        sample_csv,
        index=False,
    )

    aoi_summary.to_csv(
        aoi_csv,
        index=False,
    )

    # ========================================================
    # SAVE TEXT REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "validation_diagnostic_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ASTERRA S-EO VALIDATION "
            "SPATIAL DIAGNOSTIC\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            f"Checkpoint: {CHECKPOINT}\n"
        )

        f.write(
            f"Dataset: {DATASET}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Model input: "
            f"{MODEL_SIZE}x{MODEL_SIZE}\n\n"
        )

        f.write(
            "PER-SAMPLE RESULTS\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            df.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "PER-AOI RESULTS\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            aoi_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "INTERPRETATION NOTES\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "1. Prediction maps show model output "
            "spatial structure.\n"
        )

        f.write(
            "2. Ground-truth maps show S-EO DSM-Max.\n"
        )

        f.write(
            "3. Signed error maps show whether the "
            "model over- or under-predicts.\n"
        )

        f.write(
            "4. Absolute-error maps show spatial "
            "error magnitude.\n"
        )

        f.write(
            "5. Correlation and R2 are computed "
            "within each crop.\n"
        )

        f.write(
            "6. Test data was not used.\n"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 78)

    print(
        "SPATIAL DIAGNOSTIC COMPLETE"
    )

    print("=" * 78)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Sample metrics:"
    )

    print(
        sample_csv
    )

    print()
    print(
        "AOI metrics:"
    )

    print(
        aoi_csv
    )

    print()
    print(
        "Report:"
    )

    print(
        report_path
    )

    print()
    print(
        "Each validation sample contains:"
    )

    print(
        "  rgb.png"
    )

    print(
        "  prediction.png"
    )

    print(
        "  ground_truth.png"
    )

    print(
        "  error_signed.png"
    )

    print(
        "  absolute_error.png"
    )

    print(
        "  prediction.npy"
    )

    print(
        "  ground_truth.npy"
    )

    print(
        "  error.npy"
    )

    print(
        "  absolute_error.npy"
    )

    print(
        "  mask.npy"
    )

    print(
        "  metrics.txt"
    )


if __name__ == "__main__":

    main()