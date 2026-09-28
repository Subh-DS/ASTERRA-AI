from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ASTERRA V3 CONFIGURATION
# ============================================================

ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"

CHECKPOINT = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d_v3_1"
    / "stage5_urban3d_v3_best.pth"
)

VAL_CSV = (
    ROOT
    / "datasets"
    / "Urban3D"
    / "manifests"
    / "urban3d_val_crops.csv"
)

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

MODEL_SIZE = 518

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEPTH ANYTHING V2 IMPORT
# ============================================================

if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# MODEL
# ============================================================

def load_model():
    print("=" * 78)
    print("ASTERRA AI — STAGE 5 URBAN3D V3 INDEPENDENT VALIDATION")
    print("=" * 78)

    print(f"Device:     {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Validation: {VAL_CSV}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    if not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Validation manifest not found:\n{VAL_CSV}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    print()
    print("[CHECKPOINT]")
    print("epoch:", checkpoint.get("epoch"))
    print("best_val_mae:", checkpoint.get("best_val_mae"))
    print("stage:", checkpoint.get("stage"))
    print("stage_name:", checkpoint.get("stage_name"))
    print("target:", checkpoint.get("target"))

    state = checkpoint["model_state_dict"]

    print("model tensors:", len(state))

    # --------------------------------------------------------
    # Build exact architecture
    # --------------------------------------------------------

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    # --------------------------------------------------------
    # V3 output activation
    #
    # Original DPT head:
    # output_conv2[2] = final 32 -> 1 convolution
    # output_conv2[3] = ReLU
    #
    # V3:
    # output_conv2[3] = Softplus
    # --------------------------------------------------------

    seq = model.depth_head.scratch.output_conv2

    print()
    print("[MODEL] Original output head:")
    print(seq)

    if not isinstance(seq[3], nn.Softplus):
        seq[3] = nn.Softplus(
            beta=1.0,
            threshold=20.0,
        )

    print()
    print("[V3] output_conv2[3] = Softplus")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    print()
    print("[LOAD]")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print("Missing:")
        for key in missing:
            print("   ", key)

        raise RuntimeError(
            "Checkpoint/model mismatch."
        )

    if unexpected:
        print("Unexpected:")
        for key in unexpected:
            print("   ", key)

    # --------------------------------------------------------
    # Final projection diagnostics
    # --------------------------------------------------------

    final_conv = model.depth_head.scratch.output_conv2[2]

    w = final_conv.weight.detach().cpu().numpy()
    b = final_conv.bias.detach().cpu().numpy()

    print()
    print("[V3 FINAL PROJECTION]")
    print("shape:", w.shape)
    print("weight min:", float(w.min()))
    print("weight max:", float(w.max()))
    print("weight mean:", float(w.mean()))
    print("weight std:", float(w.std()))
    print("bias:", b)

    model.to(DEVICE)
    model.eval()

    print()
    print("[OK] V3 checkpoint loaded successfully.")

    return model


# ============================================================
# MANIFEST COLUMN DETECTION
# ============================================================

def find_column(df, candidates):
    columns = list(df.columns)

    lower = {
        str(c).lower(): c
        for c in columns
    }

    # Exact match first.
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    # Partial match second.
    for column in columns:
        name = str(column).lower()

        for candidate in candidates:
            if candidate.lower() in name:
                return column

    return None


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(value):
    """
    Resolve a manifest path.

    The Urban3D manifest normally contains absolute Windows paths.
    Relative paths are resolved against ROOT.
    """
    path = Path(str(value))

    if path.is_absolute():
        return path

    return ROOT / path


# ============================================================
# CROP INFORMATION
# ============================================================

def get_crop_info(row):
    required = ["row", "col", "crop_size"]

    missing = [
        name
        for name in required
        if name not in row.index
    ]

    if missing:
        raise RuntimeError(
            "Validation manifest is missing required crop columns: "
            + ", ".join(missing)
        )

    crop_row = int(row["row"])
    crop_col = int(row["col"])
    crop_size = int(row["crop_size"])

    if crop_row < 0 or crop_col < 0:
        raise ValueError(
            f"Invalid crop coordinates: row={crop_row}, col={crop_col}"
        )

    if crop_size <= 0:
        raise ValueError(
            f"Invalid crop_size: {crop_size}"
        )

    return crop_row, crop_col, crop_size


# ============================================================
# RGB CROP LOADING
# ============================================================

def load_rgb_crop(path, row, col, crop_size):
    with rasterio.open(path) as src:
        if src.count < 3:
            raise RuntimeError(
                f"RGB raster has only {src.count} band(s):\n{path}"
            )

        height = min(crop_size, src.height - row)
        width = min(crop_size, src.width - col)

        if height <= 0 or width <= 0:
            raise RuntimeError(
                f"Crop is outside raster bounds.\n"
                f"Raster: {path}\n"
                f"Raster size: {src.width}x{src.height}\n"
                f"row={row}, col={col}, crop_size={crop_size}"
            )

        window = rasterio.windows.Window(
            col_off=col,
            row_off=row,
            width=width,
            height=height,
        )

        rgb = src.read(
            [1, 2, 3],
            window=window,
            masked=True,
        )

        rgb = rgb.filled(0).astype(np.float32)

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    if float(rgb.max()) > 1.5:
        rgb /= 255.0

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    return rgb


# ============================================================
# DSM / DTM CROP LOADING
# ============================================================

def load_single_band_crop(path, row, col, crop_size):
    with rasterio.open(path) as src:
        height = min(crop_size, src.height - row)
        width = min(crop_size, src.width - col)

        if height <= 0 or width <= 0:
            raise RuntimeError(
                f"Crop is outside raster bounds.\n"
                f"Raster: {path}\n"
                f"Raster size: {src.width}x{src.height}\n"
                f"row={row}, col={col}, crop_size={crop_size}"
            )

        window = rasterio.windows.Window(
            col_off=col,
            row_off=row,
            width=width,
            height=height,
        )

        data = src.read(
            1,
            window=window,
            masked=True,
        )

        # Preserve invalid/nodata pixels as NaN so metrics exclude them.
        data = data.astype(np.float32).filled(np.nan)

    return data


# ============================================================
# URBAN3D TARGET
# ============================================================

def load_ndsm_target(
    dsm_path,
    dtm_path,
    row,
    col,
    crop_size,
):
    """
    Construct the exact Urban3D Stage-5 target:

        nDSM = max(DSM - DTM, 0)

    IMPORTANT:
    The validation manifest contains dsm_path and dtm_path.
    dsm_path by itself is NOT the model target.
    """

    dsm = load_single_band_crop(
        dsm_path,
        row,
        col,
        crop_size,
    )

    dtm = load_single_band_crop(
        dtm_path,
        row,
        col,
        crop_size,
    )

    if dsm.shape != dtm.shape:
        raise RuntimeError(
            "DSM/DTM crop shape mismatch:\n"
            f"DSM: {dsm.shape}\n"
            f"DTM: {dtm.shape}\n"
            f"DSM path: {dsm_path}\n"
            f"DTM path: {dtm_path}"
        )

    valid = (
        np.isfinite(dsm)
        & np.isfinite(dtm)
    )

    target = np.full(
        dsm.shape,
        np.nan,
        dtype=np.float32,
    )

    target[valid] = (
        dsm[valid] - dtm[valid]
    )

    # Urban3D Stage-5 semantics:
    # negative nDSM values become zero.
    target[valid] = np.maximum(
        target[valid],
        0.0,
    )

    return target


# ============================================================
# MODEL FORWARD
# ============================================================

@torch.inference_mode()
def predict(model, rgb):
    image = torch.from_numpy(
        rgb
    ).unsqueeze(0).to(
        DEVICE,
        dtype=torch.float32,
    )

    image = F.interpolate(
        image,
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    if DEVICE.type == "cuda":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            prediction = model(image)
    else:
        prediction = model(image)

    if isinstance(prediction, dict):
        for key in (
            "metric_depth",
            "depth",
            "out",
            "pred",
        ):
            if key in prediction:
                prediction = prediction[key]
                break
        else:
            raise RuntimeError(
                "No supported prediction key found."
            )

    if not torch.is_tensor(prediction):
        prediction = torch.as_tensor(
            prediction
        )

    if prediction.ndim == 4:
        prediction = prediction[:, 0]

    elif prediction.ndim == 3:
        prediction = prediction

    else:
        raise RuntimeError(
            f"Unexpected prediction shape: "
            f"{prediction.shape}"
        )

    prediction = prediction.float()

    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=rgb.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    prediction = prediction[
        0, 0
    ].cpu().numpy().astype(
        np.float32
    )

    return prediction


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    target,
    prediction,
):
    target = np.asarray(
        target,
        dtype=np.float32,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(target)
        & np.isfinite(prediction)
    )

    y = target[valid]
    p = prediction[valid]

    if len(y) == 0:
        raise RuntimeError(
            "No valid pixels."
        )

    # nDSM cannot be negative.
    p_clamped = np.maximum(
        p,
        0.0,
    )

    error = p_clamped - y

    absolute_error = np.abs(
        error
    )

    mae = float(
        np.mean(absolute_error)
    )

    rmse = float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )

    median_error = float(
        np.median(absolute_error)
    )

    p90_error = float(
        np.percentile(
            absolute_error,
            90,
        )
    )

    if (
        np.std(y) > 0
        and np.std(p_clamped) > 0
    ):
        correlation = float(
            np.corrcoef(
                y,
                p_clamped,
            )[0, 1]
        )
    else:
        correlation = float("nan")

    # Zero-height baseline.
    zero_mae = float(
        np.mean(np.abs(y))
    )

    zero_rmse = float(
        np.sqrt(
            np.mean(y ** 2)
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "median_error": median_error,
        "p90_error": p90_error,
        "correlation": correlation,
        "zero_mae": zero_mae,
        "zero_rmse": zero_rmse,
        "valid": int(len(y)),

        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),

        "pred_min": float(p_clamped.min()),
        "pred_max": float(p_clamped.max()),
        "pred_mean": float(p_clamped.mean()),
        "pred_std": float(p_clamped.std()),

        "pred_zero_pct": float(
            np.mean(p_clamped == 0) * 100
        ),

        "pred_positive_pct": float(
            np.mean(p_clamped > 0) * 100
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Independent validation of ASTERRA Stage-5 Urban3D V3 "
            "using the exact Urban3D nDSM target semantics."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of validation crops to evaluate.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based manifest row at which evaluation starts.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    if args.start < 0:
        raise ValueError("--start must be >= 0")

    model = load_model()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    df = pd.read_csv(
        VAL_CSV
    )

    print()
    print("=" * 78)
    print("VALIDATION MANIFEST")
    print("=" * 78)

    print("Total rows:", len(df))

    print()
    print("Columns:")
    print(list(df.columns))

    # The evaluator intentionally does NOT auto-detect a single
    # target column. Urban3D target is explicitly DSM - DTM.
    rgb_col = find_column(
        df,
        [
            "rgb_path",
            "rgb",
            "image_path",
            "image",
            "input_path",
            "input",
        ],
    )

    dsm_col = find_column(
        df,
        [
            "dsm_path",
            "dsm",
        ],
    )

    dtm_col = find_column(
        df,
        [
            "dtm_path",
            "dtm",
        ],
    )

    print()
    print("Detected RGB column:", rgb_col)
    print("Detected DSM column:", dsm_col)
    print("Detected DTM column:", dtm_col)

    if rgb_col is None:
        raise RuntimeError(
            "Could not identify RGB column."
        )

    if dsm_col is None:
        raise RuntimeError(
            "Could not identify DSM column."
        )

    if dtm_col is None:
        raise RuntimeError(
            "Could not identify DTM column."
        )

    required_crop_columns = [
        "row",
        "col",
        "crop_size",
    ]

    missing_crop_columns = [
        c
        for c in required_crop_columns
        if c not in df.columns
    ]

    if missing_crop_columns:
        raise RuntimeError(
            "Manifest is missing required crop columns: "
            + ", ".join(missing_crop_columns)
        )

    if args.start >= len(df):
        raise ValueError(
            f"--start {args.start} is outside the manifest "
            f"with {len(df)} rows."
        )

    end = min(
        args.start + args.limit,
        len(df),
    )

    selected_df = df.iloc[
        args.start:end
    ]

    limit = len(selected_df)

    print()
    print("Target definition:")
    print("  nDSM = max(DSM - DTM, 0)")
    print()
    print(
        f"Evaluating manifest rows "
        f"{args.start} through {end - 1}"
    )

    print()
    print("=" * 78)
    print(
        f"EVALUATING {limit} URBAN3D VALIDATION CROP(S)"
    )
    print("=" * 78)

    results = []

    for local_index, (index, row_data) in enumerate(
        selected_df.iterrows(),
        start=1,
    ):

        row = row_data

        rgb_path = resolve_path(
            row[rgb_col]
        )

        dsm_path = resolve_path(
            row[dsm_col]
        )

        dtm_path = resolve_path(
            row[dtm_col]
        )

        crop_row, crop_col, crop_size = get_crop_info(
            row
        )

        print()
        print("-" * 78)
        print(
            f"[{local_index}/{limit}] "
            f"manifest index={index}"
        )

        print("RGB :", rgb_path)
        print("DSM :", dsm_path)
        print("DTM :", dtm_path)

        print(
            f"Crop: row={crop_row}, "
            f"col={crop_col}, "
            f"size={crop_size}"
        )

        for path, label in (
            (rgb_path, "RGB"),
            (dsm_path, "DSM"),
            (dtm_path, "DTM"),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} not found:\n{path}"
                )

        # ----------------------------------------------------
        # Load EXACT manifest crop
        # ----------------------------------------------------

        rgb = load_rgb_crop(
            rgb_path,
            crop_row,
            crop_col,
            crop_size,
        )

        target = load_ndsm_target(
            dsm_path,
            dtm_path,
            crop_row,
            crop_col,
            crop_size,
        )

        if rgb.shape[-2:] != target.shape:
            raise RuntimeError(
                "RGB/target crop shape mismatch:\n"
                f"RGB: {rgb.shape}\n"
                f"Target: {target.shape}"
            )

        valid_target = target[
            np.isfinite(target)
        ]

        if len(valid_target) == 0:
            raise RuntimeError(
                "Target crop contains no valid pixels."
            )

        print()
        print("[CROP]")
        print("RGB shape:", rgb.shape)
        print("Target shape:", target.shape)

        print()
        print("[TARGET — nDSM = max(DSM - DTM, 0)]")
        print(
            "min:",
            float(valid_target.min())
        )
        print(
            "max:",
            float(valid_target.max())
        )
        print(
            "mean:",
            float(valid_target.mean())
        )
        print(
            "std:",
            float(valid_target.std())
        )
        print(
            "valid pixels:",
            len(valid_target),
            "/",
            target.size,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = predict(
            model,
            rgb,
        )

        print()
        print("[RAW V3 PREDICTION]")
        print(
            "min:",
            float(prediction.min())
        )
        print(
            "max:",
            float(prediction.max())
        )
        print(
            "mean:",
            float(prediction.mean())
        )
        print(
            "std:",
            float(prediction.std())
        )
        print(
            "negative %:",
            float(
                np.mean(
                    prediction < 0
                ) * 100
            )
        )
        print(
            "zero %:",
            float(
                np.mean(
                    prediction == 0
                ) * 100
            )
        )
        print(
            "positive %:",
            float(
                np.mean(
                    prediction > 0
                ) * 100
            )
        )

        # ----------------------------------------------------
        # Shape safety
        # ----------------------------------------------------

        if prediction.shape != target.shape:

            print(
                f"[RESIZE] Prediction {prediction.shape} "
                f"-> target {target.shape}"
            )

            pred_tensor = torch.from_numpy(
                prediction
            ).unsqueeze(0).unsqueeze(0)

            pred_tensor = F.interpolate(
                pred_tensor,
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            )

            prediction = pred_tensor[
                0, 0
            ].numpy().astype(
                np.float32
            )

        metrics = calculate_metrics(
            target,
            prediction,
        )

        metrics["index"] = int(index)

        metrics["crop_row"] = crop_row
        metrics["crop_col"] = crop_col
        metrics["crop_size"] = crop_size

        metrics["rgb_path"] = str(
            rgb_path
        )

        metrics["dsm_path"] = str(
            dsm_path
        )

        metrics["dtm_path"] = str(
            dtm_path
        )

        metrics["target_definition"] = (
            "max(DSM - DTM, 0)"
        )

        results.append(
            metrics
        )

        print()
        print("[METRICS]")
        print(
            f"MAE          : "
            f"{metrics['mae']:.6f} m"
        )
        print(
            f"RMSE         : "
            f"{metrics['rmse']:.6f} m"
        )
        print(
            f"Median error : "
            f"{metrics['median_error']:.6f} m"
        )
        print(
            f"P90 error    : "
            f"{metrics['p90_error']:.6f} m"
        )
        print(
            f"Correlation  : "
            f"{metrics['correlation']:.6f}"
        )
        print(
            f"Zero MAE     : "
            f"{metrics['zero_mae']:.6f} m"
        )
        print(
            f"Zero RMSE    : "
            f"{metrics['zero_rmse']:.6f} m"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        raise RuntimeError(
            "No validation results were produced."
        )

    print()
    print("=" * 78)
    print("INDEPENDENT V3 VALIDATION SUMMARY")
    print("=" * 78)

    print(
        f"Samples: {len(result_df)}"
    )

    print(
        f"Mean MAE: "
        f"{result_df['mae'].mean():.6f} m"
    )

    print(
        f"Mean RMSE: "
        f"{result_df['rmse'].mean():.6f} m"
    )

    print(
        f"Median sample MAE: "
        f"{result_df['mae'].median():.6f} m"
    )

    print(
        f"P90 sample MAE: "
        f"{result_df['mae'].quantile(.90):.6f} m"
    )

    print(
        f"Mean correlation: "
        f"{result_df['correlation'].mean():.6f}"
    )

    print()
    print("[PREDICTION DISTRIBUTION]")

    print(
        f"Mean prediction mean: "
        f"{result_df['pred_mean'].mean():.6f} m"
    )

    print(
        f"Mean prediction std: "
        f"{result_df['pred_std'].mean():.6f} m"
    )

    print()
    print("[TARGET DISTRIBUTION]")

    print(
        f"Mean target mean: "
        f"{result_df['target_mean'].mean():.6f} m"
    )

    print(
        f"Mean target std: "
        f"{result_df['target_std'].mean():.6f} m"
    )

    print()
    print("[ZERO BASELINE]")

    print(
        f"Mean zero-baseline MAE: "
        f"{result_df['zero_mae'].mean():.6f} m"
    )

    print(
        f"Mean zero-baseline RMSE: "
        f"{result_df['zero_rmse'].mean():.6f} m"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "models"
        / "asterra_stage5"
        / "urban3d_v3_1"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"{args.start}_{args.start + limit - 1}"
    )

    csv_path = (
        output_dir
        / f"v3_ndsm_independent_{suffix}.csv"
    )

    json_path = (
        output_dir
        / f"v3_ndsm_independent_{suffix}_summary.json"
    )

    result_df.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "samples": int(len(result_df)),
        "manifest_start": int(args.start),
        "manifest_end": int(args.start + limit - 1),

        "target_definition": (
            "nDSM = max(DSM - DTM, 0)"
        ),

        "mean_mae": float(
            result_df["mae"].mean()
        ),

        "mean_rmse": float(
            result_df["rmse"].mean()
        ),

        "median_sample_mae": float(
            result_df["mae"].median()
        ),

        "p90_sample_mae": float(
            result_df["mae"].quantile(.90)
        ),

        "mean_correlation": float(
            result_df["correlation"].mean()
        ),

        "mean_zero_baseline_mae": float(
            result_df["zero_mae"].mean()
        ),

        "mean_zero_baseline_rmse": float(
            result_df["zero_rmse"].mean()
        ),

        "mean_target_mean": float(
            result_df["target_mean"].mean()
        ),

        "mean_target_std": float(
            result_df["target_std"].mean()
        ),

        "mean_prediction_mean": float(
            result_df["pred_mean"].mean()
        ),

        "mean_prediction_std": float(
            result_df["pred_std"].mean()
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("[OUTPUT]")
    print("CSV :", csv_path)
    print("JSON:", json_path)

    print()
    print("=" * 78)
    print("INDEPENDENT V3 nDSM VALIDATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
