from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ASTERRA V3 DIAGNOSTIC CONFIGURATION
# ============================================================

ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"

CHECKPOINT = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d_v3"
    / "stage5_urban3d_v3_best.pth"
)

VAL_CSV = (
    ROOT
    / "datasets"
    / "Urban3D"
    / "manifests"
    / "urban3d_val_crops.csv"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d_v3"
    / "diagnostics"
)

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]
MODEL_SIZE = 518

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEPTH ANYTHING V2
# ============================================================

if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    for column in df.columns:
        name = str(column).lower()
        for candidate in candidates:
            if candidate.lower() in name:
                return column

    return None


def resolve_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_model():
    print("=" * 78)
    print("ASTERRA AI — STAGE 5 URBAN3D V3 DIAGNOSTIC")
    print("=" * 78)
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)
    print("Manifest:", VAL_CSV)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    print()
    print("[CHECKPOINT]")
    print("epoch:", checkpoint.get("epoch"))
    print("best_val_mae:", checkpoint.get("best_val_mae"))
    print("target:", checkpoint.get("target"))

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    seq = model.depth_head.scratch.output_conv2

    print()
    print("[MODEL OUTPUT HEAD BEFORE V3]")
    print(seq)

    seq[3] = nn.Softplus(
        beta=1.0,
        threshold=20.0,
    )

    state = checkpoint["model_state_dict"]

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    print()
    print("[LOAD]")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch."
        )

    final_conv = model.depth_head.scratch.output_conv2[2]

    w = final_conv.weight.detach().cpu().numpy()
    b = final_conv.bias.detach().cpu().numpy()

    print()
    print("[FINAL PROJECTION]")
    print("weight min :", float(w.min()))
    print("weight max :", float(w.max()))
    print("weight mean:", float(w.mean()))
    print("weight std :", float(w.std()))
    print("bias       :", b)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# RASTER LOADING
# ============================================================

def load_rgb_crop(path, row, col, size):
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(
            col_off=col,
            row_off=row,
            width=size,
            height=size,
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

    return np.clip(rgb, 0.0, 1.0)


def load_band_crop(path, row, col, size):
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(
            col_off=col,
            row_off=row,
            width=size,
            height=size,
        )

        data = src.read(
            1,
            window=window,
            masked=True,
        )

        data = data.astype(np.float32).filled(np.nan)

    return data


def load_target(dsm_path, dtm_path, row, col, size):
    dsm = load_band_crop(
        dsm_path,
        row,
        col,
        size,
    )

    dtm = load_band_crop(
        dtm_path,
        row,
        col,
        size,
    )

    if dsm.shape != dtm.shape:
        raise RuntimeError(
            f"DSM/DTM shape mismatch: "
            f"{dsm.shape} vs {dtm.shape}"
        )

    target = np.full(
        dsm.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(dsm)
        & np.isfinite(dtm)
    )

    target[valid] = (
        dsm[valid] - dtm[valid]
    )

    target[valid] = np.maximum(
        target[valid],
        0.0,
    )

    return target


# ============================================================
# MODEL INFERENCE
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
            pred = model(image)
    else:
        pred = model(image)

    if isinstance(pred, dict):
        for key in (
            "metric_depth",
            "depth",
            "out",
            "pred",
        ):
            if key in pred:
                pred = pred[key]
                break

    if not torch.is_tensor(pred):
        pred = torch.as_tensor(pred)

    if pred.ndim == 4:
        pred = pred[:, 0]

    pred = pred.float()

    pred = F.interpolate(
        pred.unsqueeze(1),
        size=rgb.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    return pred[0, 0].cpu().numpy().astype(np.float32)


# ============================================================
# METRICS
# ============================================================

def compute_diagnostics(target, prediction):
    valid = (
        np.isfinite(target)
        & np.isfinite(prediction)
    )

    y = target[valid].astype(np.float64)
    p = np.maximum(
        prediction[valid],
        0.0,
    ).astype(np.float64)

    error = p - y
    abs_error = np.abs(error)

    if np.std(y) > 0 and np.std(p) > 0:
        correlation = float(
            np.corrcoef(y, p)[0, 1]
        )
    else:
        correlation = float("nan")

    # Linear fit: prediction ~= slope * target + intercept
    if np.std(y) > 0:
        slope, intercept = np.polyfit(
            y,
            p,
            1,
        )
    else:
        slope, intercept = float("nan"), float("nan")

    # Pearson-like spatial association is correlation above.
    # Also report normalized prediction/target standard deviation.
    target_std = float(np.std(y))
    pred_std = float(np.std(p))

    return {
        "valid_pixels": int(len(y)),
        "mae_m": float(np.mean(abs_error)),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "median_abs_error_m": float(np.median(abs_error)),
        "p90_abs_error_m": float(np.percentile(abs_error, 90)),
        "correlation": correlation,

        "target_min_m": float(np.min(y)),
        "target_max_m": float(np.max(y)),
        "target_mean_m": float(np.mean(y)),
        "target_std_m": target_std,

        "prediction_min_m": float(np.min(p)),
        "prediction_max_m": float(np.max(p)),
        "prediction_mean_m": float(np.mean(p)),
        "prediction_std_m": pred_std,

        "std_ratio_prediction_to_target": (
            pred_std / target_std
            if target_std > 0
            else float("nan")
        ),

        "linear_slope_prediction_vs_target": float(slope),
        "linear_intercept_prediction_vs_target": float(intercept),

        "zero_baseline_mae_m": float(np.mean(np.abs(y))),
        "zero_baseline_rmse_m": float(
            np.sqrt(np.mean(y ** 2))
        ),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_diagnostics(
    rgb,
    target,
    prediction,
    metrics,
    output_dir,
    crop_index,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid = (
        np.isfinite(target)
        & np.isfinite(prediction)
    )

    pred = np.maximum(
        prediction,
        0.0,
    )

    error = np.full_like(
        target,
        np.nan,
    )

    error[valid] = (
        pred[valid] - target[valid]
    )

    abs_error = np.abs(error)

    # RGB for visualization.
    rgb_vis = np.transpose(
        rgb,
        (1, 2, 0),
    )

    # --------------------------------------------------------
    # Individual maps
    # --------------------------------------------------------

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_vis)
    plt.title("Urban3D RGB")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"crop_{crop_index:04d}_rgb.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(target, cmap="viridis")
    plt.colorbar(label="nDSM height (m)")
    plt.title("Ground Truth nDSM = max(DSM - DTM, 0)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"crop_{crop_index:04d}_ground_truth_ndsm.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(pred, cmap="viridis")
    plt.colorbar(label="Predicted height (m)")
    plt.title("ASTERRA V3 Predicted nDSM")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"crop_{crop_index:04d}_prediction.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(abs_error, cmap="magma")
    plt.colorbar(label="Absolute error (m)")
    plt.title("Absolute Error |Prediction - Ground Truth|")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"crop_{crop_index:04d}_absolute_error.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------

    gt_values = target[valid]
    pred_values = pred[valid]

    upper = float(
        max(
            np.percentile(gt_values, 99.5),
            np.percentile(pred_values, 99.5),
            1.0,
        )
    )

    bins = np.linspace(
        0,
        upper,
        60,
    )

    plt.figure(figsize=(9, 6))
    plt.hist(
        gt_values,
        bins=bins,
        alpha=0.55,
        label="Ground Truth nDSM",
        density=True,
    )
    plt.hist(
        pred_values,
        bins=bins,
        alpha=0.55,
        label="V3 Prediction",
        density=True,
    )
    plt.xlabel("Height (m)")
    plt.ylabel("Density")
    plt.title("Ground Truth vs Prediction Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / f"crop_{crop_index:04d}_histogram.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------------
    # Scatter plot
    # --------------------------------------------------------

    # Downsample only for visualization.
    rng = np.random.default_rng(42)

    max_points = 50000

    if len(gt_values) > max_points:
        indices = rng.choice(
            len(gt_values),
            size=max_points,
            replace=False,
        )

        x = gt_values[indices]
        y = pred_values[indices]
    else:
        x = gt_values
        y = pred_values

    scatter_max = float(
        max(
            np.percentile(x, 99.5),
            np.percentile(y, 99.5),
            1.0,
        )
    )

    plt.figure(figsize=(8, 8))
    plt.scatter(
        x,
        y,
        s=2,
        alpha=0.15,
    )

    plt.plot(
        [0, scatter_max],
        [0, scatter_max],
        linestyle="--",
    )

    plt.xlabel("Ground Truth nDSM (m)")
    plt.ylabel("V3 Prediction (m)")
    plt.title(
        "Ground Truth vs V3 Prediction\n"
        f"r={metrics['correlation']:.4f}, "
        f"slope={metrics['linear_slope_prediction_vs_target']:.4f}"
    )

    plt.xlim(
        0,
        scatter_max,
    )
    plt.ylim(
        0,
        scatter_max,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"crop_{crop_index:04d}_scatter.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------------
    # Error histogram
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    error_values = error[valid]

    low = float(
        np.percentile(error_values, 1)
    )
    high = float(
        np.percentile(error_values, 99)
    )

    plt.hist(
        error_values,
        bins=60,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel("Prediction - Ground Truth (m)")
    plt.ylabel("Pixel count")
    plt.title("Prediction Error Distribution")
    plt.xlim(
        low,
        high,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"crop_{crop_index:04d}_error_histogram.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------------
    # Save raw arrays for reproducibility
    # --------------------------------------------------------

    np.save(
        output_dir / f"crop_{crop_index:04d}_ground_truth_ndsm.npy",
        target.astype(np.float32),
    )

    np.save(
        output_dir / f"crop_{crop_index:04d}_prediction.npy",
        pred.astype(np.float32),
    )

    np.save(
        output_dir / f"crop_{crop_index:04d}_absolute_error.npy",
        abs_error.astype(np.float32),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose ASTERRA Stage-5 Urban3D V3 "
            "prediction collapse on one exact validation crop."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Validation manifest row to diagnose.",
    )

    args = parser.parse_args()

    if args.index < 0:
        raise ValueError(
            "--index must be >= 0"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        VAL_CSV
    )

    if args.index >= len(df):
        raise ValueError(
            f"Index {args.index} is outside manifest "
            f"with {len(df)} rows."
        )

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

    if rgb_col is None:
        raise RuntimeError(
            "RGB column not found."
        )

    if dsm_col is None:
        raise RuntimeError(
            "DSM column not found."
        )

    if dtm_col is None:
        raise RuntimeError(
            "DTM column not found."
        )

    row = df.iloc[args.index]

    rgb_path = resolve_path(
        row[rgb_col]
    )

    dsm_path = resolve_path(
        row[dsm_col]
    )

    dtm_path = resolve_path(
        row[dtm_col]
    )

    crop_row = int(row["row"])
    crop_col = int(row["col"])
    crop_size = int(row["crop_size"])

    print()
    print("=" * 78)
    print("DIAGNOSTIC CROP")
    print("=" * 78)
    print("Manifest index:", args.index)
    print("RGB:", rgb_path)
    print("DSM:", dsm_path)
    print("DTM:", dtm_path)
    print(
        f"Crop: row={crop_row}, "
        f"col={crop_col}, "
        f"size={crop_size}"
    )

    for path in (
        rgb_path,
        dsm_path,
        dtm_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    model = load_model()

    rgb = load_rgb_crop(
        rgb_path,
        crop_row,
        crop_col,
        crop_size,
    )

    target = load_target(
        dsm_path,
        dtm_path,
        crop_row,
        crop_col,
        crop_size,
    )

    prediction = predict(
        model,
        rgb,
    )

    if prediction.shape != target.shape:
        tensor = torch.from_numpy(
            prediction
        ).unsqueeze(0).unsqueeze(0)

        tensor = F.interpolate(
            tensor,
            size=target.shape,
            mode="bilinear",
            align_corners=False,
        )

        prediction = tensor[
            0, 0
        ].numpy().astype(np.float32)

    metrics = compute_diagnostics(
        target,
        prediction,
    )

    print()
    print("=" * 78)
    print("DIAGNOSTIC STATISTICS")
    print("=" * 78)

    print()
    print("[GROUND TRUTH nDSM]")
    print("min :", metrics["target_min_m"])
    print("max :", metrics["target_max_m"])
    print("mean:", metrics["target_mean_m"])
    print("std :", metrics["target_std_m"])

    print()
    print("[V3 PREDICTION]")
    print("min :", metrics["prediction_min_m"])
    print("max :", metrics["prediction_max_m"])
    print("mean:", metrics["prediction_mean_m"])
    print("std :", metrics["prediction_std_m"])

    print()
    print("[ERROR / RELATIONSHIP]")
    print("MAE:", metrics["mae_m"], "m")
    print("RMSE:", metrics["rmse_m"], "m")
    print("Median absolute error:",
          metrics["median_abs_error_m"], "m")
    print("P90 absolute error:",
          metrics["p90_abs_error_m"], "m")
    print("Correlation:",
          metrics["correlation"])
    print("Prediction/target std ratio:",
          metrics["std_ratio_prediction_to_target"])
    print("Linear slope:",
          metrics["linear_slope_prediction_vs_target"])
    print("Linear intercept:",
          metrics["linear_intercept_prediction_vs_target"])

    print()
    print("[ZERO BASELINE]")
    print(
        "MAE:",
        metrics["zero_baseline_mae_m"],
        "m",
    )
    print(
        "RMSE:",
        metrics["zero_baseline_rmse_m"],
        "m",
    )

    save_diagnostics(
        rgb=rgb,
        target=target,
        prediction=prediction,
        metrics=metrics,
        output_dir=OUTPUT_DIR,
        crop_index=args.index,
    )

    summary_path = (
        OUTPUT_DIR
        / f"crop_{args.index:04d}_diagnostic_summary.json"
    )

    payload = {
        "manifest_index": args.index,
        "rgb_path": str(rgb_path),
        "dsm_path": str(dsm_path),
        "dtm_path": str(dtm_path),
        "crop_row": crop_row,
        "crop_col": crop_col,
        "crop_size": crop_size,
        "target_definition": "max(DSM - DTM, 0)",
        "metrics": metrics,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("DIAGNOSTIC FILES")
    print("=" * 78)
    print("Directory:", OUTPUT_DIR)
    print()
    print("RGB:")
    print(
        OUTPUT_DIR
        / f"crop_{args.index:04d}_rgb.png"
    )
    print("Ground truth:")
    print(
        OUTPUT_DIR
        / f"crop_{args.index:04d}_ground_truth_ndsm.png"
    )
    print("Prediction:")
    print(
        OUTPUT_DIR
        / f"crop_{args.index:04d}_prediction.png"
    )
    print("Absolute error:")
    print(
        OUTPUT_DIR
        / f"crop_{args.index:04d}_absolute_error.png"
    )
    print("Histogram:")
    print(
        OUTPUT_DIR
        / f"crop_{args.index:04d}_histogram.png"
    )
    print("Scatter:")
    print(
        OUTPUT_DIR
        / f"crop_{args.index:04d}_scatter.png"
    )
    print("Error histogram:")
    print(
        OUTPUT_DIR
        / f"crop_{args.index:04d}_error_histogram.png"
    )
    print("Summary JSON:")
    print(summary_path)

    print()
    print("=" * 78)
    print("ASTERRA V3 DIAGNOSTIC COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()