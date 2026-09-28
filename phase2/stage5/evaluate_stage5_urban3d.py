import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# ASTERRA AI — STAGE 5 URBAN3D INDEPENDENT EVALUATION
# ============================================================

ROOT = Path(r"D:\Asterra AI")

STAGE4_BEST = ROOT / r"models\asterra_stage4\stage4_best.pth"

STAGE5_BEST = (
    ROOT
    / r"models\asterra_stage5\urban3d\stage5_urban3d_best.pth"
)

VAL_MANIFEST = (
    ROOT
    / r"datasets\Urban3D\manifests\urban3d_val_crops.csv"
)

DEPTH_REPO = ROOT / r"external\Depth-Anything-V2"

OUT_DIR = (
    ROOT
    / r"models\asterra_stage5\urban3d\evaluation"
)


# ============================================================
# MODEL CONFIG — MUST MATCH STAGE 5 TRAINING
# ============================================================

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

MODEL_SIZE = 518
CROP_SIZE = 512


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_CUDA = DEVICE.type == "cuda"


# ============================================================
# MODEL IMPORT
# ============================================================

def get_model_class():

    repo = str(DEPTH_REPO)

    if repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        raise RuntimeError(
            "Could not import DepthAnythingV2.\n\n"
            f"Repository:\n{DEPTH_REPO}\n\n"
            f"Original error:\n{e}"
        )

    return DepthAnythingV2


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_model(checkpoint_path):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print()
    print("=" * 78)
    print("LOADING MODEL")
    print("=" * 78)

    print(f"[MODEL] {checkpoint_path}")
    print(
        f"[MODEL] Size: "
        f"{checkpoint_path.stat().st_size / (1024 ** 3):.2f} GB"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    print(
        "[CHECKPOINT] Keys:",
        list(checkpoint.keys())
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Your actual Stage-5 checkpoint uses:
    #
    #     model_state_dict
    #
    # --------------------------------------------------------

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "The checkpoint does not contain "
            "'model_state_dict'.\n\n"
            f"Available keys:\n"
            f"{list(checkpoint.keys())}"
        )

    state = checkpoint["model_state_dict"]

    if not isinstance(state, dict):

        raise RuntimeError(
            "'model_state_dict' is not a dictionary."
        )

    print(
        f"[CHECKPOINT] Model parameters: {len(state)}"
    )

    # --------------------------------------------------------
    # Remove wrappers if present.
    # --------------------------------------------------------

    cleaned_state = {}

    for key, value in state.items():

        if not isinstance(key, str):
            continue

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]

        cleaned_state[new_key] = value

    state = cleaned_state

    print(
        "[CHECKPOINT] First model keys:"
    )

    for key in list(state.keys())[:10]:
        print("   ", key)

    # ========================================================
    # BUILD EXACT DEPTH ANYTHING V2 ARCHITECTURE
    # ========================================================

    DepthAnythingV2 = get_model_class()

    print()
    print("[MODEL] Building DepthAnythingV2...")
    print(f"        encoder     = {ENCODER}")
    print(f"        features    = {FEATURES}")
    print(f"        out_channels= {OUT_CHANNELS}")

    # IMPORTANT:
    # Do NOT pass max_depth.
    #
    # Your installed DepthAnythingV2 does not accept it.
    # ========================================================

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    print()
    print(
        f"[LOAD] Missing keys:    {len(missing)}"
    )

    print(
        f"[LOAD] Unexpected keys: {len(unexpected)}"
    )

    if missing:

        print("[LOAD] First missing keys:")

        for key in missing[:20]:
            print("   ", key)

        raise RuntimeError(
            "\nCheckpoint architecture mismatch.\n"
            "The trained checkpoint cannot be safely evaluated "
            "with this model definition."
        )

    if unexpected:

        print("[LOAD] First unexpected keys:")

        for key in unexpected[:20]:
            print("   ", key)

    print()
    print("[MODEL] Checkpoint loaded successfully.")

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


# ============================================================
# RGB
# ============================================================

def read_rgb(path, row, col, size):

    with rasterio.open(path) as src:

        arr = src.read(
            indexes=[1, 2, 3],
            window=(
                (row, row + size),
                (col, col + size),
            ),
        ).astype(np.float32)

    expected = (3, size, size)

    if arr.shape != expected:

        raise RuntimeError(
            f"RGB shape {arr.shape}; "
            f"expected {expected}"
        )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Urban3D RGB is uint8.
    if arr.max() > 1.5:
        arr /= 255.0

    arr = np.clip(arr, 0.0, 1.0)

    # CHW -> HWC
    arr = np.transpose(arr, (1, 2, 0))

    return arr.astype(np.float32)


# ============================================================
# TARGET
# ============================================================

def read_target(
    dsm_path,
    dtm_path,
    row,
    col,
    size,
):

    with rasterio.open(dsm_path) as src:

        dsm = src.read(
            1,
            window=(
                (row, row + size),
                (col, col + size),
            ),
        ).astype(np.float32)

        dsm_nodata = src.nodata

    with rasterio.open(dtm_path) as src:

        dtm = src.read(
            1,
            window=(
                (row, row + size),
                (col, col + size),
            ),
        ).astype(np.float32)

        dtm_nodata = src.nodata

    # --------------------------------------------------------
    # Validity
    # --------------------------------------------------------

    valid = (
        np.isfinite(dsm)
        & np.isfinite(dtm)
    )

    if dsm_nodata is not None:

        valid &= (
            dsm != dsm_nodata
        )

    else:

        valid &= (
            dsm != -32767.0
        )

    if dtm_nodata is not None:

        valid &= (
            dtm != dtm_nodata
        )

    else:

        valid &= (
            dtm != -32767.0
        )

    # --------------------------------------------------------
    # Urban3D nDSM
    #
    # nDSM = DSM - DTM
    # --------------------------------------------------------

    raw_ndsm = dsm - dtm

    valid &= np.isfinite(raw_ndsm)

    # Same training policy:
    # negative nDSM -> 0
    target = np.maximum(
        np.nan_to_num(
            raw_ndsm,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    ).astype(np.float32)

    target[~valid] = 0.0

    return (
        target,
        valid.astype(np.float32),
        raw_ndsm,
    )


# ============================================================
# MODEL OUTPUT
# ============================================================

def prediction_tensor(output):

    # Tensor
    if torch.is_tensor(output):

        pred = output

    # Dictionary
    elif isinstance(output, dict):

        pred = None

        # Prefer known names.
        for name in (
            "pred",
            "prediction",
            "out",
            "depth",
        ):

            value = output.get(name)

            if torch.is_tensor(value):

                pred = value
                break

        # Fallback
        if pred is None:

            tensors = [
                value
                for value in output.values()
                if torch.is_tensor(value)
            ]

            if not tensors:

                raise RuntimeError(
                    "Model dictionary contains no tensor output."
                )

            pred = tensors[-1]

    # Tuple/list
    elif isinstance(output, (tuple, list)):

        tensors = [
            value
            for value in output
            if torch.is_tensor(value)
        ]

        if not tensors:

            raise RuntimeError(
                "Model tuple/list contains no tensor."
            )

        pred = tensors[-1]

    else:

        raise RuntimeError(
            f"Unsupported model output type: "
            f"{type(output)}"
        )

    # --------------------------------------------------------
    # Shape normalization
    # --------------------------------------------------------

    if pred.ndim == 3:

        pred = pred.unsqueeze(1)

    if pred.ndim != 4:

        raise RuntimeError(
            f"Bad prediction shape: "
            f"{tuple(pred.shape)}"
        )

    # We need 1-channel height prediction.
    if pred.shape[1] != 1:

        pred = pred[:, :1]

    return pred


# ============================================================
# EVALUATE ONE CROP
# ============================================================

@torch.inference_mode()
def evaluate_sample(model, item):

    crop_id = item["crop_id"]

    row = int(
        float(item["row"])
    )

    col = int(
        float(item["col"])
    )

    size = int(
        float(
            item.get(
                "crop_size",
                CROP_SIZE,
            )
        )
    )

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    rgb = read_rgb(
        item["rgb_path"],
        row,
        col,
        size,
    )

    target, mask, raw = read_target(
        item["dsm_path"],
        item["dtm_path"],
        row,
        col,
        size,
    )

    # --------------------------------------------------------
    # RGB tensor
    # --------------------------------------------------------

    image = torch.from_numpy(
        np.transpose(
            rgb,
            (2, 0, 1),
        )
    ).unsqueeze(0)

    image = image.to(
        DEVICE,
        non_blocking=True,
    )

    # --------------------------------------------------------
    # Training uses 518 x 518
    # --------------------------------------------------------

    image = F.interpolate(
        image,
        size=(
            MODEL_SIZE,
            MODEL_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    if USE_CUDA:

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            output = model(image)

    else:

        output = model(image)

    pred = prediction_tensor(
        output
    ).float()

    # ========================================================
    # RAW PREDICTION DIAGNOSTICS
    # ========================================================
    raw_pred = pred.detach()

    raw_min = float(raw_pred.min().item())
    raw_max = float(raw_pred.max().item())
    raw_mean = float(raw_pred.mean().item())
    raw_std = float(raw_pred.std().item())

    raw_negative_pct = float(
        (raw_pred < 0).float().mean().item() * 100.0
    )
    raw_zero_pct = float(
        (raw_pred == 0).float().mean().item() * 100.0
    )
    raw_positive_pct = float(
        (raw_pred > 0).float().mean().item() * 100.0
    )

    print(
        f"[RAW PRED] "
        f"min={raw_min:.6f} "
        f"max={raw_max:.6f} "
        f"mean={raw_mean:.6f} "
        f"std={raw_std:.6f} "
        f"negative={raw_negative_pct:.2f}% "
        f"zero={raw_zero_pct:.2f}% "
        f"positive={raw_positive_pct:.2f}%"
    )

    # --------------------------------------------------------
    # Target tensors
    # --------------------------------------------------------

    target_t = torch.from_numpy(
        target
    )[None, None].to(
        DEVICE
    )

    mask_t = torch.from_numpy(
        mask
    )[None, None].to(
        DEVICE
    )

    # --------------------------------------------------------
    # Resize target safely.
    #
    # Weighted interpolation prevents invalid pixels from
    # contaminating valid target values.
    # --------------------------------------------------------

    target_weighted = F.interpolate(
        target_t * mask_t,
        size=pred.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    resized_mask = F.interpolate(
        mask_t,
        size=pred.shape[-2:],
        mode="nearest",
    )

    target_resized = (
        target_weighted
        / resized_mask.clamp_min(1e-6)
    )

    valid = resized_mask > 0.5

    if not torch.any(valid):

        raise RuntimeError(
            "No valid pixels after resize."
        )

    # ========================================================
    # CLAMPED PREDICTION DIAGNOSTICS
    # ========================================================
    # Keep raw_pred untouched so we can compare the actual
    # network output against the non-negative height metric.
    clamped_pred = torch.clamp(
        raw_pred,
        min=0.0,
    )

    clamped_min = float(clamped_pred.min().item())
    clamped_max = float(clamped_pred.max().item())
    clamped_mean = float(clamped_pred.mean().item())
    clamped_std = float(clamped_pred.std().item())

    print(
        f"[CLAMPED PRED] "
        f"min={clamped_min:.6f} "
        f"max={clamped_max:.6f} "
        f"mean={clamped_mean:.6f} "
        f"std={clamped_std:.6f}"
    )

    # Existing metric remains based on the non-negative
    # prediction required for Urban3D height output.
    pred = clamped_pred

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    error = (
        pred - target_resized
    )

    abs_error = error.abs()

    n_valid = int(
        valid.sum().item()
    )

    mae = float(
        abs_error[valid].mean().item()
    )

    rmse = float(
        torch.sqrt(
            (
                error[valid] ** 2
            ).mean()
        ).item()
    )

    return {

        "crop_id": crop_id,

        "mae": mae,

        "rmse": rmse,

        "valid_pixels": n_valid,

        "target_mean": float(
            target_resized[valid]
            .mean()
            .item()
        ),

        "target_max": float(
            target_resized[valid]
            .max()
            .item()
        ),

        "pred_mean": float(
            pred[valid]
            .mean()
            .item()
        ),

        "pred_max": float(
            pred[valid]
            .max()
            .item()
        ),

        # Raw network output diagnostics.
        "raw_pred_min": raw_min,
        "raw_pred_max": raw_max,
        "raw_pred_mean": raw_mean,
        "raw_pred_std": raw_std,
        "raw_negative_pct": raw_negative_pct,
        "raw_zero_pct": raw_zero_pct,
        "raw_positive_pct": raw_positive_pct,

        "clamped_pred_min": clamped_min,
        "clamped_pred_max": clamped_max,
        "clamped_pred_mean": clamped_mean,
        "clamped_pred_std": clamped_std,

        # Visualization data
        "rgb": rgb,

        "target": target_resized[
            0, 0
        ].cpu().numpy(),

        "pred": pred[
            0, 0
        ].cpu().numpy(),

        "mask": valid[
            0, 0
        ].cpu().numpy(),
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize(results):

    if not results:

        raise RuntimeError(
            "No successful evaluation samples."
        )

    total_valid = sum(
        r["valid_pixels"]
        for r in results
    )

    weighted_mae = (
        sum(
            r["mae"]
            * r["valid_pixels"]
            for r in results
        )
        / total_valid
    )

    weighted_mse = (
        sum(
            (r["rmse"] ** 2)
            * r["valid_pixels"]
            for r in results
        )
        / total_valid
    )

    weighted_rmse = math.sqrt(
        weighted_mse
    )

    sample_maes = [
        r["mae"]
        for r in results
    ]

    return {

        "samples": len(results),

        "valid_pixels": int(
            total_valid
        ),

        "mae_m": float(
            weighted_mae
        ),

        "rmse_m": float(
            weighted_rmse
        ),

        "sample_mean_mae_m": float(
            np.mean(sample_maes)
        ),

        "sample_median_mae_m": float(
            np.median(sample_maes)
        ),

        "sample_p90_mae_m": float(
            np.percentile(
                sample_maes,
                90,
            )
        ),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_visual(result, output_base):

    rgb = result["rgb"]
    target = result["target"]
    pred = result["pred"]
    mask = result["mask"]

    error = np.abs(
        pred - target
    )

    error[~mask] = 0.0

    def normalize(data):

        valid_values = data[mask]

        if len(valid_values) == 0:

            high = 1.0

        else:

            high = float(
                np.percentile(
                    valid_values,
                    99,
                )
            )

        high = max(
            high,
            1e-6,
        )

        result_img = np.clip(
            data / high,
            0.0,
            1.0,
        )

        result_img[~mask] = 0.0

        return (
            result_img * 255
        ).astype(np.uint8)

    # RGB
    Image.fromarray(
        np.clip(
            rgb * 255,
            0,
            255,
        ).astype(np.uint8)
    ).save(
        output_base
        + "_rgb.png"
    )

    # Ground truth
    Image.fromarray(
        normalize(target)
    ).save(
        output_base
        + "_gt_ndsm.png"
    )

    # Prediction
    Image.fromarray(
        normalize(pred)
    ).save(
        output_base
        + "_pred_ndsm.png"
    )

    # Absolute error
    Image.fromarray(
        normalize(error)
    ).save(
        output_base
        + "_abs_error.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Independent Urban3D evaluation "
            "for ASTERRA Stage 5."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "urban3d",
            "stage4",
            "both",
        ],
        default="urban3d",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Number of validation crops. "
            "0 = all."
        ),
    )

    parser.add_argument(
        "--visuals",
        type=int,
        default=20,
        help=(
            "Number of visual examples."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # DIRECTORIES
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    visual_dir = (
        OUT_DIR / "visuals"
    )

    visual_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 78)
    print(
        "ASTERRA AI — STAGE 5 URBAN3D "
        "INDEPENDENT EVALUATION"
    )
    print("=" * 78)

    print(
        f"Device:          {DEVICE}"
    )
    print("Mode:            RAW vs CLAMPED prediction diagnostic")

    print(
        f"Validation CSV:  {VAL_MANIFEST}"
    )

    print(
        f"Output directory:{OUT_DIR}"
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    if not VAL_MANIFEST.exists():

        raise FileNotFoundError(
            f"Validation manifest not found:\n"
            f"{VAL_MANIFEST}"
        )

    with VAL_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        items = list(
            csv.DictReader(f)
        )

    if not items:

        raise RuntimeError(
            "Validation manifest is empty."
        )

    if args.limit > 0:

        items = items[
            :args.limit
        ]

    print()
    print(
        f"[DATA] Validation crops: "
        f"{len(items)}"
    )

    # ========================================================
    # MODELS
    # ========================================================

    model_paths = []

    if args.model in (
        "urban3d",
        "both",
    ):

        model_paths.append(
            (
                "stage5_urban3d_best",
                STAGE5_BEST,
            )
        )

    if args.model in (
        "stage4",
        "both",
    ):

        model_paths.append(
            (
                "stage4_best",
                STAGE4_BEST,
            )
        )

    summaries = {}

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    for model_name, checkpoint_path in model_paths:

        model, checkpoint = load_model(
            checkpoint_path
        )

        results = []

        started = time.time()

        visual_count = 0

        print()
        print("=" * 78)
        print(
            f"EVALUATING {model_name}"
        )
        print("=" * 78)

        # ----------------------------------------------------
        # Checkpoint metadata
        # ----------------------------------------------------

        print()

        for key in (
            "stage",
            "stage_name",
            "epoch",
            "best_val_mae",
            "dataset",
            "target",
            "negative_ndsm_policy",
            "initial_checkpoint",
            "patch_size",
            "model_size",
            "encoder",
            "features",
            "out_channels",
            "full_parameter_tuning",
        ):

            if key in checkpoint:

                print(
                    f"[META] {key}: "
                    f"{checkpoint[key]}"
                )

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        for index, item in enumerate(
            items,
            start=1,
        ):

            try:

                result = evaluate_sample(
                    model,
                    item,
                )

                results.append(
                    result
                )

                # Visual examples
                if (
                    args.visuals > 0
                    and visual_count
                    < args.visuals
                ):

                    crop_id = result[
                        "crop_id"
                    ]

                    safe_id = "".join(
                        c
                        if (
                            c.isalnum()
                            or c in "-_"
                        )
                        else "_"
                        for c in crop_id
                    )

                    save_visual(
                        result,
                        str(
                            visual_dir
                            / (
                                model_name
                                + "_"
                                + safe_id
                            )
                        ),
                    )

                    visual_count += 1

                # Progress
                if (
                    index % 100 == 0
                    or index == len(items)
                ):

                    current = summarize(
                        results
                    )

                    elapsed = (
                        time.time()
                        - started
                    )

                    print(
                        f"[{index}/{len(items)}] "
                        f"MAE="
                        f"{current['mae_m']:.4f} m "
                        f"| RMSE="
                        f"{current['rmse_m']:.4f} m "
                        f"| valid="
                        f"{current['valid_pixels']:,} "
                        f"| time="
                        f"{elapsed / 60:.1f} min"
                    )

            except Exception as e:

                print(
                    f"[WARNING] "
                    f"{item.get('crop_id')}: "
                    f"{e}"
                )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        if not results:

            raise RuntimeError(
                f"No successful samples "
                f"for {model_name}."
            )

        summary = summarize(
            results
        )

        summaries[
            model_name
        ] = summary

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        summary_path = (
            OUT_DIR
            / f"{model_name}_summary.json"
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "model": model_name,
                    "checkpoint": str(
                        checkpoint_path
                    ),
                    "manifest": str(
                        VAL_MANIFEST
                    ),
                    "summary": summary,
                },
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_path = (
            OUT_DIR
            / f"{model_name}_per_crop_metrics.csv"
        )

        fields = [
            "crop_id",
            "mae",
            "rmse",
            "valid_pixels",
            "target_mean",
            "target_max",
            "pred_mean",
            "pred_max",
            "raw_pred_min",
            "raw_pred_max",
            "raw_pred_mean",
            "raw_pred_std",
            "raw_negative_pct",
            "raw_zero_pct",
            "raw_positive_pct",
            "clamped_pred_min",
            "clamped_pred_max",
            "clamped_pred_mean",
            "clamped_pred_std",
        ]

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            for result in results:

                writer.writerow(
                    {
                        key: result[key]
                        for key in fields
                    }
                )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print("-" * 78)
        print(
            f"{model_name} FINAL RESULT"
        )
        print("-" * 78)

        print(
            f"Samples:       "
            f"{summary['samples']}"
        )

        print(
            f"Valid pixels:  "
            f"{summary['valid_pixels']:,}"
        )

        print(
            f"MAE:           "
            f"{summary['mae_m']:.6f} m"
        )

        print(
            f"RMSE:          "
            f"{summary['rmse_m']:.6f} m"
        )

        print(
            f"Mean sample MAE: "
            f"{summary['sample_mean_mae_m']:.6f} m"
        )

        print(
            f"Median sample MAE: "
            f"{summary['sample_median_mae_m']:.6f} m"
        )

        print(
            f"P90 sample MAE: "
            f"{summary['sample_p90_mae_m']:.6f} m"
        )

        print(
            f"CSV: "
            f"{csv_path}"
        )

        print(
            f"JSON: "
            f"{summary_path}"
        )

        # ----------------------------------------------------
        # Free GPU
        # ----------------------------------------------------

        del model

        if USE_CUDA:

            torch.cuda.empty_cache()

    # ========================================================
    # STAGE 4 VS STAGE 5
    # ========================================================

    if (
        "stage4_best" in summaries
        and
        "stage5_urban3d_best"
        in summaries
    ):

        stage4 = summaries[
            "stage4_best"
        ]

        stage5 = summaries[
            "stage5_urban3d_best"
        ]

        mae_improvement = (
            stage4["mae_m"]
            - stage5["mae_m"]
        )

        mae_percent = (
            100.0
            * mae_improvement
            / stage4["mae_m"]
            if stage4["mae_m"] != 0
            else 0.0
        )

        rmse_improvement = (
            stage4["rmse_m"]
            - stage5["rmse_m"]
        )

        comparison = {

            "stage4_mae_m":
                stage4["mae_m"],

            "stage5_urban3d_mae_m":
                stage5["mae_m"],

            "mae_improvement_m":
                mae_improvement,

            "mae_improvement_percent":
                mae_percent,

            "stage4_rmse_m":
                stage4["rmse_m"],

            "stage5_urban3d_rmse_m":
                stage5["rmse_m"],

            "rmse_improvement_m":
                rmse_improvement,
        }

        comparison_path = (
            OUT_DIR
            / "stage4_vs_stage5_urban3d.json"
        )

        with comparison_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                comparison,
                f,
                indent=2,
            )

        print()
        print("=" * 78)
        print(
            "STAGE 4 vs STAGE 5 URBAN3D"
        )
        print("=" * 78)

        print(
            f"Stage 4 MAE: "
            f"{stage4['mae_m']:.6f} m"
        )

        print(
            f"Stage 5 MAE: "
            f"{stage5['mae_m']:.6f} m"
        )

        print(
            f"MAE improvement: "
            f"{mae_improvement:.6f} m"
        )

        print(
            f"MAE improvement: "
            f"{mae_percent:.2f}%"
        )

        print()

        print(
            f"Stage 4 RMSE: "
            f"{stage4['rmse_m']:.6f} m"
        )

        print(
            f"Stage 5 RMSE: "
            f"{stage5['rmse_m']:.6f} m"
        )

        print(
            f"RMSE improvement: "
            f"{rmse_improvement:.6f} m"
        )

        print(
            f"Comparison JSON: "
            f"{comparison_path}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 78)
    print(
        "INDEPENDENT EVALUATION COMPLETE"
    )
    print("=" * 78)

    print(
        f"Results directory:\n{OUT_DIR}"
    )


if __name__ == "__main__":

    main()
