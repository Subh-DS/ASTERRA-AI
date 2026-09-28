import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image


# ============================================================
# ASTERRA ROOT
# ============================================================

ROOT = Path(r"D:\Asterra AI")


# ============================================================
# DEPTH ANYTHING V2 SOURCE
# ============================================================

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


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DINOv2 INPUT SIZE
# ============================================================
#
# DINOv2 ViT-L/14 uses 14x14 patches.
#
# 518 / 14 = 37 exactly.
#
# Therefore 518x518 is a valid input resolution.
#
# Dataset crops remain 512x512.
# Only the RGB input to the model is resized.
#
# After inference the prediction is resized back to the
# original target dimensions.
# ============================================================

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

    # --------------------------------------------------------
    # Stage-2 output head
    #
    # Stage-2 predicts absolute S-EO DSM-Max elevation.
    #
    # The Urban3D V3.1 model had a positive-output activation.
    # For S-EO Stage-2 this was replaced by Identity so that
    # the network can predict unrestricted metric elevation.
    # --------------------------------------------------------

    model.depth_head.scratch.output_conv2[3] = nn.Identity()

    print(
        "Output activation: Identity (linear)"
    )

    print(
        "Target type      : "
        "S-EO DSM-Max absolute elevation"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()
    print("Loading checkpoint:")
    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------------
    # ASTERRA CHECKPOINT FORMAT
    #
    # The training checkpoint contains metadata plus:
    #
    #   model_state_dict
    #
    # The actual neural-network weights are stored inside
    # model_state_dict.
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state = checkpoint[
            "model_state_dict"
        ]

        print(
            "Checkpoint format : "
            "ASTERRA training checkpoint"
        )

        print(
            "Weights key       : "
            "model_state_dict"
        )

    elif (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):

        # Fallback for checkpoints that use "model".
        state = checkpoint["model"]

        print(
            "Checkpoint format : "
            "model wrapper"
        )

        print(
            "Weights key       : model"
        )

    else:

        # Fallback for a raw state_dict.
        state = checkpoint

        print(
            "Checkpoint format : "
            "raw state_dict"
        )

    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    incompatible = model.load_state_dict(
        state,
        strict=False,
    )

    missing_keys = (
        incompatible.missing_keys
    )

    unexpected_keys = (
        incompatible.unexpected_keys
    )

    print()
    print(
        "Checkpoint verification:"
    )

    print(
        f"Missing keys    : "
        f"{len(missing_keys)}"
    )

    print(
        f"Unexpected keys : "
        f"{len(unexpected_keys)}"
    )

    # --------------------------------------------------------
    # We expect a perfect architecture match.
    # --------------------------------------------------------

    if missing_keys:

        print()
        print(
            "ERROR: Missing model parameters:"
        )

        for key in missing_keys[:20]:
            print(
                "  ",
                key,
            )

        if len(missing_keys) > 20:

            print(
                f"  ... and "
                f"{len(missing_keys) - 20} more"
            )

        raise RuntimeError(
            "Checkpoint is incompatible with "
            "the ASTERRA Stage-2 architecture."
        )

    if unexpected_keys:

        print()
        print(
            "ERROR: Unexpected checkpoint parameters:"
        )

        for key in unexpected_keys[:20]:

            print(
                "  ",
                key,
            )

        if len(unexpected_keys) > 20:

            print(
                f"  ... and "
                f"{len(unexpected_keys) - 20} more"
            )

        raise RuntimeError(
            "Checkpoint contains parameters that "
            "do not belong to the ASTERRA model."
        )

    print()
    print(
        "Checkpoint loaded successfully."
    )

    print(
        "Missing keys    : 0"
    )

    print(
        "Unexpected keys : 0"
    )

    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# RGB LOADER
# ============================================================

def load_rgb(path):
    """
    Load an S-EO RGB crop.

    Dataset:
        512x512

    Model input:
        518x518

    Reason:
        DINOv2 ViT-L/14 requires image dimensions divisible
        by the 14-pixel patch size.

        512 % 14 != 0
        518 % 14 == 0

    IMPORTANT:
        The ground-truth target is NOT resized here.

    Returns:
        tensor
        original_height
        original_width
    """

    # --------------------------------------------------------
    # Load RGB
    # --------------------------------------------------------

    img = np.asarray(
        Image.open(path).convert("RGB"),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Original dimensions
    # --------------------------------------------------------

    original_h = img.shape[0]
    original_w = img.shape[1]

    # --------------------------------------------------------
    # Normalize uint8 -> [0,1]
    # --------------------------------------------------------

    if img.max() > 1.5:

        img /= 255.0

    img = np.clip(
        img,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Convert to PIL for high-quality RGB resize
    # --------------------------------------------------------

    pil_img = Image.fromarray(
        (img * 255.0).astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # 512 -> 518
    #
    # 518 = 37 * 14
    # --------------------------------------------------------

    pil_img = pil_img.resize(
        (
            MODEL_SIZE,
            MODEL_SIZE,
        ),
        Image.Resampling.BICUBIC,
    )

    # --------------------------------------------------------
    # Back to numpy [0,1]
    # --------------------------------------------------------

    img = np.asarray(
        pil_img,
        dtype=np.float32,
    ) / 255.0

    img = np.clip(
        img,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        img
    )

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    # --------------------------------------------------------
    # CHW -> BCHW
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(0)

    # --------------------------------------------------------
    # GPU / CPU
    # --------------------------------------------------------

    tensor = tensor.to(
        DEVICE
    )

    return (
        tensor,
        original_h,
        original_w,
    )


# ============================================================
# TARGET LOADER
# ============================================================

def load_target(path):

    return np.load(
        path
    ).astype(
        np.float32
    )


# ============================================================
# MASK LOADER
# ============================================================

def load_mask(path):

    return np.load(
        path
    ).astype(
        bool
    )


# ============================================================
# PER-SAMPLE METRICS
# ============================================================

def metrics(
    pred,
    gt,
    mask,
):

    p = pred[mask].astype(
        np.float64
    )

    y = gt[mask].astype(
        np.float64
    )

    err = (
        p - y
    )

    ae = np.abs(
        err
    )

    mae = np.mean(
        ae
    )

    rmse = np.sqrt(
        np.mean(
            err ** 2
        )
    )

    bias = np.mean(
        err
    )

    if (
        np.std(p) > 0
        and np.std(y) > 0
    ):

        corr = np.corrcoef(
            p,
            y,
        )[0, 1]

    else:

        corr = np.nan

    return {

        "mae":
            mae,

        "rmse":
            rmse,

        "bias":
            bias,

        "corr":
            corr,

        "pred_mean":
            np.mean(p),

        "pred_std":
            np.std(p),

        "gt_mean":
            np.mean(y),

        "gt_std":
            np.std(y),

        "median_ae":
            np.median(ae),

        "p90_ae":
            np.percentile(
                ae,
                90,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "ASTERRA — VALIDATION-ONLY "
        "AFFINE CALIBRATION"
    )

    print("=" * 78)

    print()
    print(
        "Device    :",
        DEVICE,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Dataset   :",
        DATASET,
    )

    print(
        "Model size:",
        f"{MODEL_SIZE}x{MODEL_SIZE}",
    )

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    manifest_path = (
        DATASET
        / "manifest.csv"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    # ========================================================
    # VALIDATION ONLY
    # ========================================================

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
        "Validation AOIs   :",
        sorted(
            val["aoi"].unique()
        ),
    )

    if len(val) == 0:

        raise RuntimeError(
            "No validation samples found "
            "in manifest.csv"
        )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model()

    # ========================================================
    # STORAGE
    # ========================================================

    all_pred = []

    all_gt = []

    rows = []

    # ========================================================
    # VALIDATION INFERENCE
    # ========================================================

    print()
    print("-" * 78)

    print(
        "RUNNING VALIDATION INFERENCE"
    )

    print("-" * 78)

    with torch.no_grad():

        for i, row in (
            val
            .reset_index(drop=True)
            .iterrows()
        ):

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

            # =================================================
            # LOAD RGB
            # =================================================

            (
                rgb,
                original_h,
                original_w,
            ) = load_rgb(
                image_path
            )

            # =================================================
            # MODEL INFERENCE
            # =================================================

            pred = model(
                rgb
            )

            if isinstance(
                pred,
                (tuple, list),
            ):

                pred = pred[0]

            # =================================================
            # MODEL OUTPUT
            #
            # Expected:
            #
            # [1, 1, 518, 518]
            # =================================================

            pred = pred.squeeze()

            # -------------------------------------------------
            # Safety check
            # -------------------------------------------------

            if pred.ndim != 2:

                raise RuntimeError(
                    f"Unexpected model output "
                    f"shape for {row['id']}: "
                    f"{tuple(pred.shape)}"
                )

            # =================================================
            # RESIZE PREDICTION BACK TO ORIGINAL GRID
            #
            # 518x518 -> 512x512
            #
            # Ground truth is NOT resized.
            # =================================================

            if (
                pred.shape[0]
                != original_h
                or
                pred.shape[1]
                != original_w
            ):

                pred = (
                    torch.nn.functional
                    .interpolate(
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

            # =================================================
            # CPU NUMPY
            # =================================================

            pred = (
                pred
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            # =================================================
            # LOAD GROUND TRUTH
            # =================================================

            gt = load_target(
                target_path
            )

            mask = load_mask(
                mask_path
            )

            # =================================================
            # SHAPE CHECK
            # =================================================

            if pred.shape != gt.shape:

                raise RuntimeError(
                    f"Shape mismatch: "
                    f"{row['id']} "
                    f"pred={pred.shape}, "
                    f"gt={gt.shape}"
                )

            if mask.shape != gt.shape:

                raise RuntimeError(
                    f"Mask shape mismatch: "
                    f"{row['id']} "
                    f"mask={mask.shape}, "
                    f"gt={gt.shape}"
                )

            # =================================================
            # FINITE VALIDATION
            # =================================================

            valid = (
                mask
                & np.isfinite(gt)
                & np.isfinite(pred)
            )

            if not np.any(valid):

                raise RuntimeError(
                    f"No valid pixels: "
                    f"{row['id']}"
                )

            # =================================================
            # PER-SAMPLE METRICS
            # =================================================

            m = metrics(
                pred,
                gt,
                valid,
            )

            rows.append(
                {
                    "id":
                        row["id"],

                    "aoi":
                        row["aoi"],

                    **m,
                }
            )

            # =================================================
            # COLLECT VALID PIXELS
            # =================================================

            all_pred.append(
                pred[valid]
            )

            all_gt.append(
                gt[valid]
            )

            # =================================================
            # PROGRESS
            # =================================================

            print(
                f"[{i + 1:02d}/"
                f"{len(val)}] "
                f"{row['id']} | "
                f"MAE={m['mae']:.4f} | "
                f"RMSE={m['rmse']:.4f} | "
                f"Bias={m['bias']:.4f} | "
                f"Corr={m['corr']:.4f}"
            )

    # ========================================================
    # CONCATENATE VALIDATION PIXELS
    # ========================================================

    pred_all = np.concatenate(
        all_pred
    ).astype(
        np.float64
    )

    gt_all = np.concatenate(
        all_gt
    ).astype(
        np.float64
    )

    print()
    print(
        "Total valid validation pixels:",
        len(pred_all),
    )

    # ========================================================
    # RAW GLOBAL METRICS
    # ========================================================

    raw_err = (
        pred_all
        - gt_all
    )

    raw_mae = np.mean(
        np.abs(
            raw_err
        )
    )

    raw_rmse = np.sqrt(
        np.mean(
            raw_err ** 2
        )
    )

    raw_bias = np.mean(
        raw_err
    )

    raw_corr = np.corrcoef(
        pred_all,
        gt_all,
    )[0, 1]

    # ========================================================
    # RAW RESULT
    # ========================================================

    print()
    print("=" * 78)

    print(
        "RAW VALIDATION RESULT"
    )

    print("=" * 78)

    print(
        f"MAE        : "
        f"{raw_mae:.6f}"
    )

    print(
        f"RMSE       : "
        f"{raw_rmse:.6f}"
    )

    print(
        f"Bias       : "
        f"{raw_bias:.6f}"
    )

    print(
        f"Correlation: "
        f"{raw_corr:.6f}"
    )

    # ========================================================
    # AFFINE CALIBRATION
    #
    # GT ≈ a * Prediction + b
    #
    # FIT ONLY ON VALIDATION.
    #
    # TEST SET REMAINS UNTOUCHED.
    # ========================================================

    X = np.column_stack(
        [
            pred_all,
            np.ones_like(
                pred_all
            ),
        ]
    )

    a, b = np.linalg.lstsq(
        X,
        gt_all,
        rcond=None,
    )[0]

    calibrated = (
        a
        * pred_all
        + b
    )

    cal_err = (
        calibrated
        - gt_all
    )

    cal_mae = np.mean(
        np.abs(
            cal_err
        )
    )

    cal_rmse = np.sqrt(
        np.mean(
            cal_err ** 2
        )
    )

    cal_bias = np.mean(
        cal_err
    )

    cal_corr = np.corrcoef(
        calibrated,
        gt_all,
    )[0, 1]

    # ========================================================
    # CALIBRATION RESULT
    # ========================================================

    print()
    print("=" * 78)

    print(
        "VALIDATION AFFINE CALIBRATION"
    )

    print("=" * 78)

    print()
    print(
        "Fitted relationship:"
    )

    print()

    print(
        f"GT ≈ "
        f"{a:.8f} × Prediction "
        f"+ {b:.8f}"
    )

    print()
    print(
        "Before calibration:"
    )

    print(
        f"MAE        : "
        f"{raw_mae:.6f}"
    )

    print(
        f"RMSE       : "
        f"{raw_rmse:.6f}"
    )

    print(
        f"Bias       : "
        f"{raw_bias:.6f}"
    )

    print()
    print(
        "After calibration:"
    )

    print(
        f"MAE        : "
        f"{cal_mae:.6f}"
    )

    print(
        f"RMSE       : "
        f"{cal_rmse:.6f}"
    )

    print(
        f"Bias       : "
        f"{cal_bias:.6f}"
    )

    print(
        f"Correlation: "
        f"{cal_corr:.6f}"
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    mae_improvement = (
        raw_mae
        - cal_mae
    )

    rmse_improvement = (
        raw_rmse
        - cal_rmse
    )

    if raw_mae > 0:

        mae_improvement_pct = (
            100.0
            * mae_improvement
            / raw_mae
        )

    else:

        mae_improvement_pct = np.nan

    if raw_rmse > 0:

        rmse_improvement_pct = (
            100.0
            * rmse_improvement
            / raw_rmse
        )

    else:

        rmse_improvement_pct = np.nan

    print()
    print(
        "Improvement:"
    )

    print(
        f"MAE improvement   : "
        f"{mae_improvement:.6f}"
    )

    print(
        f"MAE improvement % : "
        f"{mae_improvement_pct:.2f}%"
    )

    print(
        f"RMSE improvement  : "
        f"{rmse_improvement:.6f}"
    )

    print(
        f"RMSE improvement %: "
        f"{rmse_improvement_pct:.2f}%"
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    print()
    print(
        "DIAGNOSIS:"
    )

    if (
        cal_mae < raw_mae
        and
        cal_rmse < raw_rmse
    ):

        print(
            "Affine calibration improves "
            "validation accuracy."
        )

        print(
            "This supports the presence of "
            "a systematic scale/offset mismatch."
        )

        print(
            "ASTERRA's calibration layer may "
            "therefore be useful for converting "
            "model output into metric elevation."
        )

    else:

        print(
            "Affine calibration does not improve "
            "both validation MAE and RMSE."
        )

        print(
            "The validation error is therefore "
            "not explained by a simple global "
            "scale/offset transformation alone."
        )

        print(
            "Further investigation of model "
            "learning, domain adaptation, and "
            "spatial prediction quality is required."
        )

    # ========================================================
    # PER-SAMPLE SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        rows
    )

    print()
    print("=" * 78)

    print(
        "PER-SAMPLE VALIDATION SUMMARY"
    )

    print("=" * 78)

    if not results_df.empty:

        display_columns = [
            "id",
            "aoi",
            "mae",
            "rmse",
            "bias",
            "corr",
            "pred_mean",
            "pred_std",
            "gt_mean",
            "gt_std",
        ]

        print(
            results_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output = {

        "checkpoint":
            str(CHECKPOINT),

        "dataset":
            str(DATASET),

        "split":
            "validation",

        "samples":
            int(len(val)),

        "validation_aois":
            sorted(
                val["aoi"]
                .astype(str)
                .unique()
                .tolist()
            ),

        "model_input_size":
            MODEL_SIZE,

        "valid_pixels":
            int(
                len(pred_all)
            ),

        # ----------------------------------------------------
        # AFFINE PARAMETERS
        # ----------------------------------------------------

        "a":
            float(a),

        "b":
            float(b),

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        "raw_mae":
            float(raw_mae),

        "raw_rmse":
            float(raw_rmse),

        "raw_bias":
            float(raw_bias),

        "raw_corr":
            float(raw_corr),

        # ----------------------------------------------------
        # CALIBRATED
        # ----------------------------------------------------

        "calibrated_mae":
            float(cal_mae),

        "calibrated_rmse":
            float(cal_rmse),

        "calibrated_bias":
            float(cal_bias),

        "calibrated_corr":
            float(cal_corr),

        # ----------------------------------------------------
        # IMPROVEMENT
        # ----------------------------------------------------

        "mae_improvement":
            float(
                mae_improvement
            ),

        "mae_improvement_percent":
            float(
                mae_improvement_pct
            ),

        "rmse_improvement":
            float(
                rmse_improvement
            ),

        "rmse_improvement_percent":
            float(
                rmse_improvement_pct
            ),
    }

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        ROOT / "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SAVE TEXT REPORT
    # ========================================================

    out_path = (
        output_dir
        / "seo_validation_affine_calibration.txt"
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ASTERRA — "
            "S-EO VALIDATION AFFINE "
            "CALIBRATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        for key, value in output.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\n\n"
            "PER-SAMPLE RESULTS\n"
        )

        f.write(
            results_df.to_string(
                index=False
            )
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 78)

    print(
        "CALIBRATION EXPERIMENT COMPLETE"
    )

    print("=" * 78)

    print()
    print(
        "Saved:"
    )

    print(
        out_path
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Affine parameters were fitted "
        "ONLY on the validation split."
    )

    print(
        "The held-out test split was NOT used."
    )

    print(
        "Do NOT apply these parameters to the "
        "test set yet."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()