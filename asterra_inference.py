import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ============================================================
# ASTERRA FINAL INFERENCE ENGINE
# ============================================================

ROOT = Path(r"D:\Asterra AI")

MODEL_PATH = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d_v3_1"
    / "ASTERRA_FINAL_HEIGHT_MODEL.pth"
)

DEPTH_ANYTHING_ROOT = (
    ROOT
    / "external"
    / "Depth-Anything-V2"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# IMPORTANT
#
# Training crop:
#     512 x 512
#
# DINOv2 ViT-L/14 model input:
#     518 x 518
#
# 518 / 14 = 37
# ------------------------------------------------------------

CROP_SIZE = 512
MODEL_SIZE = 518


# ============================================================
# IMPORT DEPTH ANYTHING V2
# ============================================================

sys.path.insert(
    0,
    str(DEPTH_ANYTHING_ROOT)
)

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# MODEL
# ============================================================

def build_model():

    print()
    print("=" * 78)
    print("ASTERRA FINAL HEIGHT MODEL")
    print("=" * 78)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Device:", DEVICE)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Final model not found:\n"
            f"{MODEL_PATH}"
        )

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
    # Urban3D V3.1 uses Softplus output.
    #
    # The original DPT head contains:
    #
    # output_conv2 =
    # [
    #     Conv2d,
    #     ReLU,
    #     Conv2d,
    #     ReLU,
    #     Identity
    # ]
    #
    # For V3.1 the Softplus modification is represented by
    # replacing the ReLU at index 3.
    #
    # IMPORTANT:
    # This must match the actual checkpoint architecture.
    # --------------------------------------------------------

    model.depth_head.scratch.output_conv2[3] = nn.Softplus(
        beta=1.0,
        threshold=20.0,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print()
    print("Checkpoint loaded.")

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state = checkpoint[
            "model_state_dict"
        ]

        print(
            "Checkpoint format: "
            "ASTERRA training checkpoint"
        )

    elif (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):

        state = checkpoint["model"]

        print(
            "Checkpoint format: "
            "model dictionary"
        )

    else:

        state = checkpoint

        print(
            "Checkpoint format: "
            "raw state dictionary"
        )

    # --------------------------------------------------------
    # Strict loading is intentional.
    #
    # We do NOT want silently missing layers in the final
    # inference model.
    # --------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=False,
        )
    )

    print()
    print("Checkpoint verification:")
    print(
        "Missing keys    :",
        len(missing),
    )

    print(
        "Unexpected keys :",
        len(unexpected),
    )

    if missing:

        print()
        print("Missing:")
        for key in missing[:20]:
            print(" ", key)

        raise RuntimeError(
            "Final model has missing checkpoint keys."
        )

    if unexpected:

        print()
        print("Unexpected:")
        for key in unexpected[:20]:
            print(" ", key)

        raise RuntimeError(
            "Final model has unexpected checkpoint keys."
        )

    model = model.to(DEVICE)
    model.eval()

    print()
    print("FINAL MODEL READY")

    return model


# ============================================================
# RGB PREPROCESSING
# ============================================================

def preprocess_rgb(image_path):
    """
    Exact ASTERRA V3.1 inference preprocessing.

    Original crop:
        512 x 512

    Model input:
        518 x 518

    Ground truth is NEVER resized here.
    """

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"RGB image not found:\n"
            f"{image_path}"
        )

    # --------------------------------------------------------
    # Load RGB
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    original_width, original_height = (
        image.size
    )

    print()
    print("Input image:")
    print(" ", image_path)

    print(
        "Original size:",
        f"{original_width}x{original_height}",
    )

    # --------------------------------------------------------
    # RGB -> float32
    # --------------------------------------------------------

    img = np.asarray(
        image,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # uint8 -> [0,1]
    # --------------------------------------------------------

    if img.max() > 1.5:

        img /= 255.0

    img = np.clip(
        img,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 512 -> 518
    #
    # BICUBIC here is used for image preprocessing.
    # --------------------------------------------------------

    resized = Image.fromarray(
        (
            img * 255.0
        ).astype(
            np.uint8
        )
    )

    resized = resized.resize(
        (
            MODEL_SIZE,
            MODEL_SIZE,
        ),
        Image.Resampling.BICUBIC,
    )

    img = np.asarray(
        resized,
        dtype=np.float32,
    ) / 255.0

    img = np.clip(
        img,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # HWC -> BCHW
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        img
    ).permute(
        2,
        0,
        1,
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        non_blocking=True,
    )

    print(
        "Model input:",
        tuple(tensor.shape),
    )

    return (
        tensor,
        original_width,
        original_height,
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

@torch.no_grad()
def predict(
    model,
    image_path,
):
    """
    Run frozen ASTERRA model.

    Returns:
        nDSM prediction on the original image grid.
    """

    (
        rgb,
        original_width,
        original_height,
    ) = preprocess_rgb(
        image_path
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    prediction = model(
        rgb
    )

    if isinstance(
        prediction,
        (tuple, list),
    ):

        prediction = prediction[0]

    print(
        "Raw model output:",
        tuple(prediction.shape),
    )

    # --------------------------------------------------------
    # Remove batch/channel dimensions
    # --------------------------------------------------------

    prediction = prediction.squeeze()

    if prediction.ndim != 2:

        raise RuntimeError(
            "Unexpected model output shape: "
            f"{tuple(prediction.shape)}"
        )

    # --------------------------------------------------------
    # 518 -> original image grid
    #
    # This changes ONLY the prediction grid.
    # --------------------------------------------------------

    if (
        prediction.shape[0]
        != original_height
        or
        prediction.shape[1]
        != original_width
    ):

        prediction = (
            F.interpolate(
                prediction
                .unsqueeze(0)
                .unsqueeze(0),
                size=(
                    original_height,
                    original_width,
                ),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(
                0,
                1,
            )
        )

    prediction = (
        prediction
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Numerical diagnostics
    # --------------------------------------------------------

    finite = np.isfinite(
        prediction
    )

    if not np.any(finite):

        raise RuntimeError(
            "Model produced no finite predictions."
        )

    values = prediction[
        finite
    ]

    print()
    print("=" * 78)
    print("ASTERRA nDSM PREDICTION")
    print("=" * 78)

    print(
        "Shape :",
        prediction.shape,
    )

    print(
        "Min   :",
        float(values.min()),
    )

    print(
        "Max   :",
        float(values.max()),
    )

    print(
        "Mean  :",
        float(values.mean()),
    )

    print(
        "Std   :",
        float(values.std()),
    )

    print(
        "Median:",
        float(np.median(values)),
    )

    print(
        "P95   :",
        float(np.percentile(values, 95)),
    )

    print(
        "P99   :",
        float(np.percentile(values, 99)),
    )

    # --------------------------------------------------------
    # nDSM should be non-negative for this model.
    # We do NOT silently clip it here.
    #
    # If negative values appear, we want to see them.
    # --------------------------------------------------------

    negative_fraction = (
        np.mean(
            prediction[
                finite
            ] < 0
        )
    )

    print(
        "Negative fraction:",
        float(
            negative_fraction
        ),
    )

    return prediction


# ============================================================
# SAVE NUMPY
# ============================================================

def save_npy(
    prediction,
    output_path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        prediction,
    )

    print()
    print(
        "Saved prediction:",
        output_path,
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_height_png(
    prediction,
    output_path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    finite = np.isfinite(
        prediction
    )

    values = prediction[
        finite
    ]

    # --------------------------------------------------------
    # Robust visualization range
    # --------------------------------------------------------

    low = np.percentile(
        values,
        2,
    )

    high = np.percentile(
        values,
        98,
    )

    if high <= low:

        high = low + 1.0

    normalized = (
        prediction - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    normalized[
        ~finite
    ] = 0.0

    image = (
        normalized * 255.0
    ).astype(
        np.uint8
    )

    Image.fromarray(
        image,
        mode="L",
    ).save(
        output_path
    )

    print(
        "Saved visualization:",
        output_path,
    )


# ============================================================
# TEST
# ============================================================

def main():

    print()
    print("#" * 78)
    print("# ASTERRA FINAL INFERENCE TEST")
    print("#" * 78)

    print()
    print(
        "Final checkpoint:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Target:",
        "nDSM = DSM - DTM",
    )

    print(
        "Crop size:",
        CROP_SIZE,
    )

    print(
        "Model size:",
        MODEL_SIZE,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We deliberately use one KNOWN Urban3D validation crop.
    #
    # Replace this path only if your local filename differs.
    # --------------------------------------------------------

    image_path = (
        ROOT
        / "datasets"
        / "Urban3D"
        / "train"
        / "Inputs"
        / "JAX_Tile_004_RGB.tif"
    )

    # --------------------------------------------------------
    # Build frozen model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction = predict(
        model,
        image_path,
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "outputs"
        / "asterra_final_inference"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    npy_path = (
        output_dir
        / "JAX_Tile_004_nDSM.npy"
    )

    png_path = (
        output_dir
        / "JAX_Tile_004_nDSM.png"
    )

    save_npy(
        prediction,
        npy_path,
    )

    save_height_png(
        prediction,
        png_path,
    )

    print()
    print("#" * 78)
    print("# FINAL INFERENCE TEST COMPLETE")
    print("#" * 78)

    print()
    print(
        "Output directory:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":

    main()