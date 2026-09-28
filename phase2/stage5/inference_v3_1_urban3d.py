"""
ASTERRA AI — Stage 5 Urban3D V3 Inference
=================================

Locked production checkpoint:

D:/Asterra AI/models/asterra_stage5/urban3d_v3/stage5_urban3d_v3_best.pth

Usage:

    python inference.py --input path/to/image.tif --output path/to/height.tif

The script prints RAW model statistics before non-negative clipping.
This is intentional for production validation/debugging.

The model predicts Urban3D V3-style height / nDSM-style height-above-ground.

It does NOT directly produce absolute geodetic elevation.
Absolute elevation requires the separate ASTERRA calibration engine.

Production inference uses tiled processing because the Stage-5 model was
trained on 512x512 crops. Each tile is resized to the locked 518x518
Depth Anything V2 model resolution, then mapped back to the source grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

import torch
import torch.nn.functional as F


# ============================================================
# ASTERRA CONFIGURATION
# ============================================================

ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"

# ASTERRA Stage-5 Urban3D V3 checkpoint.
# This is the checkpoint trained for 5 epochs with the V3 output-head
# configuration and recorded best validation MAE of ~3.815 m.
CHECKPOINT = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d_v3_1"
    / "stage5_urban3d_v3_best.pth"
)

# Locked Stage-5 architecture
ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

# Locked Depth Anything V2 model resolution
MODEL_SIZE = 518

# Stage-5 training crop size
TILE_SIZE = 512

# Overlap reduces seams between neighboring predictions.
OVERLAP = 64

# Small tiles at image boundaries are padded to TILE_SIZE.
PAD_MODE = "reflect"


# ============================================================
# DEPTH ANYTHING V2 IMPORT
# ============================================================

if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# MODEL LOADING
# ============================================================

def configure_v3_output_head(model):
    """Use the V3 Softplus output activation."""
    try:
        seq = model.depth_head.scratch.output_conv2
    except AttributeError as exc:
        raise RuntimeError(
            "Expected Depth Anything V2 DPT head: model.depth_head.scratch.output_conv2"
        ) from exc

    if len(seq) <= 3:
        raise RuntimeError(f"Unexpected Depth Anything V2 output head: {seq}")

    seq[3] = torch.nn.Softplus(beta=1.0, threshold=20.0)
    print("[V3] DPT output_conv2 final activation: Softplus")


def bypass_dpt_output_relu(model):
    """Diagnostic only: expose signed final-convolution output."""
    try:
        seq = model.depth_head.scratch.output_conv2
    except AttributeError as exc:
        raise RuntimeError(
            "Expected Depth Anything V2 DPT head: model.depth_head.scratch.output_conv2"
        ) from exc

    if len(seq) <= 3:
        raise RuntimeError(f"Unexpected Depth Anything V2 output head: {seq}")

    seq[3] = torch.nn.Identity()
    print("[DIAGNOSTIC] V3 final Softplus: BYPASSED")


def forward_pre_relu(model, x):
    """
    Run the locked Depth Anything V2 encoder + DPT head without the
    final functional F.relu() (diagnostic path) in DepthAnythingV2.forward().

    This exposes the signed output of the final depth convolution.
    """
    patch_size = getattr(model, "patch_size", 14)
    patch_h = x.shape[-2] // patch_size
    patch_w = x.shape[-1] // patch_size

    if hasattr(model, "intermediate_layer_idx") and hasattr(model, "encoder"):
        layer_idx = model.intermediate_layer_idx[model.encoder]
    else:
        # Locked ASTERRA architecture is Depth Anything V2 ViT-L.
        layer_idx = [4, 11, 17, 23]

    features = model.pretrained.get_intermediate_layers(
        x,
        layer_idx,
        return_class_token=True,
    )

    depth = model.depth_head(features, patch_h, patch_w)

    if not torch.is_tensor(depth):
        depth = torch.as_tensor(depth)

    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]

    return depth

def load_model(device: torch.device, bypass_final_relu: bool = False):
    """Load the locked ASTERRA Stage-5 Urban3D V3 checkpoint."""

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Production checkpoint not found:\n{CHECKPOINT}"
        )

    print("=" * 70)
    print("ASTERRA AI — Loading Stage 5 Urban3D V3 Model")
    print("=" * 70)

    print(f"[MODEL] Encoder        : {ENCODER}")
    print(f"[MODEL] Features       : {FEATURES}")
    print(f"[MODEL] Out channels   : {OUT_CHANNELS}")
    print(f"[MODEL] Model size     : {MODEL_SIZE}")
    print(f"[MODEL] Training tile  : {TILE_SIZE}")
    print(f"[MODEL] Overlap        : {OVERLAP}")
    print(f"[MODEL] Checkpoint     : {CHECKPOINT}")
    print(f"[MODEL] Device         : {device}")

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    # V3 architecture: final DPT activation is Softplus.
    configure_v3_output_head(model)

    if bypass_final_relu:
        bypass_dpt_output_relu(model)

    payload = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(payload, dict):
        state = payload.get("model_state_dict", payload)
    else:
        state = payload

    if not isinstance(state, dict):
        raise RuntimeError(
            "Invalid checkpoint format. Expected a state dictionary."
        )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Incompatible production checkpoint.\n"
            f"Missing keys: {missing[:20]}"
        )

    if unexpected:
        print(
            f"[WARNING] Unexpected checkpoint keys: "
            f"{len(unexpected)}"
        )

    model.to(device)
    model.eval()

    epoch = None
    best_val_mae = None

    if isinstance(payload, dict):
        epoch = payload.get("epoch")
        best_val_mae = payload.get("best_val_mae")

    final_conv = model.depth_head.scratch.output_conv2[2]
    w = final_conv.weight.detach().cpu().numpy()
    b = final_conv.bias.detach().cpu().numpy()
    print(f"[V3] Final projection shape: {tuple(final_conv.weight.shape)}")
    print(f"[V3] Final weight stats: min={w.min():.8f} max={w.max():.8f} mean={w.mean():.8f} std={w.std():.8f}")
    print(f"[V3] Final bias: {b}")

    print("[OK] Checkpoint loaded successfully")
    print(f"[OK] Epoch              : {epoch}")
    print(f"[OK] Best validation MAE: {best_val_mae}")
    print("=" * 70)

    return model


# ============================================================
# RGB / GEOTIFF READING
# ============================================================

def read_rgb(path: Path):
    """
    Read an RGB/RGB-like GeoTIFF.

    1 band  -> replicate to RGB
    2 bands -> duplicate first band
    3+ bands -> use first three bands

    IMPORTANT:
    Rasterio indexes are explicitly supplied so a 4-band TIFF
    is safely read into a 3-band RGB array.
    """

    print(f"[INPUT] Reading: {path}")

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

        width = src.width
        height = src.height
        count = src.count

        print(f"[INPUT] Width        : {width}")
        print(f"[INPUT] Height       : {height}")
        print(f"[INPUT] Bands        : {count}")
        print(f"[INPUT] CRS          : {crs}")
        print(f"[INPUT] Transform    : {transform}")

        if count < 1:
            raise ValueError("Input raster contains no bands.")

        # FIX:
        # Explicitly choose source band indexes. Without indexes=,
        # Rasterio may try to read all 4 bands into a 3-band buffer.
        if count == 1:
            indexes = [1]
        elif count == 2:
            indexes = [1, 2]
        else:
            indexes = [1, 2, 3]

        print(f"[INPUT] Reading bands : {indexes}")

        arr = src.read(
            indexes=indexes,
            out_shape=(len(indexes), height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    # Convert to exactly 3 channels.
    if arr.shape[0] == 1:
        print("[INPUT] Single-band image -> replicating to RGB.")
        arr = np.repeat(arr, 3, axis=0)

    elif arr.shape[0] == 2:
        print("[INPUT] Two-band image -> duplicating first band.")
        arr = np.concatenate([arr, arr[:1]], axis=0)

    else:
        arr = arr[:3]

    if arr.shape[0] != 3:
        raise RuntimeError(
            f"RGB conversion failed. Got shape {arr.shape}"
        )

    # Same basic input convention used by the Stage-5 trainer.
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    max_value = float(np.max(arr))

    if max_value > 1.5:
        print(
            f"[INPUT] Image max={max_value:.4f}; "
            "scaling by 255."
        )
        arr /= 255.0

    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)

    print(f"[INPUT] Final RGB shape: {arr.shape}")
    print(
        f"[INPUT] Value range    : "
        f"{float(arr.min()):.6f} -> {float(arr.max()):.6f}"
    )

    return arr, profile, transform, crs


# ============================================================
# SINGLE-TILE MODEL PREDICTION
# ============================================================

@torch.inference_mode()
def predict_tile(
    model,
    tile_np: np.ndarray,
    device: torch.device,
    bypass_final_relu: bool = False,
):
    """
    Predict one CxHxW tile.

    The tile is resized to 518x518 for the locked model and
    then resized back to the original tile dimensions.
    """

    if tile_np.ndim != 3 or tile_np.shape[0] != 3:
        raise ValueError(
            f"Expected tile shape 3xHxW, got {tile_np.shape}"
        )

    tile_h = tile_np.shape[-2]
    tile_w = tile_np.shape[-1]

    image = torch.from_numpy(tile_np).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )

    image_model = F.interpolate(
        image,
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    if device.type == "cuda":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            if bypass_final_relu:
                pred = forward_pre_relu(model, image_model)
            else:
                pred = model(image_model)
    else:
        pred = model(image_model)

    if isinstance(pred, dict):
        found = False

        for key in ("metric_depth", "depth", "out", "pred"):
            if key in pred:
                pred = pred[key]
                found = True
                break

        if not found:
            raise RuntimeError(
                "Model returned a dictionary without a supported "
                "prediction key."
            )

    if not torch.is_tensor(pred):
        pred = torch.as_tensor(pred)

    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    elif pred.ndim != 4:
        raise RuntimeError(
            f"Unexpected model output shape: {pred.shape}"
        )

    pred = pred.float()

    pred = F.interpolate(
        pred,
        size=(tile_h, tile_w),
        mode="bilinear",
        align_corners=False,
    )

    result = pred[0, 0].cpu().numpy().astype(np.float32)

    # IMPORTANT:
    # Do NOT clip here. We need the raw model output to diagnose
    # whether the network is predicting negative values everywhere.
    # Non-negative clipping is applied only AFTER the complete raster
    # has been assembled and raw statistics have been printed.

    return result


# ============================================================
# TILE HELPERS
# ============================================================

def make_positions(length: int, tile_size: int, stride: int):
    """Generate tile start positions that always cover the full axis."""

    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))

    final_position = length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


def make_blend_window(size: int):
    """
    Create a smooth 2D blending window.

    Center pixels receive higher weight than tile edges,
    reducing visible seams when overlapping tiles are merged.
    """

    if size <= 1:
        return np.ones((size, size), dtype=np.float32)

    x = np.hanning(size).astype(np.float32)

    # Avoid exact zero at borders so every covered pixel
    # still receives a finite contribution.
    x = np.maximum(x, 1e-3)

    window = np.outer(x, x).astype(np.float32)

    return window


# ============================================================
# TILED INFERENCE
# ============================================================

def predict_tiled(
    model,
    image_np: np.ndarray,
    device: torch.device,
    bypass_final_relu: bool = False,
):
    """
    Run tiled inference over the complete raster.

    Training crops were 512x512, therefore production inference
    processes 512x512 tiles instead of shrinking an entire
    satellite scene to a single 518x518 image.

    Overlapping predictions are blended into the final raster.
    """

    channels, height, width = image_np.shape

    if channels != 3:
        raise ValueError(
            f"Expected 3-channel RGB input, got {image_np.shape}"
        )

    if TILE_SIZE <= OVERLAP:
        raise ValueError(
            "TILE_SIZE must be greater than OVERLAP."
        )

    stride = TILE_SIZE - OVERLAP

    y_positions = make_positions(
        height,
        TILE_SIZE,
        stride,
    )

    x_positions = make_positions(
        width,
        TILE_SIZE,
        stride,
    )

    total_tiles = len(y_positions) * len(x_positions)

    print("=" * 70)
    print("ASTERRA TILED INFERENCE")
    print("=" * 70)
    print(f"[TILE] Raster size : {width} x {height}")
    print(f"[TILE] Tile size   : {TILE_SIZE} x {TILE_SIZE}")
    print(f"[TILE] Model size  : {MODEL_SIZE} x {MODEL_SIZE}")
    print(f"[TILE] Overlap     : {OVERLAP} px")
    print(f"[TILE] Stride      : {stride} px")
    print(f"[TILE] Tiles       : {total_tiles}")

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    blend_window = make_blend_window(TILE_SIZE)

    tile_number = 0

    for y in y_positions:
        for x in x_positions:

            tile_number += 1

            y_end = min(
                y + TILE_SIZE,
                height,
            )

            x_end = min(
                x + TILE_SIZE,
                width,
            )

            actual_h = y_end - y
            actual_w = x_end - x

            tile = image_np[
                :,
                y:y_end,
                x:x_end,
            ]

            # Pad boundary tiles to 512x512.
            pad_h = TILE_SIZE - actual_h
            pad_w = TILE_SIZE - actual_w

            if pad_h > 0 or pad_w > 0:

                if PAD_MODE == "reflect":

                    # Reflect padding requires at least 2 pixels
                    # on the relevant axis. For extremely tiny
                    # rasters, fall back to edge padding.
                    if actual_h > 1 and actual_w > 1:
                        tile = np.pad(
                            tile,
                            (
                                (0, 0),
                                (0, pad_h),
                                (0, pad_w),
                            ),
                            mode="reflect",
                        )
                    else:
                        tile = np.pad(
                            tile,
                            (
                                (0, 0),
                                (0, pad_h),
                                (0, pad_w),
                            ),
                            mode="edge",
                        )

                else:
                    tile = np.pad(
                        tile,
                        (
                            (0, 0),
                            (0, pad_h),
                            (0, pad_w),
                        ),
                        mode="edge",
                    )

            pred_tile = predict_tile(
                model,
                tile.astype(np.float32),
                device,
                bypass_final_relu=bypass_final_relu,
            )

            # Crop prediction back to the real raster area.
            pred_tile = pred_tile[
                :actual_h,
                :actual_w,
            ]

            window = blend_window[
                :actual_h,
                :actual_w,
            ]

            prediction_sum[
                y:y_end,
                x:x_end,
            ] += pred_tile * window

            weight_sum[
                y:y_end,
                x:x_end,
            ] += window

            print(
                f"[TILE] {tile_number:03d}/{total_tiles:03d} "
                f"position=({x},{y}) "
                f"size={actual_w}x{actual_h}"
            )

            # Release cached CUDA memory between tiles.
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Every pixel should have received at least one prediction.
    if np.any(weight_sum <= 0):
        raise RuntimeError(
            "Tiled inference left uncovered pixels."
        )

    # --------------------------------------------------------
    # RAW MODEL OUTPUT
    # --------------------------------------------------------
    # Do not apply ReLU / non-negative clipping yet.
    # This is the actual output produced by the checkpoint.
    # --------------------------------------------------------

    result = prediction_sum / weight_sum

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    return result


# ============================================================
# GEOTIFF WRITING
# ============================================================

def write_height(
    path: Path,
    height: np.ndarray,
    profile,
    transform,
    crs,
):
    """Write the height prediction as a Float32 GeoTIFF."""

    profile = profile.copy()

    # Output is a single-band prediction raster.
    # Preserve spatial metadata from the input.
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=3,
        nodata=-9999.0,
        height=height.shape[0],
        width=height.shape[1],
        transform=transform,
        crs=crs,
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            height.astype(np.float32),
            1,
        )

        dst.set_band_description(
            1,
            "ASTERRA Urban3D V3-style height / nDSM",
        )

    print(f"[OUTPUT] GeoTIFF written: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ASTERRA Stage-5 Urban3D V3 inference"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input RGB/RGB-like GeoTIFF",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output height GeoTIFF",
    )

    parser.add_argument(
        "--bypass-final-relu",
        action="store_true",
        help=(
            "DIAGNOSTIC ONLY: bypass the final V3 Softplus and inspect the "
            "signed final-convolution prediction. In this mode the saved GeoTIFF "
            "is the signed raw diagnostic output, not a production height map."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        print(
            f"[CUDA] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"[CUDA] CUDA available: "
            f"{torch.cuda.is_available()}"
        )
    else:
        print(
            "[WARNING] CUDA is not available. "
            "Running on CPU."
        )

    # --------------------------------------------------------
    # Load exact Stage-5 Urban3D checkpoint
    # --------------------------------------------------------

    checkpoint_resolved = CHECKPOINT.resolve()
    if checkpoint_resolved.name != "stage5_urban3d_v3_best.pth":
        raise RuntimeError(
            "Refusing to run: this script is locked to V3 checkpoint "
            "stage5_urban3d_v3_best.pth."
        )

    print(f"[MODEL] Exact checkpoint: {checkpoint_resolved}")
    if args.bypass_final_relu:
        print("[MODEL] Output mode: SIGNED FINAL-CONV DIAGNOSTIC")
    else:
        print("[MODEL] Output mode: NORMAL ASTERRA STAGE-5 V3")

    model = load_model(
        device,
        bypass_final_relu=args.bypass_final_relu,
    )

    # --------------------------------------------------------
    # Read source raster
    # --------------------------------------------------------

    image, profile, transform, crs = read_rgb(
        input_path
    )

    print(
        f"[INPUT] Array shape: {image.shape}"
    )

    # --------------------------------------------------------
    # Production tiled inference
    # --------------------------------------------------------

    height = predict_tiled(
        model,
        image,
        device,
        bypass_final_relu=args.bypass_final_relu,
    )

    # --------------------------------------------------------
    # CRITICAL RAW MODEL DIAGNOSTIC
    # --------------------------------------------------------
    # The previous version clipped every tile before this point.
    # That could hide a model that is predicting negative values.
    #
    # We now inspect the raw checkpoint output FIRST.
    # --------------------------------------------------------

    height_raw = np.nan_to_num(
        height,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    negative_pct = 100.0 * float(
        np.mean(height_raw < 0.0)
    )

    zero_pct = 100.0 * float(
        np.mean(height_raw == 0.0)
    )

    positive_pct = 100.0 * float(
        np.mean(height_raw > 0.0)
    )

    print("=" * 70)
    print("RAW MODEL PREDICTION DIAGNOSTIC")
    print("=" * 70)

    print(
        f"[RAW] min             = "
        f"{float(np.min(height_raw)):.8f}"
    )

    print(
        f"[RAW] max             = "
        f"{float(np.max(height_raw)):.8f}"
    )

    print(
        f"[RAW] mean            = "
        f"{float(np.mean(height_raw)):.8f}"
    )

    print(
        f"[RAW] std             = "
        f"{float(np.std(height_raw)):.8f}"
    )

    print(
        f"[RAW] negative pixels = "
        f"{negative_pct:.4f}%"
    )

    print(
        f"[RAW] zero pixels     = "
        f"{zero_pct:.4f}%"
    )

    print(
        f"[RAW] positive pixels = "
        f"{positive_pct:.4f}%"
    )

    # --------------------------------------------------------
    # Basic collapse diagnosis
    # --------------------------------------------------------

    if float(np.std(height_raw)) < 1e-8:
        print(
            "[DIAGNOSTIC] CRITICAL: raw prediction has essentially "
            "zero variance."
        )

    elif positive_pct == 0.0:
        print(
            "[DIAGNOSTIC] CRITICAL: model predicts no positive "
            "height anywhere in this scene."
        )

    elif negative_pct > 99.0:
        print(
            "[DIAGNOSTIC] CRITICAL: >99% of raw predictions are "
            "negative. Non-negative clipping would collapse "
            "the output to almost all zeros."
        )

    else:
        print(
            "[DIAGNOSTIC] Raw output contains positive spatial "
            "signal. Continue with visual/quantitative validation."
        )

    # --------------------------------------------------------
    # Apply Urban3D semantic constraint AFTER diagnostics.
    #
    # Training target:
    #
    #     nDSM = DSM - DTM
    #
    # Negative target values were clipped to zero.
    # --------------------------------------------------------

    if args.bypass_final_relu:
        # Diagnostic mode intentionally preserves signed values.
        # Do NOT interpret these values as final physical heights.
        height = height_raw.astype(np.float32)
        print("=" * 70)
        print("SIGNED FINAL-CONV DIAGNOSTIC OUTPUT")
        print("=" * 70)
        print("[OUTPUT] WARNING: saved raster is signed final-convolution diagnostic data.")
        print("[OUTPUT] WARNING: do NOT use it as the production height map.")
    else:
        # Urban3D target semantics require non-negative height-above-ground.
        height = np.maximum(
            height_raw,
            0.0,
        ).astype(np.float32)

        print("=" * 70)
        print("CLIPPED ASTERRA OUTPUT")
        print("=" * 70)

    print(
        f"[OUTPUT] Min       : "
        f"{float(np.min(height)):.4f} m"
    )

    print(
        f"[OUTPUT] Max       : "
        f"{float(np.max(height)):.4f} m"
    )

    print(
        f"[OUTPUT] Mean      : "
        f"{float(np.mean(height)):.4f} m"
    )

    print(
        f"[OUTPUT] Std       : "
        f"{float(np.std(height)):.4f} m"
    )

    print(
        f"[OUTPUT] Size      : "
        f"{height.shape[1]} x {height.shape[0]}"
    )

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_height(
        output_path,
        height,
        profile,
        transform,
        crs,
    )

    print("=" * 70)
    print("ASTERRA INFERENCE COMPLETE")
    print("=" * 70)

    print(f"[OK] Input          : {input_path}")
    print(f"[OK] Output         : {output_path}")
    print(
        "[OK] Prediction     : "
        + ("SIGNED final-convolution diagnostic output" if args.bypass_final_relu
           else "Urban3D V3-style height / nDSM")
    )
    print(
        "[IMPORTANT] Absolute elevation is NOT produced here."
    )
    if args.bypass_final_relu:
        print(
            "[NEXT] Compare signed RAW output against the normal run. "
            "If it is strongly negative everywhere, the active ReLU explains "
            "the zero production map; validate on an Urban3D crop before any calibration."
        )
    else:
        print(
            "[NEXT] If RAW output is collapsed/negative, do NOT calibrate yet. "
            "Validate the checkpoint on an Urban3D validation crop first."
        )


if __name__ == "__main__":
    main()
