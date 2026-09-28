"""ASTERRA Stage 5 tiled nDSM inference.

The model architecture and tiled blending policy in this module are the
production inference implementation. Both the CLI entry point and the
FastAPI worker call :func:`run_inference` so they cannot drift apart.
"""

from pathlib import Path
import sys
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "asterra_stage5" / "urban3d_v3_1" / "ASTERRA_FINAL_HEIGHT_MODEL.pth"
INPUT_PATH = ROOT / "datasets" / "Urban3D" / "train" / "Inputs" / "JAX_Tile_004_RGB.tif"
OUTPUT_DIR = ROOT / "outputs" / "asterra_tiled_inference"
OUTPUT_NPY = OUTPUT_DIR / "JAX_Tile_004_nDSM_tiled.npy"
OUTPUT_TIF = OUTPUT_DIR / "JAX_Tile_004_nDSM_tiled.tif"

TILE_SIZE = 512
OVERLAP = 128
MODEL_INPUT_SIZE = 518
EXPECTED_CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEPTH_ANYTHING_DIR = ROOT / "external" / "Depth-Anything-V2"
if not DEPTH_ANYTHING_DIR.exists():
    raise FileNotFoundError(f"Depth Anything V2 directory not found: {DEPTH_ANYTHING_DIR}")
sys.path.insert(0, str(DEPTH_ANYTHING_DIR))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


def check_required_paths(input_path=INPUT_PATH, model_path=MODEL_PATH):
    for name, path in (("Model", Path(model_path)), ("Input", Path(input_path))):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")


def build_model(checkpoint_path=None, device=None):
    """Build and strictly load the frozen ASTERRA Stage 5 checkpoint."""
    checkpoint_path = Path(checkpoint_path or MODEL_PATH)
    device = device or DEVICE
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.depth_head.scratch.output_conv2[3] = nn.Softplus(beta=1, threshold=20)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("ASTERRA checkpoint is not a dictionary.")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    else:
        raise RuntimeError(
            "Could not find model weights in checkpoint. Expected model_state_dict, "
            "state_dict, or model."
        )
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "ASTERRA checkpoint loading failed: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    model = model.to(device)
    model.eval()
    return model


def get_positions(length):
    if length <= TILE_SIZE:
        return [0]
    stride = TILE_SIZE - OVERLAP
    if stride <= 0:
        raise ValueError("OVERLAP must be smaller than TILE_SIZE.")
    positions = list(range(0, length - TILE_SIZE + 1, stride))
    last = length - TILE_SIZE
    if positions[-1] != last:
        positions.append(last)
    return positions


def preprocess_rgb(tile):
    if tile.ndim != 3 or tile.shape[2] != EXPECTED_CHANNELS:
        raise RuntimeError(f"Expected HxWx3 tile, got {tile.shape}")
    tile = tile.astype(np.float32)
    finite = np.isfinite(tile)
    tile_max = float(np.nanmax(tile)) if finite.any() else 1.0
    tile_min = float(np.nanmin(tile)) if finite.any() else 0.0
    if tile_max <= 1.0 and tile_min >= 0.0:
        pass
    elif tile_max <= 255.0 and tile_min >= 0.0:
        tile = tile / 255.0
    else:
        # Direct CLI callers may provide uint16 scientific imagery. The API
        # normalizes at ingest, but keep the callable inference entry point
        # safe and useful when invoked independently.
        for band in range(EXPECTED_CHANNELS):
            values = tile[:, :, band]
            valid = np.isfinite(values)
            if not valid.any():
                tile[:, :, band] = 0.0
                continue
            low, high = np.percentile(values[valid], (2.0, 98.0)).astype(np.float32)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = float(values[valid].min()), float(values[valid].max())
            tile[:, :, band] = (values - low) / max(float(high - low), 1e-6)
    tile = np.clip(np.nan_to_num(tile, nan=0.0), 0.0, 1.0)
    tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(tensor, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mode="bicubic", align_corners=False)


@torch.inference_mode()
def predict_tile(model, tile, device=None):
    tensor = preprocess_rgb(tile).to(device or DEVICE, non_blocking=True)
    prediction = model(tensor)
    if prediction.ndim == 4:
        if prediction.shape[1] != 1:
            raise RuntimeError(f"Expected one output channel, got {prediction.shape}")
        prediction = prediction[:, 0]
    elif prediction.ndim != 3:
        raise RuntimeError(f"Unexpected model output shape: {prediction.shape}")
    prediction = F.interpolate(
        prediction.unsqueeze(1), size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False
    )[0, 0]
    prediction = prediction.cpu().numpy().astype(np.float32)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Model produced NaN or Inf values.")
    return prediction


def _pad_tile_for_model(tile, target_size=TILE_SIZE):
    """Pad a partial tile without introducing artificial black scenery.

    Map AOIs are often smaller than the model's 512 px inference window.  The
    old zero-padding path made most of the model input pure black, so the
    network interpreted the AOI boundary as a dramatic scene edge.  Reflective
    padding keeps the local colour/texture statistics continuous.  Small
    complete images are centred so their valid pixels are not all packed into
    the top-left corner of the model frame.

    Returns ``(padded, crop)`` where ``crop`` maps the model prediction back to
    the original tile.
    """
    if tile.ndim != 3 or tile.shape[2] != EXPECTED_CHANNELS:
        raise RuntimeError(f"Expected HxWx3 tile, got {tile.shape}")
    height, width = tile.shape[:2]
    if height > target_size or width > target_size:
        raise ValueError(f"Tile {tile.shape} is larger than the model window {target_size}.")
    if height == target_size and width == target_size:
        return tile, (0, 0, height, width)

    # Centre only a complete image that is smaller on both axes. If one axis
    # already fills the model window, this is an edge tile and its origin must
    # remain at the top-left while only the missing trailing edge is padded.
    if height < target_size and width < target_size:
        top = (target_size - height) // 2
        left = (target_size - width) // 2
    else:
        top = left = 0
    bottom = target_size - height - top
    right = target_size - width - left
    # A map input is required to be at least 128 px on each axis, but keep the
    # fallback safe for direct library callers that provide a 1 px strip.
    mode = "reflect" if height > 1 and width > 1 else "edge"
    padded = np.pad(tile, ((top, bottom), (left, right), (0, 0)), mode=mode)
    return padded, (top, left, height, width)


def _stats(values):
    valid = values[np.isfinite(values)]
    if not valid.size:
        raise RuntimeError("Inference produced no finite pixels.")
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "valid_pixels": int(valid.size),
        "total_pixels": int(values.size),
    }


def run_inference(
    input_path,
    output_npy,
    output_tif,
    checkpoint_path=None,
    device=None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    model=None,
):
    """Run tiled ASTERRA inference and write nDSM NPY + GeoTIFF artifacts.

    ``progress_callback`` receives ``{"completed": int, "total": int,
    "fraction": float}`` after each tile. A preloaded ``model`` may be passed
    by the long-lived API worker; otherwise the frozen checkpoint is loaded.
    """
    input_path = Path(input_path)
    output_npy = Path(output_npy)
    output_tif = Path(output_tif)
    checkpoint_path = Path(checkpoint_path or MODEL_PATH)
    device = device or DEVICE
    check_required_paths(input_path, checkpoint_path)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        if src.count < 3:
            raise RuntimeError(f"Input raster has only {src.count} band(s). ASTERRA expects RGB.")
        image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        profile = src.profile.copy()
        height, width = src.height, src.width
        source_valid = np.ones((height, width), dtype=bool)
        for band in range(1, min(src.count, 3) + 1):
            source_valid &= src.read_masks(band) > 0

    model = model or build_model(checkpoint_path=checkpoint_path, device=device)
    ys, xs = get_positions(height), get_positions(width)
    total_tiles = len(ys) * len(xs)
    prediction_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    axis = np.linspace(0, np.pi, TILE_SIZE, dtype=np.float32)
    window_1d = 0.5 - 0.5 * np.cos(axis)
    window_2d = (window_1d[:, None] * window_1d[None, :] * 0.999 + 0.001).astype(np.float32)

    tile_number = 0
    for y0 in ys:
        for x0 in xs:
            tile_number += 1
            y1, x1 = min(y0 + TILE_SIZE, height), min(x0 + TILE_SIZE, width)
            tile = image[y0:y1, x0:x1, :]
            original_h, original_w = tile.shape[:2]
            if original_h != TILE_SIZE or original_w != TILE_SIZE:
                tile, crop = _pad_tile_for_model(tile)
            else:
                crop = (0, 0, original_h, original_w)
            predicted_tile = predict_tile(model, tile, device=device)
            top, left, crop_h, crop_w = crop
            prediction = predicted_tile[top:top + crop_h, left:left + crop_w]
            local_window = window_2d[:original_h, :original_w]
            prediction_sum[y0:y1, x0:x1] += prediction * local_window
            weight_sum[y0:y1, x0:x1] += local_window
            if progress_callback:
                progress_callback({"completed": tile_number, "total": total_tiles, "fraction": tile_number / total_tiles})

    if np.any(weight_sum <= 0):
        raise RuntimeError("Some pixels received no tiled prediction.")
    final_prediction = np.maximum(prediction_sum / weight_sum, 0.0).astype(np.float32)
    final_prediction[~source_valid] = np.nan
    if not np.isfinite(final_prediction[source_valid]).all():
        raise RuntimeError("Final tiled prediction contains NaN or Inf in valid pixels.")

    np.save(output_npy, final_prediction)
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff", dtype="float32", count=1, height=height, width=width,
        nodata=-9999.0, compress="deflate", predictor=3,
    )
    output_array = np.where(source_valid, final_prediction, -9999.0).astype(np.float32)
    with rasterio.open(output_tif, "w", **output_profile) as dst:
        dst.write(output_array, 1)
        dst.set_band_description(1, "ASTERRA nDSM")
    return {
        "npy_path": str(output_npy),
        "tif_path": str(output_tif),
        "width": width,
        "height": height,
        "crs": None if profile.get("crs") is None else str(profile["crs"]),
        "transform": profile.get("transform"),
        "stats": _stats(final_prediction),
        "valid_mask": source_valid,
        "device": device,
        "model": "ASTERRA Stage 5 / Depth Anything V2 Large",
    }


def main():
    result = run_inference(INPUT_PATH, OUTPUT_NPY, OUTPUT_TIF)
    print("ASTERRA tiled inference complete")
    print(f"nDSM NPY: {result['npy_path']}")
    print(f"nDSM GeoTIFF: {result['tif_path']}")
    print(f"Shape: {result['height']} x {result['width']}")
    print(f"Stats: {result['stats']}")


if __name__ == "__main__":
    main()
