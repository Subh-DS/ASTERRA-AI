#!/usr/bin/env python
"""
ASTERRA AI - S-EO Stage 2 full-image inference
===============================================

Purpose
-------
Run the trained S-EO Stage 2 absolute-elevation model on a full RGB image
using tiled inference and overlap blending.

Model checkpoint:
    D:\Asterra AI\models\asterra_seo_v3_1_transfer\seo_best.pth

Supported input:
    .tif / .tiff
    .png
    .jpg / .jpeg

Geospatial behavior
-------------------
1. GeoTIFF input:
   - preserves width/height
   - preserves CRS
   - preserves affine transform
   - writes a georeferenced float32 GeoTIFF

2. PNG/JPEG input:
   - runs neural inference correctly
   - writes a float32 GeoTIFF only if --crs and --transform are supplied
   - otherwise writes .npy and a non-georeferenced GeoTIFF is NOT fabricated

IMPORTANT
---------
An RPC JSON by itself is not an affine GeoTIFF transform. This script does
not silently invent georeferencing from RPC metadata.

Model semantics:
    S-EO DSM-Max / absolute elevation in metres.

This is NOT an nDSM model.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(r"D:\Asterra AI")
DEFAULT_CHECKPOINT = (
    ROOT / "models" / "asterra_seo_v3_1_transfer" / "seo_best.pth"
)
DEFAULT_DEPTH_ROOT = ROOT / "external" / "Depth-Anything-V2"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "seo_stage2_inference"

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

MODEL_SIZE = 518
TILE_SIZE = 512
OVERLAP = 64
EPS = 1e-8


def setup_depth_import(depth_root: Path):
    depth_root = depth_root.resolve()
    if not depth_root.exists():
        raise FileNotFoundError(
            f"Depth Anything V2 source not found:\n{depth_root}"
        )
    if str(depth_root) not in sys.path:
        sys.path.insert(0, str(depth_root))


def build_model(depth_root: Path, device):
    setup_depth_import(depth_root)
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    oc = model.depth_head.scratch.output_conv2

    if not isinstance(oc[2], nn.Conv2d):
        raise RuntimeError(
            "Unexpected DPT head: output_conv2[2] is not Conv2d."
        )

    # Stage-2 is unrestricted absolute elevation.
    if len(oc) > 3:
        oc[3] = nn.Identity()

    return model.to(device)


def load_checkpoint(model, checkpoint_path: Path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint.")

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get(
            "state_dict",
            checkpoint.get("model", checkpoint),
        ),
    )

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    print(
        f"checkpoint tensors={len(cleaned)} "
        f"missing={len(missing)} "
        f"unexpected={len(unexpected)}"
    )

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch."
        )

    return checkpoint


def load_image(path: Path):
    """
    Return:
        rgb: HWC float32 [0,1]
        profile: GeoTIFF profile or None
        transform: affine transform or None
        crs: CRS or None
        nodata_mask: boolean HxW, True for usable pixels
    """
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read()
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        if arr.ndim != 3:
            raise RuntimeError(
                f"Unexpected GeoTIFF shape: {arr.shape}"
            )

        # First three bands are RGB.
        if arr.shape[0] < 3:
            raise RuntimeError(
                f"GeoTIFF requires at least 3 bands, got {arr.shape[0]}"
            )

        rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.float32)

        valid = np.all(np.isfinite(rgb), axis=2)

        if nodata is not None:
            valid &= np.all(rgb != nodata, axis=2)

    else:
        with Image.open(path) as im:
            rgb = np.asarray(im)

        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)

        if rgb.ndim != 3:
            raise RuntimeError(
                f"Unexpected image shape: {rgb.shape}"
            )

        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]

        if rgb.shape[-1] != 3:
            raise RuntimeError(
                f"Expected RGB image, got {rgb.shape}"
            )

        rgb = rgb.astype(np.float32)
        valid = np.all(np.isfinite(rgb), axis=2)

        profile = None
        transform = None
        crs = None

    if np.nanmax(rgb) > 1.5:
        rgb /= 255.0

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb, profile, transform, crs, valid


def tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]

    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))

    last = length - tile
    if starts[-1] != last:
        starts.append(last)

    return starts


def make_blend_window(h, w):
    """
    Hanning overlap window.

    A small floor avoids zero-weight border pixels.
    """
    wy = np.hanning(h)
    wx = np.hanning(w)

    if np.all(wy == 0):
        wy = np.ones(h)

    if np.all(wx == 0):
        wx = np.ones(w)

    window = np.outer(wy, wx).astype(np.float32)

    # Prevent zero-weight pixels.
    window = np.maximum(window, 1e-3)

    return window


@torch.inference_mode()
def predict_tile(model, tile, device):
    x = (
        torch.from_numpy(tile)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    if x.shape[-2:] != (MODEL_SIZE, MODEL_SIZE):
        x = F.interpolate(
            x,
            size=(MODEL_SIZE, MODEL_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    x = x.to(device, non_blocking=True)

    y = model(x)

    if isinstance(y, (tuple, list)):
        y = y[0]

    if y.ndim == 3:
        y = y[:, None]

    if y.ndim != 4:
        raise RuntimeError(
            f"Unexpected model output: {tuple(y.shape)}"
        )

    # Return prediction at the original tile resolution.
    y = F.interpolate(
        y[:, :1],
        size=tile.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    return y[0, 0].float().cpu().numpy()


def tiled_inference(model, rgb, valid, device):
    h, w = rgb.shape[:2]

    ys = tile_starts(h, TILE_SIZE, OVERLAP)
    xs = tile_starts(w, TILE_SIZE, OVERLAP)

    stride = TILE_SIZE - OVERLAP

    print(
        f"Image size: {w} x {h}"
    )
    print(
        f"Tile size: {TILE_SIZE}, overlap: {OVERLAP}, "
        f"stride: {stride}"
    )
    print(
        f"Tiles: {len(ys) * len(xs)}"
    )

    accum = np.zeros((h, w), dtype=np.float64)
    weights = np.zeros((h, w), dtype=np.float64)

    total = len(ys) * len(xs)
    counter = 0

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + TILE_SIZE, h)
            x1 = min(x0 + TILE_SIZE, w)

            tile = rgb[y0:y1, x0:x1]
            tile_valid = valid[y0:y1, x0:x1]

            th, tw = tile.shape[:2]

            pred = predict_tile(
                model,
                tile,
                device,
            )

            window = make_blend_window(th, tw)

            # Only blend valid input pixels.
            blend = window * tile_valid.astype(np.float32)

            finite = np.isfinite(pred)
            blend *= finite.astype(np.float32)

            accum[y0:y1, x0:x1] += (
                pred.astype(np.float64) * blend
            )

            weights[y0:y1, x0:x1] += blend

            counter += 1
            print(
                f"  tile {counter:03d}/{total:03d} "
                f"row={y0} col={x0} "
                f"pred_min={np.nanmin(pred):.3f} "
                f"pred_max={np.nanmax(pred):.3f} "
                f"pred_mean={np.nanmean(pred):.3f}"
            )

    output = np.full(
        (h, w),
        np.nan,
        dtype=np.float32,
    )

    good = weights > EPS

    output[good] = (
        accum[good] / weights[good]
    ).astype(np.float32)

    output[~valid] = np.nan

    return output


def save_geotiff(
    output_path,
    prediction,
    profile,
    transform,
    crs,
):
    import rasterio

    if profile is None:
        raise RuntimeError(
            "Input is not a georeferenced GeoTIFF. "
            "Cannot create a georeferenced output without CRS/transform."
        )

    profile = profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=prediction.shape[0],
        width=prediction.shape[1],
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
        tiled=True,
    )

    if transform is not None:
        profile["transform"] = transform

    if crs is not None:
        profile["crs"] = crs

    data = np.nan_to_num(
        prediction,
        nan=-9999.0,
        posinf=-9999.0,
        neginf=-9999.0,
    ).astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)

    print(f"\nGeoTIFF saved: {output_path}")


def save_npy(output_path, prediction):
    np.save(output_path, prediction.astype(np.float32))
    print(f"NPY saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ASTERRA S-EO Stage 2 tiled full-image inference"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input RGB GeoTIFF/PNG/JPEG.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--depth-root",
        type=Path,
        default=DEFAULT_DEPTH_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=TILE_SIZE,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{args.input}"
        )

    if args.overlap >= args.tile_size:
        raise ValueError(
            "overlap must be smaller than tile-size."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print("=" * 78)
    print("ASTERRA AI — S-EO STAGE 2 FULL IMAGE INFERENCE")
    print("=" * 78)
    print(f"Device     : {device}")

    if device.type == "cuda":
        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"CUDA       : "
            f"{torch.version.cuda}"
        )

    print(f"Input      : {args.input}")
    print(f"Checkpoint : {args.checkpoint}")
    print(
        "Semantics  : S-EO DSM-Max / absolute elevation (m)"
    )

    rgb, profile, transform, crs, valid = load_image(
        args.input
    )

    print(
        f"\nInput shape: "
        f"{rgb.shape[1]} x {rgb.shape[0]} x {rgb.shape[2]}"
    )
    print(
        f"Input valid pixels: "
        f"{int(valid.sum()):,} / {valid.size:,}"
    )

    if crs is not None:
        print(f"CRS        : {crs}")
    else:
        print("CRS        : NONE")

    if transform is not None:
        print(f"Transform  : {transform}")
    else:
        print("Transform  : NONE")

    model = build_model(
        args.depth_root,
        device,
    )

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    prediction = tiled_inference(
        model,
        rgb,
        valid,
        device,
    )

    finite = np.isfinite(prediction)

    print("\n" + "=" * 78)
    print("INFERENCE STATISTICS")
    print("=" * 78)

    if finite.any():
        print(
            f"Prediction min  : {np.nanmin(prediction):.6f} m"
        )
        print(
            f"Prediction max  : {np.nanmax(prediction):.6f} m"
        )
        print(
            f"Prediction mean : {np.nanmean(prediction):.6f} m"
        )
        print(
            f"Prediction std  : {np.nanstd(prediction):.6f} m"
        )
        print(
            f"Positive pixels : "
            f"{100*np.mean(prediction[finite] > 0):.3f}%"
        )
    else:
        raise RuntimeError(
            "Inference produced no finite predictions."
        )

    stem = args.input.stem
    geotiff_path = (
        args.output_dir
        / f"{stem}_asterra_seo_stage2_DSM.tif"
    )
    npy_path = (
        args.output_dir
        / f"{stem}_asterra_seo_stage2_DSM.npy"
    )

    # Always save raw numerical prediction.
    save_npy(
        npy_path,
        prediction,
    )

    # Save a georeferenced GeoTIFF only when the input itself supplies
    # trustworthy geospatial metadata.
    if profile is not None and crs is not None and transform is not None:
        save_geotiff(
            geotiff_path,
            prediction,
            profile,
            transform,
            crs,
        )
    else:
        print(
            "\n[INFO] Input does not contain complete GeoTIFF "
            "georeferencing. GeoTIFF was not fabricated."
        )

    print("\n" + "=" * 78)
    print("INFERENCE COMPLETE")
    print("=" * 78)
    print(f"NPY: {npy_path}")

    if (
        profile is not None
        and crs is not None
        and transform is not None
    ):
        print(f"GeoTIFF: {geotiff_path}")


if __name__ == "__main__":
    main()
