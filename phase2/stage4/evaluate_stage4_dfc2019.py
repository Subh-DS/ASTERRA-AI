from __future__ import annotations

"""
ASTERRA AI — STAGE 4 DFC2019 CHECKPOINT DIAGNOSTIC

Read-only diagnostic:
    Stage-4 best checkpoint -> DFC2019 validation sample

This script does NOT train, optimize, resume, or modify any checkpoint.
It reports raw model output statistics and DFC2019 validation metrics.
"""

import argparse
import csv
import gc
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:
    raise RuntimeError("rasterio is required: pip install rasterio") from exc


# ============================================================================
# VERIFIED ASTERRA PATHS
# ============================================================================

ROOT = Path(r"D:\Asterra AI")
DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"

STAGE4_BEST = (
    ROOT / "models" / "asterra_stage4" / "stage4_best.pth"
)

# We search these locations rather than assuming a nonexistent MANIFEST name.
DFC_ROOT = ROOT / "datasets" / "DFC2019"

MANIFEST_CANDIDATES = [
    DFC_ROOT / "manifests" / "dfc2019_val_crops.csv",
    DFC_ROOT / "manifests" / "val_crops.csv",
    DFC_ROOT / "manifests" / "validation.csv",
    DFC_ROOT / "val_crops.csv",
    DFC_ROOT / "validation.csv",
]

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]
MODEL_SIZE = 518
PATCH_SIZE = 512


# ============================================================================
# IMPORT DEPTH ANYTHING V2
# ============================================================================

if not DAV2_ROOT.exists():
    raise FileNotFoundError(
        f"Depth-Anything-V2 repository not found:\n{DAV2_ROOT}"
    )

if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================================
# HELPERS
# ============================================================================

def find_manifest() -> Path:
    for path in MANIFEST_CANDIDATES:
        if path.exists():
            return path

    # Controlled recursive search only under DFC2019.
    if DFC_ROOT.exists():
        matches = sorted(
            p for p in DFC_ROOT.rglob("*.csv")
            if any(k in p.name.lower() for k in ("val", "valid"))
        )
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not locate a DFC2019 validation CSV under:\n"
        f"{DFC_ROOT}\n\n"
        "Run this command in PowerShell to locate it:\n"
        f'Get-ChildItem "{DFC_ROOT}" -Recurse -Filter "*.csv" | '
        "Select-Object FullName"
    )


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic.")
    return torch.device("cuda")


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"DFC2019 validation manifest is empty: {path}")

    return rows


def resolve_path(value: str) -> Path:
    p = Path(value)

    if p.exists():
        return p

    # Common case: manifest stores paths relative to DFC2019 root or ASTERRA.
    candidates = [
        DFC_ROOT / value,
        ROOT / value,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return p


def first_value(row: Dict[str, str], names: List[str]) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if value:
            return value
    return None


def create_model() -> nn.Module:
    if not STAGE4_BEST.exists():
        raise FileNotFoundError(
            f"Stage-4 checkpoint not found:\n{STAGE4_BEST}"
        )

    print("[MODEL] Building DepthAnythingV2...")
    print(f"        encoder      = {ENCODER}")
    print(f"        features     = {FEATURES}")
    print(f"        out_channels = {OUT_CHANNELS}")

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    payload = torch.load(
        STAGE4_BEST,
        map_location="cpu",
        weights_only=False,
    )

    state = (
        payload.get("model_state_dict", payload)
        if isinstance(payload, dict)
        else payload
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    print(f"[LOAD] Missing keys:    {len(missing)}")
    print(f"[LOAD] Unexpected keys: {len(unexpected)}")

    if missing:
        raise RuntimeError(
            "Stage-4 checkpoint has missing model keys:\n"
            + "\n".join(map(str, missing[:20]))
        )

    if isinstance(payload, dict):
        print(f"[META] stage          = {payload.get('stage')}")
        print(f"[META] epoch          = {payload.get('epoch')}")
        print(f"[META] best_val_mae   = {payload.get('best_val_mae')}")
        print(f"[META] dataset        = {payload.get('dataset')}")

    print("[OK] Stage-4 checkpoint loaded successfully.")
    return model


def read_raster_patch(
    path: Path,
    row: int,
    col: int,
    band: int = 1,
) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(
            band,
            window=Window(col, row, PATCH_SIZE, PATCH_SIZE),
        )

    return np.asarray(arr, dtype=np.float32)


def read_rgb_patch(
    path: Path,
    row: int,
    col: int,
) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(
            window=Window(col, row, PATCH_SIZE, PATCH_SIZE)
        )

    if arr.ndim != 3 or arr.shape[0] < 3:
        raise RuntimeError(
            f"RGB requires at least 3 bands: {path}; shape={arr.shape}"
        )

    arr = arr[:3].astype(np.float32, copy=False)
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    # Same normalization convention used by the ASTERRA Stage-5 pipeline.
    if float(np.nanmax(arr)) > 1.5:
        arr /= 255.0

    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def build_sample(row_data: Dict[str, str]) -> Dict[str, Any]:
    rgb_value = first_value(
        row_data,
        ["rgb_path", "image_path", "input_path", "rgb", "image"],
    )
    target_value = first_value(
        row_data,
        ["target_path", "agl_path", "dsm_path", "height_path", "target"],
    )
    mask_value = first_value(
        row_data,
        ["mask_path", "valid_mask_path", "mask"],
    )

    if not rgb_value:
        raise RuntimeError(
            "Manifest does not contain a recognizable RGB/image path column. "
            f"Columns: {list(row_data.keys())}"
        )

    if not target_value:
        raise RuntimeError(
            "Manifest does not contain a recognizable target/AGL path column. "
            f"Columns: {list(row_data.keys())}"
        )

    rgb_path = resolve_path(rgb_value)
    target_path = resolve_path(target_value)

    row = int(float(row_data.get("row", row_data.get("y", 0))))
    col = int(float(row_data.get("col", row_data.get("x", 0))))

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")

    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    rgb = read_rgb_patch(rgb_path, row, col)

    target = read_raster_patch(
        target_path,
        row,
        col,
        band=1,
    )

    if target.shape != (PATCH_SIZE, PATCH_SIZE):
        raise RuntimeError(
            f"Target patch shape is {target.shape}, expected "
            f"{PATCH_SIZE}x{PATCH_SIZE}"
        )

    valid = np.isfinite(target)

    if mask_value:
        mask_path = resolve_path(mask_value)
        if mask_path.exists():
            external_mask = read_raster_patch(
                mask_path,
                row,
                col,
                band=1,
            )
            valid &= np.isfinite(external_mask) & (external_mask > 0)

    target = np.nan_to_num(
        target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    image = torch.from_numpy(
        np.transpose(rgb, (2, 0, 1)).copy()
    ).float().unsqueeze(0)

    target_t = torch.from_numpy(
        target.copy()
    ).float().unsqueeze(0).unsqueeze(0)

    mask_t = torch.from_numpy(
        valid.astype(np.float32).copy()
    ).float().unsqueeze(0).unsqueeze(0)

    return {
        "id": row_data.get(
            "crop_id",
            row_data.get(
                "id",
                f"{rgb_path.stem}_{row}_{col}",
            ),
        ),
        "rgb_path": rgb_path,
        "target_path": target_path,
        "row": row,
        "col": col,
        "image": image,
        "target": target_t,
        "mask": mask_t,
    }


def model_forward(
    model: nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    out = model(image)

    if isinstance(out, dict):
        for key in ("metric_depth", "depth", "out", "pred"):
            if key in out:
                out = out[key]
                break
        else:
            raise RuntimeError(
                f"Unknown model output dictionary keys: {list(out.keys())}"
            )

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not torch.is_tensor(out):
        raise RuntimeError(f"Model output is not a tensor: {type(out)}")

    if out.ndim == 3:
        out = out.unsqueeze(1)

    if out.ndim != 4:
        raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")

    if out.shape[1] != 1:
        out = out[:, :1]

    return out


def resize_target_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    weighted = target.float() * mask.float()

    weighted_resized = F.interpolate(
        weighted,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    weight_resized = F.interpolate(
        mask.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    target_resized = (
        weighted_resized
        / weight_resized.clamp_min(1e-6)
    )

    mask_resized = F.interpolate(
        mask.float(),
        size=size,
        mode="nearest",
    )

    target_resized = torch.where(
        mask_resized > 0.5,
        target_resized,
        torch.zeros_like(target_resized),
    )

    return target_resized, mask_resized


def stats(x: torch.Tensor) -> Dict[str, float]:
    v = x.detach().float()
    v = v[torch.isfinite(v)]

    if v.numel() == 0:
        return {}

    eps = 1e-8

    return {
        "min": float(v.min().cpu()),
        "max": float(v.max().cpu()),
        "mean": float(v.mean().cpu()),
        "std": float(v.std(unbiased=False).cpu()),
        "negative": float((v < -eps).float().mean().cpu() * 100),
        "zero": float((v.abs() <= eps).float().mean().cpu() * 100),
        "positive": float((v > eps).float().mean().cpu() * 100),
    }


def evaluate_sample(
    model: nn.Module,
    sample: Dict[str, Any],
    device: torch.device,
) -> None:
    image = sample["image"].to(device)
    target = sample["target"].to(device)
    mask = sample["mask"].to(device)

    # This matches the Stage-4/5 model-input resizing convention.
    image_model = F.interpolate(
        image,
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    with torch.inference_mode():
        pred = model_forward(model, image_model).float()

        target_r, mask_r = resize_target_mask(
            target,
            mask,
            pred.shape[-2:],
        )

    p = stats(pred)

    valid = (
        (mask_r > 0.5)
        & torch.isfinite(pred)
        & torch.isfinite(target_r)
    )

    if not valid.any():
        raise RuntimeError("No valid DFC2019 pixels in diagnostic sample.")

    error = (pred - target_r)[valid]

    mae = float(error.abs().mean().cpu())
    rmse = float(torch.sqrt((error * error).mean()).cpu())

    t = stats(target_r[valid])

    print()
    print("-" * 78)
    print(f"[SAMPLE] {sample['id']}")
    print(f"[RGB]    {sample['rgb_path']}")
    print(f"[TARGET] {sample['target_path']}")
    print(f"[CROP]   row={sample['row']} col={sample['col']}")
    print(f"[INPUT]  {tuple(image.shape)} -> {tuple(image_model.shape)}")
    print(f"[PRED]   {tuple(pred.shape)}")

    print()
    print("[RAW PRED]")
    print(f"  min        = {p['min']:.10f}")
    print(f"  max        = {p['max']:.10f}")
    print(f"  mean       = {p['mean']:.10f}")
    print(f"  std        = {p['std']:.10f}")
    print(f"  negative   = {p['negative']:.4f}%")
    print(f"  zero       = {p['zero']:.4f}%")
    print(f"  positive   = {p['positive']:.4f}%")

    print()
    print("[TARGET]")
    print(f"  min        = {t['min']:.10f}")
    print(f"  max        = {t['max']:.10f}")
    print(f"  mean       = {t['mean']:.10f}")
    print(f"  std        = {t['std']:.10f}")

    print()
    print("[METRICS]")
    print(f"  MAE        = {mae:.10f} m")
    print(f"  RMSE       = {rmse:.10f} m")
    print(f"  valid      = {int(valid.sum().item()):,}")

    if p["zero"] > 99.99:
        print()
        print("[DIAGNOSIS] Prediction is effectively all zero.")

    elif p["positive"] > 0.1 and p["std"] > 1e-4:
        print()
        print("[DIAGNOSIS] Non-zero spatial prediction detected.")

    else:
        print()
        print("[DIAGNOSIS] Output is non-zero but has very limited variation.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Stage-4 DFC2019 checkpoint diagnostic."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of validation samples to inspect.",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    print("=" * 78)
    print("ASTERRA AI — STAGE 4 DFC2019 CHECKPOINT DIAGNOSTIC")
    print("=" * 78)
    print(f"Device:      {get_device()}")
    print(f"Checkpoint:  {STAGE4_BEST}")

    manifest = find_manifest()
    print(f"Manifest:    {manifest}")
    print("Mode:        READ-ONLY / NO TRAINING")

    rows = load_manifest(manifest)
    print(f"[DATA] Validation rows: {len(rows)}")

    device = get_device()

    model = create_model()
    model.to(device)
    model.eval()

    checked = 0

    for row in rows:
        if checked >= args.limit:
            break

        try:
            sample = build_sample(row)
            evaluate_sample(model, sample, device)
            checked += 1
        except Exception as exc:
            print(
                f"[WARN] Failed validation row "
                f"{row.get('crop_id', row.get('id', 'UNKNOWN'))}: "
                f"{type(exc).__name__}: {exc}"
            )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print(f"DIAGNOSTIC COMPLETE — samples checked: {checked}")
    print("=" * 78)

    if checked == 0:
        raise RuntimeError(
            "No DFC2019 validation sample could be evaluated."
        )


if __name__ == "__main__":
    main()
