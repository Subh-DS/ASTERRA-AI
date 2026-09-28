"""
ASTERRA AI — STAGE 5
Urban3D Fine-Tuning
===================

Dataset:
    USSOCOM / SpaceNet Urban3D Challenge

Target:
    nDSM = DSM - DTM

Training:
    Stage-4 best checkpoint
        ↓
    Urban3D RGB + nDSM
        ↓
    ASTERRA Stage-5 Urban3D model

IMPORTANT:
    - No RPC processing
    - No synthetic targets
    - No DEM regional calibration
    - No LoRA / PEFT
    - 100% full-parameter fine-tuning
    - RGB/DSM/DTM are spatially aligned
    - Negative nDSM values are clipped to 0 for training
    - Raw nDSM remains represented in manifest statistics
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
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
    raise RuntimeError(
        "rasterio is required. Install with: pip install rasterio"
    ) from exc


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"

if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ----------------------------------------------------------------------------
# Stage-4 initialization checkpoint
# ----------------------------------------------------------------------------

STAGE4_BEST = (
    ROOT
    / "models"
    / "asterra_stage4"
    / "stage4_best.pth"
)


# ----------------------------------------------------------------------------
# Urban3D manifests
# ----------------------------------------------------------------------------

URBAN3D_ROOT = ROOT / "datasets" / "Urban3D"

TRAIN_MANIFEST = (
    URBAN3D_ROOT
    / "manifests"
    / "urban3d_train_crops.csv"
)

VAL_MANIFEST = (
    URBAN3D_ROOT
    / "manifests"
    / "urban3d_val_crops.csv"
)


# ----------------------------------------------------------------------------
# Stage-5 output
# ----------------------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST = OUTPUT_DIR / "stage5_urban3d_latest.pth"
BEST = OUTPUT_DIR / "stage5_urban3d_best.pth"
EMERGENCY = OUTPUT_DIR / "stage5_urban3d_emergency.pth"

HISTORY = OUTPUT_DIR / "stage5_urban3d_training_history.json"
CONFIG_FILE = OUTPUT_DIR / "stage5_urban3d_config.json"


# ============================================================================
# MODEL / TRAINING CONFIG
# ============================================================================

ENCODER = "vitl"

FEATURES = 256

OUT_CHANNELS = [
    256,
    512,
    1024,
    1024,
]

# Depth Anything V2 Large configuration used in Stage-4.
MODEL_SIZE = 518

# Urban3D persistent crop size.
PATCH_SIZE = 512

BATCH_SIZE = 1

# Effective batch = 1 x 4 = 4.
GRAD_ACCUMULATION = 4

EPOCHS = 3

LEARNING_RATE = 1e-6

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0

SEED = 42

PRINT_EVERY = 10

# Urban3D manifests already contain deterministic train/validation splits.
# Therefore there is no random tile split here.


# ============================================================================
# TARGET CONFIG
# ============================================================================

# Urban3D target:
#
#       nDSM = DSM - DTM
#
# Negative nDSM values are clipped to zero for training.
CLIP_NEGATIVE_NDSM = True

# Minimum valid DSM + DTM coverage required for a crop.
MIN_VALID_FRACTION = 0.70

# Expected crop size from manifest.
EXPECTED_CROP_SIZE = 512


# ============================================================================
# UTILITY
# ============================================================================

def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for ASTERRA Stage-5 Urban3D training."
        )

    return torch.device("cuda")


def atomic_json_write(
    path: Path,
    payload: Any,
) -> None:

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def atomic_torch_save(
    payload: Dict[str, Any],
    path: Path,
) -> None:

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        tmp,
    )

    tmp.replace(path)


def print_header() -> None:

    print("=" * 80)
    print("ASTERRA AI — STAGE 5")
    print("URBAN3D FULL-PARAMETER FINE-TUNING")
    print("=" * 80)

    print(f"Root:              {ROOT}")
    print(f"Stage-4 checkpoint:{STAGE4_BEST}")
    print(f"Train manifest:    {TRAIN_MANIFEST}")
    print(f"Val manifest:      {VAL_MANIFEST}")
    print(f"Output:            {OUTPUT_DIR}")

    print()

    print(f"Encoder:           {ENCODER}")
    print(f"Features:          {FEATURES}")
    print(f"Out channels:      {OUT_CHANNELS}")
    print(f"Model input:       {MODEL_SIZE}x{MODEL_SIZE}")
    print(f"Dataset crop:      {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Grad accumulation: {GRAD_ACCUMULATION}")
    print(f"Effective batch:   {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Weight decay:      {WEIGHT_DECAY}")
    print(f"Gradient clip:     {GRAD_CLIP}")
    print(f"Epochs:            {EPOCHS}")
    print()

    print("Target:")
    print("    nDSM = DSM - DTM")

    if CLIP_NEGATIVE_NDSM:
        print("    negative nDSM -> 0 for training")

    print()

    if torch.cuda.is_available():

        props = torch.cuda.get_device_properties(0)

        print(
            f"GPU:               "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM:              "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

        print(
            f"PyTorch:           "
            f"{torch.__version__}"
        )

        print(
            f"CUDA:              "
            f"{torch.version.cuda}"
        )

    else:

        print("GPU: CUDA NOT AVAILABLE")


# ============================================================================
# MODEL
# ============================================================================

def create_model(
    checkpoint: Path,
) -> nn.Module:

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Stage-4 checkpoint not found:\n{checkpoint}"
        )

    print()
    print("[MODEL] Creating Depth Anything V2 Large...")

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    print(
        f"[MODEL] Loading checkpoint:\n"
        f"        {checkpoint}"
    )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(payload, dict):

        state = payload.get(
            "model_state_dict",
            payload,
        )

    else:

        state = payload

    if not isinstance(state, dict):

        raise RuntimeError(
            "Stage-4 checkpoint does not contain "
            "a valid model state dictionary."
        )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:

        raise RuntimeError(
            "Stage-4 checkpoint is incompatible "
            "with the Stage-5 architecture.\n"
            f"Missing keys: {missing[:20]}"
        )

    if unexpected:

        print(
            f"[WARN] Unexpected checkpoint keys: "
            f"{len(unexpected)}"
        )

    print(
        "[OK] Stage-4 checkpoint loaded."
    )

    if isinstance(payload, dict):

        print(
            f"[INFO] Source stage: "
            f"{payload.get('stage')}"
        )

        print(
            f"[INFO] Source epoch: "
            f"{payload.get('epoch')}"
        )

        print(
            f"[INFO] Source best MAE: "
            f"{payload.get('best_val_mae')}"
        )

    return model


def enable_gradient_checkpointing(
    model: nn.Module,
) -> None:

    for obj in (
        model,
        getattr(
            model,
            "pretrained",
            None,
        ),
    ):

        if obj is None:
            continue

        fn = getattr(
            obj,
            "gradient_checkpointing_enable",
            None,
        )

        if callable(fn):

            try:

                fn()

                print(
                    "[OK] Gradient checkpointing enabled."
                )

                return

            except Exception as exc:

                print(
                    "[INFO] Gradient checkpointing "
                    f"unavailable: {exc}"
                )

    print(
        "[INFO] Model did not expose "
        "gradient_checkpointing_enable()."
    )


def make_trainable(
    model: nn.Module,
) -> None:

    total = 0
    trainable = 0

    for p in model.parameters():

        p.requires_grad = True

        n = p.numel()

        total += n
        trainable += n

    percentage = (
        100.0 * trainable / max(1, total)
    )

    print(
        f"[OK] Trainable parameters: "
        f"{trainable:,}/{total:,} "
        f"({percentage:.2f}%)"
    )

    if percentage < 99.99:

        raise RuntimeError(
            "Full-parameter fine-tuning requirement "
            "was not satisfied."
        )


def model_forward(
    model: nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:

    out = model(image)

    if isinstance(out, dict):

        for key in (
            "metric_depth",
            "depth",
            "out",
            "pred",
        ):

            if key in out:

                out = out[key]

                break

        else:

            raise RuntimeError(
                "Unknown model output dictionary keys: "
                f"{list(out.keys())}"
            )

    if isinstance(out, (tuple, list)):

        out = out[0]

    if not torch.is_tensor(out):

        raise RuntimeError(
            "Model output is not a tensor: "
            f"{type(out)}"
        )

    if out.ndim == 3:

        out = out.unsqueeze(1)

    if out.ndim != 4:

        raise RuntimeError(
            f"Unexpected output shape: "
            f"{tuple(out.shape)}"
        )

    if out.shape[1] != 1:

        out = out[:, :1]

    return out


# ============================================================================
# MANIFEST
# ============================================================================

def load_csv_manifest(
    path: Path,
) -> List[Dict[str, str]]:

    if not path.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{path}"
        )

    import csv

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:

        raise RuntimeError(
            f"Manifest is empty:\n{path}"
        )

    return rows


def validate_manifest_paths(
    rows: List[Dict[str, str]],
    name: str,
) -> None:

    required = [
        "rgb_path",
        "dsm_path",
        "dtm_path",
        "row",
        "col",
        "crop_size",
    ]

    for field in required:

        if field not in rows[0]:

            raise RuntimeError(
                f"{name} manifest missing field: "
                f"{field}"
            )

    missing = 0

    for row in rows:

        for field in (
            "rgb_path",
            "dsm_path",
            "dtm_path",
        ):

            p = Path(row[field])

            if not p.exists():

                missing += 1

                if missing <= 10:

                    print(
                        f"[MISSING] {p}"
                    )

    if missing:

        raise RuntimeError(
            f"{name} manifest contains "
            f"{missing} missing file references."
        )


# ============================================================================
# TIFF READING
# ============================================================================

def read_rgb_patch(
    path: Path,
    row: int,
    col: int,
) -> np.ndarray:

    with rasterio.open(path) as src:

        window = Window(
            col,
            row,
            PATCH_SIZE,
            PATCH_SIZE,
        )

        arr = src.read(
            window=window,
        )

    if arr.ndim != 3:

        raise RuntimeError(
            f"RGB must be 3D "
            f"(bands,H,W); got {arr.shape}: "
            f"{path}"
        )

    if arr.shape[0] < 3:

        raise RuntimeError(
            f"RGB has fewer than 3 bands: "
            f"{arr.shape}: {path}"
        )

    arr = arr[:3].astype(
        np.float32,
        copy=False,
    )

    arr = np.transpose(
        arr,
        (1, 2, 0),
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    # Urban3D RGB is uint8.
    if float(np.nanmax(arr)) > 1.5:

        arr /= 255.0

    arr = np.clip(
        arr,
        0.0,
        1.0,
    )

    return np.ascontiguousarray(arr)


def read_height_patch(
    path: Path,
    row: int,
    col: int,
) -> Tuple[np.ndarray, Optional[float]]:

    with rasterio.open(path) as src:

        window = Window(
            col,
            row,
            PATCH_SIZE,
            PATCH_SIZE,
        )

        arr = src.read(
            1,
            window=window,
        ).astype(
            np.float32,
            copy=False,
        )

        nodata = src.nodata

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    if nodata is not None:

        arr[arr == nodata] = np.nan

    arr[~np.isfinite(arr)] = np.nan

    return (
        np.ascontiguousarray(arr),
        nodata,
    )


# ============================================================================
# URBAN3D SAMPLE
# ============================================================================

def get_sample(
    item: Dict[str, str],
) -> Optional[Dict[str, Any]]:

    try:

        rgb_path = Path(
            item["rgb_path"]
        )

        dsm_path = Path(
            item["dsm_path"]
        )

        dtm_path = Path(
            item["dtm_path"]
        )

        row = int(
            float(item["row"])
        )

        col = int(
            float(item["col"])
        )

        crop_size = int(
            float(item["crop_size"])
        )

        if crop_size != EXPECTED_CROP_SIZE:

            raise RuntimeError(
                f"Unexpected crop size: "
                f"{crop_size}"
            )

        # ------------------------------------------------------------
        # RGB
        # ------------------------------------------------------------

        rgb = read_rgb_patch(
            rgb_path,
            row,
            col,
        )

        # ------------------------------------------------------------
        # DSM
        # ------------------------------------------------------------

        dsm, dsm_nodata = read_height_patch(
            dsm_path,
            row,
            col,
        )

        # ------------------------------------------------------------
        # DTM
        # ------------------------------------------------------------

        dtm, dtm_nodata = read_height_patch(
            dtm_path,
            row,
            col,
        )

        if dsm.shape != (
            PATCH_SIZE,
            PATCH_SIZE,
        ):

            raise RuntimeError(
                f"DSM patch shape is "
                f"{dsm.shape}"
            )

        if dtm.shape != (
            PATCH_SIZE,
            PATCH_SIZE,
        ):

            raise RuntimeError(
                f"DTM patch shape is "
                f"{dtm.shape}"
            )

        # ------------------------------------------------------------
        # Validity
        # ------------------------------------------------------------

        valid = (
            np.isfinite(dsm)
            & np.isfinite(dtm)
        )

        valid_fraction = float(
            valid.mean()
        )

        if valid_fraction < MIN_VALID_FRACTION:

            return None

        # ------------------------------------------------------------
        # nDSM = DSM - DTM
        # ------------------------------------------------------------

        ndsm = (
            dsm.astype(np.float32)
            - dtm.astype(np.float32)
        )

        ndsm[~valid] = np.nan

        # ------------------------------------------------------------
        # Negative nDSM
        #
        # Small negative values are present in the
        # Urban3D corpus due to DSM/DTM differences.
        #
        # Training target is constrained to >= 0.
        # ------------------------------------------------------------

        if CLIP_NEGATIVE_NDSM:

            ndsm[
                np.isfinite(ndsm)
                & (ndsm < 0.0)
            ] = 0.0

        # ------------------------------------------------------------
        # Tensor conversion
        # ------------------------------------------------------------

        image = torch.from_numpy(
            np.transpose(
                rgb,
                (2, 0, 1),
            ).copy()
        ).float()

        target = torch.from_numpy(
            np.nan_to_num(
                ndsm,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).copy()
        ).float().unsqueeze(0)

        mask = torch.from_numpy(
            valid.copy()
        ).float().unsqueeze(0)

        # If negative clipping was enabled, target is already
        # non-negative. Invalid pixels are zero but excluded by mask.

        return {
            "id": item.get(
                "crop_id",
                f"{Path(item['rgb_path']).stem}_{row}_{col}",
            ),
            "tile": item.get(
                "tile",
                "",
            ),
            "image": image,
            "target": target,
            "mask": mask,
            "valid_fraction": valid_fraction,
            "raw_ndsm_min": float(
                np.nanmin(ndsm)
            ),
            "raw_ndsm_max": float(
                np.nanmax(ndsm)
            ),
        }

    except Exception as exc:

        print(
            f"[WARN] Sample failed: "
            f"{item.get('crop_id', 'UNKNOWN')} | "
            f"{type(exc).__name__}: {exc}"
        )

        return None


# ============================================================================
# TARGET / MASK RESIZING
# ============================================================================

def resize_target_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:

    if target.ndim == 3:

        target = target.unsqueeze(1)

    if mask.ndim == 3:

        mask = mask.unsqueeze(1)

    target = target.float()
    mask = mask.float()

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT simply bilinear-resize target containing zero-valued
    # invalid pixels. That would contaminate valid boundary pixels.
    #
    # Instead:
    #
    #     weighted_target = target * mask
    #
    # then resize both target weights and mask weights.
    # ------------------------------------------------------------

    weighted = target * mask

    weighted_resized = F.interpolate(
        weighted,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    weight_resized = F.interpolate(
        mask,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    target_resized = (
        weighted_resized
        / weight_resized.clamp_min(1e-6)
    )

    # Nearest mask preserves the valid/invalid decision.
    mask_resized = F.interpolate(
        mask,
        size=size,
        mode="nearest",
    )

    target_resized = torch.where(
        mask_resized > 0.5,
        target_resized,
        torch.zeros_like(
            target_resized
        ),
    )

    return (
        target_resized,
        mask_resized,
    )


# ============================================================================
# LOSS
# ============================================================================

def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    if pred.ndim == 3:

        pred = pred.unsqueeze(1)

    if target.ndim == 3:

        target = target.unsqueeze(1)

    if mask.ndim == 3:

        mask = mask.unsqueeze(1)

    if pred.shape[-2:] != target.shape[-2:]:

        pred = F.interpolate(
            pred,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if mask.shape[-2:] != pred.shape[-2:]:

        mask = F.interpolate(
            mask.float(),
            size=pred.shape[-2:],
            mode="nearest",
        )

    valid = (
        (mask > 0.5)
        & torch.isfinite(pred)
        & torch.isfinite(target)
    )

    if not valid.any():

        return pred.sum() * 0.0

    error = (
        pred.float()
        - target.float()
    )

    return torch.abs(
        error
    )[valid].mean()


# ============================================================================
# AMP
# ============================================================================

def amp_context():

    if not torch.cuda.is_available():

        return torch.autocast(
            device_type="cpu",
            enabled=False,
        )

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    )


def make_scaler():

    try:

        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    except Exception:

        return torch.cuda.amp.GradScaler(
            enabled=True
        )


# ============================================================================
# OPTIMIZER
# ============================================================================

def make_optimizer(
    model: nn.Module,
):

    try:

        from transformers.optimization import (
            Adafactor,
        )

        print(
            "[OK] Optimizer: Adafactor"
        )

        return Adafactor(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    except Exception as exc:

        print(
            "[INFO] Adafactor unavailable: "
            f"{exc}"
        )

        print(
            "[OK] Optimizer: AdamW fallback"
        )

        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    best_mae: float,
    history: List[Dict[str, Any]],
    reason: str,
) -> None:

    payload: Dict[str, Any] = {

        "stage": 5,

        "stage_name":
            "Urban3D",

        "epoch":
            int(epoch),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_val_mae":
            float(best_mae),

        "history":
            history,

        "reason":
            reason,

        "initial_checkpoint":
            str(STAGE4_BEST),

        "dataset":
            "USSOCOM Urban3D",

        "target":
            "nDSM = DSM - DTM",

        "negative_ndsm_policy":
            (
                "clip_to_zero_for_training"
                if CLIP_NEGATIVE_NDSM
                else "preserve"
            ),

        "patch_size":
            PATCH_SIZE,

        "model_size":
            MODEL_SIZE,

        "encoder":
            ENCODER,

        "features":
            FEATURES,

        "out_channels":
            OUT_CHANNELS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "grad_accumulation":
            GRAD_ACCUMULATION,

        "full_parameter_tuning":
            True,

        "seed":
            SEED,
    }

    try:

        payload[
            "scaler_state_dict"
        ] = scaler.state_dict()

    except Exception:

        payload[
            "scaler_state_dict"
        ] = None

    atomic_torch_save(
        payload,
        path,
    )


# ============================================================================
# RESUME
# ============================================================================

def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scaler,
) -> Tuple[int, float, List[Dict[str, Any]]]:

    if not path.exists():

        raise FileNotFoundError(
            f"Resume checkpoint not found:\n{path}"
        )

    print()
    print(
        f"[RESUME] Loading:\n{path}"
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = payload.get(
        "model_state_dict",
        payload,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    if "optimizer_state_dict" in payload:

        optimizer.load_state_dict(
            payload[
                "optimizer_state_dict"
            ]
        )

    scaler_state = payload.get(
        "scaler_state_dict"
    )

    if scaler_state:

        try:

            scaler.load_state_dict(
                scaler_state
            )

        except Exception as exc:

            print(
                "[WARN] Could not restore "
                f"AMP scaler: {exc}"
            )

    epoch = int(
        payload.get(
            "epoch",
            0,
        )
    )

    best_mae = float(
        payload.get(
            "best_val_mae",
            float("inf"),
        )
    )

    history = payload.get(
        "history",
        [],
    )

    if not isinstance(
        history,
        list,
    ):

        history = []

    print(
        f"[OK] Resumed from epoch {epoch}"
    )

    print(
        f"[OK] Previous best MAE: "
        f"{best_mae:.6f}"
    )

    return (
        epoch + 1,
        best_mae,
        history,
    )


# ============================================================================
# PREFLIGHT
# ============================================================================

def inspect_manifest(
    train_items: List[Dict[str, str]],
    val_items: List[Dict[str, str]],
) -> None:

    print()
    print("=" * 80)
    print("STAGE-5 URBAN3D PREFLIGHT")
    print("=" * 80)

    print(
        f"[OK] Train crops: "
        f"{len(train_items)}"
    )

    print(
        f"[OK] Validation crops: "
        f"{len(val_items)}"
    )

    print()

    validate_manifest_paths(
        train_items,
        "TRAIN",
    )

    validate_manifest_paths(
        val_items,
        "VALIDATION",
    )

    print(
        "[OK] All manifest file paths exist."
    )

    print()

    test_items = (
        train_items[:4]
        + val_items[:4]
    )

    success = 0

    for item in test_items:

        sample = get_sample(
            item
        )

        if sample is None:

            continue

        print(
            f"[OK] {sample['id']} | "
            f"valid={sample['valid_fraction']:.4f} | "
            f"nDSM={sample['raw_ndsm_min']:.4f}.."
            f"{sample['raw_ndsm_max']:.4f}"
        )

        print(
            f"     RGB: "
            f"{tuple(sample['image'].shape)}"
        )

        print(
            f"     target: "
            f"{tuple(sample['target'].shape)}"
        )

        print(
            f"     mask: "
            f"{tuple(sample['mask'].shape)}"
        )

        success += 1

    if success == 0:

        raise RuntimeError(
            "Urban3D preflight could not "
            "construct any valid sample."
        )

    print()

    print(
        "[OK] Urban3D sample construction passed."
    )

    print(
        "[OK] Target definition: "
        "nDSM = DSM - DTM"
    )

    print(
        "[OK] Spatial alignment is supplied "
        "by the Urban3D RGB/DSM/DTM tiles."
    )

    print()

    print(
        "[OK] STAGE-5 URBAN3D PREFLIGHT PASSED."
    )


# ============================================================================
# GPU SMOKE TEST
# ============================================================================

def gpu_smoke_test(
    train_items: List[Dict[str, str]],
) -> bool:

    print()
    print("=" * 80)
    print("STAGE-5 URBAN3D GPU SMOKE TEST")
    print("=" * 80)

    device = get_device()

    sample = None

    for item in train_items[:50]:

        sample = get_sample(
            item
        )

        if sample is not None:

            break

    if sample is None:

        print(
            "[ERROR] Could not construct "
            "a valid Urban3D sample."
        )

        return False

    model = None
    optimizer = None
    scaler = None

    try:

        model = create_model(
            STAGE4_BEST
        )

        enable_gradient_checkpointing(
            model
        )

        make_trainable(
            model
        )

        model.to(device)

        optimizer = make_optimizer(
            model
        )

        scaler = make_scaler()

        image = (
            sample["image"]
            .unsqueeze(0)
            .to(
                device,
                non_blocking=True,
            )
        )

        target = (
            sample["target"]
            .unsqueeze(0)
            .to(
                device,
                non_blocking=True,
            )
        )

        mask = (
            sample["mask"]
            .unsqueeze(0)
            .to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with amp_context():

            image_model = F.interpolate(
                image,
                size=(
                    MODEL_SIZE,
                    MODEL_SIZE,
                ),
                mode="bilinear",
                align_corners=False,
            )

            pred = model_forward(
                model,
                image_model,
            )

            target_r, mask_r = (
                resize_target_mask(
                    target,
                    mask,
                    pred.shape[-2:],
                )
            )

            loss = masked_l1(
                pred,
                target_r,
                mask_r,
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite smoke loss: "
                f"{float(loss.detach().cpu())}"
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        print()

        print(
            f"[OK] Sample: "
            f"{sample['id']}"
        )

        print(
            f"[OK] Input: "
            f"{tuple(image.shape)}"
        )

        print(
            f"[OK] Prediction: "
            f"{tuple(pred.shape)}"
        )

        print(
            f"[OK] Loss: "
            f"{float(loss.detach().cpu()):.6f}"
        )

        print(
            f"[OK] Grad norm: "
            f"{float(grad_norm):.6f}"
        )

        print(
            "[OK] Forward pass succeeded."
        )

        print(
            "[OK] Backward pass succeeded."
        )

        print(
            "[OK] Optimizer step succeeded."
        )

        print()
        print(
            "[OK] STAGE-5 URBAN3D GPU "
            "SMOKE TEST PASSED."
        )

        return True

    except torch.cuda.OutOfMemoryError:

        print(
            "[ERROR] CUDA OOM during "
            "Urban3D smoke test."
        )

        return False

    except Exception as exc:

        print(
            "[ERROR] GPU smoke test failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return False

    finally:

        if model is not None:
            del model

        if optimizer is not None:
            del optimizer

        if scaler is not None:
            del scaler

        gc.collect()

        torch.cuda.empty_cache()


# ============================================================================
# TRAIN ONE EPOCH
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    optimizer,
    scaler,
    items: List[Dict[str, str]],
    device: torch.device,
    epoch: int,
    max_steps: int,
) -> Tuple[float, int]:

    order = list(items)

    random.shuffle(order)

    if max_steps > 0:

        order = order[
            :max_steps
        ]

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0

    count = 0

    accumulation = 0

    for step, item in enumerate(
        order,
        1,
    ):

        sample = get_sample(
            item
        )

        if sample is None:

            continue

        image = None
        target = None
        mask = None
        pred = None

        try:

            image = (
                sample["image"]
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target = (
                sample["target"]
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            mask = (
                sample["mask"]
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with amp_context():

                image_model = F.interpolate(
                    image,
                    size=(
                        MODEL_SIZE,
                        MODEL_SIZE,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

                pred = model_forward(
                    model,
                    image_model,
                )

                target_r, mask_r = (
                    resize_target_mask(
                        target,
                        mask,
                        pred.shape[-2:],
                    )
                )

                loss = masked_l1(
                    pred,
                    target_r,
                    mask_r,
                )

            if not torch.isfinite(loss):

                print(
                    f"[WARN] Non-finite loss "
                    f"at {sample['id']}; "
                    "skipping."
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                accumulation = 0

                continue

            scaled_loss = (
                loss
                / GRAD_ACCUMULATION
            )

            scaler.scale(
                scaled_loss
            ).backward()

            accumulation += 1

            should_step = (
                accumulation
                >= GRAD_ACCUMULATION
                or step
                == len(order)
            )

            if should_step:

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                accumulation = 0

            total_loss += float(
                loss.detach().cpu()
            )

            count += 1

            if step % PRINT_EVERY == 0:

                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(order)} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"valid={sample['valid_fraction']:.3f}"
                )

        except torch.cuda.OutOfMemoryError:

            print(
                f"[WARN] CUDA OOM at "
                f"{sample['id']}; "
                "clearing cache and skipping."
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulation = 0

            torch.cuda.empty_cache()

        except Exception as exc:

            print(
                f"[WARN] Training failure at "
                f"{sample['id']} | "
                f"{type(exc).__name__}: {exc}"
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulation = 0

        finally:

            if image is not None:
                del image

            if target is not None:
                del target

            if mask is not None:
                del mask

            if pred is not None:
                del pred

    # ------------------------------------------------------------
    # Flush remaining gradients
    # ------------------------------------------------------------

    if accumulation > 0:

        try:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        except Exception as exc:

            print(
                "[WARN] Final gradient "
                f"flush failed: {exc}"
            )

            optimizer.zero_grad(
                set_to_none=True
            )

    return (
        total_loss / max(
            1,
            count,
        ),
        count,
    )


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    items: List[Dict[str, str]],
    device: torch.device,
    max_samples: int,
) -> Dict[str, float]:

    if max_samples <= 0:

        selected = items

    else:

        selected = items[
            :max_samples
        ]

    total_abs = 0.0

    total_sq = 0.0

    pixels = 0

    samples = 0

    for index, item in enumerate(
        selected,
        1,
    ):

        sample = get_sample(
            item
        )

        if sample is None:

            continue

        image = None
        target = None
        mask = None
        pred = None

        try:

            image = (
                sample["image"]
                .unsqueeze(0)
                .to(device)
            )

            target = (
                sample["target"]
                .unsqueeze(0)
                .to(device)
            )

            mask = (
                sample["mask"]
                .unsqueeze(0)
                .to(device)
            )

            with amp_context():

                image_model = F.interpolate(
                    image,
                    size=(
                        MODEL_SIZE,
                        MODEL_SIZE,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

                pred = model_forward(
                    model,
                    image_model,
                )

                target_r, mask_r = (
                    resize_target_mask(
                        target,
                        mask,
                        pred.shape[-2:],
                    )
                )

            pred = pred.float()

            target_r = target_r.float()

            mask_r = mask_r.float()

            valid = (
                (mask_r > 0.5)
                & torch.isfinite(pred)
                & torch.isfinite(target_r)
            )

            if valid.any():

                error = (
                    pred
                    - target_r
                )[valid]

                total_abs += float(
                    torch.abs(
                        error
                    ).sum().cpu()
                )

                total_sq += float(
                    (
                        error * error
                    ).sum().cpu()
                )

                pixels += int(
                    valid.sum().item()
                )

                samples += 1

            if (
                index % 100 == 0
                or index == len(selected)
            ):

                current_mae = (
                    total_abs
                    / max(
                        1,
                        pixels,
                    )
                )

                print(
                    f"[VAL] "
                    f"{index}/{len(selected)} "
                    f"MAE={current_mae:.6f}"
                )

        except torch.cuda.OutOfMemoryError:

            print(
                f"[WARN] Validation OOM "
                f"at {sample['id']}; "
                "skipping."
            )

            torch.cuda.empty_cache()

        except Exception as exc:

            print(
                f"[WARN] Validation failure "
                f"at {sample['id']} | "
                f"{type(exc).__name__}: {exc}"
            )

        finally:

            if image is not None:
                del image

            if target is not None:
                del target

            if mask is not None:
                del mask

            if pred is not None:
                del pred

    if pixels == 0:

        return {
            "mae": float("inf"),
            "rmse": float("inf"),
            "samples": 0,
            "pixels": 0,
        }

    return {

        "mae":
            total_abs / pixels,

        "rmse":
            math.sqrt(
                total_sq / pixels
            ),

        "samples":
            samples,

        "pixels":
            pixels,
    }


# ============================================================================
# CONFIG SAVE
# ============================================================================

def save_config(
    train_items: List[Dict[str, str]],
    val_items: List[Dict[str, str]],
) -> None:

    payload = {

        "stage":
            5,

        "stage_name":
            "Urban3D",

        "dataset":
            "USSOCOM Urban3D",

        "train_crops":
            len(train_items),

        "validation_crops":
            len(val_items),

        "initial_checkpoint":
            str(STAGE4_BEST),

        "encoder":
            ENCODER,

        "features":
            FEATURES,

        "out_channels":
            OUT_CHANNELS,

        "model_size":
            MODEL_SIZE,

        "patch_size":
            PATCH_SIZE,

        "batch_size":
            BATCH_SIZE,

        "gradient_accumulation":
            GRAD_ACCUMULATION,

        "effective_batch_size":
            BATCH_SIZE
            * GRAD_ACCUMULATION,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRAD_CLIP,

        "target":
            "DSM - DTM",

        "negative_ndsm_clipped":
            CLIP_NEGATIVE_NDSM,

        "minimum_valid_fraction":
            MIN_VALID_FRACTION,

        "full_parameter_tuning":
            True,

        "seed":
            SEED,
    }

    atomic_json_write(
        CONFIG_FILE,
        payload,
    )


# ============================================================================
# TRAINING
# ============================================================================

def run_training(
    train_items: List[Dict[str, str]],
    val_items: List[Dict[str, str]],
    steps: int,
    val_steps: int,
    epochs: int,
    resume: Optional[Path],
) -> None:

    device = get_device()

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = create_model(
        STAGE4_BEST
    )

    enable_gradient_checkpointing(
        model
    )

    make_trainable(
        model
    )

    model.to(device)

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    optimizer = make_optimizer(
        model
    )

    scaler = make_scaler()

    # ------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------

    start_epoch = 1

    best_mae = float("inf")

    history: List[
        Dict[str, Any]
    ] = []

    if resume is not None:

        (
            start_epoch,
            best_mae,
            history,
        ) = load_resume_checkpoint(
            resume,
            model,
            optimizer,
            scaler,
        )

    # ------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------

    save_config(
        train_items,
        val_items,
    )

    # ------------------------------------------------------------
    # Training information
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("STARTING ASTERRA STAGE-5 URBAN3D FINE-TUNING")
    print("=" * 80)

    print()
    print("Stage-4 best")
    print("     ↓")
    print("Urban3D RGB")
    print("     +")
    print("nDSM = DSM - DTM")
    print("     ↓")
    print("ASTERRA Stage-5 Urban3D")
    print()

    print(
        "[OK] Full-parameter tuning: 100%"
    )

    print(
        f"[OK] Train crops: "
        f"{len(train_items)}"
    )

    print(
        f"[OK] Validation crops: "
        f"{len(val_items)}"
    )

    print(
        f"[OK] Effective batch size: "
        f"{BATCH_SIZE * GRAD_ACCUMULATION}"
    )

    print(
        f"[OK] Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"[OK] Epochs: "
        f"{epochs}"
    )

    print(
        "[OK] No synthetic elevation targets."
    )

    print(
        "[OK] No RPC geometry."
    )

    print(
        "[OK] Urban3D DSM/DTM target only."
    )

    # ------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------

    try:

        for epoch in range(
            start_epoch,
            epochs + 1,
        ):

            print()
            print("=" * 80)
            print(
                f"STAGE-5 URBAN3D EPOCH "
                f"{epoch}/{epochs}"
            )
            print("=" * 80)

            start_time = time.time()

            # ----------------------------------------------------
            # Train
            # ----------------------------------------------------

            train_loss, train_count = (
                train_one_epoch(
                    model,
                    optimizer,
                    scaler,
                    train_items,
                    device,
                    epoch,
                    steps,
                )
            )

            # ----------------------------------------------------
            # Validation
            # ----------------------------------------------------

            print()

            print(
                "[VALIDATION] Starting..."
            )

            metrics = validate(
                model,
                val_items,
                device,
                val_steps,
            )

            elapsed = (
                time.time()
                - start_time
            )

            # ----------------------------------------------------
            # Record
            # ----------------------------------------------------

            record = {

                "stage":
                    5,

                "dataset":
                    "Urban3D",

                "epoch":
                    epoch,

                "train_loss":
                    float(train_loss),

                "train_samples":
                    int(train_count),

                "validation_mae":
                    float(metrics["mae"]),

                "validation_rmse":
                    float(metrics["rmse"]),

                "validation_samples":
                    int(metrics["samples"]),

                "validation_pixels":
                    int(metrics["pixels"]),

                "learning_rate":
                    LEARNING_RATE,

                "elapsed_seconds":
                    float(elapsed),
            }

            history.append(
                record
            )

            atomic_json_write(
                HISTORY,
                history,
            )

            # ----------------------------------------------------
            # Best model
            # ----------------------------------------------------

            is_best = (
                metrics["mae"]
                < best_mae
            )

            if is_best:

                best_mae = (
                    metrics["mae"]
                )

                save_checkpoint(
                    BEST,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_mae,
                    history,
                    "best_validation_mae",
                )

                print()

                print(
                    "[OK] NEW BEST "
                    "STAGE-5 URBAN3D MODEL"
                )

                print(
                    f"[OK] Best MAE: "
                    f"{best_mae:.6f}"
                )

            # ----------------------------------------------------
            # Latest checkpoint
            # ----------------------------------------------------

            save_checkpoint(
                LATEST,
                model,
                optimizer,
                scaler,
                epoch,
                best_mae,
                history,
                "completed_epoch",
            )

            # ----------------------------------------------------
            # Epoch summary
            # ----------------------------------------------------

            print()

            print(
                f"Epoch {epoch} complete"
            )

            print(
                f"Train loss:       "
                f"{train_loss:.6f}"
            )

            print(
                f"Validation MAE:   "
                f"{metrics['mae']:.6f}"
            )

            print(
                f"Validation RMSE:  "
                f"{metrics['rmse']:.6f}"
            )

            print(
                f"Valid samples:    "
                f"{metrics['samples']}"
            )

            print(
                f"Valid pixels:     "
                f"{metrics['pixels']:,}"
            )

            print(
                f"Elapsed:          "
                f"{elapsed / 60.0:.2f} min"
            )

            print(
                f"Latest checkpoint:"
                f"\n    {LATEST}"
            )

            print(
                f"Best checkpoint:"
                f"\n    {BEST}"
            )

            # ----------------------------------------------------
            # GPU memory
            # ----------------------------------------------------

            if torch.cuda.is_available():

                peak = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )

                print(
                    f"Peak GPU memory: "
                    f"{peak:.2f} GB"
                )

                torch.cuda.reset_peak_memory_stats()

    except KeyboardInterrupt:

        print()
        print(
            "[WARNING] Training interrupted."
        )

        print(
            "[INFO] Saving emergency checkpoint..."
        )

        try:

            save_checkpoint(
                EMERGENCY,
                model,
                optimizer,
                scaler,
                max(
                    0,
                    len(history),
                ),
                best_mae,
                history,
                "keyboard_interrupt",
            )

            print(
                f"[OK] Emergency checkpoint:\n"
                f"{EMERGENCY}"
            )

        finally:

            raise

    finally:

        del model
        del optimizer
        del scaler

        gc.collect()

        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Finished
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE-5 URBAN3D TRAINING FINISHED")
    print("=" * 80)

    print(
        f"Latest:\n{LATEST}"
    )

    print(
        f"Best:\n{BEST}"
    )

    print(
        f"History:\n{HISTORY}"
    )

    print()

    print(
        f"Best validation MAE: "
        f"{best_mae:.6f}"
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "ASTERRA Stage-5 Urban3D "
            "full-parameter fine-tuning"
        )
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate Urban3D manifests, "
            "paths and sample construction."
        ),
    )

    parser.add_argument(
        "--gpu-smoke-test",
        action="store_true",
        help=(
            "Run one real GPU "
            "forward/backward/optimizer step."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help=(
            "Maximum training crops per epoch. "
            "0 = all crops."
        ),
    )

    parser.add_argument(
        "--val-steps",
        type=int,
        default=0,
        help=(
            "Maximum validation crops per epoch. "
            "0 = all validation crops."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            "Number of Stage-5 Urban3D epochs."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Stage-5 Urban3D checkpoint to resume."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    seed_everything()

    args = parse_args()

    if args.steps < 0:

        raise ValueError(
            "--steps must be >= 0"
        )

    if args.val_steps < 0:

        raise ValueError(
            "--val-steps must be >= 0"
        )

    if args.epochs <= 0:

        raise ValueError(
            "--epochs must be > 0"
        )

    print_header()

    # ------------------------------------------------------------
    # Manifests
    # ------------------------------------------------------------

    train_items = load_csv_manifest(
        TRAIN_MANIFEST
    )

    val_items = load_csv_manifest(
        VAL_MANIFEST
    )

    # ------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------

    if args.preflight:

        inspect_manifest(
            train_items,
            val_items,
        )

        return

    # ------------------------------------------------------------
    # GPU smoke
    # ------------------------------------------------------------

    if args.gpu_smoke_test:

        inspect_manifest(
            train_items[:4],
            val_items[:4],
        )

        ok = gpu_smoke_test(
            train_items
        )

        raise SystemExit(
            0 if ok else 1
        )

    # ------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------

    resume_path = None

    if args.resume:

        resume_path = Path(
            args.resume
        )

    # ------------------------------------------------------------
    # Actual training
    # ------------------------------------------------------------

    run_training(
        train_items,
        val_items,
        args.steps,
        args.val_steps,
        args.epochs,
        resume_path,
    )


if __name__ == "__main__":

    main()