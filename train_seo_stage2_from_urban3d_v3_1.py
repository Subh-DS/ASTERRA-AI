"""
ASTERRA AI — S-EO STAGE 2
LOCAL DATASET + URBAN3D V3.1 INITIALIZATION

Purpose
-------
Train the S-EO Stage-2 model from the verified Urban3D V3.1 checkpoint.

IMPORTANT TRANSFER:
    Stage 4
       ↓
    Urban3D V3
       ↓
    Urban3D V3.1
       ↓
    S-EO Stage 2   <-- this script

This script intentionally DOES NOT fall back to the old Vaihingen checkpoint.

Stage-2 dataset:
    D:\Asterra AI\datasets\S_EO_Stage2_Diverse

Expected:
    images/    = 141 RGB crops
    targets/   = 141 DSM-Max target crops
    masks/     = 141 validity masks
    manifest.csv

Target semantics:
    DSM-Max / absolute elevation representation used by the S-EO dataset.

IMPORTANT:
    Stage 5 Urban3D V3.1 predicts non-negative nDSM/height. S-EO targets
    include negative values in some crops, therefore Stage 2 uses an
    UNCONSTRAINED LINEAR final output, not ReLU/Softplus.

The V3.1 encoder + DPT representation is transferred, while only the final
32→1 projection is reinitialized around the S-EO training-target mean.

Training:
    - Full parameter fine-tuning
    - ViT-L / DINOv2 ViT-L/14
    - DPT decoder
    - batch 1
    - gradient accumulation 4
    - AMP
    - gradient checkpointing
    - masked L1
    - differential learning rates:
        encoder: 1e-6
        decoder: 1e-5
        final head: 1e-4
    - AOI-disjoint train/validation/test split comes from manifest.csv
    - test set is NEVER used for model selection

Recommended first:
    python train_seo_stage2_from_urban3d_v3_1.py --smoke-test

Then:
    python train_seo_stage2_from_urban3d_v3_1.py

Resume:
    python train_seo_stage2_from_urban3d_v3_1.py

A completed Stage-2 epoch is stored in:
    models\asterra_seo_v3_1_transfer\seo_latest.pth

Best validation checkpoint:
    models\asterra_seo_v3_1_transfer\seo_best.pth
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ============================================================================
# PROJECT
# ============================================================================

ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================================
# PATHS
# ============================================================================

DATASET_ROOT = ROOT / "datasets" / "S_EO_Stage2_Diverse"

IMAGES_DIR = DATASET_ROOT / "images"
TARGETS_DIR = DATASET_ROOT / "targets"
MASKS_DIR = DATASET_ROOT / "masks"
MANIFEST = DATASET_ROOT / "manifest.csv"

# Verified Urban3D V3.1 checkpoint.
URBAN3D_V3_1_CHECKPOINT = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d_v3_1"
    / "stage5_urban3d_v3_best.pth"
)

OUTPUT_DIR = ROOT / "models" / "asterra_seo_v3_1_transfer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT = OUTPUT_DIR / "seo_latest.pth"
BEST_CHECKPOINT = OUTPUT_DIR / "seo_best.pth"
HISTORY_FILE = OUTPUT_DIR / "seo_training_history.json"


# ============================================================================
# MODEL / TRAINING CONFIG
# ============================================================================

PATCH_SIZE = 512
MODEL_SIZE = 518

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

EPOCHS = 10

ENCODER_LR = 1e-6
DECODER_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4

AMP_ENABLED = True
GRADIENT_CHECKPOINTING = True
GRAD_CLIP = 1.0

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

SEED = 42
PRINT_EVERY = 10

# S-EO QC already established a minimum valid fraction around 0.90.
MIN_VALID_DSM = 0.80

# Do NOT clip the target elevation to zero.
# Do NOT use DSM-Min as DTM.
TARGET_CLIP_MIN = None
TARGET_CLIP_MAX = None


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_gpu_info() -> None:
    print("=" * 78)
    print("ASTERRA AI — S-EO STAGE 2")
    print("URBAN3D V3.1 → S-EO TRANSFER")
    print("=" * 78)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1024**3:.3f} GB")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("GPU: CUDA NOT AVAILABLE")

    print()
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Manifest: {MANIFEST}")
    print()
    print(f"Initialization: {URBAN3D_V3_1_CHECKPOINT}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    print(f"Patch: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Model input: {MODEL_SIZE} x {MODEL_SIZE}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRAD_ACCUMULATION}")
    print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print()
    print(f"Encoder LR: {ENCODER_LR}")
    print(f"Decoder LR: {DECODER_LR}")
    print(f"Head LR: {HEAD_LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Epochs: {EPOCHS}")
    print(f"AMP: {AMP_ENABLED}")
    print(f"Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print()
    print("Target: S-EO DSM-Max")
    print("Target activation: LINEAR / unrestricted")
    print("Loss: masked L1")
    print("=" * 78)
    print()


def print_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    print(f"--- GPU MEMORY: {label} ---")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.3f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.3f} GB")
    print(f"Peak:      {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")
    print()


# ============================================================================
# FILE MATCHING
# ============================================================================

ARRAY_EXTS = {".npy", ".npz", ".tif", ".tiff", ".png"}


def build_file_index(directory: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in ARRAY_EXTS:
            index.setdefault(p.stem, p)

    return index


def load_array(path: Path) -> np.ndarray:
    ext = path.suffix.lower()

    if ext == ".npy":
        return np.load(path)

    if ext == ".npz":
        data = np.load(path)
        keys = list(data.keys())
        if not keys:
            raise RuntimeError(f"Empty NPZ: {path}")
        return data[keys[0]]

    if ext in {".png"}:
        return np.asarray(Image.open(path))

    if ext in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as src:
                return src.read(1)
        except ImportError:
            return np.asarray(Image.open(path))

    raise ValueError(f"Unsupported array file: {path}")


def find_matching_file(
    index: Dict[str, Path],
    stem: str,
) -> Optional[Path]:
    candidates = [
        stem,
        f"{stem}_target",
        f"target_{stem}",
        f"{stem}_mask",
        f"mask_{stem}",
    ]

    for candidate in candidates:
        if candidate in index:
            return index[candidate]

    # Conservative fuzzy fallback.
    for key, path in index.items():
        if key.startswith(stem) or stem.startswith(key):
            return path

    return None


# ============================================================================
# MANIFEST
# ============================================================================

def read_manifest() -> List[Dict[str, str]]:
    import csv

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("Manifest contains zero rows.")

    return rows


def row_value(row: Dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return str(value)
    return ""


def resolve_image(row: Dict[str, str]) -> Optional[Path]:
    values = [
        row_value(row, "image", "image_path", "image_file", "rgb", "rgb_path"),
        row_value(row, "filename"),
        row_value(row, "sample_id"),
    ]

    for value in values:
        if not value:
            continue

        p = Path(value)

        if p.is_absolute() and p.exists():
            return p

        candidate = IMAGES_DIR / p.name
        if candidate.exists():
            return candidate

        matches = list(IMAGES_DIR.rglob(p.name))
        if matches:
            return matches[0]

        matches = list(IMAGES_DIR.rglob(p.stem + ".*"))
        if matches:
            return matches[0]

    return None


def resolve_target(row: Dict[str, str], target_index: Dict[str, Path], stem: str):
    explicit = row_value(
        row,
        "target",
        "target_path",
        "target_file",
        "dsm",
        "dsm_path",
    )

    if explicit:
        p = Path(explicit)
        if p.is_absolute() and p.exists():
            return p

        p2 = TARGETS_DIR / p.name
        if p2.exists():
            return p2

    return find_matching_file(target_index, stem)


def resolve_mask(row: Dict[str, str], mask_index: Dict[str, Path], stem: str):
    explicit = row_value(
        row,
        "mask",
        "mask_path",
        "mask_file",
    )

    if explicit:
        p = Path(explicit)
        if p.is_absolute() and p.exists():
            return p

        p2 = MASKS_DIR / p.name
        if p2.exists():
            return p2

    return find_matching_file(mask_index, stem)


def get_split(row: Dict[str, str]) -> str:
    return row_value(
        row,
        "split",
        "dataset_split",
        "partition",
    ).strip().lower()


def get_sample_id(row: Dict[str, str], image_path: Path) -> str:
    value = row_value(row, "sample_id", "id")
    return value or image_path.stem


def get_aoi(row: Dict[str, str], sample_id: str) -> str:
    value = row_value(row, "aoi", "aoi_name", "scene")
    if value:
        return value

    parts = sample_id.split("_")
    if len(parts) >= 2 and parts[0] in {"OMA", "UCSD", "JAX"}:
        return "_".join(parts[:2])

    return "UNKNOWN"


# ============================================================================
# LOCAL DATASET INDEX
# ============================================================================

class SEODataset:
    def __init__(self):
        self.rows = read_manifest()

        self.target_index = build_file_index(TARGETS_DIR)
        self.mask_index = build_file_index(MASKS_DIR)

        self.samples: List[Dict[str, Any]] = []

        missing_images = 0
        missing_targets = 0
        missing_masks = 0

        for row in self.rows:
            image_path = resolve_image(row)

            if image_path is None:
                missing_images += 1
                continue

            stem = image_path.stem

            target_path = resolve_target(
                row,
                self.target_index,
                stem,
            )

            mask_path = resolve_mask(
                row,
                self.mask_index,
                stem,
            )

            if target_path is None:
                missing_targets += 1
                continue

            if mask_path is None:
                missing_masks += 1

            sample_id = get_sample_id(row, image_path)
            split = get_split(row)
            aoi = get_aoi(row, sample_id)

            if split not in {"train", "validation", "val", "test"}:
                raise RuntimeError(
                    f"Unexpected split '{split}' for {sample_id}"
                )

            if split == "val":
                split = "validation"

            self.samples.append(
                {
                    "sample_id": sample_id,
                    "aoi": aoi,
                    "split": split,
                    "image": image_path,
                    "target": target_path,
                    "mask": mask_path,
                }
            )

        if missing_images or missing_targets:
            raise RuntimeError(
                "Dataset indexing failed:\n"
                f"  missing images:  {missing_images}\n"
                f"  missing targets: {missing_targets}\n"
                f"  missing masks:   {missing_masks}"
            )

        if missing_masks:
            print(
                f"[WARNING] {missing_masks} samples have no explicit mask. "
                "Finite target values will be used as the validity mask."
            )

        print(f"[OK] Manifest rows indexed: {len(self.rows)}")
        print(f"[OK] Usable samples: {len(self.samples)}")

        for split in ("train", "validation", "test"):
            items = self.by_split(split)
            print(
                f"  {split:10s}: {len(items):3d} samples | "
                f"{len({x['aoi'] for x in items}):2d} AOIs"
            )

        self._check_aoi_disjointness()

    def by_split(self, split: str) -> List[Dict[str, Any]]:
        return [x for x in self.samples if x["split"] == split]

    def _check_aoi_disjointness(self):
        sets = {
            split: {x["aoi"] for x in self.by_split(split)}
            for split in ("train", "validation", "test")
        }

        tv = sets["train"] & sets["validation"]
        tt = sets["train"] & sets["test"]
        vt = sets["validation"] & sets["test"]

        print(
            f"[OK] AOI overlap train/val={tv}, "
            f"train/test={tt}, val/test={vt}"
        )

        if tv or tt or vt:
            raise RuntimeError(
                "AOI leakage detected between train/validation/test."
            )

    def target_mean(self) -> float:
        values = []

        for sample in self.by_split("train"):
            target = load_array(sample["target"]).astype(np.float32)
            mask = (
                load_array(sample["mask"])
                if sample["mask"] is not None
                else np.isfinite(target)
            )

            target = np.squeeze(target)
            mask = np.squeeze(mask)

            if mask.shape != target.shape:
                mask = resize_array_nearest(mask, target.shape)

            valid = (
                np.isfinite(target)
                & (mask > 0)
            )

            if valid.any():
                values.append(float(target[valid].mean()))

        if not values:
            raise RuntimeError("Could not calculate training target mean.")

        return float(np.mean(values))


# ============================================================================
# ARRAY PREPARATION
# ============================================================================

def resize_array_nearest(
    arr: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    h, w = shape

    if arr.shape == shape:
        return arr

    # PIL float mode gives deterministic nearest-neighbor resizing.
    arr_float = np.asarray(arr, dtype=np.float32)
    img = Image.fromarray(arr_float, mode="F")
    img = img.resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(img)


def resize_rgb(
    rgb: np.ndarray,
    size: int,
) -> np.ndarray:
    rgb = np.asarray(rgb)

    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)

    if rgb.ndim != 3:
        raise ValueError(f"Invalid RGB shape: {rgb.shape}")

    if rgb.shape[-1] != 3:
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        else:
            raise ValueError(f"RGB must have 3 channels: {rgb.shape}")

    image = Image.fromarray(
        np.clip(rgb, 0, 255).astype(np.uint8),
        mode="RGB",
    )

    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(image)


def resize_target(
    target: np.ndarray,
    size: int,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    target = np.squeeze(target)

    if target.ndim != 2:
        raise ValueError(
            f"Invalid target shape after squeeze: {target.shape}"
        )

    tensor = torch.from_numpy(
        np.ascontiguousarray(target)
    )[None, None]

    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )

    return np.ascontiguousarray(
        tensor[0, 0].cpu().numpy(),
        dtype=np.float32,
    )


def resize_mask(
    mask: np.ndarray,
    size: int,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Invalid mask shape after squeeze: {mask.shape}"
        )

    tensor = torch.from_numpy(
        np.ascontiguousarray(mask)
    )[None, None]

    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="nearest",
    )

    return np.ascontiguousarray(
        tensor[0, 0].cpu().numpy(),
        dtype=np.float32,
    )


def prepare_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    rgb = load_array(sample["image"])
    target = load_array(sample["target"])

    rgb = resize_rgb(rgb, PATCH_SIZE)
    target = resize_target(target, PATCH_SIZE)

    if sample["mask"] is not None:
        mask = load_array(sample["mask"])
        mask = resize_mask(mask, PATCH_SIZE)
        mask = (mask > 0.5).astype(np.float32)
    else:
        mask = np.isfinite(target).astype(np.float32)

    # Target values are intentionally NOT clipped to zero.
    # S-EO validation contains negative target values.
    target = np.nan_to_num(
        target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    valid = (
        np.isfinite(target)
        & (mask > 0.5)
    )

    valid_fraction = float(valid.mean())

    if valid_fraction < MIN_VALID_DSM:
        raise ValueError(
            f"Valid fraction {valid_fraction:.4f} < "
            f"{MIN_VALID_DSM:.4f}"
        )

    image = np.asarray(rgb, dtype=np.uint8)
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(image).copy()
    ).permute(2, 0, 1).float() / 255.0

    target_tensor = torch.from_numpy(
        np.ascontiguousarray(target).copy()
    ).float()

    mask_tensor = torch.from_numpy(
        np.ascontiguousarray(mask).copy()
    ).float()

    return {
        "sample_id": sample["sample_id"],
        "aoi": sample["aoi"],
        "split": sample["split"],
        "image": image_tensor,
        "target": target_tensor,
        "mask": mask_tensor,
        "valid_fraction": valid_fraction,
    }


# ============================================================================
# MODEL
# ============================================================================

def extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dictionary.")

    for candidate in (
        "model_state_dict",
        "state_dict",
        "model",
        "weights",
    ):
        value = checkpoint.get(candidate)
        if isinstance(value, dict):
            return value

    return checkpoint


def clean_state_dict(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = {}

    for key, value in state.items():
        key = str(key)

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def configure_linear_s_eo_head(
    model: nn.Module,
    target_mean: float,
) -> None:
    """
    Convert the V3.1 positive-height head into an unrestricted linear head.

    Urban3D V3.1 used a positive output activation because nDSM >= 0.

    S-EO has observed negative target values in the prepared crops, so
    Softplus/ReLU would be mathematically inappropriate here.

    Only the final 32→1 projection is reinitialized. The encoder and DPT
    representation remain transferred from Urban3D V3.1.
    """

    output_conv2 = getattr(
        getattr(model, "depth_head", None),
        "scratch",
        None,
    )

    if output_conv2 is None:
        raise RuntimeError(
            "Could not locate model.depth_head.scratch."
        )

    output_conv2 = getattr(output_conv2, "output_conv2", None)

    if output_conv2 is None:
        raise RuntimeError(
            "Could not locate depth_head.scratch.output_conv2."
        )

    if not isinstance(output_conv2, nn.Sequential):
        raise RuntimeError(
            "Unexpected output_conv2 type: "
            f"{type(output_conv2)}"
        )

    # Known Depth Anything V2 DPT structure:
    #   [0] Conv2d
    #   [1] ReLU
    #   [2] Conv2d 32 -> 1
    #   [3] final activation
    #   [4] Identity
    if len(output_conv2) < 4:
        raise RuntimeError(
            f"Unexpected output_conv2 length: {len(output_conv2)}"
        )

    final_conv = output_conv2[2]

    if not isinstance(final_conv, nn.Conv2d):
        raise RuntimeError(
            f"Expected output_conv2[2] to be Conv2d, got {type(final_conv)}"
        )

    if final_conv.out_channels != 1:
        raise RuntimeError(
            f"Expected final output channels=1, got "
            f"{final_conv.out_channels}"
        )

    # The activation after the final conv must not force non-negative output.
    output_conv2[3] = nn.Identity()

    # Keep learned encoder/decoder, but replace the Urban3D-specific
    # projection. Initialize it around the actual S-EO training mean.
    with torch.no_grad():
        nn.init.normal_(
            final_conv.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.constant_(
            final_conv.bias,
            float(target_mean),
        )

    print()
    print("=" * 78)
    print("S-EO OUTPUT HEAD ADAPTATION")
    print("=" * 78)
    print("Transferred:")
    print("  ✓ ViT-L / DINOv2 encoder")
    print("  ✓ DPT decoder")
    print("  ✓ learned Urban3D representation")
    print()
    print("Adapted:")
    print("  ✓ final 32→1 projection reinitialized")
    print("  ✓ final ReLU/Softplus removed")
    print("  ✓ unrestricted linear elevation output")
    print(f"  ✓ initial output bias = training target mean {target_mean:.4f}")
    print("=" * 78)
    print()


def create_model(target_mean: float) -> nn.Module:
    if not URBAN3D_V3_1_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Required Urban3D V3.1 checkpoint was not found:\n"
            f"{URBAN3D_V3_1_CHECKPOINT}"
        )

    print("=" * 78)
    print("CREATING DEPTH ANYTHING V2 LARGE")
    print("=" * 78)

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    print("Loading Urban3D V3.1 checkpoint:")
    print(URBAN3D_V3_1_CHECKPOINT)

    checkpoint = torch.load(
        URBAN3D_V3_1_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = clean_state_dict(
        extract_state_dict(checkpoint)
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing or unexpected:
        for key in missing[:20]:
            print(f"  Missing: {key}")

        for key in unexpected[:20]:
            print(f"  Unexpected: {key}")

        raise RuntimeError(
            "Urban3D V3.1 checkpoint is not fully compatible with "
            "the current Depth Anything V2 Large architecture."
        )

    print("[OK] Urban3D V3.1 weights loaded perfectly.")

    configure_linear_s_eo_head(
        model,
        target_mean=target_mean,
    )

    del checkpoint
    gc.collect()

    return model


def enable_gradient_checkpointing(
    model: nn.Module,
) -> None:
    if not GRADIENT_CHECKPOINTING:
        return

    pretrained = getattr(model, "pretrained", None)
    blocks = getattr(pretrained, "blocks", None)

    if blocks is None:
        print(
            "[WARNING] Transformer blocks not found; "
            "continuing without gradient checkpoint flag."
        )
        return

    for block in blocks:
        if hasattr(block, "gradient_checkpointing"):
            block.gradient_checkpointing = True

    if hasattr(pretrained, "gradient_checkpointing"):
        pretrained.gradient_checkpointing = True

    print(
        f"[OK] Gradient checkpointing enabled for "
        f"{len(blocks)} transformer blocks."
    )


def parameter_report(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 78)
    print("PARAMETER CONFIGURATION")
    print("=" * 78)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(
        f"Trainable percentage: "
        f"{100.0 * trainable / max(total, 1):.2f}%"
    )

    if total != trainable:
        raise RuntimeError(
            "Full-parameter fine-tuning requested, but some parameters "
            "are frozen."
        )


# ============================================================================
# PARAMETER GROUPS / OPTIMIZER
# ============================================================================

def classify_parameter(
    name: str,
) -> str:
    """
    Three LR groups.

    Encoder:
        pretrained transformer

    Head:
        final 32→1 projection

    Decoder:
        DPT / scratch / depth_head except final projection
    """

    if name.startswith("pretrained."):
        return "encoder"

    if name.startswith("depth_head.scratch.output_conv2.2."):
        return "head"

    return "decoder"


def create_optimizer(model: nn.Module):
    groups = {
        "encoder": [],
        "decoder": [],
        "head": [],
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        groups[classify_parameter(name)].append(parameter)

    print()
    print("=" * 78)
    print("DIFFERENTIAL LEARNING RATES")
    print("=" * 78)
    print(
        f"Encoder parameters: {sum(p.numel() for p in groups['encoder']):,} "
        f"@ {ENCODER_LR}"
    )
    print(
        f"Decoder parameters: {sum(p.numel() for p in groups['decoder']):,} "
        f"@ {DECODER_LR}"
    )
    print(
        f"Head parameters:    {sum(p.numel() for p in groups['head']):,} "
        f"@ {HEAD_LR}"
    )
    print("=" * 78)

    return torch.optim.AdamW(
        [
            {
                "params": groups["encoder"],
                "lr": ENCODER_LR,
                "weight_decay": WEIGHT_DECAY,
                "group_name": "encoder",
            },
            {
                "params": groups["decoder"],
                "lr": DECODER_LR,
                "weight_decay": WEIGHT_DECAY,
                "group_name": "decoder",
            },
            {
                "params": groups["head"],
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
                "group_name": "head",
            },
        ]
    )


# ============================================================================
# FORWARD / LOSS
# ============================================================================

def model_forward(
    model: nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    output = model(image)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError(
            f"Model output is not a tensor: {type(output)}"
        )

    if output.ndim == 3:
        output = output.unsqueeze(1)

    if output.ndim != 4:
        raise ValueError(
            f"Unexpected model output shape: {tuple(output.shape)}"
        )

    if output.shape[1] != 1:
        output = output[:, :1]

    return output


def resize_target_and_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
):
    if target.ndim == 3:
        target = target.unsqueeze(1)

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    target = F.interpolate(
        target.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    mask = F.interpolate(
        mask.float(),
        size=size,
        mode="nearest",
    )

    return target, mask


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim == 4:
        prediction = prediction[:, 0]

    if target.ndim == 4:
        target = target[:, 0]

    if mask.ndim == 4:
        mask = mask[:, 0]

    valid = (
        (mask > 0.5)
        & torch.isfinite(prediction)
        & torch.isfinite(target)
    )

    if int(valid.sum().item()) == 0:
        raise RuntimeError("Zero valid pixels in masked L1.")

    return torch.abs(
        prediction.float() - target.float()
    )[valid].mean()


def prepare_for_model(
    sample: Dict[str, Any],
    dev: torch.device,
):
    image = sample["image"].unsqueeze(0).to(
        dev,
        non_blocking=True,
    )

    target = sample["target"].unsqueeze(0).to(
        dev,
        non_blocking=True,
    )

    mask = sample["mask"].unsqueeze(0).to(
        dev,
        non_blocking=True,
    )

    image = F.interpolate(
        image,
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return image, target, mask


def autocast_context():
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(
            torch.cuda.is_available()
            and AMP_ENABLED
        ),
    )


def create_scaler():
    if not torch.cuda.is_available() or not AMP_ENABLED:
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except TypeError:
        return torch.cuda.amp.GradScaler(
            enabled=True,
        )


# ============================================================================
# CHECKPOINTS
# ============================================================================

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

    os.replace(
        tmp,
        path,
    )


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(
            HISTORY_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

    except Exception as exc:
        print(
            f"[WARNING] History could not be read: "
            f"{type(exc).__name__}: {exc}"
        )

    return []


def build_payload(
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    train_loss: float,
    train_samples: int,
    validation: Dict[str, Any],
    best_mae: float,
    target_mean: float,
) -> Dict[str, Any]:
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    payload = {
        "stage": "S-EO",
        "stage_name": "S-EO Stage 2",
        "dataset": "S_EO_Stage2_Diverse",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "train_samples": int(train_samples),
        "validation": validation,
        "best_validation_mae": float(best_mae),

        "transfer": {
            "source_stage": "Stage 5 Urban3D V3.1",
            "source_checkpoint": str(
                URBAN3D_V3_1_CHECKPOINT
            ),
            "source_target": "nDSM / height above ground",
            "source_output": "positive-height head",
            "target_stage": "S-EO Stage 2",
            "target_semantics": "DSM-Max / absolute elevation representation",
            "target_output": "unrestricted linear elevation",
            "final_head_reinitialized": True,
            "target_mean_initial_bias": float(target_mean),
        },

        "config": {
            "encoder": ENCODER,
            "features": FEATURES,
            "out_channels": OUT_CHANNELS,
            "patch_size": PATCH_SIZE,
            "model_size": MODEL_SIZE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUMULATION,
            "effective_batch_size": (
                BATCH_SIZE * GRAD_ACCUMULATION
            ),
            "full_parameter_finetuning": True,
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "encoder_lr": ENCODER_LR,
            "decoder_lr": DECODER_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "amp": AMP_ENABLED,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "optimizer": type(optimizer).__name__,
            "loss": "masked L1",
            "target_activation": "linear",
            "target_clip_min": TARGET_CLIP_MIN,
            "target_clip_max": TARGET_CLIP_MAX,
            "dataset_root": str(DATASET_ROOT),
            "manifest": str(MANIFEST),
            "target_mean_initial_bias": float(target_mean),
        },
    }

    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    return payload


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    train_loss: float,
    train_samples: int,
    validation: Dict[str, Any],
    best_mae: float,
    target_mean: float,
) -> None:
    payload = build_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        train_loss=train_loss,
        train_samples=train_samples,
        validation=validation,
        best_mae=best_mae,
        target_mean=target_mean,
    )

    atomic_torch_save(
        payload,
        path,
    )

    print(f"[OK] Checkpoint saved: {path}")


# ============================================================================
# RESUME
# ============================================================================

def resume_from_latest(
    model: nn.Module,
    optimizer,
    scaler,
) -> Tuple[int, float]:
    if not LATEST_CHECKPOINT.exists():
        print()
        print("[INFO] No Stage-2 latest checkpoint found.")
        print("[INFO] Starting from Urban3D V3.1 transfer checkpoint.")
        return 1, float("inf")

    print()
    print("=" * 78)
    print("RESUMING S-EO STAGE 2")
    print("=" * 78)
    print(f"Latest: {LATEST_CHECKPOINT}")

    payload = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = clean_state_dict(
        extract_state_dict(payload)
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Existing Stage-2 checkpoint is incompatible with "
            "the current model."
        )

    optimizer_state = payload.get(
        "optimizer_state_dict"
    )

    if isinstance(optimizer_state, dict):
        try:
            optimizer.load_state_dict(
                optimizer_state
            )
            print("[OK] Optimizer state resumed.")
        except Exception as exc:
            print(
                "[WARNING] Optimizer state could not be resumed: "
                f"{type(exc).__name__}: {exc}"
            )

    scaler_state = payload.get(
        "scaler_state_dict"
    )

    if scaler is not None and isinstance(
        scaler_state,
        dict,
    ):
        try:
            scaler.load_state_dict(
                scaler_state
            )
            print("[OK] AMP scaler state resumed.")
        except Exception as exc:
            print(
                "[WARNING] AMP scaler state could not be resumed: "
                f"{type(exc).__name__}: {exc}"
            )

    epoch = int(
        payload.get(
            "epoch",
            0,
        )
    )

    best_mae = float(
        payload.get(
            "best_validation_mae",
            float("inf"),
        )
    )

    print(
        f"[OK] Resuming after completed epoch {epoch}."
    )
    print(
        f"[OK] Next epoch: {epoch + 1}"
    )
    print(
        f"[OK] Best validation MAE: {best_mae:.6f}"
    )

    del payload
    gc.collect()

    return epoch + 1, best_mae


# ============================================================================
# TRAIN / VALIDATE
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    optimizer,
    scaler,
    samples: List[Dict[str, Any]],
    dev: torch.device,
    epoch: int,
) -> Tuple[float, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    order = list(range(len(samples)))
    random.shuffle(order)

    loss_sum = 0.0
    successful = 0
    accumulation = 0
    updates = 0

    started = time.time()

    for step, index in enumerate(order, start=1):
        sample_meta = samples[index]

        try:
            sample = prepare_sample(
                sample_meta
            )

            image, target, mask = prepare_for_model(
                sample,
                dev,
            )

            with autocast_context():
                prediction = model_forward(
                    model,
                    image,
                )

                target_resized, mask_resized = (
                    resize_target_and_mask(
                        target,
                        mask,
                        prediction.shape[-2:],
                    )
                )

                loss = masked_l1(
                    prediction,
                    target_resized,
                    mask_resized,
                )

                if not torch.isfinite(loss):
                    print(
                        f"[WARNING] Non-finite loss at step {step}; "
                        "skipping."
                    )
                    optimizer.zero_grad(
                        set_to_none=True
                    )
                    accumulation = 0
                    continue

                scaled_loss = (
                    loss / GRAD_ACCUMULATION
                )

            if scaler is not None:
                scaler.scale(
                    scaled_loss
                ).backward()
            else:
                scaled_loss.backward()

            accumulation += 1

            should_step = (
                accumulation >= GRAD_ACCUMULATION
                or step == len(order)
            )

            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                accumulation = 0
                updates += 1

            value = float(
                loss.detach()
                .float()
                .cpu()
            )

            loss_sum += value
            successful += 1

            if (
                step == 1
                or step % PRINT_EVERY == 0
                or step == len(order)
            ):
                average = (
                    loss_sum
                    / max(successful, 1)
                )

                elapsed = (
                    time.time()
                    - started
                )

                print(
                    f"Epoch {epoch} "
                    f"Step {step}/{len(order)} | "
                    f"Loss {value:.5f} | "
                    f"Avg {average:.5f} | "
                    f"AOI {sample['aoi']} | "
                    f"Updates {updates} | "
                    f"Time {elapsed:.1f}s"
                )

                if step == 1:
                    print_memory(
                        f"epoch {epoch} step {step}"
                    )

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(
                set_to_none=True
            )
            raise

        except Exception as exc:
            optimizer.zero_grad(
                set_to_none=True
            )

            print(
                f"[WARNING] Sample failed: "
                f"{sample_meta['sample_id']} | "
                f"{type(exc).__name__}: {exc}"
            )

    if accumulation > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

    if successful == 0:
        raise RuntimeError(
            "Zero successful training samples."
        )

    return (
        loss_sum / successful,
        successful,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    dev: torch.device,
) -> Dict[str, Any]:
    model.eval()

    total_abs = 0.0
    total_sq = 0.0
    valid_pixels = 0
    sample_count = 0

    started = time.time()

    for sample_meta in samples:
        try:
            sample = prepare_sample(
                sample_meta
            )

            image, target, mask = prepare_for_model(
                sample,
                dev,
            )

            with autocast_context():
                prediction = model_forward(
                    model,
                    image,
                )

                target_resized, mask_resized = (
                    resize_target_and_mask(
                        target,
                        mask,
                        prediction.shape[-2:],
                    )
                )

            prediction = prediction.float()
            target_resized = target_resized.float()
            mask_resized = mask_resized.float()

            valid = (
                (mask_resized > 0.5)
                & torch.isfinite(prediction)
                & torch.isfinite(target_resized)
            )

            count = int(
                valid.sum().item()
            )

            if count == 0:
                continue

            diff = (
                prediction
                - target_resized
            )

            total_abs += float(
                torch.abs(diff)[valid]
                .sum()
                .cpu()
            )

            total_sq += float(
                (diff * diff)[valid]
                .sum()
                .cpu()
            )

            valid_pixels += count
            sample_count += 1

        except Exception as exc:
            print(
                f"[WARNING] Validation sample failed: "
                f"{sample_meta['sample_id']} | "
                f"{type(exc).__name__}: {exc}"
            )

    if valid_pixels == 0:
        return {
            "status": "unavailable",
            "mae": float("inf"),
            "rmse": float("inf"),
            "samples": 0,
            "valid_pixels": 0,
        }

    mae = (
        total_abs
        / valid_pixels
    )

    rmse = math.sqrt(
        total_sq
        / valid_pixels
    )

    print(
        f"Validation: samples={sample_count} | "
        f"MAE={mae:.6f} | "
        f"RMSE={rmse:.6f} | "
        f"pixels={valid_pixels:,} | "
        f"time={time.time()-started:.1f}s"
    )

    return {
        "status": "ok",
        "mae": float(mae),
        "rmse": float(rmse),
        "samples": int(sample_count),
        "valid_pixels": int(valid_pixels),
    }


# ============================================================================
# SMOKE TEST
# ============================================================================

def smoke_test(
    dataset: SEODataset,
    model: nn.Module,
    dev: torch.device,
    target_mean: float,
) -> bool:
    print()
    print("=" * 78)
    print("S-EO V3.1 TRANSFER GPU SMOKE TEST")
    print("=" * 78)

    train_samples = dataset.by_split("train")

    if not train_samples:
        print("[ERROR] No training samples.")
        return False

    sample = prepare_sample(
        train_samples[0]
    )

    image, target, mask = prepare_for_model(
        sample,
        dev,
    )

    model.to(dev)
    model.train()

    optimizer = create_optimizer(model)
    scaler = create_scaler()

    optimizer.zero_grad(
        set_to_none=True
    )

    with autocast_context():
        prediction = model_forward(
            model,
            image,
        )

        target, mask = (
            resize_target_and_mask(
                target,
                mask,
                prediction.shape[-2:],
            )
        )

        loss = masked_l1(
            prediction,
            target,
            mask,
        )

    print(
        f"Sample: {sample['sample_id']}"
    )
    print(
        f"Target mean initialization: {target_mean:.6f}"
    )
    print(
        f"Prediction shape: {tuple(prediction.shape)}"
    )
    print(
        f"Initial prediction mean: "
        f"{float(prediction.float().mean().cpu()):.6f}"
    )
    print(
        f"Initial prediction min: "
        f"{float(prediction.float().min().cpu()):.6f}"
    )
    print(
        f"Initial prediction max: "
        f"{float(prediction.float().max().cpu()):.6f}"
    )
    print(
        f"Forward loss: {float(loss.float().cpu()):.6f}"
    )

    scaled_loss = (
        loss / GRAD_ACCUMULATION
    )

    if scaler is not None:
        scaler.scale(
            scaled_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )
    else:
        scaled_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(
        set_to_none=True
    )

    print("[OK] Forward + backward + optimizer step succeeded.")
    print_memory("after smoke test")

    del optimizer
    del scaler
    del image
    del target
    del mask
    del prediction

    gc.collect()
    torch.cuda.empty_cache()

    return True


# ============================================================================
# FULL TRAINING
# ============================================================================

def run_training(
    smoke: bool = False,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Stage-2 full-parameter training."
        )

    dev = get_device()

    print_gpu_info()

    # ------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("S-EO LOCAL DATASET PREFLIGHT")
    print("=" * 78)

    dataset = SEODataset()

    train_samples = dataset.by_split(
        "train"
    )

    val_samples = dataset.by_split(
        "validation"
    )

    test_samples = dataset.by_split(
        "test"
    )

    if not train_samples:
        raise RuntimeError(
            "No training samples found."
        )

    if not val_samples:
        raise RuntimeError(
            "No validation samples found."
        )

    if not test_samples:
        raise RuntimeError(
            "No test samples found."
        )

    # ------------------------------------------------------------
    # Target statistics
    # ------------------------------------------------------------
    target_mean = dataset.target_mean()

    print()
    print(
        f"[OK] Mean training target used for initial head bias: "
        f"{target_mean:.6f} m"
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------
    model = create_model(
        target_mean=target_mean
    )

    enable_gradient_checkpointing(
        model
    )

    parameter_report(
        model
    )

    model.to(dev)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_memory(
        "after model loading"
    )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------
    optimizer = create_optimizer(
        model
    )

    scaler = create_scaler()

    # ------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------
    if smoke:
        ok = smoke_test(
            dataset,
            model,
            dev,
            target_mean,
        )

        if not ok:
            raise RuntimeError(
                "Smoke test failed."
            )

        print()
        print("[OK] Smoke test complete.")
        return

    # ------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------
    start_epoch, best_mae = (
        resume_from_latest(
            model,
            optimizer,
            scaler,
        )
    )

    history = load_history()

    if start_epoch > EPOCHS:
        print()
        print(
            f"Latest epoch={start_epoch-1} "
            f"already reaches configured EPOCHS={EPOCHS}."
        )
        print("Nothing to train.")
        return

    # ------------------------------------------------------------
    # Training information
    # ------------------------------------------------------------
    print()
    print("=" * 78)
    print("STARTING S-EO STAGE 2")
    print("=" * 78)

    print()
    print("TRANSFER CHAIN:")
    print("Depth Anything V2 Large")
    print("        ↓")
    print("GeoNRW")
    print("        ↓")
    print("Potsdam")
    print("        ↓")
    print("Vaihingen")
    print("        ↓")
    print("Urban3D V3")
    print("        ↓")
    print("Urban3D V3.1")
    print("        ↓")
    print("S-EO Stage 2  ← CURRENT")

    print()
    print("DATA:")
    print(f"Train:      {len(train_samples)} crops")
    print(f"Validation: {len(val_samples)} crops")
    print(f"Test:       {len(test_samples)} crops")
    print()
    print("Test set is NOT used during training or model selection.")
    print()
    print("TARGET:")
    print("DSM-Max / S-EO elevation representation")
    print("No DSM-Min-as-DTM conversion.")
    print("No nDSM subtraction.")
    print("No zero clipping.")
    print("Linear output permits negative target values.")
    print()
    print("OPTIMIZATION:")
    print(f"Encoder LR: {ENCODER_LR}")
    print(f"Decoder LR: {DECODER_LR}")
    print(f"Head LR:    {HEAD_LR}")
    print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUMULATION}")

    # ------------------------------------------------------------
    # Epochs
    # ------------------------------------------------------------
    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):
        print()
        print("=" * 78)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 78)

        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------
        train_loss, train_count = (
            train_one_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                samples=train_samples,
                dev=dev,
                epoch=epoch,
            )
        )

        print()
        print(
            f"Epoch {epoch} train loss: "
            f"{train_loss:.6f}"
        )

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------
        print()
        print("Running validation...")

        validation = validate(
            model=model,
            samples=val_samples,
            dev=dev,
        )

        print()
        print("=" * 78)
        print("VALIDATION RESULT")
        print("=" * 78)
        print(
            f"MAE:  {validation['mae']:.6f}"
        )
        print(
            f"RMSE: {validation['rmse']:.6f}"
        )
        print(
            f"Samples: {validation['samples']}"
        )
        print(
            f"Pixels:  {validation['valid_pixels']:,}"
        )

        # --------------------------------------------------------
        # LATEST
        # --------------------------------------------------------
        save_checkpoint(
            path=LATEST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            train_loss=train_loss,
            train_samples=train_count,
            validation=validation,
            best_mae=best_mae,
            target_mean=target_mean,
        )

        # --------------------------------------------------------
        # BEST
        # --------------------------------------------------------
        val_mae = float(
            validation.get(
                "mae",
                float("inf"),
            )
        )

        validation_available = (
            validation.get("status") == "ok"
            and math.isfinite(val_mae)
            and validation.get("samples", 0) > 0
            and validation.get("valid_pixels", 0) > 0
        )

        if (
            validation_available
            and val_mae < best_mae
        ):
            best_mae = val_mae

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                train_samples=train_count,
                validation=validation,
                best_mae=best_mae,
                target_mean=target_mean,
            )

            print()
            print(
                f"NEW BEST S-EO CHECKPOINT — "
                f"MAE {best_mae:.6f}"
            )

        else:
            print()
            print(
                f"Best checkpoint unchanged — "
                f"best MAE {best_mae:.6f}"
            )

        # --------------------------------------------------------
        # HISTORY
        # --------------------------------------------------------
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_samples": int(train_count),
                "validation": validation,
                "best_validation_mae": float(best_mae),
                "target_mean_initial_bias": float(
                    target_mean
                ),
                "source_checkpoint": str(
                    URBAN3D_V3_1_CHECKPOINT
                ),
            }
        )

        HISTORY_FILE.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

        print_memory(
            f"end of epoch {epoch}"
        )

    print()
    print("=" * 78)
    print("S-EO STAGE 2 TRAINING COMPLETE")
    print("=" * 78)
    print()
    print(
        f"Best validation MAE: "
        f"{best_mae:.6f}"
    )
    print()
    print("Best checkpoint:")
    print(BEST_CHECKPOINT)
    print()
    print("Latest checkpoint:")
    print(LATEST_CHECKPOINT)
    print()
    print("History:")
    print(HISTORY_FILE)
    print()
    print(
        "IMPORTANT: Test-set evaluation should be performed only "
        "after training/model selection is finished."
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ASTERRA S-EO Stage-2 trainer initialized from "
            "Urban3D V3.1."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one real forward/backward/optimizer step only.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    seed_everything(
        SEED
    )

    run_training(
        smoke=args.smoke_test
    )


if __name__ == "__main__":
    main()
