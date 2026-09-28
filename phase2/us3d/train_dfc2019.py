"""
ASTERRA AI — STAGE 3
US3D / IEEE GRSS DFC2019 TRACK 1
FULL-PARAMETER SINGLE-VIEW ABOVE-GROUND HEIGHT FINE-TUNING

Purpose
-------
Continue the ASTERRA Depth Anything V2 Large curriculum:

    Depth Anything V2 Large
        -> GeoNRW
        -> Potsdam
        -> Vaihingen
        -> S-EO
        -> DFC2019 / US3D Track 1  <-- CURRENT STAGE

DFC2019 Track 1 provides:
    RGB  : 1024x1024 uint8 WorldView-3 image
    AGL  : float32 above-ground height in metres
    CLS  : semantic class labels

This trainer uses RGB + AGL supervision. MSI is intentionally NOT required.

Expected local layout (the trainer also searches recursively):
    D:\Asterra AI\DFC2019\
        Track 1\
            Training data\
                Train-Track1-RGB\
                Train-Track1-Truth\

Official DFC2019 documentation identifies Track 1 as single-view semantic
3D with above-ground height targets, and specifies RGB/AGL/CLS formats.

Design
------
* Starts from models\\asterra_seo\\seo_best.pth.
* Full 335M-parameter fine-tuning.
* Batch size 1 + gradient accumulation 4.
* AMP + gradient checkpointing.
* Adafactor when available.
* Local TIFF files; no remote streaming.
* No repeated network access during training.
* A lightweight local manifest is created from filenames only.
* Validation split is persistent and tile-level, preventing geographic
  leakage between source images from the same tile.
* Uses AGL (metres) as the supervised target.
* Uses CLS to exclude DFC2019 unlabeled pixels (class 65) when available.
* Random 512x512 crop from each 1024x1024 source image.
* Default bare command trains over every discovered training pair per epoch.
* --steps can cap samples per epoch for a faster controlled run.
* --val-steps caps validation samples.
* Saves latest checkpoint after every completed epoch.
* Saves best checkpoint only when validation MAE improves.
* KeyboardInterrupt saves an emergency checkpoint.
* Existing Stage-3 checkpoints resume automatically.
* NumPy arrays are copied/contiguous before torch conversion.
* Partial gradient accumulation is flushed safely; GradScaler is never
  stepped without gradients.

Run
---
    python .\\phase2\\us3d\\train_dfc2019.py --stream-test
    python .\\phase2\\us3d\\train_dfc2019.py --gpu-smoke-test
    python .\\phase2\\us3d\\train_dfc2019.py --steps 10 --val-steps 3
    python .\\phase2\\us3d\\train_dfc2019.py

If you place this file somewhere else, the project root is still resolved
from the file location when possible, with D:\\Asterra AI as the fallback.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tifffile
except ImportError as exc:
    raise RuntimeError(
        "tifffile is required for DFC2019 TIFF files. "
        "Install it with: pip install tifffile"
    ) from exc

try:
    import rasterio
except ImportError:
    rasterio = None


# ============================================================================
# PROJECT PATHS
# ============================================================================

THIS_FILE = Path(__file__).resolve()

# Expected location:
# D:\Asterra AI\phase2\us3d\train_dfc2019.py
if THIS_FILE.parent.name.lower() == "us3d":
    ROOT = THIS_FILE.parents[2]
else:
    ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================================
# CHECKPOINTS
# ============================================================================

STAGE2_BEST = ROOT / "models" / "asterra_seo" / "seo_best.pth"

OUTPUT_DIR = ROOT / "models" / "asterra_dfc2019"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT = OUTPUT_DIR / "dfc2019_latest.pth"
BEST_CHECKPOINT = OUTPUT_DIR / "dfc2019_best.pth"
EMERGENCY_CHECKPOINT = OUTPUT_DIR / "dfc2019_emergency.pth"
HISTORY_FILE = OUTPUT_DIR / "dfc2019_training_history.json"
MANIFEST_FILE = OUTPUT_DIR / "dfc2019_manifest.json"
VALIDATION_SPLIT_FILE = OUTPUT_DIR / "dfc2019_validation_split.json"


# ============================================================================
# DATA DISCOVERY
# ============================================================================

DATASET_ROOT = ROOT / "datasets" / "DFC2019"

RGB_DIR_NAMES = {
    "train-track1-rgb",
    "train_track1_rgb",
    "track1-rgb",
    "track1_rgb",
}

TRUTH_DIR_NAMES = {
    "train-track1-truth",
    "train_track1_truth",
    "track1-truth",
    "track1_truth",
}

# DFC2019 Track 1 official file suffixes.
RGB_SUFFIX = "_RGB"
AGL_SUFFIX = "_AGL"
CLS_SUFFIX = "_CLS"

# Official Track 1 images are 1024x1024.
SOURCE_SIZE = 1024

# Training crop.
PATCH_SIZE = 512

# ViT-L/14-friendly model input.
MODEL_SIZE = 518

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

EPOCHS = 3

# 0 means: use every discovered training pair once per epoch.
DEFAULT_STEPS = 0

DEFAULT_VAL_STEPS = 10

LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-4

AMP_ENABLED = True
GRADIENT_CHECKPOINTING = True
GRAD_CLIP = 1.0

PRINT_EVERY = 10

SEED = 42

# AGL is in metres. Keep a broad but sane physical range.
AGL_MIN = 0.0
AGL_MAX = 500.0

# Require most of a crop to contain usable labeled pixels.
MIN_VALID_FRACTION = 0.50

# DFC2019 LAS classification value for unlabeled pixels.
UNLABELED_CLASS = 65

# Persistent validation fraction at geographic tile level.
VAL_FRACTION = 0.10

# Model configuration matching the Stage-2 model.
ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_header() -> None:
    print("=" * 78)
    print("ASTERRA — STAGE 3 US3D / DFC2019 TRACK 1")
    print("FULL-PARAMETER SINGLE-VIEW HEIGHT FINE-TUNING")
    print("=" * 78)

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {p.total_memory / 1024**3:.3f} GB")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("GPU: CUDA NOT AVAILABLE")

    print()
    print(f"Dataset root: {DATASET_ROOT}")
    print("Dataset: US3D / IEEE GRSS DFC2019 Track 1")
    print("Task: RGB -> above-ground height (AGL, metres)")
    print()
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Model input: {MODEL_SIZE} x {MODEL_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRAD_ACCUMULATION}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"AMP: {AMP_ENABLED}")
    print(f"Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Epochs: {EPOCHS}")
    print(f"Default steps/epoch: {'ALL PAIRS' if DEFAULT_STEPS == 0 else DEFAULT_STEPS}")
    print(f"Validation samples: {DEFAULT_VAL_STEPS}")
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
# FILE / NAME HANDLING
# ============================================================================

def normalize_stem(path: Path, suffix: str) -> Optional[str]:
    stem = path.stem

    # Remove .tif/.TIF suffix through Path.stem.
    if not stem.upper().endswith(suffix.upper()):
        return None

    return stem[:-len(suffix)]


def tile_id_from_stem(stem: str) -> str:
    """
    JAX_163_010 -> JAX_163

    Validation is held out by geographic tile, not source-image number.
    """
    parts = stem.split("_")

    if len(parts) >= 2:
        return "_".join(parts[:2])

    return stem


def find_named_dirs(root: Path, names: Sequence[str]) -> List[Path]:
    wanted = {x.lower() for x in names}
    found: List[Path] = []

    if not root.exists():
        return found

    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in wanted:
            found.append(p)

    # Stable ordering and de-duplication.
    unique = sorted({p.resolve() for p in found}, key=lambda x: str(x).lower())
    return unique


def discover_data_dirs():
    """
    Automatically locate the DFC2019 Track 1 dataset.

    Expected layout:

        D:\\Asterra AI\\datasets\\DFC2019
        └── Track 1
            └── Training data
                ├── Train-Track1-RGB
                └── Train-Track1-Truth

    Also supports the old fallback location:

        D:\\Asterra AI\\DFC2019
    """

    candidates = [
        # PRIMARY — your actual dataset location
        ROOT / "datasets" / "DFC2019",

        # FALLBACK — old layout
        ROOT / "DFC2019",

        # Explicit absolute path for this machine
        Path(r"D:\Asterra AI\datasets\DFC2019"),
    ]

    checked = []

    for root in candidates:
        root = Path(root).resolve()

        if root in checked:
            continue

        checked.append(root)

        track1_training = root / "Track 1" / "Training data"

        # Expected directories
        rgb_dir = track1_training / "Train-Track1-RGB"
        truth_dir = track1_training / "Train-Track1-Truth"

        if rgb_dir.is_dir() and truth_dir.is_dir():

            # Verify actual TIFF data exists.
            rgb_files = list(rgb_dir.rglob("*_RGB.tif"))
            agl_files = list(truth_dir.rglob("*_AGL.tif"))

            if len(rgb_files) > 0 and len(agl_files) > 0:

                print()
                print("=" * 78)
                print("DFC2019 DATASET FOUND")
                print("=" * 78)
                print(f"Dataset root : {root}")
                print(f"RGB directory: {rgb_dir}")
                print(f"Truth dir    : {truth_dir}")
                print(f"RGB TIFFs    : {len(rgb_files):,}")
                print(f"AGL TIFFs    : {len(agl_files):,}")
                print("=" * 78)
                print()

                return rgb_dir, truth_dir

    # Nothing worked.
    print()
    print("=" * 78)
    print("DFC2019 DATASET SEARCH FAILED")
    print("=" * 78)

    for path in checked:
        print(f"Checked: {path}")

    print()
    print("Your expected dataset layout is:")
    print(
        r"D:\Asterra AI\datasets\DFC2019\Track 1\Training data"
    )
    print("    ├── Train-Track1-RGB")
    print("    └── Train-Track1-Truth")
    print("=" * 78)

    raise FileNotFoundError(
        "DFC2019 Track 1 dataset could not be located. "
        "Expected D:\\Asterra AI\\datasets\\DFC2019."
    )

def build_manifest(rgb_dir: Path, truth_dir: Path) -> List[Dict[str, str]]:
    """
    Build a lightweight local manifest.

    This is NOT image data in RAM. It only records paths and IDs, so it is
    inexpensive and safe for a local DFC2019 dataset.
    """
    rgb_files = sorted(
        [
            p for p in rgb_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".tif", ".tiff"}
            and p.stem.upper().endswith(RGB_SUFFIX)
        ],
        key=lambda p: p.name.lower(),
    )

    agl_files = {}
    cls_files = {}

    for p in truth_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".tif", ".tiff"}:
            continue

        upper = p.stem.upper()

        if upper.endswith(AGL_SUFFIX):
            key = p.stem[:-len(AGL_SUFFIX)].upper()
            agl_files[key] = p

        elif upper.endswith(CLS_SUFFIX):
            key = p.stem[:-len(CLS_SUFFIX)].upper()
            cls_files[key] = p

    manifest: List[Dict[str, str]] = []
    missing_agl = 0

    for rgb_path in rgb_files:
        stem = normalize_stem(rgb_path, RGB_SUFFIX)
        if stem is None:
            continue

        key = stem.upper()
        agl_path = agl_files.get(key)

        if agl_path is None:
            missing_agl += 1
            continue

        cls_path = cls_files.get(key)

        manifest.append(
            {
                "id": stem,
                "tile": tile_id_from_stem(stem),
                "rgb": str(rgb_path),
                "agl": str(agl_path),
                "cls": str(cls_path) if cls_path else "",
            }
        )

    if not manifest:
        raise RuntimeError(
            "No RGB + AGL pairs were found.\n"
            f"RGB directory: {rgb_dir}\n"
            f"Truth directory: {truth_dir}"
        )

    print()
    print("=" * 78)
    print("DFC2019 LOCAL MANIFEST")
    print("=" * 78)
    print(f"RGB files found:       {len(rgb_files)}")
    print(f"RGB without AGL:       {missing_agl}")
    print(f"Usable RGB + AGL pairs: {len(manifest)}")
    print(f"Geographic tiles:      {len({x['tile'] for x in manifest})}")

    return manifest


def save_manifest(manifest: List[Dict[str, str]]) -> None:
    payload = {
        "version": 1,
        "dataset": "DFC2019 Track 1",
        "created_by": "ASTERRA train_dfc2019.py",
        "pairs": manifest,
    }

    tmp = MANIFEST_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    tmp.replace(MANIFEST_FILE)


def load_or_build_manifest(
    rgb_dir: Path,
    truth_dir: Path,
) -> List[Dict[str, str]]:
    # Rebuild if absent. Rebuild is cheap because only filenames are scanned.
    if MANIFEST_FILE.exists():
        try:
            payload = json.loads(
                MANIFEST_FILE.read_text(encoding="utf-8")
            )
            pairs = payload.get("pairs")
            if isinstance(pairs, list) and pairs:
                # Verify a few/all referenced paths still exist.
                valid = all(
                    Path(item["rgb"]).exists()
                    and Path(item["agl"]).exists()
                    for item in pairs
                )
                if valid:
                    print(f"[OK] Reusing local manifest: {MANIFEST_FILE}")
                    return pairs
        except Exception:
            pass

    manifest = build_manifest(rgb_dir, truth_dir)
    save_manifest(manifest)
    print(f"[OK] Manifest saved: {MANIFEST_FILE}")
    return manifest


# ============================================================================
# PERSISTENT VALIDATION SPLIT
# ============================================================================

def deterministic_fraction(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def create_or_load_validation_split(
    manifest: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Persistent geographic tile split.

    The split is based on tile IDs such as JAX_163 / OMA_250, so all source
    views from the same geographic tile remain in one partition.
    """
    all_tiles = sorted({item["tile"] for item in manifest})

    if VALIDATION_SPLIT_FILE.exists():
        try:
            payload = json.loads(
                VALIDATION_SPLIT_FILE.read_text(encoding="utf-8")
            )
            train_tiles = set(payload.get("train_tiles", []))
            val_tiles = set(payload.get("validation_tiles", []))

            if (
                train_tiles
                and val_tiles
                and train_tiles.isdisjoint(val_tiles)
                and train_tiles.union(val_tiles) == set(all_tiles)
            ):
                print()
                print("=" * 78)
                print("PERSISTENT DFC2019 VALIDATION SPLIT")
                print("=" * 78)
                print(f"Mode: geographic tile split")
                print(f"Train tiles: {len(train_tiles)}")
                print(f"Validation tiles: {len(val_tiles)}")
                print(f"Validation: {sorted(val_tiles)}")
                return payload
        except Exception:
            pass

    if len(all_tiles) < 2:
        raise RuntimeError(
            "DFC2019 Track 1 training data exposes fewer than 2 geographic "
            "tiles. A leakage-safe validation split cannot be created."
        )

    ranked = sorted(
        all_tiles,
        key=lambda t: hashlib.sha1(
            f"{SEED}:{t}".encode("utf-8")
        ).hexdigest(),
    )

    val_count = max(
        1,
        int(round(len(all_tiles) * VAL_FRACTION)),
    )

    val_count = min(
        val_count,
        len(all_tiles) - 1,
    )

    validation_tiles = sorted(ranked[:val_count])
    train_tiles = sorted(ranked[val_count:])

    payload = {
        "version": 1,
        "dataset": "DFC2019 Track 1",
        "seed": SEED,
        "fraction": VAL_FRACTION,
        "train_tiles": train_tiles,
        "validation_tiles": validation_tiles,
    }

    tmp = VALIDATION_SPLIT_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    tmp.replace(VALIDATION_SPLIT_FILE)

    print()
    print("=" * 78)
    print("CREATED PERSISTENT DFC2019 VALIDATION SPLIT")
    print("=" * 78)
    print(f"Train tiles: {len(train_tiles)}")
    print(f"Validation tiles: {len(validation_tiles)}")
    print(f"Validation tiles: {validation_tiles}")

    return payload


def split_manifest(
    manifest: List[Dict[str, str]],
    split: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train_set = set(split["train_tiles"])
    val_set = set(split["validation_tiles"])

    train = [x for x in manifest if x["tile"] in train_set]
    val = [x for x in manifest if x["tile"] in val_set]

    if not train:
        raise RuntimeError("Training split is empty.")

    if not val:
        raise RuntimeError("Validation split is empty.")

    return train, val


# ============================================================================
# TIFF READING
# ============================================================================

def read_tiff(
    path: Path,
    window: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Read TIFF either through rasterio windowing or tifffile.

    window = (x, y, width, height)
    """
    if rasterio is not None:
        with rasterio.open(path) as src:
            if window is None:
                arr = src.read()
            else:
                x, y, w, h = window
                arr = src.read(
                    window=rasterio.windows.Window(
                        x, y, w, h
                    )
                )

        return np.ascontiguousarray(arr)

    arr = tifffile.imread(str(path))
    return np.ascontiguousarray(arr)


def normalize_rgb_array(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)

    # Rasterio gives C,H,W.
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4):
        rgb = np.transpose(rgb[:3], (1, 2, 0))

    # Tifffile generally gives H,W,3.
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)

    if rgb.ndim != 3:
        raise ValueError(f"Unexpected RGB shape: {rgb.shape}")

    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]

    if rgb.shape[-1] != 3:
        raise ValueError(f"RGB must have 3 channels: {rgb.shape}")

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    # DFC2019 RGB is uint8. Clip protects against unusual reader behaviour.
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb)


def normalize_agl_array(agl: np.ndarray) -> np.ndarray:
    agl = np.asarray(agl)

    if agl.ndim == 3:
        if agl.shape[0] == 1:
            agl = agl[0]
        elif agl.shape[-1] == 1:
            agl = agl[..., 0]
        else:
            agl = agl[0]

    if agl.ndim != 2:
        raise ValueError(f"Unexpected AGL shape: {agl.shape}")

    return np.ascontiguousarray(
        np.asarray(agl, dtype=np.float32)
    )


def normalize_cls_array(cls: np.ndarray) -> np.ndarray:
    cls = np.asarray(cls)

    if cls.ndim == 3:
        if cls.shape[0] == 1:
            cls = cls[0]
        elif cls.shape[-1] == 1:
            cls = cls[..., 0]
        else:
            cls = cls[0]

    return np.ascontiguousarray(
        np.asarray(cls, dtype=np.uint8)
    )


# ============================================================================
# SAMPLE PREPARATION
# ============================================================================

def crop_coordinates(
    h: int,
    w: int,
    patch: int,
    rng: random.Random,
) -> Tuple[int, int]:
    if h < patch or w < patch:
        return 0, 0

    x = rng.randint(0, w - patch)
    y = rng.randint(0, h - patch)

    return x, y


def read_training_crop(
    item: Dict[str, str],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """
    Random aligned RGB/AGL/CLS crop from one Track-1 source image.

    For normal DFC2019 files this reads only 512x512 from each TIFF when
    rasterio is installed. If rasterio is unavailable, the full 1024x1024
    TIFF is read and cropped in RAM.
    """
    rgb_path = Path(item["rgb"])
    agl_path = Path(item["agl"])
    cls_path = Path(item["cls"]) if item.get("cls") else None

    try:
        # Get dimensions without loading the full image.
        if rasterio is not None:
            with rasterio.open(rgb_path) as src:
                h = src.height
                w = src.width
        else:
            # TIFF fallback.
            shape = tifffile.memmap(str(rgb_path)).shape
            if len(shape) == 3 and shape[0] in (3, 4):
                h, w = shape[-2], shape[-1]
            else:
                h, w = shape[:2]

        x, y = crop_coordinates(h, w, PATCH_SIZE, rng)

        rgb = read_tiff(
            rgb_path,
            (x, y, min(PATCH_SIZE, w), min(PATCH_SIZE, h)),
        )
        agl = read_tiff(
            agl_path,
            (x, y, min(PATCH_SIZE, w), min(PATCH_SIZE, h)),
        )

        cls = None
        if cls_path is not None and cls_path.exists():
            cls = read_tiff(
                cls_path,
                (x, y, min(PATCH_SIZE, w), min(PATCH_SIZE, h)),
            )

        rgb = normalize_rgb_array(rgb)
        agl = normalize_agl_array(agl)

        if cls is not None:
            cls = normalize_cls_array(cls)

        # If a source image is unexpectedly smaller than 512, resize all
        # aligned arrays to the target patch size.
        if rgb.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            from PIL import Image

            rgb_img = Image.fromarray(rgb, mode="RGB")
            rgb_img = rgb_img.resize(
                (PATCH_SIZE, PATCH_SIZE),
                Image.Resampling.BILINEAR,
            )
            rgb = np.asarray(rgb_img)

            agl_t = torch.from_numpy(
                np.array(agl, dtype=np.float32, copy=True)
            )[None, None]

            agl_t = F.interpolate(
                agl_t,
                size=(PATCH_SIZE, PATCH_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            agl = agl_t[0, 0].numpy()

            if cls is not None:
                cls_img = Image.fromarray(cls, mode="L")
                cls_img = cls_img.resize(
                    (PATCH_SIZE, PATCH_SIZE),
                    Image.Resampling.NEAREST,
                )
                cls = np.asarray(cls_img)

        # Keep AGL in metres and remove physically invalid values.
        agl = np.asarray(agl, dtype=np.float32)
        valid = np.isfinite(agl)
        valid &= agl >= AGL_MIN
        valid &= agl <= AGL_MAX

        # Exclude DFC2019 unlabeled pixels if CLS is present.
        if cls is not None:
            valid &= cls != UNLABELED_CLASS

        valid_fraction = float(valid.mean())

        if valid_fraction < MIN_VALID_FRACTION:
            return None

        clean_agl = np.where(
            valid,
            agl,
            0.0,
        ).astype(np.float32)

        rgb_tensor = torch.from_numpy(
            np.ascontiguousarray(rgb).copy()
        ).permute(2, 0, 1).float() / 255.0

        target_tensor = torch.from_numpy(
            np.ascontiguousarray(clean_agl).copy()
        ).float()

        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(valid.astype(np.float32)).copy()
        ).float()

        return {
            "image": rgb_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "id": item["id"],
            "tile": item["tile"],
            "valid_fraction": valid_fraction,
        }

    except Exception as exc:
        print(
            f"[SKIP] Could not read {item['id']}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def read_validation_sample(
    item: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """
    Deterministic center crop for validation.

    Every validation source image gets one fixed center crop, so validation
    is repeatable across epochs.
    """
    rgb_path = Path(item["rgb"])
    agl_path = Path(item["agl"])
    cls_path = Path(item["cls"]) if item.get("cls") else None

    try:
        if rasterio is not None:
            with rasterio.open(rgb_path) as src:
                h = src.height
                w = src.width
        else:
            shape = tifffile.memmap(str(rgb_path)).shape
            if len(shape) == 3 and shape[0] in (3, 4):
                h, w = shape[-2], shape[-1]
            else:
                h, w = shape[:2]

        x = max(0, (w - PATCH_SIZE) // 2)
        y = max(0, (h - PATCH_SIZE) // 2)

        rgb = normalize_rgb_array(
            read_tiff(
                rgb_path,
                (x, y, min(PATCH_SIZE, w), min(PATCH_SIZE, h)),
            )
        )

        agl = normalize_agl_array(
            read_tiff(
                agl_path,
                (x, y, min(PATCH_SIZE, w), min(PATCH_SIZE, h)),
            )
        )

        cls = None
        if cls_path is not None and cls_path.exists():
            cls = normalize_cls_array(
                read_tiff(
                    cls_path,
                    (x, y, min(PATCH_SIZE, w), min(PATCH_SIZE, h)),
                )
            )

        # Pad/resize unusually small inputs.
        if rgb.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            from PIL import Image

            rgb_img = Image.fromarray(rgb, mode="RGB")
            rgb = np.asarray(
                rgb_img.resize(
                    (PATCH_SIZE, PATCH_SIZE),
                    Image.Resampling.BILINEAR,
                )
            )

            agl_t = torch.from_numpy(
                np.array(agl, dtype=np.float32, copy=True)
            )[None, None]
            agl = F.interpolate(
                agl_t,
                size=(PATCH_SIZE, PATCH_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

            if cls is not None:
                cls = np.asarray(
                    Image.fromarray(cls, mode="L").resize(
                        (PATCH_SIZE, PATCH_SIZE),
                        Image.Resampling.NEAREST,
                    )
                )

        valid = np.isfinite(agl)
        valid &= agl >= AGL_MIN
        valid &= agl <= AGL_MAX

        if cls is not None:
            valid &= cls != UNLABELED_CLASS

        if float(valid.mean()) < MIN_VALID_FRACTION:
            return None

        clean_agl = np.where(
            valid,
            agl,
            0.0,
        ).astype(np.float32)

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(rgb).copy()
            ).permute(2, 0, 1).float() / 255.0,
            "target": torch.from_numpy(
                np.ascontiguousarray(clean_agl).copy()
            ).float(),
            "mask": torch.from_numpy(
                np.ascontiguousarray(valid.astype(np.float32)).copy()
            ).float(),
            "id": item["id"],
            "tile": item["tile"],
            "valid_fraction": float(valid.mean()),
        }

    except Exception as exc:
        print(
            f"[SKIP-VAL] {item['id']}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


# ============================================================================
# MODEL / CHECKPOINT UTILITIES
# ============================================================================

def extract_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        for name in (
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
        ):
            value = payload.get(name)
            if isinstance(value, dict):
                return value

        if all(
            isinstance(k, str) and torch.is_tensor(v)
            for k, v in payload.items()
        ):
            return payload

    raise RuntimeError(
        "Could not locate a model state_dict in checkpoint."
    )


def clean_state_dict(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    result = {}

    for key, value in state.items():
        key = str(key)
        if key.startswith("module."):
            key = key[7:]
        result[key] = value

    return result


def create_model(
    checkpoint_path: Path,
) -> nn.Module:
    print()
    print("=" * 78)
    print("CREATING DEPTH ANYTHING V2 LARGE")
    print("=" * 78)

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Required checkpoint not found:\n{checkpoint_path}"
        )

    print(f"Loading checkpoint:\n{checkpoint_path}")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = clean_state_dict(
        extract_state_dict(payload)
    )

    result = model.load_state_dict(
        state,
        strict=False,
    )

    print(f"Missing keys:    {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with Depth Anything V2 Large."
        )

    print("[OK] Checkpoint loaded perfectly.")

    return model


def enable_gradient_checkpointing(model: nn.Module) -> None:
    if not GRADIENT_CHECKPOINTING:
        return

    pretrained = getattr(model, "pretrained", None)
    blocks = getattr(pretrained, "blocks", None)

    if blocks is None:
        print(
            "[WARNING] Transformer blocks not found; continuing without "
            "explicit block checkpointing."
        )
        return

    for block in blocks:
        if hasattr(block, "gradient_checkpointing"):
            block.gradient_checkpointing = True

    if hasattr(pretrained, "gradient_checkpointing"):
        pretrained.gradient_checkpointing = True

    print(
        f"Gradient checkpointing enabled for {len(blocks)} transformer blocks."
    )


def make_all_trainable(model: nn.Module) -> None:
    total = 0
    trainable = 0

    for p in model.parameters():
        p.requires_grad_(True)
        total += p.numel()
        trainable += p.numel()

    print()
    print("=" * 78)
    print("PARAMETER CONFIGURATION")
    print("=" * 78)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable percentage: {100.0 * trainable / total:.2f}%")

    if total != trainable:
        raise RuntimeError(
            "Full-parameter tuning requested but some parameters are frozen."
        )


# ============================================================================
# OPTIMIZER / AMP
# ============================================================================

def create_optimizer(model: nn.Module):
    try:
        optimizer_cls = getattr(torch.optim, "Adafactor")
    except AttributeError:
        optimizer_cls = None

    if optimizer_cls is not None:
        try:
            opt = optimizer_cls(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
            print()
            print("Optimizer: torch.optim.Adafactor")
            print("Reason: memory-efficient full-parameter optimization")
            return opt
        except Exception as exc:
            print(f"[WARNING] Adafactor unavailable: {exc}")

    print()
    print("Optimizer: torch.optim.AdamW")
    print(
        "[WARNING] AdamW may require substantially more memory for "
        "335M-parameter full tuning."
    )

    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def autocast_context():
    if torch.cuda.is_available() and AMP_ENABLED:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )

    return torch.autocast(
        device_type="cpu",
        enabled=False,
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
# MODEL FORWARD / LOSS
# ============================================================================

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
                f"Unknown model output dictionary keys: {list(out.keys())}"
            )

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not torch.is_tensor(out):
        raise RuntimeError(
            f"Model output is not a tensor: {type(out)}"
        )

    if out.ndim == 3:
        out = out.unsqueeze(1)

    if out.ndim != 4:
        raise RuntimeError(
            f"Unexpected model output shape: {tuple(out.shape)}"
        )

    if out.shape[1] != 1:
        out = out[:, :1]

    return out


def resize_target_and_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)

    if target.ndim == 3:
        target = target.unsqueeze(1)

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    if prediction.shape[-2:] != target.shape[-2:]:
        prediction = F.interpolate(
            prediction,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    valid = mask > 0.5
    valid &= torch.isfinite(prediction)
    valid &= torch.isfinite(target)

    count = int(valid.sum().item())

    if count == 0:
        return prediction.sum() * 0.0

    return torch.abs(
        prediction.float() - target.float()
    )[valid].mean()


# ============================================================================
# SAFE OPTIMIZER STEP
# ============================================================================

def has_gradients(model: nn.Module) -> bool:
    return any(
        p.requires_grad and p.grad is not None
        for p in model.parameters()
    )


def safe_optimizer_step(
    model: nn.Module,
    optimizer,
    scaler,
) -> bool:
    """
    Prevents:
        AssertionError: No inf checks were recorded for this optimizer.

    This is especially important with gradient accumulation and interrupted
    or partial batches.
    """
    if not has_gradients(model):
        optimizer.zero_grad(set_to_none=True)
        return False

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

    optimizer.zero_grad(set_to_none=True)

    return True


# ============================================================================
# CHECKPOINTS
# ============================================================================

def atomic_torch_save(
    payload: Dict[str, Any],
    path: Path,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


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
    payload = {
        "stage": "US3D_DFC2019_TRACK1",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": float(best_mae),
        "history": history,
        "reason": reason,
        "dataset": "US3D / DFC2019 Track 1",
        "initial_checkpoint": str(STAGE2_BEST),
        "patch_size": PATCH_SIZE,
        "model_size": MODEL_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_accumulation": GRAD_ACCUMULATION,
        "full_parameter_tuning": True,
        "validation_split_file": str(VALIDATION_SPLIT_FILE),
    }

    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    atomic_torch_save(
        payload,
        path,
    )

    print(f"Checkpoint saved: {path}")


def load_resume_checkpoint(
    model: nn.Module,
    optimizer,
    scaler,
) -> Tuple[int, float, List[Dict[str, Any]]]:
    if not LATEST_CHECKPOINT.exists():
        return 1, float("inf"), []

    print()
    print("=" * 78)
    print("RESUMING STAGE 3 TRAINING")
    print("=" * 78)
    print(f"Latest checkpoint: {LATEST_CHECKPOINT}")

    payload = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = clean_state_dict(
        extract_state_dict(payload)
    )

    result = model.load_state_dict(
        state,
        strict=True,
    )

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Existing DFC2019 checkpoint is incompatible."
        )

    opt_state = payload.get("optimizer_state_dict")
    if isinstance(opt_state, dict):
        optimizer.load_state_dict(opt_state)
        print("[OK] Optimizer state resumed.")

    if scaler is not None:
        scaler_state = payload.get("scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
            print("[OK] AMP scaler state resumed.")

    epoch = int(payload.get("epoch", 0))
    best_mae = float(
        payload.get("best_val_mae", float("inf"))
    )

    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []

    print(f"[OK] Resumed completed epoch: {epoch}")
    print(
        f"[OK] Best validation MAE: "
        f"{best_mae:.6f}"
        if math.isfinite(best_mae)
        else "[OK] Best validation MAE: not established"
    )

    return epoch + 1, best_mae, history


# ============================================================================
# TRAINING / VALIDATION
# ============================================================================

def train_epoch(
    model: nn.Module,
    optimizer,
    scaler,
    items: List[Dict[str, str]],
    dev: torch.device,
    epoch: int,
    max_steps: int,
) -> Tuple[float, int]:
    model.train()

    order = list(range(len(items)))
    rng = random.Random(SEED + epoch)
    rng.shuffle(order)

    if max_steps > 0:
        order = order[:min(max_steps, len(order))]

    print(
        f"Training samples this epoch: {len(order)} "
        f"(dataset pairs available: {len(items)})"
    )

    loss_sum = 0.0
    successful = 0
    accumulation = 0
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, index in enumerate(order, start=1):
        item = items[index]

        # Try a few times with deterministic alternative RNG state if a
        # crop happens to contain too many invalid pixels.
        sample = None
        for attempt in range(4):
            sample_rng = random.Random(
                SEED
                + epoch * 1_000_003
                + index * 97
                + attempt
            )
            sample = read_training_crop(
                item,
                sample_rng,
            )
            if sample is not None:
                break

        if sample is None:
            continue

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

        try:
            with autocast_context():
                image_model = F.interpolate(
                    image,
                    size=(MODEL_SIZE, MODEL_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )

                prediction = model_forward(
                    model,
                    image_model,
                )

                target_r, mask_r = resize_target_and_mask(
                    target,
                    mask,
                    prediction.shape[-2:],
                )

                loss = masked_l1_loss(
                    prediction,
                    target_r,
                    mask_r,
                )

                if not torch.isfinite(loss):
                    print(
                        f"[WARNING] Non-finite loss at "
                        f"step {step}; sample skipped."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaled_loss = loss / GRAD_ACCUMULATION

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accumulation += 1

            # Normal update or final partial accumulation.
            if (
                accumulation >= GRAD_ACCUMULATION
                or step == len(order)
            ):
                updated = safe_optimizer_step(
                    model,
                    optimizer,
                    scaler,
                )

                if updated:
                    accumulation = 0
                else:
                    accumulation = 0

            value = float(
                loss.detach().float().cpu()
            )

            loss_sum += value
            successful += 1

            if (
                step == 1
                or step % PRINT_EVERY == 0
                or step == len(order)
            ):
                elapsed = time.time() - start
                avg = loss_sum / max(successful, 1)

                print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"Step {step}/{len(order)} | "
                    f"Loss {value:.5f} | "
                    f"Avg {avg:.5f} | "
                    f"Tile {sample['tile']} | "
                    f"Time {elapsed:.1f}s"
                )

            if step == 1 or step % 100 == 0:
                print_memory(
                    f"epoch {epoch} step {step}"
                )

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            raise

        finally:
            del image, target, mask
            del sample

    # Safety flush if the loop ended on a partial accumulation boundary.
    if accumulation > 0:
        safe_optimizer_step(
            model,
            optimizer,
            scaler,
        )

    if successful == 0:
        raise RuntimeError(
            "No successful DFC2019 training samples were produced."
        )

    return loss_sum / successful, successful


@torch.no_grad()
def validate(
    model: nn.Module,
    items: List[Dict[str, str]],
    dev: torch.device,
    max_samples: int,
) -> Dict[str, Any]:
    model.eval()

    # Deterministic order.
    ordered = sorted(
        items,
        key=lambda x: x["id"],
    )

    total_abs = 0.0
    total_sq = 0.0
    valid_pixels = 0
    samples = 0
    scanned = 0

    start = time.time()

    print()
    print("=" * 78)
    print("SHORT DFC2019 VALIDATION")
    print("=" * 78)
    print(f"Maximum validation samples: {max_samples}")

    for item in ordered:
        if samples >= max_samples:
            break

        scanned += 1

        sample = read_validation_sample(item)
        if sample is None:
            continue

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

        with autocast_context():
            image_model = F.interpolate(
                image,
                size=(MODEL_SIZE, MODEL_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            prediction = model_forward(
                model,
                image_model,
            )

            target_r, mask_r = resize_target_and_mask(
                target,
                mask,
                prediction.shape[-2:],
            )

        pred = prediction.float()
        tgt = target_r.float()
        valid = mask_r > 0.5

        valid &= torch.isfinite(pred)
        valid &= torch.isfinite(tgt)

        count = int(valid.sum().item())

        if count == 0:
            continue

        diff = pred - tgt

        total_abs += float(
            torch.abs(diff)[valid].sum().cpu()
        )

        total_sq += float(
            torch.square(diff)[valid].sum().cpu()
        )

        valid_pixels += count
        samples += 1

        if (
            samples == 1
            or samples % 5 == 0
            or samples == max_samples
        ):
            mae_now = total_abs / max(valid_pixels, 1)
            rmse_now = math.sqrt(
                total_sq / max(valid_pixels, 1)
            )

            elapsed = time.time() - start

            print(
                f"Validation {samples}/{max_samples} | "
                f"MAE {mae_now:.5f} | "
                f"RMSE {rmse_now:.5f} | "
                f"Tile {sample['tile']} | "
                f"Scanned {scanned} | "
                f"Time {elapsed:.1f}s"
            )

        del image, target, mask, prediction

    if samples == 0 or valid_pixels == 0:
        raise RuntimeError(
            "DFC2019 validation produced zero usable pixels."
        )

    mae = total_abs / valid_pixels
    rmse = math.sqrt(total_sq / valid_pixels)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "valid_pixels": int(valid_pixels),
        "samples": int(samples),
        "scanned": int(scanned),
    }


# ============================================================================
# PREFLIGHT TESTS
# ============================================================================

def stream_test(
    train_items: List[Dict[str, str]],
    count: int = 3,
) -> bool:
    print()
    print("=" * 78)
    print("DFC2019 TRACK 1 PAIR TEST")
    print("=" * 78)
    print("Reading real local RGB + AGL (+ CLS) pairs.")
    print()

    for i, item in enumerate(train_items[:count], start=1):
        try:
            # Center crop gives a stable preflight sample.
            sample = read_validation_sample(item)
            if sample is None:
                print(
                    f"[WARNING] Pair {item['id']} did not produce a "
                    "usable center crop."
                )
                continue

            print(
                f"PAIR {i:02d}: "
                f"tile={item['tile']} | "
                f"id={item['id']} | "
                f"valid={sample['valid_fraction']:.3f}"
            )
        except Exception as exc:
            print(
                f"[ERROR] Pair {item['id']} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    print()
    print("[OK] DFC2019 Track 1 RGB ↔ AGL pair test passed.")
    return True


def gpu_smoke_test(
    train_items: List[Dict[str, str]],
) -> bool:
    print()
    print("=" * 78)
    print("DFC2019 GPU ONE-BATCH SMOKE TEST")
    print("=" * 78)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available.")
        return False

    sample = None

    for item in train_items[:20]:
        sample = read_validation_sample(item)
        if sample is not None:
            break

    if sample is None:
        print("[ERROR] No usable DFC2019 sample found.")
        return False

    dev = torch.device("cuda")

    model = None
    optimizer = None
    scaler = None

    try:
        # Stage 3 starts from S-EO best, never Vaihingen.
        model = create_model(STAGE2_BEST)
        enable_gradient_checkpointing(model)
        make_all_trainable(model)
        model.to(dev)

        optimizer = create_optimizer(model)
        scaler = create_scaler()

        image = sample["image"].unsqueeze(0).to(dev)
        target = sample["target"].unsqueeze(0).to(dev)
        mask = sample["mask"].unsqueeze(0).to(dev)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            image_model = F.interpolate(
                image,
                size=(MODEL_SIZE, MODEL_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            prediction = model_forward(model, image_model)
            target_r, mask_r = resize_target_and_mask(
                target,
                mask,
                prediction.shape[-2:],
            )
            loss = masked_l1_loss(
                prediction,
                target_r,
                mask_r,
            )

            scaled = loss / GRAD_ACCUMULATION

        print(
            f"Smoke sample: {sample['id']} | "
            f"tile={sample['tile']}"
        )
        print(f"Forward loss: {float(loss.detach().cpu()):.6f}")

        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        safe_optimizer_step(
            model,
            optimizer,
            scaler,
        )

        print("[OK] Forward + backward + optimizer step succeeded.")
        print_memory("after GPU smoke test")

        return True

    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM during GPU smoke test.")
        return False

    except Exception as exc:
        print(
            f"[ERROR] GPU smoke test failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return False

    finally:
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# HISTORY
# ============================================================================

def save_history(history: List[Dict[str, Any]]) -> None:
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    tmp.replace(HISTORY_FILE)


# ============================================================================
# MAIN TRAINING
# ============================================================================

def run_training(
    requested_steps: int,
    requested_val_steps: int,
) -> None:
    dev = get_device()

    if dev.type != "cuda":
        raise RuntimeError(
            "CUDA is required for this full-parameter RTX 4050 training run."
        )

    rgb_dir, truth_dir = discover_data_dirs()

    manifest = load_or_build_manifest(
        rgb_dir,
        truth_dir,
    )

    split = create_or_load_validation_split(
        manifest,
    )

    train_items, val_items = split_manifest(
        manifest,
        split,
    )

    print()
    print("=" * 78)
    print("DATASET SPLIT")
    print("=" * 78)
    print(f"Training pairs:   {len(train_items)}")
    print(f"Validation pairs: {len(val_items)}")
    print(f"Train tiles:      {len(split['train_tiles'])}")
    print(f"Validation tiles: {len(split['validation_tiles'])}")
    print(f"Held-out tiles:   {split['validation_tiles']}")

    if not stream_test(train_items):
        raise RuntimeError(
            "DFC2019 Track 1 preflight failed."
        )

    print()
    print("=" * 78)
    print("MODEL INITIALIZATION")
    print("=" * 78)

    if LATEST_CHECKPOINT.exists():
        initial = LATEST_CHECKPOINT
        print(
            "[OK] Existing Stage-3 checkpoint found; "
            "training will resume from it."
        )
    else:
        initial = STAGE2_BEST
        print(
            "[OK] Starting Stage 3 from S-EO best checkpoint."
        )

    model = create_model(initial)
    enable_gradient_checkpointing(model)
    make_all_trainable(model)
    model.to(dev)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print_memory("after model loading")

    optimizer = create_optimizer(model)
    scaler = create_scaler()

    # Resume optimizer/history only when a Stage-3 checkpoint exists.
    if LATEST_CHECKPOINT.exists():
        start_epoch, best_mae, history = load_resume_checkpoint(
            model,
            optimizer,
            scaler,
        )
    else:
        start_epoch = 1
        best_mae = float("inf")

        if HISTORY_FILE.exists():
            try:
                history = json.loads(
                    HISTORY_FILE.read_text(encoding="utf-8")
                )
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []
        else:
            history = []

    print()
    print("=" * 78)
    print("STARTING STAGE 3 FULL-PARAMETER FINE-TUNING")
    print("=" * 78)

    print()
    print("Training chain:")
    print("Depth Anything V2 Large")
    print("        ↓")
    print("GeoNRW")
    print("        ↓")
    print("Potsdam")
    print("        ↓")
    print("Vaihingen")
    print("        ↓")
    print("S-EO")
    print("        ↓")
    print("DFC2019 / US3D Track 1  ← CURRENT STAGE")

    print()
    print("[OK] RGB + AGL local training pairs.")
    print("[OK] No MSI download required.")
    print("[OK] No remote stream.")
    print("[OK] Geographic tile validation split is persistent.")
    print("[OK] Full-parameter tuning: 100%.")
    print("[OK] Automatic Stage-3 resume.")
    print("[OK] Best checkpoint updates only on better validation MAE.")
    print(
        f"[OK] Validation maximum: {requested_val_steps} samples."
    )

    if requested_steps > 0:
        print(
            f"[OK] Training cap: {requested_steps} samples/epoch."
        )
    else:
        print(
            "[OK] Training cap: ALL discovered training pairs per epoch."
        )

    if start_epoch > EPOCHS:
        print()
        print(
            f"[INFO] Stage 3 already completed through epoch "
            f"{start_epoch - 1} under EPOCHS={EPOCHS}."
        )
        return

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            print()
            print("=" * 78)
            print(f"EPOCH {epoch}/{EPOCHS}")
            print("=" * 78)

            epoch_start = time.time()

            train_loss, train_count = train_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                items=train_items,
                dev=dev,
                epoch=epoch,
                max_steps=requested_steps,
            )

            print()
            print(
                f"Epoch {epoch} training loss: {train_loss:.6f}"
            )
            print(
                f"Epoch {epoch} training samples: {train_count}"
            )

            # Short validation. Local, so it should be much faster than S-EO.
            validation = validate(
                model=model,
                items=val_items,
                dev=dev,
                max_samples=requested_val_steps,
            )

            val_mae = validation["mae"]
            val_rmse = validation["rmse"]

            print()
            print("=" * 78)
            print("VALIDATION RESULT")
            print("=" * 78)
            print(f"Validation MAE:  {val_mae:.6f} m")
            print(f"Validation RMSE: {val_rmse:.6f} m")
            print(f"Valid pixels:    {validation['valid_pixels']:,}")
            print(f"Validation samples: {validation['samples']}")
            print(
                f"Epoch elapsed: "
                f"{time.time() - epoch_start:.1f}s"
            )

            record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_samples": int(train_count),
                "validation_mae": float(val_mae),
                "validation_rmse": float(val_rmse),
                "validation_samples": int(validation["samples"]),
                "validation_valid_pixels": int(
                    validation["valid_pixels"]
                ),
                "learning_rate": LEARNING_RATE,
                "dataset": "DFC2019 Track 1",
                "validation_tiles": split["validation_tiles"],
            }

            history.append(record)
            save_history(history)

            # REQUIRED: latest is saved after every successfully completed epoch.
            save_checkpoint(
                path=LATEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_mae=min(best_mae, val_mae),
                history=history,
                reason="completed_epoch",
            )

            # Best is updated ONLY when metric improves.
            if val_mae < best_mae:
                best_mae = val_mae

                save_checkpoint(
                    path=BEST_CHECKPOINT,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_mae=best_mae,
                    history=history,
                    reason="new_best_validation_mae",
                )

                print()
                print("NEW BEST DFC2019 CHECKPOINT")
                print(
                    f"Best validation MAE: "
                    f"{best_mae:.6f} m"
                )
            else:
                print()
                print(
                    f"[INFO] Best checkpoint retained. "
                    f"Best MAE remains {best_mae:.6f} m."
                )

            print_memory(
                f"end of epoch {epoch}"
            )

    except KeyboardInterrupt:
        print()
        print("=" * 78)
        print("TRAINING INTERRUPTED")
        print("=" * 78)

        # Save the current model state without claiming that the interrupted
        # epoch is completed. The emergency file is separate from latest.
        try:
            save_checkpoint(
                path=EMERGENCY_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=max(0, start_epoch - 1),
                best_mae=best_mae,
                history=history,
                reason="keyboard_interrupt",
            )
            print(
                "[OK] Emergency checkpoint saved."
            )
        except Exception as exc:
            print(
                f"[ERROR] Could not save emergency checkpoint: {exc}"
            )

        raise

    finally:
        del model
        del optimizer
        if scaler is not None:
            del scaler

        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("STAGE 3 TRAINING FINISHED")
    print("=" * 78)
    print(f"Latest: {LATEST_CHECKPOINT}")
    print(f"Best:   {BEST_CHECKPOINT}")
    print()
    print("Next curriculum stage:")
    print("AW3D30 + SRTM + Copernicus GLO-30 calibration")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ASTERRA Stage 3 — US3D / DFC2019 Track 1 "
            "full-parameter fine-tuning"
        )
    )

    parser.add_argument(
        "--stream-test",
        action="store_true",
        help="Test local RGB + AGL pairing without training.",
    )

    parser.add_argument(
        "--gpu-smoke-test",
        action="store_true",
        help="Run one real forward/backward optimizer step.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=(
            "Maximum training samples per epoch. "
            "0 means all discovered training pairs."
        ),
    )

    parser.add_argument(
        "--val-steps",
        type=int,
        default=DEFAULT_VAL_STEPS,
        help="Maximum validation samples per epoch.",
    )

    return parser.parse_args()


def main() -> None:
    seed_everything()

    args = parse_args()

    if args.steps < 0:
        raise ValueError("--steps must be >= 0")

    if args.val_steps <= 0:
        raise ValueError("--val-steps must be > 0")

    print_header()

    rgb_dir, truth_dir = discover_data_dirs()

    manifest = load_or_build_manifest(
        rgb_dir,
        truth_dir,
    )

    split = create_or_load_validation_split(
        manifest,
    )

    train_items, val_items = split_manifest(
        manifest,
        split,
    )

    if args.stream_test:
        ok = stream_test(train_items, count=5)
        raise SystemExit(0 if ok else 1)

    if args.gpu_smoke_test:
        ok = gpu_smoke_test(train_items)
        raise SystemExit(0 if ok else 1)

    run_training(
        requested_steps=args.steps,
        requested_val_steps=args.val_steps,
    )


if __name__ == "__main__":
    main()