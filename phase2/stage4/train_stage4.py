"""
ASTERRA AI — STAGE 4
DFC2019 AGL + AW3D30 / SRTMGL1 / COP-DEM GLO-30
REGIONAL CALIBRATION — NON-GEOREFERENCED DFC2019 SAFE VERSION

IMPORTANT
---------
The local DFC2019 Track-1 RGB TIFFs were verified to have:
    CRS = None
    identity transform

Therefore this trainer NEVER tries to reproject or pixel-align those RGB
images with AW3D30/SRTMGL1/COP-DEM.

Stage 4 therefore uses:
    RGB + DFC2019 AGL       = pixel-aligned supervised signal
    3 DEM products          = regional distribution calibration prior

The DEMs are NOT used as pixel-level labels.

Commands:
    python .\phase2\stage4\train_stage4.py --preflight
    python .\phase2\stage4\train_stage4.py --gpu-smoke-test
    python .\phase2\stage4\train_stage4.py --steps 10 --val-steps 3 --epochs 1
    python .\phase2\stage4\train_stage4.py
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
except ImportError as exc:
    raise RuntimeError("rasterio is required: pip install rasterio") from exc


# ============================================================================
# PATHS
# ============================================================================

THIS_FILE = Path(__file__).resolve()
ROOT = (
    THIS_FILE.parents[2]
    if THIS_FILE.parent.name.lower() == "stage4"
    else Path(r"D:\Asterra AI")
)

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2

STAGE3_BEST = ROOT / "models" / "asterra_dfc2019" / "dfc2019_best.pth"

OUTPUT_DIR = ROOT / "models" / "asterra_stage4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST = OUTPUT_DIR / "stage4_latest.pth"
BEST = OUTPUT_DIR / "stage4_best.pth"
EMERGENCY = OUTPUT_DIR / "stage4_emergency.pth"
HISTORY = OUTPUT_DIR / "stage4_training_history.json"
MANIFEST = OUTPUT_DIR / "stage4_manifest.json"
DEM_STATS = OUTPUT_DIR / "stage4_dem_regional_stats.json"

DATA = ROOT / "datasets"
DFC = DATA / "DFC2019"

AW3D = {
    "JAX": DATA / "AW3D30" / "JAX" / "AW3D30_JAX.tif",
    "OMA": DATA / "AW3D30" / "OMA" / "AW3D30_OMA.tif",
}
SRTM = {
    "JAX": DATA / "SRTM_JAX_test" / "rasters_SRTMGL1" / "output_SRTMGL1.tif",
    "OMA": DATA / "SRTM_OMA_test" / "rasters_SRTMGL1" / "output_SRTMGL1.tif",
}
COP = {
    "JAX": DATA / "COPDEM_JAX_test",
    "OMA": DATA / "COPDEM_OMA_test",
}


# ============================================================================
# MODEL / TRAINING CONFIG
# ============================================================================

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

# Depth Anything V2 Large accepts 518-sized model input in this project.
MODEL_SIZE = 518
PATCH_SIZE = 512

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

EPOCHS = 3
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

SEED = 42
PRINT_EVERY = 10
VAL_FRACTION = 0.10

# DFC AGL remains the dominant supervised objective.
AGL_PRIMARY_WEIGHT = 1.0

# DEMs are only a weak regional prior.
DEM_CALIBRATION_WEIGHT = 0.10

AGL_MIN = 0.0
AGL_MAX = 500.0
TARGET_MAX = 150.0
MIN_VALID_FRACTION = 0.50


# ============================================================================
# UTILITIES
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
        raise RuntimeError("CUDA is required for Stage-4 training.")
    return torch.device("cuda")


def atomic_json_write(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def print_header() -> None:
    print("=" * 78)
    print("ASTERRA — STAGE 4")
    print("DFC2019 AGL + AW3D30/SRTMGL1/COP-DEM REGIONAL CALIBRATION")
    print("=" * 78)
    print(f"Root: {ROOT}")
    print(f"Stage-3 checkpoint: {STAGE3_BEST}")
    print(f"Output: {OUTPUT_DIR}")

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {p.total_memory / 1024**3:.2f} GB")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("GPU: CUDA NOT AVAILABLE")


# ============================================================================
# MODEL
# ============================================================================

def create_model(checkpoint: Path) -> nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage-3 checkpoint not found: {checkpoint}")

    # IMPORTANT:
    # DepthAnythingV2 in the installed repository does NOT accept max_depth.
    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain model_state_dict.")

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(
            "Stage-3 checkpoint is incompatible with Stage-4 architecture.\n"
            f"Missing keys: {missing[:20]}"
        )

    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {len(unexpected)}")

    print(
        f"[OK] Loaded Stage-3 best: "
        f"epoch={payload.get('epoch')} | "
        f"best_val_mae={payload.get('best_val_mae')}"
    )

    return model


def enable_gradient_checkpointing(model: nn.Module) -> None:
    for obj in (model, getattr(model, "pretrained", None)):
        if obj is None:
            continue

        fn = getattr(obj, "gradient_checkpointing_enable", None)

        if callable(fn):
            try:
                fn()
                print("[OK] Gradient checkpointing enabled.")
                return
            except Exception as exc:
                print(f"[INFO] Gradient checkpointing unavailable: {exc}")

    print("[INFO] Model did not expose gradient_checkpointing_enable().")


def make_trainable(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def model_forward(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
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


# ============================================================================
# DFC2019 DISCOVERY
# ============================================================================

def find_named_dir(root: Path, names: set[str]) -> Optional[Path]:
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in names:
            return p
    return None


def discover_dfc_dirs() -> Tuple[Path, Path]:
    rgb = find_named_dir(
        DFC,
        {
            "train-track1-rgb",
            "train_track1_rgb",
            "track1-rgb",
            "track1_rgb",
        },
    )

    truth = find_named_dir(
        DFC,
        {
            "train-track1-truth",
            "train_track1_truth",
            "track1-truth",
            "track1_truth",
        },
    )

    if rgb is None or truth is None:
        raise RuntimeError(
            "Could not locate DFC2019 Track-1 RGB/Truth directories under:\n"
            f"{DFC}"
        )

    return rgb, truth


def build_manifest() -> List[Dict[str, str]]:
    rgb_dir, truth_dir = discover_dfc_dirs()

    rgb_files = sorted(
        [
            p
            for p in rgb_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".tif", ".tiff"}
            and p.stem.upper().endswith("_RGB")
        ],
        key=lambda p: p.name.lower(),
    )

    agl: Dict[str, Path] = {}
    cls: Dict[str, Path] = {}

    for p in truth_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".tif", ".tiff"}:
            continue

        u = p.stem.upper()

        if u.endswith("_AGL"):
            agl[p.stem[:-4].upper()] = p

        elif u.endswith("_CLS"):
            cls[p.stem[:-4].upper()] = p

    pairs: List[Dict[str, str]] = []

    for p in rgb_files:
        stem = p.stem[:-4]
        key = stem.upper()

        if key not in agl:
            continue

        parts = stem.split("_")
        tile = "_".join(parts[:2]) if len(parts) >= 2 else stem

        if stem.upper().startswith("JAX_"):
            region = "JAX"
        elif stem.upper().startswith("OMA_"):
            region = "OMA"
        else:
            region = "UNKNOWN"

        pairs.append(
            {
                "id": stem,
                "tile": tile,
                "region": region,
                "rgb": str(p),
                "agl": str(agl[key]),
                "cls": str(cls[key]) if key in cls else "",
            }
        )

    if not pairs:
        raise RuntimeError("No DFC2019 RGB + AGL pairs found.")

    print(f"[OK] DFC2019 pairs: {len(pairs)}")
    print(f"[OK] Geographic tiles: {len({x['tile'] for x in pairs})}")

    return pairs


def load_manifest() -> List[Dict[str, str]]:
    if MANIFEST.exists():
        try:
            payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
            pairs = payload.get("pairs")

            if isinstance(pairs, list) and pairs:
                valid = all(
                    Path(x["rgb"]).exists()
                    and Path(x["agl"]).exists()
                    for x in pairs
                )

                if valid:
                    print(f"[OK] Reusing Stage-4 manifest: {MANIFEST}")
                    return pairs
        except Exception:
            pass

    pairs = build_manifest()

    atomic_json_write(
        MANIFEST,
        {
            "version": 3,
            "stage": 4,
            "geolocation_status": (
                "DFC2019 RGB TIFFs are non-georeferenced; "
                "no pixel alignment with DEMs is attempted."
            ),
            "pairs": pairs,
        },
    )

    print(f"[OK] Stage-4 manifest saved: {MANIFEST}")
    return pairs


# ============================================================================
# SPLIT
# ============================================================================

def split_items(
    items: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    tiles = sorted({x["tile"] for x in items})

    ranked = sorted(
        tiles,
        key=lambda t: hashlib.sha1(
            f"{SEED}:{t}".encode("utf-8")
        ).hexdigest(),
    )

    val_count = max(1, int(round(len(ranked) * VAL_FRACTION)))
    val_count = min(val_count, max(1, len(ranked) - 1))

    val_tiles = set(ranked[:val_count])

    train = [x for x in items if x["tile"] not in val_tiles]
    val = [x for x in items if x["tile"] in val_tiles]

    return train, val, sorted(val_tiles)


# ============================================================================
# TIFF READING
# ============================================================================

def read_window(
    path: Path,
    x: int,
    y: int,
    size: int,
    band: Optional[int] = None,
) -> np.ndarray:
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(x, y, size, size)

        if band is None:
            arr = src.read(window=window)
        else:
            arr = src.read(band, window=window)

    return np.asarray(arr)


def raster_shape(path: Path) -> Tuple[int, int]:
    with rasterio.open(path) as src:
        return int(src.height), int(src.width)


def read_rgb_patch(path: Path, x: int, y: int) -> np.ndarray:
    arr = read_window(path, x, y, PATCH_SIZE, band=None)

    if arr.ndim != 3 or arr.shape[0] < 3:
        raise RuntimeError(
            f"RGB must contain >=3 bands; got {arr.shape}: {path}"
        )

    arr = arr[:3].astype(np.float32)
    arr = np.transpose(arr, (1, 2, 0))

    finite = np.isfinite(arr)
    if not finite.any():
        raise RuntimeError(f"RGB contains no finite pixels: {path}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)

    # DFC2019 RGB is uint8 in the local corpus.
    if float(np.nanmax(arr)) > 1.5:
        arr /= 255.0

    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0))


def read_agl_patch(path: Path, x: int, y: int) -> np.ndarray:
    arr = read_window(path, x, y, PATCH_SIZE, band=1).astype(np.float32)

    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < AGL_MIN) | (arr > AGL_MAX)] = np.nan

    return np.ascontiguousarray(arr)


def read_cls_patch(path: Path, x: int, y: int) -> Optional[np.ndarray]:
    if not path:
        return None

    p = Path(path)

    if not p.exists():
        return None

    try:
        arr = read_window(p, x, y, PATCH_SIZE, band=1)
        return np.asarray(arr)
    except Exception:
        return None


# ============================================================================
# DEM REGIONAL STATISTICS
# ============================================================================

def cop_tiles(region: str) -> List[Path]:
    return sorted(
        p
        for p in COP[region].rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
    )


def dem_sources(region: str) -> List[Tuple[str, Path]]:
    return [
        ("AW3D30", AW3D[region]),
        ("SRTMGL1", SRTM[region]),
        *[
            (f"COPDEM_{i+1}", p)
            for i, p in enumerate(cop_tiles(region))
        ],
    ]


def robust_relative_values(
    path: Path,
    max_values: int = 250_000,
) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, out_dtype="float32")
        nodata = src.nodata

    arr = np.asarray(arr, dtype=np.float32)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    arr[~np.isfinite(arr)] = np.nan

    values = arr[np.isfinite(arr)]

    if values.size == 0:
        return np.empty(0, dtype=np.float32)

    if values.size > max_values:
        rng = np.random.default_rng(SEED)
        values = rng.choice(
            values,
            size=max_values,
            replace=False,
        )

    baseline = float(np.percentile(values, 10.0))
    relative = values - baseline

    relative = relative[
        (relative >= TARGET_MIN)
        & (relative <= TARGET_MAX)
    ]

    return relative.astype(np.float32)


def build_dem_region_stats(region: str) -> Dict[str, Any]:
    all_values: List[np.ndarray] = []
    source_stats: Dict[str, Any] = {}

    for name, path in dem_sources(region):
        if not path.exists():
            print(f"[WARN] Missing DEM source: {path}")
            continue

        try:
            values = robust_relative_values(path)

            if values.size < 100:
                print(
                    f"[WARN] Too few valid DEM values: {path}"
                )
                continue

            q = np.percentile(
                values,
                [10, 25, 50, 75, 90],
            )

            source_stats[name] = {
                "path": str(path),
                "samples": int(values.size),
                "q10": float(q[0]),
                "q25": float(q[1]),
                "q50": float(q[2]),
                "q75": float(q[3]),
                "q90": float(q[4]),
            }

            all_values.append(values)

        except Exception as exc:
            print(
                f"[WARN] Could not read DEM {path}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not all_values:
        raise RuntimeError(
            f"No usable DEM sources for region {region}."
        )

    fused = np.concatenate(all_values)

    q = np.percentile(
        fused,
        [10, 25, 50, 75, 90],
    )

    return {
        "region": region,
        "q10": float(q[0]),
        "q25": float(q[1]),
        "q50": float(q[2]),
        "q75": float(q[3]),
        "q90": float(q[4]),
        "sources": source_stats,
        "method": (
            "regional robust relative-height distribution; "
            "no pixel alignment"
        ),
    }


def load_dem_stats() -> Dict[str, Any]:
    if DEM_STATS.exists():
        try:
            data = json.loads(
                DEM_STATS.read_text(encoding="utf-8")
            )

            if all(r in data for r in ("JAX", "OMA")):
                print(
                    f"[OK] Reusing DEM regional statistics: "
                    f"{DEM_STATS}"
                )
                return data
        except Exception:
            pass

    data = {
        "JAX": build_dem_region_stats("JAX"),
        "OMA": build_dem_region_stats("OMA"),
    }

    atomic_json_write(DEM_STATS, data)

    print(
        f"[OK] DEM regional statistics saved: {DEM_STATS}"
    )

    return data


# ============================================================================
# SAMPLE
# ============================================================================

def crop_origin(
    width: int,
    height: int,
    training: bool,
) -> Tuple[int, int]:
    if width < PATCH_SIZE or height < PATCH_SIZE:
        raise RuntimeError(
            f"Image too small: {width}x{height}"
        )

    if training:
        return (
            random.randint(0, width - PATCH_SIZE),
            random.randint(0, height - PATCH_SIZE),
        )

    return (
        (width - PATCH_SIZE) // 2,
        (height - PATCH_SIZE) // 2,
    )


def get_sample(
    item: Dict[str, str],
    dem_stats: Dict[str, Any],
    training: bool,
) -> Optional[Dict[str, Any]]:
    try:
        rgb_path = Path(item["rgb"])
        agl_path = Path(item["agl"])

        region = item.get("region", "")
        if region not in ("JAX", "OMA"):
            region = (
                "JAX"
                if item["id"].upper().startswith("JAX_")
                else "OMA"
            )

        height, width = raster_shape(rgb_path)

        x, y = crop_origin(
            width,
            height,
            training,
        )

        rgb = read_rgb_patch(
            rgb_path,
            x,
            y,
        )

        agl = read_agl_patch(
            agl_path,
            x,
            y,
        )

        agl_mask = np.isfinite(agl)

        # Optional DFC class mask.
        # DFC2019 class 65 is unlabeled/background-invalid in the
        # Stage-3 trainer, so exclude it when the CLS file is available.
        cls = read_cls_patch(
            item.get("cls", ""),
            x,
            y,
        )

        if cls is not None:
            agl_mask &= cls != 65

        valid_fraction = float(agl_mask.mean())

        if valid_fraction < MIN_VALID_FRACTION:
            return None

        # The DEMs are NOT read for this crop.
        # Their only role is the regional distribution prior.
        stats = dem_stats[region]

        dem_quantiles = np.array(
            [
                stats["q10"],
                stats["q25"],
                stats["q50"],
                stats["q75"],
                stats["q90"],
            ],
            dtype=np.float32,
        )

        image = torch.from_numpy(
            np.transpose(rgb, (2, 0, 1)).copy()
        ).float()

        target = torch.from_numpy(
            np.nan_to_num(
                agl,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).copy()
        ).float().unsqueeze(0)

        mask = torch.from_numpy(
            agl_mask.copy()
        ).float().unsqueeze(0)

        return {
            "id": item["id"],
            "tile": item["tile"],
            "region": region,
            "image": image,
            "target": target,
            "mask": mask,
            "valid_fraction": valid_fraction,
            "dem_quantiles": torch.from_numpy(
                dem_quantiles
            ).float(),
        }

    except Exception as exc:
        print(
            f"[WARN] Sample failed: {item['id']} | "
            f"{type(exc).__name__}: {exc}"
        )
        return None


# ============================================================================
# LOSSES
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

    target = F.interpolate(
        target.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    # CRITICAL FIX:
    # mask must have EXACTLY the same spatial shape as pred before indexing.
    mask = F.interpolate(
        mask.float(),
        size=size,
        mode="nearest",
    )

    return target, mask


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

    return torch.abs(
        pred.float() - target.float()
    )[valid].mean()


def distribution_calibration_loss(
    pred: torch.Tensor,
    sample: Dict[str, Any],
) -> torch.Tensor:
    """
    Weak REGIONAL calibration.

    No DEM pixel is compared with a model pixel.

    Instead:
      - sample prediction values from the current DFC crop
      - compare selected prediction quantiles against the independent
        regional DEM relative-height distribution.

    The loss is deliberately weak because DEM products are coarse elevation
    references, not DFC building-height labels.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)

    values = pred.float().reshape(-1)
    values = values[
        torch.isfinite(values)
        & (values >= 0.0)
        & (values <= AGL_MAX)
    ]

    if values.numel() < 64:
        return pred.sum() * 0.0

    # Keep computation small and deterministic enough for training.
    if values.numel() > 8192:
        idx = torch.linspace(
            0,
            values.numel() - 1,
            8192,
            device=values.device,
        ).long()
        values = values.sort().values[idx]
    else:
        values = values.sort().values

    pred_q = torch.quantile(
        values,
        torch.tensor(
            [0.10, 0.50, 0.90],
            device=values.device,
        ),
    )

    dem_q = sample["dem_quantiles"].to(
        pred.device,
        non_blocking=True,
    )[[0, 2, 4]].float()

    # DEM regional relative-height values and DFC AGL are related but not
    # identical quantities. Keep this objective weak.
    return F.smooth_l1_loss(
        pred_q,
        dem_q,
    )


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


def make_optimizer(model: nn.Module):
    try:
        from transformers.optimization import Adafactor

        print("[OK] Optimizer: Adafactor")

        return Adafactor(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    except Exception:
        print("[OK] Optimizer: AdamW fallback")

        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )


# ============================================================================
# CHECKPOINTS
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
        "stage": 4,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": float(best_mae),
        "history": history,
        "reason": reason,
        "dataset": (
            "DFC2019 AGL + AW3D30/SRTMGL1/"
            "COP-DEM GLO-30 regional calibration"
        ),
        "initial_checkpoint": str(STAGE3_BEST),
        "patch_size": PATCH_SIZE,
        "model_size": MODEL_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_accumulation": GRAD_ACCUMULATION,
        "full_parameter_tuning": True,
        "dem_calibration_weight": DEM_CALIBRATION_WEIGHT,
        "dem_alignment": (
            "regional_distribution_only; "
            "DFC2019 RGB TIFFs have no georeferencing"
        ),
        "seed": SEED,
    }

    try:
        payload["scaler_state_dict"] = scaler.state_dict()
    except Exception:
        payload["scaler_state_dict"] = None

    atomic_torch_save(payload, path)


# ============================================================================
# PREFLIGHT
# ============================================================================

def preflight(
    items: List[Dict[str, str]],
    dem_stats: Dict[str, Any],
) -> bool:
    print()
    print("=" * 78)
    print("STAGE 4 PREFLIGHT — NON-GEOREFERENCED SAFE MODE")
    print("=" * 78)

    required = [
        STAGE3_BEST,
        AW3D["JAX"],
        AW3D["OMA"],
        SRTM["JAX"],
        SRTM["OMA"],
    ]

    for p in required:
        print(
            f"{'[OK]' if p.exists() else '[MISSING]'} {p}"
        )

    cop_jax = cop_tiles("JAX")
    cop_oma = cop_tiles("OMA")

    print(f"COP-DEM JAX tiles: {len(cop_jax)}")
    print(f"COP-DEM OMA tiles: {len(cop_oma)}")

    if not STAGE3_BEST.exists():
        return False

    if not all(p.exists() for p in required):
        return False

    if not cop_jax or not cop_oma:
        return False

    print()
    print("[OK] Required Stage-4 data/checkpoint paths exist.")

    # Verify the known local DFC property without treating it as an error.
    first = items[0]

    try:
        with rasterio.open(first["rgb"]) as src:
            print(
                f"[INFO] RGB georef: CRS={src.crs}, "
                f"identity={src.transform.is_identity}"
            )

            if src.crs is None and src.transform.is_identity:
                print(
                    "[OK] RGB is non-georeferenced. "
                    "No DEM reprojection/alignment will be attempted."
                )
            else:
                print(
                    "[INFO] RGB has georeferencing, but this Stage-4 "
                    "version still uses regional DEM calibration only."
                )

    except Exception as exc:
        print(
            f"[ERROR] Could not inspect RGB georeferencing: {exc}"
        )
        return False

    train, val, val_tiles = split_items(items)

    print(f"Train items: {len(train)}")
    print(f"Validation items: {len(val)}")
    print(f"Validation tiles: {val_tiles}")

    successes = 0

    # Deterministic samples. Do not require DEM pixel alignment.
    for item in train[:25] + val[:10]:
        sample = get_sample(
            item,
            dem_stats,
            training=False,
        )

        if sample is None:
            continue

        print(
            f"[OK] Sample {sample['id']} | "
            f"region={sample['region']} | "
            f"AGL-valid={sample['valid_fraction']:.3f} | "
            f"DEM-q50={float(sample['dem_quantiles'][2]):.2f}"
        )

        successes += 1

        if successes >= 3:
            break

    if successes < 1:
        print(
            "[ERROR] Could not construct a valid "
            "RGB + DFC2019 AGL sample."
        )
        return False

    print()
    print("[OK] Stage-4 preflight passed.")
    print(
        "[IMPORTANT] AW3D30/SRTMGL1/COP-DEM are regional "
        "calibration priors, NOT pixel-aligned labels."
    )

    return True


# ============================================================================
# GPU SMOKE TEST
# ============================================================================

def gpu_smoke_test(
    items: List[Dict[str, str]],
    dem_stats: Dict[str, Any],
) -> bool:
    print()
    print("=" * 78)
    print("STAGE 4 GPU SMOKE TEST")
    print("=" * 78)

    dev = get_device()

    sample = None

    for item in items[:50]:
        sample = get_sample(
            item,
            dem_stats,
            training=False,
        )

        if sample is not None:
            break

    if sample is None:
        print("[ERROR] No valid Stage-4 sample.")
        return False

    model = None
    optimizer = None
    scaler = None

    try:
        model = create_model(STAGE3_BEST)
        enable_gradient_checkpointing(model)
        make_trainable(model)
        model.to(dev)

        optimizer = make_optimizer(model)
        scaler = make_scaler()

        image = sample["image"].unsqueeze(0).to(dev)
        target = sample["target"].unsqueeze(0).to(dev)
        mask = sample["mask"].unsqueeze(0).to(dev)

        optimizer.zero_grad(set_to_none=True)

        with amp_context():
            image_model = F.interpolate(
                image,
                size=(MODEL_SIZE, MODEL_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            pred = model_forward(
                model,
                image_model,
            )

            target_r, mask_r = resize_target_mask(
                target,
                mask,
                pred.shape[-2:],
            )

            loss_agl = masked_l1(
                pred,
                target_r,
                mask_r,
            )

            loss_dem = distribution_calibration_loss(
                pred,
                sample,
            )

            loss = (
                AGL_PRIMARY_WEIGHT * loss_agl
                + DEM_CALIBRATION_WEIGHT * loss_dem
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite smoke loss: {float(loss.detach().cpu())}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

        print(
            f"[OK] Smoke sample: {sample['id']} | "
            f"region={sample['region']}"
        )
        print(
            f"[OK] AGL loss: "
            f"{float(loss_agl.detach().cpu()):.6f}"
        )
        print(
            f"[OK] DEM calibration loss: "
            f"{float(loss_dem.detach().cpu()):.6f}"
        )
        print(
            f"[OK] Total loss: "
            f"{float(loss.detach().cpu()):.6f}"
        )
        print(
            "[OK] Forward + backward + optimizer step succeeded."
        )

        return True

    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM during smoke test.")
        return False

    except Exception as exc:
        print(
            f"[ERROR] Smoke test failed: "
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
# TRAINING
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    optimizer,
    scaler,
    items: List[Dict[str, str]],
    dem_stats: Dict[str, Any],
    dev: torch.device,
    epoch: int,
    max_steps: int,
) -> Tuple[float, int]:
    order = list(items)
    random.shuffle(order)

    if max_steps > 0:
        order = order[:max_steps]

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    count = 0
    accumulation = 0

    for step, item in enumerate(order, 1):
        sample = get_sample(
            item,
            dem_stats,
            training=True,
        )

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
            with amp_context():
                image_model = F.interpolate(
                    image,
                    size=(MODEL_SIZE, MODEL_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )

                pred = model_forward(
                    model,
                    image_model,
                )

                target_r, mask_r = resize_target_mask(
                    target,
                    mask,
                    pred.shape[-2:],
                )

                loss_agl = masked_l1(
                    pred,
                    target_r,
                    mask_r,
                )

                loss_dem = distribution_calibration_loss(
                    pred,
                    sample,
                )

                loss = (
                    AGL_PRIMARY_WEIGHT * loss_agl
                    + DEM_CALIBRATION_WEIGHT * loss_dem
                )

                if not torch.isfinite(loss):
                    print(
                        f"[WARN] Non-finite loss at "
                        f"{sample['id']}; skipping."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accumulation = 0
                    continue

                scaled_loss = (
                    loss / GRAD_ACCUMULATION
                )

            scaler.scale(
                scaled_loss
            ).backward()

            accumulation += 1

            if (
                accumulation >= GRAD_ACCUMULATION
                or step == len(order)
            ):
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                scaler.step(optimizer)
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
                    f"loss={float(loss.detach().cpu()):.5f} "
                    f"agl={float(loss_agl.detach().cpu()):.5f} "
                    f"dem_cal={float(loss_dem.detach().cpu()):.5f}"
                )

        except torch.cuda.OutOfMemoryError:
            print(
                f"[WARN] CUDA OOM at {sample['id']}; "
                "clearing cache and skipping sample."
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulation = 0
            torch.cuda.empty_cache()

        finally:
            del image
            del target
            del mask

    # Flush any partial accumulation safely.
    if accumulation > 0:
        try:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )
        except Exception as exc:
            print(
                f"[WARN] Final accumulation flush failed: {exc}"
            )
            optimizer.zero_grad(
                set_to_none=True
            )

    return (
        total_loss / max(1, count),
        count,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    items: List[Dict[str, str]],
    dem_stats: Dict[str, Any],
    dev: torch.device,
    max_samples: int,
) -> Dict[str, float]:
    total_abs = 0.0
    total_sq = 0.0
    pixels = 0
    samples = 0

    for item in items[:max_samples]:
        sample = get_sample(
            item,
            dem_stats,
            training=False,
        )

        if sample is None:
            continue

        image = sample["image"].unsqueeze(0).to(dev)
        target = sample["target"].unsqueeze(0).to(dev)
        mask = sample["mask"].unsqueeze(0).to(dev)

        try:
            with amp_context():
                image_model = F.interpolate(
                    image,
                    size=(MODEL_SIZE, MODEL_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )

                pred = model_forward(
                    model,
                    image_model,
                )

                target_r, mask_r = resize_target_mask(
                    target,
                    mask,
                    pred.shape[-2:],
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
                err = (
                    pred - target_r
                )[valid]

                total_abs += float(
                    torch.abs(err).sum().cpu()
                )

                total_sq += float(
                    (err * err).sum().cpu()
                )

                pixels += int(
                    valid.sum().item()
                )

                samples += 1

        finally:
            del image
            del target
            del mask

    if pixels == 0:
        return {
            "mae": float("inf"),
            "rmse": float("inf"),
            "samples": 0,
            "pixels": 0,
        }

    return {
        "mae": total_abs / pixels,
        "rmse": math.sqrt(
            total_sq / pixels
        ),
        "samples": samples,
        "pixels": pixels,
    }


def run_training(
    items: List[Dict[str, str]],
    dem_stats: Dict[str, Any],
    steps: int,
    val_steps: int,
    epochs: int,
) -> None:
    dev = get_device()

    train_items, val_items, val_tiles = split_items(
        items
    )

    if not preflight(
        items,
        dem_stats,
    ):
        raise RuntimeError(
            "Stage-4 preflight failed."
        )

    model = create_model(
        STAGE3_BEST
    )

    enable_gradient_checkpointing(
        model
    )

    make_trainable(model)
    model.to(dev)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer = make_optimizer(
        model
    )

    scaler = make_scaler()

    history: List[Dict[str, Any]] = []
    best_mae = float("inf")

    print()
    print("=" * 78)
    print("STARTING ACTUAL STAGE-4 FINE-TUNING")
    print("=" * 78)

    print("Stage-3 best")
    print("      ↓")
    print("DFC2019 AGL supervised fine-tuning")
    print("      ↓")
    print("Stage-4 regional DEM calibration")
    print("      ↓")
    print("AW3D30 + SRTMGL1 + COP-DEM GLO-30")
    print()

    print("[OK] Full-parameter tuning: 100%")
    print(f"[OK] LR: {LEARNING_RATE}")
    print(f"[OK] Effective batch: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"[OK] Epochs: {epochs}")
    print(f"[OK] Train samples/epoch: {'ALL' if steps == 0 else steps}")
    print(f"[OK] Validation samples/epoch: {val_steps}")
    print(f"[OK] Validation tiles: {val_tiles}")
    print(
        "[OK] DEMs are regional priors only; "
        "no fake RGB↔DEM pixel alignment."
    )

    try:
        for epoch in range(1, epochs + 1):
            print()
            print("=" * 78)
            print(f"STAGE 4 EPOCH {epoch}/{epochs}")
            print("=" * 78)

            start = time.time()

            train_loss, train_count = train_one_epoch(
                model,
                optimizer,
                scaler,
                train_items,
                dem_stats,
                dev,
                epoch,
                steps,
            )

            metrics = validate(
                model,
                val_items,
                dem_stats,
                dev,
                val_steps,
            )

            record = {
                "stage": 4,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_samples": int(train_count),
                "validation_mae": float(metrics["mae"]),
                "validation_rmse": float(metrics["rmse"]),
                "validation_samples": int(metrics["samples"]),
                "validation_pixels": int(metrics["pixels"]),
                "learning_rate": LEARNING_RATE,
                "elapsed_seconds": float(
                    time.time() - start
                ),
                "dem_calibration_weight": DEM_CALIBRATION_WEIGHT,
                "dem_alignment": "regional_distribution_only",
            }

            history.append(record)

            atomic_json_write(
                HISTORY,
                history,
            )

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

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]

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

                print(
                    f"[OK] New Stage-4 best MAE: "
                    f"{best_mae:.6f}"
                )

            print()
            print(
                f"Epoch {epoch} complete | "
                f"train_loss={train_loss:.6f} | "
                f"val_MAE={metrics['mae']:.6f} | "
                f"val_RMSE={metrics['rmse']:.6f}"
            )

            print(
                f"[OK] Latest checkpoint: {LATEST}"
            )

    except KeyboardInterrupt:
        print()
        print(
            "[WARNING] Training interrupted. "
            "Saving emergency checkpoint..."
        )

        try:
            save_checkpoint(
                EMERGENCY,
                model,
                optimizer,
                scaler,
                max(0, len(history)),
                best_mae,
                history,
                "keyboard_interrupt",
            )

            print(
                f"[OK] Emergency checkpoint: {EMERGENCY}"
            )

        finally:
            raise

    finally:
        del model
        del optimizer
        del scaler

        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("STAGE 4 TRAINING FINISHED")
    print("=" * 78)
    print(f"Latest: {LATEST}")
    print(f"Best:   {BEST}")
    print(f"History: {HISTORY}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ASTERRA Stage 4 — DFC2019 AGL + "
            "regional AW3D30/SRTMGL1/COP-DEM calibration"
        )
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run Stage-4 data/sample preflight only.",
    )

    parser.add_argument(
        "--gpu-smoke-test",
        action="store_true",
        help="Run one real GPU forward/backward/optimizer step.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Maximum training samples per epoch. 0 = all.",
    )

    parser.add_argument(
        "--val-steps",
        type=int,
        default=10,
        help="Maximum validation samples per epoch.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of Stage-4 epochs.",
    )

    return parser.parse_args()


def main() -> None:
    seed_everything()

    args = parse_args()

    if args.steps < 0:
        raise ValueError("--steps must be >= 0")

    if args.val_steps <= 0:
        raise ValueError("--val-steps must be > 0")

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    print_header()

    items = load_manifest()

    # Build/load regional DEM statistics once.
    dem_stats = load_dem_stats()

    if args.preflight:
        ok = preflight(
            items,
            dem_stats,
        )
        raise SystemExit(0 if ok else 1)

    if args.gpu_smoke_test:
        ok = gpu_smoke_test(
            items,
            dem_stats,
        )
        raise SystemExit(0 if ok else 1)

    run_training(
        items,
        dem_stats,
        args.steps,
        args.val_steps,
        args.epochs,
    )


if __name__ == "__main__":
    main()