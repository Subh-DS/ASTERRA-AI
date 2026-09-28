r"""
ASTERRA AI — STAGE 5
SPACENET MVS RPC-FRAME-AWARE ELEVATION FINE-TUNING
UPDATED / FIXED VERSION

Why this version replaces the previous Stage-5 geometry search
---------------------------------------------------------------
The local SpaceNet MVS scene TIFFs are 2001x2001, single-band uint8,
and have no CRS/geotransform/RPC metadata. The RPC information is stored
in a sidecar TXT file.

The previous implementation tried to infer the local->RPC pixel offset from
the last two numeric tail values alone. That produced a dangerous failure
mode:

    RPC footprint overlap = 1.0
    elevation valid fraction = 0.0

and therefore every sample was rejected.

This version uses the four geographic footprint values in the RPC sidecar:

    lon_min, lat_min, lon_max, lat_max

as an independent geometric constraint. It numerically fits the unknown
local-TIFF -> RPC-frame translation (and nominal RPC height) so that the
RPC-projected image footprint agrees with the supplied geographic footprint.
The fitted solution is then checked against the actual EPSG:32721 elevation
mosaic. Tail pixel hints remain only as fallback candidates.

Training is NEVER started unless at least one real image + RPC + elevation
sample is constructed.

Commands
--------
    python .\phase2\stage5\train_stage5.py --preflight
    python .\phase2\stage5\train_stage5.py --gpu-smoke-test
    python .\phase2\stage5\train_stage5.py --steps 5 --val-steps 2 --epochs 1
    python .\phase2\stage5\train_stage5.py

The script automatically resumes from:
    models\asterra_stage5\stage5_latest.pth

otherwise it starts from:
    models\asterra_stage4\stage4_best.pth
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import rasterio
    from rasterio.rpc import RPC
    from rasterio.transform import RPCTransformer
    from rasterio.windows import Window
except ImportError as exc:
    raise RuntimeError(
        "rasterio is required. Install with: python -m pip install rasterio"
    ) from exc

try:
    from pyproj import Transformer
except ImportError as exc:
    raise RuntimeError(
        "pyproj is required. Install with: python -m pip install pyproj"
    ) from exc

try:
    from scipy.optimize import least_squares
except ImportError as exc:
    raise RuntimeError(
        "scipy is required for RPC footprint fitting. "
        "Install with: python -m pip install scipy"
    ) from exc


# ============================================================================
# PROJECT
# ============================================================================

THIS_FILE = Path(__file__).resolve()
ROOT = (
    THIS_FILE.parents[2]
    if THIS_FILE.parent.name.lower() == "stage5"
    else Path(r"D:\Asterra AI")
)

DATASET_ROOT = ROOT / "datasets" / "SpaceNet_MVS"
DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"

if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================================
# CHECKPOINTS
# ============================================================================

STAGE4_BEST = ROOT / "models" / "asterra_stage4" / "stage4_best.pth"

OUTPUT_DIR = ROOT / "models" / "asterra_stage5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT = OUTPUT_DIR / "stage5_latest.pth"
BEST_CHECKPOINT = OUTPUT_DIR / "stage5_best.pth"
EMERGENCY_CHECKPOINT = OUTPUT_DIR / "stage5_emergency.pth"
MANIFEST_FILE = OUTPUT_DIR / "stage5_manifest.json"
HISTORY_FILE = OUTPUT_DIR / "stage5_training_history.json"
VALIDATION_SPLIT_FILE = OUTPUT_DIR / "stage5_validation_split.json"


# ============================================================================
# CONFIG
# ============================================================================

PATCH_SIZE = 512
MODEL_SIZE = 518
DINO_PATCH_SIZE = 14

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
EPOCHS = 3

LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

AMP_ENABLED = True
GRADIENT_CHECKPOINTING = True

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

VAL_FRACTION = 0.10
DEFAULT_STEPS = 0
DEFAULT_VAL_STEPS = 10
PRINT_EVERY = 5
SEED = 42

TARGET_MIN = -100.0
TARGET_MAX = 500.0
# DEM mosaics can end at a legitimate scene boundary. The loss and metrics
# are masked to real pixels, so reject only crops with less than one quarter
# real elevation coverage; preflight separately requires most records in each
# split to be usable.
MIN_VALID_FRACTION = 0.25

EXPECTED_MOSAIC_CRS = "EPSG:32721"

IMAGE_EXTENSIONS = {".tif", ".tiff"}

# Sparse grid used for geometry fitting.
FIT_GRID = 5
RPC_SEARCH_GRID = 7
RPC_GRID = 9

# Heights used only for fallback search if footprint fitting is unavailable.
RPC_HEIGHT_FALLBACKS = [
    0.0,
    25.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
]

# Expected geographic footprint residual tolerance.
# RPC sidecar bounds are rounded, so do not demand sub-meter agreement.
FOOTPRINT_RESIDUAL_DEG = 0.003


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MVSRecord:
    key: str
    region: str
    image_path: Path
    rpc_path: Path
    mosaic_path: Path


@dataclass
class ElevationRaster:
    region: str
    path: Path
    dataset: Any
    crs: Any
    transform: Any
    nodata: Optional[float]


# ============================================================================
# BASIC UTILITIES
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
            "CUDA is required for Stage-5 full-parameter training."
        )
    return torch.device("cuda")


def atomic_json_write(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def print_gpu_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    print()
    print(f"--- GPU MEMORY: {label} ---")
    print(
        f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB"
    )
    print(
        f"Reserved : {torch.cuda.memory_reserved() / 1024**3:.3f} GB"
    )
    print(
        f"Peak     : {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB"
    )


def print_header() -> None:
    print("=" * 80)
    print("ASTERRA AI — STAGE 5")
    print("SPACENET MVS RPC-AWARE ELEVATION FINE-TUNING — FAST ALIGNMENT")
    print("=" * 80)

    print(f"Project root : {ROOT}")
    print(f"MVS dataset  : {DATASET_ROOT}")
    print(f"Stage-4     : {STAGE4_BEST}")
    print(f"Output      : {OUTPUT_DIR}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print()
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {props.total_memory / 1024**3:.2f} GB")
        print(f"PyTorch      : {torch.__version__}")
        print(f"CUDA         : {torch.version.cuda}")
    else:
        print()
        print("GPU          : CUDA NOT AVAILABLE")

    print()
    print(f"Encoder      : {ENCODER}")
    print(f"Dataset crop : {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Model input  : {MODEL_SIZE} x {MODEL_SIZE}")
    print(f"Patch size   : {DINO_PATCH_SIZE}")
    print(
        f"{MODEL_SIZE} / {DINO_PATCH_SIZE}     : "
        f"{MODEL_SIZE // DINO_PATCH_SIZE}"
    )

    print()
    print(f"Batch        : {BATCH_SIZE}")
    print(f"Accumulation : {GRAD_ACCUMULATION}")
    print(f"Effective BS : {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay : {WEIGHT_DECAY}")
    print(f"Epochs       : {EPOCHS}")
    print(f"AMP          : {AMP_ENABLED}")
    print(f"Grad clipping: {GRAD_CLIP}")
    print()


# ============================================================================
# DISCOVERY
# ============================================================================

def is_mosaic_file(path: Path) -> bool:
    return (
        path.parent == DATASET_ROOT
        and path.name.lower()
        in {
            "masterprovisional1.tif",
            "masterprovisional2.tif",
            "masterprovisional3.tif",
        }
    )


def infer_region_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        low = part.lower()

        if low == "masterprovisional1":
            return "MasterProvisional1"

        if low == "masterprovisional2":
            return "MasterProvisional2"

        if low == "masterprovisional3":
            return "MasterProvisional3"

    return None


def normalize_scene_stem(stem: str) -> str:
    value = str(stem).strip()
    if value.lower().startswith("rpc_"):
        value = value[4:]
    return value.lower()


def discover_image_tiffs() -> List[Path]:
    files: List[Path] = []

    for p in DATASET_ROOT.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if is_mosaic_file(p):
            continue

        if infer_region_from_path(p) is None:
            continue

        files.append(p)

    return sorted(set(files), key=lambda x: str(x).lower())


def discover_rpc_files() -> List[Path]:
    files: List[Path] = []

    for p in DATASET_ROOT.rglob("*.txt"):
        if not p.is_file():
            continue

        if infer_region_from_path(p) is None:
            continue

        files.append(p)

    return sorted(set(files), key=lambda x: str(x).lower())


def build_rpc_index(files: Sequence[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}

    for p in files:
        index[normalize_scene_stem(p.stem)] = p

    return index


def find_rpc_for_image(
    image_path: Path,
    rpc_index: Dict[str, Path],
) -> Optional[Path]:
    key = normalize_scene_stem(image_path.stem)

    if key in rpc_index:
        return rpc_index[key]

    conventional = image_path.with_name(
        f"rpc_{image_path.stem}.txt"
    )

    if conventional.exists():
        return conventional

    return None


def mosaic_for_region(region: str) -> Path:
    return DATASET_ROOT / f"{region}.tif"


def discover_records() -> List[MVSRecord]:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    images = discover_image_tiffs()
    rpcs = discover_rpc_files()
    index = build_rpc_index(rpcs)

    print()
    print(f"[DISCOVERY] Actual image TIFFs : {len(images)}")
    print(f"[DISCOVERY] RPC TXT files      : {len(rpcs)}")

    records: List[MVSRecord] = []

    for image in images:
        region = infer_region_from_path(image)

        if region is None:
            continue

        rpc = find_rpc_for_image(image, index)
        mosaic = mosaic_for_region(region)

        if rpc is None or not mosaic.exists():
            continue

        records.append(
            MVSRecord(
                key=image.stem,
                region=region,
                image_path=image,
                rpc_path=rpc,
                mosaic_path=mosaic,
            )
        )

    print(f"[DISCOVERY] Valid records       : {len(records)}")
    return records


def save_manifest(records: Sequence[MVSRecord]) -> None:
    atomic_json_write(
        MANIFEST_FILE,
        {
            "version": 6,
            "stage": 5,
            "dataset": "SpaceNet MVS",
            "geometry": (
                "RPC footprint fitting using geographic bounds, "
                "then elevation-mosaic validation"
            ),
            "records": [
                {
                    "key": r.key,
                    "region": r.region,
                    "image": str(r.image_path),
                    "rpc": str(r.rpc_path),
                    "mosaic": str(r.mosaic_path),
                }
                for r in records
            ],
        },
    )


# ============================================================================
# ELEVATION MOSAICS
# ============================================================================

def open_elevation_rasters() -> Dict[str, ElevationRaster]:
    out: Dict[str, ElevationRaster] = {}

    for region in (
        "MasterProvisional1",
        "MasterProvisional2",
        "MasterProvisional3",
    ):
        path = mosaic_for_region(region)

        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        ds = rasterio.open(path)

        out[region] = ElevationRaster(
            region=region,
            path=path,
            dataset=ds,
            crs=ds.crs,
            transform=ds.transform,
            nodata=ds.nodata,
        )

    return out


def print_elevation_inventory(
    mosaics: Dict[str, ElevationRaster],
) -> None:
    print()
    print("ELEVATION MOSAICS")
    print("-" * 80)

    for region in (
        "MasterProvisional1",
        "MasterProvisional2",
        "MasterProvisional3",
    ):
        m = mosaics.get(region)

        if m is None:
            print(f"[MISSING] {region}")
            continue

        ds = m.dataset

        print(
            f"[OK] {region}: "
            f"{ds.width}x{ds.height} | "
            f"bands={ds.count} | "
            f"dtype={ds.dtypes[0]} | "
            f"CRS={ds.crs} | "
            f"nodata={ds.nodata}"
        )

        if ds.crs is None:
            raise RuntimeError(
                f"{region} elevation mosaic has no CRS."
            )


# ============================================================================
# RPC PARSING
# ============================================================================

NUMBER_RE = re.compile(
    r"""
    [+-]?
    (?:
        \d+(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][+-]?\d+)?
    """,
    re.VERBOSE,
)


def read_rpc_numbers(path: Path) -> List[float]:
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )
    return [
        float(x)
        for x in NUMBER_RE.findall(text)
    ]


def rpc_from_numbers(
    values: Sequence[float],
) -> Dict[str, Any]:
    if len(values) < 90:
        raise ValueError(
            f"RPC file contains {len(values)} numbers; "
            "at least 90 are required."
        )

    v = list(map(float, values))

    return {
        "line_off": v[0],
        "samp_off": v[1],
        "lat_off": v[2],
        "lon_off": v[3],
        "height_off": v[4],
        "line_scale": v[5],
        "samp_scale": v[6],
        "lat_scale": v[7],
        "lon_scale": v[8],
        "height_scale": v[9],
        "line_num": v[10:30],
        "line_den": v[30:50],
        "samp_num": v[50:70],
        "samp_den": v[70:90],
        "tail": list(map(float, values[90:])),
    }


def rpc_tail_bbox(
    rpc: Dict[str, Any],
) -> Optional[Tuple[float, float, float, float]]:
    """
    Local SpaceNet RPC sidecars observed in this project contain:

        lon_min, lat_min, lon_max, lat_max, ...

    Validate the first four tail values before treating them as a bbox.
    """

    tail = rpc.get("tail", [])

    if len(tail) < 4:
        return None

    lon_min, lat_min, lon_max, lat_max = map(
        float,
        tail[:4],
    )

    if not (
        -180.0 <= lon_min <= 180.0
        and -180.0 <= lon_max <= 180.0
        and -90.0 <= lat_min <= 90.0
        and -90.0 <= lat_max <= 90.0
    ):
        return None

    if lon_max <= lon_min or lat_max <= lat_min:
        return None

    return (
        lon_min,
        lat_min,
        lon_max,
        lat_max,
    )


def rpc_tail_pixel_hints(
    rpc: Dict[str, Any],
) -> List[Tuple[float, float, str]]:
    """
    Return crop-origin metadata from the MVS3D RPC sidecar.

    The crop tool writes the final two values as sample/column origin and
    line/row origin. The observed convention is therefore a=(column),
    b=(row), and the primary returned tuple is (row, column). Keep the
    alternate orientation as a diagnostic fallback for older sidecars.
    """

    tail = rpc.get("tail", [])

    if len(tail) < 6:
        return []

    a = float(tail[-2])
    b = float(tail[-1])

    if abs(a) >= 1e7 or abs(b) >= 1e7:
        return []

    # Observed convention: a=sample/column, b=line/row.
    return [
        (b, a, "rpc-tail-sample-line"),
        (a, b, "rpc-tail-line-sample-fallback"),
    ]


def rpc_to_rasterio(
    rpc: Dict[str, Any],
) -> RPC:
    return RPC(
        height_off=float(rpc["height_off"]),
        height_scale=float(rpc["height_scale"]),
        lat_off=float(rpc["lat_off"]),
        lat_scale=float(rpc["lat_scale"]),
        line_den_coeff=list(
            map(float, rpc["line_den"])
        ),
        line_num_coeff=list(
            map(float, rpc["line_num"])
        ),
        line_off=float(rpc["line_off"]),
        line_scale=float(rpc["line_scale"]),
        long_off=float(rpc["lon_off"]),
        long_scale=float(rpc["lon_scale"]),
        samp_den_coeff=list(
            map(float, rpc["samp_den"])
        ),
        samp_num_coeff=list(
            map(float, rpc["samp_num"])
        ),
        samp_off=float(rpc["samp_off"]),
        samp_scale=float(rpc["samp_scale"]),
    )


# ============================================================================
# RPC INVERSE
# ============================================================================

# ============================================================================
# RPC INVERSE — FAST PERSISTENT TRANSFORMER
# ============================================================================

_RPC_TRANSFORMER_CACHE = {}


def _get_rpc_transformer(
    rpc_obj: RPC,
) -> RPCTransformer:
    """
    Return one persistent GDAL RPC transformer per RPC object.

    Creating RPCTransformer repeatedly causes expensive GDAL environment
    initialization. Stage-5 performs many RPC evaluations during geometry
    validation, so the transformer must be reused.
    """

    key = id(rpc_obj)

    transformer = _RPC_TRANSFORMER_CACHE.get(key)

    if transformer is None:
        transformer = RPCTransformer(
            rpc_obj,
            rpc_height=0.0,
            RPC_MAX_ITERATIONS=30,
            RPC_PIXEL_ERROR_THRESHOLD=0.25,
        )

        _RPC_TRANSFORMER_CACHE[key] = transformer

    return transformer


# ============================================================================
# RPC INVERSE — PERSISTENT TRANSFORMER
# ============================================================================

_RPC_TRANSFORMER_CACHE = {}


def _get_rpc_transformer(rpc_obj: RPC):
    """
    Create one RPCTransformer per RPC object and reuse it.

    The previous implementation created and destroyed a GDAL/RPC transformer
    for every geometry evaluation. During Stage-5 preflight this multiplied
    into thousands of expensive RPC inverse operations.

    This cache is intentionally process-local.
    """

    key = id(rpc_obj)

    transformer = _RPC_TRANSFORMER_CACHE.get(key)

    if transformer is None:

        transformer = RPCTransformer(
            rpc_obj,
            rpc_height=0.0,
            RPC_MAX_ITERATIONS=50,
            RPC_PIXEL_ERROR_THRESHOLD=0.25,
        )

        _RPC_TRANSFORMER_CACHE[key] = transformer

    return transformer


def rpc_inverse(
    rpc_obj: RPC,
    rows: np.ndarray,
    cols: np.ndarray,
    height: float,
) -> Tuple[np.ndarray, np.ndarray]:

    rows = np.asarray(
        rows,
        dtype=np.float64,
    )

    cols = np.asarray(
        cols,
        dtype=np.float64,
    )

    if rows.shape != cols.shape:
        raise ValueError(
            "RPC row/column arrays must have identical shapes."
        )

    flat_r = rows.reshape(-1)
    flat_c = cols.reshape(-1)

    if flat_r.size == 0:
        raise RuntimeError(
            "RPC inverse received zero coordinates."
        )

    zs = np.full(
        flat_r.shape,
        float(height),
        dtype=np.float64,
    )

    tr = _get_rpc_transformer(rpc_obj)

    try:

        # RPCTransformer.xy() expects image rows first and columns second.
        lon, lat = tr.xy(
            flat_r,
            flat_c,
            zs=zs,
        )

    except Exception as exc:

        raise RuntimeError(
            f"RPC inverse failed: {type(exc).__name__}: {exc}"
        ) from exc

    lon = np.asarray(
        lon,
        dtype=np.float64,
    ).reshape(rows.shape)

    lat = np.asarray(
        lat,
        dtype=np.float64,
    ).reshape(rows.shape)

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (np.abs(lat) <= 90.0)
        & (np.abs(lon) <= 180.0)
    )

    if not bool(valid.any()):
        raise RuntimeError(
            "GDAL RPC inverse returned no valid coordinates."
        )

    return lat, lon

# ============================================================================
# GEO / MOSAIC COORDINATES
# ============================================================================

def geographic_to_mosaic_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    mosaic: ElevationRaster,
) -> Tuple[np.ndarray, np.ndarray]:
    if mosaic.crs is None:
        raise RuntimeError(
            f"Mosaic has no CRS: {mosaic.path}"
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        mosaic.crs,
        always_xy=True,
    )

    x, y = transformer.transform(
        lon,
        lat,
    )

    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )


def mosaic_xy_to_pixel(
    x: np.ndarray,
    y: np.ndarray,
    mosaic: ElevationRaster,
) -> Tuple[np.ndarray, np.ndarray]:
    inv = ~mosaic.transform

    col, row = inv * (
        x,
        y,
    )

    return (
        np.asarray(row, dtype=np.float64),
        np.asarray(col, dtype=np.float64),
    )


def inside_fraction(
    rows: np.ndarray,
    cols: np.ndarray,
    mosaic: ElevationRaster,
) -> float:
    valid = (
        np.isfinite(rows)
        & np.isfinite(cols)
        & (rows >= 0.0)
        & (rows < mosaic.dataset.height)
        & (cols >= 0.0)
        & (cols < mosaic.dataset.width)
    )

    return (
        float(valid.mean())
        if valid.size
        else 0.0
    )


def point_valid_fraction(
    rows: np.ndarray,
    cols: np.ndarray,
    mosaic: ElevationRaster,
) -> float:
    inside = (
        np.isfinite(rows)
        & np.isfinite(cols)
        & (rows >= 0.0)
        & (rows < mosaic.dataset.height)
        & (cols >= 0.0)
        & (cols < mosaic.dataset.width)
    )

    if not inside.any():
        return 0.0

    rr = np.rint(
        rows[inside]
    ).astype(np.int64)

    cc = np.rint(
        cols[inside]
    ).astype(np.int64)

    vals: List[float] = []

    for r, c in zip(
        rr.tolist(),
        cc.tolist(),
    ):
        try:
            value = next(
                mosaic.dataset.sample(
                    [(float(c), float(r))],
                    indexes=1,
                )
            )[0]

            vals.append(float(value))

        except Exception:
            vals.append(float("nan"))

    values = np.asarray(
        vals,
        dtype=np.float64,
    )

    valid = np.isfinite(values)

    if mosaic.nodata is not None:
        valid &= (
            values != float(mosaic.nodata)
        )

    valid &= values >= TARGET_MIN
    valid &= values <= TARGET_MAX

    return (
        float(valid.mean())
        if valid.size
        else 0.0
    )


# ============================================================================
# RPC FOOTPRINT FITTING
# ============================================================================

def local_grid(
    image_h: int,
    image_w: int,
    n: int = FIT_GRID,
) -> Tuple[np.ndarray, np.ndarray]:
    rr = np.linspace(
        0.0,
        float(image_h - 1),
        n,
        dtype=np.float64,
    )

    cc = np.linspace(
        0.0,
        float(image_w - 1),
        n,
        dtype=np.float64,
    )

    return np.meshgrid(
        rr,
        cc,
        indexing="ij",
    )


def footprint_expected_points(
    bbox: Tuple[float, float, float, float],
    image_h: int,
    image_w: int,
    n: int = FIT_GRID,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build expected geographic control points from the supplied footprint.

    We assume the image axes follow the usual north/south + west/east
    ordering, but also return the four corner targets explicitly so that
    the fit can reject an implausible orientation.
    """

    lon_min, lat_min, lon_max, lat_max = bbox

    u = np.linspace(
        0.0,
        1.0,
        n,
        dtype=np.float64,
    )

    v = np.linspace(
        0.0,
        1.0,
        n,
        dtype=np.float64,
    )

    # row=0 -> northern edge; row=max -> southern edge
    # col=0 -> western edge; col=max -> eastern edge
    lon = lon_min + v[None, :] * (
        lon_max - lon_min
    )

    lat = lat_max - u[:, None] * (
        lat_max - lat_min
    )

    lon = np.broadcast_to(
        lon,
        (n, n),
    )

    lat = np.broadcast_to(
        lat,
        (n, n),
    )

    return (
        lat.copy(),
        lon.copy(),
        np.asarray(
            [
                [lat_max, lat_max, lat_min, lat_min],
                [lon_min, lon_max, lon_min, lon_max],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.0, 0.0, float(image_h - 1), float(image_h - 1)],
                [0.0, float(image_w - 1), 0.0, float(image_w - 1)],
            ],
            dtype=np.float64,
        ),
    )


# ============================================================================
# FAST RPC LOCAL FRAME RESOLUTION
# ============================================================================

def fit_rpc_local_frame(
    rpc_obj: RPC,
    rpc: Dict[str, Any],
    image_h: int,
    image_w: int,
) -> Optional[Tuple[float, float, float, float, str]]:
    """
    Fast Stage-5 RPC frame resolver.

    SpaceNet MVS local TIFFs are image chips while the RPC describes the
    parent satellite-image coordinate frame.

    The observed SpaceNet RPC sidecar tail convention is:

        tail[-2] = parent-image sample / column
        tail[-1] = parent-image line   / row

    We therefore use those coordinates directly to derive the local TIFF
    -> parent RPC-frame translation.

    IMPORTANT:
    No scipy.optimize.least_squares() is used here.

    The previous implementation performed nonlinear RPC optimization,
    which repeatedly called RPC inverse calculations and made preflight
    extremely slow.

    Real elevation validity is NOT assumed here.
    find_best_geometry() performs the real elevation validation.
    """

    tail = rpc.get(
        "tail",
        [],
    )

    if len(tail) < 6:
        return None

    try:

        sample = float(
            tail[-2]
        )

        line = float(
            tail[-1]
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not (
        math.isfinite(sample)
        and math.isfinite(line)
    ):
        return None

    if (
        abs(sample) >= 1e7
        or abs(line) >= 1e7
    ):
        return None

    local_center_row = (
        float(image_h - 1)
        / 2.0
    )

    local_center_col = (
        float(image_w - 1)
        / 2.0
    )

    candidates = [

        (
            line - local_center_row,
            sample - local_center_col,
            float(rpc["height_off"]),
            "rpc-tail-sample-line-center",
        ),

        (
            sample - local_center_row,
            line - local_center_col,
            float(rpc["height_off"]),
            "rpc-tail-line-sample-center",
        ),

        (
            line,
            sample,
            float(rpc["height_off"]),
            "rpc-tail-sample-line-origin",
        ),

        (
            sample,
            line,
            float(rpc["height_off"]),
            "rpc-tail-line-sample-origin",
        ),
    ]

    for (
        row_offset,
        col_offset,
        height,
        mode,
    ) in candidates:

        if not (
            math.isfinite(row_offset)
            and math.isfinite(col_offset)
            and math.isfinite(height)
        ):
            continue

        if (
            abs(row_offset) >= 200000.0
            or abs(col_offset) >= 200000.0
        ):
            continue

        return (
            float(row_offset),
            float(col_offset),
            float(height),
            0.0,
            mode,
        )

    return None


def candidate_rpc_offsets(
    rpc: Dict[str, Any],
    image_h: int,
    image_w: int,
) -> List[Tuple[float, float, str]]:
    """
    Fallback candidates.

    IMPORTANT:
    The geographic footprint fit is the primary geometry resolver and ,
    These candidates are retained for files that do not expose a usable
    geographic bbox in their tail.
    """

    cr = (image_h - 1) / 2.0
    cc = (image_w - 1) / 2.0

    candidates: List[
        Tuple[float, float, str]
    ] = [
        (
            0.0,
            0.0,
            "direct",
        ),
        (
            float(rpc["line_off"]) - cr,
            float(rpc["samp_off"]) - cc,
            "rpc-center-to-local-center",
        ),
        (
            float(rpc["line_off"]) - image_h / 2.0,
            float(rpc["samp_off"]) - image_w / 2.0,
            "rpc-center-to-local-half",
        ),
    ]

    for tr, tc, name in rpc_tail_pixel_hints(rpc):
        candidates.extend(
            [
                (
                    tr,
                    tc,
                    name,
                ),
                (
                    tr - cr,
                    tc - cc,
                    name + "-center",
                ),
                (
                    tr - image_h / 2.0,
                    tc - image_w / 2.0,
                    name + "-half",
                ),
            ]
        )

    out: List[
        Tuple[float, float, str]
    ] = []

    seen = set()

    for ro, co, name in candidates:
        key = (
            round(float(ro), 3),
            round(float(co), 3),
        )

        if key in seen:
            continue

        seen.add(key)

        out.append(
            (
                float(ro),
                float(co),
                name,
            )
        )

    return out


def apply_rpc_offset(
    rows: np.ndarray,
    cols: np.ndarray,
    row_offset: float,
    col_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(rows, dtype=np.float64)
        + float(row_offset),
        np.asarray(cols, dtype=np.float64)
        + float(col_offset),
    )


# ============================================================================
# MOSAIC SAMPLING
# ============================================================================

def bilinear_sample_mosaic(
    mosaic: ElevationRaster,
    rows: np.ndarray,
    cols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read only the bounding window around requested mosaic coordinates and
    bilinearly sample it.
    """

    rows = np.asarray(
        rows,
        dtype=np.float64,
    )
    cols = np.asarray(
        cols,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(rows)
        & np.isfinite(cols)
    )

    if not finite.any():
        raise RuntimeError(
            "No finite mosaic coordinates."
        )

    rmin = max(
        0,
        int(
            math.floor(
                float(
                    np.nanmin(
                        rows[finite]
                    )
                )
            )
        ) - 2,
    )

    rmax = min(
        mosaic.dataset.height - 1,
        int(
            math.ceil(
                float(
                    np.nanmax(
                        rows[finite]
                    )
                )
            )
        ) + 2,
    )

    cmin = max(
        0,
        int(
            math.floor(
                float(
                    np.nanmin(
                        cols[finite]
                    )
                )
            )
        ) - 2,
    )

    cmax = min(
        mosaic.dataset.width - 1,
        int(
            math.ceil(
                float(
                    np.nanmax(
                        cols[finite]
                    )
                )
            )
        ) + 2,
    )

    if rmax < rmin or cmax < cmin:
        raise RuntimeError(
            "Mosaic coordinate window is empty."
        )

    window = Window(
        cmin,
        rmin,
        cmax - cmin + 1,
        rmax - rmin + 1,
    )

    arr = mosaic.dataset.read(
        1,
        window=window,
    ).astype(np.float32)

    local_r = rows - rmin
    local_c = cols - cmin

    h, w = arr.shape

    inside = (
        finite
        & (local_r >= 0.0)
        & (local_c >= 0.0)
        & (local_r <= h - 1)
        & (local_c <= w - 1)
    )

    out = np.zeros(
        rows.shape,
        dtype=np.float32,
    )

    mask = np.zeros(
        rows.shape,
        dtype=np.float32,
    )

    if not inside.any():
        return out, mask

    r = np.clip(
        local_r,
        0,
        h - 1,
    )

    c = np.clip(
        local_c,
        0,
        w - 1,
    )

    r0 = np.floor(r).astype(
        np.int64
    )
    c0 = np.floor(c).astype(
        np.int64
    )

    r1 = np.minimum(
        r0 + 1,
        h - 1,
    )
    c1 = np.minimum(
        c0 + 1,
        w - 1,
    )

    fr = r - r0
    fc = c - c0

    a = arr[
        r0,
        c0,
    ]

    b = arr[
        r0,
        c1,
    ]

    cval = arr[
        r1,
        c0,
    ]

    d = arr[
        r1,
        c1,
    ]

    nod = mosaic.nodata

    va = np.isfinite(a)
    vb = np.isfinite(b)
    vc = np.isfinite(cval)
    vd = np.isfinite(d)

    if nod is not None:
        va &= a != float(nod)
        vb &= b != float(nod)
        vc &= cval != float(nod)
        vd &= d != float(nod)

    wa = (
        (1.0 - fr)
        * (1.0 - fc)
    )
    wb = (
        (1.0 - fr)
        * fc
    )
    wc = (
        fr
        * (1.0 - fc)
    )
    wd = (
        fr
        * fc
    )

    weighted = (
        np.where(
            va,
            a * wa,
            0.0,
        )
        + np.where(
            vb,
            b * wb,
            0.0,
        )
        + np.where(
            vc,
            cval * wc,
            0.0,
        )
        + np.where(
            vd,
            d * wd,
            0.0,
        )
    )

    weights = (
        np.where(
            va,
            wa,
            0.0,
        )
        + np.where(
            vb,
            wb,
            0.0,
        )
        + np.where(
            vc,
            wc,
            0.0,
        )
        + np.where(
            vd,
            wd,
            0.0,
        )
    )

    good = (
        inside
        & (weights > 0.25)
    )

    out[good] = (
        weighted[good]
        / weights[good]
    ).astype(np.float32)

    valid_range = (
        (out >= TARGET_MIN)
        & (out <= TARGET_MAX)
    )

    mask[
        good & valid_range
    ] = 1.0

    out[
        mask < 0.5
    ] = 0.0

    return out, mask


# ============================================================================
# IMAGE / TARGET GEOMETRY
# ============================================================================

def crop_origins(
    image_h: int,
    image_w: int,
) -> List[Tuple[int, int]]:
    max_r = image_h - PATCH_SIZE
    max_c = image_w - PATCH_SIZE

    if max_r < 0 or max_c < 0:
        return []

    vals_r = sorted(
        set(
            [
                0,
                max_r // 8,
                max_r // 4,
                max_r // 2,
                (3 * max_r) // 4,
                (7 * max_r) // 8,
                max_r,
            ]
        )
    )

    vals_c = sorted(
        set(
            [
                0,
                max_c // 8,
                max_c // 4,
                max_c // 2,
                (3 * max_c) // 4,
                (7 * max_c) // 8,
                max_c,
            ]
        )
    )

    return [
        (r, c)
        for r in vals_r
        for c in vals_c
    ]


def rpc_control_points(
    row0: int,
    col0: int,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.linspace(
        0,
        size - 1,
        RPC_SEARCH_GRID,
        dtype=np.float64,
    )

    rr, cc = np.meshgrid(
        p + float(row0),
        p + float(col0),
        indexing="ij",
    )

    return rr, cc


def build_rpc_target_grid(
    rpc_obj: RPC,
    row0: int,
    col0: int,
    row_offset: float,
    col_offset: float,
    height: float,
    mosaic: ElevationRaster,
) -> Tuple[np.ndarray, np.ndarray]:
    p = np.linspace(
        0,
        PATCH_SIZE - 1,
        RPC_GRID,
        dtype=np.float64,
    )

    rr, cc = np.meshgrid(
        p + float(row0),
        p + float(col0),
        indexing="ij",
    )

    rpc_r, rpc_c = apply_rpc_offset(
        rr,
        cc,
        row_offset,
        col_offset,
    )

    lat, lon = rpc_inverse(
        rpc_obj,
        rpc_r,
        rpc_c,
        height,
    )

    x, y = geographic_to_mosaic_xy(
        lat,
        lon,
        mosaic,
    )

    mr, mc = mosaic_xy_to_pixel(
        x,
        y,
        mosaic,
    )

    # Interpolate sparse RPC/mosaic coordinate grid to the full 512x512 crop.
    def interp(
        grid: np.ndarray,
    ) -> np.ndarray:
        gh, gw = grid.shape

        yq = np.linspace(
            0,
            gh - 1,
            PATCH_SIZE,
        )

        xq = np.linspace(
            0,
            gw - 1,
            PATCH_SIZE,
        )

        yi = np.floor(
            yq
        ).astype(np.int64)

        xi = np.floor(
            xq
        ).astype(np.int64)

        yi1 = np.minimum(
            yi + 1,
            gh - 1,
        )

        xi1 = np.minimum(
            xi + 1,
            gw - 1,
        )

        fy = yq - yi
        fx = xq - xi

        g00 = grid[
            yi[:, None],
            xi[None, :],
        ]

        g01 = grid[
            yi[:, None],
            xi1[None, :],
        ]

        g10 = grid[
            yi1[:, None],
            xi[None, :],
        ]

        g11 = grid[
            yi1[:, None],
            xi1[None, :],
        ]

        top = (
            g00 * (1.0 - fx[None, :])
            + g01 * fx[None, :]
        )

        bot = (
            g10 * (1.0 - fx[None, :])
            + g11 * fx[None, :]
        )

        return (
            top * (1.0 - fy[:, None])
            + bot * fy[:, None]
        )

    return (
        interp(mr),
        interp(mc),
    )



def mosaic_guided_rpc_candidates(
    mosaic: ElevationRaster,
    rpc_obj: RPC,
    image_h: int,
    image_w: int,
) -> List[Tuple[float, float, float, str, float]]:
    """
    Resolve the local 2001x2001 scene against the elevation mosaic directly.

    IMPORTANT:
    The first four tail values in these SpaceNet MVS RPC sidecars describe
    the geographic footprint of the parent RPC image.  That footprint can be
    much larger than the local 2001x2001 TIFF.  Therefore it must NOT be
    forced to equal the local TIFF footprint.

    Instead, sample geographic points across the actual elevation mosaic,
    project those points back through the RPC at several plausible heights,
    and treat the resulting RPC row/column as the possible location of the
    local image center.  The downstream crop/validity search then selects
    the location that genuinely overlaps the elevation raster.

    Returns:
        row_offset, col_offset, height, mode, score
    """
    cr = (float(image_h) - 1.0) / 2.0
    cc = (float(image_w) - 1.0) / 2.0

    # Nine well-spread points across the mosaic.  Using the actual mosaic
    # geometry avoids assuming that the RPC sidecar bbox equals the TIFF.
    fracs = [
        (0.05, 0.05), (0.05, 0.50), (0.05, 0.95),
        (0.50, 0.05), (0.50, 0.50), (0.50, 0.95),
        (0.95, 0.05), (0.95, 0.50), (0.95, 0.95),
    ]

    rows = np.asarray(
        [f[0] * (mosaic.dataset.height - 1) for f in fracs],
        dtype=np.float64,
    )
    cols = np.asarray(
        [f[1] * (mosaic.dataset.width - 1) for f in fracs],
        dtype=np.float64,
    )

    xs, ys = mosaic.dataset.xy(rows, cols, offset="center")
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    to_wgs84 = Transformer.from_crs(
        mosaic.crs,
        "EPSG:4326",
        always_xy=True,
    )
    lons, lats = to_wgs84.transform(xs, ys)
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    candidates: List[Tuple[float, float, float, str, float]] = []

    # Buenos Aires / SpaceNet MVS elevations are low; include the RPC
    # reference height plus a broad but realistic range.
    heights = sorted(set(
        [float(RPC_HEIGHT_FALLBACKS[0]),
         25.0, 50.0, 75.0, 100.0, 150.0, 200.0,
         float(rpc_obj.height_off)]
    ))

    for h in heights:
        zs = np.full(lats.shape, h, dtype=np.float64)

        try:
            with RPCTransformer(
                rpc_obj,
                rpc_height=0.0,
                RPC_MAX_ITERATIONS=100,
                RPC_PIXEL_ERROR_THRESHOLD=0.1,
            ) as tr:
                rpc_rows, rpc_cols = tr.rowcol(
                    lons,
                    lats,
                    zs=zs,
                )
        except Exception:
            continue

        rpc_rows = np.asarray(rpc_rows, dtype=np.float64)
        rpc_cols = np.asarray(rpc_cols, dtype=np.float64)

        valid = (
            np.isfinite(rpc_rows)
            & np.isfinite(rpc_cols)
        )

        for idx in np.flatnonzero(valid):
            ro = float(rpc_rows[idx] - cr)
            co = float(rpc_cols[idx] - cc)

            if not (
                np.isfinite(ro)
                and np.isfinite(co)
                and abs(ro) < 200000.0
                and abs(co) < 200000.0
            ):
                continue

            candidates.append(
                (
                    ro,
                    co,
                    float(h),
                    "mosaic-guided-rpc-center",
                    0.0,
                )
            )

    # Add the mosaic centre independently.  It is usually the strongest
    # candidate when the scene is a chip cut from the same parent image.
    return candidates


def geometry_candidates(
    record: MVSRecord,
    mosaic: ElevationRaster,
    rpc_obj: RPC,
    rpc: Dict[str, Any],
) -> List[
    Tuple[
        float,
        float,
        float,
        str,
        float,
    ]
]:
    """
    Return candidate:

        row_offset, col_offset, height, mode, footprint_fit_score
    """

    with rasterio.open(
        record.image_path
    ) as ds:
        image_h = int(ds.height)
        image_w = int(ds.width)

    fitted = fit_rpc_local_frame(
        rpc_obj,
        rpc,
        image_h,
        image_w,
    )

    candidates: List[
        Tuple[
            float,
            float,
            float,
            str,
            float,
        ]
    ] = []

    # PRIMARY FIX:
    # Do not assume the RPC sidecar geographic bbox is the footprint of the
    # local 2001x2001 TIFF.  SpaceNet MVS stores a local image chip while the
    # RPC sidecar can describe its larger parent image.  Use the actual
    # elevation mosaic to generate physically possible RPC-frame locations.
    candidates.extend(
        mosaic_guided_rpc_candidates(
            mosaic,
            rpc_obj,
            image_h,
            image_w,
        )
    )

    if fitted is not None:
        ro, co, h, fit_score, mode = fitted

        candidates.append(
            (
                ro,
                co,
                h,
                mode,
                fit_score,
            )
        )

    for ro, co, mode in candidate_rpc_offsets(
        rpc,
        image_h,
        image_w,
    ):
        candidates.append(
            (
                ro,
                co,
                float(rpc["height_off"]),
                mode,
                999.0,
            )
        )

        # Also test physically plausible heights around RPC reference height.
        for h in RPC_HEIGHT_FALLBACKS:
            candidates.append(
                (
                    ro,
                    co,
                    float(h),
                    mode + f"-h{h:g}",
                    999.0,
                )
            )

    # Deduplicate.
    out = []
    seen = set()

    for item in candidates:
        ro, co, h, mode, fit_score = item

        key = (
            round(ro, 3),
            round(co, 3),
            round(h, 2),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    # Fitted solution first.
    out.sort(
        key=lambda x: (
            0 if x[3].startswith(
                "mosaic-guided-rpc-center"
            ) else (
                1 if x[3].startswith(
                    "rpc-geographic-footprint-fit"
                ) else 2
            ),
            x[4],
        )
    )

    return out



def find_best_geometry(
    record: MVSRecord,
    mosaic: ElevationRaster,
    rpc_obj: RPC,
    rpc: Dict[str, Any],
) -> Tuple[
    int,
    int,
    float,
    float,
    float,
    str,
    float,
    float,
]:
    """
    FAST Stage-5 RPC/elevation geometry resolver.

    Design goals
    ------------
    1. No nonlinear RPC optimization.
    2. No thousands-of-crop exhaustive search.
    3. One persistent RPCTransformer per RPC.
    4. Only a small bounded set of heights.
    5. Only five control points per candidate.
    6. Real elevation data is mandatory.
    7. Synthetic elevation is never generated.

    Returns:
        row0,
        col0,
        row_offset,
        col_offset,
        height,
        offset_name,
        overlap,
        target_valid
    """

    # ------------------------------------------------------------------
    # IMAGE SIZE
    # ------------------------------------------------------------------

    with rasterio.open(
        record.image_path
    ) as ds:

        image_h = int(ds.height)
        image_w = int(ds.width)

    if (
        image_h < PATCH_SIZE
        or image_w < PATCH_SIZE
    ):
        raise RuntimeError(
            "Image is smaller than 512x512: "
            f"{image_w}x{image_h}"
        )

    max_row = image_h - PATCH_SIZE
    max_col = image_w - PATCH_SIZE

    # ------------------------------------------------------------------
    # RPC FRAME CANDIDATES
    # ------------------------------------------------------------------

    candidates = []

    fitted = fit_rpc_local_frame(
        rpc_obj,
        rpc,
        image_h,
        image_w,
    )

    if fitted is not None:

        (
            row_offset,
            col_offset,
            height,
            fit_score,
            offset_name,
        ) = fitted

        candidates.append(
            (
                float(row_offset),
                float(col_offset),
                float(height),
                str(offset_name),
                float(fit_score),
            )
        )

    # ------------------------------------------------------------------
    # TAIL / STANDARD FALLBACK OFFSETS
    # ------------------------------------------------------------------

    for (
        row_offset,
        col_offset,
        offset_name,
    ) in candidate_rpc_offsets(
        rpc,
        image_h,
        image_w,
    ):

        candidates.append(
            (
                float(row_offset),
                float(col_offset),
                float(rpc["height_off"]),
                str(offset_name),
                999.0,
            )
        )

    # ------------------------------------------------------------------
    # DEDUPLICATE FRAME CANDIDATES
    # ------------------------------------------------------------------

    unique_candidates = []

    seen = set()

    for candidate in candidates:

        (
            row_offset,
            col_offset,
            height,
            offset_name,
            fit_score,
        ) = candidate

        key = (
            round(row_offset, 3),
            round(col_offset, 3),
            round(height, 2),
        )

        if key in seen:
            continue

        seen.add(key)

        unique_candidates.append(
            candidate
        )

    candidates = unique_candidates

    if not candidates:
        raise RuntimeError(
            "No RPC-frame candidates could be generated."
        )

    # ------------------------------------------------------------------
    # SMALL HEIGHT SEARCH
    # ------------------------------------------------------------------

    rpc_height = float(
        rpc["height_off"]
    )

    height_values = [

        rpc_height,

        0.0,

        25.0,

        50.0,

        75.0,

        100.0,

        150.0,

        200.0,
    ]

    heights = sorted(
        set(
            round(
                float(h),
                3,
            )
            for h in height_values
            if math.isfinite(
                float(h)
            )
        )
    )

    # ------------------------------------------------------------------
    # ONLY FIVE IMAGE CROP LOCATIONS
    # ------------------------------------------------------------------

    origins = [

        (
            0,
            0,
        ),

        (
            0,
            max_col,
        ),

        (
            max_row,
            0,
        ),

        (
            max_row,
            max_col,
        ),

        (
            max_row // 2,
            max_col // 2,
        ),
    ]

    origins = list(
        dict.fromkeys(
            origins
        )
    )

    # ------------------------------------------------------------------
    # FIVE CONTROL POINTS
    # ------------------------------------------------------------------

    positions = np.asarray(
        [
            [0.0, 0.0],

            [
                0.0,
                PATCH_SIZE - 1.0,
            ],

            [
                PATCH_SIZE - 1.0,
                0.0,
            ],

            [
                PATCH_SIZE - 1.0,
                PATCH_SIZE - 1.0,
            ],

            [
                (PATCH_SIZE - 1.0) / 2.0,
                (PATCH_SIZE - 1.0) / 2.0,
            ],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # PERSISTENT TRANSFORMER
    # ------------------------------------------------------------------

    transformer = _get_rpc_transformer(
        rpc_obj
    )

    best = None

    # ------------------------------------------------------------------
    # BOUNDED SEARCH
    # ------------------------------------------------------------------

    for (
        row_offset,
        col_offset,
        candidate_height,
        offset_name,
        fit_score,
    ) in candidates:

        # First test the candidate's own height.
        test_heights = [
            float(candidate_height)
        ]

        # Only add a few fallback heights.
        for height in heights:

            if abs(
                float(height)
                - float(candidate_height)
            ) < 1e-6:
                continue

            test_heights.append(
                float(height)
            )

        for height in test_heights:

            for (
                row0,
                col0,
            ) in origins:

                local_rows = (
                    float(row0)
                    + positions[:, 0]
                )

                local_cols = (
                    float(col0)
                    + positions[:, 1]
                )

                rpc_rows = (
                    local_rows
                    + float(row_offset)
                )

                rpc_cols = (
                    local_cols
                    + float(col_offset)
                )

                zs = np.full(
                    rpc_rows.shape,
                    float(height),
                    dtype=np.float64,
                )

                # ------------------------------------------------------
                # RPC INVERSE
                # ------------------------------------------------------

                try:

                    lons, lats = transformer.xy(
                        rpc_rows,
                        rpc_cols,
                        zs=zs,
                    )

                    lons = np.asarray(
                        lons,
                        dtype=np.float64,
                    )

                    lats = np.asarray(
                        lats,
                        dtype=np.float64,
                    )

                except Exception:
                    continue

                # ------------------------------------------------------
                # RPC VALIDITY
                # ------------------------------------------------------

                valid_rpc = (
                    np.isfinite(lats)
                    & np.isfinite(lons)
                    & (
                        np.abs(lats)
                        <= 90.0
                    )
                    & (
                        np.abs(lons)
                        <= 180.0
                    )
                )

                if not bool(
                    valid_rpc.any()
                ):
                    continue

                # ------------------------------------------------------
                # GEOGRAPHIC -> MOSAIC
                # ------------------------------------------------------

                try:

                    x, y = geographic_to_mosaic_xy(
                        lats,
                        lons,
                        mosaic,
                    )

                    mr, mc = mosaic_xy_to_pixel(
                        x,
                        y,
                        mosaic,
                    )

                except Exception:
                    continue

                mr = np.asarray(
                    mr,
                    dtype=np.float64,
                )

                mc = np.asarray(
                    mc,
                    dtype=np.float64,
                )

                # ------------------------------------------------------
                # GEOMETRIC OVERLAP
                # ------------------------------------------------------

                inside = (
                    valid_rpc
                    & np.isfinite(mr)
                    & np.isfinite(mc)
                    & (mr >= 0.0)
                    & (
                        mr
                        < mosaic.dataset.height
                    )
                    & (mc >= 0.0)
                    & (
                        mc
                        < mosaic.dataset.width
                    )
                )

                overlap = float(
                    inside.mean()
                )

                if overlap <= 0.0:
                    continue

                # ------------------------------------------------------
                # REAL ELEVATION VALIDITY
                # ------------------------------------------------------

                try:

                    target_valid = float(
                        point_valid_fraction(
                            mr,
                            mc,
                            mosaic,
                        )
                    )

                except Exception:
                    continue

                if target_valid <= 0.0:
                    continue

                # ------------------------------------------------------
                # SCORE
                # ------------------------------------------------------

                score = (
                    0.70
                    * target_valid
                    + 0.30
                    * overlap
                )

                # Slight preference for the observed tail frame.
                if offset_name.startswith(
                    "rpc-tail"
                ):
                    score += 0.05

                # Slight preference for fitted geometry.
                if offset_name.startswith(
                    "rpc-geographic-footprint-fit"
                ):
                    score += 0.05

                candidate = (
                    float(score),
                    float(target_valid),
                    float(overlap),
                    -abs(
                        float(height)
                        - rpc_height
                    ),
                    -float(fit_score),
                    -int(row0),
                    -int(col0),
                    int(row0),
                    int(col0),
                    float(row_offset),
                    float(col_offset),
                    float(height),
                    str(offset_name),
                )

                if (
                    best is None
                    or candidate > best
                ):
                    best = candidate

    # ------------------------------------------------------------------
    # NO VALID GEOMETRY
    # ------------------------------------------------------------------

    if best is None:

        raise RuntimeError(
            "No RPC-projected image crop overlaps "
            "the real elevation mosaic."
        )

    (
        _score,
        target_valid,
        overlap,
        _height_prior,
        _fit_score,
        _negative_row,
        _negative_col,
        row0,
        col0,
        row_offset,
        col_offset,
        height,
        offset_name,
    ) = best

    # ------------------------------------------------------------------
    # REAL-DATA SAFETY GATE
    # ------------------------------------------------------------------

    if (
        overlap < 0.05
        or target_valid < 0.05
    ):

        raise RuntimeError(
            "RPC/elevation overlap too small: "
            f"overlap={overlap:.3f}, "
            f"target_valid={target_valid:.3f}, "
            f"offset={offset_name}, "
            f"crop=({row0},{col0}), "
            f"height={height:.2f}"
        )

    return (
        int(row0),
        int(col0),
        float(row_offset),
        float(col_offset),
        float(height),
        str(offset_name),
        float(overlap),
        float(target_valid),
    )

# ============================================================================
# IMAGE / TARGET SAMPLE
# ============================================================================

def read_image_patch(
    path: Path,
    row: int,
    col: int,
) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(
            window=Window(
                col,
                row,
                PATCH_SIZE,
                PATCH_SIZE,
            )
        )

    if arr.ndim != 3:
        raise RuntimeError(
            f"Unexpected image array: {arr.shape}"
        )

    # Local SpaceNet MVS scene chips are single-band uint8.
    # Replicate to 3 channels for the existing DAV2 encoder.
    if arr.shape[0] == 1:
        arr = np.repeat(
            arr,
            3,
            axis=0,
        )

    elif arr.shape[0] >= 3:
        arr = arr[:3]

    else:
        raise RuntimeError(
            f"Unsupported image band count: {arr.shape[0]}"
        )

    arr = np.transpose(
        arr,
        (1, 2, 0),
    ).astype(np.float32)

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    if float(
        np.nanmax(arr)
    ) > 1.5:
        arr /= 255.0

    return np.clip(
        arr,
        0.0,
        1.0,
    ).astype(np.float32)


def clean_target(
    target: np.ndarray,
    mask: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    valid = (
        (mask > 0.5)
        & np.isfinite(target)
        & (target >= TARGET_MIN)
        & (target <= TARGET_MAX)
    )

    out = np.where(
        valid,
        target,
        0.0,
    ).astype(np.float32)

    return (
        out,
        valid.astype(np.float32),
    )


def build_sample(
    record: MVSRecord,
    mosaic: ElevationRaster,
    deterministic: bool,
) -> Dict[str, Any]:
    rpc = rpc_from_numbers(
        read_rpc_numbers(
            record.rpc_path
        )
    )

    rpc_obj = rpc_to_rasterio(
        rpc
    )

    (
        row0,
        col0,
        row_offset,
        col_offset,
        height,
        offset_name,
        overlap,
        target_valid_probe,
    ) = find_best_geometry(
        record,
        mosaic,
        rpc_obj,
        rpc,
    )

    image = read_image_patch(
        record.image_path,
        row0,
        col0,
    )

    rows, cols = build_rpc_target_grid(
        rpc_obj,
        row0,
        col0,
        row_offset,
        col_offset,
        height,
        mosaic,
    )

    target, mask = bilinear_sample_mosaic(
        mosaic,
        rows,
        cols,
    )

    target, mask = clean_target(
        target,
        mask,
    )

    valid_fraction = float(
        mask.mean()
    )

    if valid_fraction < MIN_VALID_FRACTION:
        raise RuntimeError(
            "Final elevation coverage too low: "
            f"{valid_fraction:.3f}"
        )

    valid_values = target[
        mask > 0.5
    ]

    if valid_values.size == 0:
        raise RuntimeError(
            "Final sample contains no valid elevation pixels."
        )

    return {
        "id": record.key,
        "region": record.region,
        "row": row0,
        "col": col0,
        "image": torch.from_numpy(
            np.ascontiguousarray(
                image
            )
        ).permute(
            2,
            0,
            1,
        ).float(),
        "target": torch.from_numpy(
            np.ascontiguousarray(
                target
            )
        ).unsqueeze(
            0
        ).float(),
        "mask": torch.from_numpy(
            np.ascontiguousarray(
                mask
            )
        ).unsqueeze(
            0
        ).float(),
        "valid_fraction": valid_fraction,
        "geo": {
            "rpc_offset_row": float(row_offset),
            "rpc_offset_col": float(col_offset),
            "rpc_offset_mode": offset_name,
            "rpc_height": float(height),
            "rpc_mosaic_overlap": float(overlap),
            "rpc_target_valid_probe": float(
                target_valid_probe
            ),
            "target_crs": str(mosaic.crs),
            "target_min": float(
                valid_values.min()
            ),
            "target_max": float(
                valid_values.max()
            ),
            "target_mean": float(
                valid_values.mean()
            ),
        },
    }



def try_build_sample(
    record: MVSRecord,
    mosaics: Dict[str, ElevationRaster],
    deterministic: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Try the record's nominal mosaic first, then every available mosaic.

    A sample is accepted only when the RPC projection produces real,
    non-nodata elevation pixels.
    """
    if not mosaics:
        print(f"[WARNING] No elevation mosaics are open for {record.key}")
        return None

    ordered = list(mosaics.items())
    nominal = getattr(record, "region", None)
    ordered.sort(key=lambda kv: 0 if kv[0] == nominal else 1)

    failures = []

    for mosaic_name, mosaic in ordered:
        try:
            sample = build_sample(record, mosaic, deterministic)

            if sample is None:
                failures.append(f"{mosaic_name}: returned None")
                continue

            vf = float(sample.get("valid_fraction", 0.0))
            if vf <= 0.0:
                failures.append(f"{mosaic_name}: valid_fraction={vf:.6f}")
                continue

            sample["mosaic_used"] = mosaic_name
            sample["original_region"] = nominal
            sample["geo"]["mosaic_used"] = mosaic_name
            return sample

        except Exception as exc:
            failures.append(
                f"{mosaic_name}: {type(exc).__name__}: {exc}"
            )

    print(
        f"[WARNING] Could not build sample {record.key}: "
        f"all mosaics failed. {' | '.join(failures[-6:])}"
    )
    return None


# ============================================================================
# VALIDATION SPLIT
# ============================================================================

def deterministic_rank(
    key: str,
) -> str:
    return hashlib.sha1(
        f"{SEED}:{key}".encode(
            "utf-8"
        )
    ).hexdigest()


def load_or_create_split(
    records: Sequence[MVSRecord],
) -> Tuple[
    List[MVSRecord],
    List[MVSRecord],
]:
    by_key = {
        r.key: r
        for r in records
    }

    keys = set(by_key)

    if VALIDATION_SPLIT_FILE.exists():
        try:
            payload = json.loads(
                VALIDATION_SPLIT_FILE.read_text(
                    encoding="utf-8"
                )
            )

            train_keys = set(
                map(
                    str,
                    payload.get(
                        "train_keys",
                        [],
                    ),
                )
            )

            val_keys = set(
                map(
                    str,
                    payload.get(
                        "validation_keys",
                        [],
                    ),
                )
            )

            if (
                train_keys
                and val_keys
                and train_keys.isdisjoint(
                    val_keys
                )
                and (
                    train_keys
                    | val_keys
                ) == keys
            ):
                print(
                    "[OK] Reusing persistent "
                    "Stage-5 validation split."
                )

                return (
                    [
                        by_key[k]
                        for k in sorted(
                            train_keys
                        )
                    ],
                    [
                        by_key[k]
                        for k in sorted(
                            val_keys
                        )
                    ],
                )

        except Exception as exc:
            print(
                "[WARNING] Could not reuse "
                "validation split: "
                f"{type(exc).__name__}: {exc}"
            )

    ranked = sorted(
        keys,
        key=deterministic_rank,
    )

    nval = max(
        1,
        int(
            round(
                len(ranked)
                * VAL_FRACTION
            )
        ),
    )

    nval = min(
        nval,
        max(
            0,
            len(ranked) - 1,
        ),
    )

    val_keys = ranked[:nval]
    train_keys = ranked[nval:]

    atomic_json_write(
        VALIDATION_SPLIT_FILE,
        {
            "version": 4,
            "stage": 5,
            "dataset": "SpaceNet MVS",
            "seed": SEED,
            "fraction": VAL_FRACTION,
            "train_keys": train_keys,
            "validation_keys": val_keys,
        },
    )

    print(
        "[OK] Created persistent "
        "Stage-5 validation split."
    )

    return (
        [
            by_key[k]
            for k in train_keys
        ],
        [
            by_key[k]
            for k in val_keys
        ],
    )


# ============================================================================
# MODEL
# ============================================================================

def create_model(
    checkpoint: Path,
) -> nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    # IMPORTANT:
    # The installed project DepthAnythingV2 constructor used in Stage 3/4
    # does not require max_depth. Keep the same architecture.
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

    if isinstance(payload, dict):
        state = payload.get(
            "model_state_dict",
            payload,
        )
    else:
        state = payload

    if not isinstance(state, dict):
        raise RuntimeError(
            "Checkpoint does not contain a usable state_dict."
        )

    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=False,
        )
    )

    if missing:
        raise RuntimeError(
            "Checkpoint is incompatible with Stage-5 model. "
            f"Missing keys: {missing[:10]}"
        )

    if unexpected:
        print(
            f"[WARNING] Unexpected checkpoint keys: "
            f"{len(unexpected)}"
        )

    epoch = (
        payload.get("epoch")
        if isinstance(payload, dict)
        else None
    )

    best = (
        payload.get("best_val_mae")
        if isinstance(payload, dict)
        else None
    )

    print(
        f"[OK] Loaded model checkpoint: "
        f"{checkpoint.name} | "
        f"epoch={epoch} | "
        f"best_val_mae={best}"
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
            except Exception:
                pass

    print(
        "[INFO] Gradient checkpointing API not exposed; "
        "continuing."
    )


def make_all_trainable(
    model: nn.Module,
) -> None:
    for p in model.parameters():
        p.requires_grad_(True)

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"[PARAMETERS] total={total:,} | "
        f"trainable={trainable:,} | "
        f"{100.0 * trainable / total:.2f}%"
    )

    if trainable != total:
        raise RuntimeError(
            "Stage-5 requires full-parameter fine-tuning."
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
                "Unknown model output dict keys: "
                f"{list(out.keys())}"
            )

    if isinstance(
        out,
        (tuple, list),
    ):
        out = out[0]

    if not torch.is_tensor(out):
        raise RuntimeError(
            f"Model output is not a tensor: "
            f"{type(out)}"
        )

    if out.ndim == 3:
        out = out.unsqueeze(1)

    if out.ndim != 4:
        raise RuntimeError(
            "Unexpected model output shape: "
            f"{tuple(out.shape)}"
        )

    if out.shape[1] != 1:
        out = out[:, :1]

    return out


# ============================================================================
# OPTIMIZATION
# ============================================================================

def create_optimizer(
    model: nn.Module,
):
    try:
        from transformers.optimization import Adafactor

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

    except Exception:
        print(
            "[WARNING] Adafactor unavailable; "
            "using AdamW."
        )

        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )


def amp_context():
    if (
        torch.cuda.is_available()
        and AMP_ENABLED
    ):
        return torch.autocast(
            "cuda",
            dtype=torch.float16,
        )

    return torch.autocast(
        "cpu",
        enabled=False,
    )


def create_scaler():
    if (
        not torch.cuda.is_available()
        or not AMP_ENABLED
    ):
        return None

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=True,
        )


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = (
        (mask > 0.5)
        & torch.isfinite(prediction)
        & torch.isfinite(target)
    )

    if not valid.any():
        raise RuntimeError(
            "Masked L1 received zero valid pixels."
        )

    return torch.abs(
        prediction[valid]
        - target[valid]
    ).mean()


def resize_target_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    shape: Tuple[int, int],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    return (
        F.interpolate(
            target,
            size=shape,
            mode="bilinear",
            align_corners=False,
        ),
        F.interpolate(
            mask,
            size=shape,
            mode="nearest",
        ),
    )


def safe_optimizer_step(
    model: nn.Module,
    optimizer,
    scaler,
) -> bool:
    has_grad = any(
        p.requires_grad
        and p.grad is not None
        for p in model.parameters()
    )

    if not has_grad:
        optimizer.zero_grad(
            set_to_none=True
        )
        return False

    if scaler is not None:
        scaler.unscale_(
            optimizer
        )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    if scaler is not None:
        scaler.step(
            optimizer
        )
        scaler.update()

    else:
        optimizer.step()

    optimizer.zero_grad(
        set_to_none=True
    )

    return True


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
    history: Sequence[Dict[str, Any]],
    reason: str,
) -> None:
    payload: Dict[str, Any] = {
        "stage": 5,
        "dataset": "SpaceNet MVS",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": float(best_mae),
        "history": list(history),
        "reason": reason,
        "config": {
            "encoder": ENCODER,
            "features": FEATURES,
            "out_channels": OUT_CHANNELS,
            "patch_size": PATCH_SIZE,
            "model_size": MODEL_SIZE,
            "dino_patch_size": DINO_PATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "grad_accumulation": GRAD_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "amp": AMP_ENABLED,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "full_parameter_finetuning": True,
            "rpc_geometry": (
                "geographic-footprint-fit + "
                "elevation-validity scoring"
            ),
            "initial_checkpoint": str(
                STAGE4_BEST
            ),
        },
    }

    if scaler is not None:
        try:
            payload[
                "scaler_state_dict"
            ] = scaler.state_dict()
        except Exception:
            payload[
                "scaler_state_dict"
            ] = None
    else:
        payload[
            "scaler_state_dict"
        ] = None

    atomic_torch_save(
        payload,
        path,
    )


# ============================================================================
# PREFLIGHT
# ============================================================================

def preflight(
    records: Sequence[MVSRecord],
    mosaics: Dict[str, ElevationRaster],
) -> bool:
    print()
    print("=" * 80)
    print("STAGE-5 PREFLIGHT")
    print("=" * 80)

    if not DATASET_ROOT.exists():
        print(
            "[FAIL] Dataset directory missing."
        )
        return False

    print(
        "[OK] Dataset directory exists."
    )

    if not STAGE4_BEST.exists():
        print(
            f"[FAIL] Stage-4 checkpoint missing: "
            f"{STAGE4_BEST}"
        )
        return False

    print(
        "[OK] Stage-4 checkpoint exists."
    )

    print()
    print(
        f"[DISCOVERY] Image TIFFs : "
        f"{len(discover_image_tiffs())}"
    )
    print(
        f"[DISCOVERY] RPC files   : "
        f"{len(discover_rpc_files())}"
    )
    print(
        f"[DISCOVERY] Training records: "
        f"{len(records)}"
    )

    print_elevation_inventory(
        mosaics
    )

    if not records:
        print(
            "[FAIL] No valid image/RPC records."
        )
        return False

    train, val = load_or_create_split(
        records
    )

    print()
    print(
        f"[SPLIT] Train records: "
        f"{len(train)}"
    )
    print(
        f"[SPLIT] Validation records: "
        f"{len(val)}"
    )

    print()
    print(
        "AUDITING ACTUAL IMAGE + RPC + ELEVATION COVERAGE"
    )

    # A single successful sample is not sufficient. It can be produced by an
    # incorrect geometry fallback while almost the entire split is rejected.
    # Audit both splits through the exact same build_sample() path used by
    # training and validation.
    sample = None
    usable_by_split = {}

    for split_name, split_records in (
        ("train", train),
        ("validation", val),
    ):
        usable = 0

        for idx, record in enumerate(
            split_records,
            start=1,
        ):
            candidate = try_build_sample(
                record,
                mosaics,
                deterministic=True,
            )

            if candidate is not None:
                usable += 1
                if sample is None:
                    sample = candidate

            if idx % 25 == 0 or idx == len(split_records):
                print(
                    f"[PREFLIGHT] {split_name}: "
                    f"checked={idx}/{len(split_records)} | "
                    f"usable={usable}"
                )

        usable_by_split[split_name] = usable

    minimum_train = max(
        1,
        math.ceil(0.50 * len(train)),
    )
    minimum_val = max(
        1,
        math.ceil(0.50 * len(val)),
    )

    print()
    print(
        f"[PREFLIGHT] Usable train samples: "
        f"{usable_by_split['train']}/{len(train)} "
        f"(minimum {minimum_train})"
    )
    print(
        f"[PREFLIGHT] Usable validation samples: "
        f"{usable_by_split['validation']}/{len(val)} "
        f"(minimum {minimum_val})"
    )

    if (
        sample is None
        or usable_by_split["train"] < minimum_train
        or usable_by_split["validation"] < minimum_val
    ):
        print(
            "[ERROR] Real image/RPC/elevation coverage is too low."
        )
        print(
            "[ERROR] Training must NOT start with this geometry."
        )
        return False

    print()
    print(
        "[OK] Sample constructed."
    )

    print(
        f"      ID             : "
        f"{sample['id']}"
    )

    print(
        f"      Region         : "
        f"{sample['region']}"
    )

    print(
        f"      Crop origin    : "
        f"({sample['row']}, "
        f"{sample['col']})"
    )

    print(
        f"      Image shape    : "
        f"{tuple(sample['image'].shape)}"
    )

    print(
        f"      Target shape   : "
        f"{tuple(sample['target'].shape)}"
    )

    print(
        f"      Mask shape     : "
        f"{tuple(sample['mask'].shape)}"
    )

    print(
        f"      Valid fraction : "
        f"{sample['valid_fraction']:.3f}"
    )

    print(
        f"      Target range   : "
        f"{sample['geo']['target_min']:.3f} .. "
        f"{sample['geo']['target_max']:.3f}"
    )

    print(
        f"      Target CRS     : "
        f"{sample['geo']['target_crs']}"
    )

    print(
        f"      RPC offset row : "
        f"{sample['geo']['rpc_offset_row']:.3f}"
    )

    print(
        f"      RPC offset col : "
        f"{sample['geo']['rpc_offset_col']:.3f}"
    )

    print(
        f"      RPC offset mode: "
        f"{sample['geo']['rpc_offset_mode']}"
    )

    print(
        f"      RPC height     : "
        f"{sample['geo']['rpc_height']:.2f}"
    )

    print(
        f"      RPC overlap    : "
        f"{sample['geo']['rpc_mosaic_overlap']:.3f}"
    )

    print(
        f"      RPC valid probe: "
        f"{sample['geo']['rpc_target_valid_probe']:.3f}"
    )

    print()
    print(
        "[OK] Geographic-footprint RPC fitting passed."
    )
    print(
        "[OK] Source TIFF affine transform was NOT used."
    )
    print(
        "[OK] 512x512 crops are resized to 518x518."
    )
    print(
        "[OK] 518 / 14 = 37."
    )
    print(
        "[OK] Training is allowed to start."
    )

    return True


# ============================================================================
# GPU SMOKE TEST
# ============================================================================

def gpu_smoke_test(
    records: Sequence[MVSRecord],
    mosaics: Dict[str, ElevationRaster],
) -> bool:
    print()
    print("=" * 80)
    print("STAGE-5 GPU SMOKE TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        print(
            "[ERROR] CUDA unavailable."
        )
        return False

    sample = None

    for record in records:
        sample = try_build_sample(
            record,
            mosaics,
            deterministic=True,
        )

        if sample is not None:
            break

    if sample is None:
        print(
            "[ERROR] No usable Stage-5 sample."
        )
        return False

    model = None
    optimizer = None
    scaler = None

    try:
        initial = (
            LATEST_CHECKPOINT
            if LATEST_CHECKPOINT.exists()
            else STAGE4_BEST
        )

        model = create_model(
            initial
        )

        enable_gradient_checkpointing(
            model
        )

        make_all_trainable(
            model
        )

        device = get_device()
        model.to(device)

        optimizer = create_optimizer(
            model
        )

        scaler = create_scaler()

        optimizer.zero_grad(
            set_to_none=True
        )

        image = sample[
            "image"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        target = sample[
            "target"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        mask = sample[
            "mask"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        image_model = F.interpolate(
            image,
            size=(
                MODEL_SIZE,
                MODEL_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        with amp_context():
            prediction = model_forward(
                model,
                image_model,
            )

            target_r, mask_r = (
                resize_target_mask(
                    target,
                    mask,
                    prediction.shape[-2:],
                )
            )

            loss = masked_l1(
                prediction,
                target_r,
                mask_r,
            )

        print(
            f"Smoke sample: "
            f"{sample['id']}"
        )

        print(
            f"Forward loss: "
            f"{float(loss.detach().cpu()):.6f}"
        )

        scaled = (
            loss
            / GRAD_ACCUMULATION
        )

        if scaler is not None:
            scaler.scale(
                scaled
            ).backward()
        else:
            scaled.backward()

        if not safe_optimizer_step(
            model,
            optimizer,
            scaler,
        ):
            raise RuntimeError(
                "GPU smoke test produced no gradients."
            )

        print(
            "[OK] Forward + backward + "
            "optimizer step succeeded."
        )

        print_gpu_memory(
            "after GPU smoke test"
        )

        return True

    except torch.cuda.OutOfMemoryError:
        print(
            "[ERROR] CUDA OOM during GPU smoke test."
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: nn.Module,
    optimizer,
    scaler,
    records: Sequence[MVSRecord],
    mosaics: Dict[str, ElevationRaster],
    device: torch.device,
    max_steps: int,
    epoch: int,
    fallback_records: Optional[Sequence[MVSRecord]] = None,
) -> Tuple[float, int]:
    model.train()

    order = list(records)
    random.Random(
        SEED + epoch
    ).shuffle(
        order
    )

    if max_steps > 0:
        order = order[:max_steps]

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_sum = 0.0
    successful = 0
    accumulation = 0

    for record in order:
        sample = try_build_sample(
            record,
            mosaics,
            deterministic=False,
        )

        if sample is None:
            continue

        image = sample[
            "image"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        target = sample[
            "target"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        mask = sample[
            "mask"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        try:
            image_model = F.interpolate(
                image,
                size=(
                    MODEL_SIZE,
                    MODEL_SIZE,
                ),
                mode="bilinear",
                align_corners=False,
            )

            with amp_context():
                prediction = model_forward(
                    model,
                    image_model,
                )

                target_r, mask_r = (
                    resize_target_mask(
                        target,
                        mask,
                        prediction.shape[-2:],
                    )
                )

                loss = masked_l1(
                    prediction,
                    target_r,
                    mask_r,
                )

            scaled = (
                loss
                / GRAD_ACCUMULATION
            )

            if scaler is not None:
                scaler.scale(
                    scaled
                ).backward()
            else:
                scaled.backward()

            accumulation += 1
            successful += 1

            loss_sum += float(
                loss.detach().cpu()
            )

            if (
                accumulation
                >= GRAD_ACCUMULATION
            ):
                safe_optimizer_step(
                    model,
                    optimizer,
                    scaler,
                )
                accumulation = 0

            if (
                successful == 1
                or successful % PRINT_EVERY == 0
            ):
                print(
                    f"Train {successful}/"
                    f"{len(order)} | "
                    f"loss="
                    f"{loss_sum / successful:.6f} | "
                    f"sample={record.key}"
                )

        finally:
            del image
            del target
            del mask
            del sample

            gc.collect()

    if accumulation > 0:
        safe_optimizer_step(
            model,
            optimizer,
            scaler,
        )

    if successful == 0:
        if fallback_records is not None:
            fallback = list(fallback_records)

            if fallback and len(fallback) != len(order):
                print(
                    "[WARNING] Training split produced zero usable "
                    "samples; retrying with all discovered records."
                )

                return train_epoch(
                    model,
                    optimizer,
                    scaler,
                    fallback,
                    mosaics,
                    device,
                    max_steps,
                    epoch,
                    fallback_records=None,
                )

        raise RuntimeError(
            "Training epoch produced zero usable samples."
        )

    return (
        loss_sum / successful,
        successful,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    records: Sequence[MVSRecord],
    mosaics: Dict[str, ElevationRaster],
    device: torch.device,
    max_samples: int,
) -> Dict[str, Any]:
    model.eval()

    order = list(records)

    total_abs = 0.0
    total_sq = 0.0
    valid_pixels = 0
    samples = 0
    scanned = 0

    for record in order:
        if (
            max_samples > 0
            and samples >= max_samples
        ):
            break

        scanned += 1

        sample = try_build_sample(
            record,
            mosaics,
            deterministic=True,
        )

        if sample is None:
            continue

        image = sample[
            "image"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        target = sample[
            "target"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        mask = sample[
            "mask"
        ].unsqueeze(
            0
        ).to(
            device,
            non_blocking=True,
        )

        try:
            image_model = F.interpolate(
                image,
                size=(
                    MODEL_SIZE,
                    MODEL_SIZE,
                ),
                mode="bilinear",
                align_corners=False,
            )

            with amp_context():
                prediction = model_forward(
                    model,
                    image_model,
                )

                target_r, mask_r = (
                    resize_target_mask(
                        target,
                        mask,
                        prediction.shape[-2:],
                    )
                )

            prediction = prediction.float()
            target_r = target_r.float()
            mask_r = mask_r.float()

            valid = (
                (mask_r > 0.5)
                & torch.isfinite(
                    prediction
                )
                & torch.isfinite(
                    target_r
                )
            )

            count = int(
                valid.sum().item()
            )

            if count == 0:
                continue

            diff = (
                prediction
                - target_r
            )

            total_abs += float(
                torch.abs(
                    diff
                )[valid].sum().cpu()
            )

            total_sq += float(
                torch.square(
                    diff
                )[valid].sum().cpu()
            )

            valid_pixels += count
            samples += 1

            if (
                samples == 1
                or samples % PRINT_EVERY == 0
            ):
                mae = (
                    total_abs
                    / max(
                        valid_pixels,
                        1,
                    )
                )

                rmse = math.sqrt(
                    total_sq
                    / max(
                        valid_pixels,
                        1,
                    )
                )

                print(
                    f"Validation "
                    f"{samples}/"
                    f"{max_samples if max_samples > 0 else len(order)} | "
                    f"MAE={mae:.6f} | "
                    f"RMSE={rmse:.6f} | "
                    f"scanned={scanned}"
                )

        finally:
            del image
            del target
            del mask
            del sample

    if (
        samples == 0
        or valid_pixels == 0
    ):
        raise RuntimeError(
            "Validation produced zero usable pixels."
        )

    return {
        "mae": (
            total_abs
            / valid_pixels
        ),
        "rmse": math.sqrt(
            total_sq
            / valid_pixels
        ),
        "valid_pixels": valid_pixels,
        "samples": samples,
        "scanned": scanned,
    }


def run_training(
    records: Sequence[MVSRecord],
    mosaics: Dict[str, ElevationRaster],
    epochs: int,
    steps: int,
    val_steps: int,
    from_stage4: bool = False,
) -> None:
    device = get_device()

    train_records, val_records = (
        load_or_create_split(
            records
        )
    )

    if not preflight(
        records,
        mosaics,
    ):
        raise RuntimeError(
            "Stage-5 preflight failed. "
            "Training was not started."
        )

    # An explicit Stage-4 start is useful when a previous Stage-5 run is
    # present. It keeps the old checkpoints recoverable and avoids requiring
    # the user to delete them just to restart fine-tuning.
    resume_stage5 = (
        LATEST_CHECKPOINT.exists()
        and not from_stage4
    )

    initial = (
        LATEST_CHECKPOINT
        if resume_stage5
        else STAGE4_BEST
    )

    if resume_stage5:
        print()
        print(
            "[RESUME] Using existing "
            "Stage-5 latest checkpoint."
        )
    else:
        print()
        print(
            "[INIT] Starting Stage-5 from "
            "Stage-4 best checkpoint."
        )

    model = create_model(
        initial
    )

    enable_gradient_checkpointing(
        model
    )

    make_all_trainable(
        model
    )

    model.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_gpu_memory(
        "after model loading"
    )

    optimizer = create_optimizer(
        model
    )

    scaler = create_scaler()

    history: List[
        Dict[str, Any]
    ] = []

    best_mae = float(
        "inf"
    )

    if resume_stage5:
        try:
            payload = torch.load(
                LATEST_CHECKPOINT,
                map_location="cpu",
                weights_only=False,
            )

            if isinstance(
                payload,
                dict,
            ):
                previous_history = (
                    payload.get(
                        "history"
                    )
                )

                if isinstance(
                    previous_history,
                    list,
                ):
                    history = previous_history

                previous_best = payload.get(
                    "best_val_mae"
                )

                if previous_best is not None:
                    best_mae = float(
                        previous_best
                    )

                optimizer_state = (
                    payload.get(
                        "optimizer_state_dict"
                    )
                )

                if isinstance(
                    optimizer_state,
                    dict,
                ):
                    try:
                        optimizer.load_state_dict(
                            optimizer_state
                        )
                        print(
                            "[OK] Resumed "
                            "Stage-5 optimizer state."
                        )
                    except Exception as exc:
                        print(
                            "[WARNING] Could not "
                            "resume optimizer state: "
                            f"{type(exc).__name__}: {exc}"
                        )

                scaler_state = (
                    payload.get(
                        "scaler_state_dict"
                    )
                )

                if (
                    scaler is not None
                    and isinstance(
                        scaler_state,
                        dict,
                    )
                ):
                    try:
                        scaler.load_state_dict(
                            scaler_state
                        )
                    except Exception:
                        pass

        except Exception as exc:
            print(
                "[WARNING] Could not read "
                "Stage-5 resume metadata: "
                f"{type(exc).__name__}: {exc}"
            )

    start_epoch = (
        len(history) + 1
    )

    try:
        for epoch in range(
            start_epoch,
            epochs + 1,
        ):
            print()
            print(
                "=" * 80
            )
            print(
                f"STAGE-5 EPOCH "
                f"{epoch}/{epochs}"
            )
            print(
                "=" * 80
            )

            train_loss, train_count = (
                train_epoch(
                    model,
                    optimizer,
                    scaler,
                    train_records,
                    mosaics,
                    device,
                    steps,
                    epoch,
                    fallback_records=records,
                )
            )

            print()
            print(
                f"Epoch {epoch} "
                f"training loss: "
                f"{train_loss:.6f}"
            )

            print(
                f"Epoch {epoch} "
                f"usable training samples: "
                f"{train_count}"
            )

            validation = validate(
                model,
                val_records,
                mosaics,
                device,
                val_steps,
            )

            val_mae = float(
                validation["mae"]
            )

            epoch_info = {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "train_samples": int(
                    train_count
                ),
                "validation": validation,
                "best_validation_mae": (
                    float(best_mae)
                    if math.isfinite(
                        best_mae
                    )
                    else None
                ),
                "timestamp": time.time(),
            }

            history.append(
                epoch_info
            )

            atomic_json_write(
                HISTORY_FILE,
                history,
            )

            save_checkpoint(
                LATEST_CHECKPOINT,
                model,
                optimizer,
                scaler,
                epoch,
                best_mae,
                history,
                "epoch_complete",
            )

            if val_mae < best_mae:
                best_mae = val_mae

                history[-1][
                    "best_validation_mae"
                ] = float(
                    best_mae
                )

                atomic_json_write(
                    HISTORY_FILE,
                    history,
                )

                save_checkpoint(
                    BEST_CHECKPOINT,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_mae,
                    history,
                    "new_best_validation_mae",
                )

                print()
                print(
                    "=" * 80
                )
                print(
                    "NEW BEST STAGE-5 CHECKPOINT"
                )
                print(
                    f"Validation MAE: "
                    f"{best_mae:.6f} m"
                )
                print(
                    "=" * 80
                )

            print()
            print(
                f"[EPOCH {epoch}] "
                f"Train loss="
                f"{train_loss:.6f} | "
                f"Val MAE="
                f"{val_mae:.6f} | "
                f"Val RMSE="
                f"{float(validation['rmse']):.6f}"
            )

            print_gpu_memory(
                f"end of epoch {epoch}"
            )

            # Free cached CUDA memory between epochs.
            gc.collect()
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print()
        print(
            "=" * 80
        )
        print(
            "STAGE-5 TRAINING INTERRUPTED"
        )
        print(
            "=" * 80
        )

        save_checkpoint(
            EMERGENCY_CHECKPOINT,
            model,
            optimizer,
            scaler,
            start_epoch,
            best_mae,
            history,
            "keyboard_interrupt",
        )

        print(
            f"[OK] Emergency checkpoint saved: "
            f"{EMERGENCY_CHECKPOINT}"
        )

        raise

    finally:
        del model
        del optimizer
        del scaler

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print(
        "=" * 80
    )
    print(
        "STAGE-5 TRAINING FINISHED"
    )
    print(
        "=" * 80
    )
    print(
        f"Best checkpoint   : "
        f"{BEST_CHECKPOINT}"
    )
    print(
        f"Latest checkpoint : "
        f"{LATEST_CHECKPOINT}"
    )
    print(
        f"History           : "
        f"{HISTORY_FILE}"
    )
    print(
        f"Validation split  : "
        f"{VALIDATION_SPLIT_FILE}"
    )

    if math.isfinite(
        best_mae
    ):
        print(
            f"Best validation MAE: "
            f"{best_mae:.6f} m"
        )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ASTERRA Stage-5 SpaceNet MVS "
            "RPC-footprint-fitted elevation trainer"
        )
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Run full dataset/RPC/elevation preflight "
            "without training."
        ),
    )

    # Accept the natural-language invocation used during troubleshooting:
    # `python train_stage5.py --preflight test`. The word is intentionally
    # optional so the original `--preflight` command remains valid.
    parser.add_argument(
        "preflight_mode",
        nargs="?",
        choices=("test",),
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--gpu-smoke-test",
        action="store_true",
        help=(
            "Run one real Stage-5 forward/backward/optimizer step."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=(
            "Training records per epoch. "
            "0 means all discovered training records."
        ),
    )

    parser.add_argument(
        "--val-steps",
        type=int,
        default=DEFAULT_VAL_STEPS,
        help=(
            "Maximum validation samples per epoch."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            "Number of Stage-5 epochs."
        ),
    )

    parser.add_argument(
        "--from-stage4",
        action="store_true",
        help=(
            "Start a fresh Stage-5 fine-tuning run from stage4_best.pth "
            "even when stage5_latest.pth already exists."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    if args.preflight_mode == "test":
        args.preflight = True

    seed_everything()

    if args.steps < 0:
        raise ValueError(
            "--steps must be >= 0."
        )

    if args.val_steps <= 0:
        raise ValueError(
            "--val-steps must be > 0."
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be > 0."
        )

    if MODEL_SIZE % DINO_PATCH_SIZE != 0:
        raise RuntimeError(
            f"MODEL_SIZE={MODEL_SIZE} is not divisible by "
            f"DINO patch size={DINO_PATCH_SIZE}."
        )

    print_header()

    records = discover_records()

    if not records:
        raise RuntimeError(
            "No SpaceNet MVS image/RPC records were discovered."
        )

    save_manifest(
        records
    )

    mosaics = open_elevation_rasters()

    try:
        if not preflight(
            records,
            mosaics,
        ):
            raise SystemExit(1)

        if args.preflight:
            print()
            print(
                "[OK] Preflight complete. "
                "Training was not started."
            )
            return

        if args.gpu_smoke_test:
            ok = gpu_smoke_test(
                records,
                mosaics,
            )

            raise SystemExit(
                0 if ok else 1
            )

        run_training(
            records=records,
            mosaics=mosaics,
            epochs=args.epochs,
            steps=args.steps,
            val_steps=args.val_steps,
            from_stage4=args.from_stage4,
        )

    finally:
        for raster in mosaics.values():
            try:
                raster.dataset.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
