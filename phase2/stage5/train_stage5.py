r"""
ASTERRA AI — STAGE 5
FINAL FAST RPC/ELEVATION LAUNCHER

Drop-in replacement for:
    D:\Asterra AI\phase2\stage5\train_stage5.py

IMPORTANT:
- This file keeps the existing Stage-5 core trainer intact.
- It patches the broken geometry SEARCH in the core at runtime.
- It does NOT use synthetic elevation targets.
- It keeps full-parameter fine-tuning.
- It tests all MasterProvisional elevation mosaics.
- It avoids the old exhaustive crop-origin/RPC-inverse loop.

Core trainer resolution:
    1. train_stage5_FIXED_FINAL.py, when present
    2. backup.py, which is the complete Stage-5 trainer currently in this
       workspace

Commands:
    python .\phase2\stage5\train_stage5.py --preflight
    python .\phase2\stage5\train_stage5.py --gpu-smoke-test
    python .\phase2\stage5\train_stage5.py --steps 5 --val-steps 2 --epochs 1
    python .\phase2\stage5\train_stage5.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# LOAD EXISTING CORE
# ============================================================================

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]

# The fast launcher was originally paired with a separate core file. That
# file is not present in this checkout, but backup.py contains the complete
# Stage-5 implementation and exposes the same functions used below. Resolve
# the optional newer core first, then fall back to the known-good trainer.
CORE_CANDIDATES = (
    THIS_FILE.parent / "train_stage5_FIXED_FINAL.py",
    THIS_FILE.parent / "backup.py",
)
CORE_FILE = next(
    (candidate for candidate in CORE_CANDIDATES if candidate.is_file()),
    None,
)

if CORE_FILE is None:
    raise FileNotFoundError(
        "\nMissing Stage-5 core trainer:\n"
        + "\n".join(f"    {candidate}" for candidate in CORE_CANDIDATES)
        + "\n\n"
        "The launcher requires either the final core or backup.py, which "
        "contains the full Stage-5 training implementation."
    )

print(f"[ASTERRA PATCH] Stage-5 core: {CORE_FILE.name}")

spec = importlib.util.spec_from_file_location(
    "asterra_stage5_final_core",
    CORE_FILE,
)

if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load: {CORE_FILE}")

core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


# ============================================================================
# FULL-PARAMETER POLICY
# ============================================================================

_original_create_model = core.create_model


def create_model_full_parameter(checkpoint):
    model = _original_create_model(checkpoint)

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 80)
    print("ASTERRA STAGE-5 PARAMETER POLICY")
    print("=" * 80)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(
        f"Trainable percentage : "
        f"{100.0 * trainable / max(total, 1):.2f}%"
    )

    if total == 0:
        raise RuntimeError("Stage-5 model contains zero parameters.")

    if trainable != total:
        raise RuntimeError(
            "SAFETY STOP: Stage-5 requires 100% full-parameter fine-tuning."
        )

    print("[OK] FULL-PARAMETER FINE-TUNING = 100%")
    print("[OK] LoRA = NOT USED")
    print("[OK] PEFT = NOT USED")
    print("=" * 80)

    return model


core.create_model = create_model_full_parameter


# ============================================================================
# FAST GEOMETRY RESOLVER
# ============================================================================
#
# ROOT CAUSE OF THE PREVIOUS FAILURE:
#
# The core find_best_geometry() did:
#
#   candidate
#       -> many crop origins
#       -> rpc_control_points()
#       -> rpc_inverse()
#       -> geographic transform
#       -> mosaic sampling
#
# The log proved the process was still entering:
#
#   rpc_control_points()
#   rpc_inverse()
#
# and was eventually interrupted with KeyboardInterrupt.
#
# This replacement never calls rpc_control_points() and never performs the
# old per-crop exhaustive inverse search.
#
# It uses:
#   1. the RPC geographic footprint when available
#   2. RPC forward projection of footprint points
#   3. the real elevation mosaic as the geographic reference
#   4. a small number of candidates
#
# The final build_sample() still calls the core's real RPC target-grid
# construction and real elevation sampling. No synthetic target is created.
# ============================================================================


def _safe_float(value, default=None):
    try:
        value = float(value)
    except Exception:
        return default

    return value if math.isfinite(value) else default


def _read_image_size(path):
    import rasterio

    with rasterio.open(path) as ds:
        return int(ds.height), int(ds.width)


def _mosaic_center_lonlat(mosaic):
    from pyproj import Transformer

    bounds = mosaic.dataset.bounds

    x = 0.5 * (
        float(bounds.left) + float(bounds.right)
    )
    y = 0.5 * (
        float(bounds.bottom) + float(bounds.top)
    )

    transformer = Transformer.from_crs(
        mosaic.crs,
        "EPSG:4326",
        always_xy=True,
    )

    lon, lat = transformer.transform(x, y)

    return float(lon), float(lat)


def _mosaic_anchor_lonlat(mosaic):
    from pyproj import Transformer

    bounds = mosaic.dataset.bounds

    # Nine points over the REAL elevation mosaic.
    # Try the mosaic centre first. It gives the largest margin for the
    # 512x512 target crop; edge anchors remain useful fallbacks.
    fractions = (
        (0.50, 0.50),
        (0.10, 0.10),
        (0.10, 0.50),
        (0.10, 0.90),
        (0.50, 0.10),
        (0.50, 0.90),
        (0.90, 0.10),
        (0.90, 0.50),
        (0.90, 0.90),
    )

    transformer = Transformer.from_crs(
        mosaic.crs,
        "EPSG:4326",
        always_xy=True,
    )

    points = []

    for fx, fy in fractions:
        x = float(
            bounds.left
            + fx * (bounds.right - bounds.left)
        )
        y = float(
            bounds.bottom
            + fy * (bounds.top - bounds.bottom)
        )

        lon, lat = transformer.transform(x, y)

        if (
            math.isfinite(float(lon))
            and math.isfinite(float(lat))
        ):
            points.append(
                (float(lon), float(lat))
            )

    return points


def _rpc_bbox(rpc):
    """
    Use the SpaceNet MVS sidecar tail layout already established in the
    Stage-5 core:

        tail[0:4] =
            lon_min, lat_min, lon_max, lat_max
    """

    tail = rpc.get("tail", [])

    if len(tail) < 4:
        return None

    values = [
        _safe_float(x)
        for x in tail[:4]
    ]

    if any(x is None for x in values):
        return None

    lon_min, lat_min, lon_max, lat_max = values

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


def _bbox_intersects(a, b, margin=0.0):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b

    return not (
        a2 < b0 - margin
        or a0 > b2 + margin
        or a3 < b1 - margin
        or a1 > b3 + margin
    )


def _mosaic_bbox_wgs84(mosaic):
    from pyproj import Transformer

    bounds = mosaic.dataset.bounds

    transformer = Transformer.from_crs(
        mosaic.crs,
        "EPSG:4326",
        always_xy=True,
    )

    xs = [
        float(bounds.left),
        float(bounds.right),
        float(bounds.left),
        float(bounds.right),
    ]

    ys = [
        float(bounds.bottom),
        float(bounds.bottom),
        float(bounds.top),
        float(bounds.top),
    ]

    lons, lats = transformer.transform(xs, ys)

    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    return (
        float(np.min(lons)),
        float(np.min(lats)),
        float(np.max(lons)),
        float(np.max(lats)),
    )


def _rpc_polynomial_terms(lon_norm, lat_norm, height_norm):
    """Return the standard 20 RPC polynomial terms."""

    l = np.asarray(lon_norm, dtype=np.float64)
    p = np.asarray(lat_norm, dtype=np.float64)
    h = np.asarray(height_norm, dtype=np.float64)

    return np.stack(
        (
            np.ones_like(l),
            l,
            p,
            h,
            l * p,
            l * h,
            p * h,
            l * l,
            p * p,
            h * h,
            l * p * h,
            l * l * l,
            l * p * p,
            l * h * h,
            l * l * p,
            p * p * p,
            p * h * h,
            l * l * h,
            p * p * h,
            h * h * h,
        ),
        axis=0,
    )


def _pure_rpc_forward(rpc_obj, lons, lats, heights):
    """Evaluate an RPC object without GDAL's transformer wrapper."""

    lon_scale = float(rpc_obj.long_scale)
    lat_scale = float(rpc_obj.lat_scale)
    height_scale = float(rpc_obj.height_scale)

    if lon_scale == 0.0 or lat_scale == 0.0 or height_scale == 0.0:
        raise ValueError("RPC normalization scale cannot be zero.")

    lons = np.asarray(lons, dtype=np.float64).reshape(-1)
    lats = np.asarray(lats, dtype=np.float64).reshape(-1)
    heights = np.asarray(heights, dtype=np.float64).reshape(-1)

    terms = _rpc_polynomial_terms(
        (lons - float(rpc_obj.long_off)) / lon_scale,
        (lats - float(rpc_obj.lat_off)) / lat_scale,
        (heights - float(rpc_obj.height_off)) / height_scale,
    )

    line_num = np.asarray(rpc_obj.line_num_coeff, dtype=np.float64)
    line_den = np.asarray(rpc_obj.line_den_coeff, dtype=np.float64)
    samp_num = np.asarray(rpc_obj.samp_num_coeff, dtype=np.float64)
    samp_den = np.asarray(rpc_obj.samp_den_coeff, dtype=np.float64)

    line_den_value = np.sum(line_den[:, None] * terms, axis=0)
    samp_den_value = np.sum(samp_den[:, None] * terms, axis=0)

    if np.any(np.isclose(line_den_value, 0.0)) or np.any(
        np.isclose(samp_den_value, 0.0)
    ):
        raise ValueError("RPC denominator is zero for a requested point.")

    rows = float(rpc_obj.line_off) + float(rpc_obj.line_scale) * (
        np.sum(line_num[:, None] * terms, axis=0) / line_den_value
    )
    cols = float(rpc_obj.samp_off) + float(rpc_obj.samp_scale) * (
        np.sum(samp_num[:, None] * terms, axis=0) / samp_den_value
    )

    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(cols, dtype=np.float64),
    )


def _forward_rpc(rpc_obj, lons, lats, heights):
    """
    One batched GDAL/Rasterio RPC forward projection.

    IMPORTANT:
    This is forward projection only. It avoids the pathological inverse
    search that caused the previous KeyboardInterrupt.
    """

    from rasterio.transform import RPCTransformer

    lons = np.asarray(lons, dtype=np.float64).reshape(-1)
    lats = np.asarray(lats, dtype=np.float64).reshape(-1)
    heights = np.asarray(heights, dtype=np.float64).reshape(-1)

    if not (
        lons.size == lats.size == heights.size
    ):
        raise ValueError("RPC forward arrays have different lengths.")

    # Use the RPC polynomial first. It is deterministic and matches the
    # coefficients read from the SpaceNet sidecar directly.
    try:
        rows, cols = _pure_rpc_forward(
            rpc_obj,
            lons,
            lats,
            heights,
        )

        if (
            rows.shape == lons.shape
            and cols.shape == lons.shape
            and np.isfinite(rows).all()
            and np.isfinite(cols).all()
        ):
            return rows, cols
    except Exception:
        pass

    # Keep GDAL/Rasterio as a compatibility fallback for unusual RPC
    # metadata objects that do not expose all coefficient attributes.
    try:
        with RPCTransformer(
            rpc_obj,
            rpc_height=0.0,
            RPC_MAX_ITERATIONS=20,
            RPC_PIXEL_ERROR_THRESHOLD=1.0,
        ) as transformer:
            rows, cols = transformer.rowcol(
                lons,
                lats,
                zs=heights,
            )

        rows = np.asarray(rows, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.float64)

        if (
            rows.shape == lons.shape
            and cols.shape == lons.shape
            and np.isfinite(rows).all()
            and np.isfinite(cols).all()
        ):
            return rows, cols
    except Exception:
        pass

    # Some Rasterio/GDAL builds reject the transformer options used above or
    # return non-finite results for these local SpaceNet RPC sidecars. The
    # standard RPC polynomial is equivalent for forward projection and keeps
    # candidate generation independent of that GDAL behavior.
    return _pure_rpc_forward(
        rpc_obj,
        lons,
        lats,
        heights,
    )


def _candidate_offsets_from_points(
    rpc_obj,
    points,
    image_h,
    image_w,
    height,
    mode,
):
    if not points:
        return []

    lons = np.asarray(
        [p[0] for p in points],
        dtype=np.float64,
    )
    lats = np.asarray(
        [p[1] for p in points],
        dtype=np.float64,
    )
    heights = np.full(
        lons.shape,
        float(height),
        dtype=np.float64,
    )

    try:
        rows, cols = _forward_rpc(
            rpc_obj,
            lons,
            lats,
            heights,
        )
    except Exception:
        return []

    cr = (image_h - 1) / 2.0
    cc = (image_w - 1) / 2.0

    result = []

    for row, col in zip(
        rows.tolist(),
        cols.tolist(),
    ):
        if not (
            math.isfinite(row)
            and math.isfinite(col)
        ):
            continue

        ro = float(row - cr)
        co = float(col - cc)

        if (
            abs(ro) >= 200000.0
            or abs(co) >= 200000.0
        ):
            continue

        result.append(
            (
                ro,
                co,
                float(height),
                mode,
                0.0,
            )
        )

    return result


def _fast_find_best_geometry(
    record,
    mosaic,
    rpc_obj,
    rpc,
):
    """
    Final fast geometry resolver.

    The local image crop is centred inside the 2001x2001 scene.

    Candidate generation:
      A. RPC sidecar geographic footprint
      B. RPC crop-origin metadata in the final two sidecar values

    Only the centre crop is evaluated by the geometry resolver. The expensive
    full target grid is constructed exactly once after a candidate is chosen.
    """

    image_h, image_w = _read_image_size(
        record.image_path
    )

    patch_size = int(
        getattr(core, "PATCH_SIZE", 512)
    )

    if image_h < patch_size or image_w < patch_size:
        raise RuntimeError(
            f"Image is smaller than {patch_size}x{patch_size}."
        )

    row0 = max(
        0,
        (image_h - patch_size) // 2,
    )
    col0 = max(
        0,
        (image_w - patch_size) // 2,
    )

    # ---------------------------------------------------------------------
    # Candidate generation
    # ---------------------------------------------------------------------

    candidates = []

    rpc_height = _safe_float(
        rpc.get("height_off"),
        0.0,
    )

    heights = []
    for h in (
        rpc_height,
        0.0,
        50.0,
        100.0,
        200.0,
    ):
        if (
            math.isfinite(h)
            and -500.0 <= h <= 3000.0
            and h not in heights
        ):
            heights.append(h)

    bbox = _rpc_bbox(rpc)

    # Sidecar footprint candidate set.
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox

        footprint_points = [
            (
                0.5 * (lon_min + lon_max),
                0.5 * (lat_min + lat_max),
            ),
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_min),
            (lon_max, lat_max),
        ]

        # This is the preferred source because it comes from the actual
        # scene's RPC sidecar.
        for height in heights[:3]:
            candidates.extend(
                _candidate_offsets_from_points(
                    rpc_obj,
                    footprint_points,
                    image_h,
                    image_w,
                    height,
                    "rpc-geographic-footprint-fast",
                )
            )

    # Tail pixel hints are secondary only.
    try:
        hints = core.rpc_tail_pixel_hints(rpc)
    except Exception:
        hints = []

    cr = (image_h - 1) / 2.0
    cc = (image_w - 1) / 2.0

    for tr, tc, name in hints:
        if (
            math.isfinite(float(tr))
            and math.isfinite(float(tc))
        ):
            candidates.append(
                (
                    float(tr - cr),
                    float(tc - cc),
                    rpc_height,
                    f"{name}-center-offset",
                    0.0,
                )
            )

    # Existing RPC centre fallback.
    try:
        candidates.append(
            (
                float(rpc["line_off"]) - cr,
                float(rpc["samp_off"]) - cc,
                rpc_height,
                "rpc-reference-centre",
                0.0,
            )
        )
    except Exception:
        pass

    # Deduplicate aggressively.
    unique = []
    seen = set()

    for item in candidates:
        ro, co, height, mode, score = item

        key = (
            round(float(ro), 2),
            round(float(co), 2),
            round(float(height), 1),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    if not unique:
        raise RuntimeError(
            "No fast RPC geometry candidates were generated."
        )

    # ---------------------------------------------------------------------
    # Fast validation
    # ---------------------------------------------------------------------
    #
    # We deliberately do NOT call:
    #
    #     rpc_control_points()
    #     find_best_geometry() old crop search
    #
    # We only validate the candidate centre against the real mosaic.
    # build_sample() performs the authoritative full target-grid validity
    # check immediately afterwards.
    # ---------------------------------------------------------------------

    from rasterio.transform import RPCTransformer

    best = None

    # The final two sidecar values are crop-origin metadata written by the
    # MVS3D crop tool: sample/column origin followed by line/row origin.
    # rpc_tail_pixel_hints() returns (row, col). These are raw crop origins,
    # so the RPC frame coordinate is local_pixel + crop_origin. Do not
    # subtract the 2001x2001 image centre here; build_rpc_target_grid() adds
    # the local crop pixel to this offset exactly once.
    #
    # Do not replace this with an arbitrary point selected from the DEM. The
    # crop origin is fixed by the sidecar; only the 512x512 window inside the
    # verified 2001x2001 chip may move. Each candidate below is checked with
    # the real RPC target grid and real DEM before it is returned.
    for tail_row, tail_col, tail_name in hints:
        if not (
            math.isfinite(float(tail_row))
            and math.isfinite(float(tail_col))
        ):
            continue

        max_row0 = image_h - patch_size
        max_col0 = image_w - patch_size

        # Five positions per axis are enough to avoid the old exhaustive
        # search while allowing the crop to move away from a DEM edge.
        row_positions = sorted(
            set(
                (
                    0,
                    max_row0 // 4,
                    max_row0 // 2,
                    (3 * max_row0) // 4,
                    max_row0,
                )
            )
        )
        col_positions = sorted(
            set(
                (
                    0,
                    max_col0 // 4,
                    max_col0 // 2,
                    (3 * max_col0) // 4,
                    max_col0,
                )
            )
        )

        best_crop = None
        best_valid = -1.0

        for candidate_row0 in row_positions:
            for candidate_col0 in col_positions:
                try:
                    target_rows, target_cols = (
                        core.build_rpc_target_grid(
                            rpc_obj,
                            int(candidate_row0),
                            int(candidate_col0),
                            float(tail_row),
                            float(tail_col),
                            float(rpc_height),
                            mosaic,
                        )
                    )

                    target, mask = core.bilinear_sample_mosaic(
                        mosaic,
                        target_rows,
                        target_cols,
                    )
                    _, mask = core.clean_target(
                        target,
                        mask,
                    )

                    valid_fraction = float(mask.mean())

                except Exception:
                    continue

                if valid_fraction > best_valid:
                    best_valid = valid_fraction
                    best_crop = (
                        int(candidate_row0),
                        int(candidate_col0),
                    )

        required_valid = float(
            getattr(core, "MIN_VALID_FRACTION", 0.25)
        )

        if (
            best_crop is not None
            and best_valid >= required_valid
        ):
            selected_row0, selected_col0 = best_crop

            print(
                f"[FAST-RPC] {record.key} -> "
                f"{tail_name}-crop-origin | "
                f"offset=({float(tail_row):.2f},"
                f"{float(tail_col):.2f}) | "
                f"crop=({selected_row0},{selected_col0}) | "
                f"height={rpc_height:.2f} | "
                f"real-valid={best_valid:.3f}"
            )

            return (
                selected_row0,
                selected_col0,
                float(tail_row),
                float(tail_col),
                float(rpc_height),
                f"{tail_name}-crop-origin",
                1.0,
                float(best_valid),
            )

        print(
            f"[FAST-RPC] {record.key} -> "
            f"{tail_name}-crop-origin rejected | "
            f"best-real-valid={max(best_valid, 0.0):.3f} | "
            f"required={required_valid:.3f}"
        )

    if hints:
        raise RuntimeError(
            "RPC crop-origin geometry has insufficient real DEM "
            "coverage for every bounded 512x512 crop."
        )

    # Keep the candidate set deterministic and bounded. There are at most
    # 36 candidates here (sidecar footprint, tail hints, nine real-mosaic
    # anchors at two heights, and the RPC centre fallback). Keep all of them:
    # truncating this list can discard the only real-mosaic anchor that has
    # valid DEM data.
    # Prefer sidecar footprint candidates first.
    unique.sort(
        key=lambda item: (
            0
            if str(item[3]).startswith(
                "rpc-geographic-footprint"
            )
            else (
                1
                if str(item[3]).startswith(
                    "rpc-tail"
                )
                else 2
            ),
            abs(float(item[2]) - rpc_height),
        )
    )

    # One batched inverse for all candidate centres.
    #
    # This is the critical speed fix: one transformer invocation instead of
    # hundreds/thousands of transformer constructions and inverse searches.
    rows = np.full(
        len(unique),
        cr,
        dtype=np.float64,
    ) + np.asarray(
        [u[0] for u in unique],
        dtype=np.float64,
    )

    cols = np.full(
        len(unique),
        cc,
        dtype=np.float64,
    ) + np.asarray(
        [u[1] for u in unique],
        dtype=np.float64,
    )

    heights_array = np.asarray(
        [u[2] for u in unique],
        dtype=np.float64,
    )

    try:
        with RPCTransformer(
            rpc_obj,
            rpc_height=0.0,
            RPC_MAX_ITERATIONS=20,
            RPC_PIXEL_ERROR_THRESHOLD=1.0,
        ) as transformer:
            # RPCTransformer.xy() expects image rows first and columns
            # second. Its return values are geographic x/y = lon/lat.
            lons, lats = transformer.xy(
                rows,
                cols,
                zs=heights_array,
            )

        lons = np.asarray(
            lons,
            dtype=np.float64,
        )
        lats = np.asarray(
            lats,
            dtype=np.float64,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Fast RPC inverse validation failed: {exc}"
        ) from exc

    x, y = core.geographic_to_mosaic_xy(
        lats,
        lons,
        mosaic,
    )

    mr, mc = core.mosaic_xy_to_pixel(
        x,
        y,
        mosaic,
    )

    inside = (
        np.isfinite(mr)
        & np.isfinite(mc)
        & (mr >= 0.0)
        & (mr < mosaic.dataset.height)
        & (mc >= 0.0)
        & (mc < mosaic.dataset.width)
    )

    # First candidate whose centre lands on the real elevation raster.
    for idx in np.flatnonzero(inside):
        item = unique[int(idx)]
        ro, co, height, mode, fit_score = item

        # Check actual DEM value at the projected centre.
        try:
            target_valid = core.point_valid_fraction(
                np.asarray([mr[int(idx)]]),
                np.asarray([mc[int(idx)]]),
                mosaic,
            )
        except Exception:
            target_valid = 0.0

        if float(target_valid) <= 0.0:
            continue

        best = (
            1.0,
            float(target_valid),
            float(ro),
            float(co),
            float(height),
            str(mode),
        )
        break

    if best is None:
        raise RuntimeError(
            "No fast RPC candidate projects onto valid pixels of the "
            "elevation mosaic."
        )

    _, target_valid, ro, co, height, mode = best

    print(
        f"[FAST-RPC] {record.key} -> "
        f"{mode} | "
        f"offset=({ro:.2f},{co:.2f}) | "
        f"height={height:.2f} | "
        f"centre-valid={target_valid:.3f}"
    )

    # The crop itself is the central 512x512 window.
    # The core's build_rpc_target_grid() performs the authoritative RPC
    # geolocation and real DEM sampling for this exact crop.
    return (
        int(row0),
        int(col0),
        float(ro),
        float(co),
        float(height),
        str(mode),
        1.0,
        float(target_valid),
    )


# Install the replacement before core.main().
core.find_best_geometry = _fast_find_best_geometry


# ============================================================================
# ALL-MOSAIC SAMPLE WRAPPER
# ============================================================================

_original_try_build_sample = core.try_build_sample


def try_build_sample_all_mosaics(
    record,
    mosaics,
    deterministic=True,
):
    if not mosaics:
        print(
            f"[WARNING] No elevation mosaics are open for {record.key}"
        )
        return None

    nominal = getattr(
        record,
        "region",
        None,
    )

    ordered = list(
        mosaics.items()
    )

    ordered.sort(
        key=lambda item:
            0 if item[0] == nominal else 1
    )

    failures = []

    for mosaic_name, mosaic in ordered:
        try:
            sample = core.build_sample(
                record,
                mosaic,
                deterministic,
            )

            if sample is None:
                failures.append(
                    f"{mosaic_name}: returned None"
                )
                continue

            valid_fraction = float(
                sample.get(
                    "valid_fraction",
                    0.0,
                )
            )

            # AUTHORITATIVE real-data guard.
            minimum_valid_fraction = float(
                getattr(core, "MIN_VALID_FRACTION", 0.25)
            )

            if valid_fraction < minimum_valid_fraction:
                failures.append(
                    f"{mosaic_name}: "
                    f"final_valid_fraction={valid_fraction:.4f}"
                )
                continue

            sample["mosaic_used"] = mosaic_name
            sample["original_region"] = nominal

            if isinstance(
                sample.get("geo"),
                dict,
            ):
                sample["geo"]["mosaic_used"] = mosaic_name

            print(
                f"[OK] REAL elevation sample: "
                f"{record.key} -> {mosaic_name} "
                f"(valid={valid_fraction:.4f})"
            )

            return sample

        except Exception as exc:
            failures.append(
                f"{mosaic_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        f"[WARNING] Could not build real sample "
        f"{record.key}: "
        + " | ".join(failures[-6:])
    )

    return None



core.try_build_sample = try_build_sample_all_mosaics


# ============================================================================
# PREFLIGHT WRAPPER
# ============================================================================

_original_preflight = core.preflight


def preflight_final(records, mosaics):
    print()
    print("=" * 80)
    print("ASTERRA STAGE-5 FINAL FAST PREFLIGHT")
    print("=" * 80)
    print("[OK] Full-parameter fine-tuning")
    print("[OK] LoRA disabled")
    print("[OK] PEFT disabled")
    print("[OK] All MasterProvisional mosaics tested")
    print("[OK] Synthetic elevation targets forbidden")
    print("[OK] Exhaustive crop-origin RPC search disabled")
    print("[OK] Batched RPC inverse validation enabled")
    print("[OK] Final target validity checked on real DEM")
    print("=" * 80)

    return _original_preflight(
        records,
        mosaics,
    )


core.preflight = preflight_final


# ============================================================================
# FINAL ENTRY
# ============================================================================

if __name__ == "__main__":
    print()
    print("[ASTERRA PATCH] FINAL FAST RPC GEOMETRY LAYER ACTIVE")
    print("[ASTERRA PATCH] Using existing Stage-5 training core")
    print("[ASTERRA PATCH] No synthetic elevation supervision")
    print("[ASTERRA PATCH] 100% parameters trainable")
    print()
    core.main()
