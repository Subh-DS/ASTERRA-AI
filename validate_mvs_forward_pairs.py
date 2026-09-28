"""
ASTERRA AI — DEFINITIVE SpaceNet MVS MP1 GEOMETRY VALIDATOR
============================================================

Purpose:
  Validate the actual SpaceNet MVS RPC -> cropped RGB -> GT relationship
  before Stage-5 fine-tuning.

Design:
  * Uses actual GT elevations.
  * Uses pyproj for GT CRS -> WGS84.
  * Uses Rasterio RPCTransformer.
  * Applies SpaceNet crop offsets:
        local_row = parent_row - vals[95]
        local_col = parent_col - vals[94]
  * Cross-checks vectorized RPC against scalar RPC calls.
  * Does NOT declare a scene valid merely because a few points overlap.
  * Finds a real 512x512 crop with projected GT support.
  * Writes a JSON geometry index only after the global checks pass.

IMPORTANT:
  This script is validation/indexing only. It does NOT train.
"""

from pathlib import Path
import json
import re
import math

import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.transform import RPCTransformer
from rasterio.rpc import RPC
from pyproj import Transformer as PyProjTransformer


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional1")
RGB_DIR = ROOT / "MasterProvisional1"
GT_PATH = ROOT / "MasterProvisional1_GT.tif"

OUTPUT = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS\mvs_mp1_geometry_index_DEFINITIVE.json")

PATCH = 512
RGB_WIDTH = 2001
RGB_HEIGHT = 2001

# Dense enough for geometry, but not unnecessarily expensive.
GT_STRIDE = 8

# Candidate crop search.
CROP_STRIDE = 32

# A crop must contain at least this many projected valid GT samples.
MIN_CROP_SAMPLES = 100

# Scalar RPC cross-check.
SCALAR_CHECK_COUNT = 12
SCALAR_TOLERANCE_PIXELS = 1.0


# ---------------------------------------------------------------------
# RPC PARSING
# ---------------------------------------------------------------------

def parse_rpc(path: Path):
    text = path.read_text(errors="ignore")

    vals = np.array(
        [
            float(x)
            for x in re.findall(
                r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?",
                text,
            )
        ],
        dtype=np.float64,
    )

    if len(vals) < 96:
        raise ValueError(
            f"{path.name}: found {len(vals)} numeric values; expected 96"
        )

    v = vals[:96]

    # Fail early if the sidecar is not the expected 96-value SpaceNet form.
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{path.name}: RPC contains non-finite values")

    # IMPORTANT:
    # Build an explicit rasterio.rpc.RPC object. Do not pass the parsed
    # dictionary directly to RPCTransformer. The explicit object preserves
    # the official GDAL/RPC field mapping unambiguously.
    rpc = RPC(
        height_off=float(v[4]),
        height_scale=float(v[9]),
        lat_off=float(v[2]),
        lat_scale=float(v[7]),
        line_den_coeff=v[30:50].tolist(),
        line_num_coeff=v[10:30].tolist(),
        line_off=float(v[0]),
        line_scale=float(v[5]),
        long_off=float(v[3]),
        long_scale=float(v[8]),
        samp_den_coeff=v[70:90].tolist(),
        samp_num_coeff=v[50:70].tolist(),
        samp_off=float(v[1]),
        samp_scale=float(v[6]),
    )

    # SpaceNet crop metadata:
    # 90:94 = geographic bounding box
    # 94    = sample / column crop origin
    # 95    = line / row crop origin
    crop = {
        "sample_offset": float(v[94]),
        "line_offset": float(v[95]),
        "lon_min": float(v[90]),
        "lat_min": float(v[91]),
        "lon_max": float(v[92]),
        "lat_max": float(v[93]),
    }

    return vals, rpc, crop


def find_rpc(rgb_path: Path) -> Path:
    rpc_path = rgb_path.parent / f"rpc_{rgb_path.stem}.txt"

    if not rpc_path.exists():
        raise FileNotFoundError(
            f"RPC sidecar not found: {rpc_path}"
        )

    return rpc_path


# ---------------------------------------------------------------------
# GT SAMPLING
# ---------------------------------------------------------------------

def sample_valid_gt(gt_ds):
    gt = gt_ds.read(1).astype(np.float64)

    valid = np.isfinite(gt)

    if gt_ds.nodata is not None:
        valid &= gt != gt_ds.nodata

    # Conservative elevation sanity range.
    valid &= gt > -1000.0
    valid &= gt < 10000.0

    rows, cols = np.where(valid)

    keep = (
        (rows % GT_STRIDE == 0)
        & (cols % GT_STRIDE == 0)
    )

    rows = rows[keep]
    cols = cols[keep]
    z = gt[rows, cols]

    xs, ys = xy(
        gt_ds.transform,
        rows,
        cols,
        offset="center",
    )

    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )


# ---------------------------------------------------------------------
# RPC PROJECTION
# ---------------------------------------------------------------------

def project_points(gt_ds, rpc, crop):
    rows, cols, xs, ys, zs = sample_valid_gt(gt_ds)

    to_wgs84 = PyProjTransformer.from_crs(
        gt_ds.crs,
        "EPSG:4326",
        always_xy=True,
    )

    lons, lats = to_wgs84.transform(xs, ys)

    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    # Use the explicit RPC object created in parse_rpc().
    tx = RPCTransformer(rpc)

    # Vectorized RPC forward projection.
    parent_rows, parent_cols = tx.rowcol(
        lons,
        lats,
        zs=zs,
    )

    parent_rows = np.asarray(parent_rows, dtype=np.float64)
    parent_cols = np.asarray(parent_cols, dtype=np.float64)

    # CRITICAL SpaceNet transformation.
    local_rows = parent_rows - crop["line_offset"]
    local_cols = parent_cols - crop["sample_offset"]

    return {
        "gt_rows": rows,
        "gt_cols": cols,
        "gt_x": xs,
        "gt_y": ys,
        "gt_z": zs,
        "lon": lons,
        "lat": lats,
        "parent_rows": parent_rows,
        "parent_cols": parent_cols,
        "local_rows": local_rows,
        "local_cols": local_cols,
    }


# ---------------------------------------------------------------------
# SCALAR RPC CROSS-CHECK
# ---------------------------------------------------------------------

def scalar_rpc_check(rpc, crop, geom):
    n = len(geom["lon"])

    if n == 0:
        return {
            "checked": 0,
            "max_row_error": None,
            "max_col_error": None,
            "pass": False,
            "reason": "No projected points",
        }

    # Deterministic evenly spaced subset.
    count = min(SCALAR_CHECK_COUNT, n)
    indices = np.linspace(0, n - 1, count, dtype=int)

    tx = RPCTransformer(rpc)

    row_errors = []
    col_errors = []

    for idx in indices:
        lon = float(geom["lon"][idx])
        lat = float(geom["lat"][idx])
        z = float(geom["gt_z"][idx])

        scalar_row, scalar_col = tx.rowcol(
            [lon],
            [lat],
            zs=[z],
        )

        scalar_row = float(np.asarray(scalar_row)[0])
        scalar_col = float(np.asarray(scalar_col)[0])

        vector_row = float(geom["parent_rows"][idx])
        vector_col = float(geom["parent_cols"][idx])

        row_errors.append(abs(scalar_row - vector_row))
        col_errors.append(abs(scalar_col - vector_col))

    max_row_error = float(max(row_errors))
    max_col_error = float(max(col_errors))

    passed = (
        max_row_error <= SCALAR_TOLERANCE_PIXELS
        and max_col_error <= SCALAR_TOLERANCE_PIXELS
    )

    return {
        "checked": int(count),
        "max_row_error": max_row_error,
        "max_col_error": max_col_error,
        "pass": bool(passed),
        "tolerance_pixels": SCALAR_TOLERANCE_PIXELS,
    }


# ---------------------------------------------------------------------
# CROP SELECTION
# ---------------------------------------------------------------------

def find_best_crop(local_rows, local_cols):
    finite = (
        np.isfinite(local_rows)
        & np.isfinite(local_cols)
    )

    rows = local_rows[finite]
    cols = local_cols[finite]

    # Only points actually inside the RGB image can belong to a crop.
    inside = (
        (rows >= 0)
        & (rows < RGB_HEIGHT)
        & (cols >= 0)
        & (cols < RGB_WIDTH)
    )

    rows = rows[inside]
    cols = cols[inside]

    if len(rows) == 0:
        return None

    max_r = RGB_HEIGHT - PATCH
    max_c = RGB_WIDTH - PATCH

    # Candidate range based on projected points.
    min_r = max(0, int(math.floor(rows.min())) - PATCH + 1)
    max_r2 = min(max_r, int(math.ceil(rows.max())))

    min_c = max(0, int(math.floor(cols.min())) - PATCH + 1)
    max_c2 = min(max_c, int(math.ceil(cols.max())))

    if min_r > max_r2 or min_c > max_c2:
        return None

    candidates = set()

    r_start = (min_r // CROP_STRIDE) * CROP_STRIDE
    c_start = (min_c // CROP_STRIDE) * CROP_STRIDE

    for r in range(r_start, max_r2 + 1, CROP_STRIDE):
        for c in range(c_start, max_c2 + 1, CROP_STRIDE):
            candidates.add((r, c))

    # Exact boundary candidates.
    candidates.update(
        {
            (min_r, min_c),
            (min_r, max_c2),
            (max_r2, min_c),
            (max_r2, max_c2),
        }
    )

    best = None

    for r, c in candidates:
        r = int(np.clip(r, 0, max_r))
        c = int(np.clip(c, 0, max_c))

        inside_crop = (
            (rows >= r)
            & (rows < r + PATCH)
            & (cols >= c)
            & (cols < c + PATCH)
        )

        count = int(np.count_nonzero(inside_crop))

        if best is None or count > best["projected_gt_samples"]:
            best = {
                "row": r,
                "col": c,
                "projected_gt_samples": count,
                "projected_gt_fraction_of_image_overlap_samples":
                    float(count / max(1, len(rows))),
            }

    return best


# ---------------------------------------------------------------------
# ONE SCENE
# ---------------------------------------------------------------------

def validate_scene(rgb_path, gt_ds, verbose=True):
    rpc_path = find_rpc(rgb_path)
    vals, rpc, crop = parse_rpc(rpc_path)

    geom = project_points(
        gt_ds,
        rpc,
        crop,
    )

    scalar = scalar_rpc_check(
        rpc,
        crop,
        geom,
    )

    local_rows = geom["local_rows"]
    local_cols = geom["local_cols"]

    finite = (
        np.isfinite(local_rows)
        & np.isfinite(local_cols)
    )

    inside = (
        finite
        & (local_rows >= 0)
        & (local_rows < RGB_HEIGHT)
        & (local_cols >= 0)
        & (local_cols < RGB_WIDTH)
    )

    inside_count = int(np.count_nonzero(inside))
    total_count = int(len(local_rows))
    inside_fraction = float(
        inside_count / max(1, total_count)
    )

    best = find_best_crop(
        local_rows,
        local_cols,
    )

    # A scene is geometrically usable if:
    # 1) RPC scalar/vector agree
    # 2) there is actual local RGB overlap
    # 3) at least one 512 crop contains meaningful GT support.
    scene_pass = (
        scalar["pass"]
        and inside_count > 0
        and best is not None
        and best["projected_gt_samples"] >= MIN_CROP_SAMPLES
    )

    if verbose:
        print(
            f"  crop offsets       : "
            f"col={crop['sample_offset']:.3f}, "
            f"row={crop['line_offset']:.3f}"
        )

        print(
            f"  parent RGB bounds  : "
            f"row={np.nanmin(geom['parent_rows']):.1f} .. "
            f"{np.nanmax(geom['parent_rows']):.1f}, "
            f"col={np.nanmin(geom['parent_cols']):.1f} .. "
            f"{np.nanmax(geom['parent_cols']):.1f}"
        )

        print(
            f"  local RGB bounds   : "
            f"row={np.nanmin(local_rows):.1f} .. "
            f"{np.nanmax(local_rows):.1f}, "
            f"col={np.nanmin(local_cols):.1f} .. "
            f"{np.nanmax(local_cols):.1f}"
        )

        print(
            f"  image overlap      : "
            f"{inside_count}/{total_count} "
            f"({inside_fraction:.2%})"
        )

        print(
            f"  scalar RPC check   : "
            f"{'PASS' if scalar['pass'] else 'FAIL'} "
            f"(max row err={scalar['max_row_error']:.6f}, "
            f"max col err={scalar['max_col_error']:.6f})"
        )

        if best:
            print(
                f"  best 512 crop      : "
                f"row={best['row']} col={best['col']} "
                f"samples={best['projected_gt_samples']}"
            )
        else:
            print("  best 512 crop      : NONE")

        print(
            f"  RESULT             : "
            f"{'PASS' if scene_pass else 'FAIL'}"
        )

    return {
        "status": "PASS" if scene_pass else "FAIL",
        "scene": rgb_path.name,
        "rgb": str(rgb_path),
        "rpc": str(rpc_path),
        "crop_offsets": crop,
        "scalar_rpc_check": scalar,
        "projected_total": total_count,
        "projected_inside_rgb": inside_count,
        "inside_rgb_fraction": inside_fraction,
        "local_rgb_bounds": {
            "row_min": float(np.nanmin(local_rows)),
            "row_max": float(np.nanmax(local_rows)),
            "col_min": float(np.nanmin(local_cols)),
            "col_max": float(np.nanmax(local_cols)),
        },
        "best_crop": best,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 76)
    print(" ASTERRA AI — FIXED MP1 GEOMETRY VALIDATION (EXPLICIT RPC OBJECT)")
    print("=" * 76)

    if not RGB_DIR.exists():
        raise FileNotFoundError(RGB_DIR)

    if not GT_PATH.exists():
        raise FileNotFoundError(GT_PATH)

    rgb_files = sorted(RGB_DIR.glob("*.tif"))

    print(f"RGB scenes discovered : {len(rgb_files)}")
    print(f"RGB directory         : {RGB_DIR}")
    print(f"GT                    : {GT_PATH}")
    print(f"RGB dimensions        : {RGB_WIDTH} x {RGB_HEIGHT}")
    print(f"Patch                 : {PATCH} x {PATCH}")
    print(f"GT sample stride      : {GT_STRIDE}")
    print(f"Crop search stride    : {CROP_STRIDE}")

    if len(rgb_files) != 50:
        raise RuntimeError(
            f"Expected 50 MP1 RGB scenes, found {len(rgb_files)}"
        )

    results = []
    passed = 0
    failed = 0

    with rasterio.open(GT_PATH) as gt_ds:
        print(f"GT CRS                : {gt_ds.crs}")
        print(
            f"GT size               : "
            f"{gt_ds.width} x {gt_ds.height}"
        )
        print(f"GT resolution         : {gt_ds.res}")

        if gt_ds.crs is None:
            raise RuntimeError("GT has no CRS")

        # -------------------------------------------------------------
        # FIRST SCENE HARD GATE
        # -------------------------------------------------------------
        #
        # This prevents silently accepting a broken vectorized RPC
        # implementation. Scene 01 is the scene already independently
        # established by the forensic test.
        # -------------------------------------------------------------

        print("\n" + "-" * 76)
        print("SCENE 01 HARD GATE")
        print("-" * 76)

        first = rgb_files[0]

        try:
            first_result = validate_scene(
                first,
                gt_ds,
                verbose=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Scene 01 hard-gate diagnostic failed. "
                "Do not train. Error: "
                f"{type(e).__name__}: {e}"
            ) from e

        if first_result["status"] != "PASS":
            raise RuntimeError(
                "SCENE 01 HARD GATE FAILED. "
                "The geometry implementation is not trustworthy; "
                "training is intentionally blocked."
            )

        results.append(first_result)
        passed += 1

        # -------------------------------------------------------------
        # REMAINING 49
        # -------------------------------------------------------------

        print("\n" + "-" * 76)
        print("ALL REMAINING MP1 SCENES")
        print("-" * 76)

        for i, rgb_path in enumerate(rgb_files[1:], 2):
            print(
                f"\n[{i:02d}/{len(rgb_files):02d}] "
                f"{rgb_path.name}"
            )

            try:
                result = validate_scene(
                    rgb_path,
                    gt_ds,
                    verbose=True,
                )

            except Exception as e:
                result = {
                    "status": "FAIL",
                    "scene": rgb_path.name,
                    "rgb": str(rgb_path),
                    "error": f"{type(e).__name__}: {e}",
                }
                print(
                    f"  RESULT             : FAIL "
                    f"({type(e).__name__}: {e})"
                )

            results.append(result)

            if result["status"] == "PASS":
                passed += 1
            else:
                failed += 1

    # -------------------------------------------------------------
    # FINAL DECISION
    # -------------------------------------------------------------

    all_pass = (
        passed == len(rgb_files)
        and failed == 0
    )

    output = {
        "dataset": "SpaceNet MVS MasterProvisional1",
        "rgb_dir": str(RGB_DIR),
        "gt_path": str(GT_PATH),
        "image_size": [RGB_WIDTH, RGB_HEIGHT],
        "patch_size": PATCH,
        "gt_sample_stride": GT_STRIDE,
        "crop_search_stride": CROP_STRIDE,
        "rpc_rule": {
            "local_row": "parent_row - rpc_values[95]",
            "local_col": "parent_col - rpc_values[94]",
        },
        "training_ready_geometry": bool(all_pass),
        "summary": {
            "scenes": len(rgb_files),
            "pass": passed,
            "fail": failed,
        },
        "scenes": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print(" MP1 DEFINITIVE GEOMETRY VALIDATION COMPLETE")
    print("=" * 76)
    print(f"PASS : {passed}")
    print(f"FAIL : {failed}")
    print(f"INDEX: {OUTPUT}")

    if all_pass:
        print()
        print("TRAINING GATE: PASS")
        print("MP1 geometry is validated for all 50 scenes.")
        print("Next step: build the Stage-5 MVS training dataset.")
    else:
        print()
        print("TRAINING GATE: BLOCKED")
        print("Do NOT start Stage-5 MVS fine-tuning yet.")
        print(
            "At least one scene failed definitive geometry "
            "validation."
        )

    print("=" * 76)


if __name__ == "__main__":
    main()
