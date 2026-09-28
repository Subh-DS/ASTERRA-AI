
import json
import re
import math
import random
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.rpc import RPC
from rasterio.transform import RPCTransformer
from pyproj import Transformer

# ============================================================
# ASTERRA AI — SPACE NET MVS FAST DATASET BUILDER
#
# FIXES:
# 1. Uses the already validated geometry index.
# 2. NEVER performs 512x512 per-pixel RPC inversion.
# 3. Inverts only a small control grid (33x33).
# 4. Interpolates ground coordinates over the dense crop.
# 5. Samples the official GT directly from those coordinates.
# 6. Caches the GT raster in RAM; no repeated ds.read(1).
# 7. Writes 3-channel RGB (replicated from official 1-band MVS).
# 8. Fails safely; never declares a partial dataset ready.
# ============================================================

ROOT = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS")
INDEX = ROOT / "mvs_mp1_geometry_index_DEFINITIVE.json"
OUT = ROOT / "processed_mp1"

PATCH = 512
CONTROL = 33
MIN_VALID_FRACTION = 0.95
MIN_CONTROL_FRACTION = 0.80
HEIGHT_TOL = 0.25
MAX_ITERS = 8
TRAIN_FRACTION = 0.80
SEED = 42
RPC_PIXEL_ERROR_THRESHOLD = 0.1

RGB_OUT = OUT / "rgb"
TARGET_OUT = OUT / "target"
MASK_OUT = OUT / "mask"
MANIFEST = OUT / "manifest.json"
TRAIN_MANIFEST = OUT / "train_manifest.json"
VAL_MANIFEST = OUT / "val_manifest.json"
FAILURES = OUT / "failures.json"


def floats_from_text(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return np.asarray([
        float(x) for x in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", text
        )
    ], dtype=np.float64)


def make_rpc(values):
    v = values[:96]
    return RPC(
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


def sample_nearest(gt_arr, gt_transform, xs, ys, nodata):
    """Fast nearest-neighbour GT lookup from an already cached array."""
    inv = ~gt_transform
    cols_f, rows_f = inv * (np.asarray(xs), np.asarray(ys))
    cols = np.rint(cols_f).astype(np.int64)
    rows = np.rint(rows_f).astype(np.int64)

    valid = (
        np.isfinite(xs) & np.isfinite(ys) &
        (rows >= 0) & (rows < gt_arr.shape[0]) &
        (cols >= 0) & (cols < gt_arr.shape[1])
    )

    out = np.full(np.asarray(xs).shape, np.nan, dtype=np.float64)
    if np.any(valid):
        vals = gt_arr[rows[valid], cols[valid]].astype(np.float64)
        if nodata is not None:
            vals[np.isclose(vals, nodata)] = np.nan
        vals[~np.isfinite(vals)] = np.nan
        out[valid] = vals

    return out, valid & np.isfinite(out)


def sample_bilinear(gt_arr, gt_transform, xs, ys, nodata):
    """Vectorized bilinear GT sampling with strict nodata handling."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    inv = ~gt_transform
    cols, rows = inv * (xs, ys)

    valid_xy = np.isfinite(cols) & np.isfinite(rows)
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1

    inside = (
        valid_xy &
        (r0 >= 0) & (c0 >= 0) &
        (r1 < gt_arr.shape[0]) & (c1 < gt_arr.shape[1])
    )

    out = np.full(xs.shape, np.nan, dtype=np.float32)
    if not np.any(inside):
        return out, np.zeros(xs.shape, dtype=bool)

    ii = np.flatnonzero(inside)
    a00 = gt_arr[r0[ii], c0[ii]].astype(np.float64)
    a01 = gt_arr[r0[ii], c1[ii]].astype(np.float64)
    a10 = gt_arr[r1[ii], c0[ii]].astype(np.float64)
    a11 = gt_arr[r1[ii], c1[ii]].astype(np.float64)

    vals = np.stack([a00, a01, a10, a11], axis=1)
    good = np.isfinite(vals).all(axis=1)
    if nodata is not None:
        good &= ~np.isclose(vals, nodata).any(axis=1)

    dr = rows[ii] - r0[ii]
    dc = cols[ii] - c0[ii]

    interp = (
        a00 * (1-dr) * (1-dc) +
        a01 * (1-dr) * dc +
        a10 * dr * (1-dc) +
        a11 * dr * dc
    )

    good &= np.isfinite(interp)
    out[ii[good]] = interp[good].astype(np.float32)

    mask = np.zeros(xs.shape, dtype=bool)
    mask[ii[good]] = True
    return out, mask


def inverse_one(tr, parent_row, parent_col, gt_arr, gt_transform, nodata,
                to_gt, seed_height):
    """Single-point fallback, used only for control-grid failures."""
    z = float(seed_height)
    best = None

    for _ in range(MAX_ITERS):
        try:
            lon, lat = tr.xy(
                [float(parent_row)], [float(parent_col)],
                zs=[z], offset="center"
            )
            lon = float(np.asarray(lon).ravel()[0])
            lat = float(np.asarray(lat).ravel()[0])

            if not (math.isfinite(lon) and math.isfinite(lat)):
                return None

            x, y = to_gt.transform(lon, lat)
            x, y = float(x), float(y)

            zn, ok = sample_nearest(
                gt_arr, gt_transform,
                np.asarray([x]), np.asarray([y]), nodata
            )
            if not bool(ok[0]):
                return None

            zn = float(zn[0])
            best = (x, y, zn)

            if abs(zn - z) <= HEIGHT_TOL:
                return best

            z = zn
        except Exception:
            return None

    return best


def inverse_control_grid(tr, rows, cols, gt_arr, gt_transform, nodata,
                         to_gt, seed_height):
    """
    Invert ONLY the control grid.
    Vectorized RPC is attempted first. If a chunk loses point cardinality,
    only that small chunk falls back to point-by-point inversion.
    """
    rows = np.asarray(rows, dtype=np.float64).ravel()
    cols = np.asarray(cols, dtype=np.float64).ravel()
    n = rows.size

    ox = np.full(n, np.nan, dtype=np.float64)
    oy = np.full(n, np.nan, dtype=np.float64)
    oz = np.full(n, np.nan, dtype=np.float64)
    ok = np.zeros(n, dtype=bool)

    # Small enough to avoid the old giant dense RPC request.
    chunk_size = 256

    for begin in range(0, n, chunk_size):
        end = min(begin + chunk_size, n)
        idx0 = np.arange(begin, end)
        rr0 = rows[idx0].copy()
        cc0 = cols[idx0].copy()
        zz0 = np.full(rr0.shape, float(seed_height), dtype=np.float64)

        unresolved = idx0.copy()
        rr = rr0.copy()
        cc = cc0.copy()
        zz = zz0.copy()
        vector_failed = False

        for _ in range(MAX_ITERS):
            if rr.size == 0:
                break
            try:
                lon, lat = tr.xy(rr, cc, zs=zz, offset="center")
                lon = np.asarray(lon, dtype=np.float64).ravel()
                lat = np.asarray(lat, dtype=np.float64).ravel()

                if lon.size != rr.size or lat.size != rr.size:
                    vector_failed = True
                    break

                x, y = to_gt.transform(lon, lat)
                x = np.asarray(x, dtype=np.float64).ravel()
                y = np.asarray(y, dtype=np.float64).ravel()
                if x.size != rr.size or y.size != rr.size:
                    vector_failed = True
                    break

                zn, inside = sample_nearest(
                    gt_arr, gt_transform, x, y, nodata
                )
                inside = np.asarray(inside, dtype=bool).ravel()
                zn = np.asarray(zn, dtype=np.float64).ravel()

                if zn.size != rr.size or inside.size != rr.size:
                    vector_failed = True
                    break

                good = (
                    np.isfinite(x) & np.isfinite(y) &
                    inside & np.isfinite(zn)
                )

                if not np.any(good):
                    vector_failed = True
                    break

                ids = unresolved[good]
                ox[ids] = x[good]
                oy[ids] = y[good]
                oz[ids] = zn[good]
                ok[ids] = True

                dz = np.abs(zn[good] - zz[good])
                if np.all(dz <= HEIGHT_TOL):
                    unresolved = unresolved[~good]
                    break

                unresolved = unresolved[good]
                rr = rr[good]
                cc = cc[good]
                zz = zn[good]

            except Exception:
                vector_failed = True
                break

        # Safe fallback for unresolved points only.
        if vector_failed:
            fallback_ids = idx0
        else:
            fallback_ids = unresolved

        for absolute in fallback_ids:
            result = inverse_one(
                tr, rows[absolute], cols[absolute],
                gt_arr, gt_transform, nodata,
                to_gt, seed_height
            )
            if result is not None:
                ox[absolute], oy[absolute], oz[absolute] = result
                ok[absolute] = True

    return ox, oy, oz, ok


def bilinear_control_to_dense(control_values):
    """Pure NumPy bilinear interpolation from CONTROL grid to 512x512."""
    cv = np.asarray(control_values, dtype=np.float64).reshape(CONTROL, CONTROL)

    u = np.linspace(0.0, CONTROL - 1.0, PATCH)
    v = np.linspace(0.0, CONTROL - 1.0, PATCH)

    U, V = np.meshgrid(u, v, indexing="ij")

    i0 = np.floor(U).astype(np.int64)
    j0 = np.floor(V).astype(np.int64)
    i1 = np.minimum(i0 + 1, CONTROL - 1)
    j1 = np.minimum(j0 + 1, CONTROL - 1)

    wi = U - i0
    wj = V - j0

    return (
        cv[i0, j0] * (1-wi) * (1-wj) +
        cv[i0, j1] * (1-wi) * wj +
        cv[i1, j0] * wi * (1-wj) +
        cv[i1, j1] * wi * wj
    )


def make_pair(record, gt_path):
    rgb_path = Path(record["rgb"])
    rpc_path = Path(record["rpc"])
    crop = record["best_crop"]

    crop_row = int(crop["row"])
    crop_col = int(crop["col"])

    values = floats_from_text(rpc_path)
    if values.size < 96:
        raise RuntimeError(f"RPC has only {values.size} numeric values")

    values = values[:96]
    rpc = make_rpc(values)

    row_offset = float(values[95])
    col_offset = float(values[94])

    # Read the GT ONCE for this scene.
    with rasterio.open(gt_path) as gt:
        gt_arr = gt.read(1).astype(np.float32)
        gt_transform = gt.transform
        gt_crs = gt.crs
        nodata = gt.nodata

    if gt_crs is None:
        raise RuntimeError("GT has no CRS")

    with rasterio.open(rgb_path) as img:
        if img.width != 2001 or img.height != 2001:
            raise RuntimeError(
                f"Expected 2001x2001 RGB, got {img.width}x{img.height}"
            )

        # Official MVS is 1-band. Replicate it to 3 channels for DPT/DepthAnything.
        rgb = img.read(
            1,
            window=rasterio.windows.Window(
                crop_col, crop_row, PATCH, PATCH
            )
        )

    if rgb.shape != (PATCH, PATCH):
        raise RuntimeError(f"RGB crop shape {rgb.shape}")

    rgb = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        rgb /= 255.0
    else:
        lo, hi = np.percentile(rgb[np.isfinite(rgb)], [1, 99])
        if hi > lo:
            rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)

    # 3-channel replicated panchromatic image.
    rgb3 = np.stack([rgb, rgb, rgb], axis=0).astype(np.float32)

    to_gt = Transformer.from_crs(
        "EPSG:4326", gt_crs, always_xy=True
    )

    # Control points in LOCAL crop coordinates.
    p = np.linspace(0, PATCH - 1, CONTROL, dtype=np.float64)
    lr, lc = np.meshgrid(p, p, indexing="ij")
    lr = lr.ravel()
    lc = lc.ravel()

    parent_rows = lr + crop_row + row_offset
    parent_cols = lc + crop_col + col_offset

    with RPCTransformer(
        rpc,
        rpc_height=float(values[4]),
        RPC_MAX_ITERATIONS=100,
        RPC_PIXEL_ERROR_THRESHOLD=RPC_PIXEL_ERROR_THRESHOLD,
    ) as tr:
        cx, cy, cz, cok = inverse_control_grid(
            tr,
            parent_rows,
            parent_cols,
            gt_arr,
            gt_transform,
            nodata,
            to_gt,
            float(values[4]),
        )

    control_fraction = float(cok.mean())
    if control_fraction < MIN_CONTROL_FRACTION:
        raise RuntimeError(
            f"control valid fraction {control_fraction:.3f} < "
            f"{MIN_CONTROL_FRACTION:.3f}"
        )

    # Missing control points are only allowed if interpolation can still
    # produce a dense valid target. Fill them using nearest valid control.
    if not np.all(cok):
        good = np.flatnonzero(cok)
        bad = np.flatnonzero(~cok)
        if good.size < int(MIN_CONTROL_FRACTION * cok.size):
            raise RuntimeError(
                f"Too few valid control points: {good.size}/{cok.size} "
                f"({good.size / cok.size:.3f}) < {MIN_CONTROL_FRACTION:.3f}"
            )
        for b in bad:
            d = (cx[good] - cx[good[b*0 if False else 0]]) if False else None
            # Nearest in LOCAL image coordinates, not ground coordinates.
            dist = (lr[good] - lr[b])**2 + (lc[good] - lc[b])**2
            k = good[np.argmin(dist)]
            cx[b], cy[b], cz[b] = cx[k], cy[k], cz[k]

    # Fast dense ground-coordinate approximation.
    dense_x = bilinear_control_to_dense(cx)
    dense_y = bilinear_control_to_dense(cy)

    # Sample the official GT at every dense coordinate.
    target, dense_ok = sample_bilinear(
        gt_arr,
        gt_transform,
        dense_x.ravel(),
        dense_y.ravel(),
        nodata,
    )
    target = target.reshape(PATCH, PATCH)
    mask = dense_ok.reshape(PATCH, PATCH)

    valid_fraction = float(mask.mean())
    if valid_fraction < MIN_VALID_FRACTION:
        raise RuntimeError(
            f"dense target valid fraction {valid_fraction:.3f} < "
            f"{MIN_VALID_FRACTION:.3f}"
        )

    # Invalidate target pixels rather than inventing values.
    target[~mask] = np.nan

    return rgb3, target, mask.astype(np.uint8), {
        "control_valid_fraction": control_fraction,
        "dense_valid_fraction": valid_fraction,
        "target_min": float(np.nanmin(target)),
        "target_max": float(np.nanmax(target)),
        "target_mean": float(np.nanmean(target)),
        "crop_row": crop_row,
        "crop_col": crop_col,
        "rpc_row_offset": row_offset,
        "rpc_col_offset": col_offset,
    }


def save_rgb(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": 3,
        "dtype": "float32",
        "compress": "deflate",
        "predictor": 2,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32))


def save_single(path, arr, dtype, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "compress": "deflate",
        "predictor": 2,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def main():
    print("=" * 100)
    print("ASTERRA AI — SPACE NET MVS FAST DATASET BUILDER — FINAL FIX")
    print("=" * 100)

    if not INDEX.exists():
        raise FileNotFoundError(f"Missing definitive index:\n{INDEX}")

    data = json.loads(INDEX.read_text(encoding="utf-8"))
    scenes = data.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("No scenes in definitive index")

    if any(s.get("status") != "PASS" or "best_crop" not in s for s in scenes):
        raise RuntimeError("Definitive index contains non-PASS scene(s)")

    gt_path = Path(data["gt_path"])
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    print(f"Index   : {INDEX}")
    print(f"Scenes  : {len(scenes)}")
    print(f"Patch   : {PATCH}x{PATCH}")
    print(f"Control : {CONTROL}x{CONTROL}")
    print(f"GT      : {gt_path}")
    print()

    # Remove only the previous generated dataset.
    # This prevents stale partial manifests/files from being reused.
    if OUT.exists():
        shutil.rmtree(OUT)

    for d in [RGB_OUT, TARGET_OUT, MASK_OUT]:
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    shuffled = scenes.copy()
    rng.shuffle(shuffled)

    n_train = int(round(len(shuffled) * TRAIN_FRACTION))
    train_ids = {id(s) for s in shuffled[:n_train]}

    records = []
    failures = []

    for i, scene in enumerate(shuffled, 1):
        name = Path(scene["scene"]).stem
        split = "train" if id(scene) in train_ids else "val"

        print(f"[{i:02d}/{len(shuffled):02d}] {name}")

        try:
            rgb, target, mask, stats = make_pair(scene, gt_path)

            rgb_path = RGB_OUT / f"{name}.tif"
            target_path = TARGET_OUT / f"{name}.tif"
            mask_path = MASK_OUT / f"{name}.tif"

            save_rgb(rgb_path, rgb)
            save_single(
                target_path,
                np.where(np.isfinite(target), target, -9999.0),
                "float32",
                -9999.0,
            )
            save_single(mask_path, mask, "uint8", 0)

            rec = {
                "scene": scene["scene"],
                "split": split,
                "rgb": str(rgb_path),
                "target": str(target_path),
                "mask": str(mask_path),
                "rgb_source": scene["rgb"],
                "rpc_source": scene["rpc"],
                "gt_source": str(gt_path),
                "crop": scene["best_crop"],
                "crop_offsets": scene["crop_offsets"],
                "geometry_validation": {
                    "scalar_rpc_check": scene["scalar_rpc_check"],
                    "projected_total": scene["projected_total"],
                    "projected_inside_rgb": scene["projected_inside_rgb"],
                    "inside_rgb_fraction": scene["inside_rgb_fraction"],
                },
                "target_stats": stats,
                "channels": 3,
                "target_type": "official_GT_elevation",
            }

            records.append(rec)

            print(
                f"  PASS | {split} | "
                f"control={stats['control_valid_fraction']:.3f} | "
                f"dense={stats['dense_valid_fraction']:.3f} | "
                f"Z={stats['target_min']:.2f}..{stats['target_max']:.2f}"
            )

        except Exception as e:
            failures.append({
                "scene": scene.get("scene"),
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"  FAIL | {type(e).__name__}: {e}")

    if failures:
        FAILURES.write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
        print()
        print("=" * 100)
        print("DATASET BUILD FAILED — NO READY MANIFEST")
        print("=" * 100)
        print(f"Success : {len(records)}")
        print(f"Failed  : {len(failures)}")
        print(f"Errors  : {FAILURES}")
        raise RuntimeError(
            f"{len(failures)} scene(s) failed. "
            f"Fix the reported issue before Stage-5 training."
        )

    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]

    payload = {
        "version": 2,
        "dataset": "SpaceNet MVS MasterProvisional1",
        "purpose": "Stage-5 RGB-to-GT elevation fine-tuning",
        "geometry_source": str(INDEX),
        "gt_source": str(gt_path),
        "patch_size": PATCH,
        "control_grid": CONTROL,
        "scene_count": len(records),
        "train_count": len(train),
        "val_count": len(val),
        "channels": 3,
        "target_policy": "official GT sampled through validated RPC geometry; no synthetic target",
        "rpc_policy": "validated best crop + official crop offsets",
        "records": records,
    }

    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    TRAIN_MANIFEST.write_text(json.dumps({"records": train}, indent=2), encoding="utf-8")
    VAL_MANIFEST.write_text(json.dumps({"records": val}, indent=2), encoding="utf-8")

    print()
    print("=" * 100)
    print("DATASET BUILD COMPLETE")
    print("=" * 100)
    print(f"TOTAL : {len(records)}")
    print(f"TRAIN : {len(train)}")
    print(f"VAL   : {len(val)}")
    print(f"OUT   : {OUT}")
    print(f"MANIFEST : {MANIFEST}")
    print()
    print("Stage-5 training has NOT been started.")


if __name__ == "__main__":
    main()
