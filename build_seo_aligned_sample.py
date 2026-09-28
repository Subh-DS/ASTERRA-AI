
"""
ASTERRA AI — S-EO RGB ↔ DSM alignment sample builder

Purpose:
    Build ONE real aligned RGB + absolute DSM training sample from S-EO
    using the validated GeoTransform + RPC geometry.

Important:
    S-EO dsm_max is preserved as ABSOLUTE ELEVATION.
    This script does NOT convert DSM-Max into nDSM.

Output:
    seo_alignment_samples/OMA_135_40/
        rgb.png
        dsm_absolute.tif
        valid_mask.tif
        metadata.json
        alignment_preview.png

No model training is performed.
"""

import io
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.crs import CRS
import rasterio
from scipy.interpolate import LinearNDInterpolator


REPO = "emasquil/shadow-eo"
AOI = "OMA_135"
ACQ = "40"

RGB_FILE = "pansharpened_crops_color_corrected.part.tar.gz.aa"
DSM_FILE = "dsm_max.tar.gz"
RPC_FILE = "rpcs.tar.gz"

OUT_ROOT = Path("seo_alignment_samples") / f"{AOI}_{ACQ}"

# Processing controls.
# 2 means every second DSM pixel is used for interpolation.
# 1 gives maximum density but uses more RAM.
DSM_SAMPLE_STEP = 2

# Minimum number of valid interpolation vertices.
MIN_VALID_POINTS = 10000


def find_sample(ds, contains):
    for sample in ds:
        if contains in sample["__key__"]:
            return sample
    raise RuntimeError(f"Sample not found: {contains}")


def parse_aux_xml(aux):
    if isinstance(aux, bytes):
        aux = aux.decode("utf-8", errors="replace")

    root = ET.fromstring(aux)
    gt = None
    srs = None

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag == "GeoTransform" and elem.text:
            gt = [float(x.strip()) for x in elem.text.replace(",", " ").split()]

        elif tag == "SRS" and elem.text:
            srs = elem.text.strip()

    if gt is None or len(gt) != 6:
        raise RuntimeError("Invalid/missing GeoTransform")

    # GDAL:
    # [origin_x, pixel_width, rotation_x,
    #  origin_y, rotation_y, pixel_height]
    #
    # Rasterio:
    # [pixel_width, rotation_x, origin_x,
    #  rotation_y, pixel_height, origin_y]
    transform = Affine(
        gt[1], gt[2], gt[0],
        gt[4], gt[5], gt[3]
    )

    crs = CRS.from_wkt(srs) if srs else CRS.from_epsg(32614)
    return transform, crs, gt


def decode_rgb(sample):
    value = sample["png"]

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))

    return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))


def decode_dsm(sample):
    arr = np.asarray(sample["tif"], dtype=np.float32)

    aux = sample.get("tif.aux.xml")
    if aux is None:
        raise RuntimeError("DSM tif.aux.xml missing")

    transform, crs, gt = parse_aux_xml(aux)
    return arr, transform, crs, gt


def parse_rpc(record):
    r = record["rpc"]

    def coeff(name):
        v = r[name]

        if isinstance(v, dict):
            return np.asarray(
                [
                    float(x)
                    for _, x in sorted(
                        v.items(),
                        key=lambda kv: int(str(kv[0]))
                    )
                ],
                dtype=np.float64
            )

        return np.asarray(v, dtype=np.float64)

    rpc = {
        "row_off": float(r["row_offset"]),
        "col_off": float(r["col_offset"]),
        "lat_off": float(r["lat_offset"]),
        "lon_off": float(r["lon_offset"]),
        "alt_off": float(r["alt_offset"]),
        "row_scale": float(r["row_scale"]),
        "col_scale": float(r["col_scale"]),
        "lat_scale": float(r["lat_scale"]),
        "lon_scale": float(r["lon_scale"]),
        "alt_scale": float(r["alt_scale"]),
        "row_num": coeff("row_num"),
        "row_den": coeff("row_den"),
        "col_num": coeff("col_num"),
        "col_den": coeff("col_den"),
    }

    for k in ("row_num", "row_den", "col_num", "col_den"):
        if len(rpc[k]) != 20:
            raise RuntimeError(f"{k}: expected 20 coefficients")

    return rpc


def rpc_terms(L, P, H):
    return np.stack([
        np.ones_like(L),
        L, P, H,
        L * P,
        L * H,
        P * H,
        L * L,
        P * P,
        H * H,
        L * P * H,
        L * L * L,
        L * P * P,
        L * H * H,
        L * L * P,
        P * P * P,
        P * H * H,
        L * L * H,
        P * P * H,
        H * H * H,
    ], axis=-1)


def rpc_project_vectorized(rpc, lon, lat, height):
    L = (lon - rpc["lon_off"]) / rpc["lon_scale"]
    P = (lat - rpc["lat_off"]) / rpc["lat_scale"]
    H = (height - rpc["alt_off"]) / rpc["alt_scale"]

    terms = rpc_terms(L, P, H)

    row_num = terms @ rpc["row_num"]
    row_den = terms @ rpc["row_den"]
    col_num = terms @ rpc["col_num"]
    col_den = terms @ rpc["col_den"]

    good = (
        np.isfinite(row_num) &
        np.isfinite(row_den) &
        np.isfinite(col_num) &
        np.isfinite(col_den) &
        (np.abs(row_den) > 1e-10) &
        (np.abs(col_den) > 1e-10)
    )

    rows = np.full(lon.shape, np.nan, dtype=np.float64)
    cols = np.full(lon.shape, np.nan, dtype=np.float64)

    rows[good] = (
        (row_num[good] / row_den[good]) * rpc["row_scale"]
        + rpc["row_off"]
    )

    cols[good] = (
        (col_num[good] / col_den[good]) * rpc["col_scale"]
        + rpc["col_off"]
    )

    return rows, cols, good


def save_geotiff(path, array, transform, crs, nodata):
    array = np.asarray(array)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3 if np.issubdtype(array.dtype, np.floating) else 2,
    ) as dst:
        dst.write(array, 1)


print("=" * 78)
print("ASTERRA — BUILD S-EO ALIGNED SAMPLE")
print("=" * 78)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. RGB
# -------------------------------------------------------------------------
print("\n[1/5] Loading RGB from Hugging Face stream...")

rgb_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={"train": RGB_FILE},
)

rgb_sample = find_sample(
    rgb_ds,
    f"pansharpened_he/{AOI}/{AOI}_{ACQ}_rgb"
)

rgb = decode_rgb(rgb_sample)
rgb_h, rgb_w = rgb.shape[:2]

print("RGB key:", rgb_sample["__key__"])
print("RGB shape:", rgb.shape)

# -------------------------------------------------------------------------
# 2. DSM
# -------------------------------------------------------------------------
print("\n[2/5] Loading DSM + auxiliary georeferencing...")

dsm_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={"train": DSM_FILE},
)

dsm_sample = find_sample(dsm_ds, AOI)

dsm, dsm_transform, dsm_crs, raw_gt = decode_dsm(dsm_sample)

print("DSM key:", dsm_sample["__key__"])
print("DSM shape:", dsm.shape)
print("DSM min:", float(np.nanmin(dsm)))
print("DSM max:", float(np.nanmax(dsm)))
print("DSM mean:", float(np.nanmean(dsm)))
print("DSM CRS:", dsm_crs)
print("GDAL GeoTransform:", raw_gt)
print("Rasterio Affine:", dsm_transform)

# -------------------------------------------------------------------------
# 3. RPC
# -------------------------------------------------------------------------
print("\n[3/5] Loading RPC...")

rpc_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={"train": RPC_FILE},
)

rpc_sample = find_sample(
    rpc_ds,
    f"{AOI}_{ACQ}_pan"
)

rpc_json = rpc_sample["json"]

if isinstance(rpc_json, str):
    rpc_json = json.loads(rpc_json)

rpc = parse_rpc(rpc_json)

print("RPC image:", rpc_json["img"])
print("RPC dimensions:", rpc_json["width"], "x", rpc_json["height"])

# -------------------------------------------------------------------------
# 4. Project DSM into RGB
# -------------------------------------------------------------------------
print("\n[4/5] Projecting DSM pixels into RGB coordinates...")

transformer = Transformer.from_crs(
    dsm_crs,
    "EPSG:4326",
    always_xy=True
)

dsm_valid = np.isfinite(dsm)

# Sample DSM pixels to keep the first real alignment test lightweight.
rr, cc = np.where(dsm_valid)

if DSM_SAMPLE_STEP > 1:
    rr = rr[::DSM_SAMPLE_STEP]
    cc = cc[::DSM_SAMPLE_STEP]

z = dsm[rr, cc].astype(np.float64)

x, y = dsm_transform * (cc + 0.5, rr + 0.5)
lon, lat = transformer.transform(x, y)

rgb_row, rgb_col, rpc_good = rpc_project_vectorized(
    rpc,
    np.asarray(lon),
    np.asarray(lat),
    z,
)

inside = (
    rpc_good &
    (rgb_row >= 0) &
    (rgb_row < rgb_h - 1) &
    (rgb_col >= 0) &
    (rgb_col < rgb_w - 1)
)

print("DSM sampled points:", len(rr))
print("Finite RPC projections:", int(rpc_good.sum()))
print("Inside RGB:", int(inside.sum()))
print("Outside RGB:", int((rpc_good & ~inside).sum()))

if int(inside.sum()) < MIN_VALID_POINTS:
    raise RuntimeError(
        f"Only {int(inside.sum())} usable aligned points. "
        f"Expected at least {MIN_VALID_POINTS}."
    )

px = rgb_col[inside]
py = rgb_row[inside]
pz = z[inside]

print("RGB projected column range:", float(px.min()), "→", float(px.max()))
print("RGB projected row range:", float(py.min()), "→", float(py.max()))
print("Projected DSM elevation range:", float(pz.min()), "→", float(pz.max()))

# -------------------------------------------------------------------------
# 5. Interpolate DSM onto RGB pixel grid
# -------------------------------------------------------------------------
print("\n[5/5] Interpolating absolute DSM onto RGB grid...")

# Restrict target to the actual projected DSM footprint.
x0 = max(0, int(math.floor(px.min())))
x1 = min(rgb_w, int(math.ceil(px.max())) + 1)
y0 = max(0, int(math.floor(py.min())))
y1 = min(rgb_h, int(math.ceil(py.max())) + 1)

print("Aligned RGB window:")
print("  x:", x0, "→", x1)
print("  y:", y0, "→", y1)
print("  size:", x1 - x0, "x", y1 - y0)

grid_x, grid_y = np.meshgrid(
    np.arange(x0, x1, dtype=np.float64) + 0.5,
    np.arange(y0, y1, dtype=np.float64) + 0.5,
)

points = np.column_stack([px, py])

print("Building LinearNDInterpolator...")
interpolator = LinearNDInterpolator(
    points,
    pz,
    fill_value=np.nan,
)

target = interpolator(grid_x, grid_y).astype(np.float32)

valid_target = np.isfinite(target)

print("Aligned target valid pixels:", int(valid_target.sum()))
print("Aligned target valid percentage:",
      float(valid_target.mean() * 100.0))
print("Target min:", float(np.nanmin(target)))
print("Target max:", float(np.nanmax(target)))
print("Target mean:", float(np.nanmean(target)))
print("Target std:", float(np.nanstd(target)))

if valid_target.sum() < 10000:
    raise RuntimeError("Too few valid interpolated target pixels.")

# Crop RGB to exactly the aligned target window.
rgb_crop = rgb[y0:y1, x0:x1]

# Save RGB.
Image.fromarray(rgb_crop).save(OUT_ROOT / "rgb.png")

# RGB image pixel coordinates are 0.5m in this S-EO sample.
# Derive output transform from the projected pixel window using the RGB
# footprint implied by the RPC geometry. For this first sample, we retain
# the DSM/ground georeferencing separately in metadata and use a simple
# pixel-space aligned target GeoTIFF transform derived from the DSM
# footprint bounding box.
#
# Since RPC is the actual image geometry, do not claim this transform is
# authoritative absolute georeferencing for the RGB crop.
aligned_transform = Affine(
    0.5, 0.0, 0.0,
    0.0, -0.5, 0.0
)

dsm_out = np.where(valid_target, target, -9999.0).astype(np.float32)
mask_out = valid_target.astype(np.uint8)

save_geotiff(
    OUT_ROOT / "dsm_absolute.tif",
    dsm_out,
    aligned_transform,
    CRS.from_epsg(4326),
    -9999.0,
)

save_geotiff(
    OUT_ROOT / "valid_mask.tif",
    mask_out,
    aligned_transform,
    CRS.from_epsg(4326),
    0,
)

metadata = {
    "dataset": REPO,
    "aoi": AOI,
    "acquisition": ACQ,
    "rgb_key": rgb_sample["__key__"],
    "dsm_key": dsm_sample["__key__"],
    "rpc_key": rpc_sample["__key__"],
    "target_semantics": "S-EO DSM-Max absolute elevation",
    "not_ndsm": True,
    "rgb_original_shape": [int(rgb_h), int(rgb_w), 3],
    "dsm_original_shape": [int(dsm.shape[0]), int(dsm.shape[1])],
    "dsm_crs": str(dsm_crs),
    "dsm_gdal_geotransform": [float(v) for v in raw_gt],
    "dsm_rasterio_affine": [float(v) for v in dsm_transform],
    "alignment_window": {
        "x0": int(x0),
        "x1": int(x1),
        "y0": int(y0),
        "y1": int(y1),
        "width": int(x1 - x0),
        "height": int(y1 - y0),
    },
    "dsm_sample_step": DSM_SAMPLE_STEP,
    "projected_points": int(len(pz)),
    "target_valid_pixels": int(valid_target.sum()),
    "target_valid_percentage": float(valid_target.mean() * 100.0),
    "target_min": float(np.nanmin(target)),
    "target_max": float(np.nanmax(target)),
    "target_mean": float(np.nanmean(target)),
    "target_std": float(np.nanstd(target)),
    "rpc_reference_check": {
        "expected_row": 16.847,
        "expected_col": 49.736,
    },
    "warning": (
        "The saved dsm_absolute.tif uses a local pixel-space transform "
        "for this alignment sample. The authoritative source georeferencing "
        "is the original S-EO DSM GeoTransform plus RPC."
    ),
}

(OUT_ROOT / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

# Simple diagnostic image: RGB and target normalized side-by-side.
# Avoid matplotlib dependency here.
rgb_small = rgb_crop.astype(np.float32)
rgb_small = np.clip(rgb_small, 0, 255).astype(np.uint8)

target_vis = target.copy()
target_vis[~valid_target] = np.nan

lo = np.nanpercentile(target_vis, 2)
hi = np.nanpercentile(target_vis, 98)

scaled = np.zeros(target_vis.shape, dtype=np.uint8)
finite = np.isfinite(target_vis)

if hi > lo:
    scaled[finite] = np.clip(
        (target_vis[finite] - lo) / (hi - lo) * 255.0,
        0,
        255
    ).astype(np.uint8)

target_rgb = np.stack([scaled] * 3, axis=-1)
target_rgb[~finite] = 0

preview = np.concatenate([rgb_small, target_rgb], axis=1)
Image.fromarray(preview).save(OUT_ROOT / "alignment_preview.png")

print("\n" + "=" * 78)
print("S-EO ALIGNMENT SAMPLE COMPLETE")
print("=" * 78)
print("Output directory:", OUT_ROOT.resolve())
print("RGB:", (OUT_ROOT / "rgb.png").resolve())
print("DSM:", (OUT_ROOT / "dsm_absolute.tif").resolve())
print("Mask:", (OUT_ROOT / "valid_mask.tif").resolve())
print("Metadata:", (OUT_ROOT / "metadata.json").resolve())
print("Preview:", (OUT_ROOT / "alignment_preview.png").resolve())

print("\nTARGET SEMANTICS:")
print("  DSM-Max absolute elevation")
print("  NOT nDSM")
print("  No DTM subtraction was performed")

print("\nNo model training was performed.")
