
import io
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.crs import CRS
import xml.etree.ElementTree as ET

REPO = "emasquil/shadow-eo"
AOI = "OMA_135"
ACQ = "40"

RGB_FILE = "pansharpened_crops_color_corrected.part.tar.gz.aa"
DSM_FILE = "dsm_max.tar.gz"
RPC_FILE = "rpcs.tar.gz"


def find_sample(ds, contains):
    for sample in ds:
        if contains in sample["__key__"]:
            return sample
    raise RuntimeError(f"Could not find sample containing: {contains}")


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
        raise RuntimeError("GeoTransform missing/invalid in tif.aux.xml")

    # IMPORTANT:
    # GDAL GeoTransform:
    # [origin_x, pixel_width, rotation_x,
    #  origin_y, rotation_y, pixel_height]
    #
    # Rasterio Affine:
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
    tif = sample["tif"]
    aux = sample.get("tif.aux.xml")

    if isinstance(tif, Image.Image):
        arr = np.asarray(tif, dtype=np.float32)
    else:
        arr = np.asarray(tif, dtype=np.float32)

    if aux is None:
        raise RuntimeError("DSM tif.aux.xml is missing")

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
            raise RuntimeError(f"{k}: expected 20 coefficients, got {len(rpc[k])}")

    return rpc


def rpc_terms(L, P, H):
    return np.array([
        1,
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
    ], dtype=np.float64)


def rpc_project(rpc, lon, lat, height):
    L = (lon - rpc["lon_off"]) / rpc["lon_scale"]
    P = (lat - rpc["lat_off"]) / rpc["lat_scale"]
    H = (height - rpc["alt_off"]) / rpc["alt_scale"]

    t = rpc_terms(L, P, H)

    row_num = np.dot(rpc["row_num"], t)
    row_den = np.dot(rpc["row_den"], t)
    col_num = np.dot(rpc["col_num"], t)
    col_den = np.dot(rpc["col_den"], t)

    if abs(row_den) < 1e-12 or abs(col_den) < 1e-12:
        return np.nan, np.nan

    row = (row_num / row_den) * rpc["row_scale"] + rpc["row_off"]
    col = (col_num / col_den) * rpc["col_scale"] + rpc["col_off"]

    return row, col


print("=" * 78)
print("S-EO RGB ↔ DSM ALIGNMENT — GEOTRANSFORM FIXED")
print("=" * 78)

# -------------------------------------------------------------------------
# RGB
# -------------------------------------------------------------------------
print("\n[1/4] Loading RGB...")
rgb_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={"train": RGB_FILE}
)

rgb_sample = find_sample(
    rgb_ds,
    f"pansharpened_he/{AOI}/{AOI}_{ACQ}_rgb"
)

rgb = decode_rgb(rgb_sample)

print("RGB key :", rgb_sample["__key__"])
print("RGB shape:", rgb.shape)
print("RGB dtype:", rgb.dtype)
print("RGB min :", rgb.min())
print("RGB max :", rgb.max())

rgb_h, rgb_w = rgb.shape[:2]

# -------------------------------------------------------------------------
# DSM
# -------------------------------------------------------------------------
print("\n[2/4] Loading DSM...")
dsm_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={"train": DSM_FILE}
)

dsm_sample = find_sample(
    dsm_ds,
    AOI
)

dsm, transform, dsm_crs, raw_gt = decode_dsm(dsm_sample)

print("DSM key :", dsm_sample["__key__"])
print("DSM shape:", dsm.shape)
print("DSM dtype:", dsm.dtype)
print("DSM min :", np.nanmin(dsm))
print("DSM max :", np.nanmax(dsm))
print("DSM mean:", np.nanmean(dsm))
print("DSM CRS :", dsm_crs)
print("GDAL GeoTransform:", raw_gt)
print("Rasterio Affine:")
print(transform)

expected = Affine(
    raw_gt[1], raw_gt[2], raw_gt[0],
    raw_gt[4], raw_gt[5], raw_gt[3]
)

print("Expected Affine:")
print(expected)

if not np.allclose(
    np.array(transform),
    np.array(expected),
    rtol=0,
    atol=1e-10
):
    raise RuntimeError("Affine transform mismatch")

# -------------------------------------------------------------------------
# RPC
# -------------------------------------------------------------------------
print("\n[3/4] Loading RPC...")
rpc_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={"train": RPC_FILE}
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

print("RPC offsets:")
print("  row:", rpc["row_off"])
print("  col:", rpc["col_off"])
print("  lat:", rpc["lat_off"])
print("  lon:", rpc["lon_off"])
print("  alt:", rpc["alt_off"])

print("RPC scales:")
print("  row:", rpc["row_scale"])
print("  col:", rpc["col_scale"])
print("  lat:", rpc["lat_scale"])
print("  lon:", rpc["lon_scale"])
print("  alt:", rpc["alt_scale"])

# -------------------------------------------------------------------------
# Projection
# -------------------------------------------------------------------------
print("\n[4/4] Projecting DSM → RGB...")

transformer = Transformer.from_crs(
    dsm_crs,
    "EPSG:4326",
    always_xy=True
)

valid = np.isfinite(dsm)

rows, cols = np.where(valid)

# Dense but manageable validation grid.
step = max(1, int(np.sqrt(len(rows) / 2500)))

rows = rows[::step]
cols = cols[::step]

print("Valid DSM pixels sampled:", len(rows))

rgb_rows = []
rgb_cols = []

# Test a few points explicitly first.
print("\nSample projection checks:")

for idx in np.linspace(0, len(rows) - 1, min(10, len(rows)), dtype=int):
    r = int(rows[idx])
    c = int(cols[idx])
    z = float(dsm[r, c])

    x, y = transform * (c + 0.5, r + 0.5)

    lon, lat = transformer.transform(x, y)

    rr, cc = rpc_project(rpc, lon, lat, z)

    print(
        f"DSM ({r:4d},{c:4d}) "
        f"z={z:8.3f} "
        f"WGS84=({lon:12.7f},{lat:11.7f}) "
        f"RGB=({rr:10.3f},{cc:10.3f})"
    )

    if np.isfinite(rr) and np.isfinite(cc):
        rgb_rows.append(rr)
        rgb_cols.append(cc)

rgb_rows = np.asarray(rgb_rows)
rgb_cols = np.asarray(rgb_cols)

if len(rgb_rows) == 0:
    raise RuntimeError("No finite RPC projections")

inside = (
    (rgb_rows >= 0) &
    (rgb_rows < rgb_h) &
    (rgb_cols >= 0) &
    (rgb_cols < rgb_w)
)

print("\n" + "=" * 78)
print("RPC ALIGNMENT RESULT")
print("=" * 78)

print("RGB dimensions:")
print("  rows:", rgb_h)
print("  cols:", rgb_w)

print("Projected finite points:", len(rgb_rows))
print("Inside RGB:", int(inside.sum()))
print("Outside RGB:", int((~inside).sum()))
print("Inside percentage:", float(inside.mean() * 100.0))

print("\nProjected coordinate ranges:")
print("  row min:", float(np.min(rgb_rows)))
print("  row max:", float(np.max(rgb_rows)))
print("  col min:", float(np.min(rgb_cols)))
print("  col max:", float(np.max(rgb_cols)))

if inside.any():
    print("\nInterior coverage:")
    interior = (
        (rgb_rows >= 20) &
        (rgb_rows < rgb_h - 20) &
        (rgb_cols >= 20) &
        (rgb_cols < rgb_w - 20)
    )
    print("  interior points:", int(interior.sum()))
    print("  interior percentage:", float(interior.mean() * 100.0))

# Known geometry sanity check.
print("\nKnown OMA_135 reference check:")
ref_lon = -95.96929316
ref_lat = 41.29886142
ref_h = 315.840

rr, cc = rpc_project(rpc, ref_lon, ref_lat, ref_h)

print("  projected row:", rr)
print("  projected col:", cc)
print("  expected row ≈ 16.847")
print("  expected col ≈ 49.736")

# Save a compact result.
result = {
    "aoi": AOI,
    "acquisition": ACQ,
    "rgb_shape": [int(rgb_h), int(rgb_w)],
    "dsm_shape": [int(dsm.shape[0]), int(dsm.shape[1])],
    "dsm_crs": str(dsm_crs),
    "gdal_geotransform": [float(x) for x in raw_gt],
    "rasterio_affine": [float(x) for x in transform],
    "sampled_valid_dsm_points": int(len(rows)),
    "finite_projected_points": int(len(rgb_rows)),
    "inside_rgb": int(inside.sum()),
    "outside_rgb": int((~inside).sum()),
    "inside_percentage": float(inside.mean() * 100.0),
    "row_min": float(np.min(rgb_rows)),
    "row_max": float(np.max(rgb_rows)),
    "col_min": float(np.min(rgb_cols)),
    "col_max": float(np.max(rgb_cols)),
    "reference_row": float(rr),
    "reference_col": float(cc),
}

out = Path("seo_alignment_geotransform_fixed_result.json")
out.write_text(json.dumps(result, indent=2), encoding="utf-8")

print("\nSaved:", out.resolve())

if inside.mean() >= 0.95:
    print("\nRPC ALIGNMENT GATE: PASS")
elif inside.mean() >= 0.80:
    print("\nRPC ALIGNMENT GATE: PASS WITH COVERAGE CHECK")
else:
    print("\nRPC ALIGNMENT GATE: FAIL")

print("\nNo training data was generated by this script.")
