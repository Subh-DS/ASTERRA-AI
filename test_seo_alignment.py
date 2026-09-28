import io
import json

import numpy as np
from PIL import Image
from datasets import load_dataset
from pyproj import Transformer
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.crs import CRS
import xml.etree.ElementTree as ET


# ============================================================
# CONFIG
# ============================================================

REPO = "emasquil/shadow-eo"

AOI = "OMA_135"
ACQ = "40"

RGB_FILE = "pansharpened_crops_color_corrected.part.tar.gz.aa"
DSM_FILE = "dsm_max.tar.gz"
RPC_FILE = "rpcs.tar.gz"

OUT = r"D:\Asterra AI\outputs\seo_alignment_oma135_40.npz"


# ============================================================
# RPC POLYNOMIAL
# Standard RPC 20-term ordering
# ============================================================

def rpc_terms(L, P, H):
    return np.array([
        1,
        L,
        P,
        H,
        L * P,
        L * H,
        P * H,
        L * L,
        P * P,
        H * H,
        P * L * H,
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


def rpc_eval(coeff, L, P, H):
    t = rpc_terms(L, P, H)
    return float(np.dot(coeff, t))


def rpc_project(rpc, lon, lat, elev):
    """
    Project WGS84 longitude/latitude/elevation into S-EO image
    coordinates using the standard RPC 20-term polynomial.
    """

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    elev = np.asarray(elev, dtype=np.float64)

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(elev)
        & np.isfinite(rpc["lon_scale"])
        & np.isfinite(rpc["lat_scale"])
        & np.isfinite(rpc["height_scale"])
        & (rpc["lon_scale"] != 0)
        & (rpc["lat_scale"] != 0)
        & (rpc["height_scale"] != 0)
    )

    L = np.zeros_like(lon, dtype=np.float64)
    P = np.zeros_like(lat, dtype=np.float64)
    H = np.zeros_like(elev, dtype=np.float64)

    L[valid] = (lon[valid] - rpc["lon_off"]) / rpc["lon_scale"]
    P[valid] = (lat[valid] - rpc["lat_off"]) / rpc["lat_scale"]
    H[valid] = (elev[valid] - rpc["height_off"]) / rpc["height_scale"]

    t = np.stack([
        np.ones_like(L),
        L, P, H,
        L * P, L * H, P * H,
        L * L, P * P, H * H,
        P * L * H,
        L * L * L,
        L * P * P,
        L * H * H,
        L * L * P,
        P * P * P,
        P * H * H,
        L * L * H,
        P * P * H,
        H * H * H,
    ], axis=0)

    row_num = np.tensordot(np.asarray(rpc["line_num"], dtype=np.float64), t, axes=(0, 0))
    row_den = np.tensordot(np.asarray(rpc["line_den"], dtype=np.float64), t, axes=(0, 0))
    col_num = np.tensordot(np.asarray(rpc["samp_num"], dtype=np.float64), t, axes=(0, 0))
    col_den = np.tensordot(np.asarray(rpc["samp_den"], dtype=np.float64), t, axes=(0, 0))

    good = (
        valid
        & np.isfinite(row_num)
        & np.isfinite(row_den)
        & np.isfinite(col_num)
        & np.isfinite(col_den)
        & (np.abs(row_den) > 1e-12)
        & (np.abs(col_den) > 1e-12)
    )

    row = np.full_like(lon, np.nan, dtype=np.float64)
    col = np.full_like(lon, np.nan, dtype=np.float64)

    row[good] = (
        (row_num[good] / row_den[good]) * rpc["line_scale"]
        + rpc["line_off"]
    )
    col[good] = (
        (col_num[good] / col_den[good]) * rpc["samp_scale"]
        + rpc["samp_off"]
    )

    return row, col

# ============================================================
# PARSE RPC JSON
# ============================================================

def parse_rpc(data):
    """
    Parse the exact S-EO RPC JSON structure.

    The actual S-EO record contains a nested `rpc` dictionary:
        row_offset, col_offset, lat_offset, lon_offset, alt_offset
        row_scale, col_scale, lat_scale, lon_scale, alt_scale
        row_num, row_den, col_num, col_den
    """

    if not isinstance(data, dict):
        raise TypeError(f"RPC metadata must be dict, got {type(data)}")

    rpc_data = data.get("rpc")
    if not isinstance(rpc_data, dict):
        raise KeyError("S-EO RPC JSON does not contain nested 'rpc' dictionary.")

    def required(name):
        if name not in rpc_data:
            raise KeyError(
                f"Missing S-EO RPC field '{name}'. "
                f"Available RPC keys: {list(rpc_data.keys())}"
            )
        return rpc_data[name]

    def coeffs(value, name):
        if isinstance(value, (list, tuple, np.ndarray)):
            out = [float(x) for x in value]
        elif isinstance(value, dict):
            out = [
                float(v) for _, v in sorted(
                    value.items(), key=lambda kv: int(str(kv[0]))
                )
            ]
        else:
            raise TypeError(f"{name} has unsupported type {type(value)}")

        if len(out) != 20:
            raise ValueError(f"{name} must contain 20 coefficients; got {len(out)}")
        return out

    parsed = {
        "line_off": float(required("row_offset")),
        "samp_off": float(required("col_offset")),
        "lat_off": float(required("lat_offset")),
        "lon_off": float(required("lon_offset")),
        "height_off": float(required("alt_offset")),

        "line_scale": float(required("row_scale")),
        "samp_scale": float(required("col_scale")),
        "lat_scale": float(required("lat_scale")),
        "lon_scale": float(required("lon_scale")),
        "height_scale": float(required("alt_scale")),

        "line_num": coeffs(required("row_num"), "row_num"),
        "line_den": coeffs(required("row_den"), "row_den"),
        "samp_num": coeffs(required("col_num"), "col_num"),
        "samp_den": coeffs(required("col_den"), "col_den"),
    }

    print("\nRPC PARSER: PASS")
    print("  Using nested S-EO 'rpc' object")
    print("  row_offset :", parsed["line_off"])
    print("  col_offset :", parsed["samp_off"])
    print("  lat_offset :", parsed["lat_off"])
    print("  lon_offset :", parsed["lon_off"])
    print("  alt_offset :", parsed["height_off"])
    print("  row_scale  :", parsed["line_scale"])
    print("  col_scale  :", parsed["samp_scale"])
    print("  lat_scale  :", parsed["lat_scale"])
    print("  lon_scale  :", parsed["lon_scale"])
    print("  alt_scale  :", parsed["height_scale"])
    print("  coefficients: 20 / 20 / 20 / 20")

    return parsed

# ============================================================
# DATASET HELPER
# ============================================================

def find_sample(ds, key_contains):
    """Return the first streamed sample whose __key__ contains key_contains."""
    for sample in ds:
        key = sample.get("__key__", "")
        if key_contains in key:
            return sample

    raise RuntimeError(
        f"Could not find sample containing key: {key_contains}"
    )


# ============================================================
# RGB DECODER
# ============================================================

def decode_rgb(rgb_value):

    # Hugging Face datasets may already decode PNG
    # into a PIL.Image.Image object.
    if isinstance(rgb_value, Image.Image):

        return np.asarray(
            rgb_value.convert("RGB")
        )

    # Some configurations return raw bytes.
    if isinstance(rgb_value, (bytes, bytearray)):

        return np.asarray(
            Image.open(
                io.BytesIO(rgb_value)
            ).convert("RGB")
        )

    # Some dataset representations may provide
    # a dictionary containing bytes.
    if isinstance(rgb_value, dict):

        if "bytes" in rgb_value:

            return np.asarray(
                Image.open(
                    io.BytesIO(rgb_value["bytes"])
                ).convert("RGB")
            )

        if "path" in rgb_value:

            return np.asarray(
                Image.open(
                    rgb_value["path"]
                ).convert("RGB")
            )

    raise TypeError(
        "Unsupported RGB sample type: "
        f"{type(rgb_value)}"
    )


# ============================================================
# DSM DECODER
# ============================================================

def parse_aux_xml_metadata(aux_xml):
    """
    Recover GeoTransform and CRS from GDAL PAM .aux.xml metadata.

    S-EO exposes the DSM as a decoded PIL TIFF, while the associated
    tif.aux.xml contains the spatial metadata needed for geospatial
    alignment.
    """

    if aux_xml is None:
        raise RuntimeError("No DSM auxiliary XML metadata was provided.")

    if isinstance(aux_xml, dict):
        if "bytes" in aux_xml:
            aux_xml = aux_xml["bytes"]
        elif "text" in aux_xml:
            aux_xml = aux_xml["text"]

    if isinstance(aux_xml, bytes):
        aux_xml = aux_xml.decode("utf-8", errors="replace")

    if not isinstance(aux_xml, str):
        raise TypeError(
            f"Unsupported aux XML type: {type(aux_xml)}"
        )

    root = ET.fromstring(aux_xml)

    geotransform_text = None
    srs_text = None

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag == "GeoTransform" and elem.text:
            geotransform_text = elem.text.strip()

        elif tag == "SRS" and elem.text:
            srs_text = elem.text.strip()

    if geotransform_text is None:
        raise RuntimeError(
            "DSM aux.xml does not contain a GeoTransform element."
        )

    values = [
        float(x.strip())
        for x in geotransform_text.replace(",", " ").split()
    ]

    if len(values) != 6:
        raise RuntimeError(
            "Expected 6 GeoTransform values, got "
            f"{len(values)}: {values}"
        )

    transform = Affine(*values)

    if srs_text:
        try:
            crs = CRS.from_wkt(srs_text)
        except Exception:
            # Some PAM files may store a CRS identifier instead of
            # full WKT. Try pyproj/rasterio parsing as a fallback.
            try:
                crs = CRS.from_string(srs_text)
            except Exception as e:
                raise RuntimeError(
                    "Could not parse CRS from DSM aux.xml."
                ) from e
    else:
        # The previously validated S-EO OMA_135 DSM metadata is
        # WGS84 / UTM zone 14N, EPSG:32614.
        #
        # This fallback is intentionally restricted to this exact
        # validated sample rather than silently guessing CRS.
        if AOI == "OMA_135":
            print(
                "WARNING: aux.xml has no SRS element. "
                "Using the previously validated OMA_135 DSM CRS: "
                "EPSG:32614."
            )
            crs = CRS.from_epsg(32614)
        else:
            raise RuntimeError(
                "DSM aux.xml has no SRS element and no validated "
                "sample-specific CRS fallback is available."
            )

    return transform, crs


def decode_dsm(tif_value, aux_xml=None):

    # ========================================================
    # CASE 1:
    # Hugging Face decoded the GeoTIFF as a PIL TIFF image.
    #
    # IMPORTANT:
    # The decoded PIL object may not preserve CRS/GeoTransform.
    # Therefore we recover spatial metadata from the associated
    # tif.aux.xml when available.
    # ========================================================
    if isinstance(tif_value, Image.Image):

        print(
            "DSM is already decoded as a PIL image."
        )

        # First attempt: use the original TIFF bytes with rasterio.
        # This is useful when the TIFF itself contains GeoTIFF tags.
        fp = getattr(tif_value, "fp", None)

        if fp is not None:

            try:
                current_pos = fp.tell()
                fp.seek(0)
                tif_bytes = fp.read()
                fp.seek(current_pos)

                if isinstance(
                    tif_bytes,
                    (bytes, bytearray),
                ) and len(tif_bytes) > 0:

                    with MemoryFile(bytes(tif_bytes)) as mem:

                        with mem.open() as src:

                            dsm = src.read(1)
                            transform = src.transform
                            crs = src.crs

                            if crs is not None:
                                print(
                                    "Recovered CRS and GeoTransform "
                                    "directly from the TIFF."
                                )

                                return (
                                    dsm,
                                    transform,
                                    crs,
                                )

                            print(
                                "TIFF bytes recovered, but embedded "
                                "CRS is missing."
                            )

            except Exception as e:

                print(
                    "Direct TIFF metadata recovery failed:"
                )
                print(" ", repr(e))

        # Second attempt: recover spatial metadata from aux.xml.
        if aux_xml is not None:

            transform, crs = parse_aux_xml_metadata(
                aux_xml
            )

            dsm = np.asarray(
                tif_value,
                dtype=np.float32,
            )

            print(
                "Recovered DSM spatial metadata from "
                "tif.aux.xml."
            )

            return (
                dsm,
                transform,
                crs,
            )

        # Third attempt: filesystem path, if one exists.
        filename = getattr(
            tif_value,
            "filename",
            None,
        )

        if filename:

            try:

                with rasterio.open(filename) as src:

                    dsm = src.read(1)
                    transform = src.transform
                    crs = src.crs

                    if crs is not None:
                        return (
                            dsm,
                            transform,
                            crs,
                        )

            except Exception as e:

                print(
                    "PIL filename recovery failed:"
                )
                print(" ", repr(e))

        raise RuntimeError(
            "DSM is a PIL TIFF image, but its CRS/GeoTransform "
            "could not be recovered from the TIFF or aux.xml."
        )

    # ========================================================
    # CASE 2:
    # Raw TIFF bytes.
    # ========================================================
    if isinstance(
        tif_value,
        (bytes, bytearray),
    ):

        with MemoryFile(bytes(tif_value)) as mem:

            with mem.open() as src:

                dsm = src.read(1)
                transform = src.transform
                crs = src.crs

                if crs is None:

                    if aux_xml is not None:

                        transform, crs = (
                            parse_aux_xml_metadata(
                                aux_xml
                            )
                        )

                    else:

                        raise RuntimeError(
                            "DSM GeoTIFF has no CRS and no "
                            "aux.xml metadata was provided."
                        )

                return (
                    dsm,
                    transform,
                    crs,
                )

    # ========================================================
    # CASE 3:
    # Dictionary containing bytes/path.
    # ========================================================
    if isinstance(
        tif_value,
        dict,
    ):

        if "bytes" in tif_value:

            return decode_dsm(
                tif_value["bytes"],
                aux_xml=aux_xml,
            )

        if "path" in tif_value:

            with rasterio.open(
                tif_value["path"]
            ) as src:

                dsm = src.read(1)
                transform = src.transform
                crs = src.crs

                if crs is None and aux_xml is not None:
                    transform, crs = (
                        parse_aux_xml_metadata(aux_xml)
                    )

                if crs is None:
                    raise RuntimeError(
                        "DSM GeoTIFF has no CRS."
                    )

                return (
                    dsm,
                    transform,
                    crs,
                )

    # ========================================================
    # CASE 4:
    # NumPy array.
    # ========================================================
    if isinstance(
        tif_value,
        np.ndarray,
    ):

        if aux_xml is not None:

            transform, crs = (
                parse_aux_xml_metadata(aux_xml)
            )

            return (
                tif_value.astype(np.float32),
                transform,
                crs,
            )

        raise RuntimeError(
            "DSM was returned as a NumPy array without "
            "GeoTIFF transform/CRS metadata. Cannot perform "
            "safe RPC alignment."
        )

    raise TypeError(
        "Unsupported DSM sample type: "
        f"{type(tif_value)}"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("S-EO RGB ↔ DSM ALIGNMENT VALIDATION")
print("=" * 70)


# ============================================================
# LOAD RGB
# ============================================================

print("\nLoading RGB stream...")

rgb_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={
        "train": RGB_FILE
    },
)

rgb_key = f"{AOI}_{ACQ}_rgb"

rgb_sample = find_sample(
    rgb_ds,
    rgb_key,
)

print(
    "RGB key :",
    rgb_sample["__key__"]
)

print(
    "RGB sample type:",
    type(rgb_sample["png"])
)

rgb = decode_rgb(
    rgb_sample["png"]
)

rgb_h, rgb_w = rgb.shape[:2]

print(
    "RGB shape:",
    rgb.shape
)

print(
    "RGB dtype:",
    rgb.dtype
)

print(
    "RGB min:",
    rgb.min()
)

print(
    "RGB max:",
    rgb.max()
)

print(
    "RGB mean:",
    rgb.mean()
)


# ============================================================
# LOAD DSM
# ============================================================

print("\nLoading DSM stream...")

dsm_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={
        "train": DSM_FILE
    },
)

dsm_key = AOI

dsm_sample = find_sample(
    dsm_ds,
    dsm_key,
)

print(
    "DSM key :",
    dsm_sample["__key__"]
)

print(
    "DSM sample type:",
    type(dsm_sample["tif"])
)

print(
    "DSM sample fields:",
    list(dsm_sample.keys())
)

dsm_aux_xml = dsm_sample.get(
    "tif.aux.xml"
)

print(
    "DSM aux.xml available:",
    dsm_aux_xml is not None
)

dsm, transform, crs = decode_dsm(
    dsm_sample["tif"],
    aux_xml=dsm_aux_xml,
)

dsm_h, dsm_w = dsm.shape

print(
    "DSM shape :",
    dsm.shape
)

print(
    "DSM dtype :",
    dsm.dtype
)

print(
    "DSM CRS   :",
    crs
)

print(
    "DSM min   :",
    np.nanmin(dsm)
)

print(
    "DSM max   :",
    np.nanmax(dsm)
)

print(
    "DSM mean  :",
    np.nanmean(dsm)
)

print(
    "DSM transform:",
    transform
)


# ============================================================
# LOAD RPC
# ============================================================

print("\nLoading RPC stream...")

rpc_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={
        "train": RPC_FILE
    },
)

rpc_key = f"{AOI}_{ACQ}_pan"

rpc_sample = find_sample(
    rpc_ds,
    rpc_key,
)

print(
    "RPC key :",
    rpc_sample["__key__"]
)

rpc_value = rpc_sample["json"]

print("RPC JSON type:", type(rpc_value))
if isinstance(rpc_value, dict):
    print("RPC JSON keys:", list(rpc_value.keys()))

if isinstance(rpc_value, dict):
    rpc_data = rpc_value
elif isinstance(rpc_value, (str, bytes, bytearray)):
    rpc_data = json.loads(rpc_value)
else:
    raise TypeError(
        f"Unsupported RPC JSON type: {type(rpc_value)}"
    )


rpc = parse_rpc(
    rpc_data
)

print("\nRPC COEFFICIENT SANITY")
print("  row_num first 5:", rpc["line_num"][:5])
print("  row_den first 5:", rpc["line_den"][:5])
print("  col_num first 5:", rpc["samp_num"][:5])
print("  col_den first 5:", rpc["samp_den"][:5])

center_row, center_col = rpc_project(
    rpc,
    np.array(rpc["lon_off"], dtype=np.float64),
    np.array(rpc["lat_off"], dtype=np.float64),
    np.array(rpc["height_off"], dtype=np.float64),
)

print("\nRPC CENTER SANITY CHECK")
print("  projected row:", float(np.asarray(center_row)))
print("  projected col:", float(np.asarray(center_col)))
print("  expected row :", float(rpc_data["height"]) / 2.0)
print("  expected col :", float(rpc_data["width"]) / 2.0)


print(
    "RPC image width :",
    rpc_data.get("width")
)

print(
    "RPC image height:",
    rpc_data.get("height")
)


# ============================================================
# VALIDATE RGB / RPC DIMENSIONS
# ============================================================

rpc_width = int(
    rpc_data.get(
        "width",
        0
    )
)

rpc_height = int(
    rpc_data.get(
        "height",
        0
    )
)

print(
    "\nRGB dimensions:",
    rgb_w,
    "x",
    rgb_h
)

print(
    "RPC dimensions:",
    rpc_width,
    "x",
    rpc_height
)

if (
    rpc_width != rgb_w
    or rpc_height != rgb_h
):

    print(
        "\nWARNING:"
    )

    print(
        "RPC image dimensions do not exactly "
        "match the RGB image dimensions."
    )

else:

    print(
        "RPC ↔ RGB dimensions: MATCH"
    )


# ============================================================
# CRS TRANSFORMATION
# ============================================================

print(
    "\nPreparing CRS transformation..."
)

if crs is None:

    raise RuntimeError(
        "DSM CRS is missing. "
        "Cannot perform safe geographic transformation."
    )

transformer = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True,
)


# ============================================================
# PROJECT DSM GRID INTO RGB
# ============================================================

print(
    "\nProjecting DSM grid into RGB coordinate space..."
)

# Every 10th DSM pixel.
step = 10

rows = []
cols = []
heights = []
rgb_rows = []
rgb_cols = []

for r in range(
    0,
    dsm_h,
    step
):

    for c in range(
        0,
        dsm_w,
        step
    ):

        z = float(
            dsm[r, c]
        )

        if not np.isfinite(z):
            continue

        # DSM pixel center -> map coordinates.
        x, y = transform * (
            c + 0.5,
            r + 0.5
        )

        # Map coordinates -> WGS84.
        lon, lat = transformer.transform(
            x,
            y
        )

        # WGS84 + elevation -> RGB pixel.
        rr, cc = rpc_project(
            rpc,
            lon,
            lat,
            z
        )

        if (
            not np.isfinite(rr)
            or not np.isfinite(cc)
        ):
            continue

        rows.append(r)
        cols.append(c)
        heights.append(z)
        rgb_rows.append(rr)
        rgb_cols.append(cc)


rows = np.asarray(
    rows,
    dtype=np.int32
)

cols = np.asarray(
    cols,
    dtype=np.int32
)

heights = np.asarray(
    heights,
    dtype=np.float32
)

rgb_rows = np.asarray(
    rgb_rows,
    dtype=np.float64
)

rgb_cols = np.asarray(
    rgb_cols,
    dtype=np.float64
)


# ============================================================
# INSIDE / OUTSIDE TEST
# ============================================================

inside = (
    (rgb_rows >= 0)
    & (rgb_rows < rgb_h)
    & (rgb_cols >= 0)
    & (rgb_cols < rgb_w)
)


print(
    "\n" + "=" * 70
)

print(
    "ALIGNMENT RESULTS"
)

print(
    "=" * 70
)

print(
    "Projected points:",
    len(rgb_rows)
)

print(
    "Inside RGB      :",
    int(inside.sum())
)

print(
    "Outside RGB     :",
    int((~inside).sum())
)

if len(rgb_rows) > 0:

    inside_pct = (
        100.0
        * inside.mean()
    )

    print(
        "Inside percentage:",
        f"{inside_pct:.2f}%"
    )

else:

    inside_pct = 0.0


if len(rgb_rows) > 0:

    print(
        "\nRGB projected coordinate range:"
    )

    print(
        "row:",
        float(rgb_rows.min()),
        "->",
        float(rgb_rows.max())
    )

    print(
        "col:",
        float(rgb_cols.min()),
        "->",
        float(rgb_cols.max())
    )


# ============================================================
# INTERIOR COVERAGE
# ============================================================

if len(rgb_rows) > 0:

    interior = (
        (rgb_rows >= 0.05 * rgb_h)
        & (rgb_rows <= 0.95 * rgb_h)
        & (rgb_cols >= 0.05 * rgb_w)
        & (rgb_cols <= 0.95 * rgb_w)
    )

    print(
        "\nInterior RGB coverage:"
    )

    print(
        int(interior.sum()),
        "/",
        len(interior)
    )

    print(
        f"{100 * interior.mean():.2f}%"
    )


# ============================================================
# BUILD ALIGNED SAMPLE
# ============================================================

valid = inside

if valid.sum() == 0:

    raise RuntimeError(
        "No DSM points project inside RGB. "
        "Alignment failed."
    )


sample_count = min(
    10000,
    int(valid.sum())
)

rng = np.random.default_rng(
    42
)

idx = np.flatnonzero(
    valid
)

if len(idx) > sample_count:

    idx = rng.choice(
        idx,
        size=sample_count,
        replace=False
    )


sample_rgb_rows = (
    rgb_rows[idx]
)

sample_rgb_cols = (
    rgb_cols[idx]
)

sample_heights = (
    heights[idx]
)


# ============================================================
# BILINEAR RGB SAMPLING
# ============================================================

def bilinear_rgb(
    image,
    rows,
    cols
):

    h, w = image.shape[:2]

    rows = np.clip(
        rows,
        0,
        h - 1.001
    )

    cols = np.clip(
        cols,
        0,
        w - 1.001
    )

    r0 = np.floor(
        rows
    ).astype(int)

    c0 = np.floor(
        cols
    ).astype(int)

    r1 = np.minimum(
        r0 + 1,
        h - 1
    )

    c1 = np.minimum(
        c0 + 1,
        w - 1
    )

    dr = (
        rows - r0
    )

    dc = (
        cols - c0
    )

    v00 = image[
        r0,
        c0
    ].astype(
        np.float32
    )

    v01 = image[
        r0,
        c1
    ].astype(
        np.float32
    )

    v10 = image[
        r1,
        c0
    ].astype(
        np.float32
    )

    v11 = image[
        r1,
        c1
    ].astype(
        np.float32
    )

    dr = dr[:, None]
    dc = dc[:, None]

    values = (
        v00 * (1 - dr) * (1 - dc)
        +
        v01 * (1 - dr) * dc
        +
        v10 * dr * (1 - dc)
        +
        v11 * dr * dc
    )

    return values


sample_rgb = bilinear_rgb(
    rgb,
    sample_rgb_rows,
    sample_rgb_cols
)


# ============================================================
# ALIGNMENT SAMPLE STATISTICS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ALIGNED SAMPLE"
)

print(
    "=" * 70
)

print(
    "Sample points:",
    len(sample_heights)
)

print(
    "Height min:",
    float(sample_heights.min())
)

print(
    "Height max:",
    float(sample_heights.max())
)

print(
    "Height mean:",
    float(sample_heights.mean())
)

print(
    "RGB sample shape:",
    sample_rgb.shape
)

print(
    "RGB sample mean:",
    float(sample_rgb.mean())
)

print(
    "RGB sample std:",
    float(sample_rgb.std())
)


# ============================================================
# SAVE VALIDATION DATA
# ============================================================

np.savez_compressed(
    OUT,
    rgb=rgb,
    dsm_heights=sample_heights,
    rgb_rows=sample_rgb_rows,
    rgb_cols=sample_rgb_cols,
    sampled_rgb=sample_rgb,
)

print(
    "\nSaved:"
)

print(
    OUT
)


# ============================================================
# FINAL GATE
# ============================================================

print(
    "\n" + "=" * 70
)

if inside_pct >= 95:

    print(
        "ALIGNMENT GATE: PASS"
    )

    print(
        "RGB ↔ DSM geometry is sufficiently "
        "consistent for Stage-2 pair generation."
    )

elif inside_pct >= 90:

    print(
        "ALIGNMENT GATE: PASS WITH MARGIN"
    )

    print(
        "RGB ↔ DSM geometry is usable, "
        "but inspect the alignment sample before training."
    )

else:

    print(
        "ALIGNMENT GATE: FAIL"
    )

    print(
        "Do NOT start Stage-2 training."
    )

print(
    "=" * 70
)
