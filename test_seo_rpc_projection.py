from datasets import load_dataset
import numpy as np
from pyproj import Transformer


AOI = "OMA_135"
ACQUISITION = "40"


# ============================================================
# 1. LOAD RPC
# ============================================================

print("=" * 70)
print("S-EO RPC PROJECTION TEST")
print("=" * 70)

print("\n[1] Loading RPC...")

rpc_ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
    data_files={"train": "rpcs.tar.gz"},
)

rpc_data = None

for sample in rpc_ds:
    key = sample["__key__"]

    if key.endswith(f"{AOI}_{ACQUISITION}_pan"):
        rpc_data = sample["json"]
        break

if rpc_data is None:
    raise RuntimeError("RPC not found.")

print("Image:", rpc_data["img"])
print("Width:", rpc_data["width"])
print("Height:", rpc_data["height"])

geojson = rpc_data["geojson"]

print("\nRPC footprint:")
print(geojson)


# ============================================================
# 2. LOAD DSM
# ============================================================

print("\n[2] Loading DSM-Max...")

dsm_ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
)

dsm_sample = None

for sample in dsm_ds:
    if f"/{AOI}/" in sample["__key__"]:
        dsm_sample = sample
        break

if dsm_sample is None:
    raise RuntimeError("DSM not found.")

dsm = np.asarray(
    dsm_sample["tif"],
    dtype=np.float64,
)

aux_xml = dsm_sample["tif.aux.xml"].decode(
    "utf-8",
    errors="ignore",
)

print("DSM key:", dsm_sample["__key__"])
print("DSM shape:", dsm.shape)


# ============================================================
# 3. EXTRACT GEOTRANSFORM
# ============================================================

import re

match = re.search(
    r"<GeoTransform>\s*([^<]+)\s*</GeoTransform>",
    aux_xml,
)

if not match:
    raise RuntimeError("GeoTransform not found.")

gt = [
    float(x.strip())
    for x in match.group(1).split(",")
]

print("\nGeoTransform:")
print(gt)


# GDAL:
#
# X = GT0 + col*GT1 + row*GT2
# Y = GT3 + col*GT4 + row*GT5

gt0, gt1, gt2, gt3, gt4, gt5 = gt


# ============================================================
# 4. UTM -> WGS84
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:32614",
    "EPSG:4326",
    always_xy=True,
)


# ============================================================
# 5. RPC IMPLEMENTATION
# ============================================================

rpc = rpc_data["rpc"]


def normalize(value, offset, scale):
    return (value - offset) / scale


def denormalize(value, offset, scale):
    return value * scale + offset


def rpc_ratio(
    P,
    L,
    H,
    numerator,
    denominator,
):
    """
    Evaluate RPC rational polynomial.

    RPC variables:
        P = normalized latitude
        L = normalized longitude
        H = normalized altitude

    Polynomial order follows standard RPC 20-term convention.
    """

    terms = np.array([
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

    return np.dot(numerator, terms) / np.dot(denominator, terms)


def project_ground_to_image(lat, lon, alt):
    """
    Ground coordinate -> RPC image row/column.
    """

    P = normalize(
        lat,
        rpc["lat_offset"],
        rpc["lat_scale"],
    )

    L = normalize(
        lon,
        rpc["lon_offset"],
        rpc["lon_scale"],
    )

    H = normalize(
        alt,
        rpc["alt_offset"],
        rpc["alt_scale"],
    )

    row_n = rpc_ratio(
        P,
        L,
        H,
        rpc["row_num"],
        rpc["row_den"],
    )

    col_n = rpc_ratio(
        P,
        L,
        H,
        rpc["col_num"],
        rpc["col_den"],
    )

    row = denormalize(
        row_n,
        rpc["row_offset"],
        rpc["row_scale"],
    )

    col = denormalize(
        col_n,
        rpc["col_offset"],
        rpc["col_scale"],
    )

    return row, col


# ============================================================
# 6. FIVE DSM TEST POINTS
# ============================================================

height, width = dsm.shape

points = [
    ("TOP_LEFT", 0, 0),
    ("TOP_RIGHT", 0, width - 1),
    ("CENTER", height // 2, width // 2),
    ("BOTTOM_LEFT", height - 1, 0),
    ("BOTTOM_RIGHT", height - 1, width - 1),
]


print("\n" + "=" * 70)
print("PROJECTING DSM → RGB")
print("=" * 70)

valid_count = 0


for name, row, col in points:

    elevation = dsm[row, col]

    x = (
        gt0
        + col * gt1
        + row * gt2
    )

    y = (
        gt3
        + col * gt4
        + row * gt5
    )

    lon, lat = transformer.transform(
        x,
        y,
    )

    img_row, img_col = project_ground_to_image(
        lat,
        lon,
        elevation,
    )

    inside = (
        0 <= img_row < rpc_data["height"]
        and
        0 <= img_col < rpc_data["width"]
    )

    if inside:
        valid_count += 1

    print("\n--------------------------------")

    print("Point:", name)

    print(
        f"DSM pixel: row={row}, col={col}"
    )

    print(
        f"Elevation: {elevation:.3f} m"
    )

    print(
        f"UTM: X={x:.3f}, Y={y:.3f}"
    )

    print(
        f"Lat/Lon: {lat:.8f}, {lon:.8f}"
    )

    print(
        f"RGB pixel: row={img_row:.3f}, "
        f"col={img_col:.3f}"
    )

    print(
        "Inside RGB:",
        "YES" if inside else "NO"
    )


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Projected points inside RGB: "
    f"{valid_count}/{len(points)}"
)

print(
    f"RGB dimensions: "
    f"{rpc_data['height']} × {rpc_data['width']}"
)

print(
    f"DSM dimensions: "
    f"{height} × {width}"
)

if valid_count == len(points):
    print("\nRPC PROJECTION TEST: PASSED")
else:
    print("\nRPC PROJECTION TEST: NEEDS INVESTIGATION")