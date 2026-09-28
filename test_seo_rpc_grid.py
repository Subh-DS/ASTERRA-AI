from datasets import load_dataset
import numpy as np
from pyproj import Transformer
import re


AOI = "OMA_135"
ACQUISITION = "40"

print("=" * 70)
print("S-EO RPC GRID VALIDATION")
print("=" * 70)


# ============================================================
# 1. RPC
# ============================================================

rpc_ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
    data_files={"train": "rpcs.tar.gz"},
)

rpc_data = None

for sample in rpc_ds:
    if sample["__key__"].endswith(
        f"{AOI}_{ACQUISITION}_pan"
    ):
        rpc_data = sample["json"]
        break

if rpc_data is None:
    raise RuntimeError("RPC not found")


print("\nRPC:")
print(rpc_data["img"])
print("width :", rpc_data["width"])
print("height:", rpc_data["height"])


# ============================================================
# 2. DSM
# ============================================================

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
    raise RuntimeError("DSM not found")


dsm = np.asarray(
    dsm_sample["tif"],
    dtype=np.float64
)

aux = dsm_sample["tif.aux.xml"].decode(
    "utf-8",
    errors="ignore"
)

match = re.search(
    r"<GeoTransform>\s*([^<]+)\s*</GeoTransform>",
    aux
)

if not match:
    raise RuntimeError("GeoTransform not found")

gt = [
    float(x.strip())
    for x in match.group(1).split(",")
]

gt0, gt1, gt2, gt3, gt4, gt5 = gt


print("\nDSM:")
print("shape:", dsm.shape)
print("valid pixels:", np.isfinite(dsm).sum())
print("invalid pixels:", np.isnan(dsm).sum())


# ============================================================
# 3. UTM → WGS84
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:32614",
    "EPSG:4326",
    always_xy=True
)


# ============================================================
# 4. RPC
# ============================================================

rpc = rpc_data["rpc"]


def rpc_ratio(P, L, H, numerator, denominator):

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
    ])

    return (
        np.dot(numerator, terms)
        /
        np.dot(denominator, terms)
    )


def project(lat, lon, alt):

    P = (
        lat - rpc["lat_offset"]
    ) / rpc["lat_scale"]

    L = (
        lon - rpc["lon_offset"]
    ) / rpc["lon_scale"]

    H = (
        alt - rpc["alt_offset"]
    ) / rpc["alt_scale"]

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

    row = (
        row_n * rpc["row_scale"]
        + rpc["row_offset"]
    )

    col = (
        col_n * rpc["col_scale"]
        + rpc["col_offset"]
    )

    return row, col


# ============================================================
# 5. SAMPLE GRID
# ============================================================

H, W = dsm.shape

# Avoid exact outer boundary.
# Sample approximately every 100 pixels.
rows = np.linspace(
    50,
    H - 51,
    10
).astype(int)

cols = np.linspace(
    50,
    W - 51,
    10
).astype(int)


results = []


print("\nProjecting grid...")

for r in rows:

    for c in cols:

        elevation = dsm[r, c]

        if not np.isfinite(elevation):
            continue

        x = (
            gt0
            + c * gt1
            + r * gt2
        )

        y = (
            gt3
            + c * gt4
            + r * gt5
        )

        lon, lat = transformer.transform(
            x,
            y
        )

        img_row, img_col = project(
            lat,
            lon,
            elevation
        )

        results.append(
            (
                r,
                c,
                elevation,
                lat,
                lon,
                img_row,
                img_col
            )
        )


results = np.asarray(results)


# ============================================================
# 6. RESULTS
# ============================================================

rows_img = results[:, 5]
cols_img = results[:, 6]

inside = (
    (rows_img >= 0)
    &
    (rows_img < rpc_data["height"])
    &
    (cols_img >= 0)
    &
    (cols_img < rpc_data["width"])
)


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(
    "Total valid projected points:",
    len(results)
)

print(
    "Inside RGB:",
    inside.sum()
)

print(
    "Outside RGB:",
    (~inside).sum()
)

print(
    "Inside percentage:",
    f"{inside.mean() * 100:.2f}%"
)

print("\nProjected RGB coordinates:")

print(
    "row min:",
    rows_img.min()
)

print(
    "row max:",
    rows_img.max()
)

print(
    "col min:",
    cols_img.min()
)

print(
    "col max:",
    cols_img.max()
)

print("\nRGB dimensions:")

print(
    "rows:",
    rpc_data["height"]
)

print(
    "cols:",
    rpc_data["width"]
)


# ============================================================
# 7. INTERIOR 80%
# ============================================================

interior = (
    (rows_img >= 0.05 * rpc_data["height"])
    &
    (rows_img <= 0.95 * rpc_data["height"])
    &
    (cols_img >= 0.05 * rpc_data["width"])
    &
    (cols_img <= 0.95 * rpc_data["width"])
)

print("\nInterior RGB coverage:")
print(
    f"{interior.sum()} / {len(results)}"
)

print(
    f"{interior.mean() * 100:.2f}%"
)


# ============================================================
# 8. CONCLUSION
# ============================================================

print("\n" + "=" * 70)

if inside.mean() >= 0.90:

    print(
        "RPC GRID VALIDATION: PASS"
    )

    print(
        "The DSM and RGB geometry are sufficiently "
        "consistent for the next alignment stage."
    )

elif inside.mean() >= 0.70:

    print(
        "RPC GRID VALIDATION: PARTIAL"
    )

    print(
        "Most of the DSM projects into the RGB image, "
        "but boundary handling needs to be implemented."
    )

else:

    print(
        "RPC GRID VALIDATION: FAIL"
    )

    print(
        "The geometry requires further investigation."
    )

print("=" * 70)