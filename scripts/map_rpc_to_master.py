import numpy as np
import rasterio
from rasterio.warp import transform
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r".\datasets\SpaceNet_MVS")

GROUP = "MasterProvisional1"

IMAGE_DIR = ROOT / GROUP
MASTER_PATH = ROOT / f"{GROUP}.tif"

OUTPUT_PATH = ROOT / f"{GROUP}_orthorectified_from_RPC.tif"


# ============================================================
# FIND IMAGE
# ============================================================

images = [
    p for p in IMAGE_DIR.glob("*.tif")
    if not p.name.startswith("rpc_")
]

if not images:
    raise RuntimeError("No satellite TIFF found.")

if len(images) != 50:
    print(f"WARNING: expected 50 images, found {len(images)}")

IMAGE_PATH = images[0]

RPC_PATH = IMAGE_DIR / f"rpc_{IMAGE_PATH.stem}.txt"

if not RPC_PATH.exists():
    raise FileNotFoundError(RPC_PATH)


# ============================================================
# RPC LOADING
# ============================================================

def load_rpc(path):

    values = [
        float(x)
        for x in path.read_text().replace(",", " ").split()
    ]

    if len(values) != 96:
        raise RuntimeError(
            f"Expected 96 RPC values, got {len(values)}"
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    line_off = values[0]
    samp_off = values[1]
    lat_off = values[2]
    lon_off = values[3]
    height_off = values[4]

    line_scale = values[5]
    samp_scale = values[6]
    lat_scale = values[7]
    lon_scale = values[8]
    height_scale = values[9]

    # --------------------------------------------------------
    # RPC coefficients
    #
    # 20 numerator coefficients
    # 20 denominator coefficients
    # for LINE
    #
    # 20 numerator coefficients
    # 20 denominator coefficients
    # for SAMPLE
    # --------------------------------------------------------

    line_num = np.asarray(values[10:30], dtype=np.float64)
    line_den = np.asarray(values[30:50], dtype=np.float64)

    samp_num = np.asarray(values[50:70], dtype=np.float64)
    samp_den = np.asarray(values[70:90], dtype=np.float64)

    # --------------------------------------------------------
    # Remaining values are footprint metadata
    # --------------------------------------------------------

    footprint = values[90:96]

    return {
        "line_off": line_off,
        "samp_off": samp_off,

        "lat_off": lat_off,
        "lon_off": lon_off,
        "height_off": height_off,

        "line_scale": line_scale,
        "samp_scale": samp_scale,

        "lat_scale": lat_scale,
        "lon_scale": lon_scale,
        "height_scale": height_scale,

        "line_num": line_num,
        "line_den": line_den,

        "samp_num": samp_num,
        "samp_den": samp_den,

        "footprint": footprint,
    }


rpc = load_rpc(RPC_PATH)


# ============================================================
# RPC MONOMIALS
# Standard RPC 20-term ordering
# ============================================================

def rpc_terms(P, L, H):

    return np.stack([
        np.ones_like(P),

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
    ], axis=0)


def rational_rpc(P, L, H, numerator, denominator):

    terms = rpc_terms(P, L, H)

    num = np.sum(
        numerator[:, None] * terms.reshape(
            terms.shape[0],
            -1
        ),
        axis=0
    )

    den = np.sum(
        denominator[:, None] * terms.reshape(
            terms.shape[0],
            -1
        ),
        axis=0
    )

    den = np.where(
        np.abs(den) < 1e-12,
        np.nan,
        den
    )

    return num / den


def project_rpc(lat, lon, height):

    P = (
        lat - rpc["lat_off"]
    ) / rpc["lat_scale"]

    L = (
        lon - rpc["lon_off"]
    ) / rpc["lon_scale"]

    H = (
        height - rpc["height_off"]
    ) / rpc["height_scale"]
    
    line_n = rational_rpc(
        P,
        L,
        H,
        rpc["line_num"],
        rpc["line_den"]
    )

    samp_n = rational_rpc(
        P,
        L,
        H,
        rpc["samp_num"],
        rpc["samp_den"]
    )

    line = (
        line_n * rpc["line_scale"]
        + rpc["line_off"]
    )

    samp = (
        samp_n * rpc["samp_scale"]
        + rpc["samp_off"]
    )

    return line, samp


# ============================================================
# LOAD MASTER DSM
# ============================================================

print()
print("=" * 70)
print("ASTERRA AI — RPC → MASTER DSM MAPPING")
print("=" * 70)

print("Satellite image:")
print(IMAGE_PATH)

print()
print("RPC:")
print(RPC_PATH)

print()
print("Master DSM:")
print(MASTER_PATH)


with rasterio.open(IMAGE_PATH) as img:

    image = img.read(1)

    image_height = img.height
    image_width = img.width

    image_dtype = img.dtypes[0]

    print()
    print("IMAGE")
    print("-" * 70)
    print("Width :", image_width)
    print("Height:", image_height)
    print("Bands :", img.count)
    print("Dtype :", image_dtype)


with rasterio.open(MASTER_PATH) as master:

    dsm = master.read(1)

    master_transform = master.transform
    master_crs = master.crs
    master_width = master.width
    master_height = master.height
    master_nodata = master.nodata

    print()
    print("MASTER DSM")
    print("-" * 70)
    print("Width :", master_width)
    print("Height:", master_height)
    print("CRS   :", master_crs)
    print("Res   :", master.res)
    print("NoData:", master_nodata)


# ============================================================
# DETERMINE VALID DSM PIXELS
# ============================================================

if master_nodata is None:

    valid = np.isfinite(dsm)

else:

    valid = (
        np.isfinite(dsm)
        &
        (dsm != master_nodata)
    )


valid &= dsm > -1000


rows, cols = np.where(valid)


print()
print("VALID DSM PIXELS")
print("-" * 70)
print("Valid:", len(rows))


if len(rows) == 0:
    raise RuntimeError("No valid DSM pixels found.")


# ============================================================
# SAMPLE VALID DSM POINTS
#
# We first use a subset to understand where the RPC
# projects the DSM into the original satellite coordinate
# system.
# ============================================================

sample_count = min(10000, len(rows))

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(rows),
    size=sample_count,
    replace=False
)

sample_rows = rows[sample_indices]
sample_cols = cols[sample_indices]

# Pixel centers
xs, ys = rasterio.transform.xy(
    master_transform,
    sample_rows,
    sample_cols,
    offset="center"
)

xs = np.asarray(xs)
ys = np.asarray(ys)

heights = dsm[
    sample_rows,
    sample_cols
].astype(np.float64)


# ============================================================
# UTM → WGS84 LAT/LON
# ============================================================

lon, lat = transform(
    master_crs,
    "EPSG:4326",
    xs.tolist(),
    ys.tolist()
)

lon = np.asarray(lon)
lat = np.asarray(lat)


# ============================================================
# PROJECT DSM → ORIGINAL RPC IMAGE SPACE
# ============================================================

raw_line, raw_samp = project_rpc(
    lat,
    lon,
    heights
)


finite = (
    np.isfinite(raw_line)
    &
    np.isfinite(raw_samp)
)

raw_line = raw_line[finite]
raw_samp = raw_samp[finite]


print()
print("=" * 70)
print("RAW RPC PROJECTION")
print("=" * 70)

print(
    "LINE range:",
    float(np.min(raw_line)),
    "to",
    float(np.max(raw_line))
)

print(
    "SAMPLE range:",
    float(np.min(raw_samp)),
    "to",
    float(np.max(raw_samp))
)

print()
print("Satellite image local coordinates should be:")
print("LINE   : 0 →", image_height - 1)
print("SAMPLE : 0 →", image_width - 1)


# ============================================================
# ESTIMATE TILE OFFSET
#
# The RPC appears to use a larger/original image coordinate
# system. Therefore:
#
# local_line  = raw_line  - line_offset
# local_samp  = raw_samp  - samp_offset
#
# We estimate the translation that places the projected
# DSM footprint over the 2001×2001 image.
# ============================================================

line_center = np.median(raw_line)
samp_center = np.median(raw_samp)

line_offset_est = (
    line_center - (image_height - 1) / 2
)

samp_offset_est = (
    samp_center - (image_width - 1) / 2
)


print()
print("=" * 70)
print("ESTIMATED TILE OFFSET")
print("=" * 70)

print(
    "Estimated LINE offset :",
    float(line_offset_est)
)

print(
    "Estimated SAMPLE offset:",
    float(samp_offset_est)
)

print()
print("RPC LINE_OFF :", rpc["line_off"])
print("RPC SAMP_OFF :", rpc["samp_off"])


# ============================================================
# APPLY OFFSET
# ============================================================

local_line = raw_line - line_offset_est
local_samp = raw_samp - samp_offset_est


print()
print("=" * 70)
print("PROJECTED LOCAL IMAGE COORDINATES")
print("=" * 70)

print(
    "LINE range:",
    float(np.min(local_line)),
    "to",
    float(np.max(local_line))
)

print(
    "SAMPLE range:",
    float(np.min(local_samp)),
    "to",
    float(np.max(local_samp))
)


inside = (
    (local_line >= 0)
    &
    (local_line < image_height)
    &
    (local_samp >= 0)
    &
    (local_samp < image_width)
)


print()
print(
    "Points inside 2001×2001 image:",
    int(np.sum(inside)),
    "/",
    len(local_line)
)

print(
    "Coverage:",
    round(
        100 * np.sum(inside) / len(local_line),
        2
    ),
    "%"
)


# ============================================================
# IMPORTANT SAFETY CHECK
# ============================================================

if np.sum(inside) / len(local_line) < 0.50:

    raise RuntimeError(
        "\n"
        "RPC projection does not align with the 2001×2001 "
        "image after the estimated translation.\n\n"
        "STOPPING instead of producing an incorrectly "
        "registered dataset.\n"
    )


# ============================================================
# FULL ORTHORECTIFICATION
#
# Output grid = MASTER DSM grid.
#
# For every DSM pixel:
#
# DSM UTM coordinate + DSM height
#              ↓
#        WGS84 lat/lon
#              ↓
#          RPC model
#              ↓
#       satellite pixel
#              ↓
#       image sampling
# ============================================================

output = np.zeros(
    (master_height, master_width),
    dtype=np.uint8
)


# Process row chunks to keep memory reasonable.

CHUNK_ROWS = 64

with rasterio.open(IMAGE_PATH) as img:

    image = img.read(1)

    for r0 in range(0, master_height, CHUNK_ROWS):

        r1 = min(
            r0 + CHUNK_ROWS,
            master_height
        )

        print(
            f"\rProcessing rows "
            f"{r0}:{r1} / {master_height}",
            end=""
        )

        rr, cc = np.meshgrid(
            np.arange(r0, r1),
            np.arange(master_width),
            indexing="ij"
        )

        rr_flat = rr.ravel()
        cc_flat = cc.ravel()

        z = dsm[
            rr_flat,
            cc_flat
        ].astype(np.float64)

        valid_chunk = (
            np.isfinite(z)
            &
            (z != master_nodata)
            &
            (z > -1000)
        )

        if not np.any(valid_chunk):
            continue

        rr_valid = rr_flat[valid_chunk]
        cc_valid = cc_flat[valid_chunk]

        z_valid = z[valid_chunk]

        x, y = rasterio.transform.xy(
            master_transform,
            rr_valid,
            cc_valid,
            offset="center"
        )

        x = np.asarray(x)
        y = np.asarray(y)

        lon_c, lat_c = transform(
            master_crs,
            "EPSG:4326",
            x.tolist(),
            y.tolist()
        )

        lon_c = np.asarray(lon_c)
        lat_c = np.asarray(lat_c)

        line, samp = project_rpc(
            lat_c,
            lon_c,
            z_valid
        )

        line = line - line_offset_est
        samp = samp - samp_offset_est

        valid_proj = (
            np.isfinite(line)
            &
            np.isfinite(samp)
            &
            (line >= 0)
            &
            (line < image_height - 1)
            &
            (samp >= 0)
            &
            (samp < image_width - 1)
        )

        if not np.any(valid_proj):
            continue

        rr2 = rr_valid[valid_proj]
        cc2 = cc_valid[valid_proj]

        line2 = line[valid_proj]
        samp2 = samp[valid_proj]

        # ----------------------------------------------------
        # Bilinear interpolation
        # ----------------------------------------------------

        y0 = np.floor(line2).astype(np.int32)
        x0 = np.floor(samp2).astype(np.int32)

        y1 = y0 + 1
        x1 = x0 + 1

        dy = line2 - y0
        dx = samp2 - x0

        v00 = image[y0, x0].astype(np.float64)
        v01 = image[y0, x1].astype(np.float64)
        v10 = image[y1, x0].astype(np.float64)
        v11 = image[y1, x1].astype(np.float64)

        values = (
            (1 - dx) * (1 - dy) * v00
            +
            dx * (1 - dy) * v01
            +
            (1 - dx) * dy * v10
            +
            dx * dy * v11
        )

        output[
            rr2,
            cc2
        ] = np.clip(
            values,
            0,
            255
        ).astype(np.uint8)


print()
print()
print("=" * 70)
print("WRITING OUTPUT")
print("=" * 70)


# ============================================================
# WRITE GEOSPATIAL OUTPUT
# ============================================================

with rasterio.open(
    MASTER_PATH
) as master:

    profile = master.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        compress="deflate",
        predictor=2,
        nodata=0
    )

    with rasterio.open(
        OUTPUT_PATH,
        "w",
        **profile
    ) as dst:

        dst.write(
            output,
            1
        )

        dst.update_tags(
            SOURCE_IMAGE=str(IMAGE_PATH),
            SOURCE_RPC=str(RPC_PATH),
            MASTER_DSM=str(MASTER_PATH),
            RPC_TILE_LINE_OFFSET=str(line_offset_est),
            RPC_TILE_SAMPLE_OFFSET=str(samp_offset_est),
            MAPPING_METHOD="RPC forward projection + bilinear sampling"
        )


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print("OUTPUT:")
print(OUTPUT_PATH)

print()
print("Output size:")
print(
    output.shape[1],
    "x",
    output.shape[0]
)

print()
print("Output uses the MASTER DSM:")
print("CRS :", master_crs)
print("Grid:", master_transform)