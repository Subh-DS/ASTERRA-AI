import rasterio
import numpy as np
from pathlib import Path
from pyproj import Transformer

ROOT = Path(r".\datasets\SpaceNet_MVS")

DSM_PATH = ROOT / "MasterProvisional1.tif"

RPC_DIR = ROOT / "MasterProvisional1"

RPC_PATH = next(RPC_DIR.glob("rpc_*.txt"))

print("=" * 75)
print("ASTERRA AI — RPC GEOMETRIC VALIDATION")
print("=" * 75)

# ------------------------------------------------------------
# LOAD DSM
# ------------------------------------------------------------

with rasterio.open(DSM_PATH) as src:
    dsm = src.read(1).astype(np.float64)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata

print("\nDSM")
print("-" * 75)
print("Shape:", dsm.shape)
print("CRS:", crs)
print("Transform:", transform)
print("NoData:", nodata)

valid = np.isfinite(dsm)

if nodata is not None:
    valid &= dsm != nodata

rows, cols = np.where(valid)

print("Valid pixels:", len(rows))

# ------------------------------------------------------------
# SAMPLE DSM POINTS
# ------------------------------------------------------------

rng = np.random.default_rng(42)

N = min(10000, len(rows))

idx = rng.choice(len(rows), size=N, replace=False)

rows = rows[idx]
cols = cols[idx]

heights = dsm[rows, cols]

# ------------------------------------------------------------
# PIXEL -> PROJECTED COORDINATES
# ------------------------------------------------------------

xs, ys = rasterio.transform.xy(
    transform,
    rows,
    cols,
    offset="center"
)

xs = np.asarray(xs)
ys = np.asarray(ys)

print("\nMASTER COORDINATES")
print("-" * 75)
print("X range:", xs.min(), "to", xs.max())
print("Y range:", ys.min(), "to", ys.max())
print("Height range:", heights.min(), "to", heights.max())

# ------------------------------------------------------------
# UTM -> LAT/LON
# ------------------------------------------------------------

transformer = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True
)

lons, lats = transformer.transform(xs, ys)

print("\nGEOGRAPHIC COORDINATES")
print("-" * 75)
print("Longitude range:", lons.min(), "to", lons.max())
print("Latitude range :", lats.min(), "to", lats.max())

# ------------------------------------------------------------
# READ RPC
# ------------------------------------------------------------

vals = [
    float(x)
    for x in RPC_PATH.read_text().replace(",", " ").split()
]

print("\nRPC")
print("-" * 75)
print("RPC:", RPC_PATH.name)
print("Total values:", len(vals))

line_off = vals[0]
samp_off = vals[1]

lat_off = vals[2]
lon_off = vals[3]
height_off = vals[4]

line_scale = vals[5]
samp_scale = vals[6]

lat_scale = vals[7]
lon_scale = vals[8]
height_scale = vals[9]

line_num = np.asarray(vals[10:30])
line_den = np.asarray(vals[30:50])

samp_num = np.asarray(vals[50:70])
samp_den = np.asarray(vals[70:90])

# ------------------------------------------------------------
# RPC MONOMIALS
# ------------------------------------------------------------

def rpc_terms(L, P, H):

    return np.column_stack([
        np.ones_like(L),

        L,
        P,
        H,

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
        H * H * H
    ])

# ------------------------------------------------------------
# NORMALIZE GEOGRAPHIC COORDINATES
# ------------------------------------------------------------

L = (lats - lat_off) / lat_scale
P = (lons - lon_off) / lon_scale
H = (heights - height_off) / height_scale

terms = rpc_terms(L, P, H)

# ------------------------------------------------------------
# RPC FORWARD PROJECTION
# ------------------------------------------------------------

line_n = (
    terms @ line_num
) / (
    terms @ line_den
)

samp_n = (
    terms @ samp_num
) / (
    terms @ samp_den
)

raw_line = line_off + line_scale * line_n
raw_samp = samp_off + samp_scale * samp_n

# ------------------------------------------------------------
# RAW RPC RANGE
# ------------------------------------------------------------

print("\nRAW RPC OUTPUT")
print("-" * 75)

print(
    "LINE:",
    raw_line.min(),
    "to",
    raw_line.max()
)

print(
    "SAMPLE:",
    raw_samp.min(),
    "to",
    raw_samp.max()
)

print(
    "LINE center:",
    np.median(raw_line)
)

print(
    "SAMPLE center:",
    np.median(raw_samp)
)

# ------------------------------------------------------------
# OFFSET USED BY CURRENT MAPPING
# ------------------------------------------------------------

LINE_OFFSET = 23916.088170479416
SAMPLE_OFFSET = 12639.267496824512

local_line = raw_line - LINE_OFFSET
local_samp = raw_samp - SAMPLE_OFFSET

# ------------------------------------------------------------
# LOCAL IMAGE RANGE
# ------------------------------------------------------------

print("\nLOCAL SATELLITE COORDINATES")
print("-" * 75)

print(
    "LINE:",
    local_line.min(),
    "to",
    local_line.max()
)

print(
    "SAMPLE:",
    local_samp.min(),
    "to",
    local_samp.max()
)

inside = (
    (local_line >= 0) &
    (local_line <= 2000) &
    (local_samp >= 0) &
    (local_samp <= 2000)
)

print("\nINSIDE 2001 × 2001")
print("-" * 75)

print("Inside:", int(inside.sum()), "/", N)

print(
    "Coverage:",
    round(100 * inside.mean(), 3),
    "%"
)

# ------------------------------------------------------------
# DISTANCE FROM IMAGE BORDER
# ------------------------------------------------------------

border_distance = np.minimum.reduce([
    local_line,
    2000 - local_line,
    local_samp,
    2000 - local_samp
])

print("\nIMAGE BORDER DISTANCE")
print("-" * 75)

print(
    "Minimum:",
    border_distance.min()
)

print(
    "Median:",
    np.median(border_distance)
)

print(
    "Maximum:",
    border_distance.max()
)

# ------------------------------------------------------------
# HEIGHT SENSITIVITY
# ------------------------------------------------------------

print("\nHEIGHT SENSITIVITY TEST")
print("-" * 75)

for delta in [-50, 0, 50]:

    HH = (
        (heights + delta - height_off)
        / height_scale
    )

    TT = rpc_terms(L, P, HH)

    ln = (
        TT @ line_num
    ) / (
        TT @ line_den
    )

    sn = (
        TT @ samp_num
    ) / (
        TT @ samp_den
    )

    rl = line_off + line_scale * ln
    rs = samp_off + samp_scale * sn

    ll = rl - LINE_OFFSET
    ss = rs - SAMPLE_OFFSET

    print(
        f"Height {delta:+4d} m -> "
        f"LINE median={np.median(ll):.3f}, "
        f"SAMPLE median={np.median(ss):.3f}"
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

out = ROOT / "MasterProvisional1_rpc_geometry_metrics.txt"

with open(out, "w", encoding="utf-8") as f:

    f.write("ASTERRA AI RPC GEOMETRIC VALIDATION\n")
    f.write("=" * 60 + "\n")

    f.write(f"DSM: {DSM_PATH}\n")
    f.write(f"RPC: {RPC_PATH}\n")

    f.write(f"Samples: {N}\n")

    f.write(
        f"Raw line range: "
        f"{raw_line.min()} to {raw_line.max()}\n"
    )

    f.write(
        f"Raw sample range: "
        f"{raw_samp.min()} to {raw_samp.max()}\n"
    )

    f.write(
        f"Local line range: "
        f"{local_line.min()} to {local_line.max()}\n"
    )

    f.write(
        f"Local sample range: "
        f"{local_samp.min()} to {local_samp.max()}\n"
    )

    f.write(
        f"Inside image: "
        f"{inside.sum()}/{N}\n"
    )

    f.write(
        f"Coverage: "
        f"{100 * inside.mean():.4f}%\n"
    )

print("\nMetrics written to:")
print(out)

print("\nDONE")
