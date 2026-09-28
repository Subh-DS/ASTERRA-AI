import os
import re
import numpy as np
import rasterio
from rasterio.rpc import RPC
from rasterio.transform import RPCTransformer
from pyproj import Transformer as PyProjTransformer


ROOT = r"D:\Asterra AI\datasets\SpaceNet_MVS"

RGB_DIR = os.path.join(
    ROOT,
    "official",
    "MasterProvisional1",
    "MasterProvisional1"
)

GT_PATH = os.path.join(
    ROOT,
    "official",
    "MasterProvisional1",
    "MasterProvisional1_GT.tif"
)


# First MP1 scene
rgb_path = sorted(
    [
        os.path.join(RGB_DIR, x)
        for x in os.listdir(RGB_DIR)
        if x.lower().endswith(".tif")
    ]
)[0]

rpc_path = os.path.join(
    RGB_DIR,
    "rpc_" + os.path.basename(rgb_path)[:-4] + ".txt"
)


print("=" * 72)
print(" ASTERRA AI — SINGLE MP1 RPC FORENSIC TEST")
print("=" * 72)

print()
print("RGB:")
print(rgb_path)

print()
print("RPC:")
print(rpc_path)

print()
print("GT:")
print(GT_PATH)


# ============================================================
# PARSE RPC
# ============================================================

with open(
    rpc_path,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:
    text = f.read()

nums = re.findall(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",
    text
)

vals = np.asarray(
    [float(x) for x in nums],
    dtype=np.float64
)

print()
print("RPC numeric values:", len(vals))

v = vals[:90]

rpc = RPC(
    height_off=v[4],
    height_scale=v[9],

    lat_off=v[2],
    lat_scale=v[7],

    line_den_coeff=v[30:50],
    line_num_coeff=v[10:30],

    line_off=v[0],
    line_scale=v[5],

    long_off=v[3],
    long_scale=v[8],

    samp_den_coeff=v[70:90],
    samp_num_coeff=v[50:70],

    samp_off=v[1],
    samp_scale=v[6],
)


print()
print("RPC normalization:")
print("LINE_OFF   =", rpc.line_off)
print("SAMP_OFF   =", rpc.samp_off)
print("LAT_OFF    =", rpc.lat_off)
print("LONG_OFF   =", rpc.long_off)
print("HEIGHT_OFF =", rpc.height_off)

print()
print("LINE_SCALE =", rpc.line_scale)
print("SAMP_SCALE =", rpc.samp_scale)
print("LAT_SCALE  =", rpc.lat_scale)
print("LONG_SCALE =", rpc.long_scale)
print("HEIGHT_SCALE =", rpc.height_scale)


# ============================================================
# RPC TAIL
# ============================================================

print()
print("RPC values 90+:")
print(vals[90:])


# ============================================================
# IMAGE
# ============================================================

with rasterio.open(rgb_path) as ds:

    print()
    print("IMAGE:")
    print("width =", ds.width)
    print("height =", ds.height)
    print("crs =", ds.crs)
    print("transform =", ds.transform)


# ============================================================
# GT
# ============================================================

with rasterio.open(GT_PATH) as gt:

    print()
    print("GT:")
    print("width =", gt.width)
    print("height =", gt.height)
    print("crs =", gt.crs)
    print("transform =", gt.transform)
    print("bounds =", gt.bounds)
    print("nodata =", gt.nodata)

    arr = gt.read(1)

    valid = np.isfinite(arr)

    if gt.nodata is not None:
        valid &= arr != gt.nodata

    valid &= arr > -1000
    valid &= arr < 10000

    rows, cols = np.where(valid)

    # Select 5 representative points:
    indices = np.linspace(
        0,
        len(rows) - 1,
        5,
        dtype=int
    )

    gt_rows = rows[indices]
    gt_cols = cols[indices]

    elevations = arr[
        gt_rows,
        gt_cols
    ].astype(np.float64)

    xs, ys = rasterio.transform.xy(
        gt.transform,
        gt_rows,
        gt_cols,
        offset="center"
    )

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    print()
    print("Representative GT points:")

    for i in range(len(gt_rows)):

        print(
            f"GT[{i}] "
            f"row={gt_rows[i]} "
            f"col={gt_cols[i]} "
            f"X={xs[i]:.3f} "
            f"Y={ys[i]:.3f} "
            f"Z={elevations[i]:.3f}"
        )

    # ========================================================
    # UTM -> WGS84
    # ========================================================

    transformer = PyProjTransformer.from_crs(
        gt.crs,
        "EPSG:4326",
        always_xy=True
    )

    lon, lat = transformer.transform(
        xs,
        ys
    )

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    print()
    print("WGS84 points:")

    for i in range(len(lon)):

        print(
            f"GT[{i}] "
            f"lon={lon[i]:.8f} "
            f"lat={lat[i]:.8f} "
            f"z={elevations[i]:.3f}"
        )

    # ========================================================
    # RPC FORWARD
    # ========================================================

    rpc_transformer = RPCTransformer(
        rpc
    )

    rows_img, cols_img = rpc_transformer.rowcol(
        lon,
        lat,
        zs=elevations
    )

    rows_img = np.asarray(rows_img)
    cols_img = np.asarray(cols_img)

    print()
    print("RPC FORWARD RESULT:")
    # ========================================================
# SPACENET CROP OFFSET
# ========================================================

if len(vals) >= 96:

    # SpaceNet tail:
    # 90:94 = geographic bbox
    # 94    = sample/column crop offset
    # 95    = line/row crop offset

    sample_offset = vals[94]
    line_offset = vals[95]

    local_rows = rows_img - line_offset
    local_cols = cols_img - sample_offset

    print()
    print("SPACENET CROPPED COORDINATES:")
    print(
        "sample/column offset =",
        sample_offset
    )
    print(
        "line/row offset      =",
        line_offset
    )

    print()

    for i in range(len(local_rows)):

        inside = (
            0 <= local_rows[i] < 2001
            and
            0 <= local_cols[i] < 2001
        )

        print(
            f"GT[{i}] "
            f"local row={local_rows[i]:.3f} "
            f"local col={local_cols[i]:.3f} "
            f"INSIDE={inside}"
        )
    print()

    for i in range(len(rows_img)):

        print(
            f"GT[{i}] "
            f"RGB row={rows_img[i]:.3f} "
            f"RGB col={cols_img[i]:.3f}"
        )

    # ========================================================
    # CROP OFFSET
    # ========================================================

    if len(vals) > 90:

        print()
        print("Possible SpaceNet crop metadata:")
        print(vals[90:])

    print()
    print("=" * 72)
    print("END")
    print("=" * 72)