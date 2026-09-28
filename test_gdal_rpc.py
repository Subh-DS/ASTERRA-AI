from pathlib import Path

from rasterio.rpc import RPC
from rasterio.transform import RPCTransformer


ROOT = Path(
    r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional1\MasterProvisional1"
)

rpc_file = next(ROOT.glob("rpc_*.txt"))

values = [
    float(x)
    for x in rpc_file.read_text().strip().split(",")
]

print("RPC:", rpc_file.name)
print("VALUES:", len(values))

if len(values) != 96:
    raise RuntimeError(f"Expected 96 RPC values, got {len(values)}")


# ------------------------------------------------------------
# SpaceNet RPC layout
# ------------------------------------------------------------

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

line_num = values[10:30]
line_den = values[30:50]

samp_num = values[50:70]
samp_den = values[70:90]

min_lon = values[90]
min_lat = values[91]
max_lon = values[92]
max_lat = values[93]

crop_x = values[94]
crop_y = values[95]


rpc = RPC(
    err_bias=0.0,
    err_rand=0.0,

    line_off=line_off,
    samp_off=samp_off,
    lat_off=lat_off,
    long_off=lon_off,
    height_off=height_off,

    line_scale=line_scale,
    samp_scale=samp_scale,
    lat_scale=lat_scale,
    long_scale=lon_scale,
    height_scale=height_scale,

    line_num_coeff=list(line_num),
    line_den_coeff=list(line_den),
    samp_num_coeff=list(samp_num),
    samp_den_coeff=list(samp_den),
)


print()
print("NORMALIZATION")
print("LINE_OFF :", line_off)
print("SAMP_OFF :", samp_off)
print("LAT_OFF  :", lat_off)
print("LON_OFF  :", lon_off)

print()
print("BBOX")
print(min_lon, min_lat, max_lon, max_lat)

print()
print("CROP ORIGIN")
print("X:", crop_x)
print("Y:", crop_y)


# ------------------------------------------------------------
# Geographic center
# ------------------------------------------------------------

lon = (min_lon + max_lon) / 2.0
lat = (min_lat + max_lat) / 2.0
height = height_off

print()
print("TEST GEO POINT")
print("lon =", lon)
print("lat =", lat)
print("height =", height)


# ------------------------------------------------------------
# GDAL-backed RPC transform
# ------------------------------------------------------------

with RPCTransformer(rpc, rpc_height=height) as transformer:

    # xy() converts geographic coordinates -> image coordinates
    rows, cols = transformer.rowcol(
        [lon],
        [lat],
        zs=[height],
    )

    row = float(rows[0])
    col = float(cols[0])

print()
print("GDAL RPC RESULT")
print("original column/sample =", col)
print("original row/line      =", row)


# ------------------------------------------------------------
# Crop transformation
# ------------------------------------------------------------

local_x = col - crop_x
local_y = row - crop_y

print()
print("LOCAL COORDINATES")
print("x =", local_x)
print("y =", local_y)

print()
print("IMAGE SIZE")
print("2001 x 2001")

inside = (
    0 <= local_x <= 2000
    and
    0 <= local_y <= 2000
)

print()
print("REGISTRATION:", "PASS" if inside else "FAIL")