from pathlib import Path
import rasterio
from rasterio.transform import RPCTransformer

rpc_file = Path(
    r".\datasets\SpaceNet_MVS\MasterProvisional1\rpc_MasterProvisional1_01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.txt"
)

nums = [
    float(x.strip())
    for x in rpc_file.read_text().replace("\n", "").split(",")
    if x.strip()
]

r = nums

rpc = {
    "LINE_OFF": r[0],
    "SAMP_OFF": r[1],
    "LAT_OFF": r[2],
    "LONG_OFF": r[3],
    "HEIGHT_OFF": r[4],
    "LINE_SCALE": r[5],
    "SAMP_SCALE": r[6],
    "LAT_SCALE": r[7],
    "LONG_SCALE": r[8],
    "HEIGHT_SCALE": r[9],
}

for i in range(20):
    rpc[f"LINE_NUM_COEFF_{i+1:02d}"] = r[10+i]

for i in range(20):
    rpc[f"LINE_DEN_COEFF_{i+1:02d}"] = r[30+i]

for i in range(20):
    rpc[f"SAMP_NUM_COEFF_{i+1:02d}"] = r[50+i]

for i in range(20):
    rpc[f"SAMP_DEN_COEFF_{i+1:02d}"] = r[70+i]

tr = RPCTransformer(rpc)

line_off = r[0]
samp_off = r[1]
line_scale = r[5]
samp_scale = r[6]

print("RPC IMAGE NORMALIZATION")
print("LINE_OFF   =", line_off)
print("SAMP_OFF   =", samp_off)
print("LINE_SCALE =", line_scale)
print("SAMP_SCALE =", samp_scale)

print()
print("TESTING COORDINATES AROUND RPC OFFSET")
print()

# IMPORTANT:
# Rasterio xy(row, col, z)
# Here row = line and col = sample.

tests = [
    ("offset", line_off, samp_off),
    ("offset-minus-half-scale",
        line_off - line_scale/2,
        samp_off - samp_scale/2),
    ("offset-plus-half-scale",
        line_off + line_scale/2,
        samp_off + samp_scale/2),
    ("offset-minus-quarter",
        line_off - line_scale/4,
        samp_off - samp_scale/4),
    ("offset-plus-quarter",
        line_off + line_scale/4,
        samp_off + samp_scale/4),
]

for name, row, col in tests:

    try:
        lon, lat = tr.xy(
            row,
            col,
            zs=31
        )

        print(
            f"{name:30s} "
            f"line={row:12.3f} "
            f"sample={col:12.3f} "
            f"-> lon={lon:.8f}, lat={lat:.8f}"
        )

    except Exception as e:

        print(
            f"{name:30s} "
            f"ERROR: {type(e).__name__}: {e}"
        )


print()
print("=" * 80)
print("NOW TESTING GEOGRAPHIC FOOTPRINT -> IMAGE COORDINATES")
print("=" * 80)

# Final 6 values from the file.
# We are treating ONLY the first four as geographic bounds for this test.
lon_min = r[90]
lat_min = r[91]
lon_max = r[92]
lat_max = r[93]

print()
print("FOOTPRINT VALUES")
print("lon_min =", lon_min)
print("lat_min =", lat_min)
print("lon_max =", lon_max)
print("lat_max =", lat_max)

print()

geo_tests = [
    ("SW", lon_min, lat_min),
    ("NW", lon_min, lat_max),
    ("SE", lon_max, lat_min),
    ("NE", lon_max, lat_max),
    ("CENTER", (lon_min+lon_max)/2, (lat_min+lat_max)/2),
]

for name, lon, lat in geo_tests:

    try:
        row, col = tr.rowcol(
            lon,
            lat,
            zs=31
        )

        print(
            f"{name:8s} "
            f"lon={lon:.8f} "
            f"lat={lat:.8f} "
            f"-> line={row:.3f}, sample={col:.3f}"
        )

    except Exception as e:

        print(
            f"{name:8s} "
            f"ERROR: {type(e).__name__}: {e}"
        )

tr.close()
