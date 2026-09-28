from pathlib import Path
import rasterio
from rasterio.transform import RPCTransformer

img = Path(r".\datasets\SpaceNet_MVS\MasterProvisional1\MasterProvisional1_01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.tif")
rpc_file = Path(r".\datasets\SpaceNet_MVS\MasterProvisional1\rpc_MasterProvisional1_01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.txt")

nums = [
    float(x.strip())
    for x in rpc_file.read_text().replace("\n", "").split(",")
    if x.strip()
]

print("RPC NUMBER COUNT =", len(nums))
print()

names = [
    "LINE_OFF","SAMP_OFF","LAT_OFF","LONG_OFF","HEIGHT_OFF",
    "LINE_SCALE","SAMP_SCALE","LAT_SCALE","LONG_SCALE","HEIGHT_SCALE"
]

print("FIRST 10 RPC VALUES")
for n, v in zip(names, nums[:10]):
    print(f"{n:12s} = {v}")

with rasterio.open(img) as ds:
    print()
    print("IMAGE SIZE")
    print("WIDTH  =", ds.width)
    print("HEIGHT =", ds.height)
    print("CRS    =", ds.crs)
    print("TRANSFORM =", ds.transform)

# Build GDAL/Rasterio RPC metadata
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

print()
print("RPC DICT CREATED")
print("RPCTransformer location: rasterio.transform")

try:
    tr = RPCTransformer(rpc)

    pixels = [
        (0,0),
        (1000,0),
        (2000,0),
        (0,1000),
        (1000,1000),
        (2000,1000),
        (0,2000),
        (1000,2000),
        (2000,2000),
    ]

    for h in [0, 30, 31, 50, 100]:

        print()
        print("=" * 70)
        print("HEIGHT =", h)
        print("=" * 70)

        for col, row in pixels:
            try:
                lon, lat = tr.xy(
                    row,
                    col,
                    zs=h
                )

                print(
                    f"pixel=({col:4d},{row:4d}) "
                    f"-> lon={lon:.8f}, lat={lat:.8f}"
                )

            except Exception as e:
                print(
                    f"pixel=({col:4d},{row:4d}) "
                    f"-> ERROR: {type(e).__name__}: {e}"
                )

    tr.close()

except Exception as e:
    print()
    print("RPC TRANSFORMER ERROR")
    print(type(e).__name__, e)
