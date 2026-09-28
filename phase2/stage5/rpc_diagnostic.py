from pathlib import Path
import rasterio
import numpy as np
from rasterio.rpc import RPC
from pyproj import Transformer

ROOT = Path(r"D:\Asterra AI")
IMG_DIR = ROOT / "datasets" / "SpaceNet_MVS"

img = next(IMG_DIR.rglob("*.tif"))
rpc_files = list(IMG_DIR.rglob("*.txt"))

print("=" * 80)
print("ASTERRA STAGE-5 RPC / MOSAIC DIAGNOSTIC")
print("=" * 80)

print("\nIMAGE:")
print(img)

with rasterio.open(img) as ds:
    print("size:", ds.width, "x", ds.height)
    print("crs:", ds.crs)
    print("bounds:", ds.bounds)
    print("tags:", ds.tags())

print("\nRPC CANDIDATES:")
for p in rpc_files[:5]:
    print(p)

print("\nELEVATION MOSAICS:")

mosaics = list(IMG_DIR.rglob("*MasterProvisional*.tif"))

for m in mosaics:
    try:
        with rasterio.open(m) as ds:
            print("\n", m.name)
            print(" size:", ds.width, "x", ds.height)
            print(" crs:", ds.crs)
            print(" bounds:", ds.bounds)
            print(" transform:", ds.transform)
            print(" nodata:", ds.nodata)

            cx = (ds.width - 1) / 2
            cy = (ds.height - 1) / 2

            x, y = ds.xy(cy, cx)

            print(" centre projected:", x, y)

            to_wgs84 = Transformer.from_crs(
                ds.crs,
                "EPSG:4326",
                always_xy=True
            )

            lon, lat = to_wgs84.transform(x, y)

            print(" centre WGS84:", lon, lat)

    except Exception as e:
        print("ERROR:", repr(e))

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

