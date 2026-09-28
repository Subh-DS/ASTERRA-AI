from pathlib import Path
import json
import math

import rasterio
from rasterio.warp import transform
from rasterio.transform import RPC, RPCTransformer


BASE = Path(
    r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional1"
)

GT_PATH = BASE / "MasterProvisional1_GT.tif"

INDEX_PATH = Path(
    r"D:\Asterra AI\datasets\SpaceNet_MVS\mp1_geometry_index.json"
)

IMAGE_SIZE = 2001
CROP_SIZE = 512
HEIGHT = 31.0

# Margin inside the GT bounds.
MARGIN = 1.0


def make_rpc(v):

    return RPC(
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


index = json.loads(
    INDEX_PATH.read_text(encoding="utf-8")
)

records = index["images"]


with rasterio.open(GT_PATH) as gt:

    gt_bounds = gt.bounds
    gt_crs = gt.crs


print("=" * 100)
print("ASTERRA AI — TRUE 512x512 RPC → GT GEOMETRY VALIDATOR")
print("=" * 100)
print(f"Scenes     : {len(records)}")
print(f"Crop       : {CROP_SIZE} x {CROP_SIZE}")
print(f"GT CRS     : {gt_crs}")
print()


passed = 0
failed = 0


for n, record in enumerate(records, 1):

    tif = Path(record["image"])
    rpc_path = Path(record["rpc"])

    safe = record["safe_512"]

    if not safe["possible"]:

        print(
            f"[{n:02d}/{len(records):02d}] "
            f"FAIL — no safe region"
        )

        failed += 1
        continue


    # --------------------------------------------------------
    # Read RPC
    # --------------------------------------------------------

    values = [
        float(x)
        for x in rpc_path.read_text().strip().split(",")
    ]

    rpc = make_rpc(values)

    transformer = RPCTransformer(rpc)


    # --------------------------------------------------------
    # Choose center of the known overlap
    # --------------------------------------------------------

    local_x_min = safe["x_min"]
    local_x_max = safe["x_max"]

    local_y_min = safe["y_min"]
    local_y_max = safe["y_max"]


    crop_x = (
        local_x_min +
        local_x_max
    ) / 2.0

    crop_y = (
        local_y_min +
        local_y_max
    ) / 2.0


    # Keep entire crop inside image.
    crop_x = max(
        0.0,
        min(
            crop_x,
            IMAGE_SIZE - CROP_SIZE
        )
    )

    crop_y = max(
        0.0,
        min(
            crop_y,
            IMAGE_SIZE - CROP_SIZE
        )
    )


    # --------------------------------------------------------
    # Four crop corners
    # --------------------------------------------------------

    corners = [
        ("TL", crop_x, crop_y),
        ("TR", crop_x + CROP_SIZE - 1, crop_y),
        ("BL", crop_x, crop_y + CROP_SIZE - 1),
        ("BR", crop_x + CROP_SIZE - 1,
              crop_y + CROP_SIZE - 1),
    ]


    geographic = []


    # --------------------------------------------------------
    # IMAGE → GEO using RPC inverse
    # --------------------------------------------------------

    for name, x, y in corners:

        try:

            lon, lat = transformer.xy(
                y,
                x,
                HEIGHT,
            )

            geographic.append(
                (name, float(lon), float(lat))
            )

        except Exception as e:

            print(
                f"[{n:02d}/{len(records):02d}] "
                f"FAIL RPC inverse: {e}"
            )

            failed += 1
            geographic = []
            break


    if len(geographic) != 4:
        continue


    # --------------------------------------------------------
    # GEO → UTM
    # --------------------------------------------------------

    lons = [p[1] for p in geographic]
    lats = [p[2] for p in geographic]

    utm_x, utm_y = transform(
        "EPSG:4326",
        gt_crs,
        lons,
        lats,
    )


    # --------------------------------------------------------
    # Check every corner against GT
    # --------------------------------------------------------

    inside = []

    for i, p in enumerate(geographic):

        name = p[0]

        x = float(utm_x[i])
        y = float(utm_y[i])

        ok = (
            gt_bounds.left + MARGIN <= x <=
            gt_bounds.right - MARGIN
            and
            gt_bounds.bottom + MARGIN <= y <=
            gt_bounds.top - MARGIN
        )

        inside.append(ok)


    # --------------------------------------------------------
    # Also test crop center
    # --------------------------------------------------------

    center_row = crop_y + (
        CROP_SIZE - 1
    ) / 2.0

    center_col = crop_x + (
        CROP_SIZE - 1
    ) / 2.0

    center_lon, center_lat = transformer.xy(
        center_row,
        center_col,
        HEIGHT,
    )

    center_utm_x, center_utm_y = transform(
        "EPSG:4326",
        gt_crs,
        [center_lon],
        [center_lat],
    )

    center_ok = (
        gt_bounds.left + MARGIN <= center_utm_x[0] <=
        gt_bounds.right - MARGIN
        and
        gt_bounds.bottom + MARGIN <= center_utm_y[0] <=
        gt_bounds.top - MARGIN
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    all_ok = all(inside) and center_ok

    if all_ok:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL"
        failed += 1


    print(
        f"[{n:02d}/{len(records):02d}] "
        f"{status:4s} "
        f"crop=({crop_x:.1f},{crop_y:.1f}) "
        f"corners={sum(inside)}/4 "
        f"center={center_ok} "
        f"{tif.name}"
    )


print()
print("=" * 100)
print("FINAL RESULT")
print("=" * 100)

print(f"PASS : {passed}")
print(f"FAIL : {failed}")
print(f"TOTAL: {passed + failed}")

if failed == 0:

    print()
    print(
        "ALL 50 CENTERED 512x512 CROPS ARE "
        "GEOMETRICALLY INSIDE THE GT MOSAIC."
    )

    print(
        "RPC geometry is ready for Stage-5 sampling."
    )

else:

    print()
    print(
        "Some centered crops extend outside the GT."
    )

    print(
        "Those scenes require a more conservative "
        "sampling position."
    )