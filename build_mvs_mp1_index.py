from pathlib import Path
import json
import math

import rasterio
from rasterio.warp import transform
from rasterio.transform import RPC, RPCTransformer


BASE = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional1")
IMG_DIR = BASE / "MasterProvisional1"
GT_PATH = BASE / "MasterProvisional1_GT.tif"

OUT = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS\mp1_geometry_index.json")

GRID = 101
SAFE_SIZE = 512
IMAGE_SIZE = 2001
HEIGHT = 31.0


def make_rpc(values):
    return RPC(
        height_off=values[4],
        height_scale=values[9],
        lat_off=values[2],
        lat_scale=values[7],
        line_den_coeff=values[30:50],
        line_num_coeff=values[10:30],
        line_off=values[0],
        line_scale=values[5],
        long_off=values[3],
        long_scale=values[8],
        samp_den_coeff=values[70:90],
        samp_num_coeff=values[50:70],
        samp_off=values[1],
        samp_scale=values[6],
    )


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None

    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return values[lo]

    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


# ------------------------------------------------------------
# GT grid
# ------------------------------------------------------------
with rasterio.open(GT_PATH) as gt:

    bounds = gt.bounds
    gt_crs = gt.crs

    gt_x = []
    gt_y = []

    for j in range(GRID):

        y = bounds.top - (
            (bounds.top - bounds.bottom) * j / (GRID - 1)
        )

        for i in range(GRID):

            x = bounds.left + (
                (bounds.right - bounds.left) * i / (GRID - 1)
            )

            gt_x.append(x)
            gt_y.append(y)

lon, lat = transform(
    gt_crs,
    "EPSG:4326",
    gt_x,
    gt_y,
)


# ------------------------------------------------------------
# Process every image
# ------------------------------------------------------------
images = sorted(IMG_DIR.glob("MasterProvisional1_*.tif"))

records = []

print("=" * 100)
print("ASTERRA AI — MP1 GEOMETRY INDEX BUILDER")
print("=" * 100)
print(f"GT       : {GT_PATH}")
print(f"GRID     : {GRID} x {GRID}")
print(f"IMAGES   : {len(images)}")
print(f"OUTPUT   : {OUT}")
print()


for number, tif in enumerate(images, 1):

    rpc_files = list(
        IMG_DIR.glob("rpc_" + tif.stem + "*.txt")
    )

    if not rpc_files:
        print(f"[{number:02d}] NO RPC: {tif.name}")
        continue

    rpc_path = rpc_files[0]

    values = [
        float(x)
        for x in rpc_path.read_text().strip().split(",")
    ]

    if len(values) < 96:
        print(
            f"[{number:02d}] BAD RPC ({len(values)} values): "
            f"{rpc_path.name}"
        )
        continue

    rpc = make_rpc(values)
    transformer = RPCTransformer(rpc)

    crop_x = values[94]
    crop_y = values[95]

    valid_points = []

    # --------------------------------------------------------
    # Project GT grid → local image coordinates
    # --------------------------------------------------------
    for p in range(len(lon)):

        try:
            row, col = transformer.rowcol(
                lon[p],
                lat[p],
                HEIGHT,
            )

            x = float(col) - crop_x
            y = float(row) - crop_y

            if (
                math.isfinite(x)
                and math.isfinite(y)
            ):
                valid_points.append(
                    {
                        "gt_x": gt_x[p],
                        "gt_y": gt_y[p],
                        "lon": lon[p],
                        "lat": lat[p],
                        "x": x,
                        "y": y,
                    }
                )

        except Exception:
            pass

    inside = [
        p for p in valid_points
        if (
            0 <= p["x"] <= IMAGE_SIZE - 1
            and
            0 <= p["y"] <= IMAGE_SIZE - 1
        )
    ]

    # --------------------------------------------------------
    # Actual overlap bounds
    # --------------------------------------------------------
    if inside:

        xs = [p["x"] for p in inside]
        ys = [p["y"] for p in inside]

        gt_inside_x = [p["gt_x"] for p in inside]
        gt_inside_y = [p["gt_y"] for p in inside]

        local_min_x = min(xs)
        local_max_x = max(xs)
        local_min_y = min(ys)
        local_max_y = max(ys)

        gt_min_x = min(gt_inside_x)
        gt_max_x = max(gt_inside_x)
        gt_min_y = min(gt_inside_y)
        gt_max_y = max(gt_inside_y)

        # ----------------------------------------------------
        # Safe 512x512 image sampling rectangle
        # ----------------------------------------------------
        safe_min_x = local_min_x
        safe_max_x = local_max_x
        safe_min_y = local_min_y
        safe_max_y = local_max_y

        safe_width = safe_max_x - safe_min_x
        safe_height = safe_max_y - safe_min_y

        if (
            safe_width >= SAFE_SIZE
            and
            safe_height >= SAFE_SIZE
        ):
            safe_crop_possible = True

            safe_x0 = math.ceil(safe_min_x)
            safe_y0 = math.ceil(safe_min_y)

            safe_x1 = math.floor(
                safe_max_x - SAFE_SIZE
            )

            safe_y1 = math.floor(
                safe_max_y - SAFE_SIZE
            )

        else:
            safe_crop_possible = False
            safe_x0 = None
            safe_y0 = None
            safe_x1 = None
            safe_y1 = None

        record = {
            "image": str(tif),
            "rpc": str(rpc_path),

            "image_width": IMAGE_SIZE,
            "image_height": IMAGE_SIZE,

            "crop_origin": {
                "x": crop_x,
                "y": crop_y,
            },

            "rpc_bbox": {
                "min_long": values[90],
                "min_lat": values[91],
                "max_long": values[92],
                "max_lat": values[93],
            },

            "gt_crs": str(gt_crs),

            "overlap": {
                "grid_points_inside": len(inside),
                "grid_points_total": len(valid_points),
                "percentage": (
                    100.0 * len(inside) / len(valid_points)
                ),

                "local_x": {
                    "min": local_min_x,
                    "max": local_max_x,
                },

                "local_y": {
                    "min": local_min_y,
                    "max": local_max_y,
                },

                "gt_x": {
                    "min": gt_min_x,
                    "max": gt_max_x,
                },

                "gt_y": {
                    "min": gt_min_y,
                    "max": gt_max_y,
                },
            },

            "safe_512": {
                "possible": safe_crop_possible,

                "x_min": safe_x0,
                "x_max": safe_x1,

                "y_min": safe_y0,
                "y_max": safe_y1,

                "width": (
                    safe_x1 - safe_x0 + 512
                    if safe_crop_possible
                    else 0
                ),

                "height": (
                    safe_y1 - safe_y0 + 512
                    if safe_crop_possible
                    else 0
                ),
            },
        }

        records.append(record)

        print(
            f"[{number:02d}/{len(images):02d}] "
            f"OVERLAP={record['overlap']['percentage']:6.2f}% "
            f"LOCAL="
            f"[{local_min_x:8.1f},{local_max_x:8.1f}] "
            f"x "
            f"[{local_min_y:8.1f},{local_max_y:8.1f}] "
            f"SAFE512={safe_crop_possible}"
        )

    else:

        print(
            f"[{number:02d}/{len(images):02d}] "
            f"NO OVERLAP"
        )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
output = {
    "dataset": "SpaceNet MVS",
    "master_provisional": "MasterProvisional1",

    "gt": {
        "path": str(GT_PATH),
        "crs": str(gt_crs),
        "grid": GRID,
    },

    "image_size": IMAGE_SIZE,
    "safe_crop_size": SAFE_SIZE,
    "rpc_height": HEIGHT,

    "images": records,
}

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUT.write_text(
    json.dumps(output, indent=2),
    encoding="utf-8",
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
safe = [
    r for r in records
    if r["safe_512"]["possible"]
]

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"RPC/image records : {len(records)}")
print(f"Safe 512 crops    : {len(safe)}")
print(f"Output            : {OUT}")

if safe:

    print()
    print("SAFE 512x512 SCENES")
    print("-" * 100)

    for r in sorted(
        safe,
        key=lambda x: x["overlap"]["percentage"],
        reverse=True,
    ):

        print(
            f"{r['overlap']['percentage']:6.2f}%  "
            f"x={r['safe_512']['x_min']}.."
            f"{r['safe_512']['x_max']}  "
            f"y={r['safe_512']['y_min']}.."
            f"{r['safe_512']['y_max']}  "
            f"{Path(r['image']).name}"
        )

print()
print("INDEX COMPLETE")