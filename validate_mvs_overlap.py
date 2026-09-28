from pathlib import Path
import rasterio
from rasterio.warp import transform
from rasterio.transform import RPC, RPCTransformer


BASE = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional1")
IMG_DIR = BASE / "MasterProvisional1"
GT_PATH = BASE / "MasterProvisional1_GT.tif"


# ------------------------------------------------------------
# Load GT
# ------------------------------------------------------------
with rasterio.open(GT_PATH) as gt:
    bounds = gt.bounds
    crs = gt.crs

    # Dense grid over entire GT mosaic.
    # 21 x 21 = 441 geographic test points.
    nx = 21
    ny = 21

    xs = [
        bounds.left + (bounds.right - bounds.left) * i / (nx - 1)
        for i in range(nx)
    ]

    ys = [
        bounds.bottom + (bounds.top - bounds.bottom) * j / (ny - 1)
        for j in range(ny)
    ]

    utm_x = []
    utm_y = []

    for y in ys:
        for x in xs:
            utm_x.append(x)
            utm_y.append(y)

    lon, lat = transform(
        crs,
        "EPSG:4326",
        utm_x,
        utm_y,
    )


# ------------------------------------------------------------
# Find image/RPC pairs
# ------------------------------------------------------------
images = sorted(IMG_DIR.glob("MasterProvisional1_*.tif"))

print("=" * 90)
print("ASTERRA AI — SPACENET MVS GT/RPC OVERLAP VALIDATOR")
print("=" * 90)
print(f"GT:       {GT_PATH.name}")
print(f"GT CRS:   {crs}")
print(f"GT SIZE:  {nx} x {ny} test grid = {len(lon)} points")
print(f"IMAGES:   {len(images)}")
print()


results = []


# ------------------------------------------------------------
# Test each image
# ------------------------------------------------------------
for idx, tif in enumerate(images, 1):

    rpc_candidates = list(
        IMG_DIR.glob("rpc_" + tif.stem + "*.txt")
    )

    if not rpc_candidates:
        print(f"[{idx:02d}/{len(images):02d}] NO RPC  {tif.name}")
        continue

    rpc_path = rpc_candidates[0]

    try:
        values = [
            float(x)
            for x in rpc_path.read_text().strip().split(",")
        ]

        if len(values) < 96:
            raise RuntimeError(
                f"RPC has only {len(values)} values"
            )

        # ----------------------------------------------------
        # Correct SpaceNet RPC field mapping
        # ----------------------------------------------------
        rpc = RPC(
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

        transformer = RPCTransformer(rpc)

        crop_x = values[94]
        crop_y = values[95]

        inside = 0
        total = len(lon)

        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")

        inside_points = []

        # ----------------------------------------------------
        # Project every GT grid point
        # ----------------------------------------------------
        for p in range(total):

            row, col = transformer.rowcol(
                lon[p],
                lat[p],
                31.0,
            )

            local_x = float(col) - crop_x
            local_y = float(row) - crop_y

            min_x = min(min_x, local_x)
            max_x = max(max_x, local_x)

            min_y = min(min_y, local_y)
            max_y = max(max_y, local_y)

            if (
                0.0 <= local_x <= 2000.0
                and
                0.0 <= local_y <= 2000.0
            ):
                inside += 1
                inside_points.append(
                    (local_x, local_y, lon[p], lat[p])
                )

        percentage = 100.0 * inside / total

        if inside > 0:
            status = "OVERLAP"
        else:
            status = "NO_OVERLAP"

        results.append({
            "name": tif.name,
            "rpc": rpc_path.name,
            "inside": inside,
            "total": total,
            "percentage": percentage,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "crop_x": crop_x,
            "crop_y": crop_y,
        })

        print(
            f"[{idx:02d}/{len(images):02d}] "
            f"{status:10s} "
            f"inside={inside:3d}/{total} "
            f"({percentage:6.2f}%) "
            f"local_x=[{min_x:9.1f},{max_x:9.1f}] "
            f"local_y=[{min_y:9.1f},{max_y:9.1f}]"
        )

    except Exception as e:

        print(
            f"[{idx:02d}/{len(images):02d}] ERROR     "
            f"{tif.name}"
        )
        print(f"       {type(e).__name__}: {e}")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

overlap = [
    r for r in results
    if r["inside"] > 0
]

no_overlap = [
    r for r in results
    if r["inside"] == 0
]

print(f"RPC/images tested : {len(results)}")
print(f"OVERLAP           : {len(overlap)}")
print(f"NO OVERLAP        : {len(no_overlap)}")
print()


# ------------------------------------------------------------
# Ranked overlap results
# ------------------------------------------------------------
print("RANKED BY GT OVERLAP")
print("-" * 90)

for r in sorted(
    overlap,
    key=lambda x: x["percentage"],
    reverse=True,
):

    print(
        f"{r['percentage']:6.2f}%  "
        f"{r['inside']:3d}/{r['total']}  "
        f"x=[{r['min_x']:.1f},{r['max_x']:.1f}]  "
        f"y=[{r['min_y']:.1f},{r['max_y']:.1f}]  "
        f"{r['name']}"
    )


print()
print("=" * 90)
print("INTERPRETATION")
print("=" * 90)

if overlap:
    print(
        "PASS: RPC/crop registration is demonstrably working "
        "for at least part of the GT mosaic."
    )
    print(
        "The next Stage-5 task is to construct training pairs "
        "from these actual overlap regions."
    )
else:
    print(
        "WARNING: No GT grid points landed inside any crop."
    )
    print(
        "Do NOT modify the RPC files yet."
    )