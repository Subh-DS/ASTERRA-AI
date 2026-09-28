from pathlib import Path
import re
import rasterio
from rasterio.transform import RPCTransformer
from pyproj import Transformer


# =============================================================================
# ASTERRA AI — STAGE 5 RPC GEOMETRY PROBE
# Purpose:
#   Correctly identify:
#       1. Scene TIFFs
#       2. RPC TXT files
#       3. MasterProvisional elevation mosaics
#   Then inspect one real scene + its RPC + all elevation mosaics.
# =============================================================================

ROOT = Path(r"D:\Asterra AI")
DATA = ROOT / "datasets" / "SpaceNet_MVS"


print("=" * 100)
print("ASTERRA AI — STAGE 5 RPC GEOMETRY PROBE")
print("=" * 100)

print("\nDATASET ROOT:")
print(DATA)
print("exists:", DATA.exists())

if not DATA.exists():
    raise RuntimeError(f"Dataset directory does not exist:\n{DATA}")


# =============================================================================
# 1. DISCOVER ALL TIFF FILES
# =============================================================================

all_tifs = sorted(DATA.rglob("*.tif"))

print("\n" + "-" * 100)
print("TIFF DISCOVERY")
print("-" * 100)

print("Total TIFF files:", len(all_tifs))

if not all_tifs:
    raise RuntimeError("No TIFF files found.")


# =============================================================================
# 2. INSPECT TIFF METADATA TO CLASSIFY FILES
#
# IMPORTANT:
# Filename alone is NOT trusted.
#
# We identify elevation mosaics using:
#   - very large raster dimensions
#   - CRS
#   - single band
#   - float32
#   - known MasterProvisional pattern
#
# Scene images are identified separately.
# =============================================================================

print("\nInspecting TIFF metadata...")

mosaics = []
scene_candidates = []

for tif in all_tifs:

    try:
        with rasterio.open(tif) as ds:

            width = ds.width
            height = ds.height
            count = ds.count
            dtype = ds.dtypes[0] if ds.count else None
            crs = ds.crs

            # -------------------------------------------------------------
            # Strong elevation-mosaic indicators
            # -------------------------------------------------------------
            is_mosaic = (
                "MasterProvisional" in tif.name
                and count == 1
                and dtype == "float32"
                and width >= 2000
                and height >= 1500
            )

            if is_mosaic:
                mosaics.append(tif)
            else:
                scene_candidates.append(tif)

    except Exception as e:
        print("[TIFF ERROR]", tif)
        print(" ", repr(e))


print("\n" + "-" * 100)
print("CLASSIFICATION")
print("-" * 100)

print("Elevation mosaics :", len(mosaics))
print("Scene candidates  :", len(scene_candidates))


# =============================================================================
# 3. PRINT ELEVATION MOSAICS
# =============================================================================

print("\nELEVATION MOSAICS:")

for p in mosaics:
    print(" ", p)


# =============================================================================
# 4. PRINT SCENE CANDIDATES
# =============================================================================

print("\nSCENE TIFF CANDIDATES:")

for p in scene_candidates[:30]:
    print(" ", p)

if len(scene_candidates) > 30:
    print(f" ... and {len(scene_candidates) - 30} more")


# =============================================================================
# 5. DISCOVER RPC TXT FILES
# =============================================================================

txt_files = sorted(DATA.rglob("*.txt"))

print("\n" + "-" * 100)
print("RPC TXT DISCOVERY")
print("-" * 100)

print("TXT files:", len(txt_files))

for p in txt_files[:30]:
    print(" ", p)

if len(txt_files) > 30:
    print(f" ... and {len(txt_files) - 30} more")


# =============================================================================
# 6. SHOW TIFF SIZE DISTRIBUTION
#
# This is important because we need to understand the actual dataset layout.
# =============================================================================

print("\n" + "-" * 100)
print("TIFF SIZE SUMMARY")
print("-" * 100)

for tif in all_tifs[:20]:

    try:
        with rasterio.open(tif) as ds:
            print(
                f"{tif.name}\n"
                f"    size={ds.width}x{ds.height} "
                f"bands={ds.count} "
                f"dtype={ds.dtypes[0] if ds.count else 'N/A'} "
                f"crs={ds.crs}"
            )

    except Exception as e:
        print("[ERROR]", tif, repr(e))


# =============================================================================
# 7. SAFETY CHECK — EXPECTED DATASET STRUCTURE
# =============================================================================

print("\n" + "=" * 100)
print("DATASET STRUCTURE CHECK")
print("=" * 100)

print("Expected approximately:")
print("  Scene TIFFs       : 139")
print("  RPC TXT files     : 139")
print("  Elevation mosaics : 3")

print("\nDetected:")
print("  Scene candidates  :", len(scene_candidates))
print("  RPC TXT files     :", len(txt_files))
print("  Elevation mosaics :", len(mosaics))


# Do NOT stop merely because counts differ.
# We continue and inspect the actual structure.


# =============================================================================
# 8. IF NO SCENE CANDIDATES, PRINT ALL TIFF DETAILS
# =============================================================================

if not scene_candidates:

    print("\n[ERROR] No scene candidates detected.")

    print("\nALL TIFF FILES WITH METADATA:")

    for tif in all_tifs:

        try:
            with rasterio.open(tif) as ds:
                print(
                    f"\n{tif}\n"
                    f"  size      : {ds.width} x {ds.height}\n"
                    f"  bands     : {ds.count}\n"
                    f"  dtype     : {ds.dtypes}\n"
                    f"  crs       : {ds.crs}\n"
                    f"  transform : {ds.transform}"
                )

        except Exception as e:
            print("ERROR:", tif, repr(e))

    raise RuntimeError(
        "Could not identify any scene TIFF. "
        "The full TIFF metadata above is required for the next correction."
    )


# =============================================================================
# 9. CHOOSE FIRST SCENE
# =============================================================================

img = scene_candidates[0]

print("\n" + "=" * 100)
print("FIRST SCENE INSPECTION")
print("=" * 100)

print("\nIMAGE:")
print(img)


# =============================================================================
# 10. INSPECT IMAGE
# =============================================================================

with rasterio.open(img) as ds:

    print("\nIMAGE METADATA:")
    print("width      :", ds.width)
    print("height     :", ds.height)
    print("bands      :", ds.count)
    print("dtype      :", ds.dtypes)
    print("crs        :", ds.crs)
    print("transform  :", ds.transform)
    print("bounds     :", ds.bounds)
    print("nodata     :", ds.nodata)

    print("\nIMAGE TAGS:")

    tags = ds.tags()

    if tags:
        for k, v in sorted(tags.items()):
            print(f"{k}: {v}")
    else:
        print("No normal TIFF tags.")

    print("\nRPC TAGS:")

    try:
        rpc_tags = ds.tags(ns="RPC")

        if rpc_tags:
            for k, v in sorted(rpc_tags.items()):
                print(f"{k}: {v}")
        else:
            print("[NO RPC TAGS IN TIFF]")

    except Exception as e:
        print("[RPC TAG ERROR]", repr(e))


# =============================================================================
# 11. MATCH RPC TXT TO IMAGE
# =============================================================================

print("\n" + "=" * 100)
print("RPC FILE MATCHING")
print("=" * 100)

stem = img.stem

print("\nScene stem:")
print(stem)

same_parent_txt = sorted(img.parent.glob("*.txt"))

print("\nTXT files beside scene:", len(same_parent_txt))

for p in same_parent_txt[:30]:
    print(" ", p.name)


# =============================================================================
# 12. FIND BEST RPC MATCH
# =============================================================================

rpc_path = None


# First: exact stem
for p in same_parent_txt:

    if p.stem == stem:
        rpc_path = p
        break


# Second: filename contained in TXT name
if rpc_path is None:

    for p in same_parent_txt:

        if stem in p.name or p.stem in stem:
            rpc_path = p
            break


# Third: global search
if rpc_path is None:

    for p in txt_files:

        if stem in p.name or p.stem in stem:
            rpc_path = p
            break


if rpc_path is None:

    print("\n[WARNING] Could not confidently match RPC TXT.")
    print("The TXT naming convention needs inspection.")

else:

    print("\n[OK] Matched RPC TXT:")
    print(rpc_path)


# =============================================================================
# 13. INSPECT RPC TEXT
# =============================================================================

if rpc_path is not None:

    print("\n" + "-" * 100)
    print("RPC CONTENT INSPECTION")
    print("-" * 100)

    text = rpc_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    print("RPC file size:", len(text), "bytes")

    important_keys = [
        "LINE_OFF",
        "SAMP_OFF",
        "LAT_OFF",
        "LONG_OFF",
        "LON_OFF",
        "HEIGHT_OFF",
        "LINE_SCALE",
        "SAMP_SCALE",
        "LAT_SCALE",
        "LONG_SCALE",
        "LON_SCALE",
        "HEIGHT_SCALE",
    ]

    for key in important_keys:

        pattern = rf"\b{re.escape(key)}\s*=\s*([-+0-9.eE]+)"

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            print(f"{key:15s}: {match.group(1)}")


# =============================================================================
# 14. TEST RASTERIO RPC READING
# =============================================================================

print("\n" + "=" * 100)
print("RASTERIO RPC TEST")
print("=" * 100)

rpc_object = None

try:

    with rasterio.open(img) as ds:

        rpc_tags = ds.tags(ns="RPC")

        if rpc_tags:

            print("[OK] RPC metadata exists inside TIFF.")

            try:
                rpc_object = RPC.from_gdal(
                    rpc_tags
                )

                print("[OK] RPC successfully parsed by Rasterio.")

            except Exception as e:

                print(
                    "[WARNING] RPC tags found, "
                    "but RPC parsing failed:"
                )
                print(repr(e))

        else:

            print(
                "[INFO] TIFF contains no embedded RPC metadata."
            )

except Exception as e:

    print("[RPC OPEN ERROR]", repr(e))


# =============================================================================
# 15. RPC TRANSFORMER TEST
# =============================================================================

if rpc_object is not None:

    print("\n" + "-" * 100)
    print("RPC TRANSFORMER TEST")
    print("-" * 100)

    try:

        transformer = RPCTransformer(
            rpc_object
        )

        with rasterio.open(img) as ds:

            cx = (ds.width - 1) / 2.0
            cy = (ds.height - 1) / 2.0

        print("Image centre pixel:")
        print("  column:", cx)
        print("  row   :", cy)

        # Test several heights.
        for height in [0.0, 100.0, 500.0, 1000.0]:

            try:

                lon, lat = transformer.xy(
                    cy,
                    cx,
                    zs=height,
                )

                print(
                    f"height={height:8.1f} -> "
                    f"lon={lon:.8f}, "
                    f"lat={lat:.8f}"
                )

            except Exception as e:

                print(
                    f"height={height:8.1f} -> ERROR: "
                    f"{repr(e)}"
                )

    except Exception as e:

        print(
            "[WARNING] RPCTransformer creation failed:"
        )
        print(repr(e))


# =============================================================================
# 16. ELEVATION MOSAIC INSPECTION
# =============================================================================

print("\n" + "=" * 100)
print("ELEVATION MOSAIC INSPECTION")
print("=" * 100)

for mosaic in mosaics:

    print("\n" + "-" * 100)
    print(mosaic.name)
    print("-" * 100)

    try:

        with rasterio.open(mosaic) as ds:

            print("size      :", ds.width, "x", ds.height)
            print("bands     :", ds.count)
            print("dtype     :", ds.dtypes)
            print("crs       :", ds.crs)
            print("bounds    :", ds.bounds)
            print("transform :", ds.transform)
            print("nodata    :", ds.nodata)

            cx = (ds.width - 1) / 2.0
            cy = (ds.height - 1) / 2.0

            x, y = ds.xy(
                cy,
                cx,
            )

            print("centre XY :", x, y)

            if ds.crs:

                try:

                    to_wgs84 = Transformer.from_crs(
                        ds.crs,
                        "EPSG:4326",
                        always_xy=True,
                    )

                    lon, lat = to_wgs84.transform(
                        x,
                        y,
                    )

                    print(
                        "centre WGS84:",
                        lon,
                        lat,
                    )

                except Exception as e:

                    print(
                        "[WGS84 TRANSFORM ERROR]",
                        repr(e),
                    )

    except Exception as e:

        print(
            "[MOSAIC ERROR]",
            repr(e),
        )


# =============================================================================
# 17. FINAL DIAGNOSTIC SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 100)

print("Total TIFFs       :", len(all_tifs))
print("Scene candidates  :", len(scene_candidates))
print("RPC TXT files     :", len(txt_files))
print("Elevation mosaics :", len(mosaics))

if len(scene_candidates) == 139:
    print("[OK] 139 scene TIFFs detected.")
else:
    print(
        "[WARNING] Scene count is not 139. "
        "Do NOT start Stage 5 yet."
    )

if len(txt_files) == 139:
    print("[OK] 139 RPC TXT files detected.")
else:
    print(
        "[WARNING] RPC count is not 139."
    )

if len(mosaics) == 3:
    print("[OK] 3 elevation mosaics detected.")
else:
    print(
        "[WARNING] Elevation mosaic count is not 3."
    )

print("\nProbe finished.")
print("=" * 100)