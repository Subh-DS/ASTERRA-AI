import json
from pathlib import Path

import rasterio


# ============================================================
# ASTERRA — VAIHINGEN DATASET INSPECTION
# ============================================================

BASE_DIR = Path(r"datasets\Vaihingen")

IMAGES_DIR = BASE_DIR / "Images"
ORTHO_DIR = BASE_DIR / "Ortho"
DSM_DIR = BASE_DIR / "DSM"

OUTPUT_DIR = Path(r"phase1\vaihingen\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "vaihingen_inspection.json"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def print_separator(char="=", width=75):
    print(char * width)


def inspect_raster(path):
    """Read important raster metadata and statistics."""

    with rasterio.open(path) as src:
        info = {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "resolution": tuple(src.res),
            "transform": str(src.transform),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "nodata": src.nodata,
        }

        # Small sample for statistics.
        # We deliberately do not load the entire huge raster.
        scale = min(
            1.0,
            2048 / src.width,
            2048 / src.height,
        )

        out_width = max(1, int(src.width * scale))
        out_height = max(1, int(src.height * scale))

        sample = src.read(
            1,
            out_shape=(out_height, out_width),
        )

        valid = sample[
            sample == sample
        ]

        if valid.size > 0:
            info["sample_min"] = float(valid.min())
            info["sample_max"] = float(valid.max())
            info["sample_mean"] = float(valid.mean())
            info["sample_nan"] = int(
                (~(sample == sample)).sum()
            )
        else:
            info["sample_min"] = None
            info["sample_max"] = None
            info["sample_mean"] = None
            info["sample_nan"] = None

        return info


def compare_geometry(image_info, dsm_info):
    """Compare image and DSM spatial geometry."""

    result = {
        "size_match": (
            image_info["width"] == dsm_info["width"]
            and image_info["height"] == dsm_info["height"]
        ),
        "crs_match": (
            image_info["crs"] == dsm_info["crs"]
        ),
        "resolution_match": (
            image_info["resolution"]
            == dsm_info["resolution"]
        ),
        "transform_match": (
            image_info["transform"]
            == dsm_info["transform"]
        ),
        "bounds_match": (
            image_info["bounds"]
            == dsm_info["bounds"]
        ),
    }

    result["fully_aligned"] = all(result.values())

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("ASTERRA — VAIHINGEN DATASET INSPECTION")
    print_separator()

    print()
    print(f"Dataset root: {BASE_DIR.resolve()}")

    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"Vaihingen dataset not found:\n{BASE_DIR.resolve()}"
        )

    # --------------------------------------------------------
    # 1. Images
    # --------------------------------------------------------

    print()
    print_separator()
    print("1. VAIHINGEN IMAGE DATA")
    print_separator()

    image_files = sorted(
        IMAGES_DIR.rglob("*.tif")
    )

    print(f"TIFF files found: {len(image_files)}")

    image_infos = []

    for path in image_files:
        print()
        print(f"IMAGE: {path.name}")

        try:
            info = inspect_raster(path)

            image_infos.append(info)

            print(f"  Size       : {info['width']} x {info['height']}")
            print(f"  Bands      : {info['count']}")
            print(f"  Dtype      : {info['dtype']}")
            print(f"  CRS        : {info['crs']}")
            print(f"  Resolution : {info['resolution']}")
            print(f"  NoData     : {info['nodata']}")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    # --------------------------------------------------------
    # 2. Orthomosaic
    # --------------------------------------------------------

    print()
    print_separator()
    print("2. VAIHINGEN ORTHOMOSAIC")
    print_separator()

    ortho_files = sorted(
        ORTHO_DIR.rglob("*.tif")
    )

    print(f"TIFF files found: {len(ortho_files)}")

    ortho_infos = []

    for path in ortho_files:
        print()
        print(f"ORTHO: {path.name}")

        try:
            info = inspect_raster(path)

            ortho_infos.append(info)

            print(f"  Size       : {info['width']} x {info['height']}")
            print(f"  Bands      : {info['count']}")
            print(f"  Dtype      : {info['dtype']}")
            print(f"  CRS        : {info['crs']}")
            print(f"  Resolution : {info['resolution']}")
            print(f"  NoData     : {info['nodata']}")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    # --------------------------------------------------------
    # 3. DSM products
    # --------------------------------------------------------

    print()
    print_separator()
    print("3. VAIHINGEN DSM PRODUCTS")
    print_separator()

    dsm_files = sorted(
        DSM_DIR.rglob("*.tif")
    )

    print(f"DSM TIFF files found: {len(dsm_files)}")

    dsm_infos = []

    for path in dsm_files:
        print()
        print(f"DSM: {path.name}")

        try:
            info = inspect_raster(path)

            dsm_infos.append(info)

            print(f"  Size       : {info['width']} x {info['height']}")
            print(f"  Bands      : {info['count']}")
            print(f"  Dtype      : {info['dtype']}")
            print(f"  CRS        : {info['crs']}")
            print(f"  Resolution : {info['resolution']}")
            print(f"  NoData     : {info['nodata']}")
            print(f"  Sample min : {info['sample_min']}")
            print(f"  Sample max : {info['sample_max']}")
            print(f"  Sample mean: {info['sample_mean']}")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    # --------------------------------------------------------
    # 4. Compare orthomosaic against DSM products
    # --------------------------------------------------------

    print()
    print_separator()
    print("4. ORTHOMOSAIC / DSM GEOMETRY COMPARISON")
    print_separator()

    comparisons = []

    if len(ortho_infos) == 0:
        print("No orthomosaic TIFF found.")

    elif len(dsm_infos) == 0:
        print("No DSM TIFF found.")

    else:

        # Use first orthomosaic.
        ortho = ortho_infos[0]

        print()
        print(f"Reference image: {ortho['path']}")

        for dsm in dsm_infos:

            comparison = compare_geometry(
                ortho,
                dsm,
            )

            record = {
                "image": ortho["path"],
                "dsm": dsm["path"],
                "geometry": comparison,
            }

            comparisons.append(record)

            print()
            print(f"DSM: {Path(dsm['path']).name}")
            print("-" * 75)

            print(
                f"Size       : "
                f"{'MATCH' if comparison['size_match'] else 'MISMATCH'}"
            )

            print(
                f"CRS        : "
                f"{'MATCH' if comparison['crs_match'] else 'MISMATCH'}"
            )

            print(
                f"Resolution : "
                f"{'MATCH' if comparison['resolution_match'] else 'MISMATCH'}"
            )

            print(
                f"Transform   : "
                f"{'MATCH' if comparison['transform_match'] else 'MISMATCH'}"
            )

            print(
                f"Bounds      : "
                f"{'MATCH' if comparison['bounds_match'] else 'MISMATCH'}"
            )

            print(
                f"FULL ALIGN : "
                f"{'YES' if comparison['fully_aligned'] else 'NO'}"
            )

    # --------------------------------------------------------
    # 5. Determine candidate training source
    # --------------------------------------------------------

    print()
    print_separator()
    print("5. TRAINING DATASET DECISION")
    print_separator()

    fully_aligned = [
        x
        for x in comparisons
        if x["geometry"]["fully_aligned"]
    ]

    if fully_aligned:

        print()
        print("FULLY ALIGNED ORTHO/DSM PAIRS FOUND:")

        for item in fully_aligned:
            print(
                f"  DSM: {Path(item['dsm']).name}"
            )

        print()
        print(
            "These are candidates for direct RGB → DSM "
            "training."
        )

    else:

        print()
        print(
            "No fully aligned orthomosaic/DSM pair was found."
        )

        print()
        print(
            "This does NOT mean Vaihingen cannot be used."
        )

        print(
            "It means the training loader will likely need "
            "spatial alignment/resampling/cropping."
        )

    # --------------------------------------------------------
    # 6. Save report
    # --------------------------------------------------------

    report = {
        "dataset": "ISPRS Vaihingen",
        "base_dir": str(BASE_DIR.resolve()),
        "images": image_infos,
        "orthomosaics": ortho_infos,
        "dsms": dsm_infos,
        "comparisons": comparisons,
        "fully_aligned_pairs": fully_aligned,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print_separator()
    print("VAIHINGEN INSPECTION COMPLETE")
    print_separator()

    print()
    print(f"Image TIFFs       : {len(image_infos)}")
    print(f"Orthomosaic TIFFs : {len(ortho_infos)}")
    print(f"DSM TIFFs         : {len(dsm_infos)}")
    print(
        f"Aligned pairs     : {len(fully_aligned)}"
    )

    print()
    print("Inspection report:")
    print(f"  {REPORT_PATH}")

    print()
    print("NEXT STEP:")
    print(
        "Review the geometry results before creating "
        "the Vaihingen patch index."
    )


if __name__ == "__main__":
    main()