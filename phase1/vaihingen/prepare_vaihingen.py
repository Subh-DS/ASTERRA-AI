import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# ASTERRA — VAIHINGEN TRAINING DATA PREPARATION
#
# Stage 1:
#   GeoNRW -> Potsdam -> Vaihingen
#
# Primary Vaihingen pair:
#   TOP_Mosaic_09cm.tif
#   DSM_09cm_matching.tif
#
# Important:
#   - Orthomosaic and DSM have matching pixel geometry.
#   - Orthomosaic CRS is missing, DSM CRS is EPSG:32633.
#   - We therefore DO NOT resample/reproject the imagery here.
#   - DSM invalid values are masked.
#   - Train/validation are spatially separated.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATASET_ROOT = Path(r"datasets\Vaihingen")

ORTHO_PATH = (
    DATASET_ROOT
    / "Ortho"
    / "TOP_Mosaic_09cm.tif"
)

DSM_PATH = (
    DATASET_ROOT
    / "DSM"
    / "DSM_09cm_matching.tif"
)

OUTPUT_DIR = Path(r"phase1\vaihingen\outputs")

TRAIN_INDEX = OUTPUT_DIR / "vaihingen_train_index.json"
VAL_INDEX = OUTPUT_DIR / "vaihingen_val_index.json"
SUMMARY_PATH = OUTPUT_DIR / "vaihingen_dataset_summary.json"


# ------------------------------------------------------------
# Patch configuration
# ------------------------------------------------------------

PATCH_SIZE = 512
STRIDE = 512

# Spatial validation:
#
# The last approximately 20% of the usable image height is
# reserved for validation.
#
# One additional patch row is left as a spatial separation
# buffer between training and validation.
VAL_FRACTION = 0.20
SPATIAL_BUFFER_ROWS = 1

# Minimum percentage of valid DSM pixels required.
#
# Invalid DSM values:
#   NaN
#   +/- Inf
#   -9999
#   unrealistic/extreme values
#   zero / near-zero values
#
# The inspection showed valid terrain elevations roughly in
# the range 0..375 m, but zero-valued areas correspond to
# invalid regions in this product. Therefore values > 1 m
# are considered usable for training.
DSM_MIN_VALID = 1.0
DSM_MAX_VALID = 500.0

MIN_VALID_FRACTION = 0.80

RANDOM_SEED = 42


# ============================================================
# Helpers
# ============================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def generate_windows(width, height):
    """
    Generate non-overlapping full 512x512 windows.

    Partial windows at the image boundaries are discarded.
    """

    windows = []

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):

        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            windows.append(
                {
                    "x": x,
                    "y": y,
                    "width": PATCH_SIZE,
                    "height": PATCH_SIZE,
                }
            )

    return windows


def validate_source_files():
    """
    Verify the expected Vaihingen files exist.
    """

    if not ORTHO_PATH.exists():
        raise FileNotFoundError(
            f"Orthomosaic not found:\n{ORTHO_PATH}"
        )

    if not DSM_PATH.exists():
        raise FileNotFoundError(
            f"DSM not found:\n{DSM_PATH}"
        )


def inspect_geometry():
    """
    Inspect and verify that the orthomosaic and DSM share
    the same raster grid.

    CRS is deliberately NOT required to match because the
    orthomosaic has no CRS metadata while its transform,
    dimensions and resolution match the DSM.
    """

    print_header("SOURCE GEOMETRY")

    with rasterio.open(ORTHO_PATH) as ortho, \
         rasterio.open(DSM_PATH) as dsm:

        print("ORTHOMOSAIC")
        print("-" * 75)
        print(f"Size       : {ortho.width} x {ortho.height}")
        print(f"Bands      : {ortho.count}")
        print(f"Dtype      : {ortho.dtypes}")
        print(f"CRS        : {ortho.crs}")
        print(f"Resolution : {ortho.res}")
        print(f"Transform  : {ortho.transform}")

        print()
        print("DSM")
        print("-" * 75)
        print(f"Size       : {dsm.width} x {dsm.height}")
        print(f"Bands      : {dsm.count}")
        print(f"Dtype      : {dsm.dtypes}")
        print(f"CRS        : {dsm.crs}")
        print(f"Resolution : {dsm.res}")
        print(f"Transform  : {dsm.transform}")

        size_match = (
            ortho.width == dsm.width
            and ortho.height == dsm.height
        )

        resolution_match = np.allclose(
            ortho.res,
            dsm.res,
            rtol=0,
            atol=1e-6,
        )

        transform_match = np.allclose(
            np.array(ortho.transform),
            np.array(dsm.transform),
            rtol=0,
            atol=1e-6,
        )

        if not size_match:
            raise RuntimeError(
                "Orthomosaic/DSM size mismatch."
            )

        if not resolution_match:
            raise RuntimeError(
                "Orthomosaic/DSM resolution mismatch."
            )

        if not transform_match:
            raise RuntimeError(
                "Orthomosaic/DSM transform mismatch."
            )

        print()
        print("GEOMETRY RESULT")
        print("-" * 75)
        print(f"Size       : MATCH")
        print(f"Resolution : MATCH")
        print(f"Transform  : MATCH")
        print(f"CRS        : {ortho.crs} vs {dsm.crs}")

        print()
        print(
            "[OK] Orthomosaic and DSM share the same pixel grid."
        )

        print(
            "[INFO] CRS metadata differs/missing, but no "
            "reprojection is required for pixel-aligned training."
        )

        return {
            "width": ortho.width,
            "height": ortho.height,
            "size_match": True,
            "resolution_match": True,
            "transform_match": True,
            "ortho_crs": (
                str(ortho.crs)
                if ortho.crs is not None
                else None
            ),
            "dsm_crs": (
                str(dsm.crs)
                if dsm.crs is not None
                else None
            ),
        }


def calculate_valid_mask(dsm_array):
    """
    Build a valid DSM mask.

    Valid values must be:
      - finite
      - greater than DSM_MIN_VALID
      - below DSM_MAX_VALID

    This removes:
      - NaN
      - Inf
      - -9999
      - zero / near-zero invalid areas
      - extreme corrupted values
    """

    valid = np.isfinite(dsm_array)

    valid &= dsm_array > DSM_MIN_VALID
    valid &= dsm_array < DSM_MAX_VALID

    return valid


def inspect_dsm_quality():
    """
    Read the DSM in chunks and calculate global quality
    statistics without loading the complete raster into RAM.
    """

    print_header("DSM QUALITY ANALYSIS")

    total_pixels = 0
    valid_pixels = 0
    invalid_pixels = 0

    nan_inf_pixels = 0
    nodata_pixels = 0
    zero_or_low_pixels = 0
    extreme_pixels = 0

    values_for_stats = []

    with rasterio.open(DSM_PATH) as dsm:

        for _, window in dsm.block_windows(1):

            arr = dsm.read(
                1,
                window=window,
                out_dtype="float32",
            )

            total_pixels += arr.size

            finite = np.isfinite(arr)

            nan_inf_pixels += int(
                np.count_nonzero(~finite)
            )

            nodata_mask = arr == -9999
            nodata_pixels += int(
                np.count_nonzero(nodata_mask)
            )

            low_mask = finite & (
                arr <= DSM_MIN_VALID
            )

            zero_or_low_pixels += int(
                np.count_nonzero(low_mask)
            )

            extreme_mask = finite & (
                arr >= DSM_MAX_VALID
            )

            extreme_pixels += int(
                np.count_nonzero(extreme_mask)
            )

            valid = calculate_valid_mask(arr)

            count_valid = int(
                np.count_nonzero(valid)
            )

            valid_pixels += count_valid

            if count_valid > 0:
                values_for_stats.append(
                    arr[valid]
                )

    if values_for_stats:

        values = np.concatenate(
            values_for_stats
        )

        stats = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p01": float(np.percentile(values, 0.1)),
            "p01_percentile": float(
                np.percentile(values, 1)
            ),
            "p05": float(
                np.percentile(values, 5)
            ),
            "p25": float(
                np.percentile(values, 25)
            ),
            "p50": float(
                np.percentile(values, 50)
            ),
            "p75": float(
                np.percentile(values, 75)
            ),
            "p95": float(
                np.percentile(values, 95)
            ),
            "p99": float(
                np.percentile(values, 99)
            ),
        }

        del values

    else:

        stats = {}

    invalid_pixels = (
        total_pixels - valid_pixels
    )

    valid_percentage = (
        100.0 * valid_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    print(f"Total pixels       : {total_pixels:,}")
    print(f"Valid pixels       : {valid_pixels:,}")
    print(f"Invalid pixels     : {invalid_pixels:,}")
    print(f"Valid percentage   : {valid_percentage:.4f}%")

    print()
    print(f"NaN / Inf          : {nan_inf_pixels:,}")
    print(f"-9999              : {nodata_pixels:,}")
    print(
        f"Low/zero values    : {zero_or_low_pixels:,}"
    )
    print(f"Extreme values     : {extreme_pixels:,}")

    if stats:

        print()
        print("VALID DSM STATISTICS")
        print("-" * 75)

        print(f"Min       : {stats['min']:.6f}")
        print(f"Max       : {stats['max']:.6f}")
        print(f"Mean      : {stats['mean']:.6f}")
        print(f"Median    : {stats['median']:.6f}")
        print(f"Std       : {stats['std']:.6f}")

        print()
        print("Percentiles")
        print(f"P0.1      : {stats['p01']:.6f}")
        print(
            f"P1        : "
            f"{stats['p01_percentile']:.6f}"
        )
        print(f"P5        : {stats['p05']:.6f}")
        print(f"P25       : {stats['p25']:.6f}")
        print(f"P50       : {stats['p50']:.6f}")
        print(f"P75       : {stats['p75']:.6f}")
        print(f"P95       : {stats['p95']:.6f}")
        print(f"P99       : {stats['p99']:.6f}")

    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
        "valid_percentage": valid_percentage,
        "nan_inf_pixels": nan_inf_pixels,
        "nodata_pixels": nodata_pixels,
        "zero_or_low_pixels": zero_or_low_pixels,
        "extreme_pixels": extreme_pixels,
        "statistics": stats,
    }


def calculate_spatial_split(height):
    """
    Create a spatial train/validation split.

    Validation occupies approximately the final 20% of
    complete patch rows.

    One patch row immediately before validation is left
    unused as a spatial buffer.
    """

    num_rows = (
        (height - PATCH_SIZE)
        // STRIDE
        + 1
    )

    requested_val_rows = max(
        1,
        int(
            np.ceil(
                num_rows * VAL_FRACTION
            )
        ),
    )

    val_start_row = (
        num_rows - requested_val_rows
    )

    train_end_row = (
        val_start_row
        - SPATIAL_BUFFER_ROWS
    )

    if train_end_row <= 0:
        raise RuntimeError(
            "Not enough patch rows for "
            "train/validation split."
        )

    return {
        "num_rows": num_rows,
        "validation_rows": requested_val_rows,
        "train_end_row": train_end_row,
        "validation_start_row": val_start_row,
    }


def evaluate_patch(
    dsm_dataset,
    window,
):
    """
    Calculate DSM validity for one patch.
    """

    raster_window = Window(
        window["x"],
        window["y"],
        window["width"],
        window["height"],
    )

    dsm = dsm_dataset.read(
        1,
        window=raster_window,
        out_dtype="float32",
    )

    valid_mask = calculate_valid_mask(
        dsm
    )

    valid_pixels = int(
        np.count_nonzero(valid_mask)
    )

    total_pixels = dsm.size

    valid_fraction = (
        valid_pixels / total_pixels
    )

    return (
        valid_fraction,
        valid_pixels,
        total_pixels,
    )


def create_patch_record(
    window,
    valid_fraction,
    valid_pixels,
    total_pixels,
):
    """
    Create the JSON representation of a patch.
    """

    return {
        "rgb": str(ORTHO_PATH),
        "dsm": str(DSM_PATH),

        "window": window,

        "valid_dsm_fraction": round(
            float(valid_fraction),
            6,
        ),

        "valid_dsm_pixels": valid_pixels,
        "total_pixels": total_pixels,

        "patch_size": PATCH_SIZE,
    }


def generate_patch_indices(
    width,
    height,
    split_info,
):
    """
    Generate train and validation patch indices.
    """

    print_header(
        "GENERATING VAIHINGEN PATCH INDEX"
    )

    windows = generate_windows(
        width,
        height,
    )

    print(
        f"Full patches available : {len(windows)}"
    )

    print(
        f"Patch size              : "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Stride                  : {STRIDE}"
    )

    print(
        f"Minimum valid DSM       : "
        f"{MIN_VALID_FRACTION * 100:.1f}%"
    )

    print()

    print(
        "Spatial split:"
    )

    print(
        f"  Total patch rows      : "
        f"{split_info['num_rows']}"
    )

    print(
        f"  Training rows         : "
        f"0 - {split_info['train_end_row'] - 1}"
    )

    print(
        f"  Buffer rows           : "
        f"{split_info['train_end_row']} - "
        f"{split_info['validation_start_row'] - 1}"
    )

    print(
        f"  Validation rows       : "
        f"{split_info['validation_start_row']} - "
        f"{split_info['num_rows'] - 1}"
    )

    train_index = []
    val_index = []

    rejected_train = 0
    rejected_val = 0

    with rasterio.open(
        DSM_PATH
    ) as dsm:

        for i, window in enumerate(
            windows,
            start=1,
        ):

            row = (
                window["y"]
                // STRIDE
            )

            valid_fraction, valid_pixels, total_pixels = (
                evaluate_patch(
                    dsm,
                    window,
                )
            )

            record = create_patch_record(
                window,
                valid_fraction,
                valid_pixels,
                total_pixels,
            )

            # ------------------------------------------------
            # Training region
            # ------------------------------------------------

            if row < split_info[
                "train_end_row"
            ]:

                if (
                    valid_fraction
                    >= MIN_VALID_FRACTION
                ):

                    train_index.append(
                        record
                    )

                else:

                    rejected_train += 1

            # ------------------------------------------------
            # Validation region
            # ------------------------------------------------

            elif row >= split_info[
                "validation_start_row"
            ]:

                if (
                    valid_fraction
                    >= MIN_VALID_FRACTION
                ):

                    val_index.append(
                        record
                    )

                else:

                    rejected_val += 1

            # ------------------------------------------------
            # Buffer region
            # ------------------------------------------------

            else:
                pass

            if (
                i % 100 == 0
                or i == len(windows)
            ):

                print(
                    f"\rProcessed patches: "
                    f"{i}/{len(windows)}",
                    end="",
                    flush=True,
                )

    print()
    print()

    print(
        f"Accepted training patches   : "
        f"{len(train_index)}"
    )

    print(
        f"Rejected training patches   : "
        f"{rejected_train}"
    )

    print(
        f"Accepted validation patches : "
        f"{len(val_index)}"
    )

    print(
        f"Rejected validation patches : "
        f"{rejected_val}"
    )

    return (
        train_index,
        val_index,
        rejected_train,
        rejected_val,
    )


def save_json(path, data):
    """
    Save formatted JSON.
    """

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print(
        "ASTERRA — VAIHINGEN TRAINING DATA PREPARATION"
    )
    print("=" * 75)

    print()
    print(
        f"Dataset root: "
        f"{DATASET_ROOT.resolve()}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    validate_source_files()

    print()
    print(
        "[OK] Orthomosaic found:"
    )
    print(
        f"     {ORTHO_PATH}"
    )

    print()
    print(
        "[OK] DSM found:"
    )
    print(
        f"     {DSM_PATH}"
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry = inspect_geometry()

    width = geometry["width"]
    height = geometry["height"]

    # --------------------------------------------------------
    # DSM quality
    # --------------------------------------------------------

    dsm_quality = inspect_dsm_quality()

    # --------------------------------------------------------
    # Spatial split
    # --------------------------------------------------------

    print_header(
        "SPATIAL TRAIN / VALIDATION SPLIT"
    )

    split_info = calculate_spatial_split(
        height
    )

    print(
        f"Raster height        : {height}"
    )

    print(
        f"Patch rows            : "
        f"{split_info['num_rows']}"
    )

    print(
        f"Validation fraction   : "
        f"{VAL_FRACTION * 100:.1f}%"
    )

    print(
        f"Buffer rows           : "
        f"{SPATIAL_BUFFER_ROWS}"
    )

    # --------------------------------------------------------
    # Generate patches
    # --------------------------------------------------------

    (
        train_index,
        val_index,
        rejected_train,
        rejected_val,
    ) = generate_patch_indices(
        width,
        height,
        split_info,
    )

    # --------------------------------------------------------
    # Save indexes
    # --------------------------------------------------------

    save_json(
        TRAIN_INDEX,
        train_index,
    )

    save_json(
        VAL_INDEX,
        val_index,
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    summary = {
        "dataset": "ISPRS Vaihingen",
        "stage": "Stage 1 — Geometry adaptation",

        "source": {
            "orthomosaic": str(
                ORTHO_PATH
            ),
            "dsm": str(
                DSM_PATH
            ),
        },

        "geometry": geometry,

        "dsm_quality": dsm_quality,

        "patch_configuration": {
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "minimum_valid_dsm_fraction":
                MIN_VALID_FRACTION,
            "dsm_min_valid":
                DSM_MIN_VALID,
            "dsm_max_valid":
                DSM_MAX_VALID,
        },

        "spatial_split": split_info,

        "dataset_counts": {
            "available_full_patches":
                len(
                    generate_windows(
                        width,
                        height,
                    )
                ),

            "training_patches":
                len(train_index),

            "validation_patches":
                len(val_index),

            "rejected_training_patches":
                rejected_train,

            "rejected_validation_patches":
                rejected_val,
        },

        "important": [
            "Original TIFFs were not modified.",
            "Original TIFFs were not copied.",
            "Training reads 512x512 windows directly from the source TIFFs.",
            "Orthomosaic and DSM are treated as pixel-grid aligned.",
            "DSM invalid pixels are excluded through validity filtering.",
            "Training and validation regions are spatially separated.",
        ],
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_header(
        "VAIHINGEN DATASET PREPARATION COMPLETE"
    )

    print()
    print(
        f"Raster size              : "
        f"{width} x {height}"
    )

    print(
        f"Full patches             : "
        f"{summary['dataset_counts']['available_full_patches']}"
    )

    print()
    print(
        f"Training patches         : "
        f"{len(train_index)}"
    )

    print(
        f"Validation patches       : "
        f"{len(val_index)}"
    )

    print()
    print(
        f"Rejected training        : "
        f"{rejected_train}"
    )

    print(
        f"Rejected validation      : "
        f"{rejected_val}"
    )

    print()
    print(
        f"Patch size               : "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Stride                   : "
        f"{STRIDE}"
    )

    print(
        f"Minimum DSM validity     : "
        f"{MIN_VALID_FRACTION * 100:.1f}%"
    )

    print()
    print("Files:")
    print(
        f"  Train index             : "
        f"{TRAIN_INDEX}"
    )

    print(
        f"  Validation index        : "
        f"{VAL_INDEX}"
    )

    print(
        f"  Dataset summary         : "
        f"{SUMMARY_PATH}"
    )

    print()
    print("IMPORTANT:")
    print(
        "  Original Vaihingen TIFFs were NOT modified."
    )

    print(
        "  Original Vaihingen TIFFs were NOT copied."
    )

    print(
        "  DSM invalid pixels are filtered during "
        "patch selection."
    )

    print(
        "  Train/validation regions are spatially "
        "separated."
    )

    print()
    print("NEXT STEP:")
    print(
        "  Review the patch counts and then create "
        "the Vaihingen fine-tuning loader."
    )


if __name__ == "__main__":
    main()