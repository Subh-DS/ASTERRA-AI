import json
import random
from pathlib import Path

import rasterio


# ============================================================
# ASTERRA — POTSDAM TRAINING DATA PREPARATION
# ============================================================

MANIFEST_PATH = Path(
    r"phase1\potsdam\outputs\potsdam_manifest.json"
)

OUTPUT_DIR = Path(
    r"phase1\potsdam\outputs"
)

TRAIN_INDEX = OUTPUT_DIR / "potsdam_train_index.json"
VAL_INDEX = OUTPUT_DIR / "potsdam_val_index.json"
EXCLUDED_TILES = OUTPUT_DIR / "potsdam_excluded_tiles.json"


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

PATCH_SIZE = 512
STRIDE = 512

# Tile-level validation split.
# Validation tiles remain completely separated
# from training tiles.

VAL_TILES = 8

RANDOM_SEED = 42


# ------------------------------------------------------------
# Known invalid Potsdam tile
# ------------------------------------------------------------
#
# 03_13 has:
#
# RGB = 6000 x 6000
# DSM = 5999 x 6000
#
# CRS, resolution and transform match, but the
# raster dimensions differ by one pixel.
#
# We therefore exclude this tile instead of modifying
# the original geospatial data.
#

KNOWN_INVALID_TILES = {
    "03_13"
}


# ============================================================
# WINDOW GENERATION
# ============================================================

def generate_windows(width, height):
    """
    Generate non-overlapping 512x512 windows.

    Only complete patches are used.
    """

    windows = []

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        STRIDE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        ):

            windows.append(
                {
                    "x": x,
                    "y": y,
                    "width": PATCH_SIZE,
                    "height": PATCH_SIZE,
                }
            )

    return windows


# ============================================================
# TILE VALIDATION
# ============================================================

def inspect_tile(tile):
    """
    Validate RGB/DSM geometry.

    Returns:
        windows, None
            if the tile is valid.

        [], reason
            if the tile is invalid.
    """

    tile_id = tile["tile_id"]

    rgb_path = Path(tile["rgb"])
    dsm_path = Path(tile["dsm"])

    # --------------------------------------------------------
    # Check known exclusions first
    # --------------------------------------------------------

    if tile_id in KNOWN_INVALID_TILES:

        return (
            [],
            "Known invalid tile: RGB/DSM dimension mismatch"
        )

    # --------------------------------------------------------
    # Open RGB
    # --------------------------------------------------------

    with rasterio.open(rgb_path) as rgb:

        rgb_width = rgb.width
        rgb_height = rgb.height
        rgb_crs = rgb.crs
        rgb_res = rgb.res
        rgb_transform = rgb.transform

    # --------------------------------------------------------
    # Open DSM
    # --------------------------------------------------------

    with rasterio.open(dsm_path) as dsm:

        dsm_width = dsm.width
        dsm_height = dsm.height
        dsm_crs = dsm.crs
        dsm_res = dsm.res
        dsm_transform = dsm.transform

    # --------------------------------------------------------
    # Dimension validation
    # --------------------------------------------------------

    if (
        rgb_width != dsm_width
        or rgb_height != dsm_height
    ):

        return (
            [],
            (
                "RGB/DSM size mismatch: "
                f"RGB={rgb_width}x{rgb_height}, "
                f"DSM={dsm_width}x{dsm_height}"
            )
        )

    # --------------------------------------------------------
    # CRS validation
    # --------------------------------------------------------

    if rgb_crs != dsm_crs:

        return (
            [],
            (
                "RGB/DSM CRS mismatch: "
                f"RGB={rgb_crs}, "
                f"DSM={dsm_crs}"
            )
        )

    # --------------------------------------------------------
    # Resolution validation
    # --------------------------------------------------------

    if rgb_res != dsm_res:

        return (
            [],
            (
                "RGB/DSM resolution mismatch: "
                f"RGB={rgb_res}, "
                f"DSM={dsm_res}"
            )
        )

    # --------------------------------------------------------
    # Transform validation
    # --------------------------------------------------------

    if rgb_transform != dsm_transform:

        return (
            [],
            "RGB/DSM transform mismatch"
        )

    # --------------------------------------------------------
    # Generate windows
    # --------------------------------------------------------

    windows = generate_windows(
        rgb_width,
        rgb_height
    )

    return windows, None


# ============================================================
# CREATE PATCH INDEX
# ============================================================

def create_patch_index(tiles, index_name):
    """
    Generate patch records for a list of valid tiles.
    """

    patch_index = []

    print()
    print(
        f"Generating {index_name} patch index..."
    )

    for tile in tiles:

        windows, reason = inspect_tile(tile)

        if reason is not None:

            print(
                f"[SKIP] {tile['tile_id']} -> {reason}"
            )

            continue

        print(
            f"[OK]   {tile['tile_id']} "
            f"-> {len(windows)} patches"
        )

        for window in windows:

            patch_index.append(
                {
                    "tile_id": tile["tile_id"],
                    "rgb": tile["rgb"],
                    "dsm": tile["dsm"],
                    "window": window,
                }
            )

    return patch_index


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "ASTERRA — POTSDAM TRAINING DATA PREPARATION"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Check manifest
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        tiles = json.load(f)

    print()
    print(
        f"Total Potsdam tiles: {len(tiles)}"
    )

    # --------------------------------------------------------
    # Remove known invalid tiles
    # --------------------------------------------------------

    valid_tiles = []
    excluded_tiles = []

    for tile in tiles:

        tile_id = tile["tile_id"]

        if tile_id in KNOWN_INVALID_TILES:

            excluded_tiles.append(
                {
                    "tile_id": tile_id,
                    "reason": (
                        "RGB/DSM dimension mismatch"
                    ),
                }
            )

            continue

        valid_tiles.append(tile)

    # --------------------------------------------------------
    # Print exclusion summary
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("DATASET GEOMETRY FILTER")
    print("=" * 75)

    print(
        f"Original tiles : {len(tiles)}"
    )

    print(
        f"Valid tiles    : {len(valid_tiles)}"
    )

    print(
        f"Excluded tiles : {len(excluded_tiles)}"
    )

    if excluded_tiles:

        print()
        print("Excluded tiles:")

        for item in excluded_tiles:

            print(
                f"  {item['tile_id']} "
                f"-> {item['reason']}"
            )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(valid_tiles) < VAL_TILES:

        raise RuntimeError(
            "Not enough valid tiles for the "
            "requested validation split."
        )

    # --------------------------------------------------------
    # Deterministic tile-level split
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    shuffled = valid_tiles.copy()

    rng.shuffle(shuffled)

    val_tiles = shuffled[:VAL_TILES]

    train_tiles = shuffled[VAL_TILES:]

    # Sort for reproducibility in generated indexes

    train_tiles = sorted(
        train_tiles,
        key=lambda x: x["tile_id"]
    )

    val_tiles = sorted(
        val_tiles,
        key=lambda x: x["tile_id"]
    )

    # --------------------------------------------------------
    # Print split
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("TILE SPLIT")
    print("=" * 75)

    print(
        f"Training tiles   : {len(train_tiles)}"
    )

    print(
        f"Validation tiles : {len(val_tiles)}"
    )

    print()

    print("Training tiles:")

    for tile in train_tiles:

        print(
            f"  {tile['tile_id']}"
        )

    print()

    print("Validation tiles:")

    for tile in val_tiles:

        print(
            f"  {tile['tile_id']}"
        )

    # --------------------------------------------------------
    # Create training patch index
    # --------------------------------------------------------

    train_index = create_patch_index(
        train_tiles,
        "training"
    )

    # --------------------------------------------------------
    # Create validation patch index
    # --------------------------------------------------------

    val_index = create_patch_index(
        val_tiles,
        "validation"
    )

    # --------------------------------------------------------
    # Save excluded tile information
    # --------------------------------------------------------

    with open(
        EXCLUDED_TILES,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            excluded_tiles,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save training index
    # --------------------------------------------------------

    with open(
        TRAIN_INDEX,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            train_index,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save validation index
    # --------------------------------------------------------

    with open(
        VAL_INDEX,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            val_index,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "POTSDAM DATASET PREPARATION COMPLETE"
    )
    print("=" * 75)

    print()

    print(
        f"Original tiles     : {len(tiles)}"
    )

    print(
        f"Valid tiles        : {len(valid_tiles)}"
    )

    print(
        f"Excluded tiles     : {len(excluded_tiles)}"
    )

    print()

    print(
        f"Train tiles        : {len(train_tiles)}"
    )

    print(
        f"Validation tiles   : {len(val_tiles)}"
    )

    print()

    print(
        f"Train patches      : {len(train_index)}"
    )

    print(
        f"Validation patches : {len(val_index)}"
    )

    print()

    print(
        f"Patch size         : "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Stride             : {STRIDE}"
    )

    print()

    print("Files:")

    print(
        f"  Train index      : {TRAIN_INDEX}"
    )

    print(
        f"  Validation index : {VAL_INDEX}"
    )

    print(
        f"  Excluded tiles   : {EXCLUDED_TILES}"
    )

    print()

    print("IMPORTANT:")

    print(
        "Original Potsdam TIFFs were NOT modified."
    )

    print(
        "Original Potsdam TIFFs were NOT copied."
    )

    print(
        "Training will read 512x512 windows "
        "directly from the original TIFFs."
    )

    print()

    print("Dataset status:")

    print(
        "  37 valid RGB/DSM pairs"
    )

    print(
        "  1 excluded pair (03_13)"
    )

    print()

    print(
        "Next step:"
    )

    print(
        "Potsdam training loader + fine-tuning."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()