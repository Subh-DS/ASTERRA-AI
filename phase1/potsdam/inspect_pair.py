import json
from pathlib import Path

import cv2
import numpy as np
import rasterio


# ============================================================
# ASTERRA — ISPRS POTSDAM RGB + DSM VALIDATION
# ============================================================

ROOT = Path(r"datasets\Potsdam")

RGB_ROOT = ROOT / "2_Ortho_RGB"
DSM_ROOT = ROOT / "1_DSM"

OUTPUT_DIR = Path(r"phase1\potsdam\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def tile_id_from_rgb(name: str):
    """
    top_potsdam_2_10_RGB.tif
        -> 2_10
    """
    stem = Path(name).stem

    if not stem.startswith("top_potsdam_"):
        return None

    stem = stem.replace("top_potsdam_", "")
    stem = stem.replace("_RGB", "")

    parts = stem.split("_")

    if len(parts) != 2:
        return None

    row = int(parts[0])
    col = int(parts[1])

    return f"{row:02d}_{col:02d}"


def tile_id_from_dsm(name: str):
    """
    dsm_potsdam_02_10.tif
        -> 02_10
    """
    stem = Path(name).stem

    if not stem.startswith("dsm_potsdam_"):
        return None

    stem = stem.replace("dsm_potsdam_", "")

    parts = stem.split("_")

    if len(parts) != 2:
        return None

    row = int(parts[0])
    col = int(parts[1])

    return f"{row:02d}_{col:02d}"


def find_files():
    print("Searching RGB files...")

    rgb_files = {}

    for path in RGB_ROOT.rglob("*.tif"):
        tile_id = tile_id_from_rgb(path.name)

        if tile_id is not None:
            rgb_files[tile_id] = path

    print(f"RGB tiles found: {len(rgb_files)}")

    print()
    print("Searching DSM files...")

    dsm_files = {}

    for path in DSM_ROOT.rglob("*.tif"):
        tile_id = tile_id_from_dsm(path.name)

        if tile_id is not None:
            dsm_files[tile_id] = path

    print(f"DSM tiles found: {len(dsm_files)}")

    return rgb_files, dsm_files


def validate_pairs(rgb_files, dsm_files):

    common_ids = sorted(
        set(rgb_files.keys()) & set(dsm_files.keys())
    )

    rgb_only = sorted(
        set(rgb_files.keys()) - set(dsm_files.keys())
    )

    dsm_only = sorted(
        set(dsm_files.keys()) - set(rgb_files.keys())
    )

    print()
    print("=" * 70)
    print("PAIRING RESULT")
    print("=" * 70)

    print(f"RGB tiles:       {len(rgb_files)}")
    print(f"DSM tiles:       {len(dsm_files)}")
    print(f"Matched pairs:   {len(common_ids)}")
    print(f"RGB without DSM: {len(rgb_only)}")
    print(f"DSM without RGB: {len(dsm_only)}")

    if not common_ids:
        raise RuntimeError("No RGB + DSM pairs found.")

    print()
    print("First 10 matched pairs:")

    for tile_id in common_ids[:10]:
        print()
        print(f"[{tile_id}]")
        print(f"RGB: {rgb_files[tile_id]}")
        print(f"DSM: {dsm_files[tile_id]}")

    return common_ids


def inspect_pair(tile_id, rgb_path, dsm_path):

    print()
    print("=" * 70)
    print(f"INSPECTING TILE: {tile_id}")
    print("=" * 70)

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    print()
    print("Opening RGB...")

    with rasterio.open(rgb_path) as src:

        rgb = src.read()

        print()
        print("RGB metadata")
        print("-" * 70)

        print(f"Width       : {src.width}")
        print(f"Height      : {src.height}")
        print(f"Bands       : {src.count}")
        print(f"Dtype       : {src.dtypes}")
        print(f"CRS         : {src.crs}")
        print(f"Resolution  : {src.res}")
        print(f"Transform   : {src.transform}")
        print(f"NoData      : {src.nodata}")

        print()
        print("RGB array")
        print("-" * 70)

        print(f"Shape       : {rgb.shape}")
        print(f"Dtype       : {rgb.dtype}")
        print(f"Min         : {rgb.min()}")
        print(f"Max         : {rgb.max()}")
        print(f"Mean        : {rgb.mean():.4f}")

        print(f"NaN         : {np.isnan(rgb).sum()}")
        print(f"Inf         : {np.isinf(rgb).sum()}")

        rgb_height = src.height
        rgb_width = src.width
        rgb_crs = src.crs
        rgb_res = src.res
        rgb_transform = src.transform

    # --------------------------------------------------------
    # DSM
    # --------------------------------------------------------

    print()
    print("Opening DSM...")

    with rasterio.open(dsm_path) as src:

        dsm = src.read(1)

        print()
        print("DSM metadata")
        print("-" * 70)

        print(f"Width       : {src.width}")
        print(f"Height      : {src.height}")
        print(f"Bands       : {src.count}")
        print(f"Dtype       : {src.dtypes}")
        print(f"CRS         : {src.crs}")
        print(f"Resolution  : {src.res}")
        print(f"Transform   : {src.transform}")
        print(f"NoData      : {src.nodata}")

        valid = np.isfinite(dsm)

        if src.nodata is not None:
            valid &= dsm != src.nodata

        valid_values = dsm[valid]

        print()
        print("DSM array")
        print("-" * 70)

        print(f"Shape       : {dsm.shape}")
        print(f"Dtype       : {dsm.dtype}")

        if len(valid_values) > 0:
            print(f"Min         : {valid_values.min():.6f}")
            print(f"Max         : {valid_values.max():.6f}")
            print(f"Mean        : {valid_values.mean():.6f}")

        print(f"NaN         : {np.isnan(dsm).sum()}")
        print(f"Inf         : {np.isinf(dsm).sum()}")

        dsm_height = src.height
        dsm_width = src.width
        dsm_crs = src.crs
        dsm_res = src.res
        dsm_transform = src.transform
        dsm_nodata = src.nodata

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RGB / DSM ALIGNMENT")
    print("=" * 70)

    size_match = (
        rgb_width == dsm_width
        and rgb_height == dsm_height
    )

    crs_match = rgb_crs == dsm_crs
    resolution_match = rgb_res == dsm_res
    transform_match = rgb_transform == dsm_transform

    print(
        f"Spatial dimensions : "
        f"{rgb_width}x{rgb_height} vs "
        f"{dsm_width}x{dsm_height} "
        f"{'MATCH' if size_match else 'MISMATCH'}"
    )

    print(
        f"CRS                : "
        f"{'MATCH' if crs_match else 'MISMATCH'}"
    )

    print(
        f"Resolution         : "
        f"{'MATCH' if resolution_match else 'MISMATCH'}"
    )

    print(
        f"Transform          : "
        f"{'MATCH' if transform_match else 'MISMATCH'}"
    )

    if not size_match:
        raise RuntimeError("RGB and DSM dimensions do not match.")

    if not crs_match:
        print("WARNING: CRS differs.")

    if not resolution_match:
        print("WARNING: Resolution differs.")

    if not transform_match:
        print("WARNING: Transform differs.")

    # --------------------------------------------------------
    # Diagnostic RGB
    # --------------------------------------------------------

    rgb_vis = np.transpose(rgb, (1, 2, 0))

    rgb_vis = np.clip(rgb_vis, 0, 255).astype(np.uint8)

    rgb_output = OUTPUT_DIR / f"potsdam_{tile_id}_rgb.png"

    cv2.imwrite(
        str(rgb_output),
        cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR),
    )

    # --------------------------------------------------------
    # Diagnostic DSM
    # --------------------------------------------------------

    dsm_valid = np.isfinite(dsm)

    if dsm_nodata is not None:
        dsm_valid &= dsm != dsm_nodata

    if np.any(dsm_valid):

        dsm_min = dsm[dsm_valid].min()
        dsm_max = dsm[dsm_valid].max()

        dsm_norm = np.zeros_like(
            dsm,
            dtype=np.float32,
        )

        dsm_norm[dsm_valid] = (
            (dsm[dsm_valid] - dsm_min)
            / (dsm_max - dsm_min + 1e-8)
        )

        dsm_png = (
            dsm_norm * 255.0
        ).clip(0, 255).astype(np.uint8)

        dsm_output = OUTPUT_DIR / f"potsdam_{tile_id}_dsm.png"

        cv2.imwrite(
            str(dsm_output),
            dsm_png,
        )

    print()
    print("Diagnostic outputs:")
    print(f"RGB: {rgb_output}")

    if np.any(dsm_valid):
        print(f"DSM: {dsm_output}")


def main():

    print("=" * 70)
    print("ASTERRA — POTSDAM RGB + DSM VALIDATION")
    print("=" * 70)

    if not RGB_ROOT.exists():
        raise FileNotFoundError(
            f"RGB directory not found: {RGB_ROOT}"
        )

    if not DSM_ROOT.exists():
        raise FileNotFoundError(
            f"DSM directory not found: {DSM_ROOT}"
        )

    rgb_files, dsm_files = find_files()

    common_ids = validate_pairs(
        rgb_files,
        dsm_files,
    )

    # Inspect one representative tile first.
    sample_id = common_ids[0]

    inspect_pair(
        sample_id,
        rgb_files[sample_id],
        dsm_files[sample_id],
    )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = []

    for tile_id in common_ids:

        manifest.append(
            {
                "tile_id": tile_id,
                "rgb": str(
                    rgb_files[tile_id].resolve()
                ),
                "dsm": str(
                    dsm_files[tile_id].resolve()
                ),
            }
        )

    manifest_path = (
        OUTPUT_DIR / "potsdam_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("POTSDAM INSPECTION COMPLETE")
    print("=" * 70)

    print(f"Matched pairs : {len(common_ids)}")
    print(f"Manifest      : {manifest_path}")
    print()
    print("Next step: Potsdam dataset caching.")


if __name__ == "__main__":
    main()