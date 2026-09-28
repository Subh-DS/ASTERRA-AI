import json
from pathlib import Path

import rasterio


MANIFEST = Path(
    r"phase1\potsdam\outputs\potsdam_manifest.json"
)


def main():

    print("=" * 75)
    print("ASTERRA — POTSDAM COMPLETE RGB/DSM GEOMETRY VALIDATION")
    print("=" * 75)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    valid = []
    invalid = []

    for i, item in enumerate(manifest, 1):

        tile_id = item["tile_id"]

        rgb_path = Path(item["rgb"])
        dsm_path = Path(item["dsm"])

        with rasterio.open(rgb_path) as rgb:
            rgb_info = {
                "width": rgb.width,
                "height": rgb.height,
                "count": rgb.count,
                "crs": rgb.crs,
                "res": rgb.res,
                "transform": rgb.transform,
            }

        with rasterio.open(dsm_path) as dsm:
            dsm_info = {
                "width": dsm.width,
                "height": dsm.height,
                "count": dsm.count,
                "crs": dsm.crs,
                "res": dsm.res,
                "transform": dsm.transform,
            }

        problems = []

        if (
            rgb_info["width"] != dsm_info["width"]
            or rgb_info["height"] != dsm_info["height"]
        ):
            problems.append(
                f"SIZE RGB={rgb_info['width']}x{rgb_info['height']} "
                f"DSM={dsm_info['width']}x{dsm_info['height']}"
            )

        if rgb_info["crs"] != dsm_info["crs"]:
            problems.append("CRS mismatch")

        if rgb_info["res"] != dsm_info["res"]:
            problems.append(
                f"RES RGB={rgb_info['res']} DSM={dsm_info['res']}"
            )

        if rgb_info["transform"] != dsm_info["transform"]:
            problems.append("TRANSFORM mismatch")

        if problems:

            invalid.append(
                {
                    "tile_id": tile_id,
                    "problems": problems,
                }
            )

            print(
                f"[INVALID] {tile_id} -> "
                + " | ".join(problems)
            )

        else:

            valid.append(tile_id)

        print(
            f"\rChecked {i}/{len(manifest)}",
            end="",
            flush=True,
        )

    print()
    print()

    print("=" * 75)
    print("VALIDATION RESULT")
    print("=" * 75)

    print(f"Total pairs : {len(manifest)}")
    print(f"Valid pairs : {len(valid)}")
    print(f"Invalid     : {len(invalid)}")

    if invalid:

        print()
        print("INVALID TILES:")

        for item in invalid:

            print(
                f"  {item['tile_id']}: "
                + " | ".join(item["problems"])
            )

    else:

        print()
        print("ALL 38 RGB + DSM PAIRS ARE GEOMETRICALLY ALIGNED.")

    output = Path(
        r"phase1\potsdam\outputs\potsdam_validation.json"
    )

    with open(output, "w", encoding="utf-8") as f:

        json.dump(
            {
                "total": len(manifest),
                "valid": valid,
                "invalid": invalid,
            },
            f,
            indent=2,
        )

    print()
    print(f"Validation report: {output}")


if __name__ == "__main__":
    main()