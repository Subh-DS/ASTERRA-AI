from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# ASTERRA — VAIHINGEN DSM QUALITY INSPECTION
# ============================================================

DSM_PATH = Path(
    r"datasets\Vaihingen\DSM\DSM_09cm_matching.tif"
)


def main():

    print("=" * 75)
    print("ASTERRA — VAIHINGEN DSM QUALITY INSPECTION")
    print("=" * 75)

    print()
    print(f"DSM: {DSM_PATH}")

    with rasterio.open(DSM_PATH) as src:

        print()
        print("METADATA")
        print("-" * 75)

        print(f"Width       : {src.width}")
        print(f"Height      : {src.height}")
        print(f"Bands       : {src.count}")
        print(f"Dtype       : {src.dtypes[0]}")
        print(f"CRS         : {src.crs}")
        print(f"Resolution  : {src.res}")
        print(f"NoData      : {src.nodata}")
        print(f"Transform   : {src.transform}")
        print(f"Bounds      : {src.bounds}")

        print()
        print("READING DSM SAMPLE...")
        print("-" * 75)

        # Read at reduced resolution so we don't load
        # the entire ~400M pixel raster into RAM.
        scale = min(
            1.0,
            2048 / src.width,
            2048 / src.height,
        )

        width = max(1, int(src.width * scale))
        height = max(1, int(src.height * scale))

        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
        )

    values = np.asarray(data.filled(np.nan), dtype=np.float64)

    total = values.size

    finite = np.isfinite(values)

    invalid_nan_inf = ~finite

    # Known DSM no-data value.
    invalid_minus9999 = values == -9999

    # Extremely suspicious values.
    invalid_extreme = (
        finite
        & (values < -1000)
    ) | (
        finite
        & (values > 1000)
    )

    valid = (
        finite
        & ~invalid_minus9999
        & ~invalid_extreme
    )

    valid_values = values[valid]

    print()
    print("=" * 75)
    print("DSM VALUE ANALYSIS")
    print("=" * 75)

    print()
    print(f"Sample pixels          : {total:,}")
    print(f"NaN / Inf              : {invalid_nan_inf.sum():,}")
    print(f"-9999 values           : {invalid_minus9999.sum():,}")
    print(f"Extreme values         : {invalid_extreme.sum():,}")
    print(f"Valid candidate pixels : {valid.sum():,}")

    print()

    if valid_values.size == 0:
        raise RuntimeError(
            "No valid DSM pixels were found."
        )

    print("VALID DSM STATISTICS")
    print("-" * 75)

    print(
        f"Min        : "
        f"{np.min(valid_values):.6f}"
    )

    print(
        f"Max        : "
        f"{np.max(valid_values):.6f}"
    )

    print(
        f"Mean       : "
        f"{np.mean(valid_values):.6f}"
    )

    print(
        f"Median     : "
        f"{np.median(valid_values):.6f}"
    )

    print(
        f"Std        : "
        f"{np.std(valid_values):.6f}"
    )

    percent_valid = (
        valid.sum() / total
    ) * 100.0

    print()
    print(
        f"Valid percentage : "
        f"{percent_valid:.4f}%"
    )

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    print()
    print("DSM PERCENTILES")
    print("-" * 75)

    percentiles = [
        0.1,
        1,
        5,
        25,
        50,
        75,
        95,
        99,
        99.9,
    ]

    results = np.percentile(
        valid_values,
        percentiles,
    )

    for p, value in zip(
        percentiles,
        results,
    ):
        print(
            f"P{p:<5} : {value:.6f}"
        )

    print()
    print("=" * 75)
    print("DSM INSPECTION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()