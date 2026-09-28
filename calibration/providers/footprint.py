from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


def get_wgs84_bounds(raster_path):

    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            raise ValueError(
                "Raster has no CRS."
            )

        bounds = src.bounds

        wgs84_bounds = transform_bounds(
            src.crs,
            "EPSG:4326",
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )

    return wgs84_bounds


def print_footprint(raster_path):

    bounds = get_wgs84_bounds(raster_path)

    left, bottom, right, top = bounds

    print("=" * 70)
    print("ASTERRA RASTER FOOTPRINT")
    print("=" * 70)

    print("Raster:")
    print(raster_path)

    print("\nWGS84 / EPSG:4326")

    print(f"West  : {left:.8f}")
    print(f"South : {bottom:.8f}")
    print(f"East  : {right:.8f}")
    print(f"North : {top:.8f}")

    print("=" * 70)


if __name__ == "__main__":

    print_footprint(
        r"D:\Asterra AI\datasets\Urban3D\train\Inputs"
        r"\JAX_Tile_004_RGB.tif"
    )