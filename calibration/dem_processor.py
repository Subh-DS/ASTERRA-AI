from pathlib import Path

import numpy as np
import rasterio

from nodata_handler import get_valid_mask


def load_dem(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"DEM file does not exist: {path}"
        )

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        mask = src.read_masks(1) > 0

        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "resolution": src.res,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }

    valid_mask = get_valid_mask(
        data,
        raster_mask=mask,
        nodata=metadata["nodata"],
    )

    return data, valid_mask, metadata


def print_dem_info(data, valid_mask, metadata):

    print("=" * 70)
    print("DEM INFORMATION")
    print("=" * 70)

    print(f"Shape       : {data.shape}")
    print(f"CRS         : {metadata['crs']}")
    print(f"Resolution  : {metadata['resolution']}")
    print(f"Nodata      : {metadata['nodata']}")
    print(f"Valid %     : {valid_mask.mean() * 100:.6f}%")

    if valid_mask.any():

        values = data[valid_mask]

        print(f"Min         : {values.min():.6f}")
        print(f"Max         : {values.max():.6f}")
        print(f"Mean        : {values.mean():.6f}")
        print(f"Std         : {values.std():.6f}")

    print("=" * 70)