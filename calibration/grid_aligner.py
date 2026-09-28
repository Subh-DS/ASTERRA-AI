import numpy as np

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling


def align_dem_to_reference(
    dem_path,
    reference_path,
    resampling=Resampling.bilinear,
):

    with rasterio.open(reference_path) as ref:

        reference_crs = ref.crs
        reference_transform = ref.transform
        reference_width = ref.width
        reference_height = ref.height

        reference_profile = ref.profile.copy()

    with rasterio.open(dem_path) as dem:

        source = dem.read(1).astype(np.float32)
        source_nodata = dem.nodata

        destination = np.full(
            (reference_height, reference_width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=dem.transform,
            src_crs=dem.crs,
            src_nodata=source_nodata,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    valid_mask = np.isfinite(destination)

    return (
        destination,
        valid_mask,
        reference_profile,
    )