import numpy as np


def get_valid_mask(
    array,
    raster_mask=None,
    nodata=None,
):

    valid = np.isfinite(array)

    if nodata is not None:
        valid &= array != nodata

    if raster_mask is not None:
        valid &= raster_mask.astype(bool)

    return valid


def sanitize_nodata(
    array,
    valid_mask,
    fill_value=np.nan,
):

    output = array.astype(
        np.float32,
        copy=True,
    )

    output[~valid_mask] = fill_value

    return output


def combine_masks(*masks):

    if not masks:
        raise ValueError(
            "At least one mask is required."
        )

    result = masks[0].astype(bool).copy()

    for mask in masks[1:]:
        result &= mask.astype(bool)

    return result