import numpy as np


def build_dsm(
    ndsm,
    dtm,
    ndsm_mask=None,
    dtm_mask=None,
):

    if ndsm.shape != dtm.shape:
        raise ValueError(
            f"nDSM shape {ndsm.shape} does not match "
            f"DTM shape {dtm.shape}"
        )

    valid = (
        np.isfinite(ndsm)
        & np.isfinite(dtm)
    )

    if ndsm_mask is not None:
        valid &= ndsm_mask

    if dtm_mask is not None:
        valid &= dtm_mask

    dsm = np.full(
        ndsm.shape,
        np.nan,
        dtype=np.float32,
    )

    dsm[valid] = (
        dtm[valid]
        + ndsm[valid]
    )

    return dsm, valid