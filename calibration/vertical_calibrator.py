from pathlib import Path
import csv

import numpy as np
from rasterio.transform import rowcol


def load_gcp_csv(gcp_path):
    """
    Load GCP CSV.

    Required columns:
        x
        y
        elevation

    Optional:
        id

    X/Y coordinates must use the output/reference CRS.
    Elevation must be in meters.
    """

    gcp_path = Path(gcp_path)

    if not gcp_path.exists():
        raise FileNotFoundError(
            f"GCP file not found: {gcp_path}"
        )

    gcps = []

    with open(gcp_path, "r", newline="", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "GCP CSV has no header."
            )

        fields = {
            field.strip().lower()
            for field in reader.fieldnames
        }

        required = {"x", "y", "elevation"}

        missing = required - fields

        if missing:
            raise ValueError(
                f"GCP CSV missing required columns: {sorted(missing)}"
            )

        for index, row in enumerate(reader, start=1):

            try:
                x = float(row["x"])
                y = float(row["y"])
                elevation = float(row["elevation"])
            except Exception as exc:
                raise ValueError(
                    f"Invalid GCP on CSV row {index}: {row}"
                ) from exc

            if not (
                np.isfinite(x)
                and np.isfinite(y)
                and np.isfinite(elevation)
            ):
                raise ValueError(
                    f"Non-finite GCP on CSV row {index}"
                )

            gcp_id = row.get("id", str(index))

            gcps.append(
                {
                    "id": gcp_id,
                    "x": x,
                    "y": y,
                    "elevation": elevation,
                }
            )

    if len(gcps) < 3:
        raise ValueError(
            "At least 3 valid GCPs are required "
            "for affine vertical calibration."
        )

    return gcps


def sample_dsm_at_gcps(
    dsm,
    valid_mask,
    transform,
    gcps,
):
    """
    Sample the in-memory DSM at GCP locations.

    Nearest-pixel sampling is used.
    """

    predicted = []
    reference = []
    used_gcps = []

    height, width = dsm.shape

    for gcp in gcps:

        row, col = rowcol(
            transform,
            gcp["x"],
            gcp["y"],
            op=round,
        )

        if (
            row < 0
            or row >= height
            or col < 0
            or col >= width
        ):
            continue

        if not valid_mask[row, col]:
            continue

        value = dsm[row, col]

        if not np.isfinite(value):
            continue

        predicted.append(
            float(value)
        )

        reference.append(
            float(gcp["elevation"])
        )

        used_gcps.append(gcp)

    if len(predicted) < 3:
        raise ValueError(
            "Fewer than 3 valid GCPs could be sampled "
            "from the DSM."
        )

    return (
        np.asarray(predicted, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        used_gcps,
    )


def fit_affine_calibration(
    predicted,
    reference,
):
    """
    Fit:

        Z_reference = a * Z_predicted + b
    """

    if predicted.size < 3:
        raise ValueError(
            "At least 3 GCP observations are required."
        )

    if np.ptp(predicted) < 1e-8:
        raise ValueError(
            "GCP predicted elevations have insufficient "
            "variation to estimate an affine scale."
        )

    A = np.column_stack(
        [
            predicted,
            np.ones_like(predicted),
        ]
    )

    coefficients, _, rank, _ = np.linalg.lstsq(
        A,
        reference,
        rcond=None,
    )

    if rank < 2:
        raise ValueError(
            "Affine calibration is underdetermined."
        )

    scale = float(coefficients[0])
    offset = float(coefficients[1])

    if not np.isfinite(scale) or not np.isfinite(offset):
        raise ValueError(
            "Invalid affine calibration parameters."
        )

    if scale <= 0:
        raise ValueError(
            f"Estimated calibration scale is {scale:.6f}. "
            "Expected a positive vertical scale. "
            "Check the GCP coordinates/elevations."
        )

    fitted = (
        scale * predicted
        + offset
    )

    residuals = fitted - reference

    mae = float(
        np.mean(np.abs(residuals))
    )

    rmse = float(
        np.sqrt(np.mean(residuals ** 2))
    )

    bias = float(
        np.mean(residuals)
    )

    if np.std(reference) > 1e-12:

        ss_res = np.sum(
            (reference - fitted) ** 2
        )

        ss_tot = np.sum(
            (reference - np.mean(reference)) ** 2
        )

        r2 = float(
            1.0 - ss_res / ss_tot
        )

    else:
        r2 = None

    return {
        "scale": scale,
        "offset": offset,
        "gcp_count": int(predicted.size),
        "gcp_mae_m": mae,
        "gcp_rmse_m": rmse,
        "gcp_bias_m": bias,
        "gcp_r2": r2,
    }


def calibrate_dsm_with_gcps(
    dsm,
    valid_mask,
    transform,
    gcp_path,
):
    """
    Perform affine vertical calibration directly
    on the in-memory DSM.

    Formula:

        Z_calibrated = a * Z_raw + b
    """

    gcps = load_gcp_csv(
        gcp_path
    )

    predicted, reference, used_gcps = sample_dsm_at_gcps(
        dsm=dsm,
        valid_mask=valid_mask,
        transform=transform,
        gcps=gcps,
    )

    calibration_info = fit_affine_calibration(
        predicted=predicted,
        reference=reference,
    )

    scale = calibration_info["scale"]
    offset = calibration_info["offset"]

    calibrated = dsm.copy()

    calibrated[valid_mask] = (
        scale * dsm[valid_mask]
        + offset
    )

    calibration_info.update(
        {
            "enabled": True,
            "method": "GCP affine vertical calibration",
            "formula": "Z_calibrated = a * Z_raw + b",
            "gcp_source": str(
                Path(gcp_path)
            ),
            "sampled_gcp_ids": [
                gcp["id"]
                for gcp in used_gcps
            ],
            "sampling_method": "nearest pixel",
        }
    )

    return (
        calibrated.astype(np.float32),
        calibration_info,
    )