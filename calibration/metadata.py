from datetime import datetime, timezone
from pathlib import Path
import json


def create_calibration_metadata(
    ndsm_path,
    dem_path,
    reference_path,
    output_path,
    dem_metadata,
    reference_profile,
    valid_pixels,
    total_pixels,
    calibration_info,
):

    transform = reference_profile.get("transform")

    if transform is not None:
        resolution = [
            abs(float(transform.a)),
            abs(float(transform.e)),
        ]
    else:
        resolution = None

    metadata = {
        "system": "ASTERRA AI",

        "product": "Metric DSM",

        "calibration_method": (
            calibration_info.get(
                "method",
                "DEM-assisted DSM reconstruction",
            )
        ),

        "formula": calibration_info.get(
            "formula",
            "DSM = DTM + nDSM",
        ),

        # Surface selection is explicit because a coarse satellite scene may
        # be intentionally rendered as DEM-backed terrain instead of claiming
        # false building detail from interpolated pixels.
        "ndsm_weight": calibration_info.get("ndsm_weight", 1.0),
        "surface_mode": calibration_info.get("surface_mode", "dsm"),

        "vertical_calibration_enabled": bool(
            calibration_info.get(
                "enabled",
                False,
            )
        ),

        "vertical_calibration_method": (
            calibration_info.get(
                "method"
            )
        ),

        "vertical_calibration_scale": (
            calibration_info.get(
                "scale"
            )
        ),

        "vertical_calibration_offset": (
            calibration_info.get(
                "offset"
            )
        ),

        "gcp_count": int(
            calibration_info.get(
                "gcp_count",
                0,
            )
        ),

        "gcp_source": calibration_info.get(
            "gcp_source"
        ),

        "gcp_mae_m": calibration_info.get(
            "gcp_mae_m"
        ),

        "gcp_rmse_m": calibration_info.get(
            "gcp_rmse_m"
        ),

        "gcp_bias_m": calibration_info.get(
            "gcp_bias_m"
        ),

        "gcp_r2": calibration_info.get(
            "gcp_r2"
        ),

        "sampling_method": calibration_info.get(
            "sampling_method"
        ),

        "vertical_reference": (
            "Defined by supplied GCP elevations"
            if calibration_info.get("enabled")
            else "Inherited from source DEM"
        ),

        "ndsm_source": str(
            Path(ndsm_path)
        ),

        "dem_source": str(
            Path(dem_path)
        ),

        "reference_source": str(
            Path(reference_path)
        ),

        "output": str(
            Path(output_path)
        ),

        "crs": (
            str(reference_profile["crs"])
            if reference_profile.get("crs")
            else None
        ),

        "resolution": resolution,

        "width": reference_profile.get(
            "width"
        ),

        "height": reference_profile.get(
            "height"
        ),

        "valid_pixels": int(
            valid_pixels
        ),

        "total_pixels": int(
            total_pixels
        ),

        "valid_percentage": (
            float(valid_pixels / total_pixels * 100)
            if total_pixels > 0
            else 0.0
        ),

        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    return metadata


def save_metadata(
    metadata,
    output_path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )
