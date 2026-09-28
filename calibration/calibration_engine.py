"""ASTERRA DEM-assisted calibration engine.

The public :func:`run_calibration` function is shared by the CLI and the API
worker. The numerical implementation remains the existing ASTERRA pipeline:
``DSM = DTM + nDSM`` followed by optional GCP affine calibration.
"""

import argparse
from pathlib import Path
import sys
from typing import Callable, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling

CALIBRATION_DIR = Path(__file__).resolve().parent
if str(CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_DIR))

from dem_processor import load_dem  # noqa: E402
from grid_aligner import align_dem_to_reference  # noqa: E402
from dsm_builder import build_dsm  # noqa: E402
from vertical_calibrator import calibrate_dsm_with_gcps  # noqa: E402
from metadata import create_calibration_metadata, save_metadata  # noqa: E402


def run_calibration(
    ndsm_path,
    dem_path,
    reference_path,
    output_path,
    gcp_path=None,
    metadata_path=None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    ndsm_weight=1.0,
):
    """Create a metric DSM GeoTIFF from an nDSM and an aligned DEM."""
    ndsm_path, dem_path, reference_path, output_path = map(
        Path, (ndsm_path, dem_path, reference_path, output_path)
    )
    gcp_path = Path(gcp_path) if gcp_path else None
    metadata_path = Path(metadata_path) if metadata_path else output_path.with_suffix(".json")
    for path, name in ((ndsm_path, "nDSM"), (dem_path, "DEM"), (reference_path, "reference")):
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
    if gcp_path and not gcp_path.exists():
        raise FileNotFoundError(f"GCP file not found: {gcp_path}")

    def progress(stage, fraction, sub):
        if progress_callback:
            progress_callback({"stage": stage, "fraction": float(fraction), "sub": sub})

    progress("calibrate", 0.05, "loading ASTERRA nDSM")
    ndsm = np.load(ndsm_path).astype(np.float32)
    ndsm_mask = np.isfinite(ndsm)

    progress("calibrate", 0.2, "loading DEM")
    _, _, dem_metadata = load_dem(dem_path)
    progress("calibrate", 0.35, "aligning DEM to reference grid")
    aligned_dem, aligned_dem_mask, reference_profile = align_dem_to_reference(
        dem_path=dem_path,
        reference_path=reference_path,
        resampling=Resampling.bilinear,
    )
    if ndsm.shape != aligned_dem.shape:
        raise ValueError(
            "ASTERRA nDSM shape does not match the aligned DEM/reference grid. "
            f"nDSM shape={ndsm.shape}, DEM shape={aligned_dem.shape}"
        )
    if not reference_profile.get("crs"):
        raise ValueError("Reference raster must have a CRS for metric calibration.")

    progress("calibrate", 0.5, "building DSM = DTM + nDSM")
    try:
        ndsm_weight = float(ndsm_weight)
    except (TypeError, ValueError):
        raise ValueError("nDSM weight must be a finite number between 0 and 1.")
    if not np.isfinite(ndsm_weight) or not 0.0 <= ndsm_weight <= 1.0:
        raise ValueError("nDSM weight must be a finite number between 0 and 1.")
    weighted_ndsm = ndsm * ndsm_weight
    dsm, valid_mask = build_dsm(
        ndsm=weighted_ndsm,
        dtm=aligned_dem,
        ndsm_mask=ndsm_mask,
        dtm_mask=aligned_dem_mask,
    )
    valid_pixels = int(valid_mask.sum())
    total_pixels = int(valid_mask.size)
    if valid_pixels == 0:
        raise ValueError("DEM and nDSM do not overlap on any valid pixels.")

    calibration_info = {
        "enabled": False,
        "method": "DEM-assisted DSM reconstruction",
        "formula": "DSM = DTM + nDSM",
        "scale": None,
        "offset": None,
        "gcp_count": 0,
        "ndsm_weight": ndsm_weight,
        "surface_mode": "terrain-dem" if ndsm_weight == 0.0 else "dsm",
    }
    if gcp_path:
        progress("calibrate", 0.7, "applying GCP affine vertical calibration")
        dsm, calibration_info = calibrate_dsm_with_gcps(
            dsm=dsm,
            valid_mask=valid_mask,
            transform=reference_profile["transform"],
            gcp_path=gcp_path,
        )
    else:
        progress("calibrate", 0.75, "no GCP affine correction requested")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff", dtype="float32", count=1, nodata=-9999.0,
        compress="deflate", predictor=3,
    )
    output_array = np.where(valid_mask, dsm, -9999.0).astype(np.float32)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_array, 1)
        dst.set_band_description(1, "ASTERRA Metric DSM")

    calibration_metadata = create_calibration_metadata(
        ndsm_path=ndsm_path,
        dem_path=dem_path,
        reference_path=reference_path,
        output_path=output_path,
        dem_metadata=dem_metadata,
        reference_profile=reference_profile,
        valid_pixels=valid_pixels,
        total_pixels=total_pixels,
        calibration_info=calibration_info,
    )
    calibration_metadata.update({
        "is_metric": True,
        "metric_valid": True,
        "unit": "meters",
        "source_dem": str(dem_path),
    })
    save_metadata(calibration_metadata, metadata_path)
    progress("calibrate", 1.0, "metric DSM written")
    return {
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": calibration_metadata,
        "stats": {
            "min": float(np.nanmin(dsm)),
            "max": float(np.nanmax(dsm)),
            "mean": float(np.nanmean(dsm)),
            "std": float(np.nanstd(dsm)),
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
        },
        "crs": str(reference_profile["crs"]),
        "width": int(reference_profile["width"]),
        "height": int(reference_profile["height"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ASTERRA DEM Calibration Engine")
    parser.add_argument("--ndsm", required=True)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gcp", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or CALIBRATION_DIR / "outputs" / "metric_dsm.tif"
    result = run_calibration(args.ndsm, args.dem, args.reference, output, args.gcp)
    print(f"Metric DSM: {result['output_path']}")
    print(f"Metadata: {result['metadata_path']}")


if __name__ == "__main__":
    main()
