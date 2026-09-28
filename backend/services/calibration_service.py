"""Adapter for the existing calibration engine and local DEM providers."""

from pathlib import Path
import json
import re

import numpy as np
import rasterio
from rasterio.warp import transform_bounds


_SRTM_TILE_NAME = re.compile(r"^[ns]\d{2}_[ew]\d{3}")


def _provider_matches(path, provider):
    name = path.name.lower()
    if provider in {None, "", "auto"}:
        return any(token in name for token in ("srtm", "aw3d", "glo", "copernicus", "dem")) or bool(_SRTM_TILE_NAME.match(name))
    if provider == "srtm" and _SRTM_TILE_NAME.match(name):
        return True
    return {"srtm": "srtm", "aw3d30": "aw3d", "glo30": "glo", "copdem": "copernicus"}.get(provider, provider) in name


def _covers(candidate, reference):
    with rasterio.open(reference) as ref, rasterio.open(candidate) as dem:
        if not ref.crs or not dem.crs:
            return False
        bounds = transform_bounds(ref.crs, dem.crs, *ref.bounds, densify_pts=21)
        return not (bounds[2] <= dem.bounds.left or bounds[0] >= dem.bounds.right or bounds[3] <= dem.bounds.bottom or bounds[1] >= dem.bounds.top)


def resolve_dem(reference_path, provider="auto", explicit_path=None, project_root=None):
    """Find a local DEM with real spatial overlap; never fabricates elevation."""
    reference_path = Path(reference_path)
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    root = Path(project_root or reference_path.parents[2])
    calibration_dem = root / "calibration" / "dem"
    if calibration_dem.exists():
        candidates.extend(calibration_dem.rglob("*.tif"))
        candidates.extend(calibration_dem.rglob("*.tiff"))
    dataset_root = root / "datasets"
    if dataset_root.exists():
        for pattern in ("COPDEM_*/*.tif", "COPDEM_*/*.tiff", "**/*AW3D*.tif", "**/*AW3D*.tiff", "**/*GLO*.tif", "**/*GLO*.tiff"):
            candidates.extend(dataset_root.glob(pattern))
    explicit_candidates = candidates[:1] if explicit_path else []
    ordered = list(explicit_candidates)
    for preferred in ("srtm", "aw3d30", "glo30", "auto"):
        ordered.extend(path for path in candidates if path not in ordered and (preferred == "auto" or _provider_matches(path, preferred)))
    for candidate in ordered:
        if not candidate.exists() or candidate.resolve() == reference_path.resolve():
            continue
        if provider not in {"auto", ""} and not _provider_matches(candidate, provider):
            continue
        try:
            if _covers(candidate, reference_path):
                return candidate
        except (OSError, rasterio.errors.RasterioIOError, ValueError):
            continue
    return None


def run_calibration(
    ndsm_path,
    dem_path,
    reference_path,
    output_path,
    gcp_path=None,
    metadata_path=None,
    progress_callback=None,
    ndsm_weight=1.0,
):
    from calibration.calibration_engine import run_calibration as engine_run_calibration
    return engine_run_calibration(
        ndsm_path,
        dem_path,
        reference_path,
        output_path,
        gcp_path,
        metadata_path,
        progress_callback,
        ndsm_weight=ndsm_weight,
    )


def run_gcp_only_calibration(prediction_tif, reference_path, output_path, gcp_path, metadata_path, progress_callback=None):
    """Calibrate a georeferenced relative surface directly from GCP elevations.

    This path is intentionally explicit: it is used only when at least three
    valid GCPs are supplied and no DEM is available.  It does not invent a
    regional affine constant and records the result as GCP-derived metric data.
    """
    from calibration.vertical_calibrator import calibrate_dsm_with_gcps

    prediction_tif, reference_path, output_path, gcp_path, metadata_path = map(
        Path, (prediction_tif, reference_path, output_path, gcp_path, metadata_path)
    )
    with rasterio.open(prediction_tif) as src:
        values = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    valid = np.isfinite(values) & (values > -9990)
    if not crs:
        raise ValueError("GCP-only metric calibration requires a georeferenced reference raster.")
    if progress_callback:
        progress_callback({"stage": "calibrate", "fraction": 0.35, "sub": "applying GCP-only vertical calibration"})
    calibrated, info = calibrate_dsm_with_gcps(values, valid, transform, gcp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0, compress="deflate", predictor=3)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.where(valid, calibrated, -9999.0).astype(np.float32), 1)
        dst.set_band_description(1, "ASTERRA GCP-calibrated DSM (m)")
    valid_values = calibrated[valid]
    metadata = {
        "system": "ASTERRA AI",
        "product": "GCP-calibrated DSM",
        "is_metric": True,
        "metric_valid": True,
        "unit": "meters",
        "crs": str(crs),
        "vertical_reference": None,
        "calibration": {"available": True, "method": "GCP affine vertical calibration", **info},
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "source": str(prediction_tif),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback({"stage": "calibrate", "fraction": 1.0, "sub": "GCP-calibrated metric DSM written"})
    return {
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "stats": {
            "min": float(valid_values.min()),
            "max": float(valid_values.max()),
            "mean": float(valid_values.mean()),
            "std": float(valid_values.std()),
            "valid_pixels": int(valid.sum()),
            "total_pixels": int(valid.size),
        },
        "crs": str(crs),
        "width": int(profile["width"]),
        "height": int(profile["height"]),
    }


def write_relative_dsm(prediction_tif, output_path, metadata_path):
    prediction_tif, output_path, metadata_path = map(Path, (prediction_tif, output_path, metadata_path))
    with rasterio.open(prediction_tif) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="float32", nodata=-9999.0, compress="deflate", predictor=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "ASTERRA relative nDSM")
    valid = data[(data > -9990) & np.isfinite(data)]
    metadata = {
        "system": "ASTERRA AI", "product": "Relative DSM", "is_metric": False, "metric_valid": False,
        "unit": "relative", "crs": None if profile.get("crs") is None else str(profile["crs"]),
        "vertical_reference": None, "calibration": {"available": False, "method": "relative nDSM; no georeferenced DEM"},
        "width": int(profile["width"]), "height": int(profile["height"]), "valid_pixels": int(valid.size),
        "total_pixels": int(data.size), "source": str(prediction_tif),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output_path": str(output_path), "metadata_path": str(metadata_path), "metadata": metadata, "stats": {"min": float(valid.min()), "max": float(valid.max()), "mean": float(valid.mean()), "valid_pixels": int(valid.size), "total_pixels": int(data.size)} if valid.size else {"valid_pixels": 0, "total_pixels": int(data.size)}, "crs": metadata["crs"], "width": metadata["width"], "height": metadata["height"]}
