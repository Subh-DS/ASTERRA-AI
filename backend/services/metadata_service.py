"""Raster ingestion, metadata, GCP, and browser-preview utilities."""

from pathlib import Path
import csv
import json

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.control import GroundControlPoint
from rasterio.coords import BoundingBox
from rasterio.transform import from_gcps, Affine
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform as transform_coords


ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def validate_upload_signature(path, extension=None):
    path = Path(path)
    suffix = (extension or path.suffix).lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Only GeoTIFF, TIFF, PNG, and JPEG uploads are supported.")
    header = path.read_bytes()[:16]
    valid = (suffix in {".tif", ".tiff"} and (header[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"})) or (suffix == ".png" and header.startswith(b"\x89PNG\r\n\x1a\n")) or (suffix in {".jpg", ".jpeg"} and header.startswith(b"\xff\xd8\xff"))
    if not valid:
        raise ValueError("The uploaded file signature does not match its extension.")


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def normalize_gcps(gcps):
    normalized = []
    for item in gcps or []:
        lat, lon = _number(item.get("lat")), _number(item.get("lon"))
        elevation = _number(item.get("elev", item.get("elevation")))
        x, y = _number(item.get("x")), _number(item.get("y"))
        pixel_x = _number(item.get("pixel_x", item.get("col")))
        pixel_y = _number(item.get("pixel_y", item.get("row")))
        if elevation is None:
            continue
        if lat is not None and lon is not None:
            if abs(lat) > 90 or abs(lon) > 180:
                raise ValueError("GCP latitude/longitude is outside valid bounds.")
            x, y = lon, lat
        if x is None or y is None:
            raise ValueError("Each GCP needs x/y or lon/lat coordinates.")
        normalized.append({"x": x, "y": y, "elevation": elevation, "pixel_x": pixel_x, "pixel_y": pixel_y})
    return normalized


def _gcps_for_georeference(gcps):
    points = []
    for item in gcps:
        if item["pixel_x"] is None or item["pixel_y"] is None:
            continue
        points.append(GroundControlPoint(row=item["pixel_y"], col=item["pixel_x"], x=item["x"], y=item["y"], z=item["elevation"]))
    return points


def _normalize_rgb_uint8(data, valid_mask=None):
    """Convert arbitrary RGB imagery to stable 8-bit display/model input.

    ASTERRA's RGB encoder expects normalized 8-bit-like imagery. Sentinel and
    other scientific GeoTIFFs commonly arrive as uint16 reflectance values;
    clipping those values to 255 destroys almost all contrast. Per-band robust
    percentiles preserve the scene's visual structure without changing the
    depth model or inventing elevation.
    """
    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError(f"Expected three RGB bands, got {data.shape}")
    array = data.astype(np.float32, copy=False)
    mask = np.ones(array.shape[1:], dtype=bool) if valid_mask is None else valid_mask > 0
    result = np.zeros(array.shape, dtype=np.uint8)
    for band in range(3):
        values = array[band]
        finite = mask & np.isfinite(values)
        if not finite.any():
            continue
        valid = values[finite]
        low_value = float(valid.min())
        high_value = float(valid.max())
        if high_value <= 1.0 and low_value >= 0.0:
            low, high = 0.0, 1.0
        elif high_value <= 255.0 and low_value >= 0.0:
            low, high = 0.0, 255.0
        else:
            low, high = np.percentile(valid, (2.0, 98.0)).astype(float)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = low_value, high_value
        scaled = (values - low) * 255.0 / max(high - low, 1e-6)
        result[band] = np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 255.0).astype(np.uint8)
    result[:, ~mask] = 0
    return result


def _metric_crs_for_reference(crs, bounds):
    """Choose the local UTM CRS for a geographic raster."""
    source_crs = CRS.from_user_input(crs)
    if source_crs.is_projected:
        # Rasterio CRS versions do not all expose pyproj's ``axis_info``.
        # ``linear_units`` is the stable rasterio property and is sufficient
        # for deciding whether an existing projected CRS is already metric.
        units = {str(getattr(source_crs, "linear_units", "") or "").lower()}
        if not units or units.intersection({"metre", "meter", "metres", "meters", "m"}):
            return source_crs
    center_x = (bounds.left + bounds.right) * 0.5
    center_y = (bounds.bottom + bounds.top) * 0.5
    lon, lat = transform_coords(source_crs, CRS.from_epsg(4326), [center_x], [center_y])
    longitude, latitude = float(lon[0]), float(lat[0])
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        return None
    zone = max(1, min(60, int((longitude + 180.0) // 6.0) + 1))
    epsg = (32600 if latitude >= 0.0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _write_raster(output_path, data, profile, valid_mask):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
        dst.write_mask(np.where(valid_mask, 255, 0).astype(np.uint8))


def materialize_input(
    source_path,
    output_path,
    gcps=None,
    max_pixels=100_000_000,
    original_name=None,
    min_dimension=0,
):
    """Normalize any supported upload to a three-band GeoTIFF.

    A plain image receives no CRS and remains explicitly relative. A valid set
    of image-pixel GCPs creates an EPSG:4326 affine reference grid. Geographic
    rasters are reprojected to local UTM before inference and mesh generation,
    so horizontal coordinates are meters rather than degrees.
    """
    source_path, output_path = Path(source_path), Path(output_path)
    suffix = Path(original_name or source_path).suffix.lower()
    validate_upload_signature(source_path, suffix)
    normalized_gcps = normalize_gcps(gcps)
    georef_gcps = _gcps_for_georeference(normalized_gcps)
    source_crs = None
    source_transform = None
    source_bounds = None
    if suffix in {".tif", ".tiff"}:
        with rasterio.open(source_path) as src:
            if src.width * src.height > max_pixels:
                raise ValueError(f"Raster exceeds configured pixel limit ({max_pixels:,}).")
            if src.count < 3:
                raise ValueError("ASTERRA expects an RGB image with at least three bands.")
            data = src.read([1, 2, 3])
            source_mask = src.read_masks(1)
            source_crs, source_transform = src.crs, src.transform
            if source_crs is None and len(georef_gcps) >= 3:
                source_crs, source_transform = CRS.from_epsg(4326), from_gcps(georef_gcps)
            crs, transform = source_crs, source_transform
            width, height = src.width, src.height
    else:
        from PIL import Image
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width * height > max_pixels:
                raise ValueError(f"Raster exceeds configured pixel limit ({max_pixels:,}).")
            data = np.transpose(np.asarray(image), (2, 0, 1))
            source_mask = np.full((height, width), 255, dtype=np.uint8)
            source_crs = CRS.from_epsg(4326) if len(georef_gcps) >= 3 else None
            source_transform = from_gcps(georef_gcps) if len(georef_gcps) >= 3 else Affine.identity()
            crs, transform = source_crs, source_transform
            source_bounds = rasterio.transform.array_bounds(height, width, transform)

    # Use the effective transform (including GCP-derived georeferencing), not
    # the upload's placeholder identity transform, when determining the UTM
    # zone and reprojection footprint.
    source_bounds = rasterio.transform.array_bounds(height, width, source_transform)

    if min_dimension and min(width, height) < min_dimension:
        raise ValueError(
            f"Input raster is {width}x{height}; at least {min_dimension} pixels on both axes "
            "are required for a useful ASTERRA 3D reconstruction. Upload the full-resolution "
            "RGB/GeoTIFF scene instead of a thumbnail or overview raster."
        )

    original_width, original_height = width, height
    normalized = _normalize_rgb_uint8(data, source_mask)
    source_bounds_obj = BoundingBox(*source_bounds)
    target_crs = _metric_crs_for_reference(crs, source_bounds_obj) if crs else None
    reprojected = bool(crs and target_crs and CRS.from_user_input(crs) != target_crs)

    if reprojected:
        transform, width, height = calculate_default_transform(
            crs, target_crs, width, height, *source_bounds
        )
        if width * height > max_pixels:
            raise ValueError(f"Reprojected raster exceeds configured pixel limit ({max_pixels:,}).")
        projected = np.zeros((3, height, width), dtype=np.uint8)
        projected_mask = np.zeros((height, width), dtype=np.uint8)
        for band in range(3):
            reproject(
                normalized[band], projected[band],
                src_transform=source_transform, src_crs=crs,
                dst_transform=transform, dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
        reproject(
            source_mask, projected_mask,
            src_transform=source_transform, src_crs=crs,
            dst_transform=transform, dst_crs=target_crs,
            src_nodata=0, dst_nodata=0, resampling=Resampling.nearest,
        )
        data, source_mask, crs = projected, projected_mask, target_crs
    else:
        data, crs = normalized, crs

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 3,
        "dtype": "uint8", "crs": crs, "transform": transform, "nodata": None,
        "compress": "deflate", "interleave": "pixel",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_raster(output_path, data, profile, source_mask)
    output_meta = inspect_raster(output_path)
    output_meta.update({
        "georeferenced": crs is not None,
        "gcps_used": len(georef_gcps) >= 3,
        "gcps": normalized_gcps,
        "source_crs": None if source_crs is None else str(source_crs),
        "reprojected_to_metric_crs": reprojected,
        "input_dtype_normalized": True,
        "source_width": int(original_width),
        "source_height": int(original_height),
    })
    return output_meta


def inspect_raster(path):
    with rasterio.open(path) as src:
        projected = bool(src.crs and src.crs.is_projected)
        return {
            "width": src.width, "height": src.height, "count": src.count,
            "crs": None if src.crs is None else str(src.crs),
            "transform": src.transform, "bounds": tuple(float(v) for v in src.bounds),
            "nodata": src.nodata, "resolution": tuple(float(v) for v in src.res),
            "pixel_size_m": tuple(float(v) for v in src.res) if projected else None,
            "projected_crs": projected,
        }


def write_gcp_csv(gcps, output_path, reference_crs=None):
    normalized = normalize_gcps(gcps)
    if len(normalized) < 3:
        return False
    xs, ys = [item["x"] for item in normalized], [item["y"] for item in normalized]
    if reference_crs and str(reference_crs) != "EPSG:4326":
        xs, ys = transform_coords("EPSG:4326", reference_crs, xs, ys)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x", "y", "elevation"])
        writer.writeheader()
        for index, (item, x, y) in enumerate(zip(normalized, xs, ys), 1):
            writer.writerow({"id": index, "x": x, "y": y, "elevation": item["elevation"]})
    return True


def write_preview_assets(source_path, dsm_path, texture_path, normal_path):
    from PIL import Image
    from visualization.glb_exporter import load_rgb_texture_image
    image = load_rgb_texture_image(source_path, texture_width=2048, texture_height=2048)
    image.save(texture_path, quality=92, optimize=True)
    with rasterio.open(dsm_path) as src:
        z = src.read(1).astype(np.float32)
        pixel_x, pixel_y = abs(float(src.res[0])), abs(float(src.res[1]))
    z[z <= -9990] = np.nan
    filled = np.nan_to_num(z, nan=float(np.nanmedian(z[np.isfinite(z)])) if np.isfinite(z).any() else 0.0)
    gy, gx = np.gradient(filled, max(pixel_y, 1e-6), max(pixel_x, 1e-6))
    nx, ny = -gx, -gy
    nz = np.ones_like(z)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack(((nx / norm + 1) * 127.5, (ny / norm + 1) * 127.5, (nz / norm + 1) * 127.5), axis=-1)
    Image.fromarray(np.clip(normal, 0, 255).astype(np.uint8), "RGB").save(normal_path)


def stats_from_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1, masked=True).astype(np.float32)
        values = array.compressed()
        if not values.size: return {"valid_pixels": 0, "total_pixels": int(array.size)}
        return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()), "valid_pixels": int(values.size), "total_pixels": int(array.size)}
