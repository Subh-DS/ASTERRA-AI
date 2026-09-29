"""Build the visualization-only bare-earth surface.

The calibrated DSM is an input measurement and must remain immutable.  A DSM
also contains roof and canopy tops, however, so it is not the right surface
for the terrain mesh when those objects are exported as separate geometry.
This module writes a second GeoTIFF on the same grid and CRS.  Only accepted
feature footprints are flattened; all unclassified terrain keeps its original
relief.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _ring(feature):
    points = feature.get("polygon_projected") or []
    if len(points) < 3:
        return None
    clean = []
    for point in points:
        try:
            value = [float(point[0]), float(point[1])]
        except (TypeError, ValueError, IndexError):
            continue
        if not clean or value != clean[-1]:
            clean.append(value)
    if len(clean) < 3:
        return None
    if clean[0] != clean[-1]:
        clean.append(clean[0])
    return {"type": "Polygon", "coordinates": [clean]}


def _tree_circle(tree):
    point = tree.get("point_projected") or []
    if len(point) != 2:
        return None
    try:
        x, y = float(point[0]), float(point[1])
        radius = max(0.5, float(tree.get("canopy_radius", 2.0)))
    except (TypeError, ValueError):
        return None
    angles = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    ring = [[x + radius * float(np.cos(a)), y + radius * float(np.sin(a))] for a in angles]
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _road_ribbon(road):
    path = road.get("path_projected") or []
    if len(path) < 2:
        return None
    try:
        width = max(0.5, float(road.get("width_m", 4.0)))
    except (TypeError, ValueError):
        width = 4.0
    half = width * 0.5
    polygons = []
    for index in range(len(path) - 1):
        try:
            x1, y1 = map(float, path[index][:2])
            x2, y2 = map(float, path[index + 1][:2])
        except (TypeError, ValueError, IndexError):
            continue
        dx, dy = x2 - x1, y2 - y1
        length = max(float(np.hypot(dx, dy)), 1e-6)
        nx, ny = -dy / length * half, dx / length * half
        polygons.append([
            [x1 - nx, y1 - ny], [x1 + nx, y1 + ny],
            [x2 + nx, y2 + ny], [x2 - nx, y2 - ny],
            [x1 - nx, y1 - ny],
        ])
    if not polygons:
        return None
    return {"type": "MultiPolygon", "coordinates": [[[point for point in polygon]] for polygon in polygons]}


def _candidate_features(buildings, environment):
    environment = environment or {}
    # Apply contextual masks first and buildings last. This is the raster
    # equivalent of the scene priority: accepted buildings win over a sports
    # ground/land-cover polygon at shared pixels.
    for layer in ("trees", "landcover", "roads", "water", "exclusion_zones"):
        for feature in environment.get(layer) or []:
            geometry = _tree_circle(feature) if layer == "trees" else (_road_ribbon(feature) if layer == "roads" else _ring(feature))
            if geometry is None:
                continue
            ground = _finite(feature.get("ground_elevation"))
            if ground is not None:
                yield ("sports_ground" if layer == "exclusion_zones" else layer), geometry, ground
    for feature in buildings or []:
        geometry = _ring(feature)
        ground = _finite(feature.get("ground_elevation"))
        if geometry is not None and ground is not None:
            yield "buildings", geometry, ground


def build_terrain_surface(dsm_path, output_path, buildings=None, environment=None):
    """Persist a same-grid bare-earth visualization surface.

    Returns provenance suitable for reconstruction metadata.  The source DSM
    is opened read-only and is never rewritten.
    """
    dsm_path = Path(dsm_path)
    output_path = Path(output_path)
    with rasterio.open(dsm_path) as src:
        raw = src.read(1, masked=True).astype(np.float32)
        raw_mask = np.ma.getmaskarray(raw).copy()
        data = raw.filled(np.nan).astype(np.float32)
        raw_mask |= ~np.isfinite(data)
        profile = src.profile.copy()
        transform = src.transform
        shape = data.shape

    layer_counts = {"buildings": 0, "sports_ground": 0, "roads": 0, "water": 0, "landcover": 0, "trees": 0}
    flattened_pixels = 0
    for layer, geometry, ground in _candidate_features(buildings, environment):
        try:
            replacement = rasterize(
                [(geometry, np.float32(ground))],
                out_shape=shape,
                transform=transform,
                fill=np.nan,
                dtype="float32",
                all_touched=True,
            )
        except (TypeError, ValueError, rasterio.errors.RasterioIOError):
            continue
        selected = np.isfinite(replacement) & ~raw_mask
        if selected.any():
            data[selected] = replacement[selected]
            flattened_pixels += int(selected.sum())
            layer_counts[layer] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
        tiled=False,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.where(raw_mask, -9999.0, data).astype(np.float32), 1)
        dst.update_tags(
            ASTERRA_SURFACE="visualization-only-bare-earth",
            ASTERRA_SOURCE_DSM=str(dsm_path),
            ASTERRA_RAW_DSM_UNCHANGED="true",
        )

    return {
        "path": str(output_path),
        "source": "calibrated-dsm-with-accepted-feature-footprints",
        "same_grid_as_dsm": True,
        "raw_dsm_unchanged": True,
        "flattened_pixels": flattened_pixels,
        "layer_counts": layer_counts,
        "priority": ["buildings", "sports_ground", "water", "roads", "landcover", "trees", "bare_terrain"],
    }
