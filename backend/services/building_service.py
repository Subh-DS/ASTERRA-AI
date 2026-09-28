"""Building footprint and massing enrichment for map reconstructions.

RGB depth alone is not a reliable building-segmentation source, especially
for 10 m Sentinel-2 scenes.  This service obtains actual OSM building
footprints, projects them into the reconstruction grid, and attaches a height
source to each footprint.  Tagged OSM heights/levels win; high-resolution
metric DSM samples are used when tags are absent.  Coarse scenes receive only
explicitly approximate context massing so the UI never presents it as a
measured building height.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes as raster_shapes
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.warp import transform as transform_coords

from ..segmentation.class_map import BUILDING


_HEIGHT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class BuildingProviderError(RuntimeError):
    """Expected, actionable building-provider failure."""


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _tag_number(value: Any) -> float | None:
    match = _HEIGHT_RE.search(str(value or ""))
    return _number(match.group(0)) if match else None


def _geometry_profile(tags: dict[str, Any]) -> str:
    """Choose a bounded architectural profile from explicit map evidence."""
    values = " ".join(
        str(tags.get(key) or "")
        for key in ("building", "name", "amenity", "religion", "building:use", "tower:type")
    ).lower()
    worship = str(tags.get("amenity") or "").lower() == "place_of_worship"
    religion = str(tags.get("religion") or "").lower()
    if any(token in values for token in (
        "temple", "mandir", "shrine", "pagoda", "gopuram", "church", "mosque", "synagogue",
    )) or (worship and religion in {"hindu", "buddhist", "jain", "sikh"}):
        return "temple"
    if any(token in values for token in ("tower", "minaret", "bell_tower")):
        return "towered"
    return "standard"


def _area(polygon: list[list[float]]) -> float:
    return abs(sum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
        for i in range(len(polygon))
    )) * 0.5


def _simplify_ring(points, tolerance=1.25):
    clean = []
    for point in points or []:
        p = [float(point[0]), float(point[1])]
        if not clean or abs(p[0] - clean[-1][0]) > 1e-6 or abs(p[1] - clean[-1][1]) > 1e-6:
            clean.append(p)
    if len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) < 1e-6 and abs(clean[0][1] - clean[-1][1]) < 1e-6:
        clean.pop()
    if len(clean) <= 4:
        return clean

    def distance(point, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return float(((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) ** 0.5)
        t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)
        t = min(1.0, max(0.0, t))
        x, y = start[0] + t * dx, start[1] + t * dy
        return float(((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5)

    def rdp(items):
        if len(items) <= 2:
            return items
        index, best = 0, 0.0
        for i in range(1, len(items) - 1):
            current = distance(items[i], items[0], items[-1])
            if current > best:
                index, best = i, current
        if best > tolerance:
            return rdp(items[:index + 1])[:-1] + rdp(items[index:])
        return [items[0], items[-1]]

    simplified = rdp(clean + [clean[0]])[:-1]
    return simplified if len(simplified) >= 3 else clean


def _clip_edge(points, axis, limit, keep_less):
    if not points:
        return []
    result = []

    def inside(point):
        return point[axis] <= limit if keep_less else point[axis] >= limit

    def intersection(start, end):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-9:
            return [float(start[0]), float(start[1])]
        t = (limit - start[axis]) / delta
        return [float(start[0] + (end[0] - start[0]) * t), float(start[1] + (end[1] - start[1]) * t)]

    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside != previous_inside:
            result.append(intersection(previous, current))
        if current_inside:
            result.append([float(current[0]), float(current[1])])
        previous, previous_inside = current, current_inside
    return result


def _clip_polygon(points, width, height):
    clipped = points
    for axis, limit, keep_less in (
        (0, float(width), True),
        (1, float(height), True),
        (0, 0.0, False),
        (1, 0.0, False),
    ):
        clipped = _clip_edge(clipped, axis, limit, keep_less)
    # Remove adjacent duplicate vertices introduced by clipping.
    clean = []
    for point in clipped:
        if not clean or abs(point[0] - clean[-1][0]) > 1e-6 or abs(point[1] - clean[-1][1]) > 1e-6:
            clean.append(point)
    if len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) < 1e-6 and abs(clean[0][1] - clean[-1][1]) < 1e-6:
        clean.pop()
    return clean


def _query_overpass(endpoint: str, aoi: dict[str, float], timeout: float) -> dict[str, Any]:
    query = (
        "[out:json][timeout:25];"
        "("
        f"way[\"building\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"way[\"amenity\"=\"place_of_worship\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"way[\"historic\"~\"^(temple|shrine|church|mosque|synagogue)$\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"node[\"amenity\"=\"place_of_worship\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"way[\"highway\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"way[\"waterway\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"way[\"natural\"~\"^(water|wood|scrub|heath|tree_row)$\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"way[\"landuse\"~\"^(forest|wood|grass|meadow|park|cemetery|allotments)$\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"way[\"leisure\"~\"^(park|garden|playground)$\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"node[\"natural\"=\"tree\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        ");"
        "out tags geom;"
    )
    endpoints = [candidate.strip() for candidate in str(endpoint).split(",") if candidate.strip()]
    if not endpoints:
        raise BuildingProviderError("no building footprint service endpoint is configured.")

    last_error: BuildingProviderError | None = None
    for candidate in endpoints:
        request = urllib.request.Request(
            candidate,
            data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "ASTERRA-AI/1.0 building-footprints",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = BuildingProviderError(f"building footprint service returned HTTP {exc.code} from {candidate}.")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = BuildingProviderError(f"building footprint service request failed for {candidate}: {exc}")

    if last_error is not None:
        if len(endpoints) > 1:
            raise BuildingProviderError(f"all building footprint services failed; last error: {last_error}") from last_error
        raise last_error
    raise BuildingProviderError("building footprint service returned no response.")


def _sample_heights(src, projected_polygon):
    geojson = {"type": "Polygon", "coordinates": [[list(point) for point in projected_polygon + [projected_polygon[0]]]]}
    try:
        min_x = min(point[0] for point in projected_polygon)
        min_y = min(point[1] for point in projected_polygon)
        max_x = max(point[0] for point in projected_polygon)
        max_y = max(point[1] for point in projected_polygon)
        pixel_x = max(abs(float(src.transform.a)), 1e-6)
        pixel_y = max(abs(float(src.transform.e)), 1e-6)
        window = window_from_bounds(
            min_x - pixel_x * 2.0,
            min_y - pixel_y * 2.0,
            max_x + pixel_x * 2.0,
            max_y + pixel_y * 2.0,
            transform=src.transform,
        ).round_offsets().round_lengths()
        full = Window(0, 0, src.width, src.height)
        window = window.intersection(full)
        if window.width < 1 or window.height < 1:
            return None, None
        data = src.read(1, window=window, masked=True)
        local_transform = src.window_transform(window)
        inside = geometry_mask(
            [geojson],
            out_shape=data.shape,
            transform=local_transform,
            invert=True,
            all_touched=True,
        )
    except (ValueError, rasterio.errors.RasterioIOError):
        return None, None
    raw = data.astype(np.float32).filled(np.nan)
    valid = np.isfinite(raw) & ~np.ma.getmaskarray(data) & (raw > -9990)
    values = raw[inside & valid]
    outside = raw[(~inside) & valid]
    if values.size < 3:
        return None, None
    # Sample the pixels around the footprint for ground. Sampling the
    # footprint crop itself would use roof elevations as the building base and
    # make every extruded building float above the terrain.
    ground = float(np.percentile(outside, 25)) if outside.size >= 3 else float(np.percentile(values, 10))
    roof = float(np.percentile(values, 90))
    return ground, roof


def fetch_buildings(
    aoi: dict[str, float],
    raster_path: Path,
    dsm_path: Path,
    *,
    endpoint: str,
    timeout: float = 20.0,
    max_buildings: int = 500,
    min_area_m2: float = 20.0,
    source_gsd_m: float | None = None,
    _payload: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return projected building massing and provider metadata.

    Provider failures are returned as metadata warnings rather than failing a
    reconstruction: terrain remains useful when OSM is unavailable.
    """
    if not endpoint:
        return [], {"enabled": False, "provider": "openstreetmap", "count": 0}
    if _payload is None:
        try:
            _payload = _query_overpass(endpoint, aoi, timeout)
        except BuildingProviderError as exc:
            return [], {"enabled": True, "provider": "openstreetmap", "count": 0, "warning": str(exc)}
    payload = _payload

    buildings: list[dict[str, Any]] = []
    seen_elements: set[tuple[str, int]] = set()
    skipped = 0
    with rasterio.open(raster_path) as raster, rasterio.open(dsm_path) as dsm:
        if not raster.crs or not dsm.crs:
            return [], {"enabled": True, "provider": "openstreetmap", "count": 0, "warning": "Building footprints require a projected raster CRS."}
        if str(raster.crs) != str(dsm.crs):
            return [], {"enabled": True, "provider": "openstreetmap", "count": 0, "warning": "Building and DSM rasters use different CRSs."}
        for element in payload.get("elements") or []:
            if len(buildings) >= max_buildings:
                break
            tags = element.get("tags") or {}
            is_worship_footprint = str(tags.get("amenity") or "").lower() == "place_of_worship"
            is_historic_worship = str(tags.get("historic") or "").lower() in {
                "temple", "shrine", "church", "mosque", "synagogue",
            }
            if (
                (not tags.get("building") or tags.get("building") == "no")
                and not is_worship_footprint
                and not is_historic_worship
            ):
                continue
            element_key = (str(element.get("type") or "way"), int(element.get("id") or 0))
            if element_key in seen_elements:
                continue
            seen_elements.add(element_key)
            geometry = element.get("geometry") or []
            is_worship_node = str(element.get("type") or "").lower() == "node" and is_worship_footprint
            if is_worship_node:
                # A place-of-worship node has no mapped polygon. Create a
                # small, AOI-clipped footprint centered on the node so the
                # temple still becomes an isolated, explicitly approximate
                # object instead of being lost in the terrain surface.
                lon, lat = _number(element.get("lon")), _number(element.get("lat"))
                if lon is None or lat is None:
                    skipped += 1
                    continue
                xs, ys = transform_coords("EPSG:4326", raster.crs, [lon], [lat])
                cx, cy = float(xs[0]), float(ys[0])
                half_size = max(6.0, min(14.0, math.sqrt(max(min_area_m2, 36.0))))
                projected = [
                    [cx - half_size, cy - half_size], [cx + half_size, cy - half_size],
                    [cx + half_size, cy + half_size], [cx - half_size, cy + half_size],
                ]
            else:
                if len(geometry) < 3:
                    skipped += 1
                    continue
                lons = [_number(point.get("lon")) for point in geometry]
                lats = [_number(point.get("lat")) for point in geometry]
                if any(value is None for value in (*lons, *lats)):
                    skipped += 1
                    continue
                xs, ys = transform_coords("EPSG:4326", raster.crs, lons, lats)
                projected = [[float(x), float(y)] for x, y in zip(xs, ys)]
                if projected[0] != projected[-1]:
                    projected.append(projected[0])
            inv = ~raster.transform
            pixels = []
            for point in projected[:-1]:
                col, row = inv * (point[0], point[1])
                pixels.append([float(col), float(row)])
            pixels = _clip_polygon(pixels, raster.width - 1, raster.height - 1)
            if len(pixels) < 3 or _area(pixels) * abs(raster.res[0] * raster.res[1]) < min_area_m2:
                skipped += 1
                continue
            projected_clipped = []
            for col, row in pixels:
                x, y = raster.transform * (col, row)
                projected_clipped.append([float(x), float(y)])
            ground, roof_sample = _sample_heights(dsm, projected_clipped)
            if ground is None:
                skipped += 1
                continue

            profile = _geometry_profile(tags)
            explicit_height = _tag_number(tags.get("height"))
            levels = _tag_number(tags.get("building:levels") or tags.get("building:levels:aboveground"))
            if explicit_height is not None and explicit_height > 0:
                height = explicit_height
                height_source = "osm:height"
            elif levels is not None and levels > 0:
                height = levels * 3.0
                height_source = "osm:levels"
            elif source_gsd_m is not None and source_gsd_m <= 5 and roof_sample is not None:
                height = max(2.5, min(45.0, roof_sample - ground))
                height_source = "metric-dsm:sampled"
            elif profile == "temple":
                # A tagged worship footprint is useful evidence even when a
                # survey height is absent. Give the procedural roof a bounded
                # prior and label it approximate instead of reducing the
                # temple to the generic 3 m context block.
                height = 9.0
                height_source = "osm:approximate-temple-prior"
            else:
                # A visible context block is useful for coarse scenes, but it
                # is explicitly approximate and never reported as measured.
                height = 3.0
                height_source = "osm:approximate-context"

            if not math.isfinite(height) or height <= 0:
                skipped += 1
                continue
            element_id = int(element.get("id") or len(buildings) + 1)
            buildings.append({
                "id": element_id,
                "name": tags.get("name") or f"Building {element_id}",
                "polygon": [[float(col), float(row)] for col, row in pixels],
                "polygon_projected": projected_clipped,
                "area_m2": float(_area(projected_clipped)),
                "ground_elevation": float(ground),
                "height": float(height),
                "roof_elevation": float(ground + height),
                "height_source": height_source,
                "geometry_profile": profile,
                "profile_source": "osm-tags" if profile != "standard" else "osm-building-footprint",
                "roof_surface_source": "pending-dsm-quality-gate",
                "roof_valid_fraction": 0.0,
                "geometry_quality": "candidate",
                "confidence": 0.95 if explicit_height is not None or levels is not None else (0.72 if profile == "temple" else 0.62),
                "source": "openstreetmap",
                "osm_tags": {
                    key: value for key, value in tags.items()
                    if key in {
                        "building", "building:part", "height", "building:levels",
                        "building:levels:aboveground", "name", "amenity", "religion", "historic",
                    }
                },
            })

    approximate = sum(item["height_source"] == "osm:approximate-context" for item in buildings)
    metadata = {
        "enabled": True,
        "provider": "openstreetmap",
        "attribution": "© OpenStreetMap contributors",
        "count": len(buildings),
        "skipped": skipped,
        "approximate_count": approximate,
        "warning": "Some building heights are approximate context massing." if approximate else None,
    }
    return buildings, metadata


def _sample_point(src, projected_point):
    """Read a conservative local ground value for a point feature."""
    try:
        inv = ~src.transform
        col, row = inv * (projected_point[0], projected_point[1])
        col = int(round(col))
        row = int(round(row))
        if col < 0 or row < 0 or col >= src.width or row >= src.height:
            return None
        sample = src.read(1, window=Window(col, row, 1, 1), masked=True)
        if sample.mask.all():
            return None
        value = float(sample[0, 0])
        return value if math.isfinite(value) and value > -9990 else None
    except (IndexError, TypeError, ValueError, rasterio.errors.RasterioIOError):
        return None


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Dependency-free local mean for the visual building fallback."""
    radius = max(1, int(radius))
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / float(size * size)


def _visual_buildings(
    raster_path: Path,
    dsm_path: Path,
    *,
    max_buildings: int,
    min_area_m2: float,
    mask_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover conservative roof candidates when OSM is unavailable.

    This is not a replacement for surveyed footprints.  It combines RGB roof
    evidence with positive local DSM relief and rejects scene-sized regions,
    which is enough to keep a temple/campus from collapsing into one terrain
    blob while remaining honest about the source in metadata.
    """
    if max_buildings <= 0:
        return [], {"enabled": True, "provider": "visual-rgb+dsm-fallback", "count": 0}
    try:
        with rasterio.open(raster_path) as raster, rasterio.open(dsm_path) as dsm:
            if not raster.crs or not dsm.crs or str(raster.crs) != str(dsm.crs):
                return [], {
                    "enabled": True,
                    "provider": "visual-rgb+dsm-fallback",
                    "count": 0,
                    "warning": "Visual building fallback requires matching projected raster CRSs.",
                }
            if raster.width != dsm.width or raster.height != dsm.height:
                return [], {
                    "enabled": True,
                    "provider": "visual-rgb+dsm-fallback",
                    "count": 0,
                    "warning": "Visual building fallback requires RGB and DSM grids to match.",
                }
            rgb = np.transpose(raster.read([1, 2, 3]), (1, 2, 0)).astype(np.float32)
            if float(np.nanmax(rgb)) > 1.5:
                rgb /= 255.0 if float(np.nanmax(rgb)) <= 255.0 else max(float(np.nanpercentile(rgb, 98)), 1.0)
            rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, 1.0)
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            high = np.maximum.reduce((r, g, b))
            low = np.minimum.reduce((r, g, b))
            saturation = (high - low) / np.maximum(high, 1e-3)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            green = (g > r * 1.04) & (g > b * 1.01)
            blue = (b > r * 1.08) & (b > g * 1.01)
            # Keep every numeric comparison grouped before applying NumPy
            # boolean operators; otherwise ``&`` can bind into the float
            # expression and the visual fallback fails before producing a
            # single candidate footprint.
            roof_like = (
                ~green
                & ~blue
                & (luminance > 0.30)
                & ((saturation < 0.48) | ((r > b * 1.10) & (g > b * 1.04)))
            )

            semantic_building = None
            if mask_path and Path(mask_path).exists():
                try:
                    with rasterio.open(mask_path) as mask_src:
                        if (
                            mask_src.width == raster.width
                            and mask_src.height == raster.height
                            and str(mask_src.crs) == str(raster.crs)
                        ):
                            semantic_building = mask_src.read(1) == BUILDING
                except (OSError, ValueError, rasterio.errors.RasterioIOError):
                    semantic_building = None

            values = dsm.read(1, masked=True).astype(np.float32).filled(np.nan)
            valid = np.isfinite(values) & (values > -9990)
            if not valid.any():
                return [], {"enabled": True, "provider": "visual-rgb+dsm-fallback", "count": 0, "warning": "DSM has no valid pixels."}
            local = _box_mean(np.nan_to_num(values, nan=float(np.nanmedian(values[valid]))), max(3, min(values.shape) // 48))
            relief = values - local
            relief_values = relief[valid & np.isfinite(relief)]
            threshold = max(0.8, float(np.percentile(relief_values, 70))) if relief_values.size else 0.8
            candidate_base = semantic_building if semantic_building is not None else roof_like
            candidate = candidate_base & valid & (relief >= threshold)

            # Suppress salt-and-pepper pixels without pulling apart real roofs.
            padded = np.pad(candidate.astype(np.uint8), 1, mode="constant")
            neighbours = sum(
                padded[1 + dy:1 + dy + candidate.shape[0], 1 + dx:1 + dx + candidate.shape[1]]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            )
            candidate &= neighbours >= 3
            if not candidate.any():
                return [], {
                    "enabled": True,
                    "provider": "visual-rgb+dsm-fallback",
                    "count": 0,
                    "warning": "No elevated roof candidates survived visual/DSM quality gates.",
                }

            pixel_area = abs(float(raster.res[0] * raster.res[1]))
            scene_area = raster.width * raster.height * pixel_area
            proposals = []
            inv = ~raster.transform
            for geometry, value in raster_shapes(candidate.astype(np.uint8), mask=candidate, transform=raster.transform, connectivity=8):
                if int(value) != 1:
                    continue
                projected_ring = geometry.get("coordinates", [[]])[0]
                if len(projected_ring) < 4:
                    continue
                area_m2 = _area(projected_ring)
                if area_m2 < min_area_m2 or area_m2 > scene_area * 0.35:
                    continue
                pixels = []
                for point in projected_ring[:-1]:
                    col, row = inv * (point[0], point[1])
                    pixels.append([float(col), float(row)])
                pixels = _simplify_ring(
                    _clip_polygon(pixels, raster.width - 1, raster.height - 1),
                    tolerance=1.25,
                )
                if len(pixels) < 3:
                    continue
                projected = []
                for point in pixels:
                    x, y = raster.transform * (point[0], point[1])
                    projected.append([float(x), float(y)])
                ground, roof = _sample_heights(dsm, projected)
                if ground is None or roof is None or roof - ground < 1.8:
                    continue
                height = min(45.0, max(2.5, float(roof - ground)))
                proposals.append({
                    "id": -800000 - len(proposals) - 1,
                    "name": "Visual roof candidate",
                    "polygon": pixels,
                    "polygon_projected": projected,
                    "area_m2": float(_area(projected)),
                    "ground_elevation": float(ground),
                    "height": float(height),
                    "roof_elevation": float(ground + height),
                    "height_source": "visual-rgb+dsm-relative-relief",
                    "geometry_profile": "standard",
                    "profile_source": "visual-rgb+dsm",
                    "roof_surface_source": "pending-dsm-quality-gate",
                    "roof_valid_fraction": 0.0,
                    "geometry_quality": "candidate",
                    "height_confidence": 0.52 if semantic_building is not None else 0.42,
                    "osm_tags": {"building": "visual-fallback"},
                    "confidence": 0.45,
                    "source": "visual-rgb+dsm",
                })

            proposals.sort(key=lambda item: item["area_m2"], reverse=True)
            proposals = proposals[:max_buildings]
            return proposals, {
                "enabled": True,
                "provider": "visual-rgb+dsm-fallback",
                "count": len(proposals),
                "approximate_count": len(proposals),
                "segmentation_guided": semantic_building is not None,
                "warning": "OSM footprints were unavailable; building geometry is a visual/DSM fallback.",
            }
    except (OSError, ValueError, rasterio.errors.RasterioIOError, IndexError) as exc:
        return [], {
            "enabled": True,
            "provider": "visual-rgb+dsm-fallback",
            "count": 0,
            "warning": f"Visual building fallback failed: {str(exc)[:220]}",
        }


def _project_line(geometry, raster):
    lons = [_number(point.get("lon")) for point in geometry]
    lats = [_number(point.get("lat")) for point in geometry]
    if any(value is None for value in (*lons, *lats)):
        return None, None
    xs, ys = transform_coords("EPSG:4326", raster.crs, lons, lats)
    projected = [[float(x), float(y)] for x, y in zip(xs, ys)]
    inv = ~raster.transform
    pixels = []
    for x, y in projected:
        col, row = inv * (x, y)
        pixels.append([
            float(min(max(col, 0.0), raster.width - 1.0)),
            float(min(max(row, 0.0), raster.height - 1.0)),
        ])
    if not any(0 <= (~raster.transform * (point[0], point[1]))[0] <= raster.width and 0 <= (~raster.transform * (point[0], point[1]))[1] <= raster.height for point in projected):
        return None, None
    return projected, pixels


def _project_polygon(geometry, raster):
    projected, pixels = _project_line(geometry, raster)
    if not projected or len(pixels) < 3:
        return None, None
    if abs(pixels[0][0] - pixels[-1][0]) < 1e-6 and abs(pixels[0][1] - pixels[-1][1]) < 1e-6:
        pixels = pixels[:-1]
    clipped = _clip_polygon(pixels, raster.width - 1, raster.height - 1)
    if len(clipped) < 3 or _area(clipped) < 1.0:
        return None, None
    projected_clipped = []
    for col, row in clipped:
        x, y = raster.transform * (col, row)
        projected_clipped.append([float(x), float(y)])
    return projected_clipped, clipped


def _environment_width(tags):
    defaults = {
        "motorway": 12.0,
        "trunk": 10.0,
        "primary": 8.0,
        "secondary": 7.0,
        "tertiary": 6.0,
        "residential": 5.0,
        "unclassified": 4.0,
        "service": 3.0,
        "living_street": 4.0,
        "pedestrian": 3.0,
        "cycleway": 2.0,
        "footway": 1.5,
        "path": 1.2,
    }
    explicit = _tag_number(tags.get("width"))
    return explicit if explicit and explicit > 0 else defaults.get(tags.get("highway"), 4.0)


def _point_in_polygon(point, polygon):
    inside = False
    x, y = point
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = current
        x2, y2 = previous
        crosses = (y1 > y) != (y2 > y)
        denominator = y2 - y1
        if crosses and abs(denominator) > 1e-12:
            if x < (x2 - x1) * (y - y1) / denominator + x1:
                inside = not inside
        previous = current
    return inside


def _parse_environment(payload, raster_path, dsm_path, *, max_features=1200, max_trees=500, source_gsd_m=None):
    environment = {"roads": [], "water": [], "landcover": [], "trees": []}
    skipped = 0
    with rasterio.open(raster_path) as raster, rasterio.open(dsm_path) as dsm:
        if not raster.crs or not dsm.crs or str(raster.crs) != str(dsm.crs):
            return environment, {
                "enabled": True,
                "provider": "openstreetmap",
                "warning": "Environment features require matching projected raster CRSs.",
                "roads_count": 0,
                "water_count": 0,
                "landcover_count": 0,
                "trees_count": 0,
            }
        for element in payload.get("elements") or []:
            if sum(len(values) for values in environment.values()) >= max_features:
                break
            tags = element.get("tags") or {}
            if tags.get("building"):
                continue
            element_type = element.get("type")
            natural = tags.get("natural")
            landuse = tags.get("landuse")
            if tags.get("natural") == "tree" and len(environment["trees"]) < max_trees:
                point = element if element_type == "node" else ((element.get("geometry") or [None])[0] or {})
                lon, lat = _number(point.get("lon")), _number(point.get("lat"))
                if lon is None or lat is None:
                    skipped += 1
                    continue
                x_values, y_values = transform_coords("EPSG:4326", raster.crs, [lon], [lat])
                projected = [float(x_values[0]), float(y_values[0])]
                inv = ~raster.transform
                col, row = inv * (projected[0], projected[1])
                if not (0 <= col < raster.width and 0 <= row < raster.height):
                    continue
                height = _tag_number(tags.get("height"))
                height_source = "osm:height" if height and height > 0 else "osm:approximate-context"
                if not height or height <= 0:
                    height = 8.0 if source_gsd_m is None or source_gsd_m > 5 else 10.0
                diameter = _tag_number(tags.get("diameter_crown") or tags.get("diameter"))
                canopy_radius = max(1.5, min(8.0, (diameter / 2.0) if diameter and diameter > 0 else height * 0.32))
                ground = _sample_point(dsm, projected)
                if ground is None:
                    skipped += 1
                    continue
                environment["trees"].append({
                    "id": int(element.get("id") or len(environment["trees"]) + 1),
                    "point": [float(min(max(col, 0.0), raster.width - 1.0)), float(min(max(row, 0.0), raster.height - 1.0))],
                    "point_projected": projected,
                    "ground_elevation": float(ground),
                    "height": float(height),
                    "canopy_radius": float(canopy_radius),
                    "height_source": height_source,
                    "source": "openstreetmap",
                    "confidence": 0.95 if height_source == "osm:height" else 0.62,
                    "name": tags.get("name") or tags.get("species") or "Tree",
                })
                continue

            geometry = element.get("geometry") or []
            if len(geometry) < 2:
                skipped += 1
                continue
            if natural == "tree_row" and len(environment["trees"]) < max_trees:
                projected, pixels = _project_line(geometry, raster)
                if projected and len(pixels) >= 2:
                    for index in range(0, len(pixels), max(1, int(10.0 / max(abs(float(raster.res[0])), 1e-6)))):
                        if len(environment["trees"]) >= max_trees:
                            break
                        ground = _sample_point(dsm, projected[index])
                        if ground is None:
                            continue
                        environment["trees"].append({
                            "id": int(element.get("id") or len(environment["trees"]) + 1),
                            "point": pixels[index],
                            "point_projected": projected[index],
                            "ground_elevation": float(ground),
                            "height": 8.0,
                            "canopy_radius": 2.5,
                            "height_source": "tree-row:approximate-context",
                            "source": "openstreetmap:tree-row",
                            "confidence": 0.55,
                            "name": tags.get("name") or "Tree row",
                        })
                continue
            if tags.get("highway"):
                projected, pixels = _project_line(geometry, raster)
                if not projected or len(pixels) < 2 or len(environment["roads"]) >= max_features:
                    continue
                ground = _sample_point(dsm, projected[len(projected) // 2])
                if ground is None:
                    continue
                environment["roads"].append({
                    "id": int(element.get("id") or len(environment["roads"]) + 1),
                    "path": pixels,
                    "path_projected": projected,
                    "width_m": float(_environment_width(tags)),
                    "ground_elevation": float(ground),
                    "class": tags.get("highway", "road"),
                    "source": "openstreetmap",
                    "confidence": 0.95,
                    "name": tags.get("name") or tags.get("ref") or "Road",
                })
                continue

            if tags.get("waterway") or natural == "water":
                projected, pixels = _project_polygon(geometry, raster) if len(geometry) >= 3 else _project_line(geometry, raster)
                if not projected or len(pixels) < 2 or len(environment["water"]) >= max_features:
                    continue
                ground = _sample_point(dsm, projected[0]) or 0.0
                environment["water"].append({
                    "id": int(element.get("id") or len(environment["water"]) + 1),
                    "polygon": pixels if len(pixels) >= 3 else None,
                    "polygon_projected": projected if len(pixels) >= 3 else None,
                    "path": pixels if len(pixels) < 3 else None,
                    "path_projected": projected if len(pixels) < 3 else None,
                    "ground_elevation": float(ground),
                    "class": "water",
                    "source": "openstreetmap",
                    "confidence": 0.95,
                    "name": tags.get("name") or "Water",
                })
                continue

            leisure = tags.get("leisure")
            if natural in {"wood", "scrub", "heath"} or landuse in {"forest", "wood", "grass", "meadow", "park", "cemetery", "allotments"} or leisure in {"park", "garden", "playground"}:
                projected, pixels = _project_polygon(geometry, raster)
                if not projected or len(environment["landcover"]) >= max_features:
                    continue
                ground = _sample_point(dsm, projected[0]) or 0.0
                environment["landcover"].append({
                    "id": int(element.get("id") or len(environment["landcover"]) + 1),
                    "polygon": pixels,
                    "polygon_projected": projected,
                    "ground_elevation": float(ground),
                    "class": "forest" if natural in {"wood", "scrub", "heath"} or landuse in {"forest", "wood"} else "park",
                    "source": "openstreetmap",
                    "confidence": 0.9,
                    "name": tags.get("name") or "Green area",
                })

        # Woodland polygons often have no individual tree nodes in OSM. Add a
        # bounded, explicitly approximate tree population inside those areas
        # so the 3D scene still reads as an environment instead of bare land.
        if len(environment["trees"]) < max_trees:
            pixel_x = max(abs(float(raster.res[0])), 1e-6)
            pixel_y = max(abs(float(raster.res[1])), 1e-6)
            step_x = max(4.0, 14.0 / pixel_x)
            step_y = max(4.0, 14.0 / pixel_y)
            for area in environment["landcover"]:
                if area.get("class") not in {"forest", "park"} or len(environment["trees"]) >= max_trees:
                    continue
                polygon = area.get("polygon") or []
                if len(polygon) < 3:
                    continue
                min_col = max(0, int(min(point[0] for point in polygon)))
                max_col = min(raster.width - 1, int(max(point[0] for point in polygon)))
                min_row = max(0, int(min(point[1] for point in polygon)))
                max_row = min(raster.height - 1, int(max(point[1] for point in polygon)))
                row = float(min_row) + step_y * 0.5
                while row <= max_row and len(environment["trees"]) < max_trees:
                    col = float(min_col) + step_x * 0.5
                    while col <= max_col and len(environment["trees"]) < max_trees:
                        if _point_in_polygon((col, row), polygon):
                            projected_point = list(map(float, raster.transform * (col, row)))
                            ground = _sample_point(dsm, projected_point)
                            if ground is not None:
                                environment["trees"].append({
                                    "id": -len(environment["trees"]) - 1,
                                    "point": [col, row],
                                    "point_projected": projected_point,
                                    "ground_elevation": float(ground),
                                    "height": 8.0,
                                    "canopy_radius": 2.5,
                                    "height_source": "landcover:approximate-context",
                                    "source": "openstreetmap:landcover-inferred",
                                    "confidence": 0.45,
                                    "name": "Estimated woodland tree",
                                })
                        col += step_x
                    row += step_y

    approximate_trees = sum(tree["height_source"] != "osm:height" for tree in environment["trees"])
    warning = "Tree heights are approximate unless OSM height tags are present."
    if approximate_trees == 0 and not environment["trees"]:
        warning = None
    return environment, {
        "enabled": True,
        "provider": "openstreetmap",
        "attribution": "© OpenStreetMap contributors",
        "roads_count": len(environment["roads"]),
        "water_count": len(environment["water"]),
        "landcover_count": len(environment["landcover"]),
        "trees_count": len(environment["trees"]),
        "approximate_tree_count": approximate_trees,
        "skipped": skipped,
        "warning": warning,
    }


def fetch_map_features(
    aoi: dict[str, float],
    raster_path: Path,
    dsm_path: Path,
    *,
    endpoint: str,
    timeout: float = 20.0,
    max_buildings: int = 500,
    max_features: int = 1200,
    max_trees: int = 500,
    min_area_m2: float = 20.0,
    source_gsd_m: float | None = None,
    mask_path: Path | None = None,
):
    """Fetch one AOI-scoped OSM payload and derive all renderable layers."""
    if not endpoint:
        fallback_buildings, fallback_info = _visual_buildings(
            raster_path,
            dsm_path,
            max_buildings=max_buildings,
            min_area_m2=min_area_m2,
            mask_path=mask_path,
        )
        return fallback_buildings, fallback_info, {"roads": [], "water": [], "landcover": [], "trees": []}, {
            "enabled": False,
            "provider": "semantic-fallback",
            "count": 0,
            "warning": "OSM environment source is disabled; semantic RGB fallback will be used.",
        }
    try:
        payload = _query_overpass(endpoint, aoi, timeout)
    except BuildingProviderError as exc:
        fallback_buildings, fallback_info = _visual_buildings(
            raster_path,
            dsm_path,
            max_buildings=max_buildings,
            min_area_m2=min_area_m2,
            mask_path=mask_path,
        )
        warning = {
            **fallback_info,
            "warning": f"{str(exc)[:260]} {fallback_info.get('warning', '')}".strip(),
            "osm_warning": str(exc),
        }
        environment_warning = {
            "enabled": True,
            "provider": "semantic-fallback",
            "roads_count": 0,
            "water_count": 0,
            "landcover_count": 0,
            "trees_count": 0,
            "warning": "OSM environment layers unavailable; semantic RGB fallback will be used.",
        }
        return fallback_buildings, warning, {"roads": [], "water": [], "landcover": [], "trees": []}, environment_warning
    buildings, building_info = fetch_buildings(
        aoi,
        raster_path,
        dsm_path,
        endpoint=endpoint,
        timeout=timeout,
        max_buildings=max_buildings,
        min_area_m2=min_area_m2,
        source_gsd_m=source_gsd_m,
        _payload=payload,
    )
    if not buildings and max_buildings > 0:
        buildings, visual_info = _visual_buildings(
            raster_path,
            dsm_path,
            max_buildings=max_buildings,
            min_area_m2=min_area_m2,
            mask_path=mask_path,
        )
        if buildings:
            building_info = {
                **visual_info,
                "osm_count": building_info.get("count", 0),
                "osm_warning": building_info.get("warning"),
            }
    environment, environment_info = _parse_environment(
        payload,
        raster_path,
        dsm_path,
        max_features=max_features,
        max_trees=max_trees,
        source_gsd_m=source_gsd_m,
    )
    return buildings, building_info, environment, environment_info


def buildings_geojson(buildings: list[dict[str, Any]], crs: str | None = None) -> dict[str, Any]:
    features = []
    for building in buildings:
        polygon = building.get("polygon_projected") or []
        if len(polygon) < 3:
            continue
        features.append({
            "type": "Feature",
            "id": building.get("id"),
            "properties": {
                key: building.get(key)
                for key in ("name", "area_m2", "ground_elevation", "height", "roof_elevation", "height_source", "geometry_profile", "confidence", "source", "osm_tags")
            },
            "geometry": {"type": "Polygon", "coordinates": [[point for point in polygon + [polygon[0]]]]},
        })
    result = {"type": "FeatureCollection", "features": features}
    if crs:
        result["crs"] = {"type": "name", "properties": {"name": crs}}
    return result


def environment_geojson(environment: dict[str, Any], crs: str | None = None) -> dict[str, Any]:
    features = []
    for road in environment.get("roads") or []:
        path = road.get("path_projected") or []
        if len(path) >= 2:
            features.append({
                "type": "Feature",
                "id": road.get("id"),
                "properties": {key: road.get(key) for key in ("name", "class", "width_m", "ground_elevation", "source", "confidence")},
                "geometry": {"type": "LineString", "coordinates": path},
            })
    for layer in ("water", "landcover"):
        for item in environment.get(layer) or []:
            polygon = item.get("polygon_projected") or []
            if len(polygon) >= 3:
                features.append({
                    "type": "Feature",
                    "id": item.get("id"),
                    "properties": {
                        "class": item.get("class", layer),
                        "name": item.get("name"),
                        "ground_elevation": item.get("ground_elevation"),
                        "source": item.get("source"),
                        "confidence": item.get("confidence"),
                    },
                    "geometry": {"type": "Polygon", "coordinates": [[point for point in polygon + [polygon[0]]]]},
                })
    for tree in environment.get("trees") or []:
        point = tree.get("point_projected") or []
        if len(point) == 2:
            features.append({
                "type": "Feature",
                "id": tree.get("id"),
                "properties": {key: tree.get(key) for key in ("name", "height", "canopy_radius", "ground_elevation", "height_source", "source", "confidence")},
                "geometry": {"type": "Point", "coordinates": point},
            })
    for region in environment.get("semantic_regions") or []:
        polygon = region.get("polygon_projected") or []
        if len(polygon) >= 3:
            features.append({
                "type": "Feature",
                "id": region.get("id"),
                "properties": {
                    "class": region.get("class", "other"),
                    "class_id": region.get("class_id"),
                    "ground_elevation": region.get("ground_elevation"),
                    "source": region.get("source", "semantic-segmentation"),
                    "confidence": region.get("confidence"),
                },
                "geometry": {"type": "Polygon", "coordinates": [[point for point in polygon + [polygon[0]]]]},
            })
    result = {"type": "FeatureCollection", "features": features}
    if crs:
        result["crs"] = {"type": "name", "properties": {"name": crs}}
    return result
