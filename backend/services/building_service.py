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
import hashlib
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize, shapes as raster_shapes
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.warp import transform as transform_coords

from ..segmentation.class_map import BUILDING, ROAD, VEGETATION, WATER


_HEIGHT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class BuildingProviderError(RuntimeError):
    """Expected, actionable building-provider failure."""


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _affine_xy(transform, x, y) -> tuple[float, float]:
    """Apply an affine transform and safely unpack its coordinate pair."""
    point = transform * (float(x), float(y))
    return float(point[0]), float(point[1])


def _tag_number(value: Any) -> float | None:
    match = _HEIGHT_RE.search(str(value or ""))
    return _number(match.group(0)) if match else None


def _geometry_profile(tags: dict[str, Any]) -> str:
    """Choose a bounded architectural profile from explicit map evidence."""
    roof_shape = str(tags.get("roof:shape") or tags.get("roof:shape:en") or "").lower().strip()
    if roof_shape in {"flat", "flat_roof"}:
        return "flat"
    if roof_shape in {"gable", "gabled", "skillion", "shed"}:
        return "gable"
    if roof_shape in {"hip", "hipped", "pyramidal", "mansard"}:
        return "hip"
    if roof_shape in {"dome", "onion", "cone"}:
        return "temple"
    values = " ".join(
        str(tags.get(key) or "")
        for key in ("building", "name", "amenity", "religion", "building:use", "building:part", "tower:type")
    ).lower()
    worship = str(tags.get("amenity") or "").lower() == "place_of_worship"
    religion = str(tags.get("religion") or "").lower()
    if any(token in values for token in (
        "temple", "mandir", "shrine", "pagoda", "gopuram", "church", "mosque", "synagogue",
    )) or (worship and religion in {"hindu", "buddhist", "jain", "sikh"}):
        return "temple"
    if any(token in values for token in ("tower", "minaret", "bell_tower")):
        return "towered"
    if any(token in values for token in ("industrial", "warehouse", "commercial", "school", "hospital", "university")):
        return "compound"
    return "flat"


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
        f"way[\"leisure\"~\"^(park|garden|playground|pitch|sports_centre|stadium|track|golf_course)$\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
        f"way[\"landuse\"~\"^(recreation_ground|grass)$\"]({aoi['south']},{aoi['west']},{aoi['north']},{aoi['east']});"
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
        # Overpass intermittently returns gateway timeouts for an otherwise
        # valid AOI. Two short retries per endpoint prevent a transient outage
        # from changing the scene into a broad semantic-only reconstruction.
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = BuildingProviderError(f"building footprint service returned HTTP {exc.code} from {candidate}.")
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = BuildingProviderError(f"building footprint service request failed for {candidate}: {exc}")
            if attempt == 0:
                time.sleep(min(0.5, max(0.05, float(timeout) * 0.025)))

    if last_error is not None:
        if len(endpoints) > 1:
            raise BuildingProviderError(f"all building footprint services failed; last error: {last_error}") from last_error
        raise last_error
    raise BuildingProviderError("building footprint service returned no response.")


def _aoi_cache_key(aoi: dict[str, float]) -> str:
    normalized = ":".join(f"{float(aoi.get(key, 0.0)):.6f}" for key in ("south", "west", "north", "east"))
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()[:32]


def _load_osm_cache(cache_root, aoi, ttl_seconds=7 * 24 * 60 * 60):
    if not cache_root:
        return None
    path = Path(cache_root) / f"{_aoi_cache_key(aoi)}.json"
    try:
        if not path.exists() or time.time() - path.stat().st_mtime > float(ttl_seconds):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = value.get("payload") if isinstance(value, dict) else None
        return payload if isinstance(payload, dict) and isinstance(payload.get("elements"), list) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _save_osm_cache(cache_root, aoi, payload):
    if not cache_root or not isinstance(payload, dict):
        return False
    try:
        root = Path(cache_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{_aoi_cache_key(aoi)}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"aoi": {key: float(aoi[key]) for key in ("south", "west", "north", "east")}, "saved_at": time.time(), "payload": payload}, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
        return True
    except (OSError, TypeError, ValueError):
        return False


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


def _sample_ground(dsm, projected):
    """Return a robust local ground elevation for a projected polygon.

    The DSM contains roofs, tree canopies, and other above-ground returns.  A
    sports or land-cover feature must therefore use the surrounding pixels
    first, with a lower percentile, instead of using one pixel or the feature
    interior as its elevation.  Keeping this sampler local to the building
    service also avoids a circular import with the segmentation service.
    """
    if len(projected) < 3:
        return None
    try:
        min_x = min(float(point[0]) for point in projected)
        min_y = min(float(point[1]) for point in projected)
        max_x = max(float(point[0]) for point in projected)
        max_y = max(float(point[1]) for point in projected)
        pad_x = max(abs(float(dsm.transform.a)), 1e-6) * 2.0
        pad_y = max(abs(float(dsm.transform.e)), 1e-6) * 2.0
        window = window_from_bounds(
            min_x - pad_x,
            min_y - pad_y,
            max_x + pad_x,
            max_y + pad_y,
            transform=dsm.transform,
        ).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, dsm.width, dsm.height))
        if window.width < 1 or window.height < 1:
            return None

        data = dsm.read(1, window=window, masked=True)
        inside = geometry_mask(
            [{
                "type": "Polygon",
                "coordinates": [[list(point) for point in projected + [projected[0]]]],
            }],
            out_shape=data.shape,
            transform=dsm.window_transform(window),
            invert=True,
            all_touched=True,
        )
        values = data.astype(np.float32).filled(np.nan)
        valid = np.isfinite(values) & ~np.ma.getmaskarray(data) & (values > -9990)
        outside = values[(~inside) & valid]
        selected = outside if outside.size >= 3 else values[inside & valid]
        if selected.size:
            return float(np.percentile(selected, 18))
        return None
    except (TypeError, ValueError, IndexError, rasterio.errors.RasterioIOError):
        return None


def _sample_local_ground(src, projected_point, radius_m=4.0):
    """Sample a low local DSM percentile so canopies/roofs are not used as ground."""
    try:
        x, y = float(projected_point[0]), float(projected_point[1])
        radius = max(1.0, float(radius_m))
        window = window_from_bounds(
            x - radius, y - radius, x + radius, y + radius,
            transform=src.transform,
        ).round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
        if window.width < 1 or window.height < 1:
            return None
        data = src.read(1, window=window, masked=True)
        values = data.astype(np.float32).filled(np.nan)
        valid = np.isfinite(values) & ~np.ma.getmaskarray(data) & (values > -9990)
        if not valid.any():
            return None
        return float(np.percentile(values[valid], 15))
    except (TypeError, ValueError, IndexError, rasterio.errors.RasterioIOError):
        return None


def _polygon_compactness(polygon):
    if len(polygon) < 3:
        return 0.0
    perimeter = sum(
        math.hypot(
            float(polygon[(index + 1) % len(polygon)][0]) - float(polygon[index][0]),
            float(polygon[(index + 1) % len(polygon)][1]) - float(polygon[index][1]),
        )
        for index in range(len(polygon))
    )
    return 4.0 * math.pi * _area(polygon) / max(perimeter * perimeter, 1e-6)


def _roof_like_mask(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / (255.0 if rgb.max(initial=0.0) <= 255.0 else max(float(np.nanpercentile(rgb, 98)), 1.0))
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, 1.0)
    red, green_band, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    high = np.maximum.reduce((red, green_band, blue))
    low = np.minimum.reduce((red, green_band, blue))
    saturation = (high - low) / np.maximum(high, 1e-3)
    luminance = 0.299 * red + 0.587 * green_band + 0.114 * blue
    green = (green_band > red * 1.04) & (green_band > blue * 1.01)
    blue_like = (blue > red * 1.08) & (blue > green_band * 1.01)
    return (
        ~green
        & ~blue_like
        & (luminance > 0.18)
        & ((saturation < 0.62) | ((red > blue * 1.08) & (green_band > blue * 1.03)))
    )


def _building_evidence(raster, dsm, rgb, projected_polygon, *, semantic=False, max_area_m2=2000.0):
    """Return evidence metrics used to reject fields and weak roof blobs."""
    area = float(_area(projected_polygon))
    compactness = _polygon_compactness(projected_polygon)
    if area <= 0 or len(projected_polygon) < 3:
        return {"accepted": False, "reason": "invalid-footprint", "area_m2": area, "compactness": compactness}
    if semantic and area > max_area_m2:
        return {"accepted": False, "reason": "oversized-semantic-region", "area_m2": area, "compactness": compactness}
    if compactness < 0.055:
        return {"accepted": False, "reason": "low-compactness", "area_m2": area, "compactness": compactness}
    ring = [list(map(float, point)) for point in projected_polygon]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    try:
        inside = geometry_mask(
            [{"type": "Polygon", "coordinates": [ring]}],
            out_shape=(dsm.height, dsm.width),
            transform=dsm.transform,
            invert=True,
            all_touched=True,
        )
        values = dsm.read(1, masked=True).astype(np.float32).filled(np.nan)
        valid = inside & np.isfinite(values) & (values > -9990)
        roof_values = values[valid]
        if roof_values.size < 3:
            return {"accepted": False, "reason": "insufficient-roof-evidence", "area_m2": area, "compactness": compactness}
        outside = (~inside) & np.isfinite(values) & (values > -9990)
        ground_values = values[outside]
        sampled_ground, sampled_roof = _sample_heights(dsm, projected_polygon)
        ground = float(sampled_ground) if sampled_ground is not None else (float(np.percentile(ground_values, 25)) if ground_values.size >= 3 else float(np.percentile(roof_values, 10)))
        roof = float(sampled_roof) if sampled_roof is not None else float(np.percentile(roof_values, 90))
        relief = roof - ground
        roof_spread = float(np.percentile(roof_values, 95) - np.percentile(roof_values, 5))
        roof_mask = _roof_like_mask(rgb)
        roof_support = float(roof_mask[inside].mean()) if roof_mask.shape == inside.shape and inside.any() else 0.0
    except (TypeError, ValueError, IndexError, rasterio.errors.RasterioIOError):
        return {"accepted": False, "reason": "invalid-raster-evidence", "area_m2": area, "compactness": compactness}

    min_relief = 1.5 if area <= 20.0 else 1.8
    min_support = 0.12 if area <= 20.0 else (0.20 if semantic else 0.14)
    if relief < min_relief:
        return {"accepted": False, "reason": "insufficient-roof-relief", "area_m2": area, "compactness": compactness, "roof_support": roof_support, "relief_m": relief}
    if roof_support < min_support:
        return {"accepted": False, "reason": "insufficient-roof-evidence", "area_m2": area, "compactness": compactness, "roof_support": roof_support, "relief_m": relief}
    if roof_spread > max(5.0, relief * 1.35) and semantic:
        return {"accepted": False, "reason": "nonplanar-field-relief", "area_m2": area, "compactness": compactness, "roof_support": roof_support, "relief_m": relief, "roof_spread_m": roof_spread}
    return {
        "accepted": True,
        "area_m2": area,
        "compactness": round(compactness, 4),
        "roof_support": round(roof_support, 4),
        "relief_m": round(relief, 3),
        "roof_spread_m": round(roof_spread, 3),
        "ground_elevation": ground,
        "roof_elevation": roof,
    }


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
            height_confidence = 0.95 if explicit_height is not None or levels is not None else (0.72 if height_source == "metric-dsm:sampled" else 0.62 if profile == "temple" else 0.45)
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
                "roof_type": profile if profile in {"flat", "gable", "hip", "compound", "temple", "towered"} else "approximate",
                "roof_plane_count": {"flat": 1, "gable": 2, "hip": 4, "compound": 4, "temple": 4, "towered": 1}.get(profile, 0),
                "profile_source": "osm-tags" if (profile != "flat" or tags.get("roof:shape")) else "osm-building-footprint",
                "roof_surface_source": "pending-dsm-quality-gate",
                "roof_valid_fraction": 0.0,
                "geometry_quality": "candidate",
                "height_confidence": height_confidence,
                "confidence": 0.95 if explicit_height is not None or levels is not None else (0.72 if profile == "temple" else 0.62),
                "source": "openstreetmap",
                "osm_tags": {
                    key: value for key, value in tags.items()
                    if key in {
                        "building", "building:part", "height", "building:levels",
                        "building:levels:aboveground", "name", "amenity", "religion", "historic",
                        "roof:shape", "building:use", "tower:type",
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


def detect_sports_ground_zones(
    raster_path: Path,
    dsm_path: Path,
    mask_path: Path | None = None,
    *,
    minimum_area_m2: float = 500.0,
    minimum_pixels: int = 80,
) -> list[dict[str, Any]]:
    """Infer flat sports grounds when map features are missing.

    A cricket pitch/oval is intentionally treated as a hidden terrain mask.
    The detector requires broad green RGB support and locally flat DSM relief;
    it never emits a visible semantic mesh or a building by itself.
    """
    try:
        with rasterio.open(raster_path) as raster, rasterio.open(dsm_path) as dsm:
            if raster.width != dsm.width or raster.height != dsm.height:
                return []
            rgb = np.transpose(raster.read([1, 2, 3]), (1, 2, 0)).astype(np.float32)
            if rgb.max(initial=0.0) > 1.5:
                rgb /= 255.0 if rgb.max(initial=0.0) <= 255.0 else max(float(np.nanpercentile(rgb, 98)), 1.0)
            rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, 1.0)
            red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            green_dominant = (green > red * 1.06) & (green > blue * 1.04) & (green > 0.16)
            values = dsm.read(1, masked=True).astype(np.float32).filled(np.nan)
            valid = np.isfinite(values) & (values > -9990)
            if not valid.any():
                return []
            local = _box_mean(np.nan_to_num(values, nan=float(np.nanmedian(values[valid]))), max(2, min(values.shape) // 64))
            relief = np.abs(values - local)
            relief_values = relief[valid & np.isfinite(relief)]
            low_relief_limit = max(1.2, float(np.percentile(relief_values, 32)) if relief_values.size else 1.2)
            flat = relief <= low_relief_limit

            semantic_allowed = np.ones_like(valid, dtype=bool)
            vegetation_fraction = None
            if mask_path and Path(mask_path).exists():
                try:
                    with rasterio.open(mask_path) as mask_src:
                        semantic = mask_src.read(1)
                        if semantic.shape == values.shape:
                            # Do not let an over-broad SegFormer building mask
                            # erase the cricket ground. Roads/water remain
                            # hard exclusions; flat green RGB+DSM evidence is
                            # allowed to override a low-confidence BUILDING
                            # label for sports-ground detection.
                            semantic_allowed = ~np.isin(semantic, (ROAD, WATER))
                            vegetation_fraction = semantic == VEGETATION
                except (OSError, ValueError, rasterio.errors.RasterioIOError):
                    pass

            candidate = green_dominant & flat & valid & semantic_allowed
            if not candidate.any():
                return []
            # Remove isolated vegetation pixels without eroding a real pitch.
            padded = np.pad(candidate.astype(np.uint8), 1, mode="constant")
            neighbours = sum(
                padded[1 + dy:1 + dy + candidate.shape[0], 1 + dx:1 + dx + candidate.shape[1]]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            )
            candidate &= neighbours >= 4

            pixel_area = abs(float(raster.res[0] * raster.res[1]))
            scene_area = raster.width * raster.height * pixel_area
            zones = []
            for index, (geometry, value) in enumerate(raster_shapes(candidate.astype(np.uint8), mask=candidate, transform=raster.transform, connectivity=8)):
                if int(value) != 1:
                    continue
                pixels = int(rasterize([(geometry, 1)], out_shape=candidate.shape, transform=raster.transform, all_touched=True).sum())
                projected_ring = geometry.get("coordinates", [[]])[0]
                if len(projected_ring) < 4 or pixels < minimum_pixels:
                    continue
                area_m2 = _area(projected_ring)
                if area_m2 < minimum_area_m2 or area_m2 > scene_area * 0.45:
                    continue
                if vegetation_fraction is not None:
                    component = rasterize([(geometry, 1)], out_shape=candidate.shape, transform=raster.transform, all_touched=True).astype(bool)
                    if float(np.count_nonzero(vegetation_fraction & component)) / max(float(component.sum()), 1.0) > 0.35:
                        continue
                projected = [[float(point[0]), float(point[1])] for point in projected_ring[:-1]]
                inv = ~raster.transform
                pixel_ring = [list(_affine_xy(inv, point[0], point[1])) for point in projected]
                inside = rasterize([(geometry, 1)], out_shape=values.shape, transform=raster.transform, all_touched=True).astype(bool)
                samples = values[inside & valid]
                if samples.size < 3:
                    continue
                ground = float(np.percentile(samples, 18))
                zones.append({
                    "id": -930000000 - index,
                    "name": "Inferred sports ground",
                    "class": "sports_ground",
                    "polygon": pixel_ring,
                    "polygon_projected": projected,
                    "ground_elevation": ground,
                    "ground_elevation_source": "local-dsm-lower-percentile",
                    "source": "rgb-dsm-inferred",
                    "confidence": 0.65,
                    "area_m2": float(area_m2),
                    "relief_threshold_m": float(low_relief_limit),
                })
            return zones
    except (OSError, ValueError, rasterio.errors.RasterioIOError, IndexError):
        return []


def _visual_buildings(
    raster_path: Path,
    dsm_path: Path,
    *,
    max_buildings: int,
    min_area_m2: float,
    mask_path: Path | None = None,
    exclusion_zones: list[dict[str, Any]] | None = None,
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
            roof_like = _roof_like_mask(rgb)

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
            candidate_base = (semantic_building & roof_like) if semantic_building is not None else roof_like
            candidate = candidate_base & valid & (relief >= threshold)
            if exclusion_zones:
                zone_shapes = []
                for zone in exclusion_zones:
                    polygon = zone.get("polygon_projected") or []
                    if len(polygon) >= 3:
                        ring = [list(map(float, point)) for point in polygon]
                        if ring[0] != ring[-1]:
                            ring.append(ring[0])
                        zone_shapes.append(({"type": "Polygon", "coordinates": [ring]}, 1))
                if zone_shapes:
                    sports_mask = rasterize(
                        zone_shapes,
                        out_shape=candidate.shape,
                        transform=raster.transform,
                        fill=0,
                        all_touched=True,
                        dtype="uint8",
                    ).astype(bool)
                    candidate &= ~sports_mask

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
            effective_min_area = max(float(min_area_m2), 4.0, pixel_area * 6.0)
            max_candidate_area = min(2000.0, max(800.0, raster.width * raster.height * pixel_area * 0.02))
            simplify_tolerance = min(1.25, max(0.25, min(abs(float(raster.res[0])), abs(float(raster.res[1]))) * 0.35))
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
                if area_m2 < effective_min_area or area_m2 > max_candidate_area:
                    continue
                pixels = []
                for point in projected_ring[:-1]:
                    col, row = inv * (point[0], point[1])
                    pixels.append([float(col), float(row)])
                pixels = _simplify_ring(
                    _clip_polygon(pixels, raster.width - 1, raster.height - 1),
                    tolerance=simplify_tolerance,
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
                height = min(45.0, max(2.2 if area_m2 <= 20.0 else 2.5, float(roof - ground)))
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
                    "geometry_profile": "flat",
                    "roof_type": "approximate",
                    "roof_plane_count": 0,
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
                "sports_zones_applied": len(exclusion_zones or []),
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


def _segments_intersect(a, b, c, d):
    def orientation(p, q, r):
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else 2

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def _polygon_intersects(first, second):
    if len(first) < 3 or len(second) < 3:
        return False
    if _point_in_polygon(first[0], second) or _point_in_polygon(second[0], first):
        return True
    for i, a in enumerate(first):
        b = first[(i + 1) % len(first)]
        for j, c in enumerate(second):
            d = second[(j + 1) % len(second)]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _polygon_overlap_fraction(raster, first, second):
    """Estimate overlap using the authoritative source raster grid."""
    if len(first or []) < 3 or len(second or []) < 3:
        return 0.0
    try:
        def shape(points):
            ring = [list(map(float, point)) for point in points]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            return {"type": "Polygon", "coordinates": [ring]}
        first_mask = rasterize([(shape(first), 1)], out_shape=(raster.height, raster.width), transform=raster.transform, all_touched=True).astype(bool)
        second_mask = rasterize([(shape(second), 1)], out_shape=(raster.height, raster.width), transform=raster.transform, all_touched=True).astype(bool)
        denominator = max(int(first_mask.sum()), 1)
        return float(np.logical_and(first_mask, second_mask).sum()) / denominator
    except (TypeError, ValueError, rasterio.errors.RasterioIOError):
        return 1.0 if _polygon_intersects(first, second) else 0.0


def filter_buildings_by_evidence(buildings, raster_path, dsm_path, environment=None, *, max_inferred_area_m2=2000.0):
    """Apply the balanced acceptance policy to mapped and inferred footprints."""
    environment = environment or {}
    zones = [item.get("polygon_projected") or [] for item in environment.get("exclusion_zones") or []]
    accepted = []
    reasons = {}
    small_count = 0
    with rasterio.open(raster_path) as raster, rasterio.open(dsm_path) as dsm:
        rgb = np.transpose(raster.read([1, 2, 3]), (1, 2, 0)).astype(np.float32)
        if rgb.max(initial=0.0) > 1.5:
            rgb /= 255.0 if rgb.max(initial=0.0) <= 255.0 else max(float(np.nanpercentile(rgb, 98)), 1.0)
        for building in buildings or []:
            source = str(building.get("source", "")).lower()
            semantic = source.startswith(("semantic-", "visual-"))
            osm_footprint = source.startswith(("openstreetmap", "osm-"))
            explicit_height = str(building.get("height_source", "")).lower().startswith(("osm:height", "osm:levels"))
            polygon = building.get("polygon_projected") or []
            overlap = max((_polygon_overlap_fraction(raster, polygon, zone) for zone in zones), default=0.0)
            in_sports_zone = overlap > 0.0
            evidence = _building_evidence(
                raster,
                dsm,
                rgb,
                polygon,
                semantic=semantic,
                max_area_m2=max_inferred_area_m2,
            )
            strong_roof = bool(
                evidence.get("accepted")
                and float(evidence.get("roof_support", 0.0)) >= 0.55
                and float(evidence.get("relief_m", 0.0)) >= 3.0
                and float(evidence.get("roof_spread_m", 99.0)) <= max(6.0, float(evidence.get("relief_m", 0.0)) * 1.5)
            )
            # OSM footprints remain authoritative in ordinary context. A
            # sports zone is the explicit exception: inferred candidates over
            # more than 20% of the field are rejected, while mapped footprints
            # must be almost fully contained and backed by unusually strong
            # roof evidence before they can survive.
            sports_conflict = (
                (semantic and overlap > 0.20)
                or (osm_footprint and overlap > 0.80 and not strong_roof)
            )
            must_pass = semantic or sports_conflict or (not osm_footprint and not explicit_height)
            if must_pass and not evidence.get("accepted"):
                reason = "sports-zone-conflict" if in_sports_zone else evidence.get("reason", "insufficient-roof-evidence")
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            if sports_conflict:
                reasons["sports-zone-conflict"] = reasons.get("sports-zone-conflict", 0) + 1
                continue
            if evidence.get("accepted"):
                building["roof_evidence"] = {key: value for key, value in evidence.items() if key not in {"accepted", "reason"}}
                if semantic:
                    building["height_confidence"] = min(float(building.get("height_confidence", 0.55)), 0.78)
                    building["geometry_quality"] = "measured" if evidence.get("relief_m", 0.0) >= 2.2 else "approximate"
            if in_sports_zone:
                building["sports_zone_supported"] = True
                building["sports_zone_overlap_fraction"] = round(overlap, 4)
            area = float(building.get("area_m2", _area(polygon) if len(polygon) >= 3 else 0.0))
            if area <= 20.0:
                small_count += 1
            accepted.append(building)
    return accepted, {
        "accepted_count": len(accepted),
        "rejected_count": sum(reasons.values()),
        "rejection_reasons": reasons,
        "small_structure_count": small_count,
        "max_inferred_area_m2": max_inferred_area_m2,
    }


def _building_clearance_mask(raster, buildings, clearance_m=1.0):
    shapes = []
    for building in buildings or []:
        polygon = building.get("polygon_projected") or []
        if len(polygon) >= 3:
            ring = [list(map(float, point)) for point in polygon]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            shapes.append(({"type": "Polygon", "coordinates": [ring]}, 1))
    if not shapes:
        return np.zeros((raster.height, raster.width), dtype=bool)
    mask = rasterize(
        shapes,
        out_shape=(raster.height, raster.width),
        transform=raster.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    try:
        from scipy.ndimage import binary_dilation
        resolution = min(abs(float(raster.res[0])), abs(float(raster.res[1])))
        mask = binary_dilation(mask, iterations=max(1, int(math.ceil(float(clearance_m) / max(resolution, 1e-6)))))
    except Exception:
        pass
    return mask


def sanitize_environment_layers(buildings, environment, raster_path=None, dsm_path=None, clearance_m=1.0):
    """Remove context features that would be drawn through accepted roofs.

    OSM land-cover polygons commonly contain the buildings they surround, and
    semantic tree proxies can land on a mapped footprint after pixel-to-world
    rounding.  Keeping the full vegetation polygon in that case makes it
    render above the building even when the terrain raster was flattened.
    """
    environment = {**(environment or {})}
    removed = {"trees": 0, "landcover": 0, "water": 0, "roads": 0, "landcover_clipped": 0, "clearance_m": float(clearance_m)}
    footprints = [item.get("polygon_projected") or [] for item in (buildings or [])]
    footprints = [ring for ring in footprints if len(ring) >= 3]

    raster = None
    blocked_mask = None
    sports_zones = [item.get("polygon_projected") or [] for item in environment.get("exclusion_zones") or []]
    sports_zones = [ring for ring in sports_zones if len(ring) >= 3]
    if raster_path and (footprints or sports_zones):
        try:
            raster = rasterio.open(raster_path)
            blocked_mask = _building_clearance_mask(raster, buildings, clearance_m)
            if sports_zones:
                sports_shapes = []
                for ring in sports_zones:
                    closed = [list(map(float, point)) for point in ring]
                    if closed[0] != closed[-1]:
                        closed.append(closed[0])
                    sports_shapes.append(({"type": "Polygon", "coordinates": [closed]}, 1))
                sports_mask = rasterize(
                    sports_shapes,
                    out_shape=(raster.height, raster.width),
                    transform=raster.transform,
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                ).astype(bool)
                blocked_mask |= sports_mask
        except (OSError, ValueError, rasterio.errors.RasterioIOError):
            raster = None
            blocked_mask = None

    def mask_contains(x, y):
        if raster is None or blocked_mask is None:
            return False
        col, row = ~raster.transform * (float(x), float(y))
        col, row = int(math.floor(col)), int(math.floor(row))
        return 0 <= row < raster.height and 0 <= col < raster.width and bool(blocked_mask[row, col])

    def clipped_polygons(feature):
        if raster is None or blocked_mask is None:
            return None
        polygon = feature.get("polygon_projected") or []
        if len(polygon) < 3:
            return None
        ring = [list(map(float, point)) for point in polygon]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        try:
            feature_mask = rasterize(
                [({"type": "Polygon", "coordinates": [ring]}, 1)],
                out_shape=(raster.height, raster.width),
                transform=raster.transform,
                fill=0,
                all_touched=True,
                dtype="uint8",
            ).astype(bool)
            safe = feature_mask & ~blocked_mask
            if not safe.any():
                return []
            output = []
            for geometry, value in raster_shapes(safe.astype(np.uint8), mask=safe, transform=raster.transform, connectivity=8):
                if int(value) != 1:
                    continue
                projected = geometry.get("coordinates", [[]])[0]
                if len(projected) < 4:
                    continue
                projected = [[float(point[0]), float(point[1])] for point in projected[:-1]]
                inv = ~raster.transform
                pixels = [list(_affine_xy(inv, point[0], point[1])) for point in projected]
                copy = {**feature, "polygon": pixels, "polygon_projected": projected}
                output.append(copy)
            return output
        except (TypeError, ValueError, rasterio.errors.RasterioIOError):
            return None

    def blocked(feature, kind):
        polygon = feature.get("polygon_projected") or []
        point = feature.get("point_projected") or []
        path = feature.get("path_projected") or []
        samples = []
        if len(point) == 2:
            try:
                x, y = float(point[0]), float(point[1])
                samples.append((x, y))
                if kind == "trees":
                    radius = max(0.5, float(feature.get("canopy_radius", 2.0)))
                    for angle in np.linspace(0.0, 2.0 * np.pi, 17)[:-1]:
                        samples.append((x + radius * np.cos(angle), y + radius * np.sin(angle)))
                    if any(mask_contains(sample[0], sample[1]) for sample in samples):
                        return True
            except (TypeError, ValueError):
                pass
        if len(polygon) >= 3:
            samples.extend((float(p[0]), float(p[1])) for p in polygon)
            samples.append((
                sum(float(p[0]) for p in polygon) / len(polygon),
                sum(float(p[1]) for p in polygon) / len(polygon),
            ))
        if polygon and len(polygon) >= 3 and clipped_polygons(feature) == []:
            return True
        for footprint in footprints:
            if len(polygon) >= 3 and _polygon_intersects(polygon, footprint):
                return True
            if any(_point_in_polygon(sample, footprint) for sample in samples):
                return True
            if len(polygon) >= 3:
                center = (
                    sum(float(p[0]) for p in footprint) / len(footprint),
                    sum(float(p[1]) for p in footprint) / len(footprint),
                )
                if _point_in_polygon(center, polygon):
                    return True
        if kind == "roads" and len(path) >= 2:
            samples.extend((float(p[0]), float(p[1])) for p in path)
            for footprint in footprints:
                if any(_point_in_polygon(sample, footprint) for sample in samples):
                    return True
                for index in range(len(path) - 1):
                    start, end = path[index], path[index + 1]
                    for edge_index, edge_start in enumerate(footprint):
                        edge_end = footprint[(edge_index + 1) % len(footprint)]
                        if _segments_intersect(start, end, edge_start, edge_end):
                            return True
        return False

    for layer in ("trees", "landcover", "water", "roads"):
        kept = []
        for feature in environment.get(layer) or []:
            if layer in {"landcover", "water"} and raster is not None:
                pieces = clipped_polygons(feature)
                if pieces == []:
                    removed[layer] += 1
                    continue
                if pieces:
                    if len(pieces) > 1 or any(_polygon_intersects(feature.get("polygon_projected") or [], footprint) for footprint in footprints):
                        removed["landcover_clipped"] += 1
                    kept.extend(pieces)
                    continue
            if blocked(feature, layer):
                removed[layer] += 1
            else:
                kept.append(feature)
        environment[layer] = kept
    if raster is not None:
        raster.close()
    if dsm_path:
        try:
            with rasterio.open(dsm_path) as dsm:
                normalized = 0
                for feature in environment.get("trees") or []:
                    point = feature.get("point_projected") or []
                    ground = _sample_local_ground(dsm, point, max(3.0, float(feature.get("canopy_radius", 2.0)) * 2.0)) if len(point) == 2 else None
                    if ground is not None:
                        feature["ground_elevation"] = ground
                        feature["ground_elevation_source"] = "local-dsm-lower-percentile"
                        normalized += 1
                for layer in ("landcover", "water"):
                    for feature in environment.get(layer) or []:
                        polygon = feature.get("polygon_projected") or []
                        if len(polygon) < 3:
                            continue
                        center = [sum(float(point[0]) for point in polygon) / len(polygon), sum(float(point[1]) for point in polygon) / len(polygon)]
                        ground = _sample_local_ground(dsm, center, max(3.0, math.sqrt(max(_area(polygon), 1.0)) * 0.35))
                        if ground is not None:
                            feature["ground_elevation"] = ground
                            feature["ground_elevation_source"] = "local-dsm-lower-percentile"
                            normalized += 1
                for zone in environment.get("exclusion_zones") or []:
                    polygon = zone.get("polygon_projected") or []
                    if len(polygon) >= 3:
                        center = [sum(float(point[0]) for point in polygon) / len(polygon), sum(float(point[1]) for point in polygon) / len(polygon)]
                        ground = _sample_local_ground(dsm, center, max(3.0, math.sqrt(max(_area(polygon), 1.0)) * 0.25))
                        if ground is not None:
                            zone["ground_elevation"] = ground
                            zone["ground_elevation_source"] = "local-dsm-lower-percentile"
                            normalized += 1
                for road in environment.get("roads") or []:
                    path = road.get("path_projected") or []
                    if len(path) >= 2:
                        center = path[len(path) // 2]
                        ground = _sample_local_ground(dsm, center, max(2.0, float(road.get("width_m", 4.0))))
                        if ground is not None:
                            road["ground_elevation"] = ground
                            road["ground_elevation_source"] = "local-dsm-lower-percentile"
                            normalized += 1
                removed["ground_normalized"] = normalized
        except (OSError, ValueError, rasterio.errors.RasterioIOError):
            removed["ground_normalized"] = 0
    return environment, removed


def validate_scene_alignment(buildings, environment, raster_path):
    """Keep only AOI features that satisfy the projected-feature contract."""
    environment = {**(environment or {})}
    report = {"checked": 0, "rejected": 0, "round_trip_failures": 0, "outside_aoi": 0}

    def valid_points(points, raster):
        if not points:
            return False
        for point in points:
            try:
                x, y = float(point[0]), float(point[1])
                col, row = ~raster.transform * (x, y)
            except (TypeError, ValueError, IndexError):
                return False
            if not (-0.5 <= col <= raster.width - 0.5 and -0.5 <= row <= raster.height - 0.5):
                return False
        return True

    def round_trip_ok(feature, raster, projected_key, pixel_key=None):
        projected = feature.get(projected_key) or []
        if projected_key == "point_projected":
            projected = [projected]
        if len(projected) < 1 or not valid_points(projected, raster):
            report["outside_aoi"] += 1
            return False
        if pixel_key:
            pixels = feature.get(pixel_key) or []
            if len(pixels) == len(projected):
                for projected_point, pixel in zip(projected, pixels):
                    expected = ~raster.transform * (float(projected_point[0]), float(projected_point[1]))
                    if abs(float(expected[0]) - float(pixel[0])) > 0.75 or abs(float(expected[1]) - float(pixel[1])) > 0.75:
                        report["round_trip_failures"] += 1
                        return False
        report["checked"] += 1
        return True

    with rasterio.open(raster_path) as raster:
        kept_buildings = []
        for building in buildings or []:
            if round_trip_ok(building, raster, "polygon_projected", "polygon"):
                kept_buildings.append(building)
            else:
                report["rejected"] += 1

        for layer in ("exclusion_zones", "landcover", "water", "semantic_regions"):
            kept = []
            for feature in environment.get(layer) or []:
                if round_trip_ok(feature, raster, "polygon_projected", "polygon"):
                    kept.append(feature)
                else:
                    report["rejected"] += 1
            environment[layer] = kept
        kept = []
        for road in environment.get("roads") or []:
            if round_trip_ok(road, raster, "path_projected", "path"):
                kept.append(road)
            else:
                report["rejected"] += 1
        environment["roads"] = kept
        kept = []
        for tree in environment.get("trees") or []:
            if round_trip_ok(tree, raster, "point_projected"):
                kept.append(tree)
            else:
                report["rejected"] += 1
        environment["trees"] = kept
    return kept_buildings, environment, report


def _parse_environment(payload, raster_path, dsm_path, *, max_features=1200, max_trees=500, source_gsd_m=None):
    environment = {"roads": [], "water": [], "landcover": [], "trees": [], "exclusion_zones": []}
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
            leisure = tags.get("leisure")
            if leisure in {"pitch", "sports_centre", "stadium", "track", "golf_course"} or landuse in {"recreation_ground"}:
                geometry = element.get("geometry") or []
                projected, pixels = _project_polygon(geometry, raster) if len(geometry) >= 3 else (None, None)
                if projected and pixels:
                    ground = _sample_ground(dsm, projected) or 0.0
                    environment["exclusion_zones"].append({
                        "id": int(element.get("id") or len(environment["exclusion_zones"]) + 1),
                        "polygon": pixels,
                        "polygon_projected": projected,
                        "class": "sports_ground",
                        "source": "openstreetmap",
                        "confidence": 0.95,
                        "ground_elevation": float(ground),
                        "ground_elevation_source": "local-dsm-lower-percentile",
                        "name": tags.get("name") or "Sports ground",
                    })
                continue
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
        "exclusion_zones_count": len(environment["exclusion_zones"]),
        "approximate_tree_count": approximate_trees,
        "skipped": skipped,
        "warning": warning,
    }


def _fallback_scene_features(
    aoi,
    raster_path,
    dsm_path,
    *,
    max_buildings,
    min_area_m2,
    mask_path=None,
    reason=None,
):
    zones = detect_sports_ground_zones(raster_path, dsm_path, mask_path)
    fallback_buildings, fallback_info = _visual_buildings(
        raster_path,
        dsm_path,
        max_buildings=max_buildings,
        min_area_m2=min_area_m2,
        mask_path=mask_path,
        exclusion_zones=zones,
    )
    warning = dict(fallback_info)
    warning["provider_source"] = "imagery-fallback"
    if reason:
        warning["osm_warning"] = str(reason)
        warning["warning"] = f"{str(reason)[:260]} {fallback_info.get('warning', '')}".strip()
    warning["sports_ground_detected"] = len(zones)
    environment_warning = {
        "enabled": True,
        "provider": "imagery-fallback",
        "roads_count": 0,
        "water_count": 0,
        "landcover_count": 0,
        "trees_count": 0,
        "exclusion_zones_count": len(zones),
        "warning": "OSM environment layers unavailable; conservative RGB/DSM fallback was used.",
    }
    return fallback_buildings, warning, {
        "roads": [],
        "water": [],
        "landcover": [],
        "trees": [],
        "exclusion_zones": zones,
    }, environment_warning


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
    cache_root: Path | None = None,
    cache_ttl_seconds: float = 7 * 24 * 60 * 60,
):
    """Fetch one AOI-scoped OSM payload and derive all renderable layers."""
    if not endpoint:
        return _fallback_scene_features(
            aoi, raster_path, dsm_path, max_buildings=max_buildings,
            min_area_m2=min_area_m2, mask_path=mask_path,
            reason="OSM environment source is disabled.",
        )
    payload = _load_osm_cache(cache_root, aoi, cache_ttl_seconds)
    payload_source = "cache" if payload is not None else "live"
    try:
        if payload is None:
            payload = _query_overpass(endpoint, aoi, timeout)
            _save_osm_cache(cache_root, aoi, payload)
    except BuildingProviderError as exc:
        return _fallback_scene_features(
            aoi, raster_path, dsm_path, max_buildings=max_buildings,
            min_area_m2=min_area_m2, mask_path=mask_path, reason=exc,
        )
    try:
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
    except Exception as exc:
        buildings = []
        building_info = {
            "enabled": True,
            "provider": "openstreetmap",
            "count": 0,
            "warning": f"Building feature parsing skipped: {str(exc)[:220]}",
        }
    building_info["provider_source"] = payload_source
    if payload_source == "cache":
        building_info["provider_warning"] = "AOI features loaded from the successful seven-day OSM cache."
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
    try:
        environment, environment_info = _parse_environment(
            payload,
            raster_path,
            dsm_path,
            max_features=max_features,
            max_trees=max_trees,
            source_gsd_m=source_gsd_m,
        )
    except Exception as exc:
        environment = {"roads": [], "water": [], "landcover": [], "trees": [], "exclusion_zones": []}
        environment_info = {
            "enabled": True,
            "provider": "openstreetmap",
            "roads_count": 0,
            "water_count": 0,
            "landcover_count": 0,
            "trees_count": 0,
            "exclusion_zones_count": 0,
            "warning": f"Environment feature parsing skipped: {str(exc)[:220]}",
        }
    environment_info["provider_source"] = payload_source
    if payload_source == "cache":
        environment_info["provider_warning"] = "AOI environment loaded from the successful seven-day OSM cache."
    if not environment.get("exclusion_zones"):
        inferred_zones = detect_sports_ground_zones(raster_path, dsm_path, mask_path)
        environment["exclusion_zones"] = inferred_zones
        environment_info["sports_ground_detected"] = len(inferred_zones)
        environment_info["sports_ground_source"] = "rgb-dsm-inferred" if inferred_zones else None
    else:
        environment_info["sports_ground_detected"] = len(environment.get("exclusion_zones") or [])
        environment_info["sports_ground_source"] = "openstreetmap"
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
                for key in ("name", "area_m2", "ground_elevation", "height", "roof_elevation", "height_source", "geometry_profile", "roof_type", "roof_plane_count", "roof_evidence", "geometry_quality", "confidence", "source", "osm_tags")
            },
            "geometry": {"type": "Polygon", "coordinates": [[point for point in polygon + [polygon[0]]]]},
        })
    result = {"type": "FeatureCollection", "features": features}
    if crs:
        result["crs"] = {"type": "name", "properties": {"name": crs}}
    return result


def environment_geojson(environment: dict[str, Any], crs: str | None = None) -> dict[str, Any]:
    features = []
    for zone in environment.get("exclusion_zones") or []:
        polygon = zone.get("polygon_projected") or []
        if len(polygon) >= 3:
            features.append({
                "type": "Feature",
                "id": zone.get("id"),
                "properties": {
                    "class": zone.get("class", "exclusion"),
                    "name": zone.get("name"),
                    "source": zone.get("source"),
                    "confidence": zone.get("confidence"),
                    "ground_elevation": zone.get("ground_elevation"),
                    "ground_elevation_source": zone.get("ground_elevation_source"),
                    "rendered": False,
                },
                "geometry": {"type": "Polygon", "coordinates": [[point for point in polygon + [polygon[0]]]]},
            })
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
