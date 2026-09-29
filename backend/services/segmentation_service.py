"""Semantic segmentation adapter for map and upload reconstruction jobs.

The model supplies semantic evidence; it does not replace the metric DSM or
OSM source.  Its output is persisted as a georeferenced mask and converted to
bounded polygons so the mesh can expose buildings, roads, water, vegetation,
and a selectable semantic wireframe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds as window_from_bounds

from ..segmentation import COLORS, NAMES, BUILDING, OTHER, VEGETATION, WATER
from ..segmentation.engine import segment_rgb
from .building_service import _area, _building_evidence, _sample_heights


def _normalise_rgb(data: np.ndarray) -> np.ndarray:
    """Convert a raster's first three bands into uint8 RGB for the model."""
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2:
        data = np.concatenate([data, data[:1]], axis=0)
    else:
        data = data[:3]
    data = np.moveaxis(data, 0, -1)
    if data.dtype == np.uint8:
        return data
    out = np.zeros(data.shape, dtype=np.uint8)
    for band in range(3):
        values = np.asarray(data[..., band], dtype=np.float32)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        lo, hi = np.percentile(values[finite], [2, 98])
        if hi <= lo:
            lo, hi = float(values[finite].min()), float(values[finite].max())
        if hi > lo:
            out[..., band] = np.clip((values - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        else:
            out[..., band] = np.clip(values, 0, 255).astype(np.uint8)
    return out


def _mask_png(mask: np.ndarray) -> Image.Image:
    colors = np.zeros((256, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        colors[int(class_id)] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(colors[np.clip(mask.astype(np.uint16), 0, 255)], mode="RGB")


def _simplify_ring(points, tolerance=1.25):
    """Remove raster stair-steps without changing the footprint materially."""
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
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return float(((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) ** 0.5)
        t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)
        t = min(1.0, max(0.0, t))
        x, y = start[0] + t * dx, start[1] + t * dy
        return float(((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5)

    def rdp(items):
        if len(items) <= 2:
            return items
        best_index, best_distance = 0, 0.0
        for index in range(1, len(items) - 1):
            current = distance(items[index], items[0], items[-1])
            if current > best_distance:
                best_index, best_distance = index, current
        if best_distance > tolerance:
            return rdp(items[:best_index + 1])[:-1] + rdp(items[best_index:])
        return [items[0], items[-1]]

    # Douglas-Peucker on a closed ring, followed by a collinear-point pass.
    simplified = rdp(clean + [clean[0]])[:-1]
    if len(simplified) < 3:
        return clean
    result = []
    for point in simplified:
        if len(result) < 2:
            result.append(point)
            continue
        a, b = result[-2], result[-1]
        cross = (b[0] - a[0]) * (point[1] - b[1]) - (b[1] - a[1]) * (point[0] - b[0])
        if abs(cross) < tolerance * 0.2:
            result[-1] = point
        else:
            result.append(point)
    return result if len(result) >= 3 else clean


def run_segmentation(settings, raster_path: Path, mask_path: Path, preview_path: Path) -> dict[str, Any]:
    """Run the configured semantic model and persist mask + preview artifacts."""
    if not settings.segmentation_enabled:
        return {
            "enabled": False,
            "available": False,
            "fallback": True,
            "reason": "disabled by ASTERRA_SEGMENTATION_ENABLED=false",
            "classes_present": [],
            "class_counts": {},
        }
    try:
        with rasterio.open(raster_path) as src:
            rgb = _normalise_rgb(src.read())
            profile = src.profile.copy()
        # Semantic segmentation is optional evidence.  Keep it bounded so a
        # first-run torch/HF load cannot hold INGEST forever.  The worker may
        # finish in the background, but it never writes artifacts directly;
        # this function owns the deterministic fallback artifacts.
        timeout_seconds = max(1.0, float(getattr(settings, "segmentation_timeout_seconds", 12.0)))
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asterra-segmentation")
        future = executor.submit(segment_rgb, rgb)
        timed_out = False
        try:
            result, info = future.result(timeout=timeout_seconds)
        except FutureTimeout:
            timed_out = True
            future.cancel()
            result, info = segment_rgb(rgb, allow_model=False)
            info = {
                **info,
                "fallback": True,
                "reason": f"segmentation timed out after {timeout_seconds:g}s; deterministic RGB fallback used",
            }
        except Exception as exc:
            # A model/import/runtime failure must not leave INGEST active or
            # discard semantic artifacts.  Run the same deterministic fallback
            # used for an unavailable checkpoint and retain the failure reason
            # as provenance for the completed job.
            result, info = segment_rgb(rgb, allow_model=False)
            info = {
                **info,
                "fallback": True,
                "reason": f"segmentation model failed: {str(exc)[:220]}; deterministic RGB fallback used",
            }
        finally:
            executor.shutdown(wait=not timed_out, cancel_futures=True)
        if result is None:
            return {**info, "classes_present": [], "class_counts": {}}
        mask = np.asarray(result.mask, dtype=np.uint8)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"segmentation mask shape {mask.shape} does not match raster {rgb.shape[:2]}")
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=OTHER,
            compress="deflate",
            interleave="band",
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask, 1)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        _mask_png(mask).save(preview_path, format="PNG", optimize=True)
        counts = {NAMES.get(int(class_id), str(int(class_id))): int((mask == class_id).sum()) for class_id in np.unique(mask) if int(class_id) != OTHER}
        return {
            **info,
            "mask_path": str(mask_path),
            "preview_path": str(preview_path),
            "classes_present": [int(value) for value in result.classes_present],
            "class_counts": counts,
            "mean_confidence": float(result.mean_confidence),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "fallback": True,
            "reason": f"segmentation failed: {str(exc)[:300]}",
            "classes_present": [],
            "class_counts": {},
        }


def _project_ring(ring, raster):
    inv = ~raster.transform
    projected = [[float(point[0]), float(point[1])] for point in ring]
    pixels = []
    for x, y in projected:
        col, row = inv * (x, y)
        pixels.append([float(col), float(row)])
    return projected, pixels


def _sample_ground(dsm, projected):
    if len(projected) < 3:
        return None
    try:
        min_x = min(point[0] for point in projected)
        min_y = min(point[1] for point in projected)
        max_x = max(point[0] for point in projected)
        max_y = max(point[1] for point in projected)
        pad_x = max(abs(float(dsm.transform.a)), 1e-6) * 2.0
        pad_y = max(abs(float(dsm.transform.e)), 1e-6) * 2.0
        window = window_from_bounds(min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y, transform=dsm.transform).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, dsm.width, dsm.height))
        if window.width < 1 or window.height < 1:
            return None
        data = dsm.read(1, window=window, masked=True)
        inside = geometry_mask(
            [{"type": "Polygon", "coordinates": [[list(point) for point in projected + [projected[0]]]]}],
            out_shape=data.shape,
            transform=dsm.window_transform(window),
            invert=True,
            all_touched=True,
        )
        values = data.astype(np.float32).filled(np.nan)
        valid = np.isfinite(values) & ~np.ma.getmaskarray(data) & (values > -9990)
        # The footprint contains roof/canopy tops. Use the surrounding annulus
        # first so semantic context and inferred homes are anchored to the
        # same bare-ground datum as OSM buildings.
        outside = values[(~inside) & valid]
        selected = outside if outside.size >= 3 else values[inside & valid]
        if selected.size:
            return float(np.percentile(selected, 18))
        return None
    except (ValueError, rasterio.errors.RasterioIOError):
        return None


def _split_dense_building_shape(shape, mask_src, mask, transform, min_area_px):
    """Split a connected settlement mask when separated roof cores exist.

    Semantic masks often merge adjacent small homes along one-pixel contacts.
    Eroding the merged component exposes independent roof cores; each core is
    dilated back only within the original component.  A split is accepted only
    when at least two substantial cores explain most of the source component,
    so fields and unresolved mega-regions remain candidates for rejection.
    """
    try:
        from scipy.ndimage import binary_dilation, binary_erosion, label

        component = rasterize(
            [(shape, 1)],
            out_shape=mask.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool) & (mask == BUILDING)
        component_pixels = int(component.sum())
        if component_pixels < max(96, int(min_area_px * 8)):
            return [shape]

        core = binary_erosion(component, structure=np.ones((3, 3), dtype=bool), iterations=1)
        labels, count = label(core, structure=np.ones((3, 3), dtype=np.uint8))
        if count < 2:
            return [shape]

        min_core = max(3, int(min_area_px * 0.35))
        pieces = []
        covered = np.zeros_like(component, dtype=bool)
        for index in range(1, count + 1):
            core_piece = labels == index
            if int(core_piece.sum()) < min_core:
                continue
            piece = binary_dilation(core_piece, structure=np.ones((3, 3), dtype=bool), iterations=1) & component
            if int(piece.sum()) < min_area_px:
                continue
            covered |= piece
            for geometry, value in shapes(
                piece.astype(np.uint8),
                mask=piece,
                transform=transform,
                connectivity=8,
            ):
                if int(value) == 1:
                    pieces.append(geometry)

        if len(pieces) < 2 or int(covered.sum()) < int(component_pixels * 0.65):
            return [shape]
        return pieces
    except (ImportError, TypeError, ValueError, rasterio.errors.RasterioIOError):
        return [shape]


def _class_regions(mask_path: Path, dsm_path: Path, *, raster_path: Path | None = None, max_regions=300, min_area_px=6):
    regions = []
    building_candidates = []
    rejected_buildings = {}
    source_rgb = None
    if raster_path and Path(raster_path).exists():
        try:
            with rasterio.open(raster_path) as source:
                source_rgb = _normalise_rgb(source.read())
        except (OSError, ValueError, rasterio.errors.RasterioIOError):
            source_rgb = None
    with rasterio.open(mask_path) as mask_src, rasterio.open(dsm_path) as dsm:
        if mask_src.width != dsm.width or mask_src.height != dsm.height:
            return regions, building_candidates, rejected_buildings
        mask = mask_src.read(1)
        transform = mask_src.transform
        pixel_area = abs(float(dsm.res[0] * dsm.res[1]))
        min_area_m2 = max(4.0, pixel_area * 6.0)
        simplify_tolerance = min(1.25, max(0.25, min(abs(float(dsm.res[0])), abs(float(dsm.res[1]))) * 0.35))
        evidence_rgb = source_rgb if source_rgb is not None and source_rgb.shape[:2] == mask.shape else np.full((mask_src.height, mask_src.width, 3), 0.5, dtype=np.float32)
        for source_shape, value in shapes(mask, mask=mask != OTHER, transform=transform, connectivity=8):
            class_id = int(value)
            if class_id == OTHER or class_id not in NAMES:
                continue
            shapes_to_process = (
                _split_dense_building_shape(source_shape, mask_src, mask, transform, min_area_px)
                if class_id == BUILDING
                else [source_shape]
            )
            for shape in shapes_to_process:
                ring = shape.get("coordinates", [[]])[0]
                if len(ring) < 4:
                    continue
                _projected_raw, pixels_raw = _project_ring(ring[:-1], mask_src)
                pixels = _simplify_ring(pixels_raw, tolerance=simplify_tolerance)
                if len(pixels) < 3 or _area(pixels) < min_area_px or _area(pixels) * pixel_area < min_area_m2:
                    continue
                projected = [list(map(float, mask_src.transform * (point[0], point[1]))) for point in pixels]
                ground = _sample_ground(dsm, projected)
                if ground is None:
                    ground = float(dsm.read(1).mean())
                region = {
                    "id": len(regions) + 1,
                    "class_id": class_id,
                    "class": NAMES[class_id],
                    "polygon": pixels,
                    "polygon_projected": projected,
                    "ground_elevation": float(ground),
                    "source": "semantic-segmentation",
                    "confidence": 0.55,
                    "geometry_quality": "cleaned-raster-component" if len(shapes_to_process) == 1 else "split-roof-component",
                }
                regions.append(region)
                if class_id == BUILDING:
                    evidence = _building_evidence(dsm, dsm, evidence_rgb, projected, semantic=True, max_area_m2=2000.0)
                    if not evidence.get("accepted"):
                        reason = evidence.get("reason", "insufficient-roof-evidence")
                        rejected_buildings[reason] = rejected_buildings.get(reason, 0) + 1
                        continue
                    sampled_ground = evidence.get("ground_elevation")
                    roof = evidence.get("roof_elevation")
                    if sampled_ground is not None:
                        ground = float(sampled_ground)
                    measured_height = float(roof) - float(ground)
                    height = max(2.2 if _area(projected) <= 20.0 else 2.5, min(45.0, measured_height))
                    building_candidates.append({
                        "id": 900000000 + len(building_candidates),
                        "name": "Segmented building",
                        "polygon": pixels,
                        "polygon_projected": projected,
                        "area_m2": _area(projected),
                        "ground_elevation": float(ground),
                        "height": height,
                        "roof_elevation": float(ground) + height,
                        "height_source": "segmentation:dsm-sampled",
                        "osm_tags": {"building": "segmentation"},
                        "segmentation_class": NAMES[class_id],
                        "geometry_profile": "flat",
                        "roof_type": "approximate",
                        "roof_plane_count": 0,
                        "profile_source": "semantic-footprint",
                        "roof_surface_source": "pending-dsm-quality-gate",
                        "roof_valid_fraction": 0.0,
                        "geometry_quality": "candidate",
                        "height_confidence": round(min(0.92, max(0.35, 0.55 + min(measured_height, 20.0) / 100.0)), 3),
                        "source": "semantic-segmentation",
                        "confidence": 0.55,
                        "roof_evidence": {key: value for key, value in evidence.items() if key not in {"accepted", "reason"}},
                    })
                if len(regions) >= max_regions:
                    break
            if len(regions) >= max_regions:
                break
    return regions, building_candidates, rejected_buildings


def add_semantic_features(environment: dict[str, Any], buildings: list[dict[str, Any]], mask_path: Path | None, dsm_path: Path, *, raster_path: Path | None = None, max_regions=300):
    """Attach semantic polygons and conservatively add missing scene layers.

    OSM is preferred when present, but semantic evidence must still produce
    explicit water/vegetation layers when OSM is unavailable.  Those layers
    are exported as separate physical/diagnostic layers and used to place
    bounded tree proxies. Semantic evidence never directly flattens terrain,
    preventing canopy/roof masks from becoming a giant overlapping hill.
    """
    environment = {**(environment or {})}
    environment.setdefault("semantic_regions", [])
    environment.setdefault("exclusion_zones", [])
    if not mask_path or not mask_path.exists():
        return environment, buildings, {"regions_count": 0, "building_candidates": 0, "approximate_tree_count": 0, "rejected_buildings": {}}
    regions, candidates, rejected_buildings = _class_regions(mask_path, dsm_path, raster_path=raster_path, max_regions=max_regions)
    environment["semantic_regions"] = regions

    environment.setdefault("water", [])
    environment.setdefault("landcover", [])
    environment.setdefault("trees", [])
    # OSM remains authoritative when present, but semantic regions must fill
    # every meaningful gap rather than only the first water/vegetation blob.
    semantic_water_added = 0
    semantic_vegetation_added = 0

    def _inside(point, polygon):
        if len(polygon) < 3:
            return False
        x, y = point
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = current
            x2, y2 = previous
            if (y1 > y) != (y2 > y) and abs(y2 - y1) > 1e-12:
                if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                    inside = not inside
            previous = current
        return inside

    def _overlaps_existing(polygon, features):
        if len(polygon) < 3:
            return False
        center = (
            sum(float(point[0]) for point in polygon) / len(polygon),
            sum(float(point[1]) for point in polygon) / len(polygon),
        )
        for feature in features or []:
            other = feature.get("polygon") or []
            if len(other) < 3:
                continue
            if _inside(center, other):
                return True
            other_center = (
                sum(float(point[0]) for point in other) / len(other),
                sum(float(point[1]) for point in other) / len(other),
            )
            if _inside(other_center, polygon):
                return True
            if any(_inside((float(point[0]), float(point[1])), other) for point in polygon):
                return True
        return False

    try:
        with rasterio.open(dsm_path) as dsm:
            step_x = max(6, int(round(12.0 / max(abs(float(dsm.res[0])), 1e-6))))
            step_y = max(6, int(round(12.0 / max(abs(float(dsm.res[1])), 1e-6))))
            ordered_regions = sorted(
                regions,
                key=lambda item: _area(item.get("polygon_projected") or []),
                reverse=True,
            )
            for region in ordered_regions:
                klass = region.get("class")
                polygon = region.get("polygon") or []
                projected = region.get("polygon_projected") or []
                if len(polygon) < 3 or len(projected) < 3:
                    continue
                if klass == "water" and not _overlaps_existing(polygon, environment["water"]) and semantic_water_added < 24:
                    environment["water"].append({
                        "id": -700000 - int(region["id"]),
                        "polygon": polygon,
                        "polygon_projected": projected,
                        "ground_elevation": region.get("ground_elevation"),
                        "class": "water",
                        "name": "Semantic water region",
                        "source": "semantic-segmentation",
                        "confidence": 0.55,
                        "geometry_quality": region.get("geometry_quality", "cleaned-raster-component"),
                    })
                    semantic_water_added += 1
                elif klass == "vegetation" and not _overlaps_existing(polygon, environment["landcover"]) and semantic_vegetation_added < 32:
                    environment["landcover"].append({
                        "id": -710000 - int(region["id"]),
                        "polygon": polygon,
                        "polygon_projected": projected,
                        "ground_elevation": region.get("ground_elevation"),
                        "class": "forest",
                        "name": "Semantic vegetation region",
                        "source": "semantic-segmentation",
                        "confidence": 0.55,
                        "geometry_quality": region.get("geometry_quality", "cleaned-raster-component"),
                    })
                    semantic_vegetation_added += 1
                    # A bounded point population makes a vegetation region
                    # readable in 3D without pretending individual tree crowns
                    # are measured from one nadir image.
                    if len(environment["trees"]) < 500 and _area(projected) >= 80.0:
                        min_x = max(0, int(min(point[0] for point in polygon)))
                        max_x = min(dsm.width - 1, int(max(point[0] for point in polygon)))
                        min_y = max(0, int(min(point[1] for point in polygon)))
                        max_y = min(dsm.height - 1, int(max(point[1] for point in polygon)))
                        for row in range(min_y, max_y + 1, step_y):
                            for col in range(min_x, max_x + 1, step_x):
                                if len(environment["trees"]) >= 500:
                                    break
                                if not _inside((col + 0.5, row + 0.5), polygon):
                                    continue
                                projected_point = list(map(float, dsm.transform * (col + 0.5, row + 0.5)))
                                # Deterministic variation keeps a vegetation
                                # region readable without claiming individual
                                # tree crowns were measured.
                                seed = ((col * 73856093) ^ (row * 19349663)) & 0xFF
                                tree_height = 5.5 + (seed / 255.0) * 5.5
                                environment["trees"].append({
                                    "id": -720000 - len(environment["trees"]),
                                    "point": [float(col + 0.5), float(row + 0.5)],
                                    "point_projected": projected_point,
                                    "ground_elevation": float(region.get("ground_elevation") or 0.0),
                                    "height": round(tree_height, 2),
                                    "canopy_radius": round(1.6 + (seed / 255.0) * 1.8, 2),
                                    "height_source": "semantic:approximate-context",
                                    "source": "semantic-segmentation:inferred-tree",
                                    "confidence": 0.4,
                                    "geometry_quality": "approximate-context",
                                    "name": "Estimated semantic tree",
                                })
                            if len(environment["trees"]) >= 500:
                                break
    except (OSError, ValueError, rasterio.errors.RasterioIOError):
        pass

    # OSM is the preferred footprint source. Semantic buildings fill only
    # unmapped gaps, preventing duplicate objects over each footprint. The
    # overlap check is done on the source raster grid, so CRS/rounding
    # differences cannot create duplicate scene objects.
    existing = buildings or []
    try:
        with rasterio.open(mask_path) as mask_src:
            grid_shape = (mask_src.height, mask_src.width)
    except (OSError, rasterio.errors.RasterioIOError):
        grid_shape = None

    def _grid_mask(polygon):
        if grid_shape is None or len(polygon) < 3:
            return None
        ring = [list(map(float, point)) for point in polygon]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        try:
            return rasterize(
                [({"type": "Polygon", "coordinates": [ring]}, 1)],
                out_shape=grid_shape,
                transform=Affine.identity(),
                fill=0,
                default_value=1,
                all_touched=True,
                dtype="uint8",
            ).astype(bool)
        except (TypeError, ValueError, rasterio.errors.RasterioIOError):
            return None

    for candidate in candidates:
        candidate_polygon = candidate.get("polygon") or []
        if len(candidate_polygon) < 3:
            continue
        cx = sum(point[0] for point in candidate_polygon) / len(candidate_polygon)
        cy = sum(point[1] for point in candidate_polygon) / len(candidate_polygon)
        candidate_mask = _grid_mask(candidate_polygon)
        duplicate = False
        for item in existing:
            item_polygon = item.get("polygon") or []
            item_mask = _grid_mask(item_polygon)
            if candidate_mask is not None and item_mask is not None:
                overlap = np.logical_and(candidate_mask, item_mask).sum()
                candidate_area = max(int(candidate_mask.sum()), 1)
                item_area = max(int(item_mask.sum()), 1)
                if float(overlap) / min(candidate_area, item_area) >= 0.35:
                    duplicate = True
                    break
            elif _point_in_bbox(cx, cy, item_polygon):
                duplicate = True
                break
        if not duplicate:
            existing.append(candidate)
    semantic_tree_count = sum(
        1 for tree in environment.get("trees") or []
        if str(tree.get("source", "")).startswith("semantic-")
    )
    return environment, existing, {
        "regions_count": len(regions),
        "building_candidates": len(candidates),
        "approximate_tree_count": semantic_tree_count,
        "rejected_buildings": rejected_buildings,
    }


def _point_in_bbox(x, y, polygon):
    if len(polygon) < 3:
        return False
    return min(point[0] for point in polygon) <= x <= max(point[0] for point in polygon) and min(point[1] for point in polygon) <= y <= max(point[1] for point in polygon)
