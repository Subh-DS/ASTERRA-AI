"""Backend hazard simulation service with physical plausibility constraints.

Computes landslide susceptibility and coastal inundation using terrain
analysis, hydrological connectivity, and physical constraints. Every pixel
is evaluated — not just AI predictions filtered after the fact.
"""

import base64
import heapq
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage


def _encode_grid(grid: np.ndarray) -> str:
    """Encode a numpy array as base64 string."""
    return base64.b64encode(grid.tobytes()).decode("ascii")


def _grid_metadata(dsm_path: Path) -> dict[str, Any]:
    """Read DSM metadata for coordinate reference."""
    with rasterio.open(dsm_path) as src:
        return {
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6] if src.transform else None,
            "bounds": {
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
            },
            "width": src.width,
            "height": src.height,
            "resolution": [float(src.res[0]), float(src.res[1])],
        }


def _compute_slope(dsm: np.ndarray, resolution: tuple[float, float]) -> np.ndarray:
    """Compute slope in degrees from DSM using gradient."""
    dy, dx = np.gradient(dsm, resolution[1], resolution[0])
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope


def _compute_aspect(dsm: np.ndarray, resolution: tuple[float, float]) -> np.ndarray:
    """Compute aspect in degrees from DSM."""
    dy, dx = np.gradient(dsm, resolution[1], resolution[0])
    aspect = np.degrees(np.arctan2(-dx, dy))
    return aspect


def _compute_curvature(dsm: np.ndarray, resolution: tuple[float, float]) -> np.ndarray:
    """Compute profile curvature from DSM."""
    dy, dx = np.gradient(dsm, resolution[1], resolution[0])
    dyy, dyx = np.gradient(dy, resolution[1], resolution[0])
    dxy, dxx = np.gradient(dx, resolution[1], resolution[0])
    # Profile curvature (curvature in the direction of steepest descent)
    curvature = -((dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy) / ((dx**2 + dy**2) * np.sqrt(dx**2 + dy**2) + 1e-10))
    return curvature


def _compute_local_relief(dsm: np.ndarray, size: int = 5) -> np.ndarray:
    """Compute local relief (max - min in neighborhood)."""
    from scipy.ndimage import maximum_filter, minimum_filter
    local_max = maximum_filter(dsm, size=size, mode="nearest")
    local_min = minimum_filter(dsm, size=size, mode="nearest")
    return local_max - local_min


def _compute_drainage_proximity(dsm: np.ndarray) -> np.ndarray:
    """Compute drainage proximity using flow accumulation approximation."""
    # Simple flow accumulation: count cells that flow into each cell
    H, W = dsm.shape
    flow = np.ones((H, W), dtype=np.float32)
    # Sort cells by elevation (lowest first)
    flat_idx = np.argsort(dsm.ravel())
    for idx in flat_idx:
        r, c = divmod(int(idx), W)
        if not np.isfinite(dsm[r, c]):
            continue
        # Check neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and np.isfinite(dsm[nr, nc]):
                if dsm[nr, nc] < dsm[r, c]:
                    flow[nr, nc] += flow[r, c]
    # Normalize
    flow = np.log1p(flow)
    flow = flow / (flow.max() + 1e-10)
    return flow


def _compute_twi(dsm: np.ndarray, resolution: tuple[float, float]) -> np.ndarray:
    """Compute Topographic Wetness Index (TWI)."""
    slope = _compute_slope(dsm, resolution)
    slope_rad = np.radians(np.maximum(slope, 0.1))
    # Approximate flow accumulation
    flow = _compute_drainage_proximity(dsm)
    twi = np.log((flow + 0.01) / np.tan(slope_rad))
    return twi


def _normalize(data: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Normalize array to [min_val, max_val] range."""
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros_like(data)
    dmin, dmax = data[valid].min(), data[valid].max()
    if dmax - dmin < 1e-10:
        return np.full_like(data, (min_val + max_val) / 2)
    normalized = (data - dmin) / (dmax - dmin)
    return normalized * (max_val - min_val) + min_val


def simulate_landslide_susceptibility(
    dsm_path: Path,
    mask_path: Path | None = None,
    rainfall_mm: float | None = None,
    stride: int = 4,
) -> dict[str, Any]:
    """Compute landslide susceptibility for every pixel using physical factors.

    Factors:
        - Slope (primary factor)
        - Elevation
        - Curvature (profile)
        - Local relief
        - Aspect
        - Drainage proximity / flow accumulation
        - Topographic Wetness Index (TWI)
        - Land cover (from segmentation mask if available)
        - Rainfall (if provided)

    Physical plausibility filter:
        - Flat areas (slope < 5°) → strongly suppressed
        - Very steep areas (slope > 60°) → rockfall, not landslide
        - Water bodies → excluded

    Args:
        dsm_path: Path to the metric DSM GeoTIFF
        mask_path: Optional path to semantic segmentation mask
        rainfall_mm: Optional rainfall in mm for triggering conditions
        stride: Grid stride for downsampling

    Returns:
        Susceptibility map with per-pixel probability and statistics
    """
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1, masked=True).astype(np.float32)
        # Properly handle nodata: fill masked values with NaN
        dsm = dsm.filled(np.nan)
        transform = src.transform
        crs = str(src.crs) if src.crs else None
        bounds = {
            "left": float(src.bounds.left),
            "bottom": float(src.bounds.bottom),
            "right": float(src.bounds.right),
            "top": float(src.bounds.top),
        }
        resolution = (float(src.res[0]), float(src.res[1]))

    # Downsample
    if stride > 1:
        dsm = dsm[::stride, ::stride]
        # Fix: scale transform by stride (not 1/stride) to maintain geographic alignment
        transform = transform * transform.scale(stride, stride)
        resolution = (resolution[0] * stride, resolution[1] * stride)

    H, W = dsm.shape
    valid = np.isfinite(dsm) & (dsm > -9990)

    # Compute terrain factors
    slope = _compute_slope(dsm, resolution)
    aspect = _compute_aspect(dsm, resolution)
    curvature = _compute_curvature(dsm, resolution)
    local_relief = _compute_local_relief(dsm, size=5)
    drainage = _compute_drainage_proximity(dsm)
    twi = _compute_twi(dsm, resolution)

    # Load land cover mask if available
    land_cover_risk = np.ones((H, W), dtype=np.float32)
    if mask_path and Path(mask_path).exists():
        try:
            with rasterio.open(mask_path) as mask_src:
                # Validate CRS matches DSM CRS
                if mask_src.crs and crs and str(mask_src.crs) != crs:
                    # Reproject mask to DSM CRS if needed
                    from rasterio.warp import reproject, Resampling
                    mask_data = mask_src.read(1)
                    reprojected = np.zeros((H, W), dtype=mask_data.dtype)
                    reproject(
                        source=mask_data,
                        destination=reprojected,
                        src_transform=mask_src.transform,
                        src_crs=mask_src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.nearest,
                    )
                    mask = reprojected
                elif mask_src.width == W and mask_src.height == H:
                    mask = mask_src.read(1)
                else:
                    # Resample mask to match DSM grid
                    from rasterio.warp import reproject, Resampling
                    mask_data = mask_src.read(1)
                    reprojected = np.zeros((H, W), dtype=mask_data.dtype)
                    reproject(
                        source=mask_data,
                        destination=reprojected,
                        src_transform=mask_src.transform,
                        src_crs=mask_src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.nearest,
                    )
                    mask = reprojected

                # Vegetation reduces risk, bare ground increases it
                # ADE20K classes: 0=other, 1=building, 2=vegetation, 3=water, 4=bare
                land_cover_risk = np.where(mask == 2, 0.6, land_cover_risk)  # vegetation
                land_cover_risk = np.where(mask == 4, 1.3, land_cover_risk)  # bare ground
                land_cover_risk = np.where(mask == 3, 0.0, land_cover_risk)  # water (excluded)
                land_cover_risk = np.where(mask == 1, 0.3, land_cover_risk)  # buildings
        except Exception:
            pass

    # Normalize factors to [0, 1]
    slope_norm = _normalize(slope, 0, 1)
    relief_norm = _normalize(local_relief, 0, 1)
    drainage_norm = _normalize(drainage, 0, 1)
    twi_norm = _normalize(twi, 0, 1)
    curvature_norm = _normalize(np.abs(curvature), 0, 1)

    # Elevation factor: mid-elevations more susceptible
    elev_norm = _normalize(dsm, 0, 1)
    elev_factor = 1.0 - np.abs(elev_norm - 0.5) * 2  # peak at mid-elevation

    # Weighted susceptibility score
    # Slope is the primary factor (40%)
    # Local relief (20%), drainage (15%), TWI (10%), curvature (10%), elevation (5%)
    susceptibility = (
        slope_norm * 0.40 +
        relief_norm * 0.20 +
        drainage_norm * 0.15 +
        twi_norm * 0.10 +
        curvature_norm * 0.10 +
        elev_factor * 0.05
    )

    # Apply land cover modifier
    susceptibility = susceptibility * land_cover_risk

    # Apply rainfall trigger if provided
    if rainfall_mm is not None and rainfall_mm > 0:
        # Rainfall increases susceptibility (sigmoid function)
        rain_factor = 1.0 / (1.0 + np.exp(-0.1 * (rainfall_mm - 50)))
        susceptibility = susceptibility * (0.5 + 0.5 * rain_factor)

    # Physical plausibility filter with configurable thresholds
    # These thresholds are based on geomorphological literature:
    # - Landslides rarely occur on slopes < 5° (too flat for gravity-driven movement)
    # - Slopes > 60° are typically rockfall zones, not landslides
    # - Areas with very low local relief are unlikely to generate landslides
    config = {
        "min_slope_deg": 5.0,      # Minimum slope for landslide susceptibility
        "max_slope_deg": 60.0,     # Maximum slope (above = rockfall)
        "min_relief_m": 2.0,       # Minimum local relief for landslide
        "flat_suppression": 0.01,  # Suppression factor for flat areas (very aggressive)
        "steep_suppression": 0.1,  # Suppression factor for very steep areas
    }

    # 1. Flat areas → strongly suppressed (near-zero)
    slope_mask = slope < config["min_slope_deg"]
    susceptibility[slope_mask] *= config["flat_suppression"]

    # 2. Very steep areas → rockfall, not landslide
    steep_mask = slope > config["max_slope_deg"]
    susceptibility[steep_mask] *= config["steep_suppression"]

    # 3. Low relief areas → suppressed
    relief_mask = local_relief < config["min_relief_m"]
    susceptibility[relief_mask] *= 0.3

    # 4. Water bodies → excluded (use land cover mask if available)
    if mask_path and Path(mask_path).exists():
        water_mask = (mask == 3) & valid
    else:
        # Fallback: lowest 5% of elevations
        water_mask = dsm < np.percentile(dsm[valid], 5) if valid.any() else np.zeros((H, W), bool)
    susceptibility[water_mask] = 0.0

    # 5. Invalid cells → excluded
    susceptibility[~valid] = 0.0

    # Clip to [0, 1] range (no re-normalization to preserve suppression)
    susceptibility = np.clip(susceptibility, 0.0, 1.0)

    # Classify susceptibility levels
    low = (susceptibility < 0.2) & valid
    moderate = (susceptibility >= 0.2) & (susceptibility < 0.4) & valid
    high = (susceptibility >= 0.4) & (susceptibility < 0.6) & valid
    very_high = (susceptibility >= 0.6) & valid

    # Find high susceptibility zones for potential landslide sources
    high_susceptibility_mask = (susceptibility >= 0.5) & valid

    # If no high susceptibility areas, use the top 5% of valid cells
    if not high_susceptibility_mask.any() and valid.any():
        threshold = np.percentile(susceptibility[valid], 95)
        high_susceptibility_mask = (susceptibility >= threshold) & valid

    # Select source points from high susceptibility areas
    source_points = []
    if high_susceptibility_mask.any():
        # Find local maxima in high susceptibility areas
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(susceptibility, size=5, mode="nearest")
        maxima = (susceptibility == local_max) & high_susceptibility_mask
        max_indices = np.argwhere(maxima)
        # Sort by susceptibility value
        max_indices = sorted(max_indices, key=lambda idx: susceptibility[idx[0], idx[1]], reverse=True)
        # Take top 3 source points
        for idx in max_indices[:3]:
            source_points.append({
                "row": int(idx[0]),
                "col": int(idx[1]),
                "susceptibility": round(float(susceptibility[idx[0], idx[1]]), 3),
                "slope_deg": round(float(slope[idx[0], idx[1]]), 1),
            })

    # Encode grids
    grids = {
        "grid_h": H,
        "grid_w": W,
        "grid_stride": stride,
        "susceptibility_b64": _encode_grid(susceptibility.astype(np.float32)),
        "slope_b64": _encode_grid(slope.astype(np.float32)),
        "mask_b64": _encode_grid(high_susceptibility_mask.astype(np.uint8)),
        "source_b64": _encode_grid(np.zeros((H, W), dtype=np.uint8)),
        "deposition_b64": _encode_grid(np.zeros((H, W), np.uint8)),
        "path_mask_b64": _encode_grid(np.zeros((H, W), np.uint8)),
    }

    # Statistics
    valid_susceptibility = susceptibility[valid]
    stats = {
        "total_cells": int(valid.sum()),
        "low_susceptibility_cells": int(low.sum()),
        "moderate_susceptibility_cells": int(moderate.sum()),
        "high_susceptibility_cells": int(high.sum()),
        "very_high_susceptibility_cells": int(very_high.sum()),
        "mean_susceptibility": round(float(valid_susceptibility.mean()), 3) if valid_susceptibility.size else 0,
        "max_susceptibility": round(float(valid_susceptibility.max()), 3) if valid_susceptibility.size else 0,
        "high_susceptibility_percentage": round((high.sum() + very_high.sum()) / max(valid.sum(), 1) * 100, 1),
        "source_points": source_points,
        "factors": {
            "slope_mean_deg": round(float(slope[valid].mean()), 1) if valid.any() else 0,
            "slope_max_deg": round(float(slope[valid].max()), 1) if valid.any() else 0,
            "local_relief_mean_m": round(float(local_relief[valid].mean()), 1) if valid.any() else 0,
            "twi_mean": round(float(twi[valid].mean()), 2) if valid.any() else 0,
        },
    }

    return {
        "simulation_type": "landslide_susceptibility",
        "grids": grids,
        "statistics": stats,
        "terrain_source": {
            "grid": [H, W],
            "crs": crs,
            "bounds": bounds,
            "resolution": list(resolution),
        },
        "polyline_px": [],
        "keyframes": [],
    }


def simulate_coastal_inundation(
    dsm_path: Path,
    water_level_m: float,
    stride: int = 4,
) -> dict[str, Any]:
    """Simulate coastal inundation with hydrological connectivity.

    Uses a flood-fill approach that respects hydrological connectivity:
    water can only reach cells that are connected to the coast (or to
    each other) through a continuous path of cells below the water level.

    Args:
        dsm_path: Path to the metric DSM GeoTIFF
        water_level_m: Water level in meters above the vertical datum
        stride: Grid stride for downsampling

    Returns:
        Simulation result with grids, statistics, and metadata
    """
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1, masked=True).astype(np.float32)
        transform = src.transform
        crs = str(src.crs) if src.crs else None
        bounds = {
            "left": float(src.bounds.left),
            "bottom": float(src.bounds.bottom),
            "right": float(src.bounds.right),
            "top": float(src.bounds.top),
        }
        resolution = (float(src.res[0]), float(src.res[1]))

    # Downsample
    if stride > 1:
        dsm = dsm[::stride, ::stride]
        transform = transform * transform.scale(1.0 / stride, 1.0 / stride)
        resolution = (resolution[0] * stride, resolution[1] * stride)

    H, W = dsm.shape
    valid = np.isfinite(dsm) & (dsm > -9990)

    # Hydrological connectivity: flood fill from coastline
    # Water can only reach cells connected to the coast through
    # a continuous path of cells below water level.
    # Coastline is detected as the boundary between valid land and nodata/invalid areas,
    # or the lowest elevation cells at the map edge.
    mask = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)

    from collections import deque
    queue = deque()

    # Detect coastline: edge cells that are valid and below water level
    # These represent where water can enter the land
    coastline_cells = []
    for r in range(H):
        for c in [0, W - 1]:
            if valid[r, c] and dsm[r, c] < water_level_m:
                coastline_cells.append((r, c))
    for c in range(W):
        for r in [0, H - 1]:
            if valid[r, c] and dsm[r, c] < water_level_m:
                coastline_cells.append((r, c))

    # If no coastline detected (e.g., entire map is land), use lowest elevation cells
    if not coastline_cells and valid.any():
        # Find cells within 10% of minimum elevation at edges
        edge_elevations = []
        for r in range(H):
            for c in [0, W - 1]:
                if valid[r, c]:
                    edge_elevations.append(dsm[r, c])
        for c in range(W):
            for r in [0, H - 1]:
                if valid[r, c]:
                    edge_elevations.append(dsm[r, c])
        if edge_elevations:
            min_edge = min(edge_elevations)
            threshold = min_edge + (water_level_m - min_edge) * 0.5
            for r in range(H):
                for c in [0, W - 1]:
                    if valid[r, c] and dsm[r, c] < threshold:
                        coastline_cells.append((r, c))
            for c in range(W):
                for r in [0, H - 1]:
                    if valid[r, c] and dsm[r, c] < threshold:
                        coastline_cells.append((r, c))

    # BFS flood fill from coastline
    for cell in coastline_cells:
        if not visited[cell[0], cell[1]]:
            queue.append(cell)
            visited[cell[0], cell[1]] = True

    while queue:
        r, c = queue.popleft()
        mask[r, c] = True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                if valid[nr, nc] and dsm[nr, nc] < water_level_m:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    # Calculate depth
    depth = np.where(mask, water_level_m - dsm, 0.0).astype(np.float32)

    # Statistics
    inundated_count = int(mask.sum())
    total_valid = int(valid.sum())
    inundated_pct = (inundated_count / max(total_valid, 1)) * 100
    max_depth = float(depth.max()) if inundated_count > 0 else 0.0
    mean_depth = float(depth[mask].mean()) if inundated_count > 0 else 0.0

    # Find deepest point for camera focus
    if inundated_count > 0:
        deepest_idx = np.unravel_index(np.argmax(depth), depth.shape)
        deepest_rc = [int(deepest_idx[0]), int(deepest_idx[1])]
    else:
        deepest_rc = [H // 2, W // 2]

    # Encode grids
    grids = {
        "grid_h": H,
        "grid_w": W,
        "grid_stride": stride,
        "mask_b64": _encode_grid(mask.astype(np.uint8)),
        "depth_b64": _encode_grid(depth),
    }

    return {
        "simulation_type": "coastal_inundation",
        "grids": grids,
        "statistics": {
            "water_level_m": float(water_level_m),
            "inundated_cells": inundated_count,
            "total_valid_cells": total_valid,
            "inundated_percentage": round(inundated_pct, 2),
            "max_depth_m": round(max_depth, 2),
            "mean_depth_m": round(mean_depth, 2),
            "deepest_point_rc": deepest_rc,
        },
        "terrain_source": {
            "grid": [H, W],
            "crs": crs,
            "bounds": bounds,
            "resolution": list(resolution),
        },
        "polyline_px": [],
        "keyframes": [],
    }


def simulate_landslide(
    dsm_path: Path,
    source_row: int | None = None,
    source_col: int | None = None,
    intensity: float = 0.5,
    stride: int = 4,
) -> dict[str, Any]:
    """Simulate a landslide from a source point on the DSM.

    Uses slope-based flow routing: material moves downhill along the
    steepest descent path until it reaches a flat area or the edge.

    Args:
        dsm_path: Path to the metric DSM GeoTIFF
        source_row: Source row (None = auto-detect from susceptibility)
        source_col: Source column (None = auto-detect from susceptibility)
        intensity: Landslide intensity (0-1)
        stride: Grid stride for downsampling

    Returns:
        Simulation result with grids, keyframes, and statistics
    """
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1, masked=True).astype(np.float32)
        transform = src.transform
        crs = str(src.crs) if src.crs else None
        bounds = {
            "left": float(src.bounds.left),
            "bottom": float(src.bounds.bottom),
            "right": float(src.bounds.right),
            "top": float(src.bounds.top),
        }
        resolution = (float(src.res[0]), float(src.res[1]))

    # Downsample
    if stride > 1:
        dsm = dsm[::stride, ::stride]
        # Fix: scale transform by stride to maintain geographic alignment
        transform = transform * transform.scale(stride, stride)
        resolution = (resolution[0] * stride, resolution[1] * stride)

    H, W = dsm.shape
    valid = np.isfinite(dsm) & (dsm > -9990)

    # Compute slope for source detection
    slope = _compute_slope(dsm, resolution)

    # Auto-detect source: find the steepest point with high susceptibility
    if source_row is None or source_col is None:
        # Mask out flat areas and very steep areas
        valid_source = valid & (slope > 5) & (slope < 60)
        if valid_source.any():
            # Find steepest valid point
            source_row, source_col = np.unravel_index(
                np.argmax(np.where(valid_source, slope, 0)), slope.shape
            )
            source_row, source_col = int(source_row), int(source_col)
        else:
            source_row, source_col = H // 2, W // 2

    # Flow routing: follow steepest descent
    path = [(source_row, source_col)]
    visited = set()
    current = (source_row, source_col)
    max_steps = min(H * W // 4, 2000)

    for _ in range(max_steps):
        visited.add(current)
        r, c = current
        current_elev = dsm[r, c] if valid[r, c] else np.inf

        best_neighbor = None
        best_slope = 0.0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if not valid[nr, nc] or (nr, nc) in visited:
                    continue
                neighbor_elev = dsm[nr, nc]
                dist = math.sqrt(dr**2 + dc**2)
                slope_val = (current_elev - neighbor_elev) / dist
                if slope_val > best_slope:
                    best_slope = slope_val
                    best_neighbor = (nr, nc)

        if best_neighbor is None or best_slope < 0.01:
            break
        path.append(best_neighbor)
        current = best_neighbor

    # Create source mask (ellipse around source)
    source_mask = np.zeros((H, W), dtype=np.uint8)
    radius = max(3, int(min(H, W) * 0.03 * intensity))
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr**2 + dc**2 <= radius**2:
                r, c = source_row + dr, source_col + dc
                if 0 <= r < H and 0 <= c < W:
                    source_mask[r, c] = 1

    # Create deposition mask (along the path)
    deposition_mask = np.zeros((H, W), dtype=np.uint8)
    for i, (r, c) in enumerate(path):
        dep_radius = max(2, int(radius * (1 - i / max(len(path), 1)) * 0.5))
        for dr in range(-dep_radius, dep_radius + 1):
            for dc in range(-dep_radius, dep_radius + 1):
                if dr**2 + dc**2 <= dep_radius**2:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        deposition_mask[nr, nc] = 1

    # Create keyframes for animation
    keyframes = []
    n_keyframes = min(8, max(3, len(path) // 10))
    for i in range(n_keyframes):
        idx = int(i * (len(path) - 1) / max(n_keyframes - 1, 1))
        r, c = path[idx]
        keyframes.append({
            "row": r,
            "col": c,
            "radius_px": max(3, int(radius * (1 - i / n_keyframes))),
        })

    polyline = [[r, c] for r, c in path]

    source_elev = float(dsm[source_row, source_col]) if valid[source_row, source_col] else 0.0
    end_elev = float(dsm[path[-1][0], path[-1][1]]) if valid[path[-1][0], path[-1][1]] else 0.0
    elevation_drop = source_elev - end_elev

    grids = {
        "grid_h": H,
        "grid_w": W,
        "grid_stride": stride,
        "source_b64": _encode_grid(source_mask),
        "deposition_b64": _encode_grid(deposition_mask),
        "path_mask_b64": _encode_grid(np.zeros((H, W), dtype=np.uint8)),
    }

    return {
        "simulation_type": "landslide",
        "grids": grids,
        "statistics": {
            "source_center_rc": [source_row, source_col],
            "source_elevation_m": round(source_elev, 2),
            "end_elevation_m": round(end_elev, 2),
            "elevation_drop_m": round(elevation_drop, 2),
            "path_length_cells": len(path),
            "affected_cells": int(source_mask.sum() + deposition_mask.sum()),
        },
        "terrain_source": {
            "grid": [H, W],
            "crs": crs,
            "bounds": bounds,
            "resolution": list(resolution),
        },
        "polyline_px": polyline,
        "keyframes": keyframes,
    }


def simulate_evacuation(
    dsm_path: Path,
    start_row: int | None = None,
    start_col: int | None = None,
    stride: int = 4,
) -> dict[str, Any]:
    """Compute evacuation routes using A* pathfinding over the DSM."""
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1, masked=True).astype(np.float32)
        transform = src.transform
        crs = str(src.crs) if src.crs else None
        bounds = {
            "left": float(src.bounds.left),
            "bottom": float(src.bounds.bottom),
            "right": float(src.bounds.right),
            "top": float(src.bounds.top),
        }
        resolution = (float(src.res[0]), float(src.res[1]))

    if stride > 1:
        dsm = dsm[::stride, ::stride]
        # Fix: scale transform by stride to maintain geographic alignment
        transform = transform * transform.scale(stride, stride)
        resolution = (resolution[0] * stride, resolution[1] * stride)

    H, W = dsm.shape
    valid = np.isfinite(dsm) & (dsm > -9990)

    if start_row is None or start_col is None:
        min_elev = np.inf
        for r in range(H):
            for c in range(W):
                if valid[r, c] and dsm[r, c] < min_elev:
                    min_elev = dsm[r, c]
                    start_row, start_col = r, c
        start_row, start_col = int(start_row), int(start_col)

    elevations = dsm[valid]
    safe_threshold = np.percentile(elevations, 90) if len(elevations) > 0 else 0
    safe_mask = valid & (dsm >= safe_threshold)

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and valid[nr, nc]:
                slope = abs(dsm[nr, nc] - dsm[r, c])
                cost = 1.0 + slope * 2.0
                neighbors.append((nr, nc, cost))
        return neighbors

    safe_zones = []
    for r in range(H):
        for c in range(W):
            if safe_mask[r, c]:
                safe_zones.append((r, c))

    safe_zones.sort(key=lambda z: heuristic((start_row, start_col), z))
    safe_zones = safe_zones[:3]

    routes = []
    for goal in safe_zones:
        open_set = [(0, (start_row, start_col))]
        came_from = {}
        g_score = {(start_row, start_col): 0}
        closed = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                routes.append({
                    "pts": [[float(dsm[r, c]), float(r), float(c)] for r, c in path],
                    "length_m": len(path) * stride * resolution[0],
                    "elevation_gain_m": float(dsm[goal[0], goal[1]] - dsm[start_row, start_col]),
                })
                break

            if current in closed:
                continue
            closed.add(current)

            for nr, nc, cost in get_neighbors(*current):
                neighbor = (nr, nc)
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

    zones = []
    for r, c in safe_zones:
        zones.append({
            "x": float(c),
            "y": float(dsm[r, c]),
            "z": float(r),
            "r": max(3, int(min(H, W) * 0.02)),
        })

    return {
        "simulation_type": "evacuation",
        "grids": {
            "grid_h": H,
            "grid_w": W,
            "grid_stride": stride,
        },
        "statistics": {
            "start_rc": [start_row, start_col],
            "start_elevation_m": round(float(dsm[start_row, start_col]), 2),
            "safe_zones_found": len(safe_zones),
            "routes_computed": len(routes),
        },
        "terrain_source": {
            "grid": [H, W],
            "crs": crs,
            "bounds": bounds,
            "resolution": list(resolution),
        },
        "routes": routes,
        "zones": zones,
        "start": {
            "x": float(start_col),
            "y": float(dsm[start_row, start_col]),
            "z": float(start_row),
        },
        "polyline_px": [],
        "keyframes": [],
    }


def run_simulation(
    dsm_path: Path,
    simulation_type: str,
    parameters: dict[str, Any],
    mask_path: Path | None = None,
) -> dict[str, Any]:
    """Run a hazard simulation based on type and parameters.

    Args:
        dsm_path: Path to the metric DSM GeoTIFF
        simulation_type: One of 'coastal_inundation', 'landslide_susceptibility', 'landslide', 'evacuation'
        parameters: Simulation-specific parameters
        mask_path: Optional path to semantic segmentation mask

    Returns:
        Simulation result dictionary
    """
    dsm_path = Path(dsm_path)
    if not dsm_path.exists():
        raise FileNotFoundError(f"DSM not found: {dsm_path}")

    if simulation_type == "coastal_inundation":
        water_level = float(parameters.get("water_level_m", 3.0))
        return simulate_coastal_inundation(dsm_path, water_level)
    elif simulation_type == "landslide_susceptibility":
        rainfall = parameters.get("rainfall_mm")
        return simulate_landslide_susceptibility(dsm_path, mask_path, rainfall)
    elif simulation_type == "landslide":
        source_row = parameters.get("source_row")
        source_col = parameters.get("source_col")
        intensity = float(parameters.get("intensity", 0.5))
        return simulate_landslide(dsm_path, source_row, source_col, intensity)
    elif simulation_type == "evacuation":
        start_row = parameters.get("start_row")
        start_col = parameters.get("start_col")
        return simulate_evacuation(dsm_path, start_row, start_col)
    else:
        raise ValueError(f"Unknown simulation type: {simulation_type}")
