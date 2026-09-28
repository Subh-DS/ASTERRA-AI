"""DEM discovery, caching, and AWS Terrain Tiles acquisition.

The calibration engine consumes a normal georeferenced DEM.  This module is
responsible for finding or creating that DEM; it deliberately does not alter
the calibration mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.error
import urllib.request

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds


_SRTM_TILE_NAME = re.compile(r"^[ns]\d{2}_[ew]\d{3}", re.IGNORECASE)
_AWS_TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_dir}/{tile}.hgt.gz"
_WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class DemResolution:
    path: Path
    metadata: dict


def _provider_matches(path: Path, provider: str | None) -> bool:
    name = path.name.lower()
    provider = (provider or "auto").lower()
    if provider in {"", "auto", "local"}:
        return any(token in name for token in ("srtm", "aw3d", "glo", "copernicus", "dem")) or bool(_SRTM_TILE_NAME.match(name))
    aliases = {
        "aws_terrain": ("srtm", "terrain", "dem"),
        "srtm": ("srtm",),
        "aw3d30": ("aw3d",),
        "glo30": ("glo",),
        "copdem": ("copernicus",),
    }
    return any(token in name for token in aliases.get(provider, (provider,))) or (provider == "srtm" and bool(_SRTM_TILE_NAME.match(name)))


def _covers(candidate: Path, reference: Path) -> bool:
    with rasterio.open(reference) as ref, rasterio.open(candidate) as dem:
        if not ref.crs or not dem.crs:
            return False
        bounds = transform_bounds(ref.crs, dem.crs, *ref.bounds, densify_pts=21)
        return not (
            bounds[2] <= dem.bounds.left
            or bounds[0] >= dem.bounds.right
            or bounds[3] <= dem.bounds.bottom
            or bounds[1] >= dem.bounds.top
        )


def _vertical_reference(path: Path) -> str | None:
    try:
        with rasterio.open(path) as src:
            tags = {str(k).lower(): str(v) for k, v in src.tags().items()}
            for key in ("vertical_reference", "vertical_datum", "vert_crs", "geoid"):
                if tags.get(key):
                    return tags[key]
    except (OSError, rasterio.errors.RasterioIOError):
        pass
    return None


def _local_candidates(reference_path: Path, project_root: Path, explicit_path: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    calibration_dem = project_root / "calibration" / "dem"
    if calibration_dem.exists():
        candidates.extend(calibration_dem.rglob("*.tif"))
        candidates.extend(calibration_dem.rglob("*.tiff"))
    dataset_root = project_root / "datasets"
    if dataset_root.exists():
        for pattern in (
            "COPDEM_*/*.tif",
            "COPDEM_*/*.tiff",
            "**/*AW3D*.tif",
            "**/*AW3D*.tiff",
            "**/*GLO*.tif",
            "**/*GLO*.tiff",
        ):
            candidates.extend(dataset_root.glob(pattern))
    # Preserve ordering while avoiding duplicate paths.
    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in seen and path.resolve() != reference_path.resolve():
            seen.add(resolved)
            result.append(path)
    return result


def _local_resolution(reference_path: Path, provider: str, explicit_path: str | Path | None, project_root: Path) -> DemResolution | None:
    candidates = _local_candidates(reference_path, project_root, explicit_path)
    explicit = Path(explicit_path).resolve() if explicit_path else None
    ordered: list[Path] = []
    if explicit:
        ordered.append(explicit)
    for preferred in ("srtm", "aw3d30", "glo30", "auto"):
        ordered.extend(
            path for path in candidates
            if path not in ordered and (preferred == "auto" or _provider_matches(path, preferred))
        )
    for candidate in ordered:
        if not candidate.exists() or not _provider_matches(candidate, provider):
            continue
        try:
            if _covers(candidate, reference_path):
                return DemResolution(
                    candidate,
                    {
                        "provider": "local",
                        "source": str(candidate),
                        "cache_hit": False,
                        "units": "meters",
                        "vertical_reference": _vertical_reference(candidate),
                    },
                )
        except (OSError, rasterio.errors.RasterioIOError, ValueError):
            continue
    return None


def _tile_name(latitude: int, longitude: int) -> tuple[str, str]:
    lat_prefix = "N" if latitude >= 0 else "S"
    lon_prefix = "E" if longitude >= 0 else "W"
    tile = f"{lat_prefix}{abs(latitude):02d}{lon_prefix}{abs(longitude):03d}"
    return f"{lat_prefix}{abs(latitude):02d}", tile


def _download(url: str, destination: Path, timeout: float, max_bytes: int) -> bool:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ASTERRA-AI/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                return False
            total = 0
            destination.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        return False
                    handle.write(chunk)
        temporary.replace(destination)
        return True
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _hgt_to_geotiff(gzip_path: Path, output_path: Path, latitude: int, longitude: int) -> bool:
    try:
        raw = gzip_path.read_bytes()
        values = np.frombuffer(gzip.decompress(raw), dtype=">i2")
        size = int(math.sqrt(values.size))
        if size * size != values.size or size not in {1201, 3601}:
            return False
        array = values.reshape((size, size)).astype(np.float32)
        array[array <= -32768] = np.nan
        resolution = 1.0 / (size - 1)
        profile = {
            "driver": "GTiff",
            "height": size,
            "width": size,
            "count": 1,
            "dtype": "float32",
            "crs": _WGS84,
            "transform": from_origin(longitude, latitude + 1, resolution, resolution),
            "nodata": -9999.0,
            "compress": "deflate",
            "predictor": 3,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".part")
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(array), array, -9999.0), 1)
            dst.set_band_description(1, "AWS Terrain bare-earth elevation (m)")
        temporary.replace(output_path)
        return True
    except (OSError, ValueError, rasterio.errors.RasterioIOError):
        return False


def _aws_resolution(reference_path: Path, cache_root: Path, timeout: float, max_download_bytes: int, progress_callback=None) -> DemResolution | None:
    with rasterio.open(reference_path) as reference:
        if not reference.crs:
            return None
        west, south, east, north = transform_bounds(reference.crs, _WGS84, *reference.bounds, densify_pts=21)
    if south < -60 or north > 60:
        return None
    south_tile = max(-60, math.floor(south))
    north_tile = min(59, math.ceil(north) - 1)
    west_tile = math.floor(west)
    east_tile = math.ceil(east) - 1
    tile_coords = [(lat, lon) for lat in range(south_tile, north_tile + 1) for lon in range(west_tile, east_tile + 1)]
    if not tile_coords or len(tile_coords) > 25:
        return None
    tile_key = "_".join(f"{lat}_{lon}" for lat, lon in tile_coords)
    mosaic_key = hashlib.sha256(tile_key.encode("utf-8")).hexdigest()[:20]
    mosaic_path = cache_root / "mosaics" / f"aws_terrain_{mosaic_key}.tif"
    if mosaic_path.exists():
        return DemResolution(mosaic_path, {"provider": "aws_terrain", "source": str(mosaic_path), "cache_hit": True, "units": "meters", "vertical_reference": None, "tile_count": len(tile_coords)})

    tile_paths: list[Path] = []
    datasets = []
    memories = []
    for index, (latitude, longitude) in enumerate(tile_coords, 1):
        lat_dir, tile = _tile_name(latitude, longitude)
        compressed = cache_root / "tiles" / lat_dir / f"{tile}.hgt.gz"
        converted = cache_root / "tiles" / lat_dir / f"{tile}.tif"
        if not converted.exists():
            if not compressed.exists():
                url = _AWS_TERRAIN_URL.format(lat_dir=lat_dir, tile=tile)
                if progress_callback:
                    progress_callback(f"downloading DEM tile {index}/{len(tile_coords)}")
                if not _download(url, compressed, timeout, max_download_bytes):
                    return None
            if not _hgt_to_geotiff(compressed, converted, latitude, longitude):
                return None
        tile_paths.append(converted)

    try:
        for path in tile_paths:
            memories.append(rasterio.open(path))
        mosaic, transform = merge(memories, nodata=-9999.0)
        profile = memories[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
        )
        mosaic_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = mosaic_path.with_suffix(mosaic_path.suffix + ".part")
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.write(mosaic.astype(np.float32))
            dst.set_band_description(1, "AWS Terrain bare-earth elevation (m)")
        temporary.replace(mosaic_path)
    except (OSError, ValueError, rasterio.errors.RasterioIOError):
        mosaic_path.unlink(missing_ok=True)
        return None
    finally:
        for dataset in datasets + memories:
            dataset.close()
        for memory in memories:
            memory.close() if isinstance(memory, MemoryFile) else None
    return DemResolution(mosaic_path, {"provider": "aws_terrain", "source": str(mosaic_path), "cache_hit": False, "units": "meters", "vertical_reference": None, "tile_count": len(tile_coords), "source_url": "AWS Terrain Tiles / elevation-tiles-prod"})


def resolve_dem_info(reference_path, provider="auto", explicit_path=None, project_root=None, cache_root=None, online=True, timeout=30.0, max_download_bytes=64 * 1024 * 1024, progress_callback=None):
    """Resolve a spatially overlapping DEM and return its provenance."""
    reference_path = Path(reference_path)
    project_root = Path(project_root or reference_path.parents[2]).resolve()
    provider = (provider or "auto").lower()
    local = _local_resolution(reference_path, provider, explicit_path, project_root)
    if local:
        if progress_callback:
            progress_callback("using local DEM")
        return local
    if provider in {"local", "srtm", "aw3d30", "glo30", "copdem"} or not online:
        return None
    cache_root = Path(cache_root or project_root / "backend" / "runtime" / "dem_cache").resolve()
    return _aws_resolution(reference_path, cache_root, timeout, max_download_bytes, progress_callback)


def resolve_dem(reference_path, provider="auto", explicit_path=None, project_root=None):
    """Backward-compatible path-only resolver for CLI callers."""
    info = resolve_dem_info(reference_path, provider, explicit_path, project_root, online=False)
    return info.path if info else None
