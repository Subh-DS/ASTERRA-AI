"""Imagery discovery and AOI extraction for map-driven reconstruction jobs.

The browser map is only a display surface.  This module owns the imagery
contract used by the reconstruction pipeline: discover STAC scenes, validate
the selected scene, and materialize three georeferenced RGB bands clipped to
the requested geographic AOI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.env import Env
from rasterio.transform import Affine, from_bounds as transform_from_bounds
from rasterio.warp import transform_bounds, reproject
from rasterio.windows import Window, from_bounds as window_from_bounds


AOI_LIMITS = {
    "min_side_m": 50.0,
    "max_side_m": 2000.0,
    "max_area_km2": 2.0,
    "max_aspect": 8.0,
}

_WGS84 = "EPSG:4326"
_DEFAULT_PUBLIC_ATTRIBUTION = "Sentinel-2 L2A via Earth Search / Copernicus"
_PUBLIC_COLLECTIONS = ("sentinel-2-c1-l2a", "sentinel-2-l2a")
_BAND_ALIASES = {
    "red": ("B04", "b04", "red", "red-10m", "red_10m", "SR_B4"),
    "green": ("B03", "b03", "green", "green-10m", "green_10m", "SR_B3"),
    "blue": ("B02", "b02", "blue", "blue-10m", "blue_10m", "SR_B2"),
}


class ImageryError(RuntimeError):
    """Base class for expected imagery-service failures."""


class ImageryValidationError(ImageryError):
    pass


class ImageryProviderError(ImageryError):
    pass


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def aoi_metrics(aoi: dict[str, Any]) -> dict[str, float]:
    north, south = float(aoi["north"]), float(aoi["south"])
    east, west = float(aoi["east"]), float(aoi["west"])
    radius = 6371000.0
    mid_lat = math.radians((north + south) / 2.0)
    width_m = abs(east - west) * math.pi / 180.0 * radius * max(math.cos(mid_lat), 1e-9)
    height_m = abs(north - south) * math.pi / 180.0 * radius
    short_side = max(min(width_m, height_m), 1e-9)
    return {
        "center_lat": (north + south) / 2.0,
        "center_lon": (east + west) / 2.0,
        "width_m": width_m,
        "height_m": height_m,
        "area_km2": width_m * height_m / 1_000_000.0,
        "aspect": max(width_m, height_m) / short_side,
    }


def validate_aoi(raw: dict[str, Any]) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ImageryValidationError("AOI must be an object with north, south, east, and west coordinates.")
    values = {key: _finite(raw.get(key)) for key in ("north", "south", "east", "west")}
    if any(value is None for value in values.values()):
        raise ImageryValidationError("AOI coordinates must be finite numbers.")
    north, south, east, west = (values[key] for key in ("north", "south", "east", "west"))
    if not (-90 <= north <= 90 and -90 <= south <= 90):
        raise ImageryValidationError("AOI latitudes must be within -90 and 90 degrees.")
    if not (-180 <= east <= 180 and -180 <= west <= 180):
        raise ImageryValidationError("AOI longitudes must be within -180 and 180 degrees.")
    if north <= south or east <= west:
        raise ImageryValidationError("AOI north/east coordinates must be greater than south/west.")
    metrics = aoi_metrics(values)
    if metrics["width_m"] < AOI_LIMITS["min_side_m"] or metrics["height_m"] < AOI_LIMITS["min_side_m"]:
        raise ImageryValidationError("AOI is too small; each side must be at least 50 metres.")
    if metrics["width_m"] > AOI_LIMITS["max_side_m"] or metrics["height_m"] > AOI_LIMITS["max_side_m"]:
        raise ImageryValidationError("AOI is too large; each side must be at most 2,000 metres.")
    if metrics["area_km2"] > AOI_LIMITS["max_area_km2"]:
        raise ImageryValidationError("AOI is too large; maximum area is 2 square kilometres.")
    if metrics["aspect"] > AOI_LIMITS["max_aspect"]:
        raise ImageryValidationError("AOI is too elongated; its aspect ratio must be at most 8:1.")
    return {**values, **metrics}


def aoi_payload(aoi: dict[str, Any]) -> dict[str, float]:
    checked = validate_aoi(aoi)
    return {
        "north": checked["north"],
        "south": checked["south"],
        "east": checked["east"],
        "west": checked["west"],
        "center_lat": checked["center_lat"],
        "center_lon": checked["center_lon"],
        "width_m": checked["width_m"],
        "height_m": checked["height_m"],
        "area_km2": checked["area_km2"],
    }


def _http_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, headers=None, timeout=30.0):
    request_headers = {"Accept": "application/json", "User-Agent": "ASTERRA-AI/1.0"}
    request_headers.update(headers or {})
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except OSError:
            pass
        raise ImageryProviderError(f"imagery catalog returned HTTP {exc.code}{': ' + detail if detail else ''}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ImageryProviderError(f"imagery catalog request failed: {exc}") from exc


def _date_value(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        return str(value)


def _asset_href(asset: dict[str, Any]) -> str | None:
    href = asset.get("href") if isinstance(asset, dict) else None
    if not isinstance(href, str) or not href.startswith(("http://", "https://")):
        return None
    return href


def _asset_resolution(asset: dict[str, Any]) -> float | None:
    raster_bands = asset.get("raster:bands") or []
    if raster_bands and isinstance(raster_bands[0], dict):
        value = _finite(raster_bands[0].get("spatial_resolution"))
        if value:
            return value
    for key in ("gsd", "resolution", "spatial_resolution"):
        value = _finite(asset.get(key))
        if value:
            return value
    return None


def _find_band_assets(assets: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]] | None:
    found = {}
    metadata = {}
    lower = {str(key).lower(): (key, value) for key, value in assets.items()}
    for band, aliases in _BAND_ALIASES.items():
        for alias in aliases:
            pair = lower.get(alias.lower())
            if pair:
                key, asset = pair
                href = _asset_href(asset)
                if href:
                    found[band] = href
                    metadata[band] = asset
                    break
    if len(found) != 3:
        return None
    return found, metadata


@dataclass(frozen=True)
class Scene:
    provider: str
    item_id: str
    collection: str
    acquisition_datetime: str | None
    cloud_cover: float | None
    resolution_m: float
    bands: tuple[str, ...]
    bbox: tuple[float, ...] | None
    assets: dict[str, str]
    synthetic: bool = False
    attribution: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "collection": self.collection,
            "acquisition_datetime": self.acquisition_datetime,
            "cloud_cover": self.cloud_cover,
            "resolution_m": self.resolution_m,
            "bands": list(self.bands),
            "bbox": list(self.bbox) if self.bbox else None,
            "synthetic": self.synthetic,
        }


class BaseProvider:
    name = "provider"
    label = "Imagery provider"
    attribution = ""
    resolution_m = 10.0
    synthetic = False

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "resolution_m": self.resolution_m,
            "bands": ["red", "green", "blue"],
            "synthetic": self.synthetic,
            "attribution": self.attribution,
        }

    def search(self, aoi, max_cloud: float | None = None) -> list[Scene]:
        raise NotImplementedError

    def resolve(self, item_id: str) -> Scene:
        raise NotImplementedError

    def download(self, scene: Scene, aoi, destination: Path, quality: str, cache_root: Path | None, timeout: float):
        raise NotImplementedError


class StacProvider(BaseProvider):
    def __init__(self, *, name, label, endpoint, collections, token=None, attribution="", default_resolution=10.0, max_candidates=6):
        self.name = name
        self.label = label
        self.endpoint = endpoint.rstrip("/")
        self.collections = tuple(collections)
        self.token = token
        self.attribution = attribution
        self.resolution_m = default_resolution
        self.max_candidates = max_candidates

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _item_scene(self, item: dict[str, Any], collection: str) -> Scene | None:
        assets = item.get("assets") or {}
        selected = _find_band_assets(assets)
        if not selected:
            return None
        hrefs, asset_meta = selected
        properties = item.get("properties") or {}
        cloud = _finite(properties.get("eo:cloud_cover"))
        resolution = next((_asset_resolution(asset_meta[b]) for b in ("red", "green", "blue") if _asset_resolution(asset_meta[b])), None)
        if resolution is None:
            resolution = _finite(properties.get("gsd")) or self.resolution_m
        bbox = item.get("bbox")
        try:
            bbox_tuple = tuple(float(value) for value in bbox) if bbox else None
        except (TypeError, ValueError):
            bbox_tuple = None
        raw_id = str(item.get("id") or "")
        if not raw_id:
            return None
        return Scene(
            provider=self.name,
            item_id=f"{collection}:{raw_id}",
            collection=collection,
            acquisition_datetime=_date_value(properties.get("datetime") or properties.get("dct:datetime")),
            cloud_cover=cloud,
            resolution_m=float(resolution),
            bands=("red", "green", "blue"),
            bbox=bbox_tuple,
            assets=hrefs,
            attribution=self.attribution,
        )

    def _search_collection(self, collection: str, aoi, max_cloud: float | None, timeout: float) -> list[Scene]:
        query: dict[str, Any] = {
            "collections": [collection],
            "bbox": [aoi["west"], aoi["south"], aoi["east"], aoi["north"]],
            "limit": max(self.max_candidates, 6),
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        }
        if max_cloud is not None:
            query["query"] = {"eo:cloud_cover": {"lte": max_cloud}}
        response = _http_json(f"{self.endpoint}/search", method="POST", payload=query, headers=self._headers(), timeout=timeout)
        features = response.get("features") or []
        scenes = []
        for item in features:
            scene = self._item_scene(item, collection)
            if scene and (scene.cloud_cover is None or max_cloud is None or scene.cloud_cover <= max_cloud):
                scenes.append(scene)
        return scenes

    def search(self, aoi, max_cloud: float | None = None, timeout: float = 30.0) -> list[Scene]:
        results: list[Scene] = []
        for collection in self.collections:
            try:
                results.extend(self._search_collection(collection, aoi, max_cloud, timeout))
            except ImageryProviderError:
                if results:
                    break
                raise
            if results:
                break
        results.sort(key=lambda scene: (
            scene.cloud_cover if scene.cloud_cover is not None else 101.0,
            scene.acquisition_datetime or "",
        ))
        return results[: self.max_candidates]

    def resolve(self, item_id: str, timeout: float = 30.0) -> Scene:
        if ":" not in item_id:
            raise ImageryValidationError("Selected imagery scene ID is invalid.")
        collection, raw_id = item_id.split(":", 1)
        if collection not in self.collections or not raw_id:
            raise ImageryValidationError("Selected imagery scene does not belong to this provider.")
        encoded_collection = urllib.parse.quote(collection, safe="")
        encoded_id = urllib.parse.quote(raw_id, safe="")
        item = _http_json(
            f"{self.endpoint}/collections/{encoded_collection}/items/{encoded_id}",
            headers=self._headers(),
            timeout=timeout,
        )
        scene = self._item_scene(item, collection)
        if not scene:
            raise ImageryProviderError("Selected imagery scene has no accessible RGB assets.")
        return scene

    def download(self, scene: Scene, aoi, destination: Path, quality: str, cache_root: Path | None, timeout: float):
        _materialize_scene(scene, aoi, destination, quality, cache_root, timeout, self.token)


class EsriWorldImageryProvider(BaseProvider):
    """Public high-resolution RGB export backed by the map's imagery source.

    This is intentionally a texture/depth source, not an elevation source.
    The server receives the exact AOI in Web Mercator and returns an RGB
    raster with matching projected bounds, so the existing georeferenced
    pipeline can consume it without a tile seam or an untracked crop.
    """

    name = "esri"
    label = "Esri World Imagery (high-resolution)"
    attribution = "Imagery © Esri, Maxar, Earthstar Geographics"
    resolution_m = 1.0
    synthetic = False

    def __init__(self, endpoint: str):
        self.endpoints = tuple(
            candidate.strip().rstrip("?")
            for candidate in str(endpoint).split(",")
            if candidate.strip()
        )
        self.endpoint = self.endpoints[0] if self.endpoints else ""
        self.last_download: dict[str, Any] = {}

    @staticmethod
    def _scene():
        return Scene(
            provider="esri",
            item_id="esri:world-imagery",
            collection="World_Imagery",
            acquisition_datetime=None,
            cloud_cover=None,
            resolution_m=1.0,
            bands=("red", "green", "blue"),
            bbox=None,
            assets={},
            attribution=EsriWorldImageryProvider.attribution,
        )

    def search(self, aoi, max_cloud: float | None = None, timeout: float = 30.0) -> list[Scene]:
        # Esri's basemap is an imagery mosaic rather than a dated STAC scene;
        # cloud filtering is not meaningful. The catalog endpoint still
        # returns one explicit selectable scene, never a hidden fallback.
        return [self._scene()]

    def resolve(self, item_id: str, timeout: float = 30.0) -> Scene:
        if item_id != "esri:world-imagery":
            raise ImageryValidationError("Selected Esri imagery scene ID is invalid.")
        return self._scene()

    def download(self, scene: Scene, aoi, destination: Path, quality: str, cache_root: Path | None, timeout: float):
        from PIL import Image, UnidentifiedImageError

        checked = validate_aoi(aoi)
        max_dimension = _quality_dimension(quality)
        target_bounds = transform_bounds(
            _WGS84,
            "EPSG:3857",
            checked["west"], checked["south"], checked["east"], checked["north"],
            densify_pts=21,
        )
        span_x = max(abs(target_bounds[2] - target_bounds[0]), 1.0)
        span_y = max(abs(target_bounds[3] - target_bounds[1]), 1.0)
        # Do not ask a public map mosaic for arbitrary sub-pixel detail. Keep
        # the export at the provider's nominal resolution, downsample large
        # AOIs to the selected quality, and only upsample when the depth model
        # requires its 128-pixel minimum input dimension.
        native_resolution = max(float(self.resolution_m), 0.1)
        native_width = max(1, int(round(span_x / native_resolution)))
        native_height = max(1, int(round(span_y / native_resolution)))
        scale = min(1.0, max_dimension / max(native_width, native_height))
        # Small AOIs otherwise arrive as 128x128 thumbnails. That is enough
        # for a map preview but not for roof/road/vegetation separation. Ask
        # the high-resolution mosaic for a browser-safe floor of 512 pixels;
        # this does not alter the honest source_gsd metadata used by the
        # pipeline, it only prevents avoidable thumbnail inference.
        minimum_dimension = min(max_dimension, 512)
        width = max(minimum_dimension, min(max_dimension, int(round(native_width * scale))))
        height = max(minimum_dimension, min(max_dimension, int(round(native_height * scale))))

        destination = Path(destination)
        cache_path = None
        if cache_root:
            key = hashlib.sha256(json.dumps({
                "schema": 3,
                "scene": scene.item_id,
                "aoi": checked,
                "quality": quality,
                "endpoint": self.endpoints,
            }, sort_keys=True).encode()).hexdigest()
            cache_path = Path(cache_root) / self.name / f"{key}.tif"
            if cache_path.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache_path, destination)
                self.last_download = {"method": "cache", "fallback_chain": []}
                return

        query = urllib.parse.urlencode({
            "bbox": ",".join(f"{value:.12f}" for value in target_bounds),
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": f"{width},{height}",
            # World_Imagery is a MapServer export endpoint. Use a format it
            # advertises directly; ``jpgpng`` is an ImageServer format and
            # causes some ArcGIS hosts to return HTTP 500.
            "format": "png",
            "f": "image",
        })
        if not self.endpoints:
            raise ImageryProviderError("Esri imagery export is not configured.")
        bands = None
        last_error = None
        export_errors = []
        for endpoint in self.endpoints:
            request = urllib.request.Request(
                f"{endpoint}?{query}",
                headers={"Accept": "image/png,image/jpeg", "User-Agent": "ASTERRA-AI/1.0"},
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        raw = response.read()
                    with Image.open(io.BytesIO(raw)) as image:
                        image = image.convert("RGB")
                        bands = np.transpose(np.asarray(image), (2, 0, 1)).astype(np.uint8)
                    if bands.shape[1] < 2 or bands.shape[2] < 2 or not np.isfinite(bands).all():
                        raise ValueError("Esri export returned an invalid RGB raster")
                    break
                except urllib.error.HTTPError as exc:
                    detail = ""
                    try:
                        detail = exc.read().decode("utf-8", errors="replace")[:180]
                    except OSError:
                        pass
                    last_error = f"HTTP {exc.code}{': ' + detail if detail else ''} from {endpoint}"
                    export_errors.append(last_error)
                    transient = exc.code == 429 or exc.code == 408 or 500 <= exc.code < 600
                    if not transient or attempt == 2:
                        break
                    time.sleep(0.25 * (2 ** attempt))
                except (urllib.error.URLError, TimeoutError, OSError, UnidentifiedImageError, ValueError) as exc:
                    last_error = f"{exc} from {endpoint}"
                    export_errors.append(last_error)
                    if attempt == 2:
                        break
                    time.sleep(0.25 * (2 ** attempt))
            if bands is not None:
                break

        fallback_chain = []
        if bands is None:
            # The browser uses the MapServer tile endpoint and it is often
            # healthy when the /export endpoint is throttled or returns 500.
            # Mosaic only the tiles intersecting the exact AOI, then crop and
            # resample back to the same georeferenced output contract.
            try:
                bands, tile_info = self._download_tiles(target_bounds, width, height, timeout, Image)
                fallback_chain = ["esri_export", "esri_tiles"]
                self.last_download = {"method": "tiles", "fallback_chain": fallback_chain, **tile_info}
            except Exception as exc:
                last_error = f"{last_error}; tile fallback failed: {str(exc)[:180]}"
        else:
            self.last_download = {"method": "export", "fallback_chain": []}
        if bands is None:
            raise ImageryProviderError(
                f"unable to download Esri imagery export or tiles; all endpoints failed: {last_error}"
            )

        out_height, out_width = bands.shape[1], bands.shape[2]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            width=out_width,
            height=out_height,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform_from_bounds(*target_bounds, out_width, out_height),
            compress="deflate",
            interleave="pixel",
        ) as dst:
            dst.write(bands)
            dst.write_mask(np.full((out_height, out_width), 255, dtype=np.uint8))
        temporary.replace(destination)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, cache_path)

    def _download_tiles(self, target_bounds, width, height, timeout, Image):
        """Mosaic the same Esri MapServer tiles used by the browser map."""
        if not self.endpoints:
            raise ImageryProviderError("Esri imagery tile fallback is not configured.")
        world = 20037508.342789244
        zoom = 18
        n = 1 << zoom
        tile_span = (2.0 * world) / n
        left, bottom, right, top = target_bounds
        tx0 = max(0, min(n - 1, int(math.floor((left + world) / tile_span))))
        tx1 = max(0, min(n - 1, int(math.floor((right + world) / tile_span))))
        ty0 = max(0, min(n - 1, int(math.floor((world - top) / tile_span))))
        ty1 = max(0, min(n - 1, int(math.floor((world - bottom) / tile_span))))
        tile_count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        if tile_count > 128:
            raise ImageryProviderError(f"tile fallback would require {tile_count} tiles")

        canvas = np.zeros((ty1 - ty0 + 1, tx1 - tx0 + 1, 256, 256, 3), dtype=np.uint8)
        last_error = None
        for endpoint in self.endpoints:
            base = endpoint.split("/export", 1)[0].rstrip("/")
            try:
                for ty in range(ty0, ty1 + 1):
                    for tx in range(tx0, tx1 + 1):
                        url = f"{base}/tile/{zoom}/{ty}/{tx}"
                        request = urllib.request.Request(
                            url,
                            headers={"Accept": "image/jpeg,image/png", "User-Agent": "ASTERRA-AI/1.0"},
                        )
                        raw = None
                        for attempt in range(3):
                            try:
                                with urllib.request.urlopen(request, timeout=timeout) as response:
                                    raw = response.read()
                                break
                            except urllib.error.HTTPError as exc:
                                last_error = f"HTTP {exc.code} from {url}"
                                if exc.code not in (408, 429) and not 500 <= exc.code < 600:
                                    raise
                                if attempt == 2:
                                    raise
                                time.sleep(0.2 * (2 ** attempt))
                            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                                last_error = str(exc)
                                if attempt == 2:
                                    raise
                                time.sleep(0.2 * (2 ** attempt))
                        with Image.open(io.BytesIO(raw)) as image:
                            tile = np.asarray(image.convert("RGB"), dtype=np.uint8)
                        if tile.shape != (256, 256, 3):
                            tile = np.asarray(
                                Image.fromarray(tile).resize((256, 256), Image.Resampling.BILINEAR),
                                dtype=np.uint8,
                            )
                        canvas[ty - ty0, tx - tx0] = tile
                break
            except Exception as exc:
                last_error = str(exc)
                canvas.fill(0)
                continue
        else:
            raise ImageryProviderError(last_error or "Esri tile download failed")

        mosaic = canvas.reshape((canvas.shape[0] * 256, canvas.shape[1] * 256, 3))
        mosaic_left = -world + tx0 * tile_span
        mosaic_top = world - ty0 * tile_span
        pixels_per_meter = 256.0 / tile_span
        x0 = max(0, int(math.floor((left - mosaic_left) * pixels_per_meter)))
        x1 = min(mosaic.shape[1], int(math.ceil((right - mosaic_left) * pixels_per_meter)))
        y0 = max(0, int(math.floor((mosaic_top - top) * pixels_per_meter)))
        y1 = min(mosaic.shape[0], int(math.ceil((mosaic_top - bottom) * pixels_per_meter)))
        crop = mosaic[y0:y1, x0:x1]
        if crop.size == 0:
            raise ImageryProviderError("Esri tile mosaic does not intersect the requested AOI")
        resized = Image.fromarray(crop, mode="RGB").resize((width, height), Image.Resampling.LANCZOS)
        bands = np.transpose(np.asarray(resized), (2, 0, 1)).astype(np.uint8)
        return bands, {"tile_zoom": zoom, "tile_count": int(tile_count)}


class MockProvider(BaseProvider):
    name = "mock"
    label = "Mock imagery"
    attribution = "Synthetic test pattern"
    resolution_m = 1.0
    synthetic = True

    def search(self, aoi, max_cloud: float | None = None, timeout: float = 30.0) -> list[Scene]:
        if max_cloud is not None and max_cloud < 0:
            return []
        return [Scene(
            provider=self.name,
            item_id="mock:synthetic-aoi",
            collection="mock",
            acquisition_datetime="2026-01-01T00:00:00+00:00",
            cloud_cover=0.0,
            resolution_m=self.resolution_m,
            bands=("red", "green", "blue"),
            bbox=(aoi["west"], aoi["south"], aoi["east"], aoi["north"]),
            assets={},
            synthetic=True,
            attribution=self.attribution,
        )]

    def resolve(self, item_id: str) -> Scene:
        if item_id != "mock:synthetic-aoi":
            raise ImageryValidationError("Selected mock imagery scene is invalid.")
        scene = self.search({"north": 1, "south": 0, "east": 1, "west": 0})[0]
        return Scene(**{**scene.__dict__, "bbox": None})

    def download(self, scene: Scene, aoi, destination: Path, quality: str, cache_root: Path | None, timeout: float):
        width = height = 256
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        ridge = np.exp(-(((xs - 0.32) ** 2) + ((ys - 0.58) ** 2)) * 18.0)
        bands = np.stack([
            np.clip(40 + 150 * xs + 45 * ridge, 0, 255),
            np.clip(55 + 125 * ys + 35 * ridge, 0, 255),
            np.clip(75 + 110 * (1 - xs) + 25 * ridge, 0, 255),
        ]).astype(np.uint8)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            destination,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs=_WGS84,
            transform=transform_from_bounds(aoi["west"], aoi["south"], aoi["east"], aoi["north"], width, height),
            compress="deflate",
            interleave="pixel",
        ) as dst:
            dst.write(bands)
            dst.write_mask(np.full((height, width), 255, dtype=np.uint8))


def _quality_dimension(quality: str) -> int:
    return {"low": 1024, "medium": 2048, "high": 4096}.get(quality, 2048)


def _safe_window(src, bounds):
    window = window_from_bounds(*bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()
    left = max(0, int(window.col_off))
    top = max(0, int(window.row_off))
    right = min(src.width, left + max(1, int(window.width)))
    bottom = min(src.height, top + max(1, int(window.height)))
    if right <= left or bottom <= top:
        raise ImageryProviderError("Imagery scene does not intersect the requested AOI.")
    return Window(left, top, right - left, bottom - top)


def _materialize_scene(scene: Scene, aoi, destination: Path, quality: str, cache_root: Path | None, timeout: float, token: str | None):
    destination = Path(destination)
    cache_path = None
    if cache_root:
        key = hashlib.sha256(json.dumps({"schema": 3, "scene": scene.item_id, "aoi": aoi, "quality": quality}, sort_keys=True).encode()).hexdigest()
        cache_path = Path(cache_root) / scene.provider / f"{key}.tif"
        if cache_path.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache_path, destination)
            return

    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.TIFF",
        "GDAL_HTTP_USERAGENT": "ASTERRA-AI/1.0",
    }
    if token:
        env_options["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"

    with Env(**env_options):
        try:
            with rasterio.open(scene.assets["red"] ) as red_src:
                target_crs = red_src.crs
                if target_crs is None:
                    raise ImageryProviderError("Imagery RGB asset has no coordinate reference system.")
                target_bounds = transform_bounds(_WGS84, target_crs, aoi["west"], aoi["south"], aoi["east"], aoi["north"], densify_pts=21)
                target_width_native = max(1, int(math.ceil(abs(target_bounds[2] - target_bounds[0]) / abs(red_src.transform.a))))
                target_height_native = max(1, int(math.ceil(abs(target_bounds[3] - target_bounds[1]) / abs(red_src.transform.e))))
                max_dimension = _quality_dimension(quality)
                scale = min(1.0, max_dimension / max(target_width_native, target_height_native))
                minimum_dimension = min(max_dimension, 512) if scene.resolution_m <= 2.0 else 128
                out_width = max(minimum_dimension, int(round(target_width_native * scale)))
                out_height = max(minimum_dimension, int(round(target_height_native * scale)))
                # Preserve the exact AOI bounds even when a small AOI must be
                # upsampled to satisfy the depth model's minimum input size.
                destination_transform = transform_from_bounds(*target_bounds, out_width, out_height)
                arrays = []
                valid_mask = None
                for band_name in ("red", "green", "blue"):
                    with rasterio.open(scene.assets[band_name]) as src:
                        source_bounds = transform_bounds(target_crs, src.crs, *target_bounds, densify_pts=21) if src.crs != target_crs else target_bounds
                        window = _safe_window(src, source_bounds)
                        read_width = max(1, min(int(window.width), max_dimension))
                        read_height = max(1, min(int(window.height), max_dimension))
                        source = src.read(1, window=window, out_shape=(read_height, read_width), resampling=Resampling.bilinear).astype(np.float32)
                        source_transform = src.window_transform(window) * Affine.scale(window.width / read_width, window.height / read_height)
                        target = np.zeros((out_height, out_width), dtype=np.float32)
                        reproject(
                            source,
                            target,
                            src_transform=source_transform,
                            src_crs=src.crs,
                            dst_transform=destination_transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear,
                        )
                        arrays.append(target)
                        if valid_mask is None:
                            valid_mask = np.isfinite(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with rasterio.open(
                temporary,
                "w",
                driver="GTiff",
                width=out_width,
                height=out_height,
                count=3,
                dtype="float32",
                crs=target_crs,
                transform=destination_transform,
                nodata=0,
                compress="deflate",
                interleave="pixel",
            ) as dst:
                dst.write(np.stack(arrays))
                dst.write_mask(np.where(valid_mask, 255, 0).astype(np.uint8))
            temporary.replace(destination)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(destination, cache_path)
        except (OSError, ValueError, rasterio.errors.RasterioIOError, KeyError) as exc:
            raise ImageryProviderError(f"unable to materialize imagery scene: {exc}") from exc


class ImageryCatalog:
    def __init__(self, settings):
        self.settings = settings
        self.last_resolution: dict[str, Any] = {"fallback_chain": []}
        self._providers: dict[str, BaseProvider] = {
            "public": StacProvider(
                name="public",
                label="Sentinel-2 public",
                endpoint=settings.imagery_stac_url,
                collections=settings.imagery_collections,
                attribution=_DEFAULT_PUBLIC_ATTRIBUTION,
                default_resolution=10.0,
                max_candidates=settings.imagery_max_candidates,
            )
        }
        if settings.imagery_esri_url:
            self._providers["esri"] = EsriWorldImageryProvider(settings.imagery_esri_url)
        if settings.imagery_highres_stac_url and settings.imagery_highres_collection:
            self._providers["highres"] = StacProvider(
                name="highres",
                label="Configured high-resolution STAC",
                endpoint=settings.imagery_highres_stac_url,
                collections=(settings.imagery_highres_collection,),
                token=settings.imagery_highres_token,
                attribution="Configured high-resolution STAC source",
                default_resolution=1.0,
                max_candidates=settings.imagery_max_candidates,
            )
        if settings.map_mock:
            self._providers["mock"] = MockProvider()

    def provider_names(self) -> list[str]:
        return list(self._providers)

    def descriptors(self) -> list[dict[str, Any]]:
        return [provider.descriptor() for provider in self._providers.values()]

    def get(self, name: str | None) -> BaseProvider:
        requested = (name or "auto").lower()
        if requested == "auto":
            # Auto means best configured source: private high-resolution STAC
            # first, then the public high-resolution mosaic, then Sentinel-2.
            # Users can still explicitly select ``public`` when they need a
            # dated, cloud-filterable Sentinel-2 scene.
            if "highres" in self._providers:
                requested = "highres"
            elif "esri" in self._providers:
                requested = "esri"
            else:
                requested = self.settings.imagery_default_provider
        provider = self._providers.get(requested)
        if not provider:
            raise ImageryValidationError(f"Imagery provider '{name}' is not configured.")
        return provider

    def _auto_candidates(self) -> list[BaseProvider]:
        """Return high-resolution sources first, then the public fallback."""
        names = ("highres", "esri", self.settings.imagery_default_provider, "public", "mock")
        providers = []
        seen = set()
        for name in names:
            if name in seen or name not in self._providers:
                continue
            seen.add(name)
            providers.append(self._providers[name])
        return providers

    def search(self, aoi: dict[str, Any], provider: str | None, max_cloud: float | None) -> dict[str, Any]:
        checked = validate_aoi(aoi)
        if max_cloud is not None and (max_cloud < 0 or max_cloud > 100):
            raise ImageryValidationError("Maximum cloud cover must be between 0 and 100 percent.")
        candidates = self._auto_candidates() if (provider or "auto").lower() == "auto" else [self.get(provider)]
        last_error = None
        for selected in candidates:
            try:
                scenes = selected.search(checked, max_cloud=max_cloud, timeout=self.settings.imagery_timeout)
            except ImageryProviderError as exc:
                last_error = exc
                continue
            if scenes or len(candidates) == 1:
                return {
                    "provider": selected.name,
                    "attribution": selected.attribution,
                    "resolution_m": selected.resolution_m,
                    "items": [scene.public() for scene in scenes],
                }
        if last_error:
            raise last_error
        raise ImageryProviderError("No suitable imagery scene was found for this AOI.")

    def resolve_for_job(self, aoi: dict[str, Any], provider: str | None, item_id: str | None, max_cloud: float | None) -> Scene:
        checked = validate_aoi(aoi)
        if max_cloud is not None and (max_cloud < 0 or max_cloud > 100):
            raise ImageryValidationError("Maximum cloud cover must be between 0 and 100 percent.")
        selected = self.get(provider)
        if item_id:
            scene = selected.resolve(item_id, timeout=self.settings.imagery_timeout)
            if scene.bbox and not scene.synthetic and not _bbox_intersects(scene.bbox, checked):
                raise ImageryValidationError("Selected imagery scene does not cover the requested AOI.")
            if scene.cloud_cover is not None and max_cloud is not None and scene.cloud_cover > max_cloud:
                raise ImageryValidationError("Selected imagery scene exceeds the requested cloud-cover limit.")
            return scene
        candidates = self._auto_candidates() if (provider or "auto").lower() == "auto" else [selected]
        last_error = None
        attempted = []
        self.last_resolution = {"fallback_chain": []}
        for candidate in candidates:
            attempted.append(candidate.name)
            try:
                scenes = candidate.search(checked, max_cloud=max_cloud, timeout=self.settings.imagery_timeout)
            except ImageryProviderError as exc:
                last_error = exc
                continue
            if scenes:
                if len(attempted) > 1:
                    self.last_resolution = {"fallback_chain": attempted.copy()}
                return scenes[0]
        if last_error:
            raise last_error
        raise ImageryProviderError("No suitable imagery scene was found for this AOI.")

    def materialize(self, scene: Scene, aoi, destination: Path, quality: str):
        provider = self.get(scene.provider)
        provider.download(
            scene,
            validate_aoi(aoi),
            Path(destination),
            quality,
            self.settings.imagery_cache_root,
            self.settings.imagery_timeout,
        )
        return dict(getattr(provider, "last_download", {}) or {})


def _bbox_intersects(bbox: tuple[float, ...], aoi: dict[str, Any]) -> bool:
    if len(bbox) < 4:
        return True
    return not (bbox[2] <= aoi["west"] or bbox[0] >= aoi["east"] or bbox[3] <= aoi["south"] or bbox[1] >= aoi["north"])


def imagery_catalog(settings) -> ImageryCatalog:
    return ImageryCatalog(settings)


def _validate_materialized_raster(path: Path) -> dict[str, Any]:
    """Reject corrupt/empty imagery before depth inference starts."""
    try:
        with rasterio.open(path) as src:
            if src.count < 3 or src.width < 2 or src.height < 2:
                raise ImageryProviderError("imagery raster is smaller than a valid RGB scene")
            sample = src.read(
                [1, 2, 3],
                out_shape=(3, min(96, src.height), min(96, src.width)),
                masked=True,
            )
            # ``sample`` is often uint8 for Esri PNG exports.  Filling an
            # integer MaskedArray directly with NaN raises
            # ``Cannot convert fill_value nan to dtype uint8`` when the
            # raster has even one masked pixel.  Promote before filling.
            values = np.asarray(sample.astype(np.float32).filled(np.nan), dtype=np.float32)
            finite = np.isfinite(values)
            if not finite.any() or float(finite.mean()) < 0.8:
                raise ImageryProviderError("imagery raster contains insufficient finite RGB data")
            if float(np.nanmax(values)) <= 0.0:
                raise ImageryProviderError("imagery raster contains no visible RGB signal")
            spread = float(np.nanstd(values))
            if not np.isfinite(spread) or spread <= 1e-6:
                raise ImageryProviderError("imagery raster contains no non-uniform texture")
            return {
                "width": int(src.width),
                "height": int(src.height),
                "bands": int(src.count),
                "rgb_std": round(spread, 4),
            }
    except (OSError, ValueError, rasterio.errors.RasterioIOError) as exc:
        if isinstance(exc, ImageryProviderError):
            raise
        raise ImageryProviderError(f"unable to validate imagery raster: {exc}") from exc


def prepare_map_scene(job: dict[str, Any], destination: Path, settings) -> dict[str, Any]:
    """Resolve and download a map scene for the existing pipeline."""
    catalog = imagery_catalog(settings)
    requested_provider = (job.get("imagery_provider") or "auto").lower()
    scene = catalog.resolve_for_job(
        job.get("aoi") or {},
        job.get("imagery_provider"),
        job.get("imagery_item_id"),
        job.get("imagery_max_cloud"),
    )
    try:
        download_info = catalog.materialize(scene, job["aoi"], destination, job.get("imagery_quality") or "medium")
        raster_info = _validate_materialized_raster(Path(destination))
    except ImageryProviderError as exc:
        # The public high-resolution mosaic is the preferred map source, but
        # one ArcGIS host can be transiently unavailable. Keep the AOI job
        # useful by retrying with a dated RGB scene instead of leaving ingest
        # permanently failed; provenance records the fallback explicitly.
        if scene.provider != "esri":
            raise
        fallback = catalog.resolve_for_job(
            job.get("aoi") or {},
            "public",
            None,
            job.get("imagery_max_cloud"),
        )
        download_info = catalog.materialize(fallback, job["aoi"], destination, job.get("imagery_quality") or "medium")
        raster_info = _validate_materialized_raster(Path(destination))
        scene = fallback
        fallback_warning = f"High-resolution imagery unavailable; used public fallback: {str(exc)[:220]}"
        fallback_chain = ["esri_export", "esri_tiles", "public"]
    else:
        fallback_warning = None
        download_chain = download_info.get("fallback_chain", [])
        resolution_chain = catalog.last_resolution.get("fallback_chain", [])
        if download_chain and resolution_chain and resolution_chain[-1] == scene.provider:
            fallback_chain = resolution_chain[:-1] + download_chain
        else:
            fallback_chain = download_chain or resolution_chain
    return scene.public() | {
        "provider": scene.provider,
        "requested_provider": requested_provider,
        "resolved_provider": scene.provider,
        "attribution": scene.attribution,
        "synthetic": scene.synthetic,
        "fallback_warning": fallback_warning,
        "quality_warning": fallback_warning or (
            "Esri export failed; browser-compatible Esri tile mosaic was used."
            if fallback_chain == ["esri_export", "esri_tiles"]
            else ("Imagery fallback chain used: " + " → ".join(fallback_chain) if fallback_chain else None)
        ),
        "fallback_chain": fallback_chain,
        "download": download_info,
        "raster": raster_info,
    }
