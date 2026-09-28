import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds

from backend.services.imagery_service import (
    ImageryProviderError,
    ImageryValidationError,
    EsriWorldImageryProvider,
    MockProvider,
    StacProvider,
    _validate_materialized_raster,
    validate_aoi,
)


AOI = {"north": 20.305, "south": 20.295, "east": 85.81, "west": 85.80}


def stac_item(item_id="scene-1", cloud=8.0, include_blue=True):
    assets = {
        "B04": {"href": "https://example.test/red.tif", "raster:bands": [{"spatial_resolution": 10}]},
        "B03": {"href": "https://example.test/green.tif", "raster:bands": [{"spatial_resolution": 10}]},
    }
    if include_blue:
        assets["B02"] = {"href": "https://example.test/blue.tif", "raster:bands": [{"spatial_resolution": 10}]}
    return {
        "id": item_id,
        "bbox": [85.7, 20.2, 85.9, 20.4],
        "properties": {"datetime": "2026-01-02T00:00:00Z", "eo:cloud_cover": cloud},
        "assets": assets,
    }


class ImageryServiceTests(unittest.TestCase):
    def test_aoi_validation_matches_frontend_limits(self):
        checked = validate_aoi(AOI)
        self.assertGreater(checked["width_m"], 50)
        with self.assertRaises(ImageryValidationError):
            validate_aoi({"north": 20.3001, "south": 20.3, "east": 85.81, "west": 85.80})

    def test_stac_search_filters_cloud_and_discards_missing_rgb(self):
        provider = StacProvider(
            name="public",
            label="Public",
            endpoint="https://example.test/v1",
            collections=("collection",),
        )
        payload = {
            "features": [
                stac_item("cloudy", cloud=80.0),
                stac_item("missing-blue", cloud=2.0, include_blue=False),
                stac_item("clear", cloud=4.0),
            ]
        }
        with patch("backend.services.imagery_service._http_json", return_value=payload):
            scenes = provider.search(AOI, max_cloud=20, timeout=1)
        self.assertEqual([scene.item_id for scene in scenes], ["collection:clear"])
        self.assertEqual(scenes[0].resolution_m, 10.0)

    def test_provider_errors_are_not_hidden(self):
        provider = StacProvider(
            name="public",
            label="Public",
            endpoint="https://example.test/v1",
            collections=("collection",),
        )
        with patch(
            "backend.services.imagery_service._http_json",
            side_effect=ImageryProviderError("imagery catalog request failed: timed out"),
        ):
            with self.assertRaisesRegex(ImageryProviderError, "timed out"):
                provider.search(AOI, max_cloud=20, timeout=1)

    def test_mock_scene_preserves_exact_aoi_georeference(self):
        provider = MockProvider()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.tif"
            scene = provider.search(AOI)[0]
            provider.download(scene, AOI, path, "medium", None, 1)
            with rasterio.open(path) as src:
                self.assertEqual((src.width, src.height), (256, 256))
                self.assertEqual(src.count, 3)
                self.assertEqual(str(src.crs), "EPSG:4326")
                self.assertAlmostEqual(src.bounds.left, AOI["west"], places=6)
                self.assertAlmostEqual(src.bounds.right, AOI["east"], places=6)
                self.assertAlmostEqual(src.bounds.bottom, AOI["south"], places=6)
                self.assertAlmostEqual(src.bounds.top, AOI["north"], places=6)

    def test_materialized_raster_validation_rejects_uniform_texture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "uniform.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=32,
                height=32,
                count=3,
                dtype="uint8",
                crs="EPSG:3857",
                transform=rasterio.transform.from_origin(0, 32, 1, 1),
            ) as dst:
                dst.write(np.full((3, 32, 32), 120, dtype="uint8"))
            with self.assertRaises(ImageryProviderError):
                _validate_materialized_raster(path)

    def test_materialized_uint8_raster_with_mask_is_dtype_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "masked.tif"
            values = np.zeros((3, 32, 32), dtype=np.uint8)
            values[0] = np.arange(32, dtype=np.uint8)[None, :]
            values[1] = np.arange(32, dtype=np.uint8)[:, None]
            values[2] = 100
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=32,
                height=32,
                count=3,
                dtype="uint8",
                crs="EPSG:3857",
                transform=rasterio.transform.from_origin(0, 32, 1, 1),
            ) as dst:
                dst.write(values)
                valid = np.full((32, 32), 255, dtype=np.uint8)
                valid[0, 0] = 0
                dst.write_mask(valid)
            info = _validate_materialized_raster(path)
            self.assertGreater(info["rgb_std"], 0.0)

    def test_esri_export_preserves_exact_projected_aoi(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (8, 6), (40, 120, 200)).save(image_bytes, format="PNG")
        response = patch("backend.services.imagery_service.urllib.request.urlopen")
        with response as urlopen:
            http_response = urlopen.return_value.__enter__.return_value
            http_response.read.return_value = image_bytes.getvalue()
            provider = EsriWorldImageryProvider("https://example.test/export")
            scene = provider.search(AOI)[0]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "esri.tif"
                provider.download(scene, AOI, path, "medium", None, 1)
                with rasterio.open(path) as src:
                    self.assertEqual(src.count, 3)
                    self.assertEqual(str(src.crs), "EPSG:3857")
                    expected = transform_bounds(
                        "EPSG:4326", "EPSG:3857",
                        AOI["west"], AOI["south"], AOI["east"], AOI["north"],
                        densify_pts=21,
                    )
                    self.assertAlmostEqual(src.bounds.left, expected[0], places=3)
                    self.assertAlmostEqual(src.bounds.bottom, expected[1], places=3)
                    self.assertAlmostEqual(src.bounds.right, expected[2], places=3)
                    self.assertAlmostEqual(src.bounds.top, expected[3], places=3)
        urlopen.assert_called_once()

    def test_esri_export_tries_configured_fallback_endpoint(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (8, 6), (40, 120, 200)).save(image_bytes, format="PNG")
        fallback_response = MagicMock()
        fallback_response.__enter__.return_value.read.return_value = image_bytes.getvalue()
        first_error = urllib.error.HTTPError(
            "https://primary.test/export", 500, "busy", {}, io.BytesIO(b"primary unavailable")
        )
        with patch(
            "backend.services.imagery_service.urllib.request.urlopen",
            side_effect=[first_error, fallback_response],
        ) as urlopen:
            provider = EsriWorldImageryProvider(
                "https://primary.test/export,https://fallback.test/export"
            )
            scene = provider.search(AOI)[0]
            with tempfile.TemporaryDirectory() as tmp:
                provider.download(scene, AOI, Path(tmp) / "esri.tif", "medium", None, 1)
        self.assertEqual(urlopen.call_count, 2)

    def test_esri_tile_mosaic_is_used_after_export_retries_fail(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (256, 256), (40, 120, 200)).save(image_bytes, format="PNG")
        tile_bytes = image_bytes.getvalue()

        def open_url(request, timeout=None):
            url = request.full_url if hasattr(request, "full_url") else str(request)
            if "/export?" in url:
                raise urllib.error.HTTPError(url, 500, "busy", {}, io.BytesIO(b"busy"))
            response = MagicMock()
            response.__enter__.return_value.read.return_value = tile_bytes
            return response

        provider = EsriWorldImageryProvider("https://example.test/export")
        scene = provider.search(AOI)[0]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("backend.services.imagery_service.urllib.request.urlopen", side_effect=open_url), \
                 patch("backend.services.imagery_service.time.sleep", return_value=None):
                provider.download(scene, AOI, Path(tmp) / "esri.tif", "medium", None, 1)
            with rasterio.open(Path(tmp) / "esri.tif") as src:
                self.assertEqual(src.count, 3)
                self.assertGreaterEqual(src.width, 512)
                self.assertGreaterEqual(src.height, 512)
        self.assertEqual(provider.last_download["method"], "tiles")
        self.assertEqual(provider.last_download["fallback_chain"], ["esri_export", "esri_tiles"])


if __name__ == "__main__":
    unittest.main()
