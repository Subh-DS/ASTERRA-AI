import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from backend.services.building_service import (
    buildings_geojson,
    detect_sports_ground_zones,
    fetch_buildings,
    fetch_map_features,
)


class BuildingServiceTests(unittest.TestCase):
    def test_osm_height_is_projected_and_preserved(self):
        payload = {
            "elements": [{
                "type": "way",
                "id": 123,
                "tags": {"building": "yes", "height": "9 m", "name": "Test Hall"},
                "geometry": [
                    {"lat": 0.0001, "lon": 0.0001},
                    {"lat": 0.0001, "lon": 0.0002},
                    {"lat": 0.0002, "lon": 0.0002},
                    {"lat": 0.0002, "lon": 0.0001},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff", "width": 128, "height": 128, "count": 3,
                "dtype": "uint8", "crs": "EPSG:3857",
                "transform": from_bounds(0, 0, 128, 128, 128, 128),
            }
            with rasterio.open(root / "rgb.tif", "w", **profile) as dst:
                dst.write(np.full((3, 128, 128), 80, dtype=np.uint8))
            with rasterio.open(root / "dsm.tif", "w", **{**profile, "count": 1, "dtype": "float32"}) as dst:
                dst.write(np.full((1, 128, 128), 20, dtype=np.float32))
            with patch("backend.services.building_service._query_overpass", return_value=payload):
                buildings, info = fetch_buildings(
                    {"north": 1, "south": -1, "east": 1, "west": -1},
                    root / "rgb.tif", root / "dsm.tif",
                    endpoint="https://example.test/overpass",
                    source_gsd_m=10,
                )
        self.assertEqual(len(buildings), 1)
        self.assertEqual(buildings[0]["height"], 9.0)
        self.assertEqual(buildings[0]["height_source"], "osm:height")
        self.assertEqual(info["provider"], "openstreetmap")
        self.assertEqual(buildings_geojson(buildings)["features"][0]["properties"]["name"], "Test Hall")

    def test_temple_name_selects_temple_geometry_profile(self):
        payload = {"elements": [{
            "type": "way", "id": 789,
            "tags": {"building": "yes", "name": "Ram Mandir"},
            "geometry": [
                {"lat": 0.0001, "lon": 0.0001}, {"lat": 0.0001, "lon": 0.0005},
                {"lat": 0.0005, "lon": 0.0005}, {"lat": 0.0005, "lon": 0.0001},
            ],
        }]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff", "width": 128, "height": 128, "count": 1,
                "dtype": "float32", "crs": "EPSG:3857",
                "transform": from_bounds(0, 0, 128, 128, 128, 128),
            }
            with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
                dst.write(np.full((1, 128, 128), 20, dtype=np.float32))
            with patch("backend.services.building_service._query_overpass", return_value=payload):
                buildings, _ = fetch_buildings(
                    {"north": 1, "south": -1, "east": 1, "west": -1},
                    root / "dsm.tif", root / "dsm.tif",
                    endpoint="https://example.test/overpass",
                    source_gsd_m=10,
                )
        self.assertEqual(len(buildings), 1)
        self.assertEqual(buildings[0]["geometry_profile"], "temple")

    def test_place_of_worship_without_building_tag_gets_temple_prior(self):
        payload = {"elements": [{
            "type": "way", "id": 790,
            "tags": {"amenity": "place_of_worship", "religion": "hindu", "name": "Temple"},
            "geometry": [
                {"lat": 0.0001, "lon": 0.0001}, {"lat": 0.0001, "lon": 0.0005},
                {"lat": 0.0005, "lon": 0.0005}, {"lat": 0.0005, "lon": 0.0001},
            ],
        }]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff", "width": 128, "height": 128, "count": 1,
                "dtype": "float32", "crs": "EPSG:3857",
                "transform": from_bounds(0, 0, 128, 128, 128, 128),
            }
            with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
                dst.write(np.full((1, 128, 128), 20, dtype=np.float32))
            with patch("backend.services.building_service._query_overpass", return_value=payload):
                buildings, _ = fetch_buildings(
                    {"north": 1, "south": -1, "east": 1, "west": -1},
                    root / "dsm.tif", root / "dsm.tif",
                    endpoint="https://example.test/overpass",
                    source_gsd_m=10,
                )
        self.assertEqual(len(buildings), 1)
        self.assertEqual(buildings[0]["geometry_profile"], "temple")
        self.assertEqual(buildings[0]["height"], 9.0)
        self.assertEqual(buildings[0]["height_source"], "osm:approximate-temple-prior")

    def test_untagged_coarse_building_is_explicitly_approximate(self):
        payload = {"elements": [{
            "type": "way", "id": 456, "tags": {"building": "yes"},
            "geometry": [
                {"lat": 0.0001, "lon": 0.0001}, {"lat": 0.0001, "lon": 0.0002},
                {"lat": 0.0002, "lon": 0.0002}, {"lat": 0.0002, "lon": 0.0001},
            ],
        }]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff", "width": 128, "height": 128, "count": 1,
                "dtype": "float32", "crs": "EPSG:3857",
                "transform": from_bounds(0, 0, 128, 128, 128, 128),
            }
            with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
                dst.write(np.full((1, 128, 128), 20, dtype=np.float32))
            with patch("backend.services.building_service._query_overpass", return_value=payload):
                buildings, info = fetch_buildings(
                    {"north": 1, "south": -1, "east": 1, "west": -1},
                    root / "dsm.tif", root / "dsm.tif",
                    endpoint="https://example.test/overpass",
                    source_gsd_m=10,
                )
        self.assertEqual(len(buildings), 1)
        self.assertEqual(buildings[0]["height"], 3.0)
        self.assertEqual(buildings[0]["height_source"], "osm:approximate-context")
        self.assertIsNotNone(info["warning"])

    def test_environment_features_are_projected_from_one_osm_payload(self):
        payload = {"elements": [
            {
                "type": "way", "id": 700, "tags": {"highway": "residential", "name": "Test Road"},
                "geometry": [
                    {"lat": 0.0001, "lon": 0.0001}, {"lat": 0.0001, "lon": 0.0008},
                ],
            },
            {
                "type": "way", "id": 701, "tags": {"natural": "water"},
                "geometry": [
                    {"lat": 0.0003, "lon": 0.0003}, {"lat": 0.0003, "lon": 0.0007},
                    {"lat": 0.0007, "lon": 0.0007}, {"lat": 0.0007, "lon": 0.0003},
                ],
            },
            {"type": "node", "id": 702, "tags": {"natural": "tree"}, "lat": 0.0008, "lon": 0.0008},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff", "width": 128, "height": 128, "count": 1,
                "dtype": "float32", "crs": "EPSG:3857",
                "transform": from_bounds(0, 0, 128, 128, 128, 128),
            }
            with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
                dst.write(np.full((1, 128, 128), 20, dtype=np.float32))
            with patch("backend.services.building_service._query_overpass", return_value=payload):
                buildings, _, environment, info = fetch_map_features(
                    {"north": 1, "south": -1, "east": 1, "west": -1},
                    root / "dsm.tif", root / "dsm.tif",
                    endpoint="https://example.test/overpass",
                    source_gsd_m=10,
                )
        self.assertEqual(buildings, [])
        self.assertEqual(len(environment["roads"]), 1)
        self.assertEqual(len(environment["water"]), 1)
        self.assertEqual(len(environment["trees"]), 1)
        self.assertEqual(info["trees_count"], 1)

    def test_sports_ground_affine_pixels_are_unpacked_as_numeric_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff", "width": 40, "height": 40, "count": 3,
                "dtype": "uint8", "crs": "EPSG:3857",
                "transform": from_bounds(0, 0, 40, 40, 40, 40),
            }
            rgb = np.full((3, 40, 40), 120, dtype=np.uint8)
            rgb[0, 8:33, 8:33] = 35
            rgb[1, 8:33, 8:33] = 145
            rgb[2, 8:33, 8:33] = 40
            with rasterio.open(root / "rgb.tif", "w", **profile) as dst:
                dst.write(rgb)
            with rasterio.open(root / "dsm.tif", "w", **{**profile, "count": 1, "dtype": "float32"}) as dst:
                dst.write(np.full((1, 40, 40), 10.0, dtype=np.float32))

            zones = detect_sports_ground_zones(
                root / "rgb.tif",
                root / "dsm.tif",
                minimum_area_m2=100.0,
                minimum_pixels=20,
            )

        self.assertTrue(zones)
        self.assertTrue(all(isinstance(value, float) for point in zones[0]["polygon"] for value in point))

    def test_malformed_osm_layers_return_warnings_and_empty_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_profile = {
                "driver": "GTiff", "width": 16, "height": 16, "count": 3,
                "dtype": "uint8", "crs": "EPSG:3857",
                "transform": from_bounds(0, 0, 16, 16, 16, 16),
            }
            with rasterio.open(root / "rgb.tif", "w", **raster_profile) as dst:
                dst.write(np.full((3, 16, 16), 120, dtype=np.uint8))
            with rasterio.open(root / "dsm.tif", "w", **{**raster_profile, "count": 1, "dtype": "float32"}) as dst:
                dst.write(np.full((1, 16, 16), 10, dtype=np.float32))

            with patch("backend.services.building_service._query_overpass", return_value={"elements": [None]}):
                buildings, building_info, environment, environment_info = fetch_map_features(
                    {"north": 1, "south": -1, "east": 1, "west": -1},
                    root / "rgb.tif",
                    root / "dsm.tif",
                    endpoint="https://example.test/overpass",
                    max_buildings=0,
                    max_features=10,
                    max_trees=0,
                    cache_root=root / "cache",
                )

        self.assertEqual(buildings, [])
        self.assertEqual(environment["roads"], [])
        self.assertIn("warning", building_info)
        self.assertIn("warning", environment_info)


if __name__ == "__main__":
    unittest.main()
