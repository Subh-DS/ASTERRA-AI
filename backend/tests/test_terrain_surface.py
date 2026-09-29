import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.services.building_service import sanitize_environment_layers
from visualization.terrain_surface import build_terrain_surface


class TerrainSurfaceTests(unittest.TestCase):
    def test_surface_flattens_accepted_objects_without_mutating_raw_dsm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "raw.tif"
            surface_path = root / "terrain_surface.tif"
            profile = {
                "driver": "GTiff", "width": 12, "height": 12, "count": 1,
                "dtype": "float32", "crs": "EPSG:3857",
                "transform": from_origin(0, 12, 1, 1),
            }
            raw = np.full((12, 12), 10.0, dtype=np.float32)
            raw[1:4, 1:4] = 28.0
            raw[8:11, 8:11] = 18.0  # surrounding terrain remains relief
            with rasterio.open(raw_path, "w", **profile) as dst:
                dst.write(raw, 1)
            original = raw.copy()
            buildings = [{
                "id": 1,
                "polygon_projected": [[1, 11], [4, 11], [4, 8], [1, 8]],
                "ground_elevation": 10.0,
            }]
            environment = {"trees": [{
                "id": 2,
                "point_projected": [9.5, 2.5],
                "ground_elevation": 10.0,
                "canopy_radius": 1.5,
            }]}
            info = build_terrain_surface(raw_path, surface_path, buildings, environment)
            with rasterio.open(raw_path) as src:
                np.testing.assert_array_equal(src.read(1), original)
            with rasterio.open(surface_path) as src:
                surface = src.read(1)
            self.assertEqual(info["raw_dsm_unchanged"], True)
            self.assertEqual(float(surface[2, 2]), 10.0)
            self.assertEqual(float(surface[9, 9]), 10.0)
            self.assertEqual(float(surface[5, 5]), 10.0)

    def test_context_proxies_inside_buildings_are_removed(self):
        buildings = [{"polygon_projected": [[0, 0], [10, 0], [10, 10], [0, 10]]}]
        environment = {
            "trees": [
                {"point_projected": [5, 5], "canopy_radius": 1.0},
                {"point_projected": [20, 20], "canopy_radius": 1.0},
            ],
            "landcover": [{"polygon_projected": [[-2, -2], [12, -2], [12, 12], [-2, 12]]}],
            "water": [],
        }
        filtered, removed = sanitize_environment_layers(buildings, environment)
        self.assertEqual(len(filtered["trees"]), 1)
        self.assertEqual(removed["trees"], 1)
        self.assertEqual(len(filtered["landcover"]), 0)
        self.assertEqual(removed["landcover"], 1)

    def test_sports_ground_is_flattened_without_touching_surrounding_hills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "raw.tif"
            surface_path = root / "terrain_surface.tif"
            profile = {
                "driver": "GTiff", "width": 16, "height": 16, "count": 1,
                "dtype": "float32", "crs": "EPSG:3857",
                "transform": from_origin(0, 16, 1, 1),
            }
            raw = np.full((16, 16), 10.0, dtype=np.float32)
            raw[3:11, 3:11] = 24.0
            raw[12:15, 12:15] = 18.0
            with rasterio.open(raw_path, "w", **profile) as dst:
                dst.write(raw, 1)
            info = build_terrain_surface(
                raw_path,
                surface_path,
                environment={"exclusion_zones": [{
                    "polygon_projected": [[3, 13], [11, 13], [11, 5], [3, 5]],
                    "ground_elevation": 10.0,
                }]},
            )
            with rasterio.open(surface_path) as src:
                surface = src.read(1)
            self.assertEqual(float(surface[6, 6]), 10.0)
            self.assertEqual(float(surface[13, 13]), 18.0)
            self.assertEqual(info["layer_counts"]["sports_ground"], 1)


if __name__ == "__main__":
    unittest.main()
