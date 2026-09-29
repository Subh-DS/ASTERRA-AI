import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import trimesh
from rasterio.transform import from_origin

from visualization.generate_3d import run_reconstruction


class BuildingGlbTests(unittest.TestCase):
    def test_reconstruction_exports_terrain_and_building_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff",
                "width": 16,
                "height": 16,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:3857",
                "transform": from_origin(0, 160, 10, 10),
            }
            with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
                dst.write(np.full((1, 16, 16), 100.0, dtype=np.float32))
            with rasterio.open(root / "rgb.tif", "w", **{**profile, "count": 3, "dtype": "uint8"}) as dst:
                dst.write(np.full((3, 16, 16), 120, dtype=np.uint8))

            result = run_reconstruction(
                root / "dsm.tif",
                root / "rgb.tif",
                root / "model.glb",
                size=16,
                buildings=[{
                    "id": 42,
                    "polygon_projected": [[30, 130], [70, 130], [70, 90], [30, 90]],
                    "ground_elevation": 100.0,
                    "roof_elevation": 109.0,
                    "height": 9.0,
                    "geometry_profile": "temple",
                }],
                environment={
                    "roads": [{
                        "id": 7,
                        "path_projected": [[10, 140], [120, 140]],
                        "width_m": 5,
                        "ground_elevation": 100,
                    }],
                    "water": [{
                        "id": 8,
                        "polygon_projected": [[80, 80], [120, 80], [120, 40], [80, 40]],
                        "ground_elevation": 100,
                    }],
                    "landcover": [],
                    "trees": [{
                        "id": 9,
                        "point_projected": [110, 25],
                        "ground_elevation": 100,
                        "height": 8,
                        "canopy_radius": 2.5,
                    }],
                },
            )

            scene = trimesh.load(root / "model.glb", force="scene")
            self.assertIn("TERRAIN", scene.geometry)
            self.assertTrue(any(name.startswith("BUILDING_42_") for name in scene.geometry))
            self.assertTrue(any(name.startswith("ROAD_7_") for name in scene.geometry))
            self.assertTrue(any(name.startswith("WATER_8_") for name in scene.geometry))
            self.assertTrue(any(name.startswith("VEGETATION_TREE_9_") for name in scene.geometry))
            building = next(scene.geometry[name] for name in scene.geometry if name.startswith("BUILDING_42_"))
            roof = next(scene.geometry[name] for name in scene.geometry if name.startswith("ROOF_BUILDING_42_"))
            self.assertTrue(building.is_watertight)
            self.assertTrue(roof.is_watertight)
            # GLB coordinates are local to the terrain vertical origin (100 m
            # in this fixture), and the exported contract is Y-up. The temple
            # roof must exceed the local 9 m building roof rather than the
            # absolute source elevation.
            self.assertGreater(float(roof.vertices[:, 1].max()), 9.0)
            self.assertEqual(result["metadata"]["coordinate_system"], "Local Y-up mesh coordinates")
            self.assertEqual(result["metadata"]["buildings_detected"], 1)
            self.assertTrue(result["metadata"]["has_environment"])
            self.assertGreaterEqual(result["metadata"]["terrain_overrides"], 2)

    def test_dsm_roof_surface_is_variable_and_semantic_layers_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff",
                "width": 24,
                "height": 24,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:3857",
                "transform": from_origin(0, 240, 1, 1),
            }
            rows, cols = np.mgrid[0:24, 0:24]
            dsm = 100.0 + cols.astype(np.float32) * 0.8 + rows.astype(np.float32) * 0.25
            with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
                dst.write(dsm[np.newaxis, ...])
            with rasterio.open(root / "rgb.tif", "w", **{**profile, "count": 3, "dtype": "uint8"}) as dst:
                dst.write(np.stack([cols * 8, rows * 8, np.full_like(cols, 120)], axis=0).astype(np.uint8))

            result = run_reconstruction(
                root / "dsm.tif",
                root / "rgb.tif",
                root / "model.glb",
                size=24,
                source_gsd_m=1.0,
                buildings=[{
                    "id": 7,
                    "polygon_projected": [[5, 230], [15, 230], [15, 220], [5, 220]],
                    "ground_elevation": 100.0,
                    "height": 3.0,
                    "roof_elevation": 103.0,
                    "geometry_profile": "standard",
                }],
                environment={
                    "roads": [],
                    "water": [],
                    "landcover": [],
                    "trees": [],
                    "semantic_regions": [{
                        "class": "vegetation",
                        "polygon_projected": [[16, 230], [20, 230], [20, 226], [16, 226]],
                        "ground_elevation": 100.0,
                    }],
                },
            )

            scene = trimesh.load(root / "model.glb", force="scene")
            roof = next(scene.geometry[name] for name in scene.geometry if name.startswith("ROOF_BUILDING_7_"))
            self.assertGreater(float(np.ptp(roof.vertices[:, 1])), 0.5)
            self.assertEqual(result["metadata"]["reconstruction"]["roof_measured"], 1)
            self.assertEqual(result["metadata"]["environment_counts"]["semantic_layers"]["vegetation"], 1)
            self.assertEqual(result["metadata"]["geometry_qa"]["building_meshes_watertight"], 1)
            self.assertEqual(result["metadata"]["geometry_qa"]["building_roofs_watertight"], 1)
            self.assertTrue(np.isfinite(roof.visual.uv).all())


if __name__ == "__main__":
    unittest.main()
