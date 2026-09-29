import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import trimesh
from rasterio.transform import from_origin

from backend.services.building_service import filter_buildings_by_evidence, sanitize_environment_layers
from visualization.generate_3d import run_reconstruction


class StructureRepairTests(unittest.TestCase):
    def _rasters(self, root, size=32):
        profile = {
            "driver": "GTiff",
            "width": size,
            "height": size,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:3857",
            "transform": from_origin(0, size, 1, 1),
        }
        dsm = np.full((size, size), 10.0, dtype=np.float32)
        rgb = np.full((3, size, size), 110, dtype=np.uint8)
        with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
            dst.write(dsm, 1)
        with rasterio.open(root / "rgb.tif", "w", **{**profile, "count": 3, "dtype": "uint8"}) as dst:
            dst.write(rgb)
        return dsm, rgb

    def test_green_cricket_field_is_rejected_but_small_roof_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dsm, rgb = self._rasters(root)
            dsm[4:28, 4:28] = 14.0
            dsm[20:24, 20:24] = 16.0
            rgb[:, 4:28, 4:28] = np.asarray([35, 125, 40], dtype=np.uint8)[:, None, None]
            rgb[:, 20:24, 20:24] = np.asarray([145, 145, 135], dtype=np.uint8)[:, None, None]
            with rasterio.open(root / "dsm.tif", "r+") as dst:
                dst.write(dsm, 1)
            with rasterio.open(root / "rgb.tif", "r+") as dst:
                dst.write(rgb)

            field = {
                "id": 900000001,
                "source": "semantic-segmentation",
                "polygon_projected": [[4, 28], [28, 28], [28, 4], [4, 4]],
                "height": 4.0,
                "ground_elevation": 10.0,
            }
            small = {
                "id": 900000002,
                "source": "semantic-segmentation",
                "polygon_projected": [[20, 12], [24, 12], [24, 8], [20, 8]],
                "height": 4.0,
                "ground_elevation": 10.0,
            }
            accepted, report = filter_buildings_by_evidence(
                [field, small], root / "rgb.tif", root / "dsm.tif", {"exclusion_zones": []}
            )
            self.assertEqual([item["id"] for item in accepted], [900000002])
            self.assertGreaterEqual(report["rejection_reasons"].get("insufficient-roof-evidence", 0), 1)

    def test_clearance_removes_roof_tree_and_clips_landcover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dsm, _ = self._rasters(root, size=20)
            dsm[5:10, 5:10] = 25.0
            with rasterio.open(root / "dsm.tif", "r+") as dst:
                dst.write(dsm, 1)
            buildings = [{
                "id": 1,
                "polygon_projected": [[5, 15], [10, 15], [10, 10], [5, 10]],
                "ground_elevation": 10.0,
            }]
            environment = {
                "trees": [{
                    "id": 2,
                    "point_projected": [7.5, 12.5],
                    "canopy_radius": 1.5,
                    "ground_elevation": 25.0,
                }],
                "landcover": [{
                    "id": 3,
                    "polygon_projected": [[2, 18], [14, 18], [14, 6], [2, 6]],
                    "ground_elevation": 10.0,
                }],
                "water": [],
            }
            filtered, removed = sanitize_environment_layers(
                buildings,
                environment,
                raster_path=root / "rgb.tif",
                dsm_path=root / "dsm.tif",
                clearance_m=1.0,
            )
            self.assertEqual(len(filtered["trees"]), 0)
            self.assertGreaterEqual(removed["trees"], 1)
            self.assertGreaterEqual(removed["landcover_clipped"], 1)
            self.assertTrue(all(item.get("ground_elevation_source") == "local-dsm-lower-percentile" for item in filtered["landcover"]))

    def test_small_structures_export_as_separate_nodes_and_tall_building_is_taller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff", "width": 32, "height": 32, "count": 1,
                "dtype": "float32", "crs": "EPSG:3857", "transform": from_origin(0, 32, 1, 1),
            }
            with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
                dst.write(np.full((1, 32, 32), 10.0, dtype=np.float32))
            with rasterio.open(root / "rgb.tif", "w", **{**profile, "count": 3, "dtype": "uint8"}) as dst:
                dst.write(np.full((3, 32, 32), 120, dtype=np.uint8))
            result = run_reconstruction(
                root / "dsm.tif", root / "rgb.tif", root / "model.glb", size=32,
                buildings=[
                    {"id": 1, "polygon_projected": [[2, 30], [6, 30], [6, 27], [2, 27]], "ground_elevation": 10.0, "height": 2.5, "roof_elevation": 12.5},
                    {"id": 2, "polygon_projected": [[12, 30], [18, 30], [18, 24], [12, 24]], "ground_elevation": 10.0, "height": 12.0, "roof_elevation": 22.0},
                ],
                environment={"roads": [], "water": [], "landcover": [], "trees": []},
            )
            scene = trimesh.load(root / "model.glb", force="scene")
            small = scene.geometry[next(name for name in scene.geometry if name.startswith("BUILDING_1_"))]
            tall = scene.geometry[next(name for name in scene.geometry if name.startswith("BUILDING_2_"))]
            self.assertGreater(float(np.ptp(tall.vertices[:, 1])), float(np.ptp(small.vertices[:, 1])))
            self.assertEqual(result["metadata"]["coordinate_system"], "Local Y-up mesh coordinates")

    def test_evidence_backed_roof_profiles_are_classified_and_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "driver": "GTiff", "width": 40, "height": 40, "count": 1,
                "dtype": "float32", "crs": "EPSG:3857", "transform": from_origin(0, 40, 1, 1),
            }
            with rasterio.open(root / "dsm.tif", "w", **profile) as dst:
                dst.write(np.full((1, 40, 40), 10.0, dtype=np.float32))
            with rasterio.open(root / "rgb.tif", "w", **{**profile, "count": 3, "dtype": "uint8"}) as dst:
                dst.write(np.full((3, 40, 40), 120, dtype=np.uint8))
            buildings = [
                {"id": 11, "polygon_projected": [[2, 38], [12, 38], [12, 30], [2, 30]], "ground_elevation": 10, "height": 6, "roof_elevation": 16, "height_source": "osm:height", "geometry_profile": "gable"},
                {"id": 12, "polygon_projected": [[16, 38], [26, 38], [26, 30], [16, 30]], "ground_elevation": 10, "height": 6, "roof_elevation": 16, "height_source": "osm:height", "geometry_profile": "hip"},
                {"id": 13, "polygon_projected": [[28, 38], [38, 38], [38, 30], [28, 30]], "ground_elevation": 10, "height": 6, "roof_elevation": 16, "height_source": "osm:height", "geometry_profile": "compound"},
            ]
            result = run_reconstruction(root / "dsm.tif", root / "rgb.tif", root / "model.glb", size=40, buildings=buildings, environment={"roads": [], "water": [], "landcover": [], "trees": []})
            scene = trimesh.load(root / "model.glb", force="scene")
            self.assertTrue(any(name.startswith("ROOF_BUILDING_11_") for name in scene.geometry))
            self.assertTrue(any(name.startswith("ROOF_BUILDING_12_") for name in scene.geometry))
            self.assertTrue(any(name.startswith("ROOF_BUILDING_13_") for name in scene.geometry))
            for name, geometry in scene.geometry.items():
                if name.startswith(("BUILDING_", "ROOF_BUILDING_")):
                    self.assertTrue(geometry.is_watertight, name)
            self.assertEqual(result["metadata"]["coordinate_system"], "Local Y-up mesh coordinates")
            self.assertEqual(result["metadata"]["scene_alignment"]["mesh_grid"]["width"], 40)


if __name__ == "__main__":
    unittest.main()
