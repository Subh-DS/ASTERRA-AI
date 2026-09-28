import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend import pipeline as pipeline_module
from backend.storage import JobStore


AOI = {"north": 20.305, "south": 20.295, "east": 85.81, "west": 85.80}


class PipelineTerminationTests(unittest.TestCase):
    def test_mocked_temple_job_reaches_terminal_state_with_hybrid_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = JobStore(root / "jobs")
            job = store.create("ram-mandir.png", "image/png")
            store.update(
                job["job_id"],
                source_type="map",
                aoi=AOI,
                imagery_provider="esri",
                imagery_quality="medium",
                imagery_item_id=None,
                imagery_max_cloud=None,
            )
            settings = SimpleNamespace(
                project_root=root,
                min_raster_dimension=1,
                max_raster_pixels=100_000,
                imagery_timeout=1,
                dem_provider="none",
                dem_path=None,
                dem_online=False,
                dem_cache_root=root / "dem-cache",
                dem_download_timeout=1,
                dem_max_download_bytes=100_000,
                buildings_enabled=True,
                buildings_overpass_url="test",
                buildings_timeout=1,
                buildings_max=10,
                buildings_min_area_m2=20,
                environment_enabled=True,
                environment_max_features=10,
                environment_max_trees=10,
                mesh_size=16,
            )
            input_meta = {
                "georeferenced": True,
                "crs": "EPSG:3857",
                "bounds": (0, 0, 96, 96),
                "resolution": (1, 1),
                "pixel_size_m": (1, 1),
                "reprojected_to_metric_crs": False,
                "input_dtype_normalized": True,
            }
            temple = {
                "id": 789,
                "name": "Ram Mandir",
                "polygon_projected": [[20, 76], [76, 76], [76, 20], [20, 20]],
                "polygon": [[20, 20], [76, 20], [76, 76], [20, 76]],
                "area_m2": 3136,
                "ground_elevation": 100,
                "height": 12,
                "roof_elevation": 112,
                "height_source": "osm:approximate-context",
                "geometry_profile": "temple",
                "confidence": 0.62,
                "source": "openstreetmap",
            }
            environment = {"roads": [], "water": [], "landcover": [], "trees": []}

            def fake_materialize(_source, output, *_args, **_kwargs):
                with rasterio.open(
                    output,
                    "w",
                    driver="GTiff",
                    width=96,
                    height=96,
                    count=3,
                    dtype="uint8",
                    crs="EPSG:3857",
                    transform=from_origin(0, 96, 1, 1),
                ) as dst:
                    dst.write(np.full((3, 96, 96), 120, dtype=np.uint8))
                return input_meta

            def fake_relative(_prediction, output, _metadata):
                with rasterio.open(
                    output,
                    "w",
                    driver="GTiff",
                    width=96,
                    height=96,
                    count=1,
                    dtype="float32",
                    crs="EPSG:3857",
                    transform=from_origin(0, 96, 1, 1),
                ) as dst:
                    dst.write(np.full((1, 96, 96), 100, dtype=np.float32))
                return {
                    "output_path": str(output),
                    "metadata": {"is_metric": False, "status": "relative"},
                    "width": 96,
                    "height": 96,
                    "stats": {},
                }

            with patch.object(
                pipeline_module,
                "prepare_map_scene",
                return_value={
                    "provider": "esri",
                    "requested_provider": "auto",
                    "resolved_provider": "esri",
                    "resolution_m": 1.0,
                    "fallback_chain": [],
                },
            ), patch.object(pipeline_module, "materialize_input", side_effect=fake_materialize), patch.object(
                pipeline_module,
                "run_segmentation",
                return_value={
                    "enabled": True,
                    "available": False,
                    "fallback": True,
                    "reason": "model unavailable",
                    "classes_present": [],
                    "class_counts": {},
                },
            ), patch.object(pipeline_module.inference_service, "run", return_value={"model": "mock", "device": "cpu"}), patch.object(
                pipeline_module, "resolve_dem_info", return_value=None
            ), patch.object(pipeline_module, "write_relative_dsm", side_effect=fake_relative), patch.object(
                pipeline_module,
                "fetch_map_features",
                return_value=(
                    [temple],
                    {"enabled": True, "count": 1, "approximate_count": 1},
                    environment,
                    {"enabled": True, "warning": None, "approximate_tree_count": 0},
                ),
            ), patch.object(
                pipeline_module,
                "add_semantic_features",
                return_value=(environment, [temple], {"regions_count": 0, "building_candidates": 0, "approximate_tree_count": 0}),
            ), patch.object(
                pipeline_module,
                "reconstruct",
                return_value={"metadata": {"vertical_origin_m": 100, "has_glb": True}},
            ), patch.object(pipeline_module, "write_preview_assets"):
                pipeline_module.run_pipeline(job["job_id"], store, settings)

            result = store.get(job["job_id"])
            self.assertEqual(result["status"], "complete")
            self.assertTrue(all(stage["state"] == "done" for stage in result["stages"].values()))
            self.assertEqual(result["result"]["reconstruction"]["scene_quality"], "hybrid")


if __name__ == "__main__":
    unittest.main()
