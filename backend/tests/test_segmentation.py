import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.segmentation.class_map import BUILDING, VEGETATION, WATER
from backend.segmentation.engine import _heuristic_segmentation, segment_rgb
from backend.services.segmentation_service import run_segmentation


class SegmentationFallbackTests(unittest.TestCase):
    def test_rgb_fallback_produces_semantic_classes_without_bitwise_type_error(self):
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[:, :32] = (35, 120, 45)       # vegetation
        rgb[:, 32:64] = (30, 80, 180)    # water
        rgb[:, 64:] = (190, 155, 110)     # warm roof/ground evidence

        result = _heuristic_segmentation(rgb)
        self.assertEqual(result.mask.shape, (96, 96))
        self.assertIn(VEGETATION, result.classes_present)
        self.assertIn(WATER, result.classes_present)

        result2, info = segment_rgb(rgb, allow_model=False)
        self.assertIsNotNone(result2)
        self.assertEqual(info["model"], "heuristic-rgb-v1")
        self.assertIn(BUILDING, result2.classes_present)

    def test_model_failure_still_persists_deterministic_mask_and_preview(self):
        class Settings:
            segmentation_enabled = True
            segmentation_timeout_seconds = 1.0

        rgb = np.zeros((3, 96, 96), dtype=np.uint8)
        rgb[:, :, :32] = np.array([[[35]], [[120]], [[45]]], dtype=np.uint8)
        rgb[:, :, 32:64] = np.array([[[30]], [[80]], [[180]]], dtype=np.uint8)
        rgb[:, :, 64:] = np.array([[[190]], [[155]], [[110]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.tif"
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=96,
                height=96,
                count=3,
                dtype="uint8",
                crs="EPSG:3857",
                transform=from_origin(0, 96, 1, 1),
            ) as dst:
                dst.write(rgb)

            real_segment = segment_rgb

            def fail_model_then_fallback(data, allow_model=True):
                if allow_model:
                    raise RuntimeError("simulated model failure")
                return real_segment(data, allow_model=False)

            with patch("backend.services.segmentation_service.segment_rgb", side_effect=fail_model_then_fallback):
                info = run_segmentation(Settings(), source, root / "mask.tif", root / "mask.png")

            self.assertTrue(info["fallback"])
            self.assertIn("model failed", info["reason"])
            self.assertTrue((root / "mask.tif").exists())
            self.assertTrue((root / "mask.png").exists())


if __name__ == "__main__":
    unittest.main()
