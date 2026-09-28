import unittest

import numpy as np

from asterra_tiled_inference import _pad_tile_for_model


class InferencePaddingTests(unittest.TestCase):
    def test_small_map_tile_is_centred_and_reflectively_padded(self):
        tile = np.ones((128, 160, 3), dtype=np.float32)
        padded, crop = _pad_tile_for_model(tile)

        self.assertEqual(padded.shape, (512, 512, 3))
        top, left, height, width = crop
        self.assertEqual((height, width), (128, 160))
        self.assertEqual(top, (512 - 128) // 2)
        self.assertEqual(left, (512 - 160) // 2)
        # A constant tile remains constant; zero padding would fail this and
        # reintroduce the artificial black scene boundary.
        np.testing.assert_allclose(padded, 1.0)

    def test_partial_edge_tile_keeps_original_corner(self):
        tile = np.arange(512 * 400 * 3, dtype=np.float32).reshape(512, 400, 3)
        padded, crop = _pad_tile_for_model(tile)

        self.assertEqual(crop, (0, 0, 512, 400))
        np.testing.assert_array_equal(padded[:, :400], tile)


if __name__ == "__main__":
    unittest.main()
