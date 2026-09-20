"""Targeted tests for SUIM mask decoding and the documented 55-pixel repair."""
import importlib.util
import tempfile
import unittest
from pathlib import Path


DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("numpy", "PIL", "torch"))


@unittest.skipUnless(DEPENDENCIES, "numpy, Pillow, and torch are required")
class DataTests(unittest.TestCase):
    def test_rgb_bits_and_near_palette_values(self):
        import numpy as np
        from PIL import Image
        from experiment.fewShot_SUIM.data import decode_mask

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.bmp"
            values = np.array([
                [(0, 0, 0), (0, 0, 254), (0, 255, 0), (0, 255, 255)],
                [(254, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255)],
            ], dtype=np.uint8)
            Image.fromarray(values, mode="RGB").save(path)
            labels, audit = decode_mask(path, (4, 2))
            self.assertEqual(labels.tolist(), [[0, 1, 2, 3], [4, 5, 6, 7]])
            self.assertEqual(audit["off_palette_pixels"], 2)

    def test_known_extra_bottom_is_cropped(self):
        import numpy as np
        from PIL import Image
        from experiment.fewShot_SUIM.data import decode_mask

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.bmp"
            values = np.full((57, 3, 3), 255, dtype=np.uint8)
            values[:2] = 0
            Image.fromarray(values, mode="RGB").save(path)
            labels, audit = decode_mask(path, (3, 2))
            self.assertEqual(labels.shape, (2, 3))
            self.assertTrue(audit["size_repaired"])
            self.assertTrue((labels == 0).all())


if __name__ == "__main__":
    unittest.main()

