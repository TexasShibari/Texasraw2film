import importlib
import unittest

import numpy as np

effects = importlib.import_module("raw2film.effects")


class CropPanTests(unittest.TestCase):
    def test_pan_moves_zoom_crop_to_image_edges(self):
        y, x = np.indices((8, 8))
        image = np.stack((x, y, np.zeros_like(x)), axis=-1).astype(np.uint8)

        left = effects.crop_image(image, zoom=2, aspect=1, pan_x=-1)
        center = effects.crop_image(image, zoom=2, aspect=1)
        right = effects.crop_image(image, zoom=2, aspect=1, pan_x=1)
        top = effects.crop_image(image, zoom=2, aspect=1, pan_y=-1)
        bottom = effects.crop_image(image, zoom=2, aspect=1, pan_y=1)

        self.assertEqual(left.shape, (4, 4, 3))
        self.assertEqual(int(left[0, 0, 0]), 0)
        self.assertEqual(int(center[0, 0, 0]), 2)
        self.assertEqual(int(right[0, 0, 0]), 4)
        self.assertEqual(int(top[0, 0, 1]), 0)
        self.assertEqual(int(bottom[0, 0, 1]), 4)

    def test_pan_is_ignored_without_zoom(self):
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))

        centered = effects.crop_image(image, zoom=1, aspect=1)
        panned = effects.crop_image(image, zoom=1, aspect=1, pan_x=1, pan_y=-1)

        np.testing.assert_array_equal(panned, centered)


if __name__ == "__main__":
    unittest.main()
