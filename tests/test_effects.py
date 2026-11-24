import unittest
from types import SimpleNamespace

import numpy as np

from raw2film.effects import blue_burn, burn, red_burn


class HighlightBurnTests(unittest.TestCase):
    def test_signed_amounts_raise_or_lower_highlights(self):
        image = np.full((20, 20, 3), 2.0, dtype=np.float32)
        negative_film = SimpleNamespace(log_H_ref=np.array([0.0, 0.0, 0.0]))

        burned = burn(image.copy(), negative_film, 1.0, 50)
        dodged = burn(image.copy(), negative_film, -1.0, 50)
        unchanged = burn(image.copy(), negative_film, 0.0, 50)

        self.assertTrue(np.all(burned < image))
        self.assertTrue(np.all(dodged > image))
        np.testing.assert_array_equal(unchanged, image)


class RedBurnTests(unittest.TestCase):
    def test_reduces_only_red_channel(self):
        image = np.ones((20, 20, 3), dtype=np.float32)
        image[..., 0] = 2.0
        image[..., 1] = 1.5
        negative_film = SimpleNamespace(
            density_measure="rgb",
            log_H_ref=np.array([1.0, 1.0, 1.0]),
        )

        result = red_burn(image.copy(), negative_film, 1.0, 50)

        self.assertTrue(np.all(result[..., 0] < image[..., 0]))
        np.testing.assert_array_equal(result[..., 1], image[..., 1])
        np.testing.assert_array_equal(result[..., 2], image[..., 2])

    def test_does_nothing_for_black_and_white(self):
        image = np.ones((20, 20, 3), dtype=np.float32)
        negative_film = SimpleNamespace(
            density_measure="bw",
            log_H_ref=np.array([1.0]),
        )

        result = red_burn(image.copy(), negative_film, 1.0, 50)

        np.testing.assert_array_equal(result, image)

    def test_reduction_is_strongest_at_midtone_reference(self):
        negative_film = SimpleNamespace(
            density_measure="rgb",
            log_H_ref=np.array([0.0]),
        )
        midtone = np.ones((20, 20, 3), dtype=np.float32)
        midtone[..., 0] = 0.0
        highlight = midtone.copy()
        highlight[..., 0] = 100.0
        shadow = midtone.copy()
        shadow[..., 0] = -1.0

        midtone_result = red_burn(midtone.copy(), negative_film, 1.0, 50)
        highlight_result = red_burn(highlight.copy(), negative_film, 1.0, 50)
        shadow_result = red_burn(shadow.copy(), negative_film, 1.0, 50)

        midtone_reduction = 0.0 - midtone_result[0, 0, 0]
        highlight_reduction = 100.0 - highlight_result[0, 0, 0]
        shadow_reduction = abs(shadow_result[0, 0, 0] - shadow[0, 0, 0])

        self.assertGreater(midtone_reduction, highlight_reduction)
        self.assertGreater(midtone_reduction, shadow_reduction)


class BlueBurnTests(unittest.TestCase):
    def test_reduces_only_blue_dominant_pixels(self):
        image = np.ones((20, 20, 3), dtype=np.float32)
        image[..., 0] = 0.4
        image[..., 1] = 0.5
        image[..., 2] = 1.0
        negative_film = SimpleNamespace(density_measure="rgb")

        result = blue_burn(image.copy(), negative_film, 1.0, 50)

        self.assertTrue(np.all(result[..., 2] < image[..., 2]))
        np.testing.assert_array_equal(result[..., 0], image[..., 0])
        np.testing.assert_array_equal(result[..., 1], image[..., 1])

    def test_does_nothing_for_balanced_pixels(self):
        image = np.ones((20, 20, 3), dtype=np.float32)
        negative_film = SimpleNamespace(density_measure="rgb")

        result = blue_burn(image.copy(), negative_film, 1.0, 50)

        np.testing.assert_array_equal(result, image)

    def test_does_nothing_for_black_and_white(self):
        image = np.ones((20, 20, 3), dtype=np.float32)
        negative_film = SimpleNamespace(density_measure="bw")

        result = blue_burn(image.copy(), negative_film, 1.0, 50)

        np.testing.assert_array_equal(result, image)


if __name__ == "__main__":
    unittest.main()
