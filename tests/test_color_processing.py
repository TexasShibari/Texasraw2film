import unittest

import numpy as np

from raw2film.color_processing import calc_exposure


class CalcExposureTests(unittest.TestCase):
    def test_uses_green_channel_percentile(self):
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
        rgb[:, :, 1] = 0.01
        sampled = np.full((4, 4), 0.18, dtype=np.float32)
        sampled[0, 0:2] = 0.01
        sampled[1, 0:2] = 0.01
        rgb[0::2, 0::2, 1] = sampled

        result = calc_exposure(rgb)

        self.assertEqual(result, 0.0)

    def test_clips_result_to_ui_range(self):
        rgb = np.zeros((4, 4, 3), dtype=np.float32)
        rgb[:, :, 1] = 0.001

        result = calc_exposure(rgb)

        self.assertEqual(result, 3.0)

    def test_metadata_bias_and_malformed_metadata(self):
        rgb = np.ones((4, 4, 3), dtype=np.float32) * 0.18
        metadata = {
            "EXIF:ISO": "200",
            "EXIF:ExposureTime": "0.008",
        }

        result = calc_exposure(rgb, metadata=metadata)

        self.assertAlmostEqual(result, 0.100 + 0.05 * np.log2(0.008 * 125))

    def test_downsampling_preserves_low_sample_behavior(self):
        rgb = np.zeros((4, 4, 3), dtype=np.float32)
        rgb[0::2, 0::2, 1] = 0.18

        result = calc_exposure(rgb)

        self.assertEqual(result, 0.0)

    def test_reduces_boost_to_protect_elevated_red(self):
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
        rgb[:, :, 1] = 0.10
        rgb[:, :, 0] = 0.70

        result = calc_exposure(rgb)

        expected = np.log2(0.92 / 0.70)
        self.assertAlmostEqual(result, float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
