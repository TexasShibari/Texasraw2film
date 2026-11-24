import unittest

import numpy as np

from raw2film.color_processing import (
    apply_adaptive_blacks,
    apply_linear_gradient_mask,
    apply_smart_saturation,
    calc_exposure,
    linear_gradient_weight,
)


class LinearGradientMaskTests(unittest.TestCase):
    def test_smooth_weight_runs_from_full_effect_to_zero(self):
        weights = linear_gradient_weight((3, 5), (0.0, 0.5), (1.0, 0.5))

        np.testing.assert_array_equal(weights[1, [0, -1]], [1.0, 0.0])
        self.assertAlmostEqual(float(weights[1, 2]), 0.5)
        self.assertTrue(np.all(np.diff(weights[1]) <= 0))

    def test_diagonal_and_degenerate_gradients(self):
        diagonal = linear_gradient_weight((3, 5), (0.0, 0.0), (1.0, 1.0))
        degenerate = linear_gradient_weight((3, 5), (0.5, 0.5), (0.5, 0.5))

        self.assertAlmostEqual(float(diagonal[1, 2]), 0.5)
        np.testing.assert_array_equal(degenerate, np.zeros((3, 5), dtype=np.float32))

    def test_mask_blends_only_inside_gradient_and_scales_to_output_shape(self):
        original = np.zeros((3, 5, 3), dtype=np.uint8)
        adjusted = np.full_like(original, 100)
        gradient = {
            "start": [0.0, 0.5],
            "end": [1.0, 0.5],
            "width": 100,
            "height": 60,
        }

        result = apply_linear_gradient_mask(original, adjusted, gradient)

        np.testing.assert_array_equal(result[1, 0], [100, 100, 100])
        np.testing.assert_array_equal(result[1, -1], [0, 0, 0])
        np.testing.assert_array_equal(result[1, 2], [50, 50, 50])
        self.assertEqual(result.dtype, original.dtype)

    def test_disabled_gradient_leaves_global_render_unchanged(self):
        original = np.zeros((2, 2, 3), dtype=np.uint8)
        adjusted = np.full_like(original, 100)
        gradient = {
            "start": [0.0, 0.5],
            "end": [1.0, 0.5],
            "enabled": False,
        }

        result = apply_linear_gradient_mask(original, adjusted, gradient)

        self.assertIs(result, original)

    def test_multiple_gradient_layers_add_to_the_global_render(self):
        global_render = np.zeros((1, 3, 3), dtype=np.uint8)
        first_adjustment = np.full_like(global_render, 10)
        second_adjustment = np.full_like(global_render, 20)
        gradient = {
            "start": [0.0, 0.0],
            "end": [1.0, 0.0],
        }

        first_layer = apply_linear_gradient_mask(
            global_render, first_adjustment, gradient
        )
        combined = apply_linear_gradient_mask(
            first_layer, second_adjustment, gradient, base=global_render
        )

        np.testing.assert_array_equal(combined[0, :, 0], [30, 15, 0])


class AdaptiveBlacksTests(unittest.TestCase):
    def test_positive_protects_black_and_lifts_darks_and_midtones(self):
        rgb = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.005, 0.005, 0.005],
                    [0.03, 0.03, 0.03],
                    [0.2, 0.2, 0.2],
                    [0.8, 0.8, 0.8],
                ]
            ],
            dtype=np.float32,
        )

        result = apply_adaptive_blacks(rgb, 100)

        np.testing.assert_array_equal(result[0, 0], rgb[0, 0])
        self.assertGreater(result[0, 1, 0], rgb[0, 1, 0])
        self.assertGreater(result[0, 2, 0], rgb[0, 2, 0])
        midtone_lift = result[0, 3, 0] - rgb[0, 3, 0]
        shadow_lift = result[0, 1, 0] - rgb[0, 1, 0]
        self.assertGreater(midtone_lift, shadow_lift)
        self.assertGreater(result[0, 4, 0], rgb[0, 4, 0])
        self.assertLess(result[0, 4, 0] - rgb[0, 4, 0], midtone_lift)

    def test_negative_adaptively_deepens_darks_with_a_soft_knee(self):
        rgb = np.array(
            [
                [
                    [0.002, 0.002, 0.002],
                    [0.01, 0.01, 0.01],
                    [0.05, 0.05, 0.05],
                    [0.18, 0.18, 0.18],
                    [0.8, 0.8, 0.8],
                ]
            ],
            dtype=np.float32,
        )

        result = apply_adaptive_blacks(rgb, -100)

        self.assertLess(result[0, 0, 0], rgb[0, 0, 0])
        self.assertGreater(result[0, 0, 0], 0.0)
        self.assertLess(result[0, 1, 0], rgb[0, 1, 0])
        self.assertLess(result[0, 3, 0], rgb[0, 3, 0] * 0.9)
        self.assertLess(result[0, 4, 0], rgb[0, 4, 0])
        self.assertGreater(result[0, 4, 0], rgb[0, 4, 0] - 0.02)

    def test_zero_amount_returns_input_unchanged(self):
        rgb = np.array([[[0.0, 0.1, 0.2]]], dtype=np.float32)

        result = apply_adaptive_blacks(rgb, 0)

        self.assertIs(result, rgb)


class SmartSaturationTests(unittest.TestCase):
    @staticmethod
    def linear_luminance(rgb):
        linear_rgb = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )
        return np.sum(linear_rgb * [0.2126, 0.7152, 0.0722], axis=-1)

    def test_increases_saturation_without_changing_luminance(self):
        rgb = np.array([[[0.8, 0.3, 0.2], [0.2, 0.5, 0.8]]], dtype=np.float32)

        result = apply_smart_saturation(rgb, 1.8)

        np.testing.assert_allclose(
            self.linear_luminance(result), self.linear_luminance(rgb), atol=1e-6
        )
        self.assertEqual(result.dtype, rgb.dtype)
        self.assertGreater(
            result[0, 0, 0] - result[0, 0, 1],
            rgb[0, 0, 0] - rgb[0, 0, 1],
        )

    def test_protects_skin_tones_and_emphasizes_reds(self):
        rgb = np.array([[[0.82, 0.58, 0.45], [0.5, 0.18, 0.15]]], dtype=np.float32)

        result = apply_smart_saturation(rgb, 1.8)

        np.testing.assert_allclose(
            self.linear_luminance(result), self.linear_luminance(rgb), atol=1e-6
        )
        input_linear = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )
        result_linear = np.where(
            result <= 0.04045,
            result / 12.92,
            ((result + 0.055) / 1.055) ** 2.4,
        )
        input_luminance = self.linear_luminance(rgb)[..., np.newaxis]
        result_luminance = self.linear_luminance(result)[..., np.newaxis]
        input_chroma = np.linalg.norm(input_linear - input_luminance, axis=-1)
        result_chroma = np.linalg.norm(result_linear - result_luminance, axis=-1)
        chroma_gain = result_chroma / input_chroma

        self.assertLess(chroma_gain[0, 0], chroma_gain[0, 1])

    def test_neutral_colors_and_default_setting_are_unchanged(self):
        rgb = np.array([[[0.4, 0.4, 0.4], [0.8, 0.3, 0.2]]], dtype=np.float32)

        result = apply_smart_saturation(rgb, 0.0)
        default_result = apply_smart_saturation(rgb, 1.0)

        np.testing.assert_allclose(result[0, 0], np.repeat(result[0, 0, :1], 3))
        np.testing.assert_array_equal(default_result, rgb)


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

    def test_protects_elevated_blue_highlights(self):
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
        rgb[:, :, 1] = 0.10
        rgb[:, :, 2] = 0.80

        result = calc_exposure(rgb)

        expected = np.log2(0.92 / 0.80)
        self.assertAlmostEqual(result, float(expected), places=6)

    def test_clipped_pixels_limit_exposure_even_when_not_in_percentile(self):
        rgb = np.zeros((100, 100, 3), dtype=np.float32)
        rgb[:, :, 1] = 0.18
        rgb[:, :, 0] = 0.60
        rgb[:2, :, 0] = 1.0
        rgb[:2, :, 1] = 1.0

        result = calc_exposure(rgb)

        self.assertLessEqual(result, np.log2(0.92 / 1.0) + 1e-6)

    def test_flash_metadata_uses_more_conservative_headroom(self):
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
        rgb[:, :, 1] = 0.18
        rgb[:, :, 0] = 0.90

        normal = calc_exposure(rgb)
        flash = calc_exposure(rgb, metadata={"EXIF:Flash": "Flash fired"})

        self.assertLess(flash, normal)
        self.assertAlmostEqual(flash, np.log2(0.88 / 0.90), places=6)


if __name__ == "__main__":
    unittest.main()
