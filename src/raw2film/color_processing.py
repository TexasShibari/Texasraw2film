"""
Helper functions for color math.
"""

import numpy as np
from spectral_film_lut.config import DEFAULT_DTYPE


def XYZ_to_kelvin(XYZ):
    """Convert from CIE XYZ 1931 to the CCT in kelvin."""
    x = XYZ[0] / np.sum(XYZ)
    y = XYZ[1] / np.sum(XYZ)
    n = (x - 0.3366) / (y - 0.1735)
    CCT = (
        -949.86315
        + 6253.80338 * np.exp(-n / 0.92159)
        + 28.70599 * np.exp(-n / 0.20039)
        + 0.00004 * np.exp(-n / 0.07125)
    )
    return CCT


def kelvin_to_XYZ(CCT):
    """Convert from a CCT in kelvin to CIE XYZ 1931."""
    # This section is ripped from the Colour Science package:
    CCT_3 = CCT**3
    CCT_2 = CCT**2

    x = np.where(
        CCT <= 4000,
        -0.2661239 * 10**9 / CCT_3
        - 0.2343589 * 10**6 / CCT_2
        + 0.8776956 * 10**3 / CCT
        + 0.179910,
        -3.0258469 * 10**9 / CCT_3
        + 2.1070379 * 10**6 / CCT_2
        + 0.2226347 * 10**3 / CCT
        + 0.24039,
    )

    x_3 = x**3
    x_2 = x**2

    cnd_l = [CCT <= 2222, np.logical_and(CCT > 2222, CCT <= 4000), CCT > 4000]
    i = -1.1063814 * x_3 - 1.34811020 * x_2 + 2.18555832 * x - 0.20219683
    j = -0.9549476 * x_3 - 1.37418593 * x_2 + 2.09137015 * x - 0.16748867
    k = 3.0817580 * x_3 - 5.8733867 * x_2 + 3.75112997 * x - 0.37001483
    y = np.select(cnd_l, [i, j, k])

    XYZ = np.array([x / y, 1, (1 - x - y) / y])
    return XYZ


def encode_ARRILogC3(x):
    """Encode the ARRI Log C3 EOTF."""
    cut, a, b, c, d, e, f = (
        0.010591,
        5.555556,
        0.052272,
        0.247190,
        0.385537,
        5.367655,
        0.092809,
    )

    return np.where(x > cut, (c / np.log(10)) * np.log(a * x + b) + d, e * x + f)


def calc_exposure(
    rgb: np.ndarray,
    ref_exposure: float = 0.18,
    metadata: dict | None = None,
    percentile: float = 75.0,
    red_protect_percentile: float = 95.0,
    red_safe_max: float = 0.92,
) -> float:
    """Exposure compensation based on a high percentile of green-channel luminance.

    Uses a high percentile (default 75) so large dark areas do not dominate
    the calculation. This keeps bright subjects from clipping on high-contrast scenes.
    """
    lum = rgb[::2, ::2, 1]
    lum = np.clip(lum, 1e-6, 1.0)

    # Optional very light bias from camera EXIF (can be removed if preferred)
    bias = 0.0
    if metadata is not None:
        try:
            iso = float(metadata.get("EXIF:ISO", 100))
            exp_time = float(metadata.get("EXIF:ExposureTime", 1 / 125))
            bias = np.log2(iso / 100) * 0.1 + np.log2(exp_time * 125) * 0.05
        except (TypeError, ValueError):
            bias = 0.0

    target_lum = np.percentile(lum, percentile)
    if target_lum <= 1e-6:
        return 0.0

    exp_comp = np.log2(ref_exposure / target_lum) + bias

    red_lum = rgb[::2, ::2, 0]
    red_lum = np.clip(red_lum, 1e-6, 1.0)
    red_high = np.percentile(red_lum, red_protect_percentile)

    predicted_red = red_high * (2**exp_comp)
    red_to_green_ratio = red_high / target_lum
    if predicted_red > red_safe_max and red_high > 1e-6 and red_to_green_ratio > 1.3:
        max_allowed_comp = np.log2(red_safe_max / red_high)
        exp_comp = min(exp_comp, max_allowed_comp)

    return float(np.clip(exp_comp, -3.0, 3.0))


def xyz_to_srgb(XYZ, M=None, output_uint8=True, clip=True, apply_matrix=True):
    """
    Convert from XYZ (D65) to sRGB using CuPy.

    Parameters:
        XYZ (cp.ndarray): Input array of shape (..., 3), dtype float32 or float64.
        output_uint8 (bool): If True, returns output in 0–255 uint8 range.
        clip (bool): If True, clips RGB values to [0, 1] before gamma.

    Returns:
        cp.ndarray: sRGB values, same shape as XYZ (or uint8 if output_uint8 is True).
    """
    # XYZ to linear RGB matrix (sRGB, D65)
    if M is None:
        M = np.array(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ],
            dtype=XYZ.dtype,
        )

    # Linear RGB
    if apply_matrix:
        RGB_linear = XYZ @ M.T
    else:
        RGB_linear = XYZ

    # Optional clipping before gamma (standard practice)
    if clip:
        RGB_linear = np.clip(RGB_linear, 0.0, 1.0)

    # Apply sRGB gamma encoding
    a = 0.055
    threshold = 0.0031308
    RGB = np.where(
        RGB_linear <= threshold,
        12.92 * RGB_linear,
        (1 + a) * np.power(RGB_linear, 1 / 2.4) - a,
    )

    # Optional output in uint8
    if output_uint8:
        RGB = np.clip(RGB, 0.0, 1.0) * 255
        RGB = RGB.get().astype(np.uint8)

    return RGB


def xyz_to_displayP3(XYZ, **kwargs):
    """Convert from CIE XYZ 1931 to diplay P3 gamut."""
    M = np.array(
        [
            [2.493496911941425, -0.9313836179191239, -0.40271078445071684],
            [-0.8294889695615747, 1.7626640603183463, 0.023624685841943577],
            [0.03584583024378447, -0.07617238926804182, 0.9568845240076872],
        ],
        dtype=XYZ.dtype,
    )

    return xyz_to_srgb(XYZ, M, **kwargs)


def rec709_to_displayP3(rgb):
    """Convert from rec. 709 to diplay P3 gamut."""
    M = np.array(
        [
            [0.822462, 0.177538, 0.000000],
            [0.033194, 0.966806, 0.000000],
            [0.017083, 0.072397, 0.910520],
        ],
        dtype=DEFAULT_DTYPE,
    )
    rgb = np.clip(rgb @ M.T, 0, 255).astype(np.uint8)
    return rgb
