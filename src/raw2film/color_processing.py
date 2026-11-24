"""
Helper functions for color math.
"""

import numpy as np
from spectral_film_lut.config import DEFAULT_DTYPE


def linear_gradient_weight(
    shape: tuple[int, int],
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray:
    """Return a smooth weight ramp from full effect at start to zero at end."""
    height, width = shape
    start_x, start_y = start
    end_x, end_y = end
    direction_x = end_x - start_x
    direction_y = end_y - start_y
    length_squared = direction_x * direction_x + direction_y * direction_y
    if length_squared <= 1e-12:
        return np.zeros((height, width), dtype=np.float32)

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[np.newaxis, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
    distance = (
        (x - start_x) * direction_x + (y - start_y) * direction_y
    ) / length_squared
    t = np.clip(distance, 0.0, 1.0)
    ramp = t * t * (3.0 - 2.0 * t)
    return (1.0 - ramp).astype(np.float32, copy=False)


def apply_linear_gradient_mask(
    image: np.ndarray,
    adjusted: np.ndarray,
    gradient: dict,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Blend a gradient layer, optionally adding its delta to an existing composite."""
    image = np.asarray(image)
    adjusted = np.asarray(adjusted)
    if not gradient.get("enabled", True):
        return image
    if base is not None:
        base = np.asarray(base)
    if (
        image.shape != adjusted.shape
        or image.ndim != 3
        or (base is not None and base.shape != image.shape)
    ):
        raise ValueError(
            "image and adjusted must have the same (height, width, channels) shape"
        )

    weight = linear_gradient_weight(
        image.shape[:2],
        tuple(gradient["start"]),
        tuple(gradient["end"]),
    )[..., np.newaxis]
    if base is None:
        blended = image.astype(np.float32) * (1.0 - weight) + adjusted.astype(
            np.float32
        ) * weight
    else:
        blended = image.astype(np.float32) + (
            adjusted.astype(np.float32) - base.astype(np.float32)
        ) * weight
    if np.issubdtype(image.dtype, np.integer):
        blended = np.rint(blended)
        blended = np.clip(
            blended, np.iinfo(image.dtype).min, np.iinfo(image.dtype).max
        )
    return blended.astype(image.dtype, copy=False)


def apply_adaptive_blacks(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Adjust linear RGB with a black-protecting lift and adaptive soft crush."""
    amount = float(np.clip(amount, -100.0, 100.0))
    if amount == 0.0:
        return rgb

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be an array with shape (height, width, 3)")

    luminance = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=rgb.dtype),
        axis=-1,
    )
    strength = abs(amount) / 100.0

    if amount > 0.0:
        positive_luminance = np.maximum(luminance, 0.0)
        lift_weight = np.divide(
            positive_luminance,
            positive_luminance + 0.02,
        ) * np.exp(-positive_luminance / 0.5)
        lift = 0.03 * strength * lift_weight
        adjusted = rgb + lift[..., np.newaxis]
    else:
        sample_step = max(1, int(np.ceil(luminance.size / 65536)))
        sampled_luminance = luminance.reshape(-1)[::sample_step]
        positive_darks = sampled_luminance[sampled_luminance > 1e-6]
        if positive_darks.size == 0:
            return rgb

        black_level = np.percentile(positive_darks, 5.0)
        knee = max(0.25, 8.0 * black_level)
        attenuation = 1.0 - 0.4 * strength * np.exp(
            -np.maximum(luminance, 0.0) / knee
        )
        scale = np.divide(
            np.maximum(luminance, 0.0),
            luminance,
            out=np.zeros_like(luminance),
            where=luminance > 0.0,
        )
        scale *= attenuation
        adjusted = rgb * scale[..., np.newaxis]

    return adjusted.astype(rgb.dtype, copy=False)


def apply_smart_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    """Adjust saturation while preserving luminance and protecting skin tones."""
    saturation = float(np.clip(saturation, 0.0, 2.0))
    if saturation == 1.0:
        return rgb

    rgb = np.asarray(rgb)
    channel_max = np.max(rgb, axis=-1)
    channel_min = np.min(rgb, axis=-1)
    channel_range = channel_max - channel_min
    safe_range = np.where(channel_range > 0, channel_range, 1.0)
    hue = np.select(
        [
            channel_max == rgb[..., 0],
            channel_max == rgb[..., 1],
            channel_max == rgb[..., 2],
        ],
        [
            60.0 * np.mod((rgb[..., 1] - rgb[..., 2]) / safe_range, 6.0),
            60.0 * ((rgb[..., 2] - rgb[..., 0]) / safe_range + 2.0),
            60.0 * ((rgb[..., 0] - rgb[..., 1]) / safe_range + 4.0),
        ],
        default=0.0,
    )
    hsv_saturation = np.divide(
        channel_range,
        channel_max,
        out=np.zeros_like(channel_range),
        where=channel_max > 0,
    )

    skin_hue_weight = np.clip((hue - 5.0) / 10.0, 0.0, 1.0) * np.clip(
        (55.0 - hue) / 15.0, 0.0, 1.0
    )
    skin_saturation_weight = np.clip((hsv_saturation - 0.08) / 0.12, 0.0, 1.0)
    skin_saturation_weight *= np.clip((0.85 - hsv_saturation) / 0.25, 0.0, 1.0)
    skin_protection = skin_hue_weight * skin_saturation_weight

    red_distance = np.minimum(hue, 360.0 - hue)
    red_weight = np.clip((30.0 - red_distance) / 20.0, 0.0, 1.0)
    red_weight *= np.clip((hsv_saturation - 0.12) / 0.25, 0.0, 1.0)
    red_weight *= 1.0 - skin_protection
    saturation_response = 1.0 - 0.99 * skin_protection
    if saturation > 1.0:
        saturation_response += 0.25 * red_weight
    requested_saturation = 1.0 + (saturation - 1.0) * saturation_response

    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    luminance_weights = np.array([0.2126, 0.7152, 0.0722], dtype=linear_rgb.dtype)
    luminance = np.sum(
        linear_rgb * luminance_weights, axis=-1, keepdims=True
    )
    chroma = linear_rgb - luminance

    max_saturation = np.full_like(chroma, np.inf)
    np.divide(
        1.0 - luminance,
        chroma,
        out=max_saturation,
        where=chroma > 0,
    )
    np.divide(
        -luminance,
        chroma,
        out=max_saturation,
        where=chroma < 0,
    )
    max_saturation = np.min(max_saturation, axis=-1, keepdims=True)
    saturation_scale = np.maximum(
        0.0, np.minimum(requested_saturation[..., np.newaxis], max_saturation)
    )
    adjusted = np.clip(luminance + chroma * saturation_scale, 0.0, 1.0)

    return np.where(
        adjusted <= 0.0031308,
        12.92 * adjusted,
        1.055 * np.maximum(adjusted, 0.0) ** (1.0 / 2.4) - 0.055,
    )


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
    highlight_percentile: float = 99.0,
    near_white_threshold: float = 0.92,
    clip_threshold: float = 0.995,
    clip_fraction_threshold: float = 0.001,
    near_white_fraction_threshold: float = 0.005,
) -> float:
    """Calculate exposure while preserving highlight headroom in all RGB channels.

    Midtone exposure is based on the green-channel percentile. After that estimate
    is calculated, each channel is checked for elevated highlights, near-white
    values, and clipped values. If a channel would be pushed beyond a safe
    highlight level, the exposure compensation is capped.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be an array with shape (height, width, 3)")

    sampled = rgb[::2, ::2]
    lum = np.clip(sampled[..., 1], 1e-6, 1.0)

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

    flash_fired = False
    if metadata is not None:
        flash_value = metadata.get("EXIF:Flash", metadata.get("Flash"))
        if isinstance(flash_value, (bool, np.bool_)):
            flash_fired = bool(flash_value)
        elif isinstance(flash_value, (int, np.integer)):
            flash_fired = bool(flash_value & 0x01)
        elif isinstance(flash_value, str):
            value = flash_value.strip().lower()
            flash_fired = "fired" in value and "did not" not in value

    safe_max = red_safe_max
    if flash_fired:
        safe_max = min(safe_max, 0.88)

    for channel in range(3):
        channel_lum = np.clip(sampled[..., channel], 1e-6, 1.0)
        protect_percentile = (
            red_protect_percentile if channel == 0 else highlight_percentile
        )
        channel_high = np.percentile(channel_lum, protect_percentile)
        if channel_high <= 1e-6:
            continue

        predicted_high = channel_high * (2**exp_comp)
        channel_to_green_ratio = channel_high / target_lum
        clipped_fraction = float(np.mean(channel_lum >= clip_threshold))
        near_white_fraction = float(
            np.mean(channel_lum >= near_white_threshold)
        )

        needs_guard = (
            (
                predicted_high > safe_max
                or (flash_fired and channel_high > safe_max)
            )
            and (
                channel_to_green_ratio > 1.3
                or clipped_fraction >= clip_fraction_threshold
                or near_white_fraction >= near_white_fraction_threshold
            )
        )
        if needs_guard:
            max_allowed_comp = np.log2(safe_max / channel_high)
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
