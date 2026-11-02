"""测试 visionlab.comm 的噪声信道与编解码。"""

import numpy as np
import pytest

from visionlab.comm import add_gaussian_noise, downsample_codec, jpeg_codec


def test_add_gaussian_noise_zero_sigma_is_identity() -> None:
    x = np.random.rand(4, 4)
    assert np.array_equal(add_gaussian_noise(x, 0.0), x)


def test_add_gaussian_noise_preserves_shape_and_range() -> None:
    x = np.zeros((10, 10))
    y = add_gaussian_noise(x, 0.5)
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_add_gaussian_noise_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError):
        add_gaussian_noise(np.zeros((2, 2)), -1.0)


def test_downsample_codec_preserves_shape_and_range() -> None:
    x = np.random.rand(28, 28)
    y = downsample_codec(x, factor=2)
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_jpeg_codec_preserves_shape_and_range() -> None:
    x = np.random.rand(28, 28)
    y = jpeg_codec(x, quality=75)
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_jpeg_codec_rejects_bad_quality() -> None:
    with pytest.raises(ValueError):
        jpeg_codec(np.zeros((2, 2)), quality=101)
