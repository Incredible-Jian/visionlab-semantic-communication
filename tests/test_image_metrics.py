"""测试 visionlab.utils.image_metrics 的 MSE / PSNR。"""

import numpy as np
import pytest

from visionlab.utils.image_metrics import mse, psnr


def test_mse_identical_is_zero() -> None:
    a = np.random.rand(8, 8)
    assert mse(a, a) == 0.0


def test_mse_known_value() -> None:
    a = np.zeros((2, 2))
    b = np.ones((2, 2))
    assert mse(a, b) == 1.0


def test_mse_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mse(np.zeros((2, 2)), np.zeros((3, 3)))


def test_psnr_identical_is_inf() -> None:
    a = np.full((4, 4), 0.5)
    assert psnr(a, a) == float("inf")


def test_psnr_known_value() -> None:
    # mse = 1，max_val=1 -> PSNR = 10*log10(1/1) = 0 dB
    a = np.zeros((2, 2))
    b = np.ones((2, 2))
    assert abs(psnr(a, b, max_val=1.0) - 0.0) < 1e-9
    # max_val=255 -> PSNR = 10*log10(255^2)
    assert abs(psnr(a, b, max_val=255.0) - 10 * np.log10(255 ** 2)) < 1e-9
