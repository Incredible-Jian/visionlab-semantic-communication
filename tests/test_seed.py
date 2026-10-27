"""测试 visionlab.utils.seed.set_seed 的可复现性。"""

import random

import numpy as np
import torch

from visionlab.utils.seed import set_seed


def test_python_random_is_reproducible() -> None:
    set_seed(0)
    a = [random.random() for _ in range(10)]
    set_seed(0)
    b = [random.random() for _ in range(10)]
    assert a == b


def test_numpy_random_is_reproducible() -> None:
    set_seed(0)
    a = np.random.rand(10)
    set_seed(0)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_torch_random_is_reproducible() -> None:
    set_seed(0)
    a = torch.randn(10)
    set_seed(0)
    b = torch.randn(10)
    assert torch.equal(a, b)
