"""测试 visionlab.utils.device.get_device。"""

import torch

from visionlab.utils.device import get_device


def test_get_device_returns_cpu() -> None:
    assert get_device() == torch.device("cpu")
