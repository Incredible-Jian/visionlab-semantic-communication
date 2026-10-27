"""设备选择工具。本项目 CPU 优先。"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """返回计算设备，固定为 ``cpu``。

    本项目默认且仅支持 CPU：即使检测到 CUDA GPU，也返回 CPU，避免在无 GPU
    环境下意外触发 CUDA 错误。后续阶段如确需 GPU，再单独扩展此函数。
    """
    return torch.device("cpu")
