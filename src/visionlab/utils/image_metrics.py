"""图像质量指标：MSE 与 PSNR（numpy 自实现，不依赖 scipy/skimage）。

输入约定：图像为浮点数组，值域通常为 [0, 1]（本项目灰度图已归一化到该区间）。
"""

from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """均方误差 MSE = mean((a - b)^2)，越小越好。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"两张图像形状不一致：{a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> float:
    """峰值信噪比 PSNR = 10 * log10(max_val^2 / MSE)，单位 dB，越大越好。

    ``max_val`` 是像素可能的最大值：归一化图像为 1.0，8-bit 图像为 255.0。
    两张图完全相同时返回 ``inf``。
    """
    m = mse(a, b)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / m))
