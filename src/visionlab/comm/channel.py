"""模拟通信信道：可控加性高斯噪声（简化模型）。"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(x: np.ndarray, sigma: float) -> np.ndarray:
    """给图像加上均值为 0、标准差为 ``sigma`` 的高斯噪声，并裁剪到 [0, 1]。

    参数
    ----
    x : np.ndarray
        输入图像，浮点数组，值域 [0, 1]。
    sigma : float
        噪声标准差（``>= 0``）。``sigma=0`` 时返回原图副本。

    说明
    ----
    噪声由全局 NumPy 随机数生成器产生，调用前可用 ``set_seed`` 固定以保证
    可复现。这里模拟的是"传输信道引入的加性噪声"，是教学用的简化模型。
    """
    x = np.asarray(x, dtype=float)
    if sigma < 0:
        raise ValueError(f"sigma 不能为负：{sigma}")
    if sigma == 0.0:
        return x.copy()
    noise = np.random.normal(0.0, sigma, size=x.shape)
    return np.clip(x + noise, 0.0, 1.0)
