"""固定随机种子，保证实验可复现。"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """为 Python、NumPy 与 PyTorch 统一设置随机种子。

    参数
    ----
    seed : int
        随机种子，任意整数。

    说明
    ----
    固定种子只能保证在**相同环境**下可复现：相同的 PyTorch 版本、相同的
    数据生成与迭代顺序。本项目 CPU 优先，暂不涉及 CUDA 的非确定性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
