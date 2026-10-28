"""卷积神经网络（CNN）图像分类器。"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN(nn.Module):
    """两层卷积 + 两层全连接的图像分类网络。

    结构：``Conv(1→16, 3×3, pad=1) → ReLU → MaxPool(2) → Conv(16→32, 3×3,
    pad=1) → ReLU → MaxPool(2) → Flatten → Linear(32*7*7 → 64) → ReLU →
    Linear(64 → num_classes)``。

    输入 ``28×28`` 经过两次 2×2 池化后变为 ``7×7``，因此全连接层输入为
    ``32 * 7 * 7 = 1568``。输出为 ``(B, num_classes)`` 的 logits。

    参数
    ----
    in_channels : int
        输入通道数，灰度图为 1。
    num_classes : int
        类别数，FashionMNIST 为 10。
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：``(B, in_channels, 28, 28) -> (B, num_classes)``。"""
        return self.classifier(self.features(x))
