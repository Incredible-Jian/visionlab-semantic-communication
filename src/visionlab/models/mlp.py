"""多层感知机（MLP）图像分类器。"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """把灰度图展平后送入全连接网络的分类器。

    输入形状 ``(B, 1, 28, 28)`` 会先展平为 ``(B, 784)``，再经过若干
    ``Linear -> ReLU -> Dropout`` 隐藏层，最后输出 ``(B, num_classes)`` 的
    类别 logits（未做 softmax，配合 ``nn.CrossEntropyLoss`` 使用）。

    参数
    ----
    input_dim : int
        展平后的输入特征维数，FashionMNIST 为 28*28=784。
    hidden_dims : tuple[int, ...]
        各隐藏层神经元数，默认 ``(128, 64)``。
    num_classes : int
        类别数，FashionMNIST 为 10。
    dropout : float
        每个隐藏层后的 Dropout 概率，0 表示关闭。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (128, 64),
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：``(B, 1, 28, 28) -> (B, num_classes)``。"""
        return self.net(x.reshape(x.shape[0], -1))
