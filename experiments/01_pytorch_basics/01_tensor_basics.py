"""实验 01：Tensor 基础操作。

演示张量的创建、形状、索引、变形、广播与矩阵运算。
运行：
    python experiments/01_pytorch_basics/01_tensor_basics.py
"""

from __future__ import annotations

import torch

from visionlab.utils.seed import set_seed


def main() -> None:
    set_seed(42)

    print("=" * 60)
    print("1. 张量创建")
    print("=" * 60)
    x = torch.tensor([1, 2, 3])       # 从列表创建（int64）
    z = torch.zeros(2, 3)             # 全 0
    a = torch.arange(6)               # 0..5
    r = torch.randn(2, 3)             # 标准正态
    print("x =", x, "| dtype =", x.dtype)
    print("zeros(2, 3):\n", z)
    print("arange(6) =", a)
    print("randn(2, 3):\n", r)

    print("\n" + "=" * 60)
    print("2. 形状与索引")
    print("=" * 60)
    m = torch.arange(12).reshape(3, 4)
    print("m (3x4):\n", m)
    print("m.shape =", tuple(m.shape), "| m.ndim =", m.ndim)
    print("m[1, 2] =", m[1, 2].item())
    print("m[:, 1] =", m[:, 1])

    print("\n" + "=" * 60)
    print("3. 变形与转置")
    print("=" * 60)
    print("m.view(4, 3):\n", m.view(4, 3))
    print("m.T (转置):\n", m.T)

    print("\n" + "=" * 60)
    print("4. 广播")
    print("=" * 60)
    v = torch.tensor([10, 20, 30, 40])   # shape (4,)
    print("m + v  (广播: (3,4) + (4,)):\n", m + v)

    print("\n" + "=" * 60)
    print("5. 矩阵运算")
    print("=" * 60)
    a2 = torch.randn(2, 3)
    b2 = torch.randn(3, 2)
    print("a2 @ b2 (2x3 @ 3x2 -> 2x2):\n", a2 @ b2)
    print("a2 * a2 (逐元素):\n", a2 * a2)


if __name__ == "__main__":
    main()
