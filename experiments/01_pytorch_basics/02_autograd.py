"""实验 02：Autograd 与梯度计算。

演示 ``requires_grad``、``backward()``、梯度取值与梯度累积。
运行：
    python experiments/01_pytorch_basics/02_autograd.py
"""

from __future__ import annotations

import torch


def main() -> None:
    print("=" * 60)
    print("1. 标量求导：f(x) = x^2，f'(x) = 2x")
    print("=" * 60)
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    print("x = 3 时 dy/dx =", x.grad.item(), "（期望 6）")

    print("\n" + "=" * 60)
    print("2. 复合函数：f(x) = x^3 + 2x^2 + x")
    print("=" * 60)
    x2 = torch.tensor(1.0, requires_grad=True)
    y2 = x2 ** 3 + 2 * x2 ** 2 + x2
    y2.backward()
    # f'(x) = 3x^2 + 4x + 1，x=1 时为 8
    print("x = 1 时 df/dx =", x2.grad.item(), "（期望 8）")

    print("\n" + "=" * 60)
    print("3. 多元函数：f(a, b) = a*b + b^2")
    print("=" * 60)
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    f = a * b + b ** 2
    f.backward()
    print("df/da =", a.grad.item(), "（期望 b = 3）")
    print("df/db =", b.grad.item(), "（期望 a + 2b = 8）")

    print("\n" + "=" * 60)
    print("4. 梯度累积：backward() 会累加到 .grad")
    print("=" * 60)
    w = torch.tensor(2.0, requires_grad=True)
    for _ in range(3):
        (w ** 2).backward()
    print("连续 3 次 backward 后 w.grad =", w.grad.item(), "（期望 12 = 3 * 4）")
    w.grad.zero_()
    (w ** 2).backward()
    print("清零后再 backward，w.grad =", w.grad.item(), "（期望 4）")


if __name__ == "__main__":
    main()
