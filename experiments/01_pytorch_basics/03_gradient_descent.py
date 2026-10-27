"""实验 03：手动梯度下降 vs torch.optim.SGD。

最小化 f(x) = (x - 3)^2（最小值在 x = 3），对比：
    - 手动更新：x -= lr * x.grad
    - 优化器：torch.optim.SGD
运行：
    python experiments/01_pytorch_basics/03_gradient_descent.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "01_pytorch_basics"


def run_manual_gd(lr: float, steps: int) -> list[float]:
    """手动梯度下降，返回每一步的损失。"""
    x = torch.tensor(0.0, requires_grad=True)
    losses: list[float] = []
    for _ in range(steps):
        loss = (x - 3.0) ** 2
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad          # 手动更新参数
        x.grad.zero_()
        losses.append(loss.item())
    return losses


def run_optim_sgd(lr: float, steps: int) -> list[float]:
    """使用 torch.optim.SGD，返回每一步的损失。"""
    x = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        optimizer.step()              # 优化器更新参数
        losses.append(loss.item())
    return losses


def main() -> None:
    lr, steps = 0.1, 30
    manual = run_manual_gd(lr, steps)
    optim = run_optim_sgd(lr, steps)

    print(f"学习率 {lr}，迭代 {steps} 步，目标 f(x) = (x-3)^2")
    print(f"{'步数':>4} {'手动GD loss':>14} {'optim.SGD loss':>16}")
    for i in range(0, steps, 3):
        print(f"{i:>4} {manual[i]:>14.6f} {optim[i]:>16.6f}")
    print(f"手动GD    最终 loss = {manual[-1]:.6f}")
    print(f"optim.SGD 最终 loss = {optim[-1]:.6f}")
    print("两者都应趋近 0（即 x -> 3）。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(manual, label="manual GD")
    plt.plot(optim, label="torch.optim.SGD")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    plt.title("Gradient descent comparison")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "03_gradient_descent.png", dpi=120)
    plt.close()
    print(f"对比图已保存到 {OUTPUT_DIR / '03_gradient_descent.png'}")


if __name__ == "__main__":
    main()
