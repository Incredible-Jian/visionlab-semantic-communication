"""实验 04：线性回归。

用 ``nn.Linear(1, 1)`` + MSE 损失 + SGD，拟合 y = 2x + 1 + 噪声。
输出：拟合权重/偏置、损失曲线、数据与拟合直线图。
运行：
    python experiments/01_pytorch_basics/04_linear_regression.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from visionlab.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "01_pytorch_basics"


def make_data(
    n: int = 100,
    true_w: float = 2.0,
    true_b: float = 1.0,
    noise: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 y = true_w * x + true_b + 高斯噪声。"""
    x = torch.linspace(-3, 3, n).reshape(-1, 1)
    y = true_w * x + true_b + noise * torch.randn_like(x)
    return x, y


def main() -> None:
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, y = make_data()
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    epochs = 200
    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    w, b = model.weight.item(), model.bias.item()
    print("真实参数：w = 2.0, b = 1.0")
    print(f"拟合参数：w = {w:.4f}, b = {b:.4f}")
    print(f"损失：初始 {losses[0]:.4f} -> 最终 {losses[-1]:.4f}")

    # 损失曲线
    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Linear regression: training loss")
    plt.savefig(OUTPUT_DIR / "04_linear_regression_loss.png", dpi=120)
    plt.close()

    # 数据与拟合直线
    plt.figure()
    with torch.no_grad():
        x_line = torch.linspace(-3, 3, 100).reshape(-1, 1)
        y_line = model(x_line)
    plt.scatter(x.numpy(), y.numpy(), s=10, label="data")
    plt.plot(x_line.numpy(), y_line.numpy(), color="red", label="fitted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear regression: fit")
    plt.savefig(OUTPUT_DIR / "04_linear_regression_fit.png", dpi=120)
    plt.close()

    print(f"图已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
