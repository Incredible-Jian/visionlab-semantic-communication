"""实验 05：神经网络训练流程（MLP）。

用两层 MLP（1 -> 32 -> 32 -> 1，Tanh）拟合带噪声的 y = sin(x)，
演示完整训练/验证流程：划分数据、逐轮记录训练/验证损失并可视化。
运行：
    python experiments/01_pytorch_basics/05_nn_training.py
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


class MLP(nn.Module):
    """两层全连接网络：1 -> 32 -> 32 -> 1，隐藏层用 Tanh。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_data(n: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """生成带噪声的 y = sin(x)，x 在 [-pi, pi]。"""
    x = torch.linspace(-3.14, 3.14, n).reshape(-1, 1)
    y = torch.sin(x) + 0.1 * torch.randn_like(x)
    return x, y


def split_data(
    x: torch.Tensor,
    y: torch.Tensor,
    ratio: float = 0.8,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """按比例随机划分为训练集与验证集。"""
    n = x.shape[0]
    n_train = int(n * ratio)
    idx = torch.randperm(n)
    train = (x[idx[:n_train]], y[idx[:n_train]])
    val = (x[idx[n_train:]], y[idx[n_train:]])
    return train, val


def main() -> None:
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, y = make_data()
    (x_tr, y_tr), (x_va, y_va) = split_data(x, y)

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 800
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_tr), y_tr)
        loss.backward()
        optimizer.step()

        # 验证（不计算梯度）
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_va), y_va)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"epoch {epoch:>4} | train loss {loss.item():.5f} "
                f"| val loss {val_loss.item():.5f}"
            )

    print(f"最终：train loss = {train_losses[-1]:.5f}, "
          f"val loss = {val_losses[-1]:.5f}")

    # 训练/验证损失曲线
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("MLP: train vs validation loss")
    plt.savefig(OUTPUT_DIR / "05_nn_training_loss.png", dpi=120)
    plt.close()

    # 拟合曲线
    plt.figure()
    with torch.no_grad():
        x_line = torch.linspace(-3.14, 3.14, 200).reshape(-1, 1)
        y_line = model(x_line)
    plt.scatter(x.numpy(), y.numpy(), s=8, label="data")
    plt.plot(x_line.numpy(), y_line.numpy(), color="red", label="prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("MLP: fit on sin(x)")
    plt.savefig(OUTPUT_DIR / "05_nn_training_fit.png", dpi=120)
    plt.close()

    print(f"图已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
