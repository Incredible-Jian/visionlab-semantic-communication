"""训练曲线与混淆矩阵可视化（matplotlib，Agg 后端，适合无显示环境）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(
    history: dict,
    output_dir: str | Path,
    name: str = "training",
) -> None:
    """绘制损失与验证准确率曲线，保存为 ``<output_dir>/<name>_history.png``。

    ``history`` 需含键 ``train_loss`` / ``val_loss`` / ``val_acc``（逐轮列表）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], label="val acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Validation accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_history.png", dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: str | Path,
    name: str = "confusion_matrix",
    normalize: bool = True,
) -> None:
    """绘制混淆矩阵热力图，保存为 ``<output_dir>/<name>.png``。

    ``cm[i, j]`` 表示真实类别 i 被预测为 j 的样本数；``normalize=True`` 时按行
    归一化为概率（每行和为 1），并在每个格子标注数值。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=120)
    plt.close(fig)
