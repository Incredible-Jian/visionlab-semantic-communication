"""实验 02：FashionMNIST 上的 CNN 图像分类。

流程与实验 01 相同，仅把模型换成两层卷积 + 两层全连接的 CNN。

运行：
    python experiments/02_image_classification/02_cnn_classification.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from visionlab.data import get_dataloaders
from visionlab.models import CNN
from visionlab.train import evaluate, load_checkpoint, save_checkpoint, train
from visionlab.utils import confusion_matrix, set_seed
from visionlab.visualization import plot_confusion_matrix, plot_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "02_image_classification"

# 可复现配置（也写入 checkpoint）
SEED = 42
BATCH_SIZE = 128
VAL_RATIO = 0.2
EPOCHS = 10
LR = 1e-3


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    # 1. 数据：训练/验证/测试 + 类别名
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        root="data", batch_size=BATCH_SIZE, val_ratio=VAL_RATIO, seed=SEED
    )
    print(f"类别（{len(class_names)}）：{class_names}")

    # 2. 模型与优化器
    model = CNN(in_channels=1, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"CNN 参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 3. 训练（逐轮在验证集评估，跟踪最佳验证准确率）
    history = train(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=EPOCHS, device=device, num_classes=10,
    )
    print(f"最佳验证准确率：{history['best_val_acc']:.4f}，"
          f"训练耗时：{history['elapsed']:.1f}s")

    # 4. 用最佳验证模型在测试集评估
    model.load_state_dict(history["best_state"])
    test = evaluate(model, test_loader, criterion, device, num_classes=10)
    print(f"测试集：loss={test['loss']:.4f} acc={test['accuracy']:.4f} "
          f"macro_f1={test['macro_f1']:.4f}")

    # 5. 可视化：损失/准确率曲线 + 混淆矩阵
    plot_history(history, OUTPUT_DIR, name="cnn")
    cm = confusion_matrix(test["y_true"], test["y_pred"], num_classes=10)
    plot_confusion_matrix(cm, class_names, OUTPUT_DIR, name="cnn_confusion_matrix")

    # 6. 保存最佳 checkpoint（模型参数 + 优化器 + 指标 + 配置）
    ckpt_path = OUTPUT_DIR / "cnn_best.pt"
    save_checkpoint(
        {
            "model_state_dict": history["best_state"],
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": EPOCHS,
            "best_val_acc": history["best_val_acc"],
            "test_acc": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "config": {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "val_ratio": VAL_RATIO,
                "epochs": EPOCHS,
                "lr": LR,
                "model": "CNN",
                "in_channels": 1,
            },
        },
        ckpt_path,
    )

    # 7. 验证：从磁盘加载到全新模型，推理结果应与保存前一致
    restored = CNN(in_channels=1, num_classes=10)
    load_checkpoint(ckpt_path, model=restored, device=device)
    restored_test = evaluate(restored, test_loader, criterion, device, num_classes=10)
    if abs(restored_test["accuracy"] - test["accuracy"]) > 1e-9:
        raise RuntimeError("checkpoint 加载后推理结果与保存前不一致！")
    print(f"重新加载 checkpoint 后测试准确率一致：{restored_test['accuracy']:.4f}")

    # 8. 展示少量样本的真实/预测标签
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        pred = model(x).argmax(dim=1)
    print("样本预测示例：")
    for i in range(5):
        print(f"  真实={class_names[y[i].item()]:<12s} 预测={class_names[pred[i].item()]}")

    print(f"\n结果已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
