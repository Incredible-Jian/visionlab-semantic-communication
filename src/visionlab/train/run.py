"""配置驱动的可复用实验运行器（阶段 D）。

``run_experiment`` 按一份 ``Config`` 完整执行「数据 → 模型 → 训练 → 测试 →
可视化 → 保存最佳模型 → 记录结果」，供 MLP / CNN 等分类实验复用。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from visionlab.config import Config, to_dict, write_yaml
from visionlab.data import get_dataloaders
from visionlab.models import CNN, MLP
from visionlab.train.trainer import evaluate, save_checkpoint, train
from visionlab.utils import confusion_matrix, set_seed
from visionlab.visualization import plot_confusion_matrix, plot_history


def build_model(model_config) -> nn.Module:
    """按 ``model_config`` 构造模型。"""
    if model_config.name == "mlp":
        return MLP(
            input_dim=28 * 28,
            hidden_dims=model_config.hidden_dims,
            num_classes=model_config.num_classes,
            dropout=model_config.dropout,
        )
    if model_config.name == "cnn":
        return CNN(
            in_channels=model_config.in_channels,
            num_classes=model_config.num_classes,
        )
    raise ValueError(f"未知模型名：{model_config.name!r}（可选 'mlp' 或 'cnn'）")


def run_experiment(config: Config, output_dir: str | Path) -> dict:
    """按配置完整运行一次实验，产出图、checkpoint 与结果文件，返回结果 dict。

    产出文件（写入 ``output_dir``）：``<name>_history.png``、
    ``<name>_confusion_matrix.png``、``<name>_best.pt``、``results.yaml``。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    set_seed(config.data.seed)

    # 1. 数据
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        root=config.data.root,
        batch_size=config.data.batch_size,
        val_ratio=config.data.val_ratio,
        seed=config.data.seed,
        num_workers=config.data.num_workers,
    )

    # 2. 模型 / 损失 / 优化器
    model = build_model(config.model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    num_params = sum(p.numel() for p in model.parameters())

    # 3. 训练
    history = train(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=config.train.epochs, device=device,
        num_classes=config.model.num_classes,
    )

    # 4. 用最佳验证模型在测试集评估
    model.load_state_dict(history["best_state"])
    test = evaluate(model, test_loader, criterion, device, config.model.num_classes)

    # 5. 可视化
    name = config.model.name
    plot_history(history, output_dir, name=name)
    cm = confusion_matrix(test["y_true"], test["y_pred"], config.model.num_classes)
    plot_confusion_matrix(cm, class_names, output_dir, name=f"{name}_confusion_matrix")

    # 6. 保存最佳 checkpoint
    ckpt_path = output_dir / f"{name}_best.pt"
    save_checkpoint(
        {
            "model_state_dict": history["best_state"],
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": config.train.epochs,
            "best_val_acc": history["best_val_acc"],
            "test_acc": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "config": to_dict(config),
        },
        ckpt_path,
    )

    # 7. 结果记录（配置 + 真实指标 + 时间戳）
    results = {
        "config": to_dict(config),
        "metrics": {
            "best_val_acc": history["best_val_acc"],
            "test_loss": test["loss"],
            "test_acc": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "elapsed_seconds": history["elapsed"],
            "num_params": num_params,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_yaml(results, output_dir / "results.yaml")

    return results
