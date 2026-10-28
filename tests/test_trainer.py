"""测试 visionlab.train.trainer 的评估 / 训练 / checkpoint 流程。"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from visionlab.train.trainer import (
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def _make_separable_loader(n: int = 64, d: int = 4) -> DataLoader:
    """构造线性可分的小数据集：标签由第 0 维特征的符号决定。"""
    torch.manual_seed(0)
    x = torch.randn(n, d)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_evaluate_returns_expected_keys() -> None:
    model = nn.Linear(4, 2)
    loader = _make_separable_loader()
    out = evaluate(model, loader, nn.CrossEntropyLoss(), "cpu", num_classes=2)
    assert {"loss", "accuracy", "macro_f1", "y_true", "y_pred", "num_samples"} <= set(out)
    assert out["num_samples"] == 64
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["y_true"].shape == (64,)
    assert out["y_pred"].shape == (64,)


def test_train_one_epoch_reduces_loss() -> None:
    model = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = _make_separable_loader()

    before = evaluate(model, loader, criterion, "cpu", 2)["loss"]
    for _ in range(30):
        train_one_epoch(model, loader, criterion, optimizer, "cpu")
    after = evaluate(model, loader, criterion, "cpu", 2)["loss"]
    assert after < before


def test_save_and_load_checkpoint_roundtrip(tmp_path) -> None:
    model = nn.Linear(4, 2)
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    expected = model(x)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        {"model_state_dict": model.state_dict(), "epoch": 5, "val_acc": 0.9},
        path,
    )

    restored = nn.Linear(4, 2)
    state = load_checkpoint(path, model=restored)
    assert state["epoch"] == 5
    assert state["val_acc"] == 0.9
    assert torch.allclose(restored(x), expected)
