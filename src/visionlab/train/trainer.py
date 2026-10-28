"""训练与评估的通用流程（阶段 C 起复用）。

提供单轮训练 ``train_one_epoch``、评估 ``evaluate``、完整训练循环 ``train``，
以及 checkpoint 的保存 / 加载 ``save_checkpoint`` / ``load_checkpoint``。
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from visionlab.utils.metrics import accuracy, macro_f1


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
    num_classes: int = 10,
) -> dict:
    """在 ``loader`` 上评估模型，返回损失与指标。

    返回字典包含 ``loss``（平均）、``accuracy``、``macro_f1``、``num_samples``，
    以及用于绘制混淆矩阵的 ``y_true`` / ``y_pred``（numpy 数组）。
    """
    model.eval()
    total_loss = 0.0
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.shape[0]
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "y_true": y_true,
        "y_pred": y_pred,
        "num_samples": len(y_true),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> float:
    """训练一轮，返回该轮平均训练损失。"""
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
        n += x.shape[0]
    return total_loss / n


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str = "cpu",
    num_classes: int = 10,
    verbose: bool = True,
) -> dict:
    """完整训练循环：逐轮训练并在验证集评估，跟踪最佳验证准确率对应的参数。

    返回 ``history`` 字典：
      ``train_loss`` / ``val_loss`` / ``val_acc`` / ``val_f1``：逐轮列表；
      ``best_val_acc``：最佳验证准确率；
      ``best_state``：最佳验证准确率对应的模型参数（深拷贝，可直接 load_state_dict）；
      ``elapsed``：总训练耗时（秒）。
    """
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }
    best_val_acc = -1.0
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device, num_classes)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["macro_f1"])
        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(
                f"epoch {epoch:>3}/{epochs} | train_loss {train_loss:.4f} | "
                f"val_loss {val['loss']:.4f} | val_acc {val['accuracy']:.4f} | "
                f"val_f1 {val['macro_f1']:.4f}"
            )
    history["best_val_acc"] = best_val_acc
    history["best_state"] = best_state
    history["elapsed"] = time.perf_counter() - start
    return history


def save_checkpoint(state: dict, path: str | Path) -> None:
    """把 ``state`` 字典保存为 checkpoint 文件（自动创建父目录）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """加载 checkpoint，可选恢复 ``model`` / ``optimizer`` 的状态。

    注意：``torch.load`` 会反序列化 pickle，只应加载自己生成的 checkpoint，
    不要加载来源不明的文件。
    """
    state = torch.load(Path(path), map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
