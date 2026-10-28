"""分类评估指标（numpy 自实现，不依赖 scikit-learn）。

约定：``y_true`` / ``y_pred`` 为长度相同的整型类别标签数组。
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """准确率 = 预测正确样本数 / 总样本数。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true 与 y_pred 形状不一致: {y_true.shape} vs {y_pred.shape}"
        )
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """返回形状为 ``(num_classes, num_classes)`` 的混淆矩阵。

    行表示真实类别，列表示预测类别，``cm[i, j]`` 是真实为 i 且预测为 j 的样本数。
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> float:
    """宏平均 F1：对每个类别分别计算 F1 后取平均（忽略样本数为 0 的类）。"""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1s.append(f1)
    return float(np.mean(f1s))
