"""测试 visionlab.utils.metrics 的准确率 / 混淆矩阵 / 宏 F1。"""

import numpy as np
import pytest

from visionlab.utils.metrics import accuracy, confusion_matrix, macro_f1


def test_accuracy_perfect() -> None:
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_partial() -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 9, 3])
    assert accuracy(y_true, y_pred) == 0.75


def test_accuracy_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([0, 1, 2]))


def test_confusion_matrix() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1])
    cm = confusion_matrix(y_true, y_pred, num_classes=2)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 1  # 真实 0 预测 0
    assert cm[0, 1] == 1  # 真实 0 预测 1
    assert cm[1, 0] == 0
    assert cm[1, 1] == 3  # 真实 1 预测 1


def test_macro_f1_perfect() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y_true, y_pred, 3) == 1.0


def test_macro_f1_known_value() -> None:
    # 二分类：类 0 的 tp=2, fp=1, fn=1 -> precision=recall=2/3 -> F1=2/3；
    # 类 1 的 tp=0 -> F1=0；宏平均 = (2/3 + 0) / 2 = 1/3。
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 0])
    assert abs(macro_f1(y_true, y_pred, 2) - 1 / 3) < 1e-9
