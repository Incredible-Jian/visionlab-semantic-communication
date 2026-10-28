"""测试 visionlab.data 的训练/验证划分逻辑。

说明：``get_dataloaders`` 需要联网下载 FashionMNIST，不在单元测试中调用，
其正确性由 ``experiments/02_image_classification/`` 的实验脚本实际运行验证。
这里只测试无需数据的纯函数 ``split_indices``。
"""

import numpy as np

from visionlab.data import split_indices


def test_split_sizes() -> None:
    train, val = split_indices(100, val_ratio=0.2, seed=42)
    assert len(val) == 20
    assert len(train) == 80


def test_split_disjoint_and_cover_all() -> None:
    train, val = split_indices(50, val_ratio=0.3, seed=0)
    assert len(np.intersect1d(train, val)) == 0
    assert len(np.union1d(train, val)) == 50


def test_split_is_reproducible() -> None:
    train_a, val_a = split_indices(100, 0.2, seed=7)
    train_b, val_b = split_indices(100, 0.2, seed=7)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)


def test_split_zero_val_ratio() -> None:
    train, val = split_indices(100, val_ratio=0.0, seed=0)
    assert len(val) == 0
    assert len(train) == 100
