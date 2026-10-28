"""FashionMNIST 数据集的下载、划分与 DataLoader 构建。

数据集：28×28 灰度图，10 类，训练 60,000 / 测试 10,000，约 30 MB。
数据下载到 ``root``（本项目用 ``data/``，已被 ``.gitignore`` 忽略）。

下载策略：先尝试 torchvision 的默认源；若失败（例如 torchvision 0.20.0 自带的
S3 源已失效、或国内网络无法直连境外源），则回退到镜像手动下载（ModelScope
阿里云优先，带 User-Agent 与 MD5 校验，国内外均可访问）。
"""

from __future__ import annotations

import gzip
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 可用的下载镜像（URL 前缀，后面拼文件名），按优先级排列。MD5 与 torchvision 官方一致。
_FASHION_MNIST_MIRRORS = [
    # ModelScope（阿里云魔搭）OpenDataLab 镜像：国内直连可达，MD5 与官方一致。
    "https://modelscope.cn/api/v1/datasets/OpenDataLab/Fashion-MNIST/repo?Revision=master&FilePath=raw/",
    # GitHub 官方仓库原始文件（境外网络可用）。
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
]

# (下载文件名, 压缩文件的 MD5)，取自 torchvision 的官方 resources 定义。
_FASHION_MNIST_RESOURCES = [
    ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
]


def split_indices(
    n: int,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把 ``0..n-1`` 随机打乱后按比例分成训练/验证索引。

    返回 ``(train_idx, val_idx)``；使用独立于全局状态的 ``default_rng(seed)``，
    保证同一 ``seed`` 下结果可复现。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    return idx[n_val:], idx[:n_val]


def _download_bytes(filename: str, md5: str) -> bytes:
    """依次尝试各镜像下载单个文件，校验 MD5 后返回字节流。"""
    last_error: Exception | None = None
    for mirror in _FASHION_MNIST_MIRRORS:
        url = mirror + filename
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if hashlib.md5(data).hexdigest() != md5:
                raise RuntimeError(f"MD5 校验失败: {url}")
            return data
        except Exception as exc:  # noqa: BLE001 - 逐个镜像尝试，失败继续下一个
            last_error = exc
    raise RuntimeError(f"所有镜像均下载失败: {filename}") from last_error


def _ensure_downloaded(root: str | Path) -> None:
    """确保 FashionMNIST 的 4 个原始文件已解压到 ``raw/``，否则从镜像下载。"""
    raw_folder = Path(root) / "FashionMNIST" / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    for filename, md5 in _FASHION_MNIST_RESOURCES:
        out_path = raw_folder / filename[:-3]  # 去掉 ".gz" 后缀
        if out_path.exists():
            continue
        data = _download_bytes(filename, md5)
        with gzip.open(io.BytesIO(data), "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())


def get_dataloaders(
    root: str | Path = "data",
    batch_size: int = 128,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """下载 FashionMNIST 并返回 ``(train, val, test, class_names)``。

    训练集 60,000 按 ``val_ratio`` 划分为训练/验证；测试集使用官方 10,000。
    ``num_workers`` 默认 0（Windows 下多进程 DataLoader 易出问题，CPU 也够用）。
    """
    transform = transforms.ToTensor()
    try:
        train_full = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform
        )
        test_ds = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transform
        )
    except RuntimeError:
        # 默认源不可用（如国内网络），回退到镜像手动下载后再加载。
        _ensure_downloaded(root)
        train_full = datasets.FashionMNIST(
            root=root, train=True, download=False, transform=transform
        )
        test_ds = datasets.FashionMNIST(
            root=root, train=False, download=False, transform=transform
        )

    train_idx, val_idx = split_indices(len(train_full), val_ratio, seed)
    train_ds = Subset(train_full, train_idx.tolist())
    val_ds = Subset(train_full, val_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_full.classes
