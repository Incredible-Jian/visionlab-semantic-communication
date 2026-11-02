"""实验 01：图像压缩重建的质量对比（PSNR / MSE）。

对比两种简化压缩方式在不同强度下的重建质量：JPEG 有损压缩（quality 1~100）
与下采样压缩（factor 越大越狠）。

> 说明：这是教学用的简化实验，不代表真实图像通信系统。

运行：
    python experiments/04_image_comm/01_compression_quality.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from visionlab.comm import downsample_codec, jpeg_codec
from visionlab.utils import set_seed
from visionlab.utils.image_metrics import mse, psnr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_image_comm"

SEED = 42
N_IMAGES = 64
JPEG_QUALITIES = [5, 10, 20, 30, 50, 70, 90]
DOWNSAMPLE_FACTORS = [2, 4, 7, 14]


def load_test_images(n: int) -> np.ndarray:
    """加载 FashionMNIST 测试集前 ``n`` 张图，返回 ``(n, 28, 28)`` 浮点数组。"""
    ds = datasets.FashionMNIST(
        root="data", train=False, download=False, transform=transforms.ToTensor()
    )
    return np.stack([ds[i][0][0].numpy() for i in range(n)])


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = load_test_images(N_IMAGES)

    # JPEG 质量扫描
    jpeg_psnr: list[float] = []
    jpeg_mse: list[float] = []
    for q in JPEG_QUALITIES:
        ps, ms = [], []
        for img in images:
            rec = jpeg_codec(img, q)
            ps.append(psnr(img, rec))
            ms.append(mse(img, rec))
        jpeg_psnr.append(float(np.mean(ps)))
        jpeg_mse.append(float(np.mean(ms)))
        print(f"JPEG quality={q:>3} -> PSNR={jpeg_psnr[-1]:.2f} dB, "
              f"MSE={jpeg_mse[-1]:.5f}")

    # 下采样扫描
    ds_psnr: list[float] = []
    ds_mse: list[float] = []
    for f in DOWNSAMPLE_FACTORS:
        ps, ms = [], []
        for img in images:
            rec = downsample_codec(img, f)
            ps.append(psnr(img, rec))
            ms.append(mse(img, rec))
        ds_psnr.append(float(np.mean(ps)))
        ds_mse.append(float(np.mean(ms)))
        print(f"downsample factor={f:>2} -> PSNR={ds_psnr[-1]:.2f} dB, "
              f"MSE={ds_mse[-1]:.5f}")

    # 绘图：PSNR 随压缩强度的变化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(JPEG_QUALITIES, jpeg_psnr, marker="o")
    ax1.set_xlabel("JPEG quality")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("JPEG compression")
    ax1.grid(True, alpha=0.3)

    ax2.plot(DOWNSAMPLE_FACTORS, ds_psnr, marker="s")
    ax2.set_xlabel("downsample factor")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Downsample compression")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_compression_quality.png", dpi=120)
    plt.close(fig)

    print(f"\n图已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
