"""实验 02：高斯噪声信道对图像质量的影响（PSNR / MSE）。

在"传输"环节加入均值为 0、标准差为 ``sigma`` 的加性高斯噪声，观察重建质量
随噪声强度的变化，并展示同一张图在不同噪声下的视觉效果。

> 说明：这是教学用的简化实验，不代表真实图像通信系统。

运行：
    python experiments/04_image_comm/02_noise_channel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from visionlab.comm import add_gaussian_noise
from visionlab.utils import set_seed
from visionlab.utils.image_metrics import mse, psnr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_image_comm"

SEED = 42
N_IMAGES = 64
SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


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

    # 噪声强度扫描
    psnr_list: list[float] = []
    mse_list: list[float] = []
    for sigma in SIGMAS:
        ps, ms = [], []
        for img in images:
            noisy = add_gaussian_noise(img, sigma)
            ps.append(psnr(img, noisy))
            ms.append(mse(img, noisy))
        psnr_list.append(float(np.mean(ps)))
        mse_list.append(float(np.mean(ms)))
        print(f"sigma={sigma:.2f} -> PSNR={psnr_list[-1]:.2f} dB, "
              f"MSE={mse_list[-1]:.5f}")

    # PSNR 随噪声强度的变化
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SIGMAS, psnr_list, marker="o")
    ax.set_xlabel("noise sigma")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Gaussian noise channel")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_noise_channel.png", dpi=120)
    plt.close(fig)

    # 展示同一张图在不同噪声下的视觉对比
    demo = images[0]
    demo_sigmas = [0.0, 0.05, 0.1, 0.2, 0.5]
    fig, axes = plt.subplots(1, len(demo_sigmas), figsize=(12, 3))
    for ax, sigma in zip(axes, demo_sigmas):
        ax.imshow(add_gaussian_noise(demo, sigma), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"sigma={sigma}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_noise_channel_demo.png", dpi=120)
    plt.close(fig)

    print(f"\n图已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
