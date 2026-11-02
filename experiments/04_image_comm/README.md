# 阶段 E：图像通信实验（简化）

用 FashionMNIST 灰度图演示一个**简化**的图像通信链路：压缩、噪声信道与重建，
并用 PSNR / MSE 衡量不同条件下的重建质量。

> **重要说明**：本阶段是教学用的简化实验，把"通信"抽象为「压缩 + 加噪 + 重建」，
> 不涉及真实信源/信道编码、调制等，**不代表真实通信系统**。

## 目录

| 文件 | 内容 |
| --- | --- |
| `01_compression_quality.py` | 压缩强度对重建质量的影响（JPEG / 下采样） |
| `02_noise_channel.py` | 高斯噪声强度对重建质量的影响 |

## 运行

```bash
python experiments/04_image_comm/01_compression_quality.py
python experiments/04_image_comm/02_noise_channel.py
```

复用 FashionMNIST 已下载的数据（`data/FashionMNIST/`），无需额外下载。图保存到
`outputs/04_image_comm/`（不提交 Git）。

## 指标解释

- **MSE（均方误差）**：``mean((原图 - 重建图)^2)``，越小越好，0 表示完全一致。
- **PSNR（峰值信噪比）**：``10 * log10(max_val^2 / MSE)``（dB），越大越好；
  完全一致时为 ``inf``。它把 MSE 映射到对数尺度，更直观地反映小幅失真的差异。

两者的关系：PSNR 本质是 MSE 的对数表示，PSNR 越高说明重建越接近原图。

## 实验内容

1. **压缩**：`comm/codec.py` 提供两种简化编解码——JPEG 有损压缩（Pillow，
   不同 quality）与下采样重建（双线性插值，不同 factor）。实验 01 扫描不同
   压缩强度，画出 PSNR 曲线。
2. **噪声**：`comm/channel.py` 提供加性高斯噪声（均值 0、标准差 sigma，裁剪到
   [0, 1]）。实验 02 扫描不同噪声强度，画 PSNR 曲线并展示示例重建图。

## 复现性

实验固定随机种子（`set_seed(42)`），噪声与结果可复现。指标用 numpy 自实现
（`src/visionlab/utils/image_metrics.py`），不依赖 scipy/skimage。
