# data/ 目录说明

本目录用于存放项目使用的数据集，但 **数据集本体不提交到 Git**（已被
`.gitignore` 排除，仅保留本 README）。

## 数据集获取方式

本项目优先使用规模小、CPU 友好的公开数据集。数据由 `src/visionlab/data/`
中的 `get_dataloaders(root="data")` 按需下载到本目录下的 `data/FashionMNIST/`
（已被 `.gitignore` 忽略，不进入版本控制）。

下载策略：先尝试 torchvision 的默认源；若失败（例如 torchvision 0.20.0 自带
的 S3 源已失效、或国内网络无法直连境外源），自动回退到 ModelScope 阿里云镜像
（`OpenDataLab/Fashion-MNIST`），并对每个文件校验 MD5，与 torchvision 官方值
一致，确保数据完整。

| 数据集 | 用途 | 规模 | 许可/来源 |
| --- | --- | --- | --- |
| FashionMNIST | 主数据集：MLP / CNN 图像分类（阶段 C） | 28×28 灰度，60k 训练 / 10k 测试，约 30 MB | [GitHub](https://github.com/zalandoresearch/fashion-mnist)（MIT） |
| CIFAR-10 | 进阶数据集：彩色图像分类（待确认后使用） | 32×32×3，50k 训练 / 10k 测试，约 170 MB | [官方](https://www.cs.toronto.edu/~kriz/cifar.html) |

> 阶段 E / F（图像通信、语义通信原型）所需图像会从上述数据集的测试子集中
> 选取少量样本，避免额外下载大型数据。

## 存储约定

- 数据集一律放在本目录（`data/`）下的子目录，**已被 `.gitignore` 忽略，不会
  进入版本控制**。
- 请不要手动删除 `data/FashionMNIST/` 下的原始文件；若误删，下次运行实验会
  自动重新下载。
- 如需更换数据集存储位置，可给 `get_dataloaders(root=...)` 传入其他路径。

## 隐私与合规

- 本项目只使用公开学术数据集，不含个人信息。
- 若未来引入任何含隐私信息的数据，必须先脱敏、取得授权，并在提交前二次
  确认不会被 `git` 追踪。
