# 实验 01：PyTorch 基础

本目录包含阶段 B 的 5 个基础实验，全部在 CPU 上运行，使用固定随机种子
（`set_seed(42)`）保证结果可复现，且只使用合成数据（**不下载任何数据集**）。

运行前请确保已激活虚拟环境，并在**项目根目录**执行命令。

## 实验清单

| 脚本 | 内容 | 运行命令 |
| --- | --- | --- |
| `01_tensor_basics.py` | 张量创建、形状、索引、变形、广播、矩阵运算 | `python experiments/01_pytorch_basics/01_tensor_basics.py` |
| `02_autograd.py` | Autograd、`backward()`、梯度取值与累积 | `python experiments/01_pytorch_basics/02_autograd.py` |
| `03_gradient_descent.py` | 手动梯度下降 vs `torch.optim.SGD` | `python experiments/01_pytorch_basics/03_gradient_descent.py` |
| `04_linear_regression.py` | `nn.Linear` + MSE + SGD 拟合直线 | `python experiments/01_pytorch_basics/04_linear_regression.py` |
| `05_nn_training.py` | MLP 训练流程（训练/验证损失记录与可视化） | `python experiments/01_pytorch_basics/05_nn_training.py` |

## 输出说明

- 实验 01、02 仅在终端打印结果，不产生文件。
- 实验 03、04、05 会在 `outputs/01_pytorch_basics/` 下保存 PNG 图像
  （损失曲线、拟合曲线等）。`outputs/` 已被 `.gitignore` 忽略，不会提交到 Git，
  删除后可随时通过重新运行脚本复现。

## 复现性说明

- 所有实验调用 `visionlab.utils.seed.set_seed(42)`，且数据生成顺序固定，
  因此在相同 PyTorch 版本（本项目为 2.5.0+cpu）下结果可复现。
- 实验不依赖 GPU，也不依赖任何外部数据集。
