# 学习路线（Roadmap）

本项目按阶段推进，每个阶段都有可运行的代码、测试与文档。**每个阶段完成后
才进入下一阶段**，不提前堆砌未经验证的模块。

## 阶段 A：工程初始化（当前阶段）

目标：建立可复用、可测试、适合公开发布的项目骨架。

验收标准：

- [x] 初始化 Git 仓库
- [x] 建立目录结构与 `.gitignore`（排除 `.venv`、数据、缓存、密钥）
- [x] 配置 `pyproject.toml`、`requirements.txt`、MIT License
- [x] 创建独立虚拟环境 `.venv` 并安装 CPU 版 PyTorch
- [x] `visionlab` 包可导入，冒烟测试通过
- [x] 完成首次 Git 提交

## 阶段 B：PyTorch 基础实验（已完成）

主题：张量操作、Autograd 梯度计算、梯度下降、`nn.Module`、MSE 损失、
反向传播与优化器、训练/验证损失记录与可视化。

已完成：

- [x] `src/visionlab/utils/seed.py`：统一固定随机种子（`set_seed`）
- [x] `src/visionlab/utils/device.py`：CPU 设备选择（`get_device`）
- [x] `experiments/01_pytorch_basics/`：5 个可运行实验
  （张量 / Autograd / 梯度下降对比 / 线性回归 / MLP 训练流程）
- [x] 训练与验证损失记录 + matplotlib 可视化（保存到 `outputs/`）
- [x] 配套 pytest 测试（6 项全部通过）与文档

## 阶段 C：机器学习与图像分类

主题：数据集与 DataLoader、训练/验证/测试集划分、MLP 图像分类、CNN 图像
分类、指标评估与可视化、模型保存与加载。

- 主数据集：FashionMNIST（CPU 友好）
- 进阶数据集：CIFAR-10
- `src/visionlab/data/`、`src/visionlab/models/`、`src/visionlab/train/`、
  `src/visionlab/visualization/`、`src/visionlab/utils/metrics.py`
- 用 numpy 自行实现准确率、混淆矩阵、精确率/召回率/F1

## 阶段 D：可靠实验流程

主题：配置管理（dataclass + YAML）、训练/评估代码复用、超参数记录、
训练曲线、最佳验证模型保存、自动化测试。

- `src/visionlab/config/`
- `configs/` 下的 YAML 配置
- 实验记录模板（超参数 + 真实结果）

## 阶段 E：图像通信实验

主题：图像/特征压缩、可控噪声干扰、图像重建、不同条件下重建质量对比，
解释 PSNR、MSE 等指标。

- `experiments/04_image_comm/`
- 明确标注这是**简化实验**，不代表真实通信系统

## 阶段 F：语义通信原型

主题：先阅读公开资料，设计最小可运行实验；明确任务、模型、输入、输出与
评估指标；对比传统图像传输基线与学习型方法；所有假设如实标注，不虚构创新点。

## 阶段 G：后续研究扩展

主题：根据实际学习进展评估 Transformer、多模态与相关论文复现。只选择能够
解释、复现、验证的问题，不为堆技术名词而加功能。
