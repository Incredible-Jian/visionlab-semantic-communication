# 学习路线（Roadmap）

本项目按阶段推进，每个阶段都有可运行的代码、测试与文档。**每个阶段完成后
才进入下一阶段**，不提前堆砌未经验证的模块。

## 阶段 A：工程初始化（已完成）

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

## 阶段 C：机器学习与图像分类（已完成）

主题：数据集与 DataLoader、训练/验证/测试集划分、MLP 图像分类、CNN 图像
分类、指标评估与可视化、模型保存与加载。

已完成：

- [x] `src/visionlab/data/fashion_mnist.py`：FashionMNIST 下载（torchvision
      默认源 + ModelScope 阿里云镜像回退、MD5 校验）、训练/验证划分、DataLoader
- [x] `src/visionlab/utils/metrics.py`：numpy 自实现准确率、混淆矩阵、宏 F1
- [x] `src/visionlab/models/`：`MLP`、`CNN`
- [x] `src/visionlab/train/trainer.py`：`train_one_epoch` / `evaluate` / `train` /
      `save_checkpoint` / `load_checkpoint`
- [x] `src/visionlab/visualization/plots.py`：损失曲线与混淆矩阵绘图
- [x] `experiments/02_image_classification/`：MLP、CNN 两个可运行实验，真实记录
      损失/准确率/耗时，保存最佳 checkpoint 并验证加载后推理一致
- [x] 配套 pytest 测试（阶段累计 23 项全部通过）

留待后续：

- [ ] CIFAR-10 进阶数据集（约 170 MB，需另行确认下载方式后再进行）

## 阶段 D：可靠实验流程（已完成）

主题：配置管理（dataclass + YAML）、训练/评估代码复用、超参数记录、
训练曲线、最佳验证模型保存、自动化测试。

已完成：

- [x] `src/visionlab/config/config.py`：`Config` / `DataConfig` / `ModelConfig` /
      `TrainConfig` dataclass，YAML 读写与未知配置项校验
- [x] `configs/`：`mlp_fashionmnist.yaml`、`cnn_fashionmnist.yaml` 示例配置
- [x] `src/visionlab/train/run.py`：`run_experiment` 按配置串联完整流程
      （数据 → 模型 → 训练 → 测试 → 画图 → 保存最佳模型 → 写结果）
- [x] `experiments/03_reliable_training/`：`run_experiment.py` 命令行入口
      （`--config configs/*.yaml`），结果写入 `outputs/03_reliable_training/<模型名>/`
- [x] `docs/experiment_record.md`：实验记录模板（超参数 + 真实结果）
- [x] 配套 pytest 测试（阶段累计 30 项全部通过）

可复现性验证：阶段 D 运行器与阶段 C 脚本使用相同超参数与随机种子时，MLP / CNN
的验证与测试指标逐位一致（MLP 测试 acc 0.8777，CNN 测试 acc 0.8980）。

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
