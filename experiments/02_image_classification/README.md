# 阶段 C：机器学习与图像分类

使用 FashionMNIST（28×28 灰度图，10 类）演示完整的图像分类流程：数据加载与
划分、MLP / CNN 模型、训练/验证/测试、指标评估与可视化、模型保存与加载。

## 目录

| 文件 | 内容 |
| --- | --- |
| `01_mlp_classification.py` | 多层感知机（MLP）图像分类 |
| `02_cnn_classification.py` | 卷积神经网络（CNN）图像分类 |

## 运行

```bash
# 在项目根目录下，用已安装 visionlab 的环境运行
python experiments/02_image_classification/01_mlp_classification.py
python experiments/02_image_classification/02_cnn_classification.py
```

首次运行会下载 FashionMNIST（约 30 MB）到 `data/FashionMNIST/`。下载优先走
torchvision 默认源，失败时自动回退到 ModelScope 阿里云镜像（见
[data/README.md](../../data/README.md)）。

## 流程说明

每个脚本都执行相同的标准流程，只替换模型：

1. **数据**：`get_dataloaders` 下载并划分训练/验证/测试集（60k 训练按 20% 分出
   验证集，官方 10k 测试集保持不变），固定随机种子保证划分可复现。
2. **模型**：定义模型、`nn.CrossEntropyLoss` 与 Adam 优化器。
3. **训练**：`train` 逐轮训练并在验证集评估，记录训练/验证损失与验证准确率、
   宏 F1，跟踪最佳验证准确率对应的模型参数。
4. **测试**：用最佳验证模型在从未参与训练的测试集上评估（避免数据泄漏）。
5. **可视化**：保存损失/准确率曲线与混淆矩阵到 `outputs/02_image_classification/`。
6. **保存**：把最佳模型参数、优化器状态、指标与超参数写入 `.pt` checkpoint。
7. **验证**：从磁盘加载 checkpoint 到全新模型，重新评估确认推理结果一致。

## 关键概念

- **训练/验证/测试划分**：训练集更新参数，验证集用于选择最佳模型与早停，
  测试集只在最终评估时使用一次，避免信息泄漏。
- **交叉熵损失**：多分类任务的标准损失，输入为模型输出的 logits，内部会做
  softmax。
- **准确率 vs 宏 F1**：准确率是整体正确率；宏 F1 先对每个类别算 F1 再平均，
  对类别不平衡更敏感，二者结合更全面地反映分类质量。
- **混淆矩阵**：行是真实类别、列是预测类别，可看出模型容易混淆哪些类别（如
  T-shirt/top 与 Shirt）。

## 结果文件

运行后会在 `outputs/02_image_classification/` 生成：

- `mlp_history.png` / `cnn_history.png`：损失与验证准确率曲线
- `mlp_confusion_matrix.png` / `cnn_confusion_matrix.png`：测试集混淆矩阵
- `mlp_best.pt` / `cnn_best.pt`：最佳模型 checkpoint

`outputs/` 已被 `.gitignore` 忽略，不进入版本控制。
