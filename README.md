# VisionLab: Intelligent Perception and Semantic Communication Laboratory

一个面向 **PyTorch / 深度学习 / 语义通信** 的实践型学习与研究项目。目标是通过
真实运行代码、修改模型、分析实验结果来掌握深度学习基础，并逐步过渡到智能感知
与语义通信方向的研究原型。

> **免责声明**：本项目是作者的个人学习与研究项目，用于积累可复现的实验与工程
> 能力。项目中的学习路线为作者自拟，**不代表导师或任何课题组已指定相关课题，
> 也不代表与导师或课题组存在正式合作**。

## 项目目标

- 从 PyTorch 基础（张量、自动微分、梯度下降）开始，逐步建立完整的实验能力。
- 覆盖 MLP / CNN、图像分类、训练/验证/测试流程、指标评估与可视化。
- 探索图像压缩、噪声干扰与重建，以及**简化**的语义通信实验（明确标注为教学
  性质的简化原型，不作夸大）。
- 强调可复现性：固定随机种子、记录实验配置、使用小规模且 CPU 友好的实验。

## 学习路线

| 阶段 | 主题 | 状态 |
| --- | --- | --- |
| A | 工程初始化 | ✅ 已完成 |
| B | PyTorch 基础实验 | ✅ 已完成 |
| C | 机器学习与图像分类 | 待开始 |
| D | 可靠实验流程 | 待开始 |
| E | 图像通信实验 | 待开始 |
| F | 语义通信原型 | 待开始 |
| G | 后续研究扩展 | 待开始 |

详见 [docs/roadmap.md](docs/roadmap.md)。

## 环境要求

- **优先支持 CPU**，不需要 NVIDIA GPU（项目默认在 CPU 上运行）。
- Python 3.10+。
- Windows / Linux / macOS 均可（本项目在 Windows 11 上开发）。

## 安装

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
# Windows（Git Bash / PowerShell）：
source .venv/Scripts/activate
# Linux / macOS：
# source .venv/bin/activate

# 2. 安装 PyTorch CPU 版
#    务必使用官方 CPU 源，避免误装体积巨大的 CUDA 版：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装项目其余依赖
pip install -r requirements.txt

# 4. 以可编辑模式安装本项目
pip install -e .
```

> 依赖说明：`torch` / `torchvision` 不在 `requirements.txt` 中，因为它们是
> CPU 版，必须走上面的官方 CPU 源；若直接 `pip install torch` 会拉取 CUDA 版。

## 快速开始

```bash
# 冒烟测试：确认包可以导入
python -c "import visionlab; print(visionlab.__version__)"

# 运行自动化测试
pytest
```

## 实验

阶段 B 的 PyTorch 基础实验位于 `experiments/01_pytorch_basics/`，全部可在
CPU 上运行、使用固定随机种子、无需下载数据集。示例：

```bash
# 张量基础
python experiments/01_pytorch_basics/01_tensor_basics.py

# 线性回归
python experiments/01_pytorch_basics/04_linear_regression.py

# 神经网络训练流程（记录并可视化训练/验证损失）
python experiments/01_pytorch_basics/05_nn_training.py
```

图像会保存到 `outputs/01_pytorch_basics/`（已被 `.gitignore` 忽略）。各实验的
输入、输出与运行命令详见 [experiments/01_pytorch_basics/README.md](experiments/01_pytorch_basics/README.md)。

## 项目结构

```
visionlab-semantic-communication/
├── README.md                 # 项目介绍、安装、快速开始
├── LICENSE                   # MIT
├── .gitignore                # 排除 .venv / 数据 / 缓存 / 密钥
├── pyproject.toml            # 包元数据 + src 布局 + pytest 配置
├── requirements.txt          # 运行时依赖（torch 走 CPU 源单独安装）
├── configs/                  # YAML 实验配置（阶段 D 起填充）
├── data/
│   └── README.md             # 数据集获取/存储/隐私说明（不提交数据本体）
├── docs/                     # 学习路线、理论解释、实验记录
├── src/visionlab/            # 可复用代码（核心）
├── experiments/              # 按主题编号的实验入口（阶段 B 起填充）
├── tests/                    # pytest 自动化测试
└── scripts/                  # 一次性工具脚本
```

## 复现性说明

- 实验通过 `src/visionlab/utils/seed.py` 统一固定随机种子（阶段 B 引入）。
- 超参数集中在 `configs/` 下的 YAML 文件中。
- 依赖版本记录于 `requirements.txt`；PyTorch 使用 CPU 版。
- 数据集来源与获取方式见 [data/README.md](data/README.md)，数据本体不提交 Git。

## License

[MIT](LICENSE)
