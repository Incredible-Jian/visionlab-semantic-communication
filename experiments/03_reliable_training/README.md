# 阶段 D：可靠实验流程

把阶段 C 里「手写超参数 + 两个重复脚本」的方式，升级为「YAML 配置驱动 +
可复用运行器 + 结果记录」，保证每次实验的配置、结果、耗时都真实可追溯、可复现。

## 运行

```bash
# MLP 实验（配置见 configs/mlp_fashionmnist.yaml）
python experiments/03_reliable_training/run_experiment.py --config configs/mlp_fashionmnist.yaml

# CNN 实验（配置见 configs/cnn_fashionmnist.yaml）
python experiments/03_reliable_training/run_experiment.py --config configs/cnn_fashionmnist.yaml
```

## 工作原理

1. `--config` 指向一个 YAML 文件，由 `src/visionlab/config/config.py` 的
   `load_config` 读入并校验（未知配置项会报错）。
2. `src/visionlab/train/run.py` 的 `run_experiment(config, output_dir)` 按配置
   串联完整流程：数据 → 模型 → 训练 → 测试 → 画图 → 保存最佳模型 → 写结果。
3. 输出目录为 `outputs/03_reliable_training/<模型名>/`，包含：
   - `<模型名>_history.png`：损失与验证准确率曲线
   - `<模型名>_confusion_matrix.png`：测试集混淆矩阵
   - `<模型名>_best.pt`：最佳验证模型 checkpoint（含配置）
   - `results.yaml`：完整配置 + 真实指标 + 时间戳

## 与阶段 C 的关系

阶段 C 的 `experiments/02_image_classification/` 用两个脚本分别写死了 MLP 与
CNN 的超参数；阶段 D 把两者统一成一个入口，超参数外置到 `configs/`，训练/评估
逻辑复用 `train/` 与 `models/`。二者结果应一致（同一超参数、同一随机种子）。

## 结果记录

每次运行的结果填写到 `docs/experiment_record.md` 模板中，禁止虚构数字；所有
指标必须来自真实运行，与 `results.yaml` 一一对应。
