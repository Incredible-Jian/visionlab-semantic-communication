"""阶段 D：配置驱动的可复现训练入口。

按 YAML 配置运行一次完整分类实验，产出曲线图、混淆矩阵、最佳模型 checkpoint
与 ``results.yaml``（配置 + 真实指标 + 时间戳）。

用法（在项目根目录）：
    python experiments/03_reliable_training/run_experiment.py --config configs/mlp_fashionmnist.yaml
    python experiments/03_reliable_training/run_experiment.py --config configs/cnn_fashionmnist.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visionlab.config import load_config
from visionlab.train import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="按 YAML 配置运行一次分类实验")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    config = load_config(config_path)
    output_dir = PROJECT_ROOT / "outputs" / "03_reliable_training" / config.model.name
    results = run_experiment(config, output_dir)

    m = results["metrics"]
    print(f"\n结果已保存到 {output_dir}")
    print(f"  最佳验证准确率 = {m['best_val_acc']:.4f}")
    print(f"  测试集准确率   = {m['test_acc']:.4f}")
    print(f"  测试集宏 F1    = {m['test_macro_f1']:.4f}")
    print(f"  训练耗时       = {m['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
