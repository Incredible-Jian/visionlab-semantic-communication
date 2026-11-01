# 实验记录模板

> 每次实验都**如实填写**，禁止虚构损失 / 准确率 / 耗时 / 图。所有指标必须来自
> 真实运行，并与 ``results.yaml`` 中的数值一一对应。

## 实验信息

- 日期：
- 阶段 / 编号：
- 目标：
- 数据集：
- 配置文件（`configs/*.yaml`）：
- 随机种子：

## 超参数

| 分组 | 参数 | 值 |
| --- | --- | --- |
| data | root / batch_size / val_ratio / seed / num_workers | |
| model | name / num_classes / hidden_dims / dropout / in_channels | |
| train | epochs / lr | |

## 结果（来自 results.yaml）

| 指标 | 值 |
| --- | --- |
| 最佳验证准确率 | |
| 测试损失 | |
| 测试准确率 | |
| 测试宏 F1 | |
| 训练耗时（秒） | |
| 参数量 | |

## 结论与观察

（模型表现、容易混淆的类别、与上一次实验的对比、任何异常或可复现性说明）

## 复现命令

```bash
python experiments/03_reliable_training/run_experiment.py --config configs/<name>.yaml
```
