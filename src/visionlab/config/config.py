"""实验配置管理：dataclass + YAML 读写（阶段 D）。

用嵌套 dataclass 描述一次实验的配置（数据 / 模型 / 训练三节），可从 YAML
加载（拒绝未知配置项），也可回写为 YAML 记录。只依赖已有的 PyYAML，不引入
额外配置库。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclass
class DataConfig:
    """数据加载配置。"""

    root: str = "data"
    batch_size: int = 128
    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0


@dataclass
class ModelConfig:
    """模型配置。``name`` 取 ``"mlp"`` 或 ``"cnn"``，其余字段按需使用。"""

    name: str = "mlp"
    num_classes: int = 10
    # MLP 专用
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    # CNN 专用
    in_channels: int = 1


@dataclass
class TrainConfig:
    """训练配置。"""

    epochs: int = 15
    lr: float = 1e-3


@dataclass
class Config:
    """一次实验的完整配置，由 data / model / train 三节组成。"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def to_dict(config: Config) -> dict:
    """把 ``Config`` 递归转成普通 dict（tuple 转 list，便于 YAML 序列化）。"""
    return _to_plain(config)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def from_dict(data: dict) -> Config:
    """从 dict 构造 ``Config``，拒绝未知配置项。"""
    if not isinstance(data, dict):
        raise TypeError(f"配置应为 dict，实际为 {type(data).__name__}")
    return Config(
        data=_from_dict(DataConfig, data.get("data", {})),
        model=_from_dict(ModelConfig, data.get("model", {})),
        train=_from_dict(TrainConfig, data.get("train", {})),
    )


def _from_dict(cls: type, data: dict) -> Any:
    """从 dict 构造单个 dataclass，拒绝未知键并处理 list→tuple。"""
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} 配置节应为 dict，实际为 {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__} 含未知配置项：{sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs: dict = {}
    for f in fields(cls):
        if f.name not in data:
            continue  # 缺失的键使用默认值
        value = data[f.name]
        if get_origin(hints[f.name]) is tuple and isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """从 YAML 文件加载配置。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return from_dict(data)


def save_config(config: Config, path: str | Path) -> None:
    """把 ``Config`` 写为 YAML 文件。"""
    write_yaml(to_dict(config), path)


def write_yaml(data: dict, path: str | Path) -> None:
    """把普通 dict 写为 YAML 文件（自动创建父目录）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
