"""测试 visionlab.config 的配置加载 / 校验 / 回写。"""

import pytest

from visionlab.config import (
    Config,
    from_dict,
    load_config,
    save_config,
    to_dict,
)


def test_default_config() -> None:
    cfg = Config()
    assert cfg.data.batch_size == 128
    assert cfg.model.name == "mlp"
    assert cfg.train.epochs == 15


def test_from_dict_overrides_and_defaults() -> None:
    cfg = from_dict({"model": {"name": "cnn"}, "train": {"epochs": 5}})
    assert cfg.model.name == "cnn"
    assert cfg.train.epochs == 5
    assert cfg.data.batch_size == 128  # 未提供的节使用默认值


def test_from_dict_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        from_dict({"model": {"unknown_field": 1}})


def test_from_dict_rejects_non_mapping_section() -> None:
    with pytest.raises(TypeError):
        from_dict({"model": "mlp"})


def test_from_dict_converts_hidden_dims_to_tuple() -> None:
    cfg = from_dict({"model": {"hidden_dims": [32, 16]}})
    assert cfg.model.hidden_dims == (32, 16)


def test_save_and_load_roundtrip(tmp_path) -> None:
    cfg = from_dict({"model": {"name": "cnn", "hidden_dims": [64]}, "train": {"epochs": 3}})
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert to_dict(loaded) == to_dict(cfg)


def test_load_config_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "not_exist.yaml")
