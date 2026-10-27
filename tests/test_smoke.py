"""冒烟测试：确认 visionlab 包可以正常导入，且版本信息有效。"""

import visionlab


def test_package_name() -> None:
    assert visionlab.__name__ == "visionlab"


def test_version_is_nonempty_string() -> None:
    assert isinstance(visionlab.__version__, str)
    assert visionlab.__version__
