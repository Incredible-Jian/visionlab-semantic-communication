"""简化图像编解码：下采样与 JPEG 有损压缩（基于 Pillow）。

本模块仅用于教学演示，模拟"压缩—重建"带来的信息损失，不代表真实通信编码。
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _to_image(x: np.ndarray) -> Image.Image:
    """浮点 [0, 1] 数组转 8-bit 灰度 PIL 图像。"""
    arr = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L")


def _from_image(img: Image.Image) -> np.ndarray:
    """PIL 图像转浮点 [0, 1] 数组。"""
    return np.asarray(img, dtype=float) / 255.0


def downsample_codec(x: np.ndarray, factor: int) -> np.ndarray:
    """下采样压缩：按 ``factor`` 缩小后再放大回原尺寸（双线性插值）。

    ``factor`` 越大压缩越狠、重建越模糊，用作对比 JPEG 的朴素基线。
    """
    if factor < 1:
        raise ValueError(f"factor 必须 >= 1：{factor}")
    img = _to_image(x)
    w, h = img.size
    small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    back = small.resize((w, h), Image.BILINEAR)
    return _from_image(back)


def jpeg_codec(x: np.ndarray, quality: int) -> np.ndarray:
    """JPEG 有损压缩：按 ``quality``（1~100，越小压缩越狠）编码后立即解码。"""
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality 必须在 1~100 之间：{quality}")
    img = _to_image(x)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = Image.open(buf)
    decoded.load()
    return _from_image(decoded)
