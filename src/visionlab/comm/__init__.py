"""图像通信的简化模块（阶段 E）。"""

from visionlab.comm.channel import add_gaussian_noise
from visionlab.comm.codec import downsample_codec, jpeg_codec

__all__ = ["add_gaussian_noise", "downsample_codec", "jpeg_codec"]
