"""测试 visionlab.models 的 MLP / CNN 前向形状与可训练性。"""

import torch

from visionlab.models import CNN, MLP


def test_mlp_forward_shape() -> None:
    model = MLP(input_dim=784, hidden_dims=(32, 16), num_classes=10)
    out = model(torch.randn(8, 1, 28, 28))
    assert out.shape == (8, 10)


def test_mlp_flattens_arbitrary_input() -> None:
    model = MLP(input_dim=16, hidden_dims=(8,), num_classes=3)
    out = model(torch.randn(5, 16))
    assert out.shape == (5, 3)


def test_cnn_forward_shape() -> None:
    model = CNN(in_channels=1, num_classes=10)
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_models_produce_gradients() -> None:
    for model in (MLP(), CNN()):
        model.train()
        loss = model(torch.randn(2, 1, 28, 28)).sum()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())
