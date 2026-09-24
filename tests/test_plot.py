"""Tests for plot utilities."""

import torch

from driving_world_model.plot import _denormalize


def test_denormalize_shape():
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5, 0.5])
    tensor = torch.zeros(1, 4, 8, 8)
    img = _denormalize(tensor, mean, std)
    assert img.shape == (8, 8, 3)


def test_denormalize_clamps_out_of_range_values():
    mean = torch.zeros(4)
    std = torch.ones(4)
    tensor = torch.full((1, 4, 2, 2), 10.0)
    img = _denormalize(tensor, mean, std)
    assert torch.all(img == 1.0)


def test_denormalize_clamps_negative_values():
    mean = torch.zeros(4)
    std = torch.ones(4)
    tensor = torch.full((1, 4, 2, 2), -10.0)
    img = _denormalize(tensor, mean, std)
    assert torch.all(img == 0.0)


def test_denormalize_drops_alpha_channel():
    mean = torch.zeros(4)
    std = torch.ones(4)
    tensor = torch.rand(1, 4, 4, 4)
    img = _denormalize(tensor, mean, std)
    assert img.shape[-1] == 3


def test_denormalize_inverts_normalization():
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5])
    std = torch.tensor([0.2, 0.2, 0.2, 0.2])
    original = torch.full((4, 3, 3), 0.7)
    normalized = ((original - mean[:, None, None]) / std[:, None, None]).unsqueeze(0)
    img = _denormalize(normalized, mean, std)
    assert torch.allclose(img, torch.full((3, 3, 3), 0.7), atol=1e-5)
