"""Tests for Linear to Conv2d patching functionality."""

import copy

import torch
import torch.nn as nn

from coremlmodels import LinearToConv2dPatcher, patch_model_linears


def test_patcher_forward_matches_linear():
    """Test that LinearToConv2dPatcher forward equals linear forward with reshape."""
    linear = nn.Linear(10, 20)
    patcher = LinearToConv2dPatcher(linear)

    x = torch.randn(2, 10)
    x_4d = x.view(2, 10, 1, 1)

    linear_out = linear(x)
    patcher_out = patcher(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(linear_out, patcher_out)


def test_patcher_forward_matches_linear_with_bias():
    """Test equivalence with bias."""
    linear = nn.Linear(10, 20, bias=True)
    patcher = LinearToConv2dPatcher(linear)

    x = torch.randn(3, 10)
    x_4d = x.view(3, 10, 1, 1)

    linear_out = linear(x)
    patcher_out = patcher(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(linear_out, patcher_out)


def test_patcher_forward_matches_linear_no_bias():
    """Test equivalence without bias."""
    linear = nn.Linear(10, 20, bias=False)
    patcher = LinearToConv2dPatcher(linear)

    x = torch.randn(1, 10)
    x_4d = x.view(1, 10, 1, 1)

    linear_out = linear(x)
    patcher_out = patcher(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(linear_out, patcher_out)


def test_patched_model_forward_matches_original():
    """Test that patched model forward equals original with reshape."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    original = copy.deepcopy(model)
    patched = patch_model_linears(model)

    x = torch.randn(2, 10)
    x_4d = x.view(2, 10, 1, 1)

    patched.eval()
    original.eval()

    with torch.no_grad():
        original_out = original(x)
        patched_out = patched(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(original_out, patched_out)


def test_patched_nested_model():
    """Test patching nested modules."""

    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 15)
            self.block = nn.Sequential(
                nn.Linear(15, 20),
                nn.ReLU(),
            )
            self.layer2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.layer1(x)
            x = self.block(x)
            x = self.layer2(x)
            return x

    model = NestedModel()
    original = copy.deepcopy(model)
    patched = patch_model_linears(model)

    x = torch.randn(2, 10)
    x_4d = x.view(2, 10, 1, 1)

    patched.eval()
    original.eval()

    with torch.no_grad():
        original_out = original(x)
        patched_out = patched(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(original_out, patched_out)


def test_skip_modules():
    """Test skipping specific modules."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    patched = patch_model_linears(model, skip_modules=["2"])

    assert isinstance(patched[0], LinearToConv2dPatcher)
    assert isinstance(patched[2], nn.Linear)
