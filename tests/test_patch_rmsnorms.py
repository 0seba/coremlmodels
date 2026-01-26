"""Tests for RMSNorm to LayerNorm-fusable ops patching functionality."""

import copy

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from coremltools.models.compute_plan import MLComputePlan

from coremlmodels import (
    RMSNormToLayerNormPatcher,
    patch_model_linears,
    patch_model_rmsnorms,
)


def test_patcher_forward_matches_rmsnorm():
    """Test that patcher forward equals RMSNorm forward."""
    rmsnorm = nn.RMSNorm(64)
    patcher = RMSNormToLayerNormPatcher(rmsnorm)

    x = torch.randn(2, 64)
    x_4d = x.view(2, 64, 1, 1)

    rmsnorm_out = rmsnorm(x)
    patcher_out = patcher(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(rmsnorm_out, patcher_out, rtol=1e-4, atol=1e-4)


def test_patcher_forward_without_weight():
    """Test equivalence without elementwise affine."""
    rmsnorm = nn.RMSNorm(64, elementwise_affine=False)
    patcher = RMSNormToLayerNormPatcher(rmsnorm)

    x = torch.randn(2, 64)
    x_4d = x.view(2, 64, 1, 1)

    rmsnorm_out = rmsnorm(x)
    patcher_out = patcher(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(rmsnorm_out, patcher_out, rtol=1e-4, atol=1e-4)


def test_patcher_with_large_values():
    """Test with values that might cause FP16 overflow."""
    rmsnorm = nn.RMSNorm(64)
    patcher = RMSNormToLayerNormPatcher(rmsnorm)

    x = torch.randn(2, 64) * 100
    x_4d = x.view(2, 64, 1, 1)

    rmsnorm.eval()
    patcher.eval()

    with torch.no_grad():
        rmsnorm_out = rmsnorm(x)
        patcher_out = patcher(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(rmsnorm_out, patcher_out, rtol=1e-4, atol=1e-4)


def test_skip_modules():
    """Test skipping specific modules."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.RMSNorm(64)
            self.norm2 = nn.RMSNorm(64)

        def forward(self, x):
            return self.norm2(self.norm1(x))

    model = Model()
    patched = patch_model_rmsnorms(model, skip_modules=["norm2"])

    assert isinstance(patched.norm1, RMSNormToLayerNormPatcher)
    assert isinstance(patched.norm2, nn.RMSNorm)


def test_custom_rmsnorm_with_variance_epsilon():
    """Test patching custom RMSNorm with variance_epsilon attribute."""

    class CustomRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * x

    custom_norm = CustomRMSNorm(64)
    patcher = RMSNormToLayerNormPatcher(custom_norm)

    x = torch.randn(2, 64)
    x_4d = x.view(2, 64, 1, 1)

    custom_out = custom_norm(x)
    patcher_out = patcher(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(custom_out, patcher_out, rtol=1e-4, atol=1e-4)


def test_coreml_conversion_with_neural_engine():
    """Test CoreML conversion runs on Neural Engine and outputs match."""

    class TransformerBlock(nn.Module):
        def __init__(self, dim=512):
            super().__init__()
            self.norm = nn.RMSNorm(dim)
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            x = self.norm(x)
            x = self.linear(x)
            return x

    dim = 512
    seq_len = 512  # Large sequence length to trigger Neural Engine scheduling

    model = TransformerBlock(dim=dim)
    original = copy.deepcopy(model)

    # Patch both Linear and RMSNorm
    patched = patch_model_rmsnorms(model)
    patched = patch_model_linears(patched)

    patched.eval()
    original.eval()

    # Input shape: (batch, channels, 1, seq_len) - seq_len in last dim
    x = torch.randn(1, dim, 1, seq_len)

    # Trace and convert to CoreML
    traced = torch.jit.trace(patched, x)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=(1, dim, 1, seq_len), name="input")],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )

    # Verify outputs match
    with torch.no_grad():
        original_out = original(x.squeeze(2).transpose(1, 2).reshape(-1, dim))
        original_out = original_out.reshape(1, seq_len, dim).transpose(1, 2)

    coreml_out = mlmodel.predict({"input": x.numpy()})["output"]
    pytorch_out = patched(x).detach().numpy()

    # FP16 tolerance
    np.testing.assert_allclose(coreml_out, pytorch_out, rtol=0.05, atol=0.05)

    # Verify Neural Engine is used for main compute operations
    compiled_path = mlmodel.get_compiled_model_path()
    compute_plan = MLComputePlan.load_from_path(compiled_path)
    prog = compute_plan.model_structure.program

    ne_ops = []
    for f in prog.functions.values():
        for op in f.block.operations:
            op_name = str(op.operator_name)
            if op_name in ("conv", "layer_norm"):
                usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(
                    op
                )
                device = usage.preferred_compute_device.__class__.__name__.replace(
                    "ML", ""
                ).replace("ComputeDevice", "")
                ne_ops.append((op_name, device))

    # Check that conv and layer_norm ops run on Neural Engine
    for op_name, device in ne_ops:
        assert device == "NeuralEngine", (
            f"{op_name} should run on NeuralEngine, got {device}"
        )
