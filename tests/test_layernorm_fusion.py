"""Test LayerNorm fusion for arbitrary axes with gamma-only patterns (RMSNorm style)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

# Import to trigger pass registration
from coremlmodels import register_extended_passes


class RMSNormConcatTrick(nn.Module):
    """RMSNorm using [x, -x] concat trick for CoreML fusion.

    Normalizes over axis=2 (the channel dimension in BCHW format).
    This simulates QK-norm in attention where we normalize query states.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        # Gamma weight for scaling
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with concat trick on axis=2.

        Input shape: (batch, heads, channels, seq_len) e.g., (1, 16, 128, 8)
        Normalize over channels (axis=2).
        """
        axis = 2

        # Concatenate [x, -x] along axis
        x_concat = torch.cat([x, -x], dim=axis)

        # LayerNorm-equivalent operations
        x_float = x_concat.float()
        channels_mean = x_float.mean(dim=axis, keepdim=True)
        zero_mean = x_float - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        variance = zero_mean_sq.mean(dim=axis, keepdim=True)
        denom = (variance + self.eps).rsqrt()
        out = zero_mean * denom

        # Apply gamma weight - reshape for broadcasting: [1, 1, 2*channels, 1]
        weight_concat = torch.cat([self.weight, torch.zeros_like(self.weight)])
        weight_reshaped = weight_concat.view(1, 1, -1, 1)
        out = out * weight_reshaped

        # Take first half (x portion) using slice
        out = out[:, :, : self.normalized_shape, :]

        return out.to(x.dtype)


def test_layernorm_fusion_with_gamma():
    """Test that RMSNorm pattern with gamma is fused into a single layer_norm op."""

    print("=" * 60)
    print("Testing LayerNorm fusion with gamma (RMSNorm pattern)")
    print("=" * 60)

    # Create model
    channels = 128
    model = RMSNormConcatTrick(normalized_shape=channels)
    model.eval()

    # Example input: (batch=1, heads=16, channels=128, seq_len=8)
    batch, heads, seq_len = 1, 16, 8
    example_input = torch.randn(batch, heads, channels, seq_len)

    print(f"\nInput shape: {example_input.shape}")
    print(f"Normalizing over axis=2 (channels={channels})")

    # Trace model
    print("\nTracing model...")
    with torch.inference_mode():
        traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    print("Converting to CoreML...")
    register_extended_passes()

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                shape=(batch, heads, channels, seq_len),
                name="input",
            ),
        ],
        outputs=[
            ct.TensorType(name="output"),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Analyze the MIL program
    print("\n" + "=" * 60)
    print("MIL Program Analysis")
    print("=" * 60)

    mil_program = mlmodel._mil_program

    layer_norm_count = 0
    mul_after_layernorm = 0

    for func in mil_program.functions.values():
        for op in func.operations:
            if op.op_type == "layer_norm":
                layer_norm_count += 1
                print(f"\nFound layer_norm operation:")
                print(f"  Output: {op.outputs[0].name}")
                print(f"  Axes: {op.axes.val if op.axes else 'None'}")
                print(f"  Has gamma: {op.gamma is not None}")
                print(f"  Has beta: {op.beta is not None}")
                if op.gamma is not None:
                    print(f"  Gamma shape: {op.gamma.shape}")

                # Check if there's a mul immediately after this layer_norm
                for child_op in op.outputs[0].child_ops:
                    if child_op.op_type == "mul":
                        mul_after_layernorm += 1
                        print(f"\n  WARNING: Found mul after layer_norm!")
                        print(f"  Mul output: {child_op.outputs[0].name}")

    print(f"\n" + "-" * 60)
    print(f"Summary:")
    print(f"  layer_norm operations: {layer_norm_count}")
    print(f"  mul operations after layer_norm: {mul_after_layernorm}")

    # Test correctness
    print(f"\n" + "=" * 60)
    print("Testing correctness")
    print("=" * 60)

    test_input = np.random.randn(batch, heads, channels, seq_len).astype(np.float32)

    # PyTorch output
    with torch.inference_mode():
        pytorch_output = model(torch.from_numpy(test_input)).numpy()

    # CoreML output
    coreml_output = mlmodel.predict({"input": test_input})["output"]

    abs_diff = np.abs(pytorch_output - coreml_output)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"CoreML output shape: {coreml_output.shape}")
    print(f"Max absolute difference: {abs_diff.max():.6f}")
    print(f"Mean absolute difference: {abs_diff.mean():.6f}")

    # Assertions
    assert layer_norm_count >= 1, "Expected at least one layer_norm operation"

    if mul_after_layernorm > 0:
        print("\n[FAIL] Gamma not fused into layer_norm - separate mul operation found!")
        return False
    else:
        print("\n[PASS] Gamma successfully fused into layer_norm!")
        return True


if __name__ == "__main__":
    success = test_layernorm_fusion_with_gamma()
    exit(0 if success else 1)
