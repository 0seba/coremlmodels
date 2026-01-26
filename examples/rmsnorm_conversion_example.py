"""Example: Converting a model with RMSNorm layers to CoreML.

This example demonstrates:
1. Creating a model with RMSNorm and Linear layers
2. Patching both layer types for Neural Engine optimization
3. Converting to CoreML
4. Verifying outputs match
5. Analyzing compute plan and MIL program for LayerNorm fusion
"""

import copy
import time

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from coremlmodels import (
    analyze_compute_plan,
    inspect_mil_program,
    patch_model_linears,
    patch_model_rmsnorms,
)


class TransformerBlockWithRMSNorm(nn.Module):
    """A simplified transformer block using RMSNorm."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.input_norm = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.output_norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.ffn(x)
        x = self.output_norm(x)
        return x


def main():
    print("CoreML RMSNorm Patching Example")
    print("=" * 50)

    # 1. Create model with large sequence length to trigger Neural Engine
    dim = 512
    seq_len = 512  # Large sequence length for Neural Engine scheduling
    model = TransformerBlockWithRMSNorm(dim=dim)
    model.eval()
    print(f"\n[1] Created TransformerBlock (dim={dim}, seq_len={seq_len})")
    print("    - 2x RMSNorm layers")
    print("    - 2x Linear layers (FFN)")

    # Keep a copy of original model for comparison
    original_model = copy.deepcopy(model)
    original_model.eval()

    # 2. Patch RMSNorm layers
    print("\n[2] Patching RMSNorm layers...")
    patched_model = patch_model_rmsnorms(model, verbose=True)

    # 3. Patch Linear layers
    print("\n[3] Patching Linear layers...")
    patched_model = patch_model_linears(patched_model, verbose=True)
    patched_model.eval()

    # 4. Prepare inputs - shape (batch, channels, 1, seq_len)
    batch_size = 1
    patched_input = torch.randn(batch_size, dim, 1, seq_len)

    # 5. Verify PyTorch equivalence before conversion
    print("\n[4] Verifying PyTorch equivalence...")
    with torch.no_grad():
        # Original model expects (batch * seq_len, dim)
        original_input = patched_input.squeeze(2).transpose(1, 2).reshape(-1, dim)
        original_out = original_model(original_input)
        original_out = original_out.reshape(batch_size, seq_len, dim).transpose(1, 2)

        patched_out = patched_model(patched_input).squeeze(2)

    max_diff = torch.max(torch.abs(original_out - patched_out)).item()
    print(f"    Max difference (PyTorch): {max_diff:.6e}")
    if max_diff < 1e-3:
        print("    [OK] PyTorch outputs match!")
    else:
        print("    [WARNING] PyTorch outputs differ significantly")

    # 6. Convert to CoreML
    print("\n[5] Converting to CoreML...")
    traced_model = torch.jit.trace(patched_model, patched_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=(batch_size, dim, 1, seq_len), name="input")],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )

    model_path = "TransformerBlock_Patched.mlpackage"
    mlmodel.save(model_path)
    print(f"    Saved CoreML model to {model_path}")

    # 7. Verify CoreML output
    print("\n[6] Verifying CoreML output...")
    prediction = mlmodel.predict({"input": patched_input.numpy()})

    coreml_out = prediction["output"].flatten()
    pytorch_out = patched_out.numpy().flatten()
    max_diff_coreml = np.max(np.abs(coreml_out - pytorch_out))
    print(f"    Max difference (CoreML vs PyTorch): {max_diff_coreml:.6f}")
    if max_diff_coreml < 0.1:
        print("    [OK] CoreML outputs match PyTorch (within FP16 tolerance)!")
    else:
        print("    [WARNING] CoreML outputs differ from PyTorch")

    # 8. Timing
    print("\n[7] Timing CoreML Inference (20 runs)...")
    for _ in range(5):  # Warmup
        mlmodel.predict({"input": patched_input.numpy()})

    start_time = time.time()
    for _ in range(20):
        mlmodel.predict({"input": patched_input.numpy()})
    coreml_time_ms = (time.time() - start_time) * 1000 / 20
    print(f"    Average inference time: {coreml_time_ms:.4f} ms")

    # 9. Analyze compute plan
    print("\n[8] Compute Plan Analysis")
    print("    Looking for 'NeuralEngine' in preferred device...")
    analyze_compute_plan(mlmodel)

    # 10. Inspect MIL program
    print("\n[9] MIL Program Inspection")
    print("    Looking for 'layer_norm' operation (fusion indicator)...")
    inspect_mil_program(mlmodel)

    print("\n" + "=" * 50)
    print("If you see 'layer_norm' in the MIL output, the fusion worked!")
    print("The RMSNorm -> LayerNorm conversion is successful.")


if __name__ == "__main__":
    main()
