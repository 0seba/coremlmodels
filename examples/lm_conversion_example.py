"""Example: Converting a HuggingFace Language Model to CoreML with KV Cache.

This example demonstrates:
1. Loading a Qwen2 model from HuggingFace
2. Patching attention, RMSNorm, and Linear layers for Neural Engine
3. Wrapping with LanguageModelWrapper for KV cache as CoreML state
4. Converting to CoreML with StateType for stateful inference
5. Comparing PyTorch and CoreML outputs
6. Analyzing the converted model

NOTE: For development/debugging, you can modify LanguageModelWrapper to
process only the first transformer layer by uncommenting the line:
    # for decoder_layer in self.layer.layers[:1]:
"""

import copy

import numpy as np
import torch
import coremltools as ct
from transformers import AutoConfig, AutoModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm

from coremlmodels import (
    analyze_compute_plan,
    create_coreml_state_specs,
    inspect_mil_program,
    LanguageModelWrapper,
    patch_model_attention,
    patch_model_linears,
    patch_model_rmsnorms,
)


def main():
    print("CoreML Language Model Conversion Example")
    print("=" * 60)

    # Configuration
    model_name = "Qwen/Qwen2-0.5B"  # Small model for testing
    seq_len = 8  # Sequence length for tracing (can be flexible at runtime)
    cache_length = 2048  # Maximum KV cache length
    batch_size = 1

    # 1. Load model
    print(f"\n[1] Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Keep a copy of the original model for comparison
    original_model = copy.deepcopy(model)
    original_model.eval()

    hidden_dim = config.hidden_size
    print(f"    Hidden dim: {hidden_dim}")
    print(f"    Num layers: {config.num_hidden_layers}")
    print(f"    Num heads: {config.num_attention_heads}")
    print(f"    Num KV heads: {config.num_key_value_heads}")

    # 2. Patch Linear layers
    print("\n[2] Patching Linear layers...")
    with torch.inference_mode():
        patch_model_linears(model, verbose=True)

    # 3. Patch RMSNorm layers
    print("\n[3] Patching RMSNorm layers...")
    with torch.inference_mode():
        patch_model_rmsnorms(
            model,
            target_classes=(Qwen2RMSNorm,),
            verbose=True,
        )

    # 4. Patch attention layers
    print("\n[4] Patching attention layers...")
    with torch.inference_mode():
        patch_model_attention(
            model,
            target_classes=(Qwen2Attention,),
            config=config,
            verbose=True,
        )

    # 5. Wrap model for CoreML conversion
    print("\n[5] Creating LanguageModelWrapper...")
    with torch.inference_mode():
        wrapped_model = LanguageModelWrapper(
            model,
            cache_length=cache_length,
            channels_first=True,
            device="cpu",
        )
        wrapped_model.eval()
    print(f"    {wrapped_model}")

    # 6. Prepare example inputs for tracing
    print("\n[6] Preparing example inputs...")
    # Channels-first format: (batch, hidden_dim, 1, seq_len)
    example_inputs = (
        torch.randn((batch_size, hidden_dim, 1, seq_len), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.int32),  # position_id
    )
    print(f"    Input shape: {example_inputs[0].shape}")
    print(f"    Position ID shape: {example_inputs[1].shape}")

    # 7. Trace the model
    print("\n[7] Tracing model with torch.jit.trace...")
    with torch.inference_mode():
        traced_model = torch.jit.trace(wrapped_model, example_inputs)
    print("    Tracing complete!")

    # 8. Convert to CoreML
    print("\n[8] Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                shape=(batch_size, hidden_dim, 1, seq_len),
                name="inputs_embeds",
            ),
            ct.TensorType(
                shape=(1,),
                name="position_id",
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="output"),
        ],
        states=create_coreml_state_specs(wrapped_model),
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    # 9. Save model
    model_path = f"qwen2_lm_seqlen_{seq_len}.mlpackage"
    mlmodel.save(model_path)
    print(f"    Saved to: {model_path}")

    # 10. Compare PyTorch and CoreML outputs
    print("\n[9] Comparing PyTorch vs CoreML outputs...")

    # Create test input in channels-first format: (batch, hidden_dim, 1, seq_len)
    test_input_cf = np.random.randn(batch_size, hidden_dim, 1, seq_len).astype(
        np.float32
    )
    test_input_tensor = torch.from_numpy(test_input_cf)

    # Convert to sequence-first format for original model: (batch, seq_len, hidden_dim)
    test_input_sf = test_input_tensor.squeeze(2).transpose(1, 2)

    # Run original PyTorch model (only first N layers for comparison)
    # Temporarily limit layers to match the wrapped model's layer count
    num_layers_to_use = 2  # Match the [:1] in LanguageModelWrapper
    with torch.inference_mode():
        # Save original layers and temporarily replace with subset
        original_layers = original_model.layers
        # original_model.layers = original_layers[:num_layers_to_use]
        original_model.layers = original_layers

        # Run full model forward
        original_output = original_model(
            inputs_embeds=test_input_sf,
            position_ids=torch.arange(seq_len).unsqueeze(0),
        )

        # Restore original layers
        original_model.layers = original_layers

        # Convert to channels-first for comparison: (batch, seq_len, hidden_dim) -> (batch, hidden_dim, 1, seq_len)
        pytorch_output = (
            original_output.last_hidden_state.transpose(1, 2).unsqueeze(2).numpy()
        )

    # Run CoreML model
    state = mlmodel.make_state()
    coreml_result = mlmodel.predict(
        {"inputs_embeds": test_input_cf, "position_id": np.array([0], dtype=np.int32)},
        state,
    )
    coreml_output = coreml_result["output"]

    # Compare outputs
    print(f"    PyTorch output shape: {pytorch_output.shape}")
    print(f"    CoreML output shape: {coreml_output.shape}")

    # Compute differences
    abs_diff = np.abs(pytorch_output - coreml_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-7)

    print("\n    Absolute difference statistics:")
    print(f"      Max:  {abs_diff.max():.6f}")
    print(f"      Mean: {abs_diff.mean():.6f}")
    print(f"      Std:  {abs_diff.std():.6f}")

    print("\n    Relative difference statistics:")
    print(f"      Max:  {rel_diff.max():.6f}")
    print(f"      Mean: {rel_diff.mean():.6f}")

    # Sample outputs for visual comparison
    print(f"\n    PyTorch sample: {pytorch_output.flatten()[:5]}")
    print(f"    CoreML sample:  {coreml_output.flatten()[:5]}")

    # Check if outputs are close (FP16 tolerance)
    if abs_diff.max() < 0.1 and rel_diff.mean() < 0.1:
        print("\n    [OK] Outputs match within FP16 tolerance!")
    else:
        print("\n    [WARNING] Outputs differ significantly - check implementation")

    # 11. Test stateful inference (second forward pass)
    print("\n[10] Testing stateful inference (second forward pass)...")
    output2 = mlmodel.predict(
        {
            "inputs_embeds": test_input_cf,
            "position_id": np.array([seq_len], dtype=np.int32),
        },
        state,
    )
    print(f"    Second output shape: {output2['output'].shape}")
    print(f"    Second output sample: {output2['output'].flatten()[:5]}")

    # 12. Analyze compute plan
    print("\n[11] Compute Plan Analysis")
    print("     Looking for Neural Engine scheduling...")
    analyze_compute_plan(mlmodel)

    # 13. Inspect MIL program
    print("\n[12] MIL Program Inspection")
    print("     Looking for conv, layer_norm operations...")
    inspect_mil_program(mlmodel)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Model saved to: {model_path}")
    print("\nExpected MIL operations:")
    print("  - conv (from Linear patching)")
    print("  - layer_norm (from RMSNorm fusion)")
    print("  - read_state/coreml_update_state (from KV cache)")


if __name__ == "__main__":
    main()
