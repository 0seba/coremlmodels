"""Example: Converting HuggingFace Language Models to CoreML with KV Cache.

This example demonstrates architecture-agnostic conversion that automatically
detects and patches the appropriate layers based on the model type.

Supported architectures:
- Qwen2 (e.g., "Qwen/Qwen2-0.5B")
- Qwen3 (e.g., "Qwen/Qwen3-0.6B")

The script:
1. Loads a model from HuggingFace
2. Auto-detects the architecture from config.model_type
3. Patches attention, RMSNorm, and Linear layers for Neural Engine
4. Wraps with LanguageModelWrapper for KV cache as CoreML state
5. Converts to CoreML with StateType for stateful inference
6. Compares PyTorch and CoreML outputs
7. Analyzes the converted model

Usage:
    # Convert Qwen2 model
    uv run python examples/lm_conversion_example.py

    # Convert Qwen3 model
    uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-0.6B

    # Custom cache length
    uv run python examples/lm_conversion_example.py --cache-length 4096
"""

import argparse
import copy

import numpy as np
import torch
import coremltools as ct
from transformers import AutoConfig, AutoModel

from coremlmodels import (
    analyze_compute_plan,
    create_coreml_state_specs,
    find_target_classes,
    get_architecture_config,
    get_supported_architectures,
    inspect_mil_program,
    LanguageModelWrapper,
    patch_model_attention,
    patch_model_linears,
    patch_model_rmsnorms,
)


def convert_language_model(
    model_name: str,
    seq_len: int = 8,
    cache_length: int = 2048,
    batch_size: int = 1,
    output_path: str | None = None,
    verbose: bool = True,
):
    """Convert a HuggingFace language model to CoreML.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2-0.5B").
        seq_len: Sequence length for tracing.
        cache_length: Maximum KV cache length.
        batch_size: Batch size for the model.
        output_path: Path to save the converted model. If None, auto-generated.
        verbose: Print detailed information during conversion.

    Returns:
        The converted CoreML model.
    """
    print("CoreML Language Model Conversion")
    print("=" * 60)

    # 1. Load config and detect architecture
    print(f"\n[1] Loading config: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type

    print(f"    Model type: {model_type}")
    print(f"    Supported architectures: {get_supported_architectures()}")

    # Get architecture-specific configuration
    arch_config = get_architecture_config(model_type)
    print("    Architecture config:")
    print(f"      - Attention classes: {arch_config.attention_class_names}")
    print(f"      - RMSNorm classes: {arch_config.rmsnorm_class_names}")
    print(f"      - Has QK-norm: {arch_config.has_qk_norm}")

    # 2. Load model
    print(f"\n[2] Loading model: {model_name}")
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

    # 3. Find target classes from the loaded model
    print("\n[3] Finding target classes...")
    attention_classes = find_target_classes(model, arch_config.attention_class_names)
    rmsnorm_classes = find_target_classes(model, arch_config.rmsnorm_class_names)
    print(f"    Attention classes: {[c.__name__ for c in attention_classes]}")
    print(f"    RMSNorm classes: {[c.__name__ for c in rmsnorm_classes]}")

    # 4. Patch Linear layers
    print("\n[4] Patching Linear layers...")
    with torch.inference_mode():
        patch_model_linears(model, verbose=verbose)

    # 5. Patch RMSNorm layers (including q_norm/k_norm if present)
    print("\n[5] Patching RMSNorm layers...")
    with torch.inference_mode():
        patch_model_rmsnorms(
            model,
            target_classes=rmsnorm_classes,
            verbose=verbose,
        )

    # 6. Patch attention layers
    print("\n[6] Patching attention layers...")
    with torch.inference_mode():
        patch_model_attention(
            model,
            target_classes=attention_classes,
            config=config,
            verbose=verbose,
        )

    # 7. Wrap model for CoreML conversion
    print("\n[7] Creating LanguageModelWrapper...")
    with torch.inference_mode():
        wrapped_model = LanguageModelWrapper(
            model,
            cache_length=cache_length,
            channels_first=True,
            device="cpu",
        )
        wrapped_model.eval()
    print(f"    {wrapped_model}")

    # 8. Prepare example inputs for tracing
    print("\n[8] Preparing example inputs...")
    # Channels-first format: (batch, hidden_dim, 1, seq_len)
    example_inputs = (
        torch.randn((batch_size, hidden_dim, 1, seq_len), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.int32),  # position_id
    )
    print(f"    Input shape: {example_inputs[0].shape}")
    print(f"    Position ID shape: {example_inputs[1].shape}")

    # 9. Trace the model
    print("\n[9] Tracing model with torch.jit.trace...")
    with torch.inference_mode():
        traced_model = torch.jit.trace(wrapped_model, example_inputs)
    print("    Tracing complete!")

    # 10. Convert to CoreML
    print("\n[10] Converting to CoreML...")
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

    # 11. Save model
    if output_path is None:
        # Generate output path from model name
        model_short_name = model_name.split("/")[-1].lower().replace("-", "_")
        output_path = f"{model_short_name}_seqlen_{seq_len}.mlpackage"

    mlmodel.save(output_path)
    print(f"    Saved to: {output_path}")

    # 12. Compare PyTorch and CoreML outputs
    print("\n[11] Comparing PyTorch vs CoreML outputs...")

    # Create test input in channels-first format: (batch, hidden_dim, 1, seq_len)
    test_input_cf = np.random.randn(batch_size, hidden_dim, 1, seq_len).astype(
        np.float32
    )
    test_input_tensor = torch.from_numpy(test_input_cf)

    # Convert to sequence-first format for original model: (batch, seq_len, hidden_dim)
    test_input_sf = test_input_tensor.squeeze(2).transpose(1, 2)

    # Run original PyTorch model
    with torch.inference_mode():
        original_output = original_model(
            inputs_embeds=test_input_sf,
            position_ids=torch.arange(seq_len).unsqueeze(0),
        )

        # Convert to channels-first for comparison
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

    # 13. Test stateful inference (second forward pass)
    print("\n[12] Testing stateful inference (second forward pass)...")
    output2 = mlmodel.predict(
        {
            "inputs_embeds": test_input_cf,
            "position_id": np.array([seq_len], dtype=np.int32),
        },
        state,
    )
    print(f"    Second output shape: {output2['output'].shape}")
    print(f"    Second output sample: {output2['output'].flatten()[:5]}")

    # 14. Analyze compute plan
    print("\n[13] Compute Plan Analysis")
    print("     Looking for Neural Engine scheduling...")
    analyze_compute_plan(mlmodel)

    # 15. Inspect MIL program
    print("\n[14] MIL Program Inspection")
    print("     Looking for conv, layer_norm operations...")
    inspect_mil_program(mlmodel)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Model saved to: {output_path}")
    print(f"Architecture: {model_type}")
    print(f"QK-norm enabled: {arch_config.has_qk_norm}")
    print("\nExpected MIL operations:")
    print("  - conv (from Linear patching)")
    print("  - layer_norm (from RMSNorm fusion)")
    print("  - read_state/coreml_update_state (from KV cache)")

    return mlmodel


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace language models to CoreML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert Qwen2 model (default)
    python examples/lm_conversion_example.py

    # Convert Qwen3 model
    python examples/lm_conversion_example.py --model Qwen/Qwen3-0.6B

    # Custom settings
    python examples/lm_conversion_example.py --model Qwen/Qwen2-0.5B --seq-len 16 --cache-length 4096
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model name (default: Qwen/Qwen2-0.5B)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Sequence length for tracing (default: 8)",
    )
    parser.add_argument(
        "--cache-length",
        type=int,
        default=2048,
        help="Maximum KV cache length (default: 2048)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the CoreML model (default: auto-generated)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity",
    )

    args = parser.parse_args()

    convert_language_model(
        model_name=args.model,
        seq_len=args.seq_len,
        cache_length=args.cache_length,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
