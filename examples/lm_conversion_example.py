"""Example: Converting HuggingFace Language Models to CoreML with KV Cache.

This example demonstrates architecture-agnostic conversion that automatically
detects and patches the appropriate layers based on the model type.

Supported architectures:
- Qwen2 (e.g., "Qwen/Qwen2-0.5B")
- Qwen3 (e.g., "Qwen/Qwen3-0.6B")

The script supports two modes:
1. Single model conversion: Converts the entire model to one CoreML package
2. Chunked conversion: Splits the model into N chunks for large models (>2GB)
   that exceed the Neural Engine limit

Additional exports:
- Embeddings: Export embedding weights as .npy file (float16)
- LM Head: Export language model head as CoreML with vocabulary chunking for
  Neural Engine compatibility (handles weight dimension limit of ~16384)

Usage:
    # Convert Qwen2 model (single)
    uv run python examples/lm_conversion_example.py

    # Convert Qwen3 model (single)
    uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-0.6B

    # Convert large model in 4 chunks (for models > 2GB)
    uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4

    # Convert only specific chunks (reduces memory usage)
    uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 0
    uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 1,2

    # Skip verification to reduce memory (useful for large models)
    uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --skip-verification

    # Minimum memory mode: convert one chunk at a time, skip model load
    uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 0 --skip-model-load

    # Export embeddings and LM head (with model conversion)
    uv run python examples/lm_conversion_example.py --export-embeddings --export-lm-head

    # Export ONLY embeddings and LM head (skip model conversion for faster development)
    uv run python examples/lm_conversion_example.py --export-embeddings --export-lm-head --components-only

    # Export LM head with custom chunk size
    uv run python examples/lm_conversion_example.py --export-lm-head --lm-head-chunk-size 8192

    # Quick debug with only 2 layers
    uv run python examples/lm_conversion_example.py --num-layers 2

    # Enable compute plan and MIL analysis
    uv run python examples/lm_conversion_example.py --analyze-compute-plan --analyze-mil

    # Cache compiled models for faster subsequent loads
    uv run python examples/lm_conversion_example.py --cache-compiled
"""

import argparse
import copy
import gc
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


# =============================================================================
# Compiled Model Caching Utilities
# =============================================================================


def get_compiled_model_path(mlpackage_path: Path) -> Path:
    """Get the compiled model path (.mlmodelc) corresponding to an .mlpackage.

    The compiled model is stored in the same directory as the .mlpackage,
    with the same name but .mlmodelc extension.

    Args:
        mlpackage_path: Path to the .mlpackage file/directory

    Returns:
        Path to the corresponding .mlmodelc directory
    """
    return mlpackage_path.with_suffix(".mlmodelc")


def cache_compiled_model(mlmodel, mlpackage_path: Path, verbose: bool = True) -> Path:
    """Cache the compiled model (.mlmodelc) to the same directory as the .mlpackage.

    CoreML compiles models to a temporary location. This function copies
    the compiled model to a persistent location for faster subsequent loads.

    Args:
        mlmodel: The loaded MLModel object
        mlpackage_path: Path to the .mlpackage file/directory
        verbose: Print progress information

    Returns:
        Path to the cached .mlmodelc directory
    """
    compiled_path = get_compiled_model_path(mlpackage_path)
    temp_compiled_path = mlmodel.get_compiled_model_path()

    if verbose:
        print(f"    Caching compiled model to: {compiled_path}")

    # Copy the compiled model directory
    shutil.copytree(temp_compiled_path, str(compiled_path), dirs_exist_ok=True)

    return compiled_path

from coremlmodels import (
    analyze_compute_plan,
    ChunkedLanguageModelWrapper,
    convert_lm_head,
    create_coreml_state_specs,
    create_chunked_coreml_state_specs,
    export_embeddings,
    find_target_classes,
    get_architecture_config,
    get_supported_architectures,
    inspect_mil_program,
    LanguageModelWrapper,
    patch_model_attention,
    patch_model_linears,
    patch_model_rmsnorms,
    register_extended_passes,
    RMSNormToLayerNormPatcher,
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelContext:
    """Holds loaded model, config, and architecture information."""

    model: nn.Module
    config: object
    arch_config: object
    attention_classes: tuple
    rmsnorm_classes: tuple
    hidden_dim: int
    model_type: str
    # Pre-computed reference output for verification (much smaller than full model copy)
    reference_output: np.ndarray | None = None
    reference_input_cf: np.ndarray | None = None


@dataclass
class VerificationResult:
    """Results from output verification."""

    abs_diff_max: float
    abs_diff_mean: float
    abs_diff_std: float
    rel_diff_max: float
    rel_diff_mean: float
    passed: bool


# =============================================================================
# Model Loading and Patching
# =============================================================================


def load_model_and_config(
    model_name: str,
    num_layers: int | None = None,
    verbose: bool = True,
    compute_reference: bool = True,
    batch_size: int = 1,
    seq_len: int = 8,
) -> ModelContext:
    """Load model from HuggingFace and detect architecture.

    Args:
        model_name: HuggingFace model name.
        num_layers: Optional layer limit for debugging.
        verbose: Print detailed information.
        compute_reference: Compute reference output before patching for verification.
            Set to False to skip (reduces memory slightly, skips verification).
        batch_size: Batch size for reference output computation.
        seq_len: Sequence length for reference output computation.

    Returns:
        ModelContext with loaded model and configuration.
    """
    if verbose:
        print(f"Loading config: {model_name}")

    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type

    if verbose:
        print(f"    Model type: {model_type}")
        print(f"    Supported architectures: {get_supported_architectures()}")

    arch_config = get_architecture_config(model_type)
    if verbose:
        print("    Architecture config:")
        print(f"      - Attention classes: {arch_config.attention_class_names}")
        print(f"      - RMSNorm classes: {arch_config.rmsnorm_class_names}")
        print(f"      - Has QK-norm: {arch_config.has_qk_norm}")

    if verbose:
        print(f"Loading model: {model_name}")

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Optionally truncate layers
    if num_layers is not None and num_layers < config.num_hidden_layers:
        if verbose:
            print(f"    Truncating from {config.num_hidden_layers} to {num_layers} layers...")
        model.layers = model.layers[:num_layers]
        config.num_hidden_layers = num_layers

    hidden_dim = config.hidden_size

    # Compute reference output before patching (much smaller than keeping full model copy)
    reference_output = None
    reference_input_cf = None
    if compute_reference:
        if verbose:
            print("    Computing reference output before patching...")
        reference_input_cf, _, reference_input_sf = create_test_inputs(batch_size, hidden_dim, seq_len)
        reference_output = run_original_model(model, reference_input_sf, seq_len)
        if verbose:
            print(f"    Reference output shape: {reference_output.shape}")
    if verbose:
        print(f"    Hidden dim: {hidden_dim}")
        print(f"    Num layers: {config.num_hidden_layers}")
        print(f"    Num heads: {config.num_attention_heads}")
        print(f"    Num KV heads: {config.num_key_value_heads}")

    # Find target classes
    if verbose:
        print("Finding target classes...")
    attention_classes = find_target_classes(model, arch_config.attention_class_names)
    rmsnorm_classes = find_target_classes(model, arch_config.rmsnorm_class_names)
    if verbose:
        print(f"    Attention classes: {[c.__name__ for c in attention_classes]}")
        print(f"    RMSNorm classes: {[c.__name__ for c in rmsnorm_classes]}")

    return ModelContext(
        model=model,
        config=config,
        arch_config=arch_config,
        attention_classes=attention_classes,
        rmsnorm_classes=rmsnorm_classes,
        hidden_dim=hidden_dim,
        model_type=model_type,
        reference_output=reference_output,
        reference_input_cf=reference_input_cf,
    )


def patch_model_layers(
    model: nn.Module,
    attention_classes: tuple,
    rmsnorm_classes: tuple,
    config: object,
    verbose: bool = True,
    start_layer_idx: int = 0,
) -> None:
    """Apply all patches (Linear, RMSNorm, Attention) to a model.

    Args:
        model: Model or module to patch.
        attention_classes: Tuple of attention class types.
        rmsnorm_classes: Tuple of RMSNorm class types.
        config: Model configuration.
        verbose: Print patching info.
        start_layer_idx: Starting layer index for attention patching.
    """
    if verbose:
        print("Patching Linear layers...")
    with torch.inference_mode():
        patch_model_linears(model, verbose=verbose)

    if verbose:
        print("Patching RMSNorm layers...")
    with torch.inference_mode():
        patch_model_rmsnorms(model, target_classes=rmsnorm_classes, verbose=verbose)

    if verbose:
        print("Patching attention layers...")
    with torch.inference_mode():
        patch_model_attention(
            model,
            target_classes=attention_classes,
            config=config,
            verbose=verbose,
            start_layer_idx=start_layer_idx,
        )


# =============================================================================
# Verification Utilities
# =============================================================================


def compute_output_differences(
    pytorch_output: np.ndarray,
    coreml_output: np.ndarray,
    abs_threshold: float = 0.1,
    rel_threshold: float = 0.1,
) -> VerificationResult:
    """Compute difference statistics between PyTorch and CoreML outputs.

    Args:
        pytorch_output: Output from PyTorch model.
        coreml_output: Output from CoreML model.
        abs_threshold: Max absolute difference threshold for pass.
        rel_threshold: Mean relative difference threshold for pass.

    Returns:
        VerificationResult with statistics and pass/fail status.
    """
    abs_diff = np.abs(pytorch_output - coreml_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-7)

    return VerificationResult(
        abs_diff_max=float(abs_diff.max()),
        abs_diff_mean=float(abs_diff.mean()),
        abs_diff_std=float(abs_diff.std()),
        rel_diff_max=float(rel_diff.max()),
        rel_diff_mean=float(rel_diff.mean()),
        passed=abs_diff.max() < abs_threshold and rel_diff.mean() < rel_threshold,
    )


def print_verification_result(
    result: VerificationResult,
    pytorch_output: np.ndarray,
    coreml_output: np.ndarray,
    label: str = "",
    indent: str = "    ",
) -> None:
    """Print verification statistics."""
    if label:
        print(f"{indent}{label}")

    print(f"{indent}Absolute difference:")
    print(f"{indent}  Max:  {result.abs_diff_max:.6f}")
    print(f"{indent}  Mean: {result.abs_diff_mean:.6f}")
    print(f"{indent}  Std:  {result.abs_diff_std:.6f}")

    print(f"{indent}Relative difference:")
    print(f"{indent}  Max:  {result.rel_diff_max:.6f}")
    print(f"{indent}  Mean: {result.rel_diff_mean:.6f}")

    print(f"{indent}PyTorch sample: {pytorch_output.flatten()[:5]}")
    print(f"{indent}CoreML sample:  {coreml_output.flatten()[:5]}")

    if result.passed:
        print(f"{indent}[OK] Outputs match within tolerance!")
    else:
        print(f"{indent}[WARNING] Outputs differ significantly")


def run_original_model(
    original_model: nn.Module,
    test_input_sf: torch.Tensor,
    seq_len: int,
) -> np.ndarray:
    """Run original PyTorch model and return channels-first output."""
    with torch.inference_mode():
        output = original_model(
            inputs_embeds=test_input_sf,
            position_ids=torch.arange(seq_len).unsqueeze(0),
        )
        return output.last_hidden_state.transpose(1, 2).unsqueeze(2).numpy()


def run_coreml_model(
    mlmodel,
    test_input_cf: np.ndarray,
    position_id: int,
    state,
    input_name: str = "inputs_embeds",
) -> np.ndarray:
    """Run CoreML model and return output."""
    result = mlmodel.predict(
        {input_name: test_input_cf, "position_id": np.array([position_id], dtype=np.int32)},
        state,
    )
    return result["output"]


def create_test_inputs(
    batch_size: int,
    hidden_dim: int,
    seq_len: int,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Create test inputs in both formats.

    Returns:
        Tuple of (channels_first_numpy, channels_first_tensor, sequence_first_tensor)
    """
    test_input_cf = np.random.randn(batch_size, hidden_dim, 1, seq_len).astype(np.float32)
    test_input_tensor = torch.from_numpy(test_input_cf)
    test_input_sf = test_input_tensor.squeeze(2).transpose(1, 2)
    return test_input_cf, test_input_tensor, test_input_sf


# =============================================================================
# CoreML Conversion
# =============================================================================


def convert_to_coreml(
    traced_model,
    batch_size: int,
    hidden_dim: int,
    seq_len: int,
    state_specs: list,
    input_name: str = "inputs_embeds",
    package_dir: str | None = None,
    skip_model_load: bool = False,
):
    """Convert traced model to CoreML.

    Args:
        traced_model: Traced PyTorch model.
        batch_size: Batch size.
        hidden_dim: Hidden dimension.
        seq_len: Sequence length.
        state_specs: CoreML state specifications.
        input_name: Name for the input tensor.
        package_dir: If provided, save model directly to this path during conversion.
            Must end with .mlpackage. Avoids creating a temporary directory.
        skip_model_load: If True, skip loading the model after conversion.
            Useful for reducing memory or converting on older macOS versions.

    Returns:
        Converted CoreML model.
    """
    return ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                shape=(batch_size, hidden_dim, 1, seq_len),
                name=input_name,
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
        states=state_specs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        package_dir=package_dir,
        skip_model_load=skip_model_load,
    )


def run_model_analysis(
    mlmodel,
    run_compute_plan: bool = False,
    run_mil_inspection: bool = False,
) -> None:
    """Run optional model analysis."""
    if run_compute_plan:
        print("\nCompute Plan Analysis")
        print("Looking for Neural Engine scheduling...")
        analyze_compute_plan(mlmodel)

    if run_mil_inspection:
        print("\nMIL Program Inspection")
        print("Looking for conv, layer_norm operations...")
        inspect_mil_program(mlmodel)


# =============================================================================
# Single Model Conversion
# =============================================================================


def convert_language_model(
    model_name: str,
    seq_len: int = 8,
    cache_length: int = 2048,
    batch_size: int = 1,
    num_layers: int | None = None,
    output_path: str | None = None,
    verbose: bool = True,
    analyze_compute: bool = False,
    analyze_mil: bool = False,
    skip_model_load: bool = False,
    cache_compiled: bool = False,
):
    """Convert a HuggingFace language model to CoreML.

    Args:
        model_name: HuggingFace model name.
        seq_len: Sequence length for tracing.
        cache_length: Maximum KV cache length.
        batch_size: Batch size.
        num_layers: Number of layers to keep (for debugging).
        output_path: Output path for CoreML model.
        verbose: Print detailed information.
        analyze_compute: Run compute plan analysis.
        analyze_mil: Run MIL program inspection.
        skip_model_load: Skip loading model after conversion to reduce memory.
        cache_compiled: Cache compiled model (.mlmodelc) for faster subsequent loads.

    Returns:
        Converted CoreML model (None if skip_model_load is True).
    """
    print("CoreML Language Model Conversion")
    print("=" * 60)
    if skip_model_load:
        print("Model loading after conversion will be skipped (memory optimization)")

    # Load model
    print("\n[1] Loading model and config...")
    ctx = load_model_and_config(
        model_name,
        num_layers,
        verbose,
        compute_reference=not skip_model_load,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    # Patch layers
    print("\n[2] Patching model layers...")
    patch_model_layers(
        ctx.model,
        ctx.attention_classes,
        ctx.rmsnorm_classes,
        ctx.config,
        verbose,
    )

    # Wrap model
    print("\n[3] Creating LanguageModelWrapper...")
    with torch.inference_mode():
        wrapped_model = LanguageModelWrapper(
            ctx.model,
            cache_length=cache_length,
            channels_first=True,
            device="cpu",
        )
        wrapped_model.eval()
    print(f"    {wrapped_model}")

    # Trace model
    print("\n[4] Tracing model...")
    example_inputs = (
        torch.randn((batch_size, ctx.hidden_dim, 1, seq_len), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.int32),
    )
    with torch.inference_mode():
        traced_model = torch.jit.trace(wrapped_model, example_inputs)
    print("    Tracing complete!")

    # Determine output path
    if output_path is None:
        model_short_name = model_name.split("/")[-1].lower().replace("-", "_")
        output_path = f"{model_short_name}_seqlen_{seq_len}.mlpackage"

    # Convert to CoreML (saves directly to package_dir)
    print("\n[5] Converting to CoreML...")
    register_extended_passes()

    mlmodel = convert_to_coreml(
        traced_model,
        batch_size,
        ctx.hidden_dim,
        seq_len,
        create_coreml_state_specs(wrapped_model),
        input_name="inputs_embeds",
        package_dir=output_path,
        skip_model_load=skip_model_load,
    )
    print(f"    Saved to: {output_path}")

    # Cache compiled model if requested
    if cache_compiled and not skip_model_load and mlmodel is not None:
        print("\n[5.5] Caching compiled model...")
        cache_compiled_model(mlmodel, Path(output_path), verbose)

    # Verification
    print("\n[6] Verifying outputs...")
    if skip_model_load:
        print("    Skipped (--skip-model-load flag set, model not loaded)")
    elif ctx.reference_output is None:
        print("    Skipped (reference output not available)")
    else:
        # Use pre-computed reference output from before patching
        pytorch_output = ctx.reference_output
        test_input_cf = ctx.reference_input_cf

        state = mlmodel.make_state()
        coreml_output = run_coreml_model(mlmodel, test_input_cf, 0, state)

        print(f"    PyTorch output shape: {pytorch_output.shape}")
        print(f"    CoreML output shape: {coreml_output.shape}")

        result = compute_output_differences(pytorch_output, coreml_output)
        print_verification_result(result, pytorch_output, coreml_output, indent="    ")

        # Test stateful inference
        print("\n[7] Testing stateful inference (second forward pass)...")
        output2 = run_coreml_model(mlmodel, test_input_cf, seq_len, state)
        print(f"    Second output shape: {output2.shape}")
        print(f"    Second output sample: {output2.flatten()[:5]}")

    # Optional analysis (only if model is loaded)
    if not skip_model_load:
        run_model_analysis(mlmodel, analyze_compute, analyze_mil)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Model saved to: {output_path}")
    print(f"Architecture: {ctx.model_type}")
    print(f"QK-norm enabled: {ctx.arch_config.has_qk_norm}")

    return mlmodel


# =============================================================================
# Chunked Model Conversion
# =============================================================================


def calculate_chunk_ranges(total_layers: int, num_chunks: int) -> list:
    """Calculate layer ranges for each chunk.

    Returns:
        List of (start, end) tuples for each chunk.
    """
    layers_per_chunk = math.ceil(total_layers / num_chunks)
    chunk_ranges = []
    for i in range(num_chunks):
        start = i * layers_per_chunk
        end = min((i + 1) * layers_per_chunk, total_layers)
        if start < total_layers:
            chunk_ranges.append((start, end))
    return chunk_ranges


def convert_single_chunk(
    chunk_idx: int,
    start_layer: int,
    end_layer: int,
    model: nn.Module,
    config: object,
    attention_classes: tuple,
    rmsnorm_classes: tuple,
    cache_length: int,
    cos_emb: torch.Tensor,
    sin_emb: torch.Tensor,
    batch_size: int,
    hidden_dim: int,
    seq_len: int,
    is_last_chunk: bool,
    output_path: Path,
    skip_model_load: bool = False,
    cache_compiled: bool = False,
):
    """Convert a single chunk to CoreML.

    Args:
        skip_model_load: Skip loading model after conversion to reduce memory.
        cache_compiled: Cache compiled model (.mlmodelc) for faster subsequent loads.

    Returns:
        Tuple of (chunk_mlmodel, chunk_wrapper). chunk_mlmodel is None if skip_model_load is True.
    """
    print(f"\n    --- Chunk {chunk_idx} (layers {start_layer}-{end_layer-1}) ---")

    # Extract and copy layers
    chunk_layers = [copy.deepcopy(layer) for layer in model.layers[start_layer:end_layer]]
    chunk_module = nn.ModuleList(chunk_layers)

    # Patch layers
    print("    Patching layers...")
    patch_model_layers(
        chunk_module,
        attention_classes,
        rmsnorm_classes,
        config,
        verbose=False,
        start_layer_idx=0,
    )

    # Get final norm for last chunk
    final_norm = None
    if is_last_chunk:
        original_norm = copy.deepcopy(model.norm)
        # Directly wrap the final norm with patcher (patch_model_rmsnorms only patches children)
        if isinstance(original_norm, rmsnorm_classes):
            final_norm = RMSNormToLayerNormPatcher(original_norm)
        else:
            final_norm = original_norm

    # Create wrapper
    print("    Creating ChunkedLanguageModelWrapper...")
    with torch.inference_mode():
        chunk_wrapper = ChunkedLanguageModelWrapper(
            layers=list(chunk_module),
            config=config,
            chunk_idx=chunk_idx,
            cache_length=cache_length,
            cos_emb=cos_emb,
            sin_emb=sin_emb,
            channels_first=True,
            is_first_chunk=(chunk_idx == 0),
            is_last_chunk=is_last_chunk,
            final_norm=final_norm,
            device="cpu",
        )
        chunk_wrapper.eval()
    print(f"    {chunk_wrapper}")

    # Trace
    print("    Tracing chunk...")
    example_inputs = (
        torch.randn((batch_size, hidden_dim, 1, seq_len), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.int32),
    )
    with torch.inference_mode():
        traced_chunk = torch.jit.trace(chunk_wrapper, example_inputs)

    # Convert and save directly using package_dir
    chunk_path = output_path / f"chunk_{chunk_idx}.mlpackage"
    print("    Converting to CoreML...")
    chunk_mlmodel = convert_to_coreml(
        traced_chunk,
        batch_size,
        hidden_dim,
        seq_len,
        create_chunked_coreml_state_specs(chunk_wrapper),
        input_name="hidden_states",
        package_dir=str(chunk_path),
        skip_model_load=skip_model_load,
    )
    print(f"    Saved to: {chunk_path}")

    # Cache compiled model if requested
    if cache_compiled and not skip_model_load and chunk_mlmodel is not None:
        cache_compiled_model(chunk_mlmodel, chunk_path, verbose=True)

    return chunk_mlmodel, chunk_wrapper


def verify_chunked_model(
    chunk_models: list,
    reference_output: np.ndarray,
    reference_input_cf: np.ndarray,
    seq_len: int,
) -> VerificationResult:
    """Verify chunked model outputs against pre-computed reference.

    This performs end-to-end verification by:
    1. Using the pre-computed reference output from the original model
    2. Chaining all CoreML chunks together with channels-first input
    3. Comparing the final outputs

    Returns:
        VerificationResult with comparison statistics.
    """
    # End-to-end verification: compare full original model vs chained CoreML chunks
    print("\n    End-to-end verification (Original PyTorch vs Chained CoreML Chunks)...")

    # Path 1: Use pre-computed reference output from original model
    pytorch_output = reference_output

    # Path 2: Chain all CoreML chunks together
    # Input/Output: channels-first (batch, hidden_dim, 1, seq_len)
    chunk_states = [model.make_state() for model in chunk_models]
    current_hidden = reference_input_cf

    for chunk_idx, (chunk_model, chunk_state) in enumerate(zip(chunk_models, chunk_states)):
        result = chunk_model.predict(
            {"hidden_states": current_hidden, "position_id": np.array([0], dtype=np.int32)},
            chunk_state,
        )
        current_hidden = result["output"]

    chained_output = current_hidden

    print(f"    PyTorch output shape: {pytorch_output.shape}")
    print(f"    Chained CoreML output shape: {chained_output.shape}")

    e2e_result = compute_output_differences(pytorch_output, chained_output, 0.5, 0.2)
    print_verification_result(e2e_result, pytorch_output, chained_output, indent="    ")

    # Test stateful inference (second forward pass to verify KV cache works)
    print("\n    Testing stateful inference (second forward pass across all chunks)...")
    current_hidden = reference_input_cf
    for chunk_model, chunk_state in zip(chunk_models, chunk_states):
        result = chunk_model.predict(
            {"hidden_states": current_hidden, "position_id": np.array([seq_len], dtype=np.int32)},
            chunk_state,
        )
        current_hidden = result["output"]

    print(f"    Second pass output shape: {current_hidden.shape}")
    print(f"    Second pass output sample: {current_hidden.flatten()[:5]}")

    return e2e_result


def convert_chunked_language_model(
    model_name: str,
    num_chunks: int,
    seq_len: int = 8,
    cache_length: int = 2048,
    batch_size: int = 1,
    num_layers: int | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
    analyze_compute: bool = False,
    analyze_mil: bool = False,
    chunk_indices: list[int] | None = None,
    skip_verification: bool = False,
    skip_model_load: bool = False,
    cache_compiled: bool = False,
):
    """Convert a HuggingFace language model to multiple CoreML chunks.

    Args:
        model_name: HuggingFace model name.
        num_chunks: Number of chunks to split the model into.
        seq_len: Sequence length for tracing.
        cache_length: Maximum KV cache length.
        batch_size: Batch size.
        num_layers: Number of layers to keep (for debugging).
        output_dir: Directory for output chunks.
        verbose: Print detailed information.
        analyze_compute: Run compute plan analysis on each chunk.
        analyze_mil: Run MIL program inspection on each chunk.
        chunk_indices: List of specific chunk indices to convert. If None, converts all.
        skip_verification: Skip output verification to reduce memory usage.
        skip_model_load: Skip loading model after conversion to reduce memory.
        cache_compiled: Cache compiled models (.mlmodelc) for faster subsequent loads.

    Returns:
        List of converted CoreML models (empty if skip_model_load is True).
    """
    print("CoreML Chunked Language Model Conversion")
    print("=" * 60)
    print(f"Splitting model into {num_chunks} chunks")
    if chunk_indices:
        print(f"Converting only chunk(s): {chunk_indices}")
    if skip_verification:
        print("Verification will be skipped (memory optimization)")
    if skip_model_load:
        print("Model loading after conversion will be skipped (memory optimization)")

    # Determine if we should compute reference output
    # Skip if: explicit skip_verification, or converting partial chunks, or skip_model_load
    converting_all_chunks = chunk_indices is None or len(chunk_indices) == num_chunks
    should_compute_reference = not skip_verification and converting_all_chunks and not skip_model_load

    # Load model
    print("\n[1] Loading model and config...")
    ctx = load_model_and_config(
        model_name,
        num_layers,
        verbose,
        compute_reference=should_compute_reference,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    total_layers = ctx.config.num_hidden_layers

    # Calculate chunk distribution
    print("\n[2] Calculating layer distribution...")
    chunk_ranges = calculate_chunk_ranges(total_layers, num_chunks)

    actual_num_chunks = len(chunk_ranges)
    if actual_num_chunks != num_chunks:
        print(f"    Note: Adjusted to {actual_num_chunks} chunks")
        num_chunks = actual_num_chunks

    for i, (start, end) in enumerate(chunk_ranges):
        print(f"    Chunk {i}: layers {start}-{end-1} ({end - start} layers)")

    # Pre-compute position embeddings
    print("\n[3] Pre-computing position embeddings...")
    position_ids = torch.arange(cache_length, dtype=torch.long).unsqueeze(0)
    dummy_values = torch.ones(1, dtype=torch.float32)
    cos_emb, sin_emb = ctx.model.rotary_emb(dummy_values, position_ids)
    cos_emb = cos_emb[0]
    sin_emb = sin_emb[0]
    print(f"    Position embedding shape: {cos_emb.shape}")

    # Register graph passes
    register_extended_passes()

    # Setup output directory
    if output_dir is None:
        model_short_name = model_name.split("/")[-1].lower().replace("-", "_")
        output_dir = f"{model_short_name}_chunked_{num_chunks}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine which chunks to convert
    indices_to_convert = chunk_indices if chunk_indices else list(range(num_chunks))
    # Validate indices
    for idx in indices_to_convert:
        if idx < 0 or idx >= num_chunks:
            raise ValueError(f"Invalid chunk index {idx}. Must be between 0 and {num_chunks - 1}")

    # Convert each chunk
    print(f"\n[4] Converting {len(indices_to_convert)} chunk(s)...")
    chunk_models = []

    for chunk_idx in indices_to_convert:
        start_layer, end_layer = chunk_ranges[chunk_idx]
        is_last_chunk = (chunk_idx == num_chunks - 1)

        chunk_mlmodel, _ = convert_single_chunk(
            chunk_idx,
            start_layer,
            end_layer,
            ctx.model,
            ctx.config,
            ctx.attention_classes,
            ctx.rmsnorm_classes,
            cache_length,
            cos_emb,
            sin_emb,
            batch_size,
            ctx.hidden_dim,
            seq_len,
            is_last_chunk,
            output_path,
            skip_model_load=skip_model_load,
            cache_compiled=cache_compiled,
        )
        if chunk_mlmodel is not None:
            chunk_models.append(chunk_mlmodel)

        # Optional analysis per chunk
        if analyze_compute or analyze_mil:
            print(f"\n    Analyzing chunk {chunk_idx}...")
            run_model_analysis(chunk_mlmodel, analyze_compute, analyze_mil)

        # Memory cleanup after each chunk
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Verification
    print("\n" + "=" * 60)
    print("[5] VERIFICATION")
    print("=" * 60)

    e2e_result = None
    if skip_verification:
        print("    Skipped (--skip-verification flag set)")
    elif skip_model_load:
        print("    Skipped (--skip-model-load flag set, models not loaded)")
    elif chunk_indices and len(chunk_indices) != num_chunks:
        print("    Skipped (not all chunks were converted)")
        print("    Run without --chunk-index to verify all chunks together")
    elif ctx.reference_output is None:
        print("    Skipped (reference output not available)")
    elif not chunk_models:
        print("    Skipped (no models available for verification)")
    else:
        e2e_result = verify_chunked_model(
            chunk_models,
            ctx.reference_output,
            ctx.reference_input_cf,
            seq_len,
        )

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Architecture: {ctx.model_type}")
    print(f"Total layers: {total_layers}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Chunks converted: {indices_to_convert}")
    print(f"Output directory: {output_dir}")
    print("\nChunk files created:")
    for i in indices_to_convert:
        print(f"  - chunk_{i}.mlpackage")
    if e2e_result is not None:
        print(f"\nEnd-to-end verification: {'PASSED' if e2e_result.passed else 'FAILED'}")
    else:
        print("\nEnd-to-end verification: SKIPPED")

    return chunk_models


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace language models to CoreML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert Qwen2 model (default, single model)
    python examples/lm_conversion_example.py

    # Convert Qwen3 model (single model)
    python examples/lm_conversion_example.py --model Qwen/Qwen3-0.6B

    # Convert large model in 4 chunks (for models > 2GB)
    python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4

    # Convert only specific chunks to reduce memory usage
    python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 0
    python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 1,2

    # Minimum memory mode: one chunk at a time, skip model load
    python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 0 --skip-model-load

    # Quick debug with only 8 layers split into 2 chunks
    python examples/lm_conversion_example.py --num-layers 8 --num-chunks 2

    # Export only embeddings and LM head (skip model conversion for faster development)
    python examples/lm_conversion_example.py --export-embeddings --export-lm-head --components-only

    # With analysis enabled
    python examples/lm_conversion_example.py --analyze-compute-plan --analyze-mil
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
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers to keep (default: all). Useful for faster debugging.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Number of chunks to split the model into (default: 1, no chunking). "
        "Use for large models (>2GB) that exceed Neural Engine limits.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the CoreML model (default: auto-generated). "
        "For chunked models, this is the output directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity",
    )
    parser.add_argument(
        "--analyze-compute-plan",
        action="store_true",
        help="Run compute plan analysis to check Neural Engine scheduling",
    )
    parser.add_argument(
        "--analyze-mil",
        action="store_true",
        help="Run MIL program inspection to check for conv, layer_norm operations",
    )
    parser.add_argument(
        "--export-embeddings",
        action="store_true",
        help="Export embedding weights as .npy file in float16 format",
    )
    parser.add_argument(
        "--export-lm-head",
        action="store_true",
        help="Export LM head as CoreML model with vocabulary chunking for Neural Engine compatibility",
    )
    parser.add_argument(
        "--lm-head-chunk-size",
        type=int,
        default=6144,
        help="Chunk size for LM head vocabulary dimension (default: 6144). "
        "Smaller values use less memory but create more chunks.",
    )
    parser.add_argument(
        "--lm-head-no-logsumexp",
        action="store_true",
        help="Disable log-sum-exp computation in LM head (only output logits)",
    )
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="Skip main model conversion and only export embeddings/LM head. "
        "Must be used with --export-embeddings and/or --export-lm-head.",
    )
    parser.add_argument(
        "--chunk-index",
        type=str,
        default=None,
        help="Convert only specific chunk(s). Can be a single index (e.g., '0') or "
        "comma-separated indices (e.g., '0,1,2'). Use with --num-chunks to specify "
        "total chunks. This reduces memory usage by not loading chunks you don't need.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip output verification step. This reduces memory usage by not keeping "
        "a copy of the original model for comparison.",
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip loading the model after conversion. Reduces memory usage and allows "
        "converting newer model formats on older macOS versions. The model can still be "
        "saved but cannot be used for prediction in the same session.",
    )
    parser.add_argument(
        "--cache-compiled",
        action="store_true",
        help="Cache compiled models (.mlmodelc) alongside .mlpackage files for faster "
        "subsequent loads. The compiled model is saved in the same directory with "
        ".mlmodelc extension. Requires model loading (incompatible with --skip-model-load).",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.components_only and not (args.export_embeddings or args.export_lm_head):
        parser.error("--components-only requires --export-embeddings and/or --export-lm-head")

    # Parse chunk indices if provided
    chunk_indices = None
    if args.chunk_index:
        try:
            chunk_indices = [int(x.strip()) for x in args.chunk_index.split(",")]
        except ValueError:
            parser.error(f"Invalid --chunk-index value: {args.chunk_index}. Must be integers separated by commas.")

    # Validate cache_compiled with skip_model_load
    if args.cache_compiled and args.skip_model_load:
        parser.error("--cache-compiled cannot be used with --skip-model-load (model must be loaded to cache)")

    # Convert main model (unless components-only mode)
    if not args.components_only and args.num_chunks > 1:
        convert_chunked_language_model(
            model_name=args.model,
            num_chunks=args.num_chunks,
            seq_len=args.seq_len,
            cache_length=args.cache_length,
            num_layers=args.num_layers,
            output_dir=args.output,
            verbose=not args.quiet,
            analyze_compute=args.analyze_compute_plan,
            analyze_mil=args.analyze_mil,
            chunk_indices=chunk_indices,
            skip_verification=args.skip_verification,
            skip_model_load=args.skip_model_load,
            cache_compiled=args.cache_compiled,
        )
    elif not args.components_only:
        convert_language_model(
            model_name=args.model,
            seq_len=args.seq_len,
            cache_length=args.cache_length,
            num_layers=args.num_layers,
            output_path=args.output,
            verbose=not args.quiet,
            analyze_compute=args.analyze_compute_plan,
            analyze_mil=args.analyze_mil,
            skip_model_load=args.skip_model_load,
            cache_compiled=args.cache_compiled,
        )

    # Export embeddings and/or LM head if requested
    if args.export_embeddings or args.export_lm_head:
        print("\n" + "=" * 60)
        print("EXPORTING ADDITIONAL COMPONENTS")
        print("=" * 60)

        # Load model if needed (for embeddings/LM head export)
        if not args.quiet:
            print(f"\nLoading model: {args.model}")
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()

        # Optionally truncate layers (for consistency with main conversion)
        if args.num_layers is not None and args.num_layers < config.num_hidden_layers:
            if not args.quiet:
                print(f"Truncating from {config.num_hidden_layers} to {args.num_layers} layers...")
            model.layers = model.layers[:args.num_layers]
            config.num_hidden_layers = args.num_layers

        # Determine output directory
        if args.output:
            if args.num_chunks > 1:
                output_dir = Path(args.output)
            else:
                output_dir = Path(args.output).parent
        else:
            model_short_name = args.model.split("/")[-1].lower().replace("-", "_")
            if args.num_chunks > 1:
                output_dir = Path(f"{model_short_name}_chunked_{args.num_chunks}")
            else:
                output_dir = Path(".")

        # Export embeddings
        if args.export_embeddings:
            print("\n[1] Exporting embeddings...")
            embeddings_path = output_dir / "embeddings.npy"
            export_embeddings(model, embeddings_path, verbose=not args.quiet)

        # Export LM head
        if args.export_lm_head:
            print("\n[2] Exporting LM head...")
            lm_head_path = output_dir / "lm_head.mlpackage"

            # Load causal LM model to get lm_head
            if not args.quiet:
                print("  Loading causal LM model for lm_head...")

            try:
                causal_model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
                causal_model.eval()

                if hasattr(causal_model, "lm_head"):
                    lm_head = causal_model.lm_head
                else:
                    print("  [WARNING] Could not find lm_head in model. Skipping LM head export.")
                    lm_head = None

                if lm_head is not None:
                    lm_head_mlmodel = convert_lm_head(
                        lm_head=lm_head,
                        batch_size=1,
                        hidden_dim=config.hidden_size,
                        seq_len=args.seq_len,
                        output_path=lm_head_path,
                        chunk_size=args.lm_head_chunk_size,
                        compute_logsumexp=not args.lm_head_no_logsumexp,
                        verbose=not args.quiet,
                        analyze_compute=args.analyze_compute_plan,
                        analyze_mil=args.analyze_mil,
                    )
                    # Cache compiled model if requested
                    if args.cache_compiled and lm_head_mlmodel is not None:
                        cache_compiled_model(lm_head_mlmodel, lm_head_path, verbose=not args.quiet)
            except Exception as e:
                print(f"  [ERROR] Failed to load or convert LM head: {e}")

        print("\n" + "=" * 60)
        print("EXPORT COMPLETE")
        print("=" * 60)
        if args.export_embeddings:
            print(f"Embeddings: {output_dir / 'embeddings.npy'}")
        if args.export_lm_head:
            print(f"LM head: {output_dir / 'lm_head.mlpackage'}")


if __name__ == "__main__":
    main()
