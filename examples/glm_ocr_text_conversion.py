"""Example: Converting GLM-OCR Text Decoder to CoreML with KV Cache.

This script converts the GLM-OCR text decoder (language model) to CoreML.
The full GLM-OCR model is loaded, then the text model is extracted, patched,
wrapped, traced, and converted.

Key differences from standard LM conversion (lm_conversion_example.py):
- Uses GlmOcrLanguageModelWrapper instead of LanguageModelWrapper
  (computes standard 1D RoPE, bypassing model's 3D mRoPE)
- Uses patch_glm_ocr_text_layers instead of patch_model_attention
  (handles 4-norm decoder layers, fused gate_up_proj MLP,
   interleaved→split-half RoPE weight permutation)

Usage:
    # Quick debug with 2 layers
    uv run python examples/glm_ocr_text_conversion.py --num-layers 2

    # Full conversion
    uv run python examples/glm_ocr_text_conversion.py

    # With custom output path
    uv run python examples/glm_ocr_text_conversion.py --output glm_ocr_text.mlpackage

    # Export embeddings and LM head
    uv run python examples/glm_ocr_text_conversion.py --export-embeddings --export-lm-head

    # Export components only (skip main text model conversion)
    uv run python examples/glm_ocr_text_conversion.py --components-only --export-lm-head

    # Skip model load (save-only, useful for older macOS)
    uv run python examples/glm_ocr_text_conversion.py --skip-model-load
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText

from coremlmodels import (
    GlmOcrLanguageModelWrapper,
    convert_lm_head,
    create_glm_ocr_state_specs,
    export_embeddings,
    find_target_classes,
    get_architecture_config,
    patch_glm_ocr_text_layers,
    patch_model_linears,
    patch_model_rmsnorms,
    register_extended_passes,
)


def load_glm_ocr_text_model(
    model_name: str,
    num_layers: int | None = None,
    verbose: bool = True,
):
    """Load GLM-OCR model and extract text model + config.

    Returns:
        Tuple of (text_model, text_config, full_model) where full_model
        is kept alive to prevent garbage collection of shared tensors.
    """
    if verbose:
        print(f"Loading config: {model_name}")

    config = AutoConfig.from_pretrained(model_name)

    if verbose:
        print(f"    Model type: {config.model_type}")
        print(f"    Text config type: {config.text_config.model_type}")

    if verbose:
        print(f"Loading full model: {model_name}")

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Extract text model
    text_model = model.language_model
    text_config = config.text_config

    # Truncate layers if requested
    if num_layers is not None and num_layers < text_config.num_hidden_layers:
        if verbose:
            print(
                f"    Truncating from {text_config.num_hidden_layers} to {num_layers} layers..."
            )
        text_model.layers = text_model.layers[:num_layers]
        text_config.num_hidden_layers = num_layers

    if verbose:
        print(f"    Hidden dim: {text_config.hidden_size}")
        print(f"    Num layers: {text_config.num_hidden_layers}")
        print(f"    Num heads: {text_config.num_attention_heads}")
        print(f"    Num KV heads: {text_config.num_key_value_heads}")
        head_dim = (
            getattr(text_config, "head_dim", None)
            or text_config.hidden_size // text_config.num_attention_heads
        )
        print(f"    Head dim: {head_dim}")
        print(f"    Intermediate size: {text_config.intermediate_size}")
        print(f"    RoPE theta: {text_config.rope_parameters.get('rope_theta', 'N/A')}")

    return text_model, text_config, model


def compute_reference_output(
    text_model: nn.Module,
    hidden_dim: int,
    batch_size: int,
    seq_len: int,
    verbose: bool = True,
):
    """Compute reference output from the original (unpatched) text model.

    Returns:
        Tuple of (reference_output_cf_numpy, test_input_cf_numpy)
        where cf = channels-first format.
    """
    if verbose:
        print("    Computing reference output before patching...")

    test_input_cf = np.random.randn(batch_size, hidden_dim, 1, seq_len).astype(
        np.float32
    )
    test_input_sf = torch.from_numpy(test_input_cf).squeeze(2).transpose(1, 2)

    with torch.inference_mode():
        output = text_model(
            inputs_embeds=test_input_sf,
            position_ids=torch.arange(seq_len).unsqueeze(0),
        )
        # Convert to channels-first
        ref_output = output.last_hidden_state.transpose(1, 2).unsqueeze(2).numpy()

    if verbose:
        print(f"    Reference output shape: {ref_output.shape}")

    return ref_output, test_input_cf


def patch_text_model(
    text_model: nn.Module,
    text_config: object,
    verbose: bool = True,
):
    """Apply all patches to GLM-OCR text model."""
    # Find target classes for RMSNorm
    arch_config = get_architecture_config("glm_ocr")
    rmsnorm_classes = find_target_classes(text_model, arch_config.rmsnorm_class_names)

    if verbose:
        print(f"    RMSNorm classes found: {[c.__name__ for c in rmsnorm_classes]}")

    # Patch Linear → Conv2d
    if verbose:
        print("Patching Linear layers...")
    with torch.inference_mode():
        patch_model_linears(text_model, verbose=verbose)

    # Patch RMSNorm → LayerNorm-fusable
    if verbose:
        print("Patching RMSNorm layers...")
    with torch.inference_mode():
        patch_model_rmsnorms(
            text_model, target_classes=rmsnorm_classes, verbose=verbose
        )

    # Patch GLM-OCR-specific layers (attention, MLP, decoder layers)
    if verbose:
        print("Patching GLM-OCR text layers...")
    with torch.inference_mode():
        patch_glm_ocr_text_layers(text_model, text_config, verbose=verbose)


def convert_glm_ocr_text(
    model_name: str,
    seq_len: int = 8,
    cache_length: int = 2048,
    batch_size: int = 1,
    num_layers: int | None = None,
    output_path: str | None = None,
    verbose: bool = True,
    skip_model_load: bool = False,
    overwrite: bool = False,
):
    """Convert GLM-OCR text decoder to CoreML.

    Returns:
        Converted CoreML model (None if skip_model_load is True).
    """
    print("GLM-OCR Text Model CoreML Conversion")
    print("=" * 60)

    # Load model
    print("\n[1] Loading model and config...")
    text_model, text_config, full_model = load_glm_ocr_text_model(
        model_name,
        num_layers,
        verbose,
    )

    hidden_dim = text_config.hidden_size

    # Compute reference output before patching
    reference_output = None
    reference_input_cf = None
    if not skip_model_load:
        reference_output, reference_input_cf = compute_reference_output(
            text_model,
            hidden_dim,
            batch_size,
            seq_len,
            verbose,
        )

    # Patch model
    print("\n[2] Patching model layers...")
    patch_text_model(text_model, text_config, verbose)

    # Wrap model
    print("\n[3] Creating GlmOcrLanguageModelWrapper...")
    with torch.inference_mode():
        wrapped_model = GlmOcrLanguageModelWrapper(
            text_model,
            config=text_config,
            cache_length=cache_length,
            channels_first=True,
            device="cpu",
        )
        wrapped_model.eval()
    print(f"    {wrapped_model}")

    # Trace model
    print("\n[4] Tracing model...")
    example_inputs = (
        torch.randn((batch_size, hidden_dim, 1, seq_len), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.int32),
    )
    with torch.inference_mode():
        traced_model = torch.jit.trace(wrapped_model, example_inputs)
    print("    Tracing complete!")

    # Determine output path
    if output_path is None:
        model_short_name = model_name.split("/")[-1].lower().replace("-", "_")
        output_path = f"{model_short_name}_text_seqlen_{seq_len}.mlpackage"

    # Check overwrite
    output_p = Path(output_path)
    if output_p.exists() and not overwrite:
        print(f"\n    Output already exists: {output_path}")
        print("    Use --overwrite to replace")
        return None

    # Convert to CoreML
    print("\n[5] Converting to CoreML...")
    register_extended_passes()

    state_specs = create_glm_ocr_state_specs(wrapped_model)
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
        states=state_specs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        package_dir=output_path,
        skip_model_load=skip_model_load,
    )
    print(f"    Saved to: {output_path}")

    # Verification
    print("\n[6] Verifying outputs...")
    if skip_model_load:
        print("    Skipped (--skip-model-load flag set)")
    elif reference_output is None:
        print("    Skipped (reference output not available)")
    else:
        state = mlmodel.make_state()
        coreml_output = mlmodel.predict(
            {
                "inputs_embeds": reference_input_cf,
                "position_id": np.array([0], dtype=np.int32),
            },
            state,
        )["output"]

        print(f"    PyTorch output shape: {reference_output.shape}")
        print(f"    CoreML output shape: {coreml_output.shape}")

        abs_diff = np.abs(reference_output - coreml_output)
        rel_diff = abs_diff / (np.abs(reference_output) + 1e-7)

        print(f"    Absolute difference:")
        print(f"      Max:  {abs_diff.max():.6f}")
        print(f"      Mean: {abs_diff.mean():.6f}")
        print(f"    Relative difference:")
        print(f"      Max:  {rel_diff.max():.6f}")
        print(f"      Mean: {rel_diff.mean():.6f}")
        print(f"    PyTorch sample: {reference_output.flatten()[:5]}")
        print(f"    CoreML sample:  {coreml_output.flatten()[:5]}")

        if abs_diff.max() < 0.1:
            print("    [OK] Outputs match within tolerance!")
        else:
            print("    [WARNING] Outputs differ significantly")

        # Test stateful inference (KV cache)
        print("\n[7] Testing stateful inference (second forward pass)...")
        output2 = mlmodel.predict(
            {
                "inputs_embeds": reference_input_cf,
                "position_id": np.array([seq_len], dtype=np.int32),
            },
            state,
        )["output"]
        print(f"    Second output shape: {output2.shape}")
        print(f"    Second output sample: {output2.flatten()[:5]}")

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Model saved to: {output_path}")
    print(f"Architecture: GLM-OCR Text")
    print(f"Layers: {text_config.num_hidden_layers}")

    return mlmodel


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLM-OCR text decoder to CoreML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="zai-org/GLM-OCR",
        help="HuggingFace model name (default: glm-org/GLM-OCR)",
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
        help="Number of transformer layers to keep (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the CoreML model",
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip loading the model after conversion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    parser.add_argument(
        "--export-embeddings",
        action="store_true",
        help="Export embedding weights as .npy file",
    )
    parser.add_argument(
        "--export-lm-head",
        action="store_true",
        help="Export LM head as CoreML model",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity",
    )
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="Only export components (--export-embeddings/--export-lm-head), "
        "skip main text model conversion",
    )

    args = parser.parse_args()

    if args.components_only and not (args.export_embeddings or args.export_lm_head):
        parser.error(
            "--components-only requires --export-embeddings and/or --export-lm-head"
        )

    # Convert text model unless user requested component-only export
    if not args.components_only:
        convert_glm_ocr_text(
            model_name=args.model,
            seq_len=args.seq_len,
            cache_length=args.cache_length,
            num_layers=args.num_layers,
            output_path=args.output,
            verbose=not args.quiet,
            skip_model_load=args.skip_model_load,
            overwrite=args.overwrite,
        )
    elif not args.quiet:
        print("Skipping main text model conversion (--components-only enabled).")

    # Export additional components
    if args.export_embeddings or args.export_lm_head:
        print("\n" + "=" * 60)
        print("EXPORTING ADDITIONAL COMPONENTS")
        print("=" * 60)

        config = AutoConfig.from_pretrained(args.model)
        text_config = config.text_config

        # Determine output directory
        if args.output:
            output_dir = Path(args.output).parent
        else:
            output_dir = Path(".")

        if args.export_embeddings:
            print("\nExporting embeddings...")
            model = AutoModel.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            model.eval()
            text_model = model.language_model
            embeddings_path = output_dir / "glm_ocr_embeddings.npy"
            export_embeddings(text_model, embeddings_path, verbose=not args.quiet)

        if args.export_lm_head:
            print("\nExporting LM head...")
            lm_head_path = output_dir / "glm_ocr_lm_head.mlpackage"

            image_text_model = AutoModelForImageTextToText.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            image_text_model.eval()

            if hasattr(image_text_model, "lm_head"):
                convert_lm_head(
                    lm_head=image_text_model.lm_head,
                    batch_size=1,
                    hidden_dim=text_config.hidden_size,
                    seq_len=args.seq_len,
                    output_path=lm_head_path,
                    verbose=not args.quiet,
                )
            else:
                print("  [WARNING] Could not find lm_head in model")


if __name__ == "__main__":
    main()
