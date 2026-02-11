"""Example: Converting GLM-OCR Vision Module to CoreML.

This script converts the GLM-OCR vision encoder to CoreML with:
- Patch-level padding (not spatial) for flexible input sizes
- 32 enumerated patch counts from 32 to 16,384
- 2D rotary position embeddings pre-computed on host
- Attention masking for padded patches
- No CoreML state needed (single forward pass)

Usage:
    # Convert with default settings (2 layers for debug)
    uv run python examples/vision_conversion_example.py --num-layers 2

    # Convert all layers
    uv run python examples/vision_conversion_example.py

    # Specify number of patches for tracing
    uv run python examples/vision_conversion_example.py --num-patches 1024

    # Skip loading (faster, saves memory)
    uv run python examples/vision_conversion_example.py --skip-model-load
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coremlmodels import (
    register_extended_passes,
    patch_model_linears,
    patch_model_rmsnorms,
)
from coremlmodels.vision_model_wrapper import (
    ENUMERATED_PATCH_COUNTS,
    VisionModelWrapper,
    compute_vision_rotary_pos_emb,
    create_padding_attention_mask,
    create_patch_mask,
    patch_vision_blocks,
)


# =============================================================================
# Model Loading
# =============================================================================


def load_vision_model(
    model_name: str = "seba/GLM-OCR",
    num_layers: int | None = None,
    verbose: bool = True,
):
    """Load the GLM-OCR vision model.

    Args:
        model_name: HuggingFace model name.
        num_layers: Limit number of transformer blocks (for debugging).
        verbose: Print info.

    Returns:
        Tuple of (vision_model, vision_config, full_config).
    """
    from transformers import AutoConfig, AutoModel

    print(f"  Loading config from {model_name}...")
    full_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    vision_config = full_config.vision_config

    if verbose:
        print("  Vision config:")
        print(f"    hidden_size: {vision_config.hidden_size}")
        print(f"    out_hidden_size: {vision_config.out_hidden_size}")
        print(f"    num_heads: {vision_config.num_heads}")
        print(f"    depth: {vision_config.depth}")
        print(f"    patch_size: {vision_config.patch_size}")
        print(f"    temporal_patch_size: {vision_config.temporal_patch_size}")
        print(f"    spatial_merge_size: {vision_config.spatial_merge_size}")
        print(f"    in_channels: {vision_config.in_channels}")

    # Limit layers for debugging
    if num_layers is not None:
        original_depth = vision_config.depth
        vision_config.depth = min(num_layers, original_depth)
        print(f"  ⚠ Limiting to {vision_config.depth}/{original_depth} layers")

    print("  Loading model weights...")
    model = AutoModel.from_pretrained(
        model_name,
        config=full_config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Extract vision model
    vision_model = model.visual
    if verbose:
        print(f"  Vision model loaded: {len(vision_model.blocks)} blocks")

    return vision_model, vision_config, full_config


# =============================================================================
# Patching
# =============================================================================


def patch_vision_model(
    vision_model: nn.Module,
    vision_config: object,
    verbose: bool = True,
):
    """Apply all patches to the vision model.

    Order:
    1. patch_model_linears (all Linear -> Conv2d)
    2. patch_model_rmsnorms (RMSNorm -> LayerNorm fusable)
    3. patch_vision_blocks (VisionBlock -> VisionBlockPatcher with AttentionPatcher)

    Args:
        vision_model: The GlmOcrVisionModel.
        vision_config: Vision config.
        verbose: Print patching info.
    """
    # Detect RMSNorm classes from the loaded model's modules
    # (avoids importing from glm_ocr which has broken relative imports)
    rmsnorm_classes = {nn.RMSNorm}
    for module in vision_model.modules():
        cls = type(module)
        if "rmsnorm" in cls.__name__.lower() and cls is not nn.RMSNorm:
            rmsnorm_classes.add(cls)
    rmsnorm_classes = tuple(rmsnorm_classes)
    if verbose:
        class_names = [c.__name__ for c in rmsnorm_classes]
        print(f"  Detected RMSNorm classes: {class_names}")

    print("  [a] Patching Linear layers -> Conv2d...")
    patch_model_linears(vision_model, verbose=verbose)

    print("  [b] Patching RMSNorm layers...")
    patch_model_rmsnorms(
        vision_model,
        target_classes=rmsnorm_classes,
        verbose=verbose,
    )

    print("  [c] Patching vision attention blocks...")
    patch_vision_blocks(vision_model, verbose=verbose)


# =============================================================================
# Reference Output
# =============================================================================


def compute_reference_output(
    vision_model: nn.Module,
    num_patches: int,
    vision_config: object,
) -> tuple:
    """Compute reference output from the original (unpatched) vision model.

    Creates a synthetic input with the specified number of patches and
    runs the original model to get reference outputs for verification.

    Args:
        vision_model: Original, unpatched vision model.
        num_patches: Number of patches to use.
        vision_config: Vision config.

    Returns:
        Tuple of (reference_output, pixel_values, grid_thw).
    """
    import math

    in_channels = vision_config.in_channels
    temporal_patch_size = vision_config.temporal_patch_size
    patch_size = vision_config.patch_size

    # Create a synthetic grid_thw
    # num_patches = t * h_patches * w_patches, with t=1 for images
    # Find h_patches, w_patches such that h*w = num_patches and both are even
    sqrt_n = int(math.sqrt(num_patches))
    # Find closest factors where both are even
    h_p = sqrt_n
    while num_patches % h_p != 0:
        h_p -= 1
    w_p = num_patches // h_p
    # Ensure both are even (for spatial_merge_size=2)
    if h_p % 2 != 0:
        h_p *= 2
        w_p //= 2
    if w_p % 2 != 0:
        w_p *= 2
        h_p //= 2

    grid_thw = torch.tensor([[1, h_p, w_p]], dtype=torch.int64)
    print(f"    Reference grid: t=1, h_p={h_p}, w_p={w_p}, total={h_p * w_p}")

    # Create random pixel input
    # Shape: (num_patches * temporal_patch_size * in_channels, patch_size * patch_size)
    # but the model expects flattened: (total_pixels,)
    # Actually: (num_patches, in_channels * temporal_patch_size * patch_size * patch_size)
    pixel_values = torch.randn(
        num_patches * in_channels * temporal_patch_size * patch_size * patch_size,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        output = vision_model(pixel_values, grid_thw=grid_thw, return_dict=True)

    return output.pooler_output.numpy(), pixel_values, grid_thw


# =============================================================================
# CoreML Conversion
# =============================================================================


def convert_vision_to_coreml(
    traced_model,
    num_patches: int,
    hidden_size: int,
    head_dim: int,
    output_path: str | None = None,
    skip_model_load: bool = False,
):
    """Convert traced vision model to CoreML.

    Args:
        traced_model: JIT-traced VisionModelWrapper.
        num_patches: Number of patches for tracing shape.
        hidden_size: Vision hidden size.
        head_dim: Head dimension.
        output_path: Path for .mlpackage output.
        skip_model_load: Skip loading after conversion.

    Returns:
        CoreML model (or None if skip_model_load).
    """
    in_channels = 3
    temporal_patch_size = 2
    patch_size = 14

    # Build enumerated shapes for patch count flexibility
    # pixel_patches: (num_patches, in_channels, temporal, patch_h, patch_w) — Conv3d-ready
    pixel_shapes = []
    cos_shapes = []
    sin_shapes = []
    mask_shapes = []
    patch_mask_shapes = []

    for n in ENUMERATED_PATCH_COUNTS:
        pixel_shapes.append(
            (n, in_channels, temporal_patch_size, patch_size, patch_size)
        )
        cos_shapes.append((n, head_dim))
        sin_shapes.append((n, head_dim))
        mask_shapes.append((1, 1, n, n))
        patch_mask_shapes.append((1, 1, 1, n))

    return ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                shape=ct.EnumeratedShapes(pixel_shapes),
                name="pixel_patches",
            ),
            ct.TensorType(
                shape=ct.EnumeratedShapes(cos_shapes),
                name="position_cos",
            ),
            ct.TensorType(
                shape=ct.EnumeratedShapes(sin_shapes),
                name="position_sin",
            ),
            ct.TensorType(
                shape=ct.EnumeratedShapes(mask_shapes),
                name="attention_mask",
            ),
            ct.TensorType(
                shape=ct.EnumeratedShapes(patch_mask_shapes),
                name="patch_mask",
            ),
        ],
        outputs=[
            ct.TensorType(name="vision_embeddings"),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        package_dir=output_path,
        skip_model_load=skip_model_load,
    )


# =============================================================================
# Verification
# =============================================================================


def verify_outputs(
    coreml_model,
    wrapped_model: VisionModelWrapper,
    num_patches: int,
    vision_config: object,
):
    """Verify CoreML outputs match PyTorch wrapped model outputs.

    Args:
        coreml_model: Loaded CoreML model.
        wrapped_model: The PyTorch VisionModelWrapper.
        num_patches: Number of patches to test with.
        vision_config: Vision config.
    """
    import math

    in_channels = vision_config.in_channels
    temporal_patch_size = vision_config.temporal_patch_size
    patch_size = vision_config.patch_size
    head_dim = vision_config.hidden_size // vision_config.num_heads

    # Use fewer real patches than num_patches to test padding
    num_real = num_patches * 3 // 4  # 75% real patches
    # Make sure num_real is divisible by 4
    num_real = (num_real // 4) * 4
    if num_real < 4:
        num_real = 4

    # Find grid dimensions
    sqrt_n = int(math.sqrt(num_real))
    h_p = sqrt_n
    while num_real % h_p != 0:
        h_p -= 1
    w_p = num_real // h_p
    if h_p % 2 != 0:
        h_p, w_p = h_p * 2, w_p // 2
    if w_p % 2 != 0:
        h_p, w_p = h_p // 2, w_p * 2

    grid_thw = torch.tensor([[1, h_p, w_p]], dtype=torch.int64)

    print(f"    Testing with {num_real}/{num_patches} real patches (grid: {h_p}×{w_p})")

    # Create inputs — Conv3d-ready shape: (num_patches, in_channels, temporal, h, w)
    pixel_data = torch.randn(
        num_real,
        in_channels,
        temporal_patch_size,
        patch_size,
        patch_size,
        dtype=torch.float32,
    )
    # Pad to num_patches
    padding = torch.zeros(
        num_patches - num_real,
        in_channels,
        temporal_patch_size,
        patch_size,
        patch_size,
        dtype=torch.float32,
    )
    pixel_patches = torch.cat([pixel_data, padding], dim=0)

    # Position embeddings (only for real patches, padding gets zeros)
    cos_real, sin_real = compute_vision_rotary_pos_emb(
        grid_thw, head_dim, vision_config.spatial_merge_size
    )
    # Pad position embeddings to num_patches
    cos_pad = torch.zeros(num_patches - num_real, head_dim)
    sin_pad = torch.zeros(num_patches - num_real, head_dim)
    position_cos = torch.cat([cos_real, cos_pad], dim=0)
    position_sin = torch.cat([sin_real, sin_pad], dim=0)

    # Masks
    attention_mask = create_padding_attention_mask(num_real, num_patches)
    patch_mask = create_patch_mask(num_real, num_patches)

    # PyTorch output
    with torch.inference_mode():
        pt_output = wrapped_model(
            pixel_patches,
            position_cos,
            position_sin,
            attention_mask,
            patch_mask,
        ).numpy()

    # CoreML output
    coreml_input = {
        "pixel_patches": pixel_patches.numpy(),
        "position_cos": position_cos.numpy(),
        "position_sin": position_sin.numpy(),
        "attention_mask": attention_mask.numpy().astype(np.float16),
        "patch_mask": patch_mask.numpy().astype(np.float16),
    }
    coreml_output = coreml_model.predict(coreml_input)["vision_embeddings"]

    # Compare
    abs_diff = np.abs(pt_output - coreml_output)
    print(f"    PyTorch shape: {pt_output.shape}")
    print(f"    CoreML shape:  {coreml_output.shape}")
    print(f"    Max abs diff:  {abs_diff.max():.6f}")
    print(f"    Mean abs diff: {abs_diff.mean():.6f}")
    print(f"    Std abs diff:  {abs_diff.std():.6f}")

    passed = abs_diff.max() < 0.5  # FP16 tolerance
    print(f"    {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed


# =============================================================================
# Main Conversion Flow
# =============================================================================


def convert_vision_model(
    model_name: str = "seba/GLM-OCR",
    num_patches: int = 128,
    num_layers: int | None = None,
    output_path: str | None = None,
    verbose: bool = True,
    skip_model_load: bool = False,
):
    """Full vision model conversion pipeline.

    Args:
        model_name: HuggingFace model name.
        num_patches: Number of patches for tracing (must be in ENUMERATED_PATCH_COUNTS).
        num_layers: Limit layers for debugging.
        output_path: Output .mlpackage path.
        verbose: Print info.
        skip_model_load: Skip loading after conversion.
    """
    print("=" * 60)
    print("GLM-OCR Vision Module → CoreML Conversion")
    print("=" * 60)

    assert num_patches in ENUMERATED_PATCH_COUNTS, (
        f"num_patches={num_patches} must be one of {ENUMERATED_PATCH_COUNTS}"
    )

    # [1] Load model
    print("\n[1] Loading model...")
    vision_model, vision_config, full_config = load_vision_model(
        model_name, num_layers, verbose
    )

    head_dim = vision_config.hidden_size // vision_config.num_heads

    # [2] Compute reference (before patching)
    print("\n[2] Computing reference output...")
    if not skip_model_load:
        ref_out, ref_pixels, ref_grid = compute_reference_output(
            vision_model, num_patches, vision_config
        )
        print(f"    Reference output shape: {ref_out.shape}")
    else:
        print("    Skipped (--skip-model-load)")

    # [3] Patch layers
    print("\n[3] Patching model layers...")
    patch_vision_model(vision_model, vision_config, verbose=verbose)

    # [4] Wrap model
    print("\n[4] Creating VisionModelWrapper...")
    with torch.inference_mode():
        wrapped_model = VisionModelWrapper(vision_model, channels_first=True)
        wrapped_model.eval()
    print(f"    {wrapped_model}")

    # [5] Trace model
    print("\n[5] Tracing model...")
    in_channels = vision_config.in_channels
    temporal_patch_size = vision_config.temporal_patch_size
    patch_size = vision_config.patch_size

    example_inputs = (
        # pixel_patches: Conv3d-ready (num_patches, in_channels, temporal, h, w)
        torch.randn(
            num_patches,
            in_channels,
            temporal_patch_size,
            patch_size,
            patch_size,
            dtype=torch.float32,
        ),
        torch.randn(num_patches, head_dim, dtype=torch.float32),  # position_cos
        torch.randn(num_patches, head_dim, dtype=torch.float32),  # position_sin
        torch.zeros(
            1, 1, num_patches, num_patches, dtype=torch.float32
        ),  # attention_mask
        torch.ones(1, 1, 1, num_patches, dtype=torch.float32),  # patch_mask
    )

    with torch.inference_mode():
        # Verify forward pass works
        test_out = wrapped_model(*example_inputs)
        print(test_out)
        print(f"    Test output shape: {test_out.shape}")

        traced_model = torch.jit.trace(wrapped_model, example_inputs)
    print("    ✅ Tracing complete!")

    # [6] Convert to CoreML
    print("\n[6] Converting to CoreML...")
    register_extended_passes()

    if output_path is None:
        suffix = f"_layers_{num_layers}" if num_layers else ""
        output_path = f"glm_ocr_vision{suffix}.mlpackage"

    mlmodel = convert_vision_to_coreml(
        traced_model,
        num_patches=num_patches,
        hidden_size=vision_config.hidden_size,
        head_dim=head_dim,
        output_path=output_path,
        skip_model_load=skip_model_load,
    )
    print(f"    Saved to: {output_path}")

    # [7] Verify
    print("\n[7] Verifying outputs...")
    if skip_model_load:
        print("    Skipped (--skip-model-load)")
    else:
        verify_outputs(mlmodel, wrapped_model, num_patches, vision_config)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Model: {output_path}")
    print(f"  Blocks: {len(vision_model.blocks)}")
    print(f"  Hidden size: {vision_config.hidden_size}")
    print(f"  Enumerated patch counts: {len(ENUMERATED_PATCH_COUNTS)} shapes")
    print(f"  Range: [{ENUMERATED_PATCH_COUNTS[0]}, {ENUMERATED_PATCH_COUNTS[-1]}]")
    print("=" * 60)

    return mlmodel


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLM-OCR Vision Module to CoreML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="seba/GLM-OCR",
        help="HuggingFace model name (default: seba/GLM-OCR)",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=128,
        help=f"Number of patches for tracing. Must be in {ENUMERATED_PATCH_COUNTS[:5]}... "
        f"(default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Limit number of transformer blocks (for debugging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .mlpackage path",
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip loading CoreML model after conversion (saves memory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed info",
    )

    args = parser.parse_args()

    convert_vision_model(
        model_name=args.model,
        num_patches=args.num_patches,
        num_layers=args.num_layers,
        output_path=args.output,
        verbose=args.verbose,
        skip_model_load=args.skip_model_load,
    )


if __name__ == "__main__":
    main()
