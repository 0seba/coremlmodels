"""Debug script: Find where NaNs arise in GLM-OCR vision CoreML conversion.

Converts only the first block up to attention (no MLP), returns all
intermediate results, and compares PyTorch vs CoreML to pinpoint the
source of NaNs.

Usage:
    uv run python examples/debug_vision_nan.py --num-patches 128
"""

import argparse
import math
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
    analyze_compute_plan,
    inspect_mil_program,
)
from coremlmodels.vision_model_wrapper import (
    ENUMERATED_PATCH_COUNTS,
    compute_vision_rotary_pos_emb,
    create_padding_attention_mask,
    create_patch_mask,
    patch_vision_blocks,
)


# =============================================================================
# Debug Wrapper — runs N blocks, detailed intermediates only on last block
# =============================================================================


class DebugVisionWrapper(nn.Module):
    """Runs patch embed + N blocks, exposes intermediates from the LAST block.

    Blocks 0..N-2 run normally (patched VisionBlockPatcher).
    Block N-1 is decomposed into individual operations with named outputs.

    Outputs (all channels-first 4D):
        patch_embed:    after Conv3d + reshape + mask
        after_norm1:    after RMSNorm (pre-attention norm) of last block
        qkv_out:        after QKV projection
        q_after_norm:   after Q-norm
        k_after_norm:   after K-norm
        q_after_rope:   after RoPE on Q
        k_after_rope:   after RoPE on K
        attn_weights:   K @ Q * scaling + mask
        attn_probs:     after softmax
        attn_output:    probs^T @ V, reshaped
        after_proj:     after output projection
        block_output:   input + attn (residual, no MLP)
        after_norm2:    after second RMSNorm (pre-MLP)
        mlp_output:     after MLP
        full_block_output: input + attn + MLP (full block)
        after_post_norm: after final post_layernorm
        after_downsample: after downsample Conv2d
        merger_output:  final output from patch merger
    """

    def __init__(self, vision_model: nn.Module, num_blocks: int):
        super().__init__()

        config = vision_model.config
        self.hidden_size = config.hidden_size
        self.out_hidden_size = config.out_hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.spatial_merge_size = config.spatial_merge_size

        # Patch embedding
        self.patch_embed = vision_model.patch_embed

        # Leading blocks (already patched as VisionBlockPatcher)
        self.leading_blocks = nn.ModuleList(
            vision_model.blocks[i] for i in range(num_blocks - 1)
        )

        # Last block — extract individual components for detailed output
        last_block = vision_model.blocks[num_blocks - 1]
        self.norm1 = last_block.norm1
        attn = last_block.attn
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.scaling = attn.scaling
        self.norm2 = last_block.norm2
        self.mlp = last_block.mlp

        # Post-processing
        self.post_layernorm = vision_model.post_layernorm
        self.downsample = vision_model.downsample
        self.merger = vision_model.merger

    def forward(
        self,
        pixel_patches: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_mask: torch.Tensor,
        grid_thw: torch.Tensor,
    ):
        bsz = 1
        half_head_dim = self.head_dim // 2

        # --- Patch Embedding ---
        hidden_states = self.patch_embed.proj(pixel_patches)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = hidden_states.unsqueeze(0).permute(0, 2, 1).unsqueeze(2)
        hidden_states = hidden_states * patch_mask
        out_patch_embed = hidden_states  # (1, hidden, 1, num_patches)

        # --- Leading blocks (run normally) ---
        position_embeddings = (position_cos, position_sin)
        for blk in self.leading_blocks:
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        # --- Last block: detailed intermediates ---
        # Norm1
        normed = self.norm1(hidden_states)
        out_after_norm1 = normed

        # QKV
        qkv = self.qkv(normed)
        out_qkv = qkv

        # Reshape to (bsz, 3, heads, head_dim, patches)
        qkv_r = qkv.view(bsz, 3, self.num_heads, self.head_dim, -1)
        query_states = qkv_r[:, 0]
        key_states = qkv_r[:, 1]
        value_states = qkv_r[:, 2]

        key_states = key_states.transpose(2, 3)
        value_states = value_states.transpose(2, 3)

        # QK-norm
        query_states = self.q_norm(query_states, axis=2)
        out_q_after_norm = query_states
        key_states = self.k_norm(key_states, axis=-1)
        out_k_after_norm = key_states

        # RoPE
        cos, sin = position_cos, position_sin
        cos_t = cos.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        sin_t = sin.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        cos_k = cos.unsqueeze(0).unsqueeze(0)
        sin_k = sin.unsqueeze(0).unsqueeze(0)

        query_states = self._apply_rotary(
            query_states, cos_t, sin_t, half_head_dim, dim=-2
        )
        out_q_after_rope = query_states
        key_states = self._apply_rotary(key_states, cos_k, sin_k, half_head_dim, dim=-1)
        out_k_after_rope = key_states

        # Attention weights
        attn_weights = key_states @ query_states
        attn_weights = attn_weights * self.scaling
        attn_weights = attn_weights + attention_mask
        out_attn_weights = attn_weights

        # Softmax
        attn_probs = attn_weights.softmax(dim=2)
        out_attn_probs = attn_probs

        # Attention output
        attn_output = attn_probs.transpose(-1, -2) @ value_states
        attn_output = attn_output.transpose(2, 3).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim, 1, -1)
        out_attn_output = attn_output

        # Output projection
        after_proj = self.proj(attn_output)
        out_after_proj = after_proj

        # Residual (attention only)
        block_output = hidden_states + after_proj
        out_block_output = block_output

        # Norm2 + MLP + Residual
        normed2 = self.norm2(block_output)
        out_after_norm2 = normed2

        mlp_out = self.mlp(normed2)
        out_mlp_output = mlp_out

        full_block = block_output + mlp_out
        out_full_block = full_block

        # --- Post-Processing ---
        # 1. Post LayerNorm
        post_normed = self.post_layernorm(full_block)
        out_after_post_norm = post_normed

        # 2. Downsample (Conv2d)
        # Reshape to (B, C, H, W) for downsample
        # full_block is (B, C, 1, L) -> view as (B, H, W, C) first?
        # Original model logic:
        # hidden_states = hidden_states.view(-1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1])
        # hidden_states = hidden_states.permute(0, 3, 1, 2)
        # But here inputs are 1D sequence of patches. We need to respect the grid.
        # However, for simplicity in debug wrapper (assuming square grid from `main`),
        # let's replicate the `vision_model_wrapper.py` logic if possible, or
        # just reshape based on `num_patches` (assuming it's a perfect square).
        # Actually `VisionModelWrapper` does:
        #   x = x.transpose(1, 2).reshape(bsz, seq_len, hidden_size)
        #   ...
        # But wait, `vision_model.downsample` expects spatial structure.
        # In our `VisionModelWrapper` (patched), `downsample` is likely a Conv2d.
        # And our `full_block` is (1, 1024, 1, 128).
        # We need to reshape it to (1, 1024, H, W).
        # We know `num_patches=128`. In `main`, H=8, W=16 (approx).
        # Wait, `main` calculates grid `h_p, w_p`. We should pass those to wrapper if we want exact spatial.
        # For now, let's just assume 128 = 8x16 or similar, or just reshape to (1, 1024, 1, 128) if downsample allows?
        # No, downsample is Conv2d with stride > 1. It needs H, W.
        # Let's derive H, W from num_patches assuming essentially square-ish.
        # But `downsample` kernel/stride is `spatial_merge_size` (usually 2).
        # So we need H, W to be divisible by `spatial_merge_size`.
        # In `main`, we enforced that.
        # Let's try to interpret (1, 128) as (8, 16) or whatever `main` used.
        # Actually `main` prints "grid: 8x12" for 96 real patches + padding?
        # The input `pixel_patches` was (num_patches, ...).
        # The `VisionModelWrapper` logic handles this via `grid_thw`.
        # Here we don't have `grid_thw`.
        # BUT: The downsample layer in `VisionModelWrapper` is patched to handle flattened input?
        # No, `VisionModelWrapper.post_process` handles the reshape.
        # We should probably replicate `VisionModelWrapper.post_process` logic here.
        # But `DebugVisionWrapper` doesn't take `grid_thw`.
        # Let's just view it as (1, C, H, W) where H*W = num_patches.
        # If num_patches=128, maybe 8x16?
        # `spatial_merge_size` is usually 2.
        # Let's hardcode a reshape valid for 128 (8x16).
        # Or better, pass `grid_size` to `forward`.
        # Let's blindly try to reshape to (1, C, H, W) where H=sqrt(N), W=N/H?
        # If N=128, sqrt=11. H=8, W=16.
        # 8*16 = 128.
        # kernel=2, stride=2.
        # (1, 1024, 8, 16) -> conv -> (1, 4096, 4, 8).
        # This seems safe.

        # Reshape to (N/4, C, 2, 2) - batched 2x2 patches
        # Input full_block: (1, 1024, 1, N)
        # Permute to (N, C) -> (N/4, 2, 2, C) -> (N/4, C, 2, 2)
        batch_size_N = full_block.shape[-1] // 4
        # (1, 1024, 1, N) -> (1, 1024, N) -> (N, 1024)
        x_flat = full_block.view(self.hidden_size, -1).permute(1, 0)
        # (N/4, 2, 2, 1024) -> (N/4, 1024, 2, 2)
        x_2d = x_flat.view(batch_size_N, 2, 2, self.hidden_size).permute(0, 3, 1, 2)

        downsampled = self.downsample(x_2d)
        # Output: (N/4, out_C, 1, 1)

        # Reshape back to (1, out_C, 1, N/4) for output signature
        out_after_downsample = downsampled.view(
            1, self.out_hidden_size, 1, batch_size_N
        )

        # 3. Merger
        # Pass (N/4, out_C, 1, 1) or (1, out_C, 1, N/4) to merger?
        # Merger is patched linear -> 1x1 conv.
        # If passed (N/4, out_C, 1, 1), it outputs (N/4, out_C, 1, 1).
        # Which corresponds to flattened sequence of length N/4.
        merged = self.merger(downsampled)
        # Reshape to (1, out_C, 1, N/4)
        out_merger_output = merged.view(1, self.out_hidden_size, 1, batch_size_N)

        return (
            out_patch_embed,
            out_after_norm1,
            out_qkv,
            out_q_after_norm,
            out_k_after_norm,
            out_q_after_rope,
            out_k_after_rope,
            out_attn_weights,
            out_attn_probs,
            out_attn_output,
            out_after_proj,
            out_block_output,
            out_after_norm2,
            out_mlp_output,
            out_full_block,
            out_after_post_norm,
            out_after_downsample,
            out_merger_output,
        )

    @staticmethod
    def _apply_rotary(x, cos, sin, half_dim, dim):
        if dim == -2:
            x1 = x[:, :, :half_dim, :]
            x2 = x[:, :, half_dim:, :]
            rotated = torch.cat((-x2, x1), dim=-2)
        else:
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)


OUTPUT_NAMES = [
    "patch_embed",
    "after_norm1",
    "qkv_out",
    "q_after_norm",
    "k_after_norm",
    "q_after_rope",
    "k_after_rope",
    "attn_weights",
    "attn_probs",
    "attn_output",
    "after_proj",
    "block_output",
    "after_norm2",
    "mlp_output",
    "full_block_output",
    "after_post_norm",
    "after_downsample",
    "merger_output",
]


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Debug NaN in vision CoreML conversion"
    )
    parser.add_argument(
        "--model", default="zai-org/GLM-OCR", help="HuggingFace model name"
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=128,
        help="Patch count for tracing (must be in ENUMERATED_PATCH_COUNTS)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of blocks to run (default: half of total)",
    )
    args = parser.parse_args()

    num_patches = args.num_patches
    assert num_patches in ENUMERATED_PATCH_COUNTS, (
        f"num_patches={num_patches} must be in ENUMERATED_PATCH_COUNTS"
    )

    # -------------------------------------------------------------------------
    # [1] Load model
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Debug: GLM-OCR Vision — finding NaN source")
    print("=" * 60)

    from transformers import AutoConfig, AutoModel

    print("\n[1] Loading model...")
    full_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vision_config = full_config.vision_config
    original_depth = vision_config.depth

    num_layers = args.num_layers if args.num_layers is not None else original_depth // 2
    vision_config.depth = num_layers
    print(f"    Using {num_layers}/{original_depth} blocks")
    print(
        f"    hidden_size={vision_config.hidden_size}, num_heads={vision_config.num_heads}"
    )

    model = AutoModel.from_pretrained(
        args.model,
        config=full_config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    vision_model = model.visual
    head_dim = vision_config.hidden_size // vision_config.num_heads

    # -------------------------------------------------------------------------
    # [2] Patch model
    # -------------------------------------------------------------------------
    print("\n[2] Patching model...")

    rmsnorm_classes = {nn.RMSNorm}
    for module in vision_model.modules():
        cls = type(module)
        if "rmsnorm" in cls.__name__.lower() and cls is not nn.RMSNorm:
            rmsnorm_classes.add(cls)
    rmsnorm_classes = tuple(rmsnorm_classes)
    print(f"    RMSNorm classes: {[c.__name__ for c in rmsnorm_classes]}")

    patch_model_linears(vision_model, verbose=False)
    print("    ✓ Linears → Conv2d")
    patch_model_rmsnorms(vision_model, target_classes=rmsnorm_classes, verbose=False)
    print("    ✓ RMSNorm patched")

    # Patch vision blocks for the leading blocks (they run normally)
    # The last block's components are extracted manually by DebugVisionWrapper
    patch_vision_blocks(vision_model, verbose=False)
    print(f"    ✓ Vision blocks patched ({num_layers} blocks)")

    # -------------------------------------------------------------------------
    # [3] Create debug wrapper
    # -------------------------------------------------------------------------
    print(
        f"\n[3] Creating DebugVisionWrapper ({num_layers} blocks, detailed on last)..."
    )
    wrapper = DebugVisionWrapper(vision_model, num_blocks=num_layers)
    wrapper.eval()

    # -------------------------------------------------------------------------
    # [4] Create test inputs
    # -------------------------------------------------------------------------
    print("\n[4] Creating test inputs...")

    in_channels = vision_config.in_channels
    temporal_patch_size = vision_config.temporal_patch_size
    patch_size = vision_config.patch_size

    # Use 75% real patches, rest padding
    num_real = (num_patches * 3 // 4 // 4) * 4
    if num_real < 4:
        num_real = 4

    # Find grid dimensions for real patches
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
    print(f"    {num_real}/{num_patches} real patches (grid: {h_p}×{w_p})")

    # Pixel patches (Conv3d-ready)
    pixel_data = torch.randn(
        num_real,
        in_channels,
        temporal_patch_size,
        patch_size,
        patch_size,
        dtype=torch.float32,
    )
    padding = torch.zeros(
        num_patches - num_real,
        in_channels,
        temporal_patch_size,
        patch_size,
        patch_size,
        dtype=torch.float32,
    )
    pixel_patches = torch.cat([pixel_data, padding], dim=0)

    # Position embeddings
    cos_real, sin_real = compute_vision_rotary_pos_emb(
        grid_thw, head_dim, vision_config.spatial_merge_size
    )
    cos_pad = torch.zeros(num_patches - num_real, head_dim)
    sin_pad = torch.zeros(num_patches - num_real, head_dim)
    position_cos = torch.cat([cos_real, cos_pad], dim=0)
    position_sin = torch.cat([sin_real, sin_pad], dim=0)

    # Masks
    attention_mask = create_padding_attention_mask(num_real, num_patches)
    patch_mask = create_patch_mask(num_real, num_patches)

    # -------------------------------------------------------------------------
    # [5] PyTorch forward
    # -------------------------------------------------------------------------
    print("\n[5] Running PyTorch forward...")
    with torch.inference_mode():
        pt_outputs = wrapper(
            pixel_patches,
            position_cos,
            position_sin,
            attention_mask,
            patch_mask,
            grid_thw,
        )
    pt_dict = {name: out.numpy() for name, out in zip(OUTPUT_NAMES, pt_outputs)}

    print("    PyTorch intermediates:")
    for name, arr in pt_dict.items():
        has_nan = np.any(np.isnan(arr))
        print(
            f"      {name:20s}  shape={str(arr.shape):30s}  nan={has_nan}  "
            f"min={np.nanmin(arr):10.4f}  max={np.nanmax(arr):10.4f}"
        )

    # -------------------------------------------------------------------------
    # [6] Trace + Convert to CoreML
    # -------------------------------------------------------------------------
    print("\n[6] Tracing...")
    example_inputs = (
        torch.randn(
            num_patches, in_channels, temporal_patch_size, patch_size, patch_size
        ),
        torch.randn(num_patches, head_dim),
        torch.randn(num_patches, head_dim),
        torch.zeros(1, 1, num_patches, num_patches),
        torch.ones(1, 1, 1, num_patches),
        grid_thw,
    )
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example_inputs)
    print("    ✓ Traced")

    print("\n[7] Converting to CoreML...")
    register_extended_passes()

    # Build enumerated shapes (limit to 8)
    valid_counts = ENUMERATED_PATCH_COUNTS[:2]
    print(f"    Using {len(valid_counts)} enumerated shapes: {valid_counts}")

    pixel_shapes = []
    cos_shapes = []
    sin_shapes = []
    mask_shapes = []
    pmask_shapes = []
    for n in valid_counts:
        pixel_shapes.append(
            (n, in_channels, temporal_patch_size, patch_size, patch_size)
        )
        cos_shapes.append((n, head_dim))
        sin_shapes.append((n, head_dim))
        mask_shapes.append((1, 1, n, n))
        pmask_shapes.append((1, 1, 1, n))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                shape=ct.EnumeratedShapes(pixel_shapes), name="pixel_patches"
            ),
            ct.TensorType(shape=ct.EnumeratedShapes(cos_shapes), name="position_cos"),
            ct.TensorType(shape=ct.EnumeratedShapes(sin_shapes), name="position_sin"),
            ct.TensorType(
                shape=ct.EnumeratedShapes(mask_shapes), name="attention_mask"
            ),
            ct.TensorType(shape=ct.EnumeratedShapes(pmask_shapes), name="patch_mask"),
            ct.TensorType(shape=(1, 3), name="grid_thw", dtype=np.int32),
        ],
        outputs=[ct.TensorType(name=name) for name in OUTPUT_NAMES],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print("    ✓ Converted")

    # Save locally
    save_path = f"debug_vision_{num_layers}layers.mlpackage"
    mlmodel.save(save_path)
    print(f"    ✓ Saved to {save_path}")

    # -------------------------------------------------------------------------
    # [8] MIL + Compute Plan Analysis
    # -------------------------------------------------------------------------
    print("\n[8] MIL Program Inspection:")
    inspect_mil_program(mlmodel)

    print("\n[9] Compute Plan Analysis:")
    analyze_compute_plan(mlmodel)

    # -------------------------------------------------------------------------
    # [10] CoreML prediction + comparison
    # -------------------------------------------------------------------------
    print("\n[10] Running CoreML prediction...")
    coreml_input = {
        "pixel_patches": pixel_patches.numpy(),
        "position_cos": position_cos.numpy(),
        "position_sin": position_sin.numpy(),
        "attention_mask": attention_mask.numpy().astype(np.float16),
        "patch_mask": patch_mask.numpy().astype(np.float16),
        "grid_thw": grid_thw.numpy().astype(np.int32),
    }
    coreml_output = mlmodel.predict(coreml_input)

    # -------------------------------------------------------------------------
    # [11] Compare all intermediates
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(
        f"{'Name':20s} | {'Shape':30s} | {'PT NaN':7s} | {'CM NaN':7s} | "
        f"{'MaxDiff':12s} | {'CM min':12s} | {'CM max':12s}"
    )
    print("-" * 100)

    for name in OUTPUT_NAMES:
        pt_arr = pt_dict[name]
        cm_arr = coreml_output[name]

        pt_nan = np.any(np.isnan(pt_arr))
        cm_nan = np.any(np.isnan(cm_arr))
        diff = np.abs(pt_arr - cm_arr)
        max_diff = np.nanmax(diff) if not (pt_nan and cm_nan) else float("nan")
        cm_min = np.nanmin(cm_arr)
        cm_max = np.nanmax(cm_arr)

        flag = " ← NaN!" if cm_nan else ""
        print(
            f"{name:20s} | {str(cm_arr.shape):30s} | {str(pt_nan):7s} | {str(cm_nan):7s} | "
            f"{max_diff:12.6f} | {cm_min:12.4f} | {cm_max:12.4f}{flag}"
        )

    print("=" * 100)
    print(
        "\nDone! The first row with 'NaN!' in CM NaN column is where NaNs first appear."
    )


if __name__ == "__main__":
    main()
