"""Vision Model Wrapper for CoreML Conversion (GLM-OCR style).

This module provides a wrapper for vision transformer models that:
1. Takes pre-extracted patches (padded to an enumerated count) as input
2. Applies patch embedding (Conv2d, converted from Conv3d)
3. Uses pre-computed 2D rotary position embeddings passed as input
4. Applies attention masking for padded patches
5. Runs transformer blocks, downsample, and patch merger

Key design: patches are padded at the patch level (not spatially), so the
model's flexible shape dimension is the number of patches. This allows a
single enumerated shape to serve multiple image aspect ratios.

The original Conv3d patch embedding with kernel (2,14,14) is converted to
a Conv2d with kernel (14,14) by folding the temporal dimension into the
channel dimension (3 channels × 2 temporal = 6 input channels).

No CoreML state is needed — vision encoding is a single forward pass.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# Enumerated Patch Counts
# ============================================================================

# Min patches: (112*112) / (2*14*14) = 32
# Max patches: capped at 16384 (CoreML dimension limit)
# 32 values with roughly geometric spacing, all multiples of 4
ENUMERATED_PATCH_COUNTS = [
    # 32,
    # 48,
    # 64,
    # 96,
    128,
    # 176,
    # 240,
    # 320,
    # 432,
    576,
    # 768,
    1024,
    # 1344,
    1792,
    2368,
    3072,
    3840,
    4608,
    5376,
    6144,
    6912,
    7680,
    8448,
    9216,
    9984,
    10752,
    11520,
    12288,
    13056,
    13824,
    14848,
    16384,
]


def get_best_patch_count(num_real_patches: int) -> int:
    """Find the smallest enumerated patch count >= num_real_patches."""
    for count in ENUMERATED_PATCH_COUNTS:
        if count >= num_real_patches:
            return count
    raise ValueError(
        f"num_real_patches={num_real_patches} exceeds maximum "
        f"enumerated count {ENUMERATED_PATCH_COUNTS[-1]}"
    )


# ============================================================================
# 2D Rotary Position Embeddings (pre-computed on host)
# ============================================================================


def compute_vision_rotary_pos_emb(
    grid_thw: torch.Tensor,
    head_dim: int,
    spatial_merge_size: int = 2,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute 2D rotary position embeddings for vision patches.

    This replicates GlmOcrVisionModel.rot_pos_emb() but returns (cos, sin)
    tensors ready for the model.

    Args:
        grid_thw: Tensor of shape (num_images, 3) with [t, h_patches, w_patches].
        head_dim: Head dimension of the attention layer.
        spatial_merge_size: Spatial merge size (default 2).
        theta: RoPE base frequency.

    Returns:
        Tuple of (cos, sin) each of shape (total_patches, head_dim).
    """
    rot_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float) / rot_dim)
    )

    pos_ids = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids, dim=0)  # (total_patches, 2)

    max_grid_size = grid_thw[:, 1:].max().item()
    seq = torch.arange(max_grid_size, dtype=torch.float)
    freqs = torch.outer(seq, inv_freq)  # (max_grid_size, rot_dim/2)

    # Index freqs by position IDs and flatten
    rotary_pos_emb = freqs[pos_ids].flatten(1)  # (total_patches, rot_dim)

    # Double the embedding (matches original: cat((emb, emb), dim=-1))
    emb = torch.cat(
        (rotary_pos_emb, rotary_pos_emb), dim=-1
    )  # (total_patches, head_dim)

    return emb.cos(), emb.sin()


# ============================================================================
# Vision Attention Patcher (non-causal, no KV cache)
# ============================================================================


class VisionAttentionPatcher(nn.Module):
    """Patches a vision attention layer for CoreML conversion.

    Unlike the LM AttentionPatcher, this is non-causal and has no KV cache.
    Attention is computed over all tokens at once with an attention mask
    for padded patches.

    The original GlmOcrVisionAttention uses:
    - Fused QKV projection
    - QK-norm (RMSNorm on head_dim)
    - 2D RoPE
    - Non-causal attention with cu_seqlens-based chunking

    This patcher replaces cu_seqlens with an explicit attention mask.
    """

    def __init__(
        self,
        attention_layer: nn.Module,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        # Keep original projections
        self.qkv = attention_layer.qkv
        self.proj = attention_layer.proj

        # QK-norm (already patched to RMSNormToLayerNormPatcher by patch_model_rmsnorms)
        self.q_norm = attention_layer.q_norm
        self.k_norm = attention_layer.k_norm

        self.module_name: Optional[str] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: (1, hidden_dim, 1, num_patches) in channels-first format.
            position_embeddings: Tuple of (cos, sin) each (num_patches, head_dim).
            attention_mask: (1, 1, num_patches, num_patches) with -inf for masked positions.

        Returns:
            (1, hidden_dim, 1, num_patches) output tensor.
        """
        bsz = hidden_states.size(0)

        # QKV projection — input: (1, hidden_dim, 1, num_patches)
        # qkv output: (1, 3*hidden_dim, 1, num_patches)
        qkv = self.qkv(hidden_states)

        # Reshape to (bsz, 3, num_heads, head_dim, num_patches)
        qkv = qkv.view(bsz, 3, self.num_heads, self.head_dim, -1)
        query_states = qkv[:, 0]  # (bsz, num_heads, head_dim, num_patches)
        key_states = qkv[:, 1]
        value_states = qkv[:, 2]

        # Transpose K,V to (bsz, num_heads, num_patches, head_dim)
        key_states = key_states.transpose(2, 3)
        value_states = value_states.transpose(2, 3)

        # QK-norm: normalize over head_dim
        # Q: (bsz, num_heads, head_dim, num_patches) → normalize axis=2
        query_states = self.q_norm(query_states, axis=2)
        # K: (bsz, num_heads, num_patches, head_dim) → normalize axis=-1
        key_states = self.k_norm(key_states, axis=-1)

        # Apply 2D RoPE
        cos, sin = position_embeddings
        # cos, sin: (num_patches, head_dim)
        # Q needs (head_dim, num_patches) format cos/sin → transpose
        cos_t = (
            cos.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, head_dim, num_patches)
        sin_t = sin.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        cos_k = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, num_patches, head_dim)
        sin_k = sin.unsqueeze(0).unsqueeze(0)

        # Apply RoPE to Q (head_dim before num_patches) and K (num_patches before head_dim)
        half_head_dim = self.head_dim // 2
        query_states = self._apply_rotary(
            query_states, cos_t, sin_t, half_head_dim, dim=-2
        )
        key_states = self._apply_rotary(key_states, cos_k, sin_k, half_head_dim, dim=-1)

        # Attention: Q^T @ K → (bsz, num_heads, num_patches_q, num_patches_k)
        # Q: (bsz, num_heads, head_dim, num_patches) → need K @ Q for (num_patches, num_patches)
        attn_weights = (
            key_states @ query_states
        )  # (bsz, num_heads, num_patches, num_patches)
        attn_weights = attn_weights * self.scaling

        # Apply attention mask (padding mask)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax over key dimension (dim=2 since shape is bsz, heads, keys, queries)
        attn_weights = attn_weights.softmax(dim=2)

        # Output: attn_weights^T @ V
        # attn_weights: (bsz, num_heads, num_patches_k, num_patches_q)
        # V: (bsz, num_heads, num_patches, head_dim)
        # attn_weights^T: (bsz, num_heads, num_patches_q, num_patches_k)
        attn_output = attn_weights.transpose(-1, -2) @ value_states
        # attn_output: (bsz, num_heads, num_patches_q, head_dim)

        # Reshape to (bsz, hidden_dim, 1, num_patches) channels-first
        attn_output = attn_output.transpose(
            2, 3
        ).contiguous()  # (bsz, num_heads, head_dim, num_patches)
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim, 1, -1)

        # Output projection
        attn_output = self.proj(attn_output)
        return attn_output

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        half_dim: int,
        dim: int,
    ) -> torch.Tensor:
        """Apply rotary position embedding.

        Uses the same rotate_half as the original vision model (split at midpoint).
        half_dim must be a Python int constant (not derived from tensor size)
        to avoid dynamic int ops that CoreML cannot convert.
        """
        if dim == -2:
            # Split along dim=-2 (head_dim dimension when it's before seq)
            x1 = x[:, :, :half_dim, :]
            x2 = x[:, :, half_dim:, :]
            rotated = torch.cat((-x2, x1), dim=-2)
        else:
            # Split along last dim
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

    def __repr__(self) -> str:
        return (
            f"VisionAttentionPatcher("
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim})"
        )


# ============================================================================
# Vision Model Wrapper
# ============================================================================


class VisionModelWrapper(nn.Module):
    """Wrapper for GLM-OCR vision model optimized for CoreML conversion.

    This wrapper:
    - Takes pre-chunked patches as input (padded to enumerated count)
    - Applies Conv2d patch embedding (converted from Conv3d)
    - Zeros out padding patch embeddings using a mask
    - Runs transformer blocks with attention masking
    - Applies post-layernorm, downsample (Conv2d), and patch merger
    - Outputs full sequence (host extracts real tokens)

    Inputs:
        pixel_patches: (num_patches, in_channels * temporal_patch_size, patch_h, patch_w)
            Pre-chunked patches with temporal dim folded into channels.
            E.g. (N, 6, 14, 14) for 3 channels × 2 temporal. Padding patches are zeros.
        position_cos: (num_patches, head_dim)
            Pre-computed 2D RoPE cosine embeddings for real patches
            (padding patches get arbitrary values since they're masked).
        position_sin: (num_patches, head_dim)
            Pre-computed 2D RoPE sine embeddings.
        attention_mask: (1, 1, num_patches, num_patches)
            Attention mask with 0 for valid, -inf for padded positions.
        patch_mask: (1, 1, 1, num_patches)
            Binary mask: 1 for real patches, 0 for padding patches.

    Output:
        merged_output: (1, out_hidden_size, 1, num_merged_tokens)
            Vision embeddings after downsample + merger.
            num_merged_tokens = num_patches / spatial_merge_size^2
    """

    def __init__(
        self,
        vision_model: nn.Module,
        channels_first: bool = True,
    ):
        super().__init__()

        config = vision_model.config

        # Patch embedding — replace Conv3d with Conv2d
        # Conv3d kernel (2,14,14) stride (2,14,14) is non-overlapping:
        # input (N, 3, 2, 14, 14) → fold temporal into channels → (N, 6, 14, 14)
        conv3d = vision_model.patch_embed.proj
        conv2d = nn.Conv2d(
            in_channels=config.in_channels * config.temporal_patch_size,  # 3*2=6
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,  # 14
            stride=config.patch_size,  # 14
            bias=conv3d.bias is not None,
        )
        # Reshape weights: (out_ch, 3, 2, 14, 14) → (out_ch, 6, 14, 14)
        with torch.no_grad():
            conv2d.weight.copy_(
                conv3d.weight.reshape(
                    config.hidden_size,
                    config.in_channels * config.temporal_patch_size,
                    config.patch_size,
                    config.patch_size,
                )
            )
            if conv3d.bias is not None:
                conv2d.bias.copy_(conv3d.bias)
        self.patch_embed_proj = conv2d

        # Store sub-modules
        self.blocks = vision_model.blocks
        self.post_layernorm = vision_model.post_layernorm
        self.downsample = vision_model.downsample
        self.merger = vision_model.merger

        # Config
        self.hidden_size = config.hidden_size
        self.out_hidden_size = config.out_hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.channels_first = channels_first

        # Derived
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

    def forward(
        self,
        pixel_patches: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_patches: (num_patches, in_channels * temporal, patch_h, patch_w)
                4D input with temporal folded into channels. E.g. (N, 6, 14, 14).
                Padding patches are zeros.
            position_cos: (num_patches, head_dim) — 2D RoPE cos.
            position_sin: (num_patches, head_dim) — 2D RoPE sin.
            attention_mask: (1, 1, num_patches, num_patches) — attention mask.
            patch_mask: (1, 1, 1, num_patches) — binary mask for real patches.

        Returns:
            (1, out_hidden_size, 1, num_merged_tokens) — vision embeddings.
        """
        # --- Patch Embedding (Conv2d) ---
        # pixel_patches: (N, 6, 14, 14) — temporal already folded into channels
        hidden_states = self.patch_embed_proj(pixel_patches)
        # Conv2d output: (N, hidden_size, 1, 1) → flatten to (N, hidden_size)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = hidden_states.unsqueeze(0).permute(0, 2, 1).unsqueeze(2)
        # Now: (1, hidden_size, 1, num_patches)

        # Zero out padding patch embeddings
        hidden_states = hidden_states * patch_mask

        # --- Transformer Blocks ---
        position_embeddings = (position_cos, position_sin)
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        # --- Post LayerNorm ---
        hidden_states = self.post_layernorm(hidden_states)

        # --- Downsample ---
        # Reshape to (N/4, C, 2, 2) batched 2×2 patches
        num_groups = hidden_states.shape[-1] // 4
        x_flat = hidden_states.view(self.hidden_size, -1).permute(1, 0)
        x_2d = x_flat.view(num_groups, 2, 2, self.hidden_size).permute(0, 3, 1, 2)

        downsampled = self.downsample(x_2d)  # (num_groups, out_hidden_size, 1, 1)

        # --- Patch Merger ---
        merged = self.merger(downsampled)  # (num_groups, out_hidden_size, 1, 1)
        output = merged.view(1, self.out_hidden_size, 1, num_groups)

        return output

    def __repr__(self) -> str:
        num_blocks = len(self.blocks)
        return (
            f"VisionModelWrapper(\n"
            f"    hidden_size={self.hidden_size},\n"
            f"    out_hidden_size={self.out_hidden_size},\n"
            f"    num_heads={self.num_heads},\n"
            f"    head_dim={self.head_dim},\n"
            f"    num_blocks={num_blocks},\n"
            f"    spatial_merge_size={self.spatial_merge_size},\n"
            f"    channels_first={self.channels_first},\n"
            f")"
        )


# ============================================================================
# Attention Mask and Patch Mask Generation
# ============================================================================


def create_padding_attention_mask(
    num_real_patches: int,
    num_total_patches: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create attention mask that blocks padding patches.

    Real patches attend to all real patches. Padding patches are masked out
    (both as queries and keys).

    Args:
        num_real_patches: Number of real (non-padding) patches.
        num_total_patches: Total number of patches (including padding).

    Returns:
        Attention mask of shape (1, 1, num_total_patches, num_total_patches).
        0 for valid attention, -1e4 for masked positions.
        Uses -1e4 instead of -inf to avoid NaN from softmax on all-masked
        rows in FP16 (softmax([-inf,...,-inf]) = 0/0 = NaN).
    """
    mask = torch.zeros(num_total_patches, num_total_patches, device=device)
    # Mask out padding patches as keys (columns)
    mask[:, num_real_patches:] = -1e4
    # Mask out padding patches as queries (rows) — they shouldn't produce meaningful output
    mask[num_real_patches:, :] = -1e4
    return mask.unsqueeze(0).unsqueeze(0)


def create_patch_mask(
    num_real_patches: int,
    num_total_patches: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create binary mask for zeroing out padding patch embeddings.

    Args:
        num_real_patches: Number of real patches.
        num_total_patches: Total patches (including padding).

    Returns:
        Mask of shape (1, 1, 1, num_total_patches).
        1 for real patches, 0 for padding.
    """
    mask = torch.zeros(1, 1, 1, num_total_patches, device=device)
    mask[:, :, :, :num_real_patches] = 1.0
    return mask


# ============================================================================
# Vision Patch Merger Patcher (handles LayerNorm in channels-first)
# ============================================================================


class VisionPatchMergerPatcher(nn.Module):
    """Wraps GlmOcrVisionPatchMerger for channels-first 4D format.

    The merger contains Linear→Conv2d layers and a LayerNorm.
    In channels-first (1, C, 1, seq), LayerNorm must normalize over dim=1 (C),
    not the last dim. This patcher handles the permutation.

    Original merger forward:
        hidden = proj(x)
        hidden = GELU(LayerNorm(hidden))
        return down_proj(act_fn(gate_proj(hidden)) * up_proj(hidden))
    """

    def __init__(self, merger: nn.Module):
        super().__init__()
        self.proj = merger.proj  # Conv2d (patched)
        self.post_projection_norm = merger.post_projection_norm  # LayerNorm
        self.gate_proj = merger.gate_proj  # Conv2d (patched)
        self.up_proj = merger.up_proj  # Conv2d (patched)
        self.down_proj = merger.down_proj  # Conv2d (patched)
        self.act1 = merger.act1  # GELU
        self.act_fn = merger.act_fn  # SiLU / SwiGLU activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass in channels-first format.

        Args:
            hidden_states: (1, hidden_dim, 1, num_groups) channels-first.

        Returns:
            (1, out_dim, 1, num_groups) channels-first.
        """
        # proj (Conv2d): (1, dim, 1, seq) → (1, dim, 1, seq)
        hidden_states = self.proj(hidden_states)

        # LayerNorm expects normalization over last dim = hidden_dim
        # Permute to (1, seq, 1, dim) → LayerNorm → permute back
        hidden_states = hidden_states.permute(0, 3, 2, 1)  # (1, seq, 1, dim)
        hidden_states = self.post_projection_norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 2, 1)  # (1, dim, 1, seq)

        hidden_states = self.act1(hidden_states)

        # SwiGLU: down_proj(act_fn(gate) * up)
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


# ============================================================================
# Vision Block Patcher (wraps GlmOcrVisionBlock for channels-first)
# ============================================================================


class VisionBlockPatcher(nn.Module):
    """Wraps a GlmOcrVisionBlock for channels-first format with attention mask.

    The original block uses (seq_len, hidden_size) format with cu_seqlens.
    This patcher converts to (1, hidden_size, 1, num_patches) channels-first
    format and uses an explicit attention mask.
    """

    def __init__(self, block: nn.Module, num_heads: int, head_dim: int):
        super().__init__()

        self.norm1 = block.norm1  # Already patched to RMSNormToLayerNormPatcher
        self.norm2 = block.norm2
        self.attn = VisionAttentionPatcher(block.attn, num_heads, head_dim)
        self.mlp = block.mlp  # Already patched (Linear→Conv2d)

        self.module_name: Optional[str] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass in channels-first format.

        Args:
            hidden_states: (1, hidden_size, 1, num_patches)
            position_embeddings: (cos, sin) for 2D RoPE
            attention_mask: (1, 1, num_patches, num_patches)

        Returns:
            (1, hidden_size, 1, num_patches)
        """
        # Pre-norm + Attention + Residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        # Pre-norm + MLP + Residual
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


# ============================================================================
# Patching Functions
# ============================================================================


def patch_vision_blocks(
    vision_model: nn.Module,
    verbose: bool = False,
) -> nn.Module:
    """Replace GlmOcrVisionBlocks with VisionBlockPatchers.

    Also patches the merger module (GlmOcrVisionPatchMerger →
    VisionPatchMergerPatcher) for channels-first 4D compatibility.

    This should be called AFTER patch_model_rmsnorms and patch_model_linears,
    so that the inner layers are already converted.

    Args:
        vision_model: The GlmOcrVisionModel instance.
        verbose: Print patching info.

    Returns:
        The vision model with patched blocks and merger.
    """
    num_heads = vision_model.config.num_heads
    head_dim = vision_model.config.hidden_size // num_heads

    for i, block in enumerate(vision_model.blocks):
        patcher = VisionBlockPatcher(block, num_heads, head_dim)
        patcher.module_name = f"blocks.{i}"
        vision_model.blocks[i] = patcher
        if verbose:
            print(f"  Patched: blocks.{i} -> {patcher.attn}")

    # Patch the merger (LayerNorm needs channels-first handling)
    if hasattr(vision_model, "merger"):
        merger_patcher = VisionPatchMergerPatcher(vision_model.merger)
        vision_model.merger = merger_patcher
        if verbose:
            print("  Patched: merger -> VisionPatchMergerPatcher")

    return vision_model
