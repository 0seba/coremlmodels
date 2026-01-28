"""Attention Layer Patching for CoreML Neural Engine Optimization.

This module provides utilities to convert attention layers to channels-first format
with Grouped Query Attention (GQA) support optimized for the Neural Engine.

Key optimizations:
1. Channels-first format (NCHW) - ANE prefers 4D tensors with channels first
2. Explicit GQA loop - Split K/V heads and process groups for better ANE scheduling
3. Manual attention computation - Avoid scaled_dot_product_attention which traces poorly
4. External position embeddings - RoPE embeddings are passed in (computed once at model level)
"""

import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Rotates half the hidden dims of the input for RoPE.

    Args:
        x: Input tensor.
        dim: Dimension along which to split and rotate. Defaults to -1.

    Returns:
        Tensor with first and second halves swapped and negated.
    """
    x1, x2 = torch.chunk(x, 2, dim)
    return torch.cat((-x2, x1), dim=dim)


def apply_rotary_pos_emb(
    embed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Apply rotary position embeddings to the input tensor.

    Args:
        embed: Input embedding tensor.
        cos: Cosine position embeddings.
        sin: Sine position embeddings.
        dim: Dimension for rotation. Defaults to -1.

    Returns:
        Tensor with rotary position embeddings applied.
    """
    return (embed * cos) + (rotate_half(embed, dim) * sin)


class AttentionPatcher(nn.Module):
    """Patches an attention layer for ANE-optimized channels-first execution.

    This patcher wraps a HuggingFace-style attention layer and converts it to
    use channels-first (NCHW) format with explicit GQA processing for better
    Neural Engine scheduling.

    Key features:
    - Expects input in channels-first format: (batch, hidden_dim, 1, seq_len)
    - Applies RoPE from external position embeddings (passed to forward)
    - Uses explicit loop over GQA groups for better ANE utilization
    - Updates KV cache in-place via index assignments

    Args:
        attention_layer: The attention layer to patch (e.g., Qwen2Attention).
        layer_idx: Index of this layer in the transformer stack for KV cache.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Dimension of each attention head.

    Attributes:
        layer: The wrapped attention layer.
        layer_idx: Layer index for KV cache indexing.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
    """

    # Default attention classes to target (extend as needed)
    DEFAULT_TARGET_CLASSES: Tuple[Type[nn.Module], ...] = ()

    def __init__(
        self,
        attention_layer: nn.Module,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()

        self.layer = attention_layer
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Store original module info for reference
        self.module_name: Optional[str] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with ANE-optimized attention computation.

        Args:
            hidden_states: Input tensor of shape (batch, hidden_dim, 1, seq_len).
            position_embeddings: Tuple of (cos, sin, cos_transposed, sin_transposed)
                                 for RoPE. Pre-indexed for current position.
            attention_mask: Causal attention mask of shape (batch, 1, seq_len, cache_len)
                           in channels-first format (transposed).
            past_key_values: Tuple of (key_cache, value_cache) tensors.
                           Shape: (num_layers, num_kv_heads, cache_len, head_dim).
            cache_position: Current position in the cache (integer).

        Returns:
            Output tensor of shape (batch, hidden_dim, 1, seq_len).
        """
        bsz, _, _, q_len = hidden_states.size()

        # Project Q, K, V using the wrapped layer's projections
        query_states = self.layer.q_proj(hidden_states)
        key_states = self.layer.k_proj(hidden_states)
        value_states = self.layer.v_proj(hidden_states)

        # Reshape for multi-head attention
        # Query: (batch, num_heads, head_dim, seq_len) - keep head_dim before seq for RoPE
        # Key/Value: (batch, num_kv_heads, seq_len, head_dim) - standard format for matmul
        query_states = query_states.view(bsz, self.num_heads, self.head_dim, q_len)
        key_states = key_states.view(
            bsz, self.num_kv_heads, self.head_dim, q_len
        ).transpose(2, 3)
        value_states = value_states.view(
            bsz, self.num_kv_heads, self.head_dim, q_len
        ).transpose(2, 3)

        # Apply rotary position embeddings
        # cos, sin: (batch, seq_len, head_dim) - standard format
        # cos_t, sin_t: (batch, head_dim, seq_len) - transposed for query
        cos, sin, cos_t, sin_t = position_embeddings

        # Query uses transposed embeddings (head_dim before seq_len)
        query_states = apply_rotary_pos_emb(query_states, cos_t, sin_t, dim=-2)
        # Key uses standard embeddings (seq_len before head_dim)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, dim=-1)

        # Update KV cache
        key_cache, value_cache = past_key_values

        # Write new keys and values to cache at current position
        # Cache shape: (num_layers, num_kv_heads, cache_len, head_dim)
        key_cache[
            self.layer_idx : self.layer_idx + 1,
            :,
            cache_position : cache_position + q_len,
        ] = key_states
        value_cache[
            self.layer_idx : self.layer_idx + 1,
            :,
            cache_position : cache_position + q_len,
        ] = value_states

        # Read full cache for this layer
        # Split by KV head for GQA processing
        key_states = key_cache[self.layer_idx : self.layer_idx + 1].split(1, dim=1)
        value_states = value_cache[self.layer_idx : self.layer_idx + 1].split(1, dim=1)

        # Split query by KV head groups
        query_states = torch.chunk(query_states, self.num_kv_heads, dim=1)

        # Compute attention for each GQA group
        attn_outputs = []
        for q, k, v in zip(query_states, key_states, value_states):
            # q: (batch, heads_per_group, head_dim, q_len)
            # k: (1, 1, cache_len, head_dim)
            # v: (1, 1, cache_len, head_dim)

            # Attention: k @ q -> (batch, 1, cache_len, q_len)
            attn_weights = k @ q
            attn_weights = attn_weights / math.sqrt(self.head_dim)

            # Apply causal mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # Softmax over cache dimension
            attn_weights = attn_weights.softmax(dim=2)

            # Output: attn_weights.T @ v -> (batch, heads_per_group, head_dim, q_len)
            # Actually: (batch, 1, q_len, cache_len) @ (1, 1, cache_len, head_dim)
            #         = (batch, 1, q_len, head_dim)
            attn_output = attn_weights.transpose(-1, -2) @ v
            attn_outputs.append(attn_output)

        # Concatenate all GQA groups
        attn_output = torch.cat(attn_outputs, dim=1)

        # Reshape: (batch, num_heads, q_len, head_dim) -> (batch, hidden_dim, 1, q_len)
        attn_output = attn_output.transpose(2, 3).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim, 1, q_len)

        # Output projection
        attn_output = self.layer.o_proj(attn_output)

        return attn_output, None

    def __repr__(self) -> str:
        return (
            f"AttentionPatcher("
            f"layer_idx={self.layer_idx}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim})"
        )


def patch_model_attention(
    model: nn.Module,
    target_classes: Tuple[Type[nn.Module], ...],
    config: object,
    skip_modules: Optional[list] = None,
    verbose: bool = False,
) -> nn.Module:
    """Iterate through model modules and patch attention layers.

    This function traverses all modules in the model and replaces attention
    layers with AttentionPatcher instances optimized for Neural Engine.

    Args:
        model: The PyTorch model to patch.
        target_classes: Tuple of attention class types to patch (e.g., Qwen2Attention).
        config: Model config object with num_attention_heads, num_key_value_heads,
                and hidden_size attributes.
        skip_modules: List of module names to skip patching. Defaults to None.
        verbose: If True, print information about patched layers. Defaults to False.

    Returns:
        The patched model with attention layers converted to ANE-optimized format.

    Example:
        >>> from transformers import Qwen2Config
        >>> from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        >>> config = Qwen2Config()
        >>> patched = patch_model_attention(model, (Qwen2Attention,), config)
    """
    if skip_modules is None:
        skip_modules = []

    # Extract config values
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    # Track layer index for KV cache
    layer_idx = [0]  # Use list to allow mutation in closure

    def _patch_module(module: nn.Module, name: str = "") -> None:
        """Recursively patch attention layers in the module."""
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if full_name in skip_modules:
                if verbose:
                    print(f"Skipping: {full_name}")
                continue

            # Check if this is a target attention class
            if isinstance(child_module, target_classes):
                patcher = AttentionPatcher(
                    attention_layer=child_module,
                    layer_idx=layer_idx[0],
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                patcher.module_name = full_name

                # Replace the attention layer with our patcher
                setattr(module, child_name, patcher)

                if verbose:
                    print(f"Patched: {full_name} -> {patcher}")

                layer_idx[0] += 1

            # Recursively process child modules
            elif isinstance(child_module, nn.Module):
                _patch_module(child_module, full_name)

    _patch_module(model)
    return model
