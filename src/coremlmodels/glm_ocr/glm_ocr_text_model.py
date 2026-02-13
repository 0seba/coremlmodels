"""GLM-OCR Text Model Patchers and Wrapper for CoreML Conversion.

This module provides patchers and a wrapper for the GLM-OCR text decoder
(language model) that handle its architectural differences from Qwen2/Qwen3:

1. Interleaved RoPE → converted to split-half via weight permutation at init
2. 4 layer norms per decoder layer (sandwich-norm pattern)
3. Fused gate_up_proj → chunk(2, dim=1) in channels-first format
4. mRoPE support via host-provided position_cos/position_sin embeddings

The interleaved→split-half weight permutation is correct because the dot
product is permutation-invariant: Q_perm · K_perm = Q · K. V is not permuted,
so attention output is unchanged. o_proj operates on attention output, not
permuted Q, so no change needed.
"""

import math
from typing import Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn

from coremlmodels.lm_model_wrapper import (
    _apply_channels_first_transforms,
    _generate_causal_mask,
    _generate_position_ids,
    _get_sequence_length,
    _index_position_embeddings,
)
from coremlmodels.patch_attention import apply_rotary_pos_emb


def compute_glm_ocr_mrope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
    mrope_section: Optional[Sequence[int]] = None,
    attention_scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GLM-OCR text mRoPE cosine/sine embeddings on host.

    Args:
        position_ids: Position IDs with shape (3, seq_len) or (3, batch, seq_len).
        head_dim: Attention head dimension.
        rope_theta: RoPE base theta.
        partial_rotary_factor: Fraction of head_dim using rotary dimensions.
        mrope_section: mRoPE sections for height/width/time interleaving.
        attention_scaling: Optional scaling factor applied to cos/sin.

    Returns:
        Tuple (cos, sin), each with shape (batch, seq_len, head_dim).
    """
    if position_ids.ndim == 2:
        # (3, seq_len) -> (3, 1, seq_len)
        position_ids = position_ids.unsqueeze(1)

    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(
            "position_ids must have shape (3, seq_len) or (3, batch, seq_len), "
            f"got {tuple(position_ids.shape)}"
        )

    if mrope_section is None:
        mrope_section = (8, 12, 12)

    dim = int(head_dim * partial_rotary_factor)
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"Invalid rotary dim {dim} for head_dim={head_dim}.")

    device = position_ids.device
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
    )

    # Match GlmOcrTextRotaryEmbedding.forward:
    # inv_freq_expanded: (3, bs, dim/2, 1), position_ids_expanded: (3, bs, 1, seq)
    inv_freq_expanded = inv_freq[None, None, :, None].expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    # freqs: (3, bs, seq_len, dim/2)

    if sum(mrope_section) != freqs.shape[-1]:
        raise ValueError(
            "mrope_section must sum to dim/2. "
            f"got sum={sum(mrope_section)} dim/2={freqs.shape[-1]}"
        )

    chunks = freqs.split(tuple(int(x) for x in mrope_section), dim=-1)
    freqs = torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling

    # When partial rotary is used, keep pass-through dimensions unchanged.
    if dim < head_dim:
        pad = head_dim - dim
        cos = torch.cat([cos, torch.ones(*cos.shape[:-1], pad, device=device)], dim=-1)
        sin = torch.cat([sin, torch.zeros(*sin.shape[:-1], pad, device=device)], dim=-1)

    return cos, sin


# ============================================================================
# Weight Permutation for Interleaved → Split-Half RoPE
# ============================================================================


def _interleaved_to_split_half_permutation(head_dim: int) -> torch.Tensor:
    """Create permutation indices to convert interleaved RoPE weights to split-half.

    Interleaved RoPE applies rotation to pairs (0,1), (2,3), (4,5), ...
    Split-half RoPE applies rotation to halves (0..D/2-1) and (D/2..D-1).

    By permuting Q/K projection weights so that interleaved positions map to
    split-half positions, we can use the existing efficient split-half RoPE code.

    Args:
        head_dim: Dimension of each attention head.

    Returns:
        Permutation tensor of shape (head_dim,) mapping interleaved → split-half.
    """
    # Interleaved: [0, 1, 2, 3, 4, 5, ...] → pairs (0,1), (2,3), ...
    # Split-half:  [0, 2, 4, ..., 1, 3, 5, ...]
    # Even indices go first, odd indices go second
    return torch.cat([
        torch.arange(0, head_dim, 2),
        torch.arange(1, head_dim, 2),
    ])


def _permute_qk_weights(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Permute Q or K projection weights from interleaved to split-half order.

    After Linear→Conv2d patching, weight shape is (num_heads*head_dim, hidden_size, 1, 1).
    We permute the output dimension per-head.

    Args:
        weight: Projection weight of shape (num_heads*head_dim, hidden_size, 1, 1).
        num_heads: Number of attention heads (query or KV heads).
        head_dim: Dimension per head.

    Returns:
        Permuted weight with same shape.
    """
    perm = _interleaved_to_split_half_permutation(head_dim)
    # Reshape to (num_heads, head_dim, hidden_size, 1, 1), permute, reshape back
    w = weight.view(num_heads, head_dim, -1, 1, 1)
    w = w[:, perm, :, :, :]
    return w.reshape_as(weight)


# ============================================================================
# GLM-OCR Text Attention Patcher
# ============================================================================


class GlmOcrTextAttentionPatcher(nn.Module):
    """Patches GLM-OCR text attention for ANE-optimized channels-first execution.

    At init time, permutes q_proj and k_proj weights from interleaved to
    split-half RoPE order. This avoids stride-2 gather ops in CoreML and
    lets us reuse the existing efficient split-half rotation code.

    Uses the same K@Q matmul order and GQA pattern as AttentionPatcher.
    """

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

        self.module_name: Optional[str] = None

        # Permute Q and K projection weights: interleaved → split-half
        with torch.no_grad():
            self.layer.q_proj.weight.copy_(
                _permute_qk_weights(self.layer.q_proj.weight, num_heads, head_dim)
            )
            self.layer.k_proj.weight.copy_(
                _permute_qk_weights(self.layer.k_proj.weight, num_kv_heads, head_dim)
            )

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

        Same interface as AttentionPatcher.forward.
        """
        bsz = hidden_states.size(0)

        # Project Q, K, V
        query_states = self.layer.q_proj(hidden_states)
        key_states = self.layer.k_proj(hidden_states)
        value_states = self.layer.v_proj(hidden_states)

        # Reshape for multi-head attention
        # Query: (batch, num_heads, head_dim, seq_len)
        # Key/Value: (batch, num_kv_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, self.num_heads, self.head_dim, -1)
        key_states = key_states.view(bsz, self.num_kv_heads, self.head_dim, -1).transpose(2, 3)
        value_states = value_states.view(bsz, self.num_kv_heads, self.head_dim, -1).transpose(2, 3)

        # Apply rotary position embeddings (split-half, thanks to weight permutation)
        cos, sin, cos_t, sin_t = position_embeddings
        query_states = apply_rotary_pos_emb(query_states, cos_t, sin_t, dim=-2)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, dim=-1)

        # Update KV cache
        key_cache, value_cache = past_key_values
        seq_len = key_states.size(2)
        key_cache[
            self.layer_idx : self.layer_idx + 1,
            :,
            cache_position : cache_position + seq_len,
        ] = key_states
        value_cache[
            self.layer_idx : self.layer_idx + 1,
            :,
            cache_position : cache_position + seq_len,
        ] = value_states

        # Read full cache for this layer, split by KV head for GQA
        key_states = key_cache[self.layer_idx : self.layer_idx + 1].split(1, dim=1)
        value_states = value_cache[self.layer_idx : self.layer_idx + 1].split(1, dim=1)
        query_states = torch.chunk(query_states, self.num_kv_heads, dim=1)

        # Compute attention for each GQA group
        attn_outputs = []
        for q, k, v in zip(query_states, key_states, value_states):
            attn_weights = k @ q
            attn_weights = attn_weights / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = attn_weights.softmax(dim=2)
            attn_output = attn_weights.transpose(-1, -2) @ v
            attn_outputs.append(attn_output)

        attn_output = torch.cat(attn_outputs, dim=1)

        # Reshape: (batch, num_heads, seq_len, head_dim) -> (batch, hidden_dim, 1, seq_len)
        attn_output = attn_output.transpose(2, 3).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim, 1, -1)

        # Output projection
        attn_output = self.layer.o_proj(attn_output)

        return attn_output, None

    def __repr__(self) -> str:
        return (
            f"GlmOcrTextAttentionPatcher("
            f"layer_idx={self.layer_idx}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim})"
        )


# ============================================================================
# GLM-OCR Text MLP Patcher
# ============================================================================


class GlmOcrTextMLPPatcher(nn.Module):
    """Patches GLM-OCR text MLP for channels-first format.

    The original MLP uses a fused gate_up_proj that projects to 2*intermediate_size,
    then chunks along dim=-1 (last dim). In channels-first format, the channels
    dimension is dim=1, so we need to chunk along dim=1 instead.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()

        self.gate_up_proj = mlp.gate_up_proj  # Already Conv2d-patched
        self.down_proj = mlp.down_proj  # Already Conv2d-patched
        self.activation_fn = mlp.activation_fn

        self.module_name: Optional[str] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass in channels-first format.

        Args:
            hidden_states: (batch, hidden_dim, 1, seq_len)

        Returns:
            (batch, hidden_dim, 1, seq_len)
        """
        up_states = self.gate_up_proj(hidden_states)
        # Chunk along channels (dim=1) instead of last dim
        gate, up_states = up_states.chunk(2, dim=1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)

    def __repr__(self) -> str:
        return "GlmOcrTextMLPPatcher()"


# ============================================================================
# GLM-OCR Text Decoder Layer Patcher
# ============================================================================


class GlmOcrTextDecoderLayerPatcher(nn.Module):
    """Patches GLM-OCR text decoder layer with 4-norm sandwich pattern.

    GLM-OCR uses 4 RMSNorms per layer:
    - input_layernorm: pre-attention
    - post_self_attn_layernorm: post-attention (before residual add)
    - post_attention_layernorm: pre-MLP
    - post_mlp_layernorm: post-MLP (before residual add)

    Forward signature matches the convention used by LanguageModelWrapper.
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        attn_patcher: GlmOcrTextAttentionPatcher,
        mlp_patcher: GlmOcrTextMLPPatcher,
    ):
        super().__init__()

        # Norms (already patched to RMSNormToLayerNormPatcher)
        self.input_layernorm = decoder_layer.input_layernorm
        self.post_self_attn_layernorm = decoder_layer.post_self_attn_layernorm
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm
        self.post_mlp_layernorm = decoder_layer.post_mlp_layernorm

        # Patched attention and MLP
        self.self_attn = attn_patcher
        self.mlp = mlp_patcher

        self.module_name: Optional[str] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_values: Tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Forward pass matching LanguageModelWrapper's decoder layer call convention.

        Args:
            hidden_states: (batch, hidden_dim, 1, seq_len)
            attention_mask: Causal mask
            position_ids: Not used (position info in position_embeddings)
            past_key_values: (key_cache, value_cache) tuple
            cache_position: Current position in cache
            position_embeddings: (cos, sin, cos_t, sin_t) for RoPE

        Returns:
            hidden_states: (batch, hidden_dim, 1, seq_len)
        """
        # Pre-attention norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        # Post-attention norm + residual
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Pre-MLP norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Post-MLP norm + residual
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def __repr__(self) -> str:
        return (
            f"GlmOcrTextDecoderLayerPatcher("
            f"attn={self.self_attn}, "
            f"mlp={self.mlp})"
        )


# ============================================================================
# GLM-OCR Language Model Wrapper
# ============================================================================


class GlmOcrLanguageModelWrapper(nn.Module):
    """Wrapper for GLM-OCR text decoder optimized for CoreML conversion.

    The wrapper keeps a scalar `position_id` for KV-cache write position and
    supports explicit RoPE embeddings (`position_cos`, `position_sin`) as
    inputs. This allows multimodal prefill with GLM-OCR mRoPE while preserving
    fixed-shape CoreML input tensors.

    If position embeddings are not provided, it falls back to internal 1D RoPE
    buffers for backward compatibility.
    """

    def __init__(
        self,
        model: nn.Module,
        config: object,
        cache_length: int = 2048,
        channels_first: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        self.layer = model
        self.cache_length = cache_length
        self.channels_first = channels_first

        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        # Initialize KV cache as registered buffers
        self.register_buffer(
            "key_cache",
            torch.zeros(num_layers, num_kv_heads, cache_length, head_dim, device=device),
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(num_layers, num_kv_heads, cache_length, head_dim, device=device),
        )

        # Pre-compute standard 1D RoPE embeddings
        # Extract rope_theta from config.rope_parameters
        rope_theta = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        )
        positions = torch.arange(cache_length, dtype=torch.float, device=device)
        freqs = torch.outer(positions, inv_freq)  # (cache_length, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (cache_length, dim)

        # If partial rotation, pad with zeros for the non-rotated portion
        if dim < head_dim:
            padding = torch.zeros(cache_length, head_dim - dim, device=device)
            cos_emb = torch.cat([emb.cos(), padding + 1.0], dim=-1)
            sin_emb = torch.cat([emb.sin(), padding], dim=-1)
        else:
            cos_emb = emb.cos()
            sin_emb = emb.sin()

        self.register_buffer("cos_emb", cos_emb)  # (cache_length, head_dim)
        self.register_buffer("sin_emb", sin_emb)  # (cache_length, head_dim)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_id: torch.Tensor,
        position_cos: Optional[torch.Tensor] = None,
        position_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the wrapped GLM-OCR text model.

        Args:
            inputs_embeds: Input embeddings in channels-first format.
                          Shape: (batch, hidden_dim, 1, seq_len).
            position_id: Starting KV-cache position.
                        Shape: (1,) containing the cache index.
            position_cos: Optional RoPE cosine embeddings for this chunk.
                          Shape: (seq_len, head_dim) or (1, seq_len, head_dim).
            position_sin: Optional RoPE sine embeddings for this chunk.
                          Shape: (seq_len, head_dim) or (1, seq_len, head_dim).

        Returns:
            Hidden states in channels-first format.
            Shape: (batch, hidden_dim, 1, seq_len).
        """
        seq_len = _get_sequence_length(inputs_embeds, self.channels_first)
        cache_position_ids = _generate_position_ids(
            seq_len, position_id, inputs_embeds.device
        )

        if position_cos is None or position_sin is None:
            # Backward-compatible fallback: standard 1D RoPE from cache positions.
            cos, sin = _index_position_embeddings(
                self.cos_emb, self.sin_emb, cache_position_ids
            )
        else:
            # Accept (seq_len, head_dim) or (1, seq_len, head_dim).
            if position_cos.ndim == 2:
                position_cos = position_cos.unsqueeze(0)
            if position_sin.ndim == 2:
                position_sin = position_sin.unsqueeze(0)

            if position_cos.ndim != 3 or position_sin.ndim != 3:
                raise ValueError(
                    "position_cos/position_sin must have shape "
                    "(seq_len, head_dim) or (1, seq_len, head_dim)"
                )

            if position_cos.shape[1] != seq_len or position_sin.shape[1] != seq_len:
                raise ValueError(
                    "position_cos/position_sin seq_len mismatch: "
                    f"{position_cos.shape[1]} / {position_sin.shape[1]} vs {seq_len}"
                )

            cos = position_cos
            sin = position_sin

        position_emb = (cos, sin)
        attention_mask = _generate_causal_mask(
            self.cache_length, cache_position_ids, inputs_embeds.device
        )

        if self.channels_first:
            position_emb, attention_mask = _apply_channels_first_transforms(
                position_emb, attention_mask
            )

        hidden_states = inputs_embeds

        for decoder_layer in self.layer.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                None,  # position_ids not needed
                (self.key_cache, self.value_cache),
                cache_position=position_id,
                position_embeddings=position_emb,
            )

        # Final layer norm
        hidden_states = self.layer.norm(hidden_states)

        return hidden_states

    def __repr__(self) -> str:
        return (
            f"GlmOcrLanguageModelWrapper("
            f"cache_length={self.cache_length}, "
            f"channels_first={self.channels_first}, "
            f"key_cache_shape={tuple(self.key_cache.shape)}, "
            f"value_cache_shape={tuple(self.value_cache.shape)})"
        )


# ============================================================================
# Patching Function
# ============================================================================


def patch_glm_ocr_text_layers(
    text_model: nn.Module,
    config: object,
    verbose: bool = False,
) -> nn.Module:
    """Patch GLM-OCR text model layers for CoreML conversion.

    This should be called AFTER patch_model_linears and patch_model_rmsnorms.
    It replaces attention, MLP, and decoder layer modules with their
    GLM-OCR-specific patchers.

    Args:
        text_model: The GlmOcrTextModel instance (with layers already
                    patched by patch_model_linears and patch_model_rmsnorms).
        config: Text model config with num_attention_heads, num_key_value_heads, etc.
        verbose: Print patching info.

    Returns:
        The patched text model.
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_heads

    for layer_idx, decoder_layer in enumerate(text_model.layers):
        # Patch attention
        attn_patcher = GlmOcrTextAttentionPatcher(
            attention_layer=decoder_layer.self_attn,
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Patch MLP
        mlp_patcher = GlmOcrTextMLPPatcher(decoder_layer.mlp)

        # Patch decoder layer
        layer_patcher = GlmOcrTextDecoderLayerPatcher(
            decoder_layer=decoder_layer,
            attn_patcher=attn_patcher,
            mlp_patcher=mlp_patcher,
        )

        text_model.layers[layer_idx] = layer_patcher

        if verbose:
            print(f"  Patched: layers.{layer_idx} -> {layer_patcher}")

    return text_model


# ============================================================================
# CoreML State Specs Helper
# ============================================================================


def create_glm_ocr_state_specs(wrapper: GlmOcrLanguageModelWrapper) -> list:
    """Create CoreML StateType specifications for GLM-OCR text model wrapper.

    Args:
        wrapper: The GlmOcrLanguageModelWrapper instance.

    Returns:
        List of coremltools.StateType objects for key_cache and value_cache.
    """
    import coremltools as ct

    return [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=wrapper.key_cache.shape),
            name="key_cache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=wrapper.value_cache.shape),
            name="value_cache",
        ),
    ]
