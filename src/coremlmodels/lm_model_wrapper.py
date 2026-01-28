"""Language Model Wrapper for CoreML Conversion with KV Cache State.

This module provides a wrapper for transformer language models that:
1. Pre-computes position embeddings at initialization (avoids FP16 precision issues)
2. Manages KV cache as registered buffers (converted to CoreML state)
3. Uses channels-first format (NCHW) for Neural Engine optimization
4. Handles position indexing at the start of forward (CPU op, then ANE graph)

The wrapper is designed to be traced with torch.jit and converted to CoreML
with ct.StateType for the KV cache tensors.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LanguageModelWrapper(nn.Module):
    """Wrapper for transformer language models optimized for CoreML conversion.

    This wrapper handles:
    - Pre-computed rotary position embeddings stored as buffers
    - KV cache as registered buffers (become CoreML state)
    - Causal attention mask generation
    - Channels-first tensor format for ANE

    The position embeddings are computed once at initialization to avoid
    FP16 precision issues with exponentials, cos, and sin operations.
    Position indexing happens at the start of forward() and typically runs
    on CPU, but subsequent operations run on the Neural Engine.

    Args:
        model: The transformer model to wrap (e.g., Qwen2Model).
        cache_length: Maximum sequence length for KV cache.
        channels_first: Whether to use channels-first format. Defaults to True.
        device: Device for buffer initialization. Defaults to "cpu".

    Attributes:
        layer: The wrapped transformer model.
        cache_length: KV cache sequence length.
        channels_first: Format flag.
        key_cache: Registered buffer for key states.
        value_cache: Registered buffer for value states.
        cos_emb: Pre-computed cosine position embeddings.
        sin_emb: Pre-computed sine position embeddings.
    """

    def __init__(
        self,
        model: nn.Module,
        cache_length: int = 2048,
        channels_first: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        self.layer = model
        self.cache_length = cache_length
        self.channels_first = channels_first

        # Extract config
        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        if "head_dim" in config:
            head_dim = config.head_dim
        else:
            head_dim = config.hidden_size // config.num_attention_heads

        # Initialize KV cache as registered buffers
        # Shape: (num_layers, num_kv_heads, cache_length, head_dim)
        # These become CoreML StateType tensors during conversion
        self.register_buffer(
            "key_cache",
            torch.zeros(
                num_layers,
                num_kv_heads,
                cache_length,
                head_dim,
                device=device,
            ),
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                num_layers,
                num_kv_heads,
                cache_length,
                head_dim,
                device=device,
            ),
        )

        # Pre-compute rotary position embeddings for all positions
        # This avoids FP16 precision issues with exp/cos/sin operations
        # The rotary_emb layer expects: (values, position_ids)
        position_ids = torch.arange(
            cache_length, dtype=torch.long, device=device
        ).unsqueeze(0)
        dummy_values = torch.ones(1, dtype=torch.float32)

        cos_emb, sin_emb = model.rotary_emb(dummy_values, position_ids)

        # Store as buffers: (cache_length, head_dim)
        # Remove batch dimension for indexing
        self.register_buffer("cos_emb", cos_emb[0])
        self.register_buffer("sin_emb", sin_emb[0])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_id: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the wrapped language model.

        Args:
            inputs_embeds: Input embeddings in channels-first format.
                          Shape: (batch, hidden_dim, 1, seq_len).
            position_id: Starting position in the sequence.
                        Shape: (1,) containing the position index.

        Returns:
            Hidden states in channels-first format.
            Shape: (batch, hidden_dim, 1, seq_len).
        """
        # Determine sequence length from input
        if self.channels_first:
            seq_len = inputs_embeds.size(-1)
        else:
            seq_len = inputs_embeds.size(1)

        # Generate position IDs for current input: [position_id, position_id + seq_len)
        position_ids = (
            torch.arange(
                seq_len, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)
            + position_id
        )
        position_ids = position_ids.view(1, -1)

        # Index into pre-computed position embeddings
        # cos_emb/sin_emb: (cache_length, head_dim)
        # position_ids: (1, seq_len)
        # Result: (1, seq_len, head_dim)
        position_emb = (self.cos_emb[position_ids], self.sin_emb[position_ids])

        # Generate causal attention mask
        # Mask positions beyond current position
        # Shape: (1, 1, cache_length) -> broadcast to (batch, 1, cache_length, seq_len)
        attention_mask = (
            torch.arange(self.cache_length, device=inputs_embeds.device)[None, None, :]
            > position_ids[..., None]
        ).float()
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.where(attention_mask == 0, -torch.inf)

        # Handle channels-first format
        if self.channels_first:
            # Transpose embeddings for channels-first attention
            # (1, seq, head_dim) -> (1, head_dim, seq)
            position_emb = (
                position_emb[0],
                position_emb[1],
                position_emb[0].transpose(-1, -2),
                position_emb[1].transpose(-1, -2),
            )
            # (1, 1, cache_len, seq_len) -> (1, 1, seq_len, cache_len)
            attention_mask = attention_mask.transpose(-1, -2)

        # Process through transformer layers
        hidden_states = inputs_embeds

        # Note: For development, you can limit to first layer:
        # for decoder_layer in self.layer.layers[:2]:
        for decoder_layer in self.layer.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                None,  # position_ids not needed - we pass embeddings directly
                (self.key_cache, self.value_cache),
                cache_position=position_id,
                position_embeddings=position_emb,
            )

        # Final layer norm
        hidden_states = self.layer.norm(hidden_states)

        return hidden_states

    def __repr__(self) -> str:
        return (
            f"LanguageModelWrapper("
            f"cache_length={self.cache_length}, "
            f"channels_first={self.channels_first}, "
            f"key_cache_shape={tuple(self.key_cache.shape)}, "
            f"value_cache_shape={tuple(self.value_cache.shape)})"
        )


def create_coreml_state_specs(wrapper: LanguageModelWrapper) -> list:
    """Create CoreML StateType specifications for the wrapper's caches.

    This helper function generates the ct.StateType objects needed for
    CoreML conversion with stateful KV cache.

    Args:
        wrapper: The LanguageModelWrapper instance.

    Returns:
        List of coremltools.StateType objects for key_cache and value_cache.

    Example:
        >>> import coremltools as ct
        >>> wrapper = LanguageModelWrapper(model, cache_length=2048)
        >>> states = create_coreml_state_specs(wrapper)
        >>> mlmodel = ct.convert(traced, ..., states=states)
    """
    # Import here to avoid dependency at module load
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
