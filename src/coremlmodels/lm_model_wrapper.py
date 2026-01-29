"""Language Model Wrapper for CoreML Conversion with KV Cache State.

This module provides wrappers for transformer language models that:
1. Pre-computes position embeddings at initialization (avoids FP16 precision issues)
2. Manages KV cache as registered buffers (converted to CoreML state)
3. Uses channels-first format (NCHW) for Neural Engine optimization
4. Handles position indexing at the start of forward (CPU op, then ANE graph)

The wrappers are designed to be traced with torch.jit and converted to CoreML
with ct.StateType for the KV cache tensors.

This module provides two wrapper classes:
- LanguageModelWrapper: Wraps the entire model with full KV cache
- ChunkedLanguageModelWrapper: Wraps a subset of layers (chunk) with its own KV cache
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# Helper Functions for Position Embeddings and Attention Masks
# ============================================================================


def _get_sequence_length(hidden_states: torch.Tensor, channels_first: bool) -> int:
    """Extract sequence length from hidden states tensor.

    Args:
        hidden_states: Input tensor.
        channels_first: If True, expects (batch, channels, 1, seq_len) format.
                       If False, expects (batch, seq_len, channels) format.

    Returns:
        Sequence length as integer.
    """
    return hidden_states.size(-1) if channels_first else hidden_states.size(1)


def _generate_position_ids(
    seq_len: int,
    position_id: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate position IDs for the current input sequence.

    Creates a tensor [position_id, position_id+1, ..., position_id+seq_len-1].

    Args:
        seq_len: Length of the current sequence.
        position_id: Starting position index (shape: (1,)).
        device: Device to create the tensor on.

    Returns:
        Position IDs tensor of shape (1, seq_len).
    """
    position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        + position_id
    )
    return position_ids.view(1, -1)


def _index_position_embeddings(
    cos_emb: torch.Tensor,
    sin_emb: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index into pre-computed position embeddings.

    Args:
        cos_emb: Pre-computed cosine embeddings (cache_length, head_dim).
        sin_emb: Pre-computed sine embeddings (cache_length, head_dim).
        position_ids: Position indices to extract (1, seq_len).

    Returns:
        Tuple of (cos_emb[position_ids], sin_emb[position_ids]).
        Each has shape (1, seq_len, head_dim).
    """
    return (cos_emb[position_ids], sin_emb[position_ids])


def _generate_causal_mask(
    cache_length: int,
    position_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate causal attention mask for autoregressive generation.

    Creates a mask where positions beyond the current position are masked out.

    Args:
        cache_length: Maximum sequence length (KV cache size).
        position_ids: Current position indices (1, seq_len).
        device: Device to create the tensor on.

    Returns:
        Attention mask of shape (1, 1, 1, cache_length).
        Masked positions contain -inf, unmasked positions contain 0.
    """
    # Create mask: cache_positions > position_ids
    # Shape: (1, 1, cache_length)
    attention_mask = (
        torch.arange(cache_length, device=device)[None, None, :]
        > position_ids[..., None]
    ).float()

    # Add extra dimension for attention heads: (1, 1, 1, cache_length)
    attention_mask = attention_mask.unsqueeze(1)

    # Replace 1s with -inf, keep 0s as 0
    attention_mask = attention_mask.where(attention_mask == 0, -torch.inf)

    return attention_mask


def _apply_channels_first_transforms(
    position_emb: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    """Apply transformations for channels-first format.

    Transposes position embeddings and attention mask to match
    channels-first (NCHW) tensor layout.

    Args:
        position_emb: Tuple of (cos, sin) embeddings with shape (1, seq, head_dim).
        attention_mask: Attention mask with shape (1, 1, 1, cache_len).

    Returns:
        Tuple of (transformed_position_emb, transformed_attention_mask).
        - position_emb becomes (cos, sin, cos_t, sin_t) where _t denotes transposed
        - attention_mask becomes (1, 1, seq_len, cache_len)
    """
    # Transpose embeddings: (1, seq, head_dim) -> (1, head_dim, seq)
    position_emb = (
        position_emb[0],
        position_emb[1],
        position_emb[0].transpose(-1, -2),
        position_emb[1].transpose(-1, -2),
    )

    # Transpose attention mask: (1, 1, 1, cache_len) -> (1, 1, seq_len, cache_len)
    # Wait, the input mask is (1, 1, 1, cache_len), need to check the actual shape
    # Looking at the code, it's (1, 1, cache_len, seq_len) that gets transposed to (1, 1, seq_len, cache_len)
    attention_mask = attention_mask.transpose(-1, -2)

    return position_emb, attention_mask


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
        # Check for explicit head_dim first (Qwen3 uses explicit head_dim != hidden_size // num_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

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
        # Extract sequence length
        seq_len = _get_sequence_length(inputs_embeds, self.channels_first)

        # Generate position IDs for current input
        position_ids = _generate_position_ids(seq_len, position_id, inputs_embeds.device)

        # Index into pre-computed position embeddings
        position_emb = _index_position_embeddings(self.cos_emb, self.sin_emb, position_ids)

        # Generate causal attention mask
        attention_mask = _generate_causal_mask(
            self.cache_length, position_ids, inputs_embeds.device
        )

        # Apply channels-first transformations if needed
        if self.channels_first:
            position_emb, attention_mask = _apply_channels_first_transforms(
                position_emb, attention_mask
            )

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


class ChunkedLanguageModelWrapper(nn.Module):
    """Wrapper for a chunk of transformer layers optimized for CoreML conversion.

    This wrapper handles a subset of transformer layers (a "chunk") for models
    that exceed the ~2GB Neural Engine limit. Each chunk:
    - Has its own KV cache sized for its layer count
    - Uses local layer indices (0 to num_chunk_layers-1)
    - Receives and outputs hidden states in channels-first format

    When converting large models, create multiple ChunkedLanguageModelWrapper
    instances, each wrapping a different slice of the model's layers.

    Args:
        layers: List of decoder layer modules for this chunk.
        config: Model configuration object.
        chunk_idx: Index of this chunk (0, 1, 2, ...).
        cache_length: Maximum sequence length for KV cache.
        cos_emb: Pre-computed cosine position embeddings (cache_length, head_dim).
        sin_emb: Pre-computed sine position embeddings (cache_length, head_dim).
        channels_first: Whether to use channels-first format. Defaults to True.
        is_first_chunk: Whether this is the first chunk (no pre-processing needed).
        is_last_chunk: Whether this is the last chunk (applies final layer norm).
        final_norm: Final layer norm module (only needed for last chunk).
        device: Device for buffer initialization. Defaults to "cpu".

    Attributes:
        layers: List of decoder layer modules.
        chunk_idx: Chunk index for naming.
        num_layers: Number of layers in this chunk.
        key_cache: Registered buffer for key states.
        value_cache: Registered buffer for value states.
    """

    def __init__(
        self,
        layers: List[nn.Module],
        config: object,
        chunk_idx: int,
        cache_length: int,
        cos_emb: torch.Tensor,
        sin_emb: torch.Tensor,
        channels_first: bool = True,
        is_first_chunk: bool = False,
        is_last_chunk: bool = False,
        final_norm: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.chunk_idx = chunk_idx
        self.num_layers = len(layers)
        self.cache_length = cache_length
        self.channels_first = channels_first
        self.is_first_chunk = is_first_chunk
        self.is_last_chunk = is_last_chunk
        self.final_norm = final_norm

        # Extract config
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        # Initialize KV cache for this chunk's layers only
        # Shape: (num_chunk_layers, num_kv_heads, cache_length, head_dim)
        self.register_buffer(
            "key_cache",
            torch.zeros(
                self.num_layers,
                num_kv_heads,
                cache_length,
                head_dim,
                device=device,
            ),
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                self.num_layers,
                num_kv_heads,
                cache_length,
                head_dim,
                device=device,
            ),
        )

        # Store pre-computed position embeddings
        self.register_buffer("cos_emb", cos_emb)
        self.register_buffer("sin_emb", sin_emb)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_id: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """Forward pass through this chunk's layers.

        Args:
            hidden_states: Input in channels-first format (batch, hidden_dim, 1, seq_len).
            position_id: Starting position in the sequence. Shape: (1,).
            debug: If True, print tensor shapes at each step.

        Returns:
            Hidden states in channels-first format (batch, hidden_dim, 1, seq_len).
        """
        if debug:
            print(f"[Chunk {self.chunk_idx}] Input hidden_states shape: {hidden_states.shape}")

        # Extract sequence length
        seq_len = _get_sequence_length(hidden_states, self.channels_first)

        # Generate position IDs for current input
        position_ids = _generate_position_ids(seq_len, position_id, hidden_states.device)

        # Index into pre-computed position embeddings
        position_emb = _index_position_embeddings(self.cos_emb, self.sin_emb, position_ids)

        # Generate causal attention mask
        attention_mask = _generate_causal_mask(
            self.cache_length, position_ids, hidden_states.device
        )

        # Apply channels-first transformations if needed
        if self.channels_first:
            position_emb, attention_mask = _apply_channels_first_transforms(
                position_emb, attention_mask
            )

        # Process through this chunk's layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            if debug:
                print(f"[Chunk {self.chunk_idx}] Before layer {layer_idx}: {hidden_states.shape}")

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                None,  # position_ids not needed - we pass embeddings directly
                (self.key_cache, self.value_cache),
                cache_position=position_id,
                position_embeddings=position_emb,
            )

            if debug:
                print(f"[Chunk {self.chunk_idx}] After layer {layer_idx}: {hidden_states.shape}")

        # Apply final layer norm only for last chunk
        if self.is_last_chunk and self.final_norm is not None:
            if debug:
                print(f"[Chunk {self.chunk_idx}] Before final_norm: {hidden_states.shape}")
                print(f"[Chunk {self.chunk_idx}] final_norm type: {type(self.final_norm).__name__}")
                if hasattr(self.final_norm, 'weight'):
                    print(f"[Chunk {self.chunk_idx}] final_norm weight shape: {self.final_norm.weight.shape}")

            hidden_states = self.final_norm(hidden_states)

            if debug:
                print(f"[Chunk {self.chunk_idx}] After final_norm: {hidden_states.shape}")

        if debug:
            print(f"[Chunk {self.chunk_idx}] Output hidden_states shape: {hidden_states.shape}")

        return hidden_states

    def __repr__(self) -> str:
        return (
            f"ChunkedLanguageModelWrapper("
            f"chunk_idx={self.chunk_idx}, "
            f"num_layers={self.num_layers}, "
            f"cache_length={self.cache_length}, "
            f"channels_first={self.channels_first}, "
            f"is_first_chunk={self.is_first_chunk}, "
            f"is_last_chunk={self.is_last_chunk}, "
            f"key_cache_shape={tuple(self.key_cache.shape)})"
        )


def create_chunked_coreml_state_specs(wrapper: ChunkedLanguageModelWrapper) -> list:
    """Create CoreML StateType specifications for a chunked wrapper's caches.

    Args:
        wrapper: The ChunkedLanguageModelWrapper instance.

    Returns:
        List of coremltools.StateType objects for this chunk's key_cache and value_cache.
        Names match the PyTorch buffer names (key_cache, value_cache).

    Example:
        >>> chunk_wrapper = ChunkedLanguageModelWrapper(layers[:8], config, chunk_idx=0, ...)
        >>> states = create_chunked_coreml_state_specs(chunk_wrapper)
        >>> mlmodel = ct.convert(traced, ..., states=states)
    """
    import coremltools as ct

    # State names must match the PyTorch buffer names exactly
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
