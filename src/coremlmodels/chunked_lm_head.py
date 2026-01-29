"""Chunked Language Model Head for Neural Engine Compatibility.

The Neural Engine has a weight dimension limit of ~16384. For large vocabulary
language models, the LM head weights (vocab_size, hidden_dim) may exceed this.

This module chunks the LM head along the vocabulary dimension into smaller pieces,
applies each chunk as a 1x1 convolution, and concatenates the results.

Optionally computes log-sum-exp for each chunk with numerical stability
(subtracting max to prevent float16 overflow on Neural Engine).
"""

import math

import torch
import torch.nn as nn

from .patch_linears import LinearToConv2dPatcher


class ChunkedLMHead(nn.Module):
    """Language model head chunked along vocabulary dimension.

    This chunks the LM head weights to avoid exceeding the Neural Engine's
    weight dimension limit. Each chunk performs a 1x1 convolution on the
    input hidden states, and results are concatenated.

    Optionally computes log-sum-exp for each chunk with numerical stability
    to prevent float16 overflow.

    Args:
        lm_head: Original Linear layer for the LM head.
        chunk_size: Size of vocabulary chunks (default: 6144).
        compute_logsumexp: Whether to compute log-sum-exp for each chunk.

    Example:
        >>> lm_head = nn.Linear(2048, 32000)  # Large vocab
        >>> chunked_head = ChunkedLMHead(lm_head, chunk_size=6144)
        >>> hidden_states = torch.randn(1, 2048, 1, 16)  # (B, D, 1, S)
        >>> temperature = torch.tensor(1.0).view(1, 1, 1, 1)
        >>> logits = chunked_head(hidden_states, temperature)  # (1, 32000, 1, 16)
    """

    def __init__(
        self,
        lm_head: nn.Linear,
        chunk_size: int = 6144,
        compute_logsumexp: bool = False,
    ):
        super().__init__()
        vocab_size, hidden_dim = lm_head.weight.shape
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.compute_logsumexp = compute_logsumexp

        # Calculate number of chunks
        self.num_chunks = math.ceil(vocab_size / chunk_size)

        # Create chunked weights as 1x1 Conv2d layers
        self.chunks = nn.ModuleList()

        for i in range(self.num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, vocab_size)
            chunk_vocab_size = end_idx - start_idx

            # Extract chunk of weights
            chunk_weight = lm_head.weight[start_idx:end_idx].detach()
            chunk_bias = None
            if lm_head.bias is not None:
                chunk_bias = lm_head.bias[start_idx:end_idx].detach()

            # Create temporary Linear layer for this chunk
            temp_linear = nn.Linear(
                hidden_dim,
                chunk_vocab_size,
                bias=(chunk_bias is not None),
            )
            temp_linear.weight.data = chunk_weight
            if chunk_bias is not None:
                temp_linear.bias.data = chunk_bias

            # Convert to Conv2d patcher
            patcher = LinearToConv2dPatcher(temp_linear, bias=(chunk_bias is not None))
            self.chunks.append(patcher)

    def forward(self, hidden_states: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor | tuple:
        """Apply chunked LM head with temperature scaling.

        Args:
            hidden_states: Input hidden states of shape (batch, hidden_dim, 1, seq_len).
            temperature: Temperature for scaling logits, shape (1, 1, 1, 1) or scalar.

        Returns:
            If compute_logsumexp=False:
                logits: Tensor of shape (batch, vocab_size, 1, seq_len)
            If compute_logsumexp=True:
                (logits, chunk_logsumexp_stable, chunk_max): Tuple of tensors
                    - logits: (batch, vocab_size, 1, seq_len)
                    - chunk_logsumexp_stable: (batch, num_chunks, 1, seq_len) - log(sum(exp(x/T - max(x/T))))
                    - chunk_max: (batch, num_chunks, 1, seq_len) - max(x/T) for each chunk
        """
        chunk_logits = []
        chunk_logsumexp_stable_values = []
        chunk_max_values = []

        # Compute inverse temperature once for efficiency (multiply is faster than divide)
        temperature_inv = 1.0 / temperature

        for chunk in self.chunks:
            # Apply chunk: (B, D, 1, S) -> (B, chunk_vocab, 1, S)
            chunk_out = chunk(hidden_states)

            # Apply temperature scaling by multiplying with inverse
            chunk_out_scaled = chunk_out * temperature_inv
            chunk_logits.append(chunk_out_scaled)

            if self.compute_logsumexp:
                # Compute stable log-sum-exp components on temperature-scaled logits
                # Instead of: logsumexp(x/T) = log(sum(exp(x/T - max(x/T)))) + max(x/T)
                # We return the two components separately to allow combining chunks outside
                chunk_max = chunk_out_scaled.max(dim=1, keepdim=True)[0]
                chunk_stable = chunk_out_scaled - chunk_max
                chunk_logsumexp_stable = torch.logsumexp(
                    chunk_stable, dim=1, keepdim=True
                )

                chunk_logsumexp_stable_values.append(chunk_logsumexp_stable)
                chunk_max_values.append(chunk_max)

        # Concatenate all chunks along vocabulary dimension
        logits = torch.cat(chunk_logits, dim=1)

        if self.compute_logsumexp:
            # Concatenate stable logsumexp and max values: (B, num_chunks, 1, S)
            logsumexp_stable = torch.cat(chunk_logsumexp_stable_values, dim=1)
            max_values = torch.cat(chunk_max_values, dim=1)
            return logits, logsumexp_stable, max_values
        else:
            return logits

    def __repr__(self) -> str:
        return (
            f"ChunkedLMHead(vocab_size={self.vocab_size}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_chunks={self.num_chunks}, "
            f"chunk_size={self.chunk_size}, "
            f"compute_logsumexp={self.compute_logsumexp})"
        )
