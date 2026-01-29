"""Tests for ChunkedLMHead module."""

import pytest
import torch
import torch.nn as nn

from coremlmodels import ChunkedLMHead


class TestChunkedLMHead:
    """Test ChunkedLMHead functionality."""

    def test_chunked_lm_head_output_shape(self):
        """Test that chunked LM head produces correct output shape."""
        hidden_dim = 512
        vocab_size = 10000
        chunk_size = 3000
        batch_size = 2
        seq_len = 16

        # Create original LM head
        lm_head = nn.Linear(hidden_dim, vocab_size, bias=True)

        # Create chunked version
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size, compute_logsumexp=False)

        # Test input (channels-first format)
        input_tensor = torch.randn(batch_size, hidden_dim, 1, seq_len)
        temperature = torch.tensor(1.0).view(1, 1, 1, 1)

        # Forward pass
        output = chunked_head(input_tensor, temperature)

        # Check output shape
        assert output.shape == (batch_size, vocab_size, 1, seq_len)

    def test_chunked_lm_head_with_logsumexp(self):
        """Test chunked LM head with logsumexp computation."""
        hidden_dim = 256
        vocab_size = 5000
        chunk_size = 1500
        batch_size = 1
        seq_len = 8

        lm_head = nn.Linear(hidden_dim, vocab_size, bias=True)
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size, compute_logsumexp=True)

        input_tensor = torch.randn(batch_size, hidden_dim, 1, seq_len)
        temperature = torch.tensor(1.0).view(1, 1, 1, 1)

        # Forward pass with logsumexp (returns 3 outputs)
        logits, logsumexp_stable, chunk_max = chunked_head(input_tensor, temperature)

        # Check shapes
        assert logits.shape == (batch_size, vocab_size, 1, seq_len)

        # Number of chunks should be ceil(vocab_size / chunk_size)
        expected_num_chunks = (vocab_size + chunk_size - 1) // chunk_size
        assert logsumexp_stable.shape == (batch_size, expected_num_chunks, 1, seq_len)
        assert chunk_max.shape == (batch_size, expected_num_chunks, 1, seq_len)

    def test_chunked_lm_head_matches_original(self):
        """Test that chunked LM head produces similar output to original."""
        hidden_dim = 128
        vocab_size = 1000
        chunk_size = 300
        batch_size = 1
        seq_len = 4

        # Create original LM head
        lm_head = nn.Linear(hidden_dim, vocab_size, bias=True)

        # Create chunked version
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size, compute_logsumexp=False)

        # Test input
        input_2d = torch.randn(batch_size, hidden_dim)  # 2D input for original
        input_4d = input_2d.view(batch_size, hidden_dim, 1, 1).expand(batch_size, hidden_dim, 1, seq_len)
        temperature = torch.tensor(1.0).view(1, 1, 1, 1)

        # Original forward (2D)
        original_output = lm_head(input_2d)

        # Chunked forward (4D)
        chunked_output = chunked_head(input_4d, temperature)

        # Compare first position (should be very close)
        # Original: (batch, vocab), Chunked: (batch, vocab, 1, seq)
        original_logits = original_output.unsqueeze(2).unsqueeze(3)  # (batch, vocab, 1, 1)
        chunked_logits_first = chunked_output[:, :, :, 0:1]  # (batch, vocab, 1, 1)

        # Should be very close (allow small numerical differences)
        max_diff = torch.abs(original_logits - chunked_logits_first).max().item()
        assert max_diff < 1e-5, f"Max difference: {max_diff}"

    def test_chunked_lm_head_no_bias(self):
        """Test chunked LM head works without bias."""
        hidden_dim = 256
        vocab_size = 2000
        chunk_size = 600

        lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size, compute_logsumexp=False)

        input_tensor = torch.randn(1, hidden_dim, 1, 8)
        temperature = torch.tensor(1.0).view(1, 1, 1, 1)
        output = chunked_head(input_tensor, temperature)

        assert output.shape == (1, vocab_size, 1, 8)

    def test_chunked_lm_head_num_chunks(self):
        """Test that number of chunks is calculated correctly."""
        hidden_dim = 512
        vocab_size = 10000
        chunk_size = 3000

        lm_head = nn.Linear(hidden_dim, vocab_size)
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size)

        # Expected: ceil(10000 / 3000) = 4 chunks
        assert chunked_head.num_chunks == 4
        assert len(chunked_head.chunks) == 4

    def test_chunked_lm_head_exact_division(self):
        """Test when vocab_size divides evenly by chunk_size."""
        hidden_dim = 256
        vocab_size = 6000
        chunk_size = 2000

        lm_head = nn.Linear(hidden_dim, vocab_size)
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size)

        # Should have exactly 3 chunks
        assert chunked_head.num_chunks == 3
        assert len(chunked_head.chunks) == 3

        input_tensor = torch.randn(1, hidden_dim, 1, 4)
        temperature = torch.tensor(1.0).view(1, 1, 1, 1)
        output = chunked_head(input_tensor, temperature)
        assert output.shape == (1, vocab_size, 1, 4)

    def test_logsumexp_numerical_stability(self):
        """Test that logsumexp computation is numerically stable."""
        hidden_dim = 128
        vocab_size = 1000
        chunk_size = 250

        lm_head = nn.Linear(hidden_dim, vocab_size, bias=True)
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size, compute_logsumexp=True)

        # Create input with large values to test stability
        input_tensor = torch.randn(1, hidden_dim, 1, 4) * 10.0
        temperature = torch.tensor(1.0).view(1, 1, 1, 1)

        # Should not produce inf/nan values
        logits, logsumexp_stable, chunk_max = chunked_head(input_tensor, temperature)

        assert not torch.isnan(logits).any(), "Logits contain NaN"
        assert not torch.isinf(logits).any(), "Logits contain Inf"
        assert not torch.isnan(logsumexp_stable).any(), "Logsumexp stable contains NaN"
        assert not torch.isinf(logsumexp_stable).any(), "Logsumexp stable contains Inf"
        assert not torch.isnan(chunk_max).any(), "Chunk max contains NaN"
        assert not torch.isinf(chunk_max).any(), "Chunk max contains Inf"

    def test_temperature_scaling(self):
        """Test that temperature scaling works correctly."""
        hidden_dim = 128
        vocab_size = 1000
        chunk_size = 300
        batch_size = 1
        seq_len = 4

        lm_head = nn.Linear(hidden_dim, vocab_size, bias=True)
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size, compute_logsumexp=False)

        input_tensor = torch.randn(batch_size, hidden_dim, 1, seq_len)

        # Test with temperature=1.0
        temp_1 = torch.tensor(1.0).view(1, 1, 1, 1)
        logits_temp_1 = chunked_head(input_tensor, temp_1)

        # Test with temperature=2.0
        temp_2 = torch.tensor(2.0).view(1, 1, 1, 1)
        logits_temp_2 = chunked_head(input_tensor, temp_2)

        # With temperature=2.0, logits should be half of temperature=1.0
        expected_logits_temp_2 = logits_temp_1 / 2.0
        max_diff = torch.abs(logits_temp_2 - expected_logits_temp_2).max().item()
        assert max_diff < 1e-5, f"Temperature scaling incorrect, max diff: {max_diff}"

        # Test with temperature=0.5 (sharper distribution)
        temp_half = torch.tensor(0.5).view(1, 1, 1, 1)
        logits_temp_half = chunked_head(input_tensor, temp_half)

        # With temperature=0.5, logits should be double of temperature=1.0
        expected_logits_temp_half = logits_temp_1 * 2.0
        max_diff_half = torch.abs(logits_temp_half - expected_logits_temp_half).max().item()
        assert max_diff_half < 1e-5, f"Temperature scaling incorrect, max diff: {max_diff_half}"

    def test_repr(self):
        """Test string representation."""
        hidden_dim = 256
        vocab_size = 5000
        chunk_size = 1500

        lm_head = nn.Linear(hidden_dim, vocab_size)
        chunked_head = ChunkedLMHead(lm_head, chunk_size=chunk_size, compute_logsumexp=True)

        repr_str = repr(chunked_head)
        assert "ChunkedLMHead" in repr_str
        assert "vocab_size=5000" in repr_str
        assert "hidden_dim=256" in repr_str
        assert "num_chunks=4" in repr_str
        assert "chunk_size=1500" in repr_str
        assert "compute_logsumexp=True" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
