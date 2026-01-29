"""Utilities for exporting embeddings and LM head to CoreML format.

This module provides functions to export model components separately:
- Embeddings: Exported as .npy files in float16 format
- LM Head: Exported as CoreML models with vocabulary chunking
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

from .chunked_lm_head import ChunkedLMHead
from .graph_passes import register_extended_passes


def export_embeddings(
    model: nn.Module,
    output_path: str | Path,
    verbose: bool = True,
) -> None:
    """Export embedding weights to .npy file in float16.

    Args:
        model: HuggingFace model with embed_tokens attribute.
        output_path: Output path for the .npy file.
        verbose: Print information.
    """
    if not hasattr(model, "embed_tokens"):
        raise ValueError("Model does not have 'embed_tokens' attribute")

    embeddings = model.embed_tokens.weight.detach().cpu().numpy().astype(np.float16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    if verbose:
        print(f"Exported embeddings to: {output_path}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dtype: {embeddings.dtype}")
        print(f"  Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")


def convert_lm_head(
    lm_head: nn.Linear,
    batch_size: int,
    hidden_dim: int,
    seq_len: int,
    output_path: str | Path,
    chunk_size: int = 6144,
    compute_logsumexp: bool = True,
    verbose: bool = True,
    analyze_compute: bool = False,
    analyze_mil: bool = False,
):
    """Convert LM head to CoreML with vocabulary chunking and temperature scaling.

    The Neural Engine has a weight dimension limit of ~16384. This function
    chunks the LM head along the vocabulary dimension to avoid this limit.

    The model accepts temperature as an input for controlling output sharpness:
    - temperature > 1.0: softer distribution (more uniform)
    - temperature = 1.0: unchanged distribution
    - temperature < 1.0: sharper distribution (more peaked)

    Args:
        lm_head: Original Linear layer for the LM head.
        batch_size: Batch size for tracing.
        hidden_dim: Hidden dimension.
        seq_len: Sequence length for tracing.
        output_path: Output path for the CoreML model.
        chunk_size: Size of vocabulary chunks (default: 6144).
        compute_logsumexp: Whether to compute log-sum-exp per chunk.
        verbose: Print detailed information.
        analyze_compute: Run compute plan analysis.
        analyze_mil: Run MIL program inspection.

    Returns:
        Converted CoreML model with inputs:
            - hidden_states: (batch, hidden_dim, 1, seq_len)
            - temperature: (1, 1, 1, 1) - scaling factor for logits
    """
    if verbose:
        print("Converting LM head to CoreML...")
        print(f"  Vocab size: {lm_head.weight.shape[0]}")
        print(f"  Hidden dim: {lm_head.weight.shape[1]}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Compute logsumexp: {compute_logsumexp}")

    # Create chunked LM head (outside inference mode to allow tracing)
    chunked_head = ChunkedLMHead(
        lm_head,
        chunk_size=chunk_size,
        compute_logsumexp=compute_logsumexp,
    )
    chunked_head.eval()

    if verbose:
        print(f"  {chunked_head}")

    # Trace model
    if verbose:
        print("  Tracing LM head...")
    example_input = torch.randn((batch_size, hidden_dim, 1, seq_len), dtype=torch.float32)
    example_temperature = torch.tensor(1.0, dtype=torch.float32).view(1, 1, 1, 1)
    with torch.no_grad():
        traced_lm_head = torch.jit.trace(chunked_head, (example_input, example_temperature))

    # Convert to CoreML
    if verbose:
        print("  Converting to CoreML...")

    register_extended_passes()

    if compute_logsumexp:
        # Three outputs: logits, stable logsumexp, and max values
        mlmodel = ct.convert(
            traced_lm_head,
            inputs=[
                ct.TensorType(
                    shape=(batch_size, hidden_dim, 1, seq_len),
                    name="hidden_states",
                ),
                ct.TensorType(
                    shape=(1, 1, 1, 1),
                    name="temperature",
                ),
            ],
            outputs=[
                ct.TensorType(name="logits"),
                ct.TensorType(name="chunk_logsumexp_stable"),
                ct.TensorType(name="chunk_max"),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )
    else:
        # Single output: logits only
        mlmodel = ct.convert(
            traced_lm_head,
            inputs=[
                ct.TensorType(
                    shape=(batch_size, hidden_dim, 1, seq_len),
                    name="hidden_states",
                ),
                ct.TensorType(
                    shape=(1, 1, 1, 1),
                    name="temperature",
                ),
            ],
            outputs=[
                ct.TensorType(name="logits"),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )

    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    if verbose:
        print(f"  Saved to: {output_path}")

    # Verification - use original lm_head as ground truth
    if verbose:
        print("  Verifying output...")
        test_input = torch.randn((batch_size, hidden_dim, 1, seq_len), dtype=torch.float32)

        # Reference: Use original Linear layer (in float32 for numerical accuracy)
        with torch.no_grad():
            # Reshape from (B, D, 1, S) to (B*S, D) for Linear layer
            test_input_2d = test_input.squeeze(2).transpose(1, 2).reshape(-1, hidden_dim)

            # Compute reference logits using original lm_head in float32
            reference_logits_2d = lm_head(test_input_2d.float())  # (B*S, vocab_size)

            # Reshape back to (B, vocab_size, 1, S) to match chunked output
            reference_logits = reference_logits_2d.reshape(batch_size, seq_len, -1).transpose(1, 2).unsqueeze(2)

            # Compute reference logsumexp from full vocabulary (in float32)
            # This is the ground truth - single logsumexp over entire vocabulary
            if compute_logsumexp:
                reference_logsumexp = torch.logsumexp(reference_logits, dim=1, keepdim=True)

        # CoreML prediction (using temperature=1.0 for verification)
        test_temperature = np.array([[[[1.0]]]], dtype=np.float32)
        coreml_out = mlmodel.predict({
            "hidden_states": test_input.numpy(),
            "temperature": test_temperature,
        })

        if compute_logsumexp:
            coreml_logits = coreml_out["logits"]
            coreml_chunk_logsumexp_stable = coreml_out["chunk_logsumexp_stable"]
            coreml_chunk_max = coreml_out["chunk_max"]

            # Reconstruct full logsumexp from chunked components
            # For each chunk: chunk_lse = chunk_logsumexp_stable + chunk_max
            # Then: full_lse = logsumexp(all_chunk_lse)
            coreml_chunk_lse = coreml_chunk_logsumexp_stable + coreml_chunk_max  # (B, num_chunks, 1, S)

            # Apply logsumexp over chunks (dim=1) to get final logsumexp
            coreml_chunk_lse_torch = torch.from_numpy(coreml_chunk_lse)
            coreml_full_logsumexp = torch.logsumexp(coreml_chunk_lse_torch, dim=1, keepdim=True).numpy()

            print(f"    Logits shape: Reference={reference_logits.shape}, CoreML={coreml_logits.shape}")
            print(f"    Logsumexp shape: Reference={reference_logsumexp.shape}, CoreML={coreml_full_logsumexp.shape}")

            # Check logits against ground truth (original Linear layer)
            logits_diff = np.abs(reference_logits.numpy() - coreml_logits).max()
            print(f"    Max logits difference (vs ground truth): {logits_diff:.6f}")

            # Check logsumexp against ground truth
            lse_diff = np.abs(reference_logsumexp.numpy() - coreml_full_logsumexp).max()
            print(f"    Max logsumexp difference (vs ground truth): {lse_diff:.6f}")
            print(f"    (Logsumexp reconstructed from {chunked_head.num_chunks} chunks)")
        else:
            coreml_logits = coreml_out["logits"]

            print(f"    Logits shape: Reference={reference_logits.shape}, CoreML={coreml_logits.shape}")
            logits_diff = np.abs(reference_logits.numpy() - coreml_logits).max()
            print(f"    Max logits difference (vs ground truth): {logits_diff:.6f}")

    # Optional analysis
    if analyze_compute or analyze_mil:
        # Import here to avoid circular dependencies
        from .analysis import analyze_compute_plan, inspect_mil_program

        if analyze_compute:
            print("\n  Compute Plan Analysis")
            print("  Looking for Neural Engine scheduling...")
            analyze_compute_plan(mlmodel)

        if analyze_mil:
            print("\n  MIL Program Inspection")
            print("  Looking for conv operations...")
            inspect_mil_program(mlmodel)

    return mlmodel
