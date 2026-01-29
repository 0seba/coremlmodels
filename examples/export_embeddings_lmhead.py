"""Example: Exporting Embeddings and LM Head.

This example demonstrates how to export embeddings and the language model head
separately from the main model conversion.

Embeddings are exported as .npy files in float16 format, while the LM head is
exported as a CoreML model with vocabulary chunking to handle the Neural Engine's
weight dimension limit (~16384).

Note: You can also use lm_conversion_example.py with --components-only flag for
faster development iteration:
    uv run python examples/lm_conversion_example.py --export-embeddings --export-lm-head --components-only

Usage:
    # Export both embeddings and LM head
    uv run python examples/export_embeddings_lmhead.py

    # Export only embeddings
    uv run python examples/export_embeddings_lmhead.py --embeddings-only

    # Export only LM head
    uv run python examples/export_embeddings_lmhead.py --lmhead-only

    # Use different model
    uv run python examples/export_embeddings_lmhead.py --model Qwen/Qwen3-0.6B

    # Customize LM head chunk size
    uv run python examples/export_embeddings_lmhead.py --chunk-size 8192
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from coremlmodels import convert_lm_head, export_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Export embeddings and LM head from HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model name (default: Qwen/Qwen2-0.5B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_components",
        help="Output directory (default: exported_components)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=6144,
        help="LM head vocabulary chunk size (default: 6144)",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Export only embeddings",
    )
    parser.add_argument(
        "--lmhead-only",
        action="store_true",
        help="Export only LM head",
    )

    args = parser.parse_args()

    export_both = not args.embeddings_only and not args.lmhead_only
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("EXPORTING MODEL COMPONENTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")

    # Load config
    config = AutoConfig.from_pretrained(args.model)

    # Export embeddings
    if args.embeddings_only or export_both:
        print("\n[1] Loading model for embeddings...")
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()

        embeddings_path = output_dir / "embeddings.npy"
        export_embeddings(model, embeddings_path)

    # Export LM head
    if args.lmhead_only or export_both:
        print("\n[2] Loading causal LM model for LM head...")
        causal_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        causal_model.eval()

        if not hasattr(causal_model, "lm_head"):
            print("[ERROR] Model does not have 'lm_head' attribute")
            return

        lm_head_path = output_dir / "lm_head.mlpackage"
        convert_lm_head(
            lm_head=causal_model.lm_head,
            batch_size=1,
            hidden_dim=config.hidden_size,
            seq_len=8,
            output_path=lm_head_path,
            chunk_size=args.chunk_size,
            compute_logsumexp=True,
            verbose=True,
        )

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    if args.embeddings_only or export_both:
        print(f"Embeddings: {output_dir / 'embeddings.npy'}")
    if args.lmhead_only or export_both:
        print(f"LM head: {output_dir / 'lm_head.mlpackage'}")


if __name__ == "__main__":
    main()
