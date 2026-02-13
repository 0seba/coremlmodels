"""Convert GLM-OCR MTP module to CoreML.

The MTP (Multi-Token Prediction) module is a single transformer layer
(layer index 16) that enables speculative decoding. It processes one
token at a time, maintaining its own KV cache separate from the main
text model's 16-layer cache.

Usage:
    # Full conversion with verification
    uv run python examples/glm_ocr_mtp_conversion.py

    # Skip model load (save-only, useful for older macOS)
    uv run python examples/glm_ocr_mtp_conversion.py --skip-model-load

    # Custom output path
    uv run python examples/glm_ocr_mtp_conversion.py --output glm_ocr_mtp.mlpackage

    # Overwrite existing output
    uv run python examples/glm_ocr_mtp_conversion.py --overwrite
"""

import argparse

from coremlmodels import convert_glm_ocr_mtp


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLM-OCR MTP module to CoreML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="zai-org/GLM-OCR",
        help="HuggingFace model name (default: zai-org/GLM-OCR)",
    )
    parser.add_argument(
        "--cache-length",
        type=int,
        default=2048,
        help="Maximum KV cache length (default: 2048)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the CoreML model",
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip loading the model after conversion (no verification)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity",
    )
    parser.add_argument(
        "--analyze-compute-plan",
        action="store_true",
        help="Run compute plan analysis to check Neural Engine scheduling",
    )
    parser.add_argument(
        "--analyze-mil",
        action="store_true",
        help="Run MIL program inspection to check for conv, layer_norm operations",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help="Run N timed iterations after conversion to measure latency",
    )

    args = parser.parse_args()

    convert_glm_ocr_mtp(
        model_name=args.model,
        seq_len=1,  # MTP processes one token at a time
        cache_length=args.cache_length,
        batch_size=1,
        output_path=args.output,
        verbose=not args.quiet,
        skip_model_load=args.skip_model_load,
        overwrite=args.overwrite,
        analyze_compute=args.analyze_compute_plan,
        analyze_mil=args.analyze_mil,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
