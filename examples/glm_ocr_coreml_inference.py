"""Run OCR with converted GLM-OCR CoreML models (no layout preprocessing).

Example:
    uv run python examples/glm_ocr_coreml_inference.py \
        --image ./page.png \
        --vision-model ./glm_ocr_vision.mlpackage \
        --text-model ./glm_ocr_text_seqlen_8.mlpackage \
        --lm-head ./glm_ocr_lm_head.mlpackage \
        --embeddings ./glm_ocr_embeddings.npy \
        --cache-compiled

    # With MTP speculative decoding:
    uv run python examples/glm_ocr_coreml_inference.py \
        --image ./page.png \
        --vision-model ./glm_ocr_vision.mlpackage \
        --text-model ./glm_ocr_text_seqlen_8.mlpackage \
        --lm-head ./glm_ocr_lm_head.mlpackage \
        --embeddings ./glm_ocr_embeddings.npy \
        --mtp-model ./glm_ocr_mtp_seqlen_1.mlpackage \
        --num-spec-steps 3 \
        --cache-compiled
"""

import argparse
import time
from pathlib import Path

from PIL import Image

from coremlmodels import GenerationConfig, GlmOcrCoreMLPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-OCR CoreML inference (image -> text).")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--vision-model",
        type=str,
        required=True,
        help="Path to converted vision .mlpackage.",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        required=True,
        help="Path to converted text decoder .mlpackage.",
    )
    parser.add_argument(
        "--lm-head",
        type=str,
        required=True,
        help="Path to converted LM head .mlpackage.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to exported embeddings .npy.",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="zai-org/GLM-OCR",
        help="HF model for processor/config (default: zai-org/GLM-OCR).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Extract all text from this image.",
        help="Instruction prompt passed with the image.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print generated OCR text incrementally as tokens are produced",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--max-vision-patches",
        type=int,
        default=None,
        help="Optional cap for vision patches. If omitted, auto-detected from "
        "vision model enumerated shapes.",
    )
    parser.add_argument(
        "--resized-image-output",
        type=str,
        default=None,
        help="Where to save resized image copy when input exceeds patch budget "
        "(default: <image_stem>_resized_for_coreml.png)",
    )
    parser.add_argument(
        "--cache-compiled",
        action="store_true",
        help="Cache compiled CoreML models (.mlmodelc) for faster subsequent loads",
    )
    parser.add_argument(
        "--mtp-model",
        type=str,
        default=None,
        help="Path to MTP .mlpackage for speculative decoding (optional).",
    )
    parser.add_argument(
        "--num-spec-steps",
        type=int,
        default=3,
        help="Number of MTP forward passes per speculation round (default: 3).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-step decode debug info: tokens, MTP drafts, and timing.",
    )

    args = parser.parse_args()

    pipeline = GlmOcrCoreMLPipeline(
        vision_model_path=Path(args.vision_model),
        text_model_path=Path(args.text_model),
        lm_head_path=Path(args.lm_head),
        embeddings_path=Path(args.embeddings),
        mtp_model_path=Path(args.mtp_model) if args.mtp_model else None,
        num_spec_steps=args.num_spec_steps,
        hf_model_name=args.hf_model,
        cache_compiled=args.cache_compiled,
        max_vision_patches=args.max_vision_patches,
    )

    generation_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    t_image_load = time.perf_counter()
    image = Image.open(args.image).convert("RGB")
    image_load_ms = (time.perf_counter() - t_image_load) * 1000.0
    resized_image_output = (
        Path(args.resized_image_output)
        if args.resized_image_output
        else Path(args.image).with_name(
            f"{Path(args.image).stem}_resized_for_coreml.png"
        )
    )
    text, timing = pipeline.run(
        image=image,
        prompt=args.prompt,
        generation_config=generation_cfg,
        resized_image_output_path=resized_image_output,
        stream_output=args.stream,
        debug=args.debug,
    )
    if args.stream:
        print()
    else:
        print(text)

    # Print timing summary
    print(f"\n--- Performance ---")
    print(f"Image load:          {image_load_ms:8.1f} ms")
    print(f"Preprocessing:       {timing.preprocessing_ms:8.1f} ms")
    print(f"Vision encoder:      {timing.vision_encode_ms:8.1f} ms")
    print(f"Prefill:             {timing.prefill_ms:8.1f} ms")
    if timing.num_tokens_generated > 0:
        print(
            f"Decode:              {timing.decode_ms:8.1f} ms"
            f"  ({timing.num_tokens_generated} tokens, {timing.decode_tokens_per_sec:.1f} tok/s)"
        )
    else:
        print(f"Decode:              {timing.decode_ms:8.1f} ms  (0 tokens)")
    if timing.mtp_prefill_positions > 0:
        avg_pf = timing.mtp_prefill_ms / timing.mtp_prefill_positions
        print(
            f"  MTP prefill:       {timing.mtp_prefill_ms:8.1f} ms"
            f"  ({timing.mtp_prefill_positions} positions, {avg_pf:.1f} ms/pos)"
        )
    if timing.mtp_draft_steps > 0:
        print(f"  MTP draft:         {timing.mtp_avg_step_ms:8.1f} ms/step  ({timing.mtp_draft_steps} steps total)")
        print(
            f"  Spec verify:       {timing.spec_avg_verify_ms:8.1f} ms/forward  ({timing.spec_verify_steps} forwards)"
        )
        print(
            f"  Acceptance rate:   {timing.spec_acceptance_rate:8.1f} drafts/forward"
            f"  ({timing.drafts_accepted}/{timing.drafts_proposed})"
        )
    print(f"Total:               {image_load_ms + timing.total_ms:8.1f} ms")


if __name__ == "__main__":
    main()
