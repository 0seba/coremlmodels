"""Run OCR with converted GLM-OCR CoreML models (no layout preprocessing).

Example:
    uv run python examples/glm_ocr_coreml_inference.py \
        --image ./page.png \
        --vision-model ./glm_ocr_vision.mlpackage \
        --text-model ./glm_ocr_text_seqlen_8.mlpackage \
        --lm-head ./glm_ocr_lm_head.mlpackage \
        --embeddings ./glm_ocr_embeddings.npy \
        --cache-compiled
"""

import argparse
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

    args = parser.parse_args()

    pipeline = GlmOcrCoreMLPipeline(
        vision_model_path=Path(args.vision_model),
        text_model_path=Path(args.text_model),
        lm_head_path=Path(args.lm_head),
        embeddings_path=Path(args.embeddings),
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

    image = Image.open(args.image).convert("RGB")
    resized_image_output = (
        Path(args.resized_image_output)
        if args.resized_image_output
        else Path(args.image).with_name(
            f"{Path(args.image).stem}_resized_for_coreml.png"
        )
    )
    text = pipeline.run(
        image=image,
        prompt=args.prompt,
        generation_config=generation_cfg,
        resized_image_output_path=resized_image_output,
        stream_output=args.stream,
    )
    if args.stream:
        print()
    else:
        print(text)


if __name__ == "__main__":
    main()
