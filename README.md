# coremlmodels

**Convert PyTorch language models to CoreML format.**

This library provides utilities to convert HuggingFace transformer models to CoreML, with support for stateful KV caching, chunked models for large architectures, and pre-compiled model caching for faster inference.

## Installation

```bash
uv sync
```

## Pre-converted Models

| Model | Input Length | Context Length | Link |
|-------|--------------|----------------|------|
| Qwen3-1.7B | 8 | 2048 | [seba/Qwen3-1.7B-CoreML-input-8-ctx-2048](https://huggingface.co/seba/Qwen3-1.7B-CoreML-input-8-ctx-2048) |
| Qwen3-4B-Instruct-2507 | 8 | 2048 | [seba/Qwen3-4B-Instruct-2507-CoreML-input-8-ctx-2048](https://huggingface.co/seba/Qwen3-4B-Instruct-2507-CoreML-input-8-ctx-2048) |

## Model Conversion

Convert a model with embeddings and LM head export:

```bash
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-1.7B --output qwen3_1.7b --num-chunks 2 --export-embeddings --export-lm-head --cache-compiled
```

For large models, convert chunks individually to reduce memory usage:

```bash
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B-Instruct-2507 --output qwen3_4b_instruct_2507 --num-chunks 4 --chunk-index 2 --skip-model-load
```

GLM-OCR conversion:

```bash
uv run python examples/glm_ocr_text_conversion.py --export-lm-head --export-embeddings
uv run python examples/vision_conversion_example.py
uv run python examples/glm_ocr_mtp_conversion.py
```

## Inference

Run inference with a converted model:

```bash
uv run python examples/inference.py --model-dir ./qwen3_4b_instruct_2507/ --model-name Qwen/Qwen3-4B-Instruct-2507 --max-new-tokens 2048 --chunked --num-chunks 4 --cache-compiled
```

GLM-OCR CoreML inference:

```bash
uv run python examples/glm_ocr_coreml_inference.py \
  --image ./assets/realworld.png \
  --vision-model ./glm_ocr_vision.mlpackage \
  --text-model ./glm_ocr_text_seqlen_8.mlpackage \
  --lm-head ./glm_ocr_lm_head.mlpackage \
  --embeddings ./glm_ocr_embeddings.npy --cache-compiled --stream
```

GLM-OCR with MTP speculative decoding (~2x faster):

```bash
uv run python examples/glm_ocr_coreml_inference.py \
  --image ./assets/realworld.png \
  --vision-model ./glm_ocr_vision.mlpackage \
  --text-model ./glm_ocr_text_seqlen_8.mlpackage \
  --lm-head ./glm_ocr_lm_head.mlpackage \
  --embeddings ./glm_ocr_embeddings.npy \
  --mtp-model ./glm_ocr_mtp_seqlen_1.mlpackage \
  --num-spec-steps 3 --cache-compiled --stream
```

## Supported Architectures

- Qwen2
- Qwen3

## Limitations

- **Fixed cache length**: KV cache size is set at conversion time and cannot be changed at runtime
- **Fixed sequence length**: Input sequence length is fixed for both prompt processing and token generation. CoreML multifunction models can address this by providing separate functions for different sequence lengths
- **Model size limit (~2GB)**: Neural Engine can only load models up to ~2GB, requiring chunked conversion for larger models
- **FP16 precision**: Computations run in FP16, which may affect numerical precision for some operations

## Documentation

For technical details, implementation guides, and development workflows, see [docs/AGENTS.md](docs/AGENTS.md).

Additional documentation:
- [CONVERSION_GUIDE.md](docs/CONVERSION_GUIDE.md) - Detailed conversion options
- [INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) - Inference configuration

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint code
uv run ruff check .
```
