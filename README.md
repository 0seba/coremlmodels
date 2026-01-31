# coremlmodels

**Convert PyTorch language models to CoreML format.**

This library provides utilities to convert HuggingFace transformer models to CoreML, with support for stateful KV caching, chunked models for large architectures, and pre-compiled model caching for faster inference.

## Installation

```bash
uv sync
```

## Model Conversion

Convert a model with embeddings and LM head export:

```bash
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-1.7B --output qwen3_1.7b --num-chunks 2 --export-embeddings --export-lm-head --cache-compiled
```

For large models, convert chunks individually to reduce memory usage:

```bash
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B-Instruct-2507 --output qwen3_4b_instruct_2507 --num-chunks 4 --chunk-index 2 --skip-model-load
```

## Inference

Run inference with a converted model:

```bash
uv run python examples/inference.py --model-dir ./qwen3_4b_instruct_2507/ --model-name Qwen/Qwen3-4B-Instruct-2507 --max-new-tokens 2048 --chunked --num-chunks 4 --cache-compiled
```

## Supported Architectures

- Qwen2
- Qwen3

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
