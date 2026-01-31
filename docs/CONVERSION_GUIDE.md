# Conversion Guide

This guide covers the conversion of HuggingFace language models to CoreML format for Neural Engine optimization.

## Overview

The conversion process transforms PyTorch models into CoreML format with:
- **Neural Engine optimization**: `nn.Linear` → `nn.Conv2d(kernel_size=1)`
- **Channels-first format**: 4D tensors `(batch, channels, 1, seq_len)`
- **Stateful KV cache**: For efficient autoregressive generation
- **Chunked model support**: For large models (>2GB) that exceed Neural Engine limits

## Quick Start

```bash
# Single model conversion
uv run python examples/lm_conversion_example.py --model Qwen/Qwen2-0.5B --export-embeddings --export-lm-head

# Large model (chunked conversion)
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --export-embeddings --export-lm-head

# Memory-efficient conversion (one chunk at a time)
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 0 --skip-model-load
```

## Supported Architectures

- **Qwen2**: `Qwen/Qwen2-0.5B`, `Qwen/Qwen2-1.5B`, etc.
- **Qwen3**: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, etc.

## Conversion Modes

### Single Model Conversion

For models under ~2GB that fit Neural Engine weight limits:

```bash
uv run python examples/lm_conversion_example.py \
    --model Qwen/Qwen2-0.5B \
    --seq-len 8 \
    --cache-length 2048 \
    --export-embeddings \
    --export-lm-head
```

**Output**: `qwen2_0.5b_seqlen_8.mlpackage` + `embeddings.npy` + `lm_head.mlpackage`

### Chunked Model Conversion

For large models that exceed Neural Engine limits:

```bash
uv run python examples/lm_conversion_example.py \
    --model Qwen/Qwen3-4B \
    --num-chunks 4 \
    --export-embeddings \
    --export-lm-head
```

**Output directory structure**:
```
qwen3_4b_chunked_4/
├── chunk_0.mlpackage
├── chunk_1.mlpackage
├── chunk_2.mlpackage
├── chunk_3.mlpackage
├── embeddings.npy
└── lm_head.mlpackage
```

## Memory Optimization

### Option 1: Convert Specific Chunks

Convert one chunk at a time to reduce peak memory:

```bash
# Convert chunk 0
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 0

# Convert chunk 1
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 1

# Convert multiple chunks
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --chunk-index 2,3
```

### Option 2: Skip Model Load

Don't load the converted model into memory after conversion:

```bash
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --skip-model-load
```

This also allows converting newer model formats on older macOS versions.

### Option 3: Skip Verification

Skip the output verification step (saves memory from not keeping reference output):

```bash
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4 --skip-verification
```

### Minimum Memory Mode

Combine all options for maximum memory savings:

```bash
uv run python examples/lm_conversion_example.py \
    --model Qwen/Qwen3-4B \
    --num-chunks 4 \
    --chunk-index 0 \
    --skip-model-load
```

Run separately for each chunk index (0, 1, 2, 3).

### Option 4: Cache Compiled Models

Cache the compiled `.mlmodelc` files for faster subsequent loads during inference:

```bash
uv run python examples/lm_conversion_example.py \
    --model Qwen/Qwen3-4B \
    --num-chunks 4 \
    --export-embeddings \
    --export-lm-head \
    --cache-compiled
```

This saves the compiled models alongside the `.mlpackage` files, significantly reducing load time for inference. See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for details on using cached models.

**Note**: `--cache-compiled` requires model loading (incompatible with `--skip-model-load`).

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2-0.5B` | HuggingFace model name |
| `--seq-len` | `8` | Sequence length for tracing |
| `--cache-length` | `2048` | Maximum KV cache length |
| `--num-layers` | all | Limit layers (for debugging) |
| `--num-chunks` | `1` | Number of chunks (1 = no chunking) |
| `--output` | auto | Output path or directory |
| `--chunk-index` | all | Specific chunk(s) to convert |
| `--skip-verification` | false | Skip output verification |
| `--skip-model-load` | false | Don't load model after conversion |
| `--cache-compiled` | false | Cache compiled models (.mlmodelc) |
| `--export-embeddings` | false | Export embeddings as .npy |
| `--export-lm-head` | false | Export LM head as CoreML |
| `--lm-head-chunk-size` | `6144` | Vocab chunk size for LM head |
| `--components-only` | false | Skip model, only export components |
| `--analyze-compute-plan` | false | Check Neural Engine scheduling |
| `--analyze-mil` | false | Inspect MIL operations |
| `--quiet` | false | Reduce output verbosity |

## Technical Details

### Neural Engine Optimization

The conversion replaces `nn.Linear` with `nn.Conv2d` using 1x1 kernels:

- **Input**: 4D tensor `(batch, channels, 1, seq_len)`
- **Output**: 4D tensor `(batch, out_channels, 1, seq_len)`
- **Weight**: Reshaped from `(out, in)` to `(out, in, 1, 1)`

This allows operations to run on the Neural Engine instead of CPU/GPU.

### KV Cache States

Each model chunk maintains stateful KV cache tensors:
- Shape: `(batch, num_kv_heads, cache_length, head_dim)`
- One K and one V cache per transformer layer
- Position tracked via `position_id` input

### Layer Patching

The conversion applies three types of patches:

1. **Linear → Conv2d**: All `nn.Linear` layers
2. **RMSNorm → LayerNorm**: With weight multiplication fused in
3. **Attention**: Custom channels-first attention with:
   - 4D tensor operations
   - Rotary position embeddings
   - Grouped-query attention support
   - QK-norm (for Qwen3)

### LM Head Chunking

Large vocabularies are split into chunks to fit Neural Engine limits:

```bash
# Default: 6144 vocab per chunk
uv run python examples/lm_conversion_example.py --export-lm-head

# Custom chunk size
uv run python examples/lm_conversion_example.py --export-lm-head --lm-head-chunk-size 8192
```

The LM head outputs:
- `logits`: Shape `(1, vocab_size, 1, seq_len)`
- `chunk_logsumexp_stable`: For stable probability computation
- `chunk_max`: Per-chunk maximum values

## Verification

After conversion, verify Neural Engine usage:

```bash
# Check compute plan
uv run python examples/lm_conversion_example.py --model Qwen/Qwen2-0.5B --analyze-compute-plan

# Inspect MIL operations
uv run python examples/lm_conversion_example.py --model Qwen/Qwen2-0.5B --analyze-mil
```

Or use the library functions:

```python
from coremlmodels import analyze_compute_plan, inspect_mil_program
import coremltools as ct

mlmodel = ct.models.MLModel("model.mlpackage")
analyze_compute_plan(mlmodel)  # Check NeuralEngine scheduling
inspect_mil_program(mlmodel)   # Inspect conv, layer_norm ops
```

## Debugging

### Quick Debug with Fewer Layers

```bash
# Only 2 layers (fast iteration)
uv run python examples/lm_conversion_example.py --num-layers 2

# 2 layers split into 2 chunks
uv run python examples/lm_conversion_example.py --num-layers 8 --num-chunks 2
```

### Export Components Only

Skip model conversion to quickly iterate on embeddings/LM head:

```bash
uv run python examples/lm_conversion_example.py --export-embeddings --export-lm-head --components-only
```

## Common Issues

### Out of Memory

- Use `--chunk-index` to convert one chunk at a time
- Use `--skip-model-load` to avoid loading after conversion
- Use `--skip-verification` to skip keeping reference output

### Neural Engine Not Used

- Check with `--analyze-compute-plan` that ops run on `NeuralEngine`
- Verify input shapes are 4D `(batch, channels, 1, seq_len)`
- Ensure weights don't exceed Neural Engine dimension limits (~16384)

### Verification Failed

Verification uses tolerances:
- Absolute diff max < 0.1 (0.5 for chunked)
- Relative diff mean < 0.1 (0.2 for chunked)

Higher tolerances for chunked models due to accumulated floating-point differences across chunks.
