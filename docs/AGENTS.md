# AI Coding Agent Guidelines

This document provides guidelines for AI coding agents working on the coremlmodels project.

## Project Overview

**coremlmodels** is a library for converting HuggingFace PyTorch models to CoreML format optimized for Apple's Neural Engine backend. The core approach involves replacing `nn.Linear` layers with equivalent 1x1 `conv2d` operations that are better aligned with Neural Engine compute units.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_patch_linears.py -v
```

## Documentation Structure

- [Project Structure](CODEBASE_STRUCTURE.md) - Directory layout and file organization
- [Coding Standards](CODING_STANDARDS.md) - Style guidelines, patterns, and conventions
- [Conversion Guide](CONVERSION_GUIDE.md) - Domain-specific conversion patterns (Linear→Conv2d)
- [Testing Guide](TESTING_GUIDE.md) - Testing requirements and patterns

## Critical Constraints

1. **Input Shapes**: The patched model layers expect 4D input tensors `(batch, channels, 1, 1)`. Do NOT reshape inputs automatically - let shape mismatches surface as errors during inference to catch bugs early.

2. **Weight Handling**: Weights are kept as views (not cloned) to avoid doubling memory usage. The reshape from `(out, in)` to `(out, in, 1, 1)` is sufficient - no transpose needed.

3. **No Auto-reshape**: Do not add automatic 2D→4D or 4D→2D conversion layers. Users are responsible for providing correct input shapes.

4. **Minimal Testing**: Tests should verify output equivalence between original and patched implementations. Keep tests focused and minimal.

5. **Detach from Gradients**: When creating patcher weights, always detach from the original linear layer's gradient computation to avoid issues.

## Communication Style

- Use markdown for formatting
- Never make up information - ask for clarification if unsure
- Format code blocks with file paths: ```path/to/file.py

## Development Workflow

1. Read the relevant documentation before making changes
2. Check existing tests for patterns
3. Make minimal changes to pass tests
4. Do not simplify code just to pass linters - correct code is preferred
5. Add descriptive error messages when debugging

## Key Files

| File | Purpose |
|------|---------|
| `src/coremlmodels/patch_linears.py` | Core Linear→Conv2d conversion logic |
| `tests/test_patch_linears.py` | Tests verifying equivalence |
| `pyproject.toml` | Project configuration and dependencies |
