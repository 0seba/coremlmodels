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

## Analysis & Verification Tools

The project provides built-in tools to verify CoreML conversion and Neural Engine usage. Agents **MUST** use these tools when creating examples or verifying new models.

```python
from coremlmodels import analyze_compute_plan, inspect_mil_program

# 1. Check Compute Device Selection (CPU, GPU, NE)
analyze_compute_plan(mlmodel)

# 2. Inspect Deep MIL Structure (Shapes, DTypes, Constants)
inspect_mil_program(mlmodel)
```

**What to look for:**
- **`analyze_compute_plan`**: Confirm `ios16.conv` layers are selected for `NeuralEngine` (NE).
- **`inspect_mil_program`**: Verify input shapes match 4D expectations `(B, C, 1, 1)` and check values of constants.

## Example Output

### Compute Plan Analysis
```text
Operation            | Identifier                     | Selected Device | Cost       | Supported Devices
------------------------------------------------------------------------------------------------------------------------
ios16.conv           | input_1_cast_fp16              | NeuralEngine    | 6.65e-01   | CPU,GPU,NE
ios16.relu           | x_3_cast_fp16                  | NeuralEngine    | 1.53e-03   | CPU,GPU,NE
```

### Deep MIL Inspection
```text
Operation: conv
  Output: input_1_cast_fp16 [1, 4096, 1, 1] (fp16)
  Inputs:
    - bias: Weights [4096] (fp16)
    - dilations: [1, 1] [2] (int32)
    - groups: 1 [] (int32)
    - pad: [0, 0, 0, 0] [4] (int32)
    - pad_type: "valid" [] (string)
    - strides: [1, 1] [2] (int32)
    - weight: Weights [4096, 4096, 1, 1] (fp16)
    - x: input [1, 4096, 1, 1] (fp16)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/coremlmodels/patch_linears.py` | Core Linear→Conv2d conversion logic |
| `src/coremlmodels/analysis.py` | Tools for Compute Plan and MIL inspection |
| `tests/test_patch_linears.py` | Tests verifying equivalence |
| `pyproject.toml` | Project configuration and dependencies |
