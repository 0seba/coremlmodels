# coremlmodels

**Convert PyTorch models to CoreML with Neural Engine optimization.**

This library provides utilities to patch standard PyTorch models (specifically replacing `nn.Linear` with `nn.Conv2d` 1x1) to ensure optimal execution on Apple Neural Engine (ANE).

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run Example**:
   ```bash
   uv run python examples/coreml_conversion_example.py
   ```

## Development

- **Run Tests**: `uv run pytest`
- **Lint/Check**: `uv run ruff check .`

## Documentation for Agents

If you are an AI agent working on this repo, please read the following documents in order:

1. [AGENTS.md](docs/AGENTS.md) - **Start Here**. Workflow guidelines and required tools.
2. [CODEBASE_STRUCTURE.md](docs/CODEBASE_STRUCTURE.md) - File layout.
3. [CONVERSION_GUIDE.md](docs/CONVERSION_GUIDE.md) - Technical constraints (4D inputs, etc.).

## Key Features

- **Linear to Conv2d Patching**: Automatically replaces layers for ANE compatibility.
- **Analysis Tools**: verify device selection and inspecting compiled MIL programs.

## Usage Example

```python
from coremlmodels import patch_model_linears, analyze_compute_plan, inspect_mil_program

# 1. Patch Model
patched = patch_model_linears(model)

# 2. Convert
mlmodel = ct.convert(...)

# 3. Analyze
analyze_compute_plan(mlmodel) # Check for Neural Engine
inspect_mil_program(mlmodel)  # Check shapes/constants
```
