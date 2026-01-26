# Project Structure

This document describes the directory layout and organization of the coremlmodels project.

## Directory Layout

```
coremlmodels/
├── src/coremlmodels/
│   ├── __init__.py           # Package exports
│   ├── analysis.py           # CoreML analysis tools
│   ├── patch_linears.py      # Linear → Conv2d conversion
│   └── patch_rmsnorm.py      # RMSNorm → LayerNorm conversion
├── tests/
│   ├── test_patch_linears.py # Linear patching tests
│   └── test_patch_rmsnorms.py# RMSNorm patching tests (includes CoreML conversion test)
├── examples/
│   ├── coreml_conversion_example.py    # Linear patching example
│   └── rmsnorm_conversion_example.py   # RMSNorm patching example
├── docs/
│   ├── AGENTS.md             # AI agent guidelines (start here)
│   ├── CODEBASE_STRUCTURE.md # This file
│   ├── CODING_STANDARDS.md   # Style and patterns
│   ├── CONVERSION_GUIDE.md   # Linear→Conv2d specifics
│   ├── RMSNORM_PATCHING.md   # RMSNorm mathematical background
│   └── TESTING_GUIDE.md      # Testing patterns
├── pyproject.toml            # Project configuration
└── uv.lock                   # Dependency lock file
```

## Core Modules

### `src/coremlmodels/patch_linears.py`

Converts `nn.Linear` to 1x1 Conv2d for Neural Engine optimization.

| Export | Type | Description |
|--------|------|-------------|
| `LinearToConv2dPatcher` | `nn.Module` | Wraps Linear, performs conv2d in forward |
| `patch_model_linears(model, skip_modules, verbose)` | Function | Recursively patches all Linear layers |

**Input/Output:** 4D tensor `(batch, channels, 1, 1)`

### `src/coremlmodels/patch_rmsnorm.py`

Converts RMSNorm to LayerNorm-equivalent operations to avoid FP16 overflow on Neural Engine.

| Export | Type | Description |
|--------|------|-------------|
| `RMSNormToLayerNormPatcher` | `nn.Module` | Wraps RMSNorm, uses [x,-x] concatenation trick |
| `patch_model_rmsnorms(model, target_classes, skip_modules, verbose)` | Function | Recursively patches RMSNorm layers |

**Input/Output:** 4D tensor `(batch, channels, 1, seq_len)`

**Supports:** `nn.RMSNorm`, custom implementations with `eps` or `variance_epsilon` attributes

### `src/coremlmodels/analysis.py`

Tools for verifying CoreML conversion and Neural Engine scheduling.

| Export | Description |
|--------|-------------|
| `analyze_compute_plan(mlmodel)` | Shows operation → device mapping (CPU/GPU/NE) |
| `inspect_mil_program(mlmodel)` | Deep inspection of MIL operations, shapes, dtypes |

## Test Files

### `tests/test_patch_linears.py`
- Numerical equivalence tests for Linear → Conv2d
- Nested module patching
- Skip modules functionality

### `tests/test_patch_rmsnorms.py`
- Numerical equivalence tests for RMSNorm → LayerNorm
- Custom RMSNorm class support (variance_epsilon)
- **CoreML conversion test with Neural Engine verification**

## Example Scripts

### `examples/rmsnorm_conversion_example.py`
End-to-end example showing:
1. Model creation with RMSNorm + Linear
2. Patching both layer types
3. CoreML conversion with FP16
4. Output verification
5. Compute plan analysis (Neural Engine confirmation)
6. MIL inspection (LayerNorm fusion confirmation)

**Run with:** `uv run python examples/rmsnorm_conversion_example.py`

## Module Dependencies

```
analysis.py          ← standalone (coremltools only)
patch_linears.py     ← standalone (torch only)
patch_rmsnorm.py     ← standalone (torch only)
```

No inter-module dependencies. Each patcher is self-contained.

## Import Conventions

```python
# Public API imports
from coremlmodels import (
    LinearToConv2dPatcher,
    patch_model_linears,
    RMSNormToLayerNormPatcher,
    patch_model_rmsnorms,
    analyze_compute_plan,
    inspect_mil_program,
)
```
