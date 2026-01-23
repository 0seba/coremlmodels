coremlmodels/docs/CODEBASE_STRUCTURE.md
```

# Project Structure

This document describes the directory layout and organization of the coremlmodels project.

## Directory Layout

```
coremlmodels/
├── src/
│   └── coremlmodels/
│       ├── __init__.py           # Package exports
│       └── patch_linears.py      # Core conversion module
├── tests/
│   └── test_patch_linears.py     # Test suite
├── docs/
│   ├── AGENTS.md                 # AI agent guidelines (this file)
│   ├── CODEBASE_STRUCTURE.md     # This file
│   ├── CODING_STANDARDS.md       # Style and patterns
│   ├── CONVERSION_GUIDE.md       # Domain-specific conversion info
│   └── TESTING_GUIDE.md          # Testing requirements
├── pyproject.toml                # Project configuration
├── uv.lock                       # Dependency lock file
└── .python-version               # Python version specification
```

## Key Files

### `src/coremlmodels/patch_linears.py`

The core module containing:

- **`LinearToConv2dPatcher`** - `nn.Module` subclass that wraps a `nn.Linear` layer and performs 1x1 conv2d instead
  - Constructor: `__init__(linear_layer: nn.Linear, bias: bool = True)`
  - Forward: `forward(x: torch.Tensor) -> torch.Tensor`
  - Input: 4D tensor `(batch, channels, 1, 1)` or `(batch, channels, H, W)`
  - Output: 4D tensor with same spatial dimensions

- **`patch_model_linears(model, skip_modules, verbose)`** - Traverses model and replaces all `nn.Linear` with `LinearToConv2dPatcher`
  - `model`: `nn.Module` to patch
  - `skip_modules`: Optional list of module names to skip
  - `verbose`: If True, print patching information
  - Returns: Modified model (in-place modification)

## Module Organization Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Minimal Dependencies**: Core modules depend only on PyTorch
3. **Explicit Over Implicit**: No automatic reshaping or type conversion
4. **Test-First**: Tests in `tests/` verify correctness

## Import Conventions

```python
# Correct
from coremlmodels import LinearToConv2dPatcher, patch_model_linears
from torch import nn

# Avoid
from coremlmodels.patch_linears import *  # No wildcard imports
```

## File Naming

- Python modules: `snake_case.py`
- Test files: `test_<module_name>.py`
- Documentation: `SCREAMING_SNAKE_CASE.md`
