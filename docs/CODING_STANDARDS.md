coremlmodels/docs/CODING_STANDARDS.md
```

# Coding Standards

This document outlines the coding standards, patterns, and conventions for the coremlmodels project.

## Philosophy

- **Minimalism**: Fewer lines of code, fewer abstractions
- **Explicit**: No magic, no auto-conversion, no hidden behavior
- **Debuggable**: Shape mismatches should fail loudly during inference
- **Tested**: Every change must have corresponding tests

## Code Style

### Formatting

- Use **ruff** for formatting (configured in `pyproject.toml`)
- Maximum line length: 100 characters
- Use 4 spaces for indentation

### Type Hints

All functions must have type hints:

```python
# Correct
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...

# Incorrect - no type hints
def forward(self, x):
    ...
```

### Docstrings

Minimal docstrings for public APIs:

```python
def patch_model_linears(
    model: nn.Module,
    skip_modules: Optional[list] = None,
    verbose: bool = False
) -> nn.Module:
    """Traverse model and replace nn.Linear with LinearToConv2dPatcher."""
    ...
```

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `LinearToConv2dPatcher` |
| Functions | snake_case | `patch_model_linears` |
| Variables | snake_case | `input_tensor` |
| Constants | UPPER_SNAKE_CASE | Not used frequently |
| Module names | snake_case | `patch_linears.py` |

## Patterns

### The Patcher Pattern

When replacing PyTorch layer behavior:

```python
class LayerPatcher(nn.Module):
    """Wraps original layer and modifies forward behavior."""
    
    def __init__(self, original_layer: nn.Module):
        super().__init__()
        self.original_layer = original_layer
        # Keep as view to avoid doubling memory usage
        self.weight = original_layer.weight.detach()
        if original_layer.bias is not None:
            self.bias = original_layer.bias.detach()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit 4D input expected
        return F.conv2d(x, self.weight, bias=self.bias, ...)
```

### Weight Handling

Detach weights without cloning to keep them as views:

```python
# Correct - view, no memory duplication
self.weight = original_layer.weight.detach()

# Avoid - cloning doubles memory usage
self.weight = original_layer.weight.detach().clone()
```

### Input Shape Philosophy

Do not add automatic reshaping:

```python
# Correct - user provides correct input shape
output = patcher(input_4d)  # input_4d is (batch, channels, 1, 1)

# Incorrect - do NOT add auto-reshape
if x.dim() == 2:
    x = x.view(x.size(0), x.size(1), 1, 1)  # Never do this
```

## Imports

Standard import order:

```python
# Standard library
from typing import Optional

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local
from coremlmodels import ...
```

No wildcard imports:

```python
# Avoid
from coremlmodels.patch_linears import *

# Use
from coremlmodels import LinearToConv2dPatcher, patch_model_linears
```

## Error Handling

- Let shape mismatches propagate as exceptions
- Add descriptive error messages for internal validation
- Do not catch ShapeMismatchError - let it fail loudly

## Testing Patterns

### Test Structure

Each test should:
1. Create original and patched versions
2. Use same random seed for reproducibility
3. Compare outputs with `torch.testing.assert_close`

```python
def test_feature():
    """Test description."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    patched = patch_model_linears(model)
    
    x = torch.randn(2, 10)
    x_4d = x.view(2, 10, 1, 1)
    
    with torch.no_grad():
        original_out = model(x)
        patched_out = patched(x_4d).squeeze(-1).squeeze(-1)
    
    torch.testing.assert_close(original_out, patched_out)
```

### Test Count

Keep test count minimal - focus on equivalence rather than exhaustive edge cases.

## File Organization

- One module per file for core functionality
- Tests in `tests/` directory
- Test file naming: `test_<module_name>.py`

## Code Review Checklist

- [ ] Type hints on all functions
- [ ] Tests pass (`uv run pytest`)
- [ ] No automatic reshaping
- [ ] Weights properly detached
- [ ] Minimal, focused changes
- [ ] No dead code or comments
- [ ] Descriptive variable names