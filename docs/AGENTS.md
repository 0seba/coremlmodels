# AI Coding Agent Guidelines

This document provides guidelines for AI coding agents working on the coremlmodels project.

## Project Overview

**coremlmodels** is a library for converting HuggingFace PyTorch models to CoreML format optimized for Apple's Neural Engine backend. The core approach involves:

1. **Linear → Conv2d**: Replace `nn.Linear` layers with equivalent 1x1 `conv2d` operations
2. **RMSNorm → LayerNorm**: Convert RMSNorm to LayerNorm-equivalent operations to avoid FP16 overflow

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run example scripts
uv run python examples/rmsnorm_conversion_example.py
```

## Documentation Reading Order

1. **This file (AGENTS.md)** - Start here for overview and constraints
2. **[Codebase Structure](CODEBASE_STRUCTURE.md)** - Directory layout and module descriptions
3. **[RMSNorm Patching](RMSNORM_PATCHING.md)** - Mathematical background for RMSNorm conversion
4. **[Coding Standards](CODING_STANDARDS.md)** - Style guidelines and patterns (reference as needed)

## Architecture Overview

### Patching Pattern

All patchers follow the same pattern:
1. Wrap original PyTorch layer in an `nn.Module` subclass
2. Transform weights in `__init__` (reshape, concatenate, etc.)
3. Implement equivalent forward pass that CoreMLTools can optimize
4. Provide `patch_model_*` function for recursive model traversal

### Input Tensor Format

All patched layers expect **4D tensors in channels-first format**:
- Shape: `(batch, channels, height, width)` or `(batch, channels, 1, seq_len)`
- For sequence models, use last dimension for sequence length to maximize Neural Engine utilization

### CoreML Conversion Flow

```
PyTorch Model → Patch Layers → torch.jit.trace → ct.convert → MLModel
```

## Key Modules

| Module | Classes/Functions | Purpose |
|--------|-------------------|---------|
| `patch_linears.py` | `LinearToConv2dPatcher`, `patch_model_linears` | Linear → 1x1 Conv2d |
| `patch_rmsnorm.py` | `RMSNormToLayerNormPatcher`, `patch_model_rmsnorms` | RMSNorm → LayerNorm ops |
| `analysis.py` | `analyze_compute_plan`, `inspect_mil_program` | CoreML verification tools |

## Critical Constraints

1. **4D Input Tensors**: Patched layers expect `(batch, channels, H, W)`. No automatic reshaping.

2. **Weight Views**: Use `.detach()` without `.clone()` to avoid doubling memory.

3. **Neural Engine Scheduling**: Use large workloads (e.g., seq_len ≥ 512) to trigger Neural Engine. Small workloads may run on CPU.

4. **Epsilon Handling**: RMSNorm implementations use different attribute names (`eps` vs `variance_epsilon`). The patcher handles both.

## Analysis & Verification Tools

**Always verify CoreML conversion with these tools:**

```python
from coremlmodels import analyze_compute_plan, inspect_mil_program

# Check which device runs each operation
analyze_compute_plan(mlmodel)

# Inspect MIL operations and shapes
inspect_mil_program(mlmodel)
```

### What to Look For

**In `analyze_compute_plan` output:**
- `Selected Device` should show `NeuralEngine` for compute-heavy ops (`conv`, `layer_norm`)
- If showing `CPU`, increase workload size (larger batch, seq_len, or dim)

**In `inspect_mil_program` output:**
- `layer_norm` operation confirms RMSNorm fusion worked
- Check tensor shapes match expectations

### Example Output (Neural Engine)

```
Operation            | Selected Device | Supported Devices
---------------------------------------------------------
ios16.layer_norm     | NeuralEngine    | CPU,GPU,NE
ios16.conv           | NeuralEngine    | CPU,GPU,NE
```

## Known Limitations & Future Work

### RMSNorm Weight Fusion

Currently, the RMSNorm weight multiplication is **not fused** into the `layer_norm` MIL operation. The weight appears as a separate `mul` operation after `layer_norm`:

```
layer_norm → mul (weight) → split
```

The `layer_norm` MIL op supports a `gamma` parameter that could fuse this multiplication. A custom graph pass could detect and fuse this pattern:

```
# Current (unfused)
layer_norm(x) → mul(weight)

# Desired (fused)  
layer_norm(x, gamma=weight)
```

**Task for future work:** Write a CoreMLTools graph pass to fuse the element-wise multiplication into the `layer_norm` gamma parameter.

## Development Workflow

1. Read relevant documentation (this file + module-specific docs)
2. Check existing tests for patterns
3. Make minimal changes
4. Verify with `analyze_compute_plan` and `inspect_mil_program`
5. Run tests: `uv run pytest -v`

## File Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Patcher modules | `patch_<layer_type>.py` | `patch_rmsnorm.py` |
| Test files | `test_patch_<layer_type>.py` | `test_patch_rmsnorms.py` |
| Example scripts | `<feature>_conversion_example.py` | `rmsnorm_conversion_example.py` |
| Documentation | `SCREAMING_SNAKE_CASE.md` | `RMSNORM_PATCHING.md` |
