# Agent Guide: coremlmodels

**Everything AI agents need to know to work on this project.**

## What This Project Does

**coremlmodels** provides patching utilities to convert PyTorch models to CoreML format optimized for Apple's Neural Engine (ANE). Current optimizations are:

1. **Convert Linear layers to 1x1 Conv2d** - ANE favors 4D tensors in channels-first format (NCHW)
2. **Patch RMSNorm layers** - Trick to mitigate ANE's FP16 accumulation issues with RMSNorm's `sqrt(mean(x²))`

## Essential Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_patch_rmsnorms.py -v

# Run conversion example
uv run python examples/rmsnorm_conversion_example.py

# Lint code
uv run ruff check .
```

## Core Concepts

### 4D Channels-First Tensors

All patched layers require **4D tensors in channels-first format (NCHW)**:

```python
# Correct: 4D tensor (batch, channels, height, width)
x = torch.randn(2, 1024, 1, 512)  # (batch, dim, 1, seq_len)

# Incorrect: 2D tensor (batch, features)
x = torch.randn(2, 1024)  # Will fail!

# User must reshape before passing to patched model
x_4d = x.view(x.size(0), x.size(1), 1, 1)
```

### 1x1 Conv2d Instead of Linear

ANE favors 4D tensors in channels-first format (NCHW). The solution: wrap the Linear layer in a patcher that performs 1x1 convolution.

**Core concepts of the implementation:**

```python
# From src/coremlmodels/patch_linears.py
class LinearToConv2dPatcher(nn.Module):
    def __init__(self, linear_layer: nn.Linear, bias: bool = True):
        super().__init__()

        # Concept 1: Create VIEW of detached weights (not cloned)
        # Using .detach() without .clone() keeps the tensor as a view
        # This avoids doubling memory usage
        self.weight = linear_layer.weight.reshape(
            linear_layer.out_features, linear_layer.in_features, 1, 1
        ).detach()  # <- View, not clone!

        # Same for bias - keep as view to save memory
        if linear_layer.bias is not None:
            self.bias = linear_layer.bias.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concept 2: Reshape weights for conv2d (already done in __init__)
        # Linear weight: (out_features, in_features)
        # Conv2d weight: (out_channels, in_channels, kernel_h, kernel_w)
        # We reshape from (out, in) to (out, in, 1, 1) for 1x1 convolution

        # Concept 3: Use F.conv2d for 1x1 convolution
        # This is mathematically equivalent to x @ W.T + b
        # But ANE schedules conv2d, not matmul
        return F.conv2d(
            x, self.weight, bias=self.bias,
            stride=1, padding=0, dilation=1, groups=1
        )
```

**Key points:**
1. **Memory efficiency:** `.detach()` creates a view, not a clone. The weight shares storage with the original, saving memory.
2. **Weight reshape:** `(out_features, in_features)` -> `(out_features, in_features, 1, 1)` for 1x1 convolution
3. **Mathematical equivalence:** `F.conv2d(x, W.view(out, in, 1, 1))` == `x @ W.T`

### The Linear Patching Function

```python
from coremlmodels import patch_model_linears

# Patch all Linear layers in a model
patched_model = patch_model_linears(model, verbose=True)

# Skip specific modules
patched_model = patch_model_linears(model, skip_modules=[model.decoder], verbose=True)
```

The function recursively traverses the model and replaces `nn.Linear` with `LinearToConv2dPatcher`, which performs 1x1 convolution in the forward pass.

### Float16 Limitation on Neural Engine

ANE operates in **FP16** and has limited dynamic range. The problem with RMSNorm:

```
RMSNorm: y = x / sqrt(mean(x²) + eps) * weight
                    ↑
            mean(x²) can overflow FP16
```

FP16 max value: ~65504
- If `mean(x²)` exceeds this during accumulation -> **overflow error**
- This happens with large hidden dimensions (>= 1024)

### The RMSNorm Patching Method

RMSNorm has an **FP16 overflow issue** on ANE. The forward pass is restructured to use operations that CoreML can **fuse into a single `layer_norm` operation**.

See [RMSNORM_PATCHING.md](RMSNORM_PATCHING.md) for the mathematical background.

**CoreML Graph Passes and Operation Fusion:**

When `ct.convert()` runs, it applies graph optimization passes that detect patterns and fuse multiple operations into a single optimized operation:

**Simplified forward method of the patched RMSNorm to ilustrate CoreML operation fusion:**

```python
# From src/coremlmodels/patch_rmsnorm.py
class RMSNormToLayerNormPatcher(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Concatenate [x, -x] along channel dimension
        x_concat = torch.cat([x, -x], dim=1)

        # Step 2: LayerNorm-equivalent ops (will be fused by CoreML)
        # CoreML detects this pattern: mean -> sub -> square -> mean -> rsqrt -> mul
        # And fuses it into: layer_norm operation
        channels_mean = x_concat.float().mean(dim=1, keepdim=True)
        zero_mean = x_concat.float() - channels_mean
        variance = (zero_mean * zero_mean).mean(dim=1, keepdim=True)
        out = zero_mean * (variance + self.eps).rsqrt()

        # Step 3: Apply weight and take first half (x portion)
        if self.weight is not None:
            out = out * self.weight
        out = torch.chunk(out, 2, dim=1)[0]

        return out.to(x.dtype)
```

**Why this matters:**
- CoreML's `layer_norm` operation handles FP16 accumulation differently, avoiding overflow
- Fused operations have lower memory bandwidth and faster execution
- The `[x, -x]` trick makes LayerNorm mathematically equivalent to RMSNorm

**Another important note: We cannot use Pytorch's builtin `LayerNorm` because it always normalizes along the last dimensions, but we structure our data in a channels-first format for better ANE compatibility**

## Available Patching Methods

```python
from coremlmodels import (
    patch_model_linears,
    patch_model_rmsnorms,
)

# Method 1: Patch all Linear layers to 1x1 Conv2d
patched = patch_model_linears(model)

# Method 2: Patch RMSNorm layers to LayerNorm-fusable ops
patched = patch_model_rmsnorms(patched)

# Both methods support:
# - skip_modules: List of module paths to skip
# - verbose: Print patched module names
patched = patch_model_linears(
    model,
    skip_modules=["model.decoder"],
    verbose=True
)

patched = patch_model_rmsnorms(
    model,
    target_classes=(nn.RMSNorm, CustomRMSNorm),  # Custom RMSNorm classes
    verbose=True
)
```



## Complete Example: Patch and Convert

```python
import torch
import torch.nn as nn
import coremltools as ct
from coremlmodels import patch_model_linears, patch_model_rmsnorms
from coremlmodels import analyze_compute_plan, inspect_mil_program

# 1. Create a model with Linear and RMSNorm
class SimpleModel(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.norm = nn.RMSNorm(dim)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x

model = SimpleModel()

# 2. Patch the model (order matters: patch Linears first, then RMSNorm)
patched = patch_model_linears(model, verbose=True)
patched = patch_model_rmsnorms(patched, verbose=True)

# 3. Create 4D input tensor (required for patched model)
batch, dim, seq_len = 1, 1024, 512
input_tensor = torch.randn(batch, dim, 1, seq_len)  # 4D NCHW format!

# 4. Trace with torch.jit
traced = torch.jit.trace(patched, input_tensor)

# 5. Convert to CoreML
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(batch, dim, 1, seq_len))],
    compute_precision=ct.precision.FLOAT16,  # FP16 required for ANE
    compute_units=ct.ComputeUnit.ALL,        # Use Neural Engine
)

# 6. Verify ANE scheduling
print("=== Compute Plan Analysis ===")
analyze_compute_plan(mlmodel)

print("\n=== MIL Program Inspection ===")
inspect_mil_program(mlmodel)
```

## Model Intermediate Language (MIL)

When you call `ct.convert()`, CoreML:
1. Traces the PyTorch model
2. Converts operations to MIL (CoreML's intermediate representation)
3. Optimizes the MIL program
4. Compiles to the device-specific format

**MIL is critical for debugging** because it shows:
- Which operations exist after optimization
- Whether CoreML fused operations (e.g., RMSNorm -> layer_norm)
- Input/output shapes at each stage
- Constant values (weights, parameters)

### MIL Analysis Tool

```python
from coremlmodels import inspect_mil_program

inspect_mil_program(mlmodel)
```

**Example output:**

```
=== MIL Program Structure ===
opset version: 2024.1

Operation: mul
  Output: var_10_cast_fp16 [1, 512, 1, 512] (fp16)
  Inputs:
    - x: input [1, 512, 1, 512] (fp16)
    - y: -0x1p+0 [] (fp16)
----------------------------------------
Operation: concat
  Output: x_concat_1_cast_fp16 [1, 1024, 1, 512] (fp16)
  Inputs:
    - axis: 1 [] (int32)
    - interleave: false [] (bool)
    - values:
      - input [1, 512, 1, 512] (fp16)
      - var_10_cast_fp16 [1, 512, 1, 512] (fp16)
----------------------------------------
Operation: layer_norm
  Output: out_1_cast_fp16 [1, 1024, 1, 512] (fp16)
  Inputs:
    - axes: [1] [1] (int32)
    - epsilon: 0x1.5p-17 [] (fp16)
    - x: x_concat_1_cast_fp16 [1, 1024, 1, 512] (fp16)
----------------------------------------
Operation: mul
  Output: out_3_cast_fp16 [1, 1024, 1, 512] (fp16)
  Inputs:
    - x: out_1_cast_fp16 [1, 1024, 1, 512] (fp16)
    - y: Weights [1, 1024, 1, 1] (fp16)
----------------------------------------
Operation: split
  Output: var_25_cast_fp16_0, [1, 512, 1, 512] (fp16)
  Inputs:
    - axis: 1 [] (int32)
    - split_sizes: [512, 512] [2] (int32)
    - x: out_3_cast_fp16 [1, 1024, 1, 512] (fp16)
----------------------------------------
Operation: conv
  Output: input_cast_fp16 [1, 2048, 1, 512] (fp16)
  Inputs:
    - bias: Weights [2048] (fp16)
    - dilations: [1, 1] [2] (int32)
    - groups: 1 [] (int32)
    - pad: [0, 0, 0, 0] [4] (int32)
    - pad_type: "valid" [] (string)
    - strides: [1, 1] [2] (int32)
    - weight: Weights [2048, 512, 1, 1] (fp16)
    - x: var_25_cast_fp16_0
----------------------------------------
Operation: gelu
  Output: x_5_cast_fp16 [1, 2048, 1, 512] (fp16)
  Inputs:
    - mode: "EXACT" [] (string)
    - x: input_cast_fp16 [1, 2048, 1, 512] (fp16)
```

**What to look for:**
- `conv` operations (from Linear patching) should exist
- `layer_norm` operation confirms RMSNorm fusion worked
- Separate `mul` after `layer_norm` means weight is NOT fused (known limitation)
- If you see `reduce_mean`, `sub`, `mul`, `rsqrt` separately -> fusion failed

### Compute Analysis Utility

```python
from coremlmodels import analyze_compute_plan

analyze_compute_plan(mlmodel)
```

**Example output:**

```
=== Compute Plan Analysis ===
Model: simple_model.mlmodel

Operation            | Identifier                     | Selected Device | Cost       | Supported Devices
------------------------------------------------------------------------------------------------------------------------
ios16.mul            | var_10_cast_fp16               | NeuralEngine    | 3.17e-02   | CPU,GPU,NE
concat               | x_concat_1_cast_fp16           | NeuralEngine    | 2.72e-02   | CPU,GPU,NE
ios16.layer_norm     | out_1_cast_fp16                | NeuralEngine    | 7.46e-02   | CPU,GPU,NE
ios16.mul            | out_3_cast_fp16                | NeuralEngine    | 6.34e-02   | CPU,GPU,NE
split                | var_25_cast_fp16_0             | NeuralEngine    | 5.59e-02   | CPU,GPU,NE
ios16.conv           | input_cast_fp16                | NeuralEngine    | 1.73e-01   | CPU,GPU,NE
...
```

**What to look for:**
- `conv` and `layer_norm` operations should show `NeuralEngine` as `Selected Device`
- If showing `CPU` or `GPU`, the model may be too small or shapes are wrong
- Large `Compute` values indicate compute-heavy ops that benefit from ANE

**If operations run on CPU instead of ANE:**
- Increase sequence length (>= 512)
- Increase batch size
- Verify input is 4D NCHW format
- Check `compute_units=ct.ComputeUnit.ALL` in conversion

## Key Modules

| Module | Classes/Functions | Purpose |
|--------|-------------------|---------|
| `patch_linears.py` | `LinearToConv2dPatcher`, `patch_model_linears` | Linear -> 1x1 Conv2d |
| `patch_rmsnorm.py` | `RMSNormToLayerNormPatcher`, `patch_model_rmsnorms` | RMSNorm -> LayerNorm ops |
| `analysis.py` | `analyze_compute_plan`, `inspect_mil_program` | CoreML verification tools |

## Known Limitations

### RMSNorm Weight Fusion

The RMSNorm weight multiplication is **not fused** into the `layer_norm` MIL operation:

```
layer_norm -> mul (weight) -> split
```

The weight appears as a separate `mul` operation after `layer_norm`. Future work could write a CoreMLTools graph pass to fuse this pattern.

## Development Workflow

1. Read relevant documentation (this file + module-specific docs)
2. Check existing tests for patterns
3. Make minimal changes
4. Verify with `analyze_compute_plan` and `inspect_mil_program`
5. Run tests: `uv run pytest -v`

## Documentation Reading Order

1. **AGENTS.md** (this file) - Overview and essential concepts
2. **[CODEBASE_STRUCTURE.md](CODEBASE_STRUCTURE.md)** - Directory layout
3. **[CODING_STANDARDS.md](CODING_STANDARDS.md)** - Code style and patterns
4. **[RMSNORM_PATCHING.md](RMSNORM_PATCHING.md)** - Mathematical background
