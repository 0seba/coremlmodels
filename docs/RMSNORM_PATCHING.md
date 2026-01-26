# RMSNorm Patching for Neural Engine

## Problem

On Apple Neural Engine (ANE), FP16 accumulation in RMSNorm causes overflow errors when calculating the root mean square. This happens because `mean(x²)` can produce very large intermediate values that exceed FP16 range.

## Solution

Convert RMSNorm to a sequence of operations that CoreMLTools will fuse into a LayerNorm operation. LayerNorm uses a different accumulation strategy that does not have this overflow issue.

## The Mathematical Trick

### RMSNorm Formula

```
y = x / sqrt(mean(x²) + eps) * weight
```

### LayerNorm Formula

```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

### The `[x, -x]` Concatenation Trick

When we concatenate `[x, -x]` along the channel dimension:

1. **Mean becomes zero**:
   ```
   mean([x, -x]) = (x + (-x)) / 2 = 0
   ```

2. **Variance equals mean(x²)**:
   Since mean = 0:
   ```
   var([x, -x]) = mean((values - 0)²) = mean([x², (-x)²]) = mean(x²)
   ```

3. **LayerNorm on [x, -x]**:
   ```
   LayerNorm(x_concat) = (x - mean) / sqrt(var + eps)
                       = (x - 0) / sqrt(mean(x²) + eps)
                       = x / sqrt(mean(x²) + eps)
   ```

4. **Taking the first half** of the output gives us exactly the RMSNorm result.

5. **Weight handling**: We use `[weight, zeros]` so only the first half gets scaled.

## Implementation

The patcher manually implements LayerNorm-equivalent operations:

```python
# 1. Concatenate [x, -x]
x_concat = torch.cat([x, -x], dim=1)

# 2. LayerNorm-equivalent ops (will be fused by CoreMLTools)
channels_mean = x_concat.mean(dim=1, keepdim=True)
zero_mean = x_concat - channels_mean
variance = (zero_mean * zero_mean).mean(dim=1, keepdim=True)
out = zero_mean * (variance + eps).rsqrt()

# 3. Apply weight and take first half
out = out * weight  # weight is [original_weight, zeros]
out = torch.chunk(out, 2, dim=1)[0]
```

## Why Not Use nn.LayerNorm Directly?

`nn.LayerNorm` expects channels-last format (normalizes over the last N dimensions), but Neural Engine operations work best with channels-first format (NCHW).

By manually implementing the operations, we:
1. Keep channels-first format throughout
2. Allow CoreMLTools to recognize and fuse the pattern
3. Maintain full control over the computation order

## Verifying Fusion

After conversion, use `inspect_mil_program(mlmodel)` to check for `layer_norm` operations in the MIL output. If you see `layer_norm` instead of separate `reduce_mean`, `sub`, `mul`, `rsqrt` operations, the fusion was successful.

## Usage

```python
from coremlmodels import patch_model_rmsnorms, patch_model_linears

# Patch RMSNorm layers
model = patch_model_rmsnorms(model, verbose=True)

# Also patch Linear layers for full 4D workflow
model = patch_model_linears(model, verbose=True)

# Convert to CoreML
mlmodel = ct.convert(traced_model, ...)
```

## Custom RMSNorm Classes

The patcher supports custom RMSNorm implementations (like `Qwen2RMSNorm`, `LlamaRMSNorm`) via the `target_classes` parameter:

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

patched = patch_model_rmsnorms(
    model,
    target_classes=(nn.RMSNorm, Qwen2RMSNorm),
    verbose=True
)
```

The patcher handles both `eps` and `variance_epsilon` attribute names automatically.

## Known Limitation: Unfused Weight Multiplication

### Current Behavior

The RMSNorm weight is applied as a **separate multiplication** after the `layer_norm` operation:

```
mul (negate x)
    ↓
concat [x, -x]
    ↓
layer_norm          ← CoreMLTools fuses mean/var/normalize ops here
    ↓
mul (weight)        ← Weight applied separately (NOT fused)
    ↓
split (take first half)
```

### MIL Output Example

```
Operation: layer_norm
  Output: out_1_cast_fp16 [1, 1024, 1, 512] (fp16)
  Inputs:
    - axes: [1]
    - epsilon: 0x1.5p-17
    - x: x_concat_1_cast_fp16
    # Note: no gamma parameter

Operation: mul
  Output: out_3_cast_fp16 [1, 1024, 1, 512] (fp16)
  Inputs:
    - x: out_1_cast_fp16
    - y: Weights [1, 1024, 1, 1]    ← This should be fused as gamma
```

### Desired Behavior

The `layer_norm` MIL operation supports a `gamma` parameter for element-wise scaling:

```
layer_norm(x, gamma=weight)  # Fused - single operation
```

### Future Work

Write a CoreMLTools graph pass to detect and fuse this pattern:

```python
# Pattern to match:
layer_norm → mul (weight broadcast along channel dim)

# Transform to:
layer_norm with gamma=weight
```

This would reduce the operation count and potentially improve Neural Engine throughput.
