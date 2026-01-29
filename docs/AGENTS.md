# Agent Guide: coremlmodels

**Everything AI agents need to know to work on this project.**

## What This Project Does

**coremlmodels** provides patching utilities to convert PyTorch models to CoreML format optimized for Apple's Neural Engine (ANE). Current optimizations are:

1. **Convert Linear layers to 1x1 Conv2d** - ANE favors 4D tensors in channels-first format (NCHW)
2. **Patch RMSNorm layers** - Trick to mitigate ANE's FP16 accumulation issues with RMSNorm's `sqrt(mean(x²))`
3. **Patch Attention layers** - Channels-first GQA with CoreML state for KV cache, supports QK-norm (Qwen3-style)
4. **Language Model Wrapper** - Full LM conversion with pre-computed position embeddings and stateful KV cache
5. **Chunked LM Head** - Vocabulary chunking to handle ANE's weight dimension limit (~16384) with temperature scaling
6. **Embeddings Export** - Export embedding weights as .npy files in float16 format
7. **Architecture Registry** - Auto-detection of model architecture for appropriate patching configuration

## Repository navigation instructions

1. **Do not read the contents of the [huggingface_models/](huggingface_models) or [reference/](reference/) directories unless I explicitly reference them or a file inside of them in my request*

## Essential Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_patch_rmsnorms.py -v

# Run simple conversion example
uv run python examples/rmsnorm_conversion_example.py

# Run full LLM conversion example (architecture-agnostic)
uv run python examples/lm_conversion_example.py                    # Qwen2 (default)
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-0.6B  # Qwen3 with QK-norm
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4  # Large model chunked

# Export embeddings and LM head separately
uv run python examples/export_embeddings_lmhead.py                 # Export both
uv run python examples/lm_conversion_example.py --export-embeddings --export-lm-head --components-only

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
    def forward(self, x: torch.Tensor, axis: int = 1) -> torch.Tensor:
        # Step 1: Concatenate [x, -x] along the normalization axis
        x_concat = torch.cat([x, -x], dim=axis)

        # Step 2: LayerNorm-equivalent ops (will be fused by CoreML)
        # CoreML detects this pattern: mean -> sub -> square -> mean -> rsqrt -> mul
        # And fuses it into: layer_norm operation
        channels_mean = x_concat.float().mean(dim=axis, keepdim=True)
        zero_mean = x_concat.float() - channels_mean
        variance = (zero_mean * zero_mean).mean(dim=axis, keepdim=True)
        out = zero_mean * (variance + self.eps).rsqrt()

        # Step 3: Apply weight (reshaped for the axis) and take first half
        if self.weight is not None:
            out = out * weight_reshaped  # Weight reshaped for broadcasting
        out = torch.chunk(out, 2, dim=axis)[0]

        return out.to(x.dtype)
```

**Flexible Axis Normalization:**

The patcher supports normalizing over any axis via the `axis` parameter:

```python
# Default: normalize over channel dimension (axis=1)
out = rmsnorm_patcher(x)  # x shape: (batch, channels, 1, seq)

# Normalize over a different axis (e.g., head_dim in attention)
out = rmsnorm_patcher(x, axis=2)  # x shape: (batch, heads, head_dim, seq)

# Last dimension: uses F.layer_norm internally (efficient)
out = rmsnorm_patcher(x, axis=-1)  # x shape: (batch, heads, seq, head_dim)
```

This flexibility is essential for QK-norm in attention layers where Q and K need normalization over `head_dim` without reshaping.

**Why this matters:**
- CoreML's `layer_norm` operation handles FP16 accumulation differently, avoiding overflow
- Fused operations have lower memory bandwidth and faster execution
- The `[x, -x]` trick makes LayerNorm mathematically equivalent to RMSNorm

**Another important note: Pytorch's builtin `LayerNorm` always normalizes along the last dimensions, but we structure our data in a channels-first format for better ANE compatibility, this means that in the general case we cannot use it and have to rely on CoreML for operator fusion, but have a separate case for the last dimension where we can use it**

### CoreML State for KV Cache

CoreML supports **stateful models** where tensors persist across inference calls. This is essential for transformer KV caches - without state, you'd need to recompute all previous keys/values on every token.

**Step 1: Define state tensors as registered buffers in PyTorch**

```python
# From src/coremlmodels/lm_model_wrapper.py
class LanguageModelWrapper(nn.Module):
    def __init__(self, model, cache_length=2048):
        super().__init__()
        # KV cache as registered buffers - these become CoreML state
        self.register_buffer(
            "key_cache",
            torch.zeros(num_layers, num_kv_heads, cache_length, head_dim),
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(num_layers, num_kv_heads, cache_length, head_dim),
        )
```

**Step 2: Inform CoreML about state during conversion**

```python
import coremltools as ct

# Create StateType specs for each stateful buffer
states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=wrapper.key_cache.shape),
        name="key_cache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=wrapper.value_cache.shape),
        name="value_cache",
    ),
]

mlmodel = ct.convert(
    traced_model,
    inputs=[...],
    outputs=[...],
    states=states,  # <-- Pass state specs here
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
)
```

**Step 3: Update and read state in forward pass**

```python
# From src/coremlmodels/patch_attention.py
def forward(self, hidden_states, past_key_values, cache_position, ...):
    key_cache, value_cache = past_key_values

    # IMPORTANT: cache_position must be a tensor of shape (1,), not a Python int!
    # CoreML requires the write index to be passed as a size [1] array
    # This is passed from the wrapper: position_id with shape (1,)

    # Get sequence length from key_states (avoids dynamic shape issues)
    seq_len = key_states.size(2)

    # Write new keys/values at current position
    key_cache[
        self.layer_idx : self.layer_idx + 1,
        :,
        cache_position : cache_position + seq_len,
    ] = key_states
    value_cache[
        self.layer_idx : self.layer_idx + 1,
        :,
        cache_position : cache_position + seq_len,
    ] = value_states

    # Read full cache for attention computation
    key_states = key_cache[self.layer_idx : self.layer_idx + 1]
    value_states = value_cache[self.layer_idx : self.layer_idx + 1]
```

**Step 4: Use state at inference time**

```python
# Create state object (holds the KV cache)
state = mlmodel.make_state()

# First forward pass at position 0
output1 = mlmodel.predict(
    {"inputs_embeds": input1, "position_id": np.array([0], dtype=np.int32)},
    state,  # Pass state to persist KV cache
)

# Second forward pass at position seq_len (uses cached KV)
output2 = mlmodel.predict(
    {"inputs_embeds": input2, "position_id": np.array([seq_len], dtype=np.int32)},
    state,  # Same state object - KV cache is preserved
)
```

### Pre-computed Position Embeddings

ANE operates in FP16 which cannot accurately represent position indices beyond ~2048 and loses precision in the exponential, cosine, and sine operations used for rotary position embeddings (RoPE).

**Solution:** Pre-compute all position embeddings at initialization in FP32, then index into them at runtime.

```python
# From src/coremlmodels/lm_model_wrapper.py
class LanguageModelWrapper(nn.Module):
    def __init__(self, model, cache_length=2048):
        # Pre-compute rotary embeddings for all positions in FP32
        position_ids = torch.arange(cache_length, dtype=torch.long).unsqueeze(0)
        cos_emb, sin_emb = model.rotary_emb(dummy_values, position_ids)

        # Store as buffers for indexing
        self.register_buffer("cos_emb", cos_emb[0])  # (cache_length, head_dim)
        self.register_buffer("sin_emb", sin_emb[0])

    def forward(self, inputs_embeds, position_id):
        # Index into pre-computed embeddings at runtime
        # This indexing op typically runs on CPU, but subsequent ops run on ANE
        position_ids = torch.arange(seq_len) + position_id
        position_emb = (self.cos_emb[position_ids], self.sin_emb[position_ids])

        # Pass to attention layers
        for decoder_layer in self.layer.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_emb,  # Pre-indexed embeddings
                ...
            )
```

**Why this matters:**
- Avoids FP16 precision loss in exp/cos/sin calculations
- Indexing operations run on CPU but create minimal overhead
- All subsequent operations run on Neural Engine without graph cuts

### Attention Patching for Language Models

For full language model conversion, attention layers need special handling:
- Channels-first format (NCHW) for ANE compatibility
- Explicit GQA (Grouped Query Attention) loop for better scheduling
- KV cache integration with CoreML state
- External position embeddings (pre-computed at model level)
- **QK-norm support** (Qwen3-style) - Auto-detected and applied before RoPE

See the implementation in `src/coremlmodels/patch_attention.py`:
- `AttentionPatcher` - Wraps attention layers for ANE-optimized execution
- `patch_model_attention()` - Recursively patches attention layers
- `rotate_half()`, `apply_rotary_pos_emb()` - RoPE utilities

**QK-norm (Qwen3-style):**

Some architectures (e.g., Qwen3) apply RMSNorm to Q and K states before RoPE. The `AttentionPatcher` auto-detects this via `hasattr(attention_layer, "q_norm")` and applies normalization using the axis-aware RMSNorm patcher:

```python
# Auto-detected in AttentionPatcher.__init__
self.has_qk_norm = hasattr(attention_layer, "q_norm") and hasattr(attention_layer, "k_norm")

# Applied in forward() before RoPE
if self.has_qk_norm:
    # Query: (batch, num_heads, head_dim, seq) - normalize over axis=2
    query_states = self.q_norm(query_states, axis=2)
    # Key: (batch, num_kv_heads, seq, head_dim) - normalize over axis=-1
    key_states = self.k_norm(key_states, axis=-1)
```

### Architecture Registry

The `registry.py` module provides architecture auto-detection for applying the correct patching configuration:

```python
from coremlmodels import get_architecture_config, find_target_classes, get_supported_architectures

# Check supported architectures
print(get_supported_architectures())  # ['qwen2', 'qwen3']

# Get config for a model type
config = get_architecture_config("qwen3")
# ArchitectureConfig(
#     attention_class_names=('Qwen3Attention',),
#     rmsnorm_class_names=('Qwen3RMSNorm',),
#     has_qk_norm=True,
# )

# Find actual classes from a loaded model
attention_classes, rmsnorm_classes = find_target_classes(model, config)
```

**Adding new architectures:**

To support a new model architecture, add an entry to `ARCHITECTURE_REGISTRY` in `registry.py`:

```python
ARCHITECTURE_REGISTRY["llama"] = ArchitectureConfig(
    attention_class_names=("LlamaAttention",),
    rmsnorm_class_names=("LlamaRMSNorm",),
    has_qk_norm=False,
)
```

## Available Patching Methods

```python
from coremlmodels import (
    patch_model_linears,
    patch_model_rmsnorms,
    patch_model_attention,
    get_architecture_config,
    find_target_classes,
    get_supported_architectures,
)

# Method 1: Patch all Linear layers to 1x1 Conv2d
patched = patch_model_linears(model)

# Method 2: Patch RMSNorm layers to LayerNorm-fusable ops
patched = patch_model_rmsnorms(patched)

# Method 3: Patch attention layers for ANE-optimized execution
patched = patch_model_attention(patched, target_classes, config)

# All patching methods support:
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

# Architecture-aware patching (recommended for HuggingFace models):
arch_config = get_architecture_config(model.config.model_type)
attention_classes, rmsnorm_classes = find_target_classes(model, arch_config)

patched = patch_model_rmsnorms(model, target_classes=rmsnorm_classes)
patched = patch_model_linears(patched)
patched = patch_model_attention(patched, target_classes=attention_classes, config=model.config)
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
| `patch_rmsnorm.py` | `RMSNormToLayerNormPatcher`, `patch_model_rmsnorms` | RMSNorm -> LayerNorm ops (axis-aware) |
| `patch_attention.py` | `AttentionPatcher`, `patch_model_attention` | Attention -> ANE-optimized channels-first with QK-norm |
| `lm_model_wrapper.py` | `LanguageModelWrapper`, `ChunkedLanguageModelWrapper`, `create_coreml_state_specs` | LM wrappers with KV cache state (full and chunked) |
| `chunked_lm_head.py` | `ChunkedLMHead` | Vocabulary-chunked LM head with temperature scaling |
| `export_utils.py` | `export_embeddings`, `convert_lm_head` | Export embeddings and LM head to CoreML |
| `registry.py` | `ArchitectureConfig`, `get_architecture_config`, `find_target_classes` | Architecture auto-detection |
| `graph_passes.py` | `fuse_layernorm_or_instancenorm`, `register_extended_passes` | Extended CoreMLTools graph passes |
| `analysis.py` | `analyze_compute_plan`, `inspect_mil_program` | CoreML verification tools |

## Extended Graph Passes

### Why Custom Graph Passes?

CoreMLTools applies graph optimization passes during `ct.convert()` that detect operation patterns and fuse them into optimized operations. However, the default `fuse_layernorm_or_instancenorm` pass only recognizes LayerNorm patterns when normalization happens over **axis=1**.

This limitation causes problems for:
- **QK-norm in attention**: Query states need normalization over axis=2 (head_dim in BCHW format)
- **Gamma-only patterns**: RMSNorm has gamma but no beta, which the original pass didn't handle for single-axis cases

Without fusion, the computation graph contains separate operations:
```
reduce_mean -> sub -> mul -> reduce_mean -> add -> rsqrt -> mul -> mul(gamma)
```

With our extended pass, these fuse into a single optimized operation:
```
layer_norm (with gamma included)
```

### Using Extended Passes

The `graph_passes.py` module provides an extended version of CoreMLTools' fusion pass. **Import it before calling `ct.convert()`**:

```python
from coremlmodels import register_extended_passes

# Register extended passes (overrides CoreMLTools defaults)
register_extended_passes()

# Now convert - LayerNorm fusion works for any single axis
mlmodel = ct.convert(traced_model, ...)
```

See `src/coremlmodels/graph_passes.py` for implementation details and inline documentation of the modifications.

### Design Guideline: Minimize Tensor Manipulations

Because the extended pass supports normalization over **any single axis**, you should **avoid using transpose, permute, or reshape** operations just to move the target axis to position 1 before normalization.

**Good** - Normalize directly over the target axis:
```python
# Query states: (batch, heads, head_dim, seq) - normalize over head_dim (axis=2)
out = rmsnorm_patcher(query_states, axis=2)
```

**Bad** - Transpose to make target axis=1, normalize, transpose back:
```python
# Don't do this! Unnecessary transposes may cause graph cuts
x = query_states.permute(0, 2, 1, 3)  # Move head_dim to axis=1
out = rmsnorm_patcher(x, axis=1)
out = out.permute(0, 2, 1, 3)  # Move back
```

This keeps the computation graph cleaner and avoids potential inefficiencies on the Neural Engine.

## Known Limitations

**Neural Engine Model Size Limit (~2GB):** Apple's Neural Engine can only load models up to approximately 2GB. For larger models (e.g., Qwen3-4B with 4 billion parameters), you must split the model into multiple chunks.

## Chunked Model Conversion for Large Models

For models exceeding the ~2GB Neural Engine limit, use the `--num-chunks` argument to split the model into multiple CoreML packages:

```bash
# Convert Qwen3-4B in 4 chunks
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-4B --num-chunks 4

# Test chunking with fewer layers (faster debugging)
uv run python examples/lm_conversion_example.py --num-layers 12 --num-chunks 3
```

**How chunking works:**
- The model's transformer layers are partitioned into N chunks
- Each chunk has its own independent KV cache (sized for its layer count)
- Chunks are converted to separate `.mlpackage` files in an output directory
- At inference time, chain the chunks sequentially: output of chunk N feeds into chunk N+1
- The final chunk applies the model's output layer norm

**Key classes:**
- `ChunkedLanguageModelWrapper` - Wraps a subset of layers with its own KV cache
- `create_chunked_coreml_state_specs()` - Creates CoreML state specs for a chunk

**Verification:** The conversion script performs end-to-end verification comparing the original PyTorch model output against the chained CoreML chunks output.

## Embeddings and LM Head Export

For separate export of embeddings and language model head components, the library provides utilities to convert these independently:

### Embeddings Export

Embeddings are exported as `.npy` files in float16 format:

```python
from coremlmodels import export_embeddings

# Export embeddings from a HuggingFace model
export_embeddings(model, output_path="embeddings.npy", verbose=True)
```

**Output:**
- `.npy` file containing embedding weights in float16 format
- Shape: `(vocab_size, hidden_dim)`

### LM Head Export with Chunking

The Neural Engine has a weight dimension limit of ~16384. For language models with large vocabularies, the LM head weights `(vocab_size, hidden_dim)` may exceed this limit. The `convert_lm_head()` function addresses this by:

1. **Chunking the vocabulary** into smaller pieces (default: 6144 tokens per chunk)
2. **Computing log-sum-exp per chunk** with numerical stability for float16
3. **Adding temperature scaling** as an input parameter

```python
from coremlmodels import convert_lm_head

mlmodel = convert_lm_head(
    lm_head=model.lm_head,
    batch_size=1,
    hidden_dim=2048,
    seq_len=8,
    output_path="lm_head.mlpackage",
    chunk_size=6144,           # Vocabulary chunk size
    compute_logsumexp=True,    # Compute log-sum-exp for each chunk
    verbose=True,
)
```

**Input Format:**
- `hidden_states`: `(batch, hidden_dim, 1, seq_len)` - Hidden states from the model
- `temperature`: `(1, 1, 1, 1)` - Temperature scaling factor for logits

**Output Format (when `compute_logsumexp=True`):**

The model returns 3 outputs that allow reconstructing the full log-sum-exp:

1. `logits`: `(batch, vocab_size, 1, seq_len)` - Temperature-scaled logits for all vocabulary tokens
2. `chunk_logsumexp_stable`: `(batch, num_chunks, 1, seq_len)` - Stable log-sum-exp component per chunk: `log(sum(exp(x/T - max(x/T))))`
3. `chunk_max`: `(batch, num_chunks, 1, seq_len)` - Maximum value per chunk: `max(x/T)`

**Reconstructing full log-sum-exp:**

```python
# For each chunk: chunk_lse = chunk_logsumexp_stable + chunk_max
chunk_lse = chunk_logsumexp_stable + chunk_max  # (batch, num_chunks, 1, seq_len)

# Apply logsumexp reduction over chunks to get final value
import torch
full_logsumexp = torch.logsumexp(torch.from_numpy(chunk_lse), dim=1, keepdim=True)
```

### Usage Examples


```bash
# Export embeddings and LM head along with full model conversion
uv run python examples/lm_conversion_example.py --export-embeddings --export-lm-head

# Export only components (skip main model conversion)
uv run python examples/lm_conversion_example.py --export-embeddings --export-lm-head --components-only
```

## Full Language Model Conversion Example

For converting complete HuggingFace language models with KV cache, see:

```bash
# Convert Qwen2 (default)
uv run python examples/lm_conversion_example.py

# Convert Qwen3 (with QK-norm)
uv run python examples/lm_conversion_example.py --model Qwen/Qwen3-0.6B

# Custom parameters
uv run python examples/lm_conversion_example.py \
    --model Qwen/Qwen2-0.5B \
    --seq-len 16 \
    --cache-length 512 \
    --output my_model.mlpackage

# Quiet mode (less verbose output)
uv run python examples/lm_conversion_example.py --quiet
```

**Supported architectures:** Qwen2, Qwen3 (see `get_supported_architectures()`)

This example demonstrates:
1. **Auto-detecting architecture** from model config using the registry
2. **Finding target classes** dynamically (attention, RMSNorm)
3. Patching Linear, RMSNorm, and Attention layers (with QK-norm if applicable)
4. Wrapping with `LanguageModelWrapper` for KV cache as CoreML state
5. Converting with `ct.StateType` for stateful inference
6. Comparing PyTorch vs CoreML outputs
7. Analyzing compute plan and MIL program

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
