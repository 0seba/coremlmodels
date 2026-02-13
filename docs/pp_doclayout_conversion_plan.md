# PP-DocLayoutV3 CoreML Conversion Plan

> **Status**: Validated Prototype (Deformable Attention Verified)
> **Goal**: Convert PP-DocLayoutV3 (RT-DETR based) to CoreML with Neural Engine support.

This document serves as the implementation guide for converting the PP-DocLayoutV3 model. It consolidates architectural research and the results of a successful proof-of-concept for the model's most critical component: multi-scale deformable attention.

> **Reference**: For detailed architecture analysis, see [pp_doclayout_research.md](./pp_doclayout_research.md).

---

## 1. Architecture Overview

**Model**: PP-DocLayoutV3 (PaddlePaddle/PP-DocLayoutV3_safetensors)
**Type**: RT-DETR (Real-Time Detection Transformer) adapted for layout analysis + mask prediction.
**Input**: Fixed size **800×800** RGB image.

### High-Level Data Flow

1.  **Input Image** `(1, 3, 800, 800)`
2.  **Backbone (HGNetV2-L)**: Extracts multi-scale feature maps.
    -   Strides: 8, 16, 32.
    -   Channels: [512, 1024, 2048].
3.  **Hybrid Encoder**:
    -   **AIFI (Attention-based Intra-scale Feature Interaction)**: Transformer encoder on the stride-32 feature map.
    -   **FPN (Feature Pyramid Network)**: Top-down pathway with upsampling.
    -   **PAN (Path Aggregation Network)**: Bottom-up pathway with fusion.
    -   **Mask Feature Head**: Specific branch for generating high-resolution mask features.
4.  **Transformer Decoder (6 Layers)**:
    -   **Self-Attention**: Standard multi-head attention between object queries.
    -   **Cross-Attention (CRITICAL)**: **Multi-Scale Deformable Attention** attending to encoder features.
    -   Predicts: Class logits, Bounding Boxes, Mask weights.
5.  **Output Heads**:
    -   `logits`: `(1, 300, 25)` - Classification scores for up to 300 detected elements.
    -   `pred_boxes`: `(1, 300, 4)` - Bounding boxes (cx, cy, w, h).
    -   `mask_logits`: `(1, 300, 200, 200)` - Instance segmentation masks.

---

## 2. Component Analysis & References

### A. Image Preprocessing (Fixed)
The image processor enforces a strict resize to 800×800, allowing us to **hardcode all spatial dimensions** in the CoreML model.

-   **Input**: `(batch, 3, 800, 800)`
-   **Normalization**: Mean=[0,0,0], Std=[1,1,1] (effectively identity after 0-1 rescale).
-   **Reference**: `transformers/models/pp_doclayout_v3/image_processing_pp_doclayout_v3_fast.py`

### B. Multi-Scale Deformable Attention (The Hard Part)
This module uses `grid_sample` to sample features from multiple spatial levels (strides 8, 16, 32) based on learned offsets.

-   Original File: `transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py` (Class `MultiScaleDeformableAttention`)
-   **Challenge 1**: Uses `grid_sample` with 6D tensors `(Batch, Queries, Heads, Levels, Points, 2)`. **CoreML Max Rank is 5.**
-   **Challenge 2**: Uses dynamic shape operations (`value.split([...])`) based on runtime spatial shapes.

### C. Post-Processing (Host Side)
Operations like Polygon Extraction (using `cv2.findContours`) **cannot** run in CoreML.
-   **Decision**: The CoreML model will output raw `logits`, `boxes`, and `masks`. All post-processing happens in Python/Swift on the host.

---

## 3. Verified Implementation Strategy

We have successfully prototyped and verified the Deformable Attention conversion in `examples/deformable_attention_test.py`.
**The following strategy MUST be used for the full model conversion:**

### ✅ Fix 1: Hardcoded Spatial Dimensions
Since input is fixed to 800×800, the feature map logical sizes are deterministic. **Do not use dynamic shape calculations.**

| Stride | Feature Map Size | Token Count | Split Index (Cumulative) |
| :--- | :--- | :--- | :--- |
| 8 | 100 × 100 | 10,000 | 0 - 10,000 |
| 16 | 50 × 50 | 2,500 | 10,000 - 12,500 |
| 32 | 25 × 25 | 625 | 12,500 - 13,125 |

**Implementation Requirement**:
In your wrapper/module, define these constants and use standard Python `int`s for all `reshape`/`view` operations. CoreML tracing fails if it sees tensor-derived shapes.

### ✅ Fix 2: Rank Reduction for `grid_sample`
To bypass the Rank-5 limit, we must split the operation by feature level.

**Correct Implementation Pattern:**

```python
# 1. Split value features (Batch, Total_Seq, Heads, Dims)
v0 = value[:, :10000, :, :]
v1 = value[:, 10000:12500, :, :]
v2 = value[:, 12500:, :, :]

# 2. Reshape sampling offsets/grids inputs (Batch, Queries, ...)
# Instead of one giant (B, Q, H, L, P, 2) tensor, split 'L' dimension
grid_0 = all_grids[:, :, 0, :, :]  # Level 0 grid (Rank 4)
grid_1 = all_grids[:, :, 1, :, :]  # Level 1 grid
...

# 3. Process each level independently (All tensors Rank <= 4)
# Reshape value to image-like (B*H, C, H, W)
val_0 = v0.flatten(2).transpose(1, 2).reshape(BxH, HEAD_DIM, 100, 100)
# Grid sample
sampled_0 = F.grid_sample(val_0, grid_0, align_corners=False)

# 4. Stack results (Creates Rank 5, which is allowed)
stacked = torch.stack([sampled_0, sampled_1, sampled_2], dim=-2)
```

**See `examples/deformable_attention_test.py` for the exact, verified code.**

---

## 4. Conversion Workflow

### Step 1: Create the Model Wrapper
Create `coremlmodels/wrappers/pp_layout_wrapper.py`.
-   **Class**: `PPLayoutCoreMLWrapper(nn.Module)`
-   **Init**: Load the HuggingFace model. Replace its `decoder.layers` attention modules with your custom `CoreMLDeformableAttention`.
-   **Forward**:
    -   Input: `pixel_values` (1, 3, 800, 800).
    -   Run backbone -> encoder -> decoder.
    -   Return: `logits`, `pred_boxes`, `masks` (Raw tensors).

### Step 2: Implement `CoreMLDeformableAttention`
Copy logic from [`examples/deformable_attention_test.py`](file:///Users/seba/Documents/mydeving/coremlmodels/examples/deformable_attention_test.py).
-   This custom module replaces `PPDocLayoutV3MultiscaleDeformableAttention`.
-   It must implement the linear projections (value_proj, sampling_offsets, etc.) just like the original, but use the **split-level grid_sample logic** in the forward pass.

### Step 3: Convert
Use `coremltools` to trace and convert.
-   **Inputs**: `[ct.TensorType(name="pixel_values", shape=(1, 3, 800, 800))]`
-   **Precision**: Float16.
-   **Compute Units**: `ctx.ComputeUnit.CPU_AND_NE`.

### Step 4: Validate
-   Compare outputs against the original PyTorch model using the `deformable_attention_test.py` verification logic (max diff < 1e-1).

---

## 5. References

### 1. Prototype & Verification Script
**File**: [`examples/deformable_attention_test.py`](file:///Users/seba/Documents/mydeving/coremlmodels/examples/deformable_attention_test.py)
**Importance**: **CRITICAL**. This file contains the *solution*. It shows exactly how to structure the deformable attention to pass CoreML compilation and run on Neural Engine. Use `DeformableAttentionFull` class as your template.

### 2. Original Model Implementation
**File**: [`transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py`](file:///Users/seba/Documents/mydeving/coremlmodels/.venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py)
**Sections**:
-   `MultiScaleDeformableAttention` (Lines ~71-123): Original incompatible implementation.
-   `PPDocLayoutV3HybridEncoder` (Lines ~867): Encoder logic (likely converts fine without changes).

### 3. Image Processing
**File**: [`transformers/models/pp_doclayout_v3/image_processing_pp_doclayout_v3_fast.py`](file:///Users/seba/Documents/mydeving/coremlmodels/.venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/image_processing_pp_doclayout_v3_fast.py)
**Importance**: confirms the 800x800 input requirement and details the post-processing logic (polygons) that must be reimplemented on the host.
