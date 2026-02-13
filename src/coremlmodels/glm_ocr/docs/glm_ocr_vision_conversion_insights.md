# GLM-OCR Vision: CoreML Conversion Insights

## 1. Key Design Decisions

### **Why Patch-Level Padding?**
CoreML's **Enumerated Shapes** work best when a single dimension varies.
- **Problem:** Images have varying aspect ratios (e.g., `100x200` vs `200x100`), making fixed `(H, W)` shapes impractical.
- **Solution:** We flattened patches into a 1D sequence `(N, C)` where `N` is the total number of patches.
- **Why it works:** The `PatchEmbed` layer uses `Conv3d` independently on each patch. By padding the *number of patches* to the nearest enumerated count (e.g., 32, 64, ..., 16384), we support any aspect ratio with a single model.
- **Implementation:** Host-side padding to `num_total_patches`, then zeroing out padding embeddings with a `patch_mask` after Conv3d.

### **Channels-First Architecture (1, C, 1, N)**
Apple's **Neural Engine (ANE)** is highly optimized for 4D tensors where `C` is a multiple of 16/32.
- **Standard Format:** `(B, N, C)` is memory-intensive on ANE.
- **Our Format:** `(1, C, 1, N)` allows `Conv2d` (patched from Linear) to process the sequence efficiently.
- **Impact:** All transformer blocks, attention, and MLP layers operate on this 4D shape.

### **Host-Side RoPE**
Rotary Position Embeddings involve complex trigonometric operations (`sin`, `cos`, `rotate_half`) that are:
1.  **Expensive** on ANE (many element-wise ops).
2.  **Prone to Precision Issues** in FP16 (sines/cosines can drift).
- **Solution:** Pre-compute `cos` and `sin` tables on the host CPU in FP32, then pass them as model inputs. The model only performs simple element-wise multiplication.

---

## 2. Pitfalls & Bugs ("What Went Wrong")

### **The "Flattened Merger" Trap**
**Symptom:** `RuntimeError: Expected 4D input to conv2d, but got [N, C]`.
- **Cause:** The original `GlmOcrVisionModel` flattens the output of `downsample` to 2D `(num_groups, hidden_size)` before passing it to `merger`.
- **Conflict:** We patched all `Linear` layers in `merger` to `Conv2d`, which **require 4D input**.
- **Fix:** Reshaped the downsample output to `(1, hidden_size, 1, num_groups)` (channels-first 4D) before passing to `merger`.

### **LayerNorm in Channels-First (The "Wrong Dimension" Bug)**
**Symptom:** Silent failure (poor/wrong outputs) or shape mismatches.
- **Cause:** `nn.LayerNorm(dim)` normalizes over the **last dimension**.
    - In standard `(B, N, dim)`, it normalizes `dim` (correct).
    - In channels-first `(1, dim, 1, N)`, it normalizes `N` (wrong!).
- **Fix:** Created `VisionPatchMergerPatcher` to explicitly wrap the merger's LayerNorm:
    1.  Permute `(1, dim, 1, N) → (1, N, 1, dim)`
    2.  Apply `LayerNorm`
    3.  Permute back `(1, N, 1, dim) → (1, dim, 1, N)`

### **Dynamic Int Ops (Tracing Failure)**
**Symptom:** `TypeError: only 0-dimensional arrays can be converted to Python scalars` during conversion.
- **Cause:** `x.size(dim) // 2` in `_apply_rotary`.
    - Tracing records `x.size(dim)` as a tensor scalar.
    - `// 2` tries to perform integer division on a tracer, creating a dynamic `int` op that CoreML doesn't support well for shape/slice indices.
- **Fix:** Passed `half_dim` as a Python `int` constant (calculated from static `config.hidden_size`) to `_apply_rotary`.

### **Relative Imports**
**Symptom:** `ImportError: attempted relative import beyond top-level package`.
- **Cause:** Importing `GlmOcrRMSNorm` directly from the local `glm_ocr` folder broke because its internal relative imports expected to be part of a package.
- **Fix:** Removed the import. Instead, dynamically inspected `model.modules()` to find any class with "RMSNorm" in its name and patched those.

---

## 3. Summary of Patched Components

| Component | Original Type | Patched To | Reason |
| :--- | :--- | :--- | :--- |
| **Linear** | `nn.Linear` | `nn.Conv2d` | ANE optimization (uses 4D input) |
| **RMSNorm** | `GlmOcrRMSNorm` | `nn.LayerNorm` (op) | ANE optimization (fusable) |
| **Attention** | `GlmOcrVisionAttention` | `VisionAttentionPatcher` | Channels-first, Non-causal, Explicit Mask |
| **Blocks** | `GlmOcrVisionBlock` | `VisionBlockPatcher` | Channels-first flow |
| **Merger** | `GlmOcrVisionPatchMerger` | `VisionPatchMergerPatcher` | Handle LayerNorm in channels-first |

---

## 4. Duplicated Table Row Investigation (Final Findings)

### **Observed Issue**
- OCR on some table images produced duplicated rows even after reconverting the vision model.

### **Initial Hypothesis**
- Padding tokens in the vision path might still influence outputs after downsampling/merger.

### **What We Verified**
- Padding suppression in vision attention/downsampling was not the primary cause of the duplicated-row behavior.
- A real vision bug did exist and was fixed:
  - The merger input reshape used a direct `view` from `(num_groups, C, 1, 1)` to `(1, C, 1, num_groups)`, which reinterpreted memory incorrectly.
  - Correct fix: transpose first, then `contiguous().view(...)`.

### **Root Cause (Primary)**
- The text pipeline used legacy 1D `position_id` semantics during multimodal prefill.
- GLM-OCR requires multimodal **3D mRoPE position indexing** (`get_rope_index`) for image+text sequences.
- Because the converted text model expected explicit RoPE embeddings, feeding only sequential 1D positions caused token-position drift in table regions, which manifested as structural duplication.

### **Fix Applied**
- Added host-side multimodal mRoPE flow to the text path:
  - Compute multimodal `position_ids` and `rope_delta` with HF reference logic.
  - Generate and pass explicit `position_cos` / `position_sin` for prefill and decode.
  - Keep backward-compatible fallback for legacy text models.

### **Supporting Robustness Changes**
- Cache loading now prefers `.mlmodelc` when present and falls back to `.mlpackage` on load failure.
- Cache-length detection now infers from available state specs instead of requiring a hardcoded `key_cache` entry.

### **Outcome**
- After mRoPE alignment, the duplicated-row issue was resolved.
- Final conclusion: this was primarily a **text positional encoding mismatch**, not a padding-mask failure in vision.
