# GLM-OCR Multi-Token Prediction (MTP) Speculative Decoding

**Research findings for implementing MTP speculative decoding in the CoreML inference pipeline.**

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [MTP Module Structure](#2-mtp-module-structure)
3. [Weight Layout in Safetensors](#3-weight-layout-in-safetensors)
4. [Multi-Step MTP: Drafting Multiple Tokens with One Layer](#4-multi-step-mtp-drafting-multiple-tokens-with-one-layer)
5. [MTP KV Cache Behavior](#5-mtp-kv-cache-behavior)
6. [CoreML Model Architecture Decisions](#6-coreml-model-architecture-decisions)
7. [Implementation Plan](#7-implementation-plan)
8. [File Reference](#8-file-reference)

---

## 1. Architecture Overview

GLM-OCR uses a **Multi-Token Prediction (MTP)** module for speculative decoding. The idea is identical to DeepSeek-V3's MTP approach:

- The **main model** (16 transformer layers) produces hidden states and generates one verified token per forward pass.
- The **MTP module** (1 extra transformer layer, stored as layer index 16) takes the main model's last hidden state + the embedding of the last predicted token and predicts the **next** token in a single cheap forward pass.
- The MTP module can be run **multiple times in sequence** (autoregressive chaining), feeding its own output hidden states back as input to draft multiple tokens ahead — just like EAGLE.
- The main model then verifies all draft tokens in a single batched forward pass.

GLM-OCR has `num_nextn_predict_layers = 1`, meaning there is exactly **one MTP layer**. But this single layer can be chained N times to draft N speculative tokens. The GLM-OCR repo's recommended launch command confirms this:

```bash
python -m sglang.launch_server --model zai-org/GLM-OCR \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \       # 3 sequential MTP forward passes
    --speculative-eagle-topk 1 \      # greedy draft (top-1)
    --speculative-num-draft-tokens 4  # total 4 draft tokens
```

### High-Level Data Flow (Single Draft Step)

```
                    Main Model (16 layers)
                    ┌─────────────────────┐
 input_embeds ──────► Transformer Blocks  ├──── hidden_states ──┬──► Final Norm ──► LM Head ──► token_main
                    │  (layers 0..15)     │                     │
                    │  KV Cache (state)   │                     │
                    └─────────────────────┘                     │
                                                                │
                    MTP Module (1 layer)                         │
                    ┌─────────────────────┐                     │
 embed(token_main)──► enorm ──┐           │                     │
                    │         ├─ concat ──► eh_proj ──► Transformer Block ──► shared_head_norm ──► LM Head ──► token_draft
 hidden_states ─────► hnorm ──┘           │  (layer 16)                                            (same weights)
                    │                     │  KV Cache (own state)
                    └─────────────────────┘
```

### Multi-Step Chaining (3 Draft Tokens)

```
Main Model hidden_states ──────────────┐
embed(token_main) ─────────────────────┤
                                       ▼
                                MTP forward pass 1 ──► draft_token_0, mtp_hidden_0
                                       │
                    mtp_hidden_0 ──────┤
                    embed(draft_0) ────┤
                                       ▼
                                MTP forward pass 2 ──► draft_token_1, mtp_hidden_1
                                       │
                    mtp_hidden_1 ──────┤
                    embed(draft_1) ────┤
                                       ▼
                                MTP forward pass 3 ──► draft_token_2, mtp_hidden_2
```

Each MTP forward pass feeds the previous MTP output hidden states (not the main model's hidden states) as `previous_hidden_states`. The MTP's own KV cache grows by one position per forward pass.

Key: the MTP module reuses the **same LM head weights** as the main model (the `shared_head.head.weight` in safetensors equals `lm_head.weight`).

---

## 2. MTP Module Structure

### 2.1 Components

From the safetensors weight names for layer 16:

| Component | Weight | Shape | Purpose |
|-----------|--------|-------|---------|
| `enorm` | `enorm.weight` | [1536] | RMSNorm on token embedding input |
| `hnorm` | `hnorm.weight` | [1536] | RMSNorm on previous hidden states |
| `eh_proj` | `eh_proj.weight` | [1536, 3072] | Linear projection: concat(enorm_out, hnorm_out) → hidden_size |
| `shared_head.norm` | `shared_head.norm.weight` | [1536] | RMSNorm before LM head (equivalent to model's final_norm) |
| `shared_head.head` | `shared_head.head.weight` | [59392, 1536] | LM head (shared with main model) |
| `embed_tokens` | `embed_tokens.weight` | [59392, 1536] | Embedding table (shared with main model) |
| Transformer block | `self_attn.*`, `mlp.*`, `*_layernorm.*` | various | Standard GLM-OCR decoder layer (4-norm sandwich pattern) |

### 2.2 Forward Pass (Pseudocode)

```python
def mtp_forward(previous_hidden_states, input_embeds, position_id, kv_cache):
    """
    Args:
        previous_hidden_states: Hidden state from main model OR previous MTP step.
            Shape: (batch, hidden_dim, 1, 1) in channels-first (single token).
        input_embeds: Embedding of the token predicted by main model or previous MTP step.
            Shape: (batch, hidden_dim, 1, 1) in channels-first (single token).
        position_id: Current position in the MTP's KV cache (int32 scalar).
        kv_cache: MTP's own KV cache state (separate from main model).

    Returns:
        mtp_hidden: Hidden states for chaining to the next MTP forward pass.
        mtp_hidden_for_head: Normalized hidden states ready for LM head.
    """
    # Step 1: Mask embeddings at position 0 (MTP doesn't need BOS context)
    # In channels-first: zero out tokens where position == 0
    input_embeds = mask_position_zero(input_embeds, position_id)

    # Step 2: Normalize both inputs independently
    input_embeds = enorm(input_embeds)          # RMSNorm
    previous_hidden_states = hnorm(previous_hidden_states)  # RMSNorm

    # Step 3: Concatenate along channel dim and project
    combined = concat([input_embeds, previous_hidden_states], dim=channels)  # → (B, 3072, 1, 1)
    hidden = eh_proj(combined)  # → (B, 1536, 1, 1)

    # Step 4: Run through standard transformer decoder layer
    # This uses its OWN KV cache, position embeddings, causal mask
    hidden = transformer_block(
        hidden,
        attention_mask=causal_mask,
        kv_cache=kv_cache,
        position_embeddings=rope(position_id),
    )

    # Step 5: Apply shared head norm
    hidden_for_head = shared_head_norm(hidden)  # RMSNorm

    return hidden, hidden_for_head
```

### 2.3 Token Prediction from MTP Output

After getting `hidden_for_head`, the token is predicted by running it through the **same LM head** used by the main model:

```python
logits = lm_head(hidden_for_head)   # → (batch, vocab_size, 1, 1)
draft_token = argmax(logits)         # greedy for draft
```

---

## 3. Weight Layout in Safetensors

The MTP weights are stored as "layer 16" (i.e., `model.language_model.layers.16.*`) in the same safetensors file as the main model. The main model only loads layers 0-15 (`num_hidden_layers = 16`).

### Standard Layer (e.g., layer 15):
```
model.language_model.layers.15.input_layernorm.weight           [1536]
model.language_model.layers.15.post_self_attn_layernorm.weight  [1536]
model.language_model.layers.15.post_attention_layernorm.weight  [1536]
model.language_model.layers.15.post_mlp_layernorm.weight        [1536]
model.language_model.layers.15.self_attn.q_proj.weight          [2048, 1536]
model.language_model.layers.15.self_attn.k_proj.weight          [1024, 1536]
model.language_model.layers.15.self_attn.v_proj.weight          [1024, 1536]
model.language_model.layers.15.self_attn.o_proj.weight          [1536, 2048]
model.language_model.layers.15.mlp.gate_up_proj.weight          [9216, 1536]
model.language_model.layers.15.mlp.down_proj.weight             [1536, 4608]
```

### MTP Layer (layer 16) — MTP-specific components marked:
```
model.language_model.layers.16.input_layernorm.weight           [1536]       ← transformer block
model.language_model.layers.16.post_self_attn_layernorm.weight  [1536]       ← transformer block
model.language_model.layers.16.post_attention_layernorm.weight  [1536]       ← transformer block
model.language_model.layers.16.post_mlp_layernorm.weight        [1536]       ← transformer block
model.language_model.layers.16.self_attn.q_proj.weight          [2048, 1536] ← transformer block
model.language_model.layers.16.self_attn.k_proj.weight          [1024, 1536] ← transformer block
model.language_model.layers.16.self_attn.v_proj.weight          [1024, 1536] ← transformer block
model.language_model.layers.16.self_attn.o_proj.weight          [1536, 2048] ← transformer block
model.language_model.layers.16.mlp.gate_up_proj.weight          [9216, 1536] ← transformer block
model.language_model.layers.16.mlp.down_proj.weight             [1536, 4608] ← transformer block
model.language_model.layers.16.enorm.weight                     [1536]       ← MTP-specific
model.language_model.layers.16.hnorm.weight                     [1536]       ← MTP-specific
model.language_model.layers.16.eh_proj.weight                   [1536, 3072] ← MTP-specific
model.language_model.layers.16.shared_head.norm.weight          [1536]       ← MTP-specific
model.language_model.layers.16.shared_head.head.weight          [59392, 1536] ← LM head (shared)
model.language_model.layers.16.embed_tokens.weight              [59392, 1536] ← Embeddings (shared)
```

The `shared_head.head.weight` and `embed_tokens.weight` are copies of / shared with the main model's `lm_head.weight` and `model.embed_tokens.weight` respectively. In our CoreML implementation we **do not need to load these** — we already have them as the separately exported `embeddings.npy` and `lm_head.mlpackage`.

### Weight Loading for CoreML Conversion

To convert the MTP module, you must:
1. Open the safetensors file directly (the MTP layer is NOT loaded by HuggingFace's `AutoModel`)
2. Build the MTP module (enorm, hnorm, eh_proj, shared_head_norm, and a `GlmOcrTextDecoderLayer`)
3. Map parameter names to safetensors keys with the `model.language_model.layers.16.` prefix
4. Load only the MTP-unique weights (skip `shared_head.head` and `embed_tokens` since we already export them separately)

---

## 4. Multi-Step MTP: Drafting Multiple Tokens with One Layer

### 4.1 The Standard (Non-MTP) Decode Loop

Current pipeline in `GlmOcrCoreMLPipeline.run()`:

```
for each token to generate:
    1. logits = lm_head(last_hidden_state)
    2. token = sample(logits)
    3. if token == EOS: break
    4. embed = embeddings[token]
    5. last_hidden_state = text_model(embed, position)  ← EXPENSIVE: full 16-layer forward
    6. position += 1
```

Each iteration requires one full main model forward pass (16 transformer layers).

### 4.2 How Multi-Step MTP Works (from vLLM's eagle.py)

Even with `num_nextn_predict_layers = 1`, vLLM runs the single MTP layer **multiple times sequentially** to draft N tokens. This is confirmed by the GLM-OCR sglang config: `--speculative-num-steps 3`.

The chaining mechanism from vLLM's `EagleProposer.propose()`:

```python
# Step 0: Initial MTP forward pass
# Input: main model's hidden_states + embed(token_main)
mtp_hidden_0 = mtp_model(
    previous_hidden_states=main_model_hidden,  # from target model
    input_embeds=embed(token_main),
    position=mtp_pos,
)
draft_token_0 = argmax(lm_head(shared_head_norm(mtp_hidden_0)))
mtp_pos += 1

# Step 1: Chain MTP with its own output
# Input: mtp_hidden_0 + embed(draft_token_0)
mtp_hidden_1 = mtp_model(
    previous_hidden_states=mtp_hidden_0,       # ← MTP's OWN output, not main model
    input_embeds=embed(draft_token_0),
    position=mtp_pos,
)
draft_token_1 = argmax(lm_head(shared_head_norm(mtp_hidden_1)))
mtp_pos += 1

# Step 2: Chain again
# Input: mtp_hidden_1 + embed(draft_token_1)
mtp_hidden_2 = mtp_model(
    previous_hidden_states=mtp_hidden_1,       # ← previous MTP output
    input_embeds=embed(draft_token_1),
    position=mtp_pos,
)
draft_token_2 = argmax(lm_head(shared_head_norm(mtp_hidden_2)))
mtp_pos += 1
```

Key vLLM code (eagle.py lines 659, 676, 688):
```python
# After each MTP forward pass:
self.hidden_states[:batch_size] = hidden_states           # Store MTP output
# ...
model_kwargs["hidden_states"] = self.hidden_states[...]   # Feed back as input
# ...
ret_hidden_states = self.model(**model_kwargs)             # Next MTP forward pass
```

### 4.3 Batch Verification with Main Model

After drafting N tokens, the main model verifies them all in a **single batched forward pass**:

```python
# Draft tokens: [token_main, draft_0, draft_1, draft_2]
# Embed all and send to main model as one chunk
embed_sequence = stack([embed(token_main), embed(draft_0), embed(draft_1), embed(draft_2)])
# Shape: (4, hidden_dim) → padded to (seq_len, hidden_dim)

hidden = text_model(embed_sequence, position)  # ONE forward pass, 4 real tokens

# Main model produces hidden states at 4 positions
# Position 0 → logits verify draft_0
# Position 1 → logits verify draft_1
# Position 2 → logits verify draft_2
# Position 3 → logits for next token (bonus)

for i, draft in enumerate([draft_0, draft_1, draft_2]):
    logits_i = lm_head(hidden[:, :, :, i:i+1])
    verify_token = sample(logits_i)
    if verify_token == draft:
        accept(draft)           # Free token!
    else:
        # Do NOT accept correction here — hidden[i+1..] are contaminated.
        # The correction emerges naturally in the next decode iteration.
        break
```

**This is where the real speedup comes from**: instead of 4 separate main model forward passes, we do 1 main model pass + 3 cheap MTP passes. If all 3 drafts are accepted, we get 4 tokens for the cost of ~1.2 main model passes.

### 4.4 MTP Model Outputs for Chaining

The MTP model must return **two outputs**:
- `mtp_hidden`: Raw hidden states — fed back as `previous_hidden_states` to the next MTP step
- `mtp_hidden_for_head`: Normalized hidden states (after `shared_head_norm`) — sent to LM head for token prediction

Both are essential. `mtp_hidden` enables chaining; `mtp_hidden_for_head` enables token prediction.

### 4.5 The Complete Speculative Decode Loop

```python
# Configuration
NUM_SPEC_STEPS = 3  # number of MTP forward passes per speculation round

for each decode round:
    # === Phase 1: Sample from main model ===
    logits_main = lm_head(last_hidden)
    token_main = sample(logits_main)
    if token_main == EOS: break
    accept(token_main)

    # === Phase 2: Draft N tokens with MTP chaining ===
    draft_tokens = []
    prev_hidden = last_hidden[:, :, :, last_token_index:last_token_index+1]
    prev_embed = to_cf(embeddings[token_main])

    for step in range(NUM_SPEC_STEPS):
        mtp_hidden, mtp_for_head = mtp_model(prev_hidden, prev_embed, mtp_pos)
        mtp_pos += 1
        draft_logits = lm_head(mtp_for_head)
        draft_token = argmax(draft_logits)
        draft_tokens.append(draft_token)

        if draft_token in eos_ids:
            break

        # Chain: MTP output becomes next input
        prev_hidden = mtp_hidden
        prev_embed = to_cf(embeddings[draft_token])

    # === Phase 3: Verify all drafts with main model (single batched forward) ===
    # Build embedding sequence: [token_main, draft_0, draft_1, ..., draft_N-1]
    all_embeds = [embeddings[token_main]]
    for dt in draft_tokens:
        all_embeds.append(embeddings[dt])
    embed_sequence = stack(all_embeds)  # shape: (1+N, hidden_dim)
    embed_chunk = pad_to_seq_len(embed_sequence)

    hidden = text_model(embed_chunk, current_position)
    num_real = len(all_embeds)

    # Verify each draft token
    num_accepted = 0
    for i, draft in enumerate(draft_tokens):
        verify_logits = lm_head(hidden[:, :, :, i:i+1])
        verify_token = sample(verify_logits)

        if verify_token == draft:
            accept(draft)
            num_accepted += 1
        else:
            # Reject: accept the correction token instead
            if verify_token not in eos_ids:
                accept(verify_token)
            num_accepted += 1  # the correction counts as an accepted position
            break

    # If ALL drafts accepted, we also get a bonus token from position N
    if num_accepted == len(draft_tokens):
        bonus_logits = lm_head(hidden[:, :, :, num_real-1:num_real])
        bonus_token = sample(bonus_logits)
        if bonus_token not in eos_ids:
            accept(bonus_token)
            num_accepted += 1

    # === Phase 4: Update positions ===
    # Only advance by accepted tokens (token_main + accepted drafts).
    # Stale KV entries from rejected drafts are beyond current_position,
    # masked out by causal attention, and overwritten on the next round.
    num_consumed = 1 + num_accepted  # token_main + accepted drafts
    current_position += num_consumed
    last_hidden = hidden
    last_token_index = num_consumed - 1  # last accepted position in hidden output

    # === Phase 5: Roll back MTP position for rejected drafts ===
    num_drafted = len(draft_tokens)
    num_rejected = num_drafted - min(num_accepted, num_drafted)
    mtp_pos -= num_rejected  # roll back so next round overwrites rejected entries
```

### 4.6 Why Batch Verification Works with Our seq_len=8

Our text model is traced with `seq_len=8`. During decode:
- Standard (no MTP): processes 1 real token + 7 padding tokens
- With MTP (3 steps): processes 4 real tokens (token_main + 3 drafts) + 4 padding tokens

This fits perfectly within seq_len=8. We could even do `--speculative-num-steps 7` for maximum utilization.

---

## 5. MTP KV Cache Behavior

### 5.1 The MTP Has Its Own Separate KV Cache

The MTP transformer block (layer 16's self_attn) maintains its **own independent KV cache**, completely separate from the main model's 16-layer KV cache.

- **Main model KV cache**: shape `(16, num_kv_heads, cache_length, head_dim)` = `(16, 8, 2048, 128)`
- **MTP KV cache**: shape `(1, num_kv_heads, cache_length, head_dim)` = `(1, 8, 2048, 128)`

### 5.2 MTP KV Cache Prefilling — ~2x Speedup

The MTP's KV cache starts empty when decoding begins. While it *can* work without prefilling (the `previous_hidden_states` from the main model provides context), acceptance rates are poor because the MTP's self-attention has no history to attend to during early decode rounds.

**Solution:** After the main model's prefill, run the MTP position-by-position over the prefill sequence. At each position `i`, feed `(main_hidden[i], embed(token[i+1]))` to the MTP. This fills its KV cache with self-attention context that mirrors the actual token sequence.

```python
# After main model prefill completes and MTP state is created:
chunk_start = 0
for hidden_cf, real_len in prefill_hidden_chunks:
    for j in range(real_len):
        abs_pos = chunk_start + j
        if abs_pos + 1 >= total_prefill_len:
            break
        prev_hidden = hidden_cf[:, :, :, j : j + 1]
        next_embed_cf = merged_embeds_seq[abs_pos + 1].reshape(1, hidden_dim, 1, 1)
        _forward_mtp(prev_hidden, next_embed_cf, mtp_position, mtp_state)
        mtp_position += 1
    chunk_start += real_len
```

**Result:** Acceptance rate increased significantly, yielding **~2x overall speedup** compared to no prefilling. The prefill overhead (~1ms per position) is more than offset by fewer main model forward passes during decode.

**Why it works:** The MTP's transformer block builds up internal K/V representations through self-attention. With a warm KV cache, the MTP can attend to the full sequence history when drafting, instead of relying solely on the main model's hidden state passed as input. This is architecturally consistent — position-0 masking is preserved during prefill.

### 5.3 MTP KV Cache Is Incremental (NOT Reset Between Rounds)

**The MTP KV cache is incremental** — it persists across speculation rounds and grows with each MTP forward pass, just like the main model's cache. It is **not reset** after each verification step.

However, **rejected draft entries need position rollback**. Here's what happens:

#### During a speculation round (3 MTP steps):

```
MTP KV Cache before round:
Position: [0] [1] [2] [3] [4] ...
Data:      K0  K1  K2  K3  --  (empty)

MTP drafts 3 tokens at positions 4, 5, 6:
Position: [0] [1] [2] [3] [4] [5] [6] ...
Data:      K0  K1  K2  K3  K4  K5  K6

Verification: only draft at position 4 accepted, positions 5 and 6 rejected.
```

#### After rejection — position rollback:

```
mtp_pos was 7 (after 3 forward passes from pos 4)
Roll back: mtp_pos = 7 - 2 (rejected) = 5

Next round starts MTP at position 5, which OVERWRITES the stale K5, K6:
Position: [0] [1] [2] [3] [4] [5'] [6'] ...
Data:      K0  K1  K2  K3  K4  K5'  K6'  (new values)
```

The overwrite is natural because our KV cache uses positional slicing: `cache[:, :, position:position+1, :]`. Writing at position 5 overwrites whatever was there.

#### How vLLM handles this:

vLLM uses a more complex slot-mapping system (since it manages batched requests with paged attention), but the effect is the same:
- `seq_lens` is reduced by `num_rejected_tokens` (eagle.py line 568-569)
- Rejected slot positions get `PADDING_SLOT_ID` preventing attention access (utils.py line 147-148)
- Next round's attention metadata reflects the rolled-back sequence length
- New KV values naturally overwrite stale ones at those positions

For our CoreML implementation with simple positional slicing, we just need to track `mtp_pos` correctly and roll it back by the number of rejected tokens.

### 5.4 MTP Position Tracking

The MTP's position counter starts at 0 when decoding begins (not at the end of prefill):
- MTP has never seen any tokens during prefill → its KV cache is empty
- Position 0 in the MTP corresponds to the first token in the first speculation round
- After each speculation round: `mtp_pos += num_drafted; mtp_pos -= num_rejected`

### 5.5 Position-0 Masking

The MTP forward pass masks `input_embeds` at position 0:
```python
inputs_embeds[positions == 0] = 0
```
This is because at position 0 (the very first MTP call), the "previous token embedding" doesn't carry meaningful information. The `previous_hidden_states` from the main model provides all the context needed.

---

## 6. CoreML Model Architecture Decisions

### 6.1 Should MTP Be a Separate .mlpackage?

**Yes, keep it separate.** Reasons:

1. **Different KV cache state**: The MTP has its own 1-layer KV cache (`(1, 8, 2048, 128)`) while the main model has 16 layers. CoreML state tensors are defined at conversion time and cannot be shared across models.

2. **Different call patterns**: The main model is called during both prefill (chunked) and decode. The MTP is only called during decode, potentially multiple times per round.

3. **Different inputs**: Main model takes `(inputs_embeds, position_id)`. MTP takes `(previous_hidden_states, input_embeds, position_id)`.

4. **Modularity**: Keeping them separate allows using the main model without MTP (fallback to standard decode) and makes debugging easier.

### 6.2 Model Split

```
embeddings.npy          ← Token embedding table (CPU lookup)
text_model.mlpackage    ← 16-layer transformer (Neural Engine, stateful KV cache)
lm_head.mlpackage       ← Chunked vocabulary projection (Neural Engine)
mtp_model.mlpackage     ← 1-layer MTP transformer (Neural Engine, own stateful KV cache)  [NEW]
```

The MTP model reuses `embeddings.npy` and `lm_head.mlpackage` — no duplication needed.

### 6.3 Can the LM Head Be Fused Into the MTP Model?

**Not recommended for the initial implementation.** Reasons:

- The same LM head is already used for the main model. Duplicating weights increases model size.
- The chunked LM head approach works well and avoids the 16384 dimension limit.
- The LM head call overhead is minimal compared to the speedup from multi-step drafting.

**Future optimization**: If profiling shows significant CoreML `predict()` call overhead, consider fusing `shared_head_norm + lm_head + argmax` into the MTP model. This would output a single token ID instead of hidden states, eliminating one CoreML round-trip per MTP step. This would break chaining though — you'd need the raw hidden states for chaining AND the token for the next embedding lookup. A compromise: output both `mtp_hidden` and `draft_token_id`.

### 6.4 MTP Model Inputs and Outputs

**Inputs:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `previous_hidden_states` | `(1, 1536, 1, 1)` | float32 | Hidden states from main model or previous MTP step (channels-first) |
| `input_embeds` | `(1, 1536, 1, 1)` | float32 | Embedding of the token from main model or previous MTP draft (channels-first) |
| `position_id` | `(1,)` | int32 | Current position in MTP's KV cache |

**Outputs:**
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `mtp_hidden` | `(1, 1536, 1, 1)` | float16 | Raw hidden states — for chaining to next MTP step |
| `mtp_hidden_for_head` | `(1, 1536, 1, 1)` | float16 | Normalized hidden states — for LM head |

**State (CoreML stateful):**
| Name | Shape | Description |
|------|-------|-------------|
| `key_cache` | `(1, 8, 2048, 128)` | MTP layer's key cache |
| `value_cache` | `(1, 8, 2048, 128)` | MTP layer's value cache |

### 6.5 LM Head Shape Mismatch with MTP Output

The LM head model is compiled with a fixed input shape `(1, hidden_dim, 1, text_seq_len)` matching the text model's sequence length (e.g., 8). The MTP model outputs `(1, hidden_dim, 1, 1)` — a single token. Passing this directly to the LM head causes a CoreML runtime shape error.

**Fix:** Pad the MTP hidden output with `np.pad(..., mode="edge")` to `(1, hidden_dim, 1, text_seq_len)` before calling the LM head. Only index 0 of the result is used.

### 6.6 Vision Conv3d to Conv2d Temporal Folding

The vision encoder's patch embedding is originally a `Conv3d` with `temporal_patch_size=2`. Since images have a single frame (temporal=1 after the processor pads to 2), fold the temporal dimension into channels:

```
Conv3d(3, hidden, kernel=(2,14,14)) → Conv2d(3*2=6, hidden, kernel=(14,14))
```

Input becomes `(num_patches, 6, 14, 14)` — standard Conv2d that ANE handles natively.

---

## 7. Implementation Plan

### 7.1 MTP CoreML Model Conversion

Create the MTP conversion module and script. The conversion needs to:

1. **Build the PyTorch MTP module** from HuggingFace's `GlmOcrTextDecoderLayer` + MTP-specific layers (enorm, hnorm, eh_proj, shared_head_norm)
2. **Load weights from safetensors** using the `model.language_model.layers.16.` prefix
3. **Patch for channels-first CoreML**:
   - `patch_model_linears()` — Linear → 1x1 Conv2d
   - `patch_model_rmsnorms()` — RMSNorm → LayerNorm-fusable ops
   - Patch attention with `GlmOcrTextAttentionPatcher` (interleaved→split-half RoPE weight permutation)
   - Patch MLP with `GlmOcrTextMLPPatcher` (fused gate_up_proj → channels-first chunk)
   - Wrap decoder layer with `GlmOcrTextDecoderLayerPatcher` (4-norm sandwich pattern)
4. **Wrap with `GlmOcrMTPWrapper`** providing stateful KV cache, RoPE precomputation, causal mask, and position-0 masking
5. **Trace and convert** to CoreML with state specs
6. **Verify** against original unpatched PyTorch module

### 7.2 Inference Pipeline Updates

Update `GlmOcrCoreMLPipeline` in `src/coremlmodels/glm_ocr_coreml_pipeline.py`:

#### 7.2.1 Constructor Changes

Add optional `mtp_model_path` parameter and `num_spec_steps` (default 3):

```python
class GlmOcrCoreMLPipeline:
    def __init__(
        self,
        vision_model_path,
        text_model_path,
        lm_head_path,
        embeddings_path,
        mtp_model_path=None,        # NEW
        num_spec_steps=3,           # NEW: how many MTP forward passes per round
        hf_model_name="zai-org/GLM-OCR",
        cache_compiled=False,
        max_vision_patches=None,
    ):
        # ... existing init ...
        self.mtp_model = None
        self.num_spec_steps = num_spec_steps
        if mtp_model_path is not None:
            self.mtp_model = self._load_model(mtp_model_path, cache_compiled)
```

#### 7.2.2 MTP Helper Methods

```python
def _forward_mtp(
    self,
    previous_hidden_states_cf: np.ndarray,
    input_embeds_cf: np.ndarray,
    position_id: int,
    mtp_state,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MTP forward pass, return (mtp_hidden, mtp_hidden_for_head)."""
    result = self.mtp_model.predict(
        {
            "previous_hidden_states": previous_hidden_states_cf,
            "input_embeds": input_embeds_cf,
            "position_id": np.array([position_id], dtype=np.int32),
        },
        mtp_state,
    )
    return result["mtp_hidden"], result["mtp_hidden_for_head"]

def _embed_token_cf(self, token_id: int) -> np.ndarray:
    """Look up embedding and reshape to channels-first (1, hidden_dim, 1, 1)."""
    embed = self.embeddings[np.array([token_id])]  # (1, hidden_dim)
    return embed.T[np.newaxis, :, np.newaxis, :].astype(np.float32)

def _draft_tokens_mtp(
    self,
    last_hidden_cf: np.ndarray,
    token_index: int,
    first_token_id: int,
    mtp_state,
    mtp_position: int,
    num_steps: int,
) -> tuple[list[int], int]:
    """Draft multiple tokens using MTP chaining.

    Returns:
        Tuple of (draft_token_ids, new_mtp_position).
    """
    draft_tokens = []
    prev_hidden = last_hidden_cf[:, :, :, token_index:token_index+1]
    prev_embed = self._embed_token_cf(first_token_id)

    for step in range(num_steps):
        mtp_hidden, mtp_for_head = self._forward_mtp(
            prev_hidden, prev_embed, mtp_position, mtp_state,
        )
        mtp_position += 1

        # Greedy decode for draft
        logits = self._compute_logits(mtp_for_head, 0, temperature=1.0)
        draft_token = int(np.argmax(logits))
        draft_tokens.append(draft_token)

        if draft_token in self.eos_token_ids:
            break

        # Chain: MTP output becomes next input
        prev_hidden = mtp_hidden
        prev_embed = self._embed_token_cf(draft_token)

    return draft_tokens, mtp_position
```

#### 7.2.3 Updated `run()` Method with Multi-Step MTP

```python
def run(self, image, prompt, generation_config=None, ...):
    cfg = generation_config or GenerationConfig()

    # ... existing prefill code (unchanged) ...

    # Initialize MTP if available
    mtp_state = None
    mtp_position = 0
    if self.mtp_model is not None:
        mtp_state = self.mtp_model.make_state()

    generated_ids = []
    while len(generated_ids) < cfg.max_new_tokens:
        # === Step 1: Sample token from main model ===
        logits = self._compute_logits(last_hidden, last_token_index, ...)
        token_main = self._sample_token(logits, ...)
        if token_main in self.eos_token_ids:
            break
        generated_ids.append(token_main)

        if mtp_state is None:
            # === No MTP: standard single-token decode ===
            token_embed = self.embeddings[np.array([token_main])]
            token_chunk, real_len = self._pad_embed_chunk(token_embed, self.text_seq_len)
            last_hidden = self._forward_text_chunk(token_chunk, current_position, state)
            current_position += real_len
            last_token_index = 0
            continue

        # === Step 2: Draft N tokens with MTP chaining ===
        draft_tokens, mtp_position = self._draft_tokens_mtp(
            last_hidden, last_token_index, token_main,
            mtp_state, mtp_position, self.num_spec_steps,
        )

        # === Step 3: Verify all drafts with main model (batched) ===
        # Build embedding sequence: [token_main, draft_0, draft_1, ...]
        all_token_ids = [token_main] + draft_tokens
        all_embeds = np.stack([self.embeddings[np.array([tid])][0] for tid in all_token_ids])
        # all_embeds shape: (num_tokens, hidden_dim)

        embed_chunk, real_len = self._pad_embed_chunk(all_embeds, self.text_seq_len)
        hidden = self._forward_text_chunk(embed_chunk, current_position, state)

        # Verify each draft token by comparing main model's logits.
        # IMPORTANT: Do NOT append correction or bonus tokens during verification.
        # After a batched forward pass, hidden[i] is only trustworthy if ALL tokens
        # at positions 0..i were correct. Once a draft is rejected at position i,
        # hidden[i+1..] are contaminated (they attended to wrong KV entries).
        # See "Verification Loop Bug" section below for details.
        num_matching_drafts = 0
        for i, draft in enumerate(draft_tokens):
            verify_logits = self._compute_logits(hidden, i, ...)
            verify_token = self._sample_token(verify_logits, ...)

            if verify_token == draft and draft not in self.eos_token_ids:
                generated_ids.append(draft)
                num_matching_drafts += 1
            else:
                if verify_token in self.eos_token_ids:
                    hit_eos = True
                # Don't append correction — it emerges naturally next iteration
                break

        # No bonus token handling needed — the next iteration samples from
        # hidden[num_matching_drafts], which naturally produces the next token
        # (correction or bonus) from a clean hidden state.

        # === Step 4: Update positions ===
        num_consumed = 1 + num_matching_drafts  # token_main + matching drafts
        current_position += num_consumed
        last_token_index = num_matching_drafts  # last CLEAN hidden state position
        last_hidden = hidden

        # Roll back MTP position for rejected drafts
        num_rejected = len(draft_tokens) - num_matching_drafts
        mtp_position -= num_rejected
```

**Main model KV cache after rejection**: When drafts are rejected, the main model's KV cache contains stale entries from the rejected draft tokens at positions beyond `current_position`. This is **not a problem** because:
1. The causal attention mask prevents any future token from attending to positions beyond `current_position` (they are effectively invisible).
2. The next round's forward pass writes new K,V at those same positions, naturally overwriting the stale entries.
3. No rollback or extra forward passes are needed — the position management handles everything.

### 7.2.4 Verification Loop Bug (Lessons Learned)

The initial implementation had three interrelated bugs that caused **wrong output** whenever the MTP was enabled:

**Bug 1 — Correction token contaminates KV cache:** When a draft was rejected at position `i`, the code appended the correction token and set `last_token_index = i+1`. But `hidden[i+1]` was computed by attending to the **rejected draft's** KV entry — not the correction token's. All subsequent tokens were generated from contaminated hidden states.

**Bug 2 — Bonus token duplicated:** When all drafts matched, the bonus token was appended to `generated_ids`, but `last_token_index` still pointed to the position that predicted the bonus. The next iteration re-sampled from that same position, producing a duplicate.

**Bug 3 — MTP position rollback off by one:** The rollback used `num_accepted_drafts` (which included the correction), rolling back one position too few in the MTP's KV cache.

**The fix:** Don't append correction or bonus tokens during verification. Set `last_token_index = num_matching_drafts`, which points to the last hidden state position that only attended to verified-correct KV entries. The correction/bonus emerges naturally when the next iteration samples from that clean position.

**Root cause insight:** In speculative decoding, after a batched verification forward pass, `hidden[i]` is only trustworthy if ALL tokens at positions `0..i` in the batch were correct. Once a draft is rejected at position `i`, `hidden[i+1..]` are contaminated because they attended to wrong KV entries. You can only use `hidden[0..i]`.

### 7.3 CLI Argument Updates

Update `examples/glm_ocr_coreml_inference.py`:

```python
parser.add_argument(
    "--mtp-model",
    type=str,
    default=None,
    help="Path to MTP .mlpackage for speculative decoding (optional).",
)
parser.add_argument(
    "--num-spec-steps",
    type=int,
    default=3,
    help="Number of MTP forward passes per speculation round (default: 3).",
)
```

And pass to pipeline:
```python
pipeline = GlmOcrCoreMLPipeline(
    ...,
    mtp_model_path=Path(args.mtp_model) if args.mtp_model else None,
    num_spec_steps=args.num_spec_steps,
)
```

### 7.4 Conversion Script

Create `examples/glm_ocr_mtp_conversion.py`:

```python
"""Convert GLM-OCR MTP module to CoreML.

Usage:
    uv run python examples/glm_ocr_mtp_conversion.py
    uv run python examples/glm_ocr_mtp_conversion.py --skip-model-load
"""
import argparse
# ... import conversion function ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="zai-org/GLM-OCR")
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert_glm_ocr_mtp(
        model_name=args.model,
        seq_len=1,           # MTP processes 1 token at a time
        cache_length=args.cache_length,
        batch_size=1,
        output_path=args.output,
        skip_model_load=args.skip_model_load,
        overwrite=args.overwrite,
    )
```

---

## 8. File Reference

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| MTP conversion module (new) | **Create** | PyTorch module, weight loading, patching, wrapping, CoreML conversion |
| `examples/glm_ocr_mtp_conversion.py` | **Create** | CLI script to convert MTP model |
| `src/coremlmodels/glm_ocr_coreml_pipeline.py` | **Modify** | Add multi-step MTP speculative decoding to inference loop |
| `examples/glm_ocr_coreml_inference.py` | **Modify** | Add `--mtp-model` and `--num-spec-steps` CLI arguments |
| `src/coremlmodels/__init__.py` | **Modify** | Export MTP conversion functions |

### Existing Files (Context)

| File | Purpose |
|------|---------|
| `src/coremlmodels/glm_ocr_text_model.py` | Patchers (attention, MLP, decoder layer) reused by MTP conversion |
| `src/coremlmodels/glm_ocr_coreml_pipeline.py` | Current inference pipeline (decode loop to be extended) |
| `examples/glm_ocr_coreml_inference.py` | CLI for inference |
| `examples/glm_ocr_text_conversion.py` | Text model conversion (reference pattern) |

### vLLM Reference Files

| File | What to Reference |
|------|-------------------|
| `third_party/vllm/vllm/model_executor/models/glm_ocr_mtp.py` | MTP architecture for GLM-OCR |
| `third_party/vllm/vllm/model_executor/models/glm4_moe_lite_mtp.py` | MTP base classes (SharedHead, forward/chaining logic) |
| `third_party/vllm/vllm/model_executor/models/glm4.py` | Decoder layer structure |
| `third_party/vllm/vllm/v1/spec_decode/eagle.py` | Multi-step propose loop, KV cache slot management |
| `third_party/vllm/vllm/v1/worker/gpu/spec_decode/eagle.py` | Worker-level orchestration, position rollback |
| `third_party/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sample.py` | Rejection sampling kernel |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| `num_hidden_layers` | 16 (main model) |
| `num_nextn_predict_layers` | 1 (single MTP layer, chained N times) |
| MTP layer index in safetensors | 16 (stored as `layers.16`) |
| `hidden_size` | 1536 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `intermediate_size` | 4608 |
| `vocab_size` | 59392 |
| `rope_theta` | 10000 |
| `rms_norm_eps` | 1e-5 |
| Main model KV cache | `(16, 8, 2048, 128)` |
| MTP KV cache | `(1, 8, 2048, 128)` |
| Recommended `num_spec_steps` | 3 (from GLM-OCR repo) |
| Max `num_spec_steps` with seq_len=8 | 7 (8 - 1 for token_main) |
