"""GLM-OCR Multi-Token Predictor (MTP) conversion helpers.

This module builds the single MTP layer (stored in safetensors as
``model.language_model.layers.<num_hidden_layers>.*``), patches it for
channels-first CoreML execution, and exports a stateful CoreML model.

The MTP module contains:
- enorm / hnorm: RMSNorms for input embedding / previous hidden states
- eh_proj: Linear projection from concat(enorm_out, hnorm_out) to hidden_size
- mtp_block: Standard GlmOcrTextDecoderLayer (4-norm sandwich pattern)
- shared_head_norm: RMSNorm before the shared LM head

The resulting CoreML model processes one token at a time (seq_len=1) and
maintains its own 1-layer KV cache, separate from the main text model's
16-layer KV cache. During multi-step speculative decoding, the MTP model
is called repeatedly, chaining its own hidden state output back as input.
"""

from __future__ import annotations

from pathlib import Path

import coremltools as ct
import numpy as np
from safetensors import safe_open
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrRMSNorm,
    GlmOcrTextDecoderLayer,
)
from transformers.utils.hub import cached_file

from coremlmodels.glm_ocr.glm_ocr_text_model import (
    GlmOcrTextAttentionPatcher,
    GlmOcrTextDecoderLayerPatcher,
    GlmOcrTextMLPPatcher,
)
from coremlmodels.graph_passes import register_extended_passes
from coremlmodels.lm_model_wrapper import (
    _generate_causal_mask,
    _index_position_embeddings,
)
from coremlmodels.patch_linears import patch_model_linears
from coremlmodels.patch_rmsnorm import patch_model_rmsnorms
from coremlmodels.registry import find_target_classes, get_architecture_config


def _cached_file(path_or_repo_id: str, filename: str) -> str:
    """Resolve model file from local cache first, then allow remote fallback."""
    try:
        return cached_file(path_or_repo_id, filename, local_files_only=True)
    except Exception:
        return cached_file(path_or_repo_id, filename)


# ============================================================================
# MTP PyTorch Module
# ============================================================================


class GlmOcrMTPModule(nn.Module):
    """Standalone GLM-OCR MTP module before CoreML wrapping.

    Contains the MTP-specific components (enorm, hnorm, eh_proj,
    shared_head_norm) and one standard GlmOcrTextDecoderLayer.

    Attributes:
        layer_idx: The layer index in safetensors (num_hidden_layers, i.e. 16).
        enorm: RMSNorm applied to input embeddings.
        hnorm: RMSNorm applied to previous hidden states.
        eh_proj: Linear projection from 2*hidden_size to hidden_size.
        mtp_block: Standard GLM-OCR decoder layer (4-norm sandwich).
        shared_head_norm: RMSNorm applied after the transformer block,
            before the shared LM head.
    """

    def __init__(self, text_config: object):
        super().__init__()
        hidden_size = int(text_config.hidden_size)
        eps = float(text_config.rms_norm_eps)
        layer_idx = int(text_config.num_hidden_layers)

        self.layer_idx = layer_idx
        self.enorm = GlmOcrRMSNorm(hidden_size, eps=eps)
        self.hnorm = GlmOcrRMSNorm(hidden_size, eps=eps)
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.mtp_block = GlmOcrTextDecoderLayer(text_config, layer_idx=layer_idx)
        self.shared_head_norm = GlmOcrRMSNorm(hidden_size, eps=eps)


# ============================================================================
# Weight Loading
# ============================================================================


def load_glm_ocr_mtp_module(
    model_name: str,
    verbose: bool = True,
) -> tuple[GlmOcrMTPModule, object]:
    """Load MTP weights from GLM-OCR safetensors into a standalone module.

    The MTP weights are stored as layer ``num_hidden_layers`` in the same
    safetensors file as the main model. This function builds the MTP module
    structure and loads only the MTP-unique weights (skipping shared_head.head
    and embed_tokens which are already exported separately).

    Args:
        model_name: HuggingFace model name or local path.
        verbose: Print loading progress.

    Returns:
        Tuple of (mtp_module, text_config).

    Raises:
        ValueError: If the model does not have MTP layers.
        KeyError: If a required MTP weight is missing from safetensors.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_config = config.text_config
    if int(getattr(text_config, "num_nextn_predict_layers", 0)) < 1:
        raise ValueError(
            "Model does not expose MTP layers (num_nextn_predict_layers < 1)."
        )

    module = GlmOcrMTPModule(text_config)
    mtp_layer_idx = int(text_config.num_hidden_layers)
    prefix = f"model.language_model.layers.{mtp_layer_idx}."
    safetensors_path = _cached_file(model_name, "model.safetensors")

    if verbose:
        print(f"Loading MTP weights from: {safetensors_path}")
        print(f"  MTP layer index: {mtp_layer_idx}")

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for name, param in module.named_parameters():
            # Map module parameter names to safetensors key names
            if name.startswith("mtp_block."):
                # Transformer block weights: strip mtp_block. prefix
                weight_name = prefix + name[len("mtp_block."):]
            elif name == "shared_head_norm.weight":
                # shared_head_norm.weight -> shared_head.norm.weight
                weight_name = prefix + "shared_head.norm.weight"
            else:
                # enorm.weight, hnorm.weight, eh_proj.weight
                weight_name = prefix + name

            if weight_name not in f.keys():
                raise KeyError(f"Missing MTP weight in safetensors: {weight_name}")

            loaded = f.get_tensor(weight_name).to(torch.float32)
            if tuple(loaded.shape) != tuple(param.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected {tuple(param.shape)}, got {tuple(loaded.shape)}"
                )
            param.data.copy_(loaded)

    module = module.float().eval()

    if verbose:
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  Loaded {num_params:,} parameters")

    return module, text_config


# ============================================================================
# Patching
# ============================================================================


def patch_glm_ocr_mtp_module(
    mtp_module: GlmOcrMTPModule,
    text_config: object,
    verbose: bool = True,
) -> GlmOcrMTPModule:
    """Patch MTP module for channels-first CoreML execution.

    Applies the same patching pipeline as the main text model:
    1. Linear -> 1x1 Conv2d
    2. RMSNorm -> LayerNorm-fusable ops
    3. Attention -> GlmOcrTextAttentionPatcher (with interleaved->split-half RoPE)
    4. MLP -> GlmOcrTextMLPPatcher (channels-first chunk)
    5. Decoder layer -> GlmOcrTextDecoderLayerPatcher (4-norm sandwich)

    Note: The attention patcher uses layer_idx=0 because the MTP has only one
    layer with its own separate KV cache.

    Args:
        mtp_module: The unpatched MTP module with loaded weights.
        text_config: Text model configuration.
        verbose: Print patching progress.

    Returns:
        The patched MTP module (mutated in-place).
    """
    arch_config = get_architecture_config("glm_ocr")
    rmsnorm_classes = find_target_classes(mtp_module, arch_config.rmsnorm_class_names)

    if verbose:
        print("Patching MTP Linear layers...")
    with torch.inference_mode():
        patch_model_linears(mtp_module, verbose=verbose)

    if verbose:
        print("Patching MTP RMSNorm layers...")
    with torch.inference_mode():
        patch_model_rmsnorms(
            mtp_module,
            target_classes=rmsnorm_classes,
            verbose=verbose,
        )

    num_heads = int(text_config.num_attention_heads)
    num_kv_heads = int(text_config.num_key_value_heads)
    head_dim = int(
        getattr(text_config, "head_dim", None)
        or text_config.hidden_size // text_config.num_attention_heads
    )

    if verbose:
        print("Patching MTP decoder layer...")
    with torch.inference_mode():
        attn_patcher = GlmOcrTextAttentionPatcher(
            attention_layer=mtp_module.mtp_block.self_attn,
            layer_idx=0,  # Only 1 layer in MTP -> index 0 in its own KV cache
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        mlp_patcher = GlmOcrTextMLPPatcher(mtp_module.mtp_block.mlp)
        mtp_module.mtp_block = GlmOcrTextDecoderLayerPatcher(
            decoder_layer=mtp_module.mtp_block,
            attn_patcher=attn_patcher,
            mlp_patcher=mlp_patcher,
        )

    return mtp_module


# ============================================================================
# Stateful Wrapper
# ============================================================================


class GlmOcrMTPWrapper(nn.Module):
    """Stateful wrapper for GLM-OCR MTP CoreML conversion.

    Provides KV cache as registered buffers (become CoreML state),
    pre-computed RoPE embeddings, and causal mask generation.
    Position-0 masking is handled by the caller (pass zeros for
    input_embeds at position 0).

    Inputs:
        previous_hidden_states: (1, hidden_dim, 1, 1) - channels-first
        input_embeds: (1, hidden_dim, 1, 1) - channels-first
        position_id: (1,) - int32 scalar

    Outputs:
        mtp_hidden: (1, hidden_dim, 1, 1) - raw hidden for chaining
        mtp_hidden_for_head: (1, hidden_dim, 1, 1) - normalized for LM head

    State:
        key_cache: (1, num_kv_heads, cache_length, head_dim)
        value_cache: (1, num_kv_heads, cache_length, head_dim)
    """

    def __init__(
        self,
        mtp_module: GlmOcrMTPModule,
        config: object,
        cache_length: int = 2048,
        channels_first: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        self.layer = mtp_module
        self.cache_length = cache_length
        self.channels_first = channels_first

        num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        # MTP has only 1 layer -> KV cache shape: (1, num_kv_heads, cache_length, head_dim)
        self.register_buffer(
            "key_cache",
            torch.zeros(1, num_kv_heads, cache_length, head_dim, device=device),
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(1, num_kv_heads, cache_length, head_dim, device=device),
        )

        # Pre-compute standard 1D RoPE embeddings (same method as GlmOcrLanguageModelWrapper)
        rope_theta = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        )
        positions = torch.arange(cache_length, dtype=torch.float, device=device)
        freqs = torch.outer(positions, inv_freq)  # (cache_length, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (cache_length, dim)

        if dim < head_dim:
            padding = torch.zeros(cache_length, head_dim - dim, device=device)
            cos_emb = torch.cat([emb.cos(), padding + 1.0], dim=-1)
            sin_emb = torch.cat([emb.sin(), padding], dim=-1)
        else:
            cos_emb = emb.cos()
            sin_emb = emb.sin()

        self.register_buffer("cos_emb", cos_emb)  # (cache_length, head_dim)
        self.register_buffer("sin_emb", sin_emb)  # (cache_length, head_dim)

    def forward(
        self,
        previous_hidden_states: torch.Tensor,
        input_embeds: torch.Tensor,
        position_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MTP forward pass.

        Args:
            previous_hidden_states: Hidden states from main model or previous
                MTP step. Shape: (1, hidden_dim, 1, 1) in channels-first.
            input_embeds: Embedding of the token predicted by main model or
                previous MTP draft. Shape: (1, hidden_dim, 1, 1) in channels-first.
                Caller should pass zeros at position 0 (pos-0 masking).
            position_id: Current position in MTP's KV cache. Shape: (1,).

        Returns:
            Tuple of (mtp_hidden, mtp_hidden_for_head):
            - mtp_hidden: Raw hidden states for chaining to next MTP step.
            - mtp_hidden_for_head: Normalized hidden states for LM head.
        """
        # Index RoPE embeddings
        position_ids = position_id.long().view(1, 1)
        position_emb = _index_position_embeddings(
            self.cos_emb, self.sin_emb, position_ids
        )
        # Channels-first: create (cos, sin, cos_t, sin_t) tuple
        position_emb = (
            position_emb[0],
            position_emb[1],
            position_emb[0].transpose(-1, -2),
            position_emb[1].transpose(-1, -2),
        )

        # Generate causal mask: (1, 1, 1, cache_length) -> transpose -> (1, 1, cache_length, 1)
        attention_mask = _generate_causal_mask(
            self.cache_length, position_ids, previous_hidden_states.device
        ).transpose(-1, -2)

        # Normalize both inputs independently
        input_embeds = self.layer.enorm(input_embeds)
        previous_hidden_states = self.layer.hnorm(previous_hidden_states)

        # Concatenate along channel dim and project: (B, 3072, 1, 1) -> (B, 1536, 1, 1)
        hidden_states = torch.cat([input_embeds, previous_hidden_states], dim=1)
        hidden_states = self.layer.eh_proj(hidden_states)

        # Run through transformer decoder layer (with its own KV cache)
        hidden_states = self.layer.mtp_block(
            hidden_states,
            attention_mask,
            None,  # position_ids not needed (in position_embeddings)
            (self.key_cache, self.value_cache),
            cache_position=position_id,
            position_embeddings=position_emb,
        )

        # Apply shared head norm for LM head input
        hidden_for_head = self.layer.shared_head_norm(hidden_states)

        return hidden_states, hidden_for_head

    def __repr__(self) -> str:
        return (
            "GlmOcrMTPWrapper("
            f"cache_length={self.cache_length}, "
            f"channels_first={self.channels_first}, "
            f"key_cache_shape={tuple(self.key_cache.shape)}, "
            f"value_cache_shape={tuple(self.value_cache.shape)})"
        )


# ============================================================================
# CoreML State Specs
# ============================================================================


def create_glm_ocr_mtp_state_specs(wrapper: GlmOcrMTPWrapper) -> list:
    """Create CoreML StateType specifications for MTP wrapper.

    Args:
        wrapper: The GlmOcrMTPWrapper instance.

    Returns:
        List of coremltools.StateType objects for key_cache and value_cache.
    """
    return [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=wrapper.key_cache.shape),
            name="key_cache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=wrapper.value_cache.shape),
            name="value_cache",
        ),
    ]


# ============================================================================
# Reference Output Computation (for verification)
# ============================================================================


def _compute_mtp_reference_output(
    mtp_module: GlmOcrMTPModule,
    text_config: object,
    batch_size: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference output from the unpatched MTP module.

    Runs the original MTP module in sequence-last format to get ground truth
    for verification against CoreML output.

    Args:
        mtp_module: The unpatched MTP module with loaded weights.
        text_config: Text model configuration.
        batch_size: Batch size.
        verbose: Print progress.

    Returns:
        Tuple of (ref_hidden_cf, ref_for_head_cf, test_prev_cf, test_embed_cf)
        where _cf means channels-first numpy arrays.
    """
    if verbose:
        print("    Computing reference output before patching...")

    hidden_dim = int(text_config.hidden_size)

    # Generate random test inputs in channels-first format
    test_prev_cf = np.random.randn(batch_size, hidden_dim, 1, 1).astype(np.float32)
    test_embed_cf = np.random.randn(batch_size, hidden_dim, 1, 1).astype(np.float32)

    # Convert to sequence-last format for the original model: (B, 1, hidden_dim)
    test_prev_sf = torch.from_numpy(test_prev_cf).squeeze(2).transpose(1, 2)
    test_embed_sf = torch.from_numpy(test_embed_cf).squeeze(2).transpose(1, 2)

    with torch.inference_mode():
        # Position-0 masking: zero input_embeds at position 0
        # (matches wrapper behavior)
        test_embed_masked = torch.zeros_like(test_embed_sf)

        # Apply enorm and hnorm
        enorm_out = mtp_module.enorm(test_embed_masked)
        hnorm_out = mtp_module.hnorm(test_prev_sf)

        # Concat and project: (B, 1, 3072) -> (B, 1, 1536)
        combined = torch.cat([enorm_out, hnorm_out], dim=-1)
        hidden = mtp_module.eh_proj(combined)

        # Compute rotary position embeddings for the decoder layer
        # The HF decoder layer expects position_embeddings=(cos, sin)
        position_ids = torch.zeros(1, 1, dtype=torch.long)
        rope_theta = text_config.rope_parameters["rope_theta"]
        num_heads = int(text_config.num_attention_heads)
        head_dim = int(
            getattr(text_config, "head_dim", None)
            or text_config.hidden_size // num_heads
        )
        partial_rotary_factor = text_config.rope_parameters.get(
            "partial_rotary_factor", 1.0
        )
        rot_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, rot_dim, 2, dtype=torch.float) / rot_dim)
        )
        freqs = torch.outer(position_ids.float().reshape(-1), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        if rot_dim < head_dim:
            pad = torch.zeros(1, head_dim - rot_dim)
            cos_emb = torch.cat([emb.cos(), pad + 1.0], dim=-1).unsqueeze(0)
            sin_emb = torch.cat([emb.sin(), pad], dim=-1).unsqueeze(0)
        else:
            cos_emb = emb.cos().unsqueeze(0)
            sin_emb = emb.sin().unsqueeze(0)
        position_embeddings = (cos_emb, sin_emb)

        # Run through the decoder layer
        layer_output = mtp_module.mtp_block(
            hidden,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
        # GlmOcrTextDecoderLayer returns a tuple; first element is hidden_states
        if isinstance(layer_output, tuple):
            hidden_out = layer_output[0]
        else:
            hidden_out = layer_output

        # Apply shared_head_norm
        hidden_for_head = mtp_module.shared_head_norm(hidden_out)

        # Convert to channels-first: (B, 1, H) -> (B, H, 1, 1)
        ref_hidden_cf = hidden_out.transpose(1, 2).unsqueeze(2).numpy()
        ref_for_head_cf = hidden_for_head.transpose(1, 2).unsqueeze(2).numpy()

    if verbose:
        print(f"    Reference mtp_hidden shape: {ref_hidden_cf.shape}")
        print(f"    Reference mtp_hidden_for_head shape: {ref_for_head_cf.shape}")

    return ref_hidden_cf, ref_for_head_cf, test_prev_cf, test_embed_cf


# ============================================================================
# Full Conversion Function
# ============================================================================


def convert_glm_ocr_mtp(
    model_name: str,
    seq_len: int = 1,
    cache_length: int = 2048,
    batch_size: int = 1,
    output_path: str | Path | None = None,
    verbose: bool = True,
    skip_model_load: bool = False,
    overwrite: bool = False,
    analyze_compute: bool = False,
    analyze_mil: bool = False,
    benchmark: int = 0,
):
    """Convert GLM-OCR MTP module to CoreML.

    Follows the same conversion pipeline as convert_glm_ocr_text:
    1. Load MTP weights from safetensors
    2. Compute reference output before patching (for verification)
    3. Patch for channels-first CoreML execution
    4. Wrap with stateful KV cache
    5. Trace and convert to CoreML
    6. Verify against original unpatched output

    Args:
        model_name: HuggingFace model name (e.g. "zai-org/GLM-OCR").
        seq_len: Sequence length for tracing (default: 1, MTP processes
            one token at a time).
        cache_length: Maximum KV cache length (default: 2048).
        batch_size: Batch size for tracing (default: 1).
        output_path: Output path for the .mlpackage. If None, auto-generated.
        verbose: Print conversion progress.
        skip_model_load: Skip loading the converted model for verification.
        overwrite: Overwrite existing output.
        analyze_compute: Run compute plan analysis.
        analyze_mil: Run MIL program inspection.
        benchmark: Number of timed iterations to run (0 to skip).

    Returns:
        Converted CoreML model (None if skip_model_load or output exists).
    """
    print("GLM-OCR MTP CoreML Conversion")
    print("=" * 60)

    # [1] Load model and weights
    print("\n[1] Loading MTP module and weights...")
    mtp_module, text_config = load_glm_ocr_mtp_module(model_name, verbose=verbose)

    hidden_dim = int(text_config.hidden_size)

    if verbose:
        print(f"    Hidden dim: {hidden_dim}")
        print(f"    Num heads: {text_config.num_attention_heads}")
        print(f"    Num KV heads: {text_config.num_key_value_heads}")
        head_dim = (
            getattr(text_config, "head_dim", None)
            or hidden_dim // text_config.num_attention_heads
        )
        print(f"    Head dim: {head_dim}")

    # [2] Compute reference output before patching
    ref_hidden = None
    ref_for_head = None
    ref_prev_cf = None
    ref_embed_cf = None

    if not skip_model_load:
        print("\n[2] Computing reference output...")
        ref_hidden, ref_for_head, ref_prev_cf, ref_embed_cf = (
            _compute_mtp_reference_output(
                mtp_module, text_config, batch_size, verbose=verbose
            )
        )
    else:
        print("\n[2] Skipping reference output (--skip-model-load)")

    # [3] Patch model
    print("\n[3] Patching MTP module...")
    patch_glm_ocr_mtp_module(mtp_module, text_config=text_config, verbose=verbose)

    # [4] Wrap with stateful KV cache
    print("\n[4] Creating GlmOcrMTPWrapper...")
    with torch.inference_mode():
        wrapper = GlmOcrMTPWrapper(
            mtp_module=mtp_module,
            config=text_config,
            cache_length=cache_length,
            channels_first=True,
            device="cpu",
        )
        wrapper.eval()
    print(f"    {wrapper}")

    # [5] Trace model
    print("\n[5] Tracing MTP model...")
    example_inputs = (
        torch.randn((batch_size, hidden_dim, 1, seq_len), dtype=torch.float32),
        torch.randn((batch_size, hidden_dim, 1, seq_len), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.int32),
    )
    with torch.inference_mode():
        traced_model = torch.jit.trace(wrapper, example_inputs)
    print("    Tracing complete!")

    # Determine output path
    if output_path is None:
        model_short_name = model_name.split("/")[-1].lower().replace("-", "_")
        output_path = f"{model_short_name}_mtp_seqlen_{seq_len}.mlpackage"

    output_p = Path(output_path)
    if output_p.exists():
        if not overwrite:
            print(f"\n    Output already exists: {output_path}")
            print("    Use --overwrite to replace")
            return None
        import shutil
        shutil.rmtree(output_p)
        print(f"    Removed existing: {output_path}")

    # [6] Convert to CoreML
    print("\n[6] Converting to CoreML...")
    register_extended_passes()

    state_specs = create_glm_ocr_mtp_state_specs(wrapper)
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                shape=(batch_size, hidden_dim, 1, seq_len),
                name="previous_hidden_states",
            ),
            ct.TensorType(
                shape=(batch_size, hidden_dim, 1, seq_len),
                name="input_embeds",
            ),
            ct.TensorType(
                shape=(1,),
                name="position_id",
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="mtp_hidden"),
            ct.TensorType(name="mtp_hidden_for_head"),
        ],
        states=state_specs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        package_dir=str(output_path),
        skip_model_load=skip_model_load,
    )
    print(f"    Saved to: {output_path}")

    # Optional analysis
    if analyze_compute or analyze_mil:
        from .analysis import analyze_compute_plan, inspect_mil_program

        if analyze_compute:
            print("\n  Compute Plan Analysis")
            analyze_compute_plan(mlmodel)

        if analyze_mil:
            print("\n  MIL Program Inspection")
            inspect_mil_program(mlmodel)

    # [7] Verify outputs
    print("\n[7] Verifying outputs...")
    if skip_model_load:
        print("    Skipped (--skip-model-load flag set)")
    elif ref_hidden is None:
        print("    Skipped (reference output not available)")
    else:
        state = mlmodel.make_state()
        # Position 0: input_embeds zeroed (pos-0 masking handled by caller)
        coreml_out = mlmodel.predict(
            {
                "previous_hidden_states": ref_prev_cf,
                "input_embeds": np.zeros_like(ref_embed_cf),
                "position_id": np.array([0], dtype=np.int32),
            },
            state,
        )
        coreml_hidden = coreml_out["mtp_hidden"]
        coreml_for_head = coreml_out["mtp_hidden_for_head"]

        print(f"    mtp_hidden shape: PyTorch={ref_hidden.shape}, CoreML={coreml_hidden.shape}")
        print(f"    mtp_hidden_for_head shape: PyTorch={ref_for_head.shape}, CoreML={coreml_for_head.shape}")

        # Check mtp_hidden
        abs_diff_hidden = np.abs(ref_hidden - coreml_hidden)
        print("    mtp_hidden difference:")
        print(f"      Max:  {abs_diff_hidden.max():.6f}")
        print(f"      Mean: {abs_diff_hidden.mean():.6f}")

        # Check mtp_hidden_for_head
        abs_diff_head = np.abs(ref_for_head - coreml_for_head)
        print("    mtp_hidden_for_head difference:")
        print(f"      Max:  {abs_diff_head.max():.6f}")
        print(f"      Mean: {abs_diff_head.mean():.6f}")

        if abs_diff_hidden.max() < 0.1 and abs_diff_head.max() < 0.1:
            print("    [OK] Outputs match within tolerance!")
        else:
            print("    [WARNING] Outputs differ significantly")

        # Test stateful inference (second forward pass at position 1)
        print("\n[8] Testing stateful inference (second forward pass)...")
        test_prev2 = np.random.randn(batch_size, hidden_dim, 1, seq_len).astype(np.float32)
        test_embed2 = np.random.randn(batch_size, hidden_dim, 1, seq_len).astype(np.float32)
        coreml_out2 = mlmodel.predict(
            {
                "previous_hidden_states": test_prev2,
                "input_embeds": test_embed2,
                "position_id": np.array([1], dtype=np.int32),
            },
            state,
        )
        print(f"    Second mtp_hidden shape: {coreml_out2['mtp_hidden'].shape}")
        print(f"    Second mtp_hidden_for_head shape: {coreml_out2['mtp_hidden_for_head'].shape}")

    # Benchmark
    if benchmark > 0 and not skip_model_load:
        import time

        print(f"\n[9] Benchmarking ({benchmark} iterations)...")
        bench_state = mlmodel.make_state()
        bench_inputs = {
            "previous_hidden_states": np.random.randn(
                batch_size, hidden_dim, 1, seq_len
            ).astype(np.float32),
            "input_embeds": np.random.randn(
                batch_size, hidden_dim, 1, seq_len
            ).astype(np.float32),
            "position_id": np.array([0], dtype=np.int32),
        }

        # Warmup (10 iterations)
        num_warmup = min(10, benchmark)
        for i in range(num_warmup):
            t0 = time.perf_counter()
            _ = mlmodel.predict(bench_inputs, bench_state)
            t1 = time.perf_counter()
            if i == 0:
                print(f"    Warmup 1/{num_warmup}: {(t1 - t0) * 1000:.2f}ms (includes JIT)")
            elif i == num_warmup - 1:
                print(f"    Warmup {num_warmup}/{num_warmup}: {(t1 - t0) * 1000:.2f}ms")

        # Timed iterations
        latencies = []
        for i in range(benchmark):
            bench_inputs["position_id"] = np.array(
                [(num_warmup + i) % cache_length], dtype=np.int32
            )
            t0 = time.perf_counter()
            _ = mlmodel.predict(bench_inputs, bench_state)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        latencies_np = np.array(latencies)
        print(f"    Iterations: {benchmark}")
        print(f"    Latency (ms):")
        print(f"      Mean:   {latencies_np.mean():.3f}")
        print(f"      Median: {np.median(latencies_np):.3f}")
        print(f"      Std:    {latencies_np.std():.3f}")
        print(f"      Min:    {latencies_np.min():.3f}")
        print(f"      Max:    {latencies_np.max():.3f}")
        print(f"      p95:    {np.percentile(latencies_np, 95):.3f}")
        print(f"      p99:    {np.percentile(latencies_np, 99):.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Model saved to: {output_path}")
    print("Architecture: GLM-OCR MTP (1 transformer layer)")
    print(f"KV cache shape: (1, {text_config.num_key_value_heads}, {cache_length}, {head_dim})")

    return mlmodel
