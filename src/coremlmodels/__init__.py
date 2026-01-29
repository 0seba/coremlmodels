"""CoreML Models - Convert HuggingFace PyTorch models to CoreML with Neural Engine backend."""

__version__ = "0.1.0"

from .analysis import analyze_compute_plan, inspect_mil_program
from .graph_passes import register_extended_passes
from .lm_model_wrapper import LanguageModelWrapper, create_coreml_state_specs
from .patch_attention import (
    AttentionPatcher,
    apply_rotary_pos_emb,
    patch_model_attention,
    rotate_half,
)
from .patch_linears import LinearToConv2dPatcher, patch_model_linears
from .patch_rmsnorm import RMSNormToLayerNormPatcher, patch_model_rmsnorms
from .registry import (
    ARCHITECTURE_REGISTRY,
    ArchitectureConfig,
    find_target_classes,
    get_architecture_config,
    get_supported_architectures,
)

__all__ = [
    # Linear patching
    "LinearToConv2dPatcher",
    "patch_model_linears",
    # RMSNorm patching
    "RMSNormToLayerNormPatcher",
    "patch_model_rmsnorms",
    # Attention patching
    "AttentionPatcher",
    "patch_model_attention",
    "rotate_half",
    "apply_rotary_pos_emb",
    # Language model wrapper
    "LanguageModelWrapper",
    "create_coreml_state_specs",
    # Architecture registry
    "ARCHITECTURE_REGISTRY",
    "ArchitectureConfig",
    "get_architecture_config",
    "find_target_classes",
    "get_supported_architectures",
    # Analysis utilities
    "analyze_compute_plan",
    "inspect_mil_program",
    # Graph passes
    "register_extended_passes",
]
