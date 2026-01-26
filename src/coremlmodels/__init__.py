"""CoreML Models - Convert HuggingFace PyTorch models to CoreML with Neural Engine backend."""

__version__ = "0.1.0"

from .analysis import analyze_compute_plan, inspect_mil_program
from .patch_linears import LinearToConv2dPatcher, patch_model_linears
from .patch_rmsnorm import RMSNormToLayerNormPatcher, patch_model_rmsnorms

__all__ = [
    "LinearToConv2dPatcher",
    "patch_model_linears",
    "RMSNormToLayerNormPatcher",
    "patch_model_rmsnorms",
    "analyze_compute_plan",
    "inspect_mil_program",
]
