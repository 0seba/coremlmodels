"""CoreML Models - Convert HuggingFace PyTorch models to CoreML with Neural Engine backend."""

__version__ = "0.1.0"

from .patch_linears import LinearToConv2dPatcher, patch_model_linears

__all__ = ["LinearToConv2dPatcher", "patch_model_linears"]
