"""RMSNorm to LayerNorm-fusable ops patching for CoreML Neural Engine Optimization.

This module provides utilities to convert RMSNorm layers to a sequence of operations
that CoreMLTools will fuse into LayerNorm, avoiding FP16 overflow on Neural Engine.

The key insight is that concatenating [x, -x] and performing LayerNorm-equivalent
operations produces the same result as RMSNorm:
1. Mean of [x, -x] is 0 (they cancel out)
2. Variance of [x, -x] equals mean(x^2) (since mean is 0)
3. LayerNorm on [x, -x] = x / sqrt(mean(x^2) + eps) = RMSNorm!
"""

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn


class RMSNormToLayerNormPatcher(nn.Module):
    """Patches an RMSNorm layer to use LayerNorm-equivalent operations.

    This class wraps an RMSNorm layer and converts its forward pass to use
    operations that CoreMLTools will fuse into a LayerNorm operation. This
    avoids FP16 accumulation overflow issues on the Neural Engine.

    The trick: concatenate [x, -x] along channel dimension, then perform
    LayerNorm-equivalent operations. The mean of [x, -x] is 0, and the
    variance equals mean(x^2), giving us RMSNorm behavior.

    Args:
        rmsnorm_layer: The RMSNorm layer to patch. Supports nn.RMSNorm and
                       other implementations with compatible interfaces.

    Attributes:
        eps: Epsilon value for numerical stability.
        weight: Concatenated weight tensor [original_weight, zeros] for
                scaling only the first half of the output.
        bias: Zero tensor (RMSNorm has no bias, but LayerNorm ops need it).
    """

    DEFAULT_TARGET_CLASSES: Tuple[Type[nn.Module], ...] = (nn.RMSNorm,)

    def __init__(self, rmsnorm_layer: nn.Module):
        super().__init__()

        self.module_name: Optional[str] = None

        # Extract epsilon - handle different attribute names
        # nn.RMSNorm can have eps=None, so check both existence and value
        if hasattr(rmsnorm_layer, "eps") and rmsnorm_layer.eps is not None:
            self.eps = rmsnorm_layer.eps
        elif hasattr(rmsnorm_layer, "variance_epsilon"):
            self.eps = rmsnorm_layer.variance_epsilon
        else:
            # Default fallback (matches PyTorch's internal default for RMSNorm)
            self.eps = 1e-5

        # Handle weight - concatenate [weight, zeros] for the [x, -x] trick
        if hasattr(rmsnorm_layer, "weight") and rmsnorm_layer.weight is not None:
            original_weight = rmsnorm_layer.weight.detach()
            # Shape: (1, 2*normalized_shape, 1, 1) for 4D input
            self.register_buffer(
                "weight",
                torch.cat(
                    [
                        original_weight.view(1, -1, 1, 1),
                        torch.zeros_like(original_weight).view(1, -1, 1, 1),
                    ],
                    dim=1,
                ),
            )
            self.normalized_shape = original_weight.shape[0]
        else:
            self.weight = None
            # Infer normalized_shape from layer if possible
            if hasattr(rmsnorm_layer, "normalized_shape"):
                self.normalized_shape = rmsnorm_layer.normalized_shape
            else:
                self.normalized_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using LayerNorm-equivalent operations.

        The sequence of operations (mean, sub, square, mean, rsqrt, mul)
        is detected by CoreMLTools and fused into a LayerNorm operation.

        Args:
            x: Input tensor of shape (batch_size, channels, 1, 1)
               Channels must match the original normalized_shape.

        Returns:
            Output tensor of shape (batch_size, channels, 1, 1)
        """
        # Concatenate [x, -x] along channel dimension
        x_concat = torch.cat([x, -x], dim=1)

        # Store input dtype for restoration
        input_dtype = x_concat.dtype

        # Perform in float32 for numerical stability during tracing
        x_float = x_concat.float()

        # LayerNorm-equivalent operations (CoreMLTools will fuse these)
        # 1. Compute channel mean
        channels_mean = x_float.mean(dim=1, keepdim=True)

        # 2. Zero-center the input
        zero_mean = x_float - channels_mean

        # 3. Compute squared values
        zero_mean_sq = zero_mean * zero_mean

        # 4. Compute variance and reciprocal sqrt
        variance = zero_mean_sq.mean(dim=1, keepdim=True)
        denom = (variance + self.eps).rsqrt()

        # 5. Normalize
        out = zero_mean * denom

        # Apply weight and bias if present
        if self.weight is not None:
            out = out * self.weight

        # Take the first half (the x portion, not -x)
        out = torch.chunk(out, 2, dim=1)[0]

        # Restore original dtype
        return out.to(input_dtype)

    def __repr__(self) -> str:
        return (
            f"RMSNormToLayerNormPatcher("
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"has_weight={self.weight is not None})"
        )


def patch_model_rmsnorms(
    model: nn.Module,
    target_classes: Optional[Tuple[Type[nn.Module], ...]] = None,
    skip_modules: Optional[List[str]] = None,
    verbose: bool = False,
) -> nn.Module:
    """Iterate through model modules and patch RMSNorm layers.

    This function traverses all modules in the model and replaces RMSNorm
    layers with RMSNormToLayerNormPatcher instances that use LayerNorm-fusable
    operations for Neural Engine compatibility.

    Args:
        model: The PyTorch model to patch.
        target_classes: Tuple of RMSNorm class types to patch. Defaults to
                        (nn.RMSNorm,). Can include custom classes like
                        Qwen2RMSNorm, LlamaRMSNorm, etc.
        skip_modules: List of module names (dot-separated paths) to skip
                      patching. Defaults to None.
        verbose: If True, print information about patched layers.
                 Defaults to False.

    Returns:
        The patched model with RMSNorm layers converted to LayerNorm-fusable
        operations.

    Example:
        >>> model = nn.Sequential(nn.RMSNorm(64), nn.ReLU(), nn.RMSNorm(64))
        >>> patched_model = patch_model_rmsnorms(model)
        >>> # RMSNorm layers are now using LayerNorm-fusable operations internally
    """
    if target_classes is None:
        target_classes = RMSNormToLayerNormPatcher.DEFAULT_TARGET_CLASSES

    if skip_modules is None:
        skip_modules = []

    def _patch_module(module: nn.Module, name: str = "") -> None:
        """Recursively patch RMSNorm layers in the module."""
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if full_name in skip_modules:
                if verbose:
                    print(f"Skipping: {full_name}")
                continue

            # Check if this is a target RMSNorm class
            if isinstance(child_module, target_classes):
                patcher = RMSNormToLayerNormPatcher(child_module)
                patcher.module_name = full_name

                # Replace the RMSNorm layer with our patcher
                setattr(module, child_name, patcher)

                if verbose:
                    print(f"Patched: {full_name} -> {patcher}")

            # Recursively process child modules
            elif isinstance(child_module, nn.Module):
                _patch_module(child_module, full_name)

    _patch_module(model)
    return model
