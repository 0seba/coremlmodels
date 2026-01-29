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
import torch.nn.functional as F


class RMSNormToLayerNormPatcher(nn.Module):
    """Patches an RMSNorm layer to use LayerNorm-equivalent operations.

    This class wraps an RMSNorm layer and converts its forward pass to use
    operations that CoreMLTools will fuse into a LayerNorm operation. This
    avoids FP16 accumulation overflow issues on the Neural Engine.

    The trick: concatenate [x, -x] along the normalization dimension, then perform
    LayerNorm-equivalent operations. The mean of [x, -x] is 0, and the
    variance equals mean(x^2), giving us RMSNorm behavior.

    Supports flexible axis selection:
    - axis=1 (default): Channel dimension normalization for 4D tensors
    - axis=-1: Last dimension normalization using F.layer_norm
    - Other axes: Uses the concat trick with dynamic weight reshaping

    Args:
        rmsnorm_layer: The RMSNorm layer to patch. Supports nn.RMSNorm and
                       other implementations with compatible interfaces.

    Attributes:
        eps: Epsilon value for numerical stability.
        weight: Original weight tensor (1D) for scaling.
        normalized_shape: Size of the normalization dimension.
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

        # Store original weight (1D) - reshaping done in forward based on axis
        if hasattr(rmsnorm_layer, "weight") and rmsnorm_layer.weight is not None:
            original_weight = rmsnorm_layer.weight.detach()
            self.register_buffer("weight", original_weight)
            self.normalized_shape = original_weight.shape[0]

            # Pre-compute weight_concat for the [x, -x] trick
            # [weight, zeros] to only scale the x portion
            weight_concat = torch.cat([original_weight, torch.zeros_like(original_weight)])
            self.register_buffer("weight_concat", weight_concat)
        else:
            self.weight = None
            self.weight_concat = None
            # Infer normalized_shape from layer if possible
            if hasattr(rmsnorm_layer, "normalized_shape"):
                self.normalized_shape = rmsnorm_layer.normalized_shape
            else:
                self.normalized_shape = None

    def forward(self, x: torch.Tensor, axis: int = 1) -> torch.Tensor:
        """Forward pass using LayerNorm-equivalent operations.

        The sequence of operations is detected by CoreMLTools and fused into
        a LayerNorm operation when axis=1. For axis=-1, uses F.layer_norm.

        Args:
            x: Input tensor. For axis=1, expects shape (batch, channels, 1, seq).
               Channels must match the original normalized_shape.
            axis: Dimension to normalize over. Defaults to 1 (channel dimension).
                  Use -1 for last dimension normalization (uses F.layer_norm).

        Returns:
            Normalized output tensor with same shape as input.
        """
        # Handle negative axis
        if axis < 0:
            axis = x.ndim + axis

        if axis == x.ndim - 1:
            # Last dimension: use F.layer_norm (efficient for this case)
            return self._forward_functional(x)
        else:
            # Other dimensions: use concat trick
            return self._forward_concat_trick(x, axis)

    def _forward_functional(self, x: torch.Tensor) -> torch.Tensor:
        """Use F.layer_norm for last dimension normalization (RMSNorm equivalent).

        This approach concatenates [x, -x] along the last dimension and applies
        F.layer_norm, which is efficient for last-dimension normalization.
        """
        input_dtype = x.dtype

        # Create [x, -x] along last dimension for RMSNorm behavior
        x_concat = torch.cat([x, -x], dim=-1)

        # Apply layer_norm on the concatenated last dimension
        # Uses pre-computed weight_concat: [weight, zeros] to only scale the x portion
        out = F.layer_norm(
            x_concat.float(),
            [x_concat.size(-1)],
            weight=self.weight_concat,
            bias=None,
            eps=self.eps,
        )

        # Take the first half (x portion)
        out = torch.chunk(out, 2, dim=-1)[0]

        return out.to(input_dtype)

    def _forward_concat_trick(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        """Use [x, -x] concat trick for arbitrary axis normalization.

        This method handles normalization over any axis by concatenating
        [x, -x] along that axis and performing LayerNorm-equivalent operations.
        The modified fuse_layernorm_or_instancenorm pass will fuse this pattern.
        """
        # Concatenate [x, -x] along the specified axis
        x_concat = torch.cat([x, -x], dim=axis)

        # Store input dtype for restoration
        input_dtype = x_concat.dtype

        # Perform in float32 for numerical stability during tracing
        x_float = x_concat.float()

        # LayerNorm-equivalent operations (modified CoreMLTools pass will fuse these)
        # 1. Compute mean along the normalization axis
        channels_mean = x_float.mean(dim=axis, keepdim=True)

        # 2. Zero-center the input
        zero_mean = x_float - channels_mean

        # 3. Compute squared values
        zero_mean_sq = zero_mean * zero_mean

        # 4. Compute variance and reciprocal sqrt
        variance = zero_mean_sq.mean(dim=axis, keepdim=True)
        denom = (variance + self.eps).rsqrt()

        # 5. Normalize
        out = zero_mean * denom

        # Apply weight if present - reshape based on axis
        if self.weight_concat is not None:
            # Reshape weight for broadcasting: [1, 1, ..., 2*normalized_shape, ..., 1]
            shape = [1] * x_concat.ndim
            shape[axis] = self.weight_concat.shape[0]
            weight_reshaped = self.weight_concat.view(*shape)

            out = out * weight_reshaped

        # Take the first half (the x portion, not -x)
        out = torch.chunk(out, 2, dim=axis)[0]

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
