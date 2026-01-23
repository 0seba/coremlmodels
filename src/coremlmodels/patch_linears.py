"""Linear to Conv2d Patching for CoreML Neural Engine Optimization.

This module provides utilities to convert Linear layers to Conv2d operations
which are better aligned with the Neural Engine compute backend in CoreML.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearToConv2dPatcher(nn.Module):
    """Patches a Linear layer to perform equivalent Conv2d operations.

    This class wraps a Linear layer and converts its forward pass to use
    torch.nn.functional.conv2d instead. The weight tensor is reshaped from
    (out_features, in_features) to (out_features, in_features, 1, 1) to
    perform 1x1 convolutions that are equivalent to the original linear operation.

    This optimization is beneficial for CoreML Neural Engine execution as
    convolutions are more efficiently mapped to the neural compute units.

    Args:
        linear_layer: The torch.nn.Linear layer to patch.
        bias: Whether to include bias in the convolution. Defaults to True.
    """

    def __init__(self, linear_layer: nn.Linear, bias: bool = True):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.has_bias = bias and linear_layer.bias is not None

        # Create conv2d weight: (out_features, in_features, 1, 1)
        # Keep as view to avoid doubling memory usage
        self.weight = linear_layer.weight.reshape(
            linear_layer.out_features, linear_layer.in_features, 1, 1
        ).detach()

        # Handle bias - keep as view
        if self.has_bias:
            self.bias = linear_layer.bias.detach()
        else:
            self.bias = None

        # Store original module info for reference
        self.module_name: Optional[str] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using 1x1 conv2d instead of linear.

        Args:
            x: Input tensor of shape (batch_size, in_features, 1, 1)

        Returns:
            Output tensor of shape (batch_size, out_features, 1, 1)
        """
        # Perform 1x1 convolution - input must be 4D (N, C, H, W)
        return F.conv2d(
            x, self.weight, bias=self.bias, stride=1, padding=0, dilation=1, groups=1
        )

    def __repr__(self) -> str:
        return (
            f"LinearToConv2dPatcher("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"has_bias={self.has_bias})"
        )


def patch_model_linears(
    model: nn.Module, skip_modules: Optional[list] = None, verbose: bool = False
) -> nn.Module:
    """Iterate through model modules and patch Linear layers to Conv2d.

    This function traverses all modules in the model and replaces Linear
    layers with LinearToConv2dPatcher instances that use 1x1 convolutions
    for better Neural Engine performance.

    Args:
        model: The PyTorch model to patch.
        skip_modules: List of module names to skip patching. Defaults to None.
        verbose: If True, print information about patched layers. Defaults to False.

    Returns:
        The patched model with Linear layers converted to Conv2d operations.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        >>> patched_model = patch_model_linears(model)
        >>> # Linear layers are now using 1x1 convolutions internally
    """
    if skip_modules is None:
        skip_modules = []

    def _patch_module(module: nn.Module, name: str = "") -> None:
        """Recursively patch linear layers in the module."""
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if full_name in skip_modules:
                if verbose:
                    print(f"Skipping: {full_name}")
                continue

            # Check if this is a Linear layer
            if isinstance(child_module, nn.Linear):
                patcher = LinearToConv2dPatcher(child_module)
                patcher.module_name = full_name

                # Replace the linear layer with our patcher
                setattr(module, child_name, patcher)

                if verbose:
                    print(f"Patched: {full_name} -> {patcher}")

            # Recursively process child modules
            elif isinstance(child_module, nn.Module):
                _patch_module(child_module, full_name)

    _patch_module(model)
    return model
