"""Architecture registry for automatic model detection and patching configuration.

This module provides a registry of supported model architectures and utilities
to automatically detect the correct classes to patch based on the model type.

The registry maps HuggingFace model types (from config.model_type) to their
corresponding attention and RMSNorm class names, along with architecture-specific
features like QK-norm support.

Example:
    >>> from transformers import AutoConfig
    >>> from coremlmodels.registry import get_architecture_config, find_target_classes
    >>>
    >>> config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    >>> arch_config = get_architecture_config(config.model_type)
    >>> print(arch_config.has_qk_norm)  # False for Qwen2
"""

from dataclasses import dataclass
from typing import Dict, Set, Tuple, Type

import torch.nn as nn


@dataclass
class ArchitectureConfig:
    """Configuration for a specific model architecture.

    Attributes:
        attention_class_names: Tuple of attention class names to patch
            (e.g., ("Qwen2Attention",)).
        rmsnorm_class_names: Tuple of RMSNorm class names to patch
            (e.g., ("Qwen2RMSNorm",)).
        has_qk_norm: Whether the architecture uses QK-norm in attention.
            Qwen3 applies RMSNorm to Q and K after projection, before RoPE.
    """

    attention_class_names: Tuple[str, ...]
    rmsnorm_class_names: Tuple[str, ...]
    has_qk_norm: bool = False


# Registry mapping model_type -> ArchitectureConfig
ARCHITECTURE_REGISTRY: Dict[str, ArchitectureConfig] = {
    "qwen2": ArchitectureConfig(
        attention_class_names=("Qwen2Attention",),
        rmsnorm_class_names=("Qwen2RMSNorm",),
        has_qk_norm=False,
    ),
    "qwen3": ArchitectureConfig(
        attention_class_names=("Qwen3Attention",),
        rmsnorm_class_names=("Qwen3RMSNorm",),
        has_qk_norm=True,
    ),
    "glm_ocr": ArchitectureConfig(
        attention_class_names=("GlmOcrTextAttention",),
        rmsnorm_class_names=("GlmOcrRMSNorm",),
        has_qk_norm=False,
    ),
}


def get_architecture_config(model_type: str) -> ArchitectureConfig:
    """Get the architecture configuration for a model type.

    Args:
        model_type: The model type string from HuggingFace config
            (e.g., "qwen2", "qwen3", "llama").

    Returns:
        ArchitectureConfig for the specified model type.

    Raises:
        ValueError: If the model type is not in the registry.

    Example:
        >>> config = get_architecture_config("qwen3")
        >>> print(config.has_qk_norm)
        True
    """
    if model_type not in ARCHITECTURE_REGISTRY:
        supported = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Supported types: {supported}"
        )
    return ARCHITECTURE_REGISTRY[model_type]


def find_target_classes(
    model: nn.Module,
    class_names: Tuple[str, ...],
) -> Tuple[Type[nn.Module], ...]:
    """Find actual class types by name from the model's modules.

    This function traverses the model and finds the actual Python class types
    that match the given class names. This allows us to use isinstance()
    checks without importing HuggingFace model internals.

    Args:
        model: The PyTorch model to search.
        class_names: Tuple of class name strings to find (e.g., ("Qwen2Attention",)).

    Returns:
        Tuple of class types found in the model that match the given names.

    Raises:
        ValueError: If none of the specified class names are found in the model.

    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("Qwen/Qwen2-0.5B")
        >>> classes = find_target_classes(model, ("Qwen2Attention",))
        >>> print(classes[0].__name__)
        Qwen2Attention
    """
    class_name_set: Set[str] = set(class_names)
    found_classes: Dict[str, Type[nn.Module]] = {}

    for module in model.modules():
        module_class_name = type(module).__name__
        if module_class_name in class_name_set and module_class_name not in found_classes:
            found_classes[module_class_name] = type(module)

    if not found_classes:
        raise ValueError(
            f"None of the target classes {class_names} were found in the model. "
            f"Check that the model architecture matches the expected type."
        )

    return tuple(found_classes.values())


def get_supported_architectures() -> Tuple[str, ...]:
    """Get a tuple of all supported architecture names.

    Returns:
        Tuple of supported model type strings.

    Example:
        >>> supported = get_supported_architectures()
        >>> print("qwen2" in supported)
        True
    """
    return tuple(sorted(ARCHITECTURE_REGISTRY.keys()))
