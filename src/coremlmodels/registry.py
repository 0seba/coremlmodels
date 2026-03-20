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
    """Traverse model modules to resolve class name strings to actual types.

    Avoids importing HuggingFace model internals — just walks the module
    tree and matches type().__name__ against the requested names.
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
    return tuple(sorted(ARCHITECTURE_REGISTRY.keys()))
