"""GLM-OCR model components for CoreML conversion and inference.

Provides text decoder patchers (4-norm sandwich, interleaved-to-split-half RoPE),
MTP speculative decoding module, and a CoreML-backed end-to-end OCR pipeline.
"""

from coremlmodels.glm_ocr.glm_ocr_text_model import (
    GlmOcrLanguageModelWrapper,
    GlmOcrTextAttentionPatcher,
    GlmOcrTextDecoderLayerPatcher,
    GlmOcrTextMLPPatcher,
    create_glm_ocr_state_specs,
    patch_glm_ocr_text_layers,
)
from coremlmodels.glm_ocr.glm_ocr_mtp_model import (
    GlmOcrMTPModule,
    GlmOcrMTPWrapper,
    convert_glm_ocr_mtp,
    create_glm_ocr_mtp_state_specs,
    load_glm_ocr_mtp_module,
    patch_glm_ocr_mtp_module,
)
from coremlmodels.glm_ocr.glm_ocr_coreml_pipeline import (
    GenerationConfig,
    GlmOcrCoreMLPipeline,
    TimingStats,
)
