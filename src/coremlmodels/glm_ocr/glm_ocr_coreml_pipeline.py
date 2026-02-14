"""GLM-OCR pipeline using converted CoreML vision + text models.

This module implements an end-to-end OCR path without document layout
preprocessing. It runs:
1. HuggingFace processor preprocessing (image resize/chunk + prompt tokens)
2. CoreML vision encoder (patch-based)
3. Embedding replacement at image placeholder token positions
4. CoreML text decoder with KV cache
5. CoreML LM head + token decoding
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Optional

import numpy as np
import coremltools as ct
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.utils.hub import cached_file

from coremlmodels.glm_ocr.glm_ocr_text_model import compute_glm_ocr_mrope_cos_sin
from coremlmodels.compiled_model_utils import load_model_with_cache
from coremlmodels.vision_model_wrapper import (
    ENUMERATED_PATCH_COUNTS,
    compute_vision_rotary_pos_emb,
    create_padding_attention_mask,
    create_patch_mask,
)

try:
    from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrModel as HFGlmOcrModel
except Exception:
    HFGlmOcrModel = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass
class TimingStats:
    preprocessing_ms: float = 0.0
    vision_encode_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    num_tokens_generated: int = 0
    # MTP-specific
    mtp_prefill_ms: float = 0.0
    mtp_prefill_positions: int = 0
    mtp_draft_steps: int = 0
    mtp_draft_total_ms: float = 0.0
    spec_verify_steps: int = 0
    spec_verify_total_ms: float = 0.0
    drafts_proposed: int = 0
    drafts_accepted: int = 0

    @property
    def decode_tokens_per_sec(self) -> float:
        if self.decode_ms <= 0:
            return 0.0
        return self.num_tokens_generated / (self.decode_ms / 1000.0)

    @property
    def mtp_avg_step_ms(self) -> float:
        if self.mtp_draft_steps <= 0:
            return 0.0
        return self.mtp_draft_total_ms / self.mtp_draft_steps

    @property
    def spec_avg_verify_ms(self) -> float:
        if self.spec_verify_steps <= 0:
            return 0.0
        return self.spec_verify_total_ms / self.spec_verify_steps

    @property
    def spec_acceptance_rate(self) -> float:
        if self.drafts_proposed <= 0:
            return 0.0
        return self.drafts_accepted / self.drafts_proposed

    @property
    def total_ms(self) -> float:
        return self.preprocessing_ms + self.vision_encode_ms + self.prefill_ms + self.decode_ms


class GlmOcrCoreMLPipeline:
    """CoreML-backed GLM-OCR inference pipeline.

    Supports optional MTP (Multi-Token Prediction) speculative decoding.
    When an MTP model is provided, the decode loop drafts multiple tokens
    per round using the cheap single-layer MTP model, then verifies them
    all in a single batched main model forward pass.
    """

    def __init__(
        self,
        vision_model_path: str | Path,
        text_model_path: str | Path,
        lm_head_path: str | Path,
        embeddings_path: str | Path,
        mtp_model_path: str | Path | None = None,
        num_spec_steps: int = 3,
        hf_model_name: str = "zai-org/GLM-OCR",
        cache_compiled: bool = False,
        verbose: bool = True,
        max_vision_patches: int | None = None,
    ) -> None:
        self.verbose = verbose

        vision_model_path = Path(vision_model_path)
        text_model_path = Path(text_model_path)
        lm_head_path = Path(lm_head_path)

        if verbose:
            print("Loading CoreML models...")
        self.vision_model, self.vision_specs = load_model_with_cache(
            vision_model_path,
            cache_compiled=cache_compiled,
            verbose=verbose,
        )
        self.text_model, self.text_specs = load_model_with_cache(
            text_model_path,
            cache_compiled=cache_compiled,
            verbose=verbose,
        )
        self.lm_head, self.lm_head_specs = load_model_with_cache(
            lm_head_path,
            cache_compiled=cache_compiled,
            verbose=verbose,
        )

        # MTP model for speculative decoding (optional)
        self.mtp_model = None
        self.num_spec_steps = num_spec_steps
        if mtp_model_path is not None:
            mtp_model_path = Path(mtp_model_path)
            if verbose:
                print("  Loading MTP model for speculative decoding...")
            self.mtp_model, _ = load_model_with_cache(
                mtp_model_path,
                cache_compiled=cache_compiled,
                verbose=verbose,
            )
        all_patch_counts = self._infer_available_vision_patch_counts(vision_model_path)
        if not all_patch_counts:
            raise ValueError("Could not determine available vision patch counts.")

        if max_vision_patches is not None:
            if max_vision_patches <= 0:
                raise ValueError("max_vision_patches must be > 0")
            active_patch_counts = tuple(
                c for c in all_patch_counts if c <= int(max_vision_patches)
            )
            if not active_patch_counts:
                raise ValueError(
                    f"Requested max_vision_patches={max_vision_patches} is below "
                    f"the smallest supported shape ({all_patch_counts[0]})."
                )
            if verbose and active_patch_counts[-1] != int(max_vision_patches):
                print(
                    f"  Requested max vision patches {max_vision_patches} adjusted to "
                    f"nearest supported value {active_patch_counts[-1]}."
                )
        else:
            active_patch_counts = all_patch_counts

        self.available_vision_patch_counts = active_patch_counts
        self.max_vision_patches = active_patch_counts[-1]
        if verbose:
            print(
                f"  Vision enumerated patch counts: {list(self.available_vision_patch_counts)} "
                f"(max={self.max_vision_patches})"
            )
            if self.max_vision_patches < 4096:
                print(
                    "  Warning: low vision patch budget detected. Large pages/tables will "
                    "be resized aggressively before encoding, which can hurt OCR fidelity."
                )

        self.embeddings = np.load(str(embeddings_path)).astype(np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings shape (vocab, hidden), got {self.embeddings.shape}")

        self.config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        self._init_preprocessing_components(
            hf_model_name=hf_model_name,
            verbose=verbose,
        )

        self.image_token_id = self.config.image_token_id
        eos_token_cfg = self.config.text_config.eos_token_id
        if isinstance(eos_token_cfg, (list, tuple, set)):
            self.eos_token_ids = {int(x) for x in eos_token_cfg}
        else:
            self.eos_token_ids = {int(eos_token_cfg)}
        self.hidden_dim = self.config.text_config.hidden_size
        self.text_head_dim = (
            getattr(self.config.text_config, "head_dim", None)
            or self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        )
        self.text_rope_theta = float(self.config.text_config.rope_parameters["rope_theta"])
        self.text_partial_rotary_factor = float(
            self.config.text_config.rope_parameters.get("partial_rotary_factor", 1.0)
        )
        self.text_mrope_section = tuple(
            int(x) for x in self.config.text_config.rope_parameters.get("mrope_section", [8, 12, 12])
        )
        self.vision_hidden_dim = self.config.vision_config.out_hidden_size
        self.vision_in_channels = self.config.vision_config.in_channels
        self.vision_temporal_patch_size = self.config.vision_config.temporal_patch_size
        self.vision_patch_size = self.config.vision_config.patch_size
        self.vision_spatial_merge_size = self.config.vision_config.spatial_merge_size
        self.vision_head_dim = self.config.vision_config.hidden_size // self.config.vision_config.num_heads

        if self.hidden_dim != self.vision_hidden_dim:
            raise ValueError(
                f"Text hidden size ({self.hidden_dim}) and vision output size ({self.vision_hidden_dim}) differ."
            )
        if self.embeddings.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Embeddings hidden size ({self.embeddings.shape[1]}) does not match model ({self.hidden_dim})."
            )

        self.text_seq_len = self._get_text_seq_len()
        self.cache_length = self._get_cache_length()
        self.text_input_names = self._get_text_input_names()
        self.text_uses_explicit_rope = {"position_cos", "position_sin"}.issubset(
            self.text_input_names
        )
        self._rope_index_helper = type("RopeIndexHelper", (), {})()
        self._rope_index_helper.config = self.config
        if verbose:
            mode = (
                "explicit (position_cos/position_sin)"
                if self.text_uses_explicit_rope
                else "legacy (position_id-only)"
            )
            print(f"  Text RoPE mode: {mode}")
            if self.text_uses_explicit_rope and HFGlmOcrModel is None:
                print(
                    "  Warning: HF GlmOcrModel class unavailable; falling back to "
                    "approximate sequential RoPE positions."
                )

    @staticmethod
    def _cached_file(pretrained_model_name_or_path: str, filename: str) -> str:
        """Resolve a model file from local cache first, then allow remote fallback."""
        try:
            return cached_file(
                pretrained_model_name_or_path,
                filename,
                local_files_only=True,
            )
        except Exception:
            return cached_file(pretrained_model_name_or_path, filename)

    def _build_glm46v_image_processor(self, hf_model_name: str):
        from transformers.models.glm46v.image_processing_glm46v import (
            Glm46VImageProcessor,
        )

        preprocessor_config_path = self._cached_file(
            hf_model_name,
            "preprocessor_config.json",
        )
        with open(preprocessor_config_path, "r", encoding="utf-8") as f:
            image_processor_config = json.load(f)

        # Processor-level fields are not constructor args for image processor.
        image_processor_config.pop("processor_class", None)
        image_processor_config.pop("image_processor_type", None)

        return Glm46VImageProcessor(**image_processor_config)

    @staticmethod
    def _extract_patch_counts_from_spec(spec) -> list[int]:
        """Extract valid patch counts from vision model `pixel_patches` input."""
        for inp in spec.description.input:
            if inp.name != "pixel_patches":
                continue
            ma = inp.type.multiArrayType

            enum_counts = []
            if len(ma.enumeratedShapes.shapes) > 0:
                for es in ma.enumeratedShapes.shapes:
                    if len(es.shape) >= 1:
                        enum_counts.append(int(es.shape[0]))
                enum_counts = sorted(set(enum_counts))
                if enum_counts:
                    return enum_counts

            static_shape = list(ma.shape)
            if len(static_shape) >= 1 and int(static_shape[0]) > 0:
                return [int(static_shape[0])]

            if len(ma.shapeRange.sizeRanges) >= 1:
                first_dim_range = ma.shapeRange.sizeRanges[0]
                if int(first_dim_range.upperBound) > 0:
                    return [int(first_dim_range.upperBound)]
                if int(first_dim_range.lowerBound) > 0:
                    return [int(first_dim_range.lowerBound)]
        return []

    def _infer_available_vision_patch_counts(self, vision_model_path: Path) -> tuple[int, ...]:
        """Infer vision patch counts from model spec; fallback to known defaults."""
        patch_counts: list[int] = []

        if hasattr(self.vision_model, "get_spec"):
            try:
                patch_counts = self._extract_patch_counts_from_spec(self.vision_model.get_spec())
            except Exception:
                patch_counts = []

        if not patch_counts and vision_model_path.suffix == ".mlpackage" and vision_model_path.exists():
            try:
                spec = ct.utils.load_spec(str(vision_model_path))
                patch_counts = self._extract_patch_counts_from_spec(spec)
            except Exception:
                patch_counts = []

        if not patch_counts:
            shape = self.vision_specs.get("inputs", {}).get("pixel_patches", {}).get("shape", [])
            if len(shape) >= 1 and int(shape[0]) > 0:
                patch_counts = [int(shape[0])]

        if not patch_counts:
            patch_counts = list(ENUMERATED_PATCH_COUNTS)

        return tuple(sorted(set(int(x) for x in patch_counts)))

    def _get_best_vision_patch_count(self, num_real_patches: int) -> int:
        """Pick smallest supported patch count >= real patch count."""
        for count in self.available_vision_patch_counts:
            if count >= num_real_patches:
                return count
        raise ValueError(
            f"num_real_patches={num_real_patches} exceeds max supported patch count "
            f"{self.available_vision_patch_counts[-1]}. "
            f"Supported counts: {list(self.available_vision_patch_counts)}"
        )

    def _init_preprocessing_components(
        self,
        hf_model_name: str,
        verbose: bool = True,
    ) -> None:
        """Initialize preprocessing. Prefer AutoProcessor, fallback to manual GLM46V path."""
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(
                hf_model_name,
                trust_remote_code=True,
            )
            # Keep tokenizer/image_processor handles for decode/introspection.
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            self.image_processor = getattr(self.processor, "image_processor", None)
        except Exception as exc:
            if verbose:
                print("  AutoProcessor loading failed; using manual GLM46V preprocessing fallback.")
                print(f"  Reason: {type(exc).__name__}: {exc}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True,
                )
            self.image_processor = self._build_glm46v_image_processor(hf_model_name)

    def _get_text_seq_len(self) -> int:
        shape = self.text_specs.get("inputs", {}).get("inputs_embeds", {}).get("shape", [])
        if len(shape) == 4:
            return int(shape[3])

        if hasattr(self.text_model, "get_spec"):
            spec = self.text_model.get_spec()
            for inp in spec.description.input:
                if inp.name == "inputs_embeds":
                    spec_shape = list(inp.type.multiArrayType.shape)
                    if len(spec_shape) == 4:
                        return int(spec_shape[3])
        raise ValueError("Could not determine `inputs_embeds` sequence length from text model spec.")

    def _get_text_input_names(self) -> set[str]:
        names = set(self.text_specs.get("inputs", {}).keys())
        if names:
            return names

        if hasattr(self.text_model, "get_spec"):
            try:
                spec = self.text_model.get_spec()
                return {inp.name for inp in spec.description.input}
            except Exception:
                return set()
        return set()

    def _get_cache_length(self) -> int:
        def _cache_len_from_shape(shape: list[int]) -> int | None:
            if len(shape) < 3:
                return None
            try:
                cache_len = int(shape[2])
            except Exception:
                return None
            return cache_len if cache_len > 0 else None

        state_map = self.text_specs.get("states", {})

        # Preferred path: explicit key_cache state in parsed specs.
        key_shape = state_map.get("key_cache", {}).get("shape", [])
        key_len = _cache_len_from_shape(key_shape)
        if key_len is not None:
            return key_len

        # Fallback path: infer from any state that has a cache-like shape.
        parsed_candidates = []
        for name, meta in state_map.items():
            cache_len = _cache_len_from_shape(meta.get("shape", []))
            if cache_len is not None:
                parsed_candidates.append((name, cache_len))
        if parsed_candidates:
            inferred = max(v for _, v in parsed_candidates)
            if self.verbose:
                names = ", ".join(f"{n}:{v}" for n, v in parsed_candidates)
                print(
                    "  key_cache state missing in parsed metadata; "
                    f"inferred cache_length={inferred} from states [{names}]."
                )
            return inferred

        # Final fallback for non-compiled MLModel objects with get_spec().
        if hasattr(self.text_model, "get_spec"):
            try:
                spec = self.text_model.get_spec()
            except Exception:
                spec = None
            if spec is not None:
                spec_candidates = []
                for st in spec.description.state:
                    spec_shape = list(st.type.multiArrayType.shape)
                    cache_len = _cache_len_from_shape(spec_shape)
                    if cache_len is not None:
                        spec_candidates.append((st.name, cache_len))

                for name, cache_len in spec_candidates:
                    if name == "key_cache":
                        return cache_len

                if spec_candidates:
                    inferred = max(v for _, v in spec_candidates)
                    if self.verbose:
                        names = ", ".join(f"{n}:{v}" for n, v in spec_candidates)
                        print(
                            "  key_cache state missing in model spec; "
                            f"inferred cache_length={inferred} from states [{names}]."
                        )
                    return inferred

        # Last-resort fallback to config when state metadata is unavailable.
        cfg_cache_len = getattr(self.config.text_config, "max_position_embeddings", None)
        if cfg_cache_len is not None:
            cfg_cache_len = int(cfg_cache_len)
            if cfg_cache_len > 0:
                if self.verbose:
                    available = list(state_map.keys())
                    print(
                        "  Warning: could not read KV cache length from model states; "
                        f"using config max_position_embeddings={cfg_cache_len}. "
                        f"Parsed states: {available}"
                    )
                return cfg_cache_len

        available = list(state_map.keys())
        raise ValueError(
            "Could not determine KV cache length from text model state spec. "
            f"Parsed states: {available}"
        )

    def _compute_multimodal_position_ids(
        self,
        input_ids: np.ndarray,
        image_grid_thw: np.ndarray,
    ) -> tuple[torch.Tensor, int]:
        """Compute GLM-OCR multimodal 3D position IDs and rope_delta."""
        seq_len = int(input_ids.shape[0])

        # Fallback path when HF reference implementation is unavailable.
        if HFGlmOcrModel is None:
            pos = (
                torch.arange(seq_len, dtype=torch.long)
                .view(1, 1, -1)
                .expand(3, 1, -1)
            )
            return pos, 0

        input_ids_t = torch.from_numpy(input_ids.astype(np.int64)).unsqueeze(0)
        image_grid_thw_t = torch.from_numpy(image_grid_thw.astype(np.int64))
        attention_mask_t = torch.ones_like(input_ids_t)

        position_ids, rope_deltas = HFGlmOcrModel.get_rope_index(
            self._rope_index_helper,
            input_ids=input_ids_t,
            image_grid_thw=image_grid_thw_t,
            video_grid_thw=None,
            attention_mask=attention_mask_t,
        )
        rope_delta = int(rope_deltas[0, 0].item())
        return position_ids.to(dtype=torch.long), rope_delta

    def _compute_chunk_rope_embeddings(
        self,
        position_ids_3d: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute mRoPE cos/sin for a chunk of position IDs."""
        cos, sin = compute_glm_ocr_mrope_cos_sin(
            position_ids=position_ids_3d,
            head_dim=self.text_head_dim,
            rope_theta=self.text_rope_theta,
            partial_rotary_factor=self.text_partial_rotary_factor,
            mrope_section=self.text_mrope_section,
        )
        # squeeze batch dim (always batch=1 in this pipeline)
        cos_np = cos[0].cpu().numpy().astype(np.float32)
        sin_np = sin[0].cpu().numpy().astype(np.float32)
        return cos_np, sin_np

    def _compute_decode_position_ids(
        self,
        cache_position: int,
        seq_len: int,
        rope_delta: int,
    ) -> torch.Tensor:
        """Build 3D position IDs for decode chunk (batch=1)."""
        start = int(cache_position) + int(rope_delta)
        base = torch.arange(seq_len, dtype=torch.long).view(1, 1, -1) + start
        return base.expand(3, 1, -1)

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if self.processor is not None:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="np",
            )
            input_ids = model_inputs["input_ids"][0].astype(np.int32)
            pixel_values = model_inputs["pixel_values"]
            image_grid_thw = model_inputs["image_grid_thw"].astype(np.int64)
            return input_ids, pixel_values, image_grid_thw

        # Manual fallback path mirroring Glm46VProcessor.__call__ for images.
        if self.tokenizer is None or self.image_processor is None:
            raise RuntimeError("Tokenizer/image processor are not initialized.")

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs = self.image_processor(
            images=[image],
            return_tensors="np",
        )
        image_grid_thw = image_inputs["image_grid_thw"].astype(np.int64)

        text_list = [text]
        image_token = getattr(self.tokenizer, "image_token", "<|image|>")
        merge_length = self.image_processor.merge_size ** 2
        index = 0
        for i in range(len(text_list)):
            while image_token in text_list[i]:
                num_image_tokens = int(np.prod(image_grid_thw[index]) // merge_length)
                text_list[i] = text_list[i].replace(
                    image_token,
                    "<|placeholder|>" * num_image_tokens,
                    1,
                )
                index += 1
            text_list[i] = text_list[i].replace("<|placeholder|>", image_token)

        text_inputs = self.tokenizer(
            text_list,
            padding=False,
            return_token_type_ids=False,
            return_tensors="np",
        )

        input_ids = text_inputs["input_ids"][0].astype(np.int32)
        pixel_values = image_inputs["pixel_values"]
        return input_ids, pixel_values, image_grid_thw

    def _prepare_inputs_with_patch_budget(
        self,
        image: Image.Image,
        prompt: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image, bool, int]:
        """Prepare inputs and ensure patch count is within configured limit."""
        current_image = image
        resized = False
        max_attempts = 8

        for _ in range(max_attempts):
            input_ids, pixel_values, image_grid_thw = self._prepare_inputs(current_image, prompt)
            num_real_patches = int(np.prod(image_grid_thw[0]))

            if num_real_patches <= self.max_vision_patches:
                return (
                    input_ids,
                    pixel_values,
                    image_grid_thw,
                    current_image,
                    resized,
                    num_real_patches,
                )

            # Downscale preserving aspect ratio, then retry preprocessing.
            resized = True
            width, height = current_image.size
            # Area-based scale approximation with small safety margin.
            scale = (self.max_vision_patches / float(num_real_patches)) ** 0.5 * 0.98
            new_width = max(28, int(width * scale))
            new_height = max(28, int(height * scale))

            # Force progress if integer rounding keeps same size.
            if new_width == width and new_height == height:
                new_width = max(28, width - 28)
                new_height = max(28, height - 28)
                if new_width == width and new_height == height:
                    break

            current_image = current_image.resize(
                (new_width, new_height),
                Image.Resampling.BICUBIC,
            )

        raise ValueError(
            f"Could not fit image into <= {self.max_vision_patches} patches "
            f"after {max_attempts} resize attempts."
        )

    def _run_vision(
        self,
        pixel_values: np.ndarray,
        image_grid_thw: np.ndarray,
    ) -> np.ndarray:
        grid_thw_t = torch.from_numpy(image_grid_thw)
        num_real_patches = int(grid_thw_t[0].prod().item())
        num_total_patches = self._get_best_vision_patch_count(num_real_patches)

        conv2d_channels = self.vision_in_channels * self.vision_temporal_patch_size
        pixel_values_t = torch.from_numpy(pixel_values).view(
            num_real_patches,
            conv2d_channels,
            self.vision_patch_size,
            self.vision_patch_size,
        )
        if num_total_patches > num_real_patches:
            pad = torch.zeros(
                num_total_patches - num_real_patches,
                conv2d_channels,
                self.vision_patch_size,
                self.vision_patch_size,
            )
            pixel_patches_t = torch.cat([pixel_values_t, pad], dim=0)
        else:
            pixel_patches_t = pixel_values_t

        cos_real, sin_real = compute_vision_rotary_pos_emb(
            grid_thw_t,
            self.vision_head_dim,
            self.vision_spatial_merge_size,
        )
        if num_total_patches > num_real_patches:
            cos_pad = torch.zeros(num_total_patches - num_real_patches, self.vision_head_dim)
            sin_pad = torch.zeros(num_total_patches - num_real_patches, self.vision_head_dim)
            position_cos_t = torch.cat([cos_real, cos_pad], dim=0)
            position_sin_t = torch.cat([sin_real, sin_pad], dim=0)
        else:
            position_cos_t = cos_real
            position_sin_t = sin_real

        attention_mask_t = create_padding_attention_mask(num_real_patches, num_total_patches)
        patch_mask_t = create_patch_mask(num_real_patches, num_total_patches)

        result = self.vision_model.predict(
            {
                "pixel_patches": pixel_patches_t.numpy().astype(np.float32),
                "position_cos": position_cos_t.numpy().astype(np.float32),
                "position_sin": position_sin_t.numpy().astype(np.float32),
                "attention_mask": attention_mask_t.numpy().astype(np.float16),
                "patch_mask": patch_mask_t.numpy().astype(np.float16),
            }
        )
        vision_cf = result["vision_embeddings"]

        num_merged_tokens = num_real_patches // (self.vision_spatial_merge_size ** 2)
        vision_cf = vision_cf[:, :, :, :num_merged_tokens]
        vision_seq = vision_cf[0, :, 0, :].T.astype(np.float32)
        return vision_seq

    def _replace_image_tokens_with_vision_embeds(
        self,
        input_ids: np.ndarray,
        vision_embeds_seq: np.ndarray,
    ) -> np.ndarray:
        inputs_embeds = self.embeddings[input_ids].copy()
        image_positions = np.where(input_ids == self.image_token_id)[0]
        if len(image_positions) != vision_embeds_seq.shape[0]:
            raise ValueError(
                "Image token count does not match extracted vision embeddings: "
                f"{len(image_positions)} != {vision_embeds_seq.shape[0]}"
            )
        inputs_embeds[image_positions] = vision_embeds_seq
        return inputs_embeds

    @staticmethod
    def _pad_embed_chunk(chunk: np.ndarray, seq_len: int) -> tuple[np.ndarray, int]:
        real_len = chunk.shape[0]
        if real_len == 0:
            raise ValueError("Cannot run empty chunk.")
        if real_len > seq_len:
            raise ValueError(f"Chunk length {real_len} exceeds model seq_len {seq_len}.")
        if real_len < seq_len:
            pad = np.repeat(chunk[-1:, :], seq_len - real_len, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk, real_len

    def _forward_text_chunk(
        self,
        embed_chunk_seq: np.ndarray,
        cache_position: int,
        state,
        position_cos_seq: np.ndarray | None = None,
        position_sin_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        chunk_cf = embed_chunk_seq.T[np.newaxis, :, np.newaxis, :].astype(np.float32)
        model_inputs = {
            "inputs_embeds": chunk_cf,
            "position_id": np.array([cache_position], dtype=np.int32),
        }
        if self.text_uses_explicit_rope:
            if position_cos_seq is None or position_sin_seq is None:
                raise ValueError(
                    "Text model expects explicit RoPE inputs, but position_cos/sin were not provided."
                )
            model_inputs["position_cos"] = position_cos_seq.astype(np.float32)
            model_inputs["position_sin"] = position_sin_seq.astype(np.float32)

        try:
            result = self.text_model.predict(model_inputs, state)
        except Exception as exc:
            err = str(exc)
            if (
                not self.text_uses_explicit_rope
                and ("position_cos" in err or "position_sin" in err)
            ):
                raise RuntimeError(
                    "The loaded text CoreML model expects explicit RoPE inputs "
                    "(`position_cos`, `position_sin`), but pipeline detected legacy mode. "
                    "Re-load from .mlpackage or clear stale compiled cache."
                ) from exc
            raise
        return result["output"]

    def _compute_logits(
        self,
        hidden_states_cf: np.ndarray,
        token_index: int,
        temperature: float,
    ) -> np.ndarray:
        temp = np.array([[[[max(temperature, 1e-6)]]]], dtype=np.float32)
        result = self.lm_head.predict({"hidden_states": hidden_states_cf, "temperature": temp})
        logits = result["logits"][0, :, 0, token_index].astype(np.float32)
        return logits

    @staticmethod
    def _sample_token(
        logits: np.ndarray,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        if not do_sample or temperature <= 0.0:
            return int(np.argmax(logits))

        scaled = logits / max(temperature, 1e-6)
        if top_k > 0 and top_k < scaled.shape[0]:
            top_idx = np.argpartition(scaled, -top_k)[-top_k:]
            mask = np.full_like(scaled, -np.inf)
            mask[top_idx] = scaled[top_idx]
            scaled = mask

        if top_p < 1.0:
            sort_idx = np.argsort(scaled)[::-1]
            sort_logits = scaled[sort_idx]
            probs = np.exp(sort_logits - np.max(sort_logits))
            probs = probs / probs.sum()
            cumsum = np.cumsum(probs)
            keep_n = int(np.searchsorted(cumsum, top_p, side="left")) + 1
            keep_idx = sort_idx[:keep_n]
            mask = np.full_like(scaled, -np.inf)
            mask[keep_idx] = scaled[keep_idx]
            scaled = mask

        probs = np.exp(scaled - np.max(scaled))
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    # ================================================================
    # MTP Speculative Decoding Helpers
    # ================================================================

    def _forward_mtp(
        self,
        previous_hidden_states_cf: np.ndarray,
        input_embeds_cf: np.ndarray,
        position_id: int,
        mtp_state,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one MTP forward pass.

        Args:
            previous_hidden_states_cf: Hidden states from main model or previous
                MTP step. Shape: (1, hidden_dim, 1, 1) channels-first.
            input_embeds_cf: Embedding of the predicted token in channels-first.
                Shape: (1, hidden_dim, 1, 1).
            position_id: Current position in MTP's KV cache.
            mtp_state: CoreML state object for MTP model.

        Returns:
            Tuple of (mtp_hidden, mtp_hidden_for_head) numpy arrays.
        """
        # Position-0 masking: zero input_embeds when position_id == 0
        if position_id == 0:
            input_embeds_cf = np.zeros_like(input_embeds_cf)

        result = self.mtp_model.predict(
            {
                "previous_hidden_states": previous_hidden_states_cf,
                "input_embeds": input_embeds_cf,
                "position_id": np.array([position_id], dtype=np.int32),
            },
            mtp_state,
        )
        return result["mtp_hidden"], result["mtp_hidden_for_head"]

    def _embed_token_cf(self, token_id: int) -> np.ndarray:
        """Look up embedding and reshape to channels-first (1, hidden_dim, 1, 1).

        Args:
            token_id: Token ID to embed.

        Returns:
            Channels-first embedding array of shape (1, hidden_dim, 1, 1).
        """
        embed = self.embeddings[token_id]  # (hidden_dim,)
        return embed.reshape(1, self.hidden_dim, 1, 1).astype(np.float32)

    def _draft_tokens_mtp(
        self,
        last_hidden_cf: np.ndarray,
        token_index: int,
        first_token_id: int,
        mtp_state,
        mtp_position: int,
        num_steps: int,
    ) -> tuple[list[int], int, float, int]:
        """Draft multiple tokens using MTP chaining.

        Runs the MTP model num_steps times sequentially, feeding each step's
        hidden output as the next step's previous_hidden_states input.

        Args:
            last_hidden_cf: Main model hidden output, channels-first.
                Shape: (1, hidden_dim, 1, seq_len).
            token_index: Index of the last accepted token in last_hidden_cf.
            first_token_id: Token sampled from main model (starting point).
            mtp_state: CoreML state for MTP model.
            mtp_position: Current position in MTP's KV cache.
            num_steps: Number of MTP forward passes to chain.

        Returns:
            Tuple of (draft_token_ids, new_mtp_position, total_mtp_ms, num_mtp_steps).
        """
        draft_tokens = []
        total_mtp_ms = 0.0
        actual_steps = 0
        # Extract hidden state for the last accepted token position
        prev_hidden = last_hidden_cf[:, :, :, token_index:token_index + 1]
        prev_embed = self._embed_token_cf(first_token_id)

        for _ in range(num_steps):
            t0 = time.perf_counter()
            mtp_hidden, mtp_for_head = self._forward_mtp(
                prev_hidden, prev_embed, mtp_position, mtp_state,
            )
            total_mtp_ms += (time.perf_counter() - t0) * 1000.0
            actual_steps += 1
            mtp_position += 1

            # Pad MTP output to match lm_head's expected seq_len
            if mtp_for_head.shape[-1] < self.text_seq_len:
                mtp_for_head = np.pad(
                    mtp_for_head,
                    ((0, 0), (0, 0), (0, 0), (0, self.text_seq_len - mtp_for_head.shape[-1])),
                    mode="edge",
                )

            # Greedy decode for draft tokens (argmax, no sampling)
            logits = self._compute_logits(mtp_for_head, 0, temperature=1.0)
            draft_token = int(np.argmax(logits))
            draft_tokens.append(draft_token)

            if draft_token in self.eos_token_ids:
                break

            # Chain: MTP output becomes next input
            prev_hidden = mtp_hidden
            prev_embed = self._embed_token_cf(draft_token)

        return draft_tokens, mtp_position, total_mtp_ms, actual_steps

    def _stream_tokens(
        self,
        generated_ids: list[int],
        streamed_text: str,
        tokenizer,
    ) -> str:
        """Stream newly generated tokens to stdout.

        Args:
            generated_ids: All generated token IDs so far.
            streamed_text: Previously streamed text (for delta computation).
            tokenizer: Tokenizer for decoding.

        Returns:
            Updated streamed_text string.
        """
        current_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if current_text.startswith(streamed_text):
            delta = current_text[len(streamed_text):]
        else:
            delta = current_text
        if delta:
            print(delta, end="", flush=True)
        return current_text

    def run(
        self,
        image: Image.Image,
        prompt: str = "Extract all text from this image.",
        generation_config: Optional[GenerationConfig] = None,
        resized_image_output_path: str | Path | None = None,
        stream_output: bool = False,
        debug: bool = False,
    ) -> tuple[str, TimingStats]:
        """Run end-to-end OCR: image preprocessing, vision, prefill, decode.

        Supports two decode modes:
        - Standard (no MTP): one token per main model forward pass.
        - Speculative (with MTP): draft N tokens with MTP chaining, verify
          all in a single batched main model forward pass.

        Returns:
            Tuple of (generated_text, timing_stats).
        """
        cfg = generation_config or GenerationConfig()
        tokenizer = self.tokenizer if self.tokenizer is not None else self.processor.tokenizer
        timing = TimingStats()

        # ============================================================
        # Preprocessing
        # ============================================================
        t0 = time.perf_counter()
        (
            input_ids,
            pixel_values,
            image_grid_thw,
            final_image,
            was_resized,
            patch_count,
        ) = self._prepare_inputs_with_patch_budget(image, prompt)

        if was_resized and resized_image_output_path is not None:
            out_path = Path(resized_image_output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            final_image.save(out_path)
            if self.verbose:
                print(
                    f"  Image resized to fit patch budget ({patch_count}/{self.max_vision_patches}) "
                    f"and saved to: {out_path}"
                )
        timing.preprocessing_ms = (time.perf_counter() - t0) * 1000.0

        # ============================================================
        # Vision encoding
        # ============================================================
        t0 = time.perf_counter()
        vision_embeds_seq = self._run_vision(pixel_values, image_grid_thw)
        merged_embeds_seq = self._replace_image_tokens_with_vision_embeds(input_ids, vision_embeds_seq)
        timing.vision_encode_ms = (time.perf_counter() - t0) * 1000.0

        rope_delta = 0
        prefill_position_cos_seq: np.ndarray | None = None
        prefill_position_sin_seq: np.ndarray | None = None
        if self.text_uses_explicit_rope:
            position_ids_3d, rope_delta = self._compute_multimodal_position_ids(
                input_ids, image_grid_thw
            )
            if int(position_ids_3d.shape[-1]) != int(merged_embeds_seq.shape[0]):
                raise ValueError(
                    "Multimodal position ID length mismatch: "
                    f"{int(position_ids_3d.shape[-1])} vs {int(merged_embeds_seq.shape[0])}"
                )
            prefill_position_cos_seq, prefill_position_sin_seq = (
                self._compute_chunk_rope_embeddings(position_ids_3d)
            )
            if debug:
                print(
                    f"  mRoPE prefill ready: len={prefill_position_cos_seq.shape[0]} "
                    f"rope_delta={rope_delta}"
                )

        # ============================================================
        # Prefill (chunked)
        # ============================================================
        t0 = time.perf_counter()
        state = self.text_model.make_state()
        current_position = 0

        use_mtp = self.mtp_model is not None
        collect_prefill_hidden = use_mtp
        prefill_hidden_chunks: list[tuple[np.ndarray, int]] = []

        last_hidden = None
        last_token_index = 0
        num_prefill_chunks = 0
        for start in range(0, merged_embeds_seq.shape[0], self.text_seq_len):
            raw_chunk = merged_embeds_seq[start : start + self.text_seq_len]
            chunk, real_len = self._pad_embed_chunk(raw_chunk, self.text_seq_len)
            chunk_position_cos = None
            chunk_position_sin = None
            if prefill_position_cos_seq is not None and prefill_position_sin_seq is not None:
                raw_pos_cos = prefill_position_cos_seq[start : start + self.text_seq_len]
                raw_pos_sin = prefill_position_sin_seq[start : start + self.text_seq_len]
                chunk_position_cos, _ = self._pad_embed_chunk(raw_pos_cos, self.text_seq_len)
                chunk_position_sin, _ = self._pad_embed_chunk(raw_pos_sin, self.text_seq_len)
            t_chunk = time.perf_counter()
            hidden = self._forward_text_chunk(
                chunk,
                current_position,
                state,
                position_cos_seq=chunk_position_cos,
                position_sin_seq=chunk_position_sin,
            )
            chunk_ms = (time.perf_counter() - t_chunk) * 1000.0
            num_prefill_chunks += 1
            if debug:
                print(f"  prefill chunk {num_prefill_chunks} pos={current_position} len={real_len}  {chunk_ms:.1f}ms")
            if collect_prefill_hidden:
                prefill_hidden_chunks.append((hidden, real_len))
            current_position += real_len
            last_hidden = hidden
            last_token_index = real_len - 1

        if last_hidden is None:
            raise ValueError("No hidden states produced during prefill.")
        timing.prefill_ms = (time.perf_counter() - t0) * 1000.0

        if current_position + cfg.max_new_tokens > self.cache_length:
            raise ValueError(
                f"Requested generation exceeds cache length: "
                f"prefill={current_position}, max_new_tokens={cfg.max_new_tokens}, cache={self.cache_length}."
            )

        # ============================================================
        # Initialize MTP state if available
        # ============================================================
        mtp_state = None
        mtp_position = 0
        if use_mtp:
            mtp_state = self.mtp_model.make_state()

            # Prefill MTP KV cache with main model's hidden states and
            # embeddings. At each position i, the MTP receives
            # (main_hidden[i], embed(token[i+1])), building up self-attention
            # context that mirrors the actual token sequence.
            if prefill_hidden_chunks:
                t_mtp_pf = time.perf_counter()
                total_prefill_len = merged_embeds_seq.shape[0]
                chunk_start = 0
                for hidden_cf, real_len in prefill_hidden_chunks:
                    for j in range(real_len):
                        abs_pos = chunk_start + j
                        if abs_pos + 1 >= total_prefill_len:
                            break
                        prev_hidden = hidden_cf[:, :, :, j : j + 1].astype(
                            np.float32
                        )
                        next_embed_cf = merged_embeds_seq[abs_pos + 1].reshape(
                            1, self.hidden_dim, 1, 1
                        ).astype(np.float32)
                        self._forward_mtp(
                            prev_hidden, next_embed_cf, mtp_position, mtp_state
                        )
                        mtp_position += 1
                    chunk_start += real_len
                prefill_hidden_chunks.clear()
                timing.mtp_prefill_ms = (
                    time.perf_counter() - t_mtp_pf
                ) * 1000.0
                timing.mtp_prefill_positions = mtp_position
                if debug:
                    avg = timing.mtp_prefill_ms / max(mtp_position, 1)
                    print(
                        f"  MTP prefill: {mtp_position} positions in "
                        f"{timing.mtp_prefill_ms:.1f}ms ({avg:.1f}ms/pos)"
                    )

        # Sampling parameters
        sample_kwargs = dict(
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )
        temp_for_logits = cfg.temperature if cfg.do_sample else 1.0

        # ============================================================
        # Decode loop
        # ============================================================
        t_decode_start = time.perf_counter()
        generated_ids: list[int] = []
        streamed_text = ""
        hit_eos = False

        while len(generated_ids) < cfg.max_new_tokens and not hit_eos:
            # === Step 1: Sample token from main model ===
            logits = self._compute_logits(last_hidden, last_token_index, temp_for_logits)
            token_main = self._sample_token(logits=logits, **sample_kwargs)

            if token_main in self.eos_token_ids:
                break

            generated_ids.append(token_main)
            if stream_output:
                streamed_text = self._stream_tokens(generated_ids, streamed_text, tokenizer)

            # === Standard decode (no MTP) ===
            if not use_mtp:
                token_embed = self.embeddings[np.array([token_main], dtype=np.int32)]
                token_chunk, real_len = self._pad_embed_chunk(token_embed, self.text_seq_len)
                decode_position_cos = None
                decode_position_sin = None
                if self.text_uses_explicit_rope:
                    decode_position_ids = self._compute_decode_position_ids(
                        current_position, self.text_seq_len, rope_delta
                    )
                    decode_position_cos, decode_position_sin = self._compute_chunk_rope_embeddings(
                        decode_position_ids
                    )
                t_fwd = time.perf_counter()
                last_hidden = self._forward_text_chunk(
                    token_chunk,
                    current_position,
                    state,
                    position_cos_seq=decode_position_cos,
                    position_sin_seq=decode_position_sin,
                )
                fwd_ms = (time.perf_counter() - t_fwd) * 1000.0
                if debug:
                    tok_str = tokenizer.decode([token_main])
                    print(f"  [M] {tok_str!r}  |  main {fwd_ms:.1f}ms")
                current_position += real_len
                last_token_index = 0
                continue

            # === Step 2: Draft N tokens with MTP chaining ===
            draft_tokens, mtp_position, draft_ms, draft_steps = self._draft_tokens_mtp(
                last_hidden, last_token_index, token_main,
                mtp_state, mtp_position, self.num_spec_steps,
            )
            timing.mtp_draft_total_ms += draft_ms
            timing.mtp_draft_steps += draft_steps

            if not draft_tokens:
                # No drafts produced (edge case). Fall back to standard decode.
                token_embed = self.embeddings[np.array([token_main], dtype=np.int32)]
                token_chunk, real_len = self._pad_embed_chunk(token_embed, self.text_seq_len)
                decode_position_cos = None
                decode_position_sin = None
                if self.text_uses_explicit_rope:
                    decode_position_ids = self._compute_decode_position_ids(
                        current_position, self.text_seq_len, rope_delta
                    )
                    decode_position_cos, decode_position_sin = self._compute_chunk_rope_embeddings(
                        decode_position_ids
                    )
                last_hidden = self._forward_text_chunk(
                    token_chunk,
                    current_position,
                    state,
                    position_cos_seq=decode_position_cos,
                    position_sin_seq=decode_position_sin,
                )
                current_position += real_len
                last_token_index = 0
                continue

            # === Step 3: Verify all drafts with main model (single batched forward) ===
            # Build embedding sequence: [token_main, draft_0, draft_1, ...]
            all_token_ids = [token_main] + draft_tokens
            all_embeds = np.stack(
                [self.embeddings[tid] for tid in all_token_ids], axis=0
            )  # (num_tokens, hidden_dim)

            embed_chunk, real_len = self._pad_embed_chunk(all_embeds, self.text_seq_len)
            verify_position_cos = None
            verify_position_sin = None
            if self.text_uses_explicit_rope:
                verify_position_ids = self._compute_decode_position_ids(
                    current_position, self.text_seq_len, rope_delta
                )
                verify_position_cos, verify_position_sin = self._compute_chunk_rope_embeddings(
                    verify_position_ids
                )
            t_verify = time.perf_counter()
            hidden = self._forward_text_chunk(
                embed_chunk,
                current_position,
                state,
                position_cos_seq=verify_position_cos,
                position_sin_seq=verify_position_sin,
            )
            verify_ms = (time.perf_counter() - t_verify) * 1000.0
            timing.spec_verify_total_ms += verify_ms
            timing.spec_verify_steps += 1
            timing.drafts_proposed += len(draft_tokens)

            # Verify each draft token against main model's logits.
            # Position i in hidden corresponds to what the main model predicts
            # AFTER seeing tokens up to all_token_ids[i]. So hidden[:,:,:,i]
            # should predict draft_tokens[i] (which is all_token_ids[i+1]).
            #
            # IMPORTANT: On rejection we do NOT append the correction token
            # here. hidden[i+1..] are contaminated by the rejected draft's
            # KV, so we can only trust hidden[0..i]. The correction (or
            # bonus when all match) will be sampled naturally at the start
            # of the next iteration from hidden[num_matching], which is the
            # last position that only attended to verified-correct KV entries.
            num_matching_drafts = 0
            for i, draft in enumerate(draft_tokens):
                verify_logits = self._compute_logits(hidden, i, temp_for_logits)
                verify_token = self._sample_token(logits=verify_logits, **sample_kwargs)

                if verify_token == draft and draft not in self.eos_token_ids:
                    # Draft accepted — matches main model's prediction
                    generated_ids.append(draft)
                    num_matching_drafts += 1
                    if stream_output:
                        streamed_text = self._stream_tokens(
                            generated_ids, streamed_text, tokenizer
                        )
                else:
                    # Draft rejected. The correction will be derived in the
                    # next iteration from hidden[i] (last clean position).
                    if verify_token in self.eos_token_ids:
                        hit_eos = True
                    break

            # When all drafts match, hidden[N] predicts the next token
            # (the "bonus"), which will be sampled in the next iteration.
            # No explicit bonus handling needed — it emerges naturally.

            # === Step 4: Update main model position ===
            # Only advance by token_main + matching drafts. The next token
            # (correction or bonus) has NOT been fed through the main model,
            # so its KV is not in the cache yet. It will be processed in
            # the next iteration when it becomes token_main.
            timing.drafts_accepted += num_matching_drafts

            num_consumed = 1 + num_matching_drafts
            current_position += num_consumed
            last_token_index = num_matching_drafts
            last_hidden = hidden

            # === Step 5: Roll back MTP position for non-matching drafts ===
            num_rejected = len(draft_tokens) - num_matching_drafts
            mtp_position -= num_rejected

            # === Debug output ===
            if debug:
                parts = [f"[M] {tokenizer.decode([token_main])!r}"]
                for d in draft_tokens[:num_matching_drafts]:
                    parts.append(f"[D] {tokenizer.decode([d])!r}")
                gen_str = "  ".join(parts)
                draft_strs = " ".join(
                    f"{tokenizer.decode([d])!r}" for d in draft_tokens
                )
                avg_draft_ms = draft_ms / max(draft_steps, 1)
                print(
                    f"  {gen_str}  |  mtp: [{draft_strs}]"
                    f"  |  verify {verify_ms:.1f}ms  draft {draft_steps}\u00d7{avg_draft_ms:.1f}ms"
                )

        # ============================================================
        # Final decode
        # ============================================================
        timing.decode_ms = (time.perf_counter() - t_decode_start) * 1000.0
        timing.num_tokens_generated = len(generated_ids)

        if not generated_ids:
            return "", timing
        final_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if stream_output and final_text.startswith(streamed_text):
            delta = final_text[len(streamed_text):]
            if delta:
                print(delta, end="", flush=True)
        return final_text, timing
