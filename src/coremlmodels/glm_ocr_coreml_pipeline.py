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
import shutil
from typing import Optional

import numpy as np
import coremltools as ct
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.utils.hub import cached_file

from .vision_model_wrapper import (
    ENUMERATED_PATCH_COUNTS,
    compute_vision_rotary_pos_emb,
    create_padding_attention_mask,
    create_patch_mask,
)


def parse_coremldata_bin(mlmodelc_path: Path) -> dict:
    """Parse coremldata.bin to recover IO/state shapes for CompiledMLModel."""
    coremldata_path = mlmodelc_path / "coremldata.bin"
    if not coremldata_path.exists():
        return {"inputs": {}, "outputs": {}, "states": {}}

    data = coremldata_path.read_bytes()

    def decode_varint(data: bytes, pos: int) -> tuple[int, int]:
        result = 0
        shift = 0
        while pos < len(data):
            byte = data[pos]
            result |= (byte & 0x7F) << shift
            pos += 1
            if not (byte & 0x80):
                break
            shift += 7
        return result, pos

    def decode_shape(data: bytes, start: int, length: int) -> list[int]:
        shape: list[int] = []
        pos = start
        end = start + length
        while pos < end:
            val, pos = decode_varint(data, pos)
            shape.append(val)
        return shape

    def parse_field(data: bytes, pos: int):
        if pos >= len(data):
            return None
        tag, pos = decode_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 0:
            value, pos = decode_varint(data, pos)
        elif wire_type == 2:
            length, pos = decode_varint(data, pos)
            value = data[pos : pos + length]
            pos += length
        elif wire_type == 1:
            value = data[pos : pos + 8]
            pos += 8
        elif wire_type == 5:
            value = data[pos : pos + 4]
            pos += 4
        else:
            return None

        return field_num, wire_type, value, pos

    proto_start = None
    for i in range(len(data) - 10):
        if data[i] == 0x0A and i + 2 < len(data) and data[i + 2] == 0x0A:
            chunk = data[i : i + 64]
            if (
                b"hidden_states" in chunk
                or b"inputs_embeds" in chunk
                or b"pixel_patches" in chunk
            ):
                proto_start = i
                break

    if proto_start is None:
        proto_start = 0x53 if len(data) > 0x53 else 0

    proto_data = data[proto_start:]
    result = {"inputs": {}, "outputs": {}, "states": {}}

    pos = 0
    while pos < len(proto_data) - 5:
        field = parse_field(proto_data, pos)
        if field is None:
            break
        field_num, wire_type, value, new_pos = field
        pos = new_pos

        if field_num not in [1, 10, 13] or wire_type != 2:
            continue

        name = "unknown"
        shape: list[int] = []

        inner_pos = 0
        while inner_pos < len(value):
            inner = parse_field(value, inner_pos)
            if inner is None:
                break
            ifnum, iwtype, ival, inner_new_pos = inner
            inner_pos = inner_new_pos

            if ifnum == 1 and iwtype == 2:
                try:
                    name = ival.decode("utf-8")
                except Exception:
                    pass
            elif ifnum == 3 and iwtype == 2:
                type_pos = 0
                while type_pos < len(ival):
                    type_field = parse_field(ival, type_pos)
                    if type_field is None:
                        break
                    tfnum, twtype, tval, type_new_pos = type_field
                    type_pos = type_new_pos

                    if tfnum == 5 and twtype == 2:
                        arr_pos = 0
                        while arr_pos < len(tval):
                            arr_field = parse_field(tval, arr_pos)
                            if arr_field is None:
                                break
                            afnum, awtype, aval, arr_new_pos = arr_field
                            arr_pos = arr_new_pos
                            if afnum == 1 and awtype == 2:
                                shape = decode_shape(
                                    tval, arr_pos - len(aval), len(aval)
                                )
                    elif tfnum == 8 and twtype == 2:
                        state_pos = 0
                        while state_pos < len(tval):
                            state_field = parse_field(tval, state_pos)
                            if state_field is None:
                                break
                            sfnum, swtype, sval, state_new_pos = state_field
                            state_pos = state_new_pos
                            if sfnum == 1 and swtype == 2:
                                nested_pos = 0
                                while nested_pos < len(sval):
                                    nf = parse_field(sval, nested_pos)
                                    if nf is None:
                                        break
                                    nfnum, nwtype, nval, nested_new_pos = nf
                                    nested_pos = nested_new_pos
                                    if nfnum == 1 and nwtype == 2:
                                        shape = decode_shape(
                                            sval,
                                            nested_pos - len(nval),
                                            len(nval),
                                        )

        if field_num == 1:
            result["inputs"][name] = {"shape": shape}
        elif field_num == 10:
            result["outputs"][name] = {"shape": shape}
        elif field_num == 13:
            result["states"][name] = {"shape": shape}

    return result


def get_compiled_model_path(mlpackage_path: Path) -> Path:
    return mlpackage_path.with_suffix(".mlmodelc")


def compiled_model_exists(mlpackage_path: Path) -> bool:
    compiled_path = get_compiled_model_path(mlpackage_path)
    return compiled_path.exists() and compiled_path.is_dir()


def cache_compiled_model(mlmodel, mlpackage_path: Path, verbose: bool = True) -> Path:
    compiled_path = get_compiled_model_path(mlpackage_path)
    temp_compiled_path = mlmodel.get_compiled_model_path()
    if verbose:
        print(f"  Caching compiled model to: {compiled_path}")
    shutil.copytree(temp_compiled_path, str(compiled_path), dirs_exist_ok=True)
    return compiled_path


def _extract_specs_from_mlmodel(mlmodel) -> dict:
    spec = mlmodel.get_spec()
    specs = {"inputs": {}, "outputs": {}, "states": {}}

    for input_desc in spec.description.input:
        shape = list(input_desc.type.multiArrayType.shape)
        specs["inputs"][input_desc.name] = {"shape": shape}

    for output_desc in spec.description.output:
        shape = list(output_desc.type.multiArrayType.shape)
        specs["outputs"][output_desc.name] = {"shape": shape}

    for state_desc in spec.description.state:
        shape = list(state_desc.type.multiArrayType.shape)
        specs["states"][state_desc.name] = {"shape": shape}

    return specs


def load_model_with_cache(
    mlpackage_path: Path,
    cache_compiled: bool = False,
    verbose: bool = True,
) -> tuple:
    """Load MLModel/CompiledMLModel with shape metadata fallback."""
    compiled_path = get_compiled_model_path(mlpackage_path)

    if compiled_model_exists(mlpackage_path):
        if verbose:
            print(f"  Found cached compiled model: {compiled_path}")
        model = ct.models.CompiledMLModel(str(compiled_path))
        specs = parse_coremldata_bin(compiled_path)
        return model, specs

    if verbose:
        print(f"  Loading from .mlpackage: {mlpackage_path}")
    mlmodel = ct.models.MLModel(str(mlpackage_path))
    specs = _extract_specs_from_mlmodel(mlmodel)

    if cache_compiled:
        cache_compiled_model(mlmodel, mlpackage_path, verbose)

    return mlmodel, specs


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False


class GlmOcrCoreMLPipeline:
    """CoreML-backed GLM-OCR inference pipeline."""

    def __init__(
        self,
        vision_model_path: str | Path,
        text_model_path: str | Path,
        lm_head_path: str | Path,
        embeddings_path: str | Path,
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

    def _get_cache_length(self) -> int:
        shape = self.text_specs.get("states", {}).get("key_cache", {}).get("shape", [])
        if len(shape) >= 3:
            return int(shape[2])

        if hasattr(self.text_model, "get_spec"):
            spec = self.text_model.get_spec()
            for st in spec.description.state:
                if st.name == "key_cache":
                    spec_shape = list(st.type.multiArrayType.shape)
                    if len(spec_shape) >= 3:
                        return int(spec_shape[2])
        raise ValueError("Could not determine KV cache length from `key_cache` state spec.")

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
        position_id: int,
        state,
    ) -> np.ndarray:
        chunk_cf = embed_chunk_seq.T[np.newaxis, :, np.newaxis, :].astype(np.float32)
        result = self.text_model.predict(
            {
                "inputs_embeds": chunk_cf,
                "position_id": np.array([position_id], dtype=np.int32),
            },
            state,
        )
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

    def run(
        self,
        image: Image.Image,
        prompt: str = "Extract all text from this image.",
        generation_config: Optional[GenerationConfig] = None,
        resized_image_output_path: str | Path | None = None,
        stream_output: bool = False,
    ) -> str:
        cfg = generation_config or GenerationConfig()
        tokenizer = self.tokenizer if self.tokenizer is not None else self.processor.tokenizer

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

        vision_embeds_seq = self._run_vision(pixel_values, image_grid_thw)
        merged_embeds_seq = self._replace_image_tokens_with_vision_embeds(input_ids, vision_embeds_seq)

        state = self.text_model.make_state()
        current_position = 0

        last_hidden = None
        last_token_index = 0
        for start in range(0, merged_embeds_seq.shape[0], self.text_seq_len):
            raw_chunk = merged_embeds_seq[start : start + self.text_seq_len]
            chunk, real_len = self._pad_embed_chunk(raw_chunk, self.text_seq_len)
            hidden = self._forward_text_chunk(chunk, current_position, state)
            current_position += real_len
            last_hidden = hidden
            last_token_index = real_len - 1

        if last_hidden is None:
            raise ValueError("No hidden states produced during prefill.")

        if current_position + cfg.max_new_tokens > self.cache_length:
            raise ValueError(
                f"Requested generation exceeds cache length: "
                f"prefill={current_position}, max_new_tokens={cfg.max_new_tokens}, cache={self.cache_length}."
            )

        generated_ids: list[int] = []
        streamed_text = ""
        for _ in range(cfg.max_new_tokens):
            logits = self._compute_logits(last_hidden, last_token_index, cfg.temperature if cfg.do_sample else 1.0)
            next_token = self._sample_token(
                logits=logits,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )
            if next_token in self.eos_token_ids:
                break
            generated_ids.append(next_token)
            if stream_output:
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
                streamed_text = current_text

            token_embed = self.embeddings[np.array([next_token], dtype=np.int32)]
            token_chunk, real_len = self._pad_embed_chunk(token_embed, self.text_seq_len)
            last_hidden = self._forward_text_chunk(token_chunk, current_position, state)
            current_position += real_len
            last_token_index = 0

        if not generated_ids:
            return ""
        final_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if stream_output and final_text.startswith(streamed_text):
            delta = final_text[len(streamed_text):]
            if delta:
                print(delta, end="", flush=True)
        return final_text
