"""CoreML Language Model Inference with Chat Interface.

This script provides a chat interface for CoreML language models converted using
lm_conversion_example.py. It supports:
- Single or chunked model inference
- Prompt processing in chunks
- Top-k and top-p (nucleus) sampling
- Stateful KV cache management
- Token generation using numpy embeddings

Dependencies: numpy, coremltools, transformers (for tokenization only)

Usage:
    # Single model inference
    python examples/inference.py --model-dir ./qwen2_0.5b_seqlen_8.mlpackage --model-name Qwen/Qwen2-0.5B

    # Chunked model inference
    python examples/inference.py --model-dir ./qwen3_4b_chunked_4 --model-name Qwen/Qwen3-4B --chunked

    # With custom sampling parameters
    python examples/inference.py --model-dir ./model --model-name Qwen/Qwen2-0.5B --temperature 0.7 --top-p 0.9 --top-k 40

    # Non-interactive mode (single prompt)
    python examples/inference.py --model-dir ./model --model-name Qwen/Qwen2-0.5B --prompt "What is AI?"

    # Cache compiled models for faster subsequent loads
    python examples/inference.py --model-dir ./model --model-name Qwen/Qwen2-0.5B --cache-compiled
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer


# =============================================================================
# Compiled Model Caching Utilities
# =============================================================================


def parse_coremldata_bin(mlmodelc_path: Path) -> dict:
    """Parse coremldata.bin from a .mlmodelc directory to extract input/output specs.

    The coremldata.bin file contains a header followed by protobuf-encoded
    input/output/state descriptors. This allows extracting input shapes from
    CompiledMLModel which doesn't have a get_spec() method.

    Args:
        mlmodelc_path: Path to the .mlmodelc directory

    Returns:
        Dict with 'inputs', 'outputs', 'states' keys, each containing
        a dict mapping name -> {'shape': [...]}
    """
    coremldata_path = mlmodelc_path / "coremldata.bin"
    if not coremldata_path.exists():
        return {'inputs': {}, 'outputs': {}, 'states': {}}

    data = coremldata_path.read_bytes()

    def decode_varint(data: bytes, pos: int) -> tuple:
        """Decode a protobuf varint, return (value, new_pos)."""
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

    def decode_shape(data: bytes, start: int, length: int) -> list:
        """Decode shape as sequence of varints."""
        shape = []
        pos = start
        end = start + length
        while pos < end:
            val, pos = decode_varint(data, pos)
            shape.append(val)
        return shape

    def parse_field(data: bytes, pos: int):
        """Parse one protobuf field, return (field_num, wire_type, value, new_pos) or None."""
        if pos >= len(data):
            return None
        tag, pos = decode_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 0:  # Varint
            value, pos = decode_varint(data, pos)
        elif wire_type == 2:  # Length-delimited
            length, pos = decode_varint(data, pos)
            value = data[pos:pos + length]
            pos += length
        elif wire_type == 1:  # 64-bit
            value = data[pos:pos + 8]
            pos += 8
        elif wire_type == 5:  # 32-bit
            value = data[pos:pos + 4]
            pos += 4
        else:
            return None  # Unknown wire type

        return field_num, wire_type, value, pos

    # Find protobuf start (look for 0x0a followed by nested 0x0a pattern)
    proto_start = None
    for i in range(len(data) - 10):
        if data[i] == 0x0A and i + 2 < len(data) and data[i + 2] == 0x0A:
            # Check for common input names nearby
            chunk = data[i:i + 50]
            if b'hidden_states' in chunk or b'inputs_embeds' in chunk or b'input' in chunk:
                proto_start = i
                break

    if proto_start is None:
        # Fallback: try common offset
        proto_start = 0x53 if len(data) > 0x53 else 0

    proto_data = data[proto_start:]
    result = {'inputs': {}, 'outputs': {}, 'states': {}}

    # Parse top-level fields
    pos = 0
    while pos < len(proto_data) - 5:
        field = parse_field(proto_data, pos)
        if field is None:
            break
        field_num, wire_type, value, new_pos = field
        pos = new_pos

        if field_num not in [1, 10, 13] or wire_type != 2:
            continue

        # Parse feature description
        name = "unknown"
        shape = []

        inner_pos = 0
        while inner_pos < len(value):
            inner = parse_field(value, inner_pos)
            if inner is None:
                break
            ifnum, iwtype, ival, inner_new_pos = inner
            inner_pos = inner_new_pos

            if ifnum == 1 and iwtype == 2:  # name
                try:
                    name = ival.decode('utf-8')
                except Exception:
                    pass
            elif ifnum == 3 and iwtype == 2:  # type
                # Parse type info to extract shape
                type_pos = 0
                while type_pos < len(ival):
                    type_field = parse_field(ival, type_pos)
                    if type_field is None:
                        break
                    tfnum, twtype, tval, type_new_pos = type_field
                    type_pos = type_new_pos

                    if tfnum == 5 and twtype == 2:  # ArrayFeatureType (tensor)
                        arr_pos = 0
                        while arr_pos < len(tval):
                            arr_field = parse_field(tval, arr_pos)
                            if arr_field is None:
                                break
                            afnum, awtype, aval, arr_new_pos = arr_field
                            arr_pos = arr_new_pos
                            if afnum == 1 and awtype == 2:  # shape
                                shape = decode_shape(tval, arr_pos - len(aval), len(aval))

                    elif tfnum == 8 and twtype == 2:  # StateType (wrapped array)
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
                                        shape = decode_shape(sval, nested_pos - len(nval), len(nval))

        # Store in appropriate category
        if field_num == 1:
            result['inputs'][name] = {'shape': shape}
        elif field_num == 10:
            result['outputs'][name] = {'shape': shape}
        elif field_num == 13:
            result['states'][name] = {'shape': shape}

    return result


def get_compiled_model_path(mlpackage_path: Path) -> Path:
    """Get the compiled model path (.mlmodelc) corresponding to an .mlpackage.

    The compiled model is stored in the same directory as the .mlpackage,
    with the same name but .mlmodelc extension.

    Args:
        mlpackage_path: Path to the .mlpackage file/directory

    Returns:
        Path to the corresponding .mlmodelc directory
    """
    return mlpackage_path.with_suffix(".mlmodelc")


def compiled_model_exists(mlpackage_path: Path) -> bool:
    """Check if a cached compiled model exists for the given .mlpackage.

    Args:
        mlpackage_path: Path to the .mlpackage file/directory

    Returns:
        True if the .mlmodelc directory exists
    """
    compiled_path = get_compiled_model_path(mlpackage_path)
    return compiled_path.exists() and compiled_path.is_dir()


def cache_compiled_model(mlmodel, mlpackage_path: Path, verbose: bool = True) -> Path:
    """Cache the compiled model (.mlmodelc) to the same directory as the .mlpackage.

    CoreML compiles models to a temporary location. This function copies
    the compiled model to a persistent location for faster subsequent loads.

    Args:
        mlmodel: The loaded MLModel object
        mlpackage_path: Path to the .mlpackage file/directory
        verbose: Print progress information

    Returns:
        Path to the cached .mlmodelc directory
    """
    compiled_path = get_compiled_model_path(mlpackage_path)
    temp_compiled_path = mlmodel.get_compiled_model_path()

    if verbose:
        print(f"  Caching compiled model to: {compiled_path}")

    # Copy the compiled model directory
    shutil.copytree(temp_compiled_path, str(compiled_path), dirs_exist_ok=True)

    return compiled_path


def load_model_with_cache(
    mlpackage_path: Path,
    cache_compiled: bool = False,
    verbose: bool = True,
) -> tuple:
    """Load a CoreML model, using cached compiled model if available.

    If a cached .mlmodelc exists, loads it directly using CompiledMLModel
    for faster initialization. Otherwise, loads the .mlpackage and optionally
    caches the compiled model for future use.

    Args:
        mlpackage_path: Path to the .mlpackage file/directory
        cache_compiled: If True, cache the compiled model after loading
        verbose: Print progress information

    Returns:
        Tuple of (model, specs_dict) where specs_dict contains 'inputs', 'outputs', 'states'
    """
    compiled_path = get_compiled_model_path(mlpackage_path)

    # Check if cached compiled model exists
    if compiled_model_exists(mlpackage_path):
        if verbose:
            print(f"  Found cached compiled model: {compiled_path}")
        model = ct.models.CompiledMLModel(str(compiled_path))
        # Parse specs from coremldata.bin since CompiledMLModel lacks get_spec()
        specs = parse_coremldata_bin(compiled_path)
        return model, specs

    # Load from .mlpackage
    if verbose:
        print(f"  Loading from .mlpackage: {mlpackage_path}")
    mlmodel = ct.models.MLModel(str(mlpackage_path))

    # Extract specs from MLModel
    spec = mlmodel.get_spec()
    specs = {'inputs': {}, 'outputs': {}, 'states': {}}

    for input_desc in spec.description.input:
        shape = list(input_desc.type.multiArrayType.shape)
        specs['inputs'][input_desc.name] = {'shape': shape}

    for output_desc in spec.description.output:
        shape = list(output_desc.type.multiArrayType.shape)
        specs['outputs'][output_desc.name] = {'shape': shape}

    for state_desc in spec.description.state:
        shape = list(state_desc.type.multiArrayType.shape)
        specs['states'][state_desc.name] = {'shape': shape}

    # Cache compiled model if requested
    if cache_compiled:
        cache_compiled_model(mlmodel, mlpackage_path, verbose)

    return mlmodel, specs


# =============================================================================
# Model Loading
# =============================================================================


class CoreMLLanguageModel:
    """Wrapper for CoreML language model inference."""

    def __init__(
        self,
        model_dir: str,
        embeddings_path: str,
        lm_head_path: str,
        is_chunked: bool = False,
        num_chunks: int = 1,
        max_context_length: Optional[int] = None,
        cache_compiled: bool = False,
        verbose: bool = True,
    ):
        """Initialize CoreML language model.

        Args:
            model_dir: Directory containing model files (single .mlpackage or chunked directory)
            embeddings_path: Path to embeddings .npy file
            lm_head_path: Path to LM head .mlpackage
            is_chunked: Whether the model is chunked
            num_chunks: Number of chunks (if chunked)
            max_context_length: Maximum context length (KV cache size)
            cache_compiled: Cache compiled models (.mlmodelc) for faster subsequent loads
            verbose: Print loading information
        """
        self.model_dir = Path(model_dir)
        self.is_chunked = is_chunked
        self.num_chunks = num_chunks
        self.max_context_length = max_context_length
        self.cache_compiled = cache_compiled
        self.verbose = verbose

        # Load embeddings
        if verbose:
            print(f"Loading embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.vocab_size, self.hidden_dim = self.embeddings.shape
        if verbose:
            print(f"  Vocab size: {self.vocab_size}")
            print(f"  Hidden dim: {self.hidden_dim}")

        # Load model(s)
        model_specs = None  # Will store specs from first model for extracting shapes
        if is_chunked:
            if verbose:
                print(f"Loading {num_chunks} model chunks...")
            self.models = []
            self.states = []
            for i in range(num_chunks):
                chunk_path = self.model_dir / f"chunk_{i}.mlpackage"
                if verbose:
                    print(f"  Loading chunk {i}:")
                model, specs = load_model_with_cache(chunk_path, cache_compiled, verbose)
                self.models.append(model)
                self.states.append(model.make_state())
                if model_specs is None:
                    model_specs = specs
        else:
            if verbose:
                print(f"Loading model:")
            model, specs = load_model_with_cache(Path(model_dir), cache_compiled, verbose)
            self.models = [model]
            self.states = [self.models[0].make_state()]
            model_specs = specs

        # Get sequence length from model input spec
        # Find the inputs_embeds or hidden_states input
        self.seq_len = None
        for input_name in ["inputs_embeds", "hidden_states"]:
            if input_name in model_specs['inputs']:
                # Shape is (batch, hidden_dim, 1, seq_len)
                shape = model_specs['inputs'][input_name]['shape']
                if len(shape) >= 4:
                    self.seq_len = shape[-1]
                    break

        if self.seq_len is None:
            raise ValueError("Could not determine sequence length from model spec")

        # Auto-detect max context length from KV cache state if not provided
        if max_context_length is None:
            # Look for state tensors (KV cache) to determine max context length
            # State shape is typically (num_layers, num_heads, max_context, head_dim)
            detected_context_length = None
            for state_name, state_info in model_specs['states'].items():
                state_shape = state_info['shape']
                # KV cache typically has shape with max_context as one dimension
                # Common patterns: [num_layers, num_heads, max_ctx, head_dim]
                # The max_context is usually the largest dimension
                if len(state_shape) >= 3:
                    # Find the dimension that's likely max_context (not small head dims or num_heads)
                    for dim in state_shape[1:]:
                        if dim > 64:  # Likely not head_dim or num_heads
                            detected_context_length = dim
                            break
                if detected_context_length:
                    break

            if detected_context_length:
                self.max_context_length = detected_context_length
                if verbose:
                    print(f"  Auto-detected max context length: {self.max_context_length}")
            else:
                self.max_context_length = 2048  # Fallback default
                if verbose:
                    print(f"  Max context length (default): {self.max_context_length}")
        else:
            self.max_context_length = max_context_length
            if verbose:
                print(f"  Max context length: {self.max_context_length}")

        if verbose:
            print(f"  Sequence length: {self.seq_len}")

        # Load LM head
        if verbose:
            print(f"Loading LM head:")
        self.lm_head, _ = load_model_with_cache(Path(lm_head_path), cache_compiled, verbose)
        if verbose:
            print("Model loading complete!")

        # Track current position in KV cache
        self.current_position = 0

    @property
    def remaining_context(self) -> int:
        """Return remaining context capacity."""
        return self.max_context_length - self.current_position

    @property
    def context_usage_percent(self) -> float:
        """Return context usage as percentage."""
        return (self.current_position / self.max_context_length) * 100

    def get_context_info(self) -> dict:
        """Return context usage information."""
        return {
            "current_position": self.current_position,
            "max_context_length": self.max_context_length,
            "remaining": self.remaining_context,
            "usage_percent": self.context_usage_percent,
        }

    def reset_state(self):
        """Reset KV cache state to initial values."""
        for i, model in enumerate(self.models):
            self.states[i] = model.make_state()
        self.current_position = 0

    def embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        """Convert token IDs to embeddings.

        Args:
            token_ids: Token IDs of shape (seq_len,)

        Returns:
            Embeddings of shape (1, hidden_dim, 1, seq_len) in channels-first format
        """
        # Look up embeddings: (seq_len, hidden_dim)
        embeds = self.embeddings[token_ids]

        # Reshape to channels-first: (1, hidden_dim, 1, seq_len)
        embeds = embeds.T[np.newaxis, :, np.newaxis, :]

        return embeds.astype(np.float32)

    def forward(self, token_ids: np.ndarray) -> tuple[np.ndarray, int]:
        """Run forward pass through the model.

        Args:
            token_ids: Token IDs of shape (seq_len,) where seq_len <= self.seq_len

        Returns:
            Tuple of (hidden_states, num_real_tokens) where:
                - hidden_states: shape (1, hidden_dim, 1, self.seq_len) - padded to fixed size
                - num_real_tokens: number of real (non-padded) tokens
        """
        num_real_tokens = len(token_ids)
        if num_real_tokens > self.seq_len:
            raise ValueError(
                f"Input sequence length {num_real_tokens} exceeds model limit {self.seq_len}"
            )

        # Pad token_ids to fixed sequence length if needed
        if num_real_tokens < self.seq_len:
            # Pad with the last token - the padding won't affect output due to causal masking
            padding = np.full(
                self.seq_len - num_real_tokens, token_ids[-1], dtype=np.int32
            )
            token_ids = np.concatenate([token_ids, padding])

        # Convert tokens to embeddings
        inputs_embeds = self.embed_tokens(token_ids)

        # Run through model(s)
        current_hidden = inputs_embeds

        if self.is_chunked:
            # Chain through all chunks
            input_name = "hidden_states"
            for chunk_idx, (model, state) in enumerate(zip(self.models, self.states)):
                result = model.predict(
                    {
                        input_name: current_hidden,
                        "position_id": np.array(
                            [self.current_position], dtype=np.int32
                        ),
                    },
                    state,
                )
                current_hidden = result["output"]
        else:
            # Single model
            result = self.models[0].predict(
                {
                    "inputs_embeds": current_hidden,
                    "position_id": np.array([self.current_position], dtype=np.int32),
                },
                self.states[0],
            )
            current_hidden = result["output"]

        # Update position by number of real tokens only
        self.current_position += num_real_tokens

        return current_hidden, num_real_tokens

    def compute_logits(
        self, hidden_states: np.ndarray, token_index: int, temperature: float = 1.0
    ) -> tuple[np.ndarray, float]:
        """Compute logits from hidden states using LM head.

        Args:
            hidden_states: Hidden states of shape (1, hidden_dim, 1, seq_len)
            token_index: Index of token position to extract logits for (0-indexed)
            temperature: Temperature for scaling logits

        Returns:
            Tuple of (logits, log_sum_exp) where:
                - logits: shape (vocab_size,) for the specified token position
                - log_sum_exp: scalar, the log normalizer for computing probabilities
        """
        # Prepare temperature input
        temp_input = np.array([[[[temperature]]]], dtype=np.float32)

        # Run LM head with full hidden states (seq_len input required)
        result = self.lm_head.predict(
            {
                "hidden_states": hidden_states,
                "temperature": temp_input,
            }
        )

        # Extract logits at the specified token position
        # Output shape: (1, vocab_size, 1, seq_len)
        logits = result["logits"][0, :, 0, token_index]  # (vocab_size,)

        # Get log_sum_exp chunks and max values at the specified position
        # Shapes: (1, num_chunks, 1, seq_len)
        lse_chunks = result["chunk_logsumexp_stable"][
            0, :, 0, token_index
        ]  # (num_chunks,)
        max_chunks = result["chunk_max"][0, :, 0, token_index]  # (num_chunks,)

        # Combine log_sum_exp chunks using stable log-sum-exp reduction
        # LSE(a, b) = max(a, b) + log(exp(a - max) + exp(b - max))
        # First, find the global max across all chunks
        global_max = np.max(max_chunks)

        # Compute the combined log_sum_exp
        # Each chunk's contribution: lse_chunk + max_chunk (to get the true lse)
        # Then reduce: global_max + log(sum(exp(true_lse - global_max)))
        true_lse_chunks = lse_chunks + max_chunks  # (num_chunks,)
        log_sum_exp = global_max + np.log(np.sum(np.exp(true_lse_chunks - global_max)))

        return logits, log_sum_exp


# =============================================================================
# Sampling Functions
# =============================================================================


def logits_to_probs(logits: np.ndarray, log_sum_exp: float) -> np.ndarray:
    """Convert logits to probabilities using precomputed log_sum_exp.

    Args:
        logits: Logits of shape (vocab_size,)
        log_sum_exp: Precomputed log normalizer

    Returns:
        Probabilities of shape (vocab_size,)
    """
    # p_i = exp(logit_i - log_sum_exp)
    return np.exp(logits - log_sum_exp)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax with numerical stability."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def top_k_filtering(logits: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Filter logits to keep only top-k values.

    Args:
        logits: Logits of shape (vocab_size,)
        top_k: Number of top values to keep

    Returns:
        Tuple of (filtered_logits, top_k_indices) where:
            - filtered_logits: top-k logits (shape: top_k,)
            - top_k_indices: indices of top-k values (shape: top_k,)
    """
    if top_k <= 0 or top_k >= len(logits):
        return logits, np.arange(len(logits))

    # Get indices of top-k values
    top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
    filtered_logits = logits[top_k_indices]

    return filtered_logits, top_k_indices


def top_p_filtering(
    logits: np.ndarray, log_sum_exp: float, top_p: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Filter logits using nucleus (top-p) sampling.

    Args:
        logits: Logits of shape (vocab_size,)
        log_sum_exp: Precomputed log normalizer
        top_p: Cumulative probability threshold

    Returns:
        Tuple of (filtered_logits, indices, new_log_sum_exp) where:
            - filtered_logits: logits within top-p (variable size)
            - indices: original indices of kept logits
            - new_log_sum_exp: recomputed log normalizer for filtered logits
    """
    if top_p >= 1.0:
        return logits, np.arange(len(logits)), log_sum_exp

    # Sort logits in descending order
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]

    # Compute cumulative probabilities using log_sum_exp
    probs = np.exp(sorted_logits - log_sum_exp)
    cumulative_probs = np.cumsum(probs)

    # Find cutoff index
    cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1

    # Keep only tokens within top-p
    kept_indices = sorted_indices[:cutoff_idx]
    filtered_logits = logits[kept_indices]

    # Recompute log_sum_exp for filtered logits (stable)
    max_filtered = np.max(filtered_logits)
    new_log_sum_exp = max_filtered + np.log(
        np.sum(np.exp(filtered_logits - max_filtered))
    )

    return filtered_logits, kept_indices, new_log_sum_exp


def sample_token(
    logits: np.ndarray,
    log_sum_exp: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits using top-k and top-p filtering.

    Args:
        logits: Logits of shape (vocab_size,)
        log_sum_exp: Precomputed log normalizer from LM head
        top_k: Number of top values to consider (0 = no filtering)
        top_p: Cumulative probability threshold (1.0 = no filtering)

    Returns:
        Sampled token ID
    """
    indices = np.arange(len(logits))

    # Apply top-k filtering first
    if top_k > 0:
        logits, top_k_indices = top_k_filtering(logits, top_k)
        indices = top_k_indices
        # Recompute log_sum_exp for top-k logits
        max_logits = np.max(logits)
        log_sum_exp = max_logits + np.log(np.sum(np.exp(logits - max_logits)))

    # Apply top-p filtering
    if top_p < 1.0:
        logits, top_p_indices, log_sum_exp = top_p_filtering(logits, log_sum_exp, top_p)
        indices = indices[top_p_indices] if top_k > 0 else top_p_indices

    # Convert to probabilities using log_sum_exp
    probs = logits_to_probs(logits, log_sum_exp)

    # Sample from filtered distribution
    sampled_idx = np.random.choice(len(probs), p=probs)

    # Map back to original token ID
    token_id = indices[sampled_idx]

    return token_id


# =============================================================================
# Generation
# =============================================================================


def generate(
    model: CoreMLLanguageModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    verbose: bool = False,
) -> str:
    """Generate text from a prompt.

    Args:
        model: CoreML language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        verbose: Print generation details

    Returns:
        Generated text
    """
    # Debug: print the formatted prompt
    print("=" * 40)
    print("INPUT PROMPT (chat template format):")
    print("=" * 40)
    print(prompt)
    print("=" * 40)

    # Timing accumulators
    prompt_forward_time = 0.0
    gen_forward_time = 0.0
    lm_head_time = 0.0
    sampling_time = 0.0

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_length = len(input_ids)
    if verbose:
        print(f"Prompt tokens: {prompt_length}")

    # Check if prompt would overflow the cache
    if prompt_length > model.remaining_context:
        print()
        print("=" * 50)
        print("ERROR: KV cache overflow!")
        print(f"  Prompt length: {prompt_length} tokens")
        print(f"  Remaining cache: {model.remaining_context} tokens")
        print(f"  Current position: {model.current_position} / {model.max_context_length}")
        print("Use /reset to clear the conversation and start fresh.")
        print("=" * 50)
        return ""

    # Process prompt in chunks
    num_chunks = (prompt_length + model.seq_len - 1) // model.seq_len

    if verbose and num_chunks > 1:
        print(f"Processing prompt in {num_chunks} chunks...")

    # Process all prompt chunks and keep the last hidden state
    last_hidden_states = None
    last_chunk_len = 0

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * model.seq_len
        end_idx = min(start_idx + model.seq_len, prompt_length)
        chunk_ids = input_ids[start_idx:end_idx]
        last_chunk_len = len(chunk_ids)

        if verbose:
            print(
                f"  Chunk {chunk_idx + 1}/{num_chunks}: tokens {start_idx} to {end_idx}"
            )

        # Run forward pass (updates KV cache)
        t0 = time.perf_counter()
        last_hidden_states, _ = model.forward(np.array(chunk_ids, dtype=np.int32))
        prompt_forward_time += time.perf_counter() - t0

    # Sample the first token using the last hidden state from prompt processing
    # Index with (last_chunk_len - 1) to get the hidden state of the last real token
    t0 = time.perf_counter()
    logits, log_sum_exp = model.compute_logits(
        last_hidden_states, token_index=last_chunk_len - 1, temperature=temperature
    )
    lm_head_time += time.perf_counter() - t0

    t0 = time.perf_counter()
    next_token = sample_token(logits, log_sum_exp, top_k=top_k, top_p=top_p)
    sampling_time += time.perf_counter() - t0

    # Generate tokens one by one
    generated_ids = [next_token]

    # Check for EOS token
    if next_token == tokenizer.eos_token_id:
        if verbose:
            print("EOS token generated at position 0")
        _print_timing_stats(
            prompt_length,
            len(generated_ids),
            prompt_forward_time,
            gen_forward_time,
            lm_head_time,
            sampling_time,
            model.get_context_info(),
        )
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Print first token in real-time
    print(tokenizer.decode([next_token]), end="", flush=True)

    if verbose:
        print("Generating tokens...")

    for i in range(1, max_new_tokens):
        # Check if cache is about to overflow
        if model.remaining_context <= 0:
            print()
            print("[Cache full - stopping generation]")
            break

        # Use the last generated token
        current_token = generated_ids[-1]

        # Run forward pass with single token (will be padded to seq_len internally)
        # KV cache position is tracked internally and only incremented by 1
        t0 = time.perf_counter()
        hidden_states, num_real = model.forward(
            np.array([current_token], dtype=np.int32)
        )
        gen_forward_time += time.perf_counter() - t0

        # Compute logits at position 0 (the real token, before padding)
        t0 = time.perf_counter()
        logits, log_sum_exp = model.compute_logits(
            hidden_states, token_index=num_real - 1, temperature=temperature
        )
        lm_head_time += time.perf_counter() - t0

        # Sample next token
        t0 = time.perf_counter()
        next_token = sample_token(logits, log_sum_exp, top_k=top_k, top_p=top_p)
        sampling_time += time.perf_counter() - t0

        generated_ids.append(next_token)

        # Check for EOS token
        if next_token == tokenizer.eos_token_id:
            if verbose:
                print(f"EOS token generated at position {i}")
            break

        # Print token in real-time
        print(tokenizer.decode([next_token]), end="", flush=True)

    print()  # Newline after generation

    # Print timing statistics with context info
    _print_timing_stats(
        prompt_length,
        len(generated_ids),
        prompt_forward_time,
        gen_forward_time,
        lm_head_time,
        sampling_time,
        model.get_context_info(),
    )

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


def _print_timing_stats(
    prompt_tokens: int,
    generated_tokens: int,
    prompt_forward_time: float,
    gen_forward_time: float,
    lm_head_time: float,
    sampling_time: float,
    context_info: dict,
):
    """Print timing statistics after generation."""
    print()
    print("-" * 50)
    print("TIMING STATISTICS")
    print("-" * 50)

    # Prompt processing stats
    if prompt_forward_time > 0:
        prompt_tok_per_sec = prompt_tokens / prompt_forward_time
        print("Prompt processing:")
        print(f"  Tokens: {prompt_tokens}")
        print(f"  Model forward time: {prompt_forward_time * 1000:.2f} ms")
        print(f"  Speed: {prompt_tok_per_sec:.2f} tokens/sec")
    print()

    # Generation stats
    total_gen_time = gen_forward_time + lm_head_time + sampling_time
    if total_gen_time > 0 and generated_tokens > 0:
        gen_tok_per_sec = generated_tokens / total_gen_time
        print("Token generation:")
        print(f"  Tokens generated: {generated_tokens}")
        print(
            f"  Model forward time: {gen_forward_time * 1000:.2f} ms "
            f"({gen_forward_time / generated_tokens * 1000:.2f} ms/token)"
        )
        print(
            f"  LM head time: {lm_head_time * 1000:.2f} ms "
            f"({lm_head_time / generated_tokens * 1000:.2f} ms/token)"
        )
        print(
            f"  Sampling time: {sampling_time * 1000:.2f} ms "
            f"({sampling_time / generated_tokens * 1000:.2f} ms/token)"
        )
        print(f"  Total generation time: {total_gen_time * 1000:.2f} ms")
        print(f"  Speed: {gen_tok_per_sec:.2f} tokens/sec")
    print()

    # Context/KV cache usage
    usage_pct = context_info["usage_percent"]
    current = context_info["current_position"]
    max_ctx = context_info["max_context_length"]
    remaining = context_info["remaining"]

    # Color coding for usage level (using ANSI escape codes)
    if usage_pct >= 90:
        status = "CRITICAL"
    elif usage_pct >= 75:
        status = "WARNING"
    else:
        status = "OK"

    print("KV Cache usage:")
    print(f"  Position: {current} / {max_ctx} ({usage_pct:.1f}%) [{status}]")
    print(f"  Remaining capacity: {remaining} tokens")
    if usage_pct >= 90:
        print("  ⚠️  Cache nearly full! Use /reset to clear conversation.")
    elif usage_pct >= 75:
        print("  ⚠️  Cache filling up. Consider resetting soon.")
    print("-" * 50)


def chat_loop(
    model: CoreMLLanguageModel,
    tokenizer,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    max_new_tokens: int = 100,
    system_prompt: Optional[str] = None,
):
    """Interactive chat loop with KV cache management.

    Args:
        model: CoreML language model
        tokenizer: Tokenizer
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        max_new_tokens: Maximum tokens to generate per response
        system_prompt: Optional system prompt to prepend
    """
    print("=" * 60)
    print("CoreML Language Model Chat Interface")
    print("=" * 60)
    print("Commands:")
    print("  /reset  - Reset conversation history and KV cache")
    print("  /quit   - Exit chat")
    print("  /help   - Show this help message")
    print("=" * 60)

    # If no system prompt provided via CLI, ask the user for one
    if system_prompt is None:
        print()
        print("No system prompt provided via --system-prompt.")
        print("Enter a system prompt below, or press Enter to skip (no system prompt):")
        print("-" * 40)
        try:
            user_system_prompt = input("System prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        print("-" * 40)

        if user_system_prompt:
            system_prompt = user_system_prompt
            print(
                f"Using system prompt: {system_prompt[:50]}"
                f"{'...' if len(system_prompt) > 50 else ''}"
            )
        else:
            print("No system prompt set.")
        print()

    # Full conversation history (for reference)
    conversation_history = []
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})

    # New messages to be tokenized and processed (starts with system prompt if any)
    new_messages = []
    if system_prompt:
        new_messages.append({"role": "system", "content": system_prompt})

    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            command = user_input[1:].lower()

            if command == "quit":
                print("Goodbye!")
                break
            elif command == "reset":
                print("Resetting conversation and KV cache...")
                conversation_history = []
                new_messages = []
                if system_prompt:
                    conversation_history.append(
                        {"role": "system", "content": system_prompt}
                    )
                    new_messages.append({"role": "system", "content": system_prompt})
                model.reset_state()
                print(f"[Cache cleared. Capacity: {model.max_context_length} tokens]")
                continue
            elif command == "help":
                print("Commands:")
                print("  /reset  - Reset conversation history and KV cache")
                print("  /quit   - Exit chat")
                print("  /help   - Show this help message")
                continue
            else:
                print(f"Unknown command: {command}")
                continue

        # Add user message to both histories
        conversation_history.append({"role": "user", "content": user_input})
        new_messages.append({"role": "user", "content": user_input})

        # Tokenize only the new messages
        new_prompt = format_chat_prompt(new_messages, tokenizer)
        new_tokens = tokenizer.encode(new_prompt, add_special_tokens=False)

        # Check if new tokens would exceed remaining cache capacity
        if len(new_tokens) > model.remaining_context:
            print()
            print("=" * 60)
            print("CONTEXT LIMIT REACHED")
            print(f"  New message requires: {len(new_tokens)} tokens")
            print(f"  Remaining capacity: {model.remaining_context} tokens")
            print(f"  Cache position: {model.current_position} / {model.max_context_length}")
            print()
            print("Starting new conversation...")
            print("=" * 60)
            # Reset everything
            model.reset_state()
            conversation_history = []
            new_messages = []
            if system_prompt:
                conversation_history.append({"role": "system", "content": system_prompt})
                new_messages.append({"role": "system", "content": system_prompt})
            # Re-add the user message that triggered the overflow
            conversation_history.append({"role": "user", "content": user_input})
            new_messages.append({"role": "user", "content": user_input})
            # Re-tokenize
            new_prompt = format_chat_prompt(new_messages, tokenizer)
            new_tokens = tokenizer.encode(new_prompt, add_special_tokens=False)

        # Debug info
        print(f"[Processing {len(new_tokens)} tokens, "
              f"cache: {model.current_position}/{model.max_context_length}]")

        # Generate response
        print("Assistant: ", end="", flush=True)
        response, cache_overflow = generate_incremental(
            model=model,
            tokenizer=tokenizer,
            new_token_ids=new_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            verbose=False,
        )

        # Add assistant response to full history
        conversation_history.append({"role": "assistant", "content": response})

        # Clear new_messages after generation (tokens are now in KV cache)
        new_messages = []

        # Check if generation stopped due to cache overflow
        if cache_overflow:
            print()
            print("=" * 60)
            print("CONTEXT LIMIT REACHED DURING GENERATION")
            print(f"  Cache position: {model.current_position} / {model.max_context_length}")
            print()
            print("Starting new conversation...")
            print("=" * 60)
            # Reset everything
            model.reset_state()
            conversation_history = []
            new_messages = []
            if system_prompt:
                conversation_history.append({"role": "system", "content": system_prompt})
                new_messages.append({"role": "system", "content": system_prompt})


def generate_incremental(
    model: CoreMLLanguageModel,
    tokenizer,
    new_token_ids: List[int],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    verbose: bool = False,
) -> tuple[str, bool]:
    """Generate text from new tokens.

    This function processes the new tokens and generates a response,
    using the existing KV cache state.

    Args:
        model: CoreML language model
        tokenizer: Tokenizer
        new_token_ids: New token IDs to process
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        verbose: Print generation details

    Returns:
        Tuple of (generated_text, cache_overflow) where cache_overflow is True
        if generation stopped due to KV cache being full.
    """
    # Timing accumulators
    prompt_forward_time = 0.0
    gen_forward_time = 0.0
    lm_head_time = 0.0
    sampling_time = 0.0
    cache_overflow = False

    prompt_length = len(new_token_ids)

    # Process new tokens in chunks
    num_chunks = (prompt_length + model.seq_len - 1) // model.seq_len

    if verbose and num_chunks > 1:
        print(f"Processing {prompt_length} new tokens in {num_chunks} chunks...")

    # Process all prompt chunks and keep the last hidden state
    last_hidden_states = None
    last_chunk_len = 0

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * model.seq_len
        end_idx = min(start_idx + model.seq_len, prompt_length)
        chunk_ids = new_token_ids[start_idx:end_idx]
        last_chunk_len = len(chunk_ids)

        if verbose:
            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: tokens {start_idx} to {end_idx}")

        # Run forward pass (updates KV cache)
        t0 = time.perf_counter()
        last_hidden_states, _ = model.forward(np.array(chunk_ids, dtype=np.int32))
        prompt_forward_time += time.perf_counter() - t0

    # Sample the first token using the last hidden state from prompt processing
    t0 = time.perf_counter()
    logits, log_sum_exp = model.compute_logits(
        last_hidden_states, token_index=last_chunk_len - 1, temperature=temperature
    )
    lm_head_time += time.perf_counter() - t0

    t0 = time.perf_counter()
    next_token = sample_token(logits, log_sum_exp, top_k=top_k, top_p=top_p)
    sampling_time += time.perf_counter() - t0

    # Generate tokens one by one
    generated_ids = [next_token]

    # Check for EOS token
    if next_token == tokenizer.eos_token_id:
        if verbose:
            print("EOS token generated at position 0")
        _print_timing_stats(
            prompt_length,
            len(generated_ids),
            prompt_forward_time,
            gen_forward_time,
            lm_head_time,
            sampling_time,
            model.get_context_info(),
        )
        return tokenizer.decode(generated_ids, skip_special_tokens=True), False

    # Print first token in real-time
    print(tokenizer.decode([next_token]), end="", flush=True)

    for i in range(1, max_new_tokens):
        # Check if cache is about to overflow
        if model.remaining_context <= 0:
            print()
            print("[Cache full - stopping generation]")
            cache_overflow = True
            break

        # Use the last generated token
        current_token = generated_ids[-1]

        # Run forward pass with single token
        t0 = time.perf_counter()
        hidden_states, num_real = model.forward(
            np.array([current_token], dtype=np.int32)
        )
        gen_forward_time += time.perf_counter() - t0

        # Compute logits
        t0 = time.perf_counter()
        logits, log_sum_exp = model.compute_logits(
            hidden_states, token_index=num_real - 1, temperature=temperature
        )
        lm_head_time += time.perf_counter() - t0

        # Sample next token
        t0 = time.perf_counter()
        next_token = sample_token(logits, log_sum_exp, top_k=top_k, top_p=top_p)
        sampling_time += time.perf_counter() - t0

        generated_ids.append(next_token)

        # Check for EOS token
        if next_token == tokenizer.eos_token_id:
            if verbose:
                print(f"EOS token generated at position {i}")
            break

        # Print token in real-time
        print(tokenizer.decode([next_token]), end="", flush=True)

    print()  # Newline after generation

    # Print timing statistics with context info
    _print_timing_stats(
        prompt_length,
        len(generated_ids),
        prompt_forward_time,
        gen_forward_time,
        lm_head_time,
        sampling_time,
        model.get_context_info(),
    )

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, cache_overflow


def format_chat_prompt(conversation_history: List[dict], tokenizer) -> str:
    """Format conversation history into a prompt.

    This tries to use the tokenizer's chat template if available,
    otherwise falls back to a simple format.

    Args:
        conversation_history: List of message dicts with 'role' and 'content'
        tokenizer: Tokenizer

    Returns:
        Formatted prompt string
    """
    # Try to use the tokenizer's chat template
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception:
            pass

    # Fallback to simple format
    lines = []
    for msg in conversation_history:
        role = msg["role"].capitalize()
        content = msg["content"]
        lines.append(f"{role}: {content}")

    lines.append("Assistant:")
    return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CoreML Language Model Inference with Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model inference
    python examples/inference.py --model-dir ./qwen2_0.5b_seqlen_8.mlpackage --model-name Qwen/Qwen2-0.5B

    # Chunked model inference
    python examples/inference.py --model-dir ./qwen3_4b_chunked_4 --model-name Qwen/Qwen3-4B --chunked --num-chunks 4

    # With custom sampling
    python examples/inference.py --model-dir ./model --model-name Qwen/Qwen2-0.5B --temperature 0.7 --top-p 0.9 --top-k 40

    # Single prompt (non-interactive)
    python examples/inference.py --model-dir ./model --model-name Qwen/Qwen2-0.5B --prompt "What is AI?"
        """,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model files (.mlpackage for single, directory for chunked)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model name for tokenizer",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings .npy file (default: model-dir/embeddings.npy)",
    )
    parser.add_argument(
        "--lm-head",
        type=str,
        default=None,
        help="Path to LM head .mlpackage (default: model-dir/lm_head.mlpackage)",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use chunked model",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Number of model chunks (required if --chunked)",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="Maximum context length / KV cache size (auto-detected from model if not specified)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k filtering parameter (0 = no filtering, default: 40)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) filtering parameter (1.0 = no filtering, default: 0.95)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend to conversation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for non-interactive mode (skips chat loop)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity",
    )
    parser.add_argument(
        "--cache-compiled",
        action="store_true",
        help="Cache compiled models (.mlmodelc) for faster subsequent loads. "
        "On first run, saves compiled models alongside .mlpackage files. "
        "On subsequent runs, automatically loads from cached .mlmodelc if available.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.chunked and args.num_chunks <= 1:
        parser.error("--chunked requires --num-chunks > 1")

    # Determine file paths
    model_dir = Path(args.model_dir)
    embeddings_path = (
        Path(args.embeddings) if args.embeddings else (model_dir / "embeddings.npy")
    )
    lm_head_path = (
        Path(args.lm_head) if args.lm_head else (model_dir / "lm_head.mlpackage")
    )

    # Check files exist
    if not embeddings_path.exists():
        print(f"Error: Embeddings file not found: {embeddings_path}")
        print("Please specify the correct path with --embeddings")
        sys.exit(1)

    if not lm_head_path.exists():
        print(f"Error: LM head model not found: {lm_head_path}")
        print("Please specify the correct path with --lm-head")
        sys.exit(1)

    if args.chunked:
        # Check chunks exist
        for i in range(args.num_chunks):
            chunk_path = model_dir / f"chunk_{i}.mlpackage"
            if not chunk_path.exists():
                print(f"Error: Chunk file not found: {chunk_path}")
                sys.exit(1)
    else:
        if not model_dir.exists() or not model_dir.suffix == ".mlpackage":
            print(f"Error: Model file not found: {model_dir}")
            sys.exit(1)

    # Load model
    print("Initializing CoreML Language Model...")
    model = CoreMLLanguageModel(
        model_dir=str(model_dir),
        embeddings_path=str(embeddings_path),
        lm_head_path=str(lm_head_path),
        is_chunked=args.chunked,
        num_chunks=args.num_chunks,
        max_context_length=args.max_context_length,
        cache_compiled=args.cache_compiled,
        verbose=not args.quiet,
    )

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not args.quiet:
        print("\nModel configuration:")
        print(f"  Vocabulary size: {model.vocab_size}")
        print(f"  Hidden dimension: {model.hidden_dim}")
        print(f"  Sequence length: {model.seq_len}")
        print(f"  Max context length: {model.max_context_length}")
        print(f"  Model type: {'Chunked' if args.chunked else 'Single'}")
        if args.chunked:
            print(f"  Number of chunks: {args.num_chunks}")
        print("\nSampling parameters:")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-k: {args.top_k}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Max new tokens: {args.max_new_tokens}")

    # Run inference
    if args.prompt:
        # Non-interactive mode
        print(f"\nPrompt: {args.prompt}")
        print("Response: ", end="", flush=True)
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verbose=not args.quiet,
        )
    else:
        # Interactive chat mode
        chat_loop(
            model=model,
            tokenizer=tokenizer,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            system_prompt=args.system_prompt,
        )


if __name__ == "__main__":
    main()
